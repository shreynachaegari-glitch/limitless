"""Evaluation harness: measured numbers, not claims.

  python -m rp.evaluate            -> docs/eval/report.md + report.json

Part A replays the six-scenario corpus in its own database (researchpulse_eval, never the live one) with
the model off, and scores every decision against hand-written expected labels. Part B measures the
claim extractor on a labelled set of sentences, including qualifiers, negations and sentences with no
claim. Part C reads the latest live runs from the main database (read-only): model-claim acceptance,
verbatim check, narration acceptance, tokens and latency.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path

MAIN_DB = os.environ.get("DATABASE_URL", "postgresql+psycopg://rp:rp@127.0.0.1:5433/researchpulse")
EVAL_DB = MAIN_DB.rsplit("/", 1)[0] + "/researchpulse_eval"

# ---- expected labels for the scenario corpus (fixture key -> per-version classification) -------
EXPECTED = {
    "novaquant": ["NEW_FINDING", "MINOR_UPDATE", "MATERIAL_REVISION"], "orion-spec": ["NEW_FINDING"],
    "paged-kv": ["NEW_FINDING"], "techwire-nova": ["DUPLICATE_REPORT"], "mldigest-nova": ["DUPLICATE_REPORT"],
    "reeval-nova": ["CONTRADICTION"], "lumen-quant": ["NEW_FINDING"], "spec-medusa": ["NEW_FINDING"],
    "spec-selfdraft": ["NEW_FINDING"], "spec-batch": ["RELATED_DEVELOPMENT"], "spec-industry": ["RELATED_DEVELOPMENT"],
    "hype-hyperserve": ["INSUFFICIENT_EVIDENCE"], "inject-kv": ["NEW_FINDING"], "sparse-attn": ["RELATED_DEVELOPMENT"],
}
EXPECTED_IRRELEVANT = {"diffusion-offtopic"}
EXPECTED_SAME_DEVELOPMENT = [{"novaquant", "techwire-nova", "mldigest-nova"}]  # everything else is its own development
EXPECTED_ALERTS = {("reeval-nova", 0), ("hype-hyperserve", 0), ("inject-kv", 0), ("novaquant", 2)}
EXPECTED_TRENDS = {"speculative decoding": "EMERGING", "quantization": "TENTATIVE", "kv cache": "NO_SIGNAL",
                   "sparse attention": "INSUFFICIENT_HISTORY"}

# ---- labelled sentences for the deterministic claim extractor ---------------------------------
CLAIM_CASES = [
    ("Nova-7B with NovaQuant achieves a 2.4x decode speedup on A100 at batch size 1.", [("Nova-7B", 2.4, "x", "A100")]),
    ("Nova-7B quantized to 4-bit loses 0.8 points accuracy on MMLU.", [("Nova-7B", 0.8, "points", "MMLU")]),
    ("Orion-13B achieves up to 3.1x decode speedup on H100.", [("Orion-13B", 3.1, "x", "H100")]),
    ("Kite-7B does not achieve a 1.2x decode speedup on A100.", [("Kite-7B", 1.2, "x", "A100")]),
    ("Atlas-34B with paged eviction achieves a 1.4x throughput gain on A100 at 32k context.", [("Atlas-34B", 1.4, "x", "A100")]),
    ("We describe a new scheduling policy for multi-tenant serving.", []),
    ("Our method is 10x faster on any model.", []),
    ("Vega-70B reaches 91.2% accuracy on GSM8K while Lumen-8B reaches 77.5% accuracy on GSM8K.",
     [("Vega-70B", 91.2, "%", "GSM8K"), ("Lumen-8B", 77.5, "%", "GSM8K")]),
    ("Version 2.1 of the toolkit was released on 3 March.", []),
    ("Nova-7B gains 2 points accuracy on ARC with the new prompt.", [("Nova-7B", 2.0, "points", "ARC")]),
]
QUALIFIER_CASES = [("Orion-13B achieves up to 3.1x decode speedup on H100.", "qualifier", "up to"),
                   ("Kite-7B does not achieve a 1.2x decode speedup on A100.", "negated", "yes")]


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": round(p, 3), "recall": round(r, 3), "tp": tp, "fp": fp, "fn": fn}


def part_a() -> dict:
    os.environ["DATABASE_URL"], os.environ["OFFLINE"] = EVAL_DB, "1"
    from sqlalchemy import create_engine, func, select, text
    with create_engine(MAIN_DB.rsplit("/", 1)[0] + "/postgres", isolation_level="AUTOCOMMIT").connect() as c:
        if not c.execute(text("SELECT 1 FROM pg_database WHERE datname='researchpulse_eval'")).scalar():
            c.execute(text("CREATE DATABASE researchpulse_eval"))
    from .db import Alert, Assessment, AssessmentEvidence, ClusterDocument, Document, DocumentVersion, Run, Trend, session
    from .pipeline import create_run, execute
    from .seed import reset

    ids = reset(drop=True)
    durations = []
    for _ in range(5):
        with session() as s:
            run = create_run(s, ids["topic_id"])
        t0 = time.time()
        execute(run.id)
        durations.append(round(time.time() - t0, 2))
    with session() as s:
        docs = {d.id: d.external_id for d in s.scalars(select(Document))}
        got, prio = {}, {}
        for a in s.scalars(select(Assessment).order_by(Assessment.id)):
            k = docs[a.document_id]
            got.setdefault(k, []).append(a.classification)
            prio[(k, len(got[k]) - 1)] = a.priority
        # classification accuracy
        total = correct = 0
        confusions = []
        for k, exp in EXPECTED.items():
            for i, e in enumerate(exp):
                total += 1
                g = (got.get(k) or [None] * (i + 1))[i] if i < len(got.get(k, [])) else None
                correct += g == e
                if g != e:
                    confusions.append(f"{k} v{i + 1}: expected {e}, got {g}")
        irrelevant_ok = all(k not in got for k in EXPECTED_IRRELEVANT)
        false_contra = sum(1 for k, v in got.items() for c in v if c == "CONTRADICTION" and "CONTRADICTION" not in EXPECTED.get(k, []))
        # duplicate clustering, pairwise
        cl = {}
        for cd in s.scalars(select(ClusterDocument)):
            cl.setdefault(docs[cd.document_id], cd.cluster_id)
        keys = [k for k in EXPECTED if k in cl]
        same = {frozenset(p) for g in EXPECTED_SAME_DEVELOPMENT for p in combinations(sorted(g), 2)}
        tp = fp = fn = 0
        for a, b in combinations(sorted(keys), 2):
            pred, exp = cl[a] == cl[b], frozenset((a, b)) in same
            tp += pred and exp
            fp += pred and not exp
            fn += exp and not pred
        # alerts
        alerted = set()
        for al in s.scalars(select(Alert)):
            a = s.get(Assessment, al.assessment_id)
            k = docs[a.document_id]
            idx = [x.id for x in s.scalars(select(Assessment).where(Assessment.document_id == a.document_id).order_by(Assessment.id))].index(a.id)
            alerted.add((k, idx))
        atp = len(alerted & EXPECTED_ALERTS)
        trends = {t.concept: t.status for t in s.scalars(select(Trend))}
        trend_ok = sum(trends.get(k) == v for k, v in EXPECTED_TRENDS.items())
        ev_total = s.scalar(select(func.count(AssessmentEvidence.id)))
        ev_ok = s.scalar(select(func.count(AssessmentEvidence.id)).where(AssessmentEvidence.verified.is_(True)))
        before = [s.scalar(select(func.count()).select_from(m)) for m in (DocumentVersion, Assessment, Alert)]
        inject_prio = prio.get(("inject-kv", 0))
    with session() as s:
        run = create_run(s, ids["topic_id"])
    execute(run.id)
    with session() as s:
        after = [s.scalar(select(func.count()).select_from(m)) for m in (DocumentVersion, Assessment, Alert)]
    return {
        "classification": {"correct": correct, "total": total, "accuracy": round(correct / total, 3), "errors": confusions},
        "off_topic_rejected": irrelevant_ok,
        "false_contradictions": false_contra,
        "duplicate_pairs": _prf(tp, fp, fn),
        "alerts": _prf(atp, len(alerted - EXPECTED_ALERTS), len(EXPECTED_ALERTS - alerted)),
        "trends": {"correct": trend_ok, "total": len(EXPECTED_TRENDS), "got": trends},
        "evidence_verified": {"verified": ev_ok, "total": ev_total},
        "prompt_injection": {"priority_given": inject_prio, "obeyed": inject_prio == "HIGH"},
        "idempotent_rerun": {"new_rows": [a - b for a, b in zip(after, before)], "ok": after == before},
        "run_seconds": durations,
    }


def part_b() -> dict:
    from . import claims
    tp = fp = fn = 0
    errs = []
    for sent, exp in CLAIM_CASES:
        got = {(c["entity"], c["value"], c["unit"], c["context"]) for c in claims.extract(sent)}
        want = set(exp)
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
        if got != want:
            errs.append({"sentence": sent, "missed": sorted(want - got), "extra": sorted(got - want)})
    q_ok = sum(1 for s, k, v in QUALIFIER_CASES if (claims.extract(s) or [{}])[0].get("conditions", {}).get(k) == v)
    return {"pattern_extractor": _prf(tp, fp, fn) | {"sentences": len(CLAIM_CASES), "errors": errs},
            "qualifiers_and_negations": {"correct": q_ok, "total": len(QUALIFIER_CASES)}}


def part_c() -> dict:
    from sqlalchemy import create_engine, text
    eng = create_engine(MAIN_DB)
    with eng.connect() as c:
        live = c.execute(text("SELECT id FROM topics WHERE name LIKE '%(live)%' LIMIT 1")).scalar()
        if not live:
            return {"note": "no live topic"}
        runs = c.execute(text("SELECT id, counters FROM runs WHERE topic_id=:t AND status IN ('SUCCEEDED','PARTIAL_SUCCESS') ORDER BY id"), {"t": live}).all()
        kept = rej = 0
        for (msg,) in c.execute(text("SELECT message FROM run_events WHERE run_id IN (SELECT id FROM runs WHERE topic_id=:t) AND message LIKE 'read %abstracts%'"), {"t": live}):
            m = re.search(r"(\d+) claims kept.*?(\d+) rejected", msg)
            if m:
                kept, rej = kept + int(m.group(1)), rej + int(m.group(2))
        exact = c.execute(text("SELECT count(*) FILTER (WHERE substr(v.text, cl.span_start+1, cl.span_end-cl.span_start) = cl.text), count(*) "
                               "FROM claims cl JOIN document_versions v ON v.id=cl.version_id WHERE cl.extractor LIKE 'llm:%'")).one()
        nar = c.execute(text("SELECT audit->>'validation' FROM assessments WHERE topic_id=:t AND audit->>'validation' <> 'template'"), {"t": live}).scalars().all()
        fetched = sum((r.counters or {}).get("fetched", 0) for r in runs)
        relevant = sum((r.counters or {}).get("assessed", 0) for r in runs)
        tokens = sum((r.counters or {}).get("llm_tokens", 0) for r in runs)
        calls = sum((r.counters or {}).get("llm_calls", 0) for r in runs)
        secs = [round((r.counters or {}).get("duration_ms", 0) / 1000, 1) for r in runs if (r.counters or {}).get("duration_ms")]
    passed = sum(1 for v in nar if v and v.startswith("passed"))
    return {
        "live_runs": len(runs), "fetched": fetched, "assessed_relevant": relevant,
        "model_claims": {"kept": kept, "rejected": rej, "acceptance": round(kept / (kept + rej), 3) if kept + rej else None,
                         "kept_exact_slices": f"{exact[0]}/{exact[1]}"},
        "narration": {"attempted": len(nar), "passed_gate": passed},
        "llm_usage": {"calls": calls, "tokens": tokens, "note": "counted for runs since usage tracking was added"},
        "run_seconds": secs,
    }


def render(r: dict) -> str:
    a, b, c = r["scenarios"], r["claims"], r["live"]
    L = [f"# ResearchPulse evaluation report", "", f"Generated {r['generated']} by `python -m rp.evaluate`. All numbers measured by this script.", "",
         "## A. Scenario corpus (model off, deterministic)", "", "| Measure | Result |", "|---|---|",
         f"| Change classification | {a['classification']['correct']}/{a['classification']['total']} correct ({a['classification']['accuracy']:.0%}) |",
         f"| Off-topic item rejected | {'yes' if a['off_topic_rejected'] else 'no'} |",
         f"| False contradictions | {a['false_contradictions']} |",
         f"| Duplicate grouping (pairwise) | precision {a['duplicate_pairs']['precision']:.0%}, recall {a['duplicate_pairs']['recall']:.0%} ({a['duplicate_pairs']['fp']} false merges, {a['duplicate_pairs']['fn']} missed) |",
         f"| Alerts | precision {a['alerts']['precision']:.0%}, recall {a['alerts']['recall']:.0%} |",
         f"| Trend status | {a['trends']['correct']}/{a['trends']['total']} correct |",
         f"| Evidence excerpts verified verbatim | {a['evidence_verified']['verified']}/{a['evidence_verified']['total']} |",
         f"| Prompt injection obeyed | {'YES' if a['prompt_injection']['obeyed'] else 'no'} (priority given: {a['prompt_injection']['priority_given']}) |",
         f"| Rerun of identical data | new rows {a['idempotent_rerun']['new_rows']} ({'idempotent' if a['idempotent_rerun']['ok'] else 'NOT idempotent'}) |",
         f"| Run time per run (s) | {', '.join(map(str, a['run_seconds']))} |", ""]
    if a["classification"]["errors"]:
        L += ["Misclassified:", *[f"- {e}" for e in a["classification"]["errors"]], ""]
    p = b["pattern_extractor"]
    L += ["## B. Claim extraction (labelled sentences)", "", "| Measure | Result |", "|---|---|",
          f"| Pattern extractor | precision {p['precision']:.0%}, recall {p['recall']:.0%} over {p['sentences']} sentences |",
          f"| Qualifiers / negations captured | {b['qualifiers_and_negations']['correct']}/{b['qualifiers_and_negations']['total']} |", ""]
    if p["errors"]:
        L += ["Extractor misses (kept as known limits):", *[f"- {e['sentence']} (missed {e['missed']}, extra {e['extra']})" for e in p["errors"]], ""]
    if "model_claims" in c:
        L += ["## C. Live runs (real arXiv, Crossref, OpenAlex; read-only)", "", "| Measure | Result |", "|---|---|",
              f"| Runs / fetched / relevant | {c['live_runs']} / {c['fetched']} / {c['assessed_relevant']} |",
              f"| Model-proposed claims kept | {c['model_claims']['kept']} kept, {c['model_claims']['rejected']} rejected ({(c['model_claims']['acceptance'] or 0):.0%} accepted) |",
              f"| Kept claims that are exact slices of the source | {c['model_claims']['kept_exact_slices']} |",
              f"| Model narration passing the grounding gate | {c['narration']['passed_gate']}/{c['narration']['attempted']} |",
              f"| Model usage | {c['llm_usage']['calls']} calls, {c['llm_usage']['tokens']:,} tokens ({c['llm_usage']['note']}) |",
              f"| Run time (s) | {', '.join(map(str, c['run_seconds'])) or 'n/a'} |", ""]
    L += ["## What this does not show", "",
          "- The scenario corpus is small and was written alongside the system; it checks the six brief scenarios, not general accuracy.",
          "- Claim recall on real abstracts is not measured: there is no hand-labelled set of real abstracts yet.",
          "- Confidence is an explained rule, not a calibrated probability.", ""]
    return "\n".join(L)


def main() -> None:
    out = Path(__file__).resolve().parents[2] / "docs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    live = part_c()  # read the main database before switching to the eval database
    report = {"generated": time.strftime("%Y-%m-%d %H:%M %Z"), "claims": part_b(), "live": live, "scenarios": part_a()}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "report.md").write_text(render(report))
    print(render(report))


if __name__ == "__main__":
    sys.exit(main())
