"""Plain-language results of one monitoring run: what needs attention, what is worth a look, what was
grouped or ignored and why. Everything here is derived from stored rows; nothing is generated."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Assessment, Document, DocumentVersion, Run, Topic, TopicDocument, Trend

# What each kind of change means, in one sentence a non-specialist can read.
KIND_EXPLAIN = {
    "NEW_FINDING": "A new result with nothing comparable in this topic's history.",
    "DUPLICATE_REPORT": "Another outlet reporting something already known; grouped, not counted twice.",
    "MINOR_UPDATE": "A source edited its text, but no claim or number changed.",
    "MATERIAL_REVISION": "A source changed a number or conclusion it had reported before.",
    "CONTRADICTION": "A new result disagrees with an earlier one under the same conditions.",
    "RELATED_DEVELOPMENT": "Same model and metric as earlier work, but different conditions, so not compared directly.",
    "INSUFFICIENT_EVIDENCE": "A claim with no primary source or measured result; treat with caution.",
}
PRIORITY_EXPLAIN = {
    "HIGH": "Act on this: a material change backed by direct evidence.",
    "REVIEW": "Check before trusting: important if true, but weak or conflicting evidence.",
    "MEDIUM": "Worth reading: a credible, relevant new development.",
    "LOW": "For the record: relevant, but not a change you need to act on.",
    "NONE": "No action: repeats and cosmetic edits, kept for the record.",
}
GROUPS = [
    ("attention", "Needs your attention", ("HIGH", "REVIEW")),
    ("worth", "Worth a look", ("MEDIUM",)),
    ("record", "For the record", ("LOW",)),
    ("quiet", "Grouped or unchanged (no action)", ("NONE",)),
]


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def run_summary(s: Session, run: Run, item_fn) -> dict:
    """item_fn(session, assessment) -> feed item dict (the API's serializer, passed in to avoid a cycle)."""
    topic = s.get(Topic, run.topic_id)
    c = run.counters or {}
    assessments = s.scalars(select(Assessment).where(Assessment.run_id == run.id)).all()
    by_prio: dict[str, list] = {}
    for a in assessments:
        by_prio.setdefault(a.priority, []).append(a)

    order = {"HIGH": 0, "REVIEW": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
    groups = []
    for key, label, prios in GROUPS:
        rows = sorted((a for p in prios for a in by_prio.get(p, [])), key=lambda a: (order[a.priority], -a.id))
        if rows:
            groups.append({"key": key, "label": label, "count": len(rows),
                           "items": [item_fn(s, a) | {"kind_explain": KIND_EXPLAIN.get(a.classification, ""),
                                                      "priority_explain": PRIORITY_EXPLAIN.get(a.priority, "")}
                                     for a in rows]})

    # Items this run judged off-topic, with the reason (the relevance gate is part of the result).
    ignored = s.execute(
        select(Document.title, TopicDocument.reason, TopicDocument.relevance)
        .join(TopicDocument, TopicDocument.document_id == Document.id)
        .join(DocumentVersion, DocumentVersion.id == TopicDocument.version_id)
        .where(TopicDocument.topic_id == run.topic_id, TopicDocument.relevant.is_(False), DocumentVersion.run_id == run.id)
        .order_by(TopicDocument.relevance.desc()).limit(6)).all()

    attention = sum(len(by_prio.get(p, [])) for p in ("HIGH", "REVIEW"))
    worth = len(by_prio.get("MEDIUM", []))
    if run.status == "FAILED":
        headline = "This run failed before it could finish"
    elif attention:
        headline = f"{_plural(attention, 'thing')} need{'s' if attention == 1 else ''} your attention"
    elif worth:
        headline = f"{_plural(worth, 'new finding')} worth a look, nothing urgent"
    elif assessments:
        headline = f"Nothing urgent: {_plural(len(assessments), 'item')} recorded"
    else:
        headline = "Nothing new since the last run"

    facts = []
    srcs = c.get("per_source") or {}
    if srcs:
        ok = [n for n, v in srcs.items() if v.get("status") == "ok"]
        per = ", ".join(f"{n} {v.get('items', 0)}" for n, v in srcs.items() if v.get("status") == "ok")
        facts.append(f"Checked {_plural(len(srcs), 'source')} ({per}): {_plural(c.get('fetched', 0), 'item')} fetched, "
                     f"{c.get('new_documents', 0)} new, {c.get('new_versions', 0)} revised, {c.get('unchanged', 0)} already known.")
        failed = [n for n, v in srcs.items() if v.get("status") != "ok"]
        if failed:
            facts.append(f"{', '.join(failed)} could not be reached; the other {_plural(len(ok), 'source')} still ran.")
    if c.get("irrelevant"):
        facts.append(f"{_plural(c['irrelevant'], 'item was', 'items were')} off-topic and ignored.")
    if c.get("duplicates_grouped"):
        facts.append(f"{_plural(c['duplicates_grouped'], 'report was', 'reports were')} grouped with developments already known.")
    if c.get("llm_claims"):
        facts.append(f"{_plural(c['llm_claims'], 'measurable claim')} read from abstracts by the model, each checked word for word against the source.")
    backfill = any((a.audit or {}).get("backfill") for a in assessments)
    if backfill:
        facts.append("First run on this topic: everything is treated as history, so no alerts were sent.")
    else:
        facts.append(f"{_plural(c.get('alerts', 0), 'alert')} sent; {c.get('suppressed', 0)} held back "
                     "(repeats, cosmetic edits, or below this topic's alert level).")
    if c.get("llm_calls"):
        facts.append(f"Model use: {_plural(c['llm_calls'], 'call')} to Nemotron, {c.get('llm_tokens', 0):,} tokens"
                     f"{', ' + str(c['llm_failures']) + ' retried or failed' if c.get('llm_failures') else ''}; run took "
                     f"{c.get('duration_ms', 0) / 1000:.1f} s.")
    if c.get("verification_failures"):
        facts.append(f"{_plural(c['verification_failures'], 'item')} failed the evidence check and were sent to review.")

    # When this run found nothing, point at the latest earlier run of the topic that did.
    last_findings = None
    if not assessments:
        last_findings = s.scalar(select(Assessment.run_id).where(Assessment.topic_id == run.topic_id, Assessment.run_id < run.id)
                                 .order_by(Assessment.run_id.desc()).limit(1))
    trends = [{"concept": t.concept, "status": t.status, "summary": t.summary}
              for t in s.scalars(select(Trend).where(Trend.topic_id == run.topic_id))]
    return {
        "run": {"id": run.id, "status": run.status, "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None},
        "topic": {"id": topic.id, "name": topic.name},
        "headline": headline,
        "facts": facts,
        "groups": groups,
        "ignored": [{"title": t, "reason": r, "relevance": round(v, 2)} for t, r, v in ignored],
        "sources": [{"name": n} | v for n, v in srcs.items()],
        "trends": trends,
        "last_findings_run": last_findings,
        "priority_explain": PRIORITY_EXPLAIN,
    }
