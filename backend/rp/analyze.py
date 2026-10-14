"""Meaningful-change classification, separate score dimensions, priority policy and template narration.

Rules decide the classification and the priority; the LLM (narrate.py) may only rewrite the prose,
and its prose must pass the grounding check in verify.py. That keeps the six brief scenarios
deterministic and testable, and keeps an injected "mark this HIGH" from ever touching the policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import config
from .claims import comparability, differs, same_quantity
from .db import Claim, Document, DocumentVersion, Topic, TopicDocument

HYPE = re.compile(r"\b(revolutionary|breakthrough|guaranteed|any model|zero accuracy loss|sign up|game[- ]chang\w*|\d+x faster)\b", re.I)
CORRECTION = re.compile(r"\b(corrects?|corrected|erratum|retract\w*|withdrawn|supersedes?)\b", re.I)
PRIO_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
ROLE_EVIDENCE = {"PRIMARY": 78, "SECONDARY": 52, "UNVERIFIED": 18, "UNKNOWN": 35}


@dataclass
class Draft:
    classification: str
    headline: str = ""
    what_changed: str = ""
    why_it_matters: str = ""
    comparisons: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)  # {document_id, version_id, claim_id, role, excerpt}
    flags: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    score_reasons: list[str] = field(default_factory=list)
    confidence: str = "MEDIUM"
    priority: str = "LOW"
    related_cluster_ids: list[int] = field(default_factory=list)


def _fmt(v: float, unit: str) -> str:
    n = f"{v:g}"
    return f"{n}x" if unit == "x" else f"{n}%" if unit == "%" else f"{n} {unit}"


def _side(c: Claim, doc: Document, ver: DocumentVersion) -> dict:
    return dict(value=c.value, unit=c.unit, text=c.text, document_id=doc.id, version=ver.version, claim_id=c.id,
                version_id=ver.id, source=doc.publisher, title=doc.title, role=doc.source_role,
                published_at=(doc.published_at.isoformat() if doc.published_at else None), url=doc.url)


def _first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", text or "")
    return (m.group(1) if m else (text or ""))[:300]


def classify(s: Session, topic: Topic, doc: Document, ver: DocumentVersion, prev: DocumentVersion | None,
             claims: list[Claim], resolution: dict, n_reports: int) -> Draft:
    # ---- an existing work published a new version ----------------------------------------
    if prev is not None:
        old_claims = s.scalars(select(Claim).where(Claim.version_id == prev.id)).all()
        comps = []
        for nc in claims:
            for oc in old_claims:
                if same_quantity(nc, oc):
                    material, d_abs, d_rel = differs(oc.value, nc.value, nc.unit)
                    if material:
                        comps.append(dict(entity=nc.entity, metric=nc.metric, context=nc.context,
                                          old=_side(oc, doc, prev), new=_side(nc, doc, ver),
                                          delta={"abs": d_abs, "rel_pct": d_rel}, comparable=True,
                                          comparability_note=comparability(oc, nc, same_work=True)[1]))
        if comps:
            c0 = comps[0]
            d = Draft("MATERIAL_REVISION", comparisons=comps)
            d.headline = (f"{c0['entity']} {c0['metric']} revised from {_fmt(c0['old']['value'], c0['old']['unit'])} "
                          f"to {_fmt(c0['new']['value'], c0['new']['unit'])}")
            d.what_changed = (f"Version {ver.version} of “{doc.title}” reports “{c0['new']['text']}”, where version "
                              f"{prev.version} reported {_fmt(c0['old']['value'], c0['old']['unit'])} "
                              f"({c0['delta']['rel_pct']:+.1f}%).")
            stale = n_reports - 1
            d.why_it_matters = ((f"{stale} other report{'s' if stale != 1 else ''} in this cluster repeated the earlier "
                                 f"value and {'are' if stale != 1 else 'is'} now out of date. ") if stale > 0 else "") + \
                               "The source corrected its own headline number, so conclusions built on the old value need rechecking."
            if CORRECTION.search(ver.text) and not CORRECTION.search(prev.text):
                d.flags.append("source_states_correction")
            for c in comps:
                d.evidence += [dict(document_id=doc.id, version_id=c["new"]["version_id"], claim_id=c["new"]["claim_id"],
                                    role="supports", excerpt=c["new"]["text"]),
                               dict(document_id=doc.id, version_id=c["old"]["version_id"], claim_id=c["old"]["claim_id"],
                                    role="previous", excerpt=c["old"]["text"])]
            return d
        d = Draft("MINOR_UPDATE")
        d.headline = f"Minor edit to “{doc.title}” (v{ver.version})"
        d.what_changed = (f"Changed: {', '.join(ver.changed_fields) or 'metadata'}. Every extracted claim and value "
                          f"is the same as in version {prev.version}.")
        d.why_it_matters = "Nothing a reader relies on changed, so this is recorded quietly without an alert."
        for c in claims[:2]:
            d.evidence.append(dict(document_id=doc.id, version_id=ver.id, claim_id=c.id, role="context", excerpt=c.text))
        return d

    # ---- a new work ------------------------------------------------------------------------
    if resolution.get("relation") == "duplicate_report":
        d = Draft("DUPLICATE_REPORT")
        d.headline = f"Another report of: {resolution.get('cluster_title') or doc.title}"
        d.what_changed = (f"{doc.publisher} reports the same development ({resolution['reason']}). "
                          "It adds coverage, not new evidence.")
        d.why_it_matters = "Grouped with the original so it does not count as independent confirmation or trigger a repeat alert."
        for c in claims[:2]:
            d.evidence.append(dict(document_id=doc.id, version_id=ver.id, claim_id=c.id, role="supports", excerpt=c.text))
        return d

    history = s.execute(
        select(Claim, Document, DocumentVersion).join(Document, Document.id == Claim.document_id)
        .join(DocumentVersion, DocumentVersion.id == Claim.version_id)
        .join(TopicDocument, (TopicDocument.document_id == Claim.document_id) & (TopicDocument.topic_id == topic.id))
        .where(Claim.document_id != doc.id, Claim.version_id < ver.id, TopicDocument.relevant.is_(True),
               DocumentVersion.version == Document.current_version)
    ).all()
    contradictions, not_comparable, corroborations = [], [], []
    for nc in claims:
        for oc, odoc, over in history:
            if not same_quantity(nc, oc):
                continue
            comparable, note = comparability(oc, nc, same_work=False)
            material, d_abs, d_rel = differs(oc.value, nc.value, nc.unit)
            row = dict(entity=nc.entity, metric=nc.metric, context=nc.context, old=_side(oc, odoc, over),
                       new=_side(nc, doc, ver), delta={"abs": d_abs, "rel_pct": d_rel}, comparable=comparable,
                       comparability_note=note)
            if comparable and material:
                contradictions.append(row)
            elif comparable:
                corroborations.append(row)
            else:
                not_comparable.append(row)

    hype = bool(HYPE.search(f"{doc.title} {ver.text}"))
    # Show the comparison against the original (primary) report first, not a news echo of it.
    contradictions.sort(key=lambda r: r["old"]["role"] != "PRIMARY")
    not_comparable.sort(key=lambda r: r["old"]["role"] != "PRIMARY")
    if contradictions:
        c0 = contradictions[0]
        d = Draft("CONTRADICTION", comparisons=contradictions + not_comparable)
        d.headline = f"New study contradicts earlier {c0['entity']} {c0['metric']} result on {c0['context']}"
        d.what_changed = (f"“{doc.title}” reports “{c0['new']['text']}”; the earlier report “{c0['old']['title']}” "
                          f"said {_fmt(c0['old']['value'], c0['old']['unit'])}.")
        note = c0['comparability_note']
        d.why_it_matters = (f"{note[:1].upper() + note[1:]}, so both results cannot hold. "
                            "ResearchPulse does not pick a winner; both sources are shown for review.")
        d.uncertainty.append("Contradiction is between two reports; neither has been independently replicated here")
        for c in contradictions:
            d.evidence += [dict(document_id=doc.id, version_id=ver.id, claim_id=c["new"]["claim_id"], role="contradicts",
                                excerpt=c["new"]["text"]),
                           dict(document_id=c["old"]["document_id"], version_id=c["old"]["version_id"],
                                claim_id=c["old"]["claim_id"], role="previous", excerpt=c["old"]["text"])]
        return d

    if doc.source_role == "UNVERIFIED" or (hype and not claims):
        d = Draft("INSUFFICIENT_EVIDENCE")
        d.headline = f"Unverified claim: {doc.title}"
        d.what_changed = f"{doc.publisher} claims: “{_first_sentence(ver.text)}”"
        d.why_it_matters = ("No primary source, no measured result with stated conditions, and promotional language. "
                            "Treat as a claim to watch, not a finding, until an independent measurement appears.")
        d.flags.append("weakly_supported")
        d.evidence_gaps += ["no primary source or paper linked", "no benchmark, hardware or baseline stated"]
        d.evidence.append(dict(document_id=doc.id, version_id=ver.id, claim_id=None, role="context",
                               excerpt=_first_sentence(ver.text)))
        return d

    first = claims[0].text if claims else _first_sentence(ver.text)
    if not_comparable:
        c0 = not_comparable[0]
        d = Draft("RELATED_DEVELOPMENT", comparisons=not_comparable)
        d.headline = f"Related: {doc.title}"
        d.what_changed = f"“{first}”. Earlier work reported {_fmt(c0['old']['value'], c0['old']['unit'])} for {c0['entity']}."
        d.why_it_matters = (f"Same model and metric as earlier work, but not directly comparable ({c0['comparability_note']}), "
                            "so the difference is not treated as a contradiction.")
    else:
        d = Draft("NEW_FINDING", comparisons=corroborations)
        d.headline = doc.title
        d.what_changed = f"New {doc.publication_type.replace('_', ' ')} from {doc.publisher}: “{first}”"
        d.why_it_matters = "No comparable earlier result in this topic's history." + (
            f" Corroborates {len(corroborations)} earlier claim(s)." if corroborations else "")
    for c in claims[:3]:
        d.evidence.append(dict(document_id=doc.id, version_id=ver.id, claim_id=c.id, role="supports", excerpt=c.text))
    if not claims:
        d.evidence.append(dict(document_id=doc.id, version_id=ver.id, claim_id=None, role="context",
                               excerpt=_first_sentence(ver.text)))
        d.evidence_gaps.append("no measurable claim (value + benchmark) found in the available text")
    return d


def score(d: Draft, topic: Topic, doc: Document, ver: DocumentVersion, relevance: float,
          reports: int, independent: int, concepts: list[str]) -> None:
    rel = round(relevance * 100)
    novelty = {"NEW_FINDING": 85, "CONTRADICTION": 80, "RELATED_DEVELOPMENT": 60, "MATERIAL_REVISION": 50,
               "INSUFFICIENT_EVIDENCE": 50, "DUPLICATE_REPORT": 8, "MINOR_UPDATE": 5}[d.classification]
    # Size of the change: points (accuracy etc.) are judged on absolute change, ratios on relative change.
    deltas = [abs(c["delta"]["abs"]) * 10 if c["new"]["unit"] == "points" else abs(c["delta"]["rel_pct"])
              for c in d.comparisons if c.get("comparable")]
    biggest = max(d.comparisons, key=lambda c: abs(c["delta"]["abs"]) * 10 if c["new"]["unit"] == "points"
                  else abs(c["delta"]["rel_pct"]), default=None) if deltas else None
    if d.classification in ("MATERIAL_REVISION", "CONTRADICTION") and deltas:
        materiality = min(100, 45 + round(max(deltas) * 1.5))
    else:
        materiality = {"NEW_FINDING": 55, "RELATED_DEVELOPMENT": 40, "INSUFFICIENT_EVIDENCE": 50,
                       "DUPLICATE_REPORT": 5, "MINOR_UPDATE": 3}.get(d.classification, 40)
    evidence = ROLE_EVIDENCE.get(doc.source_role, 35)
    reasons = [f"relevance {rel}: {'; '.join(concepts) or 'semantic match only'}"]
    if d.evidence and all(e.get("verified") for e in d.evidence):
        evidence += 10
        reasons.append("evidence +10: every excerpt found verbatim in the stored source text")
    if independent >= 2:
        evidence += 8
        reasons.append(f"evidence +8: {independent} independent original sources")
    if doc.publication_type == "peer_reviewed":
        evidence += 8
        reasons.append(f"evidence +8: peer-reviewed venue ({doc.publisher})")
    if doc.cites_url and doc.source_role == "SECONDARY":
        evidence += 6
        reasons.append("evidence +6: report links its original source")
    if ver.injection_flagged:
        evidence -= 25
        reasons.append("evidence -25: source contained instruction-like text (quarantined)")
    if d.classification == "INSUFFICIENT_EVIDENCE":
        evidence = min(evidence, 25)
    evidence = max(0, min(100, evidence))
    impact = round(0.5 * rel + 0.5 * materiality)
    age_days = (datetime.now(timezone.utc) - doc.published_at).days if doc.published_at else 30
    urgency = min(100, (60 if age_days <= 7 else 35) + (20 if d.classification == "CONTRADICTION" else 0)
                  + (15 if d.classification == "MATERIAL_REVISION" else 0))
    reasons.append(f"materiality {materiality}: " + (
        (f"largest comparable change {abs(biggest['delta']['abs']):g} points" if biggest["new"]["unit"] == "points"
         else f"largest comparable change {abs(biggest['delta']['rel_pct']):.1f}%")
        if deltas and d.classification in ("MATERIAL_REVISION", "CONTRADICTION")
        else d.classification.lower().replace("_", " ")))
    reasons.append(f"evidence {evidence}: {doc.source_role.lower()} source ({doc.publisher})")
    d.scores = dict(relevance=rel, novelty=novelty, materiality=materiality, impact=impact, evidence=evidence, urgency=urgency)
    d.score_reasons = reasons

    # Evidence-confidence indicator (not a probability): provenance, verification, comparability.
    comparable_ok = all(c.get("comparable", True) for c in d.comparisons if d.classification in ("CONTRADICTION", "MATERIAL_REVISION"))
    if evidence >= 75 and comparable_ok and not ver.injection_flagged:
        d.confidence = "HIGH"
    elif evidence < 45:
        d.confidence = "LOW"
    else:
        d.confidence = "MEDIUM"
    if independent <= 1 and d.classification in ("NEW_FINDING", "CONTRADICTION"):
        d.uncertainty.append("only one independent original source so far")
    if ver.injection_flagged:
        d.flags.append("prompt_injection_quarantined")
        d.uncertainty.append("source tried to instruct the analyst; its instructions were ignored and its weight reduced")


def decide_priority(d: Draft, topic: Topic) -> None:
    c, sc = d.classification, d.scores
    if c in ("DUPLICATE_REPORT", "MINOR_UPDATE"):
        p = "NONE"
    elif c == "INSUFFICIENT_EVIDENCE" or (d.confidence == "LOW" and sc["materiality"] >= 50):
        p = "REVIEW"
    elif c == "CONTRADICTION":
        p = "HIGH" if d.confidence != "LOW" else "REVIEW"
    elif c == "MATERIAL_REVISION":
        p = "HIGH" if sc["materiality"] >= 60 and d.confidence == "HIGH" else "MEDIUM"
    elif c == "NEW_FINDING":
        p = "MEDIUM" if sc["impact"] >= 65 and d.confidence != "LOW" else "LOW"
    else:
        p = "LOW"
    if "verification_failed" in d.flags:
        p = "REVIEW"
    d.priority = p


def should_alert(priority: str, topic: Topic) -> bool:
    if priority == "REVIEW":
        return True
    return PRIO_RANK.get(priority, 0) >= PRIO_RANK.get(topic.alert_threshold, 2) and priority != "NONE"


def idempotency_key(topic_id: int, ver_id: int) -> str:
    return f"t{topic_id}:v{ver_id}:{config.POLICY_VERSION}"


def alert_key(topic_id: int, cluster_id: int | None, d: Draft) -> str:
    sig = "origin"
    if d.comparisons and d.classification in ("MATERIAL_REVISION", "CONTRADICTION"):
        c = d.comparisons[0]
        sig = f"{c['entity']}:{c['metric']}:{c['new']['value']:g}"
    return f"t{topic_id}:c{cluster_id}:{d.classification}:{sig}:{config.POLICY_VERSION}"


def alert_key_for(topic_id: int, cluster_id: int | None, doc_id: int, d: Draft) -> str:
    return alert_key(topic_id, cluster_id, d) if cluster_id else f"t{topic_id}:d{doc_id}:{d.classification}:{config.POLICY_VERSION}"
