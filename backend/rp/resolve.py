"""Topic relevance and 4-tier duplicate/event resolution.

Tier 1 identity (DOI/arXiv/URL) and tier 2 content hash happen in ingest.py. Here:
tier 3 builds candidates (citation link, cosine, title overlap, matching claims) and tier 4 decides.
Similarity only nominates; guards veto merges of independent studies, and the ambiguous band goes to
the LLM adjudicator (or stays separate when no model is available). Preferring "related" over a
wrong merge is deliberate: a false merge hides a development, a missed merge only shows it twice.
"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import embed
from .claims import same_quantity
from .db import Claim, Cluster, ClusterDocument, Document, DocumentVersion, Topic, TopicDocument

DUP_THRESHOLD = 0.86
AMBIGUOUS_LOW = 0.72


def _terms(concept: str) -> list[str]:
    return [t.strip().lower() for t in concept.split("|") if t.strip()]


def concept_hits(concepts: list[str], text: str) -> list[str]:
    low = text.lower()
    return [c.split("|")[0] for c in concepts if any(re.search(rf"\b{re.escape(t)}", low) for t in _terms(c))]


def topic_embedding(t: Topic) -> list[float]:
    profile = f"{t.name}. {t.description} Concepts: {', '.join(c.replace('|', ', ') for c in t.concepts or [])}"
    return embed.embed([profile])[0]


# bge-small cosines sit around 0.55 even for unrelated text; rescale so the threshold means something.
COS_FLOOR, COS_SPAN = 0.55, 0.33


def relevance(t: Topic, doc: Document, ver: DocumentVersion, s: Session | None = None) -> tuple[float, bool, str, list[str]]:
    text = f"{ver.title}. {ver.text}"
    cos = embed.cosine(t.embedding, ver.embedding) if t.embedding is not None else 0.0
    hits = concept_hits(t.concepts or [], text)
    excl = concept_hits(t.exclusions or [], text)
    score = max(0.0, min(1.0, (cos - COS_FLOOR) / COS_SPAN + 0.04 * min(len(hits), 2)))
    if excl and not hits:
        return round(score, 3), False, f"matches exclusion '{excl[0]}' and none of the topic concepts", hits
    if s is not None and doc.cites_url:
        # A report about a relevant original source is relevant, however short it is.
        cited = s.scalar(select(TopicDocument.relevance).join(Document, Document.id == TopicDocument.document_id)
                         .where(TopicDocument.topic_id == t.id, TopicDocument.relevant.is_(True), Document.url == doc.cites_url))
        if cited is not None:
            return round(max(score, cited), 3), True, "reports on an original source already relevant to this topic", hits
    ok = score >= t.relevance_threshold
    why = f"semantic match {score:.2f} (cosine {cos:.2f})" + (f"; concepts: {', '.join(hits)}" if hits else "; no concept named")
    return round(score, 3), ok, why, hits


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def _authors_overlap(a: Document, b: Document) -> bool:
    return bool({x.lower() for x in a.authors or []} & {x.lower() for x in b.authors or []})


def _cluster_of(s: Session, doc_id: int, topic_id: int) -> int | None:
    return s.scalar(select(ClusterDocument.cluster_id).join(Cluster, Cluster.id == ClusterDocument.cluster_id)
                    .where(ClusterDocument.document_id == doc_id, Cluster.topic_id == topic_id).limit(1))


def resolve(s: Session, topic: Topic, doc: Document, ver: DocumentVersion, claims: list[Claim],
            adjudicate=None) -> dict:
    """Attach a new document to an existing cluster (duplicate report / origin) or open a new one.
    Returns {cluster_id, relation, reason, decided_by, score, matched_document_id}."""
    rows = s.execute(
        select(Document, DocumentVersion).join(TopicDocument, TopicDocument.document_id == Document.id)
        .join(DocumentVersion, (DocumentVersion.document_id == Document.id) & (DocumentVersion.version == Document.current_version))
        .where(TopicDocument.topic_id == topic.id, TopicDocument.relevant.is_(True), Document.id != doc.id)
    ).all()
    best = None
    for other, over in rows:
        # Citation link: a report that names the same original source is the same development.
        if doc.cites_url and (doc.cites_url == other.url or doc.cites_url == other.cites_url):
            cid = _cluster_of(s, other.id, topic.id)
            if cid:
                return dict(cluster_id=cid, relation="duplicate_report", decided_by="rule", score=1.0,
                            matched_document_id=other.id, reason=f"cites the same original source ({doc.cites_url})")
        if other.cites_url and other.cites_url == doc.url:
            cid = _cluster_of(s, other.id, topic.id)
            if cid:
                return dict(cluster_id=cid, relation="origin", decided_by="rule", score=1.0,
                            matched_document_id=other.id, reason="this is the original source an earlier report cited")
        cos = embed.cosine(ver.embedding, over.embedding) if over.embedding is not None else 0.0
        other_claims = s.scalars(select(Claim).where(Claim.version_id == over.id)).all()
        ents_a = {c.entity.lower() for c in claims}
        ents_b = {c.entity.lower() for c in other_claims}
        claim_match = any(same_quantity(a, b) and abs(a.value - b.value) < 1e-9 for a in claims for b in other_claims)
        score = 0.6 * cos + 0.2 * _jaccard(ver.title, over.title) + 0.2 * (1.0 if claim_match else 0.0)
        veto = None
        if ents_a and ents_b and not (ents_a & ents_b):
            veto = f"different models ({', '.join(sorted(ents_a))} vs {', '.join(sorted(ents_b))})"
        elif doc.source_role == "PRIMARY" and other.source_role == "PRIMARY" and not _authors_overlap(doc, other):
            veto = "two primary works by different authors are independent studies"
        if best is None or score > best[0]:
            best = (score, other, veto)
    if best and best[0] >= AMBIGUOUS_LOW and best[2] is None:
        score, other, _ = best
        cid = _cluster_of(s, other.id, topic.id)
        if cid and score >= DUP_THRESHOLD:
            return dict(cluster_id=cid, relation="duplicate_report", decided_by="rule", score=round(score, 3),
                        matched_document_id=other.id, reason=f"same development (similarity {score:.2f})")
        if cid and adjudicate is not None:
            verdict = adjudicate(doc, ver, other)
            if verdict and verdict.get("same_event"):
                return dict(cluster_id=cid, relation="duplicate_report", decided_by="llm", score=round(score, 3),
                            matched_document_id=other.id, reason=verdict.get("reason", "model judged same event"))
    cl = Cluster(topic_id=topic.id, title=ver.title, entity=(claims[0].entity if claims else None),
                 event_at=doc.published_at)
    s.add(cl)
    s.flush()
    note = ""
    if best and best[2]:
        note = f"; closest candidate kept separate: {best[2]} (similarity {best[0]:.2f})"
    elif best and best[0] >= AMBIGUOUS_LOW:
        note = f"; ambiguous candidate (similarity {best[0]:.2f}) kept separate without a model decision"
    return dict(cluster_id=cl.id, relation="origin", decided_by="rule", score=None, matched_document_id=None,
                reason="no earlier report of this development" + note, new_cluster=True)


def attach(s: Session, doc: Document, r: dict) -> None:
    s.add(ClusterDocument(cluster_id=r["cluster_id"], document_id=doc.id, relation=r["relation"],
                          reason=r["reason"], decided_by=r["decided_by"], score=r.get("score")))
    s.flush()


def cluster_members(s: Session, cluster_id: int) -> list[tuple[Document, ClusterDocument]]:
    return s.execute(select(Document, ClusterDocument).join(ClusterDocument, ClusterDocument.document_id == Document.id)
                     .where(ClusterDocument.cluster_id == cluster_id).order_by(Document.published_at)).all()


def independence(members: list[tuple[Document, ClusterDocument]]) -> tuple[int, int, str]:
    """(reports, independent original sources, note). Reports that cite one original count once."""
    n = len(members)
    originals = {d.url for d, _ in members if d.source_role == "PRIMARY" and not d.cites_url}
    originals |= {d.cites_url for d, _ in members if d.cites_url}
    k = max(1, len(originals)) if n else 0
    if n <= 1:
        return n, k, "single report"
    return n, k, f"{n} reports, based on {k} original source{'s' if k != 1 else ''}"
