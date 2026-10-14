"""Trend signals: distinct developments (clusters) per concept per window vs a baseline.

Counts clusters, not articles, so syndicated copies of one press release cannot fake a trend; clusters
backed only by unverified sources are not counted. Says INSUFFICIENT_HISTORY when it cannot tell.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .db import Assessment, ClusterDocument, Document, DocumentVersion, TopicDocument, Topic, Trend, utcnow
from .resolve import concept_hits

WINDOW = 14
BASELINE_WINDOWS = 3

LIMITS = ["Counts distinct developments (clusters), not articles",
          "Co-occurrence over time only; no causal claim",
          "Limited to the sources this topic monitors"]


def recompute(s: Session, topic: Topic) -> list[Trend]:
    rows = s.execute(
        select(Document, ClusterDocument.cluster_id, DocumentVersion.text)
        .join(TopicDocument, (TopicDocument.document_id == Document.id) & (TopicDocument.topic_id == topic.id))
        .join(ClusterDocument, ClusterDocument.document_id == Document.id)
        .join(DocumentVersion, (DocumentVersion.document_id == Document.id) & (DocumentVersion.version == Document.current_version))
        .where(TopicDocument.relevant.is_(True))
    ).all()
    now = datetime.now(timezone.utc)
    if rows:
        now = max(now, max(d.published_at for d, _, _ in rows if d.published_at) if any(d.published_at for d, _, _ in rows) else now)
    earliest = min((d.published_at for d, _, _ in rows if d.published_at), default=None)
    span_days = (now - earliest).days if earliest else 0

    by_concept: dict[str, dict[int, dict]] = defaultdict(dict)
    articles: dict[str, int] = defaultdict(int)
    for d, cid, text in rows:
        for c in concept_hits(topic.concepts or [], f"{d.title} {text}"):
            articles[c] += 1
            cl = by_concept[c].setdefault(cid, {"date": d.published_at, "title": d.title, "roles": set(), "origins": set()})
            cl["roles"].add(d.source_role)
            cl["origins"].add(d.cites_url or d.url)
            if d.published_at and (cl["date"] is None or d.published_at < cl["date"]):
                cl["date"], cl["title"] = d.published_at, d.title

    contra = s.execute(select(Assessment.cluster_id, Assessment.headline).where(
        Assessment.topic_id == topic.id, Assessment.classification == "CONTRADICTION")).all()

    s.execute(delete(Trend).where(Trend.topic_id == topic.id))
    out = []
    for concept in dict.fromkeys(c.split("|")[0].strip()[:190] for c in topic.concepts or [] if c.strip()):
        clusters = {k: v for k, v in by_concept.get(concept, {}).items()
                    if v["date"] and v["roles"] - {"UNVERIFIED"}}
        series = []
        for w in range(BASELINE_WINDOWS, -1, -1):
            hi = now - timedelta(days=WINDOW * w)
            lo = hi - timedelta(days=WINDOW)
            series.append({"window_start": lo.isoformat(),
                           "count": sum(1 for v in clusters.values() if lo < v["date"] <= hi)})
        current = series[-1]["count"]
        baseline = sum(x["count"] for x in series[:-1])
        avg = baseline / BASELINE_WINDOWS
        n = len(clusters)
        if span_days < 2 * WINDOW or n < 2:
            status = "INSUFFICIENT_HISTORY"
            summary = (f"Only {n} development{'s' if n != 1 else ''} and {span_days} days of history: "
                       "too early to call a trend.")
        # Rates are compared per equal-length window: the 42-day baseline is divided into three 14-day
        # windows, so "4 now vs 1 before" is 4 vs 0.33 per window (12x), not 4x.
        recent = [v for v in clusters.values() if v["date"] > now - timedelta(days=WINDOW)]
        independent = len(set().union(*(v["origins"] for v in recent))) if recent else 0
        ratio = current / avg if avg else None
        rate = (f"{ratio:.0f}x the earlier rate" if ratio else "none in the baseline windows")
        if span_days < 2 * WINDOW or n < 2:
            pass  # handled above
        elif current >= 3 and current >= 2 * max(avg, 0.5) and independent < 2:
            status = "TENTATIVE"
            summary = (f"{current} developments in the last {WINDOW} days, but they trace back to {independent} original "
                       "source: not enough independent evidence to call it emerging.")
        elif current >= 3 and current >= 2 * max(avg, 0.5):
            status = "EMERGING"
            summary = (f"{current} distinct developments from {independent} independent sources in the last {WINDOW} days vs "
                       f"{avg:.1f} per {WINDOW}-day window before ({rate}).")
        elif all(x["count"] >= 2 for x in series):
            status = "ESTABLISHED"
            summary = f"At least 2 distinct developments in every {WINDOW}-day window for {WINDOW * (BASELINE_WINDOWS + 1)} days."
        elif current >= 2 and current > avg:
            status = "TENTATIVE"
            summary = f"{current} developments in the last {WINDOW} days vs an average of {avg:.1f} per window before: a weak signal."
        else:
            status = "NO_SIGNAL"
            summary = f"{current} development{'s' if current != 1 else ''} in the last {WINDOW} days vs {baseline} before: no change in activity."
        if articles.get(concept, 0) > n:
            summary += f" ({articles[concept]} articles collapse to {n} distinct developments.)"
        ev = sorted(({"cluster_id": k, "title": v["title"], "published_at": v["date"].isoformat()} for k, v in clusters.items()),
                    key=lambda e: e["published_at"], reverse=True)
        t = Trend(topic_id=topic.id, concept=concept, status=status, window_days=WINDOW, current_count=current,
                  baseline_count=baseline, baseline_windows=BASELINE_WINDOWS, article_count=articles.get(concept, 0),
                  summary=summary, limitations=LIMITS, series=series, evidence=ev[:8],
                  contradicting=[{"cluster_id": cid, "headline": h} for cid, h in contra if cid in clusters],
                  updated_at=utcnow())
        s.add(t)
        out.append(t)
    s.flush()
    return out
