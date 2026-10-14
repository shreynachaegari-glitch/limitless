"""HTTP API (contract: ../API_CONTRACT.md). Run: uvicorn rp.api:app --port 8000"""

from __future__ import annotations

import json
import time
from datetime import timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select, text

from . import llm, seed
from .db import (Alert, Assessment, AssessmentEvidence, Cluster, ClusterDocument, Document, DocumentVersion, Feedback,
                 Run, RunEvent, Source, Topic, TopicDocument, Trend, init_db, session, utcnow)
from .netsafe import UnsafeURL, assert_public_url
from .pipeline import create_run
from .resolve import cluster_members, independence
from .summary import run_summary

app = FastAPI(title="ResearchPulse API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5181", "http://127.0.0.1:5181",
                                                  "http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


ALLOWED_ORIGINS = {"http://localhost:5181", "http://127.0.0.1:5181", "http://localhost:5173", "http://127.0.0.1:5173"}


@app.middleware("http")
async def _origin_guard(request, call_next):
    """CSRF guard: a page on another site can't POST/PATCH here through the user's browser."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        # Same-origin requests (the UI and its /api proxy, on localhost or behind a tunnel) always pass;
        # a page on any other site is rejected.
        same_origin = origin is not None and origin.split("://", 1)[-1] == request.headers.get("host")
        if origin is not None and origin not in ALLOWED_ORIGINS and not same_origin:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "origin not allowed"}, status_code=403)
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    with session() as s:
        if not s.scalar(select(func.count(Topic.id))):
            seed.reset()


def iso(d):
    return d.isoformat() if d else None


# ---- serialisers ------------------------------------------------------------------------------

def topic_out(t: Topic) -> dict:
    return dict(id=t.id, name=t.name, description=t.description, concepts=t.concepts, exclusions=t.exclusions,
                relevance_threshold=t.relevance_threshold, alert_threshold=t.alert_threshold, poll_minutes=t.poll_minutes,
                active=t.active, source_ids=t.source_ids, last_run_at=iso(t.last_run_at))


def source_out(s_: Source) -> dict:
    if s_.last_checked is None:
        status = "never_run"
    elif s_.last_error:
        status = "failing"
    else:
        status = "ok"
    return dict(id=s_.id, kind=s_.kind, name=s_.name, url=s_.url, query=s_.query, role_default=s_.role_default,
                enabled=s_.enabled, health=dict(status=status, last_checked=iso(s_.last_checked),
                                                last_success=iso(s_.last_success), error=s_.last_error,
                                                items_last_run=s_.items_last_run,
                                                consecutive_failures=s_.consecutive_failures))


def doc_ref(d: Document, retrieved=None) -> dict:
    return dict(document_id=d.id, title=d.title, url=d.url, publisher=d.publisher, role=d.source_role,
                published_at=iso(d.published_at), retrieved_at=iso(retrieved or d.first_seen_at), synthetic=d.synthetic)


def feed_item(s, a: Assessment) -> dict:
    doc = s.get(Document, a.document_id)
    members = cluster_members(s, a.cluster_id) if a.cluster_id else [(doc, None)]
    n, k, note = independence(members)
    return dict(
        id=a.id, topic_id=a.topic_id, cluster_id=a.cluster_id, headline=a.headline, classification=a.classification,
        priority=a.priority, confidence=a.confidence, what_changed=a.what_changed, why_it_matters=a.why_it_matters,
        scores=a.scores, source_count=n, independent_sources=k, independence_note=note,
        sources=[doc_ref(d) for d, _ in members], published_at=iso(doc.published_at), first_seen_at=iso(a.created_at),
        version=s.get(DocumentVersion, a.version_id).version, synthetic=doc.synthetic, read=a.read,
        narrator=a.narrator, flags=a.flags, change_line=_change_line(a),
    )


def _fmt_val(v: float, unit: str) -> str:
    n = f"{v:g}"
    return f"{n}x" if unit == "x" else f"{n}%" if unit == "%" else f"{n} {unit}"


def _change_line(a: Assessment) -> str | None:
    """One readable line for the headline change: 'Nova-7B decode speedup: 2.4x → 1.9x (−20.8 %)'."""
    if not a.comparisons:
        return None
    c = a.comparisons[0]
    old, new = c["old"], c["new"]
    if a.classification == "DUPLICATE_REPORT" or old["value"] == new["value"]:
        return None
    o, n = _fmt_val(old["value"], old["unit"]), _fmt_val(new["value"], new["unit"])
    rel = "" if new["unit"] == "points" else f" ({c['delta']['rel_pct']:+.1f} %)".replace("-", "−")
    what = f"{c['entity']} {c['metric']}"
    if a.classification == "MATERIAL_REVISION":
        return f"{what}: v{old['version']} said {o} → v{new['version']} says {n}{rel}"
    if a.classification == "CONTRADICTION":
        return f"{what}: earlier report {o} vs new study {n}"
    if not c.get("comparable"):
        return f"{what}: earlier work {o}, this report {n} · not directly comparable"
    return f"{what}: {o} → {n}{rel}"


# ---- overview -------------------------------------------------------------------------------

@app.get("/api/overview")
def overview():
    with session() as s:
        pc = dict(s.execute(select(Assessment.priority, func.count()).group_by(Assessment.priority)).all())
        cls = dict(s.execute(select(Assessment.classification, func.count()).group_by(Assessment.classification)).all())
        tr = dict(s.execute(select(Trend.status, func.count()).group_by(Trend.status)).all())
        last = s.scalar(select(Run).where(Run.finished_at.is_not(None)).order_by(Run.id.desc()).limit(1))
        srcs = s.scalars(select(Source)).all()
        failing = sum(1 for x in srcs if x.last_error)
        docs = s.scalar(select(func.count(Document.id))) or 0
        clusters = s.scalar(select(func.count(Cluster.id))) or 0
        dup = cls.get("DUPLICATE_REPORT", 0)
        alerts = s.scalar(select(func.count(Alert.id))) or 0
        suppressed = sum((r.counters or {}).get("suppressed", 0) for r in s.scalars(select(Run)))
        verified = s.scalar(select(func.count(AssessmentEvidence.id)).where(AssessmentEvidence.verified.is_(True))) or 0
        downgraded = pc.get("REVIEW", 0)
        meaningful = sum(cls.get(k, 0) for k in ("NEW_FINDING", "MATERIAL_REVISION", "CONTRADICTION", "RELATED_DEVELOPMENT"))
        week = utcnow() - timedelta(days=7)
        uniq7 = s.scalar(select(func.count(Cluster.id)).where(Cluster.event_at >= week)) or 0
        return dict(
            topics=s.scalar(select(func.count(Topic.id))), active_topics=s.scalar(select(func.count(Topic.id)).where(Topic.active.is_(True))),
            documents=docs, clusters=clusters, unique_developments_7d=uniq7, priority_counts=pc,
            contradictions=cls.get("CONTRADICTION", 0), trends=tr,
            unread_alerts=s.scalar(select(func.count(Alert.id)).where(Alert.read.is_(False))) or 0,
            last_run=(dict(id=last.id, status=last.status, finished_at=iso(last.finished_at), counters=last.counters) if last else None),
            source_problems=failing, llm=llm.provider(),
            agents={
                "scout": {"label": "items fetched", "value": docs, "detail": f"{len(srcs)} sources, {failing} failing"},
                "resolver": {"label": "duplicate reports grouped", "value": dup, "detail": f"{clusters} distinct developments"},
                "analyst": {"label": "meaningful changes", "value": meaningful,
                            "detail": f"{cls.get('CONTRADICTION', 0)} contradiction, {cls.get('MATERIAL_REVISION', 0)} revision, {cls.get('MINOR_UPDATE', 0)} minor"},
                "verifier": {"label": "excerpts verified", "value": verified, "detail": f"{downgraded} flagged for review"},
                "trend": {"label": "trend signals", "value": tr.get("EMERGING", 0) + tr.get("TENTATIVE", 0) + tr.get("ESTABLISHED", 0),
                          "detail": f"{tr.get('EMERGING', 0)} emerging, {tr.get('INSUFFICIENT_HISTORY', 0)} insufficient history"},
                "sentinel": {"label": "alerts sent", "value": alerts, "detail": f"{suppressed} suppressed"},
            },
        )


# ---- topics ---------------------------------------------------------------------------------

class TopicIn(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    concepts: list[str] | None = None
    exclusions: list[str] | None = None
    source_ids: list[int] | None = None
    relevance_threshold: float | None = Field(default=None, ge=0, le=1)
    alert_threshold: str | None = Field(default=None, pattern="^(HIGH|MEDIUM|LOW)$")
    poll_minutes: int | None = Field(default=None, ge=0, le=10080)
    active: bool | None = None


@app.get("/api/topics")
def topics():
    with session() as s:
        return [topic_out(t) for t in s.scalars(select(Topic).order_by(Topic.id))]


@app.post("/api/topics")
def create_topic(body: TopicIn):
    if not body.name:
        raise HTTPException(422, "name is required")
    with session() as s:
        default_sources = [x.id for x in s.scalars(select(Source).where(Source.kind.in_(("arxiv", "openalex", "crossref"))))]
        t = Topic(name=body.name, description=body.description or "", concepts=body.concepts or [],
                  exclusions=body.exclusions or [], source_ids=body.source_ids or default_sources,
                  relevance_threshold=0.5 if body.relevance_threshold is None else body.relevance_threshold,
                  alert_threshold=body.alert_threshold or "HIGH",
                  poll_minutes=body.poll_minutes or 0, active=True if body.active is None else body.active)
        if t.concepts and len({c.split('|')[0].strip().lower() for c in t.concepts}) != len(t.concepts):
            raise HTTPException(422, "concepts must have distinct first names")
        s.add(t)
        s.flush()
        return topic_out(t)


@app.patch("/api/topics/{tid}")
def patch_topic(tid: int, body: TopicIn):
    with session() as s:
        t = s.get(Topic, tid) or _404()
        data = body.model_dump(exclude_none=True)
        for k, v in data.items():
            setattr(t, k, v)
        if {"name", "description", "concepts"} & data.keys():
            t.embedding, t.profile_version = None, t.profile_version + 1  # re-embedded on the next run
        return topic_out(t)


@app.post("/api/topics/{tid}/run")
def run_topic(tid: int):
    with session() as s:
        s.get(Topic, tid) or _404()
        s.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": 100000 + tid})  # double-click → one run
        busy = s.scalar(select(Run.id).where(Run.topic_id == tid, Run.status.in_(("QUEUED", "RUNNING"))))
        if busy:
            return {"run_id": busy, "already_running": True}
        return {"run_id": create_run(s, tid).id}


def run_out(r: Run) -> dict:
    return dict(id=r.id, topic_id=r.topic_id, status=r.status, phase=r.phase, started_at=iso(r.started_at),
                finished_at=iso(r.finished_at), counters=r.counters, error=r.error)


@app.get("/api/runs/{rid}")
def get_run(rid: int):
    with session() as s:
        return run_out(s.get(Run, rid) or _404())


@app.get("/api/runs")
def list_runs(topic_id: int | None = None, limit: int = 20):
    with session() as s:
        q = select(Run).order_by(Run.id.desc()).limit(min(limit, 100))
        if topic_id:
            q = q.where(Run.topic_id == topic_id)
        return [run_out(r) for r in s.scalars(q)]


def event_out(e: RunEvent) -> dict:
    return dict(seq=e.seq, ts=iso(e.ts), agent=e.agent, kind=e.kind, message=e.message, status=e.status, ms=e.ms)


@app.get("/api/runs/{rid}/summary")
def get_run_summary(rid: int):
    """Plain-language results of one run, grouped by what the reader should do about each item."""
    with session() as s:
        return run_summary(s, s.get(Run, rid) or _404(), feed_item)


@app.get("/api/runs/{rid}/events")
def run_events(rid: int, after: int = 0):
    with session() as s:
        return [event_out(e) for e in s.scalars(select(RunEvent).where(RunEvent.run_id == rid, RunEvent.seq > after)
                                                .order_by(RunEvent.seq))]


# ---- console agent (NDJSON AgentEvent stream) ---------------------------------------------

class AgentIn(BaseModel):
    prompt: str = Field(max_length=2000)
    agent: str | None = None


PLAN = ["Scout: fetch sources", "Resolver: dedupe & cluster", "Analyst: compare with history",
        "Verifier: check evidence", "Sentinel: decide alerts", "Trend: update trends"]
PHASE_INDEX = {"scout": 0, "resolver": 1, "analyst": 2, "verifier": 3, "sentinel": 4, "trend": 5}


@app.post("/api/agent")
def agent(body: AgentIn):
    with session() as s:
        ts = s.scalars(select(Topic).where(Topic.active.is_(True)).order_by(Topic.id)).all()
        if not ts:
            raise HTTPException(409, "no active topic")
        words = set(body.prompt.lower().split())
        topic = max(ts, key=lambda t: (len(words & set(t.name.lower().replace("(", " ").replace(")", " ").split())),
                                       "live" in words and "live" in t.name.lower(), -t.id))
        s.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": 100000 + topic.id})
        busy = s.scalar(select(Run.id).where(Run.topic_id == topic.id, Run.status.in_(("QUEUED", "RUNNING"))))
        rid = busy or create_run(s, topic.id).id
        tid = topic.id
        created = s.get(Run, rid).created_at
        tname = topic.name

    def line(o):
        return json.dumps(o) + "\n"

    def stream():
        yield line({"type": "state", "state": "thinking"})
        yield line({"type": "plan", "steps": PLAN})
        yield line({"type": "token", "text": f"Monitoring “{tname}” (run {rid}).\n"})
        after, step, deadline = 0, -1, time.time() + 600
        while time.time() < deadline:
            with session() as s:
                run = s.get(Run, rid)
                if run is None or run.created_at != created:  # demo reset replaced this run
                    yield line({"type": "error", "message": "the data was reset while this run was streaming"})
                    yield line({"type": "final"})
                    return
                status, counters = run.status, run.counters  # read status BEFORE events so nothing is missed
                evs = s.scalars(select(RunEvent).where(RunEvent.run_id == rid, RunEvent.seq > after).order_by(RunEvent.seq)).all()
            for e in evs:
                after = e.seq
                if e.kind == "phase":
                    if step >= 0:
                        yield line({"type": "step", "index": step, "status": "done"})
                    step = PHASE_INDEX.get(e.agent, step)
                    yield line({"type": "route", "agentId": e.agent})
                    yield line({"type": "state", "state": "acting"})
                    yield line({"type": "step", "index": step, "status": "running"})
                elif e.kind == "tool":
                    yield line({"type": "tool", "id": f"e{e.seq}", "name": e.agent, "detail": e.message,
                                "status": "skipped" if e.status == "skipped" else "done", "ms": e.ms})
                elif e.kind in ("note", "summary"):
                    yield line({"type": "token", "text": e.message + "\n"})
            if status in ("SUCCEEDED", "PARTIAL_SUCCESS", "FAILED"):
                time.sleep(0.3)  # the final summary event is written just after the status commit
                with session() as s:
                    for e in s.scalars(select(RunEvent).where(RunEvent.run_id == rid, RunEvent.seq > after).order_by(RunEvent.seq)):
                        after = e.seq
                        if e.kind == "tool":
                            yield line({"type": "tool", "id": f"e{e.seq}", "name": e.agent, "detail": e.message,
                                        "status": "skipped" if e.status == "skipped" else "done", "ms": e.ms})
                        elif e.kind in ("note", "summary"):
                            yield line({"type": "token", "text": e.message + "\n"})
                if step >= 0:
                    yield line({"type": "step", "index": step, "status": "done"})
                if status == "FAILED":
                    yield line({"type": "error", "message": "run failed; see the run trace"})
                else:
                    c = counters or {}
                    yield line({"type": "token", "text": (
                        f"\n{c.get('assessed', 0)} items assessed, {c.get('alerts', 0)} alert(s), "
                        f"{c.get('suppressed', 0)} suppressed. Open the Feed to inspect the evidence.")})
                # Tells the interface which run's results to open.
                yield line({"type": "result", "runId": rid, "topicId": tid})
                yield line({"type": "state", "state": "done"})
                yield line({"type": "final"})
                return
            time.sleep(0.4)
        yield line({"type": "error", "message": "timed out waiting for the worker (is `python -m rp.worker` running?)"})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---- feed / changes / clusters --------------------------------------------------------------

@app.get("/api/feed")
def feed(topic_id: int | None = None, priority: str | None = None, classification: str | None = None,
         confidence: str | None = None, run_id: int | None = None, limit: int = 100):
    order = {"HIGH": 0, "REVIEW": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
    with session() as s:
        q = select(Assessment)
        if topic_id:
            q = q.where(Assessment.topic_id == topic_id)
        if run_id:
            q = q.where(Assessment.run_id == run_id)
        if priority:
            q = q.where(Assessment.priority.in_(priority.split(",")))
        if classification:
            q = q.where(Assessment.classification.in_(classification.split(",")))
        if confidence:
            q = q.where(Assessment.confidence.in_(confidence.split(",")))
        prio = case(order, value=Assessment.priority, else_=9)
        rows = s.scalars(q.order_by(prio, Assessment.id.desc()).limit(max(1, min(limit, 300)))).all()
        return [feed_item(s, a) for a in rows]


@app.get("/api/changes/{aid}")
def change(aid: int):
    with session() as s:
        a = s.get(Assessment, aid) or _404()
        out = feed_item(s, a)
        ev = []
        for e in s.scalars(select(AssessmentEvidence).where(AssessmentEvidence.assessment_id == a.id)):
            d = s.get(Document, e.document_id)
            v = s.get(DocumentVersion, e.version_id)
            ev.append(dict(document_id=e.document_id, version=v.version, role=e.role, excerpt=e.excerpt,
                           url=d.url, publisher=d.publisher, source_role=d.source_role, title=d.title, verified=e.verified))
        related = []
        if a.cluster_id:
            related = [dict(id=x.id, headline=x.headline, classification=x.classification, priority=x.priority)
                       for x in s.scalars(select(Assessment).where(Assessment.cluster_id == a.cluster_id, Assessment.id != a.id))]
        ver = s.get(DocumentVersion, a.version_id)
        out.update(comparisons=a.comparisons, evidence=ev, score_reasons=a.score_reasons, uncertainty=a.uncertainty,
                   evidence_gaps=a.evidence_gaps, related=related, flags=a.flags, audit=a.audit,
                   injection_excerpt=ver.injection_excerpt, changed_fields=ver.changed_fields)
        return out


@app.get("/api/clusters/{cid}")
def cluster(cid: int):
    with session() as s:
        c = s.get(Cluster, cid) or _404()
        members = cluster_members(s, cid)
        n, k, note = independence(members)
        return dict(id=c.id, title=c.title, independent_sources=k, independence_note=note,
                    members=[doc_ref(d) | dict(relation=cd.relation, reason=cd.reason, decided_by=cd.decided_by) for d, cd in members],
                    assessments=[dict(id=a.id, classification=a.classification, headline=a.headline, priority=a.priority)
                                 for a in s.scalars(select(Assessment).where(Assessment.cluster_id == cid))])


@app.get("/api/trends")
def trends(topic_id: int | None = None):
    with session() as s:
        q = select(Trend)
        if topic_id:
            q = q.where(Trend.topic_id == topic_id)
        rank = {"EMERGING": 0, "ESTABLISHED": 1, "TENTATIVE": 2, "NO_SIGNAL": 3, "INSUFFICIENT_HISTORY": 4}
        rows = sorted(s.scalars(q).all(), key=lambda t: rank.get(t.status, 9))
        return [dict(id=t.id, topic_id=t.topic_id, concept=t.concept, status=t.status, window_days=t.window_days,
                     current_count=t.current_count, baseline_count=t.baseline_count, baseline_windows=t.baseline_windows,
                     article_count=t.article_count, summary=t.summary, limitations=t.limitations, series=t.series,
                     evidence=t.evidence, contradicting=t.contradicting, updated_at=iso(t.updated_at)) for t in rows]


# ---- sources -------------------------------------------------------------------------------

class SourceIn(BaseModel):
    kind: str = Field(pattern="^(arxiv|openalex|crossref|rss)$")
    name: str = Field(max_length=200)
    url: str | None = Field(default=None, max_length=2000)
    query: str | None = Field(default=None, max_length=300)
    role_default: str = Field(default="SECONDARY", pattern="^(PRIMARY|SECONDARY|UNVERIFIED|UNKNOWN)$")


@app.get("/api/sources")
def sources():
    with session() as s:
        return [source_out(x) for x in s.scalars(select(Source).order_by(Source.id))]


@app.post("/api/sources")
def add_source(body: SourceIn):
    if body.kind == "rss":
        try:
            assert_public_url(body.url or "")
        except UnsafeURL as exc:
            raise HTTPException(422, str(exc))
    with session() as s:
        x = Source(kind=body.kind, name=body.name, url=body.url, query=body.query, role_default=body.role_default)
        s.add(x)
        s.flush()
        return source_out(x)


class SourcePatch(BaseModel):
    enabled: bool


@app.patch("/api/sources/{sid}")
def patch_source(sid: int, body: SourcePatch):
    with session() as s:
        x = s.get(Source, sid) or _404()
        x.enabled = body.enabled
        return source_out(x)


# ---- alerts / feedback / demo ----------------------------------------------------------------

@app.get("/api/alerts")
def alerts(unread: bool = False):
    with session() as s:
        q = select(Alert).order_by(Alert.id.desc())
        if unread:
            q = q.where(Alert.read.is_(False))
        return [dict(id=a.id, assessment_id=a.assessment_id, topic_id=a.topic_id, priority=a.priority, headline=a.headline,
                     created_at=iso(a.created_at), read=a.read, suppressed_repeats=a.suppressed_repeats) for a in s.scalars(q)]


@app.post("/api/alerts/{aid}/read")
def read_alert(aid: int):
    with session() as s:
        a = s.get(Alert, aid) or _404()
        a.read = True
        s.get(Assessment, a.assessment_id).read = True
        return {"ok": True}


class FeedbackIn(BaseModel):
    assessment_id: int
    verdict: str = Field(pattern="^(useful|not_useful|wrong_cluster|wrong_classification)$")
    note: str = Field(default="", max_length=1000)


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    with session() as s:
        s.get(Assessment, body.assessment_id) or _404()
        s.add(Feedback(assessment_id=body.assessment_id, verdict=body.verdict, note=body.note))
        return {"ok": True}


@app.post("/api/reset")
@app.post("/api/demo/reset")  # old path, kept for scripts
def demo_reset():
    return seed.reset()


@app.get("/api/health")
def health():
    with session() as s:
        s.execute(select(1))
    return {"ok": True, "llm": llm.provider()}


def _404():
    raise HTTPException(404, "not found")
