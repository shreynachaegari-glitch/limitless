"""One monitoring run: Scout → Resolver → Analyst → Verifier → Sentinel → Trend.

Deterministic code owns fetching, identity, persistence, scoring and alert policy; the model only
narrates (and adjudicates ambiguous duplicates). Each phase commits, so no transaction is held open
across network or model calls, and each step writes a run_event that the console streams.
"""

from __future__ import annotations

import time
import traceback

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from . import analyze, claims as claims_mod, config, embed, llm, narrate, resolve, trends
from .db import (Alert, Assessment, AssessmentEvidence, Claim, ClusterDocument, Cluster, Document, DocumentVersion,
                 Run, RunEvent, Source, Topic, TopicDocument, SessionLocal, utcnow)
from .ingest import ingest
from .sources import FETCHERS, SourceError
from .verify import verify_evidence


class Emitter:
    def __init__(self, run_id: int):
        self.run_id, self.seq = run_id, 0

    def __call__(self, agent: str, kind: str, message: str, status: str = "done", ms: int | None = None) -> None:
        self.seq += 1
        with SessionLocal() as es:  # separate session: events are visible while the run is still working
            es.add(RunEvent(run_id=self.run_id, seq=self.seq, agent=agent, kind=kind, message=message, status=status, ms=ms))
            es.commit()


def _phase(s: Session, run: Run, emit: Emitter, agent: str, label: str) -> None:
    run.phase = agent
    s.commit()
    emit(agent, "phase", label, "running")


def execute(run_id: int) -> None:
    from sqlalchemy import text
    from .db import engine
    from .seed import RUN_LOCK
    emit = Emitter(run_id)
    lock = engine.connect()  # holds the shared run lock so a demo reset waits for this run to finish
    lock.execute(text("SELECT pg_advisory_lock_shared(:k)"), {"k": RUN_LOCK})
    lock.commit()
    s = SessionLocal()
    try:
        if s.get(Run, run_id) is None:  # reset happened while queued
            return
        _execute(s, run_id, emit)
    except Exception as exc:
        traceback.print_exc()
        s.rollback()
        try:
            run = s.get(Run, run_id)
            if run:
                run.status, run.error, run.finished_at = "FAILED", f"{type(exc).__name__}: {exc}"[:500], utcnow()
                s.commit()
                emit("sentinel", "note", f"run failed: {type(exc).__name__}: {str(exc)[:200]}", "error")
        except Exception:
            s.rollback()
    finally:
        s.close()
        lock.execute(text("SELECT pg_advisory_unlock_shared(:k)"), {"k": RUN_LOCK})
        lock.commit()
        lock.close()


def _execute(s: Session, run_id: int, emit: Emitter) -> None:
    run = s.get(Run, run_id)
    topic = s.get(Topic, run.topic_id)
    run.status, run.started_at = "RUNNING", utcnow()
    c = dict(fetched=0, new_documents=0, new_versions=0, unchanged=0, irrelevant=0, assessed=0, duplicates_grouped=0,
             alerts=0, suppressed=0, sources_ok=0, sources_failed=0, llm_narrations=0, verification_failures=0)
    backfill = not s.scalar(select(func.count(Run.id)).where(Run.topic_id == topic.id, Run.id != run.id,
                                                            Run.status.in_(("SUCCEEDED", "PARTIAL_SUCCESS"))))
    s.commit()

    # ---- Scout ---------------------------------------------------------------------------
    _phase(s, run, emit, "scout", "Scout: fetching sources")
    results = []
    for sid in topic.source_ids or []:
        src = s.get(Source, sid)
        if not src or not src.enabled:
            continue
        t0 = time.time()
        src.last_checked = utcnow()
        try:
            items, cursor = FETCHERS[src.kind](src, topic)
        except (SourceError, Exception) as exc:  # one failing source never sinks the run
            src.last_error, src.consecutive_failures = str(exc)[:300], src.consecutive_failures + 1
            c["sources_failed"] += 1
            s.commit()
            emit("scout", "tool", f"{src.name}: fetch FAILED ({str(exc)[:120]})", "skipped", int((time.time() - t0) * 1000))
            continue
        src.cursor, src.last_success, src.last_error = cursor, utcnow(), None
        src.items_last_run, src.consecutive_failures = len(items), 0
        c["sources_ok"] += 1
        c["fetched"] += len(items)
        fresh = 0
        for it in sorted(items, key=lambda i: i.published_at or utcnow()):
            r = ingest(s, src.id, it, run.id)
            if r is None:
                c["unchanged"] += 1
                continue
            td = s.scalar(select(TopicDocument).where(TopicDocument.topic_id == topic.id,
                                                      TopicDocument.document_id == r.document.id))
            done = s.scalar(select(Assessment.id).where(Assessment.topic_id == topic.id, Assessment.version_id == r.version.id))
            if done or (td is not None and td.version_id == r.version.id):
                c["unchanged"] += 1  # this topic already judged this exact version
                continue
            if td is None or not td.relevant:
                # New to this topic (another topic may have stored it, or an earlier version was off-topic).
                r.previous, r.is_new_document = None, True
            fresh += 1
            c["new_documents" if r.is_new_document else "new_versions"] += 1
            results.append(r)
        s.commit()
        emit("scout", "tool", f"{src.name}: {len(items)} items, {fresh} new or changed, {len(items) - fresh} unchanged",
             "done", int((time.time() - t0) * 1000))
    if not results:
        emit("scout", "note", "nothing new since the last run" + (" (and a source failed)" if c["sources_failed"] else ""))

    # ---- Embed + relevance (Resolver) -------------------------------------------------------
    _phase(s, run, emit, "resolver", "Resolver: relevance, duplicates and clusters")
    if topic.embedding is None:
        topic.embedding = resolve.topic_embedding(topic)
    need = [r for r in results if r.version.embedding is None or r.version.embed_model != embed.model_name()]
    if need:
        t0 = time.time()
        vecs = embed.embed([embed.doc_text(r.version.title, r.version.text) for r in need])
        for r, v in zip(need, vecs):
            r.version.embedding, r.version.embed_model = v, embed.model_name()
        s.flush()
        emit("resolver", "tool", f"embedded {len(need)} versions with {embed.model_name()}", "done", int((time.time() - t0) * 1000))
    work = []
    for r in results:
        score, ok, why, hits = resolve.relevance(topic, r.document, r.version, s)
        td = s.scalar(select(TopicDocument).where(TopicDocument.topic_id == topic.id, TopicDocument.document_id == r.document.id))
        if td is None:
            td = TopicDocument(topic_id=topic.id, document_id=r.document.id, relevance=score, relevant=ok, reason=why,
                               concepts=hits, version_id=r.version.id)
            s.add(td)
        else:
            td.relevance, td.relevant, td.reason, td.concepts, td.version_id = score, ok, why, hits, r.version.id
        if not ok:
            c["irrelevant"] += 1
            emit("resolver", "tool", f"not relevant ({score:.2f}): {r.document.title[:70]} — {why}", "skipped")
            continue
        rows = _claims_for(s, r.document, r.version)
        if r.previous is not None:
            _claims_for(s, r.document, r.previous)  # the old version may never have been extracted
        work.append((r, rows, score, hits))
    s.commit()

    adjudicate = narrate.adjudicate_factory(s)
    resolutions = {}
    for r, rows, _, _ in work:
        if r.is_new_document:
            res = resolve.resolve(s, topic, r.document, r.version, rows, adjudicate)
            resolve.attach(s, r.document, res)
            if res["relation"] == "duplicate_report":
                c["duplicates_grouped"] += 1
            res["cluster_title"] = s.get(Cluster, res["cluster_id"]).title
            emit("resolver", "tool", (f"grouped “{r.document.title[:60]}” with an existing development ({res['reason'][:90]})"
                                      if not res.get("new_cluster") else f"new development: “{r.document.title[:60]}”"
                                      + (f" — {res['reason'].split('; ', 1)[1][:110]}" if '; ' in res['reason'] else "")))
        else:
            cid = resolve._cluster_of(s, r.document.id, topic.id)
            if cid is None:  # known to the topic but never clustered (e.g. a crash mid-run): cluster it now
                cid = resolve.resolve(s, topic, r.document, r.version, rows, None)["cluster_id"]
                resolve.attach(s, r.document, dict(cluster_id=cid, relation="origin", reason="clustered on revision",
                                                   decided_by="rule"))
            res = dict(cluster_id=cid, relation="updates", reason="new version of a known work", cluster_title=None)
            emit("resolver", "tool", f"new version v{r.version.version} of “{r.document.title[:60]}”")
        resolutions[r.version.id] = res
    s.commit()

    # ---- Analyst + Verifier ----------------------------------------------------------------
    _phase(s, run, emit, "analyst", "Analyst: comparing with stored history")
    topic_doc_ids = set(s.scalars(select(TopicDocument.document_id).where(TopicDocument.topic_id == topic.id)))
    drafts = []
    for r, rows, rel, hits in work:
        res = resolutions[r.version.id]
        members = resolve.cluster_members(s, res["cluster_id"]) if res["cluster_id"] else []
        n_reports, independent, note = resolve.independence(members)
        d = analyze.classify(s, topic, r.document, r.version, r.previous, rows, res, n_reports)
        drafts.append((r, d, rel, hits, res, n_reports, independent, note))
        emit("analyst", "tool", f"{d.classification.replace('_', ' ').lower()}: {d.headline[:90]}")
    s.commit()

    _phase(s, run, emit, "verifier", "Verifier: checking evidence and source reliability")
    for r, d, rel, hits, res, n_reports, independent, note in drafts:
        ok, gaps = verify_evidence(s, d.evidence)
        if not ok:
            d.flags.append("verification_failed")
            d.evidence_gaps += gaps
            c["verification_failures"] += 1
        analyze.score(d, topic, r.document, r.version, rel, n_reports, independent, hits)
        analyze.decide_priority(d, topic)
        if r.version.injection_flagged:
            emit("verifier", "tool", f"prompt injection quarantined in “{r.document.title[:50]}”: "
                                     f"“{(r.version.injection_excerpt or '')[:80]}” — ignored, weight reduced", "skipped")
        emit("verifier", "tool", f"{d.priority} · confidence {d.confidence} · {r.document.source_role.lower()} · "
                                 f"{sum(1 for e in d.evidence if e.get('verified'))}/{len(d.evidence)} excerpts verified — {d.headline[:60]}")
    s.commit()

    # LLM narration (bounded agent) for anything that will be surfaced.
    budget = config.LLM_MAX_NARRATIONS_PER_RUN
    rank = {"HIGH": 0, "REVIEW": 1, "MEDIUM": 2, "LOW": 3, "NONE": 9}
    for r, d, rel, hits, res, n_reports, independent, note in sorted(drafts, key=lambda x: rank.get(x[1].priority, 9)):
        audit = {"narrator": "template", "tool_calls": 0, "validation": "template"}
        if d.priority != "NONE" and budget > 0 and llm.client() is not None:
            budget -= 1
            t0 = time.time()
            try:
                audit = narrate.narrate(s, d, r.document, r.version, res["cluster_id"], note, topic_doc_ids, emit)
            except Exception as exc:  # narration is optional; never lose the assessment over it
                s.rollback()
                audit = {"narrator": "template", "tool_calls": 0, "validation": f"narrator error: {type(exc).__name__}"}
            emit("analyst", "tool", f"narration {audit['narrator']} ({audit['validation']})", "done", int((time.time() - t0) * 1000))
            if audit["narrator"] == "llm":
                c["llm_narrations"] += 1
        d._audit = audit

    # ---- Persist + Sentinel ------------------------------------------------------------------
    _phase(s, run, emit, "sentinel", "Sentinel: deciding what deserves an alert" + (" (first run: history backfill, alerts off)" if backfill else ""))
    for r, d, rel, hits, res, n_reports, independent, note in drafts:
        key = analyze.idempotency_key(topic.id, r.version.id)
        if s.scalar(select(Assessment.id).where(Assessment.idempotency_key == key)):
            continue
        a = Assessment(
            idempotency_key=key, topic_id=topic.id, cluster_id=res["cluster_id"], document_id=r.document.id,
            version_id=r.version.id, run_id=run.id, classification=d.classification, priority=d.priority,
            confidence=d.confidence, headline=d.headline, what_changed=d.what_changed, why_it_matters=d.why_it_matters,
            narrator=d._audit["narrator"], scores=d.scores, score_reasons=d.score_reasons, comparisons=d.comparisons,
            uncertainty=d.uncertainty, evidence_gaps=d.evidence_gaps, flags=d.flags,
            audit={"policy_version": config.POLICY_VERSION, "extractor": config.EXTRACTOR_VERSION,
                   "model": d._audit["narrator"] == "llm" and config.NVIDIA_MODEL or "template",
                   "embedding_model": r.version.embed_model, "tool_calls": d._audit["tool_calls"],
                   "validation": d._audit["validation"], "resolution": res.get("reason"),
                   "resolution_by": res.get("decided_by"), "independence": note, "backfill": backfill},
        )
        s.add(a)
        s.flush()
        for e in d.evidence:
            s.add(AssessmentEvidence(assessment_id=a.id, document_id=e["document_id"], version_id=e["version_id"],
                                     claim_id=e.get("claim_id"), role=e["role"], excerpt=e["excerpt"],
                                     verified=bool(e.get("verified"))))
        c["assessed"] += 1
        if d.classification == "DUPLICATE_REPORT":
            # Repeat coverage of a development: bump the existing alert instead of sending another.
            s.execute(update(Alert).where(Alert.cluster_id == res["cluster_id"]).values(suppressed_repeats=Alert.suppressed_repeats + 1))
            c["suppressed"] += 1
            continue
        if backfill or not analyze.should_alert(d.priority, topic):
            c["suppressed"] += 1
            continue
        akey = analyze.alert_key_for(topic.id, res["cluster_id"], r.document.id, d)
        existing = s.scalar(select(Alert).where(Alert.idempotency_key == akey))
        if existing:
            existing.suppressed_repeats += 1
            c["suppressed"] += 1
            continue
        s.add(Alert(idempotency_key=akey, topic_id=topic.id, assessment_id=a.id, cluster_id=res["cluster_id"],
                    priority=d.priority, headline=d.headline))
        c["alerts"] += 1
        emit("sentinel", "tool", f"ALERT {d.priority}: {d.headline[:90]}")
    s.commit()
    emit("sentinel", "tool", f"{c['alerts']} alert(s) sent, {c['suppressed']} suppressed (duplicates, minor edits, below threshold"
                             + (", backfill" if backfill else "") + ")")

    # ---- Trend -------------------------------------------------------------------------------
    _phase(s, run, emit, "trend", "Trend: counting distinct developments vs baseline")
    for t in trends.recompute(s, topic):
        emit("trend", "tool", f"{t.concept}: {t.status.replace('_', ' ').lower()} — {t.summary[:110]}")
    s.commit()

    run.counters = c
    run.status = "SUCCEEDED" if not c["sources_failed"] else ("FAILED" if not c["sources_ok"] else "PARTIAL_SUCCESS")
    run.finished_at, run.phase = utcnow(), "done"
    topic.last_run_at = run.finished_at
    s.commit()
    emit("sentinel", "summary", f"run {run.status.lower()}: {c['fetched']} fetched, {c['new_documents']} new, "
                                f"{c['new_versions']} revised, {c['unchanged']} unchanged, {c['assessed']} assessed, "
                                f"{c['alerts']} alerts")


def _claims_for(s: Session, doc: Document, ver: DocumentVersion) -> list[Claim]:
    """Claims are per version and shared by topics: extract once, reuse afterwards."""
    rows = s.scalars(select(Claim).where(Claim.version_id == ver.id)).all()
    if rows:
        return list(rows)
    rows = [Claim(document_id=doc.id, version_id=ver.id, **{k: f[k] for k in (
        "entity", "metric", "metric_key", "value", "unit", "context", "conditions", "text", "span_start", "span_end")})
        for f in claims_mod.extract(ver.text, ver.title)]
    s.add_all(rows)
    s.flush()
    return rows


def create_run(s: Session, topic_id: int, kind: str = "manual") -> Run:
    from .db import Job
    run = Run(topic_id=topic_id, status="QUEUED")
    s.add(run)
    s.flush()
    s.add(Job(topic_id=topic_id, run_id=run.id, kind=kind))
    s.commit()
    return run
