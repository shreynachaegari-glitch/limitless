"""Worker process: claims queued jobs with FOR UPDATE SKIP LOCKED + a lease (a crashed worker's job is
reclaimed when the lease expires), and a scheduler that enqueues topics whose poll interval is due.
Run: python -m rp.worker"""

from __future__ import annotations

import time
from datetime import timedelta

from sqlalchemy import select, text

from .db import Job, Run, Topic, init_db, session, utcnow
from .pipeline import create_run, execute

LEASE = timedelta(minutes=10)
MAX_ATTEMPTS = 3


def claim_job() -> Job | None:
    with session() as s:
        row = s.execute(text(
            "SELECT id FROM jobs WHERE (status = 'queued' OR (status = 'running' AND lease_until < now())) "
            "AND attempts < :max ORDER BY (kind = 'scheduled'), id FOR UPDATE SKIP LOCKED LIMIT 1"), {"max": MAX_ATTEMPTS}).first()
        if not row:
            return None
        job = s.get(Job, row[0])
        job.status, job.lease_until, job.attempts = "running", utcnow() + LEASE, job.attempts + 1
        return job


def schedule_due() -> None:
    with session() as s:
        for t in s.scalars(select(Topic).where(Topic.active.is_(True), Topic.poll_minutes > 0)):
            due = t.last_run_at is None or t.last_run_at + timedelta(minutes=t.poll_minutes) <= utcnow()
            busy = s.scalar(select(Job.id).where(Job.topic_id == t.id, Job.status.in_(("queued", "running"))))
            if due and not busy:
                create_run(s, t.id, kind="scheduled")


def main() -> None:
    init_db()
    print("worker: waiting for jobs")
    last_sched = 0.0
    while True:
        try:
            if time.time() - last_sched > 30:
                last_sched = time.time()
                schedule_due()
            job = claim_job()
        except Exception as exc:  # database restarting, schema reset in progress
            print(f"worker: poll failed ({type(exc).__name__}: {exc}); retrying", flush=True)
            time.sleep(3)
            continue
        if job is None:
            time.sleep(1.0)
            continue
        print(f"worker: run {job.run_id} (topic {job.topic_id})", flush=True)
        try:
            execute(job.run_id)
            with session() as s:
                j = s.get(Job, job.id)
                r = s.get(Run, job.run_id)
                if j and r:
                    j.status = "failed" if r.status == "FAILED" else "done"
        except Exception as exc:  # e.g. a demo reset dropped the tables mid-run; keep serving
            print(f"worker: job {job.id} aborted: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
