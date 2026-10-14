import os

os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", "postgresql+psycopg://rp:rp@127.0.0.1:5433/researchpulse_test")
os.environ["OFFLINE"] = "1"  # deterministic: template narrator, no network model calls

import pytest  # noqa: E402
from sqlalchemy import select  # noqa: E402


@pytest.fixture(scope="module")
def replay():
    """Seed and replay the five fixture runs once; return a lookup of results by fixture key."""
    from rp.db import Assessment, Document, session
    from rp.pipeline import create_run, execute
    from rp.seed import reset

    ids = reset(drop=True)
    runs = []
    for _ in range(5):
        with session() as s:
            run = create_run(s, ids["topic_id"])
        execute(run.id)
        runs.append(run.id)
    with session() as s:
        by_key = {}
        for d in s.scalars(select(Document)):
            by_key[d.external_id] = [
                dict(cls=a.classification, prio=a.priority, conf=a.confidence, flags=a.flags, cluster=a.cluster_id,
                     comps=a.comparisons, run=a.run_id, id=a.id)
                for a in s.scalars(select(Assessment).where(Assessment.document_id == d.id).order_by(Assessment.id))]
    return dict(ids=ids, runs=runs, by_key=by_key)
