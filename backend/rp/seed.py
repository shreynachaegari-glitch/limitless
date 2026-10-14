"""Reset the database and seed the demo topic (fixtures) and the live topic (arXiv + OpenAlex)."""

from __future__ import annotations

from . import fixtures
from .db import Source, Topic, init_db, session


RUN_LOCK = 4242  # advisory lock: runs hold it shared, reset holds it exclusive


def reset(drop: bool = False) -> dict:
    """Cancel queued jobs, wait for an in-flight run to finish (advisory lock), then truncate everything."""
    from sqlalchemy import text
    from .db import Base, engine
    init_db(drop=False)
    with engine.connect() as c:
        c.execute(text("UPDATE jobs SET status = 'cancelled' WHERE status = 'queued'"))
        c.commit()
        c.execute(text("SET lock_timeout = '120s'"))
        c.execute(text("SELECT pg_advisory_lock(:k)"), {"k": RUN_LOCK})
        c.commit()
        try:
            if drop:
                Base.metadata.drop_all(c)
                c.commit()
            else:
                names = ", ".join(t.name for t in Base.metadata.sorted_tables)
                c.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
                c.commit()
        finally:
            c.rollback()  # an aborted TRUNCATE must not leave the session lock held on a pooled connection
            c.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": RUN_LOCK})
            c.execute(text("RESET lock_timeout"))
            c.commit()
    init_db(drop=False)
    with session() as s:
        fx = Source(kind="fixture", name="Scenario replay", role_default="UNKNOWN")
        ax = Source(kind="arxiv", name="arXiv API", role_default="PRIMARY", url="https://export.arxiv.org/api/query")
        oa = Source(kind="openalex", name="OpenAlex", role_default="PRIMARY", url="https://api.openalex.org/works")
        cr = Source(kind="crossref", name="Crossref: Springer, IEEE, ACM, Elsevier", role_default="PRIMARY",
                    url="https://api.crossref.org/works")
        s.add_all([fx, ax, oa, cr])
        s.flush()
        demo = Topic(**fixtures.DEMO_TOPIC, source_ids=[fx.id])
        live = Topic(**fixtures.LIVE_TOPIC, source_ids=[ax.id, oa.id, cr.id])
        s.add_all([demo, live])
        s.flush()
        return {"ok": True, "topic_id": demo.id, "live_topic_id": live.id}


if __name__ == "__main__":
    print(reset())
