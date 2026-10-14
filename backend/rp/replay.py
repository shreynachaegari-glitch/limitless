"""Run the demo topic's fixture batches synchronously and print what happened (dev aid)."""

import sys

from sqlalchemy import select

from .db import Alert, Assessment, Run, Trend, session
from .pipeline import create_run, execute
from .seed import reset

if __name__ == "__main__":
    ids = reset()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for i in range(n):
        with session() as s:
            run = create_run(s, ids["topic_id"])
        execute(run.id)
        with session() as s:
            r = s.get(Run, run.id)
            print(f"\n=== run {i+1}: {r.status} {r.counters}")
            for a in s.scalars(select(Assessment).where(Assessment.run_id == run.id).order_by(Assessment.id)):
                print(f"  [{a.priority:6}] {a.classification:22} {a.confidence:6} {a.headline[:80]}")
                print(f"           {a.what_changed[:150]}")
    with session() as s:
        print("\nALERTS")
        for al in s.scalars(select(Alert)):
            print(f"  {al.priority} {al.headline[:80]} (suppressed {al.suppressed_repeats})")
        print("TRENDS")
        for t in s.scalars(select(Trend)):
            print(f"  {t.concept}: {t.status} — {t.summary}")
