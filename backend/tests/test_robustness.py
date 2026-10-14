"""Crash recovery, qualifiers/negations, and normalised trend rates."""

from sqlalchemy import func, select

from rp import analyze, claims, fixtures
from rp.db import Alert, Assessment, Run, Source, Topic, Trend, session
from rp.pipeline import create_run, execute
from rp.verify import qualifiers_kept


def _topic_on_batch(name: str, batch: int) -> int:
    with session() as s:
        fx = s.scalar(select(Source).where(Source.kind == "fixture"))
        fx.cursor = {"batch": batch, "anchor": fx.cursor["anchor"]}
        t = Topic(name=name, description=fixtures.DEMO_TOPIC["description"], concepts=fixtures.DEMO_TOPIC["concepts"],
                  exclusions=fixtures.DEMO_TOPIC["exclusions"], source_ids=[fx.id], relevance_threshold=0.45,
                  alert_threshold="HIGH")
        s.add(t)
        s.flush()
        return t.id


def test_crash_mid_run_then_retry_loses_nothing_and_duplicates_nothing(replay, monkeypatch):
    tid = _topic_on_batch("Crash recovery", 1)
    real = analyze.score
    monkeypatch.setattr(analyze, "score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("worker killed")))
    with session() as s:
        run = create_run(s, tid)
    execute(run.id)
    with session() as s:
        assert s.get(Run, run.id).status == "FAILED"
        assert s.scalar(select(func.count(Assessment.id)).where(Assessment.topic_id == tid)) == 0
    monkeypatch.setattr(analyze, "score", real)
    with session() as s:
        fx = s.scalar(select(Source).where(Source.kind == "fixture"))
        fx.cursor = {**fx.cursor, "batch": 1}  # new dict: in-place edits of a JSON column are not saved
        retry = create_run(s, tid)
    execute(retry.id)
    with session() as s:
        from rp.db import TopicDocument
        n = s.scalar(select(func.count(Assessment.id)).where(Assessment.topic_id == tid))
        relevant = s.scalar(select(func.count(TopicDocument.id)).where(TopicDocument.topic_id == tid, TopicDocument.relevant.is_(True)))
        alerts = s.scalar(select(func.count(Alert.id)).where(Alert.topic_id == tid))
        # every item the crashed run had judged relevant is assessed exactly once by the retry
        assert s.get(Run, retry.id).status == "SUCCEEDED" and relevant >= 8 and n == relevant
        fx = s.scalar(select(Source).where(Source.kind == "fixture"))
        fx.cursor = {**fx.cursor, "batch": 1}  # new dict: in-place edits of a JSON column are not saved
        again = create_run(s, tid)
    execute(again.id)
    with session() as s:
        assert s.scalar(select(func.count(Assessment.id)).where(Assessment.topic_id == tid)) == n
        assert s.scalar(select(func.count(Alert.id)).where(Alert.topic_id == tid)) == alerts


def test_qualifier_and_negation_are_captured_and_block_comparison():
    up = claims.extract("Nova-7B with NovaQuant achieves up to 2.4x decode speedup on A100.")[0]
    plain = claims.extract("Nova-7B with NovaQuant achieves a 1.9x decode speedup on A100.")[0]
    neg = claims.extract("Kite-7B does not achieve a 1.2x decode speedup on A100.")[0]
    assert up["conditions"]["qualifier"] == "up to" and neg["conditions"]["negated"] == "yes"
    ok, note = claims.comparability(up, plain, same_work=False)
    assert not ok and "qualifier" in note  # a best case is not compared with a typical result


def test_narration_that_drops_a_qualifier_is_rejected():
    assert qualifiers_kept("HyperServe is 10x faster", ["makes inference up to 10x faster"]) == (False, ["up to 10"])
    assert qualifiers_kept("up to 10x faster on one benchmark", ["up to 10x faster"])[0]


def test_trend_rates_are_compared_per_equal_window(replay):
    with session() as s:
        t = s.scalar(select(Trend).where(Trend.topic_id == replay["ids"]["topic_id"], Trend.concept == "speculative decoding"))
    assert t.status == "EMERGING"
    assert "per 14-day window" in t.summary and "12x the earlier rate" in t.summary and "independent sources" in t.summary
