"""Run results summary, the result event in the console stream, and model-assisted claim extraction."""

import json
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from rp import llm, llm_claims

ABSTRACT = ("We present FlashSpec, an adaptive speculative decoding method. On Llama-3-8B, FlashSpec achieves "
            "a 2.7x end-to-end speedup on an A100 GPU while keeping outputs identical. Memory overhead stays "
            "below 4% of the KV cache.")


def test_validator_keeps_only_verbatim_quotes_with_their_number():
    good = dict(entity="Llama-3-8B", metric="end-to-end speedup", value=2.7, unit="x", context="A100",
                method="FlashSpec", quote="FlashSpec achieves a 2.7x end-to-end speedup on an A100 GPU")
    row = llm_claims.validate(ABSTRACT, good)
    assert row and ABSTRACT[row["span_start"]:row["span_end"]] == good["quote"]
    assert row["conditions"]["method"] == "flashspec"
    assert llm_claims.validate(ABSTRACT, good | {"quote": "FlashSpec achieves a 3.1x speedup"}) is None  # not verbatim
    assert llm_claims.validate(ABSTRACT, good | {"value": 3.1}) is None  # number not in the quote
    assert llm_claims.validate(ABSTRACT, good | {"entity": ""}) is None


def test_extract_batch_with_scripted_model(monkeypatch):
    reply = {"papers": [{"id": "7", "claims": [
        {"entity": "Llama-3-8B", "metric": "speedup", "value": 2.7, "unit": "x", "context": "A100", "method": "FlashSpec",
         "quote": "FlashSpec achieves a 2.7x end-to-end speedup on an A100 GPU"},
        {"entity": "Llama-3-8B", "metric": "speedup", "value": 9.0, "unit": "x", "context": "", "method": "",
         "quote": "achieves a 9x speedup"}]}]}
    monkeypatch.setattr(llm, "client", lambda: object())
    monkeypatch.setattr(llm, "chat", lambda *a, **k: NS(content=json.dumps(reply)))
    out, st = llm_claims.extract_batch([("7", "FlashSpec", ABSTRACT)])
    assert st == {"calls": 1, "proposed": 2, "kept": 1, "rejected": 1}
    assert out["7"][0]["value"] == 2.7


def test_run_summary_groups_results(replay):
    from rp.api import app
    with TestClient(app) as c:
        contra_run = replay["by_key"]["reeval-nova"][0]["run"]
        r = c.get(f"/api/runs/{contra_run}/summary").json()
        assert r["headline"].endswith("need your attention")
        groups = {g["key"]: g for g in r["groups"]}
        attention = {i["classification"] for i in groups["attention"]["items"]}
        assert {"CONTRADICTION", "INSUFFICIENT_EVIDENCE"} <= attention
        assert all(i["kind_explain"] and i["priority_explain"] for g in r["groups"] for i in g["items"])
        assert groups["quiet"]["count"] == 2  # the two duplicate news reports
        assert any("grouped" in f for f in r["facts"])
        first = c.get(f"/api/runs/{replay['runs'][0]}/summary").json()
        assert any("First run" in f for f in first["facts"])
        assert first["ignored"] and "exclusion" in first["ignored"][0]["reason"]  # the off-topic diffusion paper


def test_feed_filters_by_run(replay):
    from rp.api import app
    with TestClient(app) as c:
        rid = replay["by_key"]["novaquant"][2]["run"]  # the run with only the v3 revision
        items = c.get(f"/api/feed?run_id={rid}").json()
        assert [i["classification"] for i in items] == ["MATERIAL_REVISION"]
