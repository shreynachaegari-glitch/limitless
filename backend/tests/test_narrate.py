"""Bounded analyst agent with a scripted fake model: tool loop, budget, and the grounding gate."""

from types import SimpleNamespace as NS

from sqlalchemy import select

from rp import llm, narrate
from rp.analyze import Draft
from rp.db import Assessment, Document, DocumentVersion, session


def _tool_call(name, args):
    return NS(content="", tool_calls=[NS(id="t1", function=NS(name=name, arguments=args))])


def _answer(text):
    return NS(content=text, tool_calls=None)


def _run(monkeypatch, replies):
    seq = iter(replies)
    monkeypatch.setattr(llm, "client", lambda: object())
    monkeypatch.setattr(llm, "chat", lambda *a, **k: next(seq))
    with session() as s:
        a = s.scalars(select(Assessment).where(Assessment.classification == "MATERIAL_REVISION")).first()
        doc, ver = s.get(Document, a.document_id), s.get(DocumentVersion, a.version_id)
        d = Draft(a.classification, headline=a.headline, what_changed=a.what_changed, why_it_matters=a.why_it_matters,
                  comparisons=a.comparisons)
        audit = narrate.narrate(s, d, doc, ver, a.cluster_id, "3 reports, based on 1 original source", {doc.id})
    return d, audit


def test_grounded_llm_answer_is_used(replay, monkeypatch):
    d, audit = _run(monkeypatch, [
        _tool_call("get_history", '{"entity": "Nova-7B", "metric": "speedup"}'),
        _answer('{"headline": "NovaQuant authors cut their Nova-7B speedup from 2.4x to 1.9x", '
                '"what_changed": "Version 3 reports 1.9x decode speedup on A100; v1 said 2.4x.", '
                '"why_it_matters": "Outlets that repeated 2.4x are now out of date."}')])
    assert audit["narrator"] == "llm" and audit["tool_calls"] == 1
    assert d.headline.startswith("NovaQuant authors")


def test_ungrounded_number_falls_back_to_template(replay, monkeypatch):
    d, audit = _run(monkeypatch, [_answer(
        '{"headline": "Speedup collapses by 60%", "what_changed": "Now 0.9x.", "why_it_matters": "Big deal."}')])
    assert audit["narrator"] == "template" and "ungrounded" in audit["validation"]
    assert "revised from 2.4x to 1.9x" in d.headline
