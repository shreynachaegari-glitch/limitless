"""Deterministic claim extraction: entity · metric · value · unit · context · conditions, with the exact
character span it came from (so evidence can be verified against stored text).

Comparability is the important half. Two numbers are only compared when entity, metric, unit and
benchmark/hardware match and the conditions (method, batch size, context length) are the same.
"Different conditions" explains a disagreement away; it is not a contradiction.
"""

from __future__ import annotations

import re

ENTITY = re.compile(r"\b[A-Z][A-Za-z]*(?:-[A-Za-z]+)*-\d+(?:\.\d+)?[BMK]\b")
CLAIM = re.compile(
    r"\b(?P<verb>loses|drops|gains|achieves|reaches|reports|improves|delivers|shows)\s+(?:a\s+|an\s+)?"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>x|×|%|points?|pp|ms|tokens/s)\s+"
    r"(?P<metric>[a-z][a-z\-]*(?:\s[a-z][a-z\-]*){0,2}?)\s+(?:on|for)\s+(?P<context>[A-Z][A-Za-z0-9\-\.]*[A-Za-z0-9])"
)
METHOD_WITH = re.compile(r"\bwith\s+(?:a\s+|an\s+|the\s+)?(?P<m>[A-Za-z0-9][\w\-]*(?:\s[a-z][\w\-]*){0,2}?)(?=\s+(?:achieves|loses|reaches|gains|delivers|improves)\b)")
METHOD_QUANT = re.compile(r"\bquantized\s+to\s+(?P<m>\d+-bit)\b")
BATCH = re.compile(r"\bat\s+batch\s+size\s+(?P<b>\d+)\b")
CTXLEN = re.compile(r"\bat\s+(?P<c>\d+k)\s+context\b")
SENT = re.compile(r".+?(?:[.!?]+(?=\s|$)|$)", re.S)


def _unit(u: str) -> str:
    u = u.lower()
    return {"×": "x", "point": "points", "pp": "points"}.get(u, u)


def _metric_key(metric: str, verb: str) -> str:
    m = re.sub(r"\s+", " ", metric.strip().lower())
    if verb in ("loses", "drops"):
        m = f"{m} loss"
    return re.sub(r"[^a-z0-9]+", "_", m).strip("_")


def extract(text: str, title: str = "") -> list[dict]:
    out: list[dict] = []
    title_entity = (ENTITY.search(title or "") or [None])[0] if title else None
    for sm in SENT.finditer(text or ""):
        sent, base = sm.group(0), sm.start()
        matches = list(CLAIM.finditer(sent))
        prev_end = 0
        for i, m in enumerate(matches):
            window = sent[prev_end:m.start()]
            after = sent[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(sent)]
            ents = ENTITY.findall(window) or ENTITY.findall(sent[m.start():]) or ENTITY.findall(sent)
            entity = ents[-1] if ents else title_entity
            if not entity:
                prev_end = m.end()
                continue
            cond: dict[str, str] = {}
            mw = list(METHOD_WITH.finditer(window + " " + m.group("verb")))
            mq = list(METHOD_QUANT.finditer(window))
            if mq:
                cond["method"] = mq[-1].group("m").lower()
            elif mw:
                cond["method"] = mw[-1].group("m").lower()
            if (b := BATCH.search(after)):
                cond["batch"] = b.group("b")
            if (c := CTXLEN.search(after)):
                cond["context_len"] = c.group("c").lower()
            # The excerpt is an exact slice of the stored text: from the clause start to the claim end.
            start = base + prev_end + (len(window) - len(window.lstrip(" ,;")))
            end = base + m.end()
            if (b or c):
                end = base + m.end() + (b or c).end()
            out.append(dict(
                entity=entity, metric=m.group("metric"), metric_key=_metric_key(m.group("metric"), m.group("verb")),
                value=float(m.group("value")), unit=_unit(m.group("unit")), context=m.group("context"),
                conditions=cond, text=text[start:end].strip(), span_start=start, span_end=end,
            ))
            prev_end = m.end()
    return out


def same_quantity(a, b) -> bool:
    g = lambda c, k: c[k] if isinstance(c, dict) else getattr(c, k)
    return (g(a, "entity").lower() == g(b, "entity").lower() and g(a, "metric_key") == g(b, "metric_key")
            and g(a, "unit") == g(b, "unit") and g(a, "context").lower() == g(b, "context").lower())


def comparability(a, b, *, same_work: bool) -> tuple[bool, str]:
    """(comparable, note). Same work across versions: conditions are the work's own. Across works the
    method must be stated on both sides and every stated condition must match."""
    g = lambda c, k: c[k] if isinstance(c, dict) else getattr(c, k)
    if not same_quantity(a, b):
        return False, "different entity, metric, unit or benchmark"
    ca, cb = g(a, "conditions") or {}, g(b, "conditions") or {}
    if same_work:
        return True, f"same work, same metric, unit and {g(a, 'context')}"
    if "method" not in ca or "method" not in cb:
        return False, "method not stated on both sides, so the numbers may describe different techniques"
    for k in sorted(set(ca) | set(cb)):
        if ca.get(k) != cb.get(k):
            return False, f"different {k.replace('_', ' ')}: {ca.get(k, 'not stated')} vs {cb.get(k, 'not stated')}"
    return True, f"same entity, method, metric, unit and {g(a, 'context')}"


def differs(old: float, new: float, unit: str) -> tuple[bool, float, float]:
    """(materially different, abs delta, relative %). Points use an absolute tolerance."""
    d = new - old
    rel = (d / old * 100.0) if old else 0.0
    if unit == "points":
        return abs(d) >= 1.0, round(d, 3), round(rel, 1)
    return abs(rel) >= 10.0, round(d, 3), round(rel, 1)
