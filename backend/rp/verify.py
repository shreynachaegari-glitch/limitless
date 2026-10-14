"""Deterministic evidence gate. The model never certifies its own citations."""

from __future__ import annotations

import re

from sqlalchemy.orm import Session

from .db import DocumentVersion

NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def check_excerpt(s: Session, version_id: int, excerpt: str) -> bool:
    v = s.get(DocumentVersion, version_id)
    return bool(v and excerpt and excerpt.strip() in v.text)


def verify_evidence(s: Session, evidence: list[dict]) -> tuple[bool, list[str]]:
    gaps = []
    for e in evidence:
        e["verified"] = check_excerpt(s, e["version_id"], e["excerpt"])
        if not e["verified"]:
            gaps.append(f"excerpt not found in stored text of document {e['document_id']} v-id {e['version_id']}")
    return not gaps, gaps


def numbers(text: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in NUM.findall(text or "")}


def grounded(narrative: str, packet_text: str) -> tuple[bool, list[str]]:
    """Every number the narrator writes must appear somewhere in the evidence packet."""
    missing = sorted(numbers(narrative) - numbers(packet_text))
    return not missing, missing


QUALIFIED = __import__("re").compile(r"\b(up to|as much as|as high as|at most|at least|nearly|approximately|roughly|more than|less than)\s+(\d+(?:\.\d+)?)", __import__("re").I)


def qualifiers_kept(narrative: str, evidence_texts: list[str]) -> tuple[bool, list[str]]:
    """A number the source qualified ("up to 10x") must keep its qualifier when the narrator repeats it;
    otherwise a best case reads as a typical result."""
    dropped = []
    low = narrative.lower()
    for t in evidence_texts:
        for q, num in QUALIFIED.findall(t or ""):
            if num in narrative and f"{q.lower()} {num}" not in low:
                dropped.append(f"{q} {num}")
    return not dropped, sorted(set(dropped))
