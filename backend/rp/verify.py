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
