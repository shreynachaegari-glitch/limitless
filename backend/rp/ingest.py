"""Normalise, identify and version incoming items. Deterministic; no model involved.

Identity order: DOI > arXiv id (version stripped) > canonical URL. Same identity + same content hash
= unchanged (no-op). Same identity + new hash = a new DocumentVersion, never a second document.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import Document, DocumentVersion, utcnow
from .sources import RawItem

_TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|ref_src$|source$)")

# Instruction-like text inside a source (pattern set ported from CodeRush 2.0 contradiction.py).
_INJECTION = re.compile("|".join([
    r"ignore (all|any|previous|prior) instructions",
    r"disregard (the|your) (system|previous) prompt",
    r"you are now",
    r"new instructions?:",
    r"system\s*:\s*override",
    r"execute the following (code|command)",
    r"reveal (your|the) (system prompt|instructions)",
    r"mark (this|it) (article )?as (high|urgent)",
]), re.IGNORECASE)


def canonical_url(url: str) -> str:
    p = urlsplit(url.strip())
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k.lower())]
    path = re.sub(r"/+$", "", p.path) or "/"
    return urlunsplit((p.scheme.lower() or "https", p.netloc.lower().removeprefix("www."), path, urlencode(q), ""))


def identity_key(item: RawItem) -> str:
    if item.doi:
        return f"doi:{item.doi.lower()}"
    if item.arxiv_id:
        return f"arxiv:{re.sub(r'v\d+$', '', item.arxiv_id)}"
    return f"url:{canonical_url(item.url)}"


def normalise_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def content_hash(title: str, text: str) -> str:
    return "sha256:" + hashlib.sha256(f"{normalise_text(title).lower()}\n{normalise_text(text)}".encode()).hexdigest()


_SENT = re.compile(r".+?(?:[.!?]+(?=\s|$)|$)", re.S)
MARK = "[quarantined: instruction-like text removed]"


def find_injection(text: str) -> str | None:
    """Every sentence that carries an instruction-like pattern, joined (the first is what the UI quotes)."""
    hits = [m.group(0).strip() for m in _SENT.finditer(text or "") if _INJECTION.search(m.group(0))]
    return " … ".join(hits) if hits else None


def quarantine(text: str, excerpt: str | None = None) -> str:
    """The text a model may see: every instruction-like sentence replaced by a marker, and tag characters
    neutralised so the text cannot close the <source> delimiter. Stored text is untouched."""
    out = "".join(MARK + " " if _INJECTION.search(m.group(0)) else m.group(0) for m in _SENT.finditer(text or ""))
    return out.replace("<", "‹").replace(">", "›")


@dataclass
class IngestResult:
    document: Document
    version: DocumentVersion
    previous: DocumentVersion | None
    is_new_document: bool
    stored_new: bool = True  # False: this exact content was already stored (maybe by another topic)


def safe_url(url: str | None) -> str | None:
    """Only http(s) links are stored; anything else (javascript:, data:, file:) is dropped."""
    return url if url and urlsplit(url.strip()).scheme.lower() in ("http", "https") else None


def ingest(s: Session, source_id: int, item: RawItem, run_id: int) -> IngestResult | None:
    """Stores a new document or version. Unchanged content is not stored again; it comes back with
    stored_new=False so each topic can still decide whether it has assessed that version.
    Returns None only for a replay of a superseded (older) version."""
    if not safe_url(item.url):
        return None  # an item without a safe link cannot be traced back to its source
    item.cites_url = safe_url(item.cites_url)
    key = identity_key(item)
    title, text = normalise_text(item.title), normalise_text(item.text)
    h = content_hash(title, text)
    doc = s.scalar(select(Document).where(Document.canonical_key == key))
    if doc is None:
        doc = Document(
            canonical_key=key, source_id=source_id, external_id=item.external_id, url=canonical_url(item.url),
            doi=item.doi, arxiv_id=item.arxiv_id, title=title, publisher=item.publisher, authors=item.authors,
            source_role=item.role, publication_type=item.publication_type,
            cites_url=canonical_url(item.cites_url) if item.cites_url else None,
            published_at=item.published_at, synthetic=item.synthetic, current_version=0,
        )
        try:
            with s.begin_nested():  # two workers may insert the same new work at once
                s.add(doc)
                s.flush()
        except IntegrityError:
            return ingest(s, source_id, item, run_id)  # the other worker won; treat as an existing document
        prev, is_new = None, True
    else:
        same = s.scalar(select(DocumentVersion).where(DocumentVersion.document_id == doc.id,
                                                      DocumentVersion.content_hash == h))
        if same is not None:
            if same.version != doc.current_version:
                return None
            before = s.scalar(select(DocumentVersion).where(DocumentVersion.document_id == doc.id,
                                                            DocumentVersion.version == same.version - 1))
            return IngestResult(doc, same, before, before is None, stored_new=False)
        prev = s.scalar(select(DocumentVersion).where(DocumentVersion.document_id == doc.id)
                        .order_by(DocumentVersion.version.desc()).limit(1))
        is_new = False
    changed = []
    if prev is not None:
        if prev.title != title:
            changed.append("title")
        if prev.text != text:
            changed.append("text")
    inj = find_injection(f"{title}. {text}")
    n = (s.scalar(select(func.max(DocumentVersion.version)).where(DocumentVersion.document_id == doc.id)) or 0) + 1
    ver = DocumentVersion(
        document_id=doc.id, version=n, content_hash=h, title=title, text=text, changed_fields=changed,
        source_updated_at=item.source_updated_at, retrieved_at=utcnow(), injection_flagged=bool(inj),
        injection_excerpt=inj, run_id=run_id,
    )
    s.add(ver)
    doc.current_version = n
    doc.title = title
    s.flush()
    return IngestResult(doc, ver, prev, is_new)
