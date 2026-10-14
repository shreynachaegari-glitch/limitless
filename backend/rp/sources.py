"""Source adapters. Each one fetches and normalises; none of them analyse or alert.

Interface: fetch(source, topic) -> (items, new_cursor). Raises SourceError on a failed fetch so the
pipeline can record "fetch failed" separately from "no new items".
arXiv/OpenAlex parsing is ported from CodeRush 2.0 (shutdown/scholar.py).
"""

from __future__ import annotations

import random
import re
import time
import defusedxml.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import feedparser
import httpx

from . import config, fixtures

UA = f"ResearchPulse/0.1 (research monitoring; mailto:{config.CONTACT_EMAIL})"
TIMEOUT = httpx.Timeout(20.0, connect=8.0)
MAX_BYTES = 5_000_000


class SourceError(Exception):
    pass


@dataclass
class RawItem:
    external_id: str
    url: str
    title: str
    text: str
    publisher: str
    role: str = "UNKNOWN"
    publication_type: str = "unknown"
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    published_at: datetime | None = None
    source_updated_at: datetime | None = None
    cites_url: str | None = None
    synthetic: bool = False


def _capped(c: httpx.Client, url: str, params: dict | None) -> httpx.Response:
    """Stream the body and stop at MAX_BYTES instead of downloading an unbounded response first."""
    with c.stream("GET", url, params=params) as r:
        buf = bytearray()
        for chunk in r.iter_bytes():
            buf += chunk
            if len(buf) > MAX_BYTES:
                raise SourceError("response too large (over 5 MB)")
        # iter_bytes() already decoded gzip/br, so the encoding headers must not be applied twice.
        headers = {k: v for k, v in r.headers.items() if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")}
        return httpx.Response(r.status_code, headers=headers, content=bytes(buf), request=r.request)


def _get(url: str, params: dict | None = None, *, attempts: int = 3, safe_redirects: bool = False) -> httpx.Response:
    """GET with timeout, Retry-After, exponential backoff + jitter, and a response-size cap.
    Only transient errors (429/5xx/network) are retried; a 4xx raises immediately."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": UA}, follow_redirects=not safe_redirects) as c:
                r = _capped(c, url, params)
                hops = 0
                # User-supplied URLs: re-check every redirect target against the SSRF guard.
                while safe_redirects and r.is_redirect and hops < 3:
                    from .netsafe import assert_public_url
                    nxt = str(httpx.URL(url).join(r.headers.get("location", "")))
                    assert_public_url(nxt)
                    r = _capped(c, nxt, None)
                    hops += 1
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", 0) or 0) or (1.5 * 2 ** i + random.random())
                last = SourceError(f"HTTP {r.status_code} from {httpx.URL(url).host}")
                time.sleep(min(wait, 10))
                continue
            if r.status_code >= 400:
                raise SourceError(f"HTTP {r.status_code} from {httpx.URL(url).host} (not retried)")
            return r
        except httpx.HTTPError as exc:
            last = SourceError(f"{type(exc).__name__}: {exc}")
            time.sleep(1.5 * 2 ** i + random.random())
    raise last or SourceError("fetch failed")


# ---- fixture ---------------------------------------------------------------------------------

def fetch_fixture(source, topic) -> tuple[list[RawItem], dict]:
    cursor = dict(source.cursor or {})
    batch_idx = int(cursor.get("batch", 0))
    anchor = cursor.get("anchor") or datetime.now(timezone.utc).isoformat()
    anchor_dt = datetime.fromisoformat(anchor)
    # Past the last batch, replay the last one unchanged: proves a rerun is a no-op.
    batch = fixtures.BATCHES[min(batch_idx, len(fixtures.BATCHES) - 1)]
    keyed = {b["key"]: b for bb in fixtures.BATCHES for b in bb}
    items = []
    for f in batch:
        cites = keyed[f["cites"]]["url"] if f.get("cites") else None
        upd = f.get("updated_days_ago")
        items.append(RawItem(
            external_id=f["key"], url=f["url"], title=f["title"], text=f["text"], publisher=f["publisher"],
            role=f["role"], publication_type=f["type"], authors=list(f["authors"]), arxiv_id=f.get("arxiv_id"),
            published_at=anchor_dt - timedelta(days=f["days_ago"]),
            source_updated_at=(anchor_dt - timedelta(days=upd)) if upd is not None else None,
            cites_url=cites, synthetic=True,
        ))
    return items, {"batch": batch_idx + 1, "anchor": anchor}


# ---- arXiv -----------------------------------------------------------------------------------

_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_last_arxiv_call = [0.0]


def _concept_query(topic) -> list[str]:
    return [c.split("|")[0] for c in (topic.concepts or [])] or [topic.name]


def fetch_arxiv(source, topic) -> tuple[list[RawItem], dict]:
    terms = [source.query] if source.query else _concept_query(topic)
    items: list[RawItem] = []
    for term in terms[:4]:
        # arXiv asks for a 3-second gap between consecutive API calls.
        gap = time.time() - _last_arxiv_call[0]
        if gap < 3.1:
            time.sleep(3.1 - gap)
        _last_arxiv_call[0] = time.time()
        q = " AND ".join(f'all:"{w}"' if " " in w else f"all:{w}" for w in [term])
        r = _get("https://export.arxiv.org/api/query",
                 {"search_query": q, "max_results": 8, "sortBy": "submittedDate", "sortOrder": "descending"})
        try:
            root = ET.fromstring(r.content)
        except Exception as exc:  # defusedxml raises ParseError or EntitiesForbidden
            raise SourceError(f"arXiv returned malformed XML: {exc}") from exc
        for e in root.findall("a:entry", _ATOM):
            link = (e.findtext("a:id", "", _ATOM) or "").strip()
            m = re.search(r"abs/([\w.\-/]+?)(v\d+)?$", link)
            aid = m.group(1) if m else link
            items.append(RawItem(
                external_id=f"arxiv:{aid}", url=f"https://arxiv.org/abs/{aid}",
                title=re.sub(r"\s+", " ", e.findtext("a:title", "", _ATOM) or "").strip(),
                text=re.sub(r"\s+", " ", e.findtext("a:summary", "", _ATOM) or "").strip(),
                publisher="arXiv", role="PRIMARY", publication_type="preprint",
                authors=[a.findtext("a:name", "", _ATOM) for a in e.findall("a:author", _ATOM)][:8],
                arxiv_id=aid, published_at=_dt(e.findtext("a:published", "", _ATOM)),
                source_updated_at=_dt(e.findtext("a:updated", "", _ATOM)),
            ))
    return _uniq(items), {"last_fetch": datetime.now(timezone.utc).isoformat()}


# ---- OpenAlex --------------------------------------------------------------------------------

def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return " ".join(w for _, w in pos)


def fetch_openalex(source, topic) -> tuple[list[RawItem], dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=60)).date().isoformat()
    items: list[RawItem] = []
    for term in ([source.query] if source.query else _concept_query(topic))[:3]:
        r = _get("https://api.openalex.org/works", {
            "search": term, "per-page": 8, "sort": "publication_date:desc",
            "filter": f"from_publication_date:{since}", "mailto": config.CONTACT_EMAIL})
        for w in r.json().get("results", []):
            loc = w.get("primary_location") or {}
            venue = ((loc.get("source") or {}) or {}).get("display_name") or "OpenAlex"
            doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
            text = _abstract(w.get("abstract_inverted_index"))
            if not text:
                continue  # metadata-only records carry nothing to compare
            items.append(RawItem(
                external_id=w.get("id", ""), url=loc.get("landing_page_url") or w.get("doi") or w.get("id", ""),
                title=(w.get("title") or "").strip(), text=text, publisher=venue, role="PRIMARY",
                publication_type="peer_reviewed" if w.get("type") == "article" else (w.get("type") or "unknown"),
                authors=[(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:8]],
                doi=doi, published_at=_dt(w.get("publication_date")),
            ))
    return _uniq(items), {"last_fetch": datetime.now(timezone.utc).isoformat()}


# ---- RSS / Atom ------------------------------------------------------------------------------

def fetch_rss(source, topic) -> tuple[list[RawItem], dict]:
    if not source.url or not source.url.startswith(("http://", "https://")):
        raise SourceError("RSS source needs an http(s) URL")
    from .netsafe import assert_public_url
    assert_public_url(source.url)
    r = _get(source.url, safe_redirects=True)
    feed = feedparser.parse(r.content)
    if feed.bozo and not feed.entries:
        raise SourceError(f"malformed feed: {feed.bozo_exception}")
    items = []
    for e in feed.entries[:30]:
        text = re.sub(r"<[^>]+>", " ", e.get("summary", "") or "")
        items.append(RawItem(
            external_id=e.get("id") or e.get("link", ""), url=e.get("link", ""),
            title=(e.get("title") or "").strip(), text=re.sub(r"\s+", " ", text).strip(),
            publisher=feed.feed.get("title", source.name), role=source.role_default or "SECONDARY",
            publication_type="news", authors=[a.get("name", "") for a in e.get("authors", [])],
            published_at=_struct(e.get("published_parsed")), source_updated_at=_struct(e.get("updated_parsed")),
        ))
    return [i for i in items if i.url and i.title], {"last_fetch": datetime.now(timezone.utc).isoformat()}


FETCHERS = {"fixture": fetch_fixture, "arxiv": fetch_arxiv, "openalex": fetch_openalex, "rss": fetch_rss}


def _dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _struct(t) -> datetime | None:
    return datetime(*t[:6], tzinfo=timezone.utc) if t else None


def _uniq(items: list[RawItem]) -> list[RawItem]:
    seen, out = set(), []
    for i in items:
        if i.external_id not in seen and i.title:
            seen.add(i.external_id)
            out.append(i)
    return out
