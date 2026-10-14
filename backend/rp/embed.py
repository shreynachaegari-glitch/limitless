"""Embeddings: BAAI/bge-small-en-v1.5 on CPU via fastembed (ONNX). Falls back to a hashed
bag-of-words vector (idea from CodeRush 2.0 search.py) if the model can't load, and records which
one produced each vector so two vector spaces are never compared."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from . import config

_model = None
_model_name: str | None = None


def _load():
    global _model, _model_name
    if _model_name is None:
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(config.EMBED_MODEL)
            _model_name = config.EMBED_MODEL
        except Exception as exc:  # offline machine, missing model files
            print(f"WARNING: embedding model unavailable ({exc}); using hashed bag-of-words fallback")
            _model_name = "hash-bow-384"
    return _model, _model_name


def model_name() -> str:
    return _load()[1]


def _hash_vec(text: str) -> list[float]:
    v = np.zeros(config.EMBED_DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % config.EMBED_DIM] += 1.0
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).tolist()


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    m, name = _load()
    if m is None:
        return [_hash_vec(t) for t in texts]
    return [np.asarray(v, dtype=np.float32).tolist() for v in m.embed(texts)]


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d else 0.0


def doc_text(title: str, text: str) -> str:
    return f"{title}. {text}"[:2000]


