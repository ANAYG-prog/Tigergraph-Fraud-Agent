"""Deterministic local text embeddings (no external API) for policy/typology GraphRAG.

Signed feature hashing of word uni/bi-grams + sublinear TF, L2-normalised. Good enough
for retrieving the right policy clause / typology section from a few dozen documents, and
reproducible across machines so the vectors stored in TigerGraph stay valid.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

DIM = 256
_vec = HashingVectorizer(n_features=DIM, alternate_sign=True, ngram_range=(1, 2), norm=None,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b", stop_words="english")

SYNONYMS = {
    "ato": "account takeover", "sar": "suspicious activity report", "cnp": "card not present",
    "mule": "money mule", "otp": "one time passcode step-up authentication",
}


def _prep(text: str) -> str:
    t = text.lower().replace("_", " ")
    for k, v in SYNONYMS.items():
        t = re.sub(rf"\b{k}\b", f"{k} {v}", t)
    return t


def embed(texts: list[str]) -> np.ndarray:
    m = _vec.transform([_prep(t) for t in texts]).toarray()
    m = np.sign(m) * np.log1p(np.abs(m))
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def cosine_top_k(q: np.ndarray, mat: np.ndarray, k: int) -> list[tuple[int, float]]:
    if mat.size == 0:
        return []
    qn = q / (np.linalg.norm(q) or 1)
    mn = mat / np.where(np.linalg.norm(mat, axis=1, keepdims=True) == 0, 1, np.linalg.norm(mat, axis=1, keepdims=True))
    s = mn @ qn
    idx = np.argsort(-s)[:k]
    return [(int(i), float(s[i])) for i in idx]
