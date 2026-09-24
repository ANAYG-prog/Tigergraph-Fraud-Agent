"""Policy / typology / regulation knowledge base for GraphRAG.

Documents are split on headings into chunks. Each chunk is a PolicyChunk vertex with an
embedding; each documented fraud pattern is a Pattern vertex linked (DESCRIBED_BY) to the
chunks that define it. Retrieval = vector search + graph expansion from the pattern
hypotheses the investigation produced, then a compact, cited context pack for the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..data.dataset import load
from .embed import cosine_top_k, embed


@dataclass
class Chunk:
    chunk_id: str
    source: str
    heading: str
    text: str


def chunk_docs(docs: list[tuple[str, str]], max_chars: int = 1400) -> list[Chunk]:
    out = []
    for src, text in docs:
        parts = re.split(r"(?m)^(#{1,4} .+|[0-9]+(?:\.[0-9]+)*\.? [A-Z][^\n]{2,80})$", text)
        heading, buf = src, ""
        pieces = []
        for p in parts:
            if p is None:
                continue
            if re.match(r"^(#{1,4} .+|[0-9]+(?:\.[0-9]+)*\.? [A-Z][^\n]{2,80})$", p.strip()):
                if buf.strip():
                    pieces.append((heading, buf.strip()))
                heading, buf = p.strip("# ").strip(), ""
            else:
                buf += p
        if buf.strip():
            pieces.append((heading, buf.strip()))
        for i, (h, b) in enumerate(pieces):
            for j in range(0, len(b), max_chars):
                out.append(Chunk(f"{src}#{i}.{j // max_chars}", src, h, b[j:j + max_chars]))
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


class KnowledgeBase:
    def __init__(self, docs=None):
        docs = docs if docs is not None else load().docs
        self.chunks = chunk_docs(docs)
        self.mat = embed([f"{c.heading}\n{c.text}" for c in self.chunks]) if self.chunks else np.zeros((0, 1))
        self.patterns = self._extract_patterns()

    # Documented patterns = headings inside the pattern/typology document(s)
    def _extract_patterns(self) -> dict[str, dict]:
        pats = {}
        for c in self.chunks:
            src = c.source.lower()
            if any(k in src for k in ("pattern", "typolog", "scheme", "modus")) and c.heading != c.source:
                name = _norm(re.sub(r"^(pattern\s*\d*[:.\-]?\s*|\d+[.)]\s*)", "", c.heading, flags=re.I))
                if name and len(name) < 60:
                    p = pats.setdefault(name, {"name": name, "title": c.heading, "chunks": [], "description": ""})
                    p["chunks"].append(c.chunk_id)
                    p["description"] = (p["description"] + " " + c.text)[:1200].strip()
        return pats

    def search(self, query: str, k: int = 6, source_filter: str | None = None) -> list[dict]:
        if not self.chunks:
            return []
        hits = cosine_top_k(embed([query])[0], self.mat, k * 3)
        out = []
        for i, s in hits:
            c = self.chunks[i]
            if source_filter and source_filter not in c.source.lower():
                continue
            out.append({"chunk_id": c.chunk_id, "source": c.source, "heading": c.heading, "text": c.text, "score": round(s, 3)})
            if len(out) >= k:
                break
        return out

    def pattern_chunks(self, name: str) -> list[dict]:
        p = self.patterns.get(name)
        if not p:
            return []
        idx = {c.chunk_id: c for c in self.chunks}
        return [{"chunk_id": cid, "source": idx[cid].source, "heading": idx[cid].heading, "text": idx[cid].text}
                for cid in p["chunks"] if cid in idx]

    def context_pack(self, question: str, pattern_names: list[str], k: int = 5) -> dict:
        """GraphRAG context: pattern definitions reached via graph links + vector-retrieved policy clauses."""
        seen, pack = set(), {"pattern_definitions": [], "policy_clauses": [], "regulatory": []}
        for n in pattern_names[:3]:
            for c in self.pattern_chunks(n)[:2]:
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"]); pack["pattern_definitions"].append(c)
        for c in self.search(question, k=k * 2):
            if c["chunk_id"] in seen:
                continue
            seen.add(c["chunk_id"])
            bucket = "regulatory" if any(x in c["source"].lower() for x in ("regul", "cfr", "law", "reference")) else "policy_clauses"
            if len(pack[bucket]) < k:
                pack[bucket].append(c)
        return pack


@lru_cache
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()
