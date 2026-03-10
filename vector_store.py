

import math
import re
from dataclasses import dataclass
from typing import Dict, List
from chunker import Chunk

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float

class SparseVectorStore:

    def __init__(self):
        self._chunks = {}
        self._vectors = {}

    def add(self, chunks, doc_id):
        self._chunks[doc_id] = {}
        self._vectors[doc_id] = {}
        tf_map = {chunk.id: self._term_freq(chunk.text) for chunk in chunks}
        idf = self._idf(tf_map)
        for chunk in chunks:
            self._chunks[doc_id][chunk.id] = chunk
            self._vectors[doc_id][chunk.id] = {
                term: tf * idf.get(term, 0.0)
                for term, tf in tf_map[chunk.id].items()
            }

    def retrieve(self, query, doc_id, top_k=5):
        if doc_id not in self._vectors:
            return []
        q_vec = self._tfidf_query(query)
        scored = [
            RetrievalResult(
                chunk=self._chunks[doc_id][cid],
                score=round(self._cosine(q_vec, vec), 4)
            )
            for cid, vec in self._vectors[doc_id].items()
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def multi_retrieve(self, queries, doc_id, top_k=3):
        seen, results = set(), []
        for q in queries:
            for r in self.retrieve(q, doc_id=doc_id, top_k=top_k):
                if r.chunk.id not in seen:
                    results.append(r)
                    seen.add(r.chunk.id)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def build_context(self, results):
        parts = []
        for r in results:
            ref = r.chunk.metadata.get("clause_ref", "?")
            parts.append(f"[Clause {ref} | Score: {r.score}]\n{r.chunk.text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _tokenize(text):
        return re.findall(r'[a-z]{2,}', text.lower())

    def _term_freq(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = len(tokens)
        return {t: c / total for t, c in counts.items()}

    def _idf(self, tf_map):
        N = len(tf_map)
        if N == 0:
            return {}
        doc_freq = {}
        for tf in tf_map.values():
            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        return {
            term: math.log((N+1) / (df+1)) + 1.0
            for term, df in doc_freq.items()
        }

    def _tfidf_query(self, query):
        tokens = self._tokenize(query)
        if not tokens:
            return {}
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = len(tokens)
        return {t: c / total for t, c in counts.items()}

    @staticmethod
    def _cosine(a, b):
        if not a or not b:
            return 0.0
        dot    = sum(a.get(t, 0.0) * v for t, v in b.items())
        norm_a = math.sqrt(sum(v*v for v in a.values()))
        norm_b = math.sqrt(sum(v*v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)