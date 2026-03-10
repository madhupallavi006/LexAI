

import json
import os
import uuid
import re

from chunker import LegalDocumentChunker
from vector_store import SparseVectorStore

LANG_SIGNATURES = {
    "Hindi":     r'[\u0900-\u097F]',
    "Telugu":    r'[\u0C00-\u0C7F]',
    "Tamil":     r'[\u0B80-\u0BFF]',
    "Bengali":   r'[\u0980-\u09FF]',
    "Kannada":   r'[\u0C80-\u0CFF]',
    "Malayalam": r'[\u0D00-\u0D7F]',
}

def detect_language(text):
    for lang, pattern in LANG_SIGNATURES.items():
        if re.search(pattern, text[:2000]):
            return lang
    return "English"

class LegalRAGPipeline:
    MAX_INGEST_CHARS = 40_000

    def __init__(self, risk_clauses_path=None):
        self._chunker  = LegalDocumentChunker(max_chars=600)
        self._store    = SparseVectorStore()
        self.documents = {}
        self._kb_path  = risk_clauses_path or os.path.join(
            os.path.dirname(__file__), "config", "risk_clauses.json"
        )
        self._load_knowledge_base()

    def ingest(self, text, doc_name="document.txt", doc_id=None):
        text     = text[:self.MAX_INGEST_CHARS]
        language = detect_language(text)
        if doc_id is None:
            doc_id = "doc_" + uuid.uuid4().hex[:8]
        chunks = self._chunker.split(text, doc_id=doc_id)
        self._store.add(chunks, doc_id=doc_id)
        self.documents[doc_id] = {
            "doc_name":    doc_name,
            "language":    language,
            "chunk_count": len(chunks),
        }
        return doc_id, len(chunks), language

    def retrieve(self, query, doc_id, top_k=5):
        return self._store.retrieve(query, doc_id=doc_id, top_k=top_k)

    def multi_retrieve(self, queries, doc_id, top_k=3):
        return self._store.multi_retrieve(queries, doc_id=doc_id, top_k=top_k)

    def build_context(self, results):
        return self._store.build_context(results)

    def get_cuad_context(self, query, top_k=4):
        if "__cuad_kb__" not in self.documents:
            return "(Knowledge base not loaded)"
        results = self._store.retrieve(query, doc_id="__cuad_kb__", top_k=top_k)
        return self._store.build_context(results)

    def cuad_kb_status(self):
        if "__cuad_kb__" not in self.documents:
            return "CUAD KB not loaded"
        n = self.documents["__cuad_kb__"]["chunk_count"]
        return f"CUAD KB loaded — {n} clause definitions"

    def get_all_clause_names(self):
        return list(self._kb_data.keys())

    def get_clause_meta(self, clause_name):
        entry = self._kb_data.get(clause_name, {})
        return entry.get("risk", "MEDIUM"), entry.get("description", "")

    def _load_knowledge_base(self):
        try:
            with open(self._kb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"⚠ KB not found at {self._kb_path}")
            self._kb_data = {}
            return
        self._kb_data = {k: v for k, v in data.items() if not k.startswith("_")}
        texts = []
        for name, entry in self._kb_data.items():
            texts.append(
                f"{name}\nRisk: {entry.get('risk','')}\n"
                f"{entry.get('description','')}\n"
                f"Keywords: {' '.join(entry.get('keywords',[]))}"
            )
        self.ingest("\n\n".join(texts), doc_name="CUAD_KB", doc_id="__cuad_kb__")