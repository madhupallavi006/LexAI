

import re
from dataclasses import dataclass, field
from typing import List

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

class LegalDocumentChunker:
    HEADING_PATTERN = re.compile(
        r'(?i)(?:^|\n)\s*'
        r'(?:section|article|clause|paragraph|part)'
        r'\s*[\d\.]+',
        re.MULTILINE
    )

    def __init__(self, max_chars=600):
        self.max_chars = max_chars

    def split(self, text, doc_id):
        raw_chunks = self._split_by_headings(text)
        if len(raw_chunks) < 3:
            raw_chunks = self._split_by_paragraphs(text)
        chunks = []
        idx = 0
        for raw in raw_chunks:
            raw = raw.strip()
            if not raw:
                continue
            for sub in self._cap(raw):
                chunks.append(Chunk(
                    id=f"{doc_id}__clause_{idx}",
                    text=sub,
                    metadata={"clause_ref": str(idx), "doc_id": doc_id}
                ))
                idx += 1
        return chunks

    def _split_by_headings(self, text):
        positions = [m.start() for m in self.HEADING_PATTERN.finditer(text)]
        if not positions:
            return [text]
        parts = []
        for i, pos in enumerate(positions):
            end = positions[i+1] if i+1 < len(positions) else len(text)
            parts.append(text[pos:end])
        return parts

    def _split_by_paragraphs(self, text):
        return [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    def _cap(self, text):
        if len(text) <= self.max_chars:
            return [text]
        parts = []
        while text:
            parts.append(text[:self.max_chars])
            text = text[self.max_chars:]
        return parts