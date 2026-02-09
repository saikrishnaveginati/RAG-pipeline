from typing import List, Dict
import re


class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        separators: List[str] | None = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

    def split_text(self, text: str) -> List[str]:
        text = self._clean(text)
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, seps: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not seps:
            return self._hard_split(text)

        sep = seps[0]
        parts = text.split(sep) if sep else list(text)

        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.extend(self._recursive_split(current, seps[1:]))
                current = part

        if current:
            chunks.extend(self._recursive_split(current, seps[1:]))

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]

    def add_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0:
            return chunks

        overlapped = []
        prev = ""

        for chunk in chunks:
            if prev:
                overlap = prev[-self.chunk_overlap :]
                chunk = overlap + chunk
            overlapped.append(chunk)
            prev = chunk

        return overlapped

    def _clean(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _dominant_block(blocks, start, end):
    best = None
    best_overlap = 0

    for block in blocks:
        overlap = max(0, min(end, block["end"]) - max(start, block["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best = block

    return best


def chunk_document(doc: Dict) -> List[Dict]:
    splitter = RecursiveTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    text = doc["text"]
    blocks = doc.get("blocks", [])

    raw_chunks = splitter.split_text(text)
    raw_chunks = splitter.add_overlap(raw_chunks)

    chunks = []
    cursor = 0

    for i, chunk_text in enumerate(raw_chunks):
        start = cursor
        end = cursor + len(chunk_text)
        cursor = end

        block = _dominant_block(blocks, start, end)

        chunks.append({
            "chunk_id": f"{doc['doc_id']}::chunk_{i}",
            "doc_id": doc["doc_id"],
            "source_path": doc["source_path"],
            "text": chunk_text,
            "page": block["page"] if block else None,
            "content_type": block["type"] if block else "unknown"
        })

    return chunks
