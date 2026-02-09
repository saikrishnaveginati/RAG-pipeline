from pathlib import Path
import json

from RAG.ingest import extract_text_from_pdf
from RAG.Chunking import chunk_document
from RAG.embeddings_free import embed_texts
from RAG.faiss_hnsw import build_and_save_hnsw_index


# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data_pdfs"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

INDEX_PATH = ARTIFACTS_DIR / "policy_hnsw.index"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"


def main():
    print("\n===== OFFLINE INDEX BUILD STARTED =====\n")

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    all_chunks = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError("No PDFs found in data_pdfs/")

    # 1️⃣ Load + chunk PDFs
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        text = extract_text_from_pdf(pdf_path)

        chunks = chunk_document({
            "doc_id": pdf_path.stem,
            "source_path": str(pdf_path),
            "text": text
        })

        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # 2️⃣ Save chunk metadata
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Chunks saved → {CHUNKS_PATH}")

    # 3️⃣ Embed
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    print(f"Embeddings shape: {embeddings.shape}")

    # 4️⃣ Build + save FAISS index
    build_and_save_hnsw_index(
        embeddings=embeddings,
        index_path=str(INDEX_PATH)
    )

    print(f"FAISS index saved → {INDEX_PATH}")
    print("\n===== OFFLINE INDEX BUILD COMPLETE =====\n")


if __name__ == "__main__":
    main()
