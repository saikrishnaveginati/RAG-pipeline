from pathlib import Path
import faiss

# ---- your modules ----
from RAG.ingest import extract_text_from_pdf
from RAG.Chunking import chunk_document
from RAG.embeddings_free import embed_texts
from RAG.faiss_hnsw import build_and_save_hnsw_index   # ✅ correct function
# ---------------------

PDF_PATH = Path(__file__).resolve().parent.parent / "data_pdfs" / "PolicyTemplate.pdf"


def main():
    # 1️⃣ Extract text from PDF
    text = extract_text_from_pdf(PDF_PATH)

    doc = {
        "doc_id": "policy_template",
        "source_path": str(PDF_PATH),
        "text": text
    }

    # 2️⃣ Chunk document
    chunks = chunk_document(doc)
    print(f"Total chunks: {len(chunks)}")

    # 3️⃣ Embed chunks
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    print(f"Embeddings shape: {embeddings.shape}")

    # 4️⃣ Build FAISS HNSW index (reuse existing code)
    index = build_and_save_hnsw_index(
        embeddings=embeddings,
        index_path=None   # set to "policy_hnsw.index" if you want to persist
    )

    # Search-time recall / latency knob
    index.hnsw.efSearch = 50

    # 5️⃣ Query
    query = "Guidance Documents"
    q_vec = embed_texts([query])

    scores, indices = index.search(q_vec, k=3)

    # 6️⃣ Print results
    print("\nTop results:")
    for rank, idx in enumerate(indices[0]):
        score = scores[0][rank]
        chunk = chunks[int(idx)]

        print(f"\nRank {rank + 1}")
        print(f"Score: {score:.4f}")
        print("-" * 60)
        print(chunk["text"][:600])
        


if __name__ == "__main__":
    main()
