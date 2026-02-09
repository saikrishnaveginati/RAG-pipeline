from pathlib import Path

from RAG.ingest import extract_text_from_pdf
from RAG.Chunking import chunk_document
from RAG.embeddings_free import embed_texts
from RAG.faiss_hnsw import build_and_save_hnsw_index

PDF_PATH = Path(__file__).resolve().parent.parent / "data_pdfs" / "PolicyTemplate.pdf"


def recall_at_k(retrieved, relevant, k):
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved, relevant):
    for i, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1.0 / i
    return 0.0


def main():
    print("\n===== START TEST =====\n")

    # 1. Load PDF
    text = extract_text_from_pdf(PDF_PATH)
    print("PDF loaded")

    doc = {
        "doc_id": "policy_template",
        "source_path": str(PDF_PATH),
        "text": text
    }

    # 2. Chunk
    chunks = chunk_document(doc)
    print(f"Chunks created: {len(chunks)}\n")

    for i, c in enumerate(chunks):
        print(f"[{i}] {c['text'][:60]}...")
    print()

    # 3. Embed
    embeddings = embed_texts([c["text"] for c in chunks])
    print(f"Embeddings shape: {embeddings.shape}\n")

    # 4. Build FAISS
    index = build_and_save_hnsw_index(
        embeddings=embeddings,
        index_path=None
    )
    index.hnsw.efSearch = 50
    print("FAISS index built\n")

    # 5. Ground truth
    ground_truth = {
        "Guidance Documents": [7],
        "Penalties for Non-Compliance": [2],
    }

    # 6. Retrieval + eval
    for query, relevant in ground_truth.items():
        print(f"\nQUERY: {query}")
        print(f"EXPECTED: {relevant}")

        q_vec = embed_texts([query])
        scores, indices = index.search(q_vec, k=5)

        retrieved = indices[0].tolist()
        scores = scores[0].tolist()

        print("\nRETRIEVED:")
        for rank, (idx, score) in enumerate(zip(retrieved, scores), start=1):
            print(f"Rank {rank} | chunk {idx} | score {score:.4f}")
            print(chunks[idx]["text"][:120], "\n")

        r = recall_at_k(retrieved, relevant, 5)
        m = reciprocal_rank(retrieved, relevant)

        print(f"Recall@5 = {r}")
        print(f"MRR = {m:.4f}")

    print("\n===== END TEST =====\n")


if __name__ == "__main__":
    main()
