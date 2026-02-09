from pathlib import Path
import numpy as np

from RAG.ingest import extract_text_from_pdf
from RAG.Chunking import chunk_document
from RAG.embeddings_free import embed_texts
from RAG.faiss_hnsw import build_and_save_hnsw_index
from RAG.reranker_cross_encoder import rerank


PDF_PATH = Path(__file__).resolve().parent.parent / "data_pdfs" / "PolicyTemplate.pdf"


def recall_at_k(retrieved, relevant, k):
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank_at_k(retrieved, relevant, k):
    for i, idx in enumerate(retrieved[:k], start=1):
        if idx in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for i, idx in enumerate(retrieved[:k], start=1):
        if idx in relevant:
            dcg += 1.0 / np.log2(i + 1)

    ideal_dcg = sum(
        1.0 / np.log2(i + 1)
        for i in range(1, min(len(relevant), k) + 1)
    )

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def main():
    print("\n===== START RETRIEVAL + RERANKING TEST =====\n")

    # 1️⃣ Load + chunk
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_document({
        "doc_id": "policy",
        "source_path": str(PDF_PATH),
        "text": text
    })

    print(f"Chunks: {len(chunks)}")

    # 2️⃣ Embed + FAISS
    embeddings = embed_texts([c["text"] for c in chunks])
    index = build_and_save_hnsw_index(embeddings, index_path=None)
    index.hnsw.efSearch = 50

    # 3️⃣ Ground truth
    ground_truth = {
        "Guidance Documents": [7],
        "Policy Statement": [0],
    }

    TOP_RERANK = 5

    for query, relevant in ground_truth.items():
        print(f"\nQUERY: {query}")
        print(f"EXPECTED: {relevant}")

        # -------- FAISS RETRIEVAL --------
        q_vec = embed_texts([query])
        scores, indices = index.search(q_vec, k=10)

        # ✅ FILTER INVALID (-1)
        retrieved = [int(i) for i in indices[0] if i != -1]

        print("\nFAISS retrieved (valid):", retrieved)

        print(
            "Recall@10:",
            recall_at_k(retrieved, relevant, 10),
            "MRR@10:",
            reciprocal_rank_at_k(retrieved, relevant, 10),
            "nDCG@10:",
            ndcg_at_k(retrieved, relevant, 10),
        )

        # -------- RERANK TOP-N ONLY --------
        rerank_candidates = retrieved[:TOP_RERANK]
        candidate_texts = [chunks[i]["text"] for i in rerank_candidates]

        rerank_scores = rerank(query, candidate_texts)

        reranked_top = [
            idx for _, idx in sorted(
                zip(rerank_scores, rerank_candidates),
                reverse=True
            )
        ]

        # ✅ Append remaining FAISS results unchanged
        reranked = reranked_top + retrieved[TOP_RERANK:]

        print("\nRERANKED (top-5 reranked):", reranked)

        print(
            "Recall@10:",
            recall_at_k(reranked, relevant, 10),
            "MRR@10:",
            reciprocal_rank_at_k(reranked, relevant, 10),
            "nDCG@10:",
            ndcg_at_k(reranked, relevant, 10),
        )

    print("\n===== END TEST =====\n")


if __name__ == "__main__":
    main()
