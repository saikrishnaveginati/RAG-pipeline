from pathlib import Path
import numpy as np

# ---- RAG modules ----
from RAG.ingest import extract_text_from_pdf
from RAG.Chunking import chunk_document
from RAG.embeddings_free import embed_texts
from RAG.faiss_hnsw import build_and_save_hnsw_index
from RAG.reranker_cross_encoder import rerank
from RAG.llm_flan_t5 import generate_answer
# ---------------------

PDF_PATH = Path(__file__).resolve().parent.parent / "data_pdfs" / "PolicyTemplate.pdf"


# ---------- EVAL METRICS ----------

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


# ---------- MAIN TEST ----------

def main():
    print("\n===== START END-TO-END RAG TEST =====\n")

    # 1️⃣ Load + Chunk PDF
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_document({
        "doc_id": "policy",
        "source_path": str(PDF_PATH),
        "text": text
    })

    print(f"Total chunks: {len(chunks)}\n")
    for i, c in enumerate(chunks):
        print(f"[{i}] {c['text'][:70]}...")
    print()

    # 2️⃣ Embed + Build FAISS HNSW
    embeddings = embed_texts([c["text"] for c in chunks])
    index = build_and_save_hnsw_index(embeddings, index_path=None)
    index.hnsw.efSearch = 50

    print(f"Embeddings shape: {embeddings.shape}")
    print("FAISS index ready\n")

    # 3️⃣ Queries + Ground Truth
    ground_truth = {
        "Guidance Documents": [7],
        "Policy Statement": [0],
    }

    TOP_RERANK = 5
    TOP_CONTEXT = 3

    # 4️⃣ Retrieval → Rerank → Eval → LLM
    for query, relevant in ground_truth.items():
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print(f"EXPECTED CHUNKS: {relevant}")

        # ---- FAISS RETRIEVAL ----
        q_vec = embed_texts([query])
        scores, indices = index.search(q_vec, k=10)

        retrieved = [int(i) for i in indices[0] if i != -1]

        print("\nFAISS retrieved:", retrieved)
        print(
            "Recall@10:", recall_at_k(retrieved, relevant, 10),
            "MRR@10:", reciprocal_rank_at_k(retrieved, relevant, 10),
            "nDCG@10:", ndcg_at_k(retrieved, relevant, 10),
        )

        # ---- RERANK TOP-N ----
        rerank_candidates = retrieved[:TOP_RERANK]
        candidate_texts = [chunks[i]["text"] for i in rerank_candidates]

        rerank_scores = rerank(query, candidate_texts)

        reranked_top = [
            idx for _, idx in sorted(
                zip(rerank_scores, rerank_candidates),
                reverse=True
            )
        ]

        reranked = reranked_top + retrieved[TOP_RERANK:]

        print("\nRERANKED:", reranked)
        print(
            "Recall@10:", recall_at_k(reranked, relevant, 10),
            "MRR@10:", reciprocal_rank_at_k(reranked, relevant, 10),
            "nDCG@10:", ndcg_at_k(reranked, relevant, 10),
        )

        # ---- LLM GENERATION ----
        contexts = [chunks[i]["text"] for i in reranked[:TOP_CONTEXT]]

        answer = generate_answer(query, contexts)

        print("\nLLM ANSWER:")
        print(answer)

    print("\n===== END END-TO-END RAG TEST =====\n")


if __name__ == "__main__":
    main()
