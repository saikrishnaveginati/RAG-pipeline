
from pathlib import Path

from RAG.ingest import extract_text_from_pdf
from RAG.Chunking import chunk_document
from RAG.embeddings_free import embed_texts
from RAG.faiss_hnsw import build_and_save_hnsw_index, load_hnsw_index
from RAG.reranker_cross_encoder import rerank
from RAG.llm_flan_t5 import generate_answer


DATA_DIR = Path(__file__).resolve().parent.parent / "data_pdfs"
INDEX_PATH = Path(__file__).resolve().parent / "policy_hnsw.index"


# -------------------------------
# OFFLINE STEP: INGEST + INDEX
# -------------------------------

def build_index():
    print("🔹 Building FAISS index from PDFs...")

    all_chunks = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
        print(f"Parsing: {pdf_path.name}")

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_document({
            "doc_id": pdf_path.stem,
            "source_path": str(pdf_path),
            "text": text
        })

        all_chunks.extend(chunks)

    print(f"Total chunks indexed: {len(all_chunks)}")

    embeddings = embed_texts([c["text"] for c in all_chunks])

    index = build_and_save_hnsw_index(
        embeddings=embeddings,
        index_path=str(INDEX_PATH)
    )

    print("✅ FAISS index saved")
    return index, all_chunks


# -------------------------------
# ONLINE STEP: QUERY → ANSWER
# -------------------------------

def answer_query(query: str, index, chunks):
    print(f"\n🔍 Query: {query}")

    # 1. Retrieve
    q_vec = embed_texts([query])
    _, indices = index.search(q_vec, k=10)

    retrieved = [int(i) for i in indices[0] if i != -1]

    # 2. Rerank (top 5)
    TOP_RERANK = 5
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

    # 3. Generate answer
    TOP_CONTEXT = 3
    contexts = [chunks[i]["text"] for i in reranked[:TOP_CONTEXT]]

    answer = generate_answer(query, contexts)

    print("\n🧠 Answer:")
    print(answer)
    return answer


# -------------------------------
# MAIN
# -------------------------------

def main():
    # Build index once (offline)
    if not INDEX_PATH.exists():
        index, chunks = build_index()
    else:
        print("🔹 Loading existing FAISS index...")
        index = load_hnsw_index(str(INDEX_PATH))
        # NOTE: chunks must be loaded from the same build step in real prod
        # Here we rebuild chunks for simplicity
        _, chunks = build_index()

    # Example queries
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer_query(query, index, chunks)


if __name__ == "__main__":
    main()
