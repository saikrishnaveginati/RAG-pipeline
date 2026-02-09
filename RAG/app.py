from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import json
from pathlib import Path

from RAG.embeddings_free import embed_texts
from RAG.reranker_cross_encoder import rerank
from RAG.llm_flan_t5 import generate_answer


# ================= CONFIG =================

ARTIFACTS_DIR = Path("/artifacts")

INDEX_PATH = ARTIFACTS_DIR / "policy_hnsw.index"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"

# =========================================


app = FastAPI(title="RAG Policy Assistant")

index = None
chunks = None


class QueryReq(BaseModel):
    query: str


def load_artifacts():
    """
    Load FAISS index + chunk metadata from local disk.
    Runs once per container lifetime.
    """
    global index, chunks

    if index is not None and chunks is not None:
        return

    print("📦 Loading FAISS index and chunks from disk")

    if not INDEX_PATH.exists():
        raise RuntimeError(f"FAISS index not found at {INDEX_PATH}")

    if not CHUNKS_PATH.exists():
        raise RuntimeError(f"Chunks file not found at {CHUNKS_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    index.hnsw.efSearch = 50

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("✅ Artifacts loaded successfully")


@app.post("/ask")
def ask(req: QueryReq):
    """
    End-to-end RAG query handler
    """
    load_artifacts()

    query = req.query

    # 1️⃣ Embed query
    q_vec = embed_texts([query])

    # 2️⃣ Retrieve from FAISS
    _, indices = index.search(q_vec, k=10)
    retrieved = [int(i) for i in indices[0] if i != -1]

    if not retrieved:
        return {
            "query": query,
            "answer": "I could not find relevant information in the documents.",
            "sources": []
        }

    # 3️⃣ Rerank top candidates
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

    # 4️⃣ Generate answer from top contexts
    TOP_CONTEXT = 3
    contexts = [chunks[i]["text"] for i in reranked[:TOP_CONTEXT]]

    answer = generate_answer(query, contexts)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "rank": r + 1,
                "doc_id": chunks[idx]["doc_id"],
                "preview": chunks[idx]["text"][:200]
            }
            for r, idx in enumerate(reranked[:TOP_CONTEXT])
        ]
    }
