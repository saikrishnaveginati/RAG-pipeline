from typing import List, Dict
from RAG.embeddings_free import embed_texts


def retrieve(
    query: str,
    index,
    chunks: List[Dict],
    k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k relevant chunks from FAISS index for a query.
    """

    # 1) Embed query (same model as documents)
    q_vec = embed_texts([query])

    # 2) FAISS ANN search
    scores, indices = index.search(q_vec, k)

    # 3) Map FAISS ids → chunk metadata
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[int(idx)]
        results.append({
            "score": float(score),
            "doc_id": chunk["doc_id"],
            "text": chunk["text"]
        })

    return results
