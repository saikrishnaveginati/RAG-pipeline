from typing import List
from sentence_transformers import CrossEncoder

# Free, open-source reranker
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = CrossEncoder(MODEL_NAME)


def rerank(
    query: str,
    candidate_texts: List[str]
) -> List[float]:
    """
    Returns relevance scores for (query, text) pairs.
    Higher score = more relevant.
    """

    pairs = [(query, text) for text in candidate_texts]
    scores = _reranker.predict(pairs)

    return scores.tolist()
