import faiss
import numpy as np
from typing import Optional


def build_and_save_hnsw_index(
    embeddings: np.ndarray,
    index_path: Optional[str] = None,
    m: int = 32,
    ef_construction: int = 200
) -> faiss.IndexHNSWFlat:
    """
    Build a FAISS HNSW index for cosine similarity.
    Assumes embeddings are L2-normalized.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D numpy array")

    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dim,
        m,
        faiss.METRIC_INNER_PRODUCT
    )

    # Build-time accuracy / memory tradeoff
    index.hnsw.efConstruction = ef_construction

    # Add vectors
    index.add(embeddings)

    # Persist index if path is provided
    if index_path is not None:
        faiss.write_index(index, index_path)

    return index


def load_hnsw_index(
    index_path: str,
    ef_search: int = 50
) -> faiss.IndexHNSWFlat:
    """
    Load a FAISS HNSW index from disk and configure search parameters.
    """

    if not index_path:
        raise ValueError("index_path must be a valid file path")

    index = faiss.read_index(index_path)

    # Search-time accuracy / latency tradeoff
    index.hnsw.efSearch = ef_search

    return index
