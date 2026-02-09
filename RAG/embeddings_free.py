from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Lightweight, fast, free
MODEL_NAME = "all-MiniLM-L6-v2"

_model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns normalized embeddings (n, 384)
    """
    texts = [t if t.strip() else " " for t in texts]
    vecs = _model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return np.array(vecs, dtype="float32")
