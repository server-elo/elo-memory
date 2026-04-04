"""Shared utilities for elo-memory."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(_sklearn_cosine(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0])


def hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Simple hash-based embedding fallback (deterministic, no model needed)."""
    arr = np.zeros(dim)
    for i, char in enumerate(text):
        idx = (ord(char) * (i + 1)) % dim
        arr[idx] += np.sin(ord(char) * 0.1) * 0.5
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr
