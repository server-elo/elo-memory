"""
Memory Interference Resolution
==============================

Handles proactive and retroactive interference between similar memories.
Implements pattern separation and completion mechanisms.

References:
- McClelland et al. (1995): Complementary learning systems
- Yassa & Stark (2011): Pattern separation in hippocampus
- Anderson & Neely (1996): Interference and inhibition in memory
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine


@dataclass
class InterferenceConfig:
    """Configuration for interference resolution."""

    similarity_threshold: float = 0.85  # When memories interfere
    separation_strength: float = 0.3  # How much to orthogonalize
    min_pattern_distance: float = 0.2  # Minimum separation
    separation_noise_scale: float = 0.1  # Noise added when patterns are too similar


class InterferenceResolver:
    """
    Resolves interference between similar memories.

    Two main mechanisms:
    1. Pattern Separation: Orthogonalize similar memories
    2. Pattern Completion: Reconstruct partial cues
    """

    def __init__(self, config: Optional[InterferenceConfig] = None):
        self.config = config or InterferenceConfig()

    def detect_interference(
        self, new_embedding: np.ndarray, existing_embeddings: List[np.ndarray]
    ) -> List[int]:
        """
        Detect which existing memories interfere with new memory.

        Args:
            new_embedding: New memory embedding
            existing_embeddings: List of existing memory embeddings

        Returns:
            Indices of interfering memories
        """
        interfering_indices = []

        for i, existing in enumerate(existing_embeddings):
            similarity = self._cosine_similarity(new_embedding, existing)
            if similarity >= self.config.similarity_threshold:
                interfering_indices.append(i)

        return interfering_indices

    def apply_pattern_separation(
        self, new_embedding: np.ndarray, interfering_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Apply pattern separation to reduce interference.

        Orthogonalizes new embedding away from interfering memory.

        Args:
            new_embedding: New memory to separate
            interfering_embedding: Interfering memory

        Returns:
            Separated embedding
        """
        # Project new onto interfering
        projection = np.dot(new_embedding, interfering_embedding) * interfering_embedding

        # Remove projection (orthogonalize)
        separated = new_embedding - self.config.separation_strength * projection

        # Normalize
        separated = separated / (np.linalg.norm(separated) + 1e-8)

        # Ensure minimum distance
        similarity = self._cosine_similarity(separated, interfering_embedding)
        if similarity > (1 - self.config.min_pattern_distance):
            # Add random noise to increase separation
            noise = np.random.randn(len(separated)) * self.config.separation_noise_scale
            separated = separated + noise
            separated = separated / (np.linalg.norm(separated) + 1e-8)

        return separated

    def pattern_complete(
        self, partial_cue: np.ndarray, stored_patterns: List[np.ndarray], threshold: float = 0.5
    ) -> Optional[np.ndarray]:
        """
        Complete partial memory cue using stored patterns.

        Args:
            partial_cue: Incomplete memory cue
            stored_patterns: Complete stored memories
            threshold: Minimum similarity to trigger completion

        Returns:
            Completed pattern or None
        """
        best_match = None
        best_similarity = threshold

        for pattern in stored_patterns:
            similarity = self._cosine_similarity(partial_cue, pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern

        return best_match

    def resolve_interference_set(
        self, new_embedding: np.ndarray, existing_embeddings: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Resolve interference for a new memory against existing set.

        Args:
            new_embedding: New memory
            existing_embeddings: Existing memories

        Returns:
            (separated_new, updated_existing)
        """
        # Detect interfering memories
        interfering_indices = self.detect_interference(new_embedding, existing_embeddings)

        if not interfering_indices:
            return new_embedding, existing_embeddings

        # Apply pattern separation
        separated_new = new_embedding.copy()
        for idx in interfering_indices:
            separated_new = self.apply_pattern_separation(separated_new, existing_embeddings[idx])

        return separated_new, existing_embeddings

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(_sklearn_cosine(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0])
