"""
Real-Time Parametric Evolution Module
=====================================

Low-rank adaptation of the embedding space based on retrieval feedback.
The agent literally gets smarter from its own experience — embeddings
evolve to improve future retrieval quality.

Core idea: learn a correction matrix W = I + A @ B.T where A, B are
low-rank matrices.  Retrieval feedback (was the result relevant?)
provides gradient signal to update A and B.

Includes replay buffer integration and safe rollback if quality degrades.

References:
- Hu et al. (2022): LoRA — Low-Rank Adaptation
- Kirkpatrick et al. (2017): Elastic Weight Consolidation
- Jang et al. (2022): Continual Learning with LoRA
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for parametric evolution."""

    rank: int = 16  # Low-rank dimension
    learning_rate: float = 0.001  # Gradient step size
    momentum: float = 0.9  # SGD momentum
    ewc_lambda: float = 0.1  # EWC regularization
    feedback_buffer_size: int = 500  # Feedback history for training
    min_feedback_for_update: int = 20  # Min feedback samples before updating
    rollback_threshold: float = -0.05  # Rollback if quality drops by this much
    checkpoint_interval: int = 50  # Save checkpoint every N updates
    max_grad_norm: float = 1.0  # Gradient clipping


@dataclass
class RetrievalFeedback:
    """Single retrieval feedback signal."""

    query_embedding: np.ndarray
    result_embedding: np.ndarray
    relevance: float  # 1.0 = relevant, 0.0 = irrelevant
    timestamp: float = 0.0


class ParametricEvolution:
    """
    Low-rank adaptation of embedding space from retrieval feedback.

    Maintains correction matrices A (embedding_dim x rank) and
    B (embedding_dim x rank) such that:
        adapted_embedding = embedding + A @ B.T @ embedding

    Gradient comes from retrieval feedback: if a result was relevant,
    the correction should increase similarity; if irrelevant, decrease it.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        config: Optional[EvolutionConfig] = None,
    ):
        self.config = config or EvolutionConfig()
        self.embedding_dim = embedding_dim
        self.rank = self.config.rank

        # Low-rank matrices (initialized small)
        self.A = np.random.randn(embedding_dim, self.rank) * 0.01
        self.B = np.random.randn(embedding_dim, self.rank) * 0.01

        # Momentum buffers
        self._momentum_A = np.zeros_like(self.A)
        self._momentum_B = np.zeros_like(self.B)

        # Fisher information for EWC
        self._fisher_A = np.zeros_like(self.A)
        self._fisher_B = np.zeros_like(self.B)
        self._anchor_A = self.A.copy()
        self._anchor_B = self.B.copy()

        # Feedback buffer
        self._feedback: deque = deque(maxlen=self.config.feedback_buffer_size)
        self._lock = threading.Lock()

        # Quality tracking
        self._quality_history: List[float] = []
        self._update_count = 0

        # Checkpoint for rollback
        self._checkpoint_A: Optional[np.ndarray] = None
        self._checkpoint_B: Optional[np.ndarray] = None
        self._checkpoint_quality: float = 0.0

    # ── Core API ─────────────────────────────────────────────────

    def adapt_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Apply learned correction to an embedding (thread-safe read)."""
        with self._lock:
            A, B = self.A.copy(), self.B.copy()
        correction = A @ (B.T @ embedding)
        adapted = embedding + correction

        # Re-normalize
        norm = np.linalg.norm(adapted)
        result: np.ndarray = adapted / norm if norm > 0 else adapted
        return result

    @staticmethod
    def _adapt_with(embedding: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Apply adaptation using given A, B matrices (no locking)."""
        correction = A @ (B.T @ embedding)
        adapted = embedding + correction
        norm = np.linalg.norm(adapted)
        return adapted / norm if norm > 0 else adapted

    def record_feedback(
        self,
        query_embedding: np.ndarray,
        result_embedding: np.ndarray,
        relevance: float,
    ) -> None:
        """Record retrieval feedback for future weight updates."""
        fb = RetrievalFeedback(
            query_embedding=query_embedding.copy(),
            result_embedding=result_embedding.copy(),
            relevance=relevance,
            timestamp=time.time(),
        )
        with self._lock:
            self._feedback.append(fb)

    def update_weights(self) -> Dict[str, Any]:
        """
        Perform a gradient update on A and B using accumulated feedback.

        Loss = -sum(relevance * cos_sim(adapt(query), adapt(result)))
             + ewc_lambda * EWC_penalty

        Returns update statistics.
        """
        with self._lock:
            feedback = list(self._feedback)
            # Snapshot A and B for consistent gradient computation
            A_snap = self.A.copy()
            B_snap = self.B.copy()

        if len(feedback) < self.config.min_feedback_for_update:
            return {"updated": False, "reason": "insufficient_feedback"}

        # Compute gradients using snapshot matrices (no race conditions)
        grad_A = np.zeros_like(A_snap)
        grad_B = np.zeros_like(B_snap)
        total_loss = 0.0
        n = len(feedback)

        for fb in feedback:
            # Adapt using snapshot matrices
            q_adapted = self._adapt_with(fb.query_embedding, A_snap, B_snap)
            r_adapted = self._adapt_with(fb.result_embedding, A_snap, B_snap)

            # Cosine similarity
            cos_sim = float(np.dot(q_adapted, r_adapted))

            # Loss: we want cos_sim to match relevance
            error = fb.relevance - cos_sim
            total_loss += error**2

            # Approximate gradient
            q_proj = B_snap.T @ fb.query_embedding
            r_proj = B_snap.T @ fb.result_embedding

            grad_A += (
                -error
                * (np.outer(fb.query_embedding, r_proj) + np.outer(fb.result_embedding, q_proj))
                / n
            )

            grad_B += (
                -error
                * (
                    np.outer(fb.query_embedding, A_snap.T @ fb.result_embedding)
                    + np.outer(fb.result_embedding, A_snap.T @ fb.query_embedding)
                )
                / n
            )

        # EWC penalty gradient
        ewc_grad_A = self.config.ewc_lambda * self._fisher_A * (A_snap - self._anchor_A)
        ewc_grad_B = self.config.ewc_lambda * self._fisher_B * (B_snap - self._anchor_B)
        grad_A += ewc_grad_A
        grad_B += ewc_grad_B

        # Gradient clipping
        grad_norm = np.sqrt(np.sum(grad_A**2) + np.sum(grad_B**2))
        if grad_norm > self.config.max_grad_norm:
            scale = self.config.max_grad_norm / grad_norm
            grad_A *= scale
            grad_B *= scale

        # SGD with momentum (thread-safe write)
        with self._lock:
            self._momentum_A = (
                self.config.momentum * self._momentum_A - self.config.learning_rate * grad_A
            )
            self._momentum_B = (
                self.config.momentum * self._momentum_B - self.config.learning_rate * grad_B
            )
            self.A += self._momentum_A
            self.B += self._momentum_B

        # Update Fisher information (running EMA of squared gradients)
        self._fisher_A = 0.9 * self._fisher_A + 0.1 * (grad_A**2)
        self._fisher_B = 0.9 * self._fisher_B + 0.1 * (grad_B**2)

        # Update EWC anchors periodically (every 100 updates) to prevent
        # unbounded drift while still providing short-term stability
        if self._update_count > 0 and self._update_count % 100 == 0:
            with self._lock:
                self._anchor_A = self.A.copy()
                self._anchor_B = self.B.copy()
                self._fisher_A = np.zeros_like(self.A)
                self._fisher_B = np.zeros_like(self.B)

        # Track quality
        avg_loss = total_loss / n
        quality = 1.0 - avg_loss  # Higher is better
        self._quality_history.append(quality)
        self._update_count += 1

        # Checkpoint and rollback check
        if self._update_count % self.config.checkpoint_interval == 0:
            self._maybe_checkpoint(quality)

        return {
            "updated": True,
            "avg_loss": avg_loss,
            "quality": quality,
            "grad_norm": float(grad_norm),
            "feedback_used": n,
            "update_count": self._update_count,
        }

    # ── Experience Distillation ──────────────────────────────────

    def distill_experiences(
        self,
        embeddings: List[np.ndarray],
        importances: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Compress a set of experience embeddings into a single
        distilled representation using the learned adaptation.
        """
        if not embeddings:
            return np.zeros(self.embedding_dim)

        weights = np.array(importances or [1.0] * len(embeddings))
        weights = weights / weights.sum()

        # Adapt each embedding
        adapted = [self.adapt_embedding(e) for e in embeddings]

        # Importance-weighted mean
        distilled = sum(w * e for w, e in zip(weights, adapted))

        norm = np.linalg.norm(distilled)
        result: np.ndarray = distilled / norm if norm > 0 else distilled
        return result

    # ── Rollback ─────────────────────────────────────────────────

    def _maybe_checkpoint(self, current_quality: float) -> None:
        """Save checkpoint or rollback if quality degraded."""
        if self._checkpoint_A is None:
            # First checkpoint
            self._checkpoint_A = self.A.copy()
            self._checkpoint_B = self.B.copy()
            self._checkpoint_quality = current_quality
            return

        quality_delta = current_quality - self._checkpoint_quality
        if quality_delta < self.config.rollback_threshold:
            logger.warning(
                "Quality degraded by %.3f, rolling back to checkpoint",
                quality_delta,
            )
            if self._checkpoint_A is not None:
                self.A = self._checkpoint_A.copy()
            if self._checkpoint_B is not None:
                self.B = self._checkpoint_B.copy()
        else:
            # Update checkpoint
            self._checkpoint_A = self.A.copy()
            self._checkpoint_B = self.B.copy()
            self._checkpoint_quality = current_quality

    def rollback(self) -> None:
        """Manually rollback to last checkpoint."""
        if self._checkpoint_A is not None and self._checkpoint_B is not None:
            self.A = self._checkpoint_A.copy()
            self.B = self._checkpoint_B.copy()
            logger.info("Rolled back to checkpoint")

    # ── Statistics ───────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        correction_magnitude = float(np.linalg.norm(self.A @ self.B.T, ord="fro"))
        return {
            "embedding_dim": self.embedding_dim,
            "rank": self.rank,
            "update_count": self._update_count,
            "feedback_buffer_size": len(self._feedback),
            "correction_magnitude": correction_magnitude,
            "quality_trend": self._quality_history[-10:] if self._quality_history else [],
            "has_checkpoint": self._checkpoint_A is not None,
        }

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data = {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "fisher_A": self._fisher_A.tolist(),
            "fisher_B": self._fisher_B.tolist(),
            "anchor_A": self._anchor_A.tolist(),
            "anchor_B": self._anchor_B.tolist(),
            "quality_history": self._quality_history,
            "update_count": self._update_count,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.A = np.array(data["A"])
            self.B = np.array(data["B"])
            self._fisher_A = np.array(data.get("fisher_A", np.zeros_like(self.A)))
            self._fisher_B = np.array(data.get("fisher_B", np.zeros_like(self.B)))
            self._anchor_A = np.array(data.get("anchor_A", self.A))
            self._anchor_B = np.array(data.get("anchor_B", self.B))
            self._quality_history = data.get("quality_history", [])
            self._update_count = data.get("update_count", 0)
        except Exception as e:
            logger.error("Failed to load evolution state: %s", e)
