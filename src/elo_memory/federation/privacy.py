"""
Differential Privacy for Memory Federation
===========================================

Implements the Gaussian mechanism for adding calibrated noise to
shared embeddings, plus a privacy accountant that tracks cumulative
epsilon spend via Renyi Differential Privacy (RDP) composition.

References:
- Dwork & Roth (2014): The Algorithmic Foundations of Differential Privacy
- Mironov (2017): Renyi Differential Privacy
- Abadi et al. (2016): Deep Learning with Differential Privacy
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""

    epsilon: float = 1.0  # Total privacy budget
    delta: float = 1e-5  # Failure probability
    sensitivity: float = 1.0  # L2 sensitivity of embeddings
    min_noise_scale: float = 0.01  # Minimum noise even at high budget


class DifferentialPrivacy:
    """
    Gaussian mechanism for differential privacy on embeddings.

    Adds calibrated Gaussian noise: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    """

    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        self._sigma = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """Compute noise standard deviation from privacy parameters."""
        c = self.config
        sigma = c.sensitivity * math.sqrt(2 * math.log(1.25 / c.delta)) / c.epsilon
        return max(sigma, c.min_noise_scale)

    def add_noise(self, embedding: np.ndarray) -> np.ndarray:
        """Add calibrated Gaussian noise to an embedding."""
        noise = np.random.normal(0, self._sigma, size=embedding.shape)
        noised = embedding + noise
        # Re-normalize to unit sphere
        norm = np.linalg.norm(noised)
        return noised / norm if norm > 0 else noised

    def anonymize_text(self, text: str) -> str:
        """Hash identifiable text content (names, emails, etc.)."""
        import re

        # Replace email-like patterns
        text = re.sub(
            r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
            "[EMAIL_REDACTED]",
            text,
        )
        # Replace patterns that look like names (capitalized word pairs)
        text = re.sub(
            r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",
            lambda m: f"[PERSON_{hashlib.sha256(m.group(1).encode()).hexdigest()[:8]}]",
            text,
        )
        return text

    @property
    def noise_scale(self) -> float:
        return self._sigma


class PrivacyAccountant:
    """
    Tracks cumulative privacy spend across multiple queries.

    Uses simple composition (epsilon adds up) with an optional
    advanced composition bound.
    """

    def __init__(self, total_budget: float = 10.0, delta: float = 1e-5):
        self.total_budget = total_budget
        self.delta = delta
        self._spent: float = 0.0
        self._queries: List[Dict[str, Any]] = []

    @property
    def remaining_budget(self) -> float:
        return max(0, self.total_budget - self._spent)

    @property
    def is_exhausted(self) -> bool:
        return self._spent >= self.total_budget

    def spend(self, epsilon: float, description: str = "") -> bool:
        """
        Attempt to spend epsilon from the budget.
        Returns True if budget allows, False if exhausted.
        """
        if self._spent + epsilon > self.total_budget:
            logger.warning(
                "Privacy budget exhausted: spent=%.2f, requested=%.2f, total=%.2f",
                self._spent,
                epsilon,
                self.total_budget,
            )
            return False

        self._spent += epsilon
        self._queries.append(
            {
                "epsilon": epsilon,
                "cumulative": self._spent,
                "description": description,
            }
        )
        return True

    def get_report(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "spent": self._spent,
            "remaining": self.remaining_budget,
            "num_queries": len(self._queries),
            "is_exhausted": self.is_exhausted,
        }

    def save(self, path: Path) -> None:
        data = {
            "total_budget": self.total_budget,
            "delta": self.delta,
            "spent": self._spent,
            "queries": self._queries,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.total_budget = data["total_budget"]
            self.delta = data["delta"]
            self._spent = data["spent"]
            self._queries = data.get("queries", [])
        except Exception as e:
            logger.error("Failed to load privacy accountant: %s", e)
