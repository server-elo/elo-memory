"""
Memory Forgetting and Decay
============================

Time-based forgetting following Ebbinghaus forgetting curve.
Implements activation-based memory decay with rehearsal.

References:
- Ebbinghaus (1885): Forgetting curve
- Anderson & Schooler (1991): Rational analysis of memory
- Wixted & Ebbesen (1991): Power law of forgetting
"""

import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass


@dataclass
class ForgettingConfig:
    """Configuration for memory forgetting."""

    decay_rate: float = 0.5  # How fast memories decay
    rehearsal_boost: float = 1.5  # Logarithmic rehearsal scaling factor
    min_activation: float = 0.1  # Minimum activation threshold
    use_power_law: bool = True  # Use power law vs exponential decay
    forgetting_steepness: float = 5.0  # Sigmoid steepness for forgetting probability


class ForgettingEngine:
    """
    Manages memory decay and forgetting.

    Activation follows power law: A(t) = A0 * (1 + t)^(-d)
    where d is decay rate, t is time since encoding.

    Rehearsal resets the clock and boosts activation.
    """

    def __init__(self, config: Optional[ForgettingConfig] = None):
        self.config = config or ForgettingConfig()

    def compute_activation(
        self,
        initial_activation: float,
        timestamp: datetime,
        rehearsal_count: int = 0,
        current_time: datetime = None,
    ) -> float:
        """
        Compute current memory activation.

        Args:
            initial_activation: Activation at encoding (e.g., surprise score)
            timestamp: When memory was encoded
            rehearsal_count: Number of times memory was retrieved
            current_time: Current time (defaults to now)

        Returns:
            Current activation level
        """
        current_time = current_time or datetime.now(timezone.utc)
        # Handle naive timestamps for backward compatibility
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        time_elapsed = (current_time - timestamp).total_seconds() / 3600  # hours

        # Apply rehearsal boost (logarithmic to prevent memory immortality)
        boosted_activation = initial_activation * (1 + self.config.rehearsal_boost * np.log1p(rehearsal_count))

        # Apply decay
        if self.config.use_power_law:
            # Power law decay: A(t) = A0 * (1 + t)^(-d)
            activation = boosted_activation * ((1 + time_elapsed) ** (-self.config.decay_rate))
        else:
            # Exponential decay: A(t) = A0 * e^(-dt)
            activation = boosted_activation * np.exp(-self.config.decay_rate * time_elapsed)

        return activation

    def should_forget(self, activation: float) -> bool:
        """
        Determine if memory should be forgotten.

        Args:
            activation: Current activation level

        Returns:
            True if activation below threshold
        """
        return activation < self.config.min_activation

    def get_forgetting_probability(self, activation: float) -> float:
        """
        Probability of forgetting given activation.

        Args:
            activation: Current activation level

        Returns:
            Probability in [0, 1]
        """
        # Sigmoid function around min_activation threshold
        x = (activation - self.config.min_activation) / self.config.min_activation
        prob = 1 / (1 + np.exp(x * self.config.forgetting_steepness))
        return prob
