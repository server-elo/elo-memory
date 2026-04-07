"""
Bayesian Surprise Engine
========================

Implementation of Bayesian surprise based on KL divergence between prior and posterior distributions.
This module detects novelty and salience in input sequences to determine what should be encoded in memory.

References:
- Itti & Baldi (2009). "Bayesian Surprise Attracts Human Attention"
- EM-LLM (ICLR 2025): Event segmentation using Bayesian surprise
- Baldassano et al. (2017): HMM-based event boundary detection

Key Concepts:
- Surprise = KL(Posterior || Prior) = information gained from new observation
- High surprise → Event boundary → Memory encoding trigger
- Prediction error → Model update → Surprise calculation
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from scipy.stats import entropy


@dataclass
class SurpriseConfig:
    """Configuration for Bayesian surprise calculation."""

    window_size: int = 50  # Number of recent observations for prior estimation
    surprise_threshold: float = 0.7  # Threshold for triggering memory encoding
    kl_method: str = "symmetric"  # "forward", "reverse", or "symmetric" (JS divergence)
    use_adaptive_threshold: bool = True  # Adapt threshold based on surprise distribution
    min_observations: int = 5  # Minimum observations before calculating surprise
    smoothing_alpha: float = 0.1  # Exponential moving average smoothing
    surprise_history_len: int = 100  # Window for adaptive threshold statistics
    observation_noise: float = 0.1  # Assumed observation variance for Bayesian update


class BayesianSurpriseEngine:
    """
    Core engine for computing Bayesian surprise from sequential observations.

    Surprise is defined as the KL divergence between prior beliefs and posterior beliefs
    after observing new data. High surprise indicates unexpected/novel events.

    Mathematical formulation:
        Surprise(D|M) = KL(P(M|D) || P(M))

    Where:
        - M: Model/hypothesis space
        - D: New observation
        - P(M): Prior distribution over models
        - P(M|D): Posterior distribution after observing D
    """

    def __init__(self, input_dim: int, config: Optional[SurpriseConfig] = None):
        """
        Args:
            input_dim: Dimensionality of input observations
            config: Configuration object for surprise calculation
        """
        self.input_dim = input_dim
        self.config = config or SurpriseConfig()

        # Observation history (sliding window for prior estimation)
        self.observation_history: deque[np.ndarray] = deque(maxlen=self.config.window_size)

        # Surprise history for adaptive thresholding
        self.surprise_history: deque[float] = deque(maxlen=self.config.surprise_history_len)

        # Running statistics
        self.step_count = 0
        self.total_surprise = 0.0
        self.mean_surprise = 0.0
        self.std_surprise = 1.0

    def calculate_kl_divergence(
        self,
        prior_mean: np.ndarray,
        prior_var: np.ndarray,
        posterior_mean: np.ndarray,
        posterior_var: np.ndarray,
    ) -> float:
        """
        Calculate KL divergence between two Gaussian distributions.

        For multivariate Gaussians with diagonal covariance:
        KL(P||Q) = 0.5 * [trace(Σ_Q^-1 Σ_P) + (μ_Q - μ_P)^T Σ_Q^-1 (μ_Q - μ_P) - k + ln(det(Σ_Q)/det(Σ_P))]

        Args:
            prior_mean: Mean of prior distribution [input_dim]
            prior_var: Variance of prior distribution [input_dim]
            posterior_mean: Mean of posterior distribution [input_dim]
            posterior_var: Variance of posterior distribution [input_dim]

        Returns:
            KL divergence (scalar)
        """
        # Ensure numerical stability
        prior_var = np.maximum(prior_var, 1e-8)
        posterior_var = np.maximum(posterior_var, 1e-8)

        if self.config.kl_method == "forward":
            # KL(Posterior || Prior)
            kl = 0.5 * np.sum(
                (posterior_var / prior_var)
                + ((prior_mean - posterior_mean) ** 2) / prior_var
                - 1
                + np.log(prior_var / posterior_var)
            )
        elif self.config.kl_method == "reverse":
            # KL(Prior || Posterior)
            kl = 0.5 * np.sum(
                (prior_var / posterior_var)
                + ((posterior_mean - prior_mean) ** 2) / posterior_var
                - 1
                + np.log(posterior_var / prior_var)
            )
        else:  # symmetric (Jensen-Shannon divergence)
            # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5(P+Q)
            m_mean = 0.5 * (prior_mean + posterior_mean)
            m_var = 0.5 * (prior_var + posterior_var)

            # Calculate forward KL divergences directly (no recursion)
            kl_prior = 0.5 * np.sum(
                (prior_var / m_var)
                + ((m_mean - prior_mean) ** 2) / m_var
                - 1
                + np.log(m_var / prior_var)
            )
            kl_posterior = 0.5 * np.sum(
                (posterior_var / m_var)
                + ((m_mean - posterior_mean) ** 2) / m_var
                - 1
                + np.log(m_var / posterior_var)
            )

            kl = 0.5 * (kl_prior + kl_posterior)

        return float(max(0.0, kl))  # Ensure non-negative

    def update_prior(self, observation: np.ndarray) -> None:
        """
        Update prior distribution based on new observation.
        Uses exponential moving average for smooth updates.

        Args:
            observation: New observation [input_dim]
        """
        self.observation_history.append(observation.copy())

    def get_prior_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate prior distribution from observation history.

        Returns:
            prior_mean: Mean of prior distribution [input_dim]
            prior_var: Variance of prior distribution [input_dim]
        """
        if len(self.observation_history) < self.config.min_observations:
            # Not enough data - use uninformative prior
            return (np.zeros(self.input_dim), np.ones(self.input_dim))

        observations = np.array(self.observation_history)
        prior_mean = np.mean(observations, axis=0)
        prior_var = np.var(observations, axis=0) + 1e-8  # Add small constant for stability

        return prior_mean, prior_var

    def get_posterior_distribution(
        self, observation: np.ndarray, prior_mean: np.ndarray, prior_var: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate posterior distribution after observing new data.
        Uses Bayesian updating with Gaussian likelihood.

        Args:
            observation: New observation [input_dim]
            prior_mean: Prior mean [input_dim]
            prior_var: Prior variance [input_dim]

        Returns:
            posterior_mean: Updated mean [input_dim]
            posterior_var: Updated variance [input_dim]
        """
        # Assume observation noise (likelihood variance)
        obs_var = self.config.observation_noise * np.ones_like(observation)

        # Bayesian update for Gaussian-Gaussian conjugate prior
        # Posterior precision = Prior precision + Observation precision
        prior_precision = 1.0 / prior_var
        obs_precision = 1.0 / obs_var

        posterior_precision = prior_precision + obs_precision
        posterior_var = 1.0 / posterior_precision

        # Posterior mean = weighted average of prior mean and observation
        posterior_mean = posterior_var * (
            prior_precision * prior_mean + obs_precision * observation
        )

        return posterior_mean, posterior_var

    def compute_surprise(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Compute Bayesian surprise for a new observation.

        Args:
            observation: New observation [input_dim]

        Returns:
            Dictionary containing:
                - surprise: Raw surprise value (KL divergence)
                - normalized_surprise: Z-score normalized surprise
                - is_novel: Boolean indicating if surprise exceeds threshold
                - threshold: Current threshold value
        """
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (self.input_dim,):
            raise ValueError(
                f"Observation shape {observation.shape} does not match input_dim ({self.input_dim},)"
            )

        # Get prior distribution from history
        prior_mean, prior_var = self.get_prior_distribution()

        # Calculate posterior after observing new data
        posterior_mean, posterior_var = self.get_posterior_distribution(
            observation, prior_mean, prior_var
        )

        # Compute KL divergence (surprise)
        surprise = self.calculate_kl_divergence(
            prior_mean, prior_var, posterior_mean, posterior_var
        )

        # Update running statistics
        self.step_count += 1
        self.total_surprise += surprise

        # Exponential moving average of surprise
        alpha = self.config.smoothing_alpha
        self.mean_surprise = alpha * surprise + (1 - alpha) * self.mean_surprise

        # Update surprise history
        self.surprise_history.append(surprise)

        # Calculate normalized surprise (Z-score)
        if len(self.surprise_history) > 10:
            surprise_array = np.array(self.surprise_history)
            self.std_surprise = np.std(surprise_array)
            normalized_surprise = (surprise - self.mean_surprise) / (self.std_surprise + 1e-8)
        else:
            normalized_surprise = 0.0

        # Adaptive threshold
        if self.config.use_adaptive_threshold and len(self.surprise_history) > 20:
            # Set threshold at 75th percentile of recent surprise values
            threshold = float(np.percentile(self.surprise_history, 75))
        else:
            threshold = float(self.config.surprise_threshold)

        # Determine if observation is novel/surprising
        is_novel = surprise > threshold

        # Update prior with new observation
        self.update_prior(observation)

        return {
            "surprise": surprise,
            "normalized_surprise": normalized_surprise,
            "is_novel": is_novel,
            "threshold": threshold,
            "mean_surprise": self.mean_surprise,
            "prior_mean": prior_mean,
            "posterior_mean": posterior_mean,
        }

    def process_sequence(self, observations: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Process a sequence of observations and compute surprise for each.

        Args:
            observations: List of observations, each [input_dim]

        Returns:
            List of surprise dictionaries, one per observation
        """
        results = []

        for obs in observations:
            surprise_info = self.compute_surprise(obs)
            results.append(surprise_info)

        return results

    def get_event_boundaries(
        self, surprise_values: List[float], method: str = "peaks"
    ) -> List[int]:
        """
        Detect event boundaries from surprise signal.

        Args:
            surprise_values: List of surprise values
            method: "peaks" or "threshold"

        Returns:
            List of indices where event boundaries occur
        """
        boundaries = []

        if method == "peaks":
            # Find local maxima in surprise signal
            local_mean = np.mean(surprise_values) if surprise_values else self.mean_surprise
            for i in range(1, len(surprise_values) - 1):
                if (
                    surprise_values[i] > surprise_values[i - 1]
                    and surprise_values[i] > surprise_values[i + 1]
                    and surprise_values[i] > local_mean
                ):
                    boundaries.append(i)
        else:  # threshold
            # Simple threshold crossing
            for i, surprise in enumerate(surprise_values):
                if surprise > self.config.surprise_threshold:
                    boundaries.append(i)

        return boundaries

    def reset(self) -> None:
        """Reset the surprise engine to initial state."""
        self.observation_history.clear()
        self.surprise_history.clear()
        self.step_count = 0
        self.total_surprise = 0.0
        self.mean_surprise = 0.0
        self.std_surprise = 1.0
