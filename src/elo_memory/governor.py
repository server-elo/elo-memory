"""
Autonomous Memory Governor
==========================

Meta-agent that uses reinforcement learning (contextual multi-armed bandit
with Thompson Sampling) to optimize memory decisions: encode, skip, promote,
demote, or prune.  Learns from delayed reward signals — was the memory
actually retrieved later?  Was a skipped observation ever needed?

The governor "learns how to learn" and progressively replaces hand-tuned
thresholds with data-driven policies.

References:
- Agrawal & Goyal (2013): Thompson Sampling for Contextual Bandits
- Auer et al. (2002): UCB for Multi-Armed Bandits
"""

from __future__ import annotations

import ast
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Actions ──────────────────────────────────────────────────────


class Action(IntEnum):
    ENCODE = 0  # Store normally
    SKIP = 1  # Do not store
    PROMOTE = 2  # Store with boosted importance
    DEMOTE = 3  # Reduce importance of existing similar memories
    PRUNE = 4  # Mark low-value existing memories for removal


ACTION_NAMES = {a: a.name for a in Action}


# ── Config ───────────────────────────────────────────────────────


@dataclass
class GovernorConfig:
    """Configuration for the Memory Governor."""

    # Context discretization bins (per feature)
    n_surprise_bins: int = 3  # low / medium / high
    n_similarity_bins: int = 3  # low / medium / high
    n_importance_bins: int = 2  # low / high

    # Reward tracking
    reward_window_hours: float = 24.0  # How long to wait for retrieval reward
    reward_check_interval: int = 50  # Check pending rewards every N decisions

    # Thompson Sampling priors
    prior_alpha: float = 1.0  # Beta prior α (successes)
    prior_beta: float = 1.0  # Beta prior β (failures)

    # Exploration
    min_exploration_rate: float = 0.05  # Minimum random exploration

    # Promotion / demotion multipliers
    promote_boost: float = 1.5
    demote_factor: float = 0.6

    # Persistence
    persist: bool = True


# ── Context Features ─────────────────────────────────────────────


@dataclass
class DecisionContext:
    """Features describing the current memory decision context."""

    surprise: float = 0.0
    max_similarity: float = 0.0  # Cosine sim to nearest existing memory
    importance: float = 0.5
    entity_count: int = 0
    storage_utilization: float = 0.0  # episodes / max_episodes
    recency_score: float = 1.0  # How recent is the last similar memory
    topic_overlap: float = 0.0  # Fraction of topics already covered

    def to_bin_key(self, config: GovernorConfig) -> Tuple[int, int, int]:
        """Discretize continuous features into bin indices."""
        s_bin = min(int(self.surprise * config.n_surprise_bins), config.n_surprise_bins - 1)
        sim_bin = min(
            int(self.max_similarity * config.n_similarity_bins), config.n_similarity_bins - 1
        )
        imp_bin = min(int(self.importance * config.n_importance_bins), config.n_importance_bins - 1)
        return (s_bin, sim_bin, imp_bin)


# ── Pending Decision ─────────────────────────────────────────────


@dataclass
class PendingDecision:
    """Tracks a decision awaiting its reward signal."""

    action: Action
    bin_key: Tuple[int, int, int]
    episode_id: Optional[str]
    timestamp: float  # time.time()
    embedding_hash: str = ""  # For tracking skipped observations


# ── Governor ─────────────────────────────────────────────────────


class MemoryGovernor:
    """
    Contextual bandit that learns optimal memory management policies.

    For each (context_bin, action) pair it maintains a Beta(α, β) distribution.
    Thompson Sampling draws from each action's posterior to select the best one.
    Rewards arrive with delay — retrieval within the reward window = success.
    """

    def __init__(
        self, config: Optional[GovernorConfig] = None, persistence_path: Optional[str] = None
    ):
        self.config = config or GovernorConfig()

        # Beta distribution parameters per (bin_key, action)
        # Key: (bin_key, action_int) → {"alpha": float, "beta": float}
        self._params: Dict[Tuple, Dict[str, float]] = {}

        # Pending decisions awaiting reward
        self._pending: List[PendingDecision] = []
        self._lock = threading.Lock()

        # Statistics
        self.total_decisions = 0
        self.total_rewards = 0
        self._action_counts: Dict[int, int] = {a.value: 0 for a in Action}

        # Retrieved episode IDs (set by external hook, pruned on resolve)
        self._retrieved_ids: set = set()
        self._retrieved_ids_max = 10000

        # Persistence
        self._persistence_path: Optional[Path] = None
        if persistence_path:
            self._persistence_path = Path(persistence_path) / "governor"
            self._persistence_path.mkdir(parents=True, exist_ok=True)
            self._load()

    # ── Core API ─────────────────────────────────────────────────

    def decide(self, context: DecisionContext) -> Action:
        """Select an action for the given context using Thompson Sampling."""
        bin_key = context.to_bin_key(self.config)

        # Exploration floor
        if np.random.random() < self.config.min_exploration_rate:
            action = Action(np.random.randint(0, len(Action)))
        else:
            # Thompson Sampling: sample from each action's Beta posterior
            best_action = Action.ENCODE
            best_sample = -1.0
            for a in Action:
                params = self._get_params(bin_key, a)
                sample = np.random.beta(params["alpha"], params["beta"])
                if sample > best_sample:
                    best_sample = sample
                    best_action = a
            action = best_action

        self._action_counts[action.value] += 1
        self.total_decisions += 1

        # Periodic reward check
        if self.total_decisions % self.config.reward_check_interval == 0:
            self._resolve_pending()

        return action

    def record_decision(
        self,
        action: Action,
        context: DecisionContext,
        episode_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Record a decision for delayed reward tracking."""
        bin_key = context.to_bin_key(self.config)
        emb_hash = ""
        if embedding is not None:
            emb_hash = str(hash(embedding.tobytes()))

        pending = PendingDecision(
            action=action,
            bin_key=bin_key,
            episode_id=episode_id,
            timestamp=time.time(),
            embedding_hash=emb_hash,
        )
        with self._lock:
            self._pending.append(pending)

    def record_retrieval(self, episode_id: str) -> None:
        """Notify the governor that an episode was retrieved (positive reward)."""
        self._retrieved_ids.add(episode_id)

    def apply_action(
        self,
        action: Action,
        importance: float,
    ) -> Tuple[bool, float]:
        """
        Apply the governor's decision.

        Returns:
            (should_store, adjusted_importance)
        """
        if action == Action.SKIP:
            return False, 0.0
        elif action == Action.PROMOTE:
            return True, min(1.0, importance * self.config.promote_boost)
        elif action == Action.DEMOTE:
            return True, importance * self.config.demote_factor
        else:  # ENCODE or PRUNE
            return True, importance

    # ── Reward Resolution ────────────────────────────────────────

    def _resolve_pending(self) -> None:
        """Check pending decisions and assign rewards."""
        now = time.time()
        cutoff = now - (self.config.reward_window_hours * 3600)
        resolved = []
        remaining = []

        with self._lock:
            for p in self._pending:
                if p.timestamp < cutoff:
                    resolved.append(p)
                else:
                    remaining.append(p)
            self._pending = remaining

        for p in resolved:
            reward = self._compute_reward(p)
            self._update_params(p.bin_key, p.action, reward)
            self.total_rewards += 1

        # Prune retrieved IDs to prevent unbounded growth.
        # Keep the most recent half to avoid corrupting pending reward signals.
        if len(self._retrieved_ids) > self._retrieved_ids_max:
            n_to_remove = len(self._retrieved_ids) - (self._retrieved_ids_max // 2)
            removed = 0
            for item in list(self._retrieved_ids):
                if removed >= n_to_remove:
                    break
                self._retrieved_ids.discard(item)
                removed += 1

    def _compute_reward(self, decision: PendingDecision) -> float:
        """Compute reward for a resolved decision."""
        if decision.action == Action.ENCODE or decision.action == Action.PROMOTE:
            # Success if the episode was retrieved
            return 1.0 if decision.episode_id in self._retrieved_ids else 0.0
        elif decision.action == Action.SKIP:
            # For skipped content, we can't directly verify retrieval without
            # episode_id. Use neutral reward (0.5) since we lack a signal.
            # This avoids the degenerate "always skip" policy.
            return 0.5
        elif decision.action == Action.DEMOTE:
            # Reward if still retrieved (we reduced cost without losing utility)
            return 0.8 if decision.episode_id in self._retrieved_ids else 0.5
        elif decision.action == Action.PRUNE:
            # Success if pruned memory was never needed
            return 0.0 if decision.episode_id in self._retrieved_ids else 1.0
        return 0.5

    # ── Beta Distribution Management ─────────────────────────────

    def _get_params(self, bin_key: Tuple, action: Action) -> Dict[str, float]:
        key = (bin_key, action.value)
        if key not in self._params:
            self._params[key] = {
                "alpha": self.config.prior_alpha,
                "beta": self.config.prior_beta,
            }
        return self._params[key]

    def _update_params(self, bin_key: Tuple[int, ...], action: Action, reward: float) -> None:
        params = self._get_params(bin_key, action)
        params["alpha"] += reward
        params["beta"] += 1.0 - reward

    # ── Policy Inspection ────────────────────────────────────────

    def get_policy_summary(self) -> Dict[str, Any]:
        """Human-readable summary of learned policies."""
        summary: Dict[str, Any] = {
            "total_decisions": self.total_decisions,
            "total_rewards": self.total_rewards,
            "action_distribution": {
                ACTION_NAMES[Action(k)]: v for k, v in self._action_counts.items()
            },
            "learned_preferences": [],
        }

        # For each context bin, show the preferred action
        seen_bins: set = set()
        for (bin_key, action_int), params in self._params.items():
            if bin_key not in seen_bins:
                seen_bins.add(bin_key)
                # Find best action for this bin
                best_action = Action.ENCODE
                best_mean = 0.0
                for a in Action:
                    p = self._get_params(bin_key, a)
                    mean = p["alpha"] / (p["alpha"] + p["beta"])
                    if mean > best_mean:
                        best_mean = mean
                        best_action = a
                summary["learned_preferences"].append(
                    {
                        "context_bin": bin_key,
                        "preferred_action": ACTION_NAMES[best_action],
                        "confidence": best_mean,
                    }
                )

        return summary

    # ── Persistence ──────────────────────────────────────────────

    def save(self) -> None:
        if not self._persistence_path:
            return
        state = {
            "params": {str(k): v for k, v in self._params.items()},
            "action_counts": self._action_counts,
            "total_decisions": self.total_decisions,
            "total_rewards": self.total_rewards,
        }
        path = self._persistence_path / "governor_state.json"
        try:
            with open(path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error("Failed to save governor state: %s", e)

    def _load(self) -> None:
        if not self._persistence_path:
            return
        path = self._persistence_path / "governor_state.json"
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                state = json.load(f)
            # Reconstruct params with tuple keys
            for k_str, v in state.get("params", {}).items():
                # Parse string representation of tuple key
                try:
                    key = ast.literal_eval(k_str)
                    self._params[key] = v
                except (ValueError, SyntaxError):
                    continue
            self._action_counts = {int(k): v for k, v in state.get("action_counts", {}).items()}
            self.total_decisions = state.get("total_decisions", 0)
            self.total_rewards = state.get("total_rewards", 0)
        except Exception as e:
            logger.warning("Failed to load governor state: %s", e)
