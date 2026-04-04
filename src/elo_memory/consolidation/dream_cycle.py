"""
Dream Consolidation Cycle
=========================

Periodic "sleep" mode where the agent replays compressed experiences,
generates synthetic augmentations (like human REM), and performs
self-supervised abstraction.

Outputs: higher-level principles, skills, pruned low-value memories.
Makes consolidation creative instead of mechanical.

References:
- Walker & Stickgold (2004): Sleep-Dependent Memory Processing
- Lewis & Durrant (2011): Overlapping Memory Replay During Sleep
- Deperrois et al. (2024): Generative Replay in REM Sleep
"""

from __future__ import annotations

import uuid as _uuid
import logging
import time as _time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..memory.episodic_store import Episode, EpisodicMemoryStore
from ..memory.forgetting import ForgettingEngine

logger = logging.getLogger(__name__)


@dataclass
class DreamConfig:
    """Configuration for dream consolidation."""
    # Replay
    replay_fraction: float = 0.3        # Fraction of episodes to replay per cycle
    replay_priority_alpha: float = 0.7  # Importance weighting for replay selection

    # Synthetic augmentation
    augmentation_noise_scale: float = 0.05   # Noise for synthetic episode generation
    augmentations_per_episode: int = 2       # Synthetic variants per replayed episode
    interpolation_count: int = 3             # Cross-episode interpolations

    # Abstraction
    cluster_threshold: float = 0.7   # Cosine threshold for clustering
    min_cluster_size: int = 3        # Min episodes to form a principle

    # Pruning
    prune_activation_threshold: float = 0.05   # Below this, mark for pruning
    max_prune_fraction: float = 0.1            # Never prune more than 10% per cycle

    # Skills
    skill_repetition_threshold: int = 3  # Min times a pattern appears to become a skill


@dataclass
class DreamResult:
    """Output of a single dream cycle."""
    episodes_replayed: int = 0
    synthetic_generated: int = 0
    principles_extracted: int = 0
    skills_learned: int = 0
    episodes_pruned: int = 0
    duration_seconds: float = 0.0
    principles: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)


class DreamConsolidation:
    """
    Creative consolidation that mimics human sleep stages.

    Stage 1 (NREM): Prioritized replay — strengthen important memories
    Stage 2 (REM):  Synthetic augmentation — generate variations
    Stage 3 (Deep): Abstraction — extract principles from clusters
    Stage 4 (Dawn): Pruning — remove low-activation memories
    """

    def __init__(self, config: Optional[DreamConfig] = None):
        self.config = config or DreamConfig()
        self._forgetting = ForgettingEngine()

        # Accumulated outputs across all dream cycles
        self.all_principles: List[Dict[str, Any]] = []
        self.all_skills: List[Dict[str, Any]] = []
        self.total_cycles = 0

    def dream(
        self,
        store: EpisodicMemoryStore,
        update_callback: Optional[Callable] = None,
    ) -> DreamResult:
        """
        Run a full dream consolidation cycle.

        Args:
            store: The episodic memory store to consolidate
            update_callback: Optional callback for strengthening episodes
        """
        start = _time.time()
        result = DreamResult()

        episodes = store.episodes
        if not episodes:
            return result

        # ── Stage 1: NREM — Prioritized Replay ──────────────────
        replay_batch = self._select_for_replay(episodes)
        for ep in replay_batch:
            if update_callback:
                update_callback(ep)
            ep.importance = min(1.0, ep.importance * 1.05)
        result.episodes_replayed = len(replay_batch)

        # ── Stage 2: REM — Synthetic Augmentation ────────────────
        synthetics = self._generate_synthetics(replay_batch, store)
        result.synthetic_generated = len(synthetics)

        # ── Stage 3: Deep — Abstraction & Principles ─────────────
        principles = self._extract_principles(episodes)
        result.principles = principles
        result.principles_extracted = len(principles)
        self.all_principles.extend(principles)

        # ── Stage 3b: Skill Extraction ───────────────────────────
        skills = self._extract_skills(episodes)
        result.skills = skills
        result.skills_learned = len(skills)
        self.all_skills.extend(skills)

        # ── Stage 4: Dawn — Pruning ─────────────────────────────
        pruned = self._prune_low_value(episodes, store)
        result.episodes_pruned = pruned

        result.duration_seconds = _time.time() - start
        self.total_cycles += 1

        logger.info(
            "Dream cycle %d complete: replayed=%d, synthetic=%d, "
            "principles=%d, skills=%d, pruned=%d (%.1fs)",
            self.total_cycles,
            result.episodes_replayed,
            result.synthetic_generated,
            result.principles_extracted,
            result.skills_learned,
            result.episodes_pruned,
            result.duration_seconds,
        )

        return result

    # ── Stage 1: Replay Selection ────────────────────────────────

    def _select_for_replay(self, episodes: List[Episode]) -> List[Episode]:
        """Select episodes for prioritized replay (NREM-like)."""
        n = max(1, int(len(episodes) * self.config.replay_fraction))

        # Priority = importance ^ alpha * surprise ^ (1-alpha)
        alpha = self.config.replay_priority_alpha
        priorities_list = []
        for ep in episodes:
            p = (ep.importance ** alpha) * ((ep.surprise + 0.1) ** (1 - alpha))
            priorities_list.append(p)

        priorities = np.array(priorities_list)
        total = priorities.sum()
        if total == 0 or not np.isfinite(total):
            probs = np.ones(len(priorities)) / len(priorities)
        else:
            probs = priorities / total

        indices = np.random.choice(
            len(episodes), size=min(n, len(episodes)), replace=False, p=probs
        )
        return [episodes[i] for i in indices]

    # ── Stage 2: Synthetic Augmentation ──────────────────────────

    def _generate_synthetics(
        self,
        replay_batch: List[Episode],
        store: EpisodicMemoryStore,
    ) -> List[Episode]:
        """Generate synthetic variations of replayed episodes (REM-like).

        Synthetic episodes are kept in-memory only and NOT stored into the
        EpisodicMemoryStore to avoid unintended capacity offloading triggers.
        """
        synthetics: List[Episode] = []
        max_synthetic = max(1, len(replay_batch) * self.config.augmentations_per_episode)

        for ep in replay_batch:
            if ep.embedding is None:
                continue

            # Noisy variants — create Episode objects without storing
            for _ in range(self.config.augmentations_per_episode):
                if len(synthetics) >= max_synthetic:
                    break
                noise = np.random.randn(*ep.embedding.shape) * self.config.augmentation_noise_scale
                aug_embedding = ep.embedding + noise
                norm = np.linalg.norm(aug_embedding)
                if norm > 0:
                    aug_embedding = aug_embedding / norm

                synthetic_ep = Episode(
                    episode_id=f"synthetic-{_uuid.uuid4().hex[:8]}",
                    content=ep.content,
                    embedding=aug_embedding,
                    surprise=ep.surprise * 0.5,
                    importance=ep.importance * 0.5,
                    timestamp=datetime.now(timezone.utc),
                    entities=list(ep.entities),
                    metadata={
                        **ep.metadata,
                        "_synthetic": True,
                        "_source_episode": ep.episode_id,
                    },
                )
                synthetics.append(synthetic_ep)

        # Cross-episode interpolation (creative recombination)
        if len(replay_batch) >= 2:
            for _ in range(self.config.interpolation_count):
                if len(synthetics) >= max_synthetic * 2:
                    break
                i, j = np.random.choice(len(replay_batch), size=2, replace=False)
                ep_a, ep_b = replay_batch[i], replay_batch[j]
                if ep_a.embedding is None or ep_b.embedding is None:
                    continue

                # SLERP interpolation
                alpha = np.random.uniform(0.3, 0.7)
                interp = self._slerp(ep_a.embedding, ep_b.embedding, alpha)

                # Merge metadata
                merged_entities = list(set(ep_a.entities + ep_b.entities))
                synthetic_ep = Episode(
                    episode_id=f"synthetic-interp-{_uuid.uuid4().hex[:8]}",
                    content={"text": f"[dream synthesis] {ep_a.metadata.get('text', '')} + {ep_b.metadata.get('text', '')}"},
                    embedding=interp,
                    surprise=0.0,
                    importance=0.0,
                    timestamp=datetime.now(timezone.utc),
                    entities=merged_entities,
                    metadata={
                        "_synthetic": True,
                        "_interpolation": True,
                        "_sources": [ep_a.episode_id, ep_b.episode_id],
                    },
                )
                synthetics.append(synthetic_ep)

        return synthetics

    @staticmethod
    def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two unit vectors."""
        v0_norm = v0 / (np.linalg.norm(v0) + 1e-8)
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
        omega = np.arccos(dot)
        if abs(omega) < 1e-6:
            result: np.ndarray = (1 - t) * v0_norm + t * v1_norm
            return result
        so = np.sin(omega)
        result2: np.ndarray = np.sin((1 - t) * omega) / so * v0_norm + np.sin(t * omega) / so * v1_norm
        return result2

    # ── Stage 3: Abstraction ─────────────────────────────────────

    def _extract_principles(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """Extract higher-level principles from episode clusters."""
        # Cluster episodes by embedding similarity
        clusters = self._cluster_episodes(episodes)
        principles = []

        for cluster in clusters:
            if len(cluster) < self.config.min_cluster_size:
                continue

            # Extract common themes
            all_topics: List[str] = []
            all_entities: List[str] = []
            texts: List[str] = []

            for ep in cluster:
                all_topics.extend(ep.metadata.get("topics", []))
                all_entities.extend(ep.entities)
                if isinstance(ep.content, dict):
                    t = ep.content.get("text", "")
                    if t:
                        texts.append(t)

            # Find most common themes
            topic_counts = Counter(all_topics)
            entity_counts = Counter(all_entities)

            common_topics = [t for t, c in topic_counts.most_common(3)]
            common_entities = [e for e, c in entity_counts.most_common(5)]

            # Compute centroid embedding
            embeddings = [ep.embedding for ep in cluster if ep.embedding is not None]
            centroid = np.mean(embeddings, axis=0) if embeddings else None

            principle = {
                "type": "principle",
                "topics": common_topics,
                "entities": common_entities,
                "episode_count": len(cluster),
                "avg_importance": float(np.mean([ep.importance for ep in cluster])),
                "centroid": centroid.tolist() if centroid is not None else None,
                "sample_texts": texts[:3],
            }
            principles.append(principle)

        return principles

    def _cluster_episodes(self, episodes: List[Episode]) -> List[List[Episode]]:
        """Simple agglomerative clustering by embedding similarity."""
        if not episodes:
            return []

        # Filter to episodes with embeddings
        with_emb = [(i, ep) for i, ep in enumerate(episodes) if ep.embedding is not None]
        if len(with_emb) < 2:
            return []

        # Greedy single-linkage clustering
        assigned = [False] * len(with_emb)
        clusters: List[List[Episode]] = []

        for i, (_, ep_i) in enumerate(with_emb):
            if assigned[i]:
                continue
            cluster = [ep_i]
            assigned[i] = True

            for j in range(i + 1, len(with_emb)):
                if assigned[j]:
                    continue
                _, ep_j = with_emb[j]
                assert ep_i.embedding is not None and ep_j.embedding is not None
                sim = float(np.dot(ep_i.embedding, ep_j.embedding))
                if sim >= self.config.cluster_threshold:
                    cluster.append(ep_j)
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    # ── Stage 3b: Skill Extraction ───────────────────────────────

    def _extract_skills(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """Identify repeated action patterns as "skills"."""
        # Look for repeated entity + topic combinations
        pattern_counts: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)

        for ep in episodes:
            topics = tuple(sorted(ep.metadata.get("topics", [])))
            entities_key = tuple(sorted(ep.entities[:3]))  # Top 3 entities
            if (topics or entities_key) and ep.episode_id is not None:
                key = (topics, entities_key)
                pattern_counts[key].append(ep.episode_id)

        skills = []
        for (topics, entities), ep_ids in pattern_counts.items():
            if len(ep_ids) >= self.config.skill_repetition_threshold:
                skills.append({
                    "type": "skill",
                    "topics": list(topics),
                    "entities": list(entities),
                    "repetitions": len(ep_ids),
                    "episode_ids": ep_ids,
                })

        return skills

    # ── Stage 4: Pruning ─────────────────────────────────────────

    def _prune_low_value(
        self,
        episodes: List[Episode],
        store: EpisodicMemoryStore,
    ) -> int:
        """Mark low-activation episodes for removal."""
        if not episodes:
            return 0

        max_prune = int(len(episodes) * self.config.max_prune_fraction)
        prune_candidates = []

        for ep in episodes:
            if ep.metadata.get("_synthetic"):
                continue  # Don't prune synthetics immediately
            activation = self._forgetting.compute_activation(
                ep.importance,
                ep.timestamp,
                rehearsal_count=ep.metadata.get("_retrieval_count", 0),
            )
            if activation < self.config.prune_activation_threshold:
                prune_candidates.append((ep, activation))

        # Sort by activation (lowest first) and prune up to max
        prune_candidates.sort(key=lambda x: x[1])
        pruned = 0

        for ep, _ in prune_candidates[:max_prune]:
            ep.importance = 0.0
            ep.metadata["_dream_pruned"] = True
            pruned += 1

        return pruned

    # ── Statistics ───────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "total_principles": len(self.all_principles),
            "total_skills": len(self.all_skills),
        }
