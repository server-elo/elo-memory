"""
Memory Consolidation Engine
============================

Sleep-like memory replay and consolidation mechanism.
Implements Systems Consolidation Theory (Squire & Alvarez 1995) and
memory replay for continual learning (McClelland et al. 1995).

Key mechanisms:
1. Prioritized replay of high-surprise episodes
2. Schema extraction from repeated patterns
3. Interference reduction via interleaved replay
4. Memory strengthening through reactivation

References:
- Squire & Alvarez (1995): Systems consolidation theory
- McClelland et al. (1995): Complementary learning systems
- Kumaran et al. (2016): Schema-based memory consolidation
- Schaul et al. (2016): Prioritized experience replay (DQN)
"""

import numpy as np
from typing import Callable, List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import heapq
from collections import defaultdict


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""

    replay_batch_size: int = 32
    replay_iterations: int = 100
    prioritization_alpha: float = 0.6  # Priority exponent (0=uniform, 1=greedy)
    importance_beta: float = 0.4  # Importance sampling correction
    schema_threshold: int = 3  # Min occurrences to form schema
    consolidation_interval: timedelta = timedelta(hours=8)  # Sleep cycle
    common_entity_ratio: float = (
        0.5  # Entity must appear in this fraction of episodes to be "common"
    )


class MemoryConsolidationEngine:
    """
    Consolidates episodic memories into schematic knowledge.

    Mimics sleep-dependent memory consolidation in the brain:
    - Replay high-priority episodes
    - Extract common patterns (schemas)
    - Reduce interference between similar memories
    - Strengthen important memories
    """

    def __init__(self, config: Optional[ConsolidationConfig] = None):
        self.config = config or ConsolidationConfig()

        # Schemas: abstracted patterns from multiple episodes
        self.schemas = []  # List of (pattern, frequency, episodes)

        # Consolidation stats
        self.last_consolidation = datetime.now(timezone.utc)
        self.total_replays = 0
        self.schemas_formed = 0

    def should_consolidate(self) -> bool:
        """Check if it's time for consolidation cycle."""
        time_since_last = datetime.now(timezone.utc) - self.last_consolidation
        return time_since_last >= self.config.consolidation_interval

    def prioritize_episodes(
        self, episodes: List, priorities: Optional[np.ndarray] = None
    ) -> Tuple[List, np.ndarray]:
        """
        Prioritize episodes for replay based on surprise and recency.

        Priority = (surprise * recency_weight) ^ alpha

        Args:
            episodes: List of episodes
            priorities: Optional pre-computed priorities

        Returns:
            Prioritized episodes and their sampling probabilities
        """
        if priorities is None:
            priorities = np.zeros(len(episodes))
            now = datetime.now(timezone.utc)

            for i, ep in enumerate(episodes):
                # Surprise component
                surprise_score = ep.surprise if hasattr(ep, "surprise") else 1.0

                # Recency component (more recent = higher priority)
                ep_ts = ep.timestamp
                if ep_ts.tzinfo is None:
                    ep_ts = ep_ts.replace(tzinfo=timezone.utc)
                time_diff = (now - ep_ts).total_seconds()
                recency_score = np.exp(-time_diff / (24 * 3600))  # Decay over days

                # Combined priority
                priorities[i] = (surprise_score * recency_score) ** self.config.prioritization_alpha

        # Normalize to probabilities
        priorities = priorities + 1e-6  # Avoid zero probabilities
        probabilities = priorities / np.sum(priorities)

        return episodes, probabilities

    def sample_for_replay(self, episodes: List, probabilities: np.ndarray, batch_size: int) -> List:
        """
        Sample episodes for replay using prioritized sampling.

        Args:
            episodes: List of episodes
            probabilities: Sampling probabilities
            batch_size: Number of episodes to sample

        Returns:
            Sampled episodes
        """
        if len(episodes) <= batch_size:
            return episodes

        # Sample without replacement
        indices = np.random.choice(
            len(episodes), size=min(batch_size, len(episodes)), replace=False, p=probabilities
        )

        return [episodes[i] for i in indices]

    def extract_schemas(self, episodes: List) -> List[Dict]:
        """
        Extract common patterns (schemas) from episodes.

        Schema = frequently co-occurring features across episodes.
        E.g., "Meeting" schema = {location: conference_room, entities: [Alice, Bob], time: afternoon}

        Args:
            episodes: List of episodes

        Returns:
            List of extracted schemas
        """
        # Group episodes by location
        location_groups = defaultdict(list)
        for ep in episodes:
            if hasattr(ep, "location") and ep.location:
                location_groups[ep.location].append(ep)

        # Extract schemas from frequent patterns
        schemas = []
        for location, location_episodes in location_groups.items():
            if len(location_episodes) >= self.config.schema_threshold:
                # Find common entities
                entity_counts = defaultdict(int)
                for ep in location_episodes:
                    if hasattr(ep, "entities"):
                        for entity in ep.entities:
                            entity_counts[entity] += 1

                # Common entities appear in >50% of episodes
                threshold = len(location_episodes) * self.config.common_entity_ratio
                common_entities = [
                    entity for entity, count in entity_counts.items() if count >= threshold
                ]

                # Average surprise (schema importance)
                avg_surprise = np.mean(
                    [ep.surprise if hasattr(ep, "surprise") else 1.0 for ep in location_episodes]
                )

                schema = {
                    "type": "location_pattern",
                    "location": location,
                    "common_entities": common_entities,
                    "frequency": len(location_episodes),
                    "avg_surprise": avg_surprise,
                    "episode_ids": [ep.episode_id for ep in location_episodes],
                }
                schemas.append(schema)

        # Topic-based schema extraction: keyword frequency across episodes with text content
        keyword_counts = defaultdict(int)
        text_episode_count = 0
        for ep in episodes:
            text = None
            if hasattr(ep, "content") and isinstance(ep.content, dict):
                text = ep.content.get("text")
            if text:
                text_episode_count += 1
                words = set(text.lower().split())
                for word in words:
                    if len(word) > 2:  # skip very short words
                        keyword_counts[word] += 1

        if text_episode_count >= self.config.schema_threshold:
            threshold = text_episode_count * self.config.common_entity_ratio
            common_keywords = [word for word, count in keyword_counts.items() if count >= threshold]
            if common_keywords:
                schemas.append(
                    {
                        "type": "topic_pattern",
                        "location": None,
                        "common_entities": common_keywords,
                        "frequency": text_episode_count,
                        "avg_surprise": float(
                            np.mean(
                                [
                                    ep.surprise if hasattr(ep, "surprise") else 1.0
                                    for ep in episodes
                                    if hasattr(ep, "content")
                                    and isinstance(ep.content, dict)
                                    and ep.content.get("text")
                                ]
                            )
                        ),
                        "episode_ids": [
                            ep.episode_id
                            for ep in episodes
                            if hasattr(ep, "content")
                            and isinstance(ep.content, dict)
                            and ep.content.get("text")
                        ],
                    }
                )

        return schemas

    def consolidate(self, episodes: List, update_callback: Optional[Callable] = None) -> Dict:
        """
        Run full consolidation cycle.

        Steps:
        1. Prioritize episodes by surprise and recency
        2. Replay high-priority episodes (strengthen representations)
        3. Extract schemas from patterns
        4. Identify interference candidates

        Args:
            episodes: List of episodes to consolidate
            update_callback: Optional callback for updating episode representations

        Returns:
            Consolidation statistics
        """
        # Prioritize episodes
        prioritized_episodes, probabilities = self.prioritize_episodes(episodes)

        # Multiple replay cycles
        replay_count = 0
        for _ in range(self.config.replay_iterations):
            # Sample batch for replay
            replay_batch = self.sample_for_replay(
                prioritized_episodes, probabilities, self.config.replay_batch_size
            )

            # Replay (strengthen representations)
            if update_callback:
                for ep in replay_batch:
                    update_callback(ep)

            replay_count += len(replay_batch)

        # Extract schemas
        new_schemas = self.extract_schemas(prioritized_episodes)
        self.schemas.extend(new_schemas)
        self.schemas_formed += len(new_schemas)

        # Update stats
        self.total_replays += replay_count
        self.last_consolidation = datetime.now(timezone.utc)

        return {
            "episodes_consolidated": len(prioritized_episodes),
            "replay_count": replay_count,
            "schemas_extracted": len(new_schemas),
            "total_schemas": len(self.schemas),
            "timestamp": self.last_consolidation,
        }

    def get_schema_summary(self) -> List[Dict]:
        """Get human-readable summary of learned schemas."""
        summaries = []
        for schema in self.schemas:
            summary = {
                "type": schema["type"],
                "pattern": f"Location: {schema['location']}",
                "frequency": schema["frequency"],
                "common_entities": schema["common_entities"],
                "importance": schema["avg_surprise"],
            }
            summaries.append(summary)
        return summaries
