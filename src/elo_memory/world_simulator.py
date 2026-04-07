"""
Multimodal Experiential World Simulator
=======================================

Stores memories as compressed, replayable experience sequences.
Episodes are chained into temporally coherent "experiences" that the
agent can re-live, re-simulate with perturbations, or query spatially.

Supports multimodal embeddings (text, image, audio, spatial) and builds
2D spatial maps from location-tagged episodes.

References:
- Ha & Schmidhuber (2018): World Models
- Hafner et al. (2020): Dream to Control (Dreamer)
- Lecun (2022): A Path Towards Autonomous Machine Intelligence
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .memory.episodic_store import Episode, EpisodicMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class Modality:
    """A single modality embedding attached to an episode."""

    name: str  # "text", "image", "audio", "spatial"
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialCoord:
    """2D/3D spatial coordinate for location mapping."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    label: str = ""


@dataclass
class Experience:
    """
    A temporally coherent sequence of episodes — one "lived experience."
    """

    experience_id: str
    episode_ids: List[str]
    start_time: datetime
    end_time: datetime
    compressed_embedding: Optional[np.ndarray] = None
    spatial_trajectory: List[SpatialCoord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "episode_ids": self.episode_ids,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "compressed_embedding": (
                self.compressed_embedding.tolist()
                if self.compressed_embedding is not None
                else None
            ),
            "spatial_trajectory": [
                {"x": s.x, "y": s.y, "z": s.z, "label": s.label} for s in self.spatial_trajectory
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Experience:
        return cls(
            experience_id=data["experience_id"],
            episode_ids=data["episode_ids"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            compressed_embedding=(
                np.array(data["compressed_embedding"]) if data.get("compressed_embedding") else None
            ),
            spatial_trajectory=[SpatialCoord(**s) for s in data.get("spatial_trajectory", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorldSimConfig:
    """Configuration for the world simulator."""

    temporal_gap_threshold_minutes: float = 60.0  # Max gap between episodes in one experience
    min_experience_episodes: int = 2  # Min episodes to form an experience
    compression_dim: int = 64  # Compressed embedding dimension
    spatial_grid_resolution: float = 1.0  # Grid cell size for spatial maps
    max_replay_steps: int = 100  # Safety cap on replay length


class WorldSimulator:
    """
    Builds and maintains a world model from episodic memory.

    Key capabilities:
    1. Auto-segments episodes into coherent experiences
    2. Compresses experience sequences into replayable representations
    3. Builds spatial maps from location-tagged episodes
    4. Supports "what-if" simulation by perturbing replay sequences
    """

    def __init__(
        self,
        store: EpisodicMemoryStore,
        config: Optional[WorldSimConfig] = None,
    ):
        self.store = store
        self.config = config or WorldSimConfig()
        self.experiences: List[Experience] = []
        self._spatial_map: Dict[str, SpatialCoord] = {}  # location → coord
        self._multimodal: Dict[str, List[Modality]] = {}  # episode_id → modalities

    # ── Experience Segmentation ──────────────────────────────────

    def segment_experiences(self) -> List[Experience]:
        """Auto-segment all episodes into temporally coherent experiences."""
        episodes = sorted(self.store.episodes, key=lambda ep: ep.timestamp)
        if not episodes:
            return []

        experiences = []
        current_group: List[Episode] = [episodes[0]]
        gap_threshold = timedelta(minutes=self.config.temporal_gap_threshold_minutes)

        for ep in episodes[1:]:
            prev = current_group[-1]
            gap = ep.timestamp - prev.timestamp
            if gap <= gap_threshold:
                current_group.append(ep)
            else:
                if len(current_group) >= self.config.min_experience_episodes:
                    experiences.append(self._create_experience(current_group))
                current_group = [ep]

        # Final group
        if len(current_group) >= self.config.min_experience_episodes:
            experiences.append(self._create_experience(current_group))

        self.experiences = experiences
        return experiences

    def _create_experience(self, episodes: List[Episode]) -> Experience:
        """Create an Experience from a group of episodes."""
        import uuid

        # Compress embeddings via mean pooling
        embeddings = [ep.embedding for ep in episodes if ep.embedding is not None]
        compressed = None
        if embeddings:
            stacked = np.vstack(embeddings)
            # PCA-like compression: use SVD to reduce dimensionality
            if stacked.shape[0] > 1 and stacked.shape[1] > self.config.compression_dim:
                try:
                    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
                    k = min(self.config.compression_dim, len(S))
                    compressed = (U[:, :k] * S[:k]).mean(axis=0)
                except np.linalg.LinAlgError:
                    compressed = stacked.mean(axis=0)[: self.config.compression_dim]
            else:
                compressed = stacked.mean(axis=0)[: self.config.compression_dim]

        # Build spatial trajectory
        trajectory = []
        for ep in episodes:
            if ep.location and ep.location in self._spatial_map:
                trajectory.append(self._spatial_map[ep.location])
            elif ep.location:
                # Auto-assign coordinates based on hash (deterministic layout)
                coord = self._location_to_coord(ep.location)
                self._spatial_map[ep.location] = coord
                trajectory.append(coord)

        return Experience(
            experience_id=f"exp_{uuid.uuid4().hex[:12]}",
            episode_ids=[ep.episode_id for ep in episodes],
            start_time=episodes[0].timestamp,
            end_time=episodes[-1].timestamp,
            compressed_embedding=compressed,
            spatial_trajectory=trajectory,
            metadata={
                "episode_count": len(episodes),
                "locations": list({ep.location for ep in episodes if ep.location}),
                "entities": list({e for ep in episodes for e in ep.entities}),
            },
        )

    # ── Replay ───────────────────────────────────────────────────

    def replay(self, experience_id: str) -> List[Dict[str, Any]]:
        """
        Re-live an experience: returns episode sequence with interpolated context.

        Each step includes the episode content, temporal delta, spatial context,
        and running summary of what has happened so far.
        """
        exp = self._find_experience(experience_id)
        if exp is None:
            return []

        steps = []
        running_entities: set = set()
        prev_time: Optional[datetime] = None

        for i, eid in enumerate(exp.episode_ids[: self.config.max_replay_steps]):
            ep = self.store._get_episode_by_id(eid)
            if ep is None:
                continue

            # Temporal delta
            delta_seconds = 0.0
            if prev_time is not None:
                delta_seconds = (ep.timestamp - prev_time).total_seconds()
            prev_time = ep.timestamp

            # Accumulate context
            running_entities.update(ep.entities)

            # Content
            content = ep.content
            if isinstance(content, dict):
                text = content.get("text", "")
            elif isinstance(content, np.ndarray):
                text = ep.metadata.get("text", "[embedding]")
            else:
                text = str(content)

            steps.append(
                {
                    "step": i,
                    "episode_id": eid,
                    "timestamp": ep.timestamp.isoformat(),
                    "delta_seconds": delta_seconds,
                    "location": ep.location,
                    "text": text,
                    "entities": ep.entities,
                    "surprise": ep.surprise,
                    "running_entities": list(running_entities),
                    "progress": (i + 1) / len(exp.episode_ids),
                }
            )

        return steps

    def simulate_variation(
        self,
        experience_id: str,
        perturbation: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        "What if" simulation: replay with a perturbation applied.

        perturbation can include:
        - remove_episode: str (episode_id to skip)
        - replace_text: {episode_id: new_text}
        - time_shift_hours: float (shift all timestamps)
        """
        steps = self.replay(experience_id)
        if not steps:
            return steps

        remove_id = perturbation.get("remove_episode")
        replacements = perturbation.get("replace_text", {})
        time_shift = perturbation.get("time_shift_hours", 0.0)

        result = []
        for step in steps:
            # Remove perturbation
            if step["episode_id"] == remove_id:
                step["text"] = "[REMOVED — counterfactual]"
                step["perturbation"] = "removed"

            # Text replacement
            if step["episode_id"] in replacements:
                step["text"] = replacements[step["episode_id"]]
                step["perturbation"] = "replaced"

            # Time shift
            if time_shift != 0:
                step["delta_seconds"] += time_shift * 3600
                step["perturbation"] = step.get("perturbation", "") + " time-shifted"

            result.append(step)

        return result

    # ── Spatial Map ──────────────────────────────────────────────

    def build_spatial_map(self) -> Dict[str, Any]:
        """Build a 2D spatial map from all location-tagged episodes."""
        location_episodes: Dict[str, List[str]] = defaultdict(list)

        for ep in self.store.episodes:
            if ep.location:
                location_episodes[ep.location].append(ep.episode_id)
                if ep.location not in self._spatial_map:
                    self._spatial_map[ep.location] = self._location_to_coord(ep.location)

        return {
            "locations": {
                loc: {
                    "coord": {"x": coord.x, "y": coord.y},
                    "episode_count": len(location_episodes.get(loc, [])),
                }
                for loc, coord in self._spatial_map.items()
            },
            "total_locations": len(self._spatial_map),
            "total_located_episodes": sum(len(v) for v in location_episodes.values()),
        }

    def _location_to_coord(self, location: str) -> SpatialCoord:
        """Deterministic location → 2D coordinate mapping."""
        h = hash(location)
        x = ((h >> 16) & 0xFFFF) / 0xFFFF * 100.0
        y = (h & 0xFFFF) / 0xFFFF * 100.0
        return SpatialCoord(x=x, y=y, label=location)

    # ── Multimodal ───────────────────────────────────────────────

    def attach_modality(
        self,
        episode_id: str,
        modality_name: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Attach an additional modality embedding to an episode."""
        mod = Modality(name=modality_name, embedding=embedding, metadata=metadata or {})
        if episode_id not in self._multimodal:
            self._multimodal[episode_id] = []
        self._multimodal[episode_id].append(mod)

    def get_modalities(self, episode_id: str) -> List[Dict[str, Any]]:
        """Get all modality embeddings for an episode."""
        mods = self._multimodal.get(episode_id, [])
        return [
            {"name": m.name, "embedding_shape": m.embedding.shape, "metadata": m.metadata}
            for m in mods
        ]

    def fused_embedding(self, episode_id: str) -> Optional[np.ndarray]:
        """Fuse all modality embeddings into a single representation."""
        ep = self.store._get_episode_by_id(episode_id)
        if ep is None:
            return None

        embeddings = []
        if ep.embedding is not None:
            embeddings.append(ep.embedding)

        for mod in self._multimodal.get(episode_id, []):
            embeddings.append(mod.embedding)

        if not embeddings:
            return None

        # Concatenate and project to fixed dim
        fused = np.concatenate([e.flatten() for e in embeddings])
        # Normalize
        norm = np.linalg.norm(fused)
        return fused / norm if norm > 0 else fused

    # ── Compression ──────────────────────────────────────────────

    def compress_experience(self, experience_id: str) -> Optional[np.ndarray]:
        """Compress an experience into a fixed-size vector."""
        exp = self._find_experience(experience_id)
        if exp is None or exp.compressed_embedding is not None:
            return exp.compressed_embedding if exp else None

        embeddings = []
        for eid in exp.episode_ids:
            ep = self.store._get_episode_by_id(eid)
            if ep and ep.embedding is not None:
                embeddings.append(ep.embedding)

        if not embeddings:
            return None

        stacked = np.vstack(embeddings)
        compressed = stacked.mean(axis=0)[: self.config.compression_dim]
        norm = np.linalg.norm(compressed)
        exp.compressed_embedding = compressed / norm if norm > 0 else compressed
        return exp.compressed_embedding

    # ── Helpers ──────────────────────────────────────────────────

    def _find_experience(self, experience_id: str) -> Optional[Experience]:
        for exp in self.experiences:
            if exp.experience_id == experience_id:
                return exp
        return None

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_experiences": len(self.experiences),
            "total_locations": len(self._spatial_map),
            "multimodal_episodes": len(self._multimodal),
            "avg_experience_length": (
                np.mean([len(e.episode_ids) for e in self.experiences]) if self.experiences else 0
            ),
        }

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data = {
            "experiences": [e.to_dict() for e in self.experiences],
            "spatial_map": {
                loc: {"x": c.x, "y": c.y, "z": c.z, "label": c.label}
                for loc, c in self._spatial_map.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.experiences = [Experience.from_dict(e) for e in data.get("experiences", [])]
            self._spatial_map = {
                loc: SpatialCoord(**coords) for loc, coords in data.get("spatial_map", {}).items()
            }
        except Exception as e:
            logger.error("Failed to load world simulator state: %s", e)
