"""
Episodic Memory Store
====================

Human-inspired episodic memory storage with temporal-spatial grounding.
Implements hybrid storage: vector database + disk offloading for massive contexts.

References:
- EM-LLM (ICLR 2025): Infinite context via episodic memory
- Tulving (1983): Episodic memory definition
- A-MEM (NeurIPS 2025): Agentic memory organization

Key Features:
1. Fast episodic encoding (single-shot learning)
2. Temporal-spatial indexing for contextual retrieval
3. Vector similarity search (ChromaDB/FAISS)
4. Disk offloading for contexts > 10M tokens
5. Automatic memory consolidation and pruning
"""

import logging
from collections import OrderedDict
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import chromadb

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """
    Single episodic memory representing a specific event.

    Attributes:
        content: The actual observation/experience
        timestamp: When the episode occurred
        location: Spatial context (optional)
        entities: Who/what was involved
        embedding: Vector representation for similarity search
        surprise: How novel/unexpected this episode was
        importance: Consolidation priority score
        metadata: Additional contextual information
    """

    content: Union[np.ndarray, Dict[str, Any]]
    timestamp: datetime
    location: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    surprise: float = 0.0
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    episode_id: Optional[str] = None

    def __post_init__(self):
        """Generate episode ID if not provided."""
        if self.episode_id is None:
            import uuid
            self.episode_id = f"ep_{self.timestamp.timestamp()}_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        """Convert episode to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "content": (
                self.content.tolist() if isinstance(self.content, np.ndarray) else self.content
            ),
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "entities": self.entities,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "surprise": self.surprise,
            "importance": self.importance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Episode":
        """Reconstruct episode from dictionary."""
        if "content" not in data or "timestamp" not in data:
            raise ValueError("Episode data must contain 'content' and 'timestamp' keys")
        content = data["content"]
        # Preserve dict content as-is; only convert lists/primitives to ndarray
        if not isinstance(content, dict):
            content = np.array(content)
        return cls(
            content=content,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=data.get("location"),
            entities=data.get("entities", []),
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            surprise=data.get("surprise", 0.0),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
            episode_id=data.get("episode_id"),
        )


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory storage."""

    max_episodes: int = 10000  # Maximum episodes in hot storage
    embedding_dim: int = 512  # Dimension of episode embeddings
    enable_disk_offload: bool = True  # Offload old episodes to disk
    offload_threshold: int = 8000  # Start offloading when this many episodes
    importance_decay: float = 0.99  # Decay factor for episode importance
    consolidation_interval: int = 100  # Consolidate every N new episodes
    consolidation_interval_episodes: int = 500  # Consolidate every N new episodes (hybrid)
    consolidation_interval_hours: float = 8.0  # OR every N hours (whichever first)
    consolidation_min_episodes: int = 100  # Minimum episodes before first consolidation
    vector_db_backend: str = "chromadb"  # "chromadb" or "faiss"
    persistence_path: Optional[str] = "./memory_store"  # Path for persistent storage
    interference_check_window: int = 100  # Recent episodes to check for interference
    query_cache_size: int = 128  # LRU cache size for similarity queries (0 to disable)


class EpisodicMemoryStore:
    """
    Episodic memory storage with efficient retrieval and consolidation.

    Mimics hippocampal episodic memory:
    - Fast single-shot encoding
    - Temporal-spatial indexing
    - Importance-based consolidation
    - Automatic capacity management
    """

    def __init__(self, config: Optional[EpisodicMemoryConfig] = None):
        """
        Args:
            config: Memory configuration
        """
        from .forgetting import ForgettingEngine
        from .interference import InterferenceResolver
        from ..online_learning import OnlineLearner

        self.config = config or EpisodicMemoryConfig()

        # In-memory episode storage (hot storage)
        self.episodes: List[Episode] = []

        # O(1) episode lookup by ID
        self._episode_index: Dict[str, Episode] = {}

        # Temporal index: timestamp -> episode_ids
        self.temporal_index: Dict[str, List[str]] = {}

        # Spatial index: location -> episode_ids
        self.spatial_index: Dict[str, List[str]] = {}

        # Entity index: entity -> episode_ids
        self.entity_index: Dict[str, List[str]] = {}

        # Statistics
        self.total_episodes_stored = 0
        self.episodes_offloaded = 0

        # Consolidation tracking (hybrid time + count)
        self.last_consolidation_time: Optional[datetime] = None
        self.episodes_since_consolidation: int = 0

        # Bio-inspired components
        self.forgetting = ForgettingEngine()
        self.interference = InterferenceResolver()
        self.online_learner = OnlineLearner()

        # LRU cache for similarity queries: cache_key -> list of episode_ids
        self._query_cache: OrderedDict[str, List[str]] = OrderedDict()
        self._query_cache_max = self.config.query_cache_size

        # Initialize vector database
        self._initialize_vector_db()

        # Setup persistence
        if self.config.persistence_path:
            self.persistence_path = Path(self.config.persistence_path)
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            # Auto-load state when persistence_path is set
            self.load_state()

    def _initialize_vector_db(self):
        """Initialize vector database for similarity search."""
        self.collection = None
        if self.config.vector_db_backend == "chromadb":
            try:
                if self.config.persistence_path:
                    chroma_path = str(Path(self.config.persistence_path) / "chroma")
                    self.chroma_client = chromadb.PersistentClient(path=chroma_path)
                else:
                    self.chroma_client = chromadb.Client()

                self.collection = self.chroma_client.get_or_create_collection(
                    name=f"episodic_memories_{self.config.embedding_dim}",
                    metadata={"description": "Human-inspired episodic memory storage"},
                )
            except Exception as e:
                logger.error("ChromaDB initialization failed, vector search disabled: %s", e)
        else:
            raise NotImplementedError("FAISS backend not yet implemented")

    _encoder = None

    @classmethod
    def _get_encoder(cls):
        """Lazy-load SentenceTransformer encoder. Returns None if not installed."""
        if cls._encoder is not None:
            return cls._encoder
        try:
            from sentence_transformers import SentenceTransformer

            cls._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            return cls._encoder
        except ImportError:
            logger.warning(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
            return None

    def search(self, query, k: int = 5) -> List["Episode"]:
        """
        Convenience method: search episodes by string or embedding.

        Args:
            query: A string or numpy embedding vector.
            k: Number of results to return.

        Returns:
            List of matching Episode objects.
        """
        if isinstance(query, str):
            encoder = self._get_encoder()
            if encoder is not None:
                embedding = encoder.encode(query)
            else:
                embedding = self._generate_embedding({"text": query})
        else:
            embedding = np.asarray(query)
        return self.retrieve_by_similarity(embedding, k=k)

    def store_episode(
        self,
        content: Union[np.ndarray, Dict[str, Any]],
        surprise: float = 0.0,
        timestamp: Optional[datetime] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> Episode:
        """
        Store a new episodic memory (single-shot learning).

        Args:
            content: Observation content (ndarray or dict with "text" key)
            surprise: Surprise/novelty score from BayesianSurpriseEngine
            timestamp: When this happened (defaults to now)
            location: Where this happened
            entities: Who/what was involved
            metadata: Additional context
            embedding: Pre-computed embedding (optional)

        Returns:
            Created Episode object
        """
        # Input validation
        if content is None:
            raise ValueError("content must not be None")
        if embedding is not None:
            embedding = np.asarray(embedding)
            if embedding.shape != (self.config.embedding_dim,):
                raise ValueError(
                    f"embedding dimension {embedding.shape} does not match "
                    f"configured embedding_dim ({self.config.embedding_dim},)"
                )
            if not np.all(np.isfinite(embedding)):
                raise ValueError("embedding contains NaN or Inf values")

        # Create episode
        episode = Episode(
            content=content,
            timestamp=timestamp or datetime.now(timezone.utc),
            location=location,
            entities=entities or [],
            embedding=embedding,
            surprise=surprise,
            importance=self._compute_importance(surprise),
            metadata=metadata or {},
        )

        # Generate embedding if not provided
        if episode.embedding is None:
            episode.embedding = self._generate_embedding(content)

        # Interference resolution: orthogonalize similar embeddings
        if episode.embedding is not None and len(self.episodes) > 0:
            existing_embeddings = [
                ep.embedding
                for ep in self.episodes[-self.config.interference_check_window :]
                if ep.embedding is not None
            ]
            if existing_embeddings:
                episode.embedding, _ = self.interference.resolve_interference_set(
                    episode.embedding, existing_embeddings
                )

        # Add to in-memory storage
        self.episodes.append(episode)

        # Update indices
        self._update_indices(episode)

        # Add to vector database
        self._add_to_vector_db(episode)

        # Let LRU evict stale cache entries naturally instead of blanket clear.
        # Similarity results shift slightly with each new episode, but cached
        # results are still useful approximations for most queries.

        # Update statistics
        self.total_episodes_stored += 1

        # Online learning: update adaptive thresholds and replay buffer
        self.online_learner.online_update(
            (
                episode.embedding
                if episode.embedding is not None
                else np.zeros(self.config.embedding_dim)
            ),
            surprise,
        )

        # Check if consolidation/offloading needed (hybrid time + count)
        self.episodes_since_consolidation += 1
        if self.should_consolidate():
            self._consolidate_memory()

        return episode

    def _compute_importance(self, surprise: float) -> float:
        """
        Compute importance score for episode.
        Higher surprise → higher importance → prioritized for consolidation.

        Args:
            surprise: Surprise value from Bayesian surprise engine

        Returns:
            Importance score [0, 1]
        """
        # Sigmoid transformation of surprise
        importance = 1.0 / (1.0 + np.exp(-surprise + 2.0))
        return np.clip(importance, 0.0, 1.0)

    def should_consolidate(self) -> bool:
        """Check if consolidation should run (hybrid time + count)."""
        now = datetime.now(timezone.utc)
        if len(self.episodes) < self.config.consolidation_min_episodes:
            return False
        if self.episodes_since_consolidation >= self.config.consolidation_interval_episodes:
            return True
        if self.last_consolidation_time is not None:
            last = self.last_consolidation_time
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            hours_since = (now - last).total_seconds() / 3600
            if hours_since >= self.config.consolidation_interval_hours:
                return True
        # First-run check: trigger if enough time has passed since the first episode
        if (
            self.last_consolidation_time is None
            and len(self.episodes) >= self.config.consolidation_min_episodes
        ):
            first_ep_time = self.episodes[0].timestamp if self.episodes else None
            if first_ep_time:
                if first_ep_time.tzinfo is None:
                    first_ep_time = first_ep_time.replace(tzinfo=timezone.utc)
                hours_since_first = (now - first_ep_time).total_seconds() / 3600
                if hours_since_first >= self.config.consolidation_interval_hours:
                    return True
        return False

    def mark_consolidated(self):
        """Mark consolidation as complete, reset counters."""
        self.last_consolidation_time = datetime.now(timezone.utc)
        self.episodes_since_consolidation = 0

    def _generate_embedding(self, content: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Generate vector embedding for content.
        In production, use a pre-trained encoder (e.g., BERT, Sentence Transformers).

        Args:
            content: Observation content (ndarray or dict)

        Returns:
            Embedding vector
        """
        # Convert dict content to array via deterministic hash
        if isinstance(content, dict):
            text = content.get("text", json.dumps(content, sort_keys=True))
            arr = np.zeros(self.config.embedding_dim)
            for i, char in enumerate(text):
                idx = (ord(char) * (i + 1)) % self.config.embedding_dim
                arr[idx] += np.sin(ord(char) * 0.1) * 0.5
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else arr

        # Simple projection for now (replace with real encoder)
        if len(content) < self.config.embedding_dim:
            # Pad with zeros
            embedding = np.zeros(self.config.embedding_dim)
            embedding[: len(content)] = content
        else:
            # Use PCA-like projection (simplified)
            embedding = content[: self.config.embedding_dim]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _update_indices(self, episode: Episode):
        """Update all indices with new episode."""
        # ID index for O(1) lookup
        self._episode_index[episode.episode_id] = episode

        # Temporal index (by date)
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        if date_key not in self.temporal_index:
            self.temporal_index[date_key] = []
        self.temporal_index[date_key].append(episode.episode_id)

        # Spatial index
        if episode.location:
            if episode.location not in self.spatial_index:
                self.spatial_index[episode.location] = []
            self.spatial_index[episode.location].append(episode.episode_id)

        # Entity index (case-insensitive)
        for entity in episode.entities:
            key = entity.lower()
            if key not in self.entity_index:
                self.entity_index[key] = []
            self.entity_index[key].append(episode.episode_id)

    def _add_to_vector_db(self, episode: Episode):
        """Add episode to vector database for similarity search."""
        if self.collection is None:
            return
        try:
            self.collection.add(
                ids=[episode.episode_id],
                embeddings=[episode.embedding.tolist()],
                metadatas=[
                    {
                        "timestamp": episode.timestamp.isoformat(),
                        "location": episode.location or "",
                        "entities": ",".join(episode.entities),
                        "surprise": episode.surprise,
                        "importance": episode.importance,
                    }
                ],
            )
        except Exception as e:
            logger.error("Failed to add episode %s to vector DB: %s", episode.episode_id, e)

    def retrieve_by_similarity(
        self, query_embedding: np.ndarray, k: int = 10, filter_criteria: Optional[Dict] = None
    ) -> List[Episode]:
        """
        Retrieve k most similar episodes by vector similarity.

        Args:
            query_embedding: Query vector
            k: Number of episodes to retrieve
            filter_criteria: Optional filters (e.g., {"location": "office"})

        Returns:
            List of similar episodes
        """
        if self.collection is None:
            return []

        # LRU cache lookup (skip cache when filters are used — too many key combos)
        cache_key = None
        if self._query_cache_max > 0 and filter_criteria is None:
            cache_key = f"{query_embedding.tobytes().hex()[:32]}:k={k}"
            if cache_key in self._query_cache:
                self._query_cache.move_to_end(cache_key)
                cached_ids = self._query_cache[cache_key]
                episodes = [self._get_episode_by_id(eid) for eid in cached_ids]
                episodes = [ep for ep in episodes if ep is not None]
                # Still apply forgetting/activation below
                for ep in episodes:
                    retrieval_count = ep.metadata.get("_retrieval_count", 0)
                    new_count = retrieval_count + 1
                    ep.metadata["_retrieval_count"] = new_count
                    ep.metadata["_activation"] = self.forgetting.compute_activation(
                        ep.importance,
                        ep.timestamp,
                        rehearsal_count=new_count,
                    )
                episodes.sort(key=lambda ep: ep.metadata.get("_activation", 0), reverse=True)
                return episodes

        # Query vector database
        where_clause = None
        if filter_criteria:
            where_clause = filter_criteria

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=k, where=where_clause
            )
        except Exception as e:
            logger.warning("Vector DB query failed: %s", e)
            return []

        # Retrieve full episodes
        episode_ids = results["ids"][0] if results["ids"] else []

        # Populate cache
        if cache_key is not None:
            self._query_cache[cache_key] = episode_ids
            if len(self._query_cache) > self._query_cache_max:
                self._query_cache.popitem(last=False)  # evict oldest

        episodes = [self._get_episode_by_id(eid) for eid in episode_ids]
        episodes = [ep for ep in episodes if ep is not None]

        # Apply forgetting: re-rank by activation (decayed importance)
        # Rehearsal count tracks how many times an episode was retrieved
        for ep in episodes:
            retrieval_count = ep.metadata.get("_retrieval_count", 0)
            new_count = retrieval_count + 1
            ep.metadata["_retrieval_count"] = new_count
            ep.metadata["_activation"] = self.forgetting.compute_activation(
                ep.importance,
                ep.timestamp,
                rehearsal_count=new_count,
            )

        # Sort by activation (highest first) so decayed memories rank lower
        episodes.sort(key=lambda ep: ep.metadata.get("_activation", 0), reverse=True)

        return episodes

    def retrieve_by_temporal_range(self, start_time: datetime, end_time: datetime) -> List[Episode]:
        """
        Retrieve episodes within a time range.

        Args:
            start_time: Start of time window
            end_time: End of time window

        Returns:
            Episodes in time range
        """
        matching_episodes = []

        for episode in self.episodes:
            ep_ts = episode.timestamp
            # Handle timezone mismatches
            if ep_ts.tzinfo is None and start_time.tzinfo is not None:
                ep_ts = ep_ts.replace(tzinfo=timezone.utc)
            elif ep_ts.tzinfo is not None and start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
                end_time = end_time.replace(tzinfo=timezone.utc)
            if start_time <= ep_ts <= end_time:
                matching_episodes.append(episode)

        return matching_episodes

    def retrieve_by_location(self, location: str) -> List[Episode]:
        """Retrieve all episodes at a specific location."""
        episode_ids = self.spatial_index.get(location, [])
        episodes = [self._get_episode_by_id(eid) for eid in episode_ids]
        return [ep for ep in episodes if ep is not None]

    def retrieve_by_entity(self, entity: str) -> List[Episode]:
        """Retrieve all episodes involving a specific entity."""
        episode_ids = self.entity_index.get(entity.lower(), [])
        episodes = [self._get_episode_by_id(eid) for eid in episode_ids]
        return [ep for ep in episodes if ep is not None]

    def _get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Retrieve episode by ID (O(1) via index)."""
        episode = self._episode_index.get(episode_id)
        if episode is not None:
            return episode

        # Check offloaded storage
        if self.config.enable_disk_offload:
            return self._load_offloaded_episode(episode_id)

        return None

    def run_consolidation(self) -> Optional[Dict]:
        """
        Run a full consolidation cycle: replay, schema extraction, importance decay.
        Call this periodically (e.g., every 60 minutes) regardless of episode count.
        Returns consolidation stats or None if skipped.
        """
        from ..consolidation.memory_consolidation import MemoryConsolidationEngine

        if not hasattr(self, "_consolidation_engine"):
            self._consolidation_engine = MemoryConsolidationEngine()

        if not self.episodes:
            return None

        # Apply forgetting decay to all episodes
        for ep in self.episodes:
            ep.metadata["_activation"] = self.forgetting.compute_activation(
                ep.importance,
                ep.timestamp,
                rehearsal_count=ep.metadata.get("_retrieval_count", 0),
            )

        # Run consolidation (replay + schema extraction)
        def _strengthen(ep):
            """Rehearsal callback: boost importance of replayed episodes."""
            ep.importance = min(1.0, ep.importance * 1.1)
            ep.metadata["_retrieval_count"] = ep.metadata.get("_retrieval_count", 0) + 1

        stats = self._consolidation_engine.consolidate(self.episodes, update_callback=_strengthen)
        return stats

    def _consolidate_memory(self):
        """
        Consolidate memory by offloading low-importance episodes to disk.
        Mimics hippocampal consolidation: important memories stay, others archived.
        """
        if not self.config.enable_disk_offload:
            return

        # Decay importance of all episodes
        for episode in self.episodes:
            episode.importance *= self.config.importance_decay

        # Sort by importance
        self.episodes.sort(key=lambda ep: ep.importance, reverse=True)

        # Offload bottom X% to disk
        n_to_offload = len(self.episodes) - self.config.max_episodes
        if n_to_offload > 0:
            episodes_to_offload = self.episodes[-n_to_offload:]
            self.episodes = self.episodes[:-n_to_offload]

            for episode in episodes_to_offload:
                self._episode_index.pop(episode.episode_id, None)
                self._offload_episode(episode)
                self.episodes_offloaded += 1

        self.mark_consolidated()

    def _offload_episode(self, episode: Episode):
        """Offload episode to disk storage."""
        if not self.persistence_path:
            return
        try:
            offload_dir = self.persistence_path / "offloaded"
            offload_dir.mkdir(exist_ok=True)
            file_path = offload_dir / f"{episode.episode_id}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(episode, f)
        except Exception as e:
            logger.error("Failed to offload episode %s: %s", episode.episode_id, e)

    def _load_offloaded_episode(self, episode_id: str) -> Optional[Episode]:
        """Load episode from disk storage."""
        if not self.persistence_path:
            return None

        file_path = self.persistence_path / "offloaded" / f"{episode_id}.pkl"
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error("Failed to load offloaded episode %s: %s", episode_id, e)

        return None

    def save_state(self):
        """Save memory state to disk."""
        if not self.persistence_path:
            return

        state = {
            "config": self.config.__dict__,
            "episodes": [ep.to_dict() for ep in self.episodes],
            "temporal_index": self.temporal_index,
            "spatial_index": self.spatial_index,
            "entity_index": self.entity_index,
            "total_episodes_stored": self.total_episodes_stored,
            "episodes_offloaded": self.episodes_offloaded,
            "last_consolidation_time": (
                self.last_consolidation_time.isoformat() if self.last_consolidation_time else None
            ),
            "episodes_since_consolidation": self.episodes_since_consolidation,
        }

        state_file = self.persistence_path / "memory_state.json"
        tmp_file = self.persistence_path / "memory_state.json.tmp"
        try:
            with open(tmp_file, "w") as f:
                json.dump(state, f)
            # Atomic rename — prevents corruption if process is killed mid-write
            import os

            os.replace(str(tmp_file), str(state_file))
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    def load_state(self):
        """Load memory state from disk."""
        if not self.persistence_path:
            return

        state_file = self.persistence_path / "memory_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load state file %s: %s", state_file, e)
            return

        try:
            # Restore episodes
            self.episodes = [Episode.from_dict(ep_dict) for ep_dict in state["episodes"]]

            # Rebuild O(1) episode ID index
            self._episode_index = {ep.episode_id: ep for ep in self.episodes}

            # Restore indices
            self.temporal_index = state["temporal_index"]
            self.spatial_index = state["spatial_index"]
            self.entity_index = state["entity_index"]

            # Restore statistics
            self.total_episodes_stored = state["total_episodes_stored"]
            self.episodes_offloaded = state["episodes_offloaded"]

            # Restore consolidation tracking
            last_consolidation_iso = state.get("last_consolidation_time")
            self.last_consolidation_time = (
                datetime.fromisoformat(last_consolidation_iso) if last_consolidation_iso else None
            )
            self.episodes_since_consolidation = state.get("episodes_since_consolidation", 0)
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Corrupted state data, starting fresh: %s", e)
            self.episodes = []
            self._episode_index = {}
            return

        # Sync episodes into ChromaDB that aren't already persisted there.
        if self.collection is not None:
            try:
                existing_ids = set()
                if self.collection.count() > 0:
                    existing_ids = set(self.collection.get()["ids"])
                for ep in self.episodes:
                    if ep.episode_id not in existing_ids and ep.embedding is not None:
                        self._add_to_vector_db(ep)
            except Exception as e:
                logger.warning("ChromaDB sync during load failed: %s", e)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        learner_stats = self.online_learner.get_statistics()
        return {
            "total_episodes": self.total_episodes_stored,
            "episodes_in_memory": len(self.episodes),
            "episodes_offloaded": self.episodes_offloaded,
            "unique_locations": len(self.spatial_index),
            "unique_entities": len(self.entity_index),
            "temporal_span_days": len(self.temporal_index),
            "mean_importance": (
                np.mean([ep.importance for ep in self.episodes]) if self.episodes else 0.0
            ),
            "adaptive_surprise_threshold": learner_stats["surprise_threshold"],
            "replay_buffer_size": learner_stats["replay_buffer_size"],
            "total_online_updates": learner_stats["total_updates"],
            "last_consolidation": (
                self.last_consolidation_time.isoformat() if self.last_consolidation_time else None
            ),
            "episodes_since_consolidation": self.episodes_since_consolidation,
            "query_cache_entries": len(self._query_cache),
            "query_cache_max": self._query_cache_max,
        }
