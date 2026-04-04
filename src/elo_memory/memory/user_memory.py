"""Per-user episodic memory with automatic embedding, dedup, conflict detection.

Wraps EpisodicMemoryStore + TwoStageRetriever into a single high-level API
suitable for agent / chatbot integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
import filelock
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .episodic_store import EpisodicMemoryConfig, EpisodicMemoryStore, Episode
from .entity_extractor import EntityExtractor
from ..retrieval.two_stage_retriever import TwoStageRetriever, RetrievalConfig
from ..utils import hash_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic detection keywords
# ---------------------------------------------------------------------------

_TOPIC_RULES: Dict[str, List[str]] = {
    "tech_stack": [
        "react",
        "vue",
        "angular",
        "svelte",
        "django",
        "flask",
        "fastapi",
        "express",
        "nextjs",
        "next.js",
        "rails",
        "laravel",
        "spring",
        "typescript",
        "javascript",
        "python",
        "golang",
        "rust",
        "kotlin",
        "swift",
        "java",
        "c#",
        "c++",
        "ruby",
        "php",
        "elixir",
        "haskell",
    ],
    "database": [
        "postgres",
        "postgresql",
        "mysql",
        "sqlite",
        "mongodb",
        "mongo",
        "redis",
        "dynamodb",
        "cassandra",
        "couchdb",
        "firestore",
        "supabase",
        "prisma",
        "drizzle",
        "sqlalchemy",
    ],
    "infrastructure": [
        "docker",
        "kubernetes",
        "k8s",
        "terraform",
        "ansible",
        "aws",
        "gcp",
        "azure",
        "vercel",
        "netlify",
        "heroku",
        "cloudflare",
        "nginx",
        "ci/cd",
        "github actions",
        "jenkins",
        "deploy",
    ],
    "payment": [
        "stripe",
        "paypal",
        "braintree",
        "square",
        "billing",
        "subscription",
        "invoice",
        "payment",
        "checkout",
        "pricing",
    ],
    "security": [
        "auth",
        "oauth",
        "jwt",
        "encryption",
        "ssl",
        "tls",
        "firewall",
        "vulnerability",
        "penetration",
        "security",
        "rbac",
        "permissions",
        "2fa",
        "mfa",
    ],
    "team": [
        "hired",
        "fired",
        "onboarding",
        "standup",
        "sprint",
        "retro",
        "team",
        "employee",
        "manager",
        "lead",
        "promoted",
        "intern",
        "contractor",
    ],
    "fundraising": [
        "funding",
        "seed",
        "series a",
        "series b",
        "investor",
        "pitch",
        "valuation",
        "runway",
        "revenue",
        "arr",
        "mrr",
        "vc",
        "angel",
        "fundraising",
        "round",
    ],
    "performance": [
        "latency",
        "throughput",
        "p99",
        "p95",
        "benchmark",
        "optimization",
        "cache",
        "caching",
        "bottleneck",
        "slow",
        "performance",
        "speed",
        "fast",
        "profiling",
    ],
    "ml": [
        "model",
        "training",
        "inference",
        "embedding",
        "llm",
        "transformer",
        "fine-tune",
        "dataset",
        "gpu",
        "cuda",
        "pytorch",
        "tensorflow",
        "machine learning",
        "neural",
    ],
    "monitoring": [
        "grafana",
        "prometheus",
        "datadog",
        "sentry",
        "logging",
        "alerting",
        "metrics",
        "dashboard",
        "observability",
        "tracing",
    ],
}

# ---------------------------------------------------------------------------
# Conflict / transition detection patterns
# ---------------------------------------------------------------------------

_TRANSITION_PATTERNS = [
    # "switched from X to Y", "moved from X to Y", "migrated from X to Y"
    re.compile(
        r"(?:I\s+)?(?:switched|moved|migrated|transitioned|changed|converted|upgraded)"
        r"\s+from\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    # "replaced X with Y"
    re.compile(
        r"(?:I\s+)?replaced\s+(.+?)\s+with\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    # "no longer using X, now using Y"
    re.compile(
        r"(?:I(?:'m|\s+am)\s+)?no\s+longer\s+(?:using|on|at)\s+(.+?),?\s*(?:now|currently)\s+(?:using|on|at)\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
]

# Patterns for implicit old values
_IMPLICIT_OLD_VALUES: Dict[str, List[str]] = {
    "promoted": ["junior", "intern", "associate", "mid-level"],
    "started dating": ["single", "alone", "not dating"],
    "got married": ["single", "dating", "engaged"],
    "got engaged": ["single", "dating"],
    "divorced": ["married"],
    "retired": ["working", "employed"],
    "graduated": ["student", "studying"],
    "quit": ["employed", "working at"],
    "fired": ["employed", "working at"],
}

# Patterns that indicate a life-change with implicit old values
_IMPLICIT_PATTERNS = [
    re.compile(
        r"(?:I\s+)?(?:got\s+)?(promoted|started\s+dating|got\s+married|got\s+engaged|divorced|retired|graduated|quit|fired)",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Derived-fact extraction
# ---------------------------------------------------------------------------

_DERIVED_FACT_PATTERNS = [
    # "switched from X to Y" → "Currently using Y"
    (
        re.compile(
            r"(?:I\s+)?(?:switched|moved|migrated|transitioned|changed|converted|upgraded)"
            r"\s+from\s+.+?\s+to\s+(.+?)(?:\s*[,]|$)",
            re.IGNORECASE,
        ),
        "Currently using {0}",
    ),
    # "replaced X with Y" → "Currently using Y"
    (
        re.compile(
            r"(?:I\s+)?replaced\s+.+?\s+with\s+(.+?)(?:\s*[,]|$)",
            re.IGNORECASE,
        ),
        "Currently using {0}",
    ),
]


class UserMemory:
    """High-level per-user memory with automatic embedding, dedup, and conflict detection."""

    # cosine threshold above which a new memory is considered a near-duplicate
    DUPLICATE_COSINE_THRESHOLD = 0.92

    def __init__(
        self,
        user_id: str,
        persistence_path: str = "./memories",
        embedding_dim: int = 384,
        surprise_threshold: float = 0.0,
    ):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        self.user_id = user_id
        self.embedding_dim = embedding_dim
        self.surprise_threshold = surprise_threshold

        # Truncate long user_ids for filesystem safety
        safe_id = user_id
        if len(safe_id) > 80:
            safe_id = hashlib.sha256(user_id.encode()).hexdigest()[:40]
        self._user_dir = Path(persistence_path) / safe_id
        self._user_dir.mkdir(parents=True, exist_ok=True)

        # File lock for multi-process safety
        self._lock = filelock.FileLock(str(self._user_dir / ".lock"), timeout=10)

        # Core stores
        store_config = EpisodicMemoryConfig(
            embedding_dim=embedding_dim,
            persistence_path=str(self._user_dir / "store"),
            consolidation_min_episodes=200,
            consolidation_interval_episodes=500,
        )
        self._store = EpisodicMemoryStore(store_config)

        retrieval_config = RetrievalConfig(
            max_retrieved=20, k_similarity=30, similarity_threshold=0.05
        )
        self._retriever = TwoStageRetriever(self._store, retrieval_config)

        self._entity_extractor = EntityExtractor()

        # Lazy-loaded sentence transformer
        self._embedder: Any = None

        # Session management
        self._session_id: str = str(uuid.uuid4())
        self._sessions: List[Dict[str, Any]] = []
        self._first_seen: Optional[str] = None
        self._last_seen: Optional[str] = None

        # All entity mentions across sessions
        self._all_entities: Dict[str, List[str]] = {}

        # Superseded episode ids
        self._superseded: set[str] = set()

        # Load persisted state
        self._load_meta()
        self._store.load_state()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, using hash-based embeddings. "
                    "Install with: pip install sentence-transformers"
                )
                self._embedder = False  # Sentinel to avoid repeated import attempts
        return self._embedder if self._embedder else None

    def _embed(self, text: str) -> np.ndarray:
        model = self._get_embedder()
        if model is not None:
            vec = model.encode(text, convert_to_numpy=True)
        else:
            # Hash-based fallback
            vec = self._hash_embedding(text)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return np.asarray(vec, dtype=np.float32)

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
        return hash_embedding(text, dim)

    # ------------------------------------------------------------------
    # Topic detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_topics(text: str) -> List[str]:
        text_lower = text.lower()
        topics = []
        for topic, keywords in _TOPIC_RULES.items():
            for kw in keywords:
                if kw in text_lower:
                    topics.append(topic)
                    break
        return topics

    # ------------------------------------------------------------------
    # Near-duplicate detection
    # ------------------------------------------------------------------

    def _is_near_duplicate(self, embedding: np.ndarray) -> bool:
        """Check if embedding is too similar to an existing episode.

        For small stores, does a direct scan which is cheap and avoids
        interference-distorted embeddings.
        For large stores, uses the vector DB with a slightly higher
        threshold to account for interference-resolution noise.
        """
        episodes = self._store.episodes
        if len(episodes) <= 200:
            # Direct scan is cheap and avoids interference-distorted embeddings
            candidates = episodes
            threshold = self.DUPLICATE_COSINE_THRESHOLD
        else:
            # Use vector DB for large stores with a slightly higher threshold
            # to account for interference-resolution noise
            candidates = self._store.retrieve_by_similarity(embedding, k=5)
            threshold = self.DUPLICATE_COSINE_THRESHOLD + 0.03

        for ep in candidates:
            if ep.episode_id in self._superseded:
                continue
            if ep.embedding is not None:
                sim = float(np.dot(embedding, ep.embedding))
                if sim > threshold:
                    return True
        return False

    # ------------------------------------------------------------------
    # Conflict / supersession detection
    # ------------------------------------------------------------------

    # Patterns that signal "I got a new X" (implicit replacement)
    _NEW_THING_RE = re.compile(
        r"\b(?:just\s+)?(?:got|picked up|bought|received|started using|new)\s+"
        r"(?:a\s+|an\s+|my\s+(?:new\s+)?)?(.+?)(?:\s*[,.]|\s+(?:yesterday|today|last|this)\b|\s*$)",
        re.IGNORECASE,
    )

    def _detect_and_supersede(self, text: str, new_episode_id: str) -> None:
        """Detect transitional statements and supersede conflicting old memories."""
        text_lower = text.lower()

        # 1. Explicit transitions: "switched from X to Y"
        for pattern in _TRANSITION_PATTERNS:
            m = pattern.search(text)
            if m:
                old_value = m.group(1).strip().lower()
                new_value = m.group(2).strip().lower()
                self._supersede_by_value(old_value, new_value, new_episode_id)

        # 2. Implicit transitions: "promoted" → supersede "junior"/"intern"
        for pattern in _IMPLICIT_PATTERNS:
            m = pattern.search(text)
            if m:
                trigger = m.group(1).strip().lower()
                for key, old_values in _IMPLICIT_OLD_VALUES.items():
                    if key in trigger:
                        for old_val in old_values:
                            self._supersede_by_value(old_val, trigger, new_episode_id)
                        break

        # 3. Semantic supersession: "Just got a new Tesla" supersedes
        # highly similar old memories about the same topic (e.g., "I drive a BMW")
        m = self._NEW_THING_RE.search(text)
        if m and len(self._store.episodes) > 1:
            new_thing = m.group(1).strip()
            new_emb = self._embed(text)
            if new_emb is not None:
                try:
                    similar = self._store.retrieve_by_similarity(new_emb, k=3)
                    for ep in similar:
                        if ep.episode_id == new_episode_id:
                            continue
                        if ep.episode_id in self._superseded:
                            continue
                        if ep.embedding is None:
                            continue
                        cos = float(
                            np.dot(new_emb, ep.embedding)
                            / (np.linalg.norm(new_emb) * np.linalg.norm(ep.embedding) + 1e-8)
                        )
                        # Moderate similarity (same topic area) + "new" language = supersede
                        if cos > 0.7:
                            self._superseded.add(ep.episode_id)
                except Exception as e:
                    logger.debug("Semantic supersession failed: %s", e)

    def _supersede_by_value(self, old_value: str, new_value: str, new_episode_id: str) -> None:
        """Scan ALL episodes for old_value mentions and supersede them."""
        old_lower = old_value.lower()
        new_lower = new_value.lower()

        for ep in self._store.episodes:
            if ep.episode_id == new_episode_id:
                continue
            if ep.episode_id in self._superseded:
                continue

            # Get episode text
            ep_text = self._episode_text(ep).lower()

            # Only supersede if old episode mentions old value but NOT the new value
            if old_lower in ep_text and new_lower not in ep_text:
                self._superseded.add(ep.episode_id)
                logger.debug(
                    "Superseded episode %s (old: %s, new: %s)",
                    ep.episode_id,
                    old_value,
                    new_value,
                )

    @staticmethod
    def _episode_text(ep: Episode) -> str:
        """Extract text content from an episode."""
        if isinstance(ep.content, dict):
            return str(ep.content.get("text", str(ep.content)))
        if isinstance(ep.content, np.ndarray):
            return str(ep.metadata.get("text", ""))
        return str(ep.content)

    # ------------------------------------------------------------------
    # Derived facts
    # ------------------------------------------------------------------

    def _generate_derived_facts(self, text: str) -> List[str]:
        """Extract derived facts from transitional statements."""
        facts = []
        for pattern, template in _DERIVED_FACT_PATTERNS:
            m = pattern.search(text)
            if m:
                facts.append(template.format(*m.groups()))
        return facts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store a memory.

        Raises:
            TypeError: if *text* is not a string.
            ValueError: if *text* is empty or whitespace-only.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        with self._lock:
            now = datetime.now(timezone.utc)
            self._last_seen = now.isoformat()
            if self._first_seen is None:
                self._first_seen = now.isoformat()

            # Embed
            embedding = self._embed(text)

            # Near-duplicate check
            if self._is_near_duplicate(embedding):
                return {
                    "stored": False,
                    "surprise": 0.0,
                    "episode_id": None,
                    "entities": [],
                    "reason": "near-duplicate",
                }

            # Extract entities
            extracted = self._entity_extractor.extract(text)
            flat_entities = self._entity_extractor.extract_flat(text)

            # Detect topics
            topics = self._detect_topics(text)

            # Build metadata
            meta = dict(metadata or {})
            meta["text"] = text
            meta["session_id"] = self._session_id
            meta["topics"] = topics
            meta["entities_structured"] = extracted

            # Store episode
            episode = self._store.store_episode(
                content={"text": text},
                embedding=embedding,
                surprise=0.0,
                entities=flat_entities,
                metadata=meta,
            )

            # Conflict detection & supersession
            self._detect_and_supersede(text, episode.episode_id)

            # Derived facts
            derived = self._generate_derived_facts(text)
            for fact_text in derived:
                fact_emb = self._embed(fact_text)
                fact_meta = {
                    "text": fact_text,
                    "session_id": self._session_id,
                    "derived_from": episode.episode_id,
                    "topics": topics,
                }
                self._store.store_episode(
                    content={"text": fact_text},
                    embedding=fact_emb,
                    surprise=0.0,
                    entities=flat_entities,
                    metadata=fact_meta,
                )

            # Update entity index
            for category, values in extracted.items():
                if values:
                    self._all_entities.setdefault(category, [])
                    for v in values:
                        if v not in self._all_entities[category]:
                            self._all_entities[category].append(v)

            return {
                "stored": True,
                "surprise": 0.0,
                "episode_id": episode.episode_id,
                "entities": flat_entities,
            }

    def recall(self, query: str, k: int = 7) -> List[Tuple[str, float]]:
        """Retrieve memories relevant to *query*.

        Returns list of (text, score) tuples.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        embedding = self._embed(query)
        results = self._retriever.retrieve(embedding)

        # Filter superseded
        results = [(ep, score) for ep, score in results if ep.episode_id not in self._superseded]

        # Topic-based supplementation
        query_topics = self._detect_topics(query)
        if query_topics:
            topic_results = self._recall_by_topics(
                query_topics, exclude_ids={ep.episode_id for ep, _ in results}
            )
            # Merge with a small boost
            for ep, base_score in topic_results:
                results.append((ep, base_score + 0.05))

        # Sort by score descending, take top k
        results.sort(key=lambda x: x[1], reverse=True)

        out: List[Tuple[str, float]] = []
        seen: set[str] = set()
        for ep, score in results:
            txt = self._episode_text(ep)
            if txt and txt not in seen:
                seen.add(txt)
                out.append((txt, float(score)))
            if len(out) >= k:
                break
        return out

    def _recall_by_topics(
        self,
        topics: List[str],
        exclude_ids: set[str] | None = None,
    ) -> List[Tuple[Episode, float]]:
        """Retrieve episodes that match given topics."""
        exclude_ids = exclude_ids or set()
        matches = []
        for ep in self._store.episodes:
            if ep.episode_id in self._superseded:
                continue
            if ep.episode_id in exclude_ids:
                continue
            ep_topics = ep.metadata.get("topics", [])
            overlap = len(set(ep_topics) & set(topics))
            if overlap > 0:
                matches.append((ep, 0.3 * overlap))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]

    def get_facts(self) -> List[Tuple[str, float]]:
        """Return non-superseded memories as (text, importance) pairs."""
        facts = []
        for ep in self._store.episodes:
            if ep.episode_id in self._superseded:
                continue
            txt = self._episode_text(ep)
            if txt:
                facts.append((txt, float(ep.importance)))
        return facts

    def get_profile(self) -> Dict[str, Any]:
        """Return a summary profile for this user."""
        return {
            "user_id": self.user_id,
            "total_memories": len(self._store.episodes),
            "sessions_count": len(self._sessions) + 1,  # +1 for current
            "first_seen": self._first_seen,
            "last_seen": self._last_seen,
            "entities": dict(self._all_entities),
        }

    def new_session(self) -> str:
        """Start a new session. Returns the new session id."""
        self._sessions.append(
            {
                "session_id": self._session_id,
                "ended": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._session_id = str(uuid.uuid4())
        self._save_meta()
        return self._session_id

    def save(self) -> None:
        """Persist state to disk."""
        with self._lock:
            self._store.save_state()
            self._save_meta()

    def close(self) -> None:
        """Save and release resources."""
        self.save()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _meta_path(self) -> Path:
        return self._user_dir / "user_meta.json"

    def _save_meta(self) -> None:
        meta = {
            "user_id": self.user_id,
            "session_id": self._session_id,
            "sessions": self._sessions,
            "first_seen": self._first_seen,
            "last_seen": self._last_seen,
            "entities": self._all_entities,
            "superseded": list(self._superseded),
        }
        tmp = self._meta_path().with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f)
        import os

        os.replace(str(tmp), str(self._meta_path()))

    def _load_meta(self) -> None:
        path = self._meta_path()
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                meta = json.load(f)
            self._sessions = meta.get("sessions", [])
            self._first_seen = meta.get("first_seen")
            self._last_seen = meta.get("last_seen")
            self._all_entities = meta.get("entities", {})
            self._superseded = set(meta.get("superseded", []))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load user meta: %s", e)
