"""
Proactive Predictive Memory Pre-fetcher
========================================

Uses a lightweight future-forecasting model to predict upcoming
goals/contexts and pre-loads relevant memories before they're needed.

Turns memory from reactive retrieval into anticipatory intelligence.

The pre-fetcher tracks:
1. Topic transition probabilities (Markov chain)
2. Temporal query patterns (time-of-day, day-of-week)
3. Entity co-occurrence (if you ask about X, you'll likely ask about Y)

References:
- Schacter & Addis (2007): Constructive Memory and Future Thinking
- Szpunar et al. (2014): Memory and the Prospective Brain
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrefetchConfig:
    """Configuration for the predictive pre-fetcher."""
    # History
    max_query_history: int = 500          # Max queries to remember
    topic_transition_smoothing: float = 0.1  # Laplace smoothing for transition probs

    # Cache
    cache_size: int = 50                  # Max pre-fetched results to cache
    cache_ttl_seconds: float = 300.0      # Cache entry time-to-live

    # Prediction
    top_k_predictions: int = 5            # Number of predicted next queries
    temporal_bins: int = 24               # Hour-of-day bins
    min_observations: int = 5             # Min observations before predicting


@dataclass
class CachedResult:
    """A pre-fetched memory result in the warm cache."""
    query_text: str
    results: List[Tuple[str, float]]  # (text, score) pairs
    created_at: float                 # time.time()
    hit_count: int = 0


class PredictivePrefetcher:
    """
    Anticipatory memory retrieval system.

    Observes query patterns and pre-loads memories that are likely
    to be needed next.
    """

    def __init__(self, config: Optional[PrefetchConfig] = None):
        self.config = config or PrefetchConfig()

        # Query history
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.config.max_query_history)

        # Topic transition matrix: topic_a → topic_b → count
        self._transitions: Dict[str, Counter] = defaultdict(Counter)
        self._last_topics: List[str] = []

        # Entity co-occurrence: entity → set of co-occurring entities
        self._entity_cooccurrence: Dict[str, Counter] = defaultdict(Counter)

        # Temporal patterns: hour_of_day → topic → count
        self._temporal_patterns: Dict[int, Counter] = defaultdict(Counter)

        # Warm cache: query_key → CachedResult
        self._cache: Dict[str, CachedResult] = {}

    # ── Observation ──────────────────────────────────────────────

    def observe_query(
        self,
        query_text: str,
        topics: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        results: Optional[List[Tuple[str, float]]] = None,
    ) -> None:
        """
        Record a query for pattern learning.

        Call this after every retrieval to build the prediction model.
        """
        now = datetime.now(timezone.utc)
        topics = topics or []
        entities = entities or []

        # Record in history
        self._history.append({
            "query": query_text,
            "topics": topics,
            "entities": entities,
            "hour": now.hour,
            "timestamp": time.time(),
        })

        # Update topic transitions
        for prev_topic in self._last_topics:
            for topic in topics:
                self._transitions[prev_topic][topic] += 1
        self._last_topics = topics

        # Update entity co-occurrence
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                self._entity_cooccurrence[e1][e2] += 1
                self._entity_cooccurrence[e2][e1] += 1

        # Update temporal patterns
        for topic in topics:
            self._temporal_patterns[now.hour][topic] += 1

        # Cache the results if provided
        if results is not None:
            cache_key = self._cache_key(query_text)
            self._cache[cache_key] = CachedResult(
                query_text=query_text,
                results=results,
                created_at=time.time(),
            )
            self._evict_cache()

    # ── Prediction ───────────────────────────────────────────────

    def predict_next_queries(
        self,
        current_topics: Optional[List[str]] = None,
        current_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict what the user is likely to query next.

        Returns a ranked list of predicted query contexts.
        """
        predictions: List[Dict[str, Any]] = []
        scores: Dict[str, float] = defaultdict(float)

        # 1. Topic transition predictions
        if current_topics:
            for topic in current_topics:
                if topic in self._transitions:
                    transition_counts = self._transitions[topic]
                    total = sum(transition_counts.values()) + self.config.topic_transition_smoothing
                    for next_topic, count in transition_counts.most_common(self.config.top_k_predictions):
                        prob = (count + self.config.topic_transition_smoothing) / total
                        scores[f"topic:{next_topic}"] += prob

        # 2. Entity co-occurrence predictions
        if current_entities:
            for entity in current_entities:
                if entity in self._entity_cooccurrence:
                    for related, count in self._entity_cooccurrence[entity].most_common(5):
                        scores[f"entity:{related}"] += count * 0.1

        # 3. Temporal predictions (what's typically queried at this hour)
        hour = datetime.now(timezone.utc).hour
        if hour in self._temporal_patterns:
            total = sum(self._temporal_patterns[hour].values())
            if total >= self.config.min_observations:
                for topic, count in self._temporal_patterns[hour].most_common(5):
                    scores[f"temporal:{topic}"] += (count / total) * 0.5

        # Rank and format predictions
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for pred_key, score in ranked[:self.config.top_k_predictions]:
            kind, value = pred_key.split(":", 1)
            predictions.append({
                "type": kind,
                "value": value,
                "confidence": min(1.0, score),
            })

        return predictions

    def predict_next_topics(self, current_topics: List[str]) -> List[Tuple[str, float]]:
        """Predict the most likely next topics with probabilities."""
        topic_scores: Dict[str, float] = {}

        for topic in current_topics:
            if topic in self._transitions:
                total = sum(self._transitions[topic].values())
                for next_topic, count in self._transitions[topic].items():
                    topic_scores[next_topic] = topic_scores.get(next_topic, 0.0) + count / max(total, 1)

        ranked = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:self.config.top_k_predictions]

    # ── Cache ────────────────────────────────────────────────────

    def prefetch(
        self,
        predictions: List[Dict[str, Any]],
        retrieval_fn: Any,
    ) -> None:
        """
        Pre-load memories for predicted queries.

        Args:
            predictions: Output of predict_next_queries()
            retrieval_fn: Callable(query_text, k) → List[(text, score)]
        """
        for pred in predictions:
            query = pred["value"]
            cache_key = self._cache_key(query)

            if cache_key in self._cache:
                continue  # Already cached

            try:
                results = retrieval_fn(query, 5)
                self._cache[cache_key] = CachedResult(
                    query_text=query,
                    results=results,
                    created_at=time.time(),
                )
            except Exception as e:
                logger.debug("Prefetch failed for %s: %s", query, e)

        self._evict_cache()

    def get_cached(self, query_text: str) -> Optional[List[Tuple[str, float]]]:
        """Return pre-fetched results if available and fresh."""
        cache_key = self._cache_key(query_text)
        entry = self._cache.get(cache_key)
        if entry is None:
            return None

        # Check TTL
        age = time.time() - entry.created_at
        if age > self.config.cache_ttl_seconds:
            del self._cache[cache_key]
            return None

        entry.hit_count += 1
        return entry.results

    def _cache_key(self, text: str) -> str:
        return text.lower().strip()[:100]

    def _evict_cache(self) -> None:
        """Evict expired and excess cache entries."""
        now = time.time()
        # Remove expired
        expired = [
            k for k, v in self._cache.items()
            if now - v.created_at > self.config.cache_ttl_seconds
        ]
        for k in expired:
            del self._cache[k]

        # Remove oldest if over capacity
        while len(self._cache) > self.config.cache_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]

    # ── Statistics ───────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        total_hits = sum(e.hit_count for e in self._cache.values())
        return {
            "query_history_size": len(self._history),
            "topic_transitions": sum(sum(c.values()) for c in self._transitions.values()),
            "entity_cooccurrences": sum(sum(c.values()) for c in self._entity_cooccurrence.values()),
            "temporal_patterns": sum(sum(c.values()) for c in self._temporal_patterns.values()),
            "cache_size": len(self._cache),
            "cache_hits": total_hits,
        }
