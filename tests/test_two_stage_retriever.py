"""Tests for Two-Stage Retriever — similarity + temporal expansion."""
import numpy as np
import pytest
from datetime import datetime, timedelta
from elo_memory.memory.episodic_store import EpisodicMemoryConfig, EpisodicMemoryStore
from elo_memory.retrieval.two_stage_retriever import TwoStageRetriever, RetrievalConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    config = EpisodicMemoryConfig(
        embedding_dim=16,
        max_episodes=200,
        persistence_path=str(tmp_path / "store"),
        consolidation_min_episodes=500,
    )
    return EpisodicMemoryStore(config)


@pytest.fixture
def retriever(store):
    return TwoStageRetriever(store)


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _rand_emb(dim=16):
    return _norm(np.random.randn(dim))


# ---------------------------------------------------------------------------
# Basic retrieval
# ---------------------------------------------------------------------------

class TestBasicRetrieval:
    def test_empty_store_returns_empty(self, retriever):
        results = retriever.retrieve(query=_rand_emb())
        assert results == []

    def test_retrieves_stored_episodes(self, store, retriever):
        base_time = datetime(2025, 6, 1, 12, 0, 0)
        for i in range(10):
            store.store_episode(
                content=np.random.randn(10).astype(np.float32),
                embedding=_rand_emb(),
                timestamp=base_time + timedelta(minutes=i * 10),
            )
        results = retriever.retrieve(query=_rand_emb())
        assert len(results) > 0
        # Each result is (Episode, score)
        for ep, score in results:
            assert isinstance(score, float)

    def test_max_retrieved_respected(self, store):
        base_time = datetime(2025, 6, 1, 12, 0, 0)
        for i in range(20):
            store.store_episode(
                content=np.random.randn(10).astype(np.float32),
                embedding=_rand_emb(),
                timestamp=base_time + timedelta(minutes=i),
            )
        cfg = RetrievalConfig(max_retrieved=3, similarity_threshold=0.0)
        retriever = TwoStageRetriever(store, config=cfg)
        results = retriever.retrieve(query=_rand_emb())
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Similarity stage
# ---------------------------------------------------------------------------

class TestSimilarityStage:
    def test_similar_episode_ranked_higher(self, store, retriever):
        base_time = datetime.now()
        target_emb = _norm(np.ones(16))

        # Store one episode close to target
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_norm(np.ones(16) + np.random.randn(16) * 0.01),
            timestamp=base_time,
        )
        # Store several random episodes
        for i in range(9):
            store.store_episode(
                content=np.random.randn(10).astype(np.float32),
                embedding=_rand_emb(),
                timestamp=base_time + timedelta(minutes=i + 1),
            )

        cfg = RetrievalConfig(
            similarity_threshold=0.0,
            enable_temporal_expansion=False,
            similarity_weight=1.0,
            recency_weight=0.0,
            importance_weight=0.0,
        )
        retriever = TwoStageRetriever(store, config=cfg)
        results = retriever.retrieve(query=target_emb)
        assert len(results) > 0
        # The first result should be the similar episode (cosine ~1.0)
        top_ep, top_score = results[0]
        cos_sim = np.dot(target_emb, top_ep.embedding) / (
            np.linalg.norm(target_emb) * np.linalg.norm(top_ep.embedding) + 1e-8
        )
        assert cos_sim > 0.8

    def test_threshold_filters_low_similarity(self, store):
        base_time = datetime.now()
        for i in range(10):
            store.store_episode(
                content=np.random.randn(10).astype(np.float32),
                embedding=_rand_emb(),
                timestamp=base_time + timedelta(minutes=i),
            )
        cfg = RetrievalConfig(similarity_threshold=0.99)
        retriever = TwoStageRetriever(store, config=cfg)
        results = retriever.retrieve(query=_rand_emb())
        # Very high threshold should eliminate most/all
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Temporal expansion
# ---------------------------------------------------------------------------

class TestTemporalExpansion:
    def test_temporal_expansion_adds_adjacent(self, store):
        t0 = datetime(2025, 6, 1, 12, 0, 0)
        target_emb = _norm(np.ones(16))

        # Seed episode that will match
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_norm(np.ones(16) + np.random.randn(16) * 0.01),
            timestamp=t0,
        )
        # Adjacent episode (1 min later — within default 5-min window)
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_rand_emb(),
            timestamp=t0 + timedelta(minutes=1),
        )
        # Distant episode (1 hour later — outside window)
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_rand_emb(),
            timestamp=t0 + timedelta(hours=1),
        )

        cfg = RetrievalConfig(
            similarity_threshold=0.0,
            enable_temporal_expansion=True,
            temporal_window=5,
            max_retrieved=10,
        )
        retriever = TwoStageRetriever(store, config=cfg)
        results = retriever.retrieve(query=target_emb, query_time=t0)
        ep_timestamps = [ep.timestamp for ep, _ in results]
        # The adjacent episode at +1 min should be included via expansion
        assert any(abs((ts - t0).total_seconds()) <= 120 for ts in ep_timestamps)

    def test_no_temporal_expansion_when_disabled(self, store):
        t0 = datetime(2025, 6, 1, 12, 0, 0)
        target_emb = _norm(np.ones(16))
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_norm(np.ones(16) + np.random.randn(16) * 0.01),
            timestamp=t0,
        )
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_rand_emb(),
            timestamp=t0 + timedelta(minutes=1),
        )
        cfg = RetrievalConfig(
            similarity_threshold=0.0,
            enable_temporal_expansion=False,
            max_retrieved=10,
        )
        retriever = TwoStageRetriever(store, config=cfg)
        results_no_expand = retriever.retrieve(query=target_emb, query_time=t0)

        cfg.enable_temporal_expansion = True
        retriever_expand = TwoStageRetriever(store, config=cfg)
        results_expand = retriever_expand.retrieve(query=target_emb, query_time=t0)

        # With expansion we should get at least as many results
        assert len(results_expand) >= len(results_no_expand)


# ---------------------------------------------------------------------------
# Final ranking
# ---------------------------------------------------------------------------

class TestFinalRanking:
    def test_scores_are_descending(self, store, retriever):
        t0 = datetime.now()
        for i in range(15):
            store.store_episode(
                content=np.random.randn(10).astype(np.float32),
                embedding=_rand_emb(),
                surprise=float(i),
                timestamp=t0 + timedelta(minutes=i),
            )
        results = retriever.retrieve(query=_rand_emb())
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_recency_weight_favors_recent(self, store):
        t_old = datetime.now() - timedelta(days=30)
        t_new = datetime.now()
        emb = _norm(np.ones(16))

        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_norm(np.ones(16) + np.random.randn(16) * 0.01),
            timestamp=t_old,
            surprise=1.0,
        )
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_norm(np.ones(16) + np.random.randn(16) * 0.01),
            timestamp=t_new,
            surprise=1.0,
        )
        cfg = RetrievalConfig(
            similarity_threshold=0.0,
            enable_temporal_expansion=False,
            similarity_weight=0.0,
            recency_weight=1.0,
            importance_weight=0.0,
        )
        retriever = TwoStageRetriever(store, config=cfg)
        results = retriever.retrieve(query=emb, query_time=t_new)
        assert len(results) >= 2
        # Most recent should rank first
        assert results[0][0].timestamp >= results[-1][0].timestamp


# ---------------------------------------------------------------------------
# Contextual / temporal cue retrieval
# ---------------------------------------------------------------------------

class TestCueRetrieval:
    def test_retrieve_by_contextual_cue_location(self, store, retriever):
        store.store_episode(content=np.random.randn(10).astype(np.float32), location="cafe")
        store.store_episode(content=np.random.randn(10).astype(np.float32), location="office")
        results = retriever.retrieve_by_contextual_cue(location="cafe")
        assert len(results) == 1
        assert results[0].location == "cafe"

    def test_retrieve_by_contextual_cue_entities(self, store, retriever):
        store.store_episode(
            content=np.random.randn(10).astype(np.float32), entities=["alice"]
        )
        store.store_episode(
            content=np.random.randn(10).astype(np.float32), entities=["bob"]
        )
        results = retriever.retrieve_by_contextual_cue(entities=["alice"])
        assert len(results) == 1

    def test_contextual_cue_deduplicates(self, store, retriever):
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            location="cafe",
            entities=["alice"],
        )
        results = retriever.retrieve_by_contextual_cue(location="cafe", entities=["alice"])
        assert len(results) == 1  # same episode, should not appear twice


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Temporal cue retrieval
# ---------------------------------------------------------------------------

class TestTemporalCueRetrieval:
    def test_yesterday(self, store, retriever):
        from datetime import datetime, timedelta
        yesterday = datetime.now() - timedelta(days=1) + timedelta(hours=2)
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_rand_emb(),
            timestamp=yesterday,
        )
        results = retriever.retrieve_by_temporal_cue("yesterday")
        assert len(results) >= 1

    def test_last_week(self, store, retriever):
        from datetime import datetime, timedelta
        three_days_ago = datetime.now() - timedelta(days=3)
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_rand_emb(),
            timestamp=three_days_ago,
        )
        results = retriever.retrieve_by_temporal_cue("last week")
        assert len(results) >= 1

    def test_default_fallback(self, store, retriever):
        from datetime import datetime, timedelta
        recent = datetime.now() - timedelta(hours=2)
        store.store_episode(
            content=np.random.randn(10).astype(np.float32),
            embedding=_rand_emb(),
            timestamp=recent,
        )
        results = retriever.retrieve_by_temporal_cue("some unknown phrase")
        assert len(results) >= 1

    def test_respects_k(self, store, retriever):
        from datetime import datetime, timedelta
        for i in range(10):
            store.store_episode(
                content=np.random.randn(10).astype(np.float32),
                embedding=_rand_emb(),
                timestamp=datetime.now() - timedelta(hours=i),
            )
        results = retriever.retrieve_by_temporal_cue("last week", k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_none_query_raises(self, store, retriever):
        with pytest.raises(ValueError, match="query must be"):
            retriever.retrieve(query=None)


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self, retriever):
        v = _rand_emb()
        assert retriever._cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self, retriever):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert retriever._cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector(self, retriever):
        v = _rand_emb(2)
        z = np.zeros(2, dtype=np.float32)
        assert retriever._cosine_similarity(v, z) == 0.0
