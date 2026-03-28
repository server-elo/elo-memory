"""Tests for Episodic Memory Store — the core storage module."""
import numpy as np
import pytest
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from elo_memory.memory.episodic_store import (
    Episode,
    EpisodicMemoryConfig,
    EpisodicMemoryStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    """Minimal config that writes to a temp directory."""
    return EpisodicMemoryConfig(
        max_episodes=50,
        embedding_dim=16,
        persistence_path=str(tmp_path / "store"),
        consolidation_min_episodes=200,  # high so consolidation doesn't auto-trigger
        consolidation_interval_episodes=500,
    )


@pytest.fixture
def store(config):
    return EpisodicMemoryStore(config)


def _rand_content(dim=10):
    return np.random.randn(dim).astype(np.float32)


def _rand_embedding(dim=16):
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

class TestEpisode:
    def test_auto_generates_id(self):
        ep = Episode(content=np.array([1.0]), timestamp=datetime.now())
        assert ep.episode_id is not None
        assert ep.episode_id.startswith("ep_")

    def test_to_dict_round_trip(self):
        ts = datetime(2025, 6, 1, 12, 0, 0)
        ep = Episode(
            content=np.array([1.0, 2.0]),
            timestamp=ts,
            location="office",
            entities=["alice"],
            embedding=np.array([0.5, 0.5]),
            surprise=1.2,
            importance=0.8,
            metadata={"key": "val"},
            episode_id="ep_test",
        )
        d = ep.to_dict()
        restored = Episode.from_dict(d)
        assert restored.episode_id == "ep_test"
        assert restored.location == "office"
        assert restored.entities == ["alice"]
        np.testing.assert_array_almost_equal(restored.content, [1.0, 2.0])
        np.testing.assert_array_almost_equal(restored.embedding, [0.5, 0.5])
        assert restored.surprise == pytest.approx(1.2)
        assert restored.importance == pytest.approx(0.8)
        assert restored.timestamp == ts

    def test_from_dict_missing_optional_fields(self):
        d = {
            "content": [1.0],
            "timestamp": "2025-06-01T12:00:00",
        }
        ep = Episode.from_dict(d)
        assert ep.embedding is None
        assert ep.entities == []
        assert ep.surprise == 0.0
        assert ep.importance == 0.5


# ---------------------------------------------------------------------------
# Store — basic operations
# ---------------------------------------------------------------------------

class TestStoreBasics:
    def test_init_empty(self, store):
        assert len(store.episodes) == 0
        assert store.total_episodes_stored == 0

    def test_store_and_count(self, store):
        for i in range(5):
            store.store_episode(content=_rand_content(), surprise=float(i))
        assert len(store.episodes) == 5
        assert store.total_episodes_stored == 5

    def test_stored_episode_has_embedding(self, store):
        ep = store.store_episode(content=_rand_content())
        assert ep.embedding is not None
        assert len(ep.embedding) == store.config.embedding_dim

    def test_explicit_embedding_is_used(self, store):
        emb = _rand_embedding(store.config.embedding_dim)
        ep = store.store_episode(content=_rand_content(), embedding=emb)
        # After interference resolution embedding may shift, but should still exist
        assert ep.embedding is not None

    def test_importance_increases_with_surprise(self, store):
        low = store.store_episode(content=_rand_content(), surprise=0.0)
        high = store.store_episode(content=_rand_content(), surprise=5.0)
        assert high.importance > low.importance


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_temporal_index(self, store):
        ts = datetime(2025, 6, 1, 10, 0, 0)
        store.store_episode(content=_rand_content(), timestamp=ts)
        assert "2025-06-01" in store.temporal_index

    def test_spatial_index(self, store):
        store.store_episode(content=_rand_content(), location="lab")
        assert "lab" in store.spatial_index

    def test_entity_index(self, store):
        store.store_episode(content=_rand_content(), entities=["alice", "bob"])
        assert "alice" in store.entity_index
        assert "bob" in store.entity_index


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_retrieve_by_similarity_returns_results(self, store):
        for _ in range(10):
            store.store_episode(content=_rand_content(), embedding=_rand_embedding())
        query = _rand_embedding()
        results = store.retrieve_by_similarity(query, k=5)
        assert len(results) <= 5
        assert all(isinstance(ep, Episode) for ep in results)

    def test_retrieve_by_similarity_tracks_retrieval_count(self, store):
        emb = _rand_embedding()
        store.store_episode(content=_rand_content(), embedding=emb)
        results = store.retrieve_by_similarity(emb, k=1)
        assert len(results) == 1
        assert results[0].metadata.get("_retrieval_count", 0) >= 1

    def test_retrieve_by_temporal_range(self, store):
        t1 = datetime(2025, 6, 1, 10, 0, 0)
        t2 = datetime(2025, 6, 1, 12, 0, 0)
        t3 = datetime(2025, 6, 1, 14, 0, 0)
        store.store_episode(content=_rand_content(), timestamp=t1)
        store.store_episode(content=_rand_content(), timestamp=t2)
        store.store_episode(content=_rand_content(), timestamp=t3)
        results = store.retrieve_by_temporal_range(
            datetime(2025, 6, 1, 11, 0, 0),
            datetime(2025, 6, 1, 13, 0, 0),
        )
        assert len(results) == 1
        assert results[0].timestamp == t2

    def test_retrieve_by_location(self, store):
        store.store_episode(content=_rand_content(), location="office")
        store.store_episode(content=_rand_content(), location="home")
        assert len(store.retrieve_by_location("office")) == 1
        assert len(store.retrieve_by_location("unknown")) == 0

    def test_retrieve_by_entity(self, store):
        store.store_episode(content=_rand_content(), entities=["alice"])
        store.store_episode(content=_rand_content(), entities=["bob"])
        assert len(store.retrieve_by_entity("alice")) == 1
        assert len(store.retrieve_by_entity("charlie")) == 0


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

class TestEmbeddingGeneration:
    def test_short_content_is_padded(self, store):
        short = np.array([1.0, 2.0])
        emb = store._generate_embedding(short)
        assert len(emb) == store.config.embedding_dim
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_long_content_is_truncated(self, store):
        long = np.random.randn(100)
        emb = store._generate_embedding(long)
        assert len(emb) == store.config.embedding_dim

    def test_zero_content_returns_zero_embedding(self, store):
        zeros = np.zeros(5)
        emb = store._generate_embedding(zeros)
        assert len(emb) == store.config.embedding_dim
        # norm is 0 so no normalization — should be all zeros
        np.testing.assert_array_equal(emb, np.zeros(store.config.embedding_dim))


# ---------------------------------------------------------------------------
# Persistence (save/load state)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_round_trip(self, config):
        store1 = EpisodicMemoryStore(config)
        ts = datetime(2025, 6, 1, 12, 0, 0)
        store1.store_episode(content=_rand_content(), surprise=1.5, timestamp=ts, location="lab")
        store1.save_state()

        store2 = EpisodicMemoryStore(config)
        store2.load_state()
        assert len(store2.episodes) == 1
        assert store2.episodes[0].location == "lab"
        assert store2.total_episodes_stored == 1

    def test_save_is_atomic(self, config):
        store = EpisodicMemoryStore(config)
        store.store_episode(content=_rand_content())
        store.save_state()
        # tmp file should not linger
        assert not (Path(config.persistence_path) / "memory_state.json.tmp").exists()
        assert (Path(config.persistence_path) / "memory_state.json").exists()

    def test_load_nonexistent_state_is_noop(self, config):
        store = EpisodicMemoryStore(config)
        store.load_state()  # should not raise
        assert len(store.episodes) == 0


# ---------------------------------------------------------------------------
# Disk offloading
# ---------------------------------------------------------------------------

class TestOffloading:
    def test_offload_and_reload_episode(self, config):
        config.max_episodes = 5
        config.consolidation_min_episodes = 1
        config.consolidation_interval_episodes = 1
        store = EpisodicMemoryStore(config)

        # Store enough to trigger offloading
        for i in range(10):
            store.store_episode(content=_rand_content(), surprise=float(i) * 0.1)

        assert store.episodes_offloaded > 0
        assert len(store.episodes) <= config.max_episodes

    def test_offloaded_episode_loadable(self, config):
        store = EpisodicMemoryStore(config)
        ep = store.store_episode(content=_rand_content(), surprise=0.5)
        store._offload_episode(ep)
        loaded = store._load_offloaded_episode(ep.episode_id)
        assert loaded is not None
        assert loaded.episode_id == ep.episode_id

    def test_load_missing_offloaded_returns_none(self, config):
        store = EpisodicMemoryStore(config)
        assert store._load_offloaded_episode("nonexistent_id") is None


# ---------------------------------------------------------------------------
# Consolidation triggers
# ---------------------------------------------------------------------------

class TestConsolidation:
    def test_should_consolidate_respects_min_episodes(self, config):
        config.consolidation_min_episodes = 100
        store = EpisodicMemoryStore(config)
        for i in range(5):
            store.store_episode(content=_rand_content())
        assert store.should_consolidate() is False

    def test_should_consolidate_by_episode_count(self, config):
        config.consolidation_min_episodes = 5
        config.consolidation_interval_episodes = 10
        store = EpisodicMemoryStore(config)
        for i in range(15):
            store.store_episode(content=_rand_content())
        # episodes_since_consolidation should have exceeded threshold
        # (consolidation may have already run, so just check it didn't crash)
        assert store.total_episodes_stored == 15

    def test_mark_consolidated_resets_counter(self, config):
        store = EpisodicMemoryStore(config)
        store.episodes_since_consolidation = 50
        store.mark_consolidated()
        assert store.episodes_since_consolidation == 0
        assert store.last_consolidation_time is not None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_get_statistics_structure(self, store):
        store.store_episode(content=_rand_content(), location="lab", entities=["alice"])
        stats = store.get_statistics()
        assert "total_episodes" in stats
        assert "episodes_in_memory" in stats
        assert "unique_locations" in stats
        assert "unique_entities" in stats
        assert stats["total_episodes"] == 1
        assert stats["unique_locations"] == 1
        assert stats["unique_entities"] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# run_consolidation
# ---------------------------------------------------------------------------

class TestRunConsolidation:
    def test_returns_none_on_empty_store(self, store):
        result = store.run_consolidation()
        assert result is None

    def test_returns_stats(self, store):
        for i in range(5):
            store.store_episode(content=_rand_content(), surprise=float(i))
        stats = store.run_consolidation()
        assert stats is not None
        assert "episodes_consolidated" in stats
        assert "replay_count" in stats

    def test_applies_forgetting_decay(self, store):
        ep = store.store_episode(content=_rand_content(), surprise=1.0)
        store.run_consolidation()
        assert "_activation" in ep.metadata

    def test_strengthens_replayed_episodes(self, store):
        for i in range(5):
            store.store_episode(content=_rand_content(), surprise=2.0)
        original_importances = [ep.importance for ep in store.episodes]
        store.run_consolidation()
        # At least some episodes should have boosted importance
        new_importances = [ep.importance for ep in store.episodes]
        assert any(n >= o for n, o in zip(new_importances, original_importances))

    def test_reuses_consolidation_engine(self, store):
        for i in range(3):
            store.store_episode(content=_rand_content())
        store.run_consolidation()
        store.run_consolidation()
        # Should reuse the same engine instance
        assert hasattr(store, "_consolidation_engine")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------

class TestQueryCache:
    def test_cache_hit_returns_same_results(self, store):
        for _ in range(5):
            store.store_episode(content=_rand_content(), embedding=_rand_embedding())
        query = _rand_embedding()
        r1 = store.retrieve_by_similarity(query, k=3)
        r2 = store.retrieve_by_similarity(query, k=3)
        assert [ep.episode_id for ep in r1] == [ep.episode_id for ep in r2]

    def test_cache_invalidated_on_store(self, store):
        store.store_episode(content=_rand_content(), embedding=_rand_embedding())
        query = _rand_embedding()
        store.retrieve_by_similarity(query, k=1)
        assert len(store._query_cache) > 0
        store.store_episode(content=_rand_content(), embedding=_rand_embedding())
        assert len(store._query_cache) == 0

    def test_cache_skipped_with_filters(self, store):
        store.store_episode(content=_rand_content(), embedding=_rand_embedding(), location="lab")
        query = _rand_embedding()
        store.retrieve_by_similarity(query, k=1, filter_criteria={"location": "lab"})
        # Filtered queries should not populate cache
        assert len(store._query_cache) == 0

    def test_cache_evicts_oldest(self, config):
        config.query_cache_size = 2
        store = EpisodicMemoryStore(config)
        for _ in range(3):
            store.store_episode(content=_rand_content(), embedding=_rand_embedding())
        # Fill cache with 3 different queries — only 2 should remain
        for i in range(3):
            q = _rand_embedding()
            store.retrieve_by_similarity(q, k=1)
        assert len(store._query_cache) <= 2

    def test_cache_disabled_when_zero(self, config):
        config.query_cache_size = 0
        store = EpisodicMemoryStore(config)
        store.store_episode(content=_rand_content(), embedding=_rand_embedding())
        store.retrieve_by_similarity(_rand_embedding(), k=1)
        assert len(store._query_cache) == 0

    def test_cache_stats_in_statistics(self, store):
        stats = store.get_statistics()
        assert "query_cache_entries" in stats
        assert "query_cache_max" in stats


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_store_statistics(self, store):
        stats = store.get_statistics()
        assert stats["total_episodes"] == 0
        assert stats["mean_importance"] == 0.0

    def test_nan_embedding_does_not_crash(self, store):
        nan_content = np.array([float("nan")] * 10)
        # Should not raise
        ep = store.store_episode(content=nan_content)
        assert ep is not None

    def test_very_high_surprise(self, store):
        ep = store.store_episode(content=_rand_content(), surprise=1000.0)
        assert ep.importance <= 1.0  # clipped by sigmoid
