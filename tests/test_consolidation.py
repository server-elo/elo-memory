"""Tests for Memory Consolidation Engine — replay, schemas, prioritization."""
import numpy as np
import pytest
from datetime import datetime, timedelta
from elo_memory.consolidation.memory_consolidation import (
    MemoryConsolidationEngine,
    ConsolidationConfig,
)
from elo_memory.memory.episodic_store import Episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(
    episode_id="ep_1",
    location=None,
    entities=None,
    surprise=1.0,
    hours_ago=0,
):
    """Create an Episode for testing."""
    return Episode(
        content=np.random.randn(10).astype(np.float32),
        timestamp=datetime.now() - timedelta(hours=hours_ago),
        location=location,
        entities=entities or [],
        embedding=None,
        surprise=surprise,
        importance=0.5,
        episode_id=episode_id,
    )


@pytest.fixture
def engine():
    return MemoryConsolidationEngine()


# ---------------------------------------------------------------------------
# Prioritization
# ---------------------------------------------------------------------------

class TestPrioritization:
    def test_returns_probabilities_summing_to_one(self, engine):
        episodes = [_make_episode(f"ep_{i}", surprise=float(i)) for i in range(10)]
        _, probs = engine.prioritize_episodes(episodes)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_higher_surprise_gets_higher_probability(self, engine):
        low = _make_episode("low", surprise=0.1)
        high = _make_episode("high", surprise=10.0)
        _, probs = engine.prioritize_episodes([low, high])
        assert probs[1] > probs[0]

    def test_more_recent_gets_higher_probability(self, engine):
        old = _make_episode("old", surprise=1.0, hours_ago=72)
        new = _make_episode("new", surprise=1.0, hours_ago=0)
        _, probs = engine.prioritize_episodes([old, new])
        assert probs[1] > probs[0]

    def test_custom_priorities_override(self, engine):
        episodes = [_make_episode(f"ep_{i}") for i in range(3)]
        custom = np.array([10.0, 1.0, 1.0])
        _, probs = engine.prioritize_episodes(episodes, priorities=custom)
        assert probs[0] > probs[1]
        assert probs[0] > probs[2]


# ---------------------------------------------------------------------------
# Replay sampling
# ---------------------------------------------------------------------------

class TestReplaySampling:
    def test_batch_size_respected(self, engine):
        episodes = [_make_episode(f"ep_{i}") for i in range(20)]
        probs = np.ones(20) / 20
        batch = engine.sample_for_replay(episodes, probs, batch_size=5)
        assert len(batch) == 5

    def test_small_pool_returns_all(self, engine):
        episodes = [_make_episode(f"ep_{i}") for i in range(3)]
        probs = np.ones(3) / 3
        batch = engine.sample_for_replay(episodes, probs, batch_size=10)
        assert len(batch) == 3

    def test_no_duplicates_in_batch(self, engine):
        episodes = [_make_episode(f"ep_{i}") for i in range(20)]
        probs = np.ones(20) / 20
        batch = engine.sample_for_replay(episodes, probs, batch_size=10)
        ids = [ep.episode_id for ep in batch]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------

class TestSchemaExtraction:
    def test_extracts_location_pattern(self, engine):
        episodes = [
            _make_episode(f"ep_{i}", location="conference", entities=["alice", "bob"])
            for i in range(5)
        ]
        schemas = engine.extract_schemas(episodes)
        assert len(schemas) == 1
        assert schemas[0]["location"] == "conference"
        assert schemas[0]["frequency"] == 5

    def test_below_threshold_no_schema(self, engine):
        engine.config.schema_threshold = 5
        episodes = [_make_episode(f"ep_{i}", location="lab") for i in range(3)]
        schemas = engine.extract_schemas(episodes)
        assert len(schemas) == 0

    def test_common_entities_in_schema(self, engine):
        # 4 episodes with alice, 4 with bob, 1 with charlie
        episodes = []
        for i in range(4):
            episodes.append(_make_episode(f"ep_a{i}", location="room", entities=["alice", "bob"]))
        episodes.append(_make_episode("ep_c", location="room", entities=["charlie"]))

        schemas = engine.extract_schemas(episodes)
        assert len(schemas) == 1
        # alice and bob appear in >50% → common; charlie does not
        assert "alice" in schemas[0]["common_entities"]
        assert "bob" in schemas[0]["common_entities"]
        assert "charlie" not in schemas[0]["common_entities"]

    def test_multiple_locations_multiple_schemas(self, engine):
        engine.config.schema_threshold = 2
        episodes = []
        for i in range(3):
            episodes.append(_make_episode(f"office_{i}", location="office"))
        for i in range(3):
            episodes.append(_make_episode(f"cafe_{i}", location="cafe"))
        schemas = engine.extract_schemas(episodes)
        locations = {s["location"] for s in schemas}
        assert "office" in locations
        assert "cafe" in locations

    def test_episodes_without_location_ignored(self, engine):
        episodes = [_make_episode(f"ep_{i}", location=None) for i in range(10)]
        schemas = engine.extract_schemas(episodes)
        assert len(schemas) == 0


# ---------------------------------------------------------------------------
# Full consolidation cycle
# ---------------------------------------------------------------------------

class TestConsolidation:
    def test_consolidate_returns_stats(self, engine):
        episodes = [
            _make_episode(f"ep_{i}", location="office", surprise=float(i))
            for i in range(10)
        ]
        stats = engine.consolidate(episodes)
        assert "episodes_consolidated" in stats
        assert "replay_count" in stats
        assert "schemas_extracted" in stats
        assert stats["episodes_consolidated"] == 10

    def test_update_callback_invoked(self, engine):
        episodes = [_make_episode(f"ep_{i}") for i in range(5)]
        call_count = {"n": 0}

        def callback(ep):
            call_count["n"] += 1

        engine.config.replay_iterations = 1
        engine.consolidate(episodes, update_callback=callback)
        assert call_count["n"] > 0

    def test_consolidation_updates_timestamp(self, engine):
        old_time = engine.last_consolidation
        episodes = [_make_episode(f"ep_{i}") for i in range(5)]
        engine.consolidate(episodes)
        assert engine.last_consolidation >= old_time

    def test_schemas_accumulate(self, engine):
        engine.config.schema_threshold = 2
        batch1 = [_make_episode(f"b1_{i}", location="office") for i in range(3)]
        batch2 = [_make_episode(f"b2_{i}", location="cafe") for i in range(3)]
        engine.consolidate(batch1)
        engine.consolidate(batch2)
        assert len(engine.schemas) >= 2


# ---------------------------------------------------------------------------
# Schema summary
# ---------------------------------------------------------------------------

class TestSchemaSummary:
    def test_summary_format(self, engine):
        engine.config.schema_threshold = 2
        episodes = [_make_episode(f"ep_{i}", location="office") for i in range(5)]
        engine.consolidate(episodes)
        summaries = engine.get_schema_summary()
        assert len(summaries) > 0
        s = summaries[0]
        assert "type" in s
        assert "pattern" in s
        assert "frequency" in s


# ---------------------------------------------------------------------------
# should_consolidate
# ---------------------------------------------------------------------------

class TestShouldConsolidate:
    def test_returns_true_after_interval(self):
        engine = MemoryConsolidationEngine(
            ConsolidationConfig(consolidation_interval=timedelta(seconds=0))
        )
        assert engine.should_consolidate() is True

    def test_returns_false_before_interval(self):
        engine = MemoryConsolidationEngine(
            ConsolidationConfig(consolidation_interval=timedelta(hours=24))
        )
        assert engine.should_consolidate() is False
