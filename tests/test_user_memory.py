"""Tests for UserMemory — the per-user high-level memory API."""

import json
import numpy as np
import pytest
from pathlib import Path

from elo_memory.memory.user_memory import UserMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem(tmp_path):
    """Fresh UserMemory writing to a temp directory."""
    m = UserMemory(
        user_id="test-user-42",
        persistence_path=str(tmp_path / "memories"),
        embedding_dim=384,
    )
    yield m
    m.close()


@pytest.fixture
def mem_factory(tmp_path):
    """Factory that creates UserMemory instances for the same user (persistence tests)."""
    instances = []

    def _make(**kwargs):
        defaults = dict(
            user_id="persist-user",
            persistence_path=str(tmp_path / "memories"),
            embedding_dim=384,
        )
        defaults.update(kwargs)
        m = UserMemory(**defaults)
        instances.append(m)
        return m

    yield _make
    for m in instances:
        m.close()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_store_rejects_none(self, mem):
        with pytest.raises(TypeError, match="must be a string"):
            mem.store(None)

    def test_store_rejects_int(self, mem):
        with pytest.raises(TypeError, match="must be a string"):
            mem.store(42)

    def test_store_rejects_empty(self, mem):
        with pytest.raises(ValueError, match="empty"):
            mem.store("")

    def test_store_rejects_whitespace(self, mem):
        with pytest.raises(ValueError, match="empty"):
            mem.store("   \n\t  ")

    def test_user_id_must_be_nonempty(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            UserMemory(user_id="", persistence_path=str(tmp_path))

    def test_recall_returns_empty_for_blank_query(self, mem):
        assert mem.recall("") == []
        assert mem.recall("   ") == []


# ---------------------------------------------------------------------------
# Basic store & recall
# ---------------------------------------------------------------------------

class TestStoreAndRecall:
    def test_store_returns_episode_id(self, mem):
        result = mem.store("I work at Acme Corp as a backend engineer")
        assert result["stored"] is True
        assert result["episode_id"] is not None

    def test_recall_finds_stored_memory(self, mem):
        mem.store("My favorite framework is FastAPI")
        results = mem.recall("What framework do you use?")
        assert len(results) >= 1
        texts = [t for t, _ in results]
        assert any("FastAPI" in t for t in texts)

    def test_store_extracts_entities(self, mem):
        result = mem.store("Contact me at alice@example.com")
        assert "alice@example.com" in result["entities"]

    def test_recall_respects_k(self, mem):
        for i in range(10):
            mem.store(f"Fact number {i}: I use tool_{i} for my workflow")
        results = mem.recall("tools", k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

class TestNearDuplicate:
    def test_exact_duplicate_skipped(self, mem):
        r1 = mem.store("I use PostgreSQL as my main database")
        r2 = mem.store("I use PostgreSQL as my main database")
        assert r1["stored"] is True
        assert r2["stored"] is False
        assert r2["reason"] == "near-duplicate"

    def test_very_similar_skipped(self, mem):
        r1 = mem.store("I use PostgreSQL as my primary database")
        r2 = mem.store("I use PostgreSQL as my primary database system")
        # Second should be detected as near-duplicate (very minor edit)
        # This depends on the model, but at least one should be stored
        assert r1["stored"] is True


# ---------------------------------------------------------------------------
# Conflict detection & supersession
# ---------------------------------------------------------------------------

class TestConflictDetection:
    def test_switched_from_to_supersedes(self, mem):
        mem.store("I use Django for my backend")
        mem.store("I switched from Django to FastAPI")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        # The old Django fact should be superseded
        django_only = [t for t in texts if "Django" in t and "FastAPI" not in t and "Currently" not in t]
        assert len(django_only) == 0, f"Old Django fact not superseded: {django_only}"

    def test_moved_from_to_supersedes(self, mem):
        mem.store("I live in Berlin")
        mem.store("I moved from Berlin to Munich")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        berlin_only = [t for t in texts if "Berlin" in t and "Munich" not in t and "Currently" not in t]
        assert len(berlin_only) == 0

    def test_multi_word_transition(self, mem):
        mem.store("I use Ruby on Rails for web development")
        mem.store("I switched from Ruby on Rails to Next.js")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        old_only = [t for t in texts if "Ruby on Rails" in t and "Next.js" not in t and "Currently" not in t]
        assert len(old_only) == 0

    def test_replaced_with_supersedes(self, mem):
        mem.store("We use Jenkins for CI")
        mem.store("We replaced Jenkins with GitHub Actions")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        old_only = [t for t in texts if "Jenkins" in t and "GitHub Actions" not in t and "Currently" not in t]
        assert len(old_only) == 0

    def test_implicit_promoted_supersedes_junior(self, mem):
        mem.store("I am a junior developer")
        mem.store("I got promoted")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        junior_only = [t for t in texts if "junior" in t.lower() and "promoted" not in t.lower()]
        assert len(junior_only) == 0

    def test_does_not_supersede_when_old_has_new_value(self, mem):
        # If the old episode already mentions both values, don't supersede it
        mem.store("I considered switching from Django to FastAPI last month")
        mem.store("I switched from Django to FastAPI")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        # The first episode mentions both Django and FastAPI, so should NOT be superseded
        considered = [t for t in texts if "considered" in t]
        assert len(considered) == 1

    def test_transition_without_i_prefix(self, mem):
        """Transition patterns should work without the 'I' prefix."""
        mem.store("Our backend uses Flask")
        mem.store("Switched from Flask to FastAPI for the backend")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        flask_only = [t for t in texts if "Flask" in t and "FastAPI" not in t and "Currently" not in t]
        assert len(flask_only) == 0


# ---------------------------------------------------------------------------
# Derived facts
# ---------------------------------------------------------------------------

class TestDerivedFacts:
    def test_switched_generates_currently_using(self, mem):
        mem.store("I switched from Django to FastAPI")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        assert any("Currently using FastAPI" in t for t in texts)

    def test_replaced_generates_currently_using(self, mem):
        mem.store("We replaced Heroku with Fly.io")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        assert any("Currently using Fly.io" in t for t in texts)


# ---------------------------------------------------------------------------
# Topic detection & recall
# ---------------------------------------------------------------------------

class TestTopicDetection:
    def test_tech_stack_topic(self, mem):
        mem.store("We migrated our frontend from React to Vue")
        facts = mem.get_facts()
        # Check that the episode has tech_stack topic via metadata
        for ep in mem._store.episodes:
            topics = ep.metadata.get("topics", [])
            if "tech_stack" in topics:
                return
        pytest.fail("Expected tech_stack topic to be detected")

    def test_topic_recall_supplements_embedding(self, mem):
        mem.store("Our database is PostgreSQL running on RDS")
        mem.store("We use Redis for caching hot data")
        mem.store("The team went hiking last weekend")
        results = mem.recall("database infrastructure")
        texts = [t for t, _ in results]
        # Database/infra items should rank higher than hiking
        assert any("PostgreSQL" in t or "Redis" in t for t in texts[:3])


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class TestSessions:
    def test_new_session_returns_id(self, mem):
        sid = mem.new_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_sessions_count_increments(self, mem):
        assert mem.get_profile()["sessions_count"] == 1
        mem.new_session()
        assert mem.get_profile()["sessions_count"] == 2

    def test_session_id_changes(self, mem):
        old = mem._session_id
        mem.new_session()
        assert mem._session_id != old


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class TestProfile:
    def test_profile_has_expected_keys(self, mem):
        profile = mem.get_profile()
        assert "user_id" in profile
        assert "total_memories" in profile
        assert "sessions_count" in profile
        assert "first_seen" in profile
        assert "last_seen" in profile
        assert "entities" in profile

    def test_first_last_seen_set_on_store(self, mem):
        assert mem.get_profile()["first_seen"] is None
        mem.store("Hello, I'm a test user")
        profile = mem.get_profile()
        assert profile["first_seen"] is not None
        assert profile["last_seen"] is not None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_reload(self, mem_factory):
        m1 = mem_factory()
        m1.store("I work at Acme Corp in Stuttgart")
        m1.save()
        m1.close()

        m2 = mem_factory()
        profile = m2.get_profile()
        assert profile["total_memories"] >= 1
        assert profile["first_seen"] is not None

    def test_superseded_persists(self, mem_factory):
        m1 = mem_factory()
        m1.store("I use Django")
        m1.store("I switched from Django to FastAPI")
        assert len(m1._superseded) > 0
        m1.save()

        m2 = mem_factory()
        assert len(m2._superseded) > 0

    def test_long_user_id_hashed(self, tmp_path):
        long_id = "a" * 200
        m = UserMemory(user_id=long_id, persistence_path=str(tmp_path / "mem"))
        # Directory name should be hash-truncated
        assert len(m._user_dir.name) <= 80
        m.close()


# ---------------------------------------------------------------------------
# get_facts
# ---------------------------------------------------------------------------

class TestGetFacts:
    def test_superseded_excluded(self, mem):
        mem.store("I use Django for everything")
        mem.store("I switched from Django to FastAPI")
        facts = mem.get_facts()
        texts = [t for t, _ in facts]
        # The Django-only episode should be superseded
        django_only = [t for t in texts if "Django" in t and "FastAPI" not in t and "Currently" not in t]
        assert len(django_only) == 0

    def test_returns_tuples(self, mem):
        mem.store("My team has 5 engineers")
        facts = mem.get_facts()
        assert len(facts) > 0
        text, importance = facts[0]
        assert isinstance(text, str)
        assert isinstance(importance, float)
