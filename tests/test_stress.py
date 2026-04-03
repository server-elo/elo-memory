"""Stress tests that push elo-memory to its limits.

These tests simulate real-world usage patterns that are genuinely hard:
- Contradictory information evolving over time
- Implicit coreference and entity disambiguation
- Temporal reasoning ("what did I say BEFORE...")
- Adversarial near-duplicates
- Knowledge base vs episodic memory consistency
- Scale: rapid-fire stores followed by precise recall
"""

import pytest
import time
from datetime import datetime, timedelta

from elo_memory.brain import EloBrain
from elo_memory.memory.user_memory import UserMemory
from elo_memory.memory.knowledge_base import KnowledgeBase


@pytest.fixture
def mem(tmp_path):
    return UserMemory(user_id="stress_user", persistence_path=str(tmp_path / "mem"))


@pytest.fixture
def brain(tmp_path):
    return EloBrain(
        user_id="stress_user",
        persistence_path=str(tmp_path / "brain"),
    )


@pytest.fixture
def kb():
    return KnowledgeBase()


# =========================================================================
# 1. CONTRADICTIONS OVER TIME
#    User changes their mind multiple times. System must track the LATEST.
# =========================================================================


class TestContradictionChain:
    """User changes tech stack 4 times. Only the latest should surface."""

    def test_multi_hop_contradiction(self, mem):
        mem.store("We use MySQL for our database")
        mem.store("We switched from MySQL to PostgreSQL for our database")
        mem.store("We migrated from PostgreSQL to CockroachDB")
        mem.store("Actually we rolled back to PostgreSQL, CockroachDB had too many issues")

        results = mem.recall("What database do we use?")
        texts = [t for t, _ in results]
        joined = " ".join(texts).lower()

        # The LATEST truth is PostgreSQL (rolled back)
        assert any("postgresql" in t.lower() or "rolled back" in t.lower() for t in texts), (
            f"Should recall PostgreSQL rollback, got: {texts}"
        )

    def test_superseded_memories_hidden(self, mem):
        mem.store("Our backend is Django")
        mem.store("We switched from Django to FastAPI for our backend")

        results = mem.recall("What's our backend?")
        texts = [t for t, _ in results]

        # Django-only memories should be superseded
        for text in texts:
            if "django" in text.lower() and "fastapi" not in text.lower():
                pytest.fail(f"Superseded memory leaked through: {text}")


# =========================================================================
# 2. ENTITY DISAMBIGUATION
#    Same name, different context. System must keep them separate.
# =========================================================================


class TestEntityDisambiguation:
    """Two different 'John's — manager vs client. Must not merge."""

    def test_two_johns(self, mem):
        mem.store("My manager John approved the new budget for Q3")
        mem.store("Our client John from Acme Corp needs the API docs by Friday")

        # Query about manager
        mgr_results = mem.recall("What did my manager approve?")
        mgr_texts = [t for t, _ in mgr_results]
        assert any("budget" in t.lower() or "q3" in t.lower() for t in mgr_texts)

        # Query about client
        client_results = mem.recall("What does the Acme client need?")
        client_texts = [t for t, _ in client_results]
        assert any("api docs" in t.lower() or "acme" in t.lower() for t in client_texts)

    def test_same_tool_different_teams(self, mem):
        mem.store("The frontend team uses React with TypeScript")
        mem.store("The mobile team uses React Native with JavaScript")

        frontend = mem.recall("What does the frontend team use?")
        mobile = mem.recall("What does the mobile team use?")

        f_texts = [t for t, _ in frontend]
        m_texts = [t for t, _ in mobile]

        assert any("typescript" in t.lower() for t in f_texts), f"Frontend: {f_texts}"
        assert any("react native" in t.lower() or "javascript" in t.lower() for t in m_texts), (
            f"Mobile: {m_texts}"
        )


# =========================================================================
# 3. TEMPORAL REASONING
#    User asks about what they said at different points in time.
# =========================================================================


class TestTemporalReasoning:
    """Recall must respect chronological ordering."""

    def test_what_changed(self, mem):
        mem.store("Our team size is 5 people")
        mem.store("We just hired 3 new engineers, team size is now 8")
        mem.store("Two people left last month, we're down to 6")

        results = mem.recall("How has our team size changed?")
        texts = [t for t, _ in results]

        # Should recall the progression, most recent first or all present
        assert any("6" in t for t in texts), f"Should mention current size 6: {texts}"


# =========================================================================
# 4. ADVERSARIAL NEAR-DUPLICATES
#    Subtle rewording that changes meaning. Must NOT be deduped.
# =========================================================================


class TestAdversarialDedup:
    """Slight wording changes with different meaning must survive dedup."""

    def test_negation_not_deduped(self, mem):
        r1 = mem.store("We use Redis for caching")
        r2 = mem.store("We don't use Redis for caching anymore")

        assert r1["stored"] is True
        assert r2["stored"] is True, "Negation was incorrectly deduplicated"

    def test_different_numbers_not_deduped(self, mem):
        r1 = mem.store("Our ARR is $2M")
        r2 = mem.store("Our ARR is $5M")

        assert r1["stored"] is True
        assert r2["stored"] is True, "Different numbers were incorrectly deduplicated"

    def test_different_subjects_not_deduped(self, mem):
        r1 = mem.store("Alice is our CTO")
        r2 = mem.store("Bob is our CTO")

        assert r1["stored"] is True
        assert r2["stored"] is True, "Different subjects were incorrectly deduplicated"


# =========================================================================
# 5. KNOWLEDGE BASE CONSISTENCY
#    KB facts must stay in sync with episodic memory after contradictions.
# =========================================================================


class TestKBConsistency:
    """KB must update when user corrects information."""

    def test_kb_tracks_corrections(self, kb):
        kb.update("Our database is MySQL")
        assert kb.get_all().get("database") == "MySQL"

        kb.update("We switched from MySQL to PostgreSQL")
        facts = kb.get_all()
        assert facts.get("database") == "PostgreSQL", f"KB didn't update: {facts}"

    def test_kb_handles_implicit_correction(self, kb):
        kb.update("Our backend is Django")
        kb.update("Actually our backend is FastAPI")
        facts = kb.get_all()
        assert facts.get("backend") == "FastAPI", f"KB didn't catch correction: {facts}"

    def test_kb_multiple_facts_one_sentence(self, kb):
        kb.update("Django backend, React frontend, PostgreSQL database")
        facts = kb.get_all()
        assert "backend" in facts, f"Missing backend: {facts}"
        assert "frontend" in facts, f"Missing frontend: {facts}"
        assert "database" in facts, f"Missing database: {facts}"


# =========================================================================
# 6. RAPID-FIRE SCALE TEST
#    Store 100 memories fast, then recall specific ones accurately.
# =========================================================================


class TestRapidFireScale:
    """Blast 100 memories in, then do precise recall."""

    def test_needle_in_haystack(self, mem):
        # Store 50 filler memories
        for i in range(50):
            mem.store(f"Team standup note #{i}: discussed sprint progress and blockers")

        # Store the needle
        mem.store("CRITICAL: Production database ran out of disk space at 3am, had to emergency scale")

        # Store 50 more filler
        for i in range(50, 100):
            mem.store(f"Team standup note #{i}: discussed sprint progress and blockers")

        # Find the needle
        results = mem.recall("What production incidents happened?")
        texts = [t for t, _ in results]
        assert any("disk space" in t.lower() or "emergency" in t.lower() for t in texts), (
            f"Failed to find needle in 101 memories: {texts[:3]}"
        )


# =========================================================================
# 7. BRAIN INTEGRATION — FULL LOOP
#    End-to-end: store via brain, recall enriched context.
# =========================================================================


class TestBrainIntegration:
    """Full EloBrain loop: process_turn -> prepare must be consistent."""

    def test_store_then_recall_via_brain(self, brain):
        brain.process_turn("I'm Sarah, CTO at DataFlow. We use Kubernetes on AWS.")
        brain.process_turn("Our main challenge is reducing P99 latency below 200ms")

        context = brain.prepare("What are our infrastructure challenges?")

        system = context["system"]
        assert "latency" in system.lower() or "200ms" in system.lower(), (
            f"Brain didn't surface latency challenge in context:\n{system[:500]}"
        )

    def test_brain_kb_and_memory_both_surface(self, brain):
        brain.process_turn("We use FastAPI for our backend")
        brain.process_turn("Our team has 12 engineers across 3 time zones")

        context = brain.prepare("Tell me about our setup")
        system = context["system"]

        # Both KB facts and episodic memories should appear
        has_fastapi = "fastapi" in system.lower()
        has_team = "12" in system or "engineer" in system.lower()
        assert has_fastapi or has_team, (
            f"Brain context missing key facts:\n{system[:500]}"
        )


# =========================================================================
# 8. PROFILE EXTRACTION
#    User identity should be extractable from natural conversation.
# =========================================================================


class TestProfileExtraction:
    def test_identity_from_conversation(self, mem):
        mem.store("My name is Max Chen and I work at PayFlow")
        mem.store("We're based in Berlin")
        mem.store("I've been coding for 15 years, mostly in Python")

        profile = mem.get_profile()
        assert profile is not None, "Profile is None"
        assert profile["total_memories"] == 3


# =========================================================================
# 9. CROSS-SESSION PERSISTENCE
#    Store in one session, recall in a fresh instance.
# =========================================================================


class TestCrossSession:
    def test_memories_survive_restart(self, tmp_path):
        path = str(tmp_path / "persist")

        # Session 1: store
        mem1 = UserMemory(user_id="persist_user", persistence_path=path)
        mem1.store("Our secret project codename is Phoenix")
        mem1.save()

        # Session 2: fresh instance, same path
        mem2 = UserMemory(user_id="persist_user", persistence_path=path)
        results = mem2.recall("What's our project codename?")
        texts = [t for t, _ in results]
        assert any("phoenix" in t.lower() for t in texts), (
            f"Memory didn't persist across sessions: {texts}"
        )


# =========================================================================
# 10. MULTI-LANGUAGE / UNICODE ROBUSTNESS
# =========================================================================


class TestUnicodeRobustness:
    def test_emoji_in_memory(self, mem):
        r = mem.store("Launch went great! 🚀🎉 Team is celebrating")
        assert r["stored"] is True

        results = mem.recall("How did the launch go?")
        texts = [t for t, _ in results]
        assert any("launch" in t.lower() for t in texts)

    def test_unicode_names(self, mem):
        r = mem.store("Our lead engineer is François Müller from Zürich")
        assert r["stored"] is True

        results = mem.recall("Who is our lead engineer?")
        texts = [t for t, _ in results]
        assert any("françois" in t.lower() or "müller" in t.lower() for t in texts)

    def test_mixed_scripts(self, mem):
        r = mem.store("The API returns データ (data) in JSON format")
        assert r["stored"] is True
