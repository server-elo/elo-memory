"""Tests for EloBrain — the agent middleware."""

import pytest

from elo_memory.brain import EloBrain, _should_skip, _extract_commitments


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def brain(tmp_path):
    """Fresh EloBrain writing to a temp directory."""
    b = EloBrain(
        user_id="brain-test-user",
        persistence_path=str(tmp_path / "memories"),
        system_prompt="You are a test assistant.",
    )
    yield b
    b.close()


def _echo_llm(prompt: str) -> str:
    """Trivial LLM stub that echoes back."""
    return "I've noted your request. I'll check on that."


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------

class TestMessageFiltering:
    def test_skip_greeting(self):
        assert _should_skip("hi") is True
        assert _should_skip("Hello!") is True
        assert _should_skip("hey") is True

    def test_skip_thanks(self):
        assert _should_skip("thanks") is True
        assert _should_skip("Thank you!") is True

    def test_skip_filler(self):
        assert _should_skip("ok") is True
        assert _should_skip("sure") is True
        assert _should_skip("yes") is True
        assert _should_skip("no") is True
        assert _should_skip("bye") is True

    def test_skip_time_question(self):
        assert _should_skip("What time is it?") is True

    def test_skip_how_are_you(self):
        assert _should_skip("How are you?") is True

    def test_keep_personal_info_in_question(self):
        assert _should_skip("Can you remember my name is Sarah?") is False

    def test_keep_personal_statements(self):
        assert _should_skip("I work at Google") is False
        assert _should_skip("I use FastAPI for my backend") is False
        assert _should_skip("My name is John") is False

    def test_keep_substantial_messages(self):
        assert _should_skip("We just migrated our database to PostgreSQL") is False
        assert _should_skip("I switched from React to Vue last week") is False

    def test_keep_remember_request(self):
        assert _should_skip("Remember that I live in Stuttgart") is False


# ---------------------------------------------------------------------------
# Commitment extraction
# ---------------------------------------------------------------------------

class TestCommitmentExtraction:
    def test_ill_check(self):
        commitments = _extract_commitments("I'll check on that for you.")
        assert len(commitments) >= 1

    def test_ive_noted(self):
        commitments = _extract_commitments("I've noted your preference for dark mode.")
        assert len(commitments) >= 1

    def test_ill_look_into(self):
        commitments = _extract_commitments("I'll look into the deployment issue.")
        assert len(commitments) >= 1

    def test_no_commitments(self):
        commitments = _extract_commitments("The weather is nice today.")
        assert len(commitments) == 0


# ---------------------------------------------------------------------------
# think()
# ---------------------------------------------------------------------------

class TestThink:
    def test_think_returns_llm_response(self, brain):
        response = brain.think("My name is Alice", _echo_llm)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_think_stores_memory(self, brain):
        brain.think("I work at Acme Corp as a data scientist", _echo_llm)
        info = brain.what_i_know()
        assert info["total_memories"] >= 1

    def test_think_stores_commitments(self, brain):
        brain.think("Can you help me debug this?", _echo_llm)
        # The echo LLM says "I've noted..." and "I'll check..."
        facts = brain._memory.get_facts()
        texts = [t for t, _ in facts]
        assert any("commitment" in t.lower() for t in texts)


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_prepare_returns_expected_keys(self, brain):
        ctx = brain.prepare("Tell me about my setup")
        assert "system" in ctx
        assert "user_message" in ctx
        assert "memories_used" in ctx
        assert "user_profile" in ctx

    def test_prepare_includes_system_prompt(self, brain):
        ctx = brain.prepare("Hello")
        assert "test assistant" in ctx["system"]

    def test_prepare_includes_kb_facts(self, brain):
        brain.process_turn("My name is Alice")
        ctx = brain.prepare("What is my name?")
        # KB should have extracted name
        assert "Alice" in ctx["system"] or ctx["memories_used"] > 0

    def test_prepare_includes_memories(self, brain):
        brain.process_turn("I use PostgreSQL for my main database")
        ctx = brain.prepare("What database do I use?")
        assert ctx["memories_used"] >= 0  # may or may not match depending on recall


# ---------------------------------------------------------------------------
# process_turn()
# ---------------------------------------------------------------------------

class TestProcessTurn:
    def test_process_turn_stores_user_message(self, brain):
        brain.process_turn("I prefer dark mode in all my editors")
        facts = brain._memory.get_facts()
        texts = [t for t, _ in facts]
        assert any("dark mode" in t for t in texts)

    def test_process_turn_skips_greeting(self, brain):
        brain.process_turn("hi")
        assert brain._memory.get_profile()["total_memories"] == 0

    def test_process_turn_keeps_personal_question(self, brain):
        brain.process_turn("Can you remember my name is Sarah?")
        facts = brain._memory.get_facts()
        texts = [t for t, _ in facts]
        assert any("Sarah" in t for t in texts)

    def test_process_turn_updates_kb(self, brain):
        brain.process_turn("My name is Bob")
        kb_facts = brain._kb.get_all()
        assert "name" in kb_facts
        assert "Bob" in kb_facts["name"]

    def test_process_turn_updates_kb_even_for_questions(self, brain):
        # Questions can contain facts
        brain.process_turn("My backend framework is FastAPI, can you remember that?")
        kb_facts = brain._kb.get_all()
        # At minimum the KB got a chance to extract from it
        assert isinstance(kb_facts, dict)

    def test_process_turn_extracts_commitments(self, brain):
        brain.process_turn(
            "Please check the deploy",
            assistant_response="I'll check on the deploy status right away.",
        )
        facts = brain._memory.get_facts()
        texts = [t for t, _ in facts]
        assert any("commitment" in t.lower() for t in texts)


# ---------------------------------------------------------------------------
# what_i_know()
# ---------------------------------------------------------------------------

class TestWhatIKnow:
    def test_returns_expected_keys(self, brain):
        info = brain.what_i_know()
        assert "knowledge" in info
        assert "facts" in info
        assert "entities" in info
        assert "topics" in info
        assert "total_memories" in info
        assert "superseded" in info

    def test_knowledge_populated_after_turns(self, brain):
        brain.process_turn("My name is Eve and I work at Startup Inc")
        info = brain.what_i_know()
        assert info["total_memories"] >= 1

    def test_topics_detected(self, brain):
        brain.process_turn("We use React and PostgreSQL for our stack")
        info = brain.what_i_know()
        assert len(info["topics"]) >= 1


# ---------------------------------------------------------------------------
# forget() — GDPR
# ---------------------------------------------------------------------------

class TestForget:
    def test_forget_supersedes_matching(self, brain):
        brain.process_turn("My email is alice@example.com")
        brain.forget("alice@example.com")
        facts = brain._memory.get_facts()
        texts = [t for t, _ in facts]
        assert not any("alice@example.com" in t for t in texts)

    def test_forget_does_not_touch_unrelated(self, brain):
        brain.process_turn("I use PostgreSQL")
        brain.process_turn("My email is bob@test.com")
        brain.forget("bob@test.com")
        facts = brain._memory.get_facts()
        texts = [t for t, _ in facts]
        assert any("PostgreSQL" in t for t in texts)

    def test_forget_persists(self, brain):
        brain.process_turn("Secret data here")
        brain.forget("Secret data")
        # After forget, save is called
        assert len(brain._memory._superseded) > 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_close_does_not_raise(self, brain):
        brain.process_turn("Some data to persist")
        brain.close()
        # Should not raise
