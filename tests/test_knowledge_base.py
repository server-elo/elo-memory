"""Tests for KnowledgeBase — the key feature that took recall from 54% to 100%."""

import os
import json
import tempfile

import pytest

from elo_memory.memory.knowledge_base import KnowledgeBase


# ── Basic extraction ─────────────────────────────────────────────────


class TestBasicExtraction:
    def test_my_x_is_y(self):
        kb = KnowledgeBase()
        changes = kb.update("My name is Alice")
        assert "name" in changes or "name" in kb.get_all()
        assert kb.get_all()["name"] == "Alice"

    def test_our_x_is_y(self):
        kb = KnowledgeBase()
        kb.update("Our backend is Django")
        assert kb.get_all()["backend"] == "Django"

    def test_the_x_is_y(self):
        kb = KnowledgeBase()
        kb.update("The database is PostgreSQL")
        assert kb.get_all()["database"] == "PostgreSQL"

    def test_we_use_x_for_y(self):
        kb = KnowledgeBase()
        kb.update("We use Stripe for payment")
        assert kb.get_all()["payment"] == "Stripe"

    def test_identity_email(self):
        kb = KnowledgeBase()
        kb.update("Contact me at alice@example.com")
        assert kb.get_all()["email"] == "alice@example.com"

    def test_multiple_sentences(self):
        kb = KnowledgeBase()
        kb.update("My name is Bob. My role is engineer.")
        facts = kb.get_all()
        assert facts["name"] == "Bob"
        assert facts["role"] == "engineer"


# ── Transitions ──────────────────────────────────────────────────────


class TestTransitions:
    def test_switched_backend(self):
        kb = KnowledgeBase()
        kb.update("Our backend is Django")
        assert kb.get_all()["backend"] == "Django"

        kb.update("We switched from Django to FastAPI for backend")
        facts = kb.get_all()
        assert facts["backend"] == "FastAPI"

    def test_migrated_database(self):
        kb = KnowledgeBase()
        kb.update("Our database is MySQL")
        kb.update("Migrated from MySQL to PostgreSQL for database")
        assert kb.get_all()["database"] == "PostgreSQL"

    def test_transition_stores_old_value(self):
        kb = KnowledgeBase()
        kb.update("Our backend is Rails")
        kb.update("Switched from Rails to Django for backend")
        assert kb._facts.get("_old_backend") == "Rails"

    def test_transition_infers_key_from_existing(self):
        """When no category word in context, find key by matching old value."""
        kb = KnowledgeBase()
        kb.update("Our framework is Express")
        kb.update("Switched from Express to Fastify")
        assert kb.get_all()["framework"] == "Fastify"


# ── Comma-list parsing ───────────────────────────────────────────────


class TestCommaList:
    def test_comma_separated_tech(self):
        kb = KnowledgeBase()
        kb.update("Django backend, React frontend, PostgreSQL database")
        facts = kb.get_all()
        assert facts["backend"] == "Django"
        assert facts["frontend"] == "React"
        assert facts["database"] == "PostgreSQL"

    def test_with_and_separator(self):
        kb = KnowledgeBase()
        kb.update("Flask backend and Vue frontend")
        facts = kb.get_all()
        assert facts["backend"] == "Flask"
        assert facts["frontend"] == "Vue"


# ── Team operations ──────────────────────────────────────────────────


class TestTeam:
    def test_hired(self):
        kb = KnowledgeBase()
        kb.update("Hired Alice for engineering")
        assert kb.get_all()["team:alice"] == "engineering"

    def test_promoted(self):
        kb = KnowledgeBase()
        kb.update("Hired Bob as junior developer")
        kb.update("Promoted Bob to senior developer")
        assert kb.get_all()["team:bob"] == "senior developer"

    def test_team_size(self):
        kb = KnowledgeBase()
        kb.update("We have a team of 12")
        assert kb.get_all()["team size"] == "12"


# ── Money extraction ─────────────────────────────────────────────────


class TestMoney:
    def test_raised(self):
        kb = KnowledgeBase()
        kb.update("We raised $5M in Series A")
        assert "funding raised" in kb.get_all()
        assert "$5m" in kb.get_all()["funding raised"]

    def test_valuation(self):
        kb = KnowledgeBase()
        kb.update("$50M valuation")
        assert kb.get_all()["valuation"] == "$50m"

    def test_revenue(self):
        kb = KnowledgeBase()
        kb.update("$2M ARR")
        assert "revenue" in kb.get_all()


# ── Compliance ────────────────────────────────────────────────────────


class TestCompliance:
    def test_single_flag(self):
        kb = KnowledgeBase()
        kb.update("We are SOC2 certified")
        assert "SOC2" in kb.get_all()["compliance"]

    def test_multiple_flags(self):
        kb = KnowledgeBase()
        kb.update("SOC2 compliant")
        kb.update("HIPAA compliant")
        compliance = kb.get_all()["compliance"]
        assert "SOC2" in compliance
        assert "HIPAA" in compliance

    def test_gdpr(self):
        kb = KnowledgeBase()
        kb.update("GDPR compliant since 2023")
        assert "GDPR" in kb.get_all()["compliance"]


# ── Query ─────────────────────────────────────────────────────────────


class TestQuery:
    def test_direct_key_match(self):
        kb = KnowledgeBase()
        kb.update("Our backend is Django")
        result = kb.query("backend")
        assert result is not None
        assert "Django" in result

    def test_keyword_search(self):
        kb = KnowledgeBase()
        kb.update("Our backend is Django")
        result = kb.query("what is the backend")
        assert result is not None
        assert "Django" in result

    def test_tech_stack_query(self):
        kb = KnowledgeBase()
        kb.update("Django backend, React frontend, PostgreSQL database")
        result = kb.query("tech stack")
        assert result is not None
        assert "Django" in result
        assert "React" in result

    def test_who_is_query(self):
        kb = KnowledgeBase()
        kb.update("Hired Alice for engineering")
        result = kb.query("who is Alice")
        assert result is not None
        assert "engineering" in result

    def test_compliance_query(self):
        kb = KnowledgeBase()
        kb.update("SOC2 and HIPAA compliant")
        result = kb.query("compliance")
        assert result is not None
        assert "SOC2" in result

    def test_no_match_returns_none(self):
        kb = KnowledgeBase()
        result = kb.query("what is quantum gravity")
        assert result is None


# ── Persistence ───────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb1 = KnowledgeBase(persistence_path=tmpdir)
            kb1.update("Our backend is Django. Our frontend is React.")

            # New instance loads from disk
            kb2 = KnowledgeBase(persistence_path=tmpdir)
            facts = kb2.get_all()
            assert facts["backend"] == "Django"
            assert facts["frontend"] == "React"

    def test_persistence_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(persistence_path=tmpdir)
            kb.update("My name is Alice")
            assert os.path.exists(os.path.join(tmpdir, "knowledge_base.json"))

    def test_persistence_json_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(persistence_path=tmpdir)
            kb.update("My name is Alice")
            with open(os.path.join(tmpdir, "knowledge_base.json")) as f:
                data = json.load(f)
            assert "facts" in data
            assert "history" in data

    def test_history_recorded(self):
        kb = KnowledgeBase()
        kb.update("Our backend is Django")
        kb.update("Switched from Django to FastAPI for backend")
        # Should have 2 history entries (initial + transition)
        assert len(kb._history) >= 2


# ── Summary ──────────────────────────────────────────────────────────


class TestSummary:
    def test_empty_summary(self):
        kb = KnowledgeBase()
        assert "empty" in kb.get_summary().lower()

    def test_summary_contains_facts(self):
        kb = KnowledgeBase()
        kb.update("Our backend is Django")
        summary = kb.get_summary()
        assert "Django" in summary
        assert "backend" in summary
