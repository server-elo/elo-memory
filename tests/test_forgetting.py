"""Tests for Memory Forgetting Engine — decay curves and rehearsal."""
import numpy as np
import pytest
from datetime import datetime, timedelta
from elo_memory.memory.forgetting import ForgettingEngine, ForgettingConfig


@pytest.fixture
def engine():
    return ForgettingEngine()


# ---------------------------------------------------------------------------
# Activation decay
# ---------------------------------------------------------------------------

class TestActivationDecay:
    def test_activation_decays_over_time(self, engine):
        ts = datetime.now() - timedelta(hours=24)
        a_0h = engine.compute_activation(1.0, ts, current_time=ts)
        a_24h = engine.compute_activation(1.0, ts, current_time=ts + timedelta(hours=24))
        assert a_24h < a_0h

    def test_old_memories_decay_below_minimum(self, engine):
        ts = datetime.now() - timedelta(days=365)
        a = engine.compute_activation(0.5, ts)
        assert a < engine.config.min_activation

    def test_activation_at_encoding_equals_initial(self, engine):
        ts = datetime.now()
        a = engine.compute_activation(2.0, ts, current_time=ts)
        assert a == pytest.approx(2.0)

    def test_power_law_vs_exponential(self):
        ts = datetime.now() - timedelta(hours=48)
        now = datetime.now()
        power = ForgettingEngine(ForgettingConfig(use_power_law=True))
        expo = ForgettingEngine(ForgettingConfig(use_power_law=False))
        a_power = power.compute_activation(1.0, ts, current_time=now)
        a_expo = expo.compute_activation(1.0, ts, current_time=now)
        # Both should have decayed, but to different amounts
        assert a_power < 1.0
        assert a_expo < 1.0
        assert a_power != pytest.approx(a_expo, abs=0.01)


# ---------------------------------------------------------------------------
# Rehearsal
# ---------------------------------------------------------------------------

class TestRehearsal:
    def test_rehearsal_boosts_activation(self, engine):
        ts = datetime.now() - timedelta(hours=12)
        a_0 = engine.compute_activation(1.0, ts, rehearsal_count=0)
        a_3 = engine.compute_activation(1.0, ts, rehearsal_count=3)
        assert a_3 > a_0

    def test_more_rehearsals_stronger_boost(self, engine):
        ts = datetime.now() - timedelta(hours=12)
        activations = [
            engine.compute_activation(1.0, ts, rehearsal_count=r) for r in range(5)
        ]
        assert activations == sorted(activations)


# ---------------------------------------------------------------------------
# Forgetting decision
# ---------------------------------------------------------------------------

class TestForgettingDecision:
    def test_should_forget_low_activation(self, engine):
        assert engine.should_forget(0.01) is True

    def test_should_not_forget_high_activation(self, engine):
        assert engine.should_forget(1.0) is False

    def test_boundary_at_min_activation(self, engine):
        # Exactly at threshold should NOT be forgotten (< not <=)
        assert engine.should_forget(engine.config.min_activation) is False


# ---------------------------------------------------------------------------
# Forgetting probability
# ---------------------------------------------------------------------------

class TestForgettingProbability:
    def test_high_activation_low_probability(self, engine):
        p = engine.get_forgetting_probability(2.0)
        assert p < 0.1

    def test_low_activation_high_probability(self, engine):
        p = engine.get_forgetting_probability(0.01)
        assert p > 0.5

    def test_probability_in_range(self, engine):
        for a in [0.001, 0.1, 0.5, 1.0, 5.0]:
            p = engine.get_forgetting_probability(a)
            assert 0.0 <= p <= 1.0
