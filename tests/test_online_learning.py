"""Tests for Online Continual Learning — replay buffer, EWC, adaptive thresholds."""
import numpy as np
import pytest
from elo_memory.online_learning import OnlineLearner, OnlineLearningConfig


@pytest.fixture
def learner():
    return OnlineLearner()


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_add_to_buffer(self, learner):
        learner.add_to_replay_buffer(np.zeros(16), surprise=1.0)
        assert len(learner.replay_buffer) == 1

    def test_buffer_respects_max_size(self):
        learner = OnlineLearner(OnlineLearningConfig(replay_buffer_size=5))
        for i in range(20):
            learner.add_to_replay_buffer(np.random.randn(16), surprise=float(i))
        assert len(learner.replay_buffer) == 5

    def test_high_priority_replaces_low(self):
        learner = OnlineLearner(OnlineLearningConfig(replay_buffer_size=3))
        for i in range(3):
            learner.add_to_replay_buffer(np.zeros(16), surprise=0.1)
        # All items have priority 0.1
        learner.add_to_replay_buffer(np.ones(16), surprise=10.0)
        priorities = [item["priority"] for item in learner.replay_buffer]
        assert 10.0 in priorities

    def test_low_priority_does_not_replace_high(self):
        learner = OnlineLearner(OnlineLearningConfig(replay_buffer_size=3))
        for i in range(3):
            learner.add_to_replay_buffer(np.zeros(16), surprise=10.0)
        learner.add_to_replay_buffer(np.ones(16), surprise=0.001)
        priorities = [item["priority"] for item in learner.replay_buffer]
        assert 0.001 not in priorities


# ---------------------------------------------------------------------------
# Replay sampling
# ---------------------------------------------------------------------------

class TestReplaySampling:
    def test_sample_empty_buffer(self, learner):
        batch = learner.sample_replay_batch()
        assert batch == []

    def test_sample_returns_correct_size(self, learner):
        for i in range(20):
            learner.add_to_replay_buffer(np.random.randn(16), surprise=float(i) + 0.1)
        batch = learner.sample_replay_batch(batch_size=5)
        assert len(batch) == 5

    def test_sample_does_not_exceed_buffer(self, learner):
        for i in range(3):
            learner.add_to_replay_buffer(np.random.randn(16), surprise=1.0)
        batch = learner.sample_replay_batch(batch_size=10)
        assert len(batch) == 3

    def test_sampled_items_have_expected_keys(self, learner):
        learner.add_to_replay_buffer(np.zeros(16), surprise=1.0, metadata={"tag": "x"})
        batch = learner.sample_replay_batch(batch_size=1)
        item = batch[0]
        assert "observation" in item
        assert "surprise" in item
        assert "priority" in item
        assert "metadata" in item


# ---------------------------------------------------------------------------
# Adaptive thresholds
# ---------------------------------------------------------------------------

class TestAdaptiveThresholds:
    def test_surprise_threshold_moves_toward_value(self, learner):
        initial = learner.surprise_threshold
        learner.update_adaptive_threshold(5.0, "surprise")
        assert learner.surprise_threshold > initial

    def test_novelty_threshold_moves_toward_value(self, learner):
        initial = learner.novelty_threshold
        learner.update_adaptive_threshold(0.1, "novelty")
        assert learner.novelty_threshold < initial

    def test_disabled_threshold_stays_constant(self):
        learner = OnlineLearner(OnlineLearningConfig(adaptive_threshold=False))
        initial = learner.surprise_threshold
        learner.update_adaptive_threshold(100.0, "surprise")
        assert learner.surprise_threshold == initial

    def test_ema_converges(self, learner):
        for _ in range(200):
            learner.update_adaptive_threshold(3.0, "surprise")
        assert learner.surprise_threshold == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# EWC loss
# ---------------------------------------------------------------------------

class TestEWCLoss:
    def test_zero_when_params_unchanged(self, learner):
        params = {"w": np.array([1.0, 2.0])}
        learner.fisher_information = {"w": np.array([1.0, 1.0])}
        loss = learner.compute_ewc_loss(params, params)
        assert loss == pytest.approx(0.0)

    def test_positive_when_params_change(self, learner):
        old = {"w": np.array([1.0, 2.0])}
        new = {"w": np.array([2.0, 3.0])}
        learner.fisher_information = {"w": np.array([1.0, 1.0])}
        loss = learner.compute_ewc_loss(new, old)
        assert loss > 0

    def test_higher_fisher_higher_penalty(self, learner):
        old = {"w": np.array([0.0])}
        new = {"w": np.array([1.0])}
        learner.fisher_information = {"w": np.array([1.0])}
        loss_low = learner.compute_ewc_loss(new, old)

        learner.fisher_information = {"w": np.array([10.0])}
        loss_high = learner.compute_ewc_loss(new, old)
        assert loss_high > loss_low

    def test_missing_fisher_ignored(self, learner):
        old = {"w": np.array([0.0])}
        new = {"w": np.array([1.0])}
        # No fisher info set
        loss = learner.compute_ewc_loss(new, old)
        assert loss == 0.0


# ---------------------------------------------------------------------------
# Fisher information update
# ---------------------------------------------------------------------------

class TestFisherUpdate:
    def test_creates_entry(self, learner):
        grad = np.array([1.0, 2.0])
        learner.update_fisher_information("w", grad)
        assert "w" in learner.fisher_information

    def test_running_average(self, learner):
        grad1 = np.array([1.0, 0.0])
        grad2 = np.array([0.0, 1.0])
        learner.update_fisher_information("w", grad1)
        learner.update_fisher_information("w", grad2)
        fi = learner.fisher_information["w"]
        # Should be a blend, not just the last gradient squared
        assert fi[0] > 0 and fi[1] > 0


# ---------------------------------------------------------------------------
# online_update integration
# ---------------------------------------------------------------------------

class TestOnlineUpdate:
    def test_increments_total_updates(self, learner):
        learner.online_update(np.random.randn(16), surprise=1.0)
        assert learner.total_updates == 1

    def test_adds_to_replay_buffer(self, learner):
        learner.online_update(np.random.randn(16), surprise=1.0)
        assert len(learner.replay_buffer) == 1

    def test_update_fn_called(self, learner):
        calls = {"n": 0}

        def fn(obs):
            calls["n"] += 1

        # Fill buffer to trigger replay
        for i in range(40):
            learner.online_update(np.random.randn(16), surprise=1.0, update_fn=fn)
        assert calls["n"] > 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_statistics_structure(self, learner):
        stats = learner.get_statistics()
        assert "total_updates" in stats
        assert "replay_count" in stats
        assert "replay_buffer_size" in stats
        assert "surprise_threshold" in stats
        assert "novelty_threshold" in stats
        assert "fisher_params" in stats

    def test_statistics_reflect_state(self, learner):
        for _ in range(5):
            learner.online_update(np.random.randn(16), surprise=1.0)
        stats = learner.get_statistics()
        assert stats["total_updates"] == 5
        assert stats["replay_buffer_size"] == 5
