"""Tests for Bayesian Surprise Engine."""
import numpy as np
import pytest
from elo_memory.surprise.bayesian_surprise import BayesianSurpriseEngine, SurpriseConfig


@pytest.fixture
def engine():
    return BayesianSurpriseEngine(input_dim=16, config=SurpriseConfig(window_size=5))


# ---------------------------------------------------------------------------
# Basic initialization and stepping
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Basic initialization and stepping
# ---------------------------------------------------------------------------

class TestInit:
    def test_engine_init(self):
        engine = BayesianSurpriseEngine(input_dim=16, config=SurpriseConfig(window_size=5))
        assert engine is not None
        assert engine.step_count == 0
        assert engine.input_dim == 16

    def test_step_count_increments(self, engine):
        engine.compute_surprise(np.zeros(16, dtype=np.float32))
        assert engine.step_count == 1
        engine.compute_surprise(np.zeros(16, dtype=np.float32))
        assert engine.step_count == 2


# ---------------------------------------------------------------------------
# compute_surprise
# ---------------------------------------------------------------------------

class TestComputeSurprise:
    def test_returns_expected_keys(self, engine):
        result = engine.compute_surprise(np.random.randn(16).astype(np.float32))
        assert "surprise" in result
        assert "is_novel" in result
        assert "normalized_surprise" in result
        assert "threshold" in result

    def test_surprise_is_non_negative(self, engine):
        for _ in range(20):
            r = engine.compute_surprise(np.random.randn(16).astype(np.float32))
            assert r["surprise"] >= 0.0

    def test_surprise_increases_for_outliers(self):
        engine = BayesianSurpriseEngine(input_dim=16, config=SurpriseConfig(window_size=5))
        # Feed normal data
        for _ in range(10):
            engine.compute_surprise(np.random.randn(16).astype(np.float32) * 0.1)
        # Feed outlier
        outlier = np.ones(16, dtype=np.float32) * 10.0
        result = engine.compute_surprise(outlier)
        assert result["surprise"] > 0

    def test_constant_input_lower_than_outlier(self):
        engine = BayesianSurpriseEngine(input_dim=16, config=SurpriseConfig(window_size=50))
        constant = np.ones(16, dtype=np.float32)
        for _ in range(30):
            r_const = engine.compute_surprise(constant)
        # An outlier should have higher surprise than the constant input
        outlier = np.ones(16, dtype=np.float32) * 100.0
        r_outlier = engine.compute_surprise(outlier)
        assert r_outlier["surprise"] > r_const["surprise"]


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_identical_distributions_zero_kl(self, engine):
        mean = np.random.randn(16)
        var = np.abs(np.random.randn(16)) + 0.1
        kl = engine.calculate_kl_divergence(mean, var, mean, var)
        assert kl == pytest.approx(0.0, abs=1e-5)

    def test_different_means_positive_kl(self, engine):
        var = np.ones(16)
        kl = engine.calculate_kl_divergence(np.zeros(16), var, np.ones(16) * 5, var)
        assert kl > 0

    def test_kl_non_negative(self, engine):
        for _ in range(10):
            pm, pv = np.random.randn(16), np.abs(np.random.randn(16)) + 0.1
            qm, qv = np.random.randn(16), np.abs(np.random.randn(16)) + 0.1
            kl = engine.calculate_kl_divergence(pm, pv, qm, qv)
            assert kl >= 0.0

    def test_symmetric_kl_is_symmetric(self):
        engine = BayesianSurpriseEngine(input_dim=8, config=SurpriseConfig(kl_method="symmetric"))
        pm, pv = np.random.randn(8), np.abs(np.random.randn(8)) + 0.1
        qm, qv = np.random.randn(8), np.abs(np.random.randn(8)) + 0.1
        kl_pq = engine.calculate_kl_divergence(pm, pv, qm, qv)
        kl_qp = engine.calculate_kl_divergence(qm, qv, pm, pv)
        assert kl_pq == pytest.approx(kl_qp, abs=1e-5)

    def test_forward_kl_not_symmetric(self):
        engine = BayesianSurpriseEngine(input_dim=8, config=SurpriseConfig(kl_method="forward"))
        pm, pv = np.array([0.0] * 8), np.array([1.0] * 8)
        qm, qv = np.array([2.0] * 8), np.array([0.5] * 8)
        kl_pq = engine.calculate_kl_divergence(pm, pv, qm, qv)
        kl_qp = engine.calculate_kl_divergence(qm, qv, pm, pv)
        assert kl_pq != pytest.approx(kl_qp, abs=0.01)


# ---------------------------------------------------------------------------
# Prior / posterior
# ---------------------------------------------------------------------------

class TestPriorPosterior:
    def test_uninformative_prior_with_few_observations(self, engine):
        mean, var = engine.get_prior_distribution()
        np.testing.assert_array_equal(mean, np.zeros(16))
        np.testing.assert_array_equal(var, np.ones(16))

    def test_prior_updates_after_observations(self):
        eng = BayesianSurpriseEngine(input_dim=16, config=SurpriseConfig(window_size=50, min_observations=5))
        for _ in range(20):
            eng.update_prior(np.ones(16) * 3.0)
        mean, var = eng.get_prior_distribution()
        np.testing.assert_array_almost_equal(mean, np.ones(16) * 3.0, decimal=1)

    def test_posterior_shifts_toward_observation(self, engine):
        prior_mean = np.zeros(16)
        prior_var = np.ones(16)
        obs = np.ones(16) * 5.0
        post_mean, _ = engine.get_posterior_distribution(obs, prior_mean, prior_var)
        # Posterior mean should be between prior and observation
        assert np.all(post_mean > prior_mean)
        assert np.all(post_mean < obs)

    def test_posterior_variance_shrinks(self, engine):
        prior_var = np.ones(16)
        obs = np.ones(16)
        _, post_var = engine.get_posterior_distribution(obs, np.zeros(16), prior_var)
        assert np.all(post_var < prior_var)


# ---------------------------------------------------------------------------
# process_sequence
# ---------------------------------------------------------------------------

class TestProcessSequence:
    def test_returns_list_of_dicts(self, engine):
        seq = [np.random.randn(16).astype(np.float32) for _ in range(10)]
        results = engine.process_sequence(seq)
        assert len(results) == 10
        assert all("surprise" in r for r in results)


# ---------------------------------------------------------------------------
# Event boundary detection
# ---------------------------------------------------------------------------

class TestEventBoundaries:
    def test_peaks_method_finds_local_maxima(self, engine):
        # Feed enough to establish a mean
        for _ in range(20):
            engine.compute_surprise(np.random.randn(16).astype(np.float32))

        values = [0.1, 0.2, 0.5, 0.2, 0.1, 0.1, 0.8, 0.1, 0.1]
        # mean_surprise is likely low, so peaks at 2 and 6 should be found
        engine.mean_surprise = 0.15
        boundaries = engine.get_event_boundaries(values, method="peaks")
        assert 2 in boundaries
        assert 6 in boundaries

    def test_threshold_method(self, engine):
        engine.config.surprise_threshold = 0.5
        values = [0.1, 0.2, 0.8, 0.1, 0.9]
        boundaries = engine.get_event_boundaries(values, method="threshold")
        assert 2 in boundaries
        assert 4 in boundaries
        assert 0 not in boundaries


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_dimension_raises(self, engine):
        with pytest.raises(ValueError, match="does not match input_dim"):
            engine.compute_surprise(np.zeros(999))

    def test_list_input_coerced(self, engine):
        # Should accept list and coerce to ndarray
        r = engine.compute_surprise([0.0] * 16)
        assert "surprise" in r

    def test_none_input_raises(self, engine):
        with pytest.raises((ValueError, TypeError)):
            engine.compute_surprise(None)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self, engine):
        for _ in range(10):
            engine.compute_surprise(np.random.randn(16).astype(np.float32))
        engine.reset()
        assert engine.step_count == 0
        assert engine.total_surprise == 0.0
        assert len(engine.observation_history) == 0
        assert len(engine.surprise_history) == 0
