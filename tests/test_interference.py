"""Tests for Interference Resolution — pattern separation and completion."""
import numpy as np
import pytest
from elo_memory.memory.interference import InterferenceResolver, InterferenceConfig


def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-8)


@pytest.fixture
def resolver():
    return InterferenceResolver()


# ---------------------------------------------------------------------------
# Interference detection
# ---------------------------------------------------------------------------

class TestInterferenceDetection:
    def test_detects_near_duplicate(self, resolver):
        v1 = _norm(np.random.randn(64))
        v2 = v1 + np.random.randn(64) * 0.01
        v2 = _norm(v2)
        indices = resolver.detect_interference(v2, [v1])
        assert 0 in indices

    def test_ignores_dissimilar(self, resolver):
        v1 = _norm(np.random.randn(64))
        v2 = _norm(np.random.randn(64))
        indices = resolver.detect_interference(v2, [v1])
        # random 64-d vectors are nearly orthogonal
        assert indices == []

    def test_multiple_interfering(self, resolver):
        base = _norm(np.ones(64))
        existing = [
            _norm(np.ones(64) + np.random.randn(64) * 0.01),
            _norm(np.random.randn(64)),  # dissimilar
            _norm(np.ones(64) + np.random.randn(64) * 0.01),
        ]
        indices = resolver.detect_interference(base, existing)
        assert 0 in indices
        assert 2 in indices
        assert 1 not in indices

    def test_empty_existing_returns_empty(self, resolver):
        v = _norm(np.random.randn(64))
        assert resolver.detect_interference(v, []) == []


# ---------------------------------------------------------------------------
# Pattern separation
# ---------------------------------------------------------------------------

class TestPatternSeparation:
    def test_reduces_similarity(self, resolver):
        v1 = _norm(np.ones(64))
        v2 = _norm(np.ones(64) + np.random.randn(64) * 0.05)
        sim_before = resolver._cosine_similarity(v1, v2)

        separated = resolver.apply_pattern_separation(v2, v1)
        sim_after = resolver._cosine_similarity(separated, v1)
        assert sim_after < sim_before

    def test_output_is_normalized(self, resolver):
        v1 = _norm(np.random.randn(64))
        v2 = _norm(np.random.randn(64) * 0.01 + v1)
        separated = resolver.apply_pattern_separation(v2, v1)
        assert np.linalg.norm(separated) == pytest.approx(1.0, abs=0.05)

    def test_separation_respects_min_distance(self, resolver):
        v1 = _norm(np.ones(64))
        # identical direction
        v2 = v1.copy()
        separated = resolver.apply_pattern_separation(v2, v1)
        sim = resolver._cosine_similarity(separated, v1)
        assert sim < 1.0


# ---------------------------------------------------------------------------
# Pattern completion
# ---------------------------------------------------------------------------

class TestPatternCompletion:
    def test_completes_to_best_match(self, resolver):
        target = _norm(np.ones(64))
        distractor = _norm(np.random.randn(64))
        partial = _norm(target * 0.8 + np.random.randn(64) * 0.1)
        result = resolver.pattern_complete(partial, [target, distractor], threshold=0.3)
        assert result is not None
        np.testing.assert_array_almost_equal(result, target)

    def test_returns_none_below_threshold(self, resolver):
        stored = [_norm(np.random.randn(64))]
        query = _norm(np.random.randn(64))
        result = resolver.pattern_complete(query, stored, threshold=0.99)
        assert result is None

    def test_empty_patterns_returns_none(self, resolver):
        query = _norm(np.random.randn(64))
        assert resolver.pattern_complete(query, [], threshold=0.0) is None


# ---------------------------------------------------------------------------
# resolve_interference_set (integration)
# ---------------------------------------------------------------------------

class TestResolveSet:
    def test_no_interference_returns_unchanged(self, resolver):
        new = _norm(np.random.randn(64))
        existing = [_norm(np.random.randn(64)) for _ in range(5)]
        separated, updated = resolver.resolve_interference_set(new, existing)
        # Should be unchanged (or very close)
        np.testing.assert_array_almost_equal(separated, new, decimal=5)

    def test_interference_produces_different_embedding(self, resolver):
        base = _norm(np.ones(64))
        existing = [_norm(np.ones(64) + np.random.randn(64) * 0.01)]
        separated, _ = resolver.resolve_interference_set(base, existing)
        assert not np.allclose(separated, base, atol=0.01)


# ---------------------------------------------------------------------------
# Cosine similarity edge cases
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_zero_vectors(self, resolver):
        z = np.zeros(10)
        v = np.ones(10)
        assert resolver._cosine_similarity(z, v) == 0.0
        assert resolver._cosine_similarity(z, z) == 0.0

    def test_identical_vectors(self, resolver):
        v = _norm(np.random.randn(32))
        assert resolver._cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self, resolver):
        v = _norm(np.random.randn(32))
        assert resolver._cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)
