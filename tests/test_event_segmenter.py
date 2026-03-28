"""Tests for Event Segmentation — prediction error, graph refinement, segmenter."""
import numpy as np
import pytest
from elo_memory.segmentation.event_segmenter import (
    PredictionErrorDetector,
    GraphBoundaryRefiner,
    EventSegmenter,
    SegmentationConfig,
)


# ---------------------------------------------------------------------------
# PredictionErrorDetector
# ---------------------------------------------------------------------------

class TestPredictionErrorDetector:
    @pytest.fixture
    def detector(self):
        return PredictionErrorDetector(threshold=2.0)

    def test_compute_prediction_error_shape(self, detector):
        obs = np.random.randn(50, 10)
        pred = np.random.randn(50, 10)
        errors = detector.compute_prediction_error(obs, pred)
        assert errors.shape == (50,)

    def test_zero_error_when_identical(self, detector):
        obs = np.random.randn(50, 10)
        errors = detector.compute_prediction_error(obs, obs)
        np.testing.assert_array_almost_equal(errors, np.zeros(50))

    def test_detects_boundaries_at_spikes(self, detector):
        errors = np.random.randn(100) * 0.3
        # Inject clear spikes
        errors[30] = 10.0
        errors[70] = 10.0
        boundaries = detector.detect_boundaries(errors)
        assert 30 in boundaries
        assert 70 in boundaries

    def test_no_boundaries_in_flat_signal(self, detector):
        errors = np.ones(100) * 0.5
        boundaries = detector.detect_boundaries(errors)
        assert boundaries == []

    def test_edge_indices_not_included(self, detector):
        """Boundaries at index 0 and len-1 should never appear (not local max)."""
        errors = np.zeros(50)
        errors[0] = 100.0
        errors[49] = 100.0
        boundaries = detector.detect_boundaries(errors)
        assert 0 not in boundaries
        assert 49 not in boundaries


# ---------------------------------------------------------------------------
# GraphBoundaryRefiner
# ---------------------------------------------------------------------------

class TestGraphBoundaryRefiner:
    @pytest.fixture
    def refiner(self):
        return GraphBoundaryRefiner(metric="modularity")

    def test_build_similarity_graph(self, refiner):
        obs = np.random.randn(20, 5)
        G = refiner.build_similarity_graph(obs, k_neighbors=3)
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() > 0

    def test_boundaries_to_communities(self, refiner):
        communities = refiner._boundaries_to_communities(10, [3, 7])
        assert len(communities) == 10
        # Three segments: [0-2], [3-6], [7-9]
        assert communities[0] == communities[1] == communities[2]
        assert communities[3] == communities[4]
        assert communities[7] == communities[8]
        assert communities[0] != communities[3]
        assert communities[3] != communities[7]

    def test_modularity_is_finite(self, refiner):
        obs = np.random.randn(30, 5)
        G = refiner.build_similarity_graph(obs, k_neighbors=3)
        mod = refiner.compute_modularity(G, [10, 20])
        assert np.isfinite(mod)

    def test_conductance_in_valid_range(self):
        refiner = GraphBoundaryRefiner(metric="conductance")
        obs = np.random.randn(30, 5)
        G = refiner.build_similarity_graph(obs, k_neighbors=3)
        cond = refiner.compute_conductance(G, [10, 20])
        assert 0.0 <= cond <= 1.0

    def test_refine_boundaries_returns_sorted(self, refiner):
        obs = np.random.randn(50, 5)
        initial = [15, 35]
        refined = refiner.refine_boundaries(obs, initial, max_iterations=3)
        assert refined == sorted(refined)

    def test_refine_empty_boundaries(self, refiner):
        obs = np.random.randn(20, 5)
        refined = refiner.refine_boundaries(obs, [], max_iterations=3)
        assert refined == []


# ---------------------------------------------------------------------------
# EventSegmenter (prediction_error mode — avoids hmmlearn dependency issues)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HiddenMarkovEventDetector
# ---------------------------------------------------------------------------

class TestHiddenMarkovEventDetector:
    def test_segment_sequence_returns_boundaries(self):
        from elo_memory.segmentation.event_segmenter import HiddenMarkovEventDetector
        detector = HiddenMarkovEventDetector(n_states=3, window_size=10)
        np.random.seed(42)
        # Two distinct clusters should produce at least one boundary
        obs = np.vstack([
            np.random.randn(30, 5) * 0.3,
            np.random.randn(30, 5) * 0.3 + 5.0,
        ])
        boundaries = detector.segment_sequence(obs)
        assert isinstance(boundaries, list)

    def test_fallback_clustering(self):
        from elo_memory.segmentation.event_segmenter import HiddenMarkovEventDetector
        detector = HiddenMarkovEventDetector(n_states=3)
        np.random.seed(42)
        obs = np.vstack([
            np.random.randn(20, 5) * 0.3,
            np.random.randn(20, 5) * 0.3 + 5.0,
        ])
        boundaries = detector._fallback_clustering(obs)
        assert isinstance(boundaries, list)
        assert all(isinstance(b, (int, np.integer)) for b in boundaries)


# ---------------------------------------------------------------------------
# EventSegmenter (full pipeline)
# ---------------------------------------------------------------------------

class TestEventSegmenter:
    def test_segment_with_prediction_error(self):
        config = SegmentationConfig(
            state_detection_method="prediction_error",
            boundary_refinement=False,
            prediction_error_threshold=2.0,
        )
        segmenter = EventSegmenter(config)

        np.random.seed(42)
        # Two clear clusters
        obs = np.vstack([
            np.random.randn(30, 8) * 0.3,
            np.random.randn(30, 8) * 0.3 + 5.0,
        ])
        # Surprise spike at boundary
        surprise = np.random.rand(60) * 0.3
        surprise[28:32] = 5.0

        result = segmenter.segment(obs, surprise_values=surprise)
        assert "boundaries" in result
        assert "events" in result
        assert "confidence" in result
        assert "n_events" in result
        assert result["n_events"] >= 1

    def test_prediction_error_requires_surprise(self):
        config = SegmentationConfig(state_detection_method="prediction_error")
        segmenter = EventSegmenter(config)
        obs = np.random.randn(20, 5)
        with pytest.raises(ValueError, match="surprise_values required"):
            segmenter.segment(obs, surprise_values=None)

    def test_filter_boundaries_min_length(self):
        config = SegmentationConfig(min_event_length=10, max_event_length=100)
        segmenter = EventSegmenter(config)
        # Boundaries that create a segment of length 3 (too short)
        filtered = segmenter._filter_boundaries([3, 6, 50], n_timesteps=100)
        # Segments: [0-3]=3, [3-6]=3, [6-50]=44, [50-100]=50
        # First two are too short, so their end boundaries should be dropped
        assert all(b > 6 for b in filtered)

    def test_extract_events_correct_count(self):
        config = SegmentationConfig()
        segmenter = EventSegmenter(config)
        obs = np.random.randn(100, 5)
        events = segmenter._extract_events(obs, [30, 60])
        assert len(events) == 3
        assert len(events[0]) == 30
        assert len(events[1]) == 30
        assert len(events[2]) == 40

    def test_confidence_with_surprise_values(self):
        config = SegmentationConfig()
        segmenter = EventSegmenter(config)
        obs = np.random.randn(100, 5)
        surprise = np.random.rand(100) * 0.5
        surprise[50] = 5.0  # spike at boundary
        confidences = segmenter._compute_confidence(obs, [50], surprise)
        assert len(confidences) == 1
        assert 0.0 <= confidences[0] <= 1.0

    def test_confidence_without_surprise_defaults(self):
        config = SegmentationConfig()
        segmenter = EventSegmenter(config)
        obs = np.random.randn(100, 5)
        confidences = segmenter._compute_confidence(obs, [50], None)
        assert confidences == [0.5]

    def test_no_boundaries_empty_confidence(self):
        config = SegmentationConfig()
        segmenter = EventSegmenter(config)
        obs = np.random.randn(100, 5)
        assert segmenter._compute_confidence(obs, [], None) == []
