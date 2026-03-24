"""Tests for Bayesian Surprise Engine."""
import numpy as np
import pytest
from elo_memory.surprise.bayesian_surprise import BayesianSurpriseEngine, SurpriseConfig


def test_engine_init():
    engine = BayesianSurpriseEngine(SurpriseConfig(input_dim=384))
    assert engine is not None


def test_compute_surprise_returns_dict():
    engine = BayesianSurpriseEngine(SurpriseConfig(input_dim=16))
    embedding = np.random.randn(16).astype(np.float32)
    result = engine.compute_surprise(embedding)
    assert "surprise" in result
    assert "is_novel" in result
    assert isinstance(result["surprise"], (int, float))
    assert isinstance(result["is_novel"], bool)


def test_surprise_increases_for_outliers():
    engine = BayesianSurpriseEngine(SurpriseConfig(input_dim=16))
    # Feed normal data
    for _ in range(20):
        normal = np.random.randn(16).astype(np.float32) * 0.1
        engine.compute_surprise(normal)
    # Feed outlier
    outlier = np.ones(16, dtype=np.float32) * 10.0
    result = engine.compute_surprise(outlier)
    assert result["surprise"] > 0


def test_step_count_increments():
    engine = BayesianSurpriseEngine(SurpriseConfig(input_dim=8))
    assert engine.step_count == 0
    engine.compute_surprise(np.zeros(8, dtype=np.float32))
    assert engine.step_count == 1
