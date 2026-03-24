"""Tests for Episodic Memory Store."""
import numpy as np
import pytest
from datetime import datetime
from elo_memory.memory.episodic_store import EpisodicMemoryStore, EpisodicMemoryConfig, Episode


def test_store_init():
    config = EpisodicMemoryConfig(embedding_dim=16)
    store = EpisodicMemoryStore(config)
    assert len(store.episodes) == 0


def test_store_episode():
    config = EpisodicMemoryConfig(embedding_dim=16)
    store = EpisodicMemoryStore(config)
    ep = store.store_episode(
        content={"text": "test memory", "metadata": {}},
        embedding=np.random.randn(16).astype(np.float32),
        surprise=0.5,
        timestamp=datetime.now(),
    )
    assert ep is not None
    assert len(store.episodes) == 1


def test_store_multiple():
    config = EpisodicMemoryConfig(embedding_dim=16)
    store = EpisodicMemoryStore(config)
    for i in range(5):
        store.store_episode(
            content={"text": f"memory {i}", "metadata": {}},
            embedding=np.random.randn(16).astype(np.float32),
            surprise=float(i) * 0.1,
            timestamp=datetime.now(),
        )
    assert len(store.episodes) == 5
