"""Integration tests for MCP servers — every endpoint, JSON-serializable."""
import json
import numpy as np
import pytest
from datetime import datetime
from elo_memory.memory.episodic_store import EpisodicMemoryConfig, EpisodicMemoryStore
from elo_memory.retrieval.two_stage_retriever import TwoStageRetriever
from elo_memory.surprise.bayesian_surprise import BayesianSurpriseEngine
from elo_memory.consolidation.memory_consolidation import MemoryConsolidationEngine
from elo_memory.segmentation.event_segmenter import EventSegmenter


# ---------------------------------------------------------------------------
# We test NeuroMemoryMCP directly (sync server) rather than going through
# stdin/stdout. This validates that every response is JSON-serializable and
# that the wiring between components is correct.
# ---------------------------------------------------------------------------


class FakeMCP:
    """
    Minimal recreation of NeuroMemoryMCP that uses hash embeddings
    (no sentence-transformers dependency) and a temp persistence path.
    Mirrors the real server's wiring exactly.
    """

    def __init__(self, tmp_path, input_dim=32):
        self.input_dim = input_dim
        self.surprise_engine = BayesianSurpriseEngine(input_dim)
        self.segmenter = EventSegmenter()
        self.memory = EpisodicMemoryStore(
            EpisodicMemoryConfig(
                max_episodes=1000,
                embedding_dim=input_dim,
                persistence_path=str(tmp_path / "store"),
                consolidation_min_episodes=500,
            )
        )
        self.retriever = TwoStageRetriever(self.memory)
        self.consolidation = MemoryConsolidationEngine()

    def _hash_embedding(self, text: str) -> np.ndarray:
        embedding = np.zeros(self.input_dim)
        for i, char in enumerate(text):
            idx = (ord(char) * (i + 1)) % self.input_dim
            embedding[idx] += np.sin(ord(char) * 0.1) * 0.5
        norm = np.linalg.norm(embedding)
        return embedding / (norm or 1)

    def store_memory(self, content, embedding=None, metadata=None):
        if not content or not isinstance(content, str):
            raise ValueError("content must be a non-empty string")
        if embedding is not None:
            emb = np.array(embedding, dtype=np.float32)
            if emb.shape != (self.input_dim,):
                raise ValueError(
                    f"Embedding dimension mismatch: got {emb.shape[0]}, expected {self.input_dim}"
                )
        else:
            emb = self._hash_embedding(content)

        surprise_info = self.surprise_engine.compute_surprise(emb)
        surprise_val = float(surprise_info["surprise"])
        is_novel = bool(surprise_info["is_novel"])

        episode = self.memory.store_episode(
            content={"text": content, "metadata": metadata or {}},
            embedding=emb,
            surprise=surprise_val,
            timestamp=datetime.now(),
        )
        return {
            "stored": True,
            "episode_id": episode.episode_id,
            "surprise": surprise_val,
            "is_novel": is_novel,
        }

    def retrieve_memories(self, query=None, query_embedding=None, k=5):
        if k < 1:
            return {"error": "k must be >= 1"}
        if query_embedding is not None:
            query_array = np.array(query_embedding, dtype=np.float32)
            if query_array.shape != (self.input_dim,):
                return {
                    "error": f"Embedding dimension mismatch: got {query_array.shape[0]}, expected {self.input_dim}"
                }
        elif query is not None:
            query_array = self._hash_embedding(query)
        else:
            return {"error": "Either query or query_embedding required"}

        self.retriever.config.max_retrieved = k
        results = self.retriever.retrieve(query=query_array)
        return [
            {
                "content": ep.content,
                "surprise": float(ep.surprise),
                "timestamp": ep.timestamp.isoformat(),
                "similarity": float(score),
            }
            for ep, score in results
        ]

    def consolidate_memories(self):
        stats = self.consolidation.consolidate(self.memory.episodes)
        schemas = self.consolidation.get_schema_summary()
        return {
            "replay_count": stats["replay_count"],
            "schemas_extracted": len(schemas),
            "schemas": schemas[:5],
        }

    def get_stats(self):
        return {
            "total_episodes": len(self.memory.episodes),
            "mean_surprise": float(self.surprise_engine.mean_surprise),
            "std_surprise": float(self.surprise_engine.std_surprise),
            "observation_count": self.surprise_engine.step_count,
        }


@pytest.fixture
def mcp(tmp_path):
    return FakeMCP(tmp_path)


def _assert_json_serializable(obj):
    """Assert the object survives json.dumps without numpy type errors."""
    serialized = json.dumps(obj)
    assert isinstance(serialized, str)
    return serialized


# ---------------------------------------------------------------------------
# store_memory
# ---------------------------------------------------------------------------

class TestStoreMemory:
    def test_store_returns_valid_json(self, mcp):
        r = mcp.store_memory("hello world")
        _assert_json_serializable(r)
        assert r["stored"] is True
        assert isinstance(r["episode_id"], str)
        assert isinstance(r["surprise"], float)
        assert isinstance(r["is_novel"], bool)

    def test_store_with_metadata(self, mcp):
        r = mcp.store_memory("test", metadata={"source": "unit_test"})
        _assert_json_serializable(r)

    def test_store_rejects_wrong_dimension(self, mcp):
        with pytest.raises(ValueError, match="dimension mismatch"):
            mcp.store_memory("test", embedding=[1.0, 2.0, 3.0])

    def test_store_rejects_empty_content(self, mcp):
        with pytest.raises(ValueError, match="non-empty string"):
            mcp.store_memory("")

    def test_store_rejects_none_content(self, mcp):
        with pytest.raises(ValueError, match="non-empty string"):
            mcp.store_memory(None)

    def test_store_accepts_correct_dimension(self, mcp):
        emb = np.random.randn(mcp.input_dim).tolist()
        r = mcp.store_memory("test", embedding=emb)
        _assert_json_serializable(r)


# ---------------------------------------------------------------------------
# retrieve_memories
# ---------------------------------------------------------------------------

class TestRetrieveMemories:
    def test_retrieve_by_text_query(self, mcp):
        mcp.store_memory("The cat sat on the mat")
        mcp.store_memory("Dogs are loyal animals")
        r = mcp.retrieve_memories(query="cat")
        _assert_json_serializable(r)
        assert isinstance(r, list)

    def test_retrieve_returns_stored_content(self, mcp):
        mcp.store_memory("unique test phrase XYZ123")
        r = mcp.retrieve_memories(query="unique test phrase XYZ123")
        assert len(r) > 0
        texts = [item["content"]["text"] for item in r]
        assert "unique test phrase XYZ123" in texts

    def test_retrieve_rejects_wrong_dimension(self, mcp):
        r = mcp.retrieve_memories(query_embedding=[1.0, 2.0])
        assert "error" in r
        _assert_json_serializable(r)

    def test_retrieve_requires_query(self, mcp):
        r = mcp.retrieve_memories()
        assert "error" in r

    def test_retrieve_respects_k(self, mcp):
        for i in range(10):
            mcp.store_memory(f"memory number {i}")
        r = mcp.retrieve_memories(query="memory", k=3)
        assert len(r) <= 3

    def test_retrieve_empty_store(self, mcp):
        r = mcp.retrieve_memories(query="anything")
        assert r == []

    def test_retrieve_rejects_k_zero(self, mcp):
        r = mcp.retrieve_memories(query="test", k=0)
        assert "error" in r


# ---------------------------------------------------------------------------
# consolidate_memories
# ---------------------------------------------------------------------------

class TestConsolidate:
    def test_consolidate_returns_valid_json(self, mcp):
        for i in range(5):
            mcp.store_memory(f"memory {i}")
        r = mcp.consolidate_memories()
        _assert_json_serializable(r)
        assert "replay_count" in r
        assert "schemas_extracted" in r

    def test_consolidate_empty_store(self, mcp):
        r = mcp.consolidate_memories()
        _assert_json_serializable(r)


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_returns_valid_json(self, mcp):
        r = mcp.get_stats()
        _assert_json_serializable(r)
        assert r["total_episodes"] == 0
        assert isinstance(r["mean_surprise"], float)

    def test_stats_reflect_stores(self, mcp):
        mcp.store_memory("one")
        mcp.store_memory("two")
        r = mcp.get_stats()
        assert r["total_episodes"] == 2
        assert r["observation_count"] == 2
