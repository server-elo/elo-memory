#!/usr/bin/env python3
"""
MCP Server for Elo Memory
Exposes memory operations as MCP tools
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from elo_memory import (
    BayesianSurpriseEngine, SurpriseConfig,
    EventSegmenter, SegmentationConfig,
    EpisodicMemoryStore, EpisodicMemoryConfig,
    TwoStageRetriever, RetrievalConfig,
    MemoryConsolidationEngine, ConsolidationConfig
)

import numpy as np

logger = logging.getLogger("elo_memory.mcp")

# Try to load sentence-transformers for embedding generation
EMBEDDING_MODEL = None
EMBEDDING_DIM = 768  # fallback default

def _load_embedding_model(model_name: str = 'BAAI/bge-small-en-v1.5', retries: int = 3) -> Optional[Any]:
    """Load SentenceTransformer with retries for network/disk failures."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed, using hash embeddings")
        return None

    for attempt in range(1, retries + 1):
        try:
            model = SentenceTransformer(model_name)
            return model
        except Exception as e:
            logger.warning("Embedding model load attempt %d/%d failed: %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(2 ** attempt)
    logger.warning("All embedding model load attempts failed, using hash embeddings")
    return None

_embedding_model_loaded = False


def _ensure_embedding_model():
    """Lazy-load embedding model on first use, not at import time."""
    global EMBEDDING_MODEL, EMBEDDING_DIM, _embedding_model_loaded
    if _embedding_model_loaded:
        return
    _embedding_model_loaded = True
    EMBEDDING_MODEL = _load_embedding_model()
    if EMBEDDING_MODEL is not None:
        EMBEDDING_DIM = EMBEDDING_MODEL.get_sentence_embedding_dimension()
    else:
        logger.info("Using hash embeddings (dim=%d)", EMBEDDING_DIM)


class NeuroMemoryMCP:
    """MCP Server for Elo Memory"""

    MAX_CONTENT_SIZE = 10_000  # Maximum characters per memory content
    MAX_K = 100  # Maximum number of results per retrieval

    def __init__(self, input_dim: int = None):
        _ensure_embedding_model()
        self.input_dim = input_dim or EMBEDDING_DIM

        # Initialize components
        self.surprise_engine = BayesianSurpriseEngine(self.input_dim)
        self.segmenter = EventSegmenter()
        self.memory = EpisodicMemoryStore(
            EpisodicMemoryConfig(max_episodes=10000, embedding_dim=self.input_dim)
        )
        self.retriever = TwoStageRetriever(self.memory)
        self.consolidation = MemoryConsolidationEngine()

    def _get_embedding(self, content: str, provided_embedding: Optional[List[float]] = None) -> np.ndarray:
        """Get embedding for content, using provided or generating new."""
        if provided_embedding is not None:
            emb = np.array(provided_embedding, dtype=np.float32)
            if emb.shape != (self.input_dim,):
                raise ValueError(
                    f"Embedding dimension mismatch: got {emb.shape[0]}, expected {self.input_dim}"
                )
            return emb

        if EMBEDDING_MODEL is not None:
            return EMBEDDING_MODEL.encode([content])[0]
        else:
            return self._hash_embedding(content)

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Simple hash-based embedding (fallback)"""
        embedding = np.zeros(self.input_dim)
        for i, char in enumerate(text):
            idx = (ord(char) * (i + 1)) % self.input_dim
            embedding[idx] += np.sin(ord(char) * 0.1) * 0.5
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / (norm or 1)

    def store_memory(self, content: str, embedding: List[float] = None, metadata: Dict = None) -> Dict:
        """Store a new memory episode (embedding optional, auto-generated if not provided)"""
        if not content or not isinstance(content, str):
            raise ValueError("content must be a non-empty string")
        if len(content) > self.MAX_CONTENT_SIZE:
            raise ValueError(
                f"content too large: {len(content)} chars (max {self.MAX_CONTENT_SIZE})"
            )
        embedding_array = self._get_embedding(content, embedding)

        # Compute surprise (for scoring, not gating)
        surprise_info = self.surprise_engine.compute_surprise(embedding_array)
        surprise_val = float(surprise_info['surprise'])
        is_novel = bool(surprise_info['is_novel'])

        # Always store — every conversation matters
        episode = self.memory.store_episode(
            content={"text": content, "metadata": metadata or {}},
            embedding=embedding_array,
            surprise=surprise_val,
            timestamp=datetime.now(timezone.utc)
        )
        return {
            "stored": True,
            "episode_id": episode.episode_id,
            "surprise": surprise_val,
            "is_novel": is_novel
        }

    def retrieve_memories(self, query: str = None, query_embedding: List[float] = None, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories (query text or embedding, one required)"""
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, self.MAX_K)
        if query_embedding is not None:
            query_array = np.array(query_embedding, dtype=np.float32)
            if query_array.shape != (self.input_dim,):
                raise ValueError(
                    f"Embedding dimension mismatch: got {query_array.shape[0]}, expected {self.input_dim}"
                )
        elif query is not None:
            query_array = self._get_embedding(query)
        else:
            raise ValueError("Either query or query_embedding required")

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

    def consolidate_memories(self) -> Dict:
        """Run memory consolidation"""
        stats = self.consolidation.consolidate(self.memory.episodes)
        schemas = self.consolidation.get_schema_summary()

        return {
            "replay_count": stats['replay_count'],
            "schemas_extracted": len(schemas),
            "schemas": schemas[:5]  # Top 5 schemas
        }

    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            "total_episodes": len(self.memory.episodes),
            "mean_surprise": float(self.surprise_engine.mean_surprise),
            "std_surprise": float(self.surprise_engine.std_surprise),
            "observation_count": self.surprise_engine.step_count
        }


import threading
_mcp_lock = threading.Lock()
_mcp_instance: Optional[NeuroMemoryMCP] = None


def handle_request(request: Dict) -> Dict:
    """Handle MCP tool request"""
    global _mcp_instance
    if _mcp_instance is None:
        with _mcp_lock:
            if _mcp_instance is None:
                _mcp_instance = NeuroMemoryMCP()
    mcp = _mcp_instance
    method = request.get('method')
    params = request.get('params', {})

    try:
        if method == 'store_memory':
            result = mcp.store_memory(
                content=params['content'],
                embedding=params.get('embedding'),
                metadata=params.get('metadata')
            )
        elif method == 'retrieve_memories':
            result = mcp.retrieve_memories(
                query=params.get('query'),
                query_embedding=params.get('embedding'),
                k=params.get('k', 5)
            )
        elif method == 'consolidate_memories':
            result = mcp.consolidate_memories()
        elif method == 'get_stats':
            result = mcp.get_stats()
        else:
            return {"error": f"Unknown method: {method}"}

        return {"result": result}

    except Exception as e:
        logger.exception("Error handling %s", method)
        return {"error": str(e)}


def main():
    """MCP Server main loop"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Neuro-Memory MCP Server started")

    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            logger.exception("Failed to process request")
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
