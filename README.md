# Elo Memory

[![PyPI version](https://badge.fury.io/py/elo-memory.svg)](https://badge.fury.io/py/elo-memory)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/server-elo/elo-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/server-elo/elo-memory/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Bio-inspired episodic memory system implementing EM-LLM (ICLR 2025).**

The missing memory layer for AI agents — automatic event detection, surprise-based encoding, and human-like memory consolidation.

**Works with:** OpenClaw | Claude Code | OpenCode | Codex | Claude | Any MCP-compatible agent

---

## Why Elo Memory?

AI agents forget everything between conversations. Elo Memory fixes that.

- **Fast** — Retrieves relevant memories in ~5ms. Agent queries by similarity, gets top 5 matches. Never reads all memories.
- **Smart storage** — Bayesian surprise engine decides in <1ms what's worth remembering. Repetitive content is skipped automatically.
- **Human-like recall** — Two-stage retrieval finds by similarity first, then expands by time context. Like how you remember "that whole day" not just one fact.
- **Self-maintaining** — Background consolidation extracts patterns. Old irrelevant memories decay naturally. No manual cleanup.
- **Works everywhere** — Python library, MCP server, or REST API. Drop into any agent framework.

---

## How It Works

```
User message → Query memory (5ms) → 5 relevant episodes → Added to prompt → Better response
                                              ↓
                                    Surprise check (1ms) → Novel? Store it. Boring? Skip it.
                                              ↓
                                    Consolidation (background) → Extract patterns, forget noise
```

---

## Quick Start

```bash
pip install elo-memory
```

```python
from elo_memory import EpisodicMemoryStore, BayesianSurpriseEngine

# Initialize memory
memory = EpisodicMemoryStore(embedding_dim=768)
surprise = BayesianSurpriseEngine(input_dim=768)

# Store an observation
embedding = encoder.encode("User loves Italian food")
surprise_info = surprise.compute_surprise(embedding)

if surprise_info['is_novel']:
    memory.store_episode(
        content={"text": "User loves Italian food"},
        embedding=embedding,
        surprise=surprise_info['surprise']
    )

# Retrieve relevant memories
results = memory.retrieve(query_embedding, k=5)
```

---

## vs Alternatives

| | Elo Memory | Mem0/Zep | Plain RAG |
|--|-----------|----------|-----------|
| Stores | Experiences with surprise | Everything | Documents |
| Retrieval | Similarity + temporal | Similarity only | Similarity only |
| Filtering | Automatic (surprise) | Manual | None |
| Forgetting | Natural decay | Manual cleanup | None |
| Speed | ~5ms query | ~50ms | ~100ms |
| Cost | Free | $70+/month | API costs |

---

## Components (8/8 Complete)

| Component | Description | Status |
|-----------|-------------|--------|
| **Bayesian Surprise Detection** | KL divergence-based novelty detection | ✅ |
| **Event Segmentation** | HMM + prediction error boundaries | ✅ |
| **Episodic Storage** | ChromaDB with temporal-spatial indexing | ✅ |
| **Two-Stage Retrieval** | Similarity + temporal expansion | ✅ |
| **Memory Consolidation** | Sleep-like replay + schema extraction | ✅ |
| **Forgetting & Decay** | Power-law activation decay | ✅ |
| **Interference Resolution** | Pattern separation/completion | ✅ |
| **Online Learning** | Experience replay + adaptive thresholds | ✅ |

---

## Documentation

- [Usage Guide](USAGE_GUIDE.md) — Real-world integration patterns
- [MCP Setup](MCP_SETUP.md) — Claude Code integration
- [Performance Comparison](COMPARISON.md) — Benchmarks vs competitors
- [Competitive Analysis](COMPETITIVE_ANALYSIS.md) — Honest assessment
- [Real-World Project](REAL_WORLD_PROJECT.md) — Build an AI assistant
- [Strategic Plan](STRATEGIC_PLAN.md) — Roadmap & vision

---

## Installation

### From PyPI
```bash
pip install elo-memory
```

### From Source
```bash
git clone https://github.com/server-elo/elo-memory.git
cd elo-memory
pip install -e ".[dev]"
```

### With API Server
```bash
pip install "elo-memory[api]"
elo-memory server --port 8000
```

---

## Running Tests

```bash
pytest tests/ -v --cov=elo_memory
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick start:
```bash
git clone https://github.com/server-elo/elo-memory.git
cd elo-memory
pip install -e ".[dev]"
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- EM-LLM (ICLR 2025) — Research foundation
- Itti & Baldi (2009) — Bayesian Surprise
- Squire & Alvarez (1995) — Systems Consolidation
- Kirkpatrick et al. (2017) — Catastrophic Forgetting

---

## Links

- **GitHub**: https://github.com/server-elo/elo-memory
- **PyPI**: https://pypi.org/project/elo-memory/
- **Documentation**: https://github.com/server-elo/elo-memory#readme
- **Issues**: https://github.com/server-elo/elo-memory/issues

---

**Status**: Production ready ✅

Made with ❤️ by the Elo Memory community.