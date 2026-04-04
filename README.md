# Elo Memory

[![PyPI version](https://badge.fury.io/py/elo-memory.svg)](https://badge.fury.io/py/elo-memory)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](https://github.com/server-elo/elo-memory/blob/main/LICENSE)
[![Tests](https://github.com/server-elo/elo-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/server-elo/elo-memory/actions)

**The memory brain for AI agents.** 3 lines to give any LLM persistent, self-improving memory that learns what to remember, reasons about cause and effect, dreams to consolidate knowledge, and proves its memories are authentic.

```python
from elo_memory import EloBrain

brain = EloBrain("user_123")
response = brain.think("I switched from Django to FastAPI because it was too slow", your_llm)
# Brain auto: recalled context, stored the fact, updated KB, superseded Django memory,
# built causal graph (slow → switched), sealed memory in hash chain, predicted next query,
# governor learned this was worth storing, evolution adapted embeddings
```

---

## Why Elo Memory?

Every other memory system stores text and searches it. Elo Memory **thinks about its own memory.**

| System | What it does |
|--------|-------------|
| **ChromaDB** | Stores text. Retrieves by similarity. No idea what's current vs outdated. |
| **Mem0** | Calls GPT to extract facts ($0.005/store). Slow (500ms), needs API key, sends data to OpenAI. |
| **Elo Memory** | Stores, reasons, evolves. Builds causal graphs. Dreams to consolidate. Adapts embeddings from feedback. Seals every memory cryptographically. All locally, 66ms, $0. |

---

## Quick Start

```bash
pip install elo-memory
```

### For AI agents (recommended)

```python
from elo_memory import EloBrain

brain = EloBrain("sarah")

# Full agent loop: recall -> prompt -> LLM -> store
response = brain.think(
    "I'm Sarah, senior engineer at Shopify. We use PostgreSQL.",
    llm_fn=lambda prompt: your_llm(prompt),
)

# What does the brain know?
state = brain.what_i_know()
# {
#   knowledge: {name: "Sarah", role: "senior engineer", company: "Shopify", database: "PostgreSQL"},
#   knowledge_gaps: [{topic: "infrastructure", missing: ["hosting", "ci/cd", "monitoring"]}],
#   causal_graph: {nodes: 3, edges: 1},
#   governor: {total_decisions: 1, learned_preferences: [...]},
#   auditor: {chain_length: 1, merkle_root: "a3f2..."},
#   prefetcher: {topic_transitions: 2, cache_size: 1},
# }
```

---

## What Happens on Every Turn

When you call `brain.think()` or `brain.process_turn()`, all 16 components fire:

### Storage Path (process_turn)

1. **Governor** decides whether to store, skip, or promote (contextual bandit with Thompson Sampling — learns from whether stored memories get retrieved later)
2. **Embeds** the message (sentence-transformers, falls back to hash)
3. **Evolution** adapts the embedding through learned LoRA-style correction
4. **Stores** the episode (skips filler: "thanks", "ok", "what time?")
5. **Updates KB** with structured facts extracted from text
6. **Detects conflicts** — "switched from X to Y" supersedes old X memories
7. **Causal engine** extracts cause/effect and adds to a persistent DAG
8. **Auditor** seals the episode into a SHA-256 hash chain
9. **Prefetcher** learns topic/entity transition patterns and warms cache for predicted next query
10. **Dream cycle** auto-triggers every 200 episodes (NREM replay, REM synthesis, abstraction, pruning)

### Retrieval Path (prepare)

1. **Prefetcher** checks warm cache before any retrieval
2. **Recalls** relevant memories + KB facts in parallel (30ms)
3. **Evolution** records retrieval feedback (query-result relevance pairs)
4. **Causal engine** answers "why" questions via graph traversal, "what if" via counterfactuals
5. **World simulator** includes experience-level context for temporal queries ("last week", "recently")
6. **Detects knowledge gaps** — knows what it doesn't know
7. **Suggests follow-ups** — predicted topics, unresolved decisions, missing context

---

## The 8 Advanced Modules

### 1. Memory Governor

Contextual bandit that learns what's worth remembering. Computes novelty, entity count, topic overlap, and storage pressure. Uses Thompson Sampling over Beta distributions. Cold-starts by absorbing everything, then progressively filters noise.

```python
brain.get_governor_policy()
# {total_decisions: 847, learned_preferences: [
#   {context_bin: (2, 0, 1), preferred_action: "ENCODE", confidence: 0.83},
#   {context_bin: (0, 2, 0), preferred_action: "SKIP", confidence: 0.71},  # low novelty, high similarity
# ]}
```

### 2. Causal Inference Engine

Builds a persistent networkx DAG from natural language. Supports forward/backward traversal, counterfactual queries, and contradiction detection.

```python
brain.process_turn("We switched to FastAPI because Django was too slow for websockets")
brain.process_turn("The speed improvement led to 50% fewer timeouts")

# Why did timeouts decrease?
brain.prepare("Why did timeouts decrease?")
# System prompt includes:
# ## Reasons (causal links)
# - [2.0] Django was too slow for websockets -> Switched to FastAPI
# - [1.0] The speed improvement -> 50% fewer timeouts

# What if we hadn't switched?
brain.counterfactual("Django was too slow")
# {removed: "Django was too slow", lost_effects: ["Switched to FastAPI", "50% fewer timeouts"]}

# Contradictions auto-detected
brain.get_causal_graph_stats()
# {nodes: 5, edges: 3, contradictions: 0, strongest_links: [...]}
```

### 3. World Simulator

Groups episodes into temporally coherent experiences. Supports replay, spatial maps, multimodal embeddings, and "what-if" simulation.

```python
# Auto-segments episodes into experiences
experiences = brain.segment_experiences()
# [{experience_id: "exp_a3f2", episode_ids: [...], locations: ["office", "home"], entities: ["alice"]}]

# Re-live an experience step by step
steps = brain.replay_experience("exp_a3f2")
# [{step: 0, text: "morning standup", location: "office", delta_seconds: 0, progress: 0.33}, ...]
```

### 4. Parametric Evolution

LoRA-style low-rank correction on embeddings. Learns from retrieval feedback (was the result relevant?). Every `prepare()` feeds training data. Every `dream()` triggers a weight update. Auto-rollback if quality degrades.

```python
brain.evolve()
# {updated: True, avg_loss: 0.12, quality: 0.88, grad_norm: 0.34}
```

### 5. Dream Consolidation

Periodic "sleep" that auto-triggers every 200 episodes. Four stages mimicking human sleep:

- **NREM**: Prioritized replay — strengthen important memories
- **REM**: Synthetic augmentation — SLERP-interpolated episode recombinations
- **Deep**: Abstraction — cluster similar episodes into principles and skills
- **Dawn**: Pruning — remove memories below activation threshold

```python
result = brain.dream()
# DreamResult(episodes_replayed=60, synthetic_generated=18, principles_extracted=3,
#             skills_learned=2, episodes_pruned=12, duration_seconds=0.003)
```

After dreaming: world simulator re-segments, evolution updates weights, causal engine decays stale links.

### 6. Predictive Prefetcher

Markov chain on topic transitions + entity co-occurrence + temporal patterns. Predicts what you'll ask next and pre-loads memories into a warm cache.

```python
# After several conversations about tech -> database -> infrastructure...
# When you mention "tech", the prefetcher already cached database and infra results.
# prepare() returns in 5ms instead of 60ms on a cache hit.
```

### 7. Memory Auditor

SHA-256 hash chain with Merkle tree. Every stored memory is sealed. Tamper detection. Full audit trail.

```python
brain.verify_integrity()
# {valid: True, checked: 847, broken_links: [], merkle_root: "a3f2...", tampered_episodes: []}

brain.get_audit_log(limit=5)
# [{episode_id: "ep_...", action: "create", timestamp: "...", actor: "system"}]
```

### 8. Cross-Agent Federation

Opt-in federated memory sharing with differential privacy. Agents share anonymized schemas (never raw episodes). Privacy budget tracking. GDPR revocation.

```python
from elo_memory import FederationClient, MemoryPool, DifferentialPrivacy

pool = MemoryPool(pool_path="./shared")
client = FederationClient("agent_1", pool, privacy_budget=10.0)

# Export a memory with calibrated noise
client.export_memory(embedding, category="tech")

# Import relevant shared knowledge from other agents
modules = client.import_relevant(query_embedding, k=5)

# Revoke all shared data (GDPR)
client.revoke_all()
```

---

## Direct Memory Access

```python
from elo_memory import UserMemory

memory = UserMemory("sarah", persistence_path="./memories")

# Store
result = memory.store("My email is sarah@shopify.com")
# {stored: True, entities: ["sarah@shopify.com", "Sarah"]}

# Recall
results = memory.recall("contact info?", k=7)
# [("My email is sarah@shopify.com", 0.47)]

# Profile
memory.get_profile()
# {user_id: "sarah", total_memories: 42, entities: {emails: [...], names: [...]}}

# Current facts only (superseded removed)
memory.get_facts()
```

---

## Performance

Benchmarked on Apple Silicon (M-series), 384-dim embeddings:

| Operation | Latency | Notes |
|---|---|---|
| Init | 129ms | All 16 components + load persisted state |
| First store | ~9s | One-time sentence-transformers model load |
| Store (warm) | **66ms** | Governor + embed + store + causal + audit + prefetch |
| Prepare (retrieval) | **62ms** | Cache check + parallel recall + causal + evolution |
| Dream cycle | 3ms | Per 10 episodes (scales with episode count) |
| Verify integrity | <1ms | Full hash chain verification |
| Close (persist) | 32ms | Save all modules to disk |

---

## vs Alternatives

| | Elo Memory | Mem0 | ChromaDB | Zep |
|--|-----------|------|----------|-----|
| Recall accuracy (24q test) | **100%** | N/A ($) | 83% | -- |
| Store latency | **66ms** | 500-2000ms | 30ms | ~100ms |
| Cost per 1000 ops | **$0** | $0.50-20 | $0 | $99/mo |
| Works offline | **Yes** | No | Yes | No |
| Needs API key | **No** | Yes | No | Yes |
| Causal reasoning | **Yes** | No | No | No |
| Learns what to store | **Yes** | No | No | No |
| Dream consolidation | **Yes** | No | No | No |
| Embedding evolution | **Yes** | No | No | No |
| Tamper-proof audit | **Yes** | No | No | No |
| Federated sharing | **Yes** | No | No | No |
| Counterfactual queries | **Yes** | No | No | No |
| Knowledge gaps | **Yes** | No | No | No |
| Conflict detection | **Yes** | Yes (LLM) | No | No |

---

## Architecture

```
User message
  |
  +---> Governor (should I store this?)
  |       |
  |       +---> Evolution (adapt embedding)
  |       |
  |       +---> Episodic Store (ChromaDB + disk offload)
  |       |       |
  |       |       +---> Auditor (seal in hash chain)
  |       |
  |       +---> Knowledge Base (structured facts)
  |       |
  |       +---> Causal Engine (build DAG)
  |       |
  |       +---> Prefetcher (learn patterns, warm cache)
  |
  +---> Every 200 episodes:
          |
          +---> Dream Cycle (replay, synthesize, abstract, prune)
          +---> World Simulator (re-segment experiences)
          +---> Evolution (update weights)
          +---> Causal Engine (decay stale links)
```

---

## API Levels

| Level | Entry Point | For |
|-------|-------------|-----|
| **Beginner** | `EloBrain` | 3 lines, everything automatic |
| **Intermediate** | `UserMemory` | Per-user isolation, sessions, profiles |
| **Advanced** | Individual modules | Governor, CausalEngine, WorldSimulator, Evolution, Auditor, Prefetcher, DreamConsolidation, Federation |
| **Expert** | Core engines | BayesianSurprise, EventSegmenter, ForgettingEngine, InterferenceResolver, OnlineLearner |

---

## Installation

```bash
# Recommended
pip install elo-memory

# From source
git clone https://github.com/server-elo/elo-memory.git
cd elo-memory && pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
# 434 tests
```

---

## Documentation

- [Usage Guide](USAGE_GUIDE.md) -- Integration patterns
- [MCP Setup](MCP_SETUP.md) -- Claude Code integration
- [Changelog](CHANGELOG.md) -- What's new

## License

BSL 1.1 (Business Source License) -- free for production use, except offering it as a competing hosted memory service. Converts to Apache 2.0 on 2029-04-03. See [LICENSE](LICENSE).

## Acknowledgments

- EM-LLM (ICLR 2025) -- Research foundation
- Itti & Baldi (2009) -- Bayesian Surprise
- Squire & Alvarez (1995) -- Systems Consolidation
- Pearl (2009) -- Causality
- Hu et al. (2022) -- LoRA
- Merkle (1988) -- Hash Trees

---

**GitHub**: https://github.com/server-elo/elo-memory
**PyPI**: https://pypi.org/project/elo-memory/
