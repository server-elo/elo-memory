# Elo Memory

[![PyPI version](https://badge.fury.io/py/elo-memory.svg)](https://badge.fury.io/py/elo-memory)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/server-elo/elo-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/server-elo/elo-memory/actions)

**The memory brain for AI agents.** 3 lines to give any LLM persistent memory that knows what to remember, what to forget, what changed, and what's missing.

```python
from elo_memory import EloBrain

brain = EloBrain("user_123")
response = brain.think("I switched from Django to FastAPI", your_llm)
# Brain auto: recalled context, stored the fact, updated KB to "backend: FastAPI",
# superseded old Django memory, extracted causal link if reason given,
# detected knowledge gaps, suggested follow-up questions
```

---

## Why Elo Memory?

Every other memory system stores text and searches it. Elo Memory **understands** it.

```
"Switched from Django to FastAPI because it was too slow"
```

| System | What it does |
|--------|-------------|
| **ChromaDB** | Stores the text. Retrieves it when query is similar. Returns old Django AND new FastAPI results with no way to know which is current. |
| **Mem0** | Calls GPT to extract facts ($0.005/store). Stores "backend: FastAPI". Slow (500ms), needs API key, sends your data to OpenAI. |
| **Elo Memory** | Stores the episode. Updates KB to `backend: FastAPI`. Supersedes Django memory. Extracts causal link: "too slow → switched to FastAPI". Detects knowledge gaps. All locally, in 30ms, for $0. |

---

## Quick Start

```bash
pip install "elo-memory[ml]"
```

### For AI agents (recommended)

```python
from elo_memory import EloBrain

brain = EloBrain("sarah")

# Full agent loop: recall → prompt → LLM → store
response = brain.think(
    "I'm Sarah, senior engineer at Shopify. We use PostgreSQL.",
    llm_fn=lambda prompt: your_llm(prompt),
)

# What does the brain know?
state = brain.what_i_know()
# {
#   knowledge: {name: "Sarah", role: "senior engineer", company: "Shopify", database: "PostgreSQL"},
#   knowledge_gaps: [{topic: "infrastructure", missing: ["hosting", "ci/cd", "monitoring"]}],
#   suggestions: ["Consider asking about: hosting, ci/cd, monitoring"],
#   facts: [...], entities: {...}, causal_links: 0, decisions_tracked: 0
# }
```

### What happens automatically on every turn

1. **Recalls** relevant memories + KB facts (30ms)
2. **Stores** the message (skips filler: "thanks", "ok", "what time?")
3. **Updates KB** with structured facts extracted from text
4. **Detects conflicts** — "switched from X to Y" supersedes old X memories
5. **Extracts entities** — names, emails, dates, amounts
6. **Tracks causality** — "because X" links cause to effect
7. **Detects knowledge gaps** — knows what it DOESN'T know
8. **Suggests follow-ups** — unresolved decisions, missing context
9. **Generates derived facts** — "Switched to FastAPI" → also indexes "Currently using FastAPI"
10. **Filters noise** — near-duplicates (>0.92 cosine) silently skipped

### Direct memory access

```python
from elo_memory import UserMemory

memory = UserMemory("sarah", persistence_path="./memories")

# Store
result = memory.store("My email is sarah@shopify.com")
# {stored: True, entities: {emails: ["sarah@shopify.com"], names: ["Sarah"]}}

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

## What No Competitor Has

### 1. Knowledge Gap Detection

```python
brain.think("I'm building a payment system with Stripe", llm)
state = brain.what_i_know()
state["knowledge_gaps"]
# [{topic: "payment", missing: ["currency", "volume", "compliance", "region"],
#   suggestion: "Consider asking about: currency, volume, compliance, region"}]
```

The brain knows what it **doesn't** know and tells the agent to ask.

### 2. Causal Reasoning

```python
brain.think("Switched to FastAPI because Django was too slow for websockets", llm)
brain.prepare("Why did we change the backend?")
# System prompt includes:
# ## Reasons (causal links)
# - Django was too slow for websockets → Switched to FastAPI
```

### 3. Conflict Resolution (no LLM needed)

```python
brain.think("I drive a BMW", llm)
brain.think("Just picked up my new Tesla yesterday", llm)
# BMW memory automatically superseded. "What car?" → Tesla only.
```

Works for explicit ("switched from X to Y") and implicit ("got a new X") contradictions.

### 4. Structured Knowledge Base

Every message automatically updates a structured KB. Queries hit the KB first (0ms), episodes as fallback.

```python
brain._kb.get_all()
# {name: "Sarah", role: "senior engineer", company: "Shopify",
#  backend: "FastAPI", database: "PostgreSQL", team_size: "12"}
```

---

## vs Alternatives

| | Elo Memory | Mem0 | ChromaDB | Zep |
|--|-----------|------|----------|-----|
| Recall accuracy (24q test) | **100%** | N/A ($) | 83% | — |
| Store latency | **30ms** | 500-2000ms | 30ms | ~100ms |
| Cost per 1000 ops | **$0** | $0.50-20 | $0 | $99/mo |
| Works offline | **Yes** | No | Yes | No |
| Needs API key | **No** | Yes | No | Yes |
| Structured KB | **Yes** | Yes (LLM) | No | No |
| Conflict detection | **Yes** | Yes (LLM) | No | No |
| Knowledge gaps | **Yes** | No | No | No |
| Causal reasoning | **Yes** | No | No | No |
| Decision tracking | **Yes** | No | No | No |

---

## Installation

```bash
# Recommended (includes sentence-transformers)
pip install "elo-memory[ml]"

# Minimal (provide your own embeddings)
pip install elo-memory

# From source
git clone https://github.com/server-elo/elo-memory.git
cd elo-memory && pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Architecture

```
User message
  │
  ├─→ Knowledge Base (instant structured facts)
  │     "backend: FastAPI", "team_size: 12"
  │
  ├─→ Episodic Store (narrative context)
  │     ChromaDB + temporal expansion
  │
  ├─→ Intelligence Layer
  │     Knowledge gaps, causal chains, decision tracking
  │
  ├─→ Conflict Detection
  │     Supersede old facts, implicit contradictions
  │
  └─→ Entity Extraction
        Names, emails, dates, amounts (regex, no LLM)
```

---

## API Levels

| Level | Entry Point | For |
|-------|-------------|-----|
| **Newbie** | `EloBrain` | 3 lines, automatic everything |
| **Intermediate** | `UserMemory` | Per-user isolation, sessions, profiles |
| **Advanced** | `EpisodicMemoryStore` | Full config control, raw components |
| **Expert** | Individual engines | Surprise, segmentation, forgetting, consolidation |

---

## Documentation

- [Usage Guide](USAGE_GUIDE.md) — Integration patterns
- [MCP Setup](MCP_SETUP.md) — Claude Code integration
- [Changelog](CHANGELOG.md) — What's new

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- EM-LLM (ICLR 2025) — Research foundation
- Itti & Baldi (2009) — Bayesian Surprise
- Squire & Alvarez (1995) — Systems Consolidation

---

**GitHub**: https://github.com/server-elo/elo-memory
**PyPI**: https://pypi.org/project/elo-memory/
