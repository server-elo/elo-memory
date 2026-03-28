#!/usr/bin/env python3
"""
Elo-Memory Retrieval Benchmarks
================================

Measures actual performance of core operations:
- Episode storage throughput
- Similarity retrieval latency (ChromaDB-backed)
- Two-stage retrieval latency (similarity + temporal expansion)
- Consolidation cycle time
- Memory scaling behavior

Run:  python benchmarks/bench_retrieval.py
"""

import sys
import time
import statistics
import numpy as np
from datetime import datetime, timedelta

from elo_memory import (
    BayesianSurpriseEngine,
    EpisodicMemoryStore,
    EpisodicMemoryConfig,
    TwoStageRetriever,
    RetrievalConfig,
    MemoryConsolidationEngine,
)


def _rand_emb(dim):
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def bench(name, fn, warmup=3, iterations=50):
    """Run a benchmark and print results."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times) * 1000
    p95 = sorted(times)[int(len(times) * 0.95)] * 1000
    mean = statistics.mean(times) * 1000
    print(f"  {name:45s}  median={med:8.2f}ms  p95={p95:8.2f}ms  mean={mean:8.2f}ms")
    return med


def main():
    import tempfile
    tmpdir = tempfile.mkdtemp()
    DIM = 128
    np.random.seed(42)

    print("=" * 80)
    print("Elo-Memory Benchmarks")
    print("=" * 80)
    print()

    # -----------------------------------------------------------------------
    # Setup: populate store with N episodes
    # -----------------------------------------------------------------------
    sizes = [100, 500, 1000, 5000]
    max_n = max(sizes)

    config = EpisodicMemoryConfig(
        max_episodes=max_n + 1000,
        embedding_dim=DIM,
        persistence_path=f"{tmpdir}/bench_store",
        consolidation_min_episodes=max_n + 1000,  # prevent auto-consolidation
    )
    store = EpisodicMemoryStore(config)
    surprise_engine = BayesianSurpriseEngine(DIM)

    # Pre-generate data
    embeddings = [_rand_emb(DIM) for _ in range(max_n)]
    contents = [np.random.randn(10).astype(np.float32) for _ in range(max_n)]
    base_time = datetime.now() - timedelta(hours=max_n)

    # -----------------------------------------------------------------------
    # Benchmark 1: Storage throughput
    # -----------------------------------------------------------------------
    print("1. STORAGE THROUGHPUT")
    print("-" * 80)

    t0 = time.perf_counter()
    for i in range(max_n):
        store.store_episode(
            content=contents[i],
            embedding=embeddings[i],
            surprise=float(np.random.rand()),
            timestamp=base_time + timedelta(minutes=i),
            location=["office", "home", "cafe"][i % 3],
            entities=[f"person_{i % 10}"],
        )
    elapsed = time.perf_counter() - t0
    print(f"  Stored {max_n} episodes in {elapsed:.2f}s ({max_n / elapsed:.0f} eps/sec)")
    print()

    # -----------------------------------------------------------------------
    # Benchmark 2: Raw similarity retrieval (ChromaDB)
    # -----------------------------------------------------------------------
    print("2. SIMILARITY RETRIEVAL (ChromaDB)")
    print("-" * 80)

    for k in [5, 10, 20]:
        query = _rand_emb(DIM)
        bench(
            f"retrieve_by_similarity(k={k}, n={max_n})",
            lambda k=k, q=query: store.retrieve_by_similarity(q, k=k),
        )
    print()

    # -----------------------------------------------------------------------
    # Benchmark 3: Two-stage retrieval
    # -----------------------------------------------------------------------
    print("3. TWO-STAGE RETRIEVAL (similarity + temporal)")
    print("-" * 80)

    retriever = TwoStageRetriever(store, RetrievalConfig(max_retrieved=10))
    query = _rand_emb(DIM)
    bench(
        f"two_stage.retrieve(n={max_n}, expansion=ON)",
        lambda: retriever.retrieve(query=query),
    )

    retriever_no_expand = TwoStageRetriever(
        store, RetrievalConfig(max_retrieved=10, enable_temporal_expansion=False)
    )
    bench(
        f"two_stage.retrieve(n={max_n}, expansion=OFF)",
        lambda: retriever_no_expand.retrieve(query=query),
    )
    print()

    # -----------------------------------------------------------------------
    # Benchmark 4: Index lookups
    # -----------------------------------------------------------------------
    print("4. INDEX LOOKUPS")
    print("-" * 80)

    bench(
        f"retrieve_by_location('office', n={max_n})",
        lambda: store.retrieve_by_location("office"),
        iterations=200,
    )
    bench(
        f"retrieve_by_entity('person_0', n={max_n})",
        lambda: store.retrieve_by_entity("person_0"),
        iterations=200,
    )
    bench(
        f"retrieve_by_temporal_range(1h window, n={max_n})",
        lambda: store.retrieve_by_temporal_range(
            datetime.now() - timedelta(hours=1), datetime.now()
        ),
        iterations=200,
    )
    print()

    # -----------------------------------------------------------------------
    # Benchmark 5: Surprise computation
    # -----------------------------------------------------------------------
    print("5. SURPRISE COMPUTATION")
    print("-" * 80)

    engine = BayesianSurpriseEngine(DIM)
    obs = _rand_emb(DIM)
    bench(
        f"compute_surprise(dim={DIM})",
        lambda: engine.compute_surprise(obs),
        iterations=200,
    )
    print()

    # -----------------------------------------------------------------------
    # Benchmark 6: Consolidation
    # -----------------------------------------------------------------------
    print("6. CONSOLIDATION")
    print("-" * 80)

    consolidation = MemoryConsolidationEngine()
    for n in [100, 500]:
        episodes = store.episodes[:n]
        bench(
            f"consolidate({n} episodes)",
            lambda eps=episodes: consolidation.consolidate(eps),
            warmup=1,
            iterations=5,
        )
    print()

    # -----------------------------------------------------------------------
    # Benchmark 7: Persistence
    # -----------------------------------------------------------------------
    print("7. PERSISTENCE (save/load)")
    print("-" * 80)

    bench(
        f"save_state({max_n} episodes)",
        lambda: store.save_state(),
        warmup=1,
        iterations=10,
    )

    config2 = EpisodicMemoryConfig(
        max_episodes=max_n + 1000,
        embedding_dim=DIM,
        persistence_path=f"{tmpdir}/bench_store",
        consolidation_min_episodes=max_n + 1000,
    )

    def load_bench():
        s = EpisodicMemoryStore(config2)
        s.load_state()

    bench(
        f"load_state({max_n} episodes)",
        load_bench,
        warmup=1,
        iterations=5,
    )
    print()

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
