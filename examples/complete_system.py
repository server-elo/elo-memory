"""
Complete Neuro-Memory System Demo
==================================

Full system integration with all components:
1. Bayesian Surprise Detection
2. Event Segmentation
3. Episodic Storage
4. Two-Stage Retrieval
5. Memory Consolidation
6. Forgetting & Decay
7. Interference Resolution
8. Online Continual Learning

This is the COMPLETE bio-inspired episodic memory system.

Install first: pip install -e ".[dev]" from repo root
"""

import numpy as np
from datetime import datetime, timedelta

# All components - uses installed package
from elo_memory import (
    BayesianSurpriseEngine, SurpriseConfig,
    EventSegmenter, SegmentationConfig,
    EpisodicMemoryStore, EpisodicMemoryConfig,
    TwoStageRetriever, RetrievalConfig,
    MemoryConsolidationEngine, ConsolidationConfig,
    ForgettingEngine, ForgettingConfig,
    InterferenceResolver, InterferenceConfig,
    OnlineLearner, OnlineLearningConfig
)


def main():
    print("=" * 80)
    print(" COMPLETE NEURO-MEMORY SYSTEM - Full Integration Test")
    print("=" * 80)
    print()

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================
    print("🔧 Initializing all components...")

    # 1. Surprise Detection
    surprise_engine = BayesianSurpriseEngine(
        input_dim=128,
        config=SurpriseConfig(window_size=30, surprise_threshold=0.7)
    )

    # 2. Event Segmentation
    segmenter = EventSegmenter(
        config=SegmentationConfig(
            hmm_states=3,
            boundary_threshold=0.6,
            min_event_length=5
        )
    )

    # 3. Episodic Storage
    memory_store = EpisodicMemoryStore(
        config=EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=128
        )
    )

    # 4. Two-Stage Retrieval
    retriever = TwoStageRetriever(
        memory_store=memory_store,
        config=RetrievalConfig(
            k_candidates=20,
            k_final=5,
            similarity_weight=0.7,
            temporal_weight=0.3
        )
    )

    # 5. Memory Consolidation
    consolidation = MemoryConsolidationEngine(
        config=ConsolidationConfig(
            replay_count=100,
            schema_threshold=0.8
        )
    )

    # 6. Forgetting Engine
    forgetting = ForgettingEngine(
        config=ForgettingConfig(
            decay_rate=0.01,
            consolidation_boost=2.0
        )
    )

    # 7. Interference Resolution
    interference = InterferenceResolver(
        config=InterferenceConfig(
            separation_threshold=0.3,
            completion_threshold=0.7
        )
    )

    # 8. Online Learning
    online_learner = OnlineLearner(
        config=OnlineLearningConfig(
            learning_rate=0.001,
            replay_buffer_size=1000
        )
    )

    print("✅ All 8 components initialized")
    print()

    # ==========================================================================
    # SIMULATE INCOMING OBSERVATIONS
    # ==========================================================================
    print("📊 Simulating observation stream...")

    np.random.seed(42)
    n_observations = 500

    # Generate synthetic embeddings with event structure
    observations = []
    current_context = 0

    for i in range(n_observations):
        # Change context every ~50 observations (simulate events)
        if i % 50 == 0:
            current_context = (current_context + 1) % 5

        # Generate embedding based on context + noise
        base = np.random.randn(128) * 0.3
        base[current_context * 25:(current_context + 1) * 25] += 1.0
        embedding = base / np.linalg.norm(base)

        observations.append({
            'id': i,
            'embedding': embedding,
            'context': current_context,
            'timestamp': datetime.now() + timedelta(seconds=i)
        })

    print(f"✅ Generated {n_observations} observations across 5 contexts")
    print()

    # ==========================================================================
    # PROCESS OBSERVATIONS
    # ==========================================================================
    print("🧠 Processing observations through memory pipeline...")

    stored_episodes = 0
    event_boundaries = []

    for obs in observations:
        # Step 1: Compute surprise
        surprise_info = surprise_engine.compute_surprise(obs['embedding'])

        # Step 2: Detect event boundaries
        is_boundary = segmenter.detect_boundary(
            embedding=obs['embedding'],
            surprise=surprise_info['surprise']
        )

        if is_boundary:
            event_boundaries.append(obs['id'])

        # Step 3: Store if novel
        if surprise_info['is_novel']:
            episode_id = memory_store.store_episode(
                content={
                    'obs_id': obs['id'],
                    'context': obs['context'],
                    'surprise': surprise_info['surprise']
                },
                embedding=obs['embedding'],
                surprise=surprise_info['surprise'],
                timestamp=obs['timestamp']
            )
            stored_episodes += 1

        # Step 4: Online learning update
        online_learner.observe(obs['embedding'], surprise_info['surprise'])

    print(f"✅ Stored {stored_episodes} episodes ({stored_episodes/n_observations*100:.1f}%)")
    print(f"✅ Detected {len(event_boundaries)} event boundaries")
    print(f"✅ Mean surprise: {surprise_engine.mean_surprise:.4f}")
    print()

    # ==========================================================================
    # MEMORY RETRIEVAL
    # ==========================================================================
    print("🔍 Testing memory retrieval...")

    # Query from context 2
    query_context = 2
    query_base = np.random.randn(128) * 0.3
    query_base[query_context * 25:(query_context + 1) * 25] += 1.0
    query_embedding = query_base / np.linalg.norm(query_base)

    results = retriever.retrieve(
        query_embedding=query_embedding,
        k=5
    )

    print(f"✅ Retrieved {len(results)} memories for context {query_context}")
    for i, result in enumerate(results[:3], 1):
        content = result.get('content', {})
        print(f"   {i}. Episode from context {content.get('context', '?')}, "
              f"surprise={result.get('surprise', 0):.3f}")
    print()

    # ==========================================================================
    # MEMORY CONSOLIDATION
    # ==========================================================================
    print("💤 Running memory consolidation (sleep phase)...")

    consolidation_stats = consolidation.consolidate(memory_store.episodes)
    schemas = consolidation.get_schema_summary()

    print(f"✅ Replayed {consolidation_stats['replay_count']} episodes")
    print(f"✅ Extracted {len(schemas)} schemas")
    if schemas:
        print(f"   Top schema: {schemas[0]['description'][:60]}...")
    print()

    # ==========================================================================
    # FORGETTING & INTERFERENCE
    # ==========================================================================
    print("🧹 Applying forgetting and interference resolution...")

    # Apply forgetting to old episodes
    forgotten = forgetting.apply_decay(memory_store.episodes)

    # Check for interference
    interference_patterns = interference.detect_interference(memory_store.episodes)

    print(f"✅ Applied forgetting decay to {len(forgotten)} episodes")
    print(f"✅ Detected {len(interference_patterns)} interference patterns")
    print()

    #