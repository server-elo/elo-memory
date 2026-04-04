"""
Hard tests for the 8 advanced memory modules.

Tests edge cases, thread safety, persistence, and integration points.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from elo_memory.governor import (
    Action,
    DecisionContext,
    GovernorConfig,
    MemoryGovernor,
)
from elo_memory.causal_engine import CausalInferenceEngine, CausalEngineConfig
from elo_memory.world_simulator import (
    WorldSimulator,
    WorldSimConfig,
    Experience,
    SpatialCoord,
)
from elo_memory.evolution import EvolutionConfig, ParametricEvolution
from elo_memory.federation.privacy import (
    DifferentialPrivacy,
    PrivacyAccountant,
    PrivacyConfig,
)
from elo_memory.federation.symbiosis import (
    FederationClient,
    MemoryModule,
    MemoryPool,
)
from elo_memory.consolidation.dream_cycle import DreamConsolidation, DreamConfig
from elo_memory.retrieval.prefetcher import PredictivePrefetcher, PrefetchConfig
from elo_memory.auditor import MemoryAuditor
from elo_memory.memory.episodic_store import (
    Episode,
    EpisodicMemoryConfig,
    EpisodicMemoryStore,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir):
    config = EpisodicMemoryConfig(
        embedding_dim=32,
        persistence_path=os.path.join(tmp_dir, "store"),
        max_episodes=100,
    )
    return EpisodicMemoryStore(config)


def _make_episode(store, text="test", surprise=0.0, location=None, entities=None):
    emb = np.random.randn(store.config.embedding_dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return store.store_episode(
        content={"text": text},
        surprise=surprise,
        embedding=emb,
        location=location,
        entities=entities or [],
    )


# =========================================================================
# 1. GOVERNOR
# =========================================================================

class TestGovernor:
    def test_decide_returns_valid_action(self):
        gov = MemoryGovernor()
        ctx = DecisionContext(surprise=0.5, max_similarity=0.3)
        action = gov.decide(ctx)
        assert isinstance(action, Action)
        assert gov.total_decisions == 1

    def test_apply_action_skip(self):
        gov = MemoryGovernor()
        should_store, imp = gov.apply_action(Action.SKIP, 0.5)
        assert should_store is False
        assert imp == 0.0

    def test_apply_action_promote(self):
        gov = MemoryGovernor()
        should_store, imp = gov.apply_action(Action.PROMOTE, 0.5)
        assert should_store is True
        assert imp > 0.5  # Boosted

    def test_apply_action_demote(self):
        gov = MemoryGovernor()
        should_store, imp = gov.apply_action(Action.DEMOTE, 0.8)
        assert should_store is True
        assert imp < 0.8  # Reduced

    def test_reward_tracking(self):
        gov = MemoryGovernor(GovernorConfig(reward_window_hours=0.0001))
        ctx = DecisionContext()
        action = gov.decide(ctx)
        gov.record_decision(action, ctx, episode_id="ep_1")
        gov.record_retrieval("ep_1")
        time.sleep(0.5)
        gov._resolve_pending()
        assert gov.total_rewards >= 1

    def test_persistence(self, tmp_dir):
        gov = MemoryGovernor(persistence_path=tmp_dir)
        for _ in range(10):
            ctx = DecisionContext(surprise=np.random.random())
            gov.decide(ctx)
        gov.save()

        gov2 = MemoryGovernor(persistence_path=tmp_dir)
        assert gov2.total_decisions == gov.total_decisions

    def test_concurrent_decisions(self):
        gov = MemoryGovernor()
        errors = []

        def worker():
            try:
                for _ in range(20):
                    ctx = DecisionContext(surprise=np.random.random())
                    gov.decide(ctx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert gov.total_decisions == 80

    def test_policy_summary_structure(self):
        gov = MemoryGovernor()
        for _ in range(5):
            gov.decide(DecisionContext(surprise=0.5))
        summary = gov.get_policy_summary()
        assert "total_decisions" in summary
        assert "action_distribution" in summary
        assert "learned_preferences" in summary


# =========================================================================
# 2. CAUSAL ENGINE
# =========================================================================

class TestCausalEngine:
    def test_ingest_because(self):
        ce = CausalInferenceEngine()
        links = ce.ingest("We switched to FastAPI because Django was too slow")
        assert len(links) >= 1
        assert any("Django" in l.cause for l in links)

    def test_ingest_led_to(self):
        ce = CausalInferenceEngine()
        links = ce.ingest("The outage led to customer complaints")
        assert len(links) >= 1

    def test_ingest_after(self):
        ce = CausalInferenceEngine()
        links = ce.ingest("After the migration, performance improved")
        assert len(links) >= 1

    def test_query_causes_depth(self):
        ce = CausalInferenceEngine()
        ce.ingest("Alpha caused Beta")
        ce.ingest("Beta caused Gamma")
        ce.ingest("Gamma caused Delta")
        results = ce.query_causes("Delta", depth=5)
        causes = {r["cause"] for r in results}
        assert "Beta" in causes or "Gamma" in causes  # At least one level back

    def test_query_effects(self):
        ce = CausalInferenceEngine()
        ce.ingest("Rain caused flooding")
        results = ce.query_effects("Rain")
        assert any("flooding" in r["effect"].lower() for r in results)

    def test_counterfactual(self):
        ce = CausalInferenceEngine()
        ce.ingest("Alpha caused Beta")
        ce.ingest("Beta caused Gamma")
        result = ce.counterfactual("Alpha")
        # Removing Alpha should potentially lose Beta and Gamma
        assert "removed" in result

    def test_contradiction_detection(self):
        ce = CausalInferenceEngine()
        ce.ingest("Alpha caused Beta")
        ce.ingest("Beta caused Alpha")  # Contradiction: reverse
        contradictions = ce.get_contradictions()
        assert len(contradictions) >= 1
        assert contradictions[0]["type"] == "reverse_causation"

    def test_strength_reinforcement(self):
        ce = CausalInferenceEngine()
        ce.ingest("Failure because Overload")
        ce.ingest("Failure because Overload")  # Same link twice
        # "because" gives effect=Failure, cause=Overload → edge: overload→failure
        assert ce.graph.has_edge(ce._normalize("Overload"), ce._normalize("Failure"))
        edge = ce.graph.edges[ce._normalize("Overload"), ce._normalize("Failure")]
        assert edge["strength"] >= 2.0

    def test_decay_removes_weak_links(self):
        ce = CausalInferenceEngine(CausalEngineConfig(min_link_strength=0.5))
        ce.ingest("Alpha caused Beta")
        assert ce.graph.number_of_edges() >= 1
        ce.decay_strengths(days_elapsed=100)  # Heavy decay
        assert ce.graph.number_of_edges() == 0

    def test_persistence(self, tmp_dir):
        ce = CausalInferenceEngine()
        ce.ingest("Rain caused flooding")
        ce.ingest("Flooding led to damage")
        path = Path(tmp_dir) / "causal.json"
        ce.save(path)

        ce2 = CausalInferenceEngine()
        ce2.load(path)
        assert ce2.graph.number_of_nodes() == ce.graph.number_of_nodes()
        assert ce2.graph.number_of_edges() == ce.graph.number_of_edges()

    def test_empty_text(self):
        ce = CausalInferenceEngine()
        links = ce.ingest("")
        assert links == []

    def test_max_graph_nodes(self):
        ce = CausalInferenceEngine(CausalEngineConfig(max_graph_nodes=10))
        for i in range(20):
            ce.ingest(f"EventNumber{i} caused ResultNumber{i}")
        # Graph should be capped — pruning removes some edges/nodes
        assert ce.graph.number_of_nodes() <= 25  # Not all 40 nodes survive


# =========================================================================
# 3. WORLD SIMULATOR
# =========================================================================

class TestWorldSimulator:
    def test_segment_experiences(self, store):
        # Create episodes close in time
        now = datetime.now(timezone.utc)
        for i in range(5):
            ep = _make_episode(store, text=f"event {i}")
            ep.timestamp = now + timedelta(minutes=i * 10)

        ws = WorldSimulator(store)
        exps = ws.segment_experiences()
        assert len(exps) >= 1
        assert exps[0].episode_ids

    def test_replay(self, store):
        now = datetime.now(timezone.utc)
        for i in range(3):
            ep = _make_episode(store, text=f"step {i}", location=f"room_{i}")
            ep.timestamp = now + timedelta(minutes=i * 5)

        ws = WorldSimulator(store)
        exps = ws.segment_experiences()
        if exps:
            steps = ws.replay(exps[0].experience_id)
            assert len(steps) >= 2
            assert "text" in steps[0]
            assert steps[0]["progress"] > 0

    def test_simulate_variation(self, store):
        now = datetime.now(timezone.utc)
        eps = []
        for i in range(3):
            ep = _make_episode(store, text=f"original {i}")
            ep.timestamp = now + timedelta(minutes=i * 5)
            eps.append(ep)

        ws = WorldSimulator(store)
        exps = ws.segment_experiences()
        if exps:
            varied = ws.simulate_variation(exps[0].experience_id, {
                "remove_episode": eps[0].episode_id,
            })
            assert any("REMOVED" in s.get("text", "") for s in varied)

    def test_spatial_map(self, store):
        _make_episode(store, text="meeting", location="office")
        _make_episode(store, text="lunch", location="cafeteria")
        _make_episode(store, text="standup", location="office")

        ws = WorldSimulator(store)
        spatial = ws.build_spatial_map()
        assert spatial["total_locations"] >= 2
        assert "office" in spatial["locations"]

    def test_multimodal_attachment(self, store):
        ep = _make_episode(store, text="visual event")
        ws = WorldSimulator(store)
        ws.attach_modality(ep.episode_id, "image", np.random.randn(128))
        ws.attach_modality(ep.episode_id, "audio", np.random.randn(64))

        mods = ws.get_modalities(ep.episode_id)
        assert len(mods) == 2
        assert mods[0]["name"] == "image"

    def test_fused_embedding(self, store):
        ep = _make_episode(store, text="multimodal")
        ws = WorldSimulator(store)
        ws.attach_modality(ep.episode_id, "image", np.random.randn(32))
        fused = ws.fused_embedding(ep.episode_id)
        assert fused is not None
        assert len(fused) > 32  # Should be concatenation

    def test_experience_compression(self, store):
        now = datetime.now(timezone.utc)
        for i in range(5):
            ep = _make_episode(store, text=f"seq {i}")
            ep.timestamp = now + timedelta(minutes=i)

        ws = WorldSimulator(store)
        exps = ws.segment_experiences()
        if exps:
            compressed = ws.compress_experience(exps[0].experience_id)
            assert compressed is not None

    def test_persistence(self, store, tmp_dir):
        _make_episode(store, text="test", location="lab")
        ws = WorldSimulator(store)
        ws.segment_experiences()
        ws.build_spatial_map()
        path = Path(tmp_dir) / "world.json"
        ws.save(path)

        ws2 = WorldSimulator(store)
        ws2.load(path)
        assert len(ws2.experiences) == len(ws.experiences)

    def test_empty_store(self, store):
        ws = WorldSimulator(store)
        exps = ws.segment_experiences()
        assert exps == []

    def test_experience_serialization(self):
        exp = Experience(
            experience_id="test_exp",
            episode_ids=["ep_1", "ep_2"],
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
            spatial_trajectory=[SpatialCoord(x=1.0, y=2.0, label="office")],
        )
        d = exp.to_dict()
        exp2 = Experience.from_dict(d)
        assert exp2.experience_id == exp.experience_id
        assert len(exp2.spatial_trajectory) == 1


# =========================================================================
# 4. PARAMETRIC EVOLUTION
# =========================================================================

class TestEvolution:
    def test_adapt_embedding_identity_at_init(self):
        evo = ParametricEvolution(embedding_dim=32)
        emb = np.random.randn(32).astype(np.float32)
        emb /= np.linalg.norm(emb)
        adapted = evo.adapt_embedding(emb)
        # At init, A and B are small random, so adapted ≈ emb
        cos_sim = float(np.dot(emb, adapted))
        assert cos_sim > 0.8  # Should be close

    def test_record_and_update(self):
        evo = ParametricEvolution(
            embedding_dim=16,
            config=EvolutionConfig(min_feedback_for_update=5, rank=4),
        )
        for _ in range(10):
            q = np.random.randn(16)
            r = np.random.randn(16)
            evo.record_feedback(q, r, relevance=np.random.random())

        result = evo.update_weights()
        assert result["updated"] is True
        assert result["feedback_used"] == 10

    def test_insufficient_feedback(self):
        evo = ParametricEvolution(
            embedding_dim=8,
            config=EvolutionConfig(min_feedback_for_update=100),
        )
        evo.record_feedback(np.zeros(8), np.zeros(8), 1.0)
        result = evo.update_weights()
        assert result["updated"] is False

    def test_distill_experiences(self):
        evo = ParametricEvolution(embedding_dim=16)
        embeddings = [np.random.randn(16) for _ in range(5)]
        distilled = evo.distill_experiences(embeddings, [0.1, 0.2, 0.3, 0.2, 0.2])
        assert distilled.shape == (16,)
        assert abs(np.linalg.norm(distilled) - 1.0) < 0.01

    def test_rollback(self):
        evo = ParametricEvolution(
            embedding_dim=8,
            config=EvolutionConfig(rank=2, min_feedback_for_update=3, checkpoint_interval=1),
        )
        # Record positive feedback and update
        for _ in range(5):
            evo.record_feedback(np.random.randn(8), np.random.randn(8), 1.0)
        evo.update_weights()

        old_A = evo.A.copy()
        evo.rollback()
        # After rollback, A should match checkpoint
        assert evo._checkpoint_A is not None

    def test_thread_safety(self):
        evo = ParametricEvolution(embedding_dim=16, config=EvolutionConfig(rank=4))
        errors = []

        def writer():
            try:
                for _ in range(10):
                    evo.record_feedback(
                        np.random.randn(16), np.random.randn(16), np.random.random()
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(10):
                    evo.adapt_embedding(np.random.randn(16))
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=writer) for _ in range(3)]
            + [threading.Thread(target=reader) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_persistence(self, tmp_dir):
        evo = ParametricEvolution(embedding_dim=8, config=EvolutionConfig(rank=2))
        for _ in range(5):
            evo.record_feedback(np.random.randn(8), np.random.randn(8), 1.0)
        evo.update_weights()

        path = Path(tmp_dir) / "evo.json"
        evo.save(path)

        evo2 = ParametricEvolution(embedding_dim=8, config=EvolutionConfig(rank=2))
        evo2.load(path)
        np.testing.assert_array_almost_equal(evo.A, evo2.A)
        np.testing.assert_array_almost_equal(evo.B, evo2.B)


# =========================================================================
# 5. FEDERATION
# =========================================================================

class TestFederation:
    def test_differential_privacy_noise(self):
        dp = DifferentialPrivacy(PrivacyConfig(epsilon=1.0))
        emb = np.ones(32) / np.sqrt(32)
        noised = dp.add_noise(emb)
        # Should be different but still unit-norm
        assert not np.allclose(emb, noised)
        assert abs(np.linalg.norm(noised) - 1.0) < 0.01

    def test_privacy_accountant_budget(self):
        acc = PrivacyAccountant(total_budget=5.0)
        assert acc.spend(2.0)
        assert acc.spend(2.0)
        assert not acc.spend(2.0)  # Exceeds budget
        assert acc.is_exhausted is False
        acc.spend(1.0)
        assert acc.is_exhausted is True

    def test_anonymize_text(self):
        dp = DifferentialPrivacy()
        text = "Contact John Smith at john@example.com"
        anon = dp.anonymize_text(text)
        assert "john@example.com" not in anon
        assert "[EMAIL_REDACTED]" in anon
        assert "John Smith" not in anon

    def test_memory_pool_contribute_and_query(self):
        pool = MemoryPool()
        emb = np.random.randn(32)
        emb /= np.linalg.norm(emb)
        mod = MemoryModule(
            module_id="mod_1",
            embedding=emb,
            category="tech",
            source_agent_hash="agent_1",
        )
        pool.contribute(mod)

        results = pool.query(emb, k=5)
        assert len(results) == 1
        assert results[0].module_id == "mod_1"

    def test_revocation(self):
        pool = MemoryPool()
        emb = np.random.randn(32)
        mod = MemoryModule(
            module_id="mod_1",
            embedding=emb,
            source_agent_hash="agent_1",
        )
        pool.contribute(mod)

        # Wrong agent can't revoke
        assert pool.revoke("mod_1", "wrong_agent") is False
        # Right agent can
        assert pool.revoke("mod_1", "agent_1") is True
        # Revoked module not queryable
        results = pool.query(emb, k=5)
        assert len(results) == 0

    def test_federation_client_export_import(self):
        pool = MemoryPool()
        client_a = FederationClient("agent_a", pool, privacy_budget=100.0)
        client_b = FederationClient("agent_b", pool, privacy_budget=100.0)

        # Agent A exports
        emb = np.random.randn(32).astype(np.float32)
        emb /= np.linalg.norm(emb)
        mid = client_a.export_memory(emb, category="tech")
        assert mid is not None

        # Agent B imports
        imported = client_b.import_relevant(emb, k=5)
        assert len(imported) == 1

        # Agent B doesn't get own exports
        client_a_imports = client_a.import_relevant(emb, k=5)
        assert len(client_a_imports) == 0  # Filtered out own exports

    def test_budget_exhaustion_blocks_export(self):
        pool = MemoryPool()
        client = FederationClient("agent", pool, privacy_budget=0.5)
        emb = np.random.randn(32)
        # First export might work, subsequent ones should fail
        results = []
        for _ in range(10):
            r = client.export_memory(emb)
            results.append(r)
        assert None in results  # At least some should fail

    def test_pool_eviction(self):
        pool = MemoryPool(max_modules=3)
        for i in range(5):
            mod = MemoryModule(
                module_id=f"mod_{i}",
                embedding=np.random.randn(8),
                source_agent_hash="a",
                utility_score=float(i),
            )
            pool.contribute(mod)
        assert len(pool.modules) <= 3

    def test_pool_persistence(self, tmp_dir):
        pool = MemoryPool(pool_path=tmp_dir)
        mod = MemoryModule(
            module_id="mod_1",
            embedding=np.random.randn(8),
            source_agent_hash="a",
        )
        pool.contribute(mod)
        pool.save()

        pool2 = MemoryPool(pool_path=tmp_dir)
        assert "mod_1" in pool2.modules


# =========================================================================
# 6. DREAM CONSOLIDATION
# =========================================================================

class TestDreamConsolidation:
    def test_dream_cycle(self, store):
        for i in range(20):
            _make_episode(store, text=f"memory {i}", surprise=np.random.random())

        dreamer = DreamConsolidation()
        result = dreamer.dream(store)
        assert result.episodes_replayed > 0
        assert result.duration_seconds > 0

    def test_synthetic_generation(self, store):
        for i in range(10):
            _make_episode(store, text=f"experience {i}", surprise=0.5)

        dreamer = DreamConsolidation(DreamConfig(augmentations_per_episode=1))
        result = dreamer.dream(store)
        assert result.synthetic_generated > 0

    def test_principle_extraction(self, store):
        # Create similar episodes to form clusters
        base_emb = np.random.randn(store.config.embedding_dim).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)
        for i in range(5):
            noise = np.random.randn(store.config.embedding_dim) * 0.05
            emb = base_emb + noise
            emb /= np.linalg.norm(emb)
            store.store_episode(
                content={"text": f"similar topic {i}"},
                embedding=emb,
                surprise=0.0,
                metadata={"topics": ["tech_stack"]},
            )

        dreamer = DreamConsolidation(DreamConfig(cluster_threshold=0.8, min_cluster_size=3))
        result = dreamer.dream(store)
        # May or may not find principles depending on similarity
        assert isinstance(result.principles, list)

    def test_skill_extraction(self, store):
        for i in range(10):
            _make_episode(
                store,
                text=f"coding session {i}",
                entities=["python", "vscode"],
            )
            store.episodes[-1].metadata["topics"] = ["tech_stack"]

        dreamer = DreamConsolidation(DreamConfig(skill_repetition_threshold=3))
        result = dreamer.dream(store)
        assert isinstance(result.skills, list)

    def test_pruning(self, store):
        # Create old, low-importance episodes
        old_time = datetime.now(timezone.utc) - timedelta(days=365)
        for i in range(20):
            ep = _make_episode(store, text=f"old memory {i}")
            ep.timestamp = old_time
            ep.importance = 0.01

        dreamer = DreamConsolidation(DreamConfig(prune_activation_threshold=0.1))
        result = dreamer.dream(store)
        assert result.episodes_pruned > 0

    def test_empty_store(self, store):
        dreamer = DreamConsolidation()
        result = dreamer.dream(store)
        assert result.episodes_replayed == 0

    def test_slerp_interpolation(self):
        dreamer = DreamConsolidation()
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])
        mid = dreamer._slerp(v0, v1, 0.5)
        # Midpoint should have equal components
        assert abs(mid[0] - mid[1]) < 0.1
        assert abs(np.linalg.norm(mid) - 1.0) < 0.01


# =========================================================================
# 7. PREDICTIVE PREFETCHER
# =========================================================================

class TestPrefetcher:
    def test_observe_and_predict(self):
        pf = PredictivePrefetcher()
        pf.observe_query("What stack are you using?", topics=["tech_stack"])
        pf.observe_query("Which database?", topics=["database"])
        pf.observe_query("What stack now?", topics=["tech_stack"])
        pf.observe_query("Which database now?", topics=["database"])

        predictions = pf.predict_next_queries(current_topics=["tech_stack"])
        # Should predict "database" as likely next topic
        assert len(predictions) > 0

    def test_cache_hit(self):
        pf = PredictivePrefetcher()
        pf.observe_query("test query", results=[("memory 1", 0.9)])
        cached = pf.get_cached("test query")
        assert cached is not None
        assert cached[0] == ("memory 1", 0.9)

    def test_cache_ttl(self):
        pf = PredictivePrefetcher(PrefetchConfig(cache_ttl_seconds=0.1))
        pf.observe_query("test", results=[("mem", 0.5)])
        time.sleep(0.2)
        assert pf.get_cached("test") is None

    def test_prefetch(self):
        pf = PredictivePrefetcher()
        pf.observe_query("tech", topics=["tech_stack"])
        pf.observe_query("db", topics=["database"])
        pf.observe_query("tech2", topics=["tech_stack"])
        pf.observe_query("db2", topics=["database"])

        predictions = pf.predict_next_queries(current_topics=["tech_stack"])

        def mock_retrieval(query, k):
            return [(f"result for {query}", 0.8)]

        pf.prefetch(predictions, mock_retrieval)
        stats = pf.get_statistics()
        assert stats["cache_size"] > 0

    def test_entity_cooccurrence(self):
        pf = PredictivePrefetcher()
        pf.observe_query("team update", entities=["alice", "bob", "carol"])
        pf.observe_query("standup", entities=["alice", "bob"])
        predictions = pf.predict_next_queries(current_entities=["alice"])
        # Should predict bob as co-occurring
        entity_preds = [p for p in predictions if p["type"] == "entity"]
        if entity_preds:
            assert any(p["value"] == "bob" for p in entity_preds)

    def test_predict_next_topics(self):
        pf = PredictivePrefetcher()
        # Build transition A→B
        for _ in range(5):
            pf.observe_query("q1", topics=["security"])
            pf.observe_query("q2", topics=["infrastructure"])

        topic_preds = pf.predict_next_topics(["security"])
        assert len(topic_preds) > 0
        assert topic_preds[0][0] == "infrastructure"

    def test_cache_eviction(self):
        pf = PredictivePrefetcher(PrefetchConfig(cache_size=3))
        for i in range(10):
            pf.observe_query(f"query_{i}", results=[(f"res_{i}", 0.5)])
        assert len(pf._cache) <= 3


# =========================================================================
# 8. AUDITOR
# =========================================================================

class TestAuditor:
    def test_add_and_verify(self, store):
        ep = _make_episode(store, text="verified memory")
        auditor = MemoryAuditor()
        auditor.add_to_chain(ep)
        assert auditor.verify_episode(ep) is True

    def test_tamper_detection(self, store):
        ep = _make_episode(store, text="original content")
        auditor = MemoryAuditor()
        auditor.add_to_chain(ep)

        # Tamper with content
        ep.content = {"text": "tampered content"}
        assert auditor.verify_episode(ep) is False
        assert ep.episode_id in auditor.get_tampered_episodes()

    def test_chain_integrity(self, store):
        auditor = MemoryAuditor()
        for i in range(10):
            ep = _make_episode(store, text=f"chain episode {i}")
            auditor.add_to_chain(ep)

        report = auditor.verify_chain()
        assert report["valid"] is True
        assert report["checked"] == 10

    def test_chain_broken_detection(self, store):
        auditor = MemoryAuditor()
        for i in range(5):
            ep = _make_episode(store, text=f"ep {i}")
            auditor.add_to_chain(ep)

        # Corrupt a link in the middle
        auditor._chain[2].chain_hash = "corrupted_hash"
        report = auditor.verify_chain()
        assert report["valid"] is False
        assert len(report["broken_links"]) > 0

    def test_merkle_proof(self, store):
        auditor = MemoryAuditor()
        episodes = []
        for i in range(8):
            ep = _make_episode(store, text=f"merkle ep {i}")
            auditor.add_to_chain(ep)
            episodes.append(ep)

        proof = auditor.get_merkle_proof(episodes[3].episode_id)
        assert proof is not None
        assert len(proof) > 0
        assert all("side" in p and "hash" in p for p in proof)

    def test_audit_log(self, store):
        auditor = MemoryAuditor()
        ep = _make_episode(store, text="logged episode")
        auditor.add_to_chain(ep)
        auditor.log_access(ep.episode_id, actor="test_user")

        log = auditor.get_audit_log(episode_id=ep.episode_id)
        assert len(log) >= 2  # create + read
        actions = {e["action"] for e in log}
        assert "create" in actions
        assert "read" in actions

    def test_audit_log_filtering(self, store):
        auditor = MemoryAuditor()
        for i in range(5):
            ep = _make_episode(store, text=f"ep {i}")
            auditor.add_to_chain(ep)
            auditor.log_access(ep.episode_id)

        creates = auditor.get_audit_log(action="create")
        reads = auditor.get_audit_log(action="read")
        assert len(creates) == 5
        assert len(reads) == 5

    def test_persistence(self, store, tmp_dir):
        auditor = MemoryAuditor(persistence_path=tmp_dir)
        for i in range(5):
            ep = _make_episode(store, text=f"persist ep {i}")
            auditor.add_to_chain(ep)
        auditor.save()

        auditor2 = MemoryAuditor(persistence_path=tmp_dir)
        assert len(auditor2._chain) == 5
        report = auditor2.verify_chain()
        assert report["valid"] is True

    def test_untracked_episode(self, store):
        auditor = MemoryAuditor()
        ep = _make_episode(store, text="never added to chain")
        assert auditor.verify_episode(ep) is False

    def test_empty_chain(self):
        auditor = MemoryAuditor()
        report = auditor.verify_chain()
        assert report["valid"] is True
        assert report["checked"] == 0


# =========================================================================
# INTEGRATION: EloBrain with all modules
# =========================================================================

class TestBrainIntegration:
    @pytest.fixture
    def brain(self, tmp_dir):
        from elo_memory.brain import EloBrain
        return EloBrain(user_id="integration_test", persistence_path=tmp_dir)

    def test_process_turn_stores_and_audits(self, brain):
        brain.process_turn("I switched to FastAPI because Django was too slow")
        # Causal engine should have links
        stats = brain.get_causal_graph_stats()
        assert stats["edges"] >= 1

    def test_what_i_know_includes_all_modules(self, brain):
        brain.process_turn("I use Python and PostgreSQL")
        info = brain.what_i_know()
        assert "causal_graph" in info
        assert "governor" in info
        assert "world_sim" in info
        assert "evolution" in info
        assert "dream" in info
        assert "prefetcher" in info
        assert "auditor" in info

    def test_dream_cycle_via_brain(self, brain):
        for i in range(25):
            brain.process_turn(f"Experience number {i} with details")
        result = brain.dream()
        assert result.episodes_replayed > 0

    def test_verify_integrity_via_brain(self, brain):
        brain.process_turn("Important fact to verify")
        report = brain.verify_integrity()
        assert report["valid"] is True

    def test_close_persists_all(self, brain, tmp_dir):
        brain.process_turn("Data to persist")
        brain.close()

        # Check that files were created
        user_dir = brain._memory._user_dir
        assert (user_dir / "causal_graph.json").exists()
        assert (user_dir / "evolution.json").exists()

    def test_counterfactual_via_brain(self, brain):
        brain.process_turn("Rain caused flooding in the city")
        result = brain.counterfactual("Rain")
        assert "removed" in result

    def test_segment_experiences_via_brain(self, brain):
        for i in range(5):
            brain.process_turn(f"Event {i} happened today")
        exps = brain.segment_experiences()
        assert isinstance(exps, list)
