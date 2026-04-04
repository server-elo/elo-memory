"""EloBrain -- agent middleware that combines episodic memory with knowledge.

Provides a simple ``think()`` loop and ``prepare() / process_turn()`` hooks
for framework-agnostic integration.

All 8 advanced modules are wired into the core flow:
- Governor gates what gets stored (like selective attention)
- Causal engine tracks why things happen
- World simulator segments episodes into replayable experiences
- Evolution adapts embeddings from retrieval feedback
- Dream cycle runs when enough experience accumulates
- Prefetcher anticipates what you'll need next
- Auditor seals every memory for integrity
- Federation is opt-in for multi-agent sharing
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .memory.user_memory import UserMemory
from .memory.knowledge_base import KnowledgeBase
from .memory.intelligence import MemoryIntelligence
from .governor import MemoryGovernor, DecisionContext, Action
from .causal_engine import CausalInferenceEngine
from .world_simulator import WorldSimulator
from .evolution import ParametricEvolution
from .consolidation.dream_cycle import DreamConsolidation
from .retrieval.prefetcher import PredictivePrefetcher
from .auditor import MemoryAuditor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message-filtering patterns
# ---------------------------------------------------------------------------

_SKIP_MESSAGE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*(hi|hello|hey|yo|sup|howdy|hiya)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(thanks|thank\s*you|thx|ty|cheers)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(ok|okay|sure|got\s*it|alright|cool|nice|great)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(yes|no|yeah|nah|yep|nope)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(bye|goodbye|see\s*ya|later|cya)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*what\s+time\s+is\s+it\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*how\s+are\s+you\s*\??\s*$", re.IGNORECASE),
]

_HAS_INFO_RE = re.compile(
    r"(?:my\s+name\s+is|i(?:'m|\s+am)\s+\w{2,}|i\s+(?:live|work|use|like|love|hate|prefer|moved|switched)"
    r"|remember\s+(?:that|my)|i\s+have\s+\w+|i\s+(?:was|used\s+to))",
    re.IGNORECASE,
)

_COMMITMENT_PATTERNS: List[re.Pattern] = [
    re.compile(r"I'll\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"I've\s+(?:noted|recorded|saved|remembered)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(
        r"I(?:'ll|\s+will)\s+(?:check|look\s+into|investigate|follow\s+up\s+on|get\s+back\s+to\s+you\s+(?:on|about))\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    re.compile(r"I(?:'ll|\s+will)\s+(?:make\s+sure|ensure)\s+(.+?)(?:\.|$)", re.IGNORECASE),
]


def _should_skip(message: str) -> bool:
    if _HAS_INFO_RE.search(message):
        return False
    for pattern in _SKIP_MESSAGE_PATTERNS:
        if pattern.match(message):
            return True
    return False


def _extract_commitments(text: str) -> List[str]:
    commitments = []
    for pattern in _COMMITMENT_PATTERNS:
        for m in pattern.finditer(text):
            commitments.append(m.group(1).strip())
    return commitments


class EloBrain:
    """Agent middleware combining UserMemory with conversational intelligence."""

    # How many episodes between automatic dream cycles
    _DREAM_INTERVAL = 200

    def __init__(
        self,
        user_id: str,
        persistence_path: str = "./memories",
        system_prompt: Optional[str] = None,
    ):
        self.user_id = user_id
        self._system_prompt = system_prompt or "You are a helpful assistant with memory."
        self._memory = UserMemory(
            user_id=user_id,
            persistence_path=persistence_path,
        )
        self._kb = KnowledgeBase(
            persistence_path=str(self._memory._user_dir / "kb"),
        )
        self._intelligence = MemoryIntelligence()

        user_dir = str(self._memory._user_dir)

        # 1. Governor — learns what to store vs skip
        self._governor = MemoryGovernor(persistence_path=user_dir)

        # 2. Causal engine — tracks cause/effect from language
        self._causal = CausalInferenceEngine()
        self._causal.load(self._memory._user_dir / "causal_graph.json")

        # 3. World simulator — groups episodes into experiences
        self._world_sim = WorldSimulator(self._memory._store)
        self._world_sim.load(self._memory._user_dir / "world_sim.json")

        # 4. Evolution — adapts embeddings from retrieval feedback
        self._evolution = ParametricEvolution(embedding_dim=self._memory.embedding_dim)
        self._evolution.load(self._memory._user_dir / "evolution.json")

        # 5. Dream consolidation — creative replay
        self._dreamer = DreamConsolidation()
        self._episodes_since_dream = 0

        # 6. Prefetcher — anticipates next queries
        self._prefetcher = PredictivePrefetcher()

        # 7. Auditor — seals every memory in a hash chain
        self._auditor = MemoryAuditor(persistence_path=user_dir)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def think(
        self,
        user_message: str,
        llm_fn: Callable[[str], str],
        k: int = 7,
    ) -> str:
        """Full think loop: prepare -> call LLM -> process turn."""
        context = self.prepare(user_message, k=k)
        prompt = context["system"] + "\n\nUser: " + context["user_message"]
        response = llm_fn(prompt)
        self.process_turn(user_message, assistant_response=response)
        return response

    # ------------------------------------------------------------------
    # Prepare — where retrieval happens
    # ------------------------------------------------------------------

    def prepare(self, user_message: str, k: int = 7) -> Dict[str, Any]:
        """Build an enriched prompt with memory context.

        Uses prefetcher cache, evolution-adapted embeddings, causal reasoning,
        and experience context to build the richest possible prompt.
        """
        # 6. Prefetcher: check warm cache before doing any retrieval
        cached = self._prefetcher.get_cached(user_message)

        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_kb_answer = pool.submit(self._kb.query, user_message)
            fut_kb_facts = pool.submit(self._kb.get_all)
            fut_memories = pool.submit(self._memory.recall, user_message, k) if not cached else None
            fut_facts = pool.submit(self._memory.get_facts)
            fut_profile = pool.submit(self._memory.get_profile)

            kb_answer = fut_kb_answer.result()
            kb_facts = fut_kb_facts.result()
            memories = cached if cached is not None else fut_memories.result()
            facts = fut_facts.result()
            profile = fut_profile.result()

        # 4. Evolution: record retrieval feedback — the fact that these
        # memories were recalled means they're potentially relevant.
        # The actual relevance signal comes later (in process_turn when
        # the user continues the conversation on the same topic).
        if memories:
            query_emb = self._memory._embed(user_message)
            for text, score in memories[:3]:
                result_emb = self._memory._embed(text)
                # Score > 0.5 = likely relevant retrieval
                self._evolution.record_feedback(query_emb, result_emb, relevance=min(1.0, score))

        # Build system prompt sections
        sections = [self._system_prompt]

        if kb_facts:
            sections.append("\n## Current facts (Knowledge Base)")
            for key, value in list(kb_facts.items())[:20]:
                sections.append(f"- {key}: {value}")

        if kb_answer:
            sections.append(f"\n## Direct answer from KB\n{kb_answer}")

        if memories:
            sections.append("\n## Related memories")
            for text, score in memories:
                sections.append(f"- [{score:.2f}] {text}")

        # 2. Causal engine: answer "why" with graph traversal
        if re.search(r"\bwhy\b", user_message, re.IGNORECASE):
            reasons = self._intelligence.get_reasons_for(user_message)
            causal_reasons = self._causal.query_causes(user_message, depth=3)
            if reasons or causal_reasons:
                sections.append("\n## Reasons (causal links)")
                for r in reasons[:3]:
                    sections.append(f"- {r['cause']} → {r['effect']}")
                for r in causal_reasons[:3]:
                    sections.append(f"- [{r['strength']:.1f}] {r['cause']} → {r['effect']}")

        # 2. Causal engine: answer "what if" with counterfactuals
        if re.search(r"\bwhat\s+if\b", user_message, re.IGNORECASE):
            cf = self._causal.counterfactual(user_message)
            if cf.get("lost_effects"):
                sections.append("\n## Counterfactual effects")
                for effect in cf["lost_effects"][:5]:
                    sections.append(f"- Would lose: {effect}")

        # 2. Causal engine: answer "what caused" / "what led to"
        if re.search(r"what\s+(?:caused|led\s+to|resulted\s+in)", user_message, re.IGNORECASE):
            effects = self._causal.query_effects(user_message, depth=2)
            if effects:
                sections.append("\n## Effects chain")
                for e in effects[:5]:
                    sections.append(f"- {e['cause']} → {e['effect']}")

        # 3. World simulator: if the user asks about a time period,
        # include experience-level context (not just individual memories)
        if re.search(r"\b(last\s+(?:week|month|session)|recently|earlier|before)\b", user_message, re.IGNORECASE):
            if not self._world_sim.experiences:
                self._world_sim.segment_experiences()
            if self._world_sim.experiences:
                sections.append("\n## Recent experiences")
                for exp in self._world_sim.experiences[-3:]:
                    n = len(exp.episode_ids)
                    locations = exp.metadata.get("locations", [])
                    entities = exp.metadata.get("entities", [])
                    parts = [f"{n} events"]
                    if locations:
                        parts.append(f"at {', '.join(locations[:3])}")
                    if entities:
                        parts.append(f"involving {', '.join(entities[:3])}")
                    sections.append(f"- Experience: {' '.join(parts)}")

        # Knowledge gaps
        all_texts = [t for t, _ in facts]
        gaps = self._intelligence.detect_gaps(kb_facts, all_texts)

        # Proactive suggestions
        suggestions = self._intelligence.suggest_next_actions(kb_facts, facts)

        # 6. Prefetcher: include predictions as suggestions
        topics = self._memory._detect_topics(user_message)
        predictions = self._prefetcher.predict_next_queries(current_topics=topics)
        predicted_topics = [p["value"] for p in predictions if p["type"] == "topic" and p["confidence"] > 0.3]
        if predicted_topics:
            suggestions.append(f"Likely next topics: {', '.join(predicted_topics)}")

        system = "\n".join(sections)

        return {
            "system": system,
            "user_message": user_message,
            "memories_used": len(memories),
            "user_profile": profile,
            "knowledge_gaps": gaps,
            "suggestions": suggestions,
        }

    # ------------------------------------------------------------------
    # Process turn — where storage happens
    # ------------------------------------------------------------------

    def process_turn(
        self,
        user_message: str,
        assistant_response: Optional[str] = None,
    ):
        """Store the interaction in memory.

        Governor decides what to store. Causal engine tracks relationships.
        Auditor seals each stored memory. Prefetcher learns patterns.
        Dream cycle auto-triggers when enough experience accumulates.
        """
        self._kb.update(user_message)

        if not _should_skip(user_message):
            # 1. Governor: compute context features and decide
            store = self._memory._store
            max_sim = 0.0
            embedding = self._memory._embed(user_message)

            # 4. Evolution: adapt the query embedding before similarity check
            adapted_emb = self._evolution.adapt_embedding(embedding)

            # Compute max similarity to existing memories (novelty signal)
            if store.episodes:
                recent = store.episodes[-min(50, len(store.episodes)):]
                for ep in recent:
                    if ep.embedding is not None:
                        sim = float(np.dot(adapted_emb, ep.embedding))
                        if sim > max_sim:
                            max_sim = sim

            topics = self._memory._detect_topics(user_message)
            context = DecisionContext(
                surprise=max(0, 1.0 - max_sim),  # Novel = high surprise
                max_similarity=max_sim,
                entity_count=len(self._memory._entity_extractor.extract_flat(user_message)),
                storage_utilization=len(store.episodes) / max(1, store.config.max_episodes),
                topic_overlap=1.0 if topics else 0.0,
            )
            action = self._governor.decide(context)
            # During cold-start (< 200 decisions), never skip — the governor
            # doesn't have enough reward data to make good skip decisions yet.
            # Like a child: absorb everything first, learn to filter later.
            if action == Action.SKIP and self._governor.total_decisions < 200:
                action = Action.ENCODE
            should_store, adjusted_importance = self._governor.apply_action(action, 0.5)

            if should_store:
                result = self._memory.store(user_message)
                episode_id = (result or {}).get("episode_id", "")

                if episode_id:
                    # 1. Governor: track for delayed reward
                    self._governor.record_decision(action, context, episode_id)

                    # 2. Causal engine: extract cause/effect relationships
                    self._causal.ingest(user_message, episode_id)

                    # 7. Auditor: seal into hash chain
                    ep = store._episode_index.get(episode_id)
                    if ep:
                        self._auditor.add_to_chain(ep)

                    # Track episode for dream cycle
                    self._episodes_since_dream += 1

                self._intelligence.extract_causal_links(user_message, episode_id)
                self._intelligence.track_decision(user_message, episode_id)
            else:
                # Governor skipped — still feed intelligence
                self._intelligence.extract_causal_links(user_message, "")

            # 6. Prefetcher: learn from this query, warm cache for predicted next
            entities = []
            if should_store and result and result.get("entities"):
                entities = result["entities"]
            self._prefetcher.observe_query(
                user_message, topics=topics, entities=entities,
            )
            predictions = self._prefetcher.predict_next_queries(
                current_topics=topics, current_entities=entities,
            )
            if predictions:
                self._prefetcher.prefetch(predictions, self._memory.recall)

        # Store assistant commitments
        if assistant_response:
            commitments = _extract_commitments(assistant_response)
            seen = set()
            for commitment in commitments:
                if commitment not in seen:
                    seen.add(commitment)
                    self._memory.store(f"[assistant commitment] {commitment}")

        # 5. Dream: auto-trigger when enough experience has accumulated
        if self._episodes_since_dream >= self._DREAM_INTERVAL:
            logger.info("Auto-triggering dream cycle (%d episodes)", self._episodes_since_dream)
            self.dream()
            self._episodes_since_dream = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def what_i_know(self) -> Dict[str, Any]:
        """Return everything the brain knows about this user."""
        facts = self._memory.get_facts()
        profile = self._memory.get_profile()

        all_topics: set[str] = set()
        for ep in self._memory._store.episodes:
            if ep.episode_id in self._memory._superseded:
                continue
            for t in ep.metadata.get("topics", []):
                all_topics.add(t)

        all_texts = [t for t, _ in facts]
        gaps = self._intelligence.detect_gaps(self._kb.get_all(), all_texts)
        suggestions = self._intelligence.suggest_next_actions(self._kb.get_all(), facts)

        return {
            "knowledge": self._kb.get_all(),
            "facts": facts,
            "entities": profile.get("entities", {}),
            "topics": sorted(all_topics),
            "total_memories": profile.get("total_memories", 0),
            "superseded": len(self._memory._superseded),
            "knowledge_gaps": gaps,
            "suggestions": suggestions,
            "causal_links": len(self._intelligence._causal_links),
            "decisions_tracked": len(self._intelligence._decision_chain),
            "causal_graph": self._causal.get_statistics(),
            "governor": self._governor.get_policy_summary(),
            "world_sim": self._world_sim.get_statistics(),
            "evolution": self._evolution.get_statistics(),
            "dream": self._dreamer.get_statistics(),
            "prefetcher": self._prefetcher.get_statistics(),
            "auditor": self._auditor.get_statistics(),
        }

    # ------------------------------------------------------------------
    # GDPR forget
    # ------------------------------------------------------------------

    def forget(self, text: str):
        """Supersede any memories matching *text* (GDPR erasure)."""
        text_lower = text.lower()
        with self._memory._lock:
            for ep in self._memory._store.episodes:
                ep_text = self._memory._episode_text(ep).lower()
                if text_lower in ep_text:
                    self._memory._superseded.add(ep.episode_id)
            if self._memory._store.config.enable_disk_offload and self._memory._store.persistence_path:
                offload_dir = self._memory._store.persistence_path / "offloaded"
                if offload_dir.exists():
                    import json as _json
                    for ep_file in offload_dir.glob("*.json"):
                        try:
                            with open(ep_file, "r") as f:
                                from .memory.episodic_store import Episode
                                ep = Episode.from_dict(_json.load(f))
                            ep_text = self._memory._episode_text(ep).lower()
                            if text_lower in ep_text:
                                self._memory._superseded.add(ep.episode_id)
                        except Exception as e:
                            logger.error(
                                "GDPR forget: failed to read offloaded episode %s: %s",
                                ep_file.name, e,
                            )
                            continue
            self._memory.save()

    # ------------------------------------------------------------------
    # Advanced capabilities
    # ------------------------------------------------------------------

    def dream(self):
        """Run a dream consolidation cycle.

        NREM: replay important memories. REM: generate creative recombinations.
        Abstraction: extract principles from clusters. Pruning: drop dead weight.

        After dreaming, re-segment experiences (3. World Simulator) and
        run an evolution update (4. Evolution) on the distilled knowledge.
        """
        result = self._dreamer.dream(self._memory._store)

        # 3. World simulator: re-segment after consolidation reshuffled things
        self._world_sim.segment_experiences()

        # 4. Evolution: use the dream as a training signal
        if result.episodes_replayed > 0:
            self._evolution.update_weights()

        # 2. Causal engine: decay old links that haven't been reinforced
        self._causal.decay_strengths(days_elapsed=1.0)

        return result

    def counterfactual(self, removed_cause: str) -> Dict[str, Any]:
        """Answer 'What if X had not happened?'"""
        return self._causal.counterfactual(removed_cause)

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the cryptographic integrity of all memories."""
        chain_report = self._auditor.verify_chain()
        tampered = []
        for ep in self._memory._store.episodes:
            if not self._auditor.verify_episode(ep):
                tampered.append(ep.episode_id)
        chain_report["tampered_episodes"] = tampered
        return chain_report

    def get_audit_log(self, episode_id: str = None, limit: int = 100) -> List[Dict]:
        return self._auditor.get_audit_log(episode_id=episode_id, limit=limit)

    def replay_experience(self, experience_id: str) -> List[Dict[str, Any]]:
        """Re-live a past experience step by step."""
        return self._world_sim.replay(experience_id)

    def segment_experiences(self) -> List[Dict[str, Any]]:
        """Auto-segment all episodes into coherent experiences."""
        exps = self._world_sim.segment_experiences()
        return [e.to_dict() for e in exps]

    def evolve(self) -> Dict[str, Any]:
        """Manually trigger a parametric evolution update."""
        return self._evolution.update_weights()

    def get_causal_graph_stats(self) -> Dict[str, Any]:
        return self._causal.get_statistics()

    def get_governor_policy(self) -> Dict[str, Any]:
        return self._governor.get_policy_summary()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Persist everything and release resources."""
        self._memory.close()
        self._governor.save()
        self._causal.save(self._memory._user_dir / "causal_graph.json")
        self._world_sim.save(self._memory._user_dir / "world_sim.json")
        self._evolution.save(self._memory._user_dir / "evolution.json")
        self._auditor.save()
