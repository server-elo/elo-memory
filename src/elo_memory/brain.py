"""EloBrain -- agent middleware that combines episodic memory with knowledge.

Provides a simple ``think()`` loop and ``prepare() / process_turn()`` hooks
for framework-agnostic integration.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from .memory.user_memory import UserMemory
from .memory.knowledge_base import KnowledgeBase
from .memory.intelligence import MemoryIntelligence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message-filtering patterns
# ---------------------------------------------------------------------------

# Messages that are pure filler / greetings / questions with no personal info
_SKIP_MESSAGE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*(hi|hello|hey|yo|sup|howdy|hiya)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(thanks|thank\s*you|thx|ty|cheers)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(ok|okay|sure|got\s*it|alright|cool|nice|great)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(yes|no|yeah|nah|yep|nope)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(bye|goodbye|see\s*ya|later|cya)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*what\s+time\s+is\s+it\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*how\s+are\s+you\s*\??\s*$", re.IGNORECASE),
]

# Regex that detects personal info worth keeping even inside a question
_HAS_INFO_RE = re.compile(
    r"(?:my\s+name\s+is|i(?:'m|\s+am)\s+\w{2,}|i\s+(?:live|work|use|like|love|hate|prefer|moved|switched)"
    r"|remember\s+(?:that|my)|i\s+have\s+\w+|i\s+(?:was|used\s+to))",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Commitment patterns (assistant promises)
# ---------------------------------------------------------------------------

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
    """Return True if *message* is pure filler with no personal information."""
    # If it contains personal info, never skip
    if _HAS_INFO_RE.search(message):
        return False
    for pattern in _SKIP_MESSAGE_PATTERNS:
        if pattern.match(message):
            return True
    return False


def _extract_commitments(text: str) -> List[str]:
    """Extract assistant commitments/promises from a response."""
    commitments = []
    for pattern in _COMMITMENT_PATTERNS:
        for m in pattern.finditer(text):
            commitments.append(m.group(1).strip())
    return commitments


class EloBrain:
    """Agent middleware combining UserMemory with conversational intelligence."""

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
    # Prepare
    # ------------------------------------------------------------------

    def prepare(self, user_message: str, k: int = 7) -> Dict[str, Any]:
        """Build an enriched prompt with memory context.

        Returns dict with keys: system, user_message, memories_used, user_profile.
        Runs independent lookups in parallel for lower latency.
        """
        # Run independent lookups concurrently
        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_kb_answer = pool.submit(self._kb.query, user_message)
            fut_kb_facts = pool.submit(self._kb.get_all)
            fut_memories = pool.submit(self._memory.recall, user_message, k)
            fut_facts = pool.submit(self._memory.get_facts)
            fut_profile = pool.submit(self._memory.get_profile)

            kb_answer = fut_kb_answer.result()
            kb_facts = fut_kb_facts.result()
            memories = fut_memories.result()
            facts = fut_facts.result()
            profile = fut_profile.result()

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

        # Why-reasoning: if query looks like a "why" question, find causal links
        if re.search(r"\bwhy\b", user_message, re.IGNORECASE):
            reasons = self._intelligence.get_reasons_for(user_message)
            if reasons:
                sections.append("\n## Reasons (causal links)")
                for r in reasons[:3]:
                    sections.append(f"- {r['cause']} → {r['effect']}")

        # Knowledge gaps: what the brain should know but doesn't
        all_texts = [t for t, _ in facts]
        gaps = self._intelligence.detect_gaps(kb_facts, all_texts)

        # Proactive suggestions
        suggestions = self._intelligence.suggest_next_actions(kb_facts, facts)

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
    # Process turn
    # ------------------------------------------------------------------

    def process_turn(
        self,
        user_message: str,
        assistant_response: Optional[str] = None,
    ):
        """Store the interaction in memory.

        Filters out pure greetings / filler unless they contain personal info.
        """
        # Always update KB (even for questions -- they may contain facts)
        self._kb.update(user_message)

        # Store user message in episodic memory (unless it's pure filler)
        if not _should_skip(user_message):
            result = self._memory.store(user_message)
            episode_id = (result or {}).get("episode_id", "")
            # Feed intelligence layer with the real episode_id
            self._intelligence.extract_causal_links(user_message, episode_id)
            self._intelligence.track_decision(user_message, episode_id)

        # Extract and store assistant commitments (dedup via near-duplicate check)
        if assistant_response:
            commitments = _extract_commitments(assistant_response)
            seen = set()
            for commitment in commitments:
                if commitment not in seen:
                    seen.add(commitment)
                    self._memory.store(f"[assistant commitment] {commitment}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def what_i_know(self) -> Dict[str, Any]:
        """Return everything the brain knows about this user."""
        facts = self._memory.get_facts()
        profile = self._memory.get_profile()

        # Collect topics (excluding superseded episodes)
        all_topics: set[str] = set()
        for ep in self._memory._store.episodes:
            if ep.episode_id in self._memory._superseded:
                continue
            for t in ep.metadata.get("topics", []):
                all_topics.add(t)

        # Intelligence: gaps, causal chains, suggestions
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
        }

    # ------------------------------------------------------------------
    # GDPR forget
    # ------------------------------------------------------------------

    def forget(self, text: str):
        """Supersede any memories matching *text* (GDPR erasure).

        Uses the UserMemory supersession API so that offloaded episodes
        are properly handled and state is persisted.
        """
        text_lower = text.lower()
        with self._memory._lock:
            # Scan in-memory episodes
            for ep in self._memory._store.episodes:
                ep_text = self._memory._episode_text(ep).lower()
                if text_lower in ep_text:
                    self._memory._superseded.add(ep.episode_id)
            # Scan offloaded episodes on disk
            if self._memory._store.config.enable_disk_offload and self._memory._store.persistence_path:
                offload_dir = self._memory._store.persistence_path / "offloaded"
                if offload_dir.exists():
                    import pickle
                    for pkl_file in offload_dir.glob("*.pkl"):
                        try:
                            with open(pkl_file, "rb") as f:
                                ep = pickle.load(f)
                            ep_text = self._memory._episode_text(ep).lower()
                            if text_lower in ep_text:
                                self._memory._superseded.add(ep.episode_id)
                        except Exception:
                            pass
            self._memory.save()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Persist everything and release resources."""
        self._memory.close()
