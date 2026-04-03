"""EloBrain -- agent middleware that combines episodic memory with knowledge.

Provides a simple ``think()`` loop and ``prepare() / process_turn()`` hooks
for framework-agnostic integration.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from .memory.user_memory import UserMemory
from .memory.knowledge_base import KnowledgeBase

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
    re.compile(r"I(?:'ll|\s+will)\s+(?:check|look\s+into|investigate|follow\s+up\s+on|get\s+back\s+to\s+you\s+(?:on|about))\s+(.+?)(?:\.|$)", re.IGNORECASE),
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
            commitments.append(m.group(0).strip())
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
        """
        # KB query first (instant structured answer)
        kb_answer = self._kb.query(user_message)
        kb_facts = self._kb.get_all()

        # Episode recall
        memories = self._memory.recall(user_message, k=k)
        facts = self._memory.get_facts()
        profile = self._memory.get_profile()

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

        system = "\n".join(sections)

        return {
            "system": system,
            "user_message": user_message,
            "memories_used": len(memories),
            "user_profile": profile,
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
            self._memory.store(user_message)

        # Extract and store assistant commitments
        if assistant_response:
            commitments = _extract_commitments(assistant_response)
            for commitment in commitments:
                self._memory.store(f"[assistant commitment] {commitment}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def what_i_know(self) -> Dict[str, Any]:
        """Return everything the brain knows about this user."""
        facts = self._memory.get_facts()
        profile = self._memory.get_profile()

        # Collect topics
        all_topics: set[str] = set()
        for ep in self._memory._store.episodes:
            for t in ep.metadata.get("topics", []):
                all_topics.add(t)

        return {
            "knowledge": self._kb.get_all(),
            "facts": facts,
            "entities": profile.get("entities", {}),
            "topics": sorted(all_topics),
            "total_memories": profile.get("total_memories", 0),
            "superseded": len(self._memory._superseded),
        }

    # ------------------------------------------------------------------
    # GDPR forget
    # ------------------------------------------------------------------

    def forget(self, text: str):
        """Supersede any memories matching *text* (GDPR erasure)."""
        text_lower = text.lower()
        for ep in self._memory._store.episodes:
            ep_text = self._memory._episode_text(ep).lower()
            if text_lower in ep_text:
                self._memory._superseded.add(ep.episode_id)
        self._memory.save()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Persist everything and release resources."""
        self._memory.close()
