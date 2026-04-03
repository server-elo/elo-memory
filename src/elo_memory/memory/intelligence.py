"""
Memory Intelligence
===================

The layer that makes elo-memory fundamentally different from every competitor.
Not just storage and retrieval — actual reasoning about what the brain knows.

Features no competitor has:
1. Knowledge gap detection — knows what it DOESN'T know
2. Causal chains — tracks WHY decisions were made
3. Decision evolution — traces how choices changed over time
4. Temporal summaries — summarizes periods, not just retrieves
5. Proactive suggestions — tells the agent what to ask next
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ── Context schemas: what SHOULD be known for a given topic ──────

_CONTEXT_SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "payment": {
        "expected": ["currency", "volume", "compliance", "processor", "region"],
        "triggers": ["payment", "billing", "checkout", "transaction", "stripe", "adyen", "paypal"],
    },
    "team": {
        "expected": ["size", "manager", "roles", "location", "hiring"],
        "triggers": ["team", "engineer", "hired", "manager", "employee"],
    },
    "project": {
        "expected": ["deadline", "progress", "stack", "status", "goal"],
        "triggers": ["project", "building", "working on", "migration", "launch"],
    },
    "infrastructure": {
        "expected": ["hosting", "orchestration", "ci/cd", "monitoring", "database"],
        "triggers": ["deploy", "hosting", "aws", "gcp", "kubernetes", "docker", "server"],
    },
    "business": {
        "expected": ["funding", "revenue", "valuation", "runway", "customers"],
        "triggers": [
            "raised",
            "revenue",
            "arr",
            "funding",
            "investor",
            "customer",
            "client",
            "startup",
            "company",
            "business",
        ],
    },
    "product": {
        "expected": ["users", "features", "metrics", "competitors", "pricing"],
        "triggers": ["product", "feature", "user", "dau", "mau", "nps", "launch"],
    },
    "security": {
        "expected": ["compliance", "encryption", "auth", "audit", "incidents"],
        "triggers": ["security", "soc2", "hipaa", "gdpr", "encryption", "auth", "vulnerability"],
    },
}

# ── Causal keywords ──────────────────────────────────────────────

_CAUSAL_RE = re.compile(
    r"\b(?:because|since|due to|in order to|so that|to avoid|the reason|"
    r"caused by|led to|resulted in|in response to|after)\b",
    re.IGNORECASE,
)

_DECISION_RE = re.compile(
    r"\b(?:decided|chose|picked|selected|went with|opted for|"
    r"switched to|moved to|migrated to|considering|planning to)\b",
    re.IGNORECASE,
)


class MemoryIntelligence:
    """Reasoning layer over the knowledge base and episodic store.

    This is what makes elo-memory fundamentally different from competitors.
    Not just retrieval — understanding.
    """

    def __init__(self) -> None:
        self._causal_links: List[Dict[str, str]] = []
        self._decision_chain: List[Dict[str, str]] = []

    # ── 1. Knowledge Gap Detection ───────────────────────────────

    def detect_gaps(
        self,
        kb_facts: Dict[str, str],
        all_texts: List[str],
    ) -> List[Dict[str, Any]]:
        """Detect what the brain SHOULD know but doesn't.

        Looks at what topics have been discussed, checks the context
        schema for that topic, and returns what's missing.
        """
        # Find which schemas are active (user has discussed the topic)
        combined_text = " ".join(all_texts).lower()
        active_schemas: List[str] = []
        for schema_name, schema in _CONTEXT_SCHEMAS.items():
            if any(trigger in combined_text for trigger in schema["triggers"]):
                active_schemas.append(schema_name)

        # For each active schema, check what's missing
        gaps: List[Dict[str, Any]] = []
        kb_text = " ".join(f"{k} {v}" for k, v in kb_facts.items()).lower()
        all_text_lower = combined_text

        # Synonym groups: any word satisfies the whole group
        synonyms = {
            "hosting": {
                "hosting",
                "deploy",
                "deployment",
                "server",
                "aws",
                "gcp",
                "azure",
                "heroku",
                "vercel",
            },
            "orchestration": {"orchestration", "kubernetes", "k8s", "docker", "ecs", "eks"},
            "ci/cd": {"ci/cd", "ci", "cd", "github actions", "jenkins", "circleci", "pipeline"},
            "monitoring": {
                "monitoring",
                "datadog",
                "grafana",
                "sentry",
                "prometheus",
                "observability",
            },
            "database": {"database", "postgresql", "postgres", "mysql", "mongodb", "redis", "db"},
            "compliance": {"compliance", "soc2", "hipaa", "gdpr", "pci", "iso"},
            "funding": {"funding", "raised", "series", "seed", "investment"},
            "revenue": {"revenue", "arr", "mrr", "income", "processing"},
            "customers": {"customers", "clients", "users", "accounts"},
            "competitors": {"competitors", "competitor", "competition", "rival"},
            "stack": {"stack", "backend", "frontend", "framework", "language"},
            "encryption": {"encryption", "encrypted", "kms", "tls", "ssl"},
            "auth": {"auth", "authentication", "oauth", "sso", "login"},
        }

        for schema_name in active_schemas:
            schema = _CONTEXT_SCHEMAS[schema_name]
            missing: List[str] = []
            for expected in schema["expected"]:
                # Check if this concept or any synonym appears in KB or memories
                check_words = synonyms.get(expected, {expected})
                found = any(w in kb_text or w in all_text_lower for w in check_words)
                if not found:
                    missing.append(expected)
            if missing:
                gaps.append(
                    {
                        "topic": schema_name,
                        "missing": missing,
                        "suggestion": f"Consider asking about: {', '.join(missing)}",
                    }
                )

        return gaps

    # ── 2. Causal Chain Extraction ───────────────────────────────

    def extract_causal_links(self, text: str, episode_id: str) -> List[Dict[str, str]]:
        """Extract cause-effect relationships from text.

        "Switched to FastAPI because Django was too slow"
        → {cause: "Django was too slow", effect: "Switched to FastAPI", episode: ...}
        """
        links: List[Dict[str, str]] = []

        # "X because Y"
        m = re.search(r"(.+?)\s+because\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
        if m:
            links.append(
                {
                    "effect": m.group(1).strip(),
                    "cause": m.group(2).strip(),
                    "episode_id": episode_id,
                    "raw": text,
                }
            )

        # "due to X, Y"
        m = re.search(r"due to\s+(.+?),\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
        if m:
            links.append(
                {
                    "cause": m.group(1).strip(),
                    "effect": m.group(2).strip(),
                    "episode_id": episode_id,
                    "raw": text,
                }
            )

        # "X so that Y" / "X in order to Y"
        m = re.search(r"(.+?)\s+(?:so that|in order to)\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
        if m:
            links.append(
                {
                    "cause": m.group(1).strip(),
                    "effect": m.group(2).strip(),
                    "episode_id": episode_id,
                    "raw": text,
                }
            )

        self._causal_links.extend(links)
        return links

    def get_reasons_for(self, query: str) -> List[Dict[str, str]]:
        """Answer 'why' questions by finding causal links."""
        query_lower = query.lower()
        matches: List[Dict[str, str]] = []
        for link in self._causal_links:
            if (
                query_lower in link["effect"].lower()
                or query_lower in link["raw"].lower()
                or any(w in link["effect"].lower() for w in query_lower.split() if len(w) > 3)
            ):
                matches.append(link)
        return matches

    # ── 3. Decision Chain Tracking ───────────────────────────────

    def track_decision(self, text: str, episode_id: str) -> Optional[Dict[str, str]]:
        """Track decisions and their evolution over time."""
        if not _DECISION_RE.search(text):
            return None

        decision = {
            "text": text,
            "episode_id": episode_id,
            "has_reason": bool(_CAUSAL_RE.search(text)),
        }
        self._decision_chain.append(decision)
        return decision

    def get_decision_history(self, topic: str) -> List[Dict[str, str]]:
        """Get how decisions about a topic evolved over time."""
        topic_lower = topic.lower()
        return [d for d in self._decision_chain if topic_lower in d["text"].lower()]

    # ── 4. Temporal Summarization ────────────────────────────────

    @staticmethod
    def summarize_period(
        memories: List[Tuple[str, float]],
        period_label: str = "",
    ) -> Dict[str, Any]:
        """Summarize a collection of memories into themes.

        Instead of returning raw memories, groups them by theme
        and counts what happened.
        """
        if not memories:
            return {"period": period_label, "themes": {}, "count": 0}

        # Categorize by simple keyword themes
        themes: Dict[str, List[str]] = defaultdict(list)
        for text, score in memories:
            text_lower = text.lower()
            categorized = False
            for category, keywords in [
                ("team", ["hired", "promoted", "fired", "team", "engineer"]),
                ("technical", ["switched", "moved", "migrated", "bug", "fixed", "deployed"]),
                ("business", ["revenue", "raised", "funding", "client", "customer", "launched"]),
                ("incidents", ["broke", "down", "outage", "incident", "security", "bug"]),
                ("personal", ["marathon", "training", "family", "vacation"]),
            ]:
                if any(kw in text_lower for kw in keywords):
                    themes[category].append(text)
                    categorized = True
                    break
            if not categorized:
                themes["other"].append(text)

        return {
            "period": period_label,
            "themes": dict(themes),
            "count": len(memories),
            "summary_lines": [
                f"{cat}: {len(items)} event{'s' if len(items) > 1 else ''}"
                for cat, items in sorted(themes.items(), key=lambda x: -len(x[1]))
            ],
        }

    # ── 5. Proactive Suggestions ─────────────────────────────────

    def suggest_next_actions(
        self,
        kb_facts: Dict[str, str],
        recent_memories: List[Tuple[str, float]],
    ) -> List[str]:
        """Suggest what the agent should proactively ask or do.

        Based on knowledge gaps, stale facts, and unresolved items.
        """
        suggestions: List[str] = []

        # From knowledge gaps
        all_texts = [t for t, _ in recent_memories]
        gaps = self.detect_gaps(kb_facts, all_texts)
        for gap in gaps[:3]:  # top 3 most relevant gaps
            suggestions.append(gap["suggestion"])

        # From unresolved decisions
        considering = [
            d
            for d in self._decision_chain
            if isinstance(d, dict)
            and "text" in d
            and ("considering" in d["text"].lower() or "thinking" in d["text"].lower())
        ]
        for d in considering[-2:]:  # last 2 unresolved
            text_val = d.get("text", "")
            if text_val:
                suggestions.append(f"Follow up: {text_val[:60]}...")

        # From causal links without resolution
        unresolved_causes = [
            link
            for link in self._causal_links
            if isinstance(link, dict)
            and "cause" in link
            and any(
                w in link["cause"].lower()
                for w in ["problem", "issue", "slow", "broken", "failing"]
            )
        ]
        for link in unresolved_causes[-2:]:
            suggestions.append(f"Check if resolved: {link['cause'][:60]}")

        return suggestions

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "causal_links": self._causal_links,
            "decision_chain": self._decision_chain,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self._causal_links = data.get("causal_links", [])
        self._decision_chain = data.get("decision_chain", [])
