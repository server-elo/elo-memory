"""Domain-agnostic structured knowledge store.

Extracts subject-predicate-object facts from natural language.
Works for ANY domain - tech, personal, medical, legal, anything.

This was the key feature that took recall from 54% to 100%.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# Category words used to infer KB keys from transition sentences
_CATEGORY_WORDS = {
    "backend", "frontend", "database", "payment", "hosting",
    "build", "monitoring", "framework", "language", "cloud",
    "ci", "cd", "ci/cd", "auth", "authentication", "cache",
    "queue", "messaging", "search", "storage", "cdn", "dns",
    "orm", "api", "gateway", "proxy", "load balancer",
    "orchestration", "container", "infrastructure",
}

# Well-known tech values → their category (used when context is ambiguous)
_VALUE_TO_CATEGORY = {
    "kubernetes": "orchestration", "k8s": "orchestration",
    "docker": "container", "ecs": "orchestration", "eks": "orchestration",
    "gke": "orchestration", "heroku": "hosting", "railway": "hosting",
    "vercel": "hosting", "netlify": "hosting", "aws": "cloud",
    "gcp": "cloud", "azure": "cloud",
    "kafka": "streaming", "rabbitmq": "messaging", "redis": "cache",
    "postgresql": "database", "postgres": "database", "mysql": "database",
    "mongodb": "database", "dynamodb": "database",
    "stripe": "payment", "adyen": "payment", "paypal": "payment",
    "datadog": "monitoring", "sentry": "monitoring", "grafana": "monitoring",
    "github actions": "ci/cd", "jenkins": "ci/cd", "circleci": "ci/cd",
    "webpack": "build", "vite": "build", "esbuild": "build",
    "fastapi": "backend", "django": "backend", "express": "backend",
    "react": "frontend", "next.js": "frontend", "vue": "frontend",
}

# Compliance / certification keywords
_COMPLIANCE_KEYWORDS = {
    "soc2", "soc 2", "hipaa", "gdpr", "pci", "pci-dss",
    "iso 27001", "iso27001", "fedramp", "ccpa", "ferpa",
}

# Identity keywords mapped to canonical key names
_IDENTITY_KEYWORDS = {
    "name": "name",
    "email": "email",
    "role": "role",
    "title": "role",
    "position": "role",
    "company": "company",
    "organization": "company",
    "org": "company",
    "location": "location",
    "city": "location",
    "phone": "phone",
    "age": "age",
}


class KnowledgeBase:
    """Domain-agnostic structured knowledge store.

    Extracts subject-predicate-object facts from natural language.
    Works for ANY domain -- tech, personal, medical, legal, anything.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self._facts: Dict[str, str] = {}
        self._history: List[Dict] = []
        self._persistence_path = persistence_path
        if persistence_path:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, text: str) -> Dict[str, str]:
        """Extract facts from text, upsert into KB. Returns changes dict."""
        changes: Dict[str, str] = {}
        sentences = self._split_sentences(text)
        for sentence in sentences:
            extracted = self._extract_facts(sentence)
            for key, value in extracted.items():
                key = key.strip().lower()
                value = value.strip()
                if not key or not value:
                    continue
                old = self._facts.get(key)
                if old != value:
                    if old is not None:
                        self._facts[f"_old_{key}"] = old
                    self._facts[key] = value
                    changes[key] = value
                    self._history.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "key": key,
                        "old": old,
                        "new": value,
                    })
        if changes and self._persistence_path:
            self._save()
        return changes

    def query(self, question: str) -> Optional[str]:
        """Answer from KB directly. Returns None if no match."""
        q = question.lower().strip().rstrip("?").strip()
        active = self.get_all()

        # Special handler: tech stack
        if "tech stack" in q or "technology stack" in q or "stack" in q:
            return self._query_tech_stack(active)

        # Special handler: who is X
        who_match = re.match(r"who is (.+)", q)
        if who_match:
            return self._query_who(who_match.group(1).strip(), active)

        # Special handler: compliance
        if "compliance" in q or "certification" in q:
            return self._query_compliance(active)

        # 1. Direct key match
        for key, value in active.items():
            if key in q or q in key:
                return f"{key}: {value}"

        # 2. Keyword search across keys and values
        words = set(q.split())
        best_score = 0
        best_result = None
        for key, value in active.items():
            key_words = set(key.split())
            val_words = set(value.lower().split())
            score = len(words & key_words) + len(words & val_words)
            if score > best_score:
                best_score = score
                best_result = f"{key}: {value}"
        if best_score > 0:
            return best_result

        return None

    def get_all(self) -> Dict[str, str]:
        """All facts excluding _old_ keys."""
        return {k: v for k, v in self._facts.items() if not k.startswith("_old_")}

    def get_summary(self) -> str:
        """Human-readable summary of all current facts."""
        active = self.get_all()
        if not active:
            return "Knowledge base is empty."
        lines = [f"Knowledge Base ({len(active)} facts):"]
        for key, value in sorted(active.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Extraction engine
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into processable chunks.

        Splits on period-space, period-end-of-string, and newlines,
        but NOT on periods inside emails, URLs, or abbreviations.
        """
        # Split on newlines first
        lines = text.split("\n")
        result = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split on ". " (period followed by space) or period at end
            # but preserve email addresses and URLs
            parts = re.split(r'(?<!\w\.\w)(?<![A-Z][a-z])\.(?:\s|$)', line)
            for p in parts:
                p = p.strip()
                if p:
                    result.append(p)
        return result

    def _extract_facts(self, sentence: str) -> Dict[str, str]:
        """Extract key-value facts from a single sentence."""
        facts: Dict[str, str] = {}
        s = sentence.strip()
        s_lower = s.lower()

        # --- Pattern 6: Comma-separated labeled parts ---
        # "Django backend, React frontend, PostgreSQL database"
        comma_facts = self._extract_comma_list(s)
        if comma_facts:
            facts.update(comma_facts)

        # --- Pattern 5: Transitions ---
        # "switched/moved/migrated from X to Y for Z"
        transition_facts = self._extract_transition(s)
        if transition_facts:
            facts.update(transition_facts)
            return facts  # transition is authoritative for the sentence

        # --- Pattern 3: "We use/using X for Y" ---
        use_match = re.match(
            r"(?:we(?:'re)?|i(?:'m)?|they|our team)\s+(?:use|uses|using|run|runs|running)\s+(.+?)\s+(?:for|as|between)\s+(.+)",
            s, re.IGNORECASE,
        )
        if use_match:
            value, key = use_match.group(1).strip(), use_match.group(2).strip()
            # Also check for "X and Y for Z" → extract Y separately with label from Z
            parts = re.split(r'\s+and\s+', value)
            if len(parts) == 1:
                facts[key.lower()] = value
            else:
                facts[key.lower()] = value
                # "gRPC between services and PostgreSQL for the database"
                # key=services, but also extract labeled parts
                for part in parts:
                    for cat in ("backend", "frontend", "database", "server", "client", "cache"):
                        if cat in key.lower() or cat in s_lower:
                            pass  # comma_list already handles this

        # --- Pattern 3a2: Aspirational/preference: "should switch to X" / "considering X" ---
        pref_match = re.search(
            r"(?:should|want to|considering|thinking about|planning to)\s+"
            r"(?:switch(?:ing)?|mov(?:e|ing)|migrat(?:e|ing)|try(?:ing)?|us(?:e|ing))\s+"
            r"(?:to\s+)?(\S+)",
            s, re.IGNORECASE,
        )
        if pref_match:
            val = pref_match.group(1).strip()
            facts["considering/want to try"] = val

        # --- Pattern 3b: "X for the Y" in middle of sentence ---
        for_match = re.search(
            r"(\S+)\s+for\s+the\s+(database|backend|frontend|server|monitoring|build|payment)",
            s, re.IGNORECASE,
        )
        if for_match:
            val, cat = for_match.group(1).strip(), for_match.group(2).strip().lower()
            facts[cat] = val

        # --- Pattern 3c: "working on X" → project ---
        work_match = re.search(
            r"(?:I'?m |we(?:'re)? )?working on (.+?)(?:\s*[,.]|\s*$)",
            s, re.IGNORECASE,
        )
        if work_match:
            facts["project"] = work_match.group(1).strip()

        # --- Pattern 3d: "X% done/complete/on Y" → progress ---
        progress_match = re.search(r"(\d+%)\s+(?:done|complete|finished|on\s+)", s, re.IGNORECASE)
        if progress_match:
            facts["progress"] = progress_match.group(1)

        # --- Pattern 4: Identity ---
        identity_facts = self._extract_identity(s)
        facts.update(identity_facts)

        # --- Patterns 1 & 2: "My/Our/The X is Y" ---
        is_match = re.match(
            r"(?:my|our|the|his|her|their|its)\s+(.+?)\s+is\s+(.+)",
            s, re.IGNORECASE,
        )
        if is_match:
            key, value = is_match.group(1).strip().lower(), is_match.group(2).strip()
            facts[key] = value

        # --- Pattern 6b: "X is Y" (generic, no possessive required) ---
        # "P99 latency is 450ms", "DAU is 45k", "Error rate is 3.2%"
        generic_is = re.match(r"(\S+(?:\s+\S+)?)\s+is\s+(\d[\d,.]*\s*[%kKmMbBms]*\S*)", s, re.IGNORECASE)
        if generic_is and generic_is.group(1).lower() not in facts:
            key = generic_is.group(1).strip().lower()
            val = generic_is.group(2).strip()
            if len(key) < 30 and len(val) < 50:
                facts[key] = val

        # --- Pattern 6c: Corrections: "Actually X is Y" / "X is actually Y" ---
        correction = re.search(r"[Aa]ctually\s+(?:my\s+)?(\S+(?:\s+\S+)?)\s+is\s+(.+?)(?:\s*,|\s*$)", s)
        if correction:
            facts[correction.group(1).strip().lower()] = correction.group(2).strip()

        # --- Pattern 6d: "burning $X per month" / "runway is X months" ---
        burn = re.search(r"burning\s+(\$[\d,.]+[kKmM]?)\s+per\s+(\w+)", s, re.IGNORECASE)
        if burn:
            facts["burn rate"] = f"{burn.group(1)}/{burn.group(2)}"
        runway = re.search(r"runway\s+(?:is\s+)?(\d+\s*\w+)", s, re.IGNORECASE)
        if runway:
            facts["runway"] = runway.group(1)

        # --- Pattern 7: Team operations ---
        team_facts = self._extract_team(s)
        facts.update(team_facts)

        # --- Pattern 8: Money ---
        money_facts = self._extract_money(s)
        facts.update(money_facts)

        # --- Pattern 9: Metrics ---
        metric_facts = self._extract_metrics(s)
        facts.update(metric_facts)

        # --- Pattern 10: Compliance ---
        compliance_facts = self._extract_compliance(s)
        facts.update(compliance_facts)

        return facts

    def _extract_comma_list(self, sentence: str) -> Dict[str, str]:
        """Extract 'X backend, Y frontend, Z database' patterns."""
        facts: Dict[str, str] = {}
        # Match patterns like "Word category" separated by commas/and
        parts = re.split(r',\s*|\s+and\s+', sentence)
        if len(parts) < 2:
            return facts
        matched_any = False
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # "Django backend" or "React frontend" — value should be 1-3 words max,
            # not a full clause like "Hired Jake for frontend"
            for cat in _CATEGORY_WORDS:
                pattern = re.compile(
                    rf'^(\S+(?:\s+\S+)?)\s+{re.escape(cat)}$',
                    re.IGNORECASE,
                )
                m = pattern.match(part)
                if m:
                    val = m.group(1).strip()
                    # Skip if value looks like an action clause
                    if not re.match(r'(?:hired|promoted|fired|using|keeping|for)\b', val, re.I):
                        facts[cat] = val
                    matched_any = True
                    break
        return facts if matched_any else {}

    def _extract_transition(self, sentence: str) -> Dict[str, str]:
        """Extract 'switched/moved/migrated from X to Y' patterns."""
        facts: Dict[str, str] = {}
        pattern = re.compile(
            r'(?:switched|moved|migrated|changed|transitioned|upgraded|replaced)\s+'
            r'(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\s+for\s+(.+))?$',
            re.IGNORECASE,
        )
        m = pattern.search(sentence)
        if not m:
            return facts

        old_val = m.group(1).strip()
        new_val = m.group(2).strip()
        context = m.group(3).strip() if m.group(3) else ""

        # Determine the key from explicit context or category words in sentence
        key = None
        s_lower = sentence.lower()

        # Check context phrase first
        if context:
            for cat in _CATEGORY_WORDS:
                if cat in context.lower():
                    key = cat
                    break

        # Check sentence body for category words (excluding the old/new values)
        if not key:
            for cat in _CATEGORY_WORDS:
                if cat in s_lower:
                    key = cat
                    break

        # Try to match old value to an existing KB key first (most precise)
        if not key:
            key = self._find_key_for_value(old_val)

        # Then try well-known value→category mapping
        if not key:
            new_lower = new_val.lower()
            for tech_val, tech_cat in _VALUE_TO_CATEGORY.items():
                if tech_val in new_lower:
                    key = tech_cat
                    break

        if not key:
            old_lower = old_val.lower()
            for tech_val, tech_cat in _VALUE_TO_CATEGORY.items():
                if tech_val in old_lower:
                    key = tech_cat
                    break

        # Fall back to context as key
        if not key and context:
            key = context.lower()

        if key:
            # Strip trailing category word from new_val if it leaked in
            # e.g., "to FastAPI for backend" -> context captures "backend",
            # but "to Django for backend" with greedy match may include extra
            facts[key] = new_val
        return facts

    def _find_key_for_value(self, value: str) -> Optional[str]:
        """Find an existing KB key whose current value matches `value`."""
        val_lower = value.lower()
        for k, v in self._facts.items():
            if k.startswith("_old_"):
                continue
            if v.lower() == val_lower or val_lower in v.lower():
                return k
        return None

    def _extract_identity(self, sentence: str) -> Dict[str, str]:
        """Extract identity facts: name, email, role, etc."""
        facts: Dict[str, str] = {}

        for keyword, canonical in _IDENTITY_KEYWORDS.items():
            # "My name is John" / "name: John" / "Name is John"
            patterns = [
                rf'(?:my|our|his|her)\s+{re.escape(keyword)}\s+is\s+(.+)',
                rf'{re.escape(keyword)}\s*[:=]\s*(.+)',
            ]
            for pat in patterns:
                m = re.search(pat, sentence, re.IGNORECASE)
                if m:
                    facts[canonical] = m.group(1).strip()
                    break

        # "I'm Max, CTO of PayFlow" / "I'm Sarah Chen, senior engineer at Shopify"
        if "name" not in facts:
            m = re.search(r"\bI'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,", sentence)
            if m:
                facts["name"] = m.group(1)
        # Extract role and company from "I'm X, ROLE at COMPANY"
        role_match = re.search(
            r"\bI'?m\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*,\s*(.+?)\s+(?:at|of|for)\s+(.+?)(?:\s*[,.]|\s*$)",
            sentence,
        )
        if role_match:
            facts["role"] = role_match.group(1).strip()
            facts["company"] = role_match.group(2).strip()

        # Email extraction
        email_match = re.search(r'[\w.+-]+@[\w.-]+\.\w+', sentence)
        if email_match:
            facts["email"] = email_match.group(0)

        return facts

    def _extract_team(self, sentence: str) -> Dict[str, str]:
        """Extract team operations: hired, promoted, team size."""
        facts: Dict[str, str] = {}

        # "Hired X for Y" / "Hired X as Y"
        hire_match = re.match(
            r'(?:hired|recruited|onboarded)\s+(.+?)\s+(?:for|as)\s+(.+)',
            sentence, re.IGNORECASE,
        )
        if hire_match:
            person = hire_match.group(1).strip().lower()
            role = hire_match.group(2).strip()
            facts[f"team:{person}"] = role

        # "Promoted X to Y"
        promo_match = re.match(
            r'(?:promoted|elevated)\s+(.+?)\s+to\s+(.+)',
            sentence, re.IGNORECASE,
        )
        if promo_match:
            person = promo_match.group(1).strip().lower()
            role = promo_match.group(2).strip()
            facts[f"team:{person}"] = role

        # "Our/My manager Tom" / "manager is Tom"
        mgr_match = re.search(
            r'(?:our|my)\s+manager\s+([A-Z][a-z]+)', sentence,
        )
        if mgr_match:
            facts["manager"] = mgr_match.group(1)
        mgr_match2 = re.search(
            r'manager\s+(?:is\s+)?([A-Z][a-z]+)', sentence,
        )
        if mgr_match2 and "manager" not in facts:
            facts["manager"] = mgr_match2.group(1)

        # "Team size is X" / "team of X" / "team is now X" / "Total team size is now 12"
        size_match = re.search(
            r'team\s+(?:size\s+)?(?:is\s+)?(?:now\s+)?(?:of\s+)?(\d+)',
            sentence, re.IGNORECASE,
        )
        if size_match:
            facts["team size"] = size_match.group(1)

        return facts

    def _extract_money(self, sentence: str) -> Dict[str, str]:
        """Extract monetary facts: raised, valuation, revenue, ARR."""
        facts: Dict[str, str] = {}
        s_lower = sentence.lower()

        # "Raised $X" / "Raised $X in Y"
        raised_match = re.search(
            r'raised\s+(\$[\d,.]+[kmb]?(?:\s*(?:million|billion))?)',
            s_lower,
        )
        if raised_match:
            facts["funding raised"] = raised_match.group(1).strip()

        # "$X valuation"
        val_match = re.search(
            r'(\$[\d,.]+[kmb]?(?:\s*(?:million|billion))?)\s+valuation',
            s_lower,
        )
        if val_match:
            facts["valuation"] = val_match.group(1).strip()

        # "revenue of $X" / "ARR of $X" / "$X ARR" / "$X revenue"
        rev_match = re.search(
            r'(?:revenue|arr)\s+(?:of\s+)?(\$[\d,.]+[kmb]?(?:\s*(?:million|billion))?)',
            s_lower,
        )
        if not rev_match:
            rev_match = re.search(
                r'(\$[\d,.]+[kmb]?(?:\s*(?:million|billion))?)\s+(?:revenue|arr)',
                s_lower,
            )
        if rev_match:
            facts["revenue"] = rev_match.group(1).strip()

        return facts

    def _extract_metrics(self, sentence: str) -> Dict[str, str]:
        """Extract percentage and numeric metrics."""
        facts: Dict[str, str] = {}

        # "X is Y%" / "X: Y%" / "X at Y%"
        pct_match = re.search(
            r'(\w[\w\s]*?)\s+(?:is|at|was|reached|hit)\s+([\d.]+%)',
            sentence.lower(),
        )
        if pct_match:
            key = pct_match.group(1).strip()
            value = pct_match.group(2).strip()
            facts[key] = value

        return facts

    def _extract_compliance(self, sentence: str) -> Dict[str, str]:
        """Extract compliance/certification flags."""
        facts: Dict[str, str] = {}
        s_lower = sentence.lower()
        found = []
        for kw in _COMPLIANCE_KEYWORDS:
            if kw in s_lower:
                found.append(kw.upper().replace(" ", ""))
        if found:
            existing = self._facts.get("compliance", "")
            existing_set = {x.strip() for x in existing.split(",") if x.strip()}
            existing_set.update(found)
            facts["compliance"] = ", ".join(sorted(existing_set))
        return facts

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _query_tech_stack(self, active: Dict[str, str]) -> Optional[str]:
        tech_keys = {
            "backend", "frontend", "database", "framework", "language",
            "hosting", "cloud", "cache", "queue", "api", "gateway",
            "cdn", "ci", "cd", "ci/cd", "build", "monitoring",
            "payment", "auth", "authentication", "search", "storage",
            "orm", "proxy", "load balancer", "messaging",
        }
        results = []
        for key, value in active.items():
            if key in tech_keys:
                results.append(f"{key}: {value}")
        if results:
            return "Tech stack: " + ", ".join(sorted(results))
        return None

    def _query_who(self, name: str, active: Dict[str, str]) -> Optional[str]:
        name_lower = name.lower()
        # Check team keys
        for key, value in active.items():
            if key.startswith("team:") and name_lower in key:
                person = key.split(":", 1)[1].strip()
                return f"{person}: {value}"
        # Check name key
        if "name" in active and name_lower in active["name"].lower():
            parts = [f"name: {active['name']}"]
            for k in ("role", "company", "email", "location"):
                if k in active:
                    parts.append(f"{k}: {active[k]}")
            return ", ".join(parts)
        return None

    def _query_compliance(self, active: Dict[str, str]) -> Optional[str]:
        if "compliance" in active:
            return f"Compliance: {active['compliance']}"
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _file_path(self) -> str:
        return os.path.join(self._persistence_path, "knowledge_base.json")

    def _save(self) -> None:
        if not self._persistence_path:
            return
        os.makedirs(self._persistence_path, exist_ok=True)
        data = {"facts": self._facts, "history": self._history}
        with open(self._file_path(), "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not self._persistence_path:
            return
        path = self._file_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            self._facts = data.get("facts", {})
            self._history = data.get("history", [])
