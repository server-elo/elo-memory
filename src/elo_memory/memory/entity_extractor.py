"""Lightweight regex-based entity extraction.

Extracts emails, names, dates, numbers with units, URLs, and version strings
from free-form text.  Designed for speed over recall -- we compile every pattern
once and guard against the most common false positives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# ── skip / reject sets ───────────────────────────────────────────────

_SKIP_WORDS: frozenset[str] = frozenset(
    {
        # Common sentence starters
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "How",
        "It",
        "Its",
        "My",
        "Your",
        "Our",
        "His",
        "Her",
        "Their",
        "We",
        "You",
        "They",
        "He",
        "She",
        "If",
        "But",
        "And",
        "Or",
        "So",
        "As",
        "At",
        "By",
        "In",
        "On",
        "To",
        "For",
        "From",
        "With",
        "About",
        "After",
        "Before",
        "Between",
        # Action verbs / gerunds that start with a capital letter
        "Using",
        "Keeping",
        "Making",
        "Getting",
        "Going",
        "Having",
        "Running",
        "Building",
        "Creating",
        "Starting",
        "Setting",
        "Adding",
        "Updating",
        "Moving",
        "Looking",
        "Working",
        "Thinking",
        "Trying",
        "Finding",
        "Taking",
        "Coming",
        "Calling",
        "Pulling",
        "Pushing",
        "Fixing",
        "Testing",
        "Writing",
        "Reading",
        "Sending",
        "Deploying",
        "Launching",
        "Here",
        "Now",
        "Then",
        "Also",
        "Just",
        "Very",
        "Each",
        "Every",
        "Some",
        "Any",
        "All",
        "Both",
        "Most",
        "Can",
        "Could",
        "Should",
        "Would",
        "Will",
        "May",
        "Might",
        "Do",
        "Does",
        "Did",
        "Has",
        "Have",
        "Had",
        "Is",
        "Are",
        "Was",
        "Were",
        "Be",
        "Been",
        "Being",
        "Not",
        "No",
        "Yes",
        "Let",
        "See",
        "Try",
        "Use",
        "Need",
        "Want",
        "Like",
        "Got",
        "Get",
        "Set",
        "Put",
        "Run",
        "Ran",
        "However",
        "Therefore",
        "Furthermore",
        "Moreover",
        "Since",
        "Until",
        "While",
        "During",
        "Because",
        "Still",
        "Already",
        "Maybe",
        "Perhaps",
        "Much",
        "Many",
        "Few",
        "Several",
        "Other",
        "Another",
        # Days / months handled separately but also skip them
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        # Common title-case words that aren't names
        "Today",
        "Tomorrow",
        "Yesterday",
        "Next",
        "Last",
        "Previous",
        "Current",
    }
)

# Role words that precede a single name: "Engineer Maria", "wife Julia"
_ROLE_WORDS: frozenset[str] = frozenset(
    {
        "engineer",
        "manager",
        "director",
        "ceo",
        "cto",
        "cfo",
        "coo",
        "vp",
        "president",
        "lead",
        "senior",
        "junior",
        "intern",
        "doctor",
        "dr",
        "professor",
        "prof",
        "wife",
        "husband",
        "daughter",
        "son",
        "brother",
        "sister",
        "friend",
        "colleague",
        "coworker",
        "boss",
        "mentor",
        "mentee",
        "hired",
        "promoted",
        "fired",
        "met",
        "called",
        "emailed",
        "architect",
        "designer",
        "developer",
        "analyst",
        "consultant",
        "nurse",
        "therapist",
        "coach",
        "teacher",
        "tutor",
        "captain",
        "lieutenant",
        "sergeant",
        "uncle",
        "aunt",
        "cousin",
        "nephew",
        "niece",
        "partner",
        "fiancee",
        "fiance",
        "mr",
        "mrs",
        "ms",
        "miss",
    }
)

# Tech products / frameworks / places that look like names
_NOT_NAMES: frozenset[str] = frozenset(
    {
        # Tech products
        "Prisma",
        "Redis",
        "Vercel",
        "Docker",
        "Firebase",
        "React",
        "Angular",
        "Vue",
        "Svelte",
        "Django",
        "Flask",
        "Rails",
        "Postgres",
        "Mongo",
        "MySQL",
        "SQLite",
        "Kafka",
        "Nginx",
        "Jenkins",
        "Travis",
        "Heroku",
        "Netlify",
        "Gatsby",
        "Hugo",
        "Webpack",
        "Babel",
        "Eslint",
        "Prettier",
        "Tailwind",
        "Bootstrap",
        "Express",
        "Fastify",
        "Deno",
        "Remix",
        "Kubernetes",
        "Terraform",
        "Ansible",
        "Vagrant",
        "Elasticsearch",
        "Grafana",
        "Prometheus",
        "Datadog",
        "Stripe",
        "Twilio",
        "Sentry",
        "Supabase",
        "Hasura",
        "Apollo",
        "GraphQL",
        "TypeScript",
        "JavaScript",
        "Python",
        "Golang",
        "Kotlin",
        "Swift",
        "Rust",
        "Chrome",
        "Firefox",
        "Safari",
        "Opera",
        "Edge",
        "Ubuntu",
        "Debian",
        "Alpine",
        "CentOS",
        "Fedora",
        "GitHub",
        "GitLab",
        "Bitbucket",
        "Jira",
        "Slack",
        "Notion",
        "Figma",
        "Postman",
        "Insomnia",
        "Node",
        "Bun",
        "Yarn",
        "Pnpm",
        "Azure",
        "Lambda",
        "Cloudflare",
        "Akamai",
        "Cypress",
        "Playwright",
        "Selenium",
        "Mocha",
        "Jest",
        # Place-name components (these appear as first word of multi-word places)
        "San",
        "Los",
        "Las",
        "New",
        "North",
        "South",
        "East",
        "West",
        "Saint",
        "Fort",
        "Mount",
        "Lake",
        "Port",
        "Cape",
        "El",
        "La",
        "De",
        "Del",
    }
)


# ── compiled patterns ────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]}]+")

_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

_LONG_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"
)

_SHORT_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)" r"\s+\d{1,2}(?:,\s*\d{4})?\b"
)

_RELATIVE_DATE_RE = re.compile(
    r"\b(?:yesterday|today|tomorrow|"
    r"(?:next|last|this)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday|week|month|year|quarter))\b",
    re.IGNORECASE,
)

_NUMBER_UNIT_RE = re.compile(
    r"(?:\$[\d,.]+[kKmMbB]?\b)"  # $500k, $1,000
    r"|(?:\b\d[\d,.]*\s*(?:%|percent))"  # 50%, 100 percent
    r"|(?:\b\d[\d,.]*\s*(?:[kKmMgGtT][bB])\b)"  # 42GB, 1.5TB
    r"|(?:\b\d[\d,.]*\s+(?:users?|people|employees?|"  # 100 users
    r"customers?|items?|requests?|queries?|"
    r"hours?|minutes?|seconds?|days?|weeks?|months?|years?|"
    r"miles?|kilometers?|meters?|feet|"
    r"bytes?|pixels?|points?|"
    r"sessions?|transactions?|events?|records?|"
    r"pages?|files?|lines?|commits?|"
    r"units?|pieces?|copies?|instances?|servers?|nodes?|"
    r"tickets?|issues?|bugs?|tasks?|stories?|sprints?))\b"
)

_VERSION_RE = re.compile(r"\b(?:v|version\s*)\d+(?:\.\d+)*\b", re.IGNORECASE)

# Multi-word capitalized names: two or more consecutive capitalised words
_NAME_MULTI_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Single name after a role word: "Engineer Maria", "wife Julia"
_NAME_ROLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_ROLE_WORDS)) + r")" r"\s+([A-Z][a-z]{1,})\b",
    re.IGNORECASE,
)


# ── extractor ─────────────────────────────────────────────────────────


class EntityExtractor:
    """Lightweight regex-based entity extraction."""

    def __init__(self) -> None:
        # All patterns are module-level constants; nothing to do here.
        pass

    # ── public API ────────────────────────────────────────────────────

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Return extracted entities grouped by type.

        Returns a dict with keys: emails, names, dates, numbers, urls, versions.
        Each value is a deduplicated list (insertion order preserved).
        """
        return {
            "emails": self._extract_emails(text),
            "names": self._extract_names(text),
            "dates": self._extract_dates(text),
            "numbers": self._extract_numbers(text),
            "urls": self._extract_urls(text),
            "versions": self._extract_versions(text),
        }

    def extract_flat(self, text: str) -> List[str]:
        """All entities as a single deduped list."""
        seen: set[str] = set()
        result: list[str] = []
        for values in self.extract(text).values():
            for v in values:
                if v not in seen:
                    seen.add(v)
                    result.append(v)
        return result

    # ── private helpers ───────────────────────────────────────────────

    @staticmethod
    def _dedup(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            normed = item.strip()
            if normed and normed not in seen:
                seen.add(normed)
                out.append(normed)
        return out

    def _extract_emails(self, text: str) -> List[str]:
        return self._dedup(_EMAIL_RE.findall(text))

    def _extract_urls(self, text: str) -> List[str]:
        # Strip trailing punctuation that gets captured
        urls = []
        for m in _URL_RE.findall(text):
            url = m.rstrip(".,;:!?")
            urls.append(url)
        return self._dedup(urls)

    def _extract_dates(self, text: str) -> List[str]:
        dates: list[str] = []
        dates.extend(_ISO_DATE_RE.findall(text))
        dates.extend(_LONG_DATE_RE.findall(text))
        dates.extend(_SHORT_DATE_RE.findall(text))
        dates.extend(_RELATIVE_DATE_RE.findall(text))
        return self._dedup(dates)

    def _extract_numbers(self, text: str) -> List[str]:
        return self._dedup(_NUMBER_UNIT_RE.findall(text))

    def _extract_versions(self, text: str) -> List[str]:
        return self._dedup(_VERSION_RE.findall(text))

    def _extract_names(self, text: str) -> List[str]:
        names: list[str] = []

        # 1) Multi-word capitalised sequences
        for m in _NAME_MULTI_RE.finditer(text):
            candidate = m.group(1)
            words = candidate.split()
            # Strip leading skip words
            while words and words[0] in _SKIP_WORDS:
                words = words[1:]
            if len(words) >= 2:
                name = " ".join(words)
                # Reject if any word is a known non-name
                if not any(w in _NOT_NAMES for w in words):
                    names.append(name)

        # 2) Single name after role word
        for m in _NAME_ROLE_RE.finditer(text):
            name = m.group(2)
            if name not in _SKIP_WORDS and name not in _NOT_NAMES:
                names.append(name)

        return self._dedup(names)
