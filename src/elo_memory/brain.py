"""EloBrain — agent middleware that combines episodic memory with knowledge.

Provides a simple ``think()`` loop and ``prepare() / process_turn()`` hooks
for framework-agnostic integration.

Also ships a self-bootstrapping CLI when run as ``python3 brain.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .memory.user_memory import UserMemory
from .memory.knowledge_base import KnowledgeBase
from .causal_engine import CausalInferenceEngine, CausalEngineConfig
from .world_simulator import WorldSimulator, WorldSimConfig
from .evolution import ParametricEvolution, EvolutionConfig
from .consolidation.dream_cycle import DreamConsolidation, DreamConfig
from .retrieval.prefetcher import PredictivePrefetcher, PrefetchConfig
from .auditor import MemoryAuditor
from .governor import MemoryGovernor, GovernorConfig
from .memory.episodic_store import EpisodicMemoryConfig

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
    if _HAS_INFO_RE.search(message):
        return False
    for pattern in _SKIP_MESSAGE_PATTERNS:
        if pattern.match(message):
            return True
    return False


def _extract_commitments(text: str) -> List[str]:
    """Extract assistant commitments/promises from a response."""
    commitments: List[str] = []
    for pattern in _COMMITMENT_PATTERNS:
        for m in pattern.finditer(text):
            commitments.append(m.group(0).strip())
    return commitments


# ===================================================================
# EloBrain class
# ===================================================================


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

        # Advanced modules
        self._causal = CausalInferenceEngine()
        self._world_sim = WorldSimulator(store=self._memory._store)
        self._evolution = ParametricEvolution()
        self._dream = DreamConsolidation()
        self._prefetcher = PredictivePrefetcher()
        self._auditor = MemoryAuditor(
            persistence_path=str(self._memory._user_dir),
        )
        self._governor = MemoryGovernor()

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
        kb_answer = self._kb.query(user_message)
        kb_facts = self._kb.get_all()

        # KB-guided recall: if KB has a value that matches the query,
        # boost episodes containing that value
        memories = self._memory.recall(user_message, k=k * 2)  # Get extra for boosting
        if kb_facts:
            query_lower = user_message.lower()
            boosted = []
            for text, score in memories:
                boost = 0
                for kb_key, kb_value in kb_facts.items():
                    val_lower = kb_value.lower()
                    if (
                        kb_key in query_lower
                        or val_lower in query_lower
                        or val_lower in text.lower()
                    ):
                        boost = max(boost, 0.15)
                boosted.append((text, score + boost))
            boosted.sort(key=lambda x: x[1], reverse=True)
            memories = boosted[:k]

        facts = self._memory.get_facts()
        profile = self._memory.get_profile()

        sections = [self._system_prompt]

        if kb_facts:
            sections.append("\n## Current facts (Knowledge Base)")
            for key, value in list(kb_facts.items())[:20]:
                # Format as natural language, not raw key: value
                formatted = self._format_kb_fact(key, value)
                sections.append(f"- {formatted}")

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

    @staticmethod
    def _format_kb_fact(key: str, value: str) -> str:
        """Format a KB fact as natural language."""
        # Convert raw key:value to readable sentences
        human_keys = {
            "name": f"Name is {value}",
            "email": f"Email is {value}",
            "role": f"Role is {value}",
            "company": f"Works at {value}",
            "location": f"Lives in {value}",
            "phone": f"Phone is {value}",
            "age": f"Age is {value}",
            "editor": f"Uses {value} as editor",
            "language": f"Uses {value} as programming language",
            "framework": f"Uses {value} as framework",
            "database": f"Uses {value} as database",
            "backend": f"Uses {value} for backend",
            "frontend": f"Uses {value} for frontend",
            "cloud": f"Uses {value} for cloud",
            "hosting": f"Uses {value} for hosting",
            "monitoring": f"Uses {value} for monitoring",
            "cache": f"Uses {value} for caching",
            "payment": f"Uses {value} for payments",
            "project": f"Working on: {value}",
            "considering/want to try": f"Considering trying {value}",
        }
        if key in human_keys:
            return human_keys[key]
        # Fallback: capitalize key and format
        readable_key = key.replace("_", " ").capitalize()
        return f"{readable_key}: {value}"

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
        self._kb.update(user_message)

        # Governor decides
        from elo_memory.governor import DecisionContext

        ctx = DecisionContext(surprise=0.5)
        action = self._governor.decide(ctx)

        if not _should_skip(user_message):
            result = self._memory.store(user_message)

            # Causal extraction
            if result.get("episode_id"):
                self._causal.ingest(user_message, result["episode_id"])
                self._auditor.add_to_chain(
                    self._memory._store._episode_index.get(result["episode_id"])
                )
                self._prefetcher.observe_query(user_message, topics=[], entities=[])

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
            # Advanced module stats
            "causal_graph": self._causal.get_statistics(),
            "governor": self._governor.get_policy_summary(),
            "world_sim": {
                "experiences": len(self._world_sim.experiences),
            },
            "evolution": self._evolution.get_statistics(),
            "dream": {"last_consolidation": None},
            "prefetcher": self._prefetcher.get_statistics(),
            "auditor": self._auditor.get_statistics(),
        }

    # ------------------------------------------------------------------
    # GDPR forget
    # ------------------------------------------------------------------

    def forget(self, text: str):
        """Supersede any memories matching *text* (GDPR erasure)."""
        text_lower = text.lower()
        # Remove from episodic memory
        for ep in self._memory._store.episodes:
            ep_text = self._memory._episode_text(ep).lower()
            if text_lower in ep_text:
                self._memory._superseded.add(ep.episode_id)
        # Also remove from KB — GDPR requires full erasure
        keys_to_remove = []
        for key, value in self._kb.get_all().items():
            if text_lower in key.lower() or text_lower in value.lower():
                keys_to_remove.append(key)
        for key in keys_to_remove:
            with self._kb._lock:
                self._kb._facts.pop(key, None)
                self._kb._facts.pop(f"_old_{key}", None)
        self._memory.save()
        if keys_to_remove:
            self._kb._save()

    # ------------------------------------------------------------------
    # Advanced module delegates
    # ------------------------------------------------------------------

    def get_causal_graph_stats(self) -> Dict[str, Any]:
        return self._causal.get_statistics()

    def counterfactual(self, removed_cause: str) -> Dict[str, Any]:
        return self._causal.counterfactual(removed_cause)

    def segment_experiences(self) -> List[Any]:
        return self._world_sim.segment_experiences()

    def replay_experience(self, experience_id: str) -> List[Dict]:
        return self._world_sim.replay(experience_id)

    def dream(self) -> Any:
        result = self._dream.dream(store=self._memory._store)
        return result

    def verify_integrity(self) -> Dict[str, Any]:
        return self._auditor.verify_chain()

    def evolve(self) -> Dict[str, Any]:
        return self._evolution.update_weights()

    def get_governor_policy(self) -> Dict[str, Any]:
        return self._governor.get_policy_summary()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Persist everything and release resources."""
        self._memory.close()
        # Persist advanced modules
        causal_path = self._memory._user_dir / "causal_graph.json"
        self._causal.save(causal_path)
        evolution_path = self._memory._user_dir / "evolution.json"
        self._evolution.save(evolution_path)
        # Persist auditor chain
        self._auditor.save()


# ===================================================================
# CLI server bootstrapping (when run as script)
# ===================================================================

ELO_DIR = os.path.expanduser("~/.elo-memory")
SERVER_SCRIPT = os.path.join(ELO_DIR, "memory_server.py")
SERVER_LOG = os.path.join(ELO_DIR, "server.log")
PID_FILE = os.path.join(ELO_DIR, "server.pid")
SERVER = "http://127.0.0.1:9876"
USER = os.environ.get("MEMORY_USER", "lorenc")


def _ensure_server():
    """Create directory and start server if needed. Returns True if server is up."""
    try:
        resp = urllib.request.urlopen(f"{SERVER}/health", timeout=2)
        return True
    except Exception:
        pass

    os.makedirs(ELO_DIR, exist_ok=True)
    os.makedirs(os.path.join(ELO_DIR, "transcripts"), exist_ok=True)

    server_script = SERVER_SCRIPT
    if not os.path.exists(server_script):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        server_script = os.path.join(this_dir, "memory_server.py")
    if not os.path.exists(server_script):
        try:
            import elo_memory

            pkg_dir = os.path.dirname(elo_memory.__file__)
            for candidate in [
                os.path.join(os.path.dirname(pkg_dir), "memory_server.py"),
                os.path.join(os.path.dirname(os.path.dirname(pkg_dir)), "memory_server.py"),
            ]:
                if os.path.exists(candidate):
                    server_script = candidate
                    break
        except ImportError:
            pass

    if not os.path.exists(server_script):
        return False

    import subprocess

    logf = open(SERVER_LOG, "w")
    proc = subprocess.Popen(
        [sys.executable, server_script, "--port", "9876", "--user", USER],
        stdout=logf,
        stderr=logf,
    )

    for _ in range(60):
        _time.sleep(0.5)
        try:
            resp = urllib.request.urlopen(f"{SERVER}/health", timeout=2)
            data = json.loads(resp.read())
            if data.get("ok"):
                with open(PID_FILE, "w") as f:
                    f.write(str(proc.pid))
                return True
        except Exception:
            continue

    return False


def _post(path, data):
    if not _ensure_server():
        _die("cannot start memory server — is memory_server.py next to brain.py?")
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        SERVER + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _get(path):
    if not _ensure_server():
        _die("cannot start memory server — is memory_server.py next to brain.py?")
    with urllib.request.urlopen(SERVER + path, timeout=10) as r:
        return json.loads(r.read())


def _die(msg):
    print(f"✗ {msg}", file=sys.stderr)
    sys.exit(1)


# ── CLI Actions ─────────────────────────────────────────────────────


def store(text):
    try:
        r = _post("/store", {"user": USER, "text": text})
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")
    if r.get("stored"):
        return f"stored [{', '.join(r.get('entities', []))}]" if r.get("entities") else "stored"
    return f"skipped ({r.get('reason', 'dup')})"


def recall(query, k=7):
    try:
        r = _get(f"/recall?user={USER}&q={urllib.parse.quote(query)}&k={k}")
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")
    return r.get("results", [])


def think(message, k=7, response=""):
    data = {"user": USER, "message": message, "k": k}
    if response:
        data["response"] = response
    try:
        return _post("/think", data)
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def extract(text):
    try:
        return _post("/extract", {"user": USER, "text": text})
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def update(old, new):
    try:
        r = _post("/update", {"user": USER, "old": old, "new": new})
        return "updated" if r.get("updated") else "failed"
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def forget(text):
    try:
        _post("/forget", {"user": USER, "text": text})
        return "forgotten"
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def briefing():
    try:
        return _get(f"/brief?user={USER}&summary=1")
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def new():
    try:
        return _get(f"/new?user={USER}")
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def facts():
    try:
        return _get(f"/facts?user={USER}")
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def stats():
    try:
        return _get(f"/stats?user={USER}")
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def dream():
    try:
        return _post("/dream", {"user": USER})
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def verify():
    try:
        return _post("/verify", {"user": USER})
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def log(query="", limit=30):
    try:
        q = f"user={USER}&limit={limit}"
        if query:
            q += f"&q={urllib.parse.quote(query)}"
        return _get(f"/transcript?{q}")
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")


def start(port=9876):
    """Start the server in background."""
    import subprocess

    try:
        _get("/health")
        return "already running"
    except Exception:
        pass
    script = os.path.expanduser("~/.elo-memory/memory_server.py")
    if not os.path.exists(script):
        return "memory_server.py not found"
    logf = open(os.path.expanduser("~/.elo-memory/server.log"), "w")
    proc = subprocess.Popen(
        [sys.executable, script, "--port", str(port), "--user", USER], stdout=logf, stderr=logf
    )
    for _ in range(30):
        _time.sleep(0.5)
        try:
            if _get("/health").get("ok"):
                pid_file = os.path.expanduser("~/.elo-memory/server.pid")
                with open(pid_file, "w") as f:
                    f.write(str(proc.pid))
                return f"started (pid {proc.pid})"
        except Exception:
            continue
    return "failed to start — check server.log"


def stop():
    import subprocess

    pid_file = os.path.expanduser("~/.elo-memory/server.pid")
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = int(f.read())
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        os.remove(pid_file)
        return "stopped"
    result = subprocess.run(["lsof", "-ti:9876"], capture_output=True, text=True)
    if result.stdout.strip():
        for pid in result.stdout.strip().split("\n"):
            os.kill(int(pid), 9)
        return "stopped"
    return "not running"


def status():
    try:
        up = _get("/health").get("uptime", 0)
        return f"up {int(up//60)}m {int(up%60)}s"
    except Exception:
        return "down"


# ── Dispatch ──────────────────────────────────────────────────────


def _dispatch_think(args):
    """Parse: think "message" [k] [--response "text"]"""
    message_parts = []
    k = 7
    response = ""
    i = 0
    while i < len(args):
        if args[i] == "--response" and i + 1 < len(args):
            response = args[i + 1]
            i += 2
        else:
            try:
                k = int(args[i])
                i += 1
            except ValueError:
                message_parts.append(args[i])
                i += 1
    message = " ".join(message_parts)
    if not message:
        _die("think requires a message")
    return think(message, k, response)


def _dispatch_update(args):
    old = new = None
    i = 0
    while i < len(args):
        if args[i] == "--old" and i + 1 < len(args):
            old = args[i + 1]
            i += 2
        elif args[i] == "--new" and i + 1 < len(args):
            new = args[i + 1]
            i += 2
        else:
            i += 1
    if not old or not new:
        _die("update requires --old and --new")
    return update(old, new)


def _dispatch_start(args):
    port = 9876
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            i += 1
    return start(port)


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    dispatch = {
        "store": lambda: store(" ".join(rest)),
        "recall": lambda: recall(rest[0], int(rest[1]) if len(rest) > 1 else 7),
        "think": lambda: _dispatch_think(rest),
        "extract": lambda: extract(" ".join(rest)),
        "update": lambda: _dispatch_update(rest),
        "forget": lambda: forget(" ".join(rest)),
        "briefing": lambda: briefing(),
        "new": lambda: new(),
        "facts": lambda: facts(),
        "stats": lambda: stats(),
        "dream": lambda: dream(),
        "verify": lambda: verify(),
        "log": lambda: log(" ".join(rest) if rest else ""),
        "start": lambda: _dispatch_start(rest),
        "stop": lambda: stop(),
        "status": lambda: status(),
    }

    fn = dispatch.get(cmd)
    if not fn:
        sys.exit(1)

    result = fn()

    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
