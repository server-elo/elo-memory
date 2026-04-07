#!/usr/bin/env python3
"""
Memory Server — persistent Elo Memory brain.

Loads the embedding model once and stays alive.
CLI clients hit a local HTTP API — no more 12s cold starts per command.
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time as _time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List
from urllib.parse import urlparse, parse_qs

from elo_memory import EloBrain

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("memory_server")


# ── Globals ──────────────────────────────────────────────────────

_brain_lock = threading.Lock()
_user_brains: Dict[str, EloBrain] = {}
MEMORY_ROOT = os.path.expanduser("~/.elo-memory")
_SERVER_START = _time.time()


def get_brain(user_id: str) -> EloBrain:
    with _brain_lock:
        if user_id not in _user_brains:
            user_dir = os.path.join(MEMORY_ROOT, user_id)
            os.makedirs(user_dir, exist_ok=True)
            brain = EloBrain(user_id, persistence_path=MEMORY_ROOT)

            # Chain all existing episodes that aren't already chained
            existing_ids = {l.episode_id for l in brain._auditor._chain}
            for ep in brain._memory._store.episodes:
                if ep.episode_id not in existing_ids:
                    brain._auditor.add_to_chain(ep)

            _user_brains[user_id] = brain
            log.info(
                "Brain loaded for user '%s' (%d episodes, %d chained)",
                user_id,
                len(brain._memory._store.episodes),
                len(brain._auditor._chain),
            )
        return _user_brains[user_id]


# ── Sentence splitting ───────────────────────────────────────────

_ABBREV = {
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Prof",
    "Sr",
    "Jr",
    "vs",
    "etc",
    "Inc",
    "Ltd",
    "Corp",
    "Co",
    "e.g",
    "i.e",
    "St",
    "Mt",
    "Ave",
    "Blvd",
    "Dept",
    "No",
    "Vol",
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
    "U.S",
    "U.K",
}


def split_sentences(text: str) -> List[str]:
    """Split text into sentences for granular storage."""
    stripped = text.strip()
    if not stripped:
        return []

    punct_count = text.count(".") + text.count("!") + text.count("?")
    if punct_count <= 1:
        return [stripped]

    sentences: List[str] = []
    current = ""
    tokens = text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        current += (" " if current else "") + token

        if current.endswith((".", "!", "?")):
            last_word = current.rstrip(".!?").split()[-1] if current.rstrip(".!?").split() else ""
            clean_word = last_word.rstrip(".")

            if clean_word in _ABBREV:
                continue

            if len(current) > 3:
                if not current.endswith((".", "!", "?")):
                    current += "."
                sentences.append(current)
                current = ""

    if current.strip():
        remainder = current.strip()
        if not remainder.endswith((".", "!", "?")):
            remainder += "."
        sentences.append(remainder)

    return sentences if sentences else [stripped]


# ── Transcript log ───────────────────────────────────────────────


class TranscriptLog:
    def __init__(self, path: str):
        self._path = path
        self._entries: List[Dict] = []
        self._lock = threading.Lock()
        os.makedirs(path, exist_ok=True)
        self._load()

    def _load(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = os.path.join(self._path, f"{today}.jsonl")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    def _current_file(self) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self._path, f"{today}.jsonl")

    def append(self, user_id: str, message: str, response: str = "", metadata: Dict = None):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user": user_id,
            "message": message,
            "response": response,
            "meta": metadata or {},
        }
        with self._lock:
            self._entries.append(entry)
            with open(self._current_file(), "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def search(self, query: str = "", user_id: str = "", limit: int = 50) -> List[Dict]:
        results = self._entries
        if query:
            q_lower = query.lower()
            results = [
                e
                for e in results
                if q_lower in e.get("message", "").lower()
                or q_lower in e.get("response", "").lower()
            ]
        if user_id:
            results = [e for e in results if e.get("user") == user_id]
        return list(reversed(results[-limit:]))

    def today(self, user_id: str = "") -> List[Dict]:
        entries = (
            [e for e in self._entries if e.get("user") == user_id] if user_id else self._entries
        )
        return list(reversed(entries[-20:]))


_transcripts = TranscriptLog(os.path.join(MEMORY_ROOT, "transcripts"))


# ── HTTP Handler ─────────────────────────────────────────────────


class MemoryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _json_response(self, code: int, data: Dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _parse_json(self) -> Dict:
        body = self._read_body()
        return json.loads(body) if body else {}

    # ── GET ──────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        try:
            if path == "/health":
                self._json_response(200, {"ok": True, "uptime": _time.time() - _SERVER_START})

            elif path == "/brief":
                user_id = params.get("user", ["default"])[0]
                show_summary = params.get("summary", ["0"])[0] == "1"
                brain = get_brain(user_id)
                profile = brain._memory.get_profile()
                facts = brain._memory.get_facts()
                kb = brain._kb.get_all()
                store = brain._memory._store

                all_topics: set[str] = set()
                for ep in store.episodes:
                    if ep.episode_id not in brain._memory._superseded:
                        for t in ep.metadata.get("topics", []):
                            all_topics.add(t)

                data = {
                    "user": user_id,
                    "total_episodes": profile.get("total_memories", 0),
                    "active_memories": len(facts),
                    "superseded": len(brain._memory._superseded),
                    "sessions": profile.get("sessions_count", 1),
                    "first_seen": profile.get("first_seen"),
                    "last_seen": profile.get("last_seen"),
                    "topics": sorted(all_topics),
                    "entities": profile.get("entities", {}),
                    "knowledge_base": kb,
                    "facts_preview": facts[:20],
                    "recent_conversations": _transcripts.today(user_id)[:10],
                    "causal_stats": brain._causal.get_statistics(),
                    "governor_stats": brain._governor.get_policy_summary(),
                }

                if show_summary:
                    # Build human-readable summary
                    lines = []
                    lines.append(f"Briefing: {user_id}")
                    lines.append(f"{'─' * 40}")
                    lines.append(
                        f"{data['total_episodes']} memories · {data['active_memories']} active · {data['superseded']} superseded"
                    )
                    lines.append(f"Topics: {', '.join(data['topics'])}")

                    # Entities
                    ents = data["entities"]
                    if ents:
                        parts = []
                        for cat, vals in sorted(ents.items()):
                            if vals:
                                parts.append(f"{cat}: {', '.join(vals[:5])}")
                        if parts:
                            lines.append(f"Known: {' | '.join(parts)}")

                    # Knowledge base
                    if kb:
                        lines.append(f"KB: {' | '.join(f'{k}={v}' for k, v in sorted(kb.items()))}")

                    # Causal
                    causal = data.get("causal_stats", {})
                    if causal.get("edges", 0) > 0:
                        lines.append(f"Causal: {causal['edges']} links")
                        for link in causal.get("strongest_links", [])[:3]:
                            lines.append(f"  {link['cause']} → {link['effect']}")

                    # Recent facts (top 10)
                    if facts:
                        lines.append(f"Recent facts ({len(facts)} total):")
                        for text, score in sorted(facts, key=lambda x: x[1], reverse=True)[:10]:
                            lines.append(f"  · {text[:120]}")

                    data["summary"] = "\n".join(lines)

                self._json_response(200, data)

            elif path == "/recall":
                user_id = params.get("user", ["default"])[0]
                query = params.get("q", [""])[0]
                k = int(params.get("k", [7])[0])
                brain = get_brain(user_id)
                results = brain._memory.recall(query, k=k)
                self._json_response(
                    200,
                    {
                        "query": query,
                        "results": [{"text": t, "score": s} for t, s in results],
                        "count": len(results),
                    },
                )

            elif path == "/facts":
                user_id = params.get("user", ["default"])[0]
                brain = get_brain(user_id)
                facts = brain._memory.get_facts()
                self._json_response(
                    200,
                    {
                        "facts": [
                            {"text": t, "importance": s}
                            for t, s in sorted(facts, key=lambda x: x[1], reverse=True)
                        ],
                        "count": len(facts),
                    },
                )

            elif path == "/stats":
                user_id = params.get("user", ["default"])[0]
                brain = get_brain(user_id)
                profile = brain._memory.get_profile()
                self._json_response(
                    200,
                    {
                        "user": user_id,
                        "episodes": profile.get("total_memories", 0),
                        "superseded": len(brain._memory._superseded),
                        "sessions": profile.get("sessions_count", 1),
                        "causal_graph": brain._causal.get_statistics(),
                        "audit_chain": brain._auditor.get_statistics(),
                        "governor": brain._governor.get_policy_summary(),
                    },
                )

            elif path == "/new":
                """What changed since last session?"""
                user_id = params.get("user", ["default"])[0]
                brain = get_brain(user_id)
                # Get conversations from today that have "store" action (new memories)
                today_convos = _transcripts.today(user_id)
                new_stores = [e for e in today_convos if e.get("meta", {}).get("action") == "store"]
                # Get the last session end time from metadata
                last_seen = brain._memory._last_seen
                brain._memory.save()

                self._json_response(
                    200,
                    {
                        "new_memories": len(new_stores),
                        "new_items": [
                            {"message": e.get("message", ""), "ts": e.get("ts", "")}
                            for e in new_stores[-10:]
                        ],
                        "last_seen": last_seen,
                        "total_active": len(brain._memory.get_facts()),
                        "superseded_since_last": len(brain._memory._superseded),
                    },
                )

            elif path == "/transcript":
                user_id = params.get("user", ["default"])[0]
                query = params.get("q", [""])[0]
                limit = int(params.get("limit", [50])[0])
                results = _transcripts.search(query=query, user_id=user_id, limit=limit)
                self._json_response(200, {"entries": results, "count": len(results)})

            elif path == "/causal":
                user_id = params.get("user", ["default"])[0]
                brain = get_brain(user_id)
                self._json_response(
                    200,
                    {
                        "graph": brain._causal.get_statistics(),
                        "contradictions": brain._causal.get_contradictions(),
                    },
                )

            else:
                self._json_response(404, {"error": "not found"})

        except Exception as e:
            log.exception("GET %s failed", path)
            self._json_response(500, {"error": str(e)})

    # ── POST ─────────────────────────────────────────────────────

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._parse_json()
            user_id = body.get("user", "default")

            if path == "/store":
                text = body.get("text", "")
                if not text:
                    self._json_response(400, {"error": "missing 'text' field"})
                    return
                brain = get_brain(user_id)

                # Update KB from full paragraph
                brain._kb.update(text)

                # Track total episodes before this store to detect derived facts
                total_before = len(brain._memory._store.episodes)

                # Split into sentences for granular storage + causal extraction
                sentences = split_sentences(text)
                stored_count = 0
                entities = []
                all_topics = []

                for sentence in sentences:
                    result = brain._memory.store(sentence)
                    if result.get("stored"):
                        stored_count += 1
                        entities.extend(result.get("entities", []))
                        all_topics.extend(result.get("topics", []))

                        # Chain the main episode
                        eid = result.get("episode_id")
                        if eid:
                            ep = brain._memory._store._episode_index.get(eid)
                            if ep:
                                brain._auditor.add_to_chain(ep)
                            # Causal extraction per sentence (not full paragraph)
                            brain._causal.ingest(sentence, eid)
                            brain._intelligence.extract_causal_links(sentence, eid)

                # Chain any derived facts that were auto-created
                total_after = len(brain._memory._store.episodes)
                for i in range(total_before, total_after):
                    ep = brain._memory._store.episodes[i]
                    if ep.episode_id not in {l.episode_id for l in brain._auditor._chain}:
                        brain._auditor.add_to_chain(ep)

                brain._memory.save()
                _transcripts.append(
                    user_id,
                    text,
                    metadata={
                        "action": "store",
                        "sentences": stored_count,
                        "topics": list(set(all_topics)),
                    },
                )

                self._json_response(
                    200,
                    {
                        "stored": stored_count > 0,
                        "sentences": stored_count,
                        "entities": list(set(entities)),
                        "topics": list(set(all_topics)),
                    },
                )

            elif path == "/think":
                user_message = body.get("message", "")
                if not user_message:
                    self._json_response(400, {"error": "missing 'message' field"})
                    return
                k = int(body.get("k", 7))
                brain = get_brain(user_id)
                context = brain.prepare(user_message, k=k)

                # Don't store the question itself as a memory — only the response
                assistant_response = body.get("response", "")
                if assistant_response:
                    brain.process_turn(user_message, assistant_response=assistant_response)
                else:
                    # Just recall, don't store
                    brain._kb.update(user_message)

                brain._memory.save()

                _transcripts.append(
                    user_id,
                    user_message,
                    response=body.get("response", ""),
                    metadata={
                        "memories_used": context.get("memories_used", 0),
                        "suggestions": context.get("suggestions", []),
                        "knowledge_gaps": context.get("knowledge_gaps", []),
                    },
                )

                self._json_response(
                    200,
                    {
                        "stored": True,
                        "context": {
                            "memories_retrieved": context.get("memories_used", 0) or 0,
                            "knowledge_base_size": len(context.get("system", "")),
                            "suggestions": context.get("suggestions", []),
                            "knowledge_gaps": context.get("knowledge_gaps", []),
                        },
                    },
                )

            elif path == "/extract":
                text = body.get("text", "")
                if not text:
                    self._json_response(400, {"error": "missing 'text' field"})
                    return
                brain = get_brain(user_id)

                # Causal + KB from full text
                brain._causal.ingest(text, "")
                brain._kb.update(text)

                # Store as sentences
                sentences = split_sentences(text)
                stored_count = 0
                entities = []
                for sentence in sentences:
                    result = brain._memory.store(sentence)
                    if result.get("stored"):
                        stored_count += 1
                        entities.extend(result.get("entities", []))
                        eid = result.get("episode_id")
                        if eid:
                            ep = brain._memory._store._episode_index.get(eid)
                            if ep:
                                brain._auditor.add_to_chain(ep)
                                brain._intelligence.extract_causal_links(sentence, eid)

                brain._memory.save()
                self._json_response(
                    200,
                    {
                        "stored": stored_count > 0,
                        "sentences": stored_count,
                        "entities": list(set(entities)),
                        "knowledge_base": brain._kb.get_all(),
                    },
                )

            elif path == "/update":
                old_text = body.get("old", "")
                new_text = body.get("new", "")
                if not old_text or not new_text:
                    self._json_response(400, {"error": "requires 'old' and 'new' fields"})
                    return
                brain = get_brain(user_id)
                with brain._memory._lock:
                    for ep in brain._memory._store.episodes:
                        if old_text.lower() in brain._memory._episode_text(ep).lower():
                            brain._memory._superseded.add(ep.episode_id)
                sentences = split_sentences(new_text)
                stored_count = 0
                for sentence in sentences:
                    result = brain._memory.store(sentence)
                    if result.get("stored"):
                        stored_count += 1
                        eid = result.get("episode_id")
                        if eid:
                            ep = brain._memory._store._episode_index.get(eid)
                            if ep:
                                brain._auditor.add_to_chain(ep)
                brain._memory.save()
                self._json_response(200, {"updated": stored_count > 0, "sentences": stored_count})

            elif path == "/forget":
                text = body.get("text", "")
                if not text:
                    self._json_response(400, {"error": "missing 'text' field"})
                    return
                brain = get_brain(user_id)
                brain.forget(text)
                brain._memory.save()
                self._json_response(200, {"forgotten": text})

            elif path == "/dream":
                brain = get_brain(user_id)
                result = brain.dream()
                brain._memory.save()
                self._json_response(
                    200,
                    {
                        "episodes_replayed": result.episodes_replayed,
                        "synthetic_generated": result.synthetic_generated,
                        "principles_extracted": result.principles_extracted,
                        "skills_learned": result.skills_learned,
                        "episodes_pruned": result.episodes_pruned,
                        "duration_seconds": result.duration_seconds,
                    },
                )

            elif path == "/verify":
                brain = get_brain(user_id)
                self._json_response(200, brain.verify_integrity())

            else:
                self._json_response(404, {"error": "not found"})

        except Exception as e:
            log.exception("POST %s failed", path)
            self._json_response(500, {"error": str(e)})


# ── Main ─────────────────────────────────────────────────────────


def _shutdown():
    """Save all brains and exit cleanly."""
    print("\nShutting down, saving brains...", file=sys.stderr)
    for brain in _user_brains.values():
        brain.close()
    print("Done.", file=sys.stderr)
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, lambda s, f: _shutdown())
    signal.signal(signal.SIGINT, lambda s, f: _shutdown())

    parser = argparse.ArgumentParser(description="Memory Server — persistent Elo Memory brain")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--user", default="default")
    args = parser.parse_args()

    print(f"Loading brain for user '{args.user}'...", file=sys.stderr)
    get_brain(args.user)
    print(f"Server starting on {args.host}:{args.port}", file=sys.stderr)

    server = HTTPServer((args.host, args.port), MemoryHandler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
