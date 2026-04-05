#!/usr/bin/env python3
"""
My Memory Brain — self-bootstrapping CLI.

Zero setup: creates ~/.elo-memory/, starts the server, runs your command.
For a new user: python3 brain.py store "hello" — it handles everything.

Usage:  python3 brain.py <action> [args...]
  store  "text"           → store a fact
  recall "query" [k]      → search memories
  think  "msg" [k]        → full turn
  think  "msg" --response "text"  → turn with answer
  extract "text"          → pull facts from text
  update  --old X --new Y → correct a memory
  forget  "text"          → erase matching
  briefing                → everything I know (with summary)
  new                     → what changed since last session
  facts                   → active facts
  stats                   → brain stats
  dream                   → consolidate
  verify                  → integrity check
  log                     → recent conversations
  start [--port N]        → start server
  stop                    → stop server
  status                  → server alive?
"""

import json
import os
import sys
import urllib.request
import urllib.error
import urllib.parse

# ── Paths ─────────────────────────────────────────────────────────

ELO_DIR = os.path.expanduser("~/.elo-memory")
# Also look within the package
SERVER_SCRIPT = os.path.join(ELO_DIR, "memory_server.py")
SERVER_LOG = os.path.join(ELO_DIR, "server.log")
PID_FILE = os.path.join(ELO_DIR, "server.pid")
SERVER = "http://127.0.0.1:9876"
USER = os.environ.get("MEMORY_USER", "lorenc")


# ── Bootstrapping ─────────────────────────────────────────────────

def _ensure_server():
    """Create directory and start server if needed. Returns True if server is up."""
    # Check if already running
    try:
        resp = urllib.request.urlopen(f"{SERVER}/health", timeout=2)
        return True
    except Exception:
        pass

    # Create directory structure
    os.makedirs(ELO_DIR, exist_ok=True)
    os.makedirs(os.path.join(ELO_DIR, "transcripts"), exist_ok=True)

    # Find memory_server.py
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

    # Start server
    logf = open(SERVER_LOG, "w")
    import subprocess
    proc = subprocess.Popen(
        [sys.executable, server_script, "--port", "9876", "--user", USER],
        stdout=logf, stderr=logf,
    )

    # Wait for it to come up (model loading takes 10-25s)
    import time
    for _ in range(60):
        time.sleep(0.5)
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


# ── HTTP helpers ──────────────────────────────────────────────────

def _post(path, data):
    if not _ensure_server():
        _die("cannot start memory server — is memory_server.py next to brain.py?")
    body = json.dumps(data).encode()
    req = urllib.request.Request(SERVER + path, data=body, headers={"Content-Type": "application/json"}, method="POST")
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


# ── Actions ───────────────────────────────────────────────────────

def store(text):
    try:
        r = _post("/store", {"user": USER, "text": text})
    except urllib.error.URLError:
        _die("memory server is down — run `brain.py start`")
    if r.get("stored"):
        return f"stored [{', '.join(r.get('entities', []))}]" if r.get('entities') else "stored"
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
    import subprocess, time
    try:
        _get("/health")
        return "already running"
    except Exception:
        pass
    script = os.path.expanduser("~/.elo-memory/memory_server.py")
    if not os.path.exists(script):
        return "memory_server.py not found"
    logf = open(os.path.expanduser("~/.elo-memory/server.log"), "w")
    proc = subprocess.Popen([sys.executable, script, "--port", str(port), "--user", USER], stdout=logf, stderr=logf)
    for _ in range(30):
        time.sleep(0.5)
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

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    dispatch = {
        "store":    lambda: store(" ".join(rest)),
        "recall":   lambda: recall(rest[0], int(rest[1]) if len(rest) > 1 else 7),
        "think":    lambda: _dispatch_think(rest),
        "extract":  lambda: extract(" ".join(rest)),
        "update":   lambda: _dispatch_update(rest),
        "forget":   lambda: forget(" ".join(rest)),
        "briefing": lambda: briefing(),
        "new":      lambda: new(),
        "facts":    lambda: facts(),
        "stats":    lambda: stats(),
        "dream":    lambda: dream(),
        "verify":   lambda: verify(),
        "log":      lambda: log(" ".join(rest) if rest else ""),
        "start":    lambda: _dispatch_start(rest),
        "stop":     lambda: stop(),
        "status":   lambda: status(),
    }

    fn = dispatch.get(cmd)
    if not fn:
        sys.exit(1)

    result = fn()

    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, default=str))


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


if __name__ == "__main__":
    main()
