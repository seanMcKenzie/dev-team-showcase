#!/usr/bin/env python3
"""
agent-status-server.py
K2S0 Agent Dashboard - Local Status Server
Runs on port 7800, serves GET /agents with live workspace data.

Start: python3 agent-status-server.py
"""

import json
import os
import sys
import time
import glob
import threading
import collections
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

OPENCLAW_JSON = Path.home() / ".openclaw" / "openclaw.json"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


def load_agent_models():
    """Load per-agent model from openclaw.json, falling back to global default."""
    models = {}
    try:
        with open(OPENCLAW_JSON, "r", encoding="utf-8") as f:
            config = json.load(f)
        global_model = (
            config.get("agents", {})
            .get("defaults", {})
            .get("model", {})
            .get("primary", DEFAULT_MODEL)
        )
        for agent in config.get("agents", {}).get("list", []):
            aid = agent.get("id")
            if aid:
                models[aid] = agent.get("model", global_model)
        # default fallback for any agent not in config
        models["__default__"] = global_model
    except Exception:
        models["__default__"] = DEFAULT_MODEL
    return models


AGENT_MODELS = load_agent_models()

# ============================================================
# CONFIG
# ============================================================
PORT = 7800
ACTIVE_THRESHOLD_SECONDS = 300  # 5 minutes
ACTIVITY_LOG_MAX = 50
POLL_INTERVAL = 5  # seconds - background watcher refresh

AGENTS = [
    {"id": "main",      "name": "K2S0",      "role": "Coordinator", "emoji": "🤖"},
    {"id": "developer", "name": "Charlie",   "role": "Developer",   "emoji": "👨‍💻"},
    {"id": "pm",        "name": "Dennis",    "role": "PM",          "emoji": "📋"},
    {"id": "qa",        "name": "Mac",       "role": "QA",          "emoji": "🔍"},
    {"id": "devops",    "name": "Frank",     "role": "DevOps",      "emoji": "🔧"},
    {"id": "research",  "name": "Sweet Dee", "role": "Research",    "emoji": "🔬"},
    {"id": "designer",  "name": "Cricket",   "role": "Designer",    "emoji": "🎨"},
]

# Map agent id → workspace directory name patterns
WORKSPACE_PATTERNS = {
    "main":      ["workspace", "workspace-main", "workspace-coordinator"],
    "developer": ["workspace-developer", "workspace-dev", "workspace-charlie"],
    "pm":        ["workspace-pm", "workspace-dennis"],
    "qa":        ["workspace-qa", "workspace-mac"],
    "devops":    ["workspace-devops", "workspace-frank"],
    "research":  ["workspace-research", "workspace-sweetdee", "workspace-sweet-dee"],
    "designer":  ["workspace-designer", "workspace-cricket"],
}

OPENCLAW_BASE = Path.home() / ".openclaw"

# ============================================================
# STATE (thread-safe via lock)
# ============================================================
lock = threading.Lock()
agent_cache = {}          # id → status dict
activity_log = collections.deque(maxlen=ACTIVITY_LOG_MAX)
file_mtime_cache = {}     # path → last known mtime
file_size_cache = {}      # path → last known byte size
file_linecount_cache = {} # path → last known line count
agent_last_active = {}    # agent_id → last epoch when detected active (for idle events)

IDLE_NOTIFY_THRESHOLD = 600  # 10 minutes


# ============================================================
# WORKSPACE SCANNING
# ============================================================
def find_workspace(agent_id):
    """Find the openclaw workspace directory for a given agent."""
    patterns = WORKSPACE_PATTERNS.get(agent_id, [])
    for pattern in patterns:
        candidate = OPENCLAW_BASE / pattern
        if candidate.is_dir():
            return candidate
    return None


def scan_md_files(workspace_dir):
    """Return list of (path, mtime, size) for all .md files in workspace."""
    if workspace_dir is None:
        return []
    results = []
    for md_file in workspace_dir.rglob("*.md"):
        try:
            stat = md_file.stat()
            results.append((md_file, stat.st_mtime, stat.st_size))
        except OSError:
            pass
    return results


def read_last_task(workspace_dir):
    """Try to extract a brief task description from the most recent memory file."""
    if workspace_dir is None:
        return None

    memory_dir = workspace_dir / "memory"
    if not memory_dir.is_dir():
        # fall back to scanning all .md files for any task hints
        return None

    # Get the most recently modified memory file
    memory_files = sorted(
        memory_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )

    for mf in memory_files[:3]:
        try:
            content = mf.read_text(encoding="utf-8", errors="ignore")
            # Look for the first non-empty, non-header line that looks like a task
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("-") or line.startswith("*"):
                    line = line.lstrip("-* ").strip()
                if len(line) > 10:
                    return line[:120]  # truncate long lines
        except Exception:
            pass

    return None


def get_total_workspace_bytes(workspace_dir):
    """Sum of all .md file sizes in the workspace."""
    if workspace_dir is None:
        return 0
    total = 0
    for md_file in workspace_dir.rglob("*.md"):
        try:
            total += md_file.stat().st_size
        except OSError:
            pass
    return total


def get_total_workspace_chars(workspace_dir):
    """Sum of character counts across all .md files in the workspace."""
    if workspace_dir is None:
        return 0
    total = 0
    for md_file in workspace_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            total += len(content)
        except OSError:
            pass
    return total


def read_file_snippet(path, n=5):
    """Return the last n non-empty lines of a file as a single string snippet."""
    try:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        tail = lines[-n:] if len(lines) >= n else lines
        return " · ".join(tail)[:300]  # join with separator, cap at 300 chars
    except Exception:
        return ""


def count_file_lines(path):
    """Count non-empty lines in a file."""
    try:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        return sum(1 for l in content.splitlines() if l.strip())
    except Exception:
        return 0


def count_memory_events(workspace_dir):
    """Count distinct memory file entries (rough: number of memory .md files)."""
    if workspace_dir is None:
        return 0
    memory_dir = workspace_dir / "memory"
    if not memory_dir.is_dir():
        return 0
    return len(list(memory_dir.glob("*.md")))


# ============================================================
# AGENT STATUS BUILD
# ============================================================
def build_agent_status(agent):
    agent_id   = agent["id"]
    workspace  = find_workspace(agent_id)
    md_files   = scan_md_files(workspace)

    now = time.time()
    last_mtime = None
    last_file  = None

    for (path, mtime, _size) in md_files:
        if last_mtime is None or mtime > last_mtime:
            last_mtime = mtime
            last_file  = path

    status = "idle"
    last_seen_iso = None

    if last_mtime is not None:
        age = now - last_mtime
        if age <= ACTIVE_THRESHOLD_SECONDS:
            status = "active"
        last_seen_iso = datetime.fromtimestamp(last_mtime, tz=timezone.utc).isoformat()

    workspace_bytes = get_total_workspace_bytes(workspace)
    workspace_chars = get_total_workspace_chars(workspace)
    last_task       = read_last_task(workspace)
    event_count     = count_memory_events(workspace)

    model_full  = AGENT_MODELS.get(agent_id, AGENT_MODELS.get("__default__", DEFAULT_MODEL))
    # short version: strip leading "anthropic/" or "openai/" prefix
    model_short = model_full.split("/", 1)[-1] if "/" in model_full else model_full

    return {
        "id":               agent_id,
        "name":             agent["name"],
        "role":             agent["role"],
        "emoji":            agent["emoji"],
        "status":           status,
        "last_seen":        last_seen_iso,
        "last_task":        last_task,
        "workspace_bytes":  workspace_bytes,
        "workspace_chars":  workspace_chars,
        "workspace_path":   str(workspace) if workspace else None,
        "event_count":      event_count,
        "model":            model_full,
        "model_short":      model_short,
        "estimated_tokens": round(workspace_chars / 4),
    }


# ============================================================
# FILE WATCHER (background thread)
# ============================================================
def detect_file_changes(agent_id, workspace):
    """Compare current mtimes/sizes against cache; emit enriched activity log events."""
    global file_mtime_cache, file_size_cache, file_linecount_cache, agent_last_active
    if workspace is None:
        return

    now = time.time()
    md_files = scan_md_files(workspace)
    agent_meta = next((a for a in AGENTS if a["id"] == agent_id), {})

    for (path, mtime, cur_size) in md_files:
        path_str  = str(path)
        prev_mtime = file_mtime_cache.get(path_str)
        prev_size  = file_size_cache.get(path_str)
        prev_lines = file_linecount_cache.get(path_str)

        is_new     = prev_mtime is None
        is_changed = (not is_new) and mtime > prev_mtime

        if is_new or is_changed:
            cur_lines  = count_file_lines(path)
            size_delta = cur_size - (prev_size if prev_size is not None else 0)
            line_delta = cur_lines - (prev_lines if prev_lines is not None else 0)
            snippet    = read_file_snippet(path, n=5)

            # relative path for display (e.g. "memory/2026-02-27.md")
            try:
                rel = str(path.relative_to(workspace))
            except ValueError:
                rel = path.name

            if is_new:
                event_type   = "task"
                severity     = "task"
                event_detail = f"Created {rel}"
            else:
                event_type   = "updated"
                severity     = "info"
                delta_str    = (f"+{line_delta}" if line_delta >= 0 else str(line_delta)) + " lines"
                event_detail = f"Updated {rel} ({delta_str})"

            ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            event = {
                "agent":        agent_id,
                "name":         agent_meta.get("name", agent_id),
                "emoji":        agent_meta.get("emoji", "🤖"),
                "timestamp":    ts,
                "file_changed": rel,
                "event_type":   event_type,
                "event_detail": event_detail,
                "snippet":      snippet,
                "size_delta":   size_delta,
                "severity":     severity,
            }
            with lock:
                activity_log.appendleft(event)
            print(f"[{ts}] {severity.upper():6s}: {agent_meta.get('name', agent_id)} → {event_detail}")

            # Update line count cache
            file_linecount_cache[path_str] = cur_lines

            # Track agent was active
            agent_last_active[agent_id] = now

        file_mtime_cache[path_str] = mtime
        file_size_cache[path_str]  = cur_size

    # Idle event: agent hasn't been active for IDLE_NOTIFY_THRESHOLD
    last_active = agent_last_active.get(agent_id)
    if last_active is not None and (now - last_active) >= IDLE_NOTIFY_THRESHOLD:
        # Only emit once per idle transition (clear last_active after emitting)
        ts = datetime.now(tz=timezone.utc).isoformat()
        idle_event = {
            "agent":        agent_id,
            "name":         agent_meta.get("name", agent_id),
            "emoji":        agent_meta.get("emoji", "🤖"),
            "timestamp":    ts,
            "file_changed": "",
            "event_type":   "idle",
            "event_detail": f"{agent_meta.get('name', agent_id)} has been idle for {int((now - last_active) // 60)}+ min",
            "snippet":      "",
            "size_delta":   0,
            "severity":     "idle",
        }
        with lock:
            activity_log.appendleft(idle_event)
        print(f"[{ts}] IDLE  : {agent_meta.get('name', agent_id)} → no changes in {int((now - last_active) // 60)}+ min")
        agent_last_active[agent_id] = None  # reset so we don't spam idle events


def watcher_loop():
    """Background thread: refresh agent cache every POLL_INTERVAL seconds."""
    # Prime caches on first run so we don't flood events for pre-existing files
    for agent in AGENTS:
        workspace = find_workspace(agent["id"])
        if workspace:
            for (path, mtime, size) in scan_md_files(workspace):
                p = str(path)
                file_mtime_cache[p]     = mtime
                file_size_cache[p]      = size
                file_linecount_cache[p] = count_file_lines(path)

    while True:
        time.sleep(POLL_INTERVAL)
        new_cache = {}
        for agent in AGENTS:
            workspace = find_workspace(agent["id"])
            detect_file_changes(agent["id"], workspace)
            status = build_agent_status(agent)
            new_cache[agent["id"]] = status

        with lock:
            agent_cache.update(new_cache)


def initial_load():
    """Synchronous first load so the HTTP server has data immediately."""
    ts = datetime.now(tz=timezone.utc).isoformat()
    for agent in AGENTS:
        status = build_agent_status(agent)
        agent_cache[agent["id"]] = status

        # Emit a boot event per agent
        boot_event = {
            "agent":        agent["id"],
            "name":         agent["name"],
            "emoji":        agent["emoji"],
            "timestamp":    ts,
            "file_changed": "",
            "event_type":   "boot",
            "event_detail": f"{agent['name']} agent online · workspace {'found' if status['workspace_path'] else 'not found'}",
            "snippet":      "",
            "size_delta":   0,
            "severity":     "info",
        }
        activity_log.appendleft(boot_event)

    print(f"[boot] Loaded {len(agent_cache)} agents.")


# ============================================================
# HTTP SERVER
# ============================================================
class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Suppress default access logs (noisy); keep errors
        if "404" in str(args) or "500" in str(args):
            super().log_message(fmt, *args)

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/agents":
            self.handle_agents()
        elif path == "/activity":
            self.handle_activity()
        elif path == "/health":
            self.handle_health()
        else:
            self.send_response(404)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "not found"}).encode())

    def handle_agents(self):
        with lock:
            data = list(agent_cache.values())
        self.send_json(data)

    def handle_activity(self):
        with lock:
            data = list(activity_log)
        self.send_json(data)

    def handle_health(self):
        self.send_json({
            "status": "ok",
            "agents": len(agent_cache),
            "uptime": int(time.time() - START_TIME),
            "activity_events": len(activity_log),
        })

    def send_json(self, data):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ============================================================
# MAIN
# ============================================================
START_TIME = time.time()

if __name__ == "__main__":
    print("=" * 56)
    print("  K2S0 Agent Status Server")
    print(f"  Listening on http://localhost:{PORT}")
    print(f"  Endpoints: /agents  /activity  /health")
    print(f"  Scanning:  {OPENCLAW_BASE}")
    print("=" * 56)

    # Initial synchronous load
    initial_load()

    # Print discovered workspaces
    for agent in AGENTS:
        ws = find_workspace(agent["id"])
        ws_str = str(ws) if ws else "(not found)"
        print(f"  {agent['emoji']}  {agent['name']:10s} → {ws_str}")
    print()

    # Start background watcher
    t = threading.Thread(target=watcher_loop, daemon=True)
    t.start()

    # Start HTTP server
    server = HTTPServer(("", PORT), DashboardHandler)
    try:
        print(f"Server running. Press Ctrl+C to stop.\n")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
        sys.exit(0)
