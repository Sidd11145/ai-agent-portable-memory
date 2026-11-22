import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .backends.base import BaseBackend, Session

load_dotenv()

_ACTIVE_SESSION_FILE = Path.home() / ".ai-memory" / "active_session"
_CLAUDE_SESSION_MAP_FILE = Path.home() / ".ai-memory" / "claude_sessions.json"
_STOPPED_FILE = Path.home() / ".ai-memory" / "STOPPED"
_ENABLED_FILE = Path.home() / ".ai-memory" / "enabled.json"
_SESSION_BACKEND_MAP_FILE = Path.home() / ".ai-memory" / "session_backends.json"
_SESSION_MARKERS_DIR = Path.home() / ".ai-memory" / "current_sessions"

# Rendezvous markers older than this are considered stale (window closed without
# a clean SessionEnd/Stop).
_MARKER_STALE_SECONDS = 7200  # 2 hours


def is_stopped() -> bool:
    return _STOPPED_FILE.exists()


def set_stopped(stopped: bool) -> None:
    _STOPPED_FILE.parent.mkdir(parents=True, exist_ok=True)
    if stopped:
        _STOPPED_FILE.write_text("stopped")
    else:
        _STOPPED_FILE.unlink(missing_ok=True)


# ── Per-(project + agent) recording opt-in ─────────────────────────────────────
# enabled.json shape: { "<resolved_cwd>": ["claude", "codex"], ... }
# Recording is OFF by default; a (directory, agent) pair must be explicitly
# enabled via `memory start`. This keeps every project/tab independent.

def _resolve_key(cwd) -> str:
    return str(Path(cwd).resolve())


def _load_enabled() -> dict:
    if _ENABLED_FILE.exists():
        try:
            data = json.loads(_ENABLED_FILE.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_enabled(data: dict) -> None:
    _ENABLED_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ENABLED_FILE.write_text(json.dumps(data, indent=2))


def is_recording(cwd, agent: str) -> bool:
    """True only if (resolved cwd, agent) was explicitly enabled and not globally stopped."""
    if is_stopped():  # global kill-switch always wins
        return False
    return agent in _load_enabled().get(_resolve_key(cwd), [])


def enable_recording(cwd, agent: str) -> None:
    data = _load_enabled()
    key = _resolve_key(cwd)
    agents = data.get(key, [])
    if agent not in agents:
        agents.append(agent)
    data[key] = agents
    _save_enabled(data)


def disable_recording(cwd, agent: str) -> None:
    data = _load_enabled()
    key = _resolve_key(cwd)
    agents = [a for a in data.get(key, []) if a != agent]
    if agents:
        data[key] = agents
    else:
        data.pop(key, None)
    _save_enabled(data)


def recording_agents(cwd) -> list[str]:
    """Agents currently enabled to record in this directory."""
    return list(_load_enabled().get(_resolve_key(cwd), []))


_KNOWN_BACKENDS = ("sqlite", "supabase", "s3", "firebase")


def resolve_backend(cwd, override: Optional[str] = None,
                    session_id: Optional[str] = None) -> Optional[str]:
    """Return the backend name to use, or None to let get_backend use its sqlite default.

    Precedence: explicit override → per-window (session) backend → MEMORY_BACKEND env → None.
    Backend choice is per-session, never per-directory.
    """
    if override:
        return override
    if session_id:
        per_session = get_backend_for_session(session_id)
        if per_session:
            return per_session
    return os.environ.get("MEMORY_BACKEND") or None


# ── Per-window (Claude session) backend routing ────────────────────────────────
# session_backends.json shape: { "ports": {"<sse_port>": "firebase"},
#                                 "sessions": {"<session_id>": "firebase"} }
# Hook payloads provide the session ID used by commands and the save path.
# Commands can also stage choices on an SSE port until the hook links its ID.

def _empty_session_backend_map() -> dict:
    return {"ports": {}, "sessions": {}, "links": {}, "decisions": {},
            "recording": {"ports": {}, "sessions": {}}}


def _load_session_backend_map() -> dict:
    if _SESSION_BACKEND_MAP_FILE.exists():
        try:
            data = json.loads(_SESSION_BACKEND_MAP_FILE.read_text())
            if not isinstance(data, dict):
                return _empty_session_backend_map()
            data.setdefault("ports", {})
            data.setdefault("sessions", {})
            data.setdefault("links", {})
            rec = data.setdefault("recording", {"ports": {}, "sessions": {}})
            rec.setdefault("ports", {})
            rec.setdefault("sessions", {})
            return data
        except Exception:
            return _empty_session_backend_map()
    return _empty_session_backend_map()


def _save_session_backend_map(data: dict) -> None:
    _SESSION_BACKEND_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = _SESSION_BACKEND_MAP_FILE.with_name(
        _SESSION_BACKEND_MAP_FILE.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(_SESSION_BACKEND_MAP_FILE)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _session_backend_transaction():
    """Lock the complete update and publish a file readers can read atomically."""
    from .codex import _session_lock

    with _session_lock(_SESSION_BACKEND_MAP_FILE.with_suffix(".lock")):
        data = _load_session_backend_map()
        yield data
        _save_session_backend_map(data)


def set_backend_for_session(key, backend: str, *, table: str = "sessions") -> None:
    """Pin a per-window backend, keyed by session_id (default) or SSE port."""
    if backend not in _KNOWN_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose: {', '.join(_KNOWN_BACKENDS)}")
    with _session_backend_transaction() as data:
        data.setdefault(table, {})[str(key)] = backend


def get_backend_for_session(key, *, table: str = "sessions") -> Optional[str]:
    """Backend pinned for this session_id/port, or None if unset."""
    if not key:
        return None
    return _load_session_backend_map().get(table, {}).get(str(key))


def clear_backend_for_session(key, *, table: str = "sessions") -> None:
    with _session_backend_transaction() as data:
        data.get(table, {}).pop(str(key), None)


# ── Per-window recording opt-in (mirrors the per-window backend routing above) ──
# Recording state lives under the "recording" key of session_backends.json:
#   {"recording": {"ports": {"<port>": ["claude"]}, "sessions": {"<sid>": ["claude"]}}}
# Commands with a session ID update recording.sessions directly. Commands with an
# unlinked port stage choices until refresh_and_promote runs. Other agents retain
# their directory recording settings. decisions tracks accepted and declined choices.

def _clear_session_staging(data: dict, session_id: str, agent: Optional[str]) -> None:
    """Discard staged choices superseded by an explicit session choice."""
    for port, linked_sid in data.get("links", {}).items():
        if linked_sid == session_id:
            data.get("ports", {}).pop(port, None)
            staged = data.get("recording", {}).get("ports", {})
            remaining = [name for name in staged.get(port, []) if agent and name != agent]
            if remaining:
                staged[port] = remaining
            else:
                staged.pop(port, None)


def set_recording_for_session(key, agent, *, table: str = "sessions",
                              backend: Optional[str] = None) -> None:
    """Add `agent` to the per-window recording list keyed by session_id (default)
    or SSE port (table='ports')."""
    if backend is not None and backend not in _KNOWN_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose: {', '.join(_KNOWN_BACKENDS)}")
    with _session_backend_transaction() as data:
        if table == "sessions":
            _clear_session_staging(data, str(key), agent)
        rec = data.setdefault("recording", {"ports": {}, "sessions": {}})
        agents = rec.setdefault(table, {}).get(str(key), [])
        if agent not in agents:
            agents.append(agent)
        rec[table][str(key)] = agents
        if backend is not None:
            data.setdefault(table, {})[str(key)] = backend
        if table == "sessions" and agent == "claude":
            data.setdefault("decisions", {})[str(key)] = True


def clear_recording_for_session(key, agent=None, *, table: str = "sessions") -> None:
    """Remove one agent (or all if agent is None) from the per-window recording
    list for this session_id/port. Drops the key when the list empties."""
    with _session_backend_transaction() as data:
        if table == "sessions":
            _clear_session_staging(data, str(key), agent)
        rec = data.setdefault("recording", {"ports": {}, "sessions": {}})
        bucket = rec.setdefault(table, {})
        if agent is None:
            bucket.pop(str(key), None)
        else:
            agents = [a for a in bucket.get(str(key), []) if a != agent]
            if agents:
                bucket[str(key)] = agents
            else:
                bucket.pop(str(key), None)
        if table == "sessions" and agent in (None, "claude"):
            data.setdefault("decisions", {})[str(key)] = False


def prepare_recording_session(session_id: str, source: str) -> None:
    """Reset choices for a new session, preserving them on resume and compaction."""
    if source not in {"startup", "clear"}:
        return
    with _session_backend_transaction() as data:
        data.get("recording", {}).get("sessions", {}).pop(session_id, None)
        data.get("sessions", {}).pop(session_id, None)
        data.setdefault("decisions", {}).pop(session_id, None)


def get_recording_decision(session_id: str) -> Optional[bool]:
    """Return the user's choice, or None while a session is awaiting a choice."""
    data = _load_session_backend_map()
    decision = data.get("decisions", {}).get(session_id)
    if decision is not None:
        return decision
    # Existing installations may have recording entries without a decision.
    if "claude" in data.get("recording", {}).get("sessions", {}).get(session_id, []):
        return True
    return None


def recording_agents_for_session(key, *, table: str = "sessions") -> list:
    """Agents recording for this session_id/port, or [] if none. Never raises."""
    try:
        if not key:
            return []
        rec = _load_session_backend_map().get("recording", {})
        return list(rec.get(table, {}).get(str(key), []))
    except Exception:
        return []


def is_recording_session(cwd, agent, *, session_id=None, port=None) -> bool:
    """Per-window recording gate with legacy per-directory fallback.

    Precedence:
      1. is_stopped() -> False  (global kill-switch always wins)
      2. resolve port -> session_id via get_session_for_port(port) if port given
      3. if a session_id is available -> the WINDOW is authoritative:
           return agent in recording.sessions[session_id]  (NO dir fallback — this
           is what stops a sibling window inheriting another window's state)
      4. if neither session_id nor port:
           - claude is strictly window-scoped; never falls back to enabled.json
           - all other agents fall back to legacy is_recording(cwd, agent)
    """
    if is_stopped():
        return False
    sid = session_id
    if not sid and port:
        sid = get_session_for_port(port)
    if sid:
        rec = _load_session_backend_map().get("recording", {})
        return agent in rec.get("sessions", {}).get(str(sid), [])
    # Claude is strictly session-scoped: without a session_id or port there is no
    # window to check, so recording is off.  A stale enabled.json entry (written by
    # an older version before per-window isolation was enforced) must not be honoured.
    if agent == "claude":
        return False
    return is_recording(cwd, agent)


def set_session_link(port, session_id) -> None:
    """Record the persistent port↔session_id link so interactive commands (which
    hold only the port) can resolve the window's session_id. Never raises."""
    try:
        if not port or not session_id:
            return
        with _session_backend_transaction() as data:
            data.setdefault("links", {})[str(port)] = str(session_id)
    except Exception:
        pass


def get_session_for_port(port) -> Optional[str]:
    """The session_id currently linked to this SSE port, or None. Never raises."""
    try:
        if not port:
            return None
        return _load_session_backend_map().get("links", {}).get(str(port))
    except Exception:
        return None


def refresh_and_promote(port, session_id, *, promote_choices: bool = True) -> None:
    """Refresh the persistent port↔session link and consume any staged
    ports[port] backend onto sessions[session_id]. Called every turn from the
    UserPromptSubmit hook and once from SessionStart (both see port+session_id).

    The link persists — only the one-shot staged port entry is consumed. This is
    what lets a mid-session `memory start --backend X` take effect on the live
    session and lets interactive `status` resolve the window's backend.
    Never raises — advisory, must not break the hook."""
    try:
        if not port or not session_id:
            return
        with _session_backend_transaction() as data:
            data.setdefault("links", {})[str(port)] = str(session_id)
            if not promote_choices:
                # Claude choices are applied using the hook's session ID. A reused
                # port cannot establish permission for the session that just opened.
                staged = data.get("recording", {}).get("ports", {})
                remaining = [agent for agent in staged.get(str(port), []) if agent != "claude"]
                if remaining:
                    staged[str(port)] = remaining
                else:
                    staged.pop(str(port), None)
                    data.get("ports", {}).pop(str(port), None)
                return
            backend = data.get("ports", {}).get(str(port))
            if backend:
                data.setdefault("sessions", {})[str(session_id)] = backend
                data.get("ports", {}).pop(str(port), None)
            # Promote staged per-window recording the same way: move recording.ports
            # [port] onto recording.sessions[session_id] and consume the staged entry.
            rec = data.setdefault("recording", {"ports": {}, "sessions": {}})
            rec.setdefault("ports", {})
            rec.setdefault("sessions", {})
            staged = rec["ports"].get(str(port))
            if staged:
                existing = rec["sessions"].get(str(session_id), [])
                rec["sessions"][str(session_id)] = sorted(set(existing) | set(staged))
                rec["ports"].pop(str(port), None)
                if "claude" in staged:
                    data.setdefault("decisions", {})[str(session_id)] = True
    except Exception:
        pass


def prune_session_backends(max_ports: int = 64, *, keep_sessions=()) -> None:
    """Drop session pins whose Claude transcript no longer exists, and cap the
    staging `ports` table. Keep sessions with transcripts, fresh markers, or an
    explicit keep_sessions entry while their first transcript is created. Never raises."""
    try:
        projects = Path.home() / ".claude" / "projects"
        live = set(keep_sessions)
        if projects.exists():
            for d in projects.iterdir():
                if not d.is_dir():
                    continue
                for f in d.iterdir():
                    if f.suffix == ".jsonl":
                        live.add(f.stem)
        with _session_backend_transaction() as data:
            if live:
                # Fresh hooks can run before their transcript file has been created.
                for marker_path in _SESSION_MARKERS_DIR.glob("*.json"):
                    try:
                        marker = json.loads(marker_path.read_text(encoding="utf-8"))
                        if time.time() - float(marker.get("updated_at", 0)) <= _MARKER_STALE_SECONDS:
                            live.add(marker["session_id"])
                    except (OSError, ValueError, KeyError):
                        continue
                data["sessions"] = {
                    k: v for k, v in data.get("sessions", {}).items() if k in live
                }
                data["links"] = {
                    p: s for p, s in data.get("links", {}).items() if s in live
                }
                rec = data.get("recording", {})
                rec["sessions"] = {
                    k: v for k, v in rec.get("sessions", {}).items() if k in live
                }
                data["decisions"] = {
                    k: v for k, v in data.get("decisions", {}).items() if k in live
                }
            ports = data.get("ports", {})
            if len(ports) > max_ports:
                data["ports"] = dict(list(ports.items())[-max_ports:])
            links = data.get("links", {})
            if len(links) > max_ports:
                data["links"] = dict(list(links.items())[-max_ports:])
    except Exception:
        pass


# ── Per-window rendezvous markers ─────────────────────────────────────────────
# SessionStart and UserPromptSubmit write per-session markers under current_sessions/.
# Commands without an explicit ID or SSE port can resolve a single matching marker.
#
# Safety invariants:
#   - One file per session_id — windows never overwrite each other's markers.
#   - `memory start` validates the marker's cwd against Path.cwd() before use.
#   - Markers older than _MARKER_STALE_SECONDS are ignored (window was hard-killed).
#   - SessionEnd removes the marker when the session closes.

def _get_windows_claude_process_identity(process_id: int) -> Optional[dict]:
    script = r"""
$ErrorActionPreference = 'Stop'
$memoryProcessId = PROCESS_ID
for ($memoryDepth = 0; $memoryDepth -lt 32; $memoryDepth++) {
    if ($memoryProcessId -le 1) { break }
    $memoryProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $memoryProcessId)
    if ($null -eq $memoryProcess) { break }
    $memoryClaudeNode = ($memoryProcess.Name -ieq 'node.exe' -and
        $memoryProcess.CommandLine -match '@anthropic-ai[\\/]claude-code[\\/]cli\.js')
    if ($memoryProcess.Name -ieq 'claude.exe' -or $memoryClaudeNode) {
        [ordered]@{
            pid = [int]$memoryProcess.ProcessId
            started_at = $memoryProcess.CreationDate.ToUniversalTime().ToString('o')
        } | ConvertTo-Json -Compress
        break
    }
    if ($memoryProcess.ParentProcessId -eq $memoryProcessId) { break }
    $memoryProcessId = [int]$memoryProcess.ParentProcessId
}
""".replace("PROCESS_ID", str(int(process_id)))
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, check=True, timeout=10)
    if not result.stdout.strip():
        return None
    identity = json.loads(result.stdout)
    if int(identity["pid"]) <= 1 or not identity["started_at"]:
        return None
    return {"pid": int(identity["pid"]), "started_at": identity["started_at"]}


def get_claude_process_identity(process_id: Optional[int] = None) -> Optional[dict]:
    """Find the owning Claude process through the calling process's ancestors."""
    pid = process_id or os.getppid()
    try:
        if os.name == "nt":
            return _get_windows_claude_process_identity(pid)
        for _ in range(32):
            if pid <= 1:
                break
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "pid=,ppid=,lstart=,comm="],
                capture_output=True, text=True, check=True, timeout=2)
            fields = result.stdout.strip().split(None, 7)
            if len(fields) != 8 or int(fields[0]) != pid:
                return None
            if Path(fields[7]).name == "claude":
                return {"pid": pid, "started_at": " ".join(fields[2:7])}
            parent_pid = int(fields[1])
            if parent_pid == pid:
                break
            pid = parent_pid
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        pass
    return None


def _process_marker_path(claude_process: dict) -> Path:
    return _SESSION_MARKERS_DIR / "processes" / (str(claude_process["pid"]) + ".json")


def _write_marker_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_session_marker(session_id: str, port: Optional[str], cwd: Path, *,
                         claude_process: Optional[dict] = None) -> None:
    """Write (or refresh) the rendezvous marker for this window. Never raises."""
    try:
        if not session_id or session_id == "unknown":
            return
        _SESSION_MARKERS_DIR.mkdir(parents=True, exist_ok=True)
        marker_path = _SESSION_MARKERS_DIR / (session_id + ".json")
        payload = {
            "session_id": session_id,
            "port": port or "",
            "cwd": str(cwd.resolve()),
            "updated_at": time.time(),
        }
        if claude_process:
            payload["claude_process"] = claude_process
        _write_marker_file(marker_path, payload)
        if claude_process:
            from .codex import _session_lock
            process_path = _process_marker_path(claude_process)
            with _session_lock(process_path.with_suffix(".lock")):
                _write_marker_file(process_path, payload)
    except Exception:
        pass


def read_session_marker_for_cwd(cwd: Path,
                                claude_session_id: Optional[str] = None,
                                port: Optional[str] = None,
                                claude_process: Optional[dict] = None,
                                ) -> Optional[dict]:
    """Return the marker for this window, or None.

    Resolution order:
      1. If claude_session_id is given, return the matching non-stale marker for this cwd —
         exact and unambiguous even when several windows share the directory.
      2. If a port is given, require a unique marker matching that port and cwd.
      3. If exactly one non-stale marker exists for this cwd, return it
         (unambiguous even without an env hint).
      4. Multiple matching markers with no session_id hint → return None.
         Callers must refuse rather than guess the wrong window.

    Never raises.
    """
    if not _SESSION_MARKERS_DIR.exists():
        return None
    resolved = str(cwd.resolve())
    if claude_process and not claude_session_id:
        try:
            marker = json.loads(_process_marker_path(claude_process).read_text(encoding="utf-8"))
            if (marker.get("claude_process") == claude_process
                    and marker.get("cwd") == resolved
                    and marker.get("session_id")):
                return marker
        except (OSError, ValueError, KeyError):
            pass
        return None
    now = time.time()
    matches: list[dict] = []
    for path in _SESSION_MARKERS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("cwd") != resolved:
            continue
        age = now - float(data.get("updated_at", 0))
        if age > _MARKER_STALE_SECONDS:
            continue
        matches.append(data)

    if not matches:
        return None

    # Exact match via the inherited env var — always unambiguous.
    if claude_session_id:
        for m in matches:
            if m.get("session_id") == claude_session_id:
                return m
        # The env var named a session not in this cwd's markers; treat as absent.
        return None

    if port:
        matches = [marker for marker in matches if marker.get("port") == str(port)]

    # No hint: unambiguous only when a single window is live here.
    if len(matches) == 1:
        return matches[0]

    # Multiple live windows, no hint → refuse to guess.
    return None


def remove_session_marker(session_id: str) -> None:
    """Delete the rendezvous marker for this session. Never raises."""
    try:
        if not session_id or session_id == "unknown":
            return
        marker_path = _SESSION_MARKERS_DIR / (session_id + ".json")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {}
        marker_path.unlink(missing_ok=True)
        claude_process = marker.get("claude_process")
        if claude_process:
            from .codex import _session_lock
            process_path = _process_marker_path(claude_process)
            with _session_lock(process_path.with_suffix(".lock")):
                try:
                    current = json.loads(process_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return
                if (current.get("session_id") == session_id
                        and current.get("claude_process") == claude_process):
                    process_path.unlink(missing_ok=True)
    except Exception:
        pass


def cleanup_stale_markers() -> None:
    """Remove markers older than _MARKER_STALE_SECONDS. Never raises."""
    try:
        if not _SESSION_MARKERS_DIR.exists():
            return
        now = time.time()
        for path in _SESSION_MARKERS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if now - float(data.get("updated_at", 0)) > _MARKER_STALE_SECONDS:
                    path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass


def get_active_session_id() -> Optional[str]:
    if _ACTIVE_SESSION_FILE.exists():
        sid = _ACTIVE_SESSION_FILE.read_text().strip()
        return sid or None
    return None


def set_active_session_id(session_id: Optional[str]) -> None:
    _ACTIVE_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    if session_id is None:
        _ACTIVE_SESSION_FILE.unlink(missing_ok=True)
    else:
        _ACTIVE_SESSION_FILE.write_text(session_id)


def _load_claude_session_map() -> dict:
    if _CLAUDE_SESSION_MAP_FILE.exists():
        try:
            return json.loads(_CLAUDE_SESSION_MAP_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_claude_session_map(mapping: dict) -> None:
    _CLAUDE_SESSION_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CLAUDE_SESSION_MAP_FILE.write_text(json.dumps(mapping))


def get_memory_session_for_claude(claude_session_id: str) -> Optional[str]:
    """Return the memory session ID currently mapped to this Claude session, if any."""
    return _load_claude_session_map().get(claude_session_id)


def set_memory_session_for_claude(claude_session_id: str, memory_session_id: str) -> None:
    """Map a Claude session ID to a memory session ID."""
    mapping = _load_claude_session_map()
    mapping[claude_session_id] = memory_session_id
    _save_claude_session_map(mapping)


def clear_memory_session_for_claude(claude_session_id: str) -> None:
    """Remove the mapping for a Claude session (called when the session ends)."""
    mapping = _load_claude_session_map()
    mapping.pop(claude_session_id, None)
    _save_claude_session_map(mapping)


def clear_all_claude_session_mappings() -> None:
    """Wipe the entire Claude session → memory session map (called on memory use clear)."""
    _save_claude_session_map({})


def get_backend(backend_override: Optional[str] = None) -> BaseBackend:
    chosen = backend_override or os.environ.get("MEMORY_BACKEND") or "sqlite"
    backend_type = chosen.lower()

    if backend_type == "sqlite":
        from .backends.sqlite import SQLiteBackend
        db_path = os.environ.get("SQLITE_DB_PATH", str(Path.home() / ".ai-memory" / "memory.db"))
        return SQLiteBackend(db_path=Path(db_path))

    elif backend_type == "supabase":
        from .backends.supabase import SupabaseBackend
        url = _require_env("SUPABASE_URL")
        key = _require_env("SUPABASE_KEY")
        return SupabaseBackend(url=url, key=key)

    elif backend_type == "s3":
        from .backends.s3 import S3Backend
        bucket = _require_env("S3_BUCKET_NAME")
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        region = os.environ.get("AWS_REGION", "us-east-1")
        return S3Backend(bucket=bucket, endpoint_url=endpoint, region=region)

    elif backend_type == "firebase":
        from .backends.firebase import FirebaseBackend
        creds = _require_env("FIREBASE_CREDENTIALS_PATH")
        project = _require_env("FIREBASE_PROJECT_ID")
        return FirebaseBackend(credentials_path=creds, project_id=project)

    else:
        raise ValueError(f"Unknown MEMORY_BACKEND: {backend_type}. Choose: sqlite, supabase, s3, firebase")


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(f"Environment variable {key} is required for this backend.")
    return val


def create_session(title: str, project: str, summary: str, tags: list[str],
                   session_id: Optional[str] = None, model: Optional[str] = None,
                   backend_override: Optional[str] = None) -> Session:
    if is_stopped():
        raise RuntimeError("Memory is stopped. Run 'memory start' to re-enable.")
    backend = get_backend(backend_override)
    backend.init()
    if session_id:
        existing = backend.get(session_id)
        if existing:
            existing.title = title
            existing.project = project
            existing.summary = summary
            existing.tags = tags
            if model:
                existing.model = model
            backend.save(existing)
            set_active_session_id(existing.id)
            return existing
    session = Session(
        id=session_id or str(uuid.uuid4()),
        project=project,
        title=title,
        summary=summary,
        tags=tags,
        model=model or "",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    backend.save(session)
    set_active_session_id(session.id)
    return session


def list_sessions(project: Optional[str] = None,
                  claude_session_id: Optional[str] = None) -> list[Session]:
    """List sessions, routing to the backend pinned for claude_session_id if set."""
    backend = get_backend(resolve_backend(Path.cwd(), session_id=claude_session_id))
    backend.init()
    return backend.list(project=project)


def load_session(session_id: str,
                 claude_session_id: Optional[str] = None) -> Optional[Session]:
    """Load a memory session, routing to the backend pinned for claude_session_id if set."""
    if is_stopped():
        raise RuntimeError("Memory is stopped. Run 'memory start' to re-enable.")
    backend = get_backend(resolve_backend(Path.cwd(), session_id=claude_session_id))
    backend.init()
    return backend.get(session_id)


def search_sessions(query: str, project: Optional[str] = None,
                    claude_session_id: Optional[str] = None) -> list[Session]:
    """Search sessions, routing to the backend pinned for claude_session_id if set."""
    backend = get_backend(resolve_backend(Path.cwd(), session_id=claude_session_id))
    backend.init()
    return backend.search(query=query, project=project)


def delete_session(session_id: str,
                   claude_session_id: Optional[str] = None) -> None:
    """Delete a memory session, routing to the backend pinned for claude_session_id if set."""
    backend = get_backend(resolve_backend(Path.cwd(), session_id=claude_session_id))
    backend.init()
    backend.delete(session_id)
