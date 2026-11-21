import json
import os
import uuid
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
    backend_type = (backend_override or os.environ.get("MEMORY_BACKEND", "sqlite")).lower()

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


def list_sessions(project: Optional[str] = None) -> list[Session]:
    backend = get_backend()
    backend.init()
    return backend.list(project=project)


def load_session(session_id: str) -> Optional[Session]:
    if is_stopped():
        raise RuntimeError("Memory is stopped. Run 'memory start' to re-enable.")
    backend = get_backend()
    backend.init()
    return backend.get(session_id)


def search_sessions(query: str, project: Optional[str] = None) -> list[Session]:
    backend = get_backend()
    backend.init()
    return backend.search(query=query, project=project)


def delete_session(session_id: str) -> None:
    backend = get_backend()
    backend.init()
    backend.delete(session_id)
