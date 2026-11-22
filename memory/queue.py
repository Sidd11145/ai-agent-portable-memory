"""Durable save queue — capture now, compress later.

Compressing a transcript requires a slow LLM API call (10-30s). Running that
inside a Stop / SessionEnd / PostCompact hook is unsafe: the agent process can
be killed on window-close before the call returns, losing the save entirely.

This module splits the work:

  enqueue()  — called from the save-event hook. Writes the RAW transcript plus
               metadata to a small JSON file. Pure disk write, no network, so it
               completes in milliseconds and survives an abrupt window-close.

  drain()    — called from the next SessionStart hook (process is alive and
               stable). Compresses each queued transcript and stores it in the
               chosen backend, deleting the queue file only on success. A failed
               item (e.g. provider unavailable) is left in place to retry later.

The store sequence here intentionally duplicates the minimal logic from
hook.save_session rather than refactoring it, so the working manual/inline save
path is never modified.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .codex import _session_lock

QUEUE_DIR = Path.home() / ".ai-memory" / "queue"


def _memory_id(source_id: str) -> str:
    """Same identity scheme as hook.save_session, so queued items map to the
    same memory entry a direct save would have produced."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "ai-agent-portable-memory:universal:" + source_id))


def _queue_path(memory_id: str) -> Path:
    return QUEUE_DIR / (memory_id + ".json")


def enqueue(source_id: str, cwd: Path, agent: str, transcript: str,
            meta_model: Optional[str] = None, backend: Optional[str] = None,
            project: Optional[str] = None) -> Optional[Path]:
    """Persist a raw transcript for later compression. Returns the queue file
    path, or None if there was nothing to save."""
    if not transcript or not transcript.strip():
        return None

    digest = hashlib.sha256(transcript.encode()).hexdigest()
    memory_id = _memory_id(f"{source_id}:{meta_model or 'default'}")
    path = _queue_path(memory_id)

    payload = {
        "source_id": source_id,
        "cwd": str(cwd),
        "agent": agent,
        "transcript": transcript,
        "meta_model": meta_model,
        "backend": backend,
        "project": project,
        "digest": digest,
        "queued_at": datetime.utcnow().isoformat(),
    }
    with _session_lock(path.with_suffix(".lock")):
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                # A changed destination still needs a save for an identical transcript.
                if all(existing.get(key) == value for key, value in payload.items()
                       if key != "queued_at"):
                    return path
            except (json.JSONDecodeError, OSError):
                pass
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    return path


def _process_one(payload: dict) -> bool:
    """Compress + store a single queued item. Returns True on success (so the
    caller can delete the file), False to leave it queued for a later retry."""
    from .core import get_backend, is_stopped, resolve_backend
    from .compressor import compress, scrub_secrets, _parse_response
    from .backends.base import Session
    from .hook import resolve_provider_model, _provider_key_present

    transcript = payload.get("transcript") or ""
    source_id = payload.get("source_id")
    agent = payload.get("agent") or "unknown"
    cwd = Path(payload.get("cwd") or Path.cwd())
    meta_model = payload.get("meta_model")
    project = payload.get("project")

    if not transcript.strip() or not source_id:
        return True  # nothing usable — drop it

    # A queued item is proof the window's per-window recording gate already
    # passed at enqueue time (hook.save path). By drain time the originating
    # window (SSE port) is gone and its recording entry has been torn down on
    # close — so re-checking per-window recording here would wrongly drop the
    # final save we were asked to keep. Only the global kill-switch can veto.
    if is_stopped():
        return False  # keep queued; drain again after 'memory start --global'

    provider, model = resolve_provider_model(agent, {"_meta_model": meta_model})
    if model is None:
        model = meta_model
    if not provider or not _provider_key_present(provider):
        return False  # provider unavailable right now — keep for retry

    backend_name = resolve_backend(cwd, payload.get("backend"), session_id=source_id)

    compressed = compress(transcript, provider=provider, model=model)
    summary = scrub_secrets(compressed.summary)
    parsed = _parse_response(summary)

    backend = get_backend(backend_name)
    backend.init()
    memory_id = _memory_id(f"{source_id}:{model or 'default'}")
    existing = backend.get(memory_id)
    now = datetime.utcnow()
    project_label = project or parsed.project or cwd.name or "default"
    session = Session(
        id=memory_id,
        project=project_label,
        title=parsed.title,
        summary=summary,
        tags=list({*parsed.tags, agent.lower()}),
        model=model or "",
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )
    backend.save(session)
    return True


def drain(limit: Optional[int] = None) -> int:
    """Compress and store queued transcripts. Returns the number successfully
    processed. Never raises — advisory, must not block the agent's turn."""
    if not QUEUE_DIR.exists():
        return 0

    processed = 0
    files = sorted(QUEUE_DIR.glob("*.json"))
    for path in files:
        if limit is not None and processed >= limit:
            break
        try:
            # Serialize drains of one item while allowing hooks to capture new
            # snapshots during the LLM call. File updates use a separate short lock.
            with _session_lock(path.with_suffix(".drain.lock")):
                with _session_lock(path.with_suffix(".lock")):
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except FileNotFoundError:
                        continue
                    except json.JSONDecodeError:
                        path.unlink(missing_ok=True)
                        continue
                if _process_one(payload):
                    with _session_lock(path.with_suffix(".lock")):
                        current = json.loads(path.read_text(encoding="utf-8"))
                        if current == payload:
                            path.unlink(missing_ok=True)
                    processed += 1
        except Exception:
            # Leave the file queued; a transient failure should not drop data.
            continue
    return processed
