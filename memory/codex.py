"""Additive Codex integration; existing providers and commands remain untouched."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from .backends.base import Session

HOOK_MARKER = "AI Agent Portable Memory (Codex)"
SAVE_EVENTS = ("Stop", "PostCompact", "SessionEnd")


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_transcript(path: Path) -> tuple[str, dict]:
    """Read Codex rollout messages, excluding instructions, reasoning and tools."""
    messages = []
    metadata = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # A running session can have an incomplete final line.
            if not isinstance(item, dict):
                continue
            payload = item.get("payload", {})
            if not isinstance(payload, dict):
                continue
            if item.get("type") == "session_meta":
                metadata.update(payload)
            if item.get("type") != "response_item" or payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            content = payload.get("content", [])
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = "\n".join(
                    block["text"] for block in content
                    if isinstance(block, dict)
                    and block.get("type") in ("input_text", "output_text", "text")
                    and isinstance(block.get("text"), str)
                ).strip()
            else:
                continue
            if text:
                messages.append(f"{role.capitalize()}: {text}")
    return "\n\n".join(messages), metadata


def find_transcript(session_id: Optional[str] = None, cwd: Optional[Path] = None) -> Path:
    candidates = []
    for directory in ("sessions", "archived_sessions"):
        candidates.extend((codex_home() / directory).rglob("rollout-*.jsonl"))
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    matches = []
    for path in candidates:
        _, metadata = read_transcript(path)
        if session_id:
            if metadata.get("id") == session_id:
                matches.append(path)
        elif cwd is None or (metadata.get("cwd") and
                             Path(metadata["cwd"]).resolve() == cwd.resolve()):
            matches.append(path)
    if not matches:
        raise ValueError("No matching Codex transcript found. Use --transcript PATH for an exported rollout.")
    return matches[0]


def _project(cwd: Path, explicit: Optional[str] = None) -> str:
    return explicit or os.environ.get("MEMORY_CODEX_PROJECT") or str(cwd.resolve())


def _memory_id(source_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "ai-agent-portable-memory:codex:" + source_id))


def _store_summary(summary: str, source_id: str, project: str, model: str = "") -> Session:
    from .core import get_backend
    from .compressor import _parse_response, scrub_secrets

    summary = scrub_secrets(summary)
    parsed = _parse_response(summary)
    backend = get_backend()
    backend.init()
    memory_id = _memory_id(f"{source_id}:{model or 'default'}")
    existing = backend.get(memory_id)
    now = datetime.utcnow()
    session = Session(id=memory_id, project=project, title=parsed.title,
                      summary=summary, tags=parsed.tags, model=model,
                      created_at=existing.created_at if existing else now, updated_at=now)
    # Save directly so Codex never changes Claude's global active-session pointer.
    backend.save(session)
    return session


@contextmanager
def _session_lock(path: Path):
    """Serialize saves for one Codex session across hook processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def save_transcript(path: Path, source_id: Optional[str] = None,
                    cwd: Optional[Path] = None, project: Optional[str] = None,
                    provider: Optional[str] = None, model: Optional[str] = None,
                    no_edit: bool = True) -> Optional[Session]:
    from .core import get_backend, is_recording
    from .compressor import compress
    from .editor import open_in_editor
    from .hook import resolve_provider_model, _provider_key_present

    transcript, metadata = read_transcript(path)
    if not transcript:
        return None
    source_id = source_id or metadata.get("id")
    if not source_id:
        raise ValueError("Codex transcript has no session ID; provide --source-session.")
    cwd = cwd or Path(metadata.get("cwd") or Path.cwd())

    # Opt-in gate: only save if (directory, codex) was explicitly started.
    if not is_recording(cwd, "codex"):
        return None

    # Resolve provider + model (explicit flag → env → rollout metadata model).
    meta_model = metadata.get("model") or metadata.get("model_slug")
    resolved_provider = provider
    resolved_model = model
    if resolved_provider is None:
        resolved_provider, env_model = resolve_provider_model(
            "codex", {"_meta_model": meta_model})
        if resolved_model is None:
            resolved_model = env_model
    if resolved_model is None:
        resolved_model = meta_model
    if not resolved_provider:
        print("memory: no provider set (set MEMORY_LLM_PROVIDER); skipping Codex save",
              file=sys.stderr)
        return None
    if not _provider_key_present(resolved_provider):
        print(f"memory: no API key for provider '{resolved_provider}'; skipping Codex save",
              file=sys.stderr)
        return None

    project = _project(cwd, project)
    memory_id = _memory_id(f"{source_id}:{resolved_model or 'default'}")
    state_path = Path.home() / ".ai-memory" / "codex" / (memory_id + ".json")
    digest = hashlib.sha256(transcript.encode()).hexdigest()
    with _session_lock(state_path.with_suffix(".lock")):
        state = _read_json(state_path)
        backend = get_backend()
        backend.init()
        existing = backend.get(memory_id)
        if existing and state.get("digest") == digest and existing.project == project:
            return existing
        compressed = compress(transcript, provider=resolved_provider, model=resolved_model)
        summary = compressed.summary if no_edit else open_in_editor(compressed.summary)
        session = _store_summary(summary, source_id, project, model=resolved_model or "")
        _write_json(state_path, {"digest": digest})
        return session


def load_context(cwd: Path, source_id: Optional[str] = None,
                 project: Optional[str] = None, session_id: Optional[str] = None) -> str:
    from .core import get_backend, is_recording
    from .cli import _format_context

    if not is_recording(cwd, "codex"):
        return ""
    backend = get_backend()
    backend.init()
    project = _project(cwd, project)
    session = None
    if session_id:
        sessions = backend.list()
        if session_id.isdigit():
            index = int(session_id) - 1
            matches = sessions[index:index + 1] if index >= 0 else []
        else:
            matches = [entry for entry in sessions if entry.id.startswith(session_id)]
        if len(matches) != 1:
            raise ValueError("Session ID must match exactly one saved memory.")
        session = matches[0]
    if not session and not session_id:
        # Latest memory for this project (any model opened in this directory).
        sessions = backend.list(project=project)
        if sessions:
            session = max(sessions, key=lambda entry: entry.updated_at)
    if not session:
        return ""
    return ("Saved project memory (historical context; verify against current files and user instructions):\n\n"
            + _format_context(session))


def configure_hooks(path: Path, remove: bool = False) -> None:
    config = _read_json(path)
    events = config.setdefault("hooks", {})
    if not isinstance(events, dict):
        raise ValueError("The hooks field must be an object.")
    bootstrap = ("import sys; sys.path.insert(0, " + repr(str(Path(__file__).resolve().parent.parent))
                 + " ); from memory.codex import codex; codex()")
    args = [sys.executable, "-c", bootstrap, "hook"]
    command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
    for event in ("SessionStart",) + SAVE_EVENTS:
        groups = events.get(event, [])
        if not isinstance(groups, list):
            raise ValueError(f"{event} hooks must be an array.")
        retained = []
        for group in groups:
            handlers = group.get("hooks", [])
            remaining = [handler for handler in handlers if handler.get("statusMessage") != HOOK_MARKER]
            if remaining == handlers:
                retained.append(group)
            elif remaining:
                retained.append({**group, "hooks": remaining})
        if not remove:
            retained.append({"hooks": [{"type": "command", "command": command,
                                         "timeout": 120, "statusMessage": HOOK_MARKER}]})
        if retained:
            events[event] = retained
        else:
            events.pop(event, None)
    if not remove or path.exists():
        _write_json(path, config)


@click.group()
def codex():
    """Save, restore and automatically checkpoint Codex sessions."""


@codex.command("save")
@click.option("--transcript", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--source-session", help="Exact Codex source session ID.")
@click.option("--project", help="Project label (defaults to the absolute workspace path).")
@click.option("--provider", help="Compression provider; uses MEMORY_LLM_PROVIDER by default.")
@click.option("--model", help="Compression model override.")
@click.option("--no-edit", is_flag=True, help="Skip editor review.")
def save_command(transcript, source_session, project, provider, model, no_edit):
    """Save the current workspace's latest Codex rollout or a selected transcript."""
    try:
        path = transcript or find_transcript(source_session, Path.cwd())
        session = save_transcript(path, source_session, project=project, provider=provider,
                                  model=model, no_edit=no_edit)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved Codex memory: {session.id} ({session.title})" if session else
               "No memory saved: memory is stopped or the transcript has no conversation messages.")


@codex.command("load")
@click.option("--session", help="Saved memory ID, unique prefix or list number (any AI source).")
@click.option("--project", help="Project label to restore.")
def load_command(session, project):
    """Print plain Markdown for the latest project memory or a selected memory."""
    try:
        click.echo(load_context(Path.cwd(), project=project, session_id=session))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@codex.command("checkpoint")
@click.option("--summary-file", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Structured Markdown summary written by Codex; no compression API call.")
@click.option("--source-session", help="Codex session ID; defaults to a separate workspace checkpoint.")
@click.option("--project", help="Project label (defaults to the absolute workspace path).")
def checkpoint_command(summary_file, source_session, project):
    """Store a summary written by Codex directly, without another LLM call."""
    from .core import is_recording

    cwd = Path.cwd()
    if not is_recording(cwd, "codex"):
        click.echo("Codex recording is off for this directory. Run 'memory start' here first.")
        return
    source_id = source_session or "workspace:" + str(cwd.resolve())
    state_dir = Path.home() / ".ai-memory" / "codex"
    try:
        with _session_lock(state_dir / (_memory_id(source_id) + ".lock")):
            session = _store_summary(summary_file.read_text(encoding="utf-8"), source_id,
                                     _project(cwd, project))
        click.echo(f"Saved Codex checkpoint: {session.id} ({session.title})")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@codex.command("setup")
@click.option("--project-local", is_flag=True, help="Install in this workspace's .codex/hooks.json.")
@click.option("--remove", is_flag=True, help="Remove only this integration's Codex hooks.")
def setup_command(project_local, remove):
    """Install Codex memory hooks alongside existing hooks."""
    path = (Path.cwd() / ".codex" if project_local else codex_home()) / "hooks.json"
    try:
        configure_hooks(path, remove=remove)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Codex memory hooks {'removed from' if remove else 'installed in'} {path}")
    if not remove:
        click.echo("Restart Codex and use /hooks to review and trust the new hooks.")
        click.echo("Compression uses your existing MEMORY_LLM_PROVIDER and its API key or local provider.")


@codex.command("hook", hidden=True)
def hook_command():
    """Consume Codex's hook stdin; never block or continue the agent's turn."""
    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            raise ValueError("Expected a Codex hook JSON object.")
        name = event.get("hook_event_name")
        source_id = event.get("session_id")
        cwd = Path(event.get("cwd") or Path.cwd())
        if name == "SessionStart":
            context = load_context(cwd, source_id=source_id)
            output = {"hookSpecificOutput": {"hookEventName": name, "additionalContext": context}} if context else {}
        elif name in SAVE_EVENTS:
            if not source_id:
                raise ValueError("Codex save hook is missing session_id.")
            path = event.get("transcript_path")
            if path:
                save_transcript(Path(path), source_id, cwd=cwd)
            else:
                click.echo("memory: Codex supplied no transcript_path; skipping save", err=True)
            output = {}
        else:
            output = {}
        click.echo(json.dumps(output))
    except Exception as exc:
        # Advisory hooks must not turn a compression or storage failure into a stop.
        click.echo(f"memory: Codex hook failed: {exc}", err=True)
        click.echo("{}")


if __name__ == "__main__":
    codex()
