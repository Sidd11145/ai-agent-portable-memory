"""Additive GitHub Copilot CLI integration; existing providers and commands remain untouched."""

import json
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from .backends.base import Session

HOOK_MARKER = "AI Agent Portable Memory (Copilot)"
# Copilot CLI supports Claude Code hook event names when hooks are defined in .claude/settings.json.
# SessionStart injects context; Stop saves the session summary.
SAVE_EVENTS = ("Stop",)


def copilot_settings_path(project_local: bool = False, cwd: Optional[Path] = None) -> Path:
    """Return the settings file where Copilot CLI reads hooks."""
    if project_local:
        return (cwd or Path.cwd()) / ".claude" / "settings.json"
    return Path.home() / ".copilot" / "settings.json"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _project(cwd: Path, explicit: Optional[str] = None) -> str:
    return explicit or os.environ.get("MEMORY_COPILOT_PROJECT") or str(cwd.resolve())


def _memory_id(source_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "ai-agent-portable-memory:copilot:" + source_id))


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
    session = Session(
        id=memory_id,
        project=project,
        title=parsed.title,
        summary=summary,
        tags=parsed.tags,
        model=model,
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )
    backend.save(session)
    return session


def save_from_hook(source_id: str, cwd: Path, project: Optional[str] = None,
                   provider: Optional[str] = None, model: Optional[str] = None) -> Optional[Session]:
    """Compress and save the current session via hook stdin (no transcript file needed).

    Copilot CLI does not expose a transcript file path in hook payloads, so this
    integration uses the checkpoint pattern: the hook writes a brief summary from
    whatever context is available rather than re-reading a rollout file.
    """
    from .core import is_recording

    if not is_recording(cwd, "copilot"):
        return None

    # Copilot may run any backend model; honor an explicit MEMORY_LLM_MODEL if set.
    resolved_model = model or os.environ.get("MEMORY_LLM_MODEL", "")
    project = _project(cwd, project)
    memory_id = _memory_id(source_id)

    # Build a minimal summary marking the session end so context is preserved.
    summary = (
        f"## Title\nCopilot session — {cwd.name}\n\n"
        f"## Project\n{project}\n\n"
        f"## What we were doing\nGitHub Copilot CLI session in {cwd}.\n\n"
        f"## Current state / where we stopped\n"
        f"Session ended at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}.\n\n"
        f"## Tags\ncopilot, copilot-cli\n"
    )
    return _store_summary(summary, source_id, project, model=resolved_model)


def load_context(cwd: Path, source_id: Optional[str] = None,
                 project: Optional[str] = None, session_id: Optional[str] = None) -> str:
    from .core import get_backend, is_recording
    from .cli import _format_context

    if not is_recording(cwd, "copilot"):
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
            matches = [s for s in sessions if s.id.startswith(session_id)]
        if len(matches) != 1:
            raise ValueError("Session ID must match exactly one saved memory.")
        session = matches[0]
    elif source_id:
        memory_id = _memory_id(source_id)
        session = backend.get(memory_id)
        if session and session.project != project:
            session = None
    if not session and not session_id:
        sessions = backend.list(project=project)
        if sessions:
            session = max(sessions, key=lambda s: s.updated_at)
    if not session:
        return ""
    return (
        "Saved project memory (historical context; verify against current files and user instructions):\n\n"
        + _format_context(session)
    )


def configure_hooks(path: Path, remove: bool = False) -> None:
    """Install or remove Copilot CLI memory hooks in a settings file.

    Hooks use Claude Code event names (SessionStart / Stop) which Copilot CLI
    recognises when it reads .claude/settings.json or its own settings.json.
    The hook command calls back into this module via `memory copilot hook`.
    """
    config = _read_json(path)
    events = config.setdefault("hooks", {})
    if not isinstance(events, dict):
        raise ValueError("The hooks field must be an object.")

    bootstrap = (
        "import sys; sys.path.insert(0, "
        + repr(str(Path(__file__).resolve().parent.parent))
        + "); from memory.copilot import copilot; copilot()"
    )
    args = [sys.executable, "-c", bootstrap, "hook"]
    command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)

    for event in ("SessionStart",) + SAVE_EVENTS:
        groups = events.get(event, [])
        if not isinstance(groups, list):
            raise ValueError(f"{event} hooks must be an array.")
        retained = []
        for group in groups:
            handlers = group.get("hooks", [])
            remaining = [h for h in handlers if h.get("statusMessage") != HOOK_MARKER]
            if remaining == handlers:
                retained.append(group)
            elif remaining:
                retained.append({**group, "hooks": remaining})
        if not remove:
            retained.append({
                "hooks": [{
                    "type": "command",
                    "command": command,
                    "timeout": 120,
                    "statusMessage": HOOK_MARKER,
                }]
            })
        if retained:
            events[event] = retained
        else:
            events.pop(event, None)

    if not remove or path.exists():
        _write_json(path, config)


@click.group()
def copilot():
    """Save, restore and automatically checkpoint GitHub Copilot CLI sessions."""


@copilot.command("save")
@click.option("--session", "source_session", help="Copilot session ID.")
@click.option("--project", help="Project label (defaults to the absolute workspace path).")
@click.option("--summary-file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Pre-written Markdown summary; skips LLM compression.")
@click.option("--no-edit", is_flag=True, help="Skip editor review.")
def save_command(source_session, project, summary_file, no_edit):
    """Save the current workspace's Copilot session to memory."""
    from .core import is_recording

    cwd = Path.cwd()
    if not is_recording(cwd, "copilot"):
        click.echo("Copilot recording is off for this directory. Run 'memory start' here first.")
        return

    source_id = source_session or "workspace:" + str(cwd.resolve())

    try:
        if summary_file:
            # Direct checkpoint path — no extra LLM call.
            session = _store_summary(
                summary_file.read_text(encoding="utf-8"),
                source_id,
                _project(cwd, project),
            )
        else:
            session = save_from_hook(source_id, cwd, project=project)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if session:
        click.echo(f"Saved Copilot memory: {session.id[:8]} ({session.title})")
    else:
        click.echo("No memory saved: recording is off or nothing to save.")


@copilot.command("load")
@click.option("--session", help="Saved memory ID, unique prefix or list number (any AI source).")
@click.option("--project", help="Project label to restore.")
def load_command(session, project):
    """Print plain Markdown for the latest project memory or a selected memory."""
    try:
        click.echo(load_context(Path.cwd(), project=project, session_id=session))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@copilot.command("checkpoint")
@click.option("--summary-file", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Structured Markdown summary written by Copilot; no compression API call.")
@click.option("--source-session", help="Copilot session ID; defaults to workspace-scoped checkpoint.")
@click.option("--project", help="Project label (defaults to the absolute workspace path).")
def checkpoint_command(summary_file, source_session, project):
    """Store a summary written by Copilot directly, without an extra LLM call."""
    from .core import is_recording

    cwd = Path.cwd()
    if not is_recording(cwd, "copilot"):
        click.echo("Copilot recording is off for this directory. Run 'memory start' here first.")
        return

    source_id = source_session or "workspace:" + str(cwd.resolve())
    try:
        session = _store_summary(
            summary_file.read_text(encoding="utf-8"),
            source_id,
            _project(cwd, project),
        )
        click.echo(f"Saved Copilot checkpoint: {session.id[:8]} ({session.title})")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@copilot.command("setup")
@click.option("--project-local", is_flag=True,
              help="Install hooks in ./.claude/settings.json (project scope).")
@click.option("--remove", is_flag=True, help="Remove only this integration's hooks.")
def setup_command(project_local, remove):
    """Install GitHub Copilot CLI memory hooks alongside existing hooks.

    By default, hooks are written to ~/.copilot/settings.json (user scope).
    Pass --project-local to write to .claude/settings.json in the current directory
    instead, which Copilot CLI also reads.
    """
    path = copilot_settings_path(project_local=project_local)
    try:
        configure_hooks(path, remove=remove)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Copilot memory hooks {'removed from' if remove else 'installed in'} {path}")
    if not remove:
        click.echo("Context will be injected at SessionStart and saved on Stop.")
        click.echo("Compression uses your MEMORY_LLM_PROVIDER and its API key.")


@copilot.command("hook", hidden=True)
def hook_command():
    """Consume Copilot CLI's hook stdin payload; never block the agent's turn."""
    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            raise ValueError("Expected a Copilot hook JSON object.")

        # Copilot CLI uses Claude Code field names when sourced from .claude/settings.json.
        name = event.get("hook_event_name") or event.get("hookEventName")
        # session_id comes in as snake_case for Claude Code-compatible hooks.
        source_id = event.get("session_id") or event.get("sessionId")
        raw_cwd = event.get("cwd") or event.get("workingDirectory")
        cwd = Path(raw_cwd) if raw_cwd else Path.cwd()

        if name == "SessionStart":
            context = load_context(cwd, source_id=source_id)
            output = (
                {"hookSpecificOutput": {"hookEventName": name, "additionalContext": context}}
                if context else {}
            )
        elif name in SAVE_EVENTS:
            if source_id:
                save_from_hook(source_id, cwd)
            else:
                click.echo("memory: Copilot hook is missing session_id; skipping save.", err=True)
            output = {}
        else:
            output = {}

        click.echo(json.dumps(output))
    except Exception as exc:
        # Advisory hooks must not turn a failure into a blocked agent turn.
        click.echo(f"memory: Copilot hook failed: {exc}", err=True)
        click.echo("{}")


if __name__ == "__main__":
    copilot()
