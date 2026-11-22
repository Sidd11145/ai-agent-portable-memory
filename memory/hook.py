"""Universal agent hook — one install, works with any AI agent.

Supports: Claude Code, GitHub Copilot CLI, Codex, Cursor, Windsurf, and
any agent that speaks the Claude Code hook protocol (SessionStart / Stop /
PostCompact).

Usage
-----
Install once (user-global):

    memory install

Install for this project only:

    memory install --local

Then set your LLM key (pick any one):

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GROQ_API_KEY=gsk_...

That's it. The hook auto-detects which agent fired it at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .backends.base import Session

HOOK_MARKER = "AI Agent Portable Memory"

# Hook events we care about — same names work for Claude Code and Copilot CLI.
_SESSION_START = "SessionStart"
_SAVE_EVENTS = {"Stop", "PostCompact", "SessionEnd"}
_LINK_EVENTS = {"UserPromptSubmit"}

# Env vars each agent sets for its *own* selected model. Reading these is not
# "guessing the provider from the agent name" — it is reading the model the user
# chose when they opened that agent. Absent that, the user sets MEMORY_LLM_*.
_AGENT_MODEL_ENV = {
    "claude": ("ANTHROPIC_MODEL",),
    "codex": ("CODEX_MODEL",),
}
_MODEL_ENV_PROVIDER = {
    "ANTHROPIC_MODEL": "anthropic",
    "CODEX_MODEL": "openai",
}

# Which env var(s) hold the credential for each provider. ollama is local (no key).
_PROVIDER_KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "ollama": (),
}


# ── Agent detection ───────────────────────────────────────────────────────────

def detect_agent() -> str:
    """Return the name of the currently running AI agent based on environment signals."""
    env = os.environ

    # Copilot CLI sets COPILOT_* variables
    if env.get("COPILOT_SESSION_ID") or env.get("GH_COPILOT_VERSION"):
        return "copilot"

    # Codex sets CODEX_* or OPENAI_CODEX_* variables
    if env.get("CODEX_SESSION_ID") or env.get("CODEX_HOME"):
        return "codex"

    # Claude Code sets CLAUDE_* or ANTHROPIC_* proxy variables
    if (env.get("CLAUDE_SESSION_ID") or env.get("ANTHROPIC_AUTH_TOKEN")
            or env.get("CLAUDE_CODE_ENTRYPOINT")):
        return "claude"

    # Cursor: VS Code fork, sets CURSOR_TRACE_ID or CURSOR_SESSION_ID
    if env.get("CURSOR_TRACE_ID") or env.get("CURSOR_SESSION_ID"):
        return "cursor"

    # Windsurf: sets WINDSURF_SESSION_ID
    if env.get("WINDSURF_SESSION_ID"):
        return "windsurf"

    return "unknown"


# ── Settings file resolution ──────────────────────────────────────────────────

def _settings_paths(local: bool = False, cwd: Optional[Path] = None) -> list[Path]:
    """Return settings files to install hooks into.

    Each entry is only included if that agent appears to be installed on this machine,
    so we never write into another tool's config uninvited.

    local=True writes only to ./.claude/settings.json (project scope, read by both
    Claude Code and Copilot CLI from the repo root).
    """
    base = cwd or Path.cwd()

    if local:
        return [base / ".claude" / "settings.json"]

    paths: list[Path] = []

    # Copilot CLI — present if ~/.copilot/ exists or `copilot` binary is on PATH.
    import shutil
    copilot_home = Path.home() / ".copilot"
    if copilot_home.exists() or shutil.which("copilot") or shutil.which("gh-copilot"):
        paths.append(copilot_home / "settings.json")

    # Claude Code — present if ~/.claude/ exists or `claude` binary is on PATH.
    claude_home = Path.home() / ".claude"
    if claude_home.exists() or shutil.which("claude"):
        paths.append(claude_home / "settings.json")

    return paths


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


# ── Hook command builder ──────────────────────────────────────────────────────

def _hook_command() -> str:
    """Return the shell command that runs the universal hook handler."""
    bootstrap = (
        "import sys; sys.path.insert(0, "
        + repr(str(Path(__file__).resolve().parent.parent))
        + "); from memory.hook import _run_hook; _run_hook()"
    )
    args = [sys.executable, "-c", bootstrap]
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


# ── Install / uninstall ───────────────────────────────────────────────────────

def configure_hooks(path: Path, remove: bool = False) -> None:
    """Add or remove the universal memory hook in a single settings file."""
    config = _read_json(path)
    events = config.setdefault("hooks", {})
    if not isinstance(events, dict):
        events = {}
        config["hooks"] = events

    command = _hook_command()

    for event in (_SESSION_START, *sorted(_SAVE_EVENTS), *sorted(_LINK_EVENTS)):
        groups: list = events.get(event, [])
        if not isinstance(groups, list):
            groups = []

        # Strip our existing hook from every group.
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

    if not events:
        config.pop("hooks", None)

    if not remove or path.exists():
        _write_json(path, config)


# ── Session state ─────────────────────────────────────────────────────────────

def _memory_id(source_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "ai-agent-portable-memory:universal:" + source_id))


def _digest_path(memory_id: str) -> Path:
    return Path.home() / ".ai-memory" / "hook" / (memory_id + ".json")


def _project(cwd: Path, explicit: Optional[str] = None) -> str:
    return (explicit
            or os.environ.get("MEMORY_PROJECT")
            or cwd.name  # just the folder name, not the full path
            or "default")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_session(source_id: str, cwd: Path, transcript: str = "",
                 agent: str = "unknown", project: Optional[str] = None,
                 provider: Optional[str] = None, model: Optional[str] = None,
                 event: Optional[dict] = None) -> Optional[Session]:
    """Compress and save a session. Returns None if not recording or nothing to save."""
    from .core import get_backend, is_recording, resolve_backend
    from .compressor import compress, _parse_response, scrub_secrets

    # Opt-in gate: only save if this (directory, agent) pair was explicitly started.
    if not is_recording(cwd, agent):
        return None

    if not transcript.strip():
        return None

    # Resolve provider + model — never guess from the agent name.
    resolved_provider = provider
    resolved_model = model
    if resolved_provider is None:
        resolved_provider, env_model = resolve_provider_model(agent, event)
        if resolved_model is None:
            resolved_model = env_model
    if not resolved_provider:
        print("memory: no provider set (set MEMORY_LLM_PROVIDER in this tab); skipping save",
              file=sys.stderr)
        return None
    if not _provider_key_present(resolved_provider):
        print(f"memory: no API key for provider '{resolved_provider}'; skipping save "
              f"(set its key or MEMORY_LLM_PROVIDER)", file=sys.stderr)
        return None

    project_label = _project(cwd, project)
    # Memory identity is (project, model): the same source under a different model
    # is a distinct memory, but shares the project label.
    memory_id = _memory_id(f"{source_id}:{resolved_model or 'default'}")
    digest = hashlib.sha256(transcript.encode()).hexdigest()
    digest_file = _digest_path(memory_id)
    backend_name = resolve_backend(cwd)

    # Skip if transcript hasn't changed since last save.
    if digest_file.exists():
        try:
            state = json.loads(digest_file.read_text(encoding="utf-8"))
            if state.get("digest") == digest:
                backend = get_backend(backend_name)
                backend.init()
                existing = backend.get(memory_id)
                if existing:
                    return existing
        except (json.JSONDecodeError, OSError):
            pass

    compressed = compress(transcript, provider=resolved_provider, model=resolved_model)
    summary = scrub_secrets(compressed.summary)
    parsed = _parse_response(summary)

    backend = get_backend(backend_name)
    backend.init()
    existing = backend.get(memory_id)
    now = datetime.utcnow()
    session = Session(
        id=memory_id,
        project=project_label,
        title=parsed.title,
        summary=summary,
        tags=list({*parsed.tags, agent.lower()}),
        model=resolved_model or "",
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )
    backend.save(session)

    digest_file.parent.mkdir(parents=True, exist_ok=True)
    digest_file.write_text(json.dumps({"digest": digest, "agent": agent}), encoding="utf-8")
    return session


def resolve_provider_model(agent: str, event: Optional[dict] = None) -> tuple[Optional[str], Optional[str]]:
    """Return (provider, model) to compress with, or (None, None) to skip.

    Never guesses a provider from the agent name. Resolution order:
      (a) explicit MEMORY_LLM_PROVIDER / MEMORY_LLM_MODEL set by the user per tab
      (b) the model the agent itself exposes in its environment (e.g. Claude's
          ANTHROPIC_MODEL), mapped to its provider
      (c) the model actually seen in the transcript, threaded in via
          event["_meta_model"] by the transcript reader (ground truth)
      (d) nothing known → (None, None); the caller skips with a clear message

    The transcript's own model (c) is the ground truth of what ran, so it fills
    in the model whenever the user hasn't set an explicit MEMORY_LLM_MODEL
    override — even if the provider was resolved from MEMORY_LLM_PROVIDER or a
    (possibly stale) agent env var.
    """
    event = event or {}
    meta_model = event.get("_meta_model")

    prov = os.environ.get("MEMORY_LLM_PROVIDER")
    model = os.environ.get("MEMORY_LLM_MODEL")
    if prov:
        return prov, model or meta_model

    for env_var in _AGENT_MODEL_ENV.get(agent, ()):
        val = os.environ.get(env_var)
        if val:
            return _MODEL_ENV_PROVIDER.get(env_var), meta_model or val

    if meta_model:
        if agent == "codex":
            return "openai", meta_model
        if agent == "claude":
            return "anthropic", meta_model

    return None, None


def _provider_key_present(provider: str) -> bool:
    """True if the credential for this provider is available (ollama needs none)."""
    env_vars = _PROVIDER_KEY_ENV.get(provider)
    if env_vars is None:
        return False  # unknown provider — be strict, don't call anything
    if not env_vars:
        return True  # local provider (ollama)
    return any(os.environ.get(v) for v in env_vars)


def _message_count(transcript: str) -> int:
    """Count User:/Assistant: message blocks produced by the transcript readers."""
    return sum(
        1 for block in transcript.split("\n\n")
        if block.startswith("User:") or block.startswith("Assistant:")
    )


# ── Load ──────────────────────────────────────────────────────────────────────

def load_context(cwd: Path, project: Optional[str] = None,
                 session_id: Optional[str] = None, agent: Optional[str] = None) -> str:
    from .core import get_backend, is_recording_session, resolve_backend
    from .cli import _format_context

    # Opt-in gate: only inject context if this window (session) was started.
    resolved_agent = agent or detect_agent()
    if not is_recording_session(cwd, resolved_agent, session_id=session_id,
                                port=os.environ.get("CLAUDE_CODE_SSE_PORT")):
        return ""

    backend = get_backend(resolve_backend(cwd, session_id=session_id))
    backend.init()
    project_label = _project(cwd, project)

    session = None
    if session_id:
        all_sessions = backend.list()
        if session_id.isdigit():
            index = int(session_id) - 1
            matches = all_sessions[index:index + 1] if index >= 0 else []
        else:
            matches = [s for s in all_sessions if s.id.startswith(session_id)]
        if len(matches) == 1:
            session = matches[0]
    else:
        candidates = backend.list(project=project_label)
        if candidates:
            session = max(candidates, key=lambda s: s.updated_at)

    if not session:
        return ""

    return (
        "Saved project memory (historical context; verify against current files and user instructions):\n\n"
        + _format_context(session)
    )


# ── Universal hook handler ────────────────────────────────────────────────────

def _recording_setup_context(session_id: str, cwd: Path) -> str:
    """Give Claude the exact commands for applying this session's user choice."""
    from .core import get_recording_decision

    if get_recording_decision(session_id) is not None:
        return ""
    start_command = shlex.join([
        "memory", "start", "--agent", "claude", "--session-id", session_id])
    stop_command = shlex.join([
        "memory", "stop", "--agent", "claude", "--session-id", session_id])
    return (
        "Portable memory setup for this session. Recording is awaiting your choice.\n"
        "At your next response, ask the user whether to record this session now. "
        "Use AskUserQuestion when available. If the user agrees, ask which storage "
        "backend to use, offering SQLite (local), Firebase, Supabase, and S3. "
        "If the user already supplied these choices, apply them. If you already "
        "asked and are awaiting an answer, wait for that answer.\n"
        f"Run commands from {shlex.quote(str(cwd))}.\n"
        f"To enable recording, run {start_command} --backend <chosen-backend>, "
        "where chosen-backend is sqlite, firebase, supabase, or s3.\n"
        f"If the user declines or wants to decide later, run {stop_command}. "
        "They can enable recording later with the start command.\n"
        "Confirm the command result before reporting recording as enabled."
    )


def _run_hook() -> None:
    """Entry point called by every agent's hook command.

    Reads the hook payload from stdin (Claude Code / Copilot CLI protocol),
    detects which agent fired it, then loads or saves accordingly.
    Never raises — advisory hooks must not block the agent's turn.
    """
    try:
        raw = sys.stdin.read()
        event: dict = json.loads(raw) if raw.strip() else {}

        # Both Claude Code and Copilot CLI use snake_case field names when
        # hooks are defined under Claude Code event names.
        name = event.get("hook_event_name") or event.get("hookEventName", "")
        source_id = (event.get("session_id") or event.get("sessionId")
                     or event.get("source_id") or "unknown")
        raw_cwd = event.get("cwd") or event.get("workingDirectory")
        cwd = Path(raw_cwd) if raw_cwd else Path.cwd()
        agent = detect_agent()

        if name == _SESSION_START:
            from .core import (prepare_recording_session, refresh_and_promote,
                               prune_session_backends, write_session_marker,
                               get_claude_process_identity)
            sid = None if source_id == "unknown" else source_id
            if sid and agent == "claude":
                prepare_recording_session(sid, event.get("source") or "startup")

            port = os.environ.get("CLAUDE_CODE_SSE_PORT")
            refresh_and_promote(port, sid, promote_choices=agent != "claude")
            if sid:
                write_session_marker(sid, port, cwd, claude_process=(
                    get_claude_process_identity() if agent == "claude" else None))
            prune_session_backends(keep_sessions=[sid] if sid else [])
            setup = _recording_setup_context(sid, cwd) if sid and agent == "claude" else ""

            # Drain any transcripts queued by a previous (possibly killed) session.
            # Safe here: the process is alive and stable, so the slow LLM call can
            # run without racing window-close.
            try:
                from . import queue as _queue
                _queue.drain()
            except Exception:
                pass
            try:
                context = load_context(cwd, agent=agent, session_id=sid)
            except Exception as exc:
                print(f"memory: context load failed: {exc}", file=sys.stderr)
                context = ""
            if sid and agent == "claude":
                context = (
                    f"Portable memory session ID: {sid}\n" + setup + "\n" + context
                )
            output = (
                {"hookSpecificOutput": {"hookEventName": name, "additionalContext": context}}
                if context else {}
            )

        elif name in _LINK_EVENTS:
            # Refresh the link and marker each turn. Claude's commands apply
            # choices directly to the session ID supplied in this hook payload.
            # Pure disk — no LLM, network, or save.
            from .core import refresh_and_promote, write_session_marker, get_claude_process_identity
            port = os.environ.get("CLAUDE_CODE_SSE_PORT")
            sid = None if source_id == "unknown" else source_id
            refresh_and_promote(port, sid, promote_choices=agent != "claude")
            if sid:
                write_session_marker(sid, port, cwd, claude_process=(
                    get_claude_process_identity() if agent == "claude" else None))
            setup = _recording_setup_context(sid, cwd) if sid and agent == "claude" else ""
            output = (
                {"hookSpecificOutput": {"hookEventName": name, "additionalContext": setup}}
                if setup else {}
            )

        elif name in _SAVE_EVENTS:
            transcript = _read_transcript(event, agent, cwd)
            if transcript and source_id != "unknown":
                # Smart compaction: Stop/SessionEnd always save; PostCompact only
                # saves when the conversation is long enough, to avoid recompressing
                # on every context compaction in long sessions.
                should_save = True
                if name == "PostCompact":
                    threshold = int(os.environ.get("MEMORY_MIN_MESSAGES", "10"))
                    should_save = _message_count(transcript) >= threshold
                if should_save:
                    # Capture the raw transcript instantly and compress later on the
                    # next SessionStart. A plain disk write survives window-close,
                    # whereas an inline LLM call here would be killed mid-flight.
                    from .core import is_recording_session, resolve_backend
                    from . import queue as _queue
                    if is_recording_session(
                            cwd, agent, session_id=source_id,
                            port=os.environ.get("CLAUDE_CODE_SSE_PORT")):
                        _queue.enqueue(
                            source_id, cwd, agent, transcript,
                            meta_model=event.get("_meta_model"),
                            backend=resolve_backend(cwd, session_id=source_id) or "sqlite",
                        )
            # Stop finishes a response. SessionEnd finishes the session, including
            # sessions whose transcript is empty or below the compaction threshold.
            if name == "SessionEnd" and source_id != "unknown":
                from .core import (clear_recording_for_session, remove_session_marker,
                                   get_session_for_port)
                clear_recording_for_session(source_id, agent)
                port = os.environ.get("CLAUDE_CODE_SSE_PORT")
                if port and get_session_for_port(port) == source_id:
                    clear_recording_for_session(port, agent, table="ports")
                remove_session_marker(source_id)
            output = {}

        else:
            output = {}

        print(json.dumps(output))

    except Exception as exc:
        print(f"memory: hook error: {exc}", file=sys.stderr)
        print("{}")


def _read_transcript(event: dict, agent: str, cwd: Path) -> str:
    """Try to read the session transcript for the detected agent.

    Side effect: for Codex, stashes the model from the rollout metadata into
    event["_meta_model"] so resolve_provider_model can match it.
    """
    # Codex provides transcript_path directly in the hook payload.
    path = event.get("transcript_path")
    if path and Path(path).exists():
        from .codex import read_transcript as _read_codex
        transcript, metadata = _read_codex(Path(path))
        model = metadata.get("model") or metadata.get("model_slug")
        if model:
            event["_meta_model"] = model
        return transcript

    # Claude Code: read from ~/.claude/projects/
    if agent == "claude":
        from .cli import _read_claude_transcript
        session_id = event.get("session_id") or event.get("sessionId")
        transcript, model, _ = _read_claude_transcript(session_id)
        if model:
            event["_meta_model"] = model
        return transcript

    return ""
