import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from memory import codex as integration
from memory.cli import cli
from memory.compressor import CompressedSummary


SUMMARY = "## Title\nCodex checkpoint\n\n## Project\ndefault\n\n## Next steps\nFinish the feature.\n\n## Tags\npython, codex"


def make_runner():
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:  # Click 8.2+ always captures stdout and stderr separately.
        return CliRunner()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from memory import core

    task_home = tmp_path / "home"
    task_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: task_home))
    monkeypatch.setenv("CODEX_HOME", str(task_home / ".codex"))
    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "memory.db"))
    # Local provider needs no API key, so saves reach the (mocked) compressor
    # regardless of the host environment. Tests that assert a specific provider
    # still override this via the explicit --provider flag.
    monkeypatch.setenv("MEMORY_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("MEMORY_CODEX_PROJECT", raising=False)
    monkeypatch.setattr(core, "_STOPPED_FILE", task_home / ".ai-memory" / "STOPPED")
    monkeypatch.setattr(core, "_ACTIVE_SESSION_FILE", task_home / ".ai-memory" / "active_session")
    monkeypatch.setattr(core, "_ENABLED_FILE", task_home / ".ai-memory" / "enabled.json")
    # These integration tests assume Codex recording is ON everywhere; the opt-in
    # gate is exercised separately in test_recording.py.
    monkeypatch.setattr(core, "is_recording", lambda cwd, agent: not core.is_stopped())
    monkeypatch.setattr("memory.codex.is_recording",
                        lambda cwd, agent: not core.is_stopped(), raising=False)
    calls = []

    def compress(transcript, provider, model):
        calls.append((transcript, provider, model))
        return CompressedSummary("Codex checkpoint", "default", SUMMARY, ["python", "codex"])

    monkeypatch.setattr("memory.compressor.compress", compress)
    return tmp_path, calls


def rollout(path, source_id="codex-session-1", cwd="/workspace/one", answer="Implemented it."):
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {"type": "session_meta", "payload": {"id": source_id, "cwd": cwd}},
        {"type": "response_item", "payload": {"type": "message", "role": "developer",
                                                  "content": [{"type": "input_text", "text": "Private instructions."}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": "Build a feature."}]}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "Build a feature."}},
        {"type": "response_item", "payload": {"type": "function_call_output", "output": "Tool output."}},
        {"type": "response_item", "payload": {"type": "reasoning", "text": "Private reasoning."}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                  "content": [{"type": "output_text", "text": answer}]}},
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + '\n{"partial":', encoding="utf-8")
    return path


def test_transcript_reads_conversation_without_duplicate_events_or_instructions(tmp_path):
    path = rollout(tmp_path / "rollout.jsonl")
    transcript, metadata = integration.read_transcript(path)
    assert transcript == "User: Build a feature.\n\nAssistant: Implemented it."
    assert metadata["id"] == "codex-session-1"


def test_discovery_filters_workspace_and_supports_archived_sessions(isolated):
    home = integration.codex_home()
    one = rollout(home / "sessions" / "2026" / "rollout-one.jsonl")
    rollout(home / "sessions" / "2026" / "rollout-two.jsonl", "codex-session-2", "/workspace/two")
    archived = rollout(home / "archived_sessions" / "rollout-old.jsonl", "old-session", "/workspace/old")
    assert integration.find_transcript(cwd=Path("/workspace/one")) == one
    assert integration.find_transcript("old-session") == archived
    with pytest.raises(ValueError, match="No matching"):
        integration.find_transcript("missing")


def test_saves_deduplicate_update_and_preserve_claude_active_memory(isolated):
    from memory.core import get_active_session_id, get_backend, set_active_session_id

    root, calls = isolated
    set_active_session_id("claude-active-id")
    path = rollout(root / "rollout.jsonl")
    first = integration.save_transcript(path)
    assert first.project == str(Path("/workspace/one").resolve())
    assert integration.save_transcript(path).id == first.id
    assert len(calls) == 1
    rollout(path, answer="Finished tests too.")
    updated = integration.save_transcript(path)
    assert updated.id == first.id
    assert updated.created_at == first.created_at
    assert len(calls) == 2
    second = integration.save_transcript(rollout(root / "other.jsonl", "codex-session-2"))
    assert second.id != first.id
    assert len(get_backend().list()) == 2
    assert get_active_session_id() == "claude-active-id"
    get_backend().delete(first.id)
    assert integration.save_transcript(path).id == first.id
    assert len(calls) == 4


def test_restore_uses_project_scope_and_accepts_memories_from_other_sources(isolated):
    from memory.core import create_session

    root, _ = isolated
    session = integration.save_transcript(rollout(root / "rollout.jsonl"))
    assert "Finish the feature." in integration.load_context(Path("/workspace/one"))
    assert integration.load_context(Path("/workspace/two"), source_id="codex-session-1") == ""
    shared = create_session("Claude checkpoint", "shared-project", SUMMARY, [])
    assert "Claude checkpoint" in integration.load_context(Path("/workspace/two"), session_id=shared.id[:8])
    assert "Codex checkpoint" in integration.load_context(Path("/workspace/one"), session_id=session.id)
    with pytest.raises(ValueError, match="exactly one"):
        integration.load_context(Path("/workspace/one"), session_id="missing")


def test_installer_is_idempotent_preserves_other_hooks_and_runs_bootstrap(isolated):
    root, _ = isolated
    path = root / "hooks.json"
    existing = {"description": "Keep me", "hooks": {"Stop": [{"matcher": "", "hooks": [
        {"type": "command", "command": "echo existing"}]}], "PreToolUse": []}}
    path.write_text(json.dumps(existing))
    integration.configure_hooks(path)
    first = json.loads(path.read_text())
    integration.configure_hooks(path)
    assert json.loads(path.read_text()) == first
    assert first["hooks"]["Stop"][0] == existing["hooks"]["Stop"][0]
    command = first["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    result = subprocess.run(command, shell=True, input='{"hook_event_name":"Unknown"}',
                            cwd=root, text=True, capture_output=True)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {}
    integration.configure_hooks(path, remove=True)
    assert json.loads(path.read_text()) == existing


def test_removal_preserves_other_handlers_in_a_shared_group(isolated):
    root, _ = isolated
    path = root / "hooks.json"
    other = {"type": "command", "command": "echo other"}
    path.write_text(json.dumps({"hooks": {"Stop": [{"matcher": "", "hooks": [other,
        {"type": "command", "command": "ours", "statusMessage": integration.HOOK_MARKER}]}]}}))
    integration.configure_hooks(path, remove=True)
    assert json.loads(path.read_text())["hooks"]["Stop"] == [{"matcher": "", "hooks": [other]}]


def test_cli_hooks_save_restore_and_deduplicate_across_events(isolated):
    root, calls = isolated
    path = rollout(root / "rollout.jsonl")
    event = {"hook_event_name": "Stop", "session_id": "codex-session-1",
             "cwd": "/workspace/one", "transcript_path": str(path)}
    runner = make_runner()
    for name in integration.SAVE_EVENTS:
        result = runner.invoke(cli, ["codex", "hook"], input=json.dumps({**event, "hook_event_name": name}))
        assert result.exit_code == 0
        assert json.loads(result.stdout) == {}
    assert len(calls) == 1
    result = runner.invoke(cli, ["codex", "hook"], input=json.dumps({**event, "hook_event_name": "SessionStart"}))
    output = json.loads(result.stdout)["hookSpecificOutput"]
    assert output["hookEventName"] == "SessionStart"
    assert "Finish the feature." in output["additionalContext"]
    assert "decision" not in json.loads(result.stdout)


def test_hook_failures_are_advisory_and_do_not_write_state(isolated, monkeypatch):
    root, _ = isolated
    path = rollout(root / "rollout.jsonl")

    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("memory.compressor.compress", fail)
    result = make_runner().invoke(cli, ["codex", "hook"], input=json.dumps({
        "hook_event_name": "Stop", "session_id": "codex-session-1",
        "cwd": "/workspace/one", "transcript_path": str(path)}))
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {}
    assert "provider unavailable" in result.stderr
    assert not list((Path.home() / ".ai-memory" / "codex").glob("*.json"))


def test_missing_transcript_and_stopped_memory_skip_compression(isolated):
    from memory.core import set_stopped

    root, calls = isolated
    event = {"hook_event_name": "Stop", "session_id": "one", "cwd": str(root), "transcript_path": None}
    result = make_runner().invoke(cli, ["codex", "hook"], input=json.dumps(event))
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {}
    assert "no transcript_path" in result.stderr
    set_stopped(True)
    assert integration.save_transcript(rollout(root / "rollout.jsonl")) is None
    assert integration.load_context(root) == ""
    assert calls == []


def test_manual_codex_save_exposes_provider_options_and_does_not_use_editor(isolated):
    root, calls = isolated
    result = make_runner().invoke(cli, ["codex", "save", "--transcript",
        str(rollout(root / "rollout.jsonl")), "--provider", "ollama",
        "--model", "local-model", "--project", "shared", "--no-edit"])
    assert result.exit_code == 0, result.output
    assert "Saved Codex memory" in result.output
    assert calls[0][1:] == ("ollama", "local-model")
    context = make_runner().invoke(cli, ["codex", "load", "--project", "shared"])
    assert context.exit_code == 0
    assert "**Project:** shared" in context.output


def test_direct_checkpoint_needs_no_compression_call_and_updates_workspace_memory(isolated, monkeypatch):
    from memory.core import get_backend, get_active_session_id, set_active_session_id

    root, calls = isolated
    monkeypatch.chdir(root)
    set_active_session_id("claude-active-id")
    summary = root / "summary.md"
    summary.write_text(SUMMARY + "\nExample key: sk-proj-testing-only-key")
    first = make_runner().invoke(cli, ["codex", "checkpoint", "--summary-file", str(summary)])
    assert first.exit_code == 0, first.output
    session = get_backend().list()[0]
    assert "sk-proj-testing-only-key" not in session.summary
    assert "[OPENAI_API_KEY]" in session.summary
    summary.write_text(SUMMARY.replace("Finish the feature.", "Feature complete."))
    second = make_runner().invoke(cli, ["codex", "checkpoint", "--summary-file", str(summary)])
    assert second.exit_code == 0, second.output
    sessions = get_backend().list()
    assert len(sessions) == 1
    assert sessions[0].id == session.id
    assert "Feature complete." in integration.load_context(root)
    assert get_active_session_id() == "claude-active-id"
    assert calls == []


def test_invalid_direct_summary_does_not_save_a_memory(isolated):
    from memory.core import get_backend

    root, calls = isolated
    summary = root / "summary.md"
    summary.write_text("No title section.")
    result = make_runner().invoke(cli, ["codex", "checkpoint", "--summary-file", str(summary)])
    assert result.exit_code == 1
    backend = get_backend()
    backend.init()
    assert backend.list() == []
    assert calls == []


def test_setup_command_selects_only_requested_scope(isolated, monkeypatch):
    root, _ = isolated
    monkeypatch.chdir(root)
    result = make_runner().invoke(cli, ["codex", "setup", "--project-local"])
    assert result.exit_code == 0, result.output
    assert (root / ".codex" / "hooks.json").exists()
    assert not (integration.codex_home() / "hooks.json").exists()
    assert "/hooks" in result.output


def test_existing_cli_options_remain_available():
    runner = make_runner()
    save = runner.invoke(cli, ["save", "--help"])
    assert save.exit_code == 0
    for option in ("--from-claude", "--from-chatgpt", "--from-cursor", "--from-windsurf", "--from-file"):
        assert option in save.output
    setup = runner.invoke(cli, ["setup", "--help"])
    assert setup.exit_code == 0
    for option in ("--claude-hook", "--compact-hook", "--remove-claude-hook", "--remove-compact-hook"):
        assert option in setup.output
