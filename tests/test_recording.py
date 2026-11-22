import json
import os
from unittest.mock import patch

import pytest

from memory import core
from memory import hook


# ── Per-(dir + agent) recording state ──────────────────────────────────────────

@pytest.fixture
def state(tmp_path, monkeypatch):
    """Redirect core's state files into a tmp dir so tests never touch ~/.ai-memory."""
    monkeypatch.setattr(core, "_ENABLED_FILE", tmp_path / "enabled.json")
    monkeypatch.setattr(core, "_STOPPED_FILE", tmp_path / "STOPPED")
    monkeypatch.setattr(core, "_SESSION_BACKEND_MAP_FILE", tmp_path / "session_backends.json")
    monkeypatch.setattr(core, "_SESSION_MARKERS_DIR", tmp_path / "current_sessions")
    return tmp_path


def test_recording_off_by_default(state, tmp_path):
    assert core.is_recording(tmp_path, "claude") is False


def test_enable_then_recording(state, tmp_path):
    core.enable_recording(tmp_path, "claude")
    assert core.is_recording(tmp_path, "claude") is True
    # A different agent in the same dir is still off.
    assert core.is_recording(tmp_path, "codex") is False


def test_two_agents_one_dir_independent(state, tmp_path):
    core.enable_recording(tmp_path, "claude")
    core.enable_recording(tmp_path, "codex")
    assert sorted(core.recording_agents(tmp_path)) == ["claude", "codex"]
    core.disable_recording(tmp_path, "codex")
    assert core.recording_agents(tmp_path) == ["claude"]
    assert core.is_recording(tmp_path, "codex") is False


def test_disable_last_agent_drops_key(state, tmp_path):
    core.enable_recording(tmp_path, "claude")
    core.disable_recording(tmp_path, "claude")
    assert core.recording_agents(tmp_path) == []


def test_two_dirs_isolated(state, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    core.enable_recording(dir_a, "claude")
    assert core.is_recording(dir_a, "claude") is True
    assert core.is_recording(dir_b, "claude") is False


def test_global_kill_switch_overrides(state, tmp_path):
    core.enable_recording(tmp_path, "claude")
    core.set_stopped(True)
    assert core.is_recording(tmp_path, "claude") is False
    core.set_stopped(False)
    assert core.is_recording(tmp_path, "claude") is True


# ── Provider / model resolution (never guess from agent name) ───────────────────

def test_resolve_explicit_env_wins():
    with patch.dict(os.environ, {"MEMORY_LLM_PROVIDER": "openai",
                                 "MEMORY_LLM_MODEL": "gpt-4o"}, clear=False):
        assert hook.resolve_provider_model("copilot", {}) == ("openai", "gpt-4o")


def test_resolve_from_agent_exposed_model():
    env = {k: v for k, v in os.environ.items()
           if k not in ("MEMORY_LLM_PROVIDER", "MEMORY_LLM_MODEL")}
    env["ANTHROPIC_MODEL"] = "claude-sonnet-4-6"
    with patch.dict(os.environ, env, clear=True):
        assert hook.resolve_provider_model("claude", {}) == ("anthropic", "claude-sonnet-4-6")


def test_resolve_codex_meta_model():
    env = {k: v for k, v in os.environ.items()
           if k not in ("MEMORY_LLM_PROVIDER", "MEMORY_LLM_MODEL", "CODEX_MODEL")}
    with patch.dict(os.environ, env, clear=True):
        assert hook.resolve_provider_model("codex", {"_meta_model": "gpt-5"}) == ("openai", "gpt-5")


def test_resolve_nothing_known_returns_none():
    env = {k: v for k, v in os.environ.items()
           if k not in ("MEMORY_LLM_PROVIDER", "MEMORY_LLM_MODEL", "ANTHROPIC_MODEL", "CODEX_MODEL")}
    with patch.dict(os.environ, env, clear=True):
        assert hook.resolve_provider_model("copilot", {}) == (None, None)


def test_provider_key_present():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x"}, clear=False):
        assert hook._provider_key_present("openai") is True
    assert hook._provider_key_present("ollama") is True  # local, no key
    assert hook._provider_key_present("made-up-provider") is False


# ── Skip-save when no provider resolvable ───────────────────────────────────────

def test_save_session_skips_without_provider(state, tmp_path, capsys):
    core.enable_recording(tmp_path, "copilot")
    env = {k: v for k, v in os.environ.items()
           if k not in ("MEMORY_LLM_PROVIDER", "MEMORY_LLM_MODEL", "ANTHROPIC_MODEL", "CODEX_MODEL")}
    with patch.dict(os.environ, env, clear=True):
        result = hook.save_session("src", tmp_path, transcript="User: hi\n\nAssistant: yo",
                                   agent="copilot")
    assert result is None
    assert "no provider set" in capsys.readouterr().err


def test_save_session_skips_when_not_recording(state, tmp_path):
    # Not enabled → returns None without even resolving a provider.
    assert hook.save_session("src", tmp_path, transcript="User: hi",
                             agent="claude") is None


# ── Compaction gate (message count) ─────────────────────────────────────────────

def test_message_count():
    transcript = "User: a\n\nAssistant: b\n\nUser: c"
    assert hook._message_count(transcript) == 3
    assert hook._message_count("") == 0


# ── (project, model) identity ───────────────────────────────────────────────────

class _Compressed:
    def __init__(self, summary):
        self.summary = summary


def test_same_source_two_models_two_sessions(state, tmp_path, monkeypatch):
    """Saving one source under model A then model B yields two rows, one project."""
    from memory.backends.sqlite import SQLiteBackend

    db = SQLiteBackend(db_path=tmp_path / "mem.db")
    db.init()

    monkeypatch.setattr(hook, "_digest_path",
                        lambda mid: tmp_path / "digests" / (mid + ".json"))
    monkeypatch.setattr("memory.core.get_backend", lambda name=None: db)
    monkeypatch.setattr(hook, "get_backend", lambda name=None: db, raising=False)
    # save_session imports get_backend from .core inside the function, so patch there.
    import memory.core as _core
    monkeypatch.setattr(_core, "get_backend", lambda name=None: db)

    def fake_compress(transcript, provider=None, model=None):
        return _Compressed(f"## Title\nWork with {model}\n\n## Project\nmyproj\n\n## Tags\nx")
    monkeypatch.setattr("memory.compressor.compress", fake_compress)
    monkeypatch.setattr(hook, "_provider_key_present", lambda provider: True)

    core.enable_recording(tmp_path, "claude")

    s_a = hook.save_session("src", tmp_path, transcript="User: hi\n\nAssistant: yo",
                            agent="claude", provider="anthropic", model="model-a",
                            project="myproj")
    s_b = hook.save_session("src", tmp_path, transcript="User: hi again\n\nAssistant: yo2",
                            agent="claude", provider="anthropic", model="model-b",
                            project="myproj")

    assert s_a is not None and s_b is not None
    assert s_a.id != s_b.id
    assert s_a.model == "model-a"
    assert s_b.model == "model-b"
    listed = db.list(project="myproj")
    assert len(listed) == 2
    assert {x.model for x in listed} == {"model-a", "model-b"}


# ── Per-window (Claude session) backend routing ─────────────────────────────────

def test_session_backend_round_trip(state):
    assert core.get_backend_for_session("sid-1") is None
    core.set_backend_for_session("sid-1", "firebase")
    assert core.get_backend_for_session("sid-1") == "firebase"
    core.clear_backend_for_session("sid-1")
    assert core.get_backend_for_session("sid-1") is None


def test_set_session_backend_rejects_unknown(state):
    with pytest.raises(ValueError):
        core.set_backend_for_session("sid-1", "not-a-backend")


def test_resolve_backend_precedence(state, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MEMORY_BACKEND"}
    with patch.dict(os.environ, env, clear=True):
        # Nothing set anywhere → None (get_backend picks its sqlite default).
        assert core.resolve_backend(tmp_path) is None
        # Env default applies.
        with patch.dict(os.environ, {"MEMORY_BACKEND": "s3"}, clear=False):
            assert core.resolve_backend(tmp_path) == "s3"
            # Explicit override beats env.
            assert core.resolve_backend(tmp_path, override="sqlite") == "sqlite"


def test_resolve_override_beats_session(state, tmp_path):
    core.set_backend_for_session("sid-1", "firebase")
    assert core.resolve_backend(tmp_path, override="sqlite", session_id="sid-1") == "sqlite"


def test_resolve_precedence_session_over_env(state, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MEMORY_BACKEND"}
    with patch.dict(os.environ, {**env, "MEMORY_BACKEND": "s3"}, clear=True):
        core.set_backend_for_session("sid-1", "firebase")
        # Session pin wins over env.
        assert core.resolve_backend(tmp_path, session_id="sid-1") == "firebase"
        # Without session_id, env applies.
        assert core.resolve_backend(tmp_path) == "s3"


def test_refresh_and_promote_stages_then_promotes(state):
    core.set_backend_for_session("35163", "firebase", table="ports")
    core.refresh_and_promote("35163", "sid-1")
    # Backend promoted onto the session.
    assert core.get_backend_for_session("sid-1") == "firebase"
    # Port entry consumed.
    assert core.get_backend_for_session("35163", table="ports") is None
    # Link is persistent — NOT deleted on promote.
    assert core.get_session_for_port("35163") == "sid-1"


def test_refresh_and_promote_link_survives(state):
    # A refresh with no staged backend just (re)records the link, idempotently.
    core.refresh_and_promote("35163", "sid-1")
    assert core.get_session_for_port("35163") == "sid-1"
    assert core.get_backend_for_session("sid-1") is None
    core.refresh_and_promote("35163", "sid-1")  # idempotent
    assert core.get_session_for_port("35163") == "sid-1"


def test_refresh_and_promote_restage_picks_new_backend(state):
    core.set_backend_for_session("35163", "firebase", table="ports")
    core.refresh_and_promote("35163", "sid-1")
    assert core.get_backend_for_session("sid-1") == "firebase"
    # Re-stage a different backend on the same port; next refresh promotes it.
    core.set_backend_for_session("35163", "supabase", table="ports")
    core.refresh_and_promote("35163", "sid-1")
    assert core.get_backend_for_session("sid-1") == "supabase"
    assert core.get_backend_for_session("35163", table="ports") is None


def test_refresh_and_promote_noop_when_missing(state):
    # Missing args → no-op, no raise, no link written.
    core.refresh_and_promote(None, "sid-1")
    core.refresh_and_promote("35163", None)
    assert core.get_session_for_port("35163") is None
    assert core.get_backend_for_session("sid-1") is None


def test_session_link_round_trip(state):
    assert core.get_session_for_port("35163") is None
    core.set_session_link("35163", "sid-1")
    assert core.get_session_for_port("35163") == "sid-1"
    # Never raises on missing args.
    core.set_session_link(None, "sid-1")
    core.set_session_link("35163", None)


def test_prune_removes_dead_sessions(state, tmp_path, monkeypatch):
    # Fake a Claude projects dir with one live transcript.
    home = tmp_path / "home"
    proj = home / ".claude" / "projects" / "-some-dir"
    proj.mkdir(parents=True)
    (proj / "live-sid.jsonl").write_text("{}")
    monkeypatch.setattr(core.Path, "home", staticmethod(lambda: home))

    core.set_backend_for_session("live-sid", "firebase")
    core.set_backend_for_session("dead-sid", "s3")
    core.set_session_link("port-live", "live-sid")
    core.set_session_link("port-dead", "dead-sid")
    core.prune_session_backends()
    assert core.get_backend_for_session("live-sid") == "firebase"
    assert core.get_backend_for_session("dead-sid") is None
    # Links to dead sessions are pruned; links to live sessions survive.
    assert core.get_session_for_port("port-live") == "live-sid"
    assert core.get_session_for_port("port-dead") is None


def test_prune_caps_links(state, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    monkeypatch.setattr(core.Path, "home", staticmethod(lambda: home))
    for i in range(70):
        core.set_session_link(f"port-{i}", f"sid-{i}")
    core.prune_session_backends(max_ports=64)
    remaining = core._load_session_backend_map()["links"]
    assert len(remaining) == 64
    assert "port-69" in remaining
    assert "port-0" not in remaining


def test_prune_caps_ports(state, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    monkeypatch.setattr(core.Path, "home", staticmethod(lambda: home))
    for i in range(70):
        core.set_backend_for_session(f"port-{i}", "sqlite", table="ports")
    core.prune_session_backends(max_ports=64)
    remaining = core._load_session_backend_map()["ports"]
    assert len(remaining) == 64
    # Most recent are kept.
    assert "port-69" in remaining
    assert "port-0" not in remaining


def test_prune_never_raises_without_projects(state, tmp_path, monkeypatch):
    home = tmp_path / "no-home"
    monkeypatch.setattr(core.Path, "home", staticmethod(lambda: home))
    core.set_backend_for_session("sid-1", "firebase")
    core.prune_session_backends()  # must not raise, must not wipe
    assert core.get_backend_for_session("sid-1") == "firebase"


# ── UserPromptSubmit link-refresh dispatch ──────────────────────────────────────

def test_userpromptsubmit_discards_unbound_claude_choices(state, tmp_path, monkeypatch, capsys):
    """The per-turn link event refreshes the link, discards unbound choices,
    and creates NO queue item (pure disk, no save)."""
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "35163")
    # If enqueue were called, fail loudly.
    import memory.queue as _q
    monkeypatch.setattr(_q, "enqueue",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no save on link event")))

    core.set_backend_for_session("35163", "firebase", table="ports")

    event = {"hook_event_name": "UserPromptSubmit", "session_id": "sid-1",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    assert core.get_backend_for_session("sid-1") is None
    assert core.get_session_for_port("35163") == "sid-1"
    output = json.loads(capsys.readouterr().out)
    assert "--session-id sid-1" in output["hookSpecificOutput"]["additionalContext"]


def test_userpromptsubmit_never_raises_on_bad_input(state, monkeypatch):
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr("sys.stdin", _FakeStdin("not json"))
    hook._run_hook()  # must not raise


# ── configure_hooks registers/removes UserPromptSubmit ──────────────────────────

def test_configure_hooks_registers_link_event(tmp_path):
    path = tmp_path / "settings.json"
    hook.configure_hooks(path)
    config = json.loads(path.read_text())
    events = config["hooks"]
    for ev in ("SessionStart", "Stop", "PostCompact", "SessionEnd", "UserPromptSubmit"):
        assert ev in events, f"{ev} not registered"
    # Removal cleanly strips every event including UserPromptSubmit.
    hook.configure_hooks(path, remove=True)
    config = json.loads(path.read_text())
    assert config.get("hooks", {}) == {}


class _FakeStdin:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


# ── Per-window (per-session) recording state ────────────────────────────────────

def test_recording_session_round_trip(state):
    assert core.recording_agents_for_session("sid-1") == []
    core.set_recording_for_session("sid-1", "claude")
    assert core.recording_agents_for_session("sid-1") == ["claude"]
    core.set_recording_for_session("sid-1", "codex")
    assert set(core.recording_agents_for_session("sid-1")) == {"claude", "codex"}
    core.clear_recording_for_session("sid-1", "claude")
    assert core.recording_agents_for_session("sid-1") == ["codex"]
    core.clear_recording_for_session("sid-1")  # clear all
    assert core.recording_agents_for_session("sid-1") == []


def test_recording_session_ports_table(state):
    core.set_recording_for_session("35163", "claude", table="ports")
    assert core.recording_agents_for_session("35163", table="ports") == ["claude"]
    assert core.recording_agents_for_session("35163", table="sessions") == []


def test_is_recording_session_no_identity_falls_back_to_dir(state, tmp_path):
    # Non-Claude agents with no session_id/port fall back to legacy per-dir enabled.json.
    assert core.is_recording_session(tmp_path, "codex") is False
    core.enable_recording(tmp_path, "codex")
    assert core.is_recording_session(tmp_path, "codex") is True


def test_is_recording_session_claude_never_falls_back_to_dir(state, tmp_path):
    # Claude is strictly session-scoped: a stale enabled.json entry must never
    # cause recording to appear enabled without an active session/port.
    core.enable_recording(tmp_path, "claude")  # simulate a stale legacy entry
    assert core.is_recording_session(tmp_path, "claude") is False
    # Even with the stale dir entry, a session_id is still required.
    assert core.is_recording_session(tmp_path, "claude", session_id="sid-1") is False
    # Only an explicit per-window entry enables recording.
    core.set_recording_for_session("sid-1", "claude")
    assert core.is_recording_session(tmp_path, "claude", session_id="sid-1") is True


def test_is_recording_session_window_authoritative(state, tmp_path):
    # Dir enabled, but a known session with no window entry must NOT inherit it.
    core.enable_recording(tmp_path, "claude")
    assert core.is_recording_session(tmp_path, "claude", session_id="sid-1") is False
    core.set_recording_for_session("sid-1", "claude")
    assert core.is_recording_session(tmp_path, "claude", session_id="sid-1") is True


def test_is_recording_session_resolves_port_to_session(state, tmp_path):
    core.set_session_link("35163", "sid-1")
    core.set_recording_for_session("sid-1", "claude")
    assert core.is_recording_session(tmp_path, "claude", port="35163") is True
    assert core.is_recording_session(tmp_path, "codex", port="35163") is False


def test_is_recording_session_global_kill_wins(state, tmp_path):
    core.set_recording_for_session("sid-1", "claude")
    core.set_stopped(True)
    assert core.is_recording_session(tmp_path, "claude", session_id="sid-1") is False


def test_two_windows_same_dir_isolated(state, tmp_path):
    # THE regression: two windows (sessions) on one dir record independently.
    core.set_recording_for_session("sid-A", "claude")
    assert core.is_recording_session(tmp_path, "claude", session_id="sid-A") is True
    assert core.is_recording_session(tmp_path, "claude", session_id="sid-B") is False


def test_refresh_and_promote_promotes_recording(state):
    core.set_recording_for_session("35163", "claude", table="ports")
    core.refresh_and_promote("35163", "sid-1")
    # Moved onto the session; port entry consumed; link persists.
    assert core.recording_agents_for_session("sid-1") == ["claude"]
    assert core.recording_agents_for_session("35163", table="ports") == []
    assert core.get_session_for_port("35163") == "sid-1"


def test_refresh_and_promote_backend_and_recording_together(state):
    core.set_backend_for_session("35163", "firebase", table="ports")
    core.set_recording_for_session("35163", "claude", table="ports")
    core.refresh_and_promote("35163", "sid-1")
    assert core.get_backend_for_session("sid-1") == "firebase"
    assert core.recording_agents_for_session("sid-1") == ["claude"]


def test_prune_drops_dead_recording_sessions(state, tmp_path, monkeypatch):
    home = tmp_path / "home"
    proj = home / ".claude" / "projects" / "-some-dir"
    proj.mkdir(parents=True)
    (proj / "live-sid.jsonl").write_text("{}")
    monkeypatch.setattr(core.Path, "home", staticmethod(lambda: home))

    core.set_recording_for_session("live-sid", "claude")
    core.set_recording_for_session("dead-sid", "claude")
    core.prune_session_backends()
    assert core.recording_agents_for_session("live-sid") == ["claude"]
    assert core.recording_agents_for_session("dead-sid") == []


# ── SessionStart / Stop hook: per-window, no cross-window clobber ────────────────

def _no_network_sessionstart(monkeypatch):
    """Neutralize the slow/IO parts of the SessionStart path for hook tests."""
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    import memory.queue as _q
    monkeypatch.setattr(_q, "drain", lambda *a, **k: 0)
    monkeypatch.setattr(hook, "load_context", lambda *a, **k: "")
    # prune reads the real ~/.claude/projects; stub it so it can't drop the
    # synthetic session IDs used in these tests.
    monkeypatch.setattr(core, "prune_session_backends", lambda *a, **k: None)


def test_sessionstart_resets_only_this_session(state, tmp_path, monkeypatch):
    _no_network_sessionstart(monkeypatch)
    core.set_recording_for_session("sid-other", "claude")  # a sibling window

    event = {"hook_event_name": "SessionStart", "session_id": "sid-A",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    # This window reset to off; the sibling window is untouched.
    assert core.recording_agents_for_session("sid-A") == []
    assert core.recording_agents_for_session("sid-other") == ["claude"]


def test_sessionstart_does_not_touch_dir_enabled(state, tmp_path, monkeypatch):
    _no_network_sessionstart(monkeypatch)
    core.enable_recording(tmp_path, "claude")  # legacy per-dir setup

    event = {"hook_event_name": "SessionStart", "session_id": "sid-A",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    # The per-dir store must be left intact (no clobber).
    assert core.recording_agents(tmp_path) == ["claude"]


def test_sessionstart_discards_staged_start(state, tmp_path, monkeypatch, capsys):
    # A port can be reused by a new session. Its staged recording is insufficient
    # to establish the new session's choice.
    _no_network_sessionstart(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "35163")
    core.set_recording_for_session("35163", "claude", table="ports")

    event = {"hook_event_name": "SessionStart", "session_id": "sid-A",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    assert core.recording_agents_for_session("sid-A") == []
    assert core.recording_agents_for_session("35163", table="ports") == []
    assert core.get_recording_decision("sid-A") is None
    assert "AskUserQuestion" in capsys.readouterr().out


def test_stop_event_keeps_sessions_recording(state, tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr(hook, "_read_transcript",
                        lambda event, agent, cwd: "User: hi\n\nAssistant: yo")
    import memory.queue as _q
    monkeypatch.setattr(_q, "enqueue", lambda *a, **k: None)

    core.set_recording_for_session("sid-A", "claude")
    core.set_recording_for_session("sid-other", "claude")

    event = {"hook_event_name": "Stop", "session_id": "sid-A", "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    # Completing a response keeps this session and its sibling recording.
    assert core.recording_agents_for_session("sid-A") == ["claude"]
    assert core.recording_agents_for_session("sid-other") == ["claude"]


# ── Claude session-scope regression tests ──────────────────────────────────────

def test_two_simultaneous_claude_windows_fully_isolated(state, tmp_path):
    """Two Claude windows on the same directory record completely independently.
    Starting/stopping recording in one window must not affect the other."""
    # Window A starts recording.
    core.set_recording_for_session("win-A", "claude")
    # Window B has never started recording.
    assert core.is_recording_session(tmp_path, "claude", session_id="win-A") is True
    assert core.is_recording_session(tmp_path, "claude", session_id="win-B") is False
    # Stopping window A must not affect window B.
    core.clear_recording_for_session("win-A", "claude")
    assert core.is_recording_session(tmp_path, "claude", session_id="win-A") is False
    assert core.is_recording_session(tmp_path, "claude", session_id="win-B") is False


def test_stale_enabled_json_claude_entry_ignored(state, tmp_path):
    """A 'claude' entry left in enabled.json by an older version must be silently
    ignored — it must never cause any new session to record unexpectedly."""
    core.enable_recording(tmp_path, "claude")  # simulate old installation
    # No session_id or port → must be off, not inherited from enabled.json.
    assert core.is_recording_session(tmp_path, "claude") is False
    # With an unknown session_id → still off (no per-window entry).
    assert core.is_recording_session(tmp_path, "claude", session_id="new-sid") is False
    # With an unknown port whose linked session has no entry → still off.
    core.set_session_link("port-new", "new-sid")
    assert core.is_recording_session(tmp_path, "claude", port="port-new") is False


def test_sessionstart_new_session_starts_recording_off(state, tmp_path, monkeypatch):
    """A brand-new Claude session must start with recording OFF even if the
    previous session in the same directory had recording enabled."""
    _no_network_sessionstart(monkeypatch)
    # Simulate: previous session was recording (enabled.json legacy entry exists).
    core.enable_recording(tmp_path, "claude")
    # Brand-new session starts.
    event = {"hook_event_name": "SessionStart", "session_id": "brand-new-sid",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    # New session must be off — legacy dir entry must not bleed through.
    assert core.is_recording_session(tmp_path, "claude", session_id="brand-new-sid") is False


def test_sessionend_saves_and_clears_exactly_this_session(state, tmp_path, monkeypatch):
    """SessionEnd must enqueue the transcript for THIS window and then clear ONLY
    that window's recording state.  A sibling window's state must be untouched."""
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr(hook, "_read_transcript",
                        lambda event, agent, cwd: "User: q\n\nAssistant: a")
    enqueued = {}
    import memory.queue as _q
    monkeypatch.setattr(_q, "enqueue",
                        lambda src, *a, **k: enqueued.__setitem__("src", src) or None)

    core.set_recording_for_session("ending-sid", "claude")
    core.set_recording_for_session("sibling-sid", "claude")

    event = {"hook_event_name": "SessionEnd", "session_id": "ending-sid",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    # Transcript was enqueued for the ending session.
    assert enqueued.get("src") == "ending-sid"
    # Ending session's recording state is cleared.
    assert core.recording_agents_for_session("ending-sid") == []
    # Sibling window is completely unaffected.
    assert core.recording_agents_for_session("sibling-sid") == ["claude"]


def test_session_mapping_cannot_cause_new_session_to_append_old_entry(state, tmp_path):
    """After a session ends and its recording state is cleared, a new session with
    a different ID must start fresh and not share state with the old session."""
    core.set_recording_for_session("old-sid", "claude")
    # Old session ends → state cleared.
    core.clear_recording_for_session("old-sid", "claude")
    # New session starts — must have its own independent state.
    assert core.recording_agents_for_session("new-sid") == []
    assert core.is_recording_session(tmp_path, "claude", session_id="new-sid") is False
    # Enabling new session must not affect old session's (empty) entry.
    core.set_recording_for_session("new-sid", "claude")
    assert core.recording_agents_for_session("old-sid") == []
    assert core.recording_agents_for_session("new-sid") == ["claude"]


def test_status_recording_label_suppresses_stale_claude_dir_entry(
        state, tmp_path, monkeypatch):
    """_status_recording_label must not show 'claude (this dir, legacy)' even when
    enabled.json has a claude entry — Claude is window-scoped, so a plain terminal
    (no CLAUDE_CODE_SSE_PORT) must show no recording for claude."""
    from memory.cli import _status_recording_label
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    core.enable_recording(tmp_path, "claude")
    label = _status_recording_label(tmp_path)
    # Must be None (no recording) — not "claude (this dir, legacy)".
    assert label is None


def test_status_recording_label_shows_legacy_for_non_claude(
        state, tmp_path, monkeypatch):
    """Non-Claude agents still use enabled.json and must appear in the status label."""
    from memory.cli import _status_recording_label
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    core.enable_recording(tmp_path, "codex")
    label = _status_recording_label(tmp_path)
    assert label is not None
    assert "codex" in label


def test_missing_port_refuses_claude_start(state, tmp_path, monkeypatch):
    """Running 'memory start' for claude without CLAUDE_CODE_SSE_PORT must refuse
    with an error message and must NOT write a claude entry into enabled.json."""
    from click.testing import CliRunner
    from memory.cli import cli
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["start", "--agent", "claude"])
    # Must fail with a clear message.
    assert result.exit_code == 0  # click exits 0 by default after printing
    assert "no live Claude window" in result.output.lower() or \
           "CLAUDE_CODE_SSE_PORT" in result.output
    # Must NOT have written claude to enabled.json.
    assert core.is_recording(tmp_path, "claude") is False


# ── Rendezvous marker: core.py functions ───────────────────────────────────────

@pytest.fixture
def marker_state(tmp_path, monkeypatch):
    """Redirect all state files AND the markers dir into tmp_path."""
    monkeypatch.setattr(core, "_ENABLED_FILE", tmp_path / "enabled.json")
    monkeypatch.setattr(core, "_STOPPED_FILE", tmp_path / "STOPPED")
    monkeypatch.setattr(core, "_SESSION_BACKEND_MAP_FILE", tmp_path / "session_backends.json")
    monkeypatch.setattr(core, "_SESSION_MARKERS_DIR", tmp_path / "current_sessions")
    return tmp_path


def test_write_and_read_marker_round_trip(marker_state, tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("sid-abc", "35163", cwd)
    result = core.read_session_marker_for_cwd(cwd)
    assert result is not None
    assert result["session_id"] == "sid-abc"
    assert result["port"] == "35163"
    assert result["cwd"] == str(cwd.resolve())


def test_read_marker_missing_dir_returns_none(marker_state, tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    assert core.read_session_marker_for_cwd(cwd) is None


def test_remove_marker(marker_state, tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("sid-abc", "35163", cwd)
    assert core.read_session_marker_for_cwd(cwd) is not None
    core.remove_session_marker("sid-abc")
    assert core.read_session_marker_for_cwd(cwd) is None


def test_remove_marker_noop_on_unknown(marker_state, tmp_path):
    core.remove_session_marker("ghost-sid")  # must not raise


def test_write_marker_noop_on_unknown_id(marker_state, tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("unknown", "35163", cwd)
    assert core.read_session_marker_for_cwd(cwd) is None


def test_write_marker_idempotent(marker_state, tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("sid-abc", "35163", cwd)
    core.write_session_marker("sid-abc", "35163", cwd)
    result = core.read_session_marker_for_cwd(cwd)
    assert result["session_id"] == "sid-abc"
    # Only one file should exist.
    files = list((tmp_path / "current_sessions").glob("*.json"))
    assert len(files) == 1


def test_two_windows_same_project_isolated_markers(marker_state, tmp_path):
    """Two windows on the same cwd get separate marker files and are never confused.
    With no session-id hint both are present but read_session_marker_for_cwd
    returns None — ambiguous, callers must refuse rather than guess."""
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("win-A", "11111", cwd)
    core.write_session_marker("win-B", "22222", cwd)
    # Both files exist independently.
    files = {f.stem for f in (tmp_path / "current_sessions").glob("*.json")}
    assert "win-A" in files
    assert "win-B" in files
    # Ambiguous — no hint → None (refuse to guess).
    assert core.read_session_marker_for_cwd(cwd) is None
    # With a hint the correct window is returned.
    result = core.read_session_marker_for_cwd(cwd, claude_session_id="win-B")
    assert result is not None
    assert result["session_id"] == "win-B"


def test_two_windows_different_projects_isolated_markers(marker_state, tmp_path):
    """Markers for different cwds are only returned for their own cwd."""
    cwd_a = tmp_path / "proj-a"
    cwd_b = tmp_path / "proj-b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    core.write_session_marker("sid-A", "11111", cwd_a)
    core.write_session_marker("sid-B", "22222", cwd_b)
    result_a = core.read_session_marker_for_cwd(cwd_a)
    result_b = core.read_session_marker_for_cwd(cwd_b)
    assert result_a["session_id"] == "sid-A"
    assert result_b["session_id"] == "sid-B"


def test_stale_marker_is_ignored(marker_state, tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    import time
    core.write_session_marker("sid-stale", "35163", cwd)
    # Manually backdate the marker's updated_at to beyond the stale threshold.
    marker_path = tmp_path / "current_sessions" / "sid-stale.json"
    data = json.loads(marker_path.read_text())
    data["updated_at"] = time.time() - core._MARKER_STALE_SECONDS - 1
    marker_path.write_text(json.dumps(data))
    assert core.read_session_marker_for_cwd(cwd) is None


def test_cleanup_stale_markers(marker_state, tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    import time
    core.write_session_marker("sid-stale", "35163", cwd)
    core.write_session_marker("sid-fresh", "99999", cwd)
    stale_path = tmp_path / "current_sessions" / "sid-stale.json"
    data = json.loads(stale_path.read_text())
    data["updated_at"] = time.time() - core._MARKER_STALE_SECONDS - 1
    stale_path.write_text(json.dumps(data))
    core.cleanup_stale_markers()
    files = {f.stem for f in (tmp_path / "current_sessions").glob("*.json")}
    assert "sid-stale" not in files
    assert "sid-fresh" in files


def test_read_marker_cwd_mismatch_ignored(marker_state, tmp_path):
    cwd_a = tmp_path / "proj-a"
    cwd_b = tmp_path / "proj-b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    core.write_session_marker("sid-A", "11111", cwd_a)
    # Reading from cwd_b must not return sid-A.
    assert core.read_session_marker_for_cwd(cwd_b) is None


# ── Rendezvous marker: hook writes marker on UserPromptSubmit ──────────────────

def test_userpromptsubmit_writes_marker(marker_state, tmp_path, monkeypatch):
    """UserPromptSubmit must write a rendezvous marker for the live session."""
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "35163")

    event = {"hook_event_name": "UserPromptSubmit", "session_id": "sid-live",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    result = core.read_session_marker_for_cwd(tmp_path)
    assert result is not None
    assert result["session_id"] == "sid-live"
    assert result["port"] == "35163"
    assert result["cwd"] == str(tmp_path.resolve())


def test_userpromptsubmit_does_not_write_marker_for_unknown_session(
        marker_state, tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "35163")
    event = {"hook_event_name": "UserPromptSubmit", "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    assert core.read_session_marker_for_cwd(tmp_path) is None


def test_sessionend_removes_marker(marker_state, tmp_path, monkeypatch):
    """SessionEnd must remove the rendezvous marker for this session."""
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr(hook, "_read_transcript", lambda *a, **k: "")
    core.write_session_marker("ending-sid", "35163", tmp_path)

    event = {"hook_event_name": "SessionEnd", "session_id": "ending-sid",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    assert core.read_session_marker_for_cwd(tmp_path) is None


def test_stop_event_keeps_marker(marker_state, tmp_path, monkeypatch):
    """Finishing a response keeps the current session discoverable."""
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr(hook, "_read_transcript", lambda *a, **k: "")
    core.write_session_marker("stop-sid", "35163", tmp_path)

    event = {"hook_event_name": "Stop", "session_id": "stop-sid",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    assert core.read_session_marker_for_cwd(tmp_path)["session_id"] == "stop-sid"


def test_postcompact_does_not_remove_marker(marker_state, tmp_path, monkeypatch):
    """PostCompact must NOT remove the marker — the session is still live."""
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr(hook, "_read_transcript", lambda *a, **k: "")
    core.write_session_marker("live-sid", "35163", tmp_path)

    event = {"hook_event_name": "PostCompact", "session_id": "live-sid",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    result = core.read_session_marker_for_cwd(tmp_path)
    assert result is not None
    assert result["session_id"] == "live-sid"


def test_sessionend_only_removes_own_marker(marker_state, tmp_path, monkeypatch):
    """SessionEnd for one window must not remove another window's marker."""
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr(hook, "_read_transcript", lambda *a, **k: "")
    cwd_b = tmp_path / "proj-b"
    cwd_b.mkdir()
    core.write_session_marker("sibling-sid", "99999", cwd_b)

    event = {"hook_event_name": "SessionEnd", "session_id": "ending-sid",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    result = core.read_session_marker_for_cwd(cwd_b)
    assert result is not None
    assert result["session_id"] == "sibling-sid"


# ── Rendezvous marker: `memory start` via ! shell ─────────────────────────────

def test_start_via_rendezvous_marker_enables_recording(marker_state, tmp_path, monkeypatch):
    """When CLAUDE_CODE_SSE_PORT is absent but a valid marker exists for cwd,
    `memory start` must attach recording to the marker's session_id."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    core.write_session_marker("live-sid", "35163", tmp_path)

    runner = CliRunner()
    with runner.isolated_filesystem():
        # Run from tmp_path so cwd matches the marker.
        import os as _os
        orig = _os.getcwd()
        _os.chdir(tmp_path)
        try:
            result = runner.invoke(cli, ["start", "--agent", "claude"])
        finally:
            _os.chdir(orig)

    assert result.exit_code == 0
    assert "can't start" not in result.output.lower()
    assert core.recording_agents_for_session("live-sid") == ["claude"]
    # Must NOT have written to enabled.json.
    assert core.is_recording(tmp_path, "claude") is False


def test_start_via_marker_attaches_backend_to_session(marker_state, tmp_path, monkeypatch):
    """Backend selection via marker must land on the session, not the directory."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    core.write_session_marker("live-sid", "35163", tmp_path)

    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["start", "--agent", "claude", "--backend", "firebase"])
    finally:
        _os.chdir(orig)

    assert result.exit_code == 0
    assert core.get_backend_for_session("live-sid") == "firebase"


def test_start_two_windows_same_project_isolated(marker_state, tmp_path, monkeypatch):
    """Two windows on the same cwd — starting via CLAUDE_SESSION_ID must only
    attach recording to the identified window, not the other."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setenv("CLAUDE_SESSION_ID", "win-B")

    core.write_session_marker("win-A", "11111", tmp_path)
    core.write_session_marker("win-B", "22222", tmp_path)

    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["start", "--agent", "claude"])
    finally:
        _os.chdir(orig)

    assert result.exit_code == 0
    assert core.recording_agents_for_session("win-B") == ["claude"]
    # Win-A was not touched.
    assert core.recording_agents_for_session("win-A") == []


def test_start_two_windows_different_projects_isolated(marker_state, tmp_path, monkeypatch):
    """Markers for different cwds must never be resolved for the wrong project."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")

    cwd_a = tmp_path / "proj-a"
    cwd_b = tmp_path / "proj-b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    core.write_session_marker("sid-A", "11111", cwd_a)
    core.write_session_marker("sid-B", "22222", cwd_b)

    # Start from proj-b → must only touch sid-B.
    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()
    _os.chdir(cwd_b)
    try:
        result = runner.invoke(cli, ["start", "--agent", "claude"])
    finally:
        _os.chdir(orig)

    assert result.exit_code == 0
    assert core.recording_agents_for_session("sid-B") == ["claude"]
    assert core.recording_agents_for_session("sid-A") == []


def test_start_no_port_no_marker_refuses_claude(marker_state, tmp_path, monkeypatch):
    """With no SSE port AND no valid marker, start must refuse for claude."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    # No marker written.

    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["start", "--agent", "claude"])
    finally:
        _os.chdir(orig)

    assert "can't start" in result.output.lower() or "CLAUDE_CODE_SSE_PORT" in result.output
    assert core.is_recording(tmp_path, "claude") is False


def test_start_stale_marker_refuses_claude(marker_state, tmp_path, monkeypatch):
    """A stale marker must be treated as absent — must not attach recording."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")

    import time
    core.write_session_marker("stale-sid", "35163", tmp_path)
    stale_path = tmp_path / "current_sessions" / "stale-sid.json"
    data = json.loads(stale_path.read_text())
    data["updated_at"] = time.time() - core._MARKER_STALE_SECONDS - 1
    stale_path.write_text(json.dumps(data))

    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["start", "--agent", "claude"])
    finally:
        _os.chdir(orig)

    assert "can't start" in result.output.lower() or "CLAUDE_CODE_SSE_PORT" in result.output
    assert core.recording_agents_for_session("stale-sid") == []


def test_start_no_dir_fallback_via_marker(marker_state, tmp_path, monkeypatch):
    """Recording via rendezvous must NEVER write to enabled.json."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    core.write_session_marker("live-sid", "35163", tmp_path)

    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        runner.invoke(cli, ["start", "--agent", "claude"])
    finally:
        _os.chdir(orig)

    # enabled.json must not contain claude for this cwd.
    assert "claude" not in core.recording_agents(tmp_path)


def test_different_backends_for_different_windows_via_marker(marker_state, tmp_path, monkeypatch):
    """Each window can be assigned its own backend via the marker mechanism."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")

    cwd_a = tmp_path / "proj-a"
    cwd_b = tmp_path / "proj-b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    core.write_session_marker("sid-A", "11111", cwd_a)
    core.write_session_marker("sid-B", "22222", cwd_b)

    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()

    _os.chdir(cwd_a)
    try:
        runner.invoke(cli, ["start", "--agent", "claude", "--backend", "firebase"])
    finally:
        _os.chdir(orig)

    _os.chdir(cwd_b)
    try:
        runner.invoke(cli, ["start", "--agent", "claude", "--backend", "sqlite"])
    finally:
        _os.chdir(orig)

    assert core.get_backend_for_session("sid-A") == "firebase"
    assert core.get_backend_for_session("sid-B") == "sqlite"


# ── Marker resolution with CLAUDE_SESSION_ID hint ─────────────────────────────

def test_marker_resolved_by_claude_session_id(marker_state, tmp_path):
    """When CLAUDE_SESSION_ID matches one of two live markers in the same cwd,
    the correct window's marker is returned unambiguously."""
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("win-A", "11111", cwd)
    core.write_session_marker("win-B", "22222", cwd)
    result = core.read_session_marker_for_cwd(cwd, claude_session_id="win-A")
    assert result is not None
    assert result["session_id"] == "win-A"


def test_marker_ambiguous_multiple_no_hint_returns_none(marker_state, tmp_path):
    """Two live markers for the same cwd and no env hint → None (refuse to guess)."""
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("win-A", "11111", cwd)
    core.write_session_marker("win-B", "22222", cwd)
    assert core.read_session_marker_for_cwd(cwd) is None


def test_marker_unambiguous_single_window_no_hint(marker_state, tmp_path):
    """A single live marker with no env hint is still returned (unambiguous)."""
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("solo-win", "55555", cwd)
    result = core.read_session_marker_for_cwd(cwd)
    assert result is not None
    assert result["session_id"] == "solo-win"


def test_marker_session_id_mismatch_returns_none(marker_state, tmp_path):
    """CLAUDE_SESSION_ID that doesn't match any marker for this cwd → None."""
    cwd = tmp_path / "proj"
    cwd.mkdir()
    core.write_session_marker("win-A", "11111", cwd)
    result = core.read_session_marker_for_cwd(cwd, claude_session_id="ghost-sid")
    assert result is None


# ── Per-window backend used by list/load/search/delete ─────────────────────────

def test_list_sessions_uses_per_window_backend(state, tmp_path, monkeypatch):
    """list_sessions with a claude_session_id that has a pinned backend must
    call get_backend with that backend name, not the global default."""
    seen = {}

    class FakeBackend:
        def init(self): pass
        def list(self, project=None): return []

    def fake_get_backend(name=None):
        seen["name"] = name
        return FakeBackend()

    monkeypatch.setattr(core, "get_backend", fake_get_backend)
    core.set_backend_for_session("win-X", "firebase")
    core.list_sessions(claude_session_id="win-X")
    assert seen["name"] == "firebase"


def test_list_sessions_no_session_id_uses_default(state, tmp_path, monkeypatch):
    """list_sessions without a claude_session_id uses the default backend."""
    env = {k: v for k, v in os.environ.items() if k != "MEMORY_BACKEND"}
    seen = {}

    class FakeBackend:
        def init(self): pass
        def list(self, project=None): return []

    def fake_get_backend(name=None):
        seen["name"] = name
        return FakeBackend()

    monkeypatch.setattr(core, "get_backend", fake_get_backend)
    with patch.dict(os.environ, env, clear=True):
        core.list_sessions()
    assert seen["name"] is None  # resolve_backend returns None → get_backend sqlite default


def test_load_session_uses_per_window_backend(state, tmp_path, monkeypatch):
    """load_session with a claude_session_id routes to the pinned backend."""
    seen = {}

    class FakeBackend:
        def init(self): pass
        def get(self, sid): return None

    def fake_get_backend(name=None):
        seen["name"] = name
        return FakeBackend()

    monkeypatch.setattr(core, "get_backend", fake_get_backend)
    core.set_backend_for_session("win-X", "supabase")
    core.load_session("mem-sid-123", claude_session_id="win-X")
    assert seen["name"] == "supabase"


def test_search_sessions_uses_per_window_backend(state, tmp_path, monkeypatch):
    """search_sessions with a claude_session_id routes to the pinned backend."""
    seen = {}

    class FakeBackend:
        def init(self): pass
        def search(self, query, project=None): return []

    def fake_get_backend(name=None):
        seen["name"] = name
        return FakeBackend()

    monkeypatch.setattr(core, "get_backend", fake_get_backend)
    core.set_backend_for_session("win-Y", "s3")
    core.search_sessions("hello", claude_session_id="win-Y")
    assert seen["name"] == "s3"


def test_delete_session_uses_per_window_backend(state, tmp_path, monkeypatch):
    """delete_session with a claude_session_id routes to the pinned backend."""
    seen = {}

    class FakeBackend:
        def init(self): pass
        def delete(self, sid): pass

    def fake_get_backend(name=None):
        seen["name"] = name
        return FakeBackend()

    monkeypatch.setattr(core, "get_backend", fake_get_backend)
    core.set_backend_for_session("win-Z", "firebase")
    core.delete_session("mem-sid-999", claude_session_id="win-Z")
    assert seen["name"] == "firebase"


# ── load_context uses per-window backend ───────────────────────────────────────

def test_load_context_uses_per_window_backend(state, tmp_path, monkeypatch):
    """load_context must use the backend pinned for the current session, not
    always the global default."""
    seen = {}

    class FakeBackend:
        def init(self): pass
        def list(self, project=None): return []

    def fake_get_backend(name=None):
        seen["name"] = name
        return FakeBackend()

    monkeypatch.setattr(core, "get_backend", fake_get_backend)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)

    # Window has recording enabled and a pinned backend.
    core.set_recording_for_session("ctx-sid", "claude")
    core.set_backend_for_session("ctx-sid", "firebase")

    hook.load_context(tmp_path, session_id="ctx-sid", agent="claude")
    assert seen["name"] == "firebase"


def test_load_context_no_session_id_uses_default(state, tmp_path, monkeypatch):
    """load_context with no session_id (recording off) returns empty string
    and must not call get_backend at all (gated before backend use)."""
    called = {}

    class FakeBackend:
        def init(self): pass
        def list(self, project=None): return []

    def fake_get_backend(name=None):
        called["yes"] = True
        return FakeBackend()

    monkeypatch.setattr(core, "get_backend", fake_get_backend)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)

    # No recording entry for the session → load_context returns "" immediately.
    result = hook.load_context(tmp_path, session_id="unknown-sid", agent="claude")
    assert result == ""
    assert "yes" not in called


# ── memory start uses CLAUDE_SESSION_ID when multiple windows share cwd ────────

def test_start_uses_claude_session_id_to_pick_correct_window(
        marker_state, tmp_path, monkeypatch):
    """When CLAUDE_SESSION_ID is set and multiple markers exist for the cwd,
    `memory start` must attach recording to the identified window, not win-B."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")

    # Two windows in the same project.
    core.write_session_marker("win-A", "11111", tmp_path)
    core.write_session_marker("win-B", "22222", tmp_path)

    # CLAUDE_SESSION_ID identifies win-A.
    monkeypatch.setenv("CLAUDE_SESSION_ID", "win-A")

    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["start", "--agent", "claude"])
    finally:
        _os.chdir(orig)

    assert result.exit_code == 0
    assert "can't start" not in result.output.lower()
    # Only win-A gets recording; win-B is untouched.
    assert core.recording_agents_for_session("win-A") == ["claude"]
    assert core.recording_agents_for_session("win-B") == []


def test_start_refuses_when_ambiguous_and_no_session_id_env(
        marker_state, tmp_path, monkeypatch):
    """When multiple windows share cwd and CLAUDE_SESSION_ID is unset,
    `memory start` must refuse rather than guess."""
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")

    core.write_session_marker("win-A", "11111", tmp_path)
    core.write_session_marker("win-B", "22222", tmp_path)

    runner = CliRunner()
    import os as _os
    orig = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["start", "--agent", "claude"])
    finally:
        _os.chdir(orig)

    # Must refuse: neither window gets recording.
    assert core.recording_agents_for_session("win-A") == []
    assert core.recording_agents_for_session("win-B") == []


def test_sessionstart_asks_for_recording_without_environment_identity(
        state, tmp_path, monkeypatch, capsys):
    _no_network_sessionstart(monkeypatch)
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    event = {"hook_event_name": "SessionStart", "session_id": "new-window",
             "cwd": str(tmp_path), "source": "startup"}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "AskUserQuestion" in context
    assert "memory start --agent claude --session-id new-window" in context
    assert "memory stop --agent claude --session-id new-window" in context
    assert all(backend in context for backend in ("sqlite", "firebase", "supabase", "s3"))
    assert core.recording_agents_for_session("new-window") == []
    assert core.read_session_marker_for_cwd(tmp_path)["session_id"] == "new-window"


@pytest.mark.parametrize("source", ["resume", "compact"])
def test_sessionstart_preserves_recording_on_resume_and_compaction(
        state, tmp_path, monkeypatch, capsys, source):
    _no_network_sessionstart(monkeypatch)
    core.set_recording_for_session("active-window", "claude", backend="firebase")
    event = {"hook_event_name": "SessionStart", "session_id": "active-window",
             "cwd": str(tmp_path), "source": source}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()

    assert core.recording_agents_for_session("active-window") == ["claude"]
    assert core.get_backend_for_session("active-window") == "firebase"
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "active-window" in context
    assert "AskUserQuestion" not in context


def test_explicit_session_commands_isolate_windows_and_keep_hooks(
        state, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "port-B")
    monkeypatch.setenv("CLAUDE_SESSION_ID", "win-B")
    core.set_session_link("port-B", "win-B")
    core.write_session_marker("win-A", None, tmp_path)
    core.write_session_marker("win-B", None, tmp_path)
    core.set_recording_for_session("win-B", "claude", backend="sqlite")
    runner = CliRunner()
    result = runner.invoke(cli, ["start", "--session-id", "win-A", "--backend", "firebase"])
    assert result.exit_code == 0, result.output
    assert "Recording started" in result.output
    assert core.get_backend_for_session("win-A") == "firebase"
    assert core.get_backend_for_session("win-B") == "sqlite"
    settings = tmp_path / ".claude" / "settings.json"
    installed = settings.read_text()

    status = runner.invoke(cli, ["status", "--session-id", "win-A"])
    assert status.exit_code == 0, status.output
    assert "win-A" in status.output
    assert "firebase" in status.output
    assert "this window" in status.output
    stop = runner.invoke(cli, ["stop", "--session-id", "win-A"])
    assert stop.exit_code == 0, stop.output
    assert core.recording_agents_for_session("win-A") == []
    assert core.recording_agents_for_session("win-B") == ["claude"]
    assert settings.read_text() == installed


def test_declining_recording_is_remembered_and_can_be_changed(
        state, tmp_path, monkeypatch, capsys):
    from click.testing import CliRunner
    from memory.cli import cli

    _no_network_sessionstart(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["stop", "--session-id", "declined"])
    assert result.exit_code == 0, result.output
    assert core.get_recording_decision("declined") is False
    event = {"hook_event_name": "UserPromptSubmit", "session_id": "declined",
             "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "AskUserQuestion" not in context
    assert "memory save" in context

    event.update(hook_event_name="SessionStart", source="compact")
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    assert "AskUserQuestion" not in capsys.readouterr().out
    result = runner.invoke(cli, ["start", "--session-id", "declined", "--backend", "sqlite"])
    assert result.exit_code == 0, result.output
    assert core.get_recording_decision("declined") is True


def test_status_resolves_a_marker_and_reports_ambiguity(
        state, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    core.write_session_marker("only-window", None, tmp_path)
    core.set_recording_for_session("only-window", "claude", backend="firebase")
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "only-window" in result.output
    assert "firebase" in result.output
    core.write_session_marker("second-window", None, tmp_path)
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "unknown (use --session-id)" in result.output


def test_response_snapshots_continue_until_session_end(
        state, tmp_path, monkeypatch):
    import memory.queue as queue

    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr(hook, "_read_transcript", lambda *args: "User: hello")
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    snapshots = []
    monkeypatch.setattr(queue, "enqueue",
                        lambda sid, *args, **kwargs: snapshots.append((sid, kwargs["backend"])))
    core.set_recording_for_session("live", "claude", backend="firebase")
    core.set_recording_for_session("sibling", "claude", backend="sqlite")
    for event_name in ("Stop", "Stop", "SessionEnd", "Stop"):
        event = {"hook_event_name": event_name, "session_id": "live", "cwd": str(tmp_path)}
        monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
        hook._run_hook()
    assert snapshots == [("live", "firebase")] * 3
    assert core.recording_agents_for_session("live") == []
    assert core.recording_agents_for_session("sibling") == ["claude"]


def test_sessionend_clears_recording_even_with_empty_transcript(
        state, tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setattr(hook, "_read_transcript", lambda *args: "")
    core.set_recording_for_session("empty", "claude", backend="sqlite")
    core.write_session_marker("empty", None, tmp_path)
    event = {"hook_event_name": "SessionEnd", "session_id": "empty", "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    assert core.recording_agents_for_session("empty") == []
    assert core.read_session_marker_for_cwd(tmp_path) is None


def test_explicit_stop_cannot_be_undone_by_staged_recording(state):
    core.set_session_link("port-A", "win-A")
    core.set_recording_for_session("port-A", "claude", table="ports", backend="firebase")
    core.clear_recording_for_session("win-A", "claude")
    core.refresh_and_promote("port-A", "win-A")
    assert core.get_recording_decision("win-A") is False
    assert core.recording_agents_for_session("win-A") == []


def _write_concurrent_choices(state_path, worker, barrier):
    core._SESSION_BACKEND_MAP_FILE = state_path
    barrier.wait(timeout=20)
    for index in range(10):
        core.set_recording_for_session(f"worker-{worker}-{index}", "claude", backend="sqlite")


def test_concurrent_processes_preserve_all_session_choices(state):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    workers = [context.Process(target=_write_concurrent_choices,
                               args=(core._SESSION_BACKEND_MAP_FILE, index, barrier))
               for index in range(4)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
    saved = core._load_session_backend_map()
    expected = {f"worker-{worker}-{index}" for worker in range(4) for index in range(10)}
    assert set(saved["sessions"]) == expected
    assert set(saved["recording"]["sessions"]) == expected
    assert set(saved["decisions"]) == expected


@pytest.mark.parametrize("source", ["startup", "clear"])
def test_new_session_reasks_and_clears_old_backend_for_its_id(
        state, tmp_path, monkeypatch, capsys, source):
    _no_network_sessionstart(monkeypatch)
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    core.set_recording_for_session("new", "claude", backend="firebase")
    core.set_recording_for_session("sibling", "claude", backend="sqlite")
    event = {"hook_event_name": "SessionStart", "session_id": "new",
             "cwd": str(tmp_path), "source": source}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    assert "AskUserQuestion" in capsys.readouterr().out
    assert core.recording_agents_for_session("new") == []
    assert core.get_backend_for_session("new") is None
    assert core.recording_agents_for_session("sibling") == ["claude"]
    assert core.get_backend_for_session("sibling") == "sqlite"


def test_start_for_other_agents_keeps_directory_recording(
        state, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    core.write_session_marker("claude-window", None, tmp_path)
    result = CliRunner().invoke(cli, ["start", "--agent", "codex"])
    assert result.exit_code == 0, result.output
    assert core.is_recording(tmp_path, "codex") is True
    assert core.recording_agents_for_session("claude-window") == []


def test_new_session_with_leftover_port_state_stays_off_until_user_chooses(
        state, tmp_path, monkeypatch, capsys):
    from click.testing import CliRunner
    from memory.cli import cli
    import memory.queue as queue

    _no_network_sessionstart(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "35163")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
    core.set_recording_for_session("old-window", "claude", backend="firebase")
    core.set_session_link("35163", "old-window")
    core.set_recording_for_session("35163", "claude", table="ports", backend="sqlite")
    event = {"hook_event_name": "SessionStart", "session_id": "new-window",
             "cwd": str(tmp_path), "source": "startup"}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    assert "AskUserQuestion" in capsys.readouterr().out
    status = CliRunner().invoke(cli, ["status"])
    assert status.exit_code == 0, status.output
    assert "off (awaiting your choice)" in status.output
    assert "not selected (awaiting your choice)" in status.output
    assert "sqlite" not in status.output
    assert core.get_backend_for_session("new-window") is None
    assert core.get_recording_decision("new-window") is None
    assert core.recording_agents_for_session("old-window") == ["claude"]
    assert core.get_backend_for_session("old-window") == "firebase"

    snapshots = []
    monkeypatch.setattr(hook, "_read_transcript", lambda *args: "User: hello")
    monkeypatch.setattr(queue, "enqueue", lambda *args, **kwargs: snapshots.append(args))
    event["hook_event_name"] = "Stop"
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    assert snapshots == []


def test_shared_sse_port_cannot_select_between_two_windows(
        state, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "shared-port")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    core.write_session_marker("win-A", "shared-port", tmp_path)
    core.write_session_marker("win-B", "shared-port", tmp_path)
    core.set_session_link("shared-port", "win-B")
    runner = CliRunner()
    start = runner.invoke(cli, ["start", "--backend", "firebase"])
    assert "Can't start Claude recording" in start.output
    assert core.recording_agents_for_session("win-A") == []
    assert core.recording_agents_for_session("win-B") == []
    status = runner.invoke(cli, ["status"])
    assert "unknown (use --session-id)" in status.output


def test_distinct_sse_port_resolves_a_unique_marker(
        state, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "port-A")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    core.write_session_marker("win-A", "port-A", tmp_path)
    core.write_session_marker("win-B", "port-B", tmp_path)
    result = CliRunner().invoke(cli, ["start", "--backend", "firebase"])
    assert result.exit_code == 0, result.output
    assert core.get_backend_for_session("win-A") == "firebase"
    assert core.recording_agents_for_session("win-A") == ["claude"]
    assert core.recording_agents_for_session("win-B") == []


def test_status_identifies_its_claude_process_without_a_port(
        state, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from memory.cli import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    process_a = {"pid": 111, "started_at": "Sun Nov 22 14:00:00 2026"}
    process_b = {"pid": 222, "started_at": "Sun Nov 22 14:01:00 2026"}
    core.write_session_marker("win-A", None, tmp_path, claude_process=process_a)
    core.write_session_marker("win-B", None, tmp_path, claude_process=process_b)
    core.set_recording_for_session("win-B", "claude", backend="firebase")
    monkeypatch.setattr(core, "get_claude_process_identity", lambda: process_a)
    status = CliRunner().invoke(cli, ["status"])
    assert status.exit_code == 0, status.output
    assert "win-A" in status.output
    assert "off (awaiting your choice)" in status.output
    assert "not selected (awaiting your choice)" in status.output
    assert "firebase" not in status.output


def test_process_id_reuse_cannot_attach_to_an_old_session(state, tmp_path):
    old_process = {"pid": 111, "started_at": "Sun Nov 22 14:00:00 2026"}
    new_process = {"pid": 111, "started_at": "Sun Nov 22 15:00:00 2026"}
    core.write_session_marker("old-session", None, tmp_path, claude_process=old_process)
    assert core.read_session_marker_for_cwd(tmp_path, claude_process=new_process) is None


def test_old_session_end_preserves_the_new_session_in_the_same_process(state, tmp_path):
    process = {"pid": 111, "started_at": "Sun Nov 22 14:00:00 2026"}
    core.write_session_marker("old-session", None, tmp_path, claude_process=process)
    core.write_session_marker("new-session", None, tmp_path, claude_process=process)
    core.remove_session_marker("old-session")
    marker = core.read_session_marker_for_cwd(tmp_path, claude_process=process)
    assert marker["session_id"] == "new-session"
    core.remove_session_marker("new-session")
    assert core.read_session_marker_for_cwd(tmp_path, claude_process=process) is None


def test_running_process_can_resolve_its_session_after_marker_timeout(state, tmp_path):
    process = {"pid": 111, "started_at": "Sun Nov 22 14:00:00 2026"}
    core.write_session_marker("long-running", None, tmp_path, claude_process=process)
    pointer = core._SESSION_MARKERS_DIR / "processes" / "111.json"
    payload = json.loads(pointer.read_text())
    payload["updated_at"] = 0
    pointer.write_text(json.dumps(payload))
    marker = core.read_session_marker_for_cwd(tmp_path, claude_process=process)
    assert marker["session_id"] == "long-running"


def test_claude_process_identity_walks_shell_ancestors(monkeypatch):
    import subprocess

    responses = {
        "111": "111 222 Sun Nov 22 14:00:00 2026 /bin/zsh",
        "222": "222 333 Sun Nov 22 13:59:00 2026 claude",
    }
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments[2])
        return subprocess.CompletedProcess(arguments, 0, stdout=responses[arguments[2]])

    monkeypatch.setattr(core.subprocess, "run", run)
    identity = core.get_claude_process_identity(process_id=111)
    assert identity == {"pid": 222, "started_at": "Sun Nov 22 13:59:00 2026"}
    assert calls == ["111", "222"]


def test_sessionstart_binds_process_before_the_first_user_prompt(
        state, tmp_path, monkeypatch, capsys):
    _no_network_sessionstart(monkeypatch)
    process = {"pid": 111, "started_at": "Sun Nov 22 14:00:00 2026"}
    monkeypatch.setattr(core, "get_claude_process_identity", lambda: process)
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    event = {"hook_event_name": "SessionStart", "session_id": "new-window",
             "cwd": str(tmp_path), "source": "startup"}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    assert "AskUserQuestion" in capsys.readouterr().out
    assert core.read_session_marker_for_cwd(tmp_path, claude_process=process)["session_id"] == "new-window"
    assert core.recording_agents_for_session("new-window") == []


def test_stop_reads_an_actual_claude_transcript_and_queues_it(
        state, tmp_path, monkeypatch):
    import memory.queue as queue

    monkeypatch.setattr(queue, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.setenv("MEMORY_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("MEMORY_LLM_MODEL", raising=False)
    path = tmp_path / "claude-session.jsonl"
    messages = [
        {"type": "user", "message": {"content": "Remember the project settings"}},
        {"type": "assistant", "message": {"model": "claude-sonnet-4-6",
         "content": [{"type": "text", "text": "Project settings confirmed"}]}},
    ]
    path.write_text("\n".join(json.dumps(message) for message in messages))
    core.set_recording_for_session("claude-session", "claude", backend="firebase")
    event = {"hook_event_name": "Stop", "session_id": "claude-session",
             "cwd": str(tmp_path), "transcript_path": str(path)}
    monkeypatch.setattr("sys.stdin", _FakeStdin(json.dumps(event)))
    hook._run_hook()
    pending = list(queue.QUEUE_DIR.glob("*.json"))
    assert len(pending) == 1
    payload = json.loads(pending[0].read_text())
    assert payload["transcript"] == (
        "User: Remember the project settings\n\nAssistant: Project settings confirmed")
    assert payload["meta_model"] == "claude-sonnet-4-6"
    assert payload["provider"] == "anthropic"
    assert payload["backend"] == "firebase"


def test_universal_hook_still_reads_codex_rollouts(tmp_path):
    path = tmp_path / "codex.jsonl"
    entries = [
        {"type": "session_meta", "payload": {"id": "codex-session", "model": "gpt-5"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "Hello from Codex"}]}},
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries))
    event = {"transcript_path": str(path)}
    assert hook._read_transcript(event, "codex", tmp_path) == "User: Hello from Codex"
    assert event["_meta_model"] == "gpt-5"


@pytest.fixture
def current_session_save(state, tmp_path, monkeypatch):
    from memory.backends.sqlite import SQLiteBackend
    from memory.compressor import CompressedSummary
    import memory.queue as queue

    home = tmp_path / "home"
    monkeypatch.setattr(core.Path, "home", staticmethod(lambda: home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core, "_ACTIVE_SESSION_FILE", tmp_path / "active_session")
    monkeypatch.setattr(queue, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(hook, "detect_agent", lambda: "claude")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SSE_PORT", raising=False)
    monkeypatch.setenv("MEMORY_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("MEMORY_LLM_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
    process = {"pid": 111, "started_at": "Sun Nov 22 14:00:00 2026"}
    monkeypatch.setattr(core, "get_claude_process_identity", lambda: process)
    core.write_session_marker("win-A", None, tmp_path, claude_process=process)
    core.write_session_marker("win-B", None, tmp_path, claude_process={
        "pid": 222, "started_at": "Sun Nov 22 14:01:00 2026"})
    core.set_recording_for_session("win-A", "claude", backend="firebase")
    core.set_recording_for_session("win-B", "claude", backend="sqlite")
    project = home / ".claude" / "projects" / "test-project"
    project.mkdir(parents=True)
    for sid, text in (("win-A", "Window A conversation"), ("win-B", "Window B conversation")):
        entries = [
            {"type": "user", "message": {"content": text}},
            {"type": "assistant", "message": {"model": "claude-sonnet-4-6",
             "content": [{"type": "text", "text": "Confirmed"}]}},
        ]
        (project / (sid + ".jsonl")).write_text("\n".join(json.dumps(entry) for entry in entries))
    db = SQLiteBackend(tmp_path / "memory.db")
    db.init()
    backend_calls = []

    def get_backend(name=None):
        backend_calls.append(name)
        return db

    monkeypatch.setattr(core, "get_backend", get_backend)
    compressed = []

    def compress(transcript, provider=None, model=None):
        compressed.append((transcript, provider, model))
        return CompressedSummary(
            "Saved conversation", "project-A",
            "## Title\nSaved conversation\n\n## Project\nproject-A\n\n## Tags\ntest", ["test"])

    monkeypatch.setattr("memory.compressor.compress", compress)
    monkeypatch.setattr("memory.editor.open_in_editor", lambda *args: pytest.fail("Unexpected editor"))
    return db, backend_calls, compressed


def test_bare_save_uses_current_session_settings_and_updates_one_entry(
        current_session_save, tmp_path):
    from click.testing import CliRunner
    from memory.cli import cli
    import memory.queue as queue

    db, backends, compressed = current_session_save
    other_snapshot = queue.enqueue("win-B", tmp_path, "claude", "User: Other pending work",
                                   meta_model="claude-sonnet-4-6", backend="sqlite")
    core._ACTIVE_SESSION_FILE.write_text("other-window-memory")
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(cli, ["save"])
        assert result.exit_code == 0, result.output
        assert "Session saved!" in result.output
        assert "firebase" in result.output
    assert len(db.list()) == 1
    assert db.list()[0].id == queue._memory_id("win-A:claude-sonnet-4-6")
    assert backends == ["firebase", "firebase"]
    assert all(call == ("User: Window A conversation\n\nAssistant: Confirmed",
                        "anthropic", "claude-sonnet-4-6") for call in compressed)
    assert other_snapshot.exists()
    assert core._ACTIVE_SESSION_FILE.read_text() == "other-window-memory"


def test_current_session_save_preserves_explicit_overrides(current_session_save):
    from click.testing import CliRunner
    from memory.cli import cli

    db, backends, compressed = current_session_save
    result = CliRunner().invoke(cli, ["save", "--provider", "openai", "--model", "gpt-5",
                                      "--backend", "s3"])
    assert result.exit_code == 0, result.output
    assert compressed[0][1:] == ("openai", "gpt-5")
    assert backends == ["s3"]
    assert db.list()[0].model == "gpt-5"
    assert core.get_backend_for_session("win-A") == "firebase"


def test_current_session_save_failure_keeps_transcript_for_retry(
        current_session_save, monkeypatch):
    from click.testing import CliRunner
    from memory.cli import cli
    import memory.queue as queue

    db, _, _ = current_session_save

    class UnavailableBackend:
        def init(self):
            pass

        def get(self, memory_id):
            return None

        def save(self, session):
            raise RuntimeError("Backend temporarily unavailable")

    monkeypatch.setattr(core, "get_backend", lambda name=None: UnavailableBackend())
    runner = CliRunner()
    result = runner.invoke(cli, ["save"])
    assert result.exit_code == 1
    assert "Transcript retained for retry" in result.output
    pending = list(queue.QUEUE_DIR.glob("*.json"))
    assert len(pending) == 1
    assert json.loads(pending[0].read_text())["source_id"] == "win-A"
    monkeypatch.setattr(core, "get_backend", lambda name=None: db)
    result = runner.invoke(cli, ["save"])
    assert result.exit_code == 0, result.output
    assert len(db.list()) == 1
    assert list(queue.QUEUE_DIR.glob("*.json")) == []


def test_pasted_transcript_is_preserved_inside_claude(current_session_save):
    from click.testing import CliRunner
    from memory.cli import cli

    db, _, compressed = current_session_save
    result = CliRunner().invoke(cli, ["save", "--no-edit"], input="User: Explicit pasted transcript")
    assert result.exit_code == 0, result.output
    assert compressed[0][0] == "User: Explicit pasted transcript"
    assert len(db.list()) == 1

