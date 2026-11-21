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
    monkeypatch.setattr("memory.core.get_backend", lambda: db)
    monkeypatch.setattr(hook, "get_backend", lambda: db, raising=False)
    # save_session imports get_backend from .core inside the function, so patch there.
    import memory.core as _core
    monkeypatch.setattr(_core, "get_backend", lambda: db)

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

