import json

import pytest

from memory import queue as mqueue
from memory import core, hook


@pytest.fixture
def qstate(tmp_path, monkeypatch):
    """Redirect all state (queue dir, enabled map, backend map) into a tmp dir."""
    monkeypatch.setattr(mqueue, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(core, "_ENABLED_FILE", tmp_path / "enabled.json")
    monkeypatch.setattr(core, "_STOPPED_FILE", tmp_path / "STOPPED")
    monkeypatch.setattr(core, "_SESSION_BACKEND_MAP_FILE", tmp_path / "session_backends.json")
    return tmp_path


def test_enqueue_writes_raw_transcript_no_network(qstate, tmp_path, monkeypatch):
    # If enqueue touched the network it would call compress; make that explode.
    monkeypatch.setattr("memory.compressor.compress",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network in enqueue")))
    path = mqueue.enqueue("sess-1", tmp_path, "claude",
                          "User: hi\n\nAssistant: yo", meta_model="model-x",
                          backend="firebase")
    assert path is not None and path.exists()
    payload = json.loads(path.read_text())
    assert payload["transcript"] == "User: hi\n\nAssistant: yo"
    assert payload["meta_model"] == "model-x"
    assert payload["backend"] == "firebase"
    assert payload["source_id"] == "sess-1"


def test_enqueue_dedup_identical(qstate, tmp_path):
    p1 = mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi", meta_model="m")
    p2 = mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi", meta_model="m")
    assert p1 == p2
    assert len(list((tmp_path / "queue").glob("*.json"))) == 1


def test_enqueue_empty_returns_none(qstate, tmp_path):
    assert mqueue.enqueue("sess-1", tmp_path, "claude", "   ") is None


def test_drain_compresses_stores_and_deletes(qstate, tmp_path, monkeypatch):
    from memory.backends.sqlite import SQLiteBackend
    from memory.compressor import CompressedSummary

    db = SQLiteBackend(db_path=tmp_path / "mem.db")
    db.init()
    monkeypatch.setattr("memory.core.get_backend", lambda name=None: db)

    def fake_compress(transcript, provider=None, model=None):
        return CompressedSummary("Queued work", "myproj",
                                 "## Title\nQueued work\n\n## Project\nmyproj\n\n## Tags\nx",
                                 ["x"])
    monkeypatch.setattr("memory.compressor.compress", fake_compress)
    monkeypatch.setattr(hook, "_provider_key_present", lambda provider: True)

    core.set_recording_for_session("sess-1", "claude")
    mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi\n\nAssistant: yo",
                   meta_model="model-a")

    processed = mqueue.drain()
    assert processed == 1
    # Queue file removed on success.
    assert list((tmp_path / "queue").glob("*.json")) == []
    # Memory landed in the backend.
    rows = db.list()
    assert len(rows) == 1
    assert rows[0].model == "model-a"
    assert "claude" in rows[0].tags


def test_drain_leaves_item_when_provider_unavailable(qstate, tmp_path, monkeypatch):
    core.set_recording_for_session("sess-1", "claude")
    # No provider resolvable, no key → _process_one returns False, file kept.
    monkeypatch.setattr(hook, "resolve_provider_model", lambda agent, event: (None, None))
    mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi\n\nAssistant: yo")

    processed = mqueue.drain()
    assert processed == 0
    assert len(list((tmp_path / "queue").glob("*.json"))) == 1


def test_drain_stores_item_after_window_teardown(qstate, tmp_path, monkeypatch):
    # Lifecycle: a window records, closes (final transcript enqueued), and its
    # per-window recording entry is torn down on close. The NEXT SessionStart
    # drains — by then the originating window's recording entry is gone. The
    # queued final save must still land, not be dropped. The queue file itself
    # is proof the recording gate passed at enqueue time.
    from memory.backends.sqlite import SQLiteBackend
    from memory.compressor import CompressedSummary

    db = SQLiteBackend(db_path=tmp_path / "mem.db")
    db.init()
    monkeypatch.setattr("memory.core.get_backend", lambda name=None: db)
    monkeypatch.setattr("memory.compressor.compress",
                        lambda t, provider=None, model=None: CompressedSummary(
                            "t", "p", "## Title\nt\n\n## Project\np\n\n## Tags\nx", ["x"]))
    monkeypatch.setattr(hook, "_provider_key_present", lambda provider: True)

    # NO recording entry for this session (already torn down on close).
    mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi\n\nAssistant: yo",
                   meta_model="model-a")
    processed = mqueue.drain()
    assert processed == 1
    assert list((tmp_path / "queue").glob("*.json")) == []
    rows = db.list()
    assert len(rows) == 1  # the final save landed


def test_drain_keeps_item_when_globally_stopped(qstate, tmp_path, monkeypatch):
    # The global kill-switch is the only thing that vetoes a drain. The item is
    # kept (not dropped) so it can be stored after 'memory start --global'.
    monkeypatch.setattr("memory.compressor.compress",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not compress while stopped")))
    mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi\n\nAssistant: yo")
    core.set_stopped(True)
    processed = mqueue.drain()
    assert processed == 0
    assert len(list((tmp_path / "queue").glob("*.json"))) == 1


def test_drain_keeps_item_when_session_recording_on(qstate, tmp_path, monkeypatch):
    # Per-window recording on for the session, with a stored backend → kept and
    # routed to that backend even though the dir was never enabled.
    from memory.backends.sqlite import SQLiteBackend
    from memory.compressor import CompressedSummary

    db = SQLiteBackend(db_path=tmp_path / "mem.db")
    db.init()
    seen = {}
    monkeypatch.setattr("memory.core.get_backend",
                        lambda name=None: (seen.__setitem__("name", name) or db))
    monkeypatch.setattr("memory.compressor.compress",
                        lambda t, provider=None, model=None: CompressedSummary(
                            "t", "p", "## Title\nt\n\n## Project\np\n\n## Tags\nx", ["x"]))
    monkeypatch.setattr(hook, "_provider_key_present", lambda provider: True)

    core.set_recording_for_session("sess-1", "claude")
    mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi\n\nAssistant: yo",
                   meta_model="m", backend="firebase")
    processed = mqueue.drain()
    assert processed == 1
    assert list((tmp_path / "queue").glob("*.json")) == []
    assert seen["name"] == "firebase"


def test_drain_routes_to_recorded_backend(qstate, tmp_path, monkeypatch):
    from memory.backends.sqlite import SQLiteBackend
    from memory.compressor import CompressedSummary

    db = SQLiteBackend(db_path=tmp_path / "mem.db")
    db.init()
    seen = {}

    def fake_get_backend(name=None):
        seen["name"] = name
        return db
    monkeypatch.setattr("memory.core.get_backend", fake_get_backend)
    monkeypatch.setattr("memory.compressor.compress",
                        lambda t, provider=None, model=None: CompressedSummary(
                            "t", "p", "## Title\nt\n\n## Project\np\n\n## Tags\nx", ["x"]))
    monkeypatch.setattr(hook, "_provider_key_present", lambda provider: True)

    core.set_recording_for_session("sess-1", "claude")
    # Item recorded with an explicit backend; drain must route to it.
    mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi\n\nAssistant: yo",
                   meta_model="m", backend="firebase")
    mqueue.drain()
    assert seen["name"] == "firebase"


def test_drain_empty_queue_is_noop(qstate):
    assert mqueue.drain() == 0


def test_session_pin_resolves_over_env_at_drain(qstate, tmp_path, monkeypatch):
    # A per-window session pin must win over MEMORY_BACKEND env when the
    # queued item carries no explicit backend (legacy backend=None).
    from memory.backends.sqlite import SQLiteBackend
    from memory.compressor import CompressedSummary

    db = SQLiteBackend(db_path=tmp_path / "mem.db")
    db.init()
    seen = {}
    monkeypatch.setattr("memory.core.get_backend",
                        lambda name=None: (seen.__setitem__("name", name) or db))
    monkeypatch.setattr("memory.compressor.compress",
                        lambda t, provider=None, model=None: CompressedSummary(
                            "t", "p", "## Title\nt\n\n## Project\np\n\n## Tags\nx", ["x"]))
    monkeypatch.setattr(hook, "_provider_key_present", lambda provider: True)
    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_LLM_PROVIDER", "anthropic")

    core.set_recording_for_session("sess-1", "claude")
    core.set_backend_for_session("sess-1", "firebase")  # per-window pin wins

    # Enqueue with no explicit backend; drain re-resolves with session_id.
    mqueue.enqueue("sess-1", tmp_path, "claude", "User: hi\n\nAssistant: yo")
    mqueue.drain()
    assert seen["name"] == "firebase"


def test_drain_keeps_a_new_snapshot_captured_during_compression(
        qstate, tmp_path, monkeypatch):
    path = mqueue.enqueue("live", tmp_path, "claude", "User: first", backend="firebase")

    def process(payload):
        assert payload["transcript"] == "User: first"
        mqueue.enqueue("live", tmp_path, "claude", "User: second", backend="firebase")
        return True

    monkeypatch.setattr(mqueue, "_process_one", process)
    assert mqueue.drain() == 1
    assert json.loads(path.read_text())["transcript"] == "User: second"
    monkeypatch.setattr(mqueue, "_process_one", lambda payload: True)
    assert mqueue.drain() == 1
    assert not path.exists()


def test_identical_transcript_can_change_its_destination(qstate, tmp_path):
    path = mqueue.enqueue("live", tmp_path, "claude", "User: hello", backend="sqlite")
    mqueue.enqueue("live", tmp_path, "claude", "User: hello", backend="firebase")
    assert json.loads(path.read_text())["backend"] == "firebase"


def test_simultaneous_drains_process_each_snapshot_once(qstate, tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    mqueue.enqueue("live", tmp_path, "claude", "User: hello", backend="sqlite")
    started = threading.Barrier(2)
    processed = []

    def process(payload):
        processed.append(payload["transcript"])
        time.sleep(0.05)
        return True

    def drain():
        started.wait(timeout=5)
        return mqueue.drain()

    monkeypatch.setattr(mqueue, "_process_one", process)
    with ThreadPoolExecutor(max_workers=2) as executor:
        workers = [executor.submit(drain) for _ in range(2)]
        results = [worker.result(timeout=5) for worker in workers]
    assert sum(results) == 1
    assert processed == ["User: hello"]
