import pytest
from datetime import datetime
from pathlib import Path

from memory.backends.sqlite import SQLiteBackend
from memory.backends.base import Session


@pytest.fixture
def backend(tmp_path):
    db = SQLiteBackend(db_path=tmp_path / "test.db")
    db.init()
    return db


def make_session(**kwargs) -> Session:
    defaults = dict(
        id="abc123",
        project="testproject",
        title="Test Session",
        summary="## Title\nTest\n## Project\ntestproject",
        tags=["python", "test"],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return Session(**defaults)


def test_save_and_get(backend):
    s = make_session()
    backend.save(s)
    result = backend.get("abc123")
    assert result is not None
    assert result.title == "Test Session"
    assert result.project == "testproject"
    assert result.tags == ["python", "test"]


def test_get_missing_returns_none(backend):
    assert backend.get("nonexistent") is None


def test_list_all(backend):
    backend.save(make_session(id="1", title="First"))
    backend.save(make_session(id="2", title="Second"))
    sessions = backend.list()
    assert len(sessions) == 2


def test_list_filter_by_project(backend):
    backend.save(make_session(id="1", project="alpha", title="Alpha"))
    backend.save(make_session(id="2", project="beta", title="Beta"))
    results = backend.list(project="alpha")
    assert len(results) == 1
    assert results[0].title == "Alpha"


def test_delete(backend):
    backend.save(make_session())
    backend.delete("abc123")
    assert backend.get("abc123") is None


def test_delete_nonexistent_does_not_raise(backend):
    backend.delete("ghost")


def test_update_existing(backend):
    s = make_session()
    backend.save(s)
    s.title = "Updated Title"
    backend.save(s)
    result = backend.get("abc123")
    assert result.title == "Updated Title"


def test_search(backend):
    backend.save(make_session(id="1", title="JWT Auth flow", summary="## Title\nJWT Auth flow"))
    backend.save(make_session(id="2", title="React dashboard", summary="## Title\nReact dashboard"))
    results = backend.search("JWT")
    assert len(results) == 1
    assert results[0].title == "JWT Auth flow"


def test_tags_roundtrip(backend):
    s = make_session(tags=["react", "typescript", "vite"])
    backend.save(s)
    result = backend.get("abc123")
    assert result.tags == ["react", "typescript", "vite"]


def test_empty_tags(backend):
    s = make_session(tags=[])
    backend.save(s)
    result = backend.get("abc123")
    assert result.tags == []
