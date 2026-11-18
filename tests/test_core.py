import pytest
import os
from unittest.mock import patch
from memory.core import get_backend, _require_env
from memory.backends.sqlite import SQLiteBackend


def test_get_backend_defaults_to_sqlite(tmp_path):
    with patch.dict(os.environ, {"MEMORY_BACKEND": "sqlite", "SQLITE_DB_PATH": str(tmp_path / "m.db")}):
        backend = get_backend()
    assert isinstance(backend, SQLiteBackend)


def test_get_backend_unknown_raises():
    with patch.dict(os.environ, {"MEMORY_BACKEND": "unknown"}):
        with pytest.raises(ValueError, match="Unknown MEMORY_BACKEND"):
            get_backend()


def test_require_env_raises_when_missing():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("SOME_KEY", None)
        with pytest.raises(EnvironmentError, match="SOME_KEY"):
            _require_env("SOME_KEY")


def test_require_env_returns_value():
    with patch.dict(os.environ, {"MY_KEY": "hello"}):
        assert _require_env("MY_KEY") == "hello"
