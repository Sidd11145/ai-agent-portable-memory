from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import BaseBackend, Session

DEFAULT_DB_PATH = Path.home() / ".ai-memory" / "memory.db"


class SQLiteBackend(BaseBackend):
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    project     TEXT NOT NULL DEFAULT 'default',
                    title       TEXT NOT NULL,
                    summary     TEXT NOT NULL,
                    tags        TEXT NOT NULL DEFAULT '[]',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts
                USING fts5(id UNINDEXED, title, summary, content='sessions', content_rowid='rowid')
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
                    INSERT INTO sessions_fts(rowid, id, title, summary)
                    VALUES (new.rowid, new.id, new.title, new.summary);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
                    INSERT INTO sessions_fts(sessions_fts, rowid, id, title, summary)
                    VALUES ('delete', old.rowid, old.id, old.title, old.summary);
                END
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            project=row["project"],
            title=row["title"],
            summary=row["summary"],
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def save(self, session: Session) -> None:
        now = datetime.utcnow().isoformat()
        if not session.id:
            session.id = str(uuid.uuid4())
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM sessions WHERE id = ?", (session.id,)).fetchone()
            if existing:
                conn.execute("""
                    UPDATE sessions SET project=?, title=?, summary=?, tags=?, updated_at=?
                    WHERE id=?
                """, (session.project, session.title, session.summary,
                      json.dumps(session.tags), now, session.id))
                conn.execute("""
                    INSERT INTO sessions_fts(sessions_fts, rowid, id, title, summary)
                    SELECT 'delete', rowid, id, title, summary FROM sessions WHERE id=?
                """, (session.id,))
                conn.execute("""
                    INSERT INTO sessions_fts(rowid, id, title, summary)
                    SELECT rowid, id, title, summary FROM sessions WHERE id=?
                """, (session.id,))
            else:
                session.created_at = session.created_at or datetime.utcnow()
                session.updated_at = datetime.utcnow()
                conn.execute("""
                    INSERT INTO sessions (id, project, title, summary, tags, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session.id, session.project, session.title, session.summary,
                      json.dumps(session.tags), session.created_at.isoformat(), now))

    def get(self, session_id: str) -> Optional[Session]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def list(self, project: Optional[str] = None) -> list[Session]:
        with self._connect() as conn:
            if project:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE project = ? ORDER BY updated_at DESC", (project,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [self._row_to_session(r) for r in rows]

    def search(self, query: str, project: Optional[str] = None) -> list[Session]:
        with self._connect() as conn:
            if project:
                rows = conn.execute("""
                    SELECT s.* FROM sessions s
                    JOIN sessions_fts f ON s.id = f.id
                    WHERE sessions_fts MATCH ? AND s.project = ?
                    ORDER BY rank
                """, (query, project)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT s.* FROM sessions s
                    JOIN sessions_fts f ON s.id = f.id
                    WHERE sessions_fts MATCH ?
                    ORDER BY rank
                """, (query,)).fetchall()
        return [self._row_to_session(r) for r in rows]

    def delete(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
