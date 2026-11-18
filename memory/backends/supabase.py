import json
import uuid
from datetime import datetime
from typing import Optional

from .base import BaseBackend, Session


class SupabaseBackend(BaseBackend):
    """
    Requires: pip install ai-agent-portable-memory[supabase]
    Env vars: SUPABASE_URL, SUPABASE_KEY

    Run this SQL in your Supabase project first:
        CREATE TABLE sessions (
            id          TEXT PRIMARY KEY,
            project     TEXT NOT NULL DEFAULT 'default',
            title       TEXT NOT NULL,
            summary     TEXT NOT NULL,
            tags        JSONB NOT NULL DEFAULT '[]',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX ON sessions USING GIN (to_tsvector('english', title || ' ' || summary));
    """

    def __init__(self, url: str, key: str):
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError("Run: pip install ai-agent-portable-memory[supabase]")
        self.client = create_client(url, key)

    def init(self) -> None:
        pass  # Table must be created manually via SQL above

    def _row_to_session(self, row: dict) -> Session:
        return Session(
            id=row["id"],
            project=row["project"],
            title=row["title"],
            summary=row["summary"],
            tags=row["tags"] if isinstance(row["tags"], list) else json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00")),
        )

    def save(self, session: Session) -> None:
        if not session.id:
            session.id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        data = {
            "id": session.id,
            "project": session.project,
            "title": session.title,
            "summary": session.summary,
            "tags": session.tags,
            "updated_at": now,
        }
        self.client.table("sessions").upsert(data).execute()

    def get(self, session_id: str) -> Optional[Session]:
        resp = self.client.table("sessions").select("*").eq("id", session_id).single().execute()
        return self._row_to_session(resp.data) if resp.data else None

    def list(self, project: Optional[str] = None) -> list[Session]:
        q = self.client.table("sessions").select("*").order("updated_at", desc=True)
        if project:
            q = q.eq("project", project)
        resp = q.execute()
        return [self._row_to_session(r) for r in resp.data]

    def search(self, query: str, project: Optional[str] = None) -> list[Session]:
        q = self.client.table("sessions").select("*").text_search(
            "title,summary", query, config="english"
        )
        if project:
            q = q.eq("project", project)
        resp = q.execute()
        return [self._row_to_session(r) for r in resp.data]

    def delete(self, session_id: str) -> None:
        self.client.table("sessions").delete().eq("id", session_id).execute()
