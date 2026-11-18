import uuid
from datetime import datetime
from typing import Optional

from .base import BaseBackend, Session


class FirebaseBackend(BaseBackend):
    """
    Requires: pip install ai-agent-portable-memory[firebase]
    Uses Firestore as the database.

    Env vars:
        FIREBASE_CREDENTIALS_PATH   path to serviceAccountKey.json
        FIREBASE_PROJECT_ID

    Firestore collection: "sessions"
    """

    def __init__(self, credentials_path: str, project_id: str):
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
        except ImportError:
            raise ImportError("Run: pip install ai-agent-portable-memory[firebase]")

        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {"projectId": project_id})

        self.db = firestore.client()
        self.col = self.db.collection("sessions")

    def init(self) -> None:
        pass  # Firestore creates collections automatically

    def _doc_to_session(self, doc) -> Session:
        d = doc.to_dict()
        return Session(
            id=doc.id,
            project=d.get("project", "default"),
            title=d["title"],
            summary=d["summary"],
            tags=d.get("tags", []),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )

    def save(self, session: Session) -> None:
        if not session.id:
            session.id = str(uuid.uuid4())
        now = datetime.utcnow()
        data = {
            "project": session.project,
            "title": session.title,
            "summary": session.summary,
            "tags": session.tags,
            "created_at": session.created_at or now,
            "updated_at": now,
        }
        self.col.document(session.id).set(data)

    def get(self, session_id: str) -> Optional[Session]:
        doc = self.col.document(session_id).get()
        return self._doc_to_session(doc) if doc.exists else None

    def list(self, project: Optional[str] = None) -> list[Session]:
        q = self.col.order_by("updated_at", direction="DESCENDING")
        if project:
            q = q.where("project", "==", project)
        return [self._doc_to_session(d) for d in q.stream()]

    def search(self, query: str, project: Optional[str] = None) -> list[Session]:
        # Firestore has no full-text search — client-side filter on loaded docs
        query_lower = query.lower()
        return [
            s for s in self.list(project)
            if query_lower in s.title.lower() or query_lower in s.summary.lower()
        ]

    def delete(self, session_id: str) -> None:
        self.col.document(session_id).delete()
