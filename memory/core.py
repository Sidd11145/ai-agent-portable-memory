import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .backends.base import BaseBackend, Session

load_dotenv()


def get_backend() -> BaseBackend:
    backend_type = os.environ.get("MEMORY_BACKEND", "sqlite").lower()

    if backend_type == "sqlite":
        from .backends.sqlite import SQLiteBackend
        db_path = os.environ.get("SQLITE_DB_PATH", str(Path.home() / ".ai-memory" / "memory.db"))
        return SQLiteBackend(db_path=Path(db_path))

    elif backend_type == "supabase":
        from .backends.supabase import SupabaseBackend
        url = _require_env("SUPABASE_URL")
        key = _require_env("SUPABASE_KEY")
        return SupabaseBackend(url=url, key=key)

    elif backend_type == "s3":
        from .backends.s3 import S3Backend
        bucket = _require_env("S3_BUCKET_NAME")
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        region = os.environ.get("AWS_REGION", "us-east-1")
        return S3Backend(bucket=bucket, endpoint_url=endpoint, region=region)

    elif backend_type == "firebase":
        from .backends.firebase import FirebaseBackend
        creds = _require_env("FIREBASE_CREDENTIALS_PATH")
        project = _require_env("FIREBASE_PROJECT_ID")
        return FirebaseBackend(credentials_path=creds, project_id=project)

    else:
        raise ValueError(f"Unknown MEMORY_BACKEND: {backend_type}. Choose: sqlite, supabase, s3, firebase")


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(f"Environment variable {key} is required for this backend.")
    return val


def create_session(title: str, project: str, summary: str, tags: list[str]) -> Session:
    backend = get_backend()
    backend.init()
    session = Session(
        id=str(uuid.uuid4()),
        project=project,
        title=title,
        summary=summary,
        tags=tags,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    backend.save(session)
    return session


def list_sessions(project: Optional[str] = None) -> list[Session]:
    backend = get_backend()
    backend.init()
    return backend.list(project=project)


def load_session(session_id: str) -> Optional[Session]:
    backend = get_backend()
    backend.init()
    return backend.get(session_id)


def search_sessions(query: str, project: Optional[str] = None) -> list[Session]:
    backend = get_backend()
    backend.init()
    return backend.search(query=query, project=project)


def delete_session(session_id: str) -> None:
    backend = get_backend()
    backend.init()
    backend.delete(session_id)
