import json
import uuid
from datetime import datetime
from typing import Optional

from .base import BaseBackend, Session


class S3Backend(BaseBackend):
    """
    Requires: pip install ai-agent-portable-memory[s3]
    Works with AWS S3, Cloudflare R2, MinIO, or any S3-compatible storage.

    Env vars:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        S3_BUCKET_NAME
        S3_ENDPOINT_URL   (optional — for R2/MinIO, e.g. https://<id>.r2.cloudflarestorage.com)
        AWS_REGION        (optional, default us-east-1)

    Layout in bucket:
        sessions/<id>.json
        index.json          (lightweight index for list/search)
    """

    def __init__(self, bucket: str, endpoint_url: Optional[str] = None, region: str = "us-east-1"):
        try:
            import boto3
        except ImportError:
            raise ImportError("Run: pip install ai-agent-portable-memory[s3]")
        kwargs = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        self.s3 = boto3.client("s3", **kwargs)
        self.bucket = bucket

    def init(self) -> None:
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception:
            self.s3.create_bucket(Bucket=self.bucket)
        # Initialize empty index if missing
        try:
            self.s3.head_object(Bucket=self.bucket, Key="index.json")
        except Exception:
            self._write_index([])

    def _write_index(self, index: list[dict]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key="index.json",
            Body=json.dumps(index, default=str),
            ContentType="application/json",
        )

    def _read_index(self) -> list[dict]:
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key="index.json")
            return json.loads(resp["Body"].read())
        except Exception:
            return []

    def _session_key(self, session_id: str) -> str:
        return f"sessions/{session_id}.json"

    def save(self, session: Session) -> None:
        if not session.id:
            session.id = str(uuid.uuid4())
        now = datetime.utcnow()
        session.updated_at = now
        if not session.created_at:
            session.created_at = now

        data = {
            "id": session.id,
            "project": session.project,
            "title": session.title,
            "summary": session.summary,
            "tags": session.tags,
            "model": session.model,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._session_key(session.id),
            Body=json.dumps(data, indent=2),
            ContentType="application/json",
        )

        # Update index (store lightweight metadata only)
        index = self._read_index()
        index = [e for e in index if e["id"] != session.id]
        index.insert(0, {
            "id": session.id,
            "project": session.project,
            "title": session.title,
            "tags": session.tags,
            "model": session.model,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        })
        self._write_index(index)

    def get(self, session_id: str) -> Optional[Session]:
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self._session_key(session_id))
            data = json.loads(resp["Body"].read())
            return Session(
                id=data["id"],
                project=data["project"],
                title=data["title"],
                summary=data["summary"],
                tags=data["tags"],
                model=data.get("model", ""),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
            )
        except Exception:
            return None

    def list(self, project: Optional[str] = None) -> list[Session]:
        index = self._read_index()
        if project:
            index = [e for e in index if e["project"] == project]
        # Return lightweight sessions from index (no summary — load on demand)
        return [
            Session(
                id=e["id"],
                project=e["project"],
                title=e["title"],
                summary="",
                tags=e["tags"],
                model=e.get("model", ""),
                created_at=datetime.fromisoformat(e["created_at"]),
                updated_at=datetime.fromisoformat(e["updated_at"]),
            )
            for e in index
        ]

    def search(self, query: str, project: Optional[str] = None) -> list[Session]:
        query_lower = query.lower()
        results = []
        for session in self.list(project):
            full = self.get(session.id)
            if full and (query_lower in full.title.lower() or query_lower in full.summary.lower()):
                results.append(full)
        return results

    def delete(self, session_id: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=self._session_key(session_id))
        index = [e for e in self._read_index() if e["id"] != session_id]
        self._write_index(index)
