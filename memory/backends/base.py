from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Session:
    id: str
    project: str
    title: str
    summary: str
    created_at: datetime
    updated_at: datetime
    tags: list[str] = field(default_factory=list)


class BaseBackend(ABC):
    @abstractmethod
    def init(self) -> None:
        """Initialize storage (create tables, buckets, etc.)"""

    @abstractmethod
    def save(self, session: Session) -> None:
        """Insert or update a session."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[Session]:
        """Fetch a single session by ID."""

    @abstractmethod
    def list(self, project: Optional[str] = None) -> list[Session]:
        """List all sessions, optionally filtered by project."""

    @abstractmethod
    def search(self, query: str, project: Optional[str] = None) -> list[Session]:
        """Full-text search over titles and summaries."""

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete a session."""
