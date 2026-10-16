"""SQLModel tables and engine (SQLite by default; any SQLAlchemy URL via VAKTA_DB_URL)."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterator, Optional

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from pipeline.settings import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    email: Optional[str] = Field(default=None, index=True, unique=True)
    password_hash: Optional[str] = None  # None = legacy name-only profile, claimable on sign-up
    created_at: datetime = Field(default_factory=utcnow)


class AuthToken(SQLModel, table=True):
    __tablename__ = "auth_token"

    token_hash: str = Field(primary_key=True)  # sha256 of the bearer token; the raw token is never stored
    user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime


class SpeechSession(SQLModel, table=True):
    __tablename__ = "speech_session"

    id: str = Field(primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    title: str = "Untitled session"
    source: str = "upload"  # webcam | upload
    created_at: datetime = Field(default_factory=utcnow)
    status: str = "queued"  # queued | processing | done | failed
    error: Optional[str] = None
    duration_s: Optional[float] = None
    confidence: Optional[float] = None
    authoritativeness: Optional[float] = None
    engagement: Optional[float] = None
    congruence: Optional[float] = None
    persona: Optional[str] = None
    keep_video: bool = False
    video_deleted: bool = False


class DrillProgress(SQLModel, table=True):
    __tablename__ = "drill_progress"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="speech_session.id", index=True)
    drill_id: str
    done: bool = False
    done_at: Optional[datetime] = None


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    args = {"check_same_thread": False} if settings.db_url.startswith("sqlite") else {}
    engine = create_engine(settings.db_url, connect_args=args)
    SQLModel.metadata.create_all(engine)
    if settings.db_url.startswith("sqlite"):
        _migrate_sqlite(engine)
    return engine


def _migrate_sqlite(engine: Engine) -> None:
    """Bring databases created before sign-in existed up to date (create_all never alters tables)."""
    from sqlalchemy import text

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(user)"))}
        if "email" not in cols:
            conn.execute(text("ALTER TABLE user ADD COLUMN email VARCHAR"))
        if "password_hash" not in cols:
            conn.execute(text("ALTER TABLE user ADD COLUMN password_hash VARCHAR"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_user_email ON user (email)"))
        unique_name = [r for r in conn.execute(text("PRAGMA index_list(user)")) if r[1] == "ix_user_name" and r[2]]
        if unique_name:  # display names no longer need to be unique
            conn.execute(text("DROP INDEX ix_user_name"))
            conn.execute(text("CREATE INDEX ix_user_name ON user (name)"))


def get_db() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
