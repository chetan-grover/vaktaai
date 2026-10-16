"""Request/response models for the API."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import re

from pydantic import AfterValidator, BaseModel, Field, field_validator

# SQLite drops tzinfo; every stored timestamp is UTC, so mark it as such for the browser.
UtcDatetime = Annotated[datetime, AfterValidator(lambda d: d if d.tzinfo else d.replace(tzinfo=timezone.utc))]


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _email(value: str) -> str:
    value = value.strip().lower()
    if not EMAIL_RE.match(value):
        raise ValueError("enter a valid email address")
    return value


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)

    _check_email = field_validator("email")(_email)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("enter your name")
        return v.strip()


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)

    _check_email = field_validator("email")(_email)


class UserOut(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    created_at: UtcDatetime


class AuthOut(BaseModel):
    token: str
    user: UserOut


class SessionOut(BaseModel):
    id: str
    user_id: int
    title: str
    source: str
    created_at: UtcDatetime
    status: str
    error: Optional[str] = None
    duration_s: Optional[float] = None
    confidence: Optional[float] = None
    authoritativeness: Optional[float] = None
    engagement: Optional[float] = None
    congruence: Optional[float] = None
    persona: Optional[str] = None
    keep_video: bool = False
    video_deleted: bool = False
    progress: Optional[dict[str, Any]] = None


class SessionPatch(BaseModel):
    title: Optional[str] = Field(default=None, max_length=120)
    keep_video: Optional[bool] = None


class DrillPatch(BaseModel):
    done: bool
