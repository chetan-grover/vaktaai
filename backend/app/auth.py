"""Email + password accounts with bearer tokens.

Passwords: PBKDF2-HMAC-SHA256 (stdlib, 260k iterations, per-user salt).
Tokens: 32 random bytes, returned once; only their SHA-256 is stored, valid for 30 days.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import timedelta, timezone

from fastapi import Depends, Header, HTTPException, Query
from sqlmodel import Session, select

from .db import AuthToken, User, get_db, utcnow

ITERATIONS = 260_000
TOKEN_DAYS = 30


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return f"pbkdf2_sha256${ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        _, iterations, salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations)).hex()
    except ValueError:
        return False
    return hmac.compare_digest(check, digest)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue_token(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(AuthToken(token_hash=_token_hash(token), user_id=user.id, expires_at=utcnow() + timedelta(days=TOKEN_DAYS)))
    db.commit()
    return token


def revoke_token(db: Session, token: str) -> None:
    row = db.get(AuthToken, _token_hash(token))
    if row:
        db.delete(row)
        db.commit()


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def _user_for(db: Session, token: str | None) -> User:
    if not token:
        raise HTTPException(401, "sign in required")
    row = db.get(AuthToken, _token_hash(token))
    if row is None:
        raise HTTPException(401, "session expired, please sign in again")
    expires = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=timezone.utc)  # SQLite drops tz
    if expires < utcnow():
        raise HTTPException(401, "session expired, please sign in again")
    user = db.get(User, row.user_id)
    if user is None:
        raise HTTPException(401, "account no longer exists")
    return user


def current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> User:
    return _user_for(db, _bearer(authorization))


def current_user_or_query(authorization: str | None = Header(default=None), token: str | None = Query(default=None),
                          db: Session = Depends(get_db)) -> User:
    """For <video src>, which cannot send headers: also accept ?token=."""
    return _user_for(db, _bearer(authorization) or token)


def find_user_by_email(db: Session, email: str) -> User | None:
    return db.exec(select(User).where(User.email == email.strip().lower())).first()
