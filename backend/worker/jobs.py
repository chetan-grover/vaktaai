"""Job runner: executes the pipeline for one session and records the outcome in the DB.

Runs inside FastAPI BackgroundTasks (blueprint: keep infra minimal). The function is plain
and synchronous, so it can be handed to RQ unchanged if a queue is ever needed.
"""
from __future__ import annotations

import logging
import shutil
import threading
from datetime import timedelta

from sqlmodel import Session, select

from app.db import SpeechSession, get_engine, utcnow
from pipeline.runner import SessionCancelled, run_pipeline
from pipeline.settings import get_settings

log = logging.getLogger("vakta.worker")
# On a laptop CPU two pipelines in parallel make both crawl, so jobs run one at a time.
_JOB_LOCK = threading.Lock()


def _update(session_id: str, **fields) -> None:
    with Session(get_engine()) as db:
        row = db.get(SpeechSession, session_id)
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
        db.add(row)
        db.commit()


def run_session_job(session_id: str) -> None:
    with _JOB_LOCK:
        _run_session_job(session_id)


def resume_unfinished() -> list[str]:
    """Sessions left queued/processing by a server restart, oldest first."""
    with Session(get_engine()) as db:
        rows = db.exec(select(SpeechSession).where(SpeechSession.status.in_(["queued", "processing"]))
                       .order_by(SpeechSession.created_at)).all()
        return [r.id for r in rows]


def run_queue(session_ids: list[str]) -> None:
    for session_id in session_ids:
        run_session_job(session_id)


def _run_session_job(session_id: str) -> None:
    if not (get_settings().sessions_dir / session_id).exists():
        return  # deleted while waiting in the queue
    _update(session_id, status="processing", error=None)
    try:
        report = run_pipeline(session_id)
    except SessionCancelled:
        log.info("session %s deleted during analysis; stopped", session_id)
        delete_session_files(session_id)
        return
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        log.exception("session %s failed", session_id)
        _update(session_id, status="failed", error=str(exc)[:500])
        return
    traits = report.get("traits") or {}
    persona = None
    if report.get("persona_features"):
        try:
            from analytics.persona.infer import load_map

            persona = load_map(str(get_settings().models_dir)).assign(report["persona_features"])["name"]
        except Exception:  # noqa: BLE001
            persona = None
    _update(
        session_id,
        status="done",
        duration_s=report.get("duration_s"),
        confidence=(traits.get("confidence") or {}).get("score"),
        authoritativeness=(traits.get("authoritativeness") or {}).get("score"),
        engagement=(traits.get("engagement") or {}).get("score"),
        congruence=(report.get("congruence") or {}).get("score"),
        persona=persona,
    )


def delete_video_files(session_id: str) -> None:
    folder = get_settings().sessions_dir / session_id
    for pattern in ("input.*", "video.mp4", "audio.wav"):
        for path in folder.glob(pattern):
            path.unlink(missing_ok=True)


def purge_expired_videos() -> int:
    """NFR3: drop raw video older than the retention window unless the user kept it."""
    cutoff = utcnow() - timedelta(days=get_settings().retention_days)
    purged = 0
    with Session(get_engine()) as db:
        rows = db.exec(select(SpeechSession).where(SpeechSession.keep_video == False,  # noqa: E712
                                                   SpeechSession.video_deleted == False)).all()  # noqa: E712
        for row in rows:
            created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=cutoff.tzinfo)
            if created < cutoff:
                delete_video_files(row.id)
                row.video_deleted = True
                db.add(row)
                purged += 1
        db.commit()
    return purged


def delete_session_files(session_id: str) -> None:
    shutil.rmtree(get_settings().sessions_dir / session_id, ignore_errors=True)
