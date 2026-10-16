"""VaktaAI API. Run from backend/: uvicorn app.main:app --reload  (docs at /docs)."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from pipeline.base import INPUT_PREFIX, SessionContext
from pipeline.runner import REGISTRY, read_progress, rerun
from pipeline.settings import get_settings
from pipeline.utils import ffmpeg_exe
from worker.jobs import (delete_session_files, delete_video_files, purge_expired_videos, resume_unfinished, run_queue,
                         run_session_job)

from .auth import (_bearer, current_user, current_user_or_query, find_user_by_email, hash_password, issue_token,
                   revoke_token, verify_password)
from .db import DrillProgress, SpeechSession, User, get_db, get_engine, utcnow
from .schemas import AuthOut, DrillPatch, LoginIn, RegisterIn, SessionOut, SessionPatch, UserOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
ALLOWED_EXT = {".mp4", ".webm", ".mov", ".mkv"}
CHUNK = 1 << 20
TREND_METRICS = {
    "wpm": ("voice", "summary", "wpm"),
    "filler_rate_per_min": ("voice", "summary", "filler_rate_per_min"),
    "pitch_std_st": ("voice", "summary", "pitch_std_st"),
    "eye_contact_ratio": ("gaze", "summary", "eye_contact_ratio"),
    "gaze_coverage": ("gaze", "summary", "coverage"),
    "bpm_rise": ("physio", "summary", "bpm_rise"),
    "transition_quality": ("content", "rhetoric", "transition_quality"),
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_settings().sessions_dir.mkdir(parents=True, exist_ok=True)
    get_engine()
    purge_expired_videos()
    pending = resume_unfinished()
    if pending:
        threading.Thread(target=run_queue, args=(pending,), daemon=True, name="vakta-resume").start()
    yield


app = FastAPI(title="VaktaAI", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- helpers
def _session_or_404(db: Session, session_id: str, user: User) -> SpeechSession:
    """Other users' sessions look exactly like missing ones."""
    row = db.get(SpeechSession, session_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(404, "session not found")
    return row


def _out(row: SpeechSession, with_progress: bool = False) -> SessionOut:
    progress = read_progress(SessionContext.for_session(row.id)) if with_progress else None
    return SessionOut(**row.model_dump(), progress=progress)


def _load_report(session_id: str) -> dict[str, Any] | None:
    path = get_settings().sessions_dir / session_id / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _iso(dt) -> str:
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def _dig(d: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        d = d.get(key) if isinstance(d, dict) else None
    return d


def _model_status() -> dict[str, bool]:
    m = get_settings().models_dir
    return {
        "physnet": any((m / "rppg").glob("*PhysNet*.pth")) if (m / "rppg").exists() else False,
        "l2cs": any((m / "gaze" / f).exists() for f in ("l2cs_gaze360_resnet50.safetensors", "L2CSNet_gaze360.pkl")),
        "gaze_gru": (m / "gaze" / "gaze_gru.pt").exists(),
        "e3_head": (m / "fusion" / "e3_head.pt").exists(),
        "persona_vae": (m / "persona" / "meta.json").exists(),
        "llm_endpoint": bool(get_settings().llm_url),
    }


# --------------------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ffmpeg": ffmpeg_exe(), "models": _model_status(),
            "whisper": get_settings().whisper_model}


@app.post("/api/auth/register", response_model=AuthOut, status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)) -> AuthOut:
    if find_user_by_email(db, body.email):
        raise HTTPException(409, "an account with this email already exists; sign in instead")
    # A profile from the old name-only login is claimed by the first sign-up with the same name, keeping its sessions.
    user = db.exec(select(User).where(User.name == body.name, User.password_hash == None)).first()  # noqa: E711
    user = user or User(name=body.name)
    user.email, user.password_hash = body.email, hash_password(body.password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return AuthOut(token=issue_token(db, user), user=UserOut.model_validate(user, from_attributes=True))


@app.post("/api/auth/login", response_model=AuthOut)
def login(body: LoginIn, db: Session = Depends(get_db)) -> AuthOut:
    user = find_user_by_email(db, body.email)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "incorrect email or password")
    return AuthOut(token=issue_token(db, user), user=UserOut.model_validate(user, from_attributes=True))


@app.post("/api/auth/logout", status_code=204, response_model=None)
def logout(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> None:
    token = _bearer(authorization)
    if token:
        revoke_token(db, token)


@app.get("/api/auth/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user


@app.post("/api/sessions", response_model=SessionOut, status_code=201)
async def create_session(background: BackgroundTasks, file: UploadFile = File(...),
                         title: str = Form("Untitled session"), source: str = Form("upload"),
                         calibration_s: float | None = Form(None), db: Session = Depends(get_db),
                         user: User = Depends(current_user)) -> SessionOut:
    ext = Path(file.filename or "").suffix.lower() or (".webm" if source == "webcam" else "")
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"unsupported file type '{ext}'; use MP4, WebM or MOV")
    session_id = uuid.uuid4().hex[:12]
    ctx = SessionContext.for_session(session_id)
    limit = get_settings().max_upload_mb * CHUNK
    written = 0
    with (ctx.session_dir / f"{INPUT_PREFIX}{ext}").open("wb") as out:
        while chunk := await file.read(CHUNK):
            written += len(chunk)
            if written > limit:
                out.close()
                delete_session_files(session_id)
                raise HTTPException(413, f"file larger than {get_settings().max_upload_mb} MB")
            out.write(chunk)
    ctx.write_json("meta.json", {"title": title, "source": source, "calibration_s": calibration_s,
                                 "original_filename": file.filename, "bytes": written})
    row = SpeechSession(id=session_id, user_id=user.id, title=title.strip() or "Untitled session",
                        source="webcam" if source == "webcam" else "upload")
    db.add(row)
    db.commit()
    db.refresh(row)
    background.add_task(run_session_job, session_id)
    return _out(row)


@app.get("/api/sessions", response_model=list[SessionOut])
def list_sessions(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[SessionOut]:
    rows = db.exec(select(SpeechSession).where(SpeechSession.user_id == user.id)
                   .order_by(SpeechSession.created_at.desc())).all()
    return [_out(r) for r in rows]


@app.get("/api/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> SessionOut:
    return _out(_session_or_404(db, session_id, user), with_progress=True)


@app.get("/api/sessions/{session_id}/report")
def get_report(session_id: str, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> dict[str, Any]:
    row = _session_or_404(db, session_id, user)
    report = _load_report(session_id)
    if report is None:
        raise HTTPException(409, f"report not ready (status: {row.status})")
    done = db.exec(select(DrillProgress).where(DrillProgress.session_id == session_id)).all()
    report["drill_progress"] = {d.drill_id: d.done for d in done}
    report["session"] = _out(row).model_dump(mode="json")
    return report


@app.get("/api/sessions/{session_id}/video")
def get_video(session_id: str, db: Session = Depends(get_db),
              user: User = Depends(current_user_or_query)) -> FileResponse:
    _session_or_404(db, session_id, user)
    ctx = SessionContext.for_session(session_id)
    path = ctx.video_path if ctx.video_path.exists() else next(ctx.session_dir.glob(f"{INPUT_PREFIX}.*"), None)
    if path is None or not path.exists():
        raise HTTPException(404, "video deleted")
    return FileResponse(path, media_type="video/mp4" if path.suffix == ".mp4" else None)


@app.get("/api/sessions/{session_id}/modules/{name}")
def get_module(session_id: str, name: str, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> dict[str, Any]:
    _session_or_404(db, session_id, user)
    env = SessionContext.for_session(session_id).read_envelope(name)
    if env is None:
        raise HTTPException(404, "module output not found")
    return env


@app.post("/api/sessions/{session_id}/reanalyse", response_model=SessionOut)
def reanalyse(session_id: str, background: BackgroundTasks, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> SessionOut:
    """Run the full pipeline again (e.g. after new models or tuned thresholds). Needs the raw video."""
    row = _session_or_404(db, session_id, user)
    if row.video_deleted or not any(SessionContext.for_session(session_id).session_dir.glob(f"{INPUT_PREFIX}.*")):
        raise HTTPException(409, "the raw video was deleted, so this session cannot be re-analysed")
    if row.status in ("queued", "processing"):
        raise HTTPException(409, "this session is already being analysed")
    row.status = "queued"
    db.add(row)
    db.commit()
    db.refresh(row)
    background.add_task(run_session_job, session_id)
    return _out(row)


@app.post("/api/sessions/{session_id}/rerun/{stage}")
def rerun_stage(session_id: str, stage: str, background: BackgroundTasks, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> dict[str, str]:
    _session_or_404(db, session_id, user)
    if stage not in REGISTRY or stage == "ingest":
        raise HTTPException(400, f"stage must be one of {sorted(set(REGISTRY) - {'ingest'})}")
    background.add_task(rerun, session_id, stage)
    return {"status": "queued", "stage": stage}


@app.patch("/api/sessions/{session_id}", response_model=SessionOut)
def patch_session(session_id: str, body: SessionPatch, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> SessionOut:
    row = _session_or_404(db, session_id, user)
    if body.title is not None:
        row.title = body.title
    if body.keep_video is not None:
        row.keep_video = body.keep_video
    db.add(row)
    db.commit()
    db.refresh(row)
    return _out(row)


@app.patch("/api/sessions/{session_id}/drills/{drill_id}")
def mark_drill(session_id: str, drill_id: str, body: DrillPatch, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> dict[str, Any]:
    _session_or_404(db, session_id, user)
    item = db.exec(select(DrillProgress).where(DrillProgress.session_id == session_id,
                                               DrillProgress.drill_id == drill_id)).first()
    item = item or DrillProgress(session_id=session_id, drill_id=drill_id)
    item.done = body.done
    item.done_at = utcnow() if body.done else None
    db.add(item)
    db.commit()
    return {"drill_id": drill_id, "done": item.done}


@app.delete("/api/sessions/{session_id}", status_code=204, response_model=None)
def delete_session(session_id: str, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> None:
    """FR10: remove the session, its report and the raw video."""
    row = _session_or_404(db, session_id, user)
    for d in db.exec(select(DrillProgress).where(DrillProgress.session_id == session_id)).all():
        db.delete(d)
    db.delete(row)
    db.commit()
    delete_session_files(session_id)


@app.delete("/api/sessions/{session_id}/video", response_model=SessionOut)
def delete_video(session_id: str, db: Session = Depends(get_db),
        user: User = Depends(current_user)) -> SessionOut:
    row = _session_or_404(db, session_id, user)
    delete_video_files(session_id)
    row.video_deleted = True
    db.add(row)
    db.commit()
    db.refresh(row)
    return _out(row)


@app.get("/api/me/progress")
def user_progress(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    """Trend data (FR9): traits and key metrics per finished session, oldest first."""
    rows = db.exec(select(SpeechSession).where(SpeechSession.user_id == user.id, SpeechSession.status == "done")
                   .order_by(SpeechSession.created_at)).all()
    points = []
    for r in rows:
        report = _load_report(r.id) or {}
        points.append({"session_id": r.id, "title": r.title, "created_at": _iso(r.created_at),
                       "confidence": r.confidence, "authoritativeness": r.authoritativeness,
                       "engagement": r.engagement, "congruence": r.congruence, "persona": r.persona,
                       **{k: _dig(report, path) for k, path in TREND_METRICS.items()}})
    return {"points": points}


@app.get("/api/me/persona")
def user_persona(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    from analytics.persona.infer import persona_trajectory

    rows = db.exec(select(SpeechSession).where(SpeechSession.user_id == user.id, SpeechSession.status == "done")
                   .order_by(SpeechSession.created_at)).all()
    sessions = [{"session_id": r.id, "created_at": _iso(r.created_at), "title": r.title,
                 "features": (_load_report(r.id) or {}).get("persona_features")} for r in rows]
    return persona_trajectory(sessions, str(get_settings().models_dir))
