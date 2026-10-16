"""Pipeline orchestration: ingest, then three parallel streams, then fusion -> timeline -> drills -> report.

    audio stream:      asr -> voice -> audio_va        (content starts as soon as asr is done)
    transcript stream: content (E4, waits on asr)
    video stream:      face -> gaze -> face_va -> physio
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .affect import AudioAffectAnalyzer, FaceAffectAnalyzer
from .asr import AsrAnalyzer
from .base import Analyzer, SessionContext
from .content import ContentAnalyzer
from .drills import DrillsAnalyzer
from .face import FaceAnalyzer
from .fusion import FusionAnalyzer
from .gaze import GazeAnalyzer
from .ingest import IngestAnalyzer
from .physio import PhysioAnalyzer
from .report import build_report
from .timeline import TimelineAnalyzer
from .voice import VoiceAnalyzer

log = logging.getLogger("vakta.pipeline")
ProgressFn = Callable[[dict[str, Any]], None]

AUDIO_STREAM: list[type[Analyzer]] = [AsrAnalyzer, VoiceAnalyzer, AudioAffectAnalyzer]
VIDEO_STREAM: list[type[Analyzer]] = [FaceAnalyzer, GazeAnalyzer, FaceAffectAnalyzer, PhysioAnalyzer]
TRANSCRIPT_STREAM: list[type[Analyzer]] = [ContentAnalyzer]
FUSION_STAGES: list[type[Analyzer]] = [FusionAnalyzer, TimelineAnalyzer, DrillsAnalyzer]
ALL: list[type[Analyzer]] = [IngestAnalyzer, *AUDIO_STREAM, *TRANSCRIPT_STREAM, *VIDEO_STREAM, *FUSION_STAGES]
REGISTRY: dict[str, type[Analyzer]] = {cls.name: cls for cls in ALL}
STAGE_NAMES = [cls.name for cls in ALL] + ["report"]


class SessionCancelled(RuntimeError):
    """The session was deleted while it was being analysed."""


class Progress:
    """Thread-safe per-stage status, mirrored to progress.json and an optional callback (FR2)."""

    def __init__(self, ctx: SessionContext, on_change: ProgressFn | None = None) -> None:
        self.ctx = ctx
        self.on_change = on_change
        self.lock = threading.Lock()
        self.stages: dict[str, dict[str, Any]] = {n: {"status": "pending"} for n in STAGE_NAMES}

    def set(self, stage: str, status: str, **extra: Any) -> None:
        with self.lock:
            if not self.ctx.session_dir.exists():
                raise SessionCancelled(self.ctx.session_id)
            self.stages[stage] = {"status": status, **extra}
            snapshot = {"stages": dict(self.stages), "done": sum(s["status"] not in ("pending", "running")
                                                                 for s in self.stages.values()),
                        "total": len(self.stages)}
            try:
                self.ctx.write_json("progress.json", snapshot)
            except OSError as exc:
                log.warning("could not write progress for %s: %s", self.ctx.session_id, exc)
        if self.on_change:
            self.on_change(snapshot)


def run_stage(cls: type[Analyzer], ctx: SessionContext, progress: Progress | None = None) -> dict[str, Any]:
    if progress:
        progress.set(cls.name, "running")
    env = cls().run(ctx)
    if progress:
        progress.set(cls.name, env["status"], elapsed_s=env["elapsed_s"], error=env.get("error"))
    return env


def run_pipeline(session_id: str, on_progress: ProgressFn | None = None) -> dict[str, Any]:
    ctx = SessionContext.for_session(session_id)
    progress = Progress(ctx, on_progress)
    ingest = run_stage(IngestAnalyzer, ctx, progress)
    if ingest["status"] != "ok":
        for name in STAGE_NAMES[1:]:
            progress.set(name, "skipped", error="ingest failed")
        raise RuntimeError(ingest.get("error") or "ingest failed")

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="vakta") as pool:
        asr_done = threading.Event()

        def audio() -> None:
            try:
                run_stage(AsrAnalyzer, ctx, progress)
            finally:
                asr_done.set()
            for cls in AUDIO_STREAM[1:]:
                run_stage(cls, ctx, progress)

        def transcript() -> None:
            asr_done.wait()
            for cls in TRANSCRIPT_STREAM:
                run_stage(cls, ctx, progress)

        def video() -> None:
            for cls in VIDEO_STREAM:
                run_stage(cls, ctx, progress)

        for fut in [pool.submit(audio), pool.submit(transcript), pool.submit(video)]:
            fut.result()

    for cls in FUSION_STAGES:
        run_stage(cls, ctx, progress)
    progress.set("report", "running")
    report = build_report(ctx)
    progress.set("report", "ok")
    return report


def rerun(session_id: str, stage: str) -> dict[str, Any]:
    """Re-run one module from its saved inputs, then refresh fusion, timeline, drills and the report."""
    if stage not in REGISTRY:
        raise KeyError(f"unknown stage '{stage}'")
    ctx = SessionContext.for_session(session_id)
    REGISTRY[stage]().run(ctx)
    for cls in FUSION_STAGES:
        if cls.name != stage:
            cls().run(ctx)
    return build_report(ctx)


def analyse_file(video: Path, stages: list[str], session_id: str | None = None) -> SessionContext:
    """Run `stages` (in order, ingest implied) on any video file; used by training/ and eval/.
    Files land in data/sessions/<session_id>; existing ok outputs are reused."""
    session_id = session_id or f"eval-{video.stem}-{hashlib.md5(str(video).encode()).hexdigest()[:6]}"
    ctx = SessionContext.for_session(session_id)
    if not any(ctx.session_dir.glob("input.*")):
        shutil.copy(video, ctx.session_dir / f"input{video.suffix.lower()}")
    for name in ["ingest", *[s for s in stages if s != "ingest"]]:
        env = ctx.read_envelope(name)
        if env is None or env.get("status") != "ok":
            REGISTRY[name]().run(ctx)
    return ctx


def read_progress(ctx: SessionContext) -> dict[str, Any] | None:
    path = ctx.path("progress.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
