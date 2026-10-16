"""F2 ingest: normalise the upload to 30 fps H.264 + 16 kHz mono WAV, and sample frames."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .base import Analyzer, SessionContext
from .settings import get_settings
from .utils import rnd, run_ffmpeg

TARGET_FPS = 30
MAX_WIDTH = 640
SAMPLE_RATE = 16_000


def normalise_video(src: Path, dst: Path) -> None:
    run_ffmpeg([
        "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"fps={TARGET_FPS},scale='min({MAX_WIDTH},iw)':-2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst),
    ])


def extract_audio(src: Path, dst: Path) -> bool:
    try:
        run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)])
    except RuntimeError:
        return False
    return dst.exists() and dst.stat().st_size > 1000


def probe_video(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {"fps": float(fps), "n_frames": frames, "width": width, "height": height,
            "duration_s": frames / fps if fps else 0.0}


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, int(sr)


def iter_frames(video_path: Path, fps: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, t_seconds, BGR frame), optionally subsampled to `fps`."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    step = 1.0 / fps if fps else 0.0
    next_t = 0.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / src_fps
            if not step or t + 1e-6 >= next_t:
                yield idx, t, frame
                next_t += step
            idx += 1
    finally:
        cap.release()


class IngestAnalyzer(Analyzer):
    name = "ingest"

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        settings = get_settings()
        src = ctx.input_path
        normalise_video(src, ctx.video_path)
        meta = probe_video(ctx.video_path)
        has_audio = extract_audio(src, ctx.audio_path)
        audio_duration = 0.0
        if has_audio:
            samples, sr = load_audio(ctx.audio_path)
            audio_duration = len(samples) / sr
        duration = max(meta["duration_s"], audio_duration)
        if duration < settings.min_duration_s:
            raise ValueError(f"recording is {duration:.1f}s; minimum is {settings.min_duration_s:.0f}s")
        if duration > settings.max_duration_s + 5:
            raise ValueError(f"recording is {duration:.1f}s; maximum is {settings.max_duration_s:.0f}s")
        return {
            "source": src.name,
            "duration_s": rnd(duration),
            "video": {k: rnd(v) if isinstance(v, float) else v for k, v in meta.items()},
            "has_audio": has_audio,
            "audio_duration_s": rnd(audio_duration),
            "sample_rate": SAMPLE_RATE,
            "analysis_fps": settings.analysis_fps,
        }
