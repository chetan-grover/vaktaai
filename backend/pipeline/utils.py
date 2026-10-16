"""Small shared helpers: JSON conversion, ffmpeg lookup, device choice, series maths."""
from __future__ import annotations

import math
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .settings import get_settings


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and NaN/inf into plain JSON values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def rnd(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    f = float(value)
    return None if math.isnan(f) or math.isinf(f) else round(f, digits)


@lru_cache
def ffmpeg_exe() -> str:
    """System ffmpeg if on PATH, else the binary bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: Sequence[str]) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-400:]}")


def resolve_device() -> str:
    """'cuda' when requested/available, else 'cpu'. Never fails without torch."""
    wanted = get_settings().device
    if wanted == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@lru_cache
def configure_torch() -> None:
    """Two streams run torch at once (audio V-A, gaze/rPPG); give each half the cores, not all of them."""
    import os

    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def window_starts(duration: float, size: float, hop: float) -> list[float]:
    if duration <= 0:
        return []
    starts = list(np.arange(0.0, max(duration - size, 0.0) + 1e-9, hop))
    return [float(s) for s in starts] or [0.0]


def per_second(t: Iterable[float], values: Iterable[float | None], duration: float,
               reducer=np.nanmedian) -> list[float | None]:
    """Resample an irregular series to one value per second (None where empty)."""
    t_arr = np.asarray(list(t), dtype=float)
    v_arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    out: list[float | None] = []
    for sec in range(int(math.ceil(duration))):
        mask = (t_arr >= sec) & (t_arr < sec + 1) & ~np.isnan(v_arr)
        out.append(rnd(reducer(v_arr[mask])) if mask.any() else None)
    return out


def trapezoid(x: float | None, lo0: float, lo1: float, hi1: float, hi0: float) -> float | None:
    """Goodness in [0,1]: 0 below lo0, ramps to 1 at lo1, flat to hi1, ramps to 0 at hi0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if x <= lo0 or x >= hi0:
        return 0.0
    if x < lo1:
        return (x - lo0) / (lo1 - lo0)
    if x > hi1:
        return (hi0 - x) / (hi0 - hi1)
    return 1.0


def nanmean(values: Iterable[float | None]) -> float | None:
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def nanstd(values: Iterable[float | None]) -> float | None:
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.std()) if arr.size > 1 else None
