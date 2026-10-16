"""Runtime configuration, read from VAKTA_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VAKTA_{name}", default)


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(f"VAKTA_{name}")
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    models_dir: Path
    db_url: str
    whisper_model: str
    whisper_language: str | None
    device: str
    analysis_fps: float
    segment_seconds: float
    min_duration_s: float
    max_duration_s: float
    max_upload_mb: int
    retention_days: int
    allow_downloads: bool
    llm_url: str | None
    llm_model: str
    seed: int

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


@lru_cache
def get_settings() -> Settings:
    data_dir = Path(_env("DATA_DIR", str(ROOT / "data")))
    return Settings(
        data_dir=data_dir,
        models_dir=Path(_env("MODELS_DIR", str(ROOT / "models"))),
        db_url=_env("DB_URL", f"sqlite:///{(data_dir / 'vakta.db').as_posix()}"),
        whisper_model=_env("WHISPER_MODEL", "small"),
        whisper_language=os.environ.get("VAKTA_WHISPER_LANGUAGE") or None,
        device=_env("DEVICE", "auto"),
        analysis_fps=float(_env("ANALYSIS_FPS", "10")),
        segment_seconds=float(_env("SEGMENT_SECONDS", "5")),
        min_duration_s=float(_env("MIN_DURATION_S", "30")),
        max_duration_s=float(_env("MAX_DURATION_S", "300")),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", "200")),
        retention_days=int(_env("RETENTION_DAYS", "7")),
        allow_downloads=_flag("ALLOW_DOWNLOADS", True),
        llm_url=os.environ.get("VAKTA_LLM_URL") or None,
        llm_model=_env("LLM_MODEL", "vakta-rhetoric"),
        seed=int(_env("SEED", "13")),
    )
