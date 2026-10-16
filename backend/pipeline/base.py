"""Analyzer interface shared by every pipeline module.

Each analyzer reads the JSON written by earlier modules, computes its own result
and writes data/sessions/<id>/<name>.json as an envelope:

    {"module", "version", "status": ok|skipped|error, "elapsed_s", "data", "error"}

A failing analyzer writes status=error instead of raising, so one broken module
never kills the report (NFR5).
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import get_settings
from .utils import to_jsonable

log = logging.getLogger("vakta.pipeline")

SCHEMA_VERSION = 1
INPUT_PREFIX = "input"
REPLACE_RETRIES = 8


class AnalyzerSkipped(Exception):
    """Raise inside analyze() when the module cannot run on this input (no face, no audio...)."""


@dataclass
class SessionContext:
    session_id: str
    session_dir: Path

    @classmethod
    def for_session(cls, session_id: str) -> "SessionContext":
        session_dir = get_settings().sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return cls(session_id=session_id, session_dir=session_dir)

    @property
    def input_path(self) -> Path:
        matches = sorted(self.session_dir.glob(f"{INPUT_PREFIX}.*"))
        if not matches:
            raise FileNotFoundError(f"no input video in {self.session_dir}")
        return matches[0]

    @property
    def video_path(self) -> Path:
        """Normalised 30 fps H.264 copy written by ingest."""
        return self.session_dir / "video.mp4"

    @property
    def audio_path(self) -> Path:
        """16 kHz mono PCM written by ingest."""
        return self.session_dir / "audio.wav"

    def path(self, filename: str) -> Path:
        return self.session_dir / filename

    def module_path(self, name: str) -> Path:
        return self.session_dir / f"{name}.json"

    def read_envelope(self, name: str) -> dict[str, Any] | None:
        path = self.module_path(name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read(self, name: str) -> dict[str, Any] | None:
        """Data of a module that finished with status ok, else None."""
        env = self.read_envelope(name)
        if env is None or env.get("status") != "ok":
            return None
        return env.get("data")

    def write_json(self, filename: str, payload: Any) -> None:
        """Atomic write (tmp + replace). On Windows the replace fails while another process has the
        target open (the API polling it, OneDrive, antivirus), so retry briefly, then write in place."""
        path = self.session_dir / filename
        text = json.dumps(to_jsonable(payload), ensure_ascii=False)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(REPLACE_RETRIES):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        path.write_text(text, encoding="utf-8")
        tmp.unlink(missing_ok=True)


class Analyzer(ABC):
    name: str = ""
    version: str = "1"
    requires: tuple[str, ...] = ()

    @abstractmethod
    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        """Return the module's data dict. Raise AnalyzerSkipped when not applicable."""

    def need(self, ctx: SessionContext, name: str) -> dict[str, Any]:
        data = ctx.read(name)
        if data is None:
            raise AnalyzerSkipped(f"{self.name} needs '{name}', which did not complete")
        return data

    def run(self, ctx: SessionContext) -> dict[str, Any]:
        start = time.perf_counter()
        envelope: dict[str, Any] = {"module": self.name, "version": self.version, "schema": SCHEMA_VERSION}
        try:
            data = self.analyze(ctx)
            envelope.update(status="ok", data=data, error=None)
        except AnalyzerSkipped as exc:
            envelope.update(status="skipped", data=None, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - modules must not crash the job
            log.exception("analyzer %s failed", self.name)
            envelope.update(status="error", data=None, error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc(limit=6))
        envelope["elapsed_s"] = round(time.perf_counter() - start, 3)
        ctx.write_json(f"{self.name}.json", envelope)
        return envelope
