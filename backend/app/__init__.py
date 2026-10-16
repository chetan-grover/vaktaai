"""FastAPI application. Puts the repo root on sys.path so `analytics.persona` imports
when uvicorn is started from backend/ (as CLAUDE.md says)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
