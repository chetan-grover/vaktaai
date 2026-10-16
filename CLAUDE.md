# VaktaAI
Multimodal speech coach. Spec: docs/blueprint.md (read sections 2-6 before coding).

## Stack
Backend: Python 3.11+ (3.12 in .venv), FastAPI, SQLModel, pytest. Frontend: React + Vite + TS + Tailwind + Recharts.

## Layout
- backend/pipeline/: one Analyzer per module; runner.py orders the 3 streams; report.py assembles report.json.
- backend/pipeline/config/: traits.yaml (rule-fusion weights + bands), drills.yaml (weakness rules + drill library).
- backend/pipeline/nets.py: torch modules shared by training/ and inference (PhysNet, gaze GRU, E3 fusion head).
- analytics/persona/: E5 features, VAE, training, inference (with rule fallback).
- training/: Colab notebooks (e3, e4, e5) + e2_gaze_smoother.py. eval/: scripts for every number in section 9.

## Rules
- Every analyser subclasses pipeline/base.py Analyzer and writes data/sessions/<id>/<name>.json.
- Never load a model at import time; lazy-load and cache (functools.lru_cache in the module).
- Every learned component has a fallback (POS, head-pose gaze, landmark/prosody V-A, rule fusion,
  heuristic rhetoric, rule personas). A missing model must degrade the module, never fail the report.
- Every new module ships with a pytest. Tests must not download weights (conftest sets VAKTA_ALLOW_DOWNLOADS=0).
- Do not download datasets or large weights without asking.
- GPU code must fall back to CPU (pipeline.utils.resolve_device).
- Keep functions small; type hints everywhere.
- rPPG and emotion outputs are "indicative, not medical or lie detection" (NFR7) - keep the disclaimer in report.py.
- Every API route except /api/health and /api/auth/* takes `user: User = Depends(current_user)` and looks sessions up
  with `_session_or_404(db, id, user)` (ownership check). The video route also accepts `?token=` for <video src>.
- Jobs run one at a time (worker/jobs.py lock); torch modules call `configure_torch()` to share CPU cores.

## Commands
- Setup: python -m venv .venv; .venv/Scripts/pip install -r backend/requirements.txt (+ requirements-ml.txt for torch models)
- Run API: uvicorn app.main:app --reload (from backend/)
- Tests: pytest -q (from repo root)
- Frontend: npm run dev (from frontend/); API proxied from /api to :8000
- Models: python scripts/download_models.py --list
