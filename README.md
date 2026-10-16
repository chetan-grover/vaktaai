# VaktaAI

Multimodal speech coach: record or upload a 30 s–5 min talk and get a report covering voice, face, audience gaze,
webcam heart rate, voice/face emotional congruence, rhetoric, drills, and a persona map across sessions.
Spec: [docs/blueprint.md](docs/blueprint.md).

## Quick start (Windows, from the repo root)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend/requirements.txt
# optional learned models (torch etc.), see the comments in the file for the 3-step order:
.venv\Scripts\pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
.venv\Scripts\pip install -r backend/requirements-ml.txt
.venv\Scripts\pip install --no-deps --no-build-isolation hsemotion-onnx==0.3.1

python scripts/download_models.py --list          # sizes + licences
python scripts/download_models.py --get whisper-small hsemotion physnet audio-va

cd backend;  ..\.venv\Scripts\uvicorn app.main:app --reload     # API on :8000, docs at /docs
cd frontend; npm install; npm run dev                            # UI on http://localhost:5173
.venv\Scripts\pytest -q                                          # tests (no downloads; synthetic fixture video)
```

ffmpeg is taken from PATH, or from the `imageio-ffmpeg` wheel if it isn't installed.

## Accounts

Open the app and choose **Create account** (name, email, password ≥ 8 characters). Passwords are stored as
PBKDF2-SHA256 hashes and sign-in returns a bearer token (30 days). Every session route checks ownership, so users
only ever see their own recordings. A profile from the earlier name-only login is kept: sign up once with the same
name and its sessions move into the new account.

To try changes against a copy of your data, run a second API with `VAKTA_DATA_DIR=<copy> uvicorn app.main:app --port 8001`
and a second UI with `VAKTA_API=http://127.0.0.1:8001 npx vite --port 5174`.

## How it works

`ingest` → three parallel streams → `fusion` → `timeline` → `drills` → `report.json`

| Stream | Modules (backend/pipeline) | Learned model | Fallback when the model is missing |
|---|---|---|---|
| audio | `asr` (F3), `voice` (F4, F5, S2), `affect.audio_va` (E3) | faster-whisper; audeering wav2vec2 V-A | – ; prosody proxy |
| transcript | `content` (F9, E4) | QLoRA Llama-3.2-3B via `VAKTA_LLM_URL` | regex/lexicon rhetoric analyser |
| video | `face` (F6, F8), `gaze` (E2), `affect.face_va` (E3), `physio` (E1) | Face Mesh; L2CS + GRU; HSEmotion; PhysNet | Haar boxes; head-pose+iris + mode filter; landmark proxy; POS |
| fusion | `fusion` (F10-F12, F19), `timeline` (F13), `drills` (F14) | E3 cross-attention head | YAML rule baseline (`config/traits.yaml`) |
| history | `analytics/persona` (E5) | VAE + GMM | rule axes (energy × polish) |

Every analyser writes `data/sessions/<id>/<module>.json` with `status: ok | skipped | error`. A failed
module never stops the report, and the report lists which model or fallback produced each number (NFR4/NFR5).
Proxy V-A values feed only the congruence score, never absolute trait scores.

## Training and evaluation

| What | Where | Output |
|---|---|---|
| E2 GRU smoother | `python training/e2_gaze_smoother.py --clips <9-target clips>` | `models/gaze/gaze_gru.pt` |
| E3 fusion head | `training/e3_fusion.ipynb` (Colab T4) | `models/fusion/e3_head.pt` |
| E4 rhetoric QLoRA | `training/e4_qlora.ipynb` → `training/serve_rhetoric.py` | `models/rhetoric/merged` |
| E5 personas | `eval/extract_features.py` → `python -m analytics.persona.train` or `training/e5_vae.ipynb` | `models/persona/` |

Section 9's numbers come from `eval/` (results land in `eval/results/` as CSV + PNG):
`asr_eval.py`, `rppg_eval.py`, `gaze_eval.py`, `fusion_eval.py`, `rhetoric_eval.py`, `persona_eval.py`,
`user_study.py`, `latency.py`.

## Configuration (environment variables)

`VAKTA_WHISPER_MODEL` (small) · `VAKTA_WHISPER_LANGUAGE` (auto) · `VAKTA_DEVICE` (auto|cpu) ·
`VAKTA_LLM_URL` (unset = heuristic rhetoric) · `VAKTA_ALLOW_DOWNLOADS` (1) · `VAKTA_RETENTION_DAYS` (7) ·
`VAKTA_DATA_DIR`, `VAKTA_MODELS_DIR`, `VAKTA_MIN_DURATION_S` (30), `VAKTA_MAX_DURATION_S` (300).

Heart-rate, stress and emotion outputs are indicative only: they are not medical measurements and not lie detection.
