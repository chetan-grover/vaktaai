# VaktaAI – Project Blueprint & Build Plan
Oct 15, 2026 · @Chetan

## 1. Final scope: 5 enhancements in, 2 as stretch, 2 dropped
Five of the nine suggested enhancements are in (the mail asks for at least four), so one can slip without breaking
the requirement. Selection rule: a pretrained or openly downloadable model must exist, the dataset must be accessible
without a long licence wait, and training must fit on a free Colab/Kaggle GPU.

| # | Enhancement (teacher's list) | Decision | How it is done |
|---|---|---|---|
| E1 | #6 rPPG heart rate + HRV from webcam (PhysNet / DeepPhys, UBFC-rPPG / PURE) | Core | Pretrained PhysNet weights from rPPG-Toolbox on MediaPipe face crops; BPM + HRV (RMSSD) per 10 s window. Fine-tune optional. |
| E2 | #2 Spatio-temporal gaze vs virtual audience grid (MPIIGaze / GazeCapture) | Core | Pretrained L2CS-Net yaw/pitch per frame → 3x3 virtual audience grid; fixation coverage, dwell time, neglected zones; small GRU smoother trained on own labelled clips. |
| E3 | #3 Cross-attention fusion into valence-arousal space (Wav2Vec 2.0 + visual) | Core | Wav2Vec2 dimensional-emotion model (audio V-A) + AffectNet-trained facial V-A; small cross-attention head trained on RAVDESS / CREMA-D. Congruence = audio–face V-A distance per segment. Replaces the rule-based mismatch. |
| E4 | #4 LLM rhetoric analysis (anaphora, ethos/pathos/logos, transitions) | Core | QLoRA fine-tune of Llama-3.2-3B (or Mistral-7B) in 4-bit on ~300-500 TED segments labelled with a JSON rubric. Fallback: same model, few-shot prompting. |
| E5 | #9 VAE persona clustering over session history | Core | Small VAE on per-session feature vectors (own sessions + public speech clips); cluster latent space into personas ("Fast Talker", "Monotone Presenter"); plot each user's trajectory. |
| S1 | #7 Reactive virtual audience avatar (SadTalker / LivePortrait) | Stretch | Pre-render 5-6 reaction clips offline with LivePortrait; switch clips from live energy/gaze. No real-time generation. |
| S2 | #8 Code-switching / Hinglish robustness (FLEURS / SEAME) | Stretch | Whisper large-v3 + Hindi filler lexicon ("matlab", "toh", "basically", "na"); WER on FLEURS Hindi + own Hinglish clips. No fine-tuning. |
| X1 | #1 ViViT micro-expressions (CASME II / SAMM) | Dropped | Signed licence requests; ~250 clips at 200 fps – a 30 fps webcam cannot capture the signal. High risk, low demo value. |
| X2 | #5 CLIP-style audio-video contrastive model (VoxCeleb) | Dropped | Needs multi-GPU days. E3 already gives a learned, quantitative incongruence score. |

In the report, phrase drops as scope decisions with reasons. E3's congruence distance is the defensible version of #5's "Incongruence Distance Score".

## 2. Functionality list (21 core + 2 stretch)
| # | Functionality | Status | Module |
|---|---|---|---|
| F1 | Browser video + audio recording / file upload | Original | frontend |
| F2 | Audio extraction + preprocessing (ffmpeg, 16 kHz mono) | Original | pipeline/ingest |
| F3 | Transcription with word timestamps (faster-whisper) | Original | pipeline/asr |
| F4 | Voice features: pitch variance, pace (WPM), pauses, energy | Original | pipeline/voice |
| F5 | Filler-word detection + timestamps | Original | pipeline/voice |
| F6 | Facial expression variability | Upgraded by E3 (frame-level V-A) | pipeline/face (+ affect) |
| F7 | Eye contact | Upgraded by E2 (audience-grid coverage) | pipeline/gaze |
| F8 | Blink rate (EAR from Face Mesh) | Original | pipeline/face |
| F9 | Content structure | Upgraded by E4 | pipeline/content |
| F10 | Multimodal fusion engine | Upgraded by E3 (rule layer kept as baseline) | pipeline/fusion |
| F11 | Trait scores: confidence, authoritativeness, engagement | Original | pipeline/fusion |
| F12 | Cross-modal mismatch detection (novel) | Driven by E3 congruence distance | pipeline/fusion |
| F13 | Segment-level confidence-dip timeline (novel) | Original | pipeline/timeline |
| F14 | Trait-to-drill mapping engine (novel) | Original; also triggered by E1, E2, E4 | pipeline/drills |
| F15 | Feedback report dashboard | Original | frontend |
| F16 | Session history + progress tracking | Original | backend/db |
| F17 | E1 heart rate + HRV and stress curve on the timeline | New | pipeline/physio |
| F18 | E2 virtual-audience gaze heatmap, neglected-zone alerts | New | pipeline/gaze |
| F19 | E3 V-A trajectory + emotional-congruence score | New | pipeline/fusion |
| F20 | E4 rhetoric report: devices, ethos/pathos/logos, transitions, rewrites | New | pipeline/content |
| F21 | E5 persona map and progress arc | New | analytics/persona |
| S1 | Reactive virtual audience during practice | Stretch | frontend/practice |
| S2 | Hinglish filler detection + code-switch robustness | Stretch | pipeline/voice |

## 3. Requirement analysis
One 1-5 minute recording → full report in < 3 min on a single GPU (< 8 min CPU), stored for comparison.

Users: student practising (primary); mentor reviewing reports (secondary).
UC1 record/upload → report. UC2 inspect timeline, jump to dip/mismatch/HR spike. UC3 open drills, mark done.
UC4 compare sessions, persona arc. UC5 (stretch) live practice with reactive audience.

| ID | Functional requirement |
|---|---|
| FR1 | Webcam (WebM) or upload (MP4/WebM/MOV), 30 s – 5 min, ≤ 200 MB |
| FR2 | Asynchronous analysis with per-stage progress |
| FR3 | Word-level transcript, filler list, voice metrics with timestamps |
| FR4 | Per-second face metrics: valence, arousal, blinks, gaze cell, BPM |
| FR5 | Segments (5 s windows + sentence boundaries), trait scores per segment and overall |
| FR6 | Flag incongruent segments where audio–face V-A distance exceeds a calibrated threshold |
| FR7 | Rhetoric report in a fixed JSON schema |
| FR8 | Each weakness → 1-3 drills with the evidence segment |
| FR9 | Sessions per user; trend charts and persona position |
| FR10 | Delete any session incl. raw video |

| ID | Non-functional | Target |
|---|---|---|
| NFR1 | Latency | 3-min speech ≤ 3 min on a T4-class GPU |
| NFR2 | Accuracy (reported) | WER ≤ 12 %; rPPG MAE ≤ 6 BPM on UBFC subset; gaze-cell acc ≥ 75 % |
| NFR3 | Privacy | Local processing; raw video auto-deleted after 7 days unless kept; no third-party API gets video |
| NFR4 | Reproducibility | Pinned requirements, seeds, model versions logged per report |
| NFR5 | Modularity | Common analyser interface; any one can fail without killing the report |
| NFR6 | Usability | One screen + timeline; every score has a one-line "why" |
| NFR7 | Ethics | rPPG/emotion labelled "indicative, not medical or lie detection" |

Constraints: solo developer; laptop + free Colab/Kaggle GPU; single frontal speaker, lit room, 720p/30 fps; English or Hinglish.

## 4. Architecture and stack
Upload → ingest → three parallel streams (transcript: E4 + structure, waits on ASR; audio: Whisper, voice/fillers,
E3 audio V-A; video: Face Mesh + blink, E3 face V-A, E2 gaze grid, E1 rPPG) → fusion engine (rule baseline + E3
head) → timeline, drills, session store → report dashboard; E5 persona VAE over the session store.

| Layer | Choice |
|---|---|
| Frontend | React + Vite + TypeScript, Tailwind, Recharts, MediaRecorder API |
| API | FastAPI + Pydantic |
| Jobs | FastAPI BackgroundTasks first; RQ + Redis only if needed |
| Storage | SQLite (SQLModel) + files on disk per session |
| ML runtime | PyTorch, HF Transformers, PEFT/bitsandbytes, MediaPipe, faster-whisper |
| LLM serving | HF pipeline or Ollama with the merged adapter |
| Training | Colab / Kaggle notebooks in training/ |

## 5. Models, datasets, what gets trained
Only three things are trained (E3 head, E4 QLoRA adapter, E5 VAE) plus the small E2 GRU; everything else is pretrained inference.

| Module | Model | Data | Train/infer |
|---|---|---|---|
| ASR | faster-whisper small / large-v3 | – | infer |
| Voice | Parselmouth, librosa | – | rules |
| Face | MediaPipe Face Mesh | – | infer |
| E1 | PhysNet (rPPG-Toolbox) | UBFC-rPPG for eval | infer + evaluate |
| E2 | L2CS-Net + GRU smoother | own 20-30 clips, 9 targets | infer + train smoother |
| E3 audio | wav2vec2 dimensional (MSP-Podcast) | – | infer |
| E3 face | HSEmotion / EmoNet (AffectNet) | – | infer |
| E3 fusion | 2-layer cross-attention head (~2 M params) | RAVDESS + CREMA-D | train (Colab T4, 1-2 h) |
| E4 | Llama-3.2-3B-Instruct 4-bit QLoRA | 300-500 TED segments, silver labels, 100 hand-checked | fine-tune (2-3 h) |
| E5 | VAE (~24 features, latent 2-4) + GMM | own sessions + ~200 public clips | train (CPU) |
| Drills | YAML rules (+ optional LLM wording) | ~30 hand-written drills | rules |

Until the E3 head is trained, the rule-based fusion runs as the baseline; keep it – "learned vs rule-based" is the evaluation table.

## 6. Repo structure
See README.md. Each analyser writes `data/sessions/<id>/<module>.json`; fusion and timeline read only those files.

## 7. Timeline (Oct 16 – Dec 3, 2026)
Setup Oct 16-19 · core F1-F9 Oct 19-28 · baseline fusion + UI Oct 26-Nov 4 · **baseline demo Nov 4** ·
E1+E2 Nov 2-11 · E3 training Nov 6-16 · E4 QLoRA Nov 9-18 · E5 Nov 16-22 · evaluation + user study Nov 20-28 ·
report/demo prep Nov 26-Dec 3 · **final demo Dec 3**.

## 8. Building with Claude Code
Plan first, one module per session, tests first, review diffs, commit after each green test, clear context between modules.
Claude Code: scaffolding, glue, tests, UI, notebook boilerplate. You: thresholds, labelling (gaze clips, 100 rhetoric
samples), Colab training, sanity checks, report. Skip: auth beyond simple login, Docker/K8s, cloud, mobile.

## 9. Evaluation, risks, viva demo
| Component | Metric | Baseline |
|---|---|---|
| ASR + fillers | WER, filler P/R on 10 own clips | Whisper small vs large-v3 |
| E1 rPPG | MAE / RMSE BPM on UBFC-rPPG | POS / CHROM |
| E2 gaze | 9-cell accuracy on own clips | raw L2CS without GRU; Face-Mesh head pose only |
| E3 fusion | emotion acc / CCC on RAVDESS held-out actors; mismatch detection on 20 own acted clips | rule-based fusion |
| E4 rhetoric | agreement with 100 hand labels (F1 per device, MAE balance) | same base model, prompting only |
| E5 personas | silhouette; qualitative sanity check | k-means on raw features |
| Whole system | before/after drills study, 5-8 friends, 2 sessions each | descriptive |

| Risk | Fallback |
|---|---|
| rPPG noisy (motion / light) | report only high signal-quality windows; mark others unreliable |
| Colab quota | Kaggle (30 GPU-h/week); smaller models |
| E3 doesn't transfer from acted emotion | keep rule baseline live; present comparison honestly |
| LLM breaks JSON | pydantic validation + retry + prompting fallback |
| Time overrun | drop S1/S2 first, then E5 training (use k-means), never E1-E4 |
| Ethics questions | NFR7 disclaimer; no lie-detection claims; consent for participants |

Viva demo (7 min): record 60 s live, rushed, looking at one side → report (dip, neglected cells, HR rise, mismatch,
rhetoric) → drills → pre-recorded "after drills" session (trait deltas, persona arc) → evaluation table. Keep a recorded backup.
