# models/ (git-ignored except this file)

| Path | What | How it gets here |
|---|---|---|
| `whisper/` | faster-whisper weights | `python scripts/download_models.py --get whisper-small` (or first analysis, if downloads allowed) |
| `audio_va/` | audeering wav2vec2 V-A model (E3 audio) | `--get audio-va` |
| `rppg/PURE_PhysNet_DiffNormalized.pth` | PhysNet (E1) | `--get physnet` |
| `gaze/l2cs_gaze360_resnet50.safetensors` | L2CS-Net Gaze360 (E2), py-feat re-pack | `--get l2cs` (the original `.pkl` also works) |
| `gaze/gaze_gru.pt` | GRU smoother (E2) | `python training/e2_gaze_smoother.py --clips ...` |
| `fusion/e3_head.pt` | cross-attention head (E3) | `training/e3_fusion.ipynb` on Colab |
| `rhetoric/merged/` | QLoRA-tuned Llama-3.2-3B (E4) | `training/e4_qlora.ipynb`, served by `training/serve_rhetoric.py` |
| `persona/` | VAE + GMM (E5) | `python -m analytics.persona.train ...` or `training/e5_vae.ipynb` |

Every module runs without its learned weights using a fallback (POS rPPG, Face Mesh gaze,
landmark/prosody V-A, rule fusion, heuristic rhetoric, rule personas); the report shows which ran.
