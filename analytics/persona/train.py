"""E5 training: VAE on session vectors -> GMM in latent space -> named personas.

    python -m analytics.persona.train --csv data/persona/public_features.csv --include-sessions

Inputs: CSVs with the FEATURE_NAMES columns (eval/extract_features.py writes them from public
speech clips) plus, optionally, every analysed session's report.json.
Outputs (models/persona/): vae.pt, gmm.joblib, meta.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from analytics.persona.features import FEATURE_NAMES, NORMS, session_vector, to_array  # noqa: E402
from analytics.persona.naming import name_profile  # noqa: E402


def read_csv(path: Path) -> list[dict[str, float | None]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({n: float(row[n]) if row.get(n) not in (None, "", "nan") else None for n in FEATURE_NAMES})
    return rows


def session_vectors(sessions_dir: Path) -> list[dict[str, float | None]]:
    out = []
    for report_path in sorted(sessions_dir.glob("*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        out.append(report.get("persona_features") or session_vector(report))
    return out


def fit_norms(vectors: list[dict[str, float | None]]) -> dict[str, tuple[float, float]]:
    """Dataset mean/std per feature, falling back to the prior where data is thin.
    The std is floored at a quarter of the prior so near-constant features are not blown up into noise."""
    norms = {}
    for name in FEATURE_NAMES:
        vals = np.array([v[name] for v in vectors if v.get(name) is not None], dtype=float)
        if vals.size >= 10:
            norms[name] = (float(vals.mean()), max(float(vals.std()), 0.25 * NORMS[name][1]))
        else:
            norms[name] = NORMS[name]
    return norms


def train_vae(x: np.ndarray, latent: int = 2, epochs: int = 600, beta: float = 0.5, seed: int = 13):
    import torch

    from analytics.persona.model import SessionVAE, vae_loss

    torch.manual_seed(seed)
    model = SessionVAE(in_dim=x.shape[1], latent=latent)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    data = torch.tensor(x, dtype=torch.float32)
    batch = min(64, len(data))
    for _ in range(epochs):
        perm = torch.randperm(len(data))
        model.train()
        for i in range(0, len(data), batch):
            xb = data[perm[i:i + batch]]
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, xb, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(data)
    return model, mu.numpy()


def fit_gmm(z: np.ndarray, k: int | None, seed: int):
    from sklearn.mixture import GaussianMixture

    candidates = [k] if k else list(range(3, max(4, min(7, len(z) // 5) + 1)))
    best = None
    for n in candidates:
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=seed, reg_covar=1e-3).fit(z)
        if best is None or gmm.bic(z) < best.bic(z):
            best = gmm
    return best


def name_components(x: np.ndarray, labels: np.ndarray, z: np.ndarray) -> list[dict[str, Any]]:
    personas, taken = [], set()
    order = sorted(set(labels.tolist()), key=lambda c: -(labels == c).sum())
    for comp in order:
        members = x[labels == comp]
        profile = dict(zip(FEATURE_NAMES, members.mean(axis=0).tolist()))
        name, desc = name_profile(profile, taken)
        if name in taken:
            name = f"{name} {len(taken) + 1}"
        taken.add(name)
        personas.append({"component": int(comp), "name": name, "description": desc,
                         "size": int((labels == comp).sum()),
                         "centroid": z[labels == comp].mean(axis=0).round(4).tolist()})
    return sorted(personas, key=lambda p: p["component"])


def train(vectors: list[dict[str, float | None]], out_dir: Path, latent: int = 2, k: int | None = None,
          epochs: int = 600, seed: int = 13) -> dict[str, Any]:
    import joblib
    import torch

    if len(vectors) < 15:
        raise ValueError(f"need at least 15 sessions/clips to train personas, got {len(vectors)}")
    norms = fit_norms(vectors)
    x = np.stack([to_array(v, norms) for v in vectors])
    model, z = train_vae(x, latent=latent, epochs=epochs, seed=seed)
    gmm = fit_gmm(z, k, seed)
    labels = gmm.predict(z)
    personas = name_components(x, labels, z)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, out_dir / "vae.pt")
    joblib.dump(gmm, out_dir / "gmm.joblib")
    meta = {"kind": "vae+gmm", "latent": latent, "n_train": len(vectors), "norms": norms, "personas": personas}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"meta": meta, "x": x, "z": z, "labels": labels}


def main() -> None:
    from pipeline.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, action="append", default=[])
    parser.add_argument("--include-sessions", action="store_true")
    parser.add_argument("--latent", type=int, default=2)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()
    settings = get_settings()
    vectors = [row for path in args.csv for row in read_csv(path)]
    if args.include_sessions:
        vectors += session_vectors(settings.sessions_dir)
    result = train(vectors, settings.models_dir / "persona", args.latent, args.k, args.epochs, settings.seed)
    for p in result["meta"]["personas"]:
        print(f"{p['component']}: {p['name']:<24} n={p['size']:<4} {p['description']}")


if __name__ == "__main__":
    main()
