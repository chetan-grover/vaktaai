"""E5: silhouette of VAE + GMM personas vs k-means on raw features (same k), plus a latent-space plot.

    python eval/persona_eval.py --csv data/persona/public_features.csv --include-sessions
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from common import save_plot, write_csv

from analytics.persona.features import to_array
from analytics.persona.infer import kmeans_raw
from analytics.persona.train import fit_norms, read_csv, session_vectors, train
from pipeline.settings import get_settings


def main() -> None:
    from sklearn.metrics import silhouette_score

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, action="append", default=[])
    ap.add_argument("--include-sessions", action="store_true")
    args = ap.parse_args()
    vectors = [row for p in args.csv for row in read_csv(p)]
    if args.include_sessions:
        vectors += session_vectors(get_settings().sessions_dir)
    with tempfile.TemporaryDirectory() as tmp:
        result = train(vectors, Path(tmp), latent=2, seed=get_settings().seed)
    z, labels = result["z"], result["labels"]
    k = len(set(labels.tolist()))
    x = np.stack([to_array(v, fit_norms(vectors)) for v in vectors])
    km = kmeans_raw(vectors, k)
    write_csv("e5_persona.csv", [
        {"system": "VAE + GMM", "k": k, "silhouette_own_space": silhouette_score(z, labels),
         "silhouette_raw_space": silhouette_score(x, labels)},
        {"system": "k-means on raw features", "k": k, "silhouette_own_space": silhouette_score(x, km),
         "silhouette_raw_space": silhouette_score(x, km)},
    ])
    import matplotlib.pyplot as plt

    names = {p["component"]: p["name"] for p in result["meta"]["personas"]}
    fig, ax = plt.subplots(figsize=(6, 5))
    for comp in sorted(set(labels.tolist())):
        m = labels == comp
        ax.scatter(z[m, 0], z[m, 1], s=12, label=names.get(comp, comp))
    ax.legend(fontsize=8)
    ax.set_title("VAE latent space, GMM personas")
    save_plot(fig, "e5_persona_map.png")


if __name__ == "__main__":
    main()
