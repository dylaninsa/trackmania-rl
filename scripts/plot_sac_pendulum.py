"""Plots the learning curves written by ``validate_sac_pendulum.py``.

Reads every ``results/sac_pendulum_seed*.csv`` and draws the evaluation return, the temperature alpha, the mean
Q value and the policy entropy against the number of steps, one line per seed.

Example::

    uv run python scripts/plot_sac_pendulum.py
    uv run python scripts/plot_sac_pendulum.py --input-dir results --output results/sac_pendulum.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SUCCESS_RETURN = -250.0
RANDOM_RETURN = -1200.0
PANELS = (("eval_return", "Retour d'évaluation (déterministe)"), ("alpha", "Température α"),
          ("q_mean", "Q moyen (min des deux critiques)"), ("entropy", "Entropie de la politique"))


def read_run(path):
    """Reads one CSV file written by ``validate_sac_pendulum.py``.

    Args:
        path (pathlib.Path): CSV file.

    Returns:
        dict[str, np.ndarray]: One float array per column.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def main():
    """Draws the figure and saves it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="results")
    parser.add_argument("--output", default="results/sac_pendulum.png")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob("sac_pendulum_seed*.csv"))
    if not files:
        raise SystemExit(f"Aucun fichier sac_pendulum_seed*.csv dans {args.input_dir}")
    runs = {f.stem.removeprefix("sac_pendulum_"): read_run(f) for f in files}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        for name, run in runs.items():
            ax.plot(run["step"], run[key], marker="o", markersize=3, alpha=0.8, label=name)
        if key == "eval_return":
            ax.axhline(SUCCESS_RETURN, color="green", linestyle="--", label="seuil de réussite")
            ax.axhline(RANDOM_RETURN, color="gray", linestyle=":", label="aléatoire (≈)")
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("pas d'environnement")
    fig.suptitle("SAC sur Pendulum-v1")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Figure enregistrée dans {output}")


if __name__ == "__main__":
    main()