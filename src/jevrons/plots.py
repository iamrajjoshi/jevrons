"""Figures for the write-up. Usage: uv run python -m jevrons.plots stage3"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures"


def stage3():
    from jevrons.stage3 import GRID, XTE, YTE

    runs = [("ste_jev", 0, "Trained through Jev"), ("swap", 0, "Trained exact, run on Jev")]
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.6), constrained_layout=True)
    for ax, (arm, seed, title) in zip(axes, runs):
        r = json.loads((ROOT / f"runs/stage3/result-{arm}-{seed}.json").read_text())
        p = np.array(r["grid_p"]).reshape(len(GRID), len(GRID))
        im = ax.imshow(p, origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r", vmin=0, vmax=1)
        ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, ls="--", lw=1.5, color="k"))
        inside = YTE.ravel() > 0.5
        ax.scatter(*XTE[inside].T, s=6, c="k", marker="o")
        ax.scatter(*XTE[~inside].T, s=6, c="k", marker="x", lw=0.6)
        ax.set_title(f"{title}\nheld-out accuracy {r['test_accuracy']:.0%}")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Jev output p (inside)")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage3-boundary.png", dpi=160)
    print(FIG / "stage3-boundary.png")


if __name__ == "__main__":
    {"stage3": stage3}[sys.argv[1]]()
