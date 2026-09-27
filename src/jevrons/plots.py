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


def stage5():
    pairs, arms = ("0v1", "3v8"), (("jev", "Trained through Jev"), ("swap", "Trained exact, swapped to Jev"))
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    width = 0.2
    for k, pair in enumerate(pairs):
        for a, (arm, label) in enumerate(arms):
            r = json.loads((ROOT / f"runs/stage5/result-{pair}-{arm}.json").read_text())
            for m, (value, hatch) in enumerate(((r["val_exact_step"], "//"), (r["val_jev"]["accuracy"], ""))):
                x = k + (a * 2 + m - 1.5) * width
                ax.bar(x, value, width, color=("#b2182b", "#2166ac")[a], hatch=hatch, alpha=0.55 if hatch else 1,
                       label=f"{label}, {'exact neuron' if hatch else 'live Jev'}" if k == 0 else None)
                ax.text(x, value + 0.01, f"{value:.0%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(pairs)), ["0 vs 1", "3 vs 8"])
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Validation accuracy (500 images)")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage5-swap.png", dpi=160)
    print(FIG / "stage5-swap.png")


def margin():
    """Jev as a function: fire rate vs normalized margin m = z / spread, with a fitted noisy threshold."""
    from math import erf
    rows = json.loads((ROOT / "runs/probe/full2_rows.json").read_text())  # mode, fmt, k, m, z, bias, p
    rows = [r for r in rows if r[0] == "bias0"]
    m, fired = np.array([r[3] for r in rows]), np.array([r[6] >= 0.5 for r in rows], float)
    Phi = np.vectorize(lambda t: 0.5 * (1 + erf(t / np.sqrt(2))))
    ks = np.linspace(0.3, 5, 200)
    nll = [-np.mean(fired * np.log(Phi(k * m) + 1e-9) + (1 - fired) * np.log(1 - Phi(k * m) + 1e-9)) for k in ks]
    k = ks[int(np.argmin(nll))]
    edges = np.linspace(-3, 3, 25)
    mid = (edges[1:] + edges[:-1]) / 2
    idx = np.digitize(m, edges) - 1
    rate = [fired[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(mid))]
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.plot(mid, rate, "o", color="#1e1e1e", ms=5, label=f"Jev, {len(rows)} neurons, bias 0")
    t = np.linspace(-3, 3, 300)
    ax.plot(t, Phi(k * t), color="#e4507a", lw=2, label=f"noisy threshold  Φ({k:.2f} m)")
    ax.step(t, (t > 0).astype(float), color="#999999", lw=1, ls="--", where="post", label="exact neuron")
    ax.set_xlabel("normalized margin  m = z / spread of the terms")
    ax.set_ylabel("fraction of calls where Jev said fire")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "jev-as-a-function.png", dpi=160)
    print(FIG / "jev-as-a-function.png", "fitted k =", round(k, 2))


if __name__ == "__main__":
    {"stage3": stage3, "stage5": stage5, "margin": margin}[sys.argv[1]]()
