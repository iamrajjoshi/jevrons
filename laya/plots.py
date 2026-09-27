"""Figures for docs/results/stage10-teaching-laya.md from runs/stage10/eval-*.json and training logs.
Usage: uv run python plots.py base truth jev   (eval names; the first is base laya)"""

import json
import sys
from math import erf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import data  # noqa: E402

RUNS, FIG = data.ROOT / "runs/stage10", data.ROOT / "docs/figures"
JEV = {10: (3.2, 0.20), 30: (2.2, 0.15), 75: (1.8, -0.10), 150: (1.8, -0.6), 250: (1.5, -1.1)}  # runs/margin/summary.json
JEV_ACC = {10: 0.951, 30: 0.936, 75: 0.933, 150: 0.891, 250: 0.809}
COLORS = {"base": "#9a9a9a", "jev-copy": "#e4507a", "truth": "#2a7de1", "Jev": "#222222"}
Phi = np.vectorize(lambda t: 0.5 * (1 + erf(t / np.sqrt(2))))


def label(name):
    return "base laya" if name == "base" else ("laya, copy-Jev arm" if name.startswith("jev") else "laya, true-math arm")


def color(name):
    return COLORS["base"] if name == "base" else COLORS["jev-copy"] if name.startswith("jev") else COLORS["truth"]


def main(names):
    ev = {n: json.loads((RUNS / f"eval-{n}.json").read_text()) for n in names}
    FIG.mkdir(parents=True, exist_ok=True)
    mm = np.linspace(-3, 3, 200)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True, sharey=True)
    for ax, k in zip(axes, (10, 30, 75)):
        for n in names:
            c = ev[n]["curves"][f"probe-{k}"]
            b = np.array(c["binned"])
            ax.plot(b[:, 0] + 0.125, b[:, 1], "o", ms=3, color=color(n))
            ax.plot(mm, Phi(c["k"] * (mm - c["m0"])), color=color(n), label=f"{label(n)}: k {c['k']:.1f}, m0 {c['m0']:+.2f}")
        ax.plot(mm, Phi(JEV[k][0] * (mm - JEV[k][1])), "--", color=COLORS["Jev"], label=f"Jev: k {JEV[k][0]}, m0 {JEV[k][1]:+.2f}")
        ax.axvline(0, color="#ddd", lw=0.8, zorder=0)
        ax.set_title(f"{k} terms + bias (margin probe states)", fontsize=9)
        ax.set_xlabel("m = total / spread")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    axes[0].set_ylabel("dots: mean p per 0.25 bin\nlines: Φ(k(m − m0)) fitted to yes/no")
    fig.savefig(FIG / "stage10-curves.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    ax.axvspan(2, 60, color="#eef4fc", zorder=0, label="trained term counts (10, 30 held out)")
    for n in names:
        ks = sorted(int(key.split("-")[1]) for key in ev[n]["curves"] if key.startswith("gen-"))
        acc = [ev[n]["curves"][f"gen-{k}"]["sign_accuracy"] for k in ks]
        ax.plot(ks, acc, "o-", color=color(n), label=label(n) + " (fresh probe states)")
    ax.plot(list(JEV_ACC), list(JEV_ACC.values()), "s--", color=COLORS["Jev"], label="Jev (margin probe)")
    ax.axvline(97, color="#bbb", lw=0.8, ls=":")
    ax.text(98, 0.55, "512-token\ncontext ends", fontsize=7, color="#888")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 30, 60, 90, 150, 250], ["2", "5", "10", "30", "60", "90", "150", "250"])
    ax.set_xlabel("terms in the sum (150: context raised to 1,024)")
    ax.set_ylabel("sign accuracy, m uniform on [−3, 3]")
    ax.set_ylim(0.4, 1.01)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(FIG / "stage10-accuracy.png", dpi=160)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for n in names[1:]:
        rows = [json.loads(line) for line in (RUNS / n / "log.jsonl").open()]
        steps = [r for r in rows if "loss" in r]
        loss = np.convolve([r["loss"] for r in steps], np.ones(50) / 50, "valid")
        axes[0].plot(np.arange(len(loss)) + 50, loss, color=color(n), label=label(n))
        evs = [r["eval"] for r in rows if "eval" in r]
        for key, ls in (("acc_s10", "-"), ("acc_s75", ":"), ("acc_real", "--")):
            axes[1].plot([e["step"] for e in evs], [e[key] for e in evs], ls, marker="o", ms=3, color=color(n),
                         label=f"{label(n)}: {key[4:]}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step (batch 16)")
    axes[0].set_ylabel("training loss (50-step mean)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("validation accuracy vs true sign")
    axes[1].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.savefig(FIG / "stage10-training.png", dpi=160)

    jev_traps = json.loads((data.ROOT / "runs/stage6b/traps.json").read_text())["families"]
    fams = ["last", "first", "vote", "long90", "zeros60"]
    titles = {"last": "big number last\n(75 terms)", "first": "big number first\n(75)", "vote": "sign vote\n(75)",
              "long90": "long list, m = −1\n(90; Jev: 250)", "zeros60": "zero padding\n(60; Jev: 150)"}
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    x = np.arange(len(fams))
    series = [("Jev", {f: jev_traps[{"long90": "long", "zeros60": "zeros"}.get(f, f)] for f in fams})] + \
             [(n, ev[n]["traps"]) for n in names]
    w = 0.8 / len(series)
    for s, (n, t) in enumerate(series):
        col = COLORS["Jev"] if n == "Jev" else color(n)
        xs = x - 0.4 + w * (s + 0.5)
        ax.bar(xs, [t[f]["trap_misfire"] for f in fams], w * 0.9, color=col, label=("Jev" if n == "Jev" else label(n)) + ": trap")
        ax.scatter(xs, [t[f]["control_misfire"] for f in fams], marker="_", s=120, color="#f5a623", zorder=3,
                   label="matched control" if s == 0 else None)
    ax.set_xticks(x, [titles[f] for f in fams], fontsize=8)
    ax.set_ylabel("misfire rate (200 states per bar)")
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center")
    fig.savefig(FIG / "stage10-traps.png", dpi=160)
    print("figures written")


if __name__ == "__main__":
    main(sys.argv[1:])
