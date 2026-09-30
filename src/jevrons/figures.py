"""Every figure in docs/blog/jevrons.mdx, in one style (jevrons.figstyle), from the run files.

Offline: reads runs/ and replays local exact-neuron training where a journal lacks a run. No API calls.
Every number drawn comes from a data file; where a figure labels a value the post cites, it's asserted.
Usage: uv run python -m jevrons.figures [--theme docs|light|dark|both] [name ...]
  (default: every figure, docs theme into docs/figures; light/dark/both write the blog's variants to
  docs/figures/site/ as name.png/.svg and name-dark.png/.svg; names are the keys of FIGURES)
"""

import json
import sys
from math import erf
from pathlib import Path

import numpy as np

from jevrons import figstyle as S

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
Phi = np.vectorize(lambda t: 0.5 * (1 + erf(t / np.sqrt(2))))


def js(rel):
    return json.loads((RUNS / rel).read_text())


def check(value, cited, fmt="{:.0%}"):
    """The figure labels `value`; the post cites it as `cited`. Fail loudly if they drift apart
    (by more than the cited precision allows, so 0.385 may be cited as 39%)."""
    got = fmt.format(value)
    pct = cited.endswith("%")
    num = float(cited.rstrip("%")) / (100 if pct else 1)
    decimals = len(cited.rstrip("%").partition(".")[2])
    assert abs(value - num) <= 0.5 * 10 ** -decimals / (100 if pct else 1) + 1e-9, f"figure value {got} != cited {cited}"
    return got


def weights(rel):
    d = np.load(RUNS / rel)
    return [d[f"arr_{i}"] for i in range(len(d.files))]


# ---------------------------------------------------------------- shared marks

def dumbbell(ax, rows, lo, hi, chance=None):
    """One row per network: hollow dot = exact neuron, filled dot = live Jev, joined by a line.
    rows: [(label, colour, exact, jev)], top to bottom. The live-Jev value is labelled in the row's
    colour, the exact-neuron value in grey, each on its own side of the pair."""
    for y, (label, color, exact, jev) in enumerate(rows):
        y = -y
        ax.plot([exact, jev], [y, y], color=color, lw=1.6, alpha=0.45, zorder=1, solid_capstyle="butt")
        ax.plot(exact, y, "o", ms=10, mfc=S.PAPER, mec=color, mew=1.8, zorder=2)
        ax.plot(jev, y, "o", ms=6.5, color=color, zorder=3)  # smaller and on top: a tie reads as a dot in a ring
        strong = S.JEV_TEXT if color == S.JEV else S.INK
        first = -1 if exact <= jev else 1  # exact on the left when it's the smaller value
        for v, c, weight, side in ((exact, S.MUTE, "normal", first), (jev, strong, "medium", -first)):
            ax.annotate(f"{v:.1%}", (v, y), xytext=(side * 10, 0), textcoords="offset points", va="center",
                        ha="right" if side < 0 else "left", fontsize=9.5, color=c, fontweight=weight)
    ax.set_yticks([-i for i in range(len(rows))], [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    S.pct(ax, "x")
    ax.grid(axis="x", color=S.FAINT, lw=0.8)
    ax.set_axisbelow(True)
    if chance is not None:
        ax.axvline(chance, color=S.MUTE, lw=0.8, ls=(0, (2, 2)), zorder=0)


def dumbbell_key(ax):
    """Key for the dumbbell marks, as a one-row legend above the axes."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=9, mfc=S.PAPER, mec=S.INK2, mew=1.6),
         Line2D([], [], ls="", marker="o", ms=6.5, color=S.INK2)]
    ax.legend(h, ["exact neuron", "live Jev"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9.5,
              handletextpad=0.1, columnspacing=1.2, borderaxespad=0.2, borderpad=0)


# ---------------------------------------------------------------- Does it add?

def margin_by_terms():
    rows = js("margin/rows.json")  # [terms, m, p_rep0, p_rep1]
    fits = js("margin/summary.json")["offset_fit"]
    for n, (k, m0) in {10: (3.2, 0.20), 250: (1.5, -1.10)}.items():  # the post's table
        assert (fits[str(n)]["k"], fits[str(n)]["threshold_m0"]) == (k, m0)
    terms = sorted({r[0] for r in rows})
    lo = 0.45 if S.THEME == "dark" else 0.28  # the pale end of the ramp fades into dark paper sooner
    colors = [S.SEQ_JEV(v) for v in np.linspace(lo, 0.92, len(terms))]
    edges = np.linspace(-3, 3, 25)
    mid = (edges[1:] + edges[:-1]) / 2
    t = np.linspace(-3, 3, 400)
    fig, ax = S.figure(4.1)
    ax.axhline(0.5, color=S.RULE, lw=0.8, zorder=0)
    ax.plot([-3, 0, 0, 3], [0, 0, 1, 1], color=S.INK, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.08, 0.03, "exact neuron", color=S.INK2, fontsize=9.5, va="bottom")
    label_y = {250: 0.62, 150: 0.47, 75: 0.33, 30: 0.2, 10: 0.5}
    for color, n in zip(colors, terms):
        m = np.array([r[1] for r in rows if r[0] == n])
        fired = np.array([[r[2] >= 0.5, r[3] >= 0.5] for r in rows if r[0] == n], float).mean(1)
        idx = np.digitize(m, edges) - 1
        rate = np.array([fired[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(mid))])
        k, m0 = fits[str(n)]["k"], fits[str(n)]["threshold_m0"]
        ax.plot(mid, rate, "o", color=color, ms=3.6, alpha=0.9, zorder=2)
        ax.plot(t, Phi(k * (t - m0)), color=color, lw=2, zorder=3)
        y = label_y[n]
        x = m0 + float(np.sqrt(2) * _erfinv(2 * y - 1)) / k
        right = n == 10  # the shortest list is the rightmost curve at this height; label it on the open side
        ax.text(x + (0.09 if right else -0.09), y, "10 terms" if right else str(n), color=color if n > 30 else S.JEV_TEXT,
                fontsize=10, ha="left" if right else "right", va="center", fontweight="medium")
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("normalized margin  m = z / spread")
    ax.set_ylabel("share of calls where Jev said fire")
    S.pct(ax)
    S.ygrid(ax)
    S.save(fig, "jev-margin-by-terms.png")


def _erfinv(y):
    """Inverse error function by Newton on math.erf; only used to place labels on fitted curves."""
    x = 0.0
    for _ in range(60):
        x -= (erf(x) - y) / (2 / np.sqrt(np.pi) * np.exp(-x * x))
    return x


# ---------------------------------------------------------------- The bias trick

def bias():
    """When the bias and the true total have opposite signs, which one does Jev side with?"""
    rows = js("probe/wording_rows.json")  # [format, wording, terms, z, bias, p]; both formats share the states
    terms = sorted({r[2] for r in rows})
    fig, ax = S.figure(3.3)
    x = np.arange(len(terms))
    w = 0.36
    for off, fmt, color, name in ((-w / 2 - 0.02, "products", S.JEV, "bias as its own field"),
                                  (w / 2 + 0.02, "folded", S.INK, "bias folded into the list")):
        vals, ns = [], []
        for n in terms:
            dis = [r for r in rows if r[0] == fmt and r[2] == n and (r[3] > 0) != (r[4] > 0)]
            vals.append(np.mean([(r[5] >= 0.5) == (r[4] > 0) for r in dis]))
            ns.append(len(dis))
        ax.bar(x + off, vals, w, color=color, zorder=2)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=9.5,
                    color=S.JEV_TEXT if color == S.JEV else S.INK2)
        pooled = sum(v * n for v, n in zip(vals, ns)) / sum(ns)
        ax.text(len(terms) - 0.45, pooled, f"{name}\n{pooled:.0%} of {sum(ns)} calls", ha="left", va="center",
                fontsize=9.5, color=S.JEV_TEXT if color == S.JEV else S.INK2, linespacing=1.3)
        ax.plot([len(terms) - 0.6, len(terms) - 0.5], [pooled, pooled], color=color, lw=2, clip_on=False)
    ax.axhline(0.5, color=S.MUTE, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(-0.55, 0.97, "calls where the bias and the true total have opposite signs", fontsize=9.5, color=S.INK2, va="top")
    ax.set_xticks(x, [f"{n} terms" if i == 0 else str(n) for i, n in enumerate(terms)])
    ax.set_xlim(-0.6, len(terms) + 0.9)
    ax.spines["bottom"].set_bounds(-0.6, len(terms) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Jev sided with the bias")
    S.pct(ax)
    S.ygrid(ax)
    S.save(fig, "jev-bias-overtrust.png")


# ---------------------------------------------------------------- XOR

def xor():
    arms = (("ste_products", "straight-through,\nbias as its own field"), ("ste_folded", "straight-through,\nbias folded in"),
            ("spsa_products", "SPSA,\nbias as its own field"))
    target = [0, 1, 1, 0]
    inputs = ["0, 0", "0, 1", "1, 0", "1, 1"]
    fig, axes = S.figure(2.9, ncols=3, sharey=True)
    for ax, (arm, title) in zip(axes, arms):
        res = [js(f"stage2/result-XOR-{arm}-{s}.json") for s in (0, 1, 2)]
        for i, tgt in enumerate(target):
            ax.plot([i - 0.3, i + 0.3], [tgt, tgt], color=S.INK, lw=1.2, zorder=1)
            for s, r in enumerate(res):
                p = r["mean_p"][i]
                wrong = (p >= 0.5) != bool(tgt)
                ax.plot(i + (s - 1) * 0.18, p, "o", ms=6.5, color=S.JEV if wrong else S.GREY, zorder=3,
                        mec=S.PAPER, mew=0.8)
        solved = sum(r["all_correct_frac"] * len(r["pass_accuracy"]) for r in res)
        total = sum(len(r["pass_accuracy"]) for r in res)
        ax.set_title(title, fontsize=10, color=S.INK, loc="left", linespacing=1.25)
        ax.text(0.0, -0.36, f"{solved:.0f} of {total} passes fully correct", transform=ax.transAxes, fontsize=9.5, color=S.INK2)
        ax.axhline(0.5, color=S.MUTE, lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_xticks(range(4), inputs)
        ax.set_xlim(-0.55, 3.55)
        ax.set_xlabel("input", fontsize=10)
    folded = sum(js(f"stage2/result-XOR-ste_folded-{s}.json")["all_correct_frac"] * 5 for s in (0, 1, 2))
    assert round(folded) == 14
    axes[0].text(1.0, 0.94, "target", fontsize=9, color=S.INK2, ha="center", va="top")
    axes[2].text(3.5, 0.62, "wrong side\nof 0.5", fontsize=9, color=S.JEV_TEXT, ha="right", va="bottom", linespacing=1.1)
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_ylabel("Jev's mean output p")
    axes[0].set_yticks([0, 0.5, 1])
    S.save(fig, "stage2-xor.png")


# ---------------------------------------------------------------- A circle

def stage3_boundary():
    from jevrons.stage3 import GRID, XTE, YTE
    runs = [("ste_jev", "trained through Jev"), ("swap", "trained exact, run on Jev")]
    fig, axes = S.figure(3.9, ncols=2)
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, (arm, title) in zip(axes, runs):
        r = js(f"stage3/result-{arm}-0.json")
        p = np.array(r["grid_p"]).reshape(len(GRID), len(GRID))
        step = GRID[1] - GRID[0]
        im = ax.imshow(p, origin="lower", extent=(-1 - step / 2, 1 + step / 2) * 2, cmap=S.FIRE, vmin=0, vmax=1,
                       interpolation="none")
        ax.plot(0.8 * np.cos(th), 0.8 * np.sin(th), color=S.INK, lw=1.2, ls=(0, (4, 3)))
        inside = YTE.ravel() > 0.5
        ax.scatter(*XTE[inside].T, s=5, color=S.INK, lw=0, alpha=0.75)
        ax.scatter(*XTE[~inside].T, s=9, color=S.INK, marker="x", lw=0.6, alpha=0.55)
        ax.set_title(f"{title}\nheld-out accuracy {r['test_accuracy']:.0%}", fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        for s in ax.spines.values():
            s.set_visible(False)
    check(js("stage3/result-ste_jev-0.json")["test_accuracy"], "84%")
    check(js("stage3/result-swap-0.json")["test_accuracy"], "47%")
    cb = fig.colorbar(im, ax=axes, shrink=0.75, aspect=22, pad=0.03)
    cb.set_label("Jev's output p (fire = inside)", fontsize=10, color=S.INK2)
    cb.set_ticks([0, 0.5, 1])
    cb.outline.set_visible(False)
    S.save(fig, "stage3-boundary.png")


# ---------------------------------------------------------------- Digits with the sum in code

def wording():
    """The scalar neuron: Jev's p against z for the statement wording and the question wording."""
    d = js("probe/scalar_curve.json")
    z, p = np.array(d["z"]), np.array(d["p_mean"])
    za = np.array(sorted(float(k) for k in d["alt"]))
    pa = np.array([d["alt"][k] for k in sorted(d["alt"], key=float)])
    neg = p[z < 0]
    lo_hi = f"{neg.min():.2f}-{neg.max():.2f}"
    assert lo_hi == "0.66-0.74", lo_hi
    assert f"{pa[za <= 0].min():.2f}-{pa[za <= 0].max():.2f}" == "0.01-0.05" and np.all(pa[za >= 0.1] == 0.99)
    w = np.abs(z) <= 10
    fig, ax = S.figure(3.4)
    ax.axhline(0.5, color=S.RULE, lw=0.8, zorder=0)
    ax.plot([-10, 0, 0, 10], [0, 0, 1, 1], color=S.INK, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot(z[w], p[w], color=S.JEV, lw=1.8, zorder=3)
    ax.plot(za, pa, color=S.INK, lw=1.8, zorder=2)
    ax.text(-9.7, neg.max() + 0.04, "statement: \u201cz is greater than zero.\u201d", color=S.JEV_TEXT, fontsize=10,
            va="bottom")
    ax.text(-9.7, neg.min() - 0.04, f"p = {neg.min():.2f} to {neg.max():.2f} for every negative z", color=S.JEV_TEXT,
            fontsize=9.5, va="top")
    ax.text(-9.7, 0.08, "question: \u201cIs the number z positive?\u201d", color=S.INK, fontsize=10, va="bottom")
    ax.text(9.7, 0.9, "exact step", color=S.INK2, fontsize=9.5, ha="right", va="top")
    ax.set_xlim(-10.2, 10.2)
    ax.set_xticks(range(-10, 11, 5))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("z, the sum computed in code")
    ax.set_ylabel("Jev's answer p")
    ax.set_yticks([0, 0.5, 1])
    S.ygrid(ax)
    S.save(fig, "jev-wording-trap.png")


# ---------------------------------------------------------------- Digits with Jev doing the sum

def stage5_swap():
    rows = []
    for pair, name in (("0v1", "0 vs 1"), ("3v8", "3 vs 8")):
        for arm, who, color in (("jev", "trained through Jev", S.JEV), ("swap", "trained exact", S.INK)):
            r = js(f"stage5/result-{pair}-{arm}.json")
            rows.append((f"{name}, {who}", color, r["val_exact_step"], r["val_jev"]["accuracy"]))
    check(rows[3][3], "67.8%", "{:.1%}")
    check(rows[2][3], "94.2%", "{:.1%}")
    fig, ax = S.figure(2.7)
    dumbbell(ax, rows, 0.5, 1.0, chance=0.5)
    ax.axhline(-1.5, color=S.RULE, lw=0.8)
    ax.set_xlabel("validation accuracy, 500 images")
    dumbbell_key(ax)
    S.save(fig, "stage5-swap.png")


def stage5_training():
    fig, axes = S.figure(4.0, ncols=2, nrows=2, sharex=True)
    k = 5
    for c, (pair, name) in enumerate((("0v1", "0 vs 1"), ("3v8", "3 vs 8"))):
        for arm, color, label in (("swap", S.INK, "trained exact"), ("jev", S.JEV, "trained through Jev")):
            cur = js(f"stage5/curve-{pair}-{arm}.json")
            step = np.array([r["step"] for r in cur]) + 1
            for r_, key in ((0, "acc"), (1, "loss")):
                if key == "loss" and arm == "swap":
                    continue
                v = np.array([r[key] for r in cur])
                sm = np.convolve(v, np.ones(k) / k, "valid")
                ax = axes[r_, c]
                ax.plot(step, v, color=color, lw=0.8, alpha=0.3)
                ax.plot(step[k - 1:], sm, color=color, lw=2)
                if c == 1 and r_ == 0:
                    ax.text(75, 0.36 if arm == "jev" else 0.47, label, fontsize=9.5, ha="right", va="center",
                            color=S.JEV_TEXT if arm == "jev" else S.INK2)
        axes[0, c].set_title(name)
        axes[0, c].set_ylim(0.3, 1.02)
        S.pct(axes[0, c])
        axes[1, c].set_ylim(0, 1.5)
        axes[1, c].set_xlabel("training step (batch of 32)")
        for r_ in (0, 1):
            S.ygrid(axes[r_, c])
            axes[r_, c].set_xlim(0, 76)
    axes[0, 1].set_xlim(0, 76)
    axes[0, 0].set_ylabel("batch accuracy\non its own neuron")
    axes[1, 0].set_ylabel("loss, trained\nthrough Jev")
    axes[1, 0].text(75, 1.4, "exact-trained loss not drawn:\na step neuron outputs 0 or 1,\nso its loss only counts\nmistakes",
                    fontsize=9, color=S.MUTE, ha="right", va="top", linespacing=1.25)
    S.save(fig, "stage5-training.png")


def stage5_margins():
    """Hidden margins on 3 vs 8 validation images, live Jev calls, split by whether Jev fired as intended."""
    from jevrons.digits import mnist, split
    from jevrons.stage6b import hidden_margins
    xtr, ytr, _, _ = mnist()
    _, va = split(ytr, (3, 8), 500, 500)
    c = np.load(RUNS / "stage6c/calls.npz")
    base = (c["src"] == 0) & (c["pair"] == 1) & (c["layer"] == 0) & (c["split"] == 1)
    bins = np.linspace(-6, 6, 49)
    fig, axes = S.figure(4.2, nrows=2, sharex=True, sharey=True)
    for ax, (arm, code, title) in zip(axes, (("jev", 0, "trained through Jev"), ("swap", 1, "trained exact, run on Jev"))):
        W1, b1 = weights(f"stage5/weights-3v8-{arm}.npz")[:2]
        z_all, s_all, _ = hidden_margins(xtr[va], W1, b1)
        sel = base & (c["arm"] == code)
        i, j = c["i"][sel].astype(int), c["j"][sel].astype(int)
        z, p = c["z"][sel], c["p"][sel]
        err = np.abs(z - z_all[i, j])  # the journal's z is exact, the replay sums rounded products
        assert np.median(err) < 0.02 and err.max() < 0.2, "journal calls don't line up with the replayed sums"
        m = z / s_all[i, j]
        ok = (p >= 0.5) == (z > 0)
        mc = np.clip(m, bins[0] + 1e-9, bins[-1] - 1e-9)
        ax.hist([mc[ok], mc[~ok]], bins, stacked=True, color=[S.GREY, S.JEV], rwidth=0.9, zorder=2)
        ax.axvline(0, color=S.INK, lw=0.8, zorder=3)
        ax.axvspan(-1, 1, color=S.FAINT, zorder=0, lw=0)
        near = np.mean(np.abs(m) < 1)
        res = js(f"stage5/result-3v8-{arm}.json")["val_jev"]["fires_as_intended"]["hidden"]
        assert abs(ok.mean() - res) < 1e-3, (ok.mean(), res)  # the result file counts z = 0 slightly differently
        ax.set_title(title)
        ax.text(0.99, 0.95, f"fires as intended {res:.1%}", transform=ax.transAxes, ha="right", va="top", fontsize=10,
                color=S.INK2)
        ax.text(0.99, 0.78, f"Jev misfired {1 - res:.1%}", transform=ax.transAxes, ha="right", va="top", fontsize=10,
                color=S.JEV_TEXT)
        ax.text(0, 1.01, f"{near:.0%} within one spread of zero", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9.5, color=S.INK2)
        ax.set_ylabel("hidden calls")
        S.ygrid(ax)
    check(js("stage5/result-3v8-jev.json")["val_jev"]["fires_as_intended"]["hidden"], "93.0%", "{:.1%}")
    check(js("stage5/result-3v8-swap.json")["val_jev"]["fires_as_intended"]["hidden"], "86.9%", "{:.1%}")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.22)
    axes[1].set_xlabel("hidden margin  m = z / spread  (clipped at ±6)")
    axes[1].set_xlim(-6.1, 6.1)
    S.save(fig, "stage5-hidden-margins.png")


# ---------------------------------------------------------------- All ten digits

def stage6_swap():
    fig, axes = S.figure(6.4, nrows=2)
    for ax, split_ in zip(axes, ("val", "test")):
        rows = []
        for seed in (0, 1, 2):
            for arm, who, color in (("jev", "trained through Jev", S.JEV), ("swap", "trained exact", S.INK)):
                r = js(f"stage6/result-seed{seed}-{arm}.json")
                r = r["val"]["10"] if split_ == "val" else r["test"]
                rows.append((f"seed {seed}, {who}", color, r["exact_step"], r["accuracy"]))
        dumbbell(ax, rows, 0.3, 0.95, chance=None)
        for y in (-1.5, -3.5):
            ax.axhline(y, color=S.RULE, lw=0.8)
        ax.set_xlabel("validation, 500 images" if split_ == "val" else "test, 1,000 images")
    test = [js(f"stage6/result-seed{s}-{a}.json")["test"] for s in (0, 1, 2) for a in ("jev", "swap")]
    check(np.mean([t["accuracy"] for t in test[::2]]), "68.5%", "{:.1%}")
    check(np.mean([t["accuracy"] for t in test[1::2]]), "48.0%", "{:.1%}")
    dumbbell_key(axes[0])
    S.save(fig, "stage6-swap.png")


def _stage6_predictions(arm):
    """Test predictions per seed, recomputed from the logged outputs with the run's tie-break."""
    from jevrons.digits import mnist, test_subset
    from jevrons.net import decide
    _, _, _, yte = mnist()
    y = yte[test_subset(yte, 1000)]
    out = []
    for s in (0, 1, 2):
        r = js(f"stage6/result-seed{s}-{arm}.json")["test"]
        pred = decide(np.array(r["outputs"]))
        out.append((y, pred, r))
    return out


def stage6_digits():
    ink = js("stage6c/pixels.json")["ink_pixels_per_digit"]
    fig = S.figure(6.3)[0]
    fig.clf()
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
    ax = fig.add_subplot(gs[0, :])
    digits = np.arange(10)
    means = {}
    for arm, color, off, label in (("jev", S.JEV, -0.14, "trained through Jev"), ("swap", S.INK, 0.14, "trained exact, run on Jev")):
        per = np.array([js(f"stage6/result-seed{s}-{arm}.json")["test"]["per_digit"] for s in (0, 1, 2)])
        means[arm] = per.mean(0)
        ax.vlines(digits + off, per.min(0), per.max(0), color=color, lw=1.4, alpha=0.5)
        ax.plot(digits + off, per.mean(0), "o", color=color, ms=7, mec=S.PAPER, mew=0.8, label=label)
    for d in (0, 3, 8):
        ax.text(d + 0.3, means["swap"][d], f"{means['swap'][d]:.0%}", ha="left", va="center", fontsize=9.5, color=S.INK2)
        ax.text(d - 0.14, means["jev"][d] + 0.05, f"{means['jev'][d]:.0%}", ha="center", va="bottom", fontsize=9.5,
                color=S.JEV_TEXT)
    for d, cited in zip((0, 3, 8), ("7%", "17%", "5%")):
        check(means["swap"][d], cited)
    for d, cited in zip((0, 3, 8), ("82%", "66%", "43%")):
        check(means["jev"][d], cited)
    ax.set_xticks(digits, [f"{d}\n{ink[d]:.0f}" for d in digits])
    ax.text(-0.75, -0.33, "active\npixels", transform=ax.get_xaxis_transform(), ha="right", fontsize=9, color=S.MUTE,
            linespacing=1.1)
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test accuracy per digit")
    S.pct(ax)
    S.ygrid(ax)
    ax.legend(loc="lower right", ncol=2, fontsize=9.5, handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1,
              bbox_to_anchor=(1, 1.0))
    for c, (arm, cmap, title) in enumerate((("jev", S.SEQ_JEV, "trained through Jev"), ("swap", S.SEQ_INK, "trained exact, run on Jev"))):
        preds = _stage6_predictions(arm)
        conf = np.zeros((10, 10))
        for y, pred, r in preds:
            np.add.at(conf, (y, pred), 1)
            per = [np.mean(pred[y == d] == d) for d in range(10)]
            assert np.allclose(per, r["per_digit"], atol=0.021), "recomputed predictions drift from the logged ones"
        conf /= conf.sum(1, keepdims=True)
        axc = fig.add_subplot(gs[1, c])
        axc.imshow(conf, cmap=cmap, vmin=0, vmax=1, interpolation="none")
        for i in range(10):
            for j in range(10):
                if conf[i, j] >= 0.1:
                    axc.text(j, i, f"{conf[i, j] * 100:.0f}", ha="center", va="center", fontsize=9,
                             color=S.text_on(cmap(conf[i, j])))
        axc.set_xticks(digits)
        axc.set_yticks(digits)
        axc.tick_params(length=0, labelsize=9)
        for s in axc.spines.values():
            s.set_visible(False)
        axc.set_title(title, fontsize=10.5)
        axc.set_xlabel("predicted", fontsize=10)
        if c == 0:
            axc.set_ylabel("true digit", fontsize=10)
    fig.text(1.0, -0.015, "cells: % of each true digit, 3 seeds pooled, shown when 10% or more", ha="right", va="top",
             fontsize=9, color=S.MUTE)
    S.save(fig, "stage6-digits.png")


# ---------------------------------------------------------------- Five times the data

def stage7():
    j, s = js("stage7/result-jev.json")["test"], js("stage7/result-swap.json")["test"]
    check(j["accuracy"], "84.6%", "{:.1%}")
    check(j["exact_step"], "70.0%", "{:.1%}")
    check(s["accuracy"], "52.6%", "{:.1%}")
    check(s["exact_step"], "83.9%", "{:.1%}")
    fig = S.figure(5.4)[0]
    fig.clf()
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.9])
    ax = fig.add_subplot(gs[0])
    six = {a: [js(f"stage6/result-seed{sd}-{a}.json")["test"] for sd in (0, 1, 2)] for a in ("jev", "swap")}
    rows = [("5,000 images, trained through Jev", S.JEV, j["exact_step"], j["accuracy"]),
            ("5,000 images, trained exact", S.INK, s["exact_step"], s["accuracy"]),
            ("1,000 images, trained through Jev", S.JEV_LIGHT, np.mean([t["exact_step"] for t in six["jev"]]),
             np.mean([t["accuracy"] for t in six["jev"]])),
            ("1,000 images, trained exact", S.GREY, np.mean([t["exact_step"] for t in six["swap"]]),
             np.mean([t["accuracy"] for t in six["swap"]]))]
    dumbbell(ax, rows, 0.3, 0.95)
    ax.axhline(-1.5, color=S.RULE, lw=0.8)
    ax.set_xlabel("test accuracy (2,000 images;\n1,000-image rows are the mean of 3 ten-digit seeds)", fontsize=10)
    dumbbell_key(ax)
    ax2 = fig.add_subplot(gs[1])
    digits = np.arange(10)
    for arm, res, c7, c6, off in (("jev", j, S.JEV, S.JEV_LIGHT, -0.15), ("swap", s, S.INK, S.GREY, 0.15)):
        before = np.mean([t["per_digit"] for t in six[arm]], 0)
        after = np.array(res["per_digit"])
        ax2.vlines(digits + off, before, after, color=c6, lw=1.4, zorder=1)
        ax2.plot(digits + off, before, "o", ms=5.5, mfc=S.PAPER, mec=c6, mew=1.4, zorder=2)
        ax2.plot(digits + off, after, "o", ms=7, color=c7, mec=S.PAPER, mew=0.8, zorder=3)
    for d, cited in ((0, "6%"), (3, "17%")):
        check(s["per_digit"][d], cited)
        ax2.text(d + 0.32, s["per_digit"][d], f"{s['per_digit'][d]:.1%}", va="center", fontsize=9.5, color=S.INK2)
    from matplotlib.lines import Line2D
    ax2.legend([Line2D([], [], ls="", marker="o", ms=7, color=S.JEV), Line2D([], [], ls="", marker="o", ms=7, color=S.INK)],
               ["trained through Jev", "trained exact"], loc="lower left", bbox_to_anchor=(0.47, 0.02), ncol=1, fontsize=9.5,
               handletextpad=0.1, columnspacing=1.2)
    ax2.text(0.0, 1.03, "per digit, on live Jev: hollow = 1,000 images (mean of 3 ten-digit seeds),\nfilled = 5,000 images",
             transform=ax2.transAxes, fontsize=9.5, color=S.INK2)
    ax2.set_xticks(digits)
    ax2.set_xlabel("digit")
    ax2.set_xlim(-0.6, 9.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("test accuracy")
    S.pct(ax2)
    S.ygrid(ax2)
    S.save(fig, "stage7-swap.png")


def stage8_sparse():
    """Dense (stage 7) against sparse (stage 8a/8c) on the scoreboard, and misfires by margin band."""
    r7 = {a: js(f"stage7/result-{a}.json")["test"] for a in ("jev", "swap")}
    r8 = {a: js(f"stage8a/result-{a}.json")["test"] for a in ("jev", "swap")}
    pre, ft = js("stage8c/result-pretrained-test.json"), js("stage8c/result.json")["test"]
    rows = [("dense, trained through Jev", S.JEV, r7["jev"]), ("dense, trained exact", S.INK, r7["swap"]),
            ("sparse, trained through Jev", S.JEV, r8["jev"]), ("sparse, trained exact", S.INK, r8["swap"]),
            ("sparse, pretrained on simulated Jev", S.JEV_LIGHT, pre), ("sparse, pretrained, then trained on Jev", S.JEV, ft)]
    cited = (("84.6%", "70.0%"), ("52.6%", "83.9%"), ("85.3%", "75.7%"), ("65.8%", "72.6%"), ("83.8%", "57.1%"),
             ("83.8%", "79.0%"))  # (on Jev, exact neuron) per row
    for (_, _, r), (on_jev, exact) in zip(rows, cited):
        check(r["accuracy"], on_jev, "{:.1%}")
        check(r["exact_step"], exact, "{:.1%}")
    bt = js("stage8a/by-terms.json")
    fig = S.figure(6.2)[0]
    fig.clf()
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0])
    dumbbell(ax, [(label, color, r["exact_step"], r["accuracy"]) for label, color, r in rows], 0.45, 0.95)
    for y in (-1.5, -3.5):
        ax.axhline(y, color=S.RULE, lw=0.8)
    ax.set_xlabel("test accuracy (dense 784-32-10: 2,000 images;\nsparse 784-64-10, 96 pixels per neuron: 1,000)",
                  fontsize=10)
    dumbbell_key(ax)
    ax2 = fig.add_subplot(gs[1])
    bands = list(bt["jev"]["by_abs_m"])
    x = np.arange(len(bands))

    def pooled(keys):
        n = np.array([[bt[k]["by_abs_m"][b][0] for b in bands] for k in keys], float)
        rate = np.array([[bt[k]["by_abs_m"][b][1] for b in bands] for k in keys])
        terms = np.average([bt[k]["mean_terms"] for k in keys], weights=n.sum(1))
        return (n * rate).sum(0) / n.sum(0), terms

    for keys, shade in ((["stage 7 dense (both arms)"], 0.92), (["jev", "swap"], 0.45)):
        rate, terms = pooled(keys)
        color = S.SEQ_JEV(shade)
        name = "dense" if len(keys) == 1 else "sparse"
        ax2.plot(x, rate, "o-", color=color, ms=5, lw=2, label=f"{name}, about {terms:.0f} numbers per call")
    ax2.legend(loc="upper right", fontsize=9.5, handlelength=1.6)
    ax2.set_xticks(x, [b.replace("3-99", "3+").replace("-", "\u2013") for b in bands])
    ax2.set_xlabel("hidden margin band  |m| = |z| / spread  (both arms pooled)")
    ax2.set_ylabel("hidden misfires")
    ax2.set_ylim(0, 0.42)
    S.pct(ax2)
    S.ygrid(ax2)
    S.save(fig, "stage8-sparse.png")


def training_curves():
    """The two headline networks, dense (stage 7) and sparse (stage 8a): batch accuracy and loss by epoch,
    validation on Jev at the evaluated epochs, and the final test accuracy on Jev."""
    nets = (("stage7", "dense 784-32-10, 5,000 images", {"jev": "84.6%", "swap": "52.6%"}),
            ("stage8a", "sparse 784-64-10, 1,000 images", {"jev": "85.3%", "swap": "65.8%"}))
    fig, axes = S.figure(8.2, nrows=4, gridspec_kw={"height_ratios": [1.5, 1, 1.5, 1]})
    for c, (run, title, cited) in enumerate(nets):
        top, bottom = axes[2 * c], axes[2 * c + 1]
        top.sharex(bottom)
        top.tick_params(labelbottom=False)
        for arm, color in (("swap", S.INK), ("jev", S.JEV)):
            cur = js(f"{run}/curve-{arm}.json")
            res = js(f"{run}/result-{arm}.json")
            epochs = max(r["epoch"] for r in cur) + 1
            per = len(cur) / epochs
            x = (np.arange(len(cur)) + 1) / per
            k = max(3, int(per // 4))
            for ax, key in ((top, "acc"), (bottom, "loss")):
                if key == "loss" and arm == "swap":
                    continue  # a step neuron's loss is a clipped count of mistakes, not comparable
                v = np.array([r[key] for r in cur])
                ax.plot(x, v, color=color, lw=0.6, alpha=0.12)
                ax.plot(x[k - 1:], np.convolve(v, np.ones(k) / k, "valid"), color=color, lw=1.8)
            val = sorted((int(e), r["accuracy"]) for e, r in res["val"].items())
            top.plot([e for e, _ in val], [a for _, a in val], "o", ms=7, mfc=S.PAPER, mec=color, mew=1.8, zorder=4)
            test = res["test"]["accuracy"]
            check(test, cited[arm], "{:.1%}")
            top.plot([epochs + 0.35], [test], "D", ms=6.5, color=color, zorder=4, clip_on=False)
            top.annotate(f"{test:.1%}", (epochs + 0.35, test), xytext=(8, 0), textcoords="offset points", va="center",
                         fontsize=9.5, fontweight="medium", color=S.JEV_TEXT if arm == "jev" else S.INK)
        top.set_title(title, fontsize=10.5)
        top.set_ylim(0.3, 1.02)
        top.set_xlim(0, epochs + 0.35)
        S.pct(top)
        bottom.set_ylim(0, 0.62)
        bottom.set_xlabel("epoch")
        bottom.set_xticks(range(0, epochs + 1, 2))
        for ax in (top, bottom):
            S.ygrid(ax)
        top.set_ylabel("accuracy")
        bottom.set_ylabel("training loss\n(trained through Jev)", fontsize=10.5)
    fig.text(0.0, -0.01, "Accuracy panels: lines are training batch accuracy, each network on its own neuron (smoothed over a quarter "
             "epoch, raw behind); rings are validation on Jev; diamonds are test on Jev.", ha="left", va="top", fontsize=9,
             color=S.INK2, wrap=True)
    axes[2].text(3.1, 0.93, "trained through Jev", fontsize=9.5, color=S.JEV_TEXT, ha="center")
    axes[2].text(3.1, 0.47, "trained exact\n(its twin)", fontsize=9.5, color=S.INK, ha="center", va="top", linespacing=1.1)
    axes[3].text(0.98, 0.9, "the twin's loss isn't drawn: a step neuron outputs\n0 or 1, so its loss only counts mistakes",
                    transform=axes[3].transAxes, ha="right", va="top", fontsize=9, color=S.MUTE, linespacing=1.3)
    S.save(fig, "training-curves.png")


def sparse_confusion():
    """Stage 8a: confusion on live Jev over the 1,000 test digits, trained through Jev and its exact-trained twin."""
    from jevrons.digits import mnist, test_subset
    from jevrons.net import decide
    from jevrons.stage8a import N_TEST
    _, _, _, yte = mnist()
    y = yte[test_subset(yte, N_TEST)]
    # The result files round each output to 2 decimals, which creates ties that the run broke on unrounded
    # values; the journal (gitignored) has every output call, so rebuild the exact outputs from it.
    outputs = {a: np.full((N_TEST, 10), np.nan) for a in ("jev", "swap")}
    with open(RUNS / "stage8a/journal.jsonl", "rb") as fh:
        for line in fh:
            if b'"split": "test"' not in line or b'"layer": 1' not in line:
                continue
            r = json.loads(line)
            t = r["tag"]
            if t.get("layer") == 1 and r.get("answers"):
                outputs[t["arm"]][t["i"], t["j"]] = np.mean([v["above"] if isinstance(v, dict) else v
                                                             for v in r["answers"].values()])
    assert not any(np.isnan(o).any() for o in outputs.values()), "journal is missing test output calls"
    fig, axes = S.figure(8.0, nrows=2)
    for ax, (arm, title, cited) in zip(axes, (("jev", "trained through Jev", "85.3%"), ("swap", "trained exact, run on Jev", "65.8%"))):
        r = js(f"stage8a/result-{arm}.json")["test"]
        pred = decide(outputs[arm])
        assert np.allclose([np.mean(pred[y == d] == d) for d in range(10)], r["per_digit"])
        check(np.mean(pred == y), cited, "{:.1%}")
        counts = np.zeros((10, 10), int)
        np.add.at(counts, (y, pred), 1)
        frac = counts / counts.sum(1, keepdims=True)
        rgb = np.ones((10, 10, 3))
        for i in range(10):
            for j in range(10):
                cmap = S.SEQ_INK if i == j else S.SEQ_JEV
                rgb[i, j] = cmap(frac[i, j] if i == j else min(frac[i, j] / 0.4, 1) * 0.6)[:3]
        ax.imshow(rgb, interpolation="none")
        for i in range(10):
            for j in range(10):
                if counts[i, j] and (i == j or frac[i, j] >= 0.05):
                    faint = i != j and frac[i, j] < 0.2
                    ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=9,
                            color=S.JEV_TEXT if faint else S.text_on(rgb[i, j]))
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.tick_params(length=0, labelsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{title}\n{r['accuracy']:.1%} of 1,000 test digits", fontsize=10.5, linespacing=1.3)
        ax.set_xlabel("predicted", fontsize=10)
    for ax in axes:
        ax.set_ylabel("true digit", fontsize=10)
    fig.text(0.5, -0.01, "cells: image counts; diagonal shaded grey, off-diagonal shaded pink by share of the row;\n"
             "off-diagonal counts shown at 5% of the row or more", ha="center", va="top", fontsize=9, color=S.MUTE)
    S.save(fig, "stage8a-confusion.png")


# ---------------------------------------------------------------- Two ways the weights adapt

def margin_dynamics():
    from jevrons.stage6b import p_intended, replay_swap
    dyn = js("stage6b/dynamics.json")
    panels = [("stage 6 seed 0", "ten digits, seed 0", "stage6/result-seed0-swap.json"),
              ("stage 6 seed 1", "ten digits, seed 1", "stage6/result-seed1-swap.json"),
              ("stage 5 0v1", "0 vs 1", "stage5/result-0v1-swap.json"),
              ("stage 5 3v8", "3 vs 8", "stage5/result-3v8-swap.json")]
    fig, grid = S.figure(7.8, ncols=2, nrows=4)
    axes = grid.T  # axes[metric, dataset]: one row per dataset, stacked
    for c, (key, title, swap_res) in enumerate(panels):
        rows = dyn[key]
        swap = rows["swap"]
        if not swap:  # the journal analysis ran before this twin existed; replay it (deterministic, local)
            by_epoch, same = replay_swap(6, int(key[-1]))
            assert same, "replayed twin differs from its saved weights"
            swap = []
            for e, (z, s, k) in sorted(by_epoch.items()):
                m = z / np.where(s > 0, s, 1)
                swap.append({"epoch": e + 1, "frac_abs_m_lt_1": float(np.mean(np.abs(m) < 1)),
                             "curve_predicted": float(p_intended(m, z, k).mean())})
        r = js(swap_res)
        measured = r["val"]["10"]["fires_as_intended"]["hidden"] if "val" in r else r["val_jev"]["fires_as_intended"]["hidden"]
        for arm, data, color in (("jev", rows["jev"], S.JEV), ("swap", swap, S.INK)):
            ep = [d["epoch"] for d in data]
            axes[0, c].plot(ep, [d["frac_abs_m_lt_1"] for d in data], "o-", color=color, ms=3.5, lw=1.8)
            axes[1, c].plot(ep, [d["curve_predicted"] for d in data], color=color, lw=1.2, ls=(0, (3, 2)))
            if arm == "jev":
                axes[1, c].plot(ep, [d["fires_as_intended"] for d in data], "o-", color=color, ms=3.5, lw=1.8)
        axes[1, c].plot([ep[-1]], [measured], "o", ms=6, mfc=S.PAPER, mec=S.INK, mew=1.6)
        axes[0, c].set_title(title, fontsize=10.5)
        axes[0, c].set_ylim(0, 0.52)
        axes[1, c].set_ylim(0.8, 1.0)
        n = len(ep)
        for r_ in (0, 1):
            ax = axes[r_, c]
            ax.set_xticks([1, n // 2, n] if n > 5 else range(1, n + 1))
            ax.tick_params(labelsize=9)
            S.ygrid(ax)
            S.pct(ax)
    for ax in grid[-1]:
        ax.set_xlabel("epoch", fontsize=10)
    for ax in grid[:, 0]:
        ax.set_ylabel("hidden sums with |m| < 1", fontsize=10)
    for ax in grid[:, 1]:
        ax.set_ylabel("hidden fires as intended", fontsize=10)
    s38 = dyn["stage 5 3v8"]
    for arm, first, last in (("jev", "32%", "13%"), ("swap", "49%", "38%")):  # the post says 39%; the data is 38.45%
        check(s38[arm][0]["frac_abs_m_lt_1"], first)
        check(s38[arm][-1]["frac_abs_m_lt_1"], last)
    s01 = dyn["stage 5 0v1"]
    for arm, first, last in (("jev", "31%", "2%"), ("swap", "36%", "18%")):
        check(s01[arm][0]["frac_abs_m_lt_1"], first)
        check(s01[arm][-1]["frac_abs_m_lt_1"], last)
    ax = axes[0, 3]
    ax.text(5, s38["swap"][-1]["frac_abs_m_lt_1"] + 0.05, "trained exact", ha="right", fontsize=9.5, color=S.INK2)
    ax.text(5, s38["jev"][-1]["frac_abs_m_lt_1"] - 0.05, "trained through Jev", ha="right", va="top", fontsize=9.5,
            color=S.JEV_TEXT)
    axes[1, 2].text(0.97, 0.04, "solid: measured on Jev\ndashed: predicted by\nthe margin curve\nhollow: exact twin,\nmeasured",
                    transform=axes[1, 2].transAxes, fontsize=9, color=S.INK2, linespacing=1.3, ha="right", va="bottom")
    S.save(fig, "stage6b-margin-dynamics.png")


def weights_3v8():
    from jevrons.stage5 import init
    from jevrons.stage6b import mosaic
    W0 = init(0)[0][0]
    Wj = weights("stage5/weights-3v8-jev.npz")[0]
    Ws = weights("stage5/weights-3v8-swap.npz")[0]
    dJ, dS = Wj - W0, Ws - W0
    check(np.linalg.norm(dJ), "42.3", "{:.1f}")
    check(np.linalg.norm(dS), "18.0", "{:.1f}")
    vw = float(np.quantile(np.abs(np.concatenate([W0, Wj, Ws])), 0.99))
    vd = float(np.quantile(np.abs(np.concatenate([dJ, dS])), 0.99))
    panels = [("random start", W0, vw), ("change, trained through Jev", dJ, vd),
              ("trained through Jev", Wj, vw), ("change, trained exact", dS, vd),
              ("trained exact", Ws, vw), ("difference of the changes, Jev minus exact", dJ - dS, vd)]
    fig, axes = S.figure(6.7, ncols=2, nrows=3)
    for ax, (title, W, v) in zip(axes.ravel(), panels):
        img = np.ma.masked_invalid(mosaic(W))
        cmap = S.DIVERGING.copy()
        cmap.set_bad(S.PAPER)
        im = ax.imshow(img, cmap=cmap, vmin=-v, vmax=v, interpolation="none")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if W is dJ or W is dS:
            ax.text(1.0, 1.02, f"Frobenius norm {np.linalg.norm(W):.1f}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9.5, color=S.JEV_TEXT if W is dJ else S.INK2)
        if ax in (axes[2, 0], axes[2, 1]):
            cb = fig.colorbar(im, ax=axes[:, list(axes[2]).index(ax)], orientation="horizontal", shrink=0.6, aspect=30,
                              pad=0.02)
            cb.set_ticks([-round(v, 1), 0, round(v, 1)])
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=9)
            cb.set_label("weight" if ax is axes[2, 0] else "change from the start (same scale for all three)",
                         fontsize=9.5, color=S.INK2)
    S.save(fig, "stage6b-weights-s5-3v8.png", svg=False, colors=128)


# ---------------------------------------------------------------- Quirks

def traps():
    fam = js("stage6b/traps.json")["families"]
    check(fam["last"]["control_misfire"], "82.5%", "{:.1%}")
    check(fam["first"]["trap_misfire"], "27.5%", "{:.1%}")
    check(fam["last"]["trap_misfire"], "4.5%", "{:.1%}")
    check(fam["vote"]["trap_misfire"], "99.0%", "{:.1%}")
    check(fam["long"]["trap_misfire"], "74.5%", "{:.1%}")
    # (label, value, is_trap): the three big-number placements share one list, so they read as one group
    groups = [("one big positive among 74 small negatives", [
                  ("in the middle", fam["last"]["control_misfire"], True),
                  ("first", fam["first"]["trap_misfire"], True),
                  ("last", fam["last"]["trap_misfire"], True)]),
              ("60 small positives, 15 large negatives", [
                  ("sign-vote trap", fam["vote"]["trap_misfire"], True),
                  ("matched control", fam["vote"]["control_misfire"], False)]),
              ("250 terms, one spread from zero", [
                  ("m = −1", fam["long"]["trap_misfire"], True),
                  ("mirror, m = +1", fam["long"]["control_misfire"], False)]),
              ("40 real terms at m = +0.3", [
                  ("110 zeros padded in", fam["zeros"]["trap_misfire"], True),
                  ("no padding", fam["zeros"]["control_misfire"], False)])]
    fig, ax = S.figure(4.5)
    y, ticks, labels = 0, [], []
    for head, bars in groups:
        ax.text(-0.005, y, head, fontsize=10, color=S.INK, fontweight="medium", ha="right", va="center",
                transform=ax.get_yaxis_transform())
        y -= 0.8
        for label, v, trap in bars:
            ax.barh(y, v, 0.68, color=S.JEV if trap else S.GREY, zorder=2)
            ax.text(v + 0.012, y, f"{v:.1%}".replace(".0%", "%"), va="center", fontsize=9.5,
                    color=S.JEV_TEXT if trap else S.INK2)
            ticks.append(y)
            labels.append(label)
            y -= 0.8
        y -= 0.45
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.08)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    S.pct(ax, "x")
    ax.grid(axis="x", color=S.FAINT, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlabel("misfire rate, 200 states per bar")
    S.save(fig, "stage6b-traps.png")


# ---------------------------------------------------------------- Teaching laya to add

def laya():
    return {"base": (S.GREY, "base laya"), "truth": (S.INK, "fine-tuned laya")}


def stage10_curves():
    fits = js("margin/summary.json")["offset_fit"]
    ev = {n: js(f"stage10/eval-{n}.json") for n in laya()}
    mm = np.linspace(-3, 3, 400)
    fig, axes = S.figure(3.2, ncols=3, sharey=True)
    for ax, n in zip(axes, (10, 30, 75)):
        ax.axhline(0.5, color=S.RULE, lw=0.8, zorder=0)
        ax.axvline(0, color=S.RULE, lw=0.8, zorder=0)
        for name, (color, _) in laya().items():
            c = ev[name]["curves"][f"probe-{n}"]
            b = np.array(c["binned"])
            ax.plot(b[:, 0] + 0.125, b[:, 1], "o", ms=2.8, color=color, alpha=0.9)
            ax.plot(mm, Phi(c["k"] * (mm - c["m0"])), color=color, lw=1.8)
        k, m0 = fits[str(n)]["k"], fits[str(n)]["threshold_m0"]
        ax.plot(mm, Phi(k * (mm - m0)), color=S.JEV, lw=1.8, ls=(0, (4, 2.5)))
        kt = ev["truth"]["curves"][f"probe-{n}"]["k"]
        ax.set_title(f"{n} terms", fontsize=10.5)
        ax.text(0.98, 0.3, f"fine-tuned\nk {kt:.1f}\nJev k {k:.1f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=S.INK2, linespacing=1.4)
        ax.set_xlim(-3, 3)
        ax.set_xticks([-3, 0, 3])
        ax.set_xlabel("m = z / spread", fontsize=10.5)
    check(ev["truth"]["curves"]["probe-10"]["k"], "13.5", "{:.1f}")
    check(ev["truth"]["curves"]["probe-10"]["sign_accuracy"], "98.7%", "{:.1%}")
    axes[0].set_ylabel("share of calls that fire", fontsize=10.5)
    S.pct(axes[0])
    ax = axes[0]
    ax.text(-2.85, 0.93, "base laya", fontsize=9, color=S.MUTE, va="center")
    ax.annotate("fine-tuned laya", xy=(0.02, 0.62), xytext=(-2.9, 0.62), fontsize=9, color=S.INK, va="center",
                arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.6))
    ax.annotate("Jev", xy=(0.12, 0.3), xytext=(-2.9, 0.3), fontsize=9, color=S.JEV_TEXT, va="center",
                arrowprops=dict(arrowstyle="-", color=S.JEV, lw=0.6))
    S.save(fig, "stage10-curves.png")


def stage10_accuracy():
    """Results doc only: sign accuracy against term count."""
    ev = {n: js(f"stage10/eval-{n}.json") for n in laya()}
    jev = js("margin/summary.json")["by_terms"]
    fig, ax = S.figure(3.4)
    ax.axvspan(2, 60, color=S.FAINT, lw=0, zorder=0)
    ax.text(2.2, 0.42, "trained term counts", fontsize=9, color=S.MUTE)
    for name, (color, label) in laya().items():
        ks = sorted(int(k.split("-")[1]) for k in ev[name]["curves"] if k.startswith("gen-"))
        acc = [ev[name]["curves"][f"gen-{k}"]["sign_accuracy"] for k in ks]
        ax.plot(ks, acc, "o-", color=color, ms=4, lw=1.8)
        ax.annotate(label, (ks[-1], acc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5,
                    color=S.INK2)
    ks = sorted(int(k) for k in jev)
    ax.plot(ks, [jev[str(k)]["sign_accuracy"] for k in ks], "o", color=S.JEV, ms=4, lw=1.8, ls=(0, (4, 2.5)))
    ax.annotate("Jev", (ks[-1], jev[str(ks[-1])]["sign_accuracy"]), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9.5, color=S.JEV_TEXT)
    ax.axvline(97, color=S.MUTE, lw=0.8, ls=(0, (1, 2)))
    ax.text(99, 0.43, "512-token\ncontext ends", fontsize=9, color=S.MUTE, linespacing=1.1)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 30, 60, 90, 150, 250], ["2", "5", "10", "30", "60", "90", "150", "250"])
    ax.minorticks_off()
    ax.set_xlim(1.8, 600)
    ax.spines["bottom"].set_bounds(1.8, 260)
    ax.set_ylim(0.4, 1.01)
    S.pct(ax)
    S.ygrid(ax)
    ax.set_xlabel("terms in the sum (150: context raised to 1,024)")
    ax.set_ylabel("sign accuracy")
    S.save(fig, "stage10-accuracy.png")


def stage10_training():
    """Results doc only: fine-tuning loss and validation accuracy by step, true-math arm."""
    fig, axes = S.figure(3.2, ncols=2)
    color, label = laya()["truth"]
    rows = [json.loads(line) for line in (RUNS / "stage10/truth/log.jsonl").open()]
    steps = [r for r in rows if "loss" in r]
    loss = np.convolve([r["loss"] for r in steps], np.ones(50) / 50, "valid")
    axes[0].plot(np.arange(len(loss)) + 50, loss, color=color, lw=1.6)
    axes[0].text(2100, 0.2, label, fontsize=9, color=S.INK2)
    evs = [r["eval"] for r in rows if "eval" in r]
    for key, ls in (("acc_s10", "-"), ("acc_s75", (0, (1, 1.5))), ("acc_real", (0, (4, 2)))):
        axes[1].plot([e["step"] for e in evs], [e[key] for e in evs], color=color, ls=ls, lw=1.5)
    axes[1].text(0.98, 0.05, "vs the true sign\nsolid: 10 terms\ndotted: 75 terms\ndashed: real journal states",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=9, color=S.INK2, linespacing=1.3)
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.03, 0.1, 0.3, 1], ["0.03", "0.1", "0.3", "1"])
    axes[0].minorticks_off()
    axes[0].set_xlabel("step (batch 16)")
    axes[0].set_ylabel("training loss, 50-step mean")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_ylim(0, 1.02)
    S.pct(axes[1])
    for ax in axes:
        S.ygrid(ax)
    S.save(fig, "stage10-training.png")


def stage10_traps():
    """Results doc only: trap misfires for Jev and the two laya models; ticks are the matched controls."""
    ev = {n: js(f"stage10/eval-{n}.json")["traps"] for n in laya()}
    jt = js("stage6b/traps.json")["families"]
    fams = [("last", "last", "big number last\n(75 terms)"), ("first", "first", "big number first\n(75)"),
            ("vote", "vote", "sign vote\n(75)"), ("long90", "long", "long list, m = \u22121\n(90; Jev: 250)"),
            ("zeros60", "zeros", "zero padding\n(60; Jev: 150)")]
    series = [("Jev", S.JEV, {f: jt[j] for f, j, _ in fams})] + [(laya()[n][1], laya()[n][0], ev[n]) for n in laya()]
    fig, ax = S.figure(3.4)
    x = np.arange(len(fams))
    w = 0.8 / len(series)
    for i, (label, color, t) in enumerate(series):
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, [t[f]["trap_misfire"] for f, _, _ in fams], w * 0.88, color=color, label=label, zorder=2)
        ax.scatter(xs, [t[f]["control_misfire"] for f, _, _ in fams], marker="_", s=90, lw=1.6, color=S.INK2, zorder=3,
                   label="matched control" if i == 0 else None)
    ax.set_xticks(x, [f[2] for f in fams], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.05)
    S.pct(ax)
    S.ygrid(ax)
    ax.set_ylabel("misfire rate, 200 states per bar")
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, handlelength=1, columnspacing=1,
              handletextpad=0.4)
    S.save(fig, "stage10-traps.png")


# ---------------------------------------------------------------- How far the weights bend

def bend():
    res = js("stage6c/bend.json")
    real = res["real"]
    for name, cited in (("plain Jev", "1.77"), ("8g E", "1.13"), ("10 laya-neuron (sparse)", "0.31")):
        check(real[name]["bend"]["W1"]["rel_dist"], cited, "{:.2f}")
    # (label, offset in points, alignment) for the points named in the left panel
    named = {"plain Jev": ("plain Jev", (-8, 0), "right"), "8g B": ("curve backprop", (7, -9), "left"),
             "8g C": ("bias shift", (-8, 3), "right"), "8g E": ("two questions averaged", (0, 12), "center"),
             "10 laya-neuron (sparse)": ("fine-tuned laya", (8, 0), "left"), "9c laya-hidden": ("base laya", (0, 13), "center")}
    fig, axes = S.figure(7.4, nrows=3)
    sims = list(res["sim"].values())
    for graded, color, ls, lab in ((True, S.GREY, "-", "simulated, graded"), (False, S.MUTE, (0, (3, 2)), "simulated, sampled")):
        pts = [v for v in sims if v["graded"] == graded]
        lams = sorted({v["lam"] for v in pts})
        for ax, xkey, ykey in ((axes[0], "swap_gap_pts", "rel_dist"), (axes[1], "twin_hidden_misfire", "rel_dist"),
                               (axes[2], "twin_hidden_misfire", "cos")):
            scale = 1 if xkey == "swap_gap_pts" else 100
            xs = [np.mean([v[xkey] for v in pts if v["lam"] == lam]) * scale for lam in lams]
            ys = [np.mean([v["bend"]["W1"][ykey] for v in pts if v["lam"] == lam]) for lam in lams]
            ax.plot(xs, ys, marker="o", color=color, ms=3, lw=1.1, ls=ls, zorder=1)
            if ax is axes[1]:
                ax.annotate(lab, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9, color=S.INK2,
                            va="center")
    for name, v in real.items():
        jev = name.startswith(("plain", "8g"))
        color = S.JEV if jev else S.INK
        marker = "s" if name.startswith("9c") else "o"
        for ax, x, y in ((axes[0], v["swap_gap_pts"], v["bend"]["W1"]["rel_dist"]),
                         (axes[1], 100 * v["twin_hidden_misfire"], v["bend"]["W1"]["rel_dist"]),
                         (axes[2], 100 * v["twin_hidden_misfire"], v["bend"]["W1"]["cos"])):
            ax.plot(x, y, marker, color=color, ms=6.5, mec=S.PAPER, mew=0.8, zorder=3)
            if ax is axes[0] and name in named:
                text, off, ha = named[name]
                ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=9, ha=ha, va="center",
                            color=S.JEV_TEXT if jev else S.INK)
    ax = axes[2]
    ax.text(14, 0.5, "Jev\nneurons", fontsize=9, color=S.JEV_TEXT, ha="center", va="top", linespacing=1.1)
    ax.text(65, 0.44, "base\nlaya", fontsize=9, color=S.INK, ha="center", va="bottom", linespacing=1.1)
    nf = res["noise_floor"]["lam=1.0"]["W1_rel_dist_between_seeds"]
    for ax in axes[:2]:
        ax.axhline(nf, color=S.MUTE, lw=0.8, ls=(0, (1, 2)), zorder=0)
        ax.set_ylim(0, 1.95)
        S.ygrid(ax)
    axes[0].text(46, nf - 0.03, "noise floor", fontsize=9, color=S.MUTE, ha="right", va="top")
    axes[0].set_ylabel("bend of the first-layer change", fontsize=10.5)
    axes[0].set_xlabel("swap gap (points)", fontsize=10.5)
    axes[0].set_xlim(-3, 47)
    axes[1].set_ylabel("bend of the first-layer change", fontsize=10.5)
    axes[1].set_xlabel("twin's hidden misfires", fontsize=10.5)
    axes[2].set_xlabel("twin's hidden misfires", fontsize=10.5)
    axes[2].set_ylabel("cosine with the twin's change", fontsize=10.5)
    axes[2].set_ylim(0, 1)
    S.ygrid(axes[2])
    for ax in axes[1:]:
        ax.set_xlim(0, 72)
        ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.PercentFormatter(100, decimals=0))
    for ax in axes:
        ax.tick_params(labelsize=9)
    S.save(fig, "stage6c-bend.png")


def pixels():
    from jevrons import stage5, stage6
    from jevrons.digits import mnist, split
    xtr, ytr, _, _ = mnist()
    tr, _ = split(ytr, range(10), 1000, 500)
    res = js("stage6c/pixels.json")
    X, y = xtr[tr], ytr[tr]
    freq = {d: (X[y == d] > 0).mean(0) for d in range(10)}
    mean_f = np.mean(list(freq.values()), 0)
    live = mean_f > 0.01

    def diff_map(init, jev, swap):
        w0 = init[0][0]
        return np.linalg.norm((weights(jev)[0] - w0) - (weights(swap)[0] - w0), axis=1)

    maps = [diff_map(stage6.init(s), f"stage6/weights-seed{s}-jev.npz", f"stage6/weights-seed{s}-swap.npz") for s in range(3)]
    m38 = diff_map(stage5.init(0), "stage5/weights-3v8-jev.npz", "stage5/weights-3v8-swap.npz")
    corr0 = float(np.corrcoef(maps[0][live], freq[0][live])[0, 1])
    assert abs(corr0 - res["s6 seed0"]["digits"]["0"]["corr_D_vs_on_freq_live"]) < 1e-9, "pixel maps drift from pixels.json"
    Dm = np.mean(maps, 0)
    fig, axes = S.figure(3.6, ncols=5, nrows=2)
    for ax in axes.ravel():
        ax.axis("off")

    def show(ax, img, cmap, title, sym=False, magnitude=False):
        if magnitude:  # the size of the difference: mask pixels never on, stretch over the live range
            img = np.ma.masked_where(~live, img)
            lo, hi = np.quantile(img.compressed(), [0.02, 0.99])
        else:
            hi = np.abs(img).max()
            lo = -hi if sym else 0
        cm = cmap.copy()
        cm.set_bad(S.PAPER)
        ax.imshow(img.reshape(28, 28), cmap=cm, vmin=lo, vmax=hi, interpolation="none")
        ax.set_title(title, fontsize=9, loc="center", linespacing=1.3)

    def r_(v):
        return f"r = {v:+.2f}".replace("-", "\u2212")

    show(axes[0, 0], Dm, S.SEQ_JEV, "Jev minus exact\nten digits, 3 seeds", magnitude=True)
    show(axes[1, 0], m38, S.SEQ_JEV, "Jev minus exact\n3 vs 8", magnitude=True)
    for c, d in enumerate((0, 3, 8, 1), start=1):
        r = np.mean([res[f"s6 seed{s}"]["digits"][str(d)]["corr_D_vs_on_freq_live"] for s in range(3)])
        show(axes[0, c], freq[d], S.SEQ_INK, f"{d}: how often on\n{r_(r)}")
        r = np.mean([res[f"s6 seed{s}"]["digits"][str(d)]["corr_D_vs_excess_on_freq"] for s in range(3)])
        show(axes[1, c], freq[d] - mean_f, S.DIVERGING, f"{d}: on more than avg.\n{r_(r)}", sym=True)
    fig.text(1.0, -0.01, "r: correlation with the ten-digit map (top left) over pixels that are ever on, mean of 3 seeds",
             ha="right", va="top", fontsize=9, color=S.MUTE)
    S.save(fig, "stage6c-pixels.png", svg=False)


FIGURES = {
    "margin_by_terms": margin_by_terms, "bias": bias, "wording": wording, "xor": xor, "stage3": stage3_boundary,
    "stage5": stage5_swap, "stage5_training": stage5_training, "stage5_margins": stage5_margins,
    "stage6": stage6_swap, "stage6_digits": stage6_digits, "stage7": stage7, "stage8_sparse": stage8_sparse,
    "training_curves": training_curves, "sparse_confusion": sparse_confusion,
    "dynamics": margin_dynamics, "weights": weights_3v8, "traps": traps, "stage10": stage10_curves,
    "stage10_accuracy": stage10_accuracy, "stage10_training": stage10_training, "stage10_traps": stage10_traps,
    "bend": bend, "pixels": pixels,
}

def main(argv):
    args = list(argv)
    theme = "docs"
    if "--theme" in args:
        i = args.index("--theme")
        theme = args[i + 1]
        del args[i:i + 2]
    themes = ("light", "dark") if theme == "both" else (theme,)
    assert all(t in S.THEMES for t in themes), f"--theme is one of docs, light, dark, both; got {theme}"
    for t in themes:
        S.use(t)
        for name in args or FIGURES:
            FIGURES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
