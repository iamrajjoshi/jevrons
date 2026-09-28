"""One visual system for every Jevrons figure, matched to demo/web/style.css.

Colour has one meaning everywhere: pink is Jev (a network trained through Jev, or Jev's own answers),
ink is the exact neuron or the exact-trained network, and a lighter tint of either is the secondary
case (the swapped twin, an earlier stage, a simulation). Diverging maps (weights) use slate and brick
so they never borrow Jev's pink. Figures are sized for the blog's 720 px column and exported at 2x.
JEVRONS_FIG_THEME=dark renders on the demo's dark paper instead.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FIG = Path(__file__).resolve().parents[2] / "docs" / "figures"
WIDTH = 7.2  # inches; 720 px column at 100 px/in, exported at 200 dpi
DPI = 200

_THEMES = {
    "light": dict(paper="#f6f6f3", ink="#141414", ink2="#4d4d4a", mute="#8f8f8a", rule="#d9d9d4", faint="#e9e9e4",
                  jev="#e0467a", jev_text="#b8285a", jev_light="#f2a9c1", grey="#b4b4ae", slate="#3f6e9a", brick="#b5553c"),
    "dark": dict(paper="#0f0f0f", ink="#ededea", ink2="#b3b3ae", mute="#75756f", rule="#2d2d2b", faint="#1c1c1b",
                 jev="#f2769c", jev_text="#f58fb0", jev_light="#8c3f59", grey="#5c5c58", slate="#6f9bc6", brick="#d27a5f"),
}
T = _THEMES[os.environ.get("JEVRONS_FIG_THEME", "light")]
PAPER, INK, INK2, MUTE, RULE, FAINT = T["paper"], T["ink"], T["ink2"], T["mute"], T["rule"], T["faint"]
JEV, JEV_TEXT, JEV_LIGHT = T["jev"], T["jev_text"], T["jev_light"]
EXACT, GREY = INK, T["grey"]
SLATE, BRICK = T["slate"], T["brick"]

SEQ_INK = LinearSegmentedColormap.from_list("seq_ink", [PAPER, INK])
SEQ_JEV = LinearSegmentedColormap.from_list("seq_jev", [PAPER, JEV, "#5e1230" if PAPER != "#0f0f0f" else "#ffd6e3"])
DIVERGING = LinearSegmentedColormap.from_list("div", [SLATE, PAPER, BRICK])  # signed values: weights
FIRE = LinearSegmentedColormap.from_list("fire", [SLATE, PAPER, JEV])  # Jev's p, 0.5 at the midpoint


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Inter", "Arial", "DejaVu Sans"],
        "font.size": 9.5, "axes.labelsize": 11, "axes.titlesize": 10.5, "axes.titleweight": "medium",
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9, "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.labelpad": 5,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": MUTE, "axes.linewidth": 0.8,
        "xtick.color": MUTE, "ytick.color": MUTE, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": FAINT, "grid.linewidth": 0.8,
        "lines.linewidth": 2, "lines.markersize": 5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "axes.unicode_minus": True,
        "svg.fonttype": "path", "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.06, "figure.constrained_layout.w_pad": 0.06,
        "axes.prop_cycle": matplotlib.cycler(color=[JEV, INK, GREY, SLATE]),
    })


def figure(height, ncols=1, nrows=1, width=WIDTH, **kw):
    apply()
    return plt.subplots(nrows, ncols, figsize=(width, height), **kw)


def ygrid(ax):
    """Light horizontal gridlines behind the data."""
    ax.grid(axis="y", color=FAINT, lw=0.8)
    ax.set_axisbelow(True)


def pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, name, svg=True, colors=None):
    """PNG at 2x (plus SVG when the figure is mostly vector). `colors` quantizes the PNG to a palette,
    for pixel-map figures whose noise defeats PNG compression."""
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    if colors:
        from PIL import Image
        Image.open(out).convert("RGB").quantize(colors, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)
    if svg:
        with plt.rc_context({"svg.hashsalt": name}):  # stable ids and no date, so unchanged figures stay unchanged
            fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    plt.close(fig)
    kb = out.stat().st_size / 1024
    print(f"{out.relative_to(FIG.parents[1])}  {kb:.0f} KB" + (f", svg {out.with_suffix('.svg').stat().st_size / 1024:.0f} KB" if svg else ""))
