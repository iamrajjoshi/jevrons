"""One visual system for every Jevrons figure.

Colour has one meaning everywhere: teal (the site's link colour) is Jev (a network trained through Jev,
or Jev's own answers), ink is the exact neuron or the exact-trained network, and a lighter tint of either
is the secondary case (the swapped twin, an earlier stage, a simulation). Diverging maps (weights) use
slate and brick so they never borrow Jev's teal; Jev's own p runs from the site's amber (false) to teal (true).

Three themes. "docs" is opaque on the demo's cool paper, for docs/figures, which GitHub shows on white.
"light" and "dark" are for rajjoshi.me: transparent, in the site's warm ink and cream, and drawn at the
blog column's real size (672 px, exported at 2x) so type reads at 1:1.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FIG = Path(__file__).resolve().parents[2] / "docs" / "figures"
SITE = FIG / "site"  # light and dark variants for the blog, copied into the site's assets; not tracked
WIDTH = 6.72  # inches; the blog's 672 px column at 100 px/in, exported at 200 dpi (2x)
DPI = 200

THEMES = {
    "docs": dict(paper="#f6f6f3", bg="#f6f6f3", ink="#141414", ink2="#4d4d4a", mute="#6f6f6a", rule="#d9d9d4",
                 faint="#e9e9e4", jev="#0f766e", jev_text="#0f766e", jev_light="#3a9c92", jev_deep="#0a3f3a",
                 grey="#b4b4ae", slate="#3f6e9a", brick="#b5553c", amber="#765018"),
    # The site's paper (#fff9eb) and ink (#28221c); ink2 and mute are its muted and faint text colours,
    # jev its link teal and amber its warning callout.
    "light": dict(paper="#fff9eb", bg="none", ink="#28221c", ink2="#56534d", mute="#756f67", rule="#d9cdb8",
                  faint="#efe5d2", jev="#0f766e", jev_text="#0f766e", jev_light="#3d9f95", jev_deep="#0a3f3a",
                  grey="#aea390", slate="#3f6e9a", brick="#b5553c", amber="#765018"),
    "dark": dict(paper="#28221c", bg="none", ink="#fff9eb", ink2="#b9ad9a", mute="#9a8f80", rule="#4d443a",
                 faint="#352e27", jev="#67c7ba", jev_text="#83d4ca", jev_light="#3f8a80", jev_deep="#a3ddd4",
                 grey="#786e61", slate="#6f9bc6", brick="#d27a5f", amber="#d4a45e"),
}
THEME = "docs"


def use(theme):
    """Switch every colour and colormap below to `theme` (figures read them at draw time)."""
    global THEME, T, BG, PAPER, INK, INK2, MUTE, RULE, FAINT, JEV, JEV_TEXT, JEV_LIGHT, EXACT, GREY, SLATE, BRICK
    global SEQ_INK, SEQ_JEV, DIVERGING, FIRE
    THEME, T = theme, THEMES[theme]
    BG, PAPER, INK, INK2, MUTE, RULE, FAINT = T["bg"], T["paper"], T["ink"], T["ink2"], T["mute"], T["rule"], T["faint"]
    JEV, JEV_TEXT, JEV_LIGHT = T["jev"], T["jev_text"], T["jev_light"]
    EXACT, GREY = INK, T["grey"]
    SLATE, BRICK = T["slate"], T["brick"]
    SEQ_INK = LinearSegmentedColormap.from_list("seq_ink", [PAPER, INK])
    SEQ_JEV = LinearSegmentedColormap.from_list("seq_jev", [PAPER, JEV, T["jev_deep"]])
    DIVERGING = LinearSegmentedColormap.from_list("div", [SLATE, PAPER, BRICK])  # signed values: weights
    FIRE = LinearSegmentedColormap.from_list("fire", [T["amber"], PAPER, JEV])  # Jev's p, 0.5 at the midpoint


use(THEME)


def contrast(a, b):
    """WCAG contrast ratio of two colours."""
    from matplotlib.colors import to_rgb
    lum = lambda c: sum(w * (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for w, v in zip((0.2126, 0.7152, 0.0722), to_rgb(c)))
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def text_on(rgb, prefer=None):
    """Text kwargs for a number on a cell of colour `rgb`: `prefer` if it reads at 4.5:1, else INK or PAPER,
    whichever reads better. Mid-tone cells where neither reaches 4.5:1 get a thin halo of the other one."""
    from matplotlib import patheffects
    rgb = tuple(rgb[:3])
    if prefer and contrast(prefer, rgb) >= 4.5:
        return dict(color=prefer)
    color, halo = sorted((INK, PAPER), key=lambda c: contrast(c, rgb), reverse=True)
    if contrast(color, rgb) >= 4.5:
        return dict(color=color)
    return dict(color=color, path_effects=[patheffects.withStroke(linewidth=1.6, foreground=halo)])


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Inter", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 10.5, "axes.titleweight": "medium",
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.labelpad": 5,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
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
    for pixel-map figures whose noise defeats PNG compression. The site themes write to docs/figures/site/
    as name.png/.svg (light) and name-dark.png/.svg (dark); their SVG is written for every figure, with
    text as paths, pixel maps embedded unscaled (drawn pixelated), and its size in CSS px at the column's 100 px/in."""
    site = THEME != "docs"
    out = (SITE / (name.replace(".png", "-dark.png") if THEME == "dark" else name)) if site else FIG / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.06)
    if colors and not site:  # the site re-encodes every image anyway
        from PIL import Image
        Image.open(out).convert("RGB").quantize(colors, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)
    svg = svg or site
    if svg:
        vec = out.with_suffix(".svg")
        with plt.rc_context({"svg.hashsalt": name}):  # stable ids and no date, so unchanged figures stay unchanged
            fig.savefig(vec, bbox_inches="tight", pad_inches=0.06 if site else 0.12, metadata={"Date": None})
        if site:  # matplotlib sizes SVGs in pt (72/in); the page is laid out at 100 px/in, like the PNG at 2x
            import re
            px = lambda m: f'{m[1]}="{float(m[2]) * 100 / 72:.2f}px"'
            head, body = vec.read_text().split("<svg", 1)
            tag, rest = body.split(">", 1)
            vec.write_text(head + "<svg" + re.sub(r'(width|height)="([\d.]+)pt"', px, tag) + ">" + rest)
    plt.close(fig)
    kb = lambda p: f"{p.stat().st_size / 1024:.0f} KB"
    print(f"{out.relative_to(FIG.parents[1])}  {kb(out)}" + (f", svg {kb(out.with_suffix('.svg'))}" if svg else ""))
