"""One visual system for every Jevrons figure.

Colour has one meaning everywhere: teal (the site's link colour) is Jev (a network trained through Jev, or
Jev's own answers); coral is exact (the exact neuron, perfect math, the exact-trained twin, laya fine-tuned on
exact answers); ink or cream is a neutral reference (targets, base laya, simulations). Grey is only for axes,
grids and labels, never data: a secondary series (1,000-image rows, simulations) is drawn hollow or dashed in its
parent's colour. Jev's own p runs from the calm page (false) to strong teal (true). Signed maps (weights) run
slate to brown, so they borrow neither teal nor coral. Lists of different lengths (10 to 250 terms) each get an
Okabe-Ito-derived colour, TERMS, kept apart from each other and from coral, for deuteranopes too.

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

# terms: one colour per Jev list length (10, 30, 75, 150, 250 terms), as (line, text) pairs: ochre, sky blue,
# bluish green, reddish purple, blue. Neighbours are >= 15 CIEDE2000 apart and every one is >= 12 from coral,
# in normal vision and under deuteranopia; lines are >= 3:1 on the page and text >= 4.5:1.
_TERMS_LIGHT = (("#7a5c00", "#7a5c00"), ("#2b86c0", "#1a6fa8"), ("#009e73", "#00805e"), ("#a8508a", "#a8508a"),
                ("#004f80", "#004f80"))
THEMES = {
    "docs": dict(paper="#f6f6f3", bg="#f6f6f3", ink="#141414", ink2="#4d4d4a", mute="#6f6f6a", rule="#d9d9d4",
                 faint="#e9e9e4", jev="#0f766e", jev_text="#0f766e", jev_deep="#0a3f3a",
                 exact="#d9480f", exact_text="#c43f0b", exact_deep="#6e2406", slate="#3f6e9a", brown="#8a5a1a",
                 fire=((0, "#f6f6f3"), (0.5, "#cfe8df"), (0.72, "#2fa596"), (1, "#0b5a53")), terms=_TERMS_LIGHT),
    # The site's paper (#fff9eb) and ink (#28221c); ink2 and mute are its muted and faint text colours,
    # jev its link teal.
    "light": dict(paper="#fff9eb", bg="none", ink="#28221c", ink2="#56534d", mute="#756f67", rule="#d9cdb8",
                  faint="#efe5d2", jev="#0f766e", jev_text="#0f766e", jev_deep="#0a3f3a",
                  exact="#d9480f", exact_text="#c43f0b", exact_deep="#6e2406", slate="#3f6e9a", brown="#8a5a1a",
                  fire=((0, "#fff9eb"), (0.5, "#cfe8df"), (0.72, "#2fa596"), (1, "#0b5a53")), terms=_TERMS_LIGHT),
    "dark": dict(paper="#28221c", bg="none", ink="#fff9eb", ink2="#b9ad9a", mute="#9a8f80", rule="#4d443a",
                 faint="#352e27", jev="#67c7ba", jev_text="#83d4ca", jev_deep="#a3ddd4",
                 exact="#ff8a5c", exact_text="#ff8a5c", exact_deep="#ffd6c4", slate="#6f9bc6", brown="#b8944a",
                 fire=((0, "#28221c"), (0.5, "#33443c"), (0.72, "#1f9a8a"), (1, "#6fe0cf")),
                 terms=(("#9a7a10", "#b8962c"), ("#56b4e9", "#56b4e9"), ("#1fbf8f", "#1fbf8f"), ("#d98cbb", "#d98cbb"),
                        ("#3f8fd8", "#3f8fd8"))),
}
THEME = "docs"
TERM_COUNTS = (10, 30, 75, 150, 250)


def use(theme):
    """Switch every colour and colormap below to `theme` (figures read them at draw time)."""
    global THEME, T, BG, PAPER, INK, INK2, MUTE, RULE, FAINT, JEV, JEV_TEXT, EXACT, EXACT_TEXT, SLATE, BROWN
    global SEQ_JEV, SEQ_EXACT, SEQ_SLATE, DIVERGING, FIRE, TERMS, TERMS_TEXT
    THEME, T = theme, THEMES[theme]
    BG, PAPER, INK, INK2, MUTE, RULE, FAINT = T["bg"], T["paper"], T["ink"], T["ink2"], T["mute"], T["rule"], T["faint"]
    JEV, JEV_TEXT, EXACT, EXACT_TEXT = T["jev"], T["jev_text"], T["exact"], T["exact_text"]
    SLATE, BROWN = T["slate"], T["brown"]
    TERMS = dict(zip(TERM_COUNTS, (line for line, _ in T["terms"])))
    TERMS_TEXT = dict(zip(TERM_COUNTS, (text for _, text in T["terms"])))
    SEQ_JEV = LinearSegmentedColormap.from_list("seq_jev", [PAPER, JEV, T["jev_deep"]])
    SEQ_EXACT = LinearSegmentedColormap.from_list("seq_exact", [PAPER, EXACT, T["exact_deep"]])
    SEQ_SLATE = LinearSegmentedColormap.from_list("seq_slate", [PAPER, SLATE, INK])  # neutral image data (pixels)
    DIVERGING = LinearSegmentedColormap.from_list("div", [SLATE, PAPER, BROWN])  # signed values: weights
    FIRE = LinearSegmentedColormap.from_list("fire", T["fire"])  # Jev's p: calm below 0.5, strong teal at 1


def halo(width=1.6, color=None):
    """Path effect: a thin outline in the page colour, so marks stay visible on any cell of a map."""
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground=color or PAPER)]


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
    rgb = tuple(rgb[:3])
    if prefer and contrast(prefer, rgb) >= 4.5:
        return dict(color=prefer)
    color, other = sorted((INK, PAPER), key=lambda c: contrast(c, rgb), reverse=True)
    if contrast(color, rgb) >= 4.5:
        return dict(color=color)
    return dict(color=color, path_effects=halo(1.6, other))


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
        "axes.prop_cycle": matplotlib.cycler(color=[JEV, EXACT, INK, SLATE]),
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
