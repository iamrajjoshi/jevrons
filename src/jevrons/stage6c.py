"""Stage 6c: the trained weights as a probe of Jev's bias and errors. Offline; no Jev or Ollaya calls.

Every trained network has an exact twin (same init, data and schedule, exact step neuron). This compares
each network's net change from init with its twin's, and reads Jev's measured curve against them.

  bend      how far each 3 vs 8 arm moved away from its twin, against how faulty its neuron is, with a
            family of simulated neurons of known faultiness (Jev's measured curve, scaled) as the reference
  misfire   per-neuron predicted misfires (margin curve) against measured ones (journals), by direction
  lean      why Jev-trained sums drift up: output-layer compensation, near-constant units, graded-p dynamics
  readoff   can Jev's threshold drift m0(n) be read from the trained weights alone?
  pixels    which pixels' weights differ most between arms, against which digits use those pixels
Usage: uv run python -m jevrons.stage6c [bend misfire lean readoff pixels]  (default: all)
Needs runs/stage6c/calls.npz (uv run python -m jevrons.stage6c_calls). Numbers: runs/stage6c/<part>.json;
figures: docs/figures/stage6c-*.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jevrons import figstyle as S  # noqa: E402  (docs palette; these figures keep their own layout)
from jevrons import stage5, stage6  # noqa: E402
from jevrons.digits import mnist, split  # noqa: E402
from jevrons.net import StepNeuron, fit, forward  # noqa: E402
from jevrons.stage6b import hidden_margins, mosaic  # noqa: E402
from jevrons.stage8g import curve, slope_for, terms_and_spread  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT, FIG = ROOT / "runs" / "stage6c", ROOT / "docs" / "figures"
ACCENT, INK, EXACT, BLUE = S.JEV, S.INK, S.EXACT, S.SLATE
XTR, YTR, _, _ = mnist()
TR38, VA38 = split(YTR, (3, 8), 500, 500)
TR10, VA10 = split(YTR, range(10), 1000, 500)
SWAP38_EXACT = 0.928  # stage 5: the 3v8 exact twin on an exact neuron


def Phi(t):
    """Normal CDF via Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7), vectorized."""
    x = np.abs(t) / np.sqrt(2)
    u = 1 / (1 + 0.3275911 * x)
    y = 1 - ((((1.061405429 * u - 1.453152027) * u + 1.421413741) * u - 0.284496736) * u + 0.254829592) * u * np.exp(-x * x)
    return 0.5 * (1 + np.sign(t) * y)


def save(part, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{part}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(json.dumps(obj, indent=1, default=float)[:6000])


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print(FIG / name)


def load(path):
    w = np.load(ROOT / path)
    return [w[f"arr_{i}"] for i in range(4)]


def flat(params):
    return [q for layer in params for q in layer]


def pairs(w):
    return [(w[0], w[1]), (w[2], w[3])]


def acc(w, X, y, neuron=None):
    out = forward(pairs(w), X, neuron or StepNeuron(), {})[0][-1]
    return float(np.mean((out[:, 0] >= 0.5) == (y >= 0.5))) if out.shape[1] == 1 else float(np.mean(np.argmax(out, 1) == y))


def data38():
    return XTR[TR38], (YTR[TR38] == 8).astype(float)[:, None], XTR[VA38], (YTR[VA38] == 8).astype(float)


def s10_init():
    """Stage 10's sparse start: stage5.init(0) with each hidden neuron's top 64 initial weights kept (laya/payoff.py)."""
    w = flat(stage5.init(0))
    active = (np.mean(XTR[TR38] > 0, 0) >= 0.05).astype(float)
    idx = np.argsort(-np.abs(w[0]) * active[:, None], 0)[:64]
    mask = np.zeros_like(w[0])
    np.put_along_axis(mask, idx, 1.0, 0)
    return [w[0] * mask, *w[1:]], mask


class CurveNeuron:
    """Jev's measured curve with its faultiness scaled by lam: P(fire) = Phi(k(n)/lam * (m - lam * m0(n))).
    graded: return that probability (Jev reports a graded p); else sample fire/silence as 0.97/0.03.
    lean=False drops the threshold drift and keeps the blur. floor squeezes a graded p into [floor, 1 - floor],
    as Jev's p is (it rarely leaves 0.04-0.96)."""

    def __init__(self, lam=1.0, graded=True, lean=True, seed=0, floor=0.0):
        self.lam, self.graded, self.lean, self.rng, self.floor = lam, graded, lean, np.random.default_rng(seed), floor

    def __call__(self, X, W, b, tag=None):
        z = X @ W + b
        n, spread = terms_and_spread(X, W, b)
        k, m0 = curve(n)
        p = Phi(k / self.lam * (z / spread - (self.lam * m0 if self.lean else 0)))
        return self.floor + (1 - 2 * self.floor) * p if self.graded else (self.rng.random(p.shape) < p) * 0.94 + 0.03


def train38(neuron, slope=None, start=None):
    X, Y, _, _ = data38()
    return flat(fit(start or stage5.init(0), X, Y, neuron, stage5.EPOCHS, stage5.BATCH, stage5.LR, stage5.TAU, stage5.SEED,
                    slope=slope))


def train10(neuron, seed=0):
    return flat(fit(stage6.init(seed), XTR[TR10], np.eye(10)[YTR[TR10]], neuron, stage6.EPOCHS, stage6.BATCH, stage6.LR,
                    stage6.TAU, seed))


def bend(w0, wa, ws):
    """Change of an arm from init against its exact twin's, per layer. explained = cos^2 (least-squares share)."""
    out = {}
    for name, idx in (("W1", [0]), ("b1", [1]), ("W2", [2]), ("b2", [3]), ("all", [0, 1, 2, 3])):
        dA = np.concatenate([(wa[i] - w0[i]).ravel() for i in idx])
        dS = np.concatenate([(ws[i] - w0[i]).ravel() for i in idx])
        c = float(dA @ dS / (np.linalg.norm(dA) * np.linalg.norm(dS) + 1e-12))
        out[name] = {"rel_dist": float(np.linalg.norm(dA - dS) / np.linalg.norm(dS)), "cos": c, "explained": c * c,
                     "scale": float(np.linalg.norm(dA) / np.linalg.norm(dS))}
    return out


def sign_disagreement(wa, ws, X):
    """Share of (image, hidden neuron) whose intended sum has a different sign in the two networks."""
    return float(np.mean(((X @ wa[0] + wa[1]) > 0) != ((X @ ws[0] + ws[1]) > 0)))


def expected_misfire(w, X, neuron):
    """Mean P(hidden answer disagrees with the true sign) under a CurveNeuron (its expectation, not a sample)."""
    z = X @ w[0] + w[1]
    n, spread = terms_and_spread(X, w[0], w[1])
    k, m0 = curve(n)
    lam = neuron.lam
    pf = Phi(k / lam * (z / spread - (lam * m0 if neuron.lean else 0)))
    return float(np.mean(np.where(z > 0, 1 - pf, pf)))


# ---------------------------------------------------------------- 1. bend against faultiness

def real_arms():
    """3 vs 8 arms: (weights, twin, init, swap gap in points, twin hidden misfire on the arm's neuron, trained hidden misfire)."""
    i5 = flat(stage5.init(0))
    twin = load("runs/stage5/weights-3v8-swap.npz")
    s10, _ = s10_init()
    g8 = {a: json.loads((ROOT / f"runs/stage8g/result-{a}.json").read_text()) for a in "BCDE"}
    s5 = {a: json.loads((ROOT / f"runs/stage5/result-3v8-{a}.json").read_text()) for a in ("jev", "swap")}
    p10 = json.loads((ROOT / "runs/stage10/payoff-truth-n64/result.json").read_text())
    s9 = json.loads((ROOT / "runs/stage9c/summary.json").read_text())
    plain_twin_mis = 1 - s5["swap"]["val_jev"]["fires_as_intended"]["hidden"]
    _, _, Xv, _ = data38()
    from jevrons.stage9c import laya_curve
    zt = np.round(Xv @ twin[0] + twin[1], 2)
    laya_mis = float(np.mean((laya_curve(zt) >= 0.5) != (zt > 0)))  # laya is deterministic: its table is its answer
    arms = {"plain Jev": ("runs/stage5/weights-3v8-jev.npz", 25.0, plain_twin_mis, 1 - s5["jev"]["val_jev"]["fires_as_intended"]["hidden"])}
    for a in "BCDE":
        gap = round(100 * (SWAP38_EXACT - g8[a]["swap"]["accuracy"]), 1) if "swap" in g8[a] else 25.0
        tm = 1 - g8[a]["swap"]["fires_as_intended"]["hidden"] if "swap" in g8[a] else plain_twin_mis
        arms[f"8g {a}"] = (f"runs/stage8g/weights-{a}.npz", gap, tm, 1 - g8[a]["trained"]["fires_as_intended"]["hidden"])
    for a in ("laya-hidden", "all-laya"):
        arms[f"9c {a}"] = (f"runs/stage9c/weights-{a}.npz", round(100 * (SWAP38_EXACT - s9[f"{a}/swap"]["val_on_laya"]), 1), laya_mis, None)
    out = {name: (load(p), twin, i5, gap, tm, trm) for name, (p, gap, tm, trm) in arms.items()}
    out["10 laya-neuron (sparse)"] = (load("runs/stage10/payoff-truth-n64/weights-laya.npz"),
                                      load("runs/stage10/payoff-truth-n64/weights-swap.npz"), s10,
                                      round(100 * (p10["swap"]["val_exact_step"] - p10["swap"]["val_laya"]["accuracy"]), 1),
                                      1 - p10["swap"]["val_laya"]["fires_as_intended"]["hidden"],
                                      1 - p10["laya"]["val_laya"]["fires_as_intended"]["hidden"])
    return out


def run_bend():
    X, Y, Xv, yv = data38()
    i5 = flat(stage5.init(0))
    twin = load("runs/stage5/weights-3v8-swap.npz")
    replay = train38(StepNeuron())
    assert all(np.allclose(a, b) for a, b in zip(replay, twin)), "exact twin replay differs from saved weights"
    res = {"real": {}, "recipe_twins": {}, "sim": {}}
    for name, (wa, ws, w0, gap, tm, trm) in real_arms().items():
        res["real"][name] = {"swap_gap_pts": gap, "twin_hidden_misfire": tm, "trained_hidden_misfire": trm,
                             "bend": bend(w0, wa, ws), "hidden_sign_disagreement_val": sign_disagreement(wa, ws, Xv)}
    # stage 8g changed the backward pass as well as the neuron; an exact twin with each arm's backward pass
    # separates the two (the neuron is then the only difference)
    for name, comp in (("8g B", False), ("8g C", True), ("8g D", True), ("8g E", True)):
        rt = train38(StepNeuron(), slope_for(comp))
        wa = load(f"runs/stage8g/weights-{name[-1]}.npz")
        res["recipe_twins"][name] = {"bend_vs_recipe_twin": bend(i5, wa, rt), "recipe_twin_vs_plain_twin": bend(i5, rt, twin),
                                     "recipe_twin_exact_acc": acc(rt, Xv, yv)}
    # reference: Jev's measured curve with faultiness scaled by lam, trained with stage 5's recipe
    for graded in (True, False):
        for lam in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            for seed in ((0,) if graded else (0, 1, 2)):
                nrn = CurveNeuron(lam, graded, True, seed)
                wa = train38(nrn)
                ev = [CurveNeuron(lam, graded, True, 100 + r) for r in range(1 if graded else 5)]
                gap = 100 * (SWAP38_EXACT - np.mean([acc(twin, Xv, yv, e) for e in ev]))
                res["sim"][f"{'graded' if graded else 'binary'} lam={lam} seed={seed}"] = {
                    "graded": graded, "lam": lam, "seed": seed, "swap_gap_pts": float(gap),
                    "twin_hidden_misfire": expected_misfire(twin, Xv, nrn), "trained_hidden_misfire": expected_misfire(wa, Xv, nrn),
                    "trained_acc_on_neuron": float(np.mean([acc(wa, Xv, yv, e) for e in ev])),
                    "bend": bend(i5, wa, twin), "hidden_sign_disagreement_val": sign_disagreement(wa, twin, Xv), "_w": wa}
    # noise floor: two binary runs at the same lam differ only in the neuron's coin flips
    res["noise_floor"] = {}
    for lam in (0.5, 1.0, 2.0):
        ws = [res["sim"][f"binary lam={lam} seed={s}"]["_w"] for s in range(3)]
        rel = [np.linalg.norm((ws[a][0] - ws[b][0])) / np.linalg.norm(twin[0] - i5[0]) for a, b in ((0, 1), (0, 2), (1, 2))]
        res["noise_floor"][f"lam={lam}"] = {"W1_rel_dist_between_seeds": float(np.mean(rel))}
    for v in res["sim"].values():
        v.pop("_w")
    save("bend", res)
    plot_bend(res)


def plot_bend(res):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    sims = res["sim"]
    for graded, color, lab in ((True, INK, "simulated: Jev's curve scaled, graded p"), (False, BLUE, "simulated: scaled, sampled yes/no")):
        pts = [v for v in sims.values() if v["graded"] == graded]
        lams = sorted({v["lam"] for v in pts})
        for ax, xkey in ((axes[0], "swap_gap_pts"), (axes[1], "twin_hidden_misfire")):
            xs = [np.mean([v[xkey] for v in pts if v["lam"] == lam]) for lam in lams]
            ys = [np.mean([v["bend"]["W1"]["rel_dist"] for v in pts if v["lam"] == lam]) for lam in lams]
            if xkey == "twin_hidden_misfire":
                xs = [100 * x for x in xs]
            ax.plot(xs, ys, "o-", color=color, ms=3, lw=1, label=lab)
            for x, y, lam in zip(xs, ys, lams):
                ax.annotate(f"λ={lam:g}", (x, y), fontsize=6, color=color, xytext=(3, -8), textcoords="offset points")
        xs = [np.mean([v["twin_hidden_misfire"] for v in pts if v["lam"] == lam]) * 100 for lam in lams]
        axes[2].plot(xs, [np.mean([v["bend"]["W1"]["cos"] for v in pts if v["lam"] == lam]) for lam in lams], "o-", color=color, ms=3, lw=1)
    for name, v in res["real"].items():
        color = ACCENT if name.startswith(("plain", "8g")) else INK if name.startswith("9c") else EXACT
        marker = "s" if name.startswith("10") else ("D" if name.startswith("9c") else "o")
        for ax, x, y in ((axes[0], v["swap_gap_pts"], v["bend"]["W1"]["rel_dist"]),
                         (axes[1], 100 * v["twin_hidden_misfire"], v["bend"]["W1"]["rel_dist"]),
                         (axes[2], 100 * v["twin_hidden_misfire"], v["bend"]["W1"]["cos"])):
            ax.plot(x, y, marker, color=color, ms=6)
            ax.annotate(name, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    nf = res["noise_floor"]["lam=1.0"]["W1_rel_dist_between_seeds"]
    for ax in axes[:2]:
        ax.axhline(nf, color=BLUE, lw=0.7, ls=":")
        ax.set_ylabel("bend: ‖ΔW1 − ΔW1 twin‖ / ‖ΔW1 twin‖")
    axes[0].text(0.5, nf, " coin-flip noise floor (λ=1, sampled)", fontsize=6, color=BLUE, va="bottom")
    axes[0].set_xlabel("swap gap (points): twin on exact neuron − twin on this neuron")
    axes[1].set_xlabel("twin's hidden misfire rate on this neuron (%)")
    axes[2].set_xlabel("twin's hidden misfire rate on this neuron (%)")
    axes[2].set_ylabel("cosine of ΔW1 with the twin's ΔW1")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("3 vs 8: how far each network's first layer bent away from its exact twin, against how faulty its neuron is\n"
                 "pink = Jev arms (stage 5, 8g), black = laya (9c scalar ◆, 10 sparse ■, own twin)", fontsize=10)
    savefig(fig, "stage6c-bend.png")


# ---------------------------------------------------------------- 2. per-neuron misfires, predicted and measured

def calls():
    return dict(np.load(OUT / "calls.npz"))


def measured(c, sel, n_img, n_neur):
    """Per (image, neuron): mean measured p over passes, for the selected hidden calls."""
    i, j = c["i"][sel].astype(int), c["j"][sel].astype(int)
    tot = np.bincount(i * n_neur + j, c["p"][sel], n_img * n_neur)
    cnt = np.bincount(i * n_neur + j, None, n_img * n_neur)
    return (tot / np.maximum(cnt, 1)).reshape(n_img, n_neur), cnt.reshape(n_img, n_neur)


def curve_p(z, spread, n):
    k, m0 = curve(n)
    return Phi(k * (z / np.where(spread > 0, spread, 1) - m0))


def trained_logit_p(z, spread, n):
    """stage 6b's margin-only logistic fit on trained hidden neurons (in-sample for stages 5-6)."""
    rows = json.loads((ROOT / "runs/stage6b/error.json").read_text())["answer_model"]["trained hidden (stages 5-6)"]
    edges = [60, 120, 200]
    band = np.digitize(n - 1, edges)  # the bands count terms without the bias
    a = np.array([r["margin_only_intercept"] for r in rows])[band]
    s = np.array([r["margin_only_slope"] for r in rows])[band]
    return 1 / (1 + np.exp(-(a + s * z / np.where(spread > 0, spread, 1))))


def hidden_nets():
    """Plain yes/no Jev networks with journaled validation calls: (label, weights, Xv, call selector)."""
    c = calls()
    base = (c["layer"] == 0) & (c["split"] == 1)
    nets = []
    for pair, code in (("3v8", 1), ("0v1", 0)):
        _, va = split(YTR, tuple(int(d) for d in pair.split("v")), 500, 500)
        for arm, a in (("jev", 0), ("swap", 1)):
            nets.append((f"s5 {pair} {arm}", load(f"runs/stage5/weights-{pair}-{arm}.npz"), XTR[va],
                         base & (c["src"] == 0) & (c["pair"] == code) & (c["arm"] == a)))
    for seed in (0, 1, 2):
        for arm, a in (("jev", 0), ("swap", 1)):
            sel = base & (c["src"] == 1) & (c["seed"] == seed) & (c["arm"] == a)
            if arm == "jev":
                sel &= c["epoch"] == 9  # the final-weights validation pass
            nets.append((f"s6 seed{seed} {arm}", load(f"runs/stage6/weights-seed{seed}-{arm}.npz"), XTR[VA10], sel))
    nets.append(("8g B", load("runs/stage8g/weights-B.npz"), XTR[VA38], base & (c["src"] == 2) & (c["arm"] == 2)))
    return c, nets


def run_misfire():
    c, nets = hidden_nets()
    res, scatter = {}, {}
    for label, w, Xv, sel in nets:
        p, cnt = measured(c, sel, len(Xv), w[0].shape[1])
        z, spread, n = hidden_margins(Xv, w[0], w[1])
        n = n + 1  # the folded bias is one more number
        zj = np.zeros_like(z)
        np.add.at(zj, (c["i"][sel].astype(int), c["j"][sel].astype(int)), c["z"][sel])
        dz = np.abs(zj / np.maximum(cnt, 1) - z)  # journal z matches the weights up to per-product rounding ties
        assert dz.mean() < 0.05 and dz.max() < 0.5, (label, dz.mean(), dz.max())  # 8g rounds x*w, not x and w
        pos, fired = z > 0, p >= 0.5
        mis = fired != pos
        pred = {"curve": curve_p(z, spread, n), "6b logistic": trained_logit_p(z, spread, n)}
        r = {"calls": int(cnt.sum()), "measured": {"misfire": float(mis.mean()), "false_fire": float(fired[~pos].mean()),
                                                   "false_silence": float((~fired[pos]).mean()), "intended_fire_rate": float(pos.mean()),
                                                   "fire_rate": float(fired.mean())}}
        for pname, pf in pred.items():
            pm = np.where(pos, 1 - pf, pf)
            per_meas, per_pred = mis.mean(0), pm.mean(0)
            r[pname] = {"misfire": float(pm.mean()), "false_fire": float(pf[~pos].mean()), "false_silence": float((1 - pf[pos]).mean()),
                        "per_neuron_corr": float(np.corrcoef(per_meas, per_pred)[0, 1]),
                        "per_neuron_mean_abs_err": float(np.mean(np.abs(per_meas - per_pred)))}
        m = z / np.where(spread > 0, spread, 1)
        r["per_neuron"] = {"measured_misfire": mis.mean(0).round(4).tolist(), "curve_misfire": np.where(pos, 1 - pred["curve"], pred["curve"]).mean(0).round(4).tolist(),
                           "measured_false_fire_share": float(np.sum(fired & ~pos) / max(np.sum(mis), 1))}
        # asymmetry: are negative sums kept farther from zero than positive ones, and more so on long images?
        kimg = n[:, 0]
        terc = np.quantile(kimg, [1 / 3, 2 / 3])
        r["by_length"] = {}
        for nm, wsel in (("short", kimg <= terc[0]), ("mid", (kimg > terc[0]) & (kimg <= terc[1])), ("long", kimg > terc[1])):
            mm, pp = m[wsel], pos[wsel]
            r["by_length"][nm] = {"terms": [int(kimg[wsel].min()), int(kimg[wsel].max())],
                                  "median_m_negative": float(np.median(mm[~pp])), "median_m_positive": float(np.median(mm[pp])),
                                  "share_negative_in_danger_zone": float(np.mean((mm[~pp] > curve(kimg[wsel])[1].min() - 0.5))),
                                  "measured_false_fire": float(fired[wsel][~pp].mean()), "measured_false_silence": float((~fired[wsel][pp]).mean())}
        # per neuron: is a neuron's negative-side margin bigger when the images it sits near zero on are longer?
        near = np.abs(m) < 1
        nn = np.array([kimg[near[:, j]].mean() if near[:, j].any() else np.nan for j in range(m.shape[1])])
        negm = np.array([np.median(-m[~pos[:, j], j]) if (~pos[:, j]).sum() > 5 else np.nan for j in range(m.shape[1])])
        ok = ~np.isnan(nn) & ~np.isnan(negm)
        r["per_neuron_corr_len_near_zero_vs_negative_margin"] = float(np.corrcoef(nn[ok], negm[ok])[0, 1]) if ok.sum() > 3 else None
        res[label] = r
        scatter[label] = (mis.mean(0), np.where(pos, 1 - pred["curve"], pred["curve"]).mean(0), fired, pos, z, spread, n)
    # init: where the curve says the starting weights sit (no measurement)
    for label, init, Xv in (("s5 3v8 init", flat(stage5.init(0)), XTR[VA38]), ("s6 seed0 init", flat(stage6.init(0)), XTR[VA10])):
        z, spread, n = hidden_margins(Xv, init[0], init[1])
        pf, pos = curve_p(z, spread, n + 1), z > 0
        res[label] = {"curve": {"misfire": float(np.where(pos, 1 - pf, pf).mean()), "false_fire": float(pf[~pos].mean()),
                                "false_silence": float((1 - pf[pos]).mean())}, "intended_fire_rate": float(pos.mean())}
    save("misfire", res)
    plot_misfire(res, scatter)


def plot_misfire(res, scatter):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for label, color, mk in (("s5 3v8 jev", ACCENT, "o"), ("s5 3v8 swap", EXACT, "o"), ("s6 seed0 jev", ACCENT, "^"),
                             ("s6 seed0 swap", EXACT, "^"), ("s6 seed1 jev", ACCENT, "v"), ("s6 seed1 swap", EXACT, "v"),
                             ("s6 seed2 jev", ACCENT, "<"), ("s6 seed2 swap", EXACT, "<")):
        meas, pred = scatter[label][:2]
        axes[0].plot(100 * pred, 100 * meas, mk, color=color, ms=3.5, alpha=0.7, mfc="none" if "s6" in label else color,
                     label=label if "seed1" not in label and "seed2" not in label else None)
    lim = axes[0].get_xlim()[1]
    axes[0].plot([0, 60], [0, 60], color=S.MUTE, lw=0.7)  # guide
    axes[0].set_xlim(0, 60)
    axes[0].set_ylim(0, 60)
    axes[0].set_xlabel("predicted misfire, margin curve (%)")
    axes[0].set_ylabel("measured misfire on validation (%)")
    axes[0].set_title("per hidden neuron (s6: all 3 seeds, open markers)", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    labels = ["s5 3v8 jev", "s5 3v8 swap", "s5 0v1 jev", "s5 0v1 swap"] + [f"s6 seed{s} {a}" for s in range(3) for a in ("jev", "swap")]
    x = np.arange(len(labels))
    for off, key, color, name in ((-0.2, "false_fire", ACCENT, "false fire (total ≤ 0, Jev says yes)"),
                                  (0.2, "false_silence", BLUE, "false silence (total > 0, Jev says no)")):
        axes[1].bar(x + off, [100 * res[lb]["measured"][key] for lb in labels], 0.4, color=color, label=f"measured {name}")
        axes[1].plot(x + off, [100 * res[lb]["curve"][key] for lb in labels], "_", color=INK, ms=12, mew=1.5)
    axes[1].set_xticks(x, [lb.replace("s6 ", "").replace("s5 ", "") for lb in labels], rotation=60, fontsize=7)
    axes[1].set_ylabel("rate (%), bars measured, ticks = margin-curve prediction")
    axes[1].legend(fontsize=7, frameon=False)
    for label, color in (("s6 seed0 jev", ACCENT), ("s6 seed0 swap", EXACT)):
        _, _, fired, pos, z, spread, n = scatter[label]
        m = (z / spread).ravel()
        axes[2].hist(np.clip(m, -5, 5), np.linspace(-5, 5, 81), histtype="step", color=color, density=True, lw=1.3, label=label)
    axes[2].axvline(0, color=INK, lw=0.5)
    axes[2].axvline(float(curve(np.array([150]))[1][0]), color=ACCENT, lw=0.8, ls="--", label="measured threshold m0 at 150 terms")
    axes[2].set_xlabel("hidden m = z / spread, validation")
    axes[2].legend(fontsize=7, frameon=False)
    fig.suptitle("Where Jev misfires on trained networks: predicted from its margin curve vs measured in the journals", fontsize=10)
    savefig(fig, "stage6c-misfire.png")


# ---------------------------------------------------------------- 3. the yes-lean seen from the weights

def drift_stats(w0, w, Xv):
    z0 = Xv @ w0[0] + w0[1]
    z = Xv @ w[0] + w[1]
    return {"median_change_mean_z": float(np.median(z.mean(0) - z0.mean(0))), "intended_fire_rate": float(np.mean(z > 0)),
            "norm_dW2": float(np.linalg.norm(w[2] - w0[2])), "mean_db2": float(np.mean(w[3] - w0[3])),
            "mean_db1": float(np.mean(w[1] - w0[1]))}


def run_lean():
    c, nets = hidden_nets()
    res = {"output_layer": {}, "constant_units": {}, "sim": {}}
    # (a) does the output layer absorb the extra firing? Extra = measured hidden p minus the exact step.
    for label, w, Xv, sel in nets:
        if label.startswith("8g"):
            continue
        p, _ = measured(c, sel, len(Xv), w[0].shape[1])
        z = Xv @ w[0] + w[1]
        step = (z > 0).astype(float)
        extra = p - step
        shift = extra @ w[2]  # per image and output: change in the output sum from Jev's hidden answers
        zo_step = step @ w[2] + w[3]
        spread_o = np.sqrt((step ** 2) @ (w[2] ** 2) + w[3] ** 2)
        w0 = flat(stage5.init(0) if label.startswith("s5") else stage6.init(int(label[7])))
        res["output_layer"][label] = {
            "mean_extra_p_on_negative": float(extra[z <= 0].mean()), "mean_extra_p_on_positive": float(extra[z > 0].mean()),
            "output_sum_shift_mean": shift.mean(0).round(3).tolist(), "output_sum_shift_in_spreads": (shift / spread_o).mean(0).round(3).tolist(),
            "db2": (w[3] - w0[3]).round(3).tolist(), "b2": w[3].round(3).tolist(),
            "shift_cancelled_by_db2": ((w[3] - w0[3]) / np.where(shift.mean(0) != 0, -shift.mean(0), 1)).round(3).tolist(),
            "output_decision_flip_rate_from_hidden_answers": float(np.mean((zo_step > 0) != (zo_step + shift > 0)))}
        # (b) near-constant hidden units: Jev's p barely varies, or the neuron always or never fires
        fr, sd, ifr = (p >= 0.5).mean(0), p.std(0), (z > 0).mean(0)
        m = z / np.maximum(np.sqrt((Xv ** 2) @ (w[0] ** 2) + w[1] ** 2), 1e-9)
        res["constant_units"][label] = {"jev_fires_ge95pct": int(np.sum(fr >= 0.95)), "jev_fires_le5pct": int(np.sum(fr <= 0.05)),
                                        "sd_p_lt_0.1": int(np.sum(sd < 0.1)), "intended_on_ge95pct": int(np.sum(ifr >= 0.95)),
                                        "intended_off_ge95pct": int(np.sum(ifr <= 0.05)), "median_sd_p": float(np.median(sd)),
                                        "neurons_median_m_gt_2": int(np.sum(np.median(m, 0) > 2)),
                                        "neurons_median_m_lt_-2": int(np.sum(np.median(m, 0) < -2))}
    for label, w0, Xv in (("s5 3v8 init", flat(stage5.init(0)), XTR[VA38]), ("s6 seed0 init", flat(stage6.init(0)), XTR[VA10])):
        ifr = ((Xv @ w0[0] + w0[1]) > 0).mean(0)
        res["constant_units"][label] = {"intended_on_ge95pct": int(np.sum(ifr >= 0.95)), "intended_off_ge95pct": int(np.sum(ifr <= 0.05))}
    # (c) which property of the neuron makes sums drift up? Simulated neurons, 2 x 2: graded p or sampled yes/no,
    # with or without the threshold drift, stage 5 3v8 and stage 6 seed 0 recipes. Compared with the real arms.
    i5, i6 = flat(stage5.init(0)), flat(stage6.init(0))
    Xv38, Xv10 = XTR[VA38], XTR[VA10]
    res["sim"]["3v8 exact twin"] = drift_stats(i5, load("runs/stage5/weights-3v8-swap.npz"), Xv38)
    res["sim"]["3v8 real Jev"] = drift_stats(i5, load("runs/stage5/weights-3v8-jev.npz"), Xv38)
    res["sim"]["10 exact twin"] = drift_stats(i6, load("runs/stage6/weights-seed0-swap.npz"), Xv10)
    res["sim"]["10 real Jev"] = drift_stats(i6, load("runs/stage6/weights-seed0-jev.npz"), Xv10)
    for kind, graded, floor in (("graded", True, 0.0), ("graded in [0.03, 0.97]", True, 0.03), ("sampled", False, 0.0)):
        for lean in (True, False):
            tag = f"{kind}, {'lean' if lean else 'no lean'}"
            res["sim"][f"3v8 {tag}"] = drift_stats(i5, train38(CurveNeuron(1.0, graded, lean, 0, floor)), Xv38)
            res["sim"][f"10 {tag}"] = drift_stats(i6, train10(CurveNeuron(1.0, graded, lean, 0, floor)), Xv10)
    save("lean", res)
    plot_lean(res)


def plot_lean(res):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    names = ["exact twin", "real Jev", "graded, lean", "graded, no lean", "graded in [0.03, 0.97], lean",
             "graded in [0.03, 0.97], no lean", "sampled, lean", "sampled, no lean"]
    colors = [EXACT, ACCENT, INK, INK, S.PAPER, S.PAPER, BLUE, BLUE]  # simulations: ink, hollow ink, slate
    edges = [EXACT, ACCENT, INK, INK, INK, INK, BLUE, BLUE]
    hatches = ["", "", "", "//", "", "//", "", "//"]
    for ax, key, title in ((axes[0], "median_change_mean_z", "median change in a hidden neuron's mean z"),
                           (axes[1], "norm_dW2", "‖ΔW2‖ (output layer)"), (axes[2], "intended_fire_rate", "intended hidden fire rate")):
        for t, task in enumerate(("3v8", "10")):
            vals = [res["sim"][f"{task} {n}"][key] for n in names]
            x = t * (len(names) + 1) + np.arange(len(names))
            ax.bar(x, vals, color=colors, hatch=hatches, edgecolor=edges, lw=0.9)
        ax.set_xticks([3.5, 12.5], ["3 vs 8 (stage 5)", "ten digits (stage 6 seed 0)"])
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color=INK, lw=0.5)
    from matplotlib.patches import Patch
    axes[0].legend([Patch(facecolor=c, edgecolor=e, hatch=h) for c, e, h in zip(colors, edges, hatches)], names, fontsize=7, frameon=False)
    fig.suptitle("Which neuron property makes Jev-trained sums drift up? Simulated neurons with Jev's measured curve "
                 "(graded p vs sampled yes/no; with vs without threshold drift)", fontsize=10)
    savefig(fig, "stage6c-lean.png")


# ---------------------------------------------------------------- 4. reading Jev's threshold off the weights

def trough(m, lo=-2.5, hi=1.5, bw=0.15):
    """Location of the lowest point of a Gaussian-kernel density of m between lo and hi."""
    g = np.linspace(lo, hi, 401)
    m = m[(m > lo - 1) & (m < hi + 1)]
    dens = np.exp(-0.5 * ((g[:, None] - m[None]) / bw) ** 2).sum(1)
    return float(g[np.argmin(dens)])


def run_readoff():
    rng = np.random.default_rng(0)
    runs = [("s5 3v8", "runs/stage5/weights-3v8", np.concatenate([TR38, VA38]))]
    runs += [(f"s6 seed{s}", f"runs/stage6/weights-seed{s}", np.concatenate([TR10, VA10])) for s in range(3)]
    res = {"trough": {}, "likelihood": {}}
    for label, prefix, idx in runs:
        X = XTR[idx]
        kimg = (np.round(X, 2) != 0).sum(1) + 1
        edges = np.quantile(kimg, [0, 1 / 3, 2 / 3, 1])
        res["trough"][label] = {}
        for arm in ("jev", "swap"):
            w = load(f"{prefix}-{arm}.npz")
            z, spread, _ = hidden_margins(X, w[0], w[1])
            m = z / np.where(spread > 0, spread, 1)
            rows = []
            for a, b in zip(edges, edges[1:]):
                sel = (kimg >= a) & (kimg <= b)
                t = trough(m[sel].ravel())
                if t in (-2.5, 1.5):  # lowest point at the edge of the window: no gap in the sums
                    rows.append({"terms": [int(a), int(b)], "median_terms": float(np.median(kimg[sel])), "trough_m": None})
                    continue
                boots = [trough(m[sel][rng.integers(0, sel.sum(), sel.sum())].ravel()) for _ in range(100)]
                nmed = float(np.median(kimg[sel]))
                rows.append({"terms": [int(a), int(b)], "median_terms": nmed, "trough_m": t, "ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
                             "margin_probe_m0": float(curve(np.array([nmed]))[1][0])})
            res["trough"][label][arm] = rows
    # 6b's trained-hidden logistic fit gives its own 50/50 point per band: m0 = -intercept / slope
    rows = json.loads((ROOT / "runs/stage6b/error.json").read_text())["answer_model"]["trained hidden (stages 5-6)"]
    res["trained_logistic_m0_by_band"] = {r["k"]: -r["margin_only_intercept"] / r["margin_only_slope"] for r in rows}
    # likelihood read-off: which constant threshold m0 (and blur k) in the hidden neurons makes the trained weights
    # score best on their own training set? Output neurons keep Jev's measured curve.
    grid_m0, grid_k = np.round(np.arange(-3.0, 1.01, 0.1), 2), np.array([0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    def fit_threshold(w, X, Y):
        z, spread, _ = hidden_margins(X, w[0], w[1])
        m = z / np.where(spread > 0, spread, 1)
        loss = np.zeros((len(grid_k), len(grid_m0)))
        for a, k in enumerate(grid_k):
            for b, m0 in enumerate(grid_m0):
                out = CurveNeuron(1.0, True, True)(Phi(k * (m - m0)), w[2], w[3])
                pc = np.clip(out, 1e-3, 1 - 1e-3)
                loss[a, b] = float(np.mean(-Y * np.log(pc) - (1 - Y) * np.log(1 - pc)))
        a, b = np.unravel_index(np.argmin(loss), loss.shape)
        return {"best_k": float(grid_k[a]), "best_m0": float(grid_m0[b]),
                "best_m0_at_each_k": {str(k): float(grid_m0[np.argmin(loss[i])]) for i, k in enumerate(grid_k)},
                "loss_at_best": float(loss[a, b]), "grid_m0": grid_m0.tolist(), "grid_k": grid_k.tolist(), "loss": loss.round(4).tolist()}

    X38, Y38 = XTR[TR38], (YTR[TR38] == 8).astype(float)[:, None]
    for label, prefix, X, Y in (("s5 3v8", "runs/stage5/weights-3v8", X38, Y38),
                                ("s6 seed0", "runs/stage6/weights-seed0", XTR[TR10], np.eye(10)[YTR[TR10]])):
        res["likelihood"][label] = {arm: fit_threshold(load(f"{prefix}-{arm}.npz"), X, Y) for arm in ("jev", "swap")}
    # control: networks trained through simulated neurons whose threshold is known (Jev's curve, or no drift)
    res["likelihood"]["simulated 3v8"] = {}
    for kind, graded in (("graded", True), ("sampled", False)):
        for lean in (True, False):
            w = train38(CurveNeuron(1.0, graded, lean, 0))
            res["likelihood"]["simulated 3v8"][f"{kind}, {'lean' if lean else 'no lean'}"] = fit_threshold(w, X38, Y38)
    res["likelihood"]["simulated 3v8"]["true m0 at median 3v8 image (165 terms), lean"] = float(curve(np.array([165]))[1][0])
    save("readoff", res)
    plot_readoff(res)


def plot_readoff(res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ns = np.linspace(60, 260, 50)
    axes[0].plot(ns, curve(ns)[1], color=INK, lw=1.2, label="Jev's measured threshold m0(n) (margin probe)")
    bands = res["trained_logistic_m0_by_band"]
    for (k, v), x in zip(bands.items(), (40, 90, 160, 230)):
        axes[0].plot(x, v, "s", color=INK, mfc=S.PAPER, ms=5, label="Jev's 50/50 point on trained states (6b logistic)" if k == "30-59" else None)
    for label, mk in (("s5 3v8", "o"), ("s6 seed0", "^"), ("s6 seed1", "v"), ("s6 seed2", "<")):
        for arm, color in (("jev", ACCENT), ("swap", EXACT)):
            rows = res["trough"][label][arm]
            x = [r["median_terms"] for r in rows]
            rows = [r for r in rows if r["trough_m"] is not None]
            if not rows:
                continue
            x = [r["median_terms"] for r in rows]
            y = [r["trough_m"] for r in rows]
            err = np.array([[r["trough_m"] - r["ci95"][0], r["ci95"][1] - r["trough_m"]] for r in rows]).T
            axes[0].errorbar(x, y, err, fmt=mk + "-", color=color, ms=4, lw=0.8, capsize=2,
                             label=f"{'Jev-trained' if arm == 'jev' else 'exact twin'}: trough of hidden m ({label})" if label in ("s5 3v8", "s6 seed0") else None)
    axes[0].axhline(0, color=S.MUTE, lw=0.5)
    axes[0].set_xlabel("terms in the neuron's list (image length band, median)")
    axes[0].set_ylabel("m = z / spread")
    axes[0].set_title("where the trained networks leave a gap in their hidden sums", fontsize=9)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    for label, ls in (("s5 3v8", "-"), ("s6 seed0", "--")):
        for arm, color in (("jev", ACCENT), ("swap", EXACT)):
            r = res["likelihood"][label][arm]
            loss = np.array(r["loss"])
            kbest = r["grid_k"].index(r["best_k"])
            axes[1].plot(r["grid_m0"], loss[kbest] - loss[kbest].min(), ls, color=color,
                         label=f"{label} {'Jev-trained' if arm == 'jev' else 'exact twin'} (best k {r['best_k']:g}, m0 {r['best_m0']:+.1f})")
    axes[1].axvline(float(curve(np.array([150]))[1][0]), color=INK, lw=0.7, ls=":", label="Jev's m0 at 150 terms")
    axes[1].set_xlabel("assumed hidden threshold m0 (constant)")
    axes[1].set_ylabel("training loss above its minimum")
    axes[1].set_title("which threshold makes the trained weights score best", fontsize=9)
    axes[1].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Reading Jev's threshold drift off the weights", fontsize=10)
    savefig(fig, "stage6c-readoff.png")


# ---------------------------------------------------------------- 5. which pixels pay

def run_pixels():
    res = {}
    freq = {d: (XTR[TR10][YTR[TR10] == d] > 0).mean(0) for d in range(10)}
    ink = {d: float((XTR[TR10][YTR[TR10] == d] > 0).sum(1).mean()) for d in range(10)}
    maps = {}
    for label, w0, pj, ps in [(f"s6 seed{s}", flat(stage6.init(s)), f"runs/stage6/weights-seed{s}-jev.npz", f"runs/stage6/weights-seed{s}-swap.npz") for s in range(3)] + \
                             [("s5 3v8", flat(stage5.init(0)), "runs/stage5/weights-3v8-jev.npz", "runs/stage5/weights-3v8-swap.npz")]:
        wj, ws = load(pj), load(ps)
        diff = (wj[0] - w0[0]) - (ws[0] - w0[0])
        D = np.linalg.norm(diff, axis=1)  # per pixel, over hidden neurons
        maps[label] = D
        live = np.array(list(freq.values())).mean(0) > 0.01
        r = {"digits": {}}
        for d in (range(10) if label.startswith("s6") else (3, 8)):
            on = freq[d] >= 0.5
            r["digits"][str(d)] = {"ink_pixels": ink[d], "pixels_on_ge50pct": int(on.sum()),
                                   "share_of_diff_sq_on_them": float(np.sum(D[on] ** 2) / np.sum(D ** 2)),
                                   "share_of_live_pixels": float(on.sum() / live.sum()),
                                   "corr_D_vs_on_freq_live": float(np.corrcoef(D[live], freq[d][live])[0, 1])}
        # pixels a digit uses more than the average digit does
        mean_f = np.mean([freq[d] for d in range(10)], 0)
        for d in (range(10) if label.startswith("s6") else (3, 8)):
            r["digits"][str(d)]["corr_D_vs_excess_on_freq"] = float(np.corrcoef(D[live], (freq[d] - mean_f)[live])[0, 1])
        r["corr_D_vs_mean_on_freq"] = float(np.corrcoef(D[live], mean_f[live])[0, 1])
        res[label] = r
    # consistency of the map across seeds
    res["corr_D_between_seeds"] = {f"{a}-{b}": float(np.corrcoef(maps[f"s6 seed{a}"], maps[f"s6 seed{b}"])[0, 1]) for a, b in ((0, 1), (0, 2), (1, 2))}
    # per-digit accuracy of the swapped network on Jev, stage 6 (results files)
    pd = [json.loads((ROOT / f"runs/stage6/result-seed{s}-swap.json").read_text())["test"]["per_digit"] for s in range(3)]
    res["swap_on_jev_per_digit_test_mean"] = np.mean(pd, 0).round(3).tolist()
    res["ink_pixels_per_digit"] = [ink[d] for d in range(10)]
    save("pixels", res)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6), constrained_layout=True)
    Dm = np.mean([maps[f"s6 seed{s}"] for s in range(3)], 0)
    panels = [("‖ΔW Jev − ΔW exact‖ per pixel\nten digits, mean of 3 seeds", Dm, S.SEQ_JEV),
              ("same, 3 vs 8 (stage 5)", maps["s5 3v8"], S.SEQ_JEV)]
    panels += [(f"digit {d}: share of images with the pixel on\ncorr with map {res['s6 seed0']['digits'][str(d)]['corr_D_vs_on_freq_live']:+.2f} (seed 0)", freq[d], S.SEQ_SLATE)
               for d in (0, 3, 8, 1)]
    panels += [(f"digit {d}: pixels used more than average\n(corr {np.mean([res[f's6 seed{s}']['digits'][str(d)]['corr_D_vs_excess_on_freq'] for s in range(3)]):+.2f}, mean over seeds)",
                freq[d] - np.mean([freq[e] for e in range(10)], 0), S.DIVERGING) for d in (0, 3, 8, 1)]
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        v = np.abs(img).max()
        ax.imshow(img.reshape(28, 28), cmap=cmap, vmin=-v if cmap is S.DIVERGING else 0, vmax=v)
        ax.set_title(title, fontsize=7.5)
        ax.axis("off")
    fig.suptitle("Which pixels' weights differ most between Jev-trained and exact-trained networks", fontsize=10)
    savefig(fig, "stage6c-pixels.png")


PARTS = {"bend": run_bend, "misfire": run_misfire, "lean": run_lean, "readoff": run_readoff, "pixels": run_pixels}

if __name__ == "__main__":
    for part in sys.argv[1:] or PARTS:
        PARTS[part]()
