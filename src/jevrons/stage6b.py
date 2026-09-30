"""Stage 6b offline analysis over runs/stage6b/features.npz (build it with jevrons.stage6b_features).

  error     what Jev's answer depends on beyond the margin; does it skim the list?
  calib     reliability of p against the true sign
  dynamics  margins and fires-as-intended by epoch, Jev-trained vs exact-trained (swap replayed locally)
  weights   weight maps, net change from init, biases, margins, output-layer use
  repeats   repeat-call noise by margin and term count; typesafe vs vercel at scale
Usage: uv run python -m jevrons.stage6b [error calib dynamics weights repeats]  (default: all)
Numbers go to runs/stage6b/<part>.json, figures to docs/figures/stage6b-*.png.
"""

import json
import sys
from math import erf
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jevrons import figstyle as S  # noqa: E402  (docs palette; these figures keep their own layout)

ROOT = Path(__file__).resolve().parents[2]
OUT, FIG = ROOT / "runs" / "stage6b", ROOT / "docs" / "figures"
ACCENT, INK, GREY = S.JEV, S.INK, S.GREY
TERM_BINS = [S.TERMS[n] for n in S.TERM_COUNTS]  # one colour per KBINS band, shortest lists first
KBINS = ((10, 30), (30, 60), (60, 120), (120, 200), (200, 1000))
RANDOM = ("s1_full", "s1_full2", "s1_wording", "margin", "backend", "backend6000", "backend12000")
Phi = np.vectorize(lambda t: 0.5 * (1 + erf(t / np.sqrt(2))))


def load():
    d = dict(np.load(OUT / "features.npz"))
    names = list(d.pop("sources"))
    d["name"] = np.array(names)[d["src"].astype(int)]
    d["fired"] = d["p"] >= 0.5
    d["pos"] = d["z"] > 0
    d["test"] = d["hash"] % 5 == 0  # held-out by state, so repeats of one state never straddle the split
    return d


def save(part, obj):
    (OUT / f"{part}.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.item()))
    print(json.dumps(obj, indent=1, default=lambda o: o.item())[:4000])


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print(FIG / name)


def logit_fit(X, y, l2=1.0, iters=50):
    """L2-regularized logistic regression by damped Newton. Returns (coef with intercept first, se)."""
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])

    def loss(w):
        t = X @ w
        return np.sum(np.logaddexp(0, t) - y * t) + 0.5 * l2 * w @ w

    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, X.T @ (p - y) + l2 * w)
        f, a = loss(w), 1.0
        while loss(w - a * step) > f and a > 1e-4:
            a /= 2
        w = w - a * step
        if np.max(np.abs(a * step)) < 1e-7:
            break
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


def predict(w, X):
    return 1 / (1 + np.exp(-(w[0] + X @ w[1:])))


def auc(score, y):
    r = np.argsort(np.argsort(score)) + 1.0
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(c - h), float(c + h))


# ---------------------------------------------------------------- 1. error model

ANSWER_FEATURES = ("first third", "middle third", "last third", "final number (bias slot)",
                   "sign vote", "share of zero terms", "largest term")


def answer_design(d):
    """Signed predictors of Jev's yes, all in units of the spread except the vote and zero share.
    For the folded format the final number is the bias; thirds are over the other terms."""
    s = np.where(d["spread"] > 0, d["spread"], 1)
    vote = d["frac_pos"] - (1 - d["frac_pos"] - d["frac_zero"])
    return np.column_stack([d["sum_first"], d["sum_mid"], d["sum_last"], d["bias"] / s, vote, d["frac_zero"],
                            d["dom"] * d["dom_sign"]]).astype(float)


def error():
    d = load()
    X = answer_design(d)
    y = d["fired"].astype(float)
    folded = (d["fmt"] == 3) & ~np.isin(d["name"], ("s2", "s3"))  # stages 2-3 have 1-6 terms
    groups = {"all folded": folded, "random folded probes": folded & np.isin(d["name"], RANDOM),
              "trained hidden (stages 5-6)": folded & np.isin(d["name"], ("s5", "s6")) & (d["tag_layer"] == 0)}
    res = {"answer_model": {}, "skim": {}, "separate_bias": {}}
    for g, sel in groups.items():
        rows = []
        for lo, hi in KBINS:
            w = sel & (d["k"] >= lo) & (d["k"] < hi)
            tr, te = w & ~d["test"], w & d["test"]
            if tr.sum() < 500:
                continue
            c, se = logit_fit(X[tr], y[tr])
            cm, _ = logit_fit(d["m"][tr, None], y[tr])
            pf, pm = predict(c, X[te]), predict(cm, d["m"][te, None])
            mis = d["fired"][te] != d["pos"][te]
            # P(misfire) under each model: P(yes) when the true total is <= 0, else P(no)
            mf, mm = np.where(d["pos"][te], 1 - pf, pf), np.where(d["pos"][te], 1 - pm, pm)
            rows.append({"k": f"{lo}-{hi - 1}" if hi < 1000 else f"{lo}+", "n_calls": int(w.sum()), "n_test": int(te.sum()),
                         "misfire_rate": float(mis.mean()),
                         "intercept": float(c[0]), **{f: [float(c[i + 1]), float(se[i + 1])] for i, f in enumerate(ANSWER_FEATURES)},
                         "margin_only_slope": float(cm[1]), "margin_only_intercept": float(cm[0]),
                         "heldout_agree_truth": float(np.mean(d["pos"][te] == d["fired"][te])),
                         "heldout_agree_margin_model": float(np.mean((pm > 0.5) == d["fired"][te])),
                         "heldout_agree_full_model": float(np.mean((pf > 0.5) == d["fired"][te])),
                         "misfire_auc_margin_model": auc(mm, mis) if 0 < mis.sum() < len(mis) else None,
                         "misfire_auc_full_model": auc(mf, mis) if 0 < mis.sum() < len(mis) else None})
        res["answer_model"][g] = rows

    # stage 1 separate-bias formats, pooled over term counts: the original bias-trust finding
    sep = np.isin(d["name"], ("s1_full", "s1_full2", "s1_wording")) & (d["fmt"] != 3)
    for fmt, code in (("paired", 0), ("parallel", 1), ("products", 2)):
        w = sep & (d["fmt"] == code) & (d["bias"] != 0)
        s = np.where(d["spread"] > 0, d["spread"], 1)
        c, se = logit_fit(np.column_stack([(d["z"] - d["bias"]) / s, d["bias"] / s])[w], y[w])
        res["separate_bias"][fmt] = {"n": int(w.sum()), "terms_weight": [float(c[1]), float(se[1])],
                                     "bias_weight": [float(c[2]), float(se[2])]}

    # skim test: when a partial sum disagrees with the true sign, which one does Jev follow?
    heur = [f"{h}{n}" for h in ("pre", "suf", "rnd") for n in (5, 10, 20, 40)] + ["pre_half", "suf_half"]
    skim_groups = {"random probes, all formats": np.isin(d["name"], RANDOM) & (d["k"] >= 30),
                   "trained hidden (stages 5-6)": np.isin(d["name"], ("s5", "s6")) & (d["tag_layer"] == 0) & (d["k"] >= 30)}
    for g, sel in skim_groups.items():
        out = {}
        for h in heur:
            hs = d[h] > 0
            dis = sel & (hs != d["pos"])
            n, kf = int(dis.sum()), int((d["fired"] == hs)[dis].sum())
            out[h] = {"agree_with_jev": float((hs == d["fired"])[sel].mean()), "agree_with_truth": float((hs == d["pos"])[sel].mean()),
                      "n_disagree_with_truth": n, "jev_follows_heuristic": kf / n if n else None, "ci": wilson(kf, n)}
        res["skim"][g] = out
    # stage 1 hint: first/last 10 + bias on the original probe (k = 10 lists are the whole list)
    s1 = d["name"] == "s1_full"
    res["stage1_hint"] = {h: {str(k): float(((d[h] > 0) == d["fired"])[s1 & (d["k"] == k)].mean()) for k in (10, 30, 75, 150, 250)}
                          for h in ("pre10", "suf10")}
    res["stage1_hint"]["truth"] = {str(k): float((d["pos"] == d["fired"])[s1 & (d["k"] == k)].mean()) for k in (10, 30, 75, 150, 250)}
    save("error", res)
    plot_error(res)


def plot_error(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    rows = res["answer_model"]["all folded"]
    xs = np.arange(len(rows))
    for f, color, off in (("first third", INK, -0.3), ("middle third", S.MUTE, -0.15), ("last third", S.RULE, 0),
                          ("largest term", GREY, 0.15), ("final number (bias slot)", ACCENT, 0.3)):
        v, e = np.array([r[f] for r in rows]).T
        ax.bar(xs + off, v, 0.15, yerr=1.96 * e, color=color, label=f)
    ax.plot(xs, [r["intercept"] for r in rows], "o-", color=INK, ms=4, lw=1, label="intercept (yes-lean)")
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(xs, [r["k"] for r in rows])
    ax.set_xlabel("terms in the list")
    ax.set_ylabel("logit weight per unit of (sum / spread)")
    ax.set_title("What moves Jev's yes (folded format, all stages)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axes[1]
    for g, marker in (("random probes, all formats", "o"), ("trained hidden (stages 5-6)", "s")):
        sk = res["skim"][g]
        for h, color in (("pre", INK), ("suf", GREY), ("rnd", ACCENT)):
            ns = (5, 10, 20, 40)
            v = [sk[f"{h}{n}"]["jev_follows_heuristic"] for n in ns]
            lo = [v[i] - sk[f"{h}{n}"]["ci"][0] for i, n in enumerate(ns)]
            hi = [sk[f"{h}{n}"]["ci"][1] - v[i] for i, n in enumerate(ns)]
            ax.errorbar(ns, v, yerr=[lo, hi], marker=marker, color=color, ms=4, lw=1, capsize=2,
                        ls="-" if marker == "o" else "--",
                        label=f"{ {'pre': 'first n', 'suf': 'last n', 'rnd': 'random n'}[h]} + bias, "
                              f"{'random probes' if marker == 'o' else 'trained nets'}")
    ax.axhline(0.5, color=INK, lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xticks((5, 10, 20, 40), ["5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("terms in the partial sum")
    ax.set_ylabel("Jev sides with the partial sum")
    ax.set_title("When a partial sum has the wrong sign (lists of 30+ terms)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    savefig(fig, "stage6b-error-model.png")


# ---------------------------------------------------------------- 2. calibration

def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for i in range(bins):
        w = idx == i
        if w.sum():
            rows.append([float(p[w].mean()), float(y[w].mean()), int(w.sum())])
            ece += w.mean() * abs(p[w].mean() - y[w].mean())
    return rows, float(ece), float(np.mean((p - y) ** 2))


def calib():
    d = load()
    y = d["pos"].astype(float)
    res = {}
    sets = {"all full-state calls": np.ones(len(y), bool), "random probes": np.isin(d["name"], RANDOM),
            "trained nets (stages 2-6)": np.isin(d["name"], ("s2", "s3", "s5", "s6"))}
    sets.update({f"{lo}-{hi - 1} terms" if hi < 1000 else f"{lo}+ terms": (d["k"] >= lo) & (d["k"] < hi) for lo, hi in KBINS})
    sets["under 10 terms"] = d["k"] < 10
    for name, w in sets.items():
        rows, ece, brier = reliability(d["p"][w], y[w])
        band = w & (d["p"] >= 0.75) & (d["p"] < 0.85)
        res[name] = {"n": int(w.sum()), "ece": ece, "brier": brier, "curve": rows,
                     "p_0.75_0.85_true_rate": float(y[band].mean()) if band.any() else None,
                     "p_0.15_0.25_true_rate": float(y[w & (d["p"] >= 0.15) & (d["p"] < 0.25)].mean())}
    # the same p means different things at different margins: calibration within |m| bands
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 99)):
        w = (np.abs(d["m"]) >= lo) & (np.abs(d["m"]) < hi)
        rows, ece, brier = reliability(d["p"][w], y[w])
        res[f"|m| {lo}-{hi}"] = {"n": int(w.sum()), "ece": ece, "brier": brier, "curve": rows}
    save("calib", res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, keys, title in ((axes[0], [k for k in res if k.endswith("terms")], "by term count"),
                            (axes[1], ["random probes", "trained nets (stages 2-6)"], "random probes vs trained nets")):
        colors = TERM_BINS + [GREY]  # the five KBINS bands, then "under 10 terms"
        for key, color in zip(keys, colors if len(keys) > 2 else (INK, ACCENT)):
            c = np.array(res[key]["curve"])
            ax.plot(c[:, 0], c[:, 1], "o-", color=color, ms=3.5, lw=1.4,
                    label=f"{key}  (n={res[key]['n']:,}, ECE {res[key]['ece']:.3f})")
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
        ax.set_xlabel("Jev's p (mean within bin)")
        ax.set_ylabel("fraction with true total > 0")
        ax.set_title(f"Reliability, {title}", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    savefig(fig, "stage6b-calibration.png")


# ---------------------------------------------------------------- 3 and 5 shared: margins from weights

def margin_curve():
    fits = json.loads((ROOT / "runs/margin/summary.json").read_text())["offset_fit"]
    ks = np.array(sorted(int(k) for k in fits))
    return ks, fits


def p_intended(m, z, k):
    """Predicted P(Jev fires as intended) from the margin-probe fit at the nearest term count."""
    ks, fits = margin_curve()
    near = ks[np.argmin(np.abs(np.log(np.maximum(k, 1))[:, None] - np.log(ks)[None]), 1)]
    kk = np.array([fits[str(v)]["k"] for v in near])
    m0 = np.array([fits[str(v)]["threshold_m0"] for v in near])
    pf = Phi(kk * (m - m0))
    return np.where(z > 0, pf, 1 - pf)


def hidden_margins(X, W, b):
    """z, spread and term count per (image, hidden neuron), rounded exactly as JevNeuron sends them."""
    Xr, Wr, br = np.round(X, 2), np.round(W, 2), np.round(b, 2)
    z, s2 = [], []
    for a in range(0, len(X), 50):
        P = np.round(Xr[a:a + 50, :, None] * Wr[None], 2)
        z.append(P.sum(1) + br)
        s2.append((P * P).sum(1) + br * br)
    z, spread = np.concatenate(z), np.sqrt(np.concatenate(s2))
    k = np.repeat((Xr != 0).sum(1)[:, None], W.shape[1], 1)
    return z, spread, k


class RecordingStep:
    """Exact step neuron that records each hidden call's margin, as the journal does for the Jev arm."""

    def __init__(self):
        self.by_epoch = {}

    def __call__(self, X, W, b, tag=None):
        if tag["layer"] == 0:
            z, s, k = hidden_margins(X, W, b)
            self.by_epoch.setdefault(tag["epoch"], []).append(np.stack([z.ravel(), s.ravel(), k.ravel()]))
        return (X @ W + b > 0).astype(float)


def replay_swap(stage, seed_or_pair):
    """Retrain the swap arm locally (it is deterministic) and return per-epoch hidden (z, spread, k)."""
    from jevrons.digits import mnist, split
    from jevrons.net import fit
    xtr, ytr, _, _ = mnist()
    rec = RecordingStep()
    if stage == 6:
        from jevrons import stage6 as S
        tr, _ = split(ytr, range(10), 1000, 500)
        params = fit(S.init(seed_or_pair), xtr[tr], np.eye(10)[ytr[tr]], rec, S.EPOCHS, S.BATCH, S.LR, S.TAU, seed_or_pair,
                     None, {"seed": seed_or_pair, "arm": "swap", "split": "train"})
        saved = np.load(ROOT / f"runs/stage6/weights-seed{seed_or_pair}-swap.npz")
    else:
        from jevrons import stage5 as S
        pair = tuple(int(c) for c in seed_or_pair.split("v"))
        tr, _ = split(ytr, pair, 500, 500)
        params = fit(S.init(S.SEED), xtr[tr], (ytr[tr] == pair[1]).astype(float)[:, None], rec, S.EPOCHS, S.BATCH, S.LR,
                     S.TAU, S.SEED, None, {"pair": seed_or_pair, "arm": "swap", "split": "train"})
        saved = np.load(ROOT / f"runs/stage5/weights-{seed_or_pair}-swap.npz")
    same = all(np.allclose(a, saved[f"arr_{i}"]) for i, a in enumerate(q for layer in params for q in layer))
    return {e: np.concatenate(v, 1) for e, v in rec.by_epoch.items()}, same


def dynamics():
    d = load()
    res, panels = {}, []
    runs = [(6, s) for s in (0, 1, 2)] + [(5, p) for p in ("0v1", "3v8")]
    for stage, run in runs:
        if stage == 6:
            sel = (d["name"] == "s6") & (d["tag_seed"] == run)
            label = f"stage 6 seed {run}"
        else:
            sel = (d["name"] == "s5") & (d["pair"] == {"0v1": 0, "3v8": 1}[run])
            label = f"stage 5 {run}"
        jev = sel & (d["arm"] == 0) & (d["split"] == 0) & (d["tag_layer"] == 0)
        if not jev.any():
            continue
        rows = {"jev": [], "swap": []}
        for e in sorted(set(d["tag_epoch"][jev].astype(int))):
            w = jev & (d["tag_epoch"] == e)
            am = np.abs(d["m"][w])
            pred = p_intended(d["m"][w], d["z"][w], d["k"][w])
            rows["jev"].append({"epoch": e + 1, "n": int(w.sum()), "fires_as_intended": float((d["fired"] == d["pos"])[w].mean()),
                                "curve_predicted": float(pred.mean()), "median_abs_m": float(np.median(am)),
                                "frac_abs_m_lt_1": float(np.mean(am < 1)), "q10_abs_m": float(np.quantile(am, 0.1)),
                                "intended_fire_rate": float(d["pos"][w].mean()), "jev_fire_rate": float(d["fired"][w].mean())})
        swap_done = (ROOT / (f"runs/stage6/weights-seed{run}-swap.npz" if stage == 6 else f"runs/stage5/weights-{run}-swap.npz")).exists()
        if swap_done:
            by_epoch, same = replay_swap(stage, run)
            res[f"{label} swap replay matches saved weights"] = same
            for e, (z, s, k) in sorted(by_epoch.items()):
                m = z / np.where(s > 0, s, 1)
                rows["swap"].append({"epoch": e + 1, "n": int(len(m)), "curve_predicted": float(p_intended(m, z, k).mean()),
                                     "median_abs_m": float(np.median(np.abs(m))), "frac_abs_m_lt_1": float(np.mean(np.abs(m) < 1)),
                                     "q10_abs_m": float(np.quantile(np.abs(m), 0.1)), "intended_fire_rate": float(np.mean(z > 0))})
            sv = sel & (d["arm"] == 1) & (d["split"] == 1) & (d["tag_layer"] == 0)
            rows["swap_val_on_jev"] = {"fires_as_intended": float((d["fired"] == d["pos"])[sv].mean()),
                                       "curve_predicted": float(p_intended(d["m"][sv], d["z"][sv], d["k"][sv]).mean()),
                                       "median_abs_m": float(np.median(np.abs(d["m"][sv])))}
        jv = sel & (d["arm"] == 0) & (d["split"] == 1) & (d["tag_layer"] == 0)
        rows["jev_val_on_jev"] = {"fires_as_intended": float((d["fired"] == d["pos"])[jv].mean()),
                                  "curve_predicted": float(p_intended(d["m"][jv], d["z"][jv], d["k"][jv]).mean()),
                                  "median_abs_m": float(np.median(np.abs(d["m"][jv])))}
        res[label] = rows
        panels.append((label, rows))
    save("dynamics", res)
    fig, axes = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 6.2), constrained_layout=True, squeeze=False)
    for c, (label, rows) in enumerate(panels):
        for arm, color, name in (("jev", ACCENT, "trained through Jev"), ("swap", INK, "trained exact")):
            if not rows[arm]:
                continue
            ep = [r["epoch"] for r in rows[arm]]
            axes[0, c].plot(ep, [r["median_abs_m"] for r in rows[arm]], "o-", color=color, ms=3, label=name)
            axes[0, c].fill_between(ep, [r["q10_abs_m"] for r in rows[arm]], [r["median_abs_m"] for r in rows[arm]], color=color, alpha=0.12)
            axes[1, c].plot(ep, [r["curve_predicted"] for r in rows[arm]], "--", color=color, lw=1.2, label=f"{name}, predicted by margin curve")
            if arm == "jev":
                axes[1, c].plot(ep, [r["fires_as_intended"] for r in rows[arm]], "o-", color=color, ms=3, label=f"{name}, measured")
        if "swap_val_on_jev" in rows:
            axes[1, c].plot([ep[-1]], [rows["swap_val_on_jev"]["fires_as_intended"]], "x", color=INK, ms=7, label="trained exact, measured on val")
        axes[0, c].set_title(label, fontsize=10)
        axes[0, c].set_ylim(bottom=0)
        axes[1, c].set_ylim(0.75, 1.0)
        axes[1, c].set_xlabel("epoch")
    axes[0, 0].set_ylabel("hidden |m| (median; band from 10th pct)")
    axes[1, 0].set_ylabel("hidden fires as intended")
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].legend(fontsize=6.5, frameon=False, loc="lower right")
    savefig(fig, "stage6b-margin-dynamics.png")


# ---------------------------------------------------------------- 5. weight maps and net change

def mosaic(W, rows=4):
    cols = W.shape[1] // rows
    img = np.full((rows * 29 - 1, cols * 29 - 1), np.nan)
    for j in range(W.shape[1]):
        r, c = divmod(j, cols)
        img[r * 29:r * 29 + 28, c * 29:c * 29 + 28] = W[:, j].reshape(28, 28)
    return img


def weight_analysis(label, W0, Wj, Ws, Xv, Xtr, jev_val=None, swap_val=None):
    """Everything about how two arms that share an init differ. jev_val/swap_val: per-neuron measured
    fires-as-intended on validation from the journal, if available."""
    (W1_0, b1_0, W2_0, b2_0), (W1_j, b1_j, W2_j, b2_j), (W1_s, b1_s, W2_s, b2_s) = W0, Wj, Ws
    dJ, dS = W1_j - W1_0, W1_s - W1_0
    freq = (np.round(Xtr, 2) != 0).mean(0)  # share of training images where each pixel is in the state
    on, border = freq >= 0.1, freq < 0.01

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    alpha = float((dJ.ravel() @ dS.ravel()) / (dS.ravel() @ dS.ravel()))
    shared = alpha * dS
    per_cos = [cos(dJ[:, j], dS[:, j]) for j in range(dJ.shape[1])]
    out = {"norm_dW1": {"jev": float(np.linalg.norm(dJ)), "swap": float(np.linalg.norm(dS))},
           "norm_dW2": {"jev": float(np.linalg.norm(W2_j - W2_0)), "swap": float(np.linalg.norm(W2_s - W2_0))},
           "per_neuron_norm_dW1_median": {"jev": float(np.median(np.linalg.norm(dJ, axis=0))), "swap": float(np.median(np.linalg.norm(dS, axis=0)))},
           "cosine_dW1_overall": cos(dJ.ravel(), dS.ravel()), "cosine_dW1_per_neuron_median": float(np.median(per_cos)),
           "cosine_dW1_per_neuron_range": [float(min(per_cos)), float(max(per_cos))],
           "shared_scale_alpha": alpha, "jev_change_explained_by_swap": float(1 - np.sum((dJ - shared) ** 2) / np.sum(dJ ** 2)),
           "share_of_change_on_digit_pixels": {a: float(np.sum(v[on] ** 2) / np.sum(v ** 2)) for a, v in (("jev", dJ), ("swap", dS))},
           "share_of_change_on_border_pixels": {a: float(np.sum(v[border] ** 2) / np.sum(v ** 2)) for a, v in (("jev", dJ), ("swap", dS))},
           "n_on_digit_pixels": int(on.sum()), "n_border_pixels": int(border.sum()),
           "mean_dW1_on_digit": {a: float(v[on].mean()) for a, v in (("jev", dJ), ("swap", dS))},
           "frac_dW1_positive_on_digit": {a: float((v[on] > 0).mean()) for a, v in (("jev", dJ), ("swap", dS))},
           "sparsity_abs_w_lt_0.05_on_digit": {a: float((np.abs(v[on]) < 0.05).mean()) for a, v in (("init", W1_0), ("jev", W1_j), ("swap", W1_s))},
           "mean_abs_w_on_digit": {a: float(np.abs(v[on]).mean()) for a, v in (("init", W1_0), ("jev", W1_j), ("swap", W1_s))},
           "frac_w_positive_on_digit": {a: float((v[on] > 0).mean()) for a, v in (("init", W1_0), ("jev", W1_j), ("swap", W1_s))},
           "b1": {a: {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
                  for a, v in (("init", b1_0), ("jev", b1_j), ("swap", b1_s))},
           "db1_mean": {"jev": float((b1_j - b1_0).mean()), "swap": float((b1_s - b1_0).mean())},
           "db2": {"jev": (b2_j - b2_0).round(3).tolist(), "swap": (b2_s - b2_0).round(3).tolist()}}
    # margins and intended firing on real validation images
    margins = {}
    for a, (W, b) in (("init", (W1_0, b1_0)), ("jev", (W1_j, b1_j)), ("swap", (W1_s, b1_s))):
        z, s, k = hidden_margins(Xv, W, b)
        m = z / np.where(s > 0, s, 1)
        margins[a] = (z, m, k)
        out.setdefault("val_hidden", {})[a] = {
            "mean_abs_m_per_neuron_median": float(np.median(np.abs(m).mean(0))), "median_abs_m": float(np.median(np.abs(m))),
            "frac_abs_m_lt_1": float(np.mean(np.abs(m) < 1)), "intended_fire_rate": float(np.mean(z > 0)),
            "mean_z_per_neuron_median": float(np.median(z.mean(0))),
            "neurons_never_intended_on": int(np.sum((z > 0).mean(0) < 0.01)),
            "curve_predicted_fires_as_intended": float(p_intended(m.ravel(), z.ravel(), k.ravel()).mean())}
    # yes-lean test: is the Jev arm's intended fire rate lower than the exact arm's on long lists?
    kimg = margins["init"][2][:, 0]
    terc = np.quantile(kimg, [1 / 3, 2 / 3])
    for name, w in (("short", kimg <= terc[0]), ("mid", (kimg > terc[0]) & (kimg <= terc[1])), ("long", kimg > terc[1])):
        out.setdefault("intended_fire_rate_by_image_length", {})[name] = {
            "terms_range": [int(kimg[w].min()), int(kimg[w].max())],
            **{a: float(np.mean(margins[a][0][w] > 0)) for a in ("init", "jev", "swap")},
            **{f"mean_m_{a}": float(margins[a][1][w].mean()) for a in ("init", "jev", "swap")}}
    # change in each neuron's mean z caused by training, split into weights and bias
    for a, (W, b) in (("jev", (W1_j, b1_j)), ("swap", (W1_s, b1_s))):
        dz = margins[a][0].mean(0) - margins["init"][0].mean(0)
        out.setdefault("delta_mean_z_per_neuron", {})[a] = {"median": float(np.median(dz)), "frac_negative": float(np.mean(dz < 0)),
                                                            "from_bias_median": float(np.median(b - b1_0))}
    # output layer: which hidden neurons carry weight
    for a, W2 in (("jev", W2_j), ("swap", W2_s)):
        r = np.linalg.norm(W2, axis=1)
        out.setdefault("output_layer", {})[a] = {"row_norm_median": float(np.median(r)),
                                                 "participation_ratio": float(r.sum() ** 2 / (r ** 2).sum()),
                                                 "top8_share_of_sq_norm": float(np.sort(r ** 2)[-8:].sum() / (r ** 2).sum())}
    for a, rel in (("jev", jev_val), ("swap", swap_val)):
        if rel is not None:
            W2 = W2_j if a == "jev" else W2_s
            out["output_layer"][a]["corr_row_norm_vs_measured_reliability"] = float(np.corrcoef(np.linalg.norm(W2, axis=1), rel)[0, 1])
            out["output_layer"][a]["measured_reliability_median"] = float(np.median(rel))
    return out, margins, per_cos


def weights():
    from jevrons import stage5, stage6
    from jevrons.digits import mnist, split
    d = load()
    xtr, ytr, _, _ = mnist()
    res = {}
    runs = []
    for seed in (0, 1, 2):
        if (ROOT / f"runs/stage6/weights-seed{seed}-jev.npz").exists() and (ROOT / f"runs/stage6/weights-seed{seed}-swap.npz").exists():
            tr, va = split(ytr, range(10), 1000, 500)
            runs.append((f"stage 6 seed {seed}", f"s6-seed{seed}", stage6.init(seed), f"runs/stage6/weights-seed{seed}", tr, va,
                         (d["name"] == "s6") & (d["tag_seed"] == seed)))
    tr, va = split(ytr, (3, 8), 500, 500)
    runs.append(("stage 5 3v8", "s5-3v8", stage5.init(0), "runs/stage5/weights-3v8", tr, va, (d["name"] == "s5") & (d["pair"] == 1)))
    for label, short, init, prefix, tr, va, sel in runs:
        W0 = [q for layer in init for q in layer]
        Wj = [np.load(ROOT / f"{prefix}-jev.npz")[f"arr_{i}"] for i in range(4)]
        Ws = [np.load(ROOT / f"{prefix}-swap.npz")[f"arr_{i}"] for i in range(4)]
        rel = {}
        for a, code in (("jev", 0), ("swap", 1)):
            w = sel & (d["arm"] == code) & (d["split"] == 1) & (d["tag_layer"] == 0)
            if short.startswith("s6") and a == "jev":
                w &= d["tag_epoch"] == 9  # the final-epoch validation pass
            if w.any():
                ok = (d["fired"] == d["pos"])[w]
                j = d["tag_j"][w].astype(int)
                rel[a] = np.bincount(j, ok, 32) / np.maximum(np.bincount(j, minlength=32), 1)
        out, margins, per_cos = weight_analysis(label, W0, Wj, Ws, xtr[va], xtr[tr], rel.get("jev"), rel.get("swap"))
        res[label] = out
        plot_weights(label, short, W0, Wj, Ws, margins, per_cos)
    save("weights", res)


def plot_weights(label, short, W0, Wj, Ws, margins, per_cos):
    panels = [("init  W", W0[0]), ("trained through Jev  W", Wj[0]), ("trained exact  W", Ws[0]),
              ("Jev arm  ΔW = W − W_init", Wj[0] - W0[0]), ("exact arm  ΔW", Ws[0] - W0[0]),
              ("ΔW Jev − ΔW exact", (Wj[0] - W0[0]) - (Ws[0] - W0[0]))]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.6), constrained_layout=True)
    for ax, (title, W) in zip(axes.T.ravel(), panels):
        v = np.nanquantile(np.abs(W), 0.99) or 1
        im = ax.imshow(mosaic(W), cmap=S.DIVERGING, vmin=-v, vmax=v)
        ax.set_title(f"{title}   (±{v:.2f})", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"{label}: first-layer weights per hidden neuron, 28×28 (brick positive, slate negative)", fontsize=10)
    savefig(fig, f"stage6b-weights-{short}.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    bins = np.linspace(-6, 6, 61)
    for a, color in (("init", GREY), ("swap", INK), ("jev", ACCENT)):
        m = margins[a][1]
        name = {"init": "init", "swap": "trained exact", "jev": "trained through Jev"}[a]
        axes[0].hist(np.clip(m.ravel(), -6, 6), bins, histtype="step", color=color, lw=1.5, density=True, label=name)
        axes[1].plot(np.sort(np.abs(m).mean(0)), "o-", color=color, ms=3, label=name)
    axes[0].set_xlabel("hidden m = z / spread on validation images")
    axes[0].set_ylabel("density")
    axes[0].axvline(0, color=INK, lw=0.5)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("hidden neuron (sorted)")
    axes[1].set_ylabel("mean |m| over validation images")
    b = [W0[1], Ws[1], Wj[1]]
    axes[2].hist(b, np.linspace(min(map(np.min, b)), max(map(np.max, b)), 20), color=[GREY, INK, ACCENT],
                 label=["init", "trained exact", "trained through Jev"])
    axes[2].set_xlabel("hidden bias")
    axes[2].set_title(f"ΔW cosine, Jev vs exact arm, per neuron: median {np.median(per_cos):.2f}", fontsize=9)
    axes[2].set_ylabel("neurons")
    axes[2].legend(fontsize=7, frameon=False)
    fig.suptitle(label, fontsize=10)
    savefig(fig, f"stage6b-margins-{short}.png")


# ---------------------------------------------------------------- 6. repeats and backends

def matched_backend(d, w0):
    """typesafe minus vercel, fire rate and fires-as-intended, averaged over matched cells of
    (term count, |m|, true sign, neuron index) with at least 20 calls from each backend."""
    idx = np.flatnonzero(w0)
    keys = np.column_stack([np.digitize(d["k"][idx], [60, 120, 160, 200]), np.digitize(np.abs(d["m"][idx]), [0.25, 0.5, 1, 1.5, 2, 3, 5]),
                            d["pos"][idx], d["tag_j"][idx]])
    inv = np.unique(keys, axis=0, return_inverse=True)[1].ravel()
    be, f, ok = d["backend"][idx], d["fired"][idx], (d["fired"] == d["pos"])[idx]
    na, nb = np.bincount(inv, be == 0), np.bincount(inv, be == 1)
    fa, fb = np.bincount(inv, f * (be == 0)) / np.maximum(na, 1), np.bincount(inv, f * (be == 1)) / np.maximum(nb, 1)
    oa, ob = np.bincount(inv, ok * (be == 0)) / np.maximum(na, 1), np.bincount(inv, ok * (be == 1)) / np.maximum(nb, 1)
    wt = (na + nb) * ((na >= 20) & (nb >= 20))
    wt = wt / wt.sum()
    se = float(np.sqrt(np.sum(wt ** 2 * (fa * (1 - fa) / np.maximum(na, 1) + fb * (1 - fb) / np.maximum(nb, 1)))))
    return {"cells": int((wt > 0).sum()), "calls_in_cells": int(np.sum((na + nb)[wt > 0])),
            "fire_rate_diff": float(wt @ (fa - fb)), "fires_as_intended_diff": float(wt @ (oa - ob)), "se": se}


def repeats():
    d = load()
    res = {"repeat_noise": {}, "backend_at_scale": {}}
    order = np.argsort(d["hash"], kind="stable")
    h = d["hash"][order]
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    sizes = np.diff(np.r_[starts, len(h)])
    grp = np.repeat(np.arange(len(starts)), sizes)
    g = np.empty(len(h), int)
    g[order] = grp
    n = np.bincount(g)
    fsum = np.bincount(g, d["fired"])
    psum, p2sum = np.bincount(g, d["p"]), np.bincount(g, d["p"] ** 2)
    # per group: probability two calls on the same state disagree (unbiased from counts), and p sd
    idx = np.unique(g, return_index=True)[1]  # one representative call per distinct state
    G = {"n": n[g[idx]], "flip": (2 * fsum * (n - fsum) / np.maximum(n * (n - 1), 1))[g[idx]],
         "sd": np.sqrt(np.maximum(p2sum / n - (psum / n) ** 2, 0) * n / np.maximum(n - 1, 1))[g[idx]],
         "am": np.abs(d["m"][idx]), "k": d["k"][idx], "name": d["name"][idx], "pos": d["pos"][idx],
         "mean_fire": (fsum / n)[g[idx]]}
    multi = G["n"] >= 2
    res["repeat_noise"]["states_with_repeats_by_source"] = {s: int(np.sum(multi & (G["name"] == s))) for s in np.unique(G["name"])}
    kb = [(0, 10), (10, 60), (60, 120), (120, 200), (200, 1000)]
    mb = [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 99)]
    table = []
    for klo, khi in kb:
        for mlo, mhi in mb:
            w = multi & (G["k"] >= klo) & (G["k"] < khi) & (G["am"] >= mlo) & (G["am"] < mhi)
            if w.sum() >= 30:
                table.append({"k": [klo, khi], "abs_m": [mlo, mhi], "states": int(w.sum()),
                              "flip_prob": float(G["flip"][w].mean()), "p_sd": float(G["sd"][w].mean())})
    res["repeat_noise"]["by_k_and_margin"] = table
    for s in ("s1_full", "margin", "backend", "s5", "s3"):
        w = multi & (G["name"] == s)
        if w.any():
            res["repeat_noise"][f"overall {s}"] = {"states": int(w.sum()), "flip_prob": float(G["flip"][w].mean()),
                                                   "p_sd": float(G["sd"][w].mean())}
    # share of all repeat disagreement that sits within |m| < 0.5
    tot = G["flip"][multi].sum()
    res["repeat_noise"]["share_of_flips_at_abs_m_lt_0.5"] = float(G["flip"][multi & (G["am"] < 0.5)].sum() / tot)
    res["repeat_noise"]["share_of_states_at_abs_m_lt_0.5"] = float(np.mean(G["am"][multi] < 0.5))
    # backends at scale: stages 5-6, cells of (term count, |m|, true sign, neuron index); weighted difference
    sel = np.isin(d["name"], ("s5", "s6")) & (d["backend"] >= 0)
    for part, w0 in (("hidden", sel & (d["tag_layer"] == 0)), ("output", sel & (d["tag_layer"] == 1))):
        res["backend_at_scale"][part] = {
            "calls": {"typesafe": int((w0 & (d["backend"] == 0)).sum()), "vercel": int((w0 & (d["backend"] == 1)).sum())},
            "raw_fires_as_intended": {b: float((d["fired"] == d["pos"])[w0 & (d["backend"] == c)].mean()) for b, c in (("typesafe", 0), ("vercel", 1))},
            **{f"matched {name}": matched_backend(d, w0 & extra) for name, extra in
               (("all", True), ("true total <= 0", ~d["pos"]), ("true total > 0", d["pos"]),
                ("stage 5", d["name"] == "s5"), ("stage 6", d["name"] == "s6"))}}
    # same state on both backends (stage 5 validation passes and the backend check)
    w = np.flatnonzero((d["name"] == "s5") & (d["split"] == 1))
    gg, bb, pp = g[w], d["backend"][w], d["p"][w]
    o = np.argsort(gg, kind="stable")
    gg, bb, pp = gg[o], bb[o], pp[o]
    pair = np.flatnonzero(gg[1:] == gg[:-1])
    same = bb[pair] == bb[pair + 1]
    dp = np.abs(pp[pair] - pp[pair + 1])
    fl = (pp[pair] >= 0.5) != (pp[pair + 1] >= 0.5)
    both = {"pairs": int(len(pair)), "n_same": int(same.sum()), "n_cross": int((~same).sum()),
            "same_backend_mean_abs_dp": float(dp[same].mean()), "cross_backend_mean_abs_dp": float(dp[~same].mean()),
            "same_backend_flip": float(fl[same].mean()), "cross_backend_flip": float(fl[~same].mean())}
    res["backend_at_scale"]["stage5_val_repeat_pairs"] = both
    save("repeats", res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for (klo, khi), color in zip(kb, TERM_BINS):
        rows = [r for r in table if r["k"] == [klo, khi]]
        if rows:
            x = [np.mean(r["abs_m"]) if r["abs_m"][1] < 99 else 5 for r in rows]
            lab = f"{klo}-{khi - 1} terms" if khi < 1000 else f"{klo}+ terms"
            axes[0].plot(x, [r["flip_prob"] for r in rows], "o-", color=color, ms=3.5, label=lab)
            axes[1].plot(x, [r["p_sd"] for r in rows], "o-", color=color, ms=3.5, label=lab)
    axes[0].set_ylabel("P(two identical calls disagree on yes/no)")
    axes[1].set_ylabel("sd of p across identical calls")
    for ax in axes:
        ax.set_xlabel("|m| (bin centre)")
        ax.legend(fontsize=7, frameon=False)
    savefig(fig, "stage6b-repeat-noise.png")


if __name__ == "__main__":
    parts = {"error": error, "calib": calib, "dynamics": dynamics, "weights": weights, "repeats": repeats}
    for name in sys.argv[1:] or parts:
        parts[name]()
