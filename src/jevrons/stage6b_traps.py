"""Stage 6b, step 4: adversarial states built from the error model, each paired with a matched control.

Every trap instance has a control with the same term count, true total and spread (so the same |m|),
differing only in the trap feature. Folded format, standard question. Spend is capped at $1 and the
client is paced slowly so stage 6 keeps its rate limit. A rerun reuses answers already journaled.
Usage: uv run python -m jevrons.stage6b_traps [--dry | --plot]
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jevrons.jev import BACKENDS, USD_PER_INPUT_TOKEN, Jev  # noqa: E402
from jevrons.states import neuron_question  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT, FIG = ROOT / "runs" / "stage6b", ROOT / "docs" / "figures"
N = 200


def two_groups(a, b, m, S):
    """Scale positive shape a and negative shape b (both > 0) so that sum(u*a) - sum(v*b) = m*S and
    the spread is S. Bisection on the angle between the groups' shares of the spread."""
    p1, p2, n1, n2 = a.sum(), (a * a).sum(), b.sum(), (b * b).sum()
    lo, hi = 0.0, np.pi / 2
    for _ in range(60):
        th = (lo + hi) / 2
        if np.cos(th) * p1 / np.sqrt(p2) - np.sin(th) * n1 / np.sqrt(n2) > m:
            lo = th
        else:
            hi = th
    return a * S * np.cos(th) / np.sqrt(p2), -b * S * np.sin(th) / np.sqrt(n2)


def affine(g, z, S):
    """Shift and scale g so it sums to z with spread S."""
    k = len(g)
    return (g - g.mean()) * np.sqrt((S * S - z * z / k) / np.sum((g - g.mean()) ** 2)) + z / k


def r2(v):
    return [round(float(x), 2) for x in v]


def build(rng):
    """{family: [(trap list, control list)] * N}; A and B share one control arm (big number in the middle)."""
    fam = {k: [] for k in ("last", "first", "vote", "long", "zeros")}
    for _ in range(N):
        # A/B: one large positive, 74 small negatives, total at m = -0.5; only the big number's position differs
        big, neg = two_groups(np.array([1.0]), rng.uniform(0.5, 1.5, 74), -0.5, 5.0)
        rest = list(rng.permutation(neg))
        mid = int(rng.integers(20, 55))
        ctrl = rest[:mid] + list(big) + rest[mid:]
        fam["last"].append((rest + list(big), ctrl))
        fam["first"].append((list(big) + rest, ctrl))
        # C: 60 small positives vs 15 large negatives (sign vote says yes), total at m = -0.5;
        # control has the same total and spread with 38 positives and 37 negatives
        pos, neg = two_groups(rng.uniform(0.5, 1.5, 60), rng.uniform(0.5, 1.5, 15), -0.5, 4.0)
        pos2, neg2 = two_groups(rng.uniform(0.5, 1.5, 38), rng.uniform(0.5, 1.5, 37), -0.5, 4.0)
        fam["vote"].append((list(rng.permutation(np.r_[pos, neg])), list(rng.permutation(np.r_[pos2, neg2]))))
        # D: 250 mixed-sign terms one spread below zero (m = -1), control is the mirror image (m = +1)
        t = affine(rng.normal(0, 1, 250), -5.0, 5.0)
        fam["long"].append((list(t), list(-t)))
        # E: 40 real terms padded with 110 zeros, m = +0.3; control is 150 real terms with the same total and spread
        real = affine(rng.normal(0, 1, 40), 1.2, 4.0)
        slots = np.zeros(150)
        slots[np.sort(rng.choice(150, 40, replace=False))] = real
        fam["zeros"].append((list(slots), list(affine(rng.normal(0, 1, 150), 1.2, 4.0))))
    return fam


def jobs():
    fam = build(np.random.default_rng(66))
    out = []
    for f, pairs in fam.items():
        for i, (trap, ctrl) in enumerate(pairs):
            out.append((f, "trap", i, r2(trap)))
            if f != "first":  # "first" reuses the "last" family's control
                out.append((f, "control", i, r2(ctrl)))
    return out


def stats(v):
    v = np.asarray(v, float)
    z = sum(v)
    s = np.sqrt(np.sum(v * v))
    return z, z / s


def main(dry=False):
    js = jobs()
    tokens = sum(300 + 5 * len(v) for *_, v in js)
    print(f"{len(js)} calls, estimated {tokens:,} tokens, ${tokens * USD_PER_INPUT_TOKEN:.3f}")
    for f in ("last", "first", "vote", "long", "zeros"):
        for arm in ("trap", "control"):
            ms = [stats(v)[1] for ff, a, _, v in js if ff == f and a == arm]
            if ms:
                print(f"  {f:6s} {arm:8s} n={len(ms)} mean m {np.mean(ms):+.3f} sd {np.std(ms):.3f}")
    if dry:
        return
    journal = OUT / "journal.jsonl"
    done = {}
    if journal.exists():
        for line in journal.open():
            r = json.loads(line)
            t = r["tag"]
            done[(t["family"], t["arm"], t["i"])] = r["answers"]["fires"]
    BACKENDS["typesafe"].per_min, BACKENDS["vercel"].per_min = 150, 450  # leave the rate limit to stage 6
    jev = Jev(journal, max_usd=1.0)
    q = neuron_question("folded")

    def one(j):
        f, arm, i, v = j
        if (f, arm, i) in done:
            return done[(f, arm, i)]
        z, m = stats(v)
        return jev.ask({"products": v}, q, {"part": "trap", "family": f, "arm": arm, "i": i, "z": round(z, 6), "m": m})["fires"]

    with ThreadPoolExecutor(16) as ex:
        ps = list(ex.map(one, js))
    print(f"new calls {jev.calls}, tokens {jev.tokens:,}, ${jev.usd:.4f}")
    p = {(f, arm, i): pi for (f, arm, i, _), pi in zip(js, ps)}
    zs = {(f, arm, i): stats(v)[0] for f, arm, i, v in js}
    mis = {key: float((p[key] >= 0.5) != (zs[key] > 0)) for key in p}
    res = {"calls_this_run": jev.calls, "usd_this_run": round(jev.usd, 4), "tokens_this_run": jev.tokens, "families": {}}
    for f in ("last", "first", "vote", "long", "zeros"):
        cf = "last" if f == "first" else f
        t = np.array([mis[(f, "trap", i)] for i in range(N)])
        c = np.array([mis[(cf, "control", i)] for i in range(N)])
        dlt = t - c
        res["families"][f] = {"trap_misfire": float(t.mean()), "control_misfire": float(c.mean()),
                              "diff": float(dlt.mean()), "diff_ci": [float(dlt.mean() - 1.96 * dlt.std(ddof=1) / np.sqrt(N)),
                                                                     float(dlt.mean() + 1.96 * dlt.std(ddof=1) / np.sqrt(N))],
                              "trap_mean_p": float(np.mean([p[(f, "trap", i)] for i in range(N)])),
                              "control_mean_p": float(np.mean([p[(cf, "control", i)] for i in range(N)])),
                              "trap_mean_m": float(np.mean([stats(v)[1] for ff, a, _, v in js if ff == f and a == "trap"])),
                              "control_mean_m": float(np.mean([stats(v)[1] for ff, a, _, v in js if ff == cf and a == "control"]))}
    (OUT / "traps.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    plot(res)


def plot(res):
    labels = {"last": "one big + last\n(control: same list, big + mid)", "first": "one big + first\n(same control)", "vote": "sign vote says yes\n(60 small +, 15 big −)",
              "long": "250 terms, m = −1\n(control: mirror, m = +1)", "zeros": "110 zeros padding\n40 real terms, m = +0.3"}
    fams = list(labels)
    fig, ax = plt.subplots(figsize=(8.5, 4), constrained_layout=True)
    x = np.arange(len(fams))
    from jevrons.stage6b import wilson
    for off, key, color, name in ((-0.18, "control_misfire", "#9a9a9a", "matched control"), (0.18, "trap_misfire", "#e4507a", "trap")):
        v = np.array([res["families"][f][key] for f in fams])
        ci = np.array([wilson(round(val * N), N) for val in v])
        ax.bar(x + off, v, 0.34, color=color, label=name, yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=3)
        for xi, vi, hi in zip(x + off, v, ci[:, 1]):
            ax.text(xi, hi + 0.015, f"{vi:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, [labels[f] for f in fams], fontsize=8)
    ax.set_ylabel(f"misfire rate ({N} states per bar, 95% CI)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage6b-traps.png", dpi=160)
    print(FIG / "stage6b-traps.png")


if __name__ == "__main__":
    if "--plot" in sys.argv:  # redraw from traps.json, no calls
        plot(json.loads((OUT / "traps.json").read_text()))
    else:
        main("--dry" in sys.argv)
