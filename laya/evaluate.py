"""Evaluate a laya checkpoint the way Jev was evaluated. Offline (PyTorch on MPS), no server calls.

  curves   P(fire) = Phi(k (m - m0)) fitted per term count, plus sign accuracy, on
           (a) the margin probe's own 1,800 states (10/30/75 terms, Jev's answers journaled) and
           (b) fresh states from the same generator at 10..90 terms (seed 99), 400 per count (2 and 5 terms
           use the training generator, since the probe's can't reach |m| = 3 with so few numbers), and
           150 terms with the context raised to 1,024 tokens (Ollaya's laya:en stops at 512)
  real     6,000 held-out real journal states: accuracy vs the true sign, agreement with Jev, |p - p_jev|
  traps    stage 6b trap families at their original size where they fit (75 terms), and length-limited
           analogs of the two that don't (long list at 90 terms, zero padding at 60)
  scalar   z in [-10, 10] at 0.1, both scalar wordings
  speed    one call at a time and batched
Usage: uv run python evaluate.py base|<run name>   -> runs/stage10/eval-<name>.json
"""

import json
import sys
import time
from math import erf

import numpy as np
import torch

import data
import laya_model as L
from jevrons.margin_probe import neurons
from jevrons.stage6b_traps import affine, build, r2
from jevrons.states import SCALAR_Q, SCALAR_Q_ALT

OUT = data.ROOT / "runs/stage10"
Phi = np.vectorize(lambda t: 0.5 * (1 + erf(t / np.sqrt(2))))
CURVE_K = (2, 5, 10, 20, 30, 45, 60, 75, 90)


def fit_curve(m, fired):
    """Grid MLE of P(fire) = Phi(k (m - m0)), the margin probe's offset fit."""
    m, fired = np.asarray(m, float), np.asarray(fired, float)
    best = (np.inf, 0, 0)
    for k in np.r_[np.linspace(0.1, 6, 60), np.linspace(6.5, 40, 68)]:
        for m0 in np.linspace(-2, 2, 81):
            p = np.clip(Phi(k * (m - m0)), 1e-6, 1 - 1e-6)
            nll = -np.mean(fired * np.log(p) + (1 - fired) * np.log(1 - p))
            if nll < best[0]:
                best = (nll, k, m0)
    return {"k": round(float(best[1]), 2), "m0": round(float(best[2]), 3),
            "sign_accuracy": float(np.mean(fired == (m > 0))),
            "fires_when_negative": float(np.mean(fired[m <= 0])), "fires_when_positive": float(np.mean(fired[m > 0])), "n": len(m)}


def curve_states():
    """{label: [(state, m)]}; label 'probe-<k>' is the margin probe's journaled states, 'gen-<k>' fresh ones."""
    out = {}
    _, test = data.real_states()
    for r in test:
        if r["src"] == "margin":
            out.setdefault(f"probe-{len(r['products']) - 1}", []).append(({"products": r["products"]}, data.margin(r["products"]), r["p_jev"]))
    rng = np.random.default_rng(99)
    for k in (*CURVE_K, 150):
        if k < 10:  # the probe's rejection sampler can't reach |m| = 3 with so few numbers; use the training generator
            out[f"gen-{k}"] = [(s, data.margin(s["products"]), None) for s, _ in (data.synth(rng, k, family="probe") for _ in range(400))]
        else:
            out[f"gen-{k}"] = [(s, m, None) for s, _, m in neurons(rng, data.pixel_pool(), k, 400)]
    return out


def traps():
    """{family: (trap states, control states)} with each state a products list."""
    fam = build(np.random.default_rng(66))
    out = {f: ([r2(t) for t, _ in fam[f]], [r2(c) for _, c in fam[f]]) for f in ("last", "first", "vote")}
    rng = np.random.default_rng(67)
    long_t, zeros_t, zeros_c = [], [], []
    for _ in range(200):
        t = affine(rng.normal(0, 1, 90), -5.0, 5.0)  # 90 mixed terms one spread below zero; control = mirror
        long_t.append(r2(t))
        real = affine(rng.normal(0, 1, 20), 1.2, 4.0)  # 20 real terms + 40 zeros at m = +0.3; control 60 real terms
        slots = np.zeros(60)
        slots[np.sort(rng.choice(60, 20, replace=False))] = real
        zeros_t.append(r2(slots))
        zeros_c.append(r2(affine(rng.normal(0, 1, 60), 1.2, 4.0)))
    out["long90"] = (long_t, [[-v for v in t] for t in long_t])
    out["zeros60"] = (zeros_t, zeros_c)
    return out


def main(name):
    weights = None if name == "base" else OUT / name / "model.safetensors"
    tok, model = L.tokenizer(), L.load(weights)
    ask = lambda states, q=data.QUESTION, batch=32: L.p_yes(model, tok, states, q, batch)  # noqa: E731
    res = {"model": name, "curves": {}}

    for label, rows in curve_states().items():
        if label == "gen-150":
            L.MAX_LEN = 1024  # beyond Ollaya's laya:en context; tests the encoder at a length it never trained on
        p = ask([s for s, _, _ in rows])
        L.MAX_LEN = 512
        ok = ~np.isnan(p)
        m = np.array([mm for _, mm, _ in rows])[ok]
        c = fit_curve(m, (p[ok] >= 0.5).astype(float))
        c["fit_frac"] = float(ok.mean())
        if rows[0][2] is not None:
            pj = np.array([r[2] for r in rows])[ok]
            c["jev"] = fit_curve(m, (pj >= 0.5).astype(float))
            c["agree_with_jev"] = float(np.mean((p[ok] >= 0.5) == (pj >= 0.5)))
            c["mae_vs_jev"] = float(np.mean(np.abs(p[ok] - pj)))
        c["binned"] = [[float(lo), float(np.mean(p[ok][(m >= lo) & (m < lo + 0.25)]))] for lo in np.arange(-3, 3, 0.25)
                       if np.any((m >= lo) & (m < lo + 0.25))]
        res["curves"][label] = c
        print(label, {k: v for k, v in c.items() if k != "binned"}, flush=True)

    _, test = data.real_states()
    test = [r for r in test if r["src"] != "margin"]
    test = [test[i] for i in np.random.default_rng(5).choice(len(test), 6000, replace=False)]  # 6,000 of 45,526
    p = ask([{"products": r["products"]} for r in test])
    ok = ~np.isnan(p)
    z = np.array([r["z"] for r in test])[ok]
    pj = np.array([r["p_jev"] for r in test])[ok]
    src = np.array([r["src"] for r in test])[ok]
    p = p[ok]
    res["real"] = {s: {"n": int(np.sum(sel)), "acc_true_sign": float(np.mean((p[sel] >= 0.5) == (z[sel] > 0))),
                       "jev_acc_true_sign": float(np.mean((pj[sel] >= 0.5) == (z[sel] > 0))),
                       "agree_with_jev": float(np.mean((p[sel] >= 0.5) == (pj[sel] >= 0.5))),
                       "mae_vs_jev": float(np.mean(np.abs(p[sel] - pj[sel])))}
                   for s, sel in [("all", np.ones(len(p), bool))] + [(s, src == s) for s in sorted(set(src))]}
    res["real"]["fit_frac"] = float(ok.mean())
    print("real", res["real"], flush=True)

    res["traps"] = {}
    for fam, (trap, ctrl) in traps().items():
        pt, pc = ask([{"products": v} for v in trap]), ask([{"products": v} for v in ctrl])
        mis = lambda ps, vs: np.array([(pp >= 0.5) != (sum(v) > 0) for pp, v in zip(ps, vs)], float)  # noqa: E731
        mt, mc = mis(pt, trap), mis(pc, ctrl)
        d = mt - mc
        res["traps"][fam] = {"trap_misfire": float(mt.mean()), "control_misfire": float(mc.mean()), "diff": float(d.mean()),
                             "diff_ci": [float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                                         float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))],
                             "n_terms": len(trap[0]), "fit": bool(not np.isnan(pt).any() and not np.isnan(pc).any())}
        print("trap", fam, res["traps"][fam], flush=True)

    zs = [round(v, 1) for v in np.arange(-10, 10.0001, 0.1)]
    res["scalar"] = {}
    for wording, q in (("v1", SCALAR_Q["pos"]), ("v2", SCALAR_Q_ALT["pos"])):
        p = ask([{"z": zz} for zz in zs], q)
        res["scalar"][wording] = {"sign_accuracy": float(np.mean([(pp >= 0.5) == (zz > 0) for zz, pp in zip(zs, p)])),
                                  "mean_p_neg": float(np.mean([pp for zz, pp in zip(zs, p) if zz < 0])),
                                  "mean_p_pos": float(np.mean([pp for zz, pp in zip(zs, p) if zz > 0]))}
    p = ask([{"products": [zz]} for zz in zs])
    res["scalar"]["one_number_list"] = {"sign_accuracy": float(np.mean([(pp >= 0.5) == (zz > 0) for zz, pp in zip(zs, p)]))}
    print("scalar", res["scalar"], flush=True)

    states = [s for s, _, _ in curve_states()["gen-30"][:64]]
    ask(states[:4])
    t0 = time.perf_counter()
    for s in states[:32]:
        ask([s])
    single = (time.perf_counter() - t0) / 32
    t0 = time.perf_counter()
    ask(states, batch=64)
    res["speed"] = {"device": str(L.device()), "single_call_s_30_terms": round(single, 4),
                    "batched_per_state_s_30_terms": round((time.perf_counter() - t0) / len(states), 4)}
    print("speed", res["speed"], flush=True)
    (OUT / f"eval-{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    with torch.no_grad():
        main(sys.argv[1])
