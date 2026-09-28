"""Stage 8a: sparse Jev neurons on ten digits.

Each hidden neuron connects to a fixed subset of K pixels, so its folded state holds only the connected
pixels that are nonzero in the image (about 20-60 numbers instead of ~150), where Jev adds more accurately
and each call costs fewer tokens. The neuron is stage 8g's arm E (yes/no and a neutral choice averaged in
one call) and backprop goes through Jev's measured curve, symmetric, at each call's own term count.

  local   offline checks: fit a curve mock to arm E's graded answers (stage 7 journal), then compare random
          and local receptive fields at several K and H on the exact neuron and on that mock; cost estimates
  pilot   stage 10's exact-trained sparse 3 vs 8 network (64 weights per hidden neuron) on the arm E Jev neuron
  run     ten digits: trained through Jev, and the swap control (same init and mask, exact-trained, run on Jev)
  by_terms  hidden misfires on the test calls by numbers sent and by margin, against stage 7
Usage: uv run python -m jevrons.stage8a [local | pilot | run | by_terms]
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevrons.digits import mnist, split, test_subset
from jevrons.jev import USD_PER_INPUT_TOKEN, Jev
from jevrons.net import StepNeuron, fit, forward
from jevrons.stage6 import evaluate, exact_accuracy
from jevrons.stage6c import Phi
from jevrons.stage8g import QUESTIONS, curve, phi
from jevrons.states import r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage8a"
KIND, K, H = "random", 96, 64  # chosen by `local` (docs/results/stage8a-sparse.md)
SEED, BATCH, LR, TAU, SCALE = 0, 32, 0.02, 3.0, 0.3
N_TRAIN, N_VAL, N_TEST, EPOCHS = 1000, 500, 1000, 6
VAL_EPOCHS = (1, 3, 5)  # zero-based: after epochs 2, 4 and 6 (epoch 2 matches stage 8c's fine-tuning spend)
TOKENS_FIXED, TOKENS_PER_TERM = 379.3, 5.33  # least squares on stage 7's journal (arm E questions)


def active_pixels(X):
    return np.mean(X > 0, 0) >= 0.05


def connections(kind, k, h, active, seed=SEED):
    """784 x h 0/1 mask. random: k active pixels per neuron. local: the k active pixels nearest a random active centre."""
    rng = np.random.default_rng(seed + 100)
    idx = np.flatnonzero(active)
    rc = np.array(divmod(idx, 28)).T
    mask = np.zeros((784, h))
    for j in range(h):
        if kind == "random":
            mask[rng.choice(idx, k, replace=False), j] = 1
        else:
            d = ((rc - rc[rng.integers(len(idx))]) ** 2).sum(1) + 1e-3 * rng.random(len(idx))
            mask[idx[np.argsort(d)[:k]], j] = 1
    return mask


def init(seed, mask, scale=SCALE):
    rng = np.random.default_rng(seed)
    h = mask.shape[1]
    return [(rng.normal(0, scale, (784, h)) * mask, np.zeros(h)), (rng.normal(0, 1 / np.sqrt(h), (h, 10)), np.zeros(10))]


def terms_and_spread(a_in, W, b):
    """Per (example, neuron): numbers in the folded state (nonzero input and nonzero weight, plus the bias), and their spread."""
    n = (np.round(a_in, 2) != 0).astype(float) @ (np.round(W, 2) != 0).astype(float) + 1
    return n, np.sqrt((a_in**2) @ (W**2) + b**2) + 1e-6


def slope(layer, z, a_in, W, b):
    """stage 8g's symmetric measured-curve backward (slope_for(True)), at each call's own term count."""
    n, spread = terms_and_spread(a_in, W, b)
    k, _ = curve(n)
    return k / spread * phi(k * z / spread)


class SparseENeuron:
    """Arm E Jev neuron (stage8g.CalibratedJevNeuron without compensation) that sends only connected, nonzero terms."""

    def __init__(self, jev, threads=128):
        self.jev, self.threads = jev, threads
        self.questions = {k: QUESTIONS[k] for k in ("v1", "choice")}

    def __call__(self, X, W, b, tag=None):
        jobs = [(i, j) for i in range(X.shape[0]) for j in range(W.shape[1])]
        live = np.round(W, 2) != 0

        def one(ij):
            i, j = ij
            keep = (np.round(X[i], 2) != 0) & live[:, j]
            state = {"products": [r2(v) for v in X[i][keep] * W[keep, j]] + [r2(b[j])]}
            ans = self.jev.ask(state, self.questions, {**(tag or {}), "i": i, "j": j, "z": round(sum(state["products"]), 2)})
            return float(np.mean([a["above"] if isinstance(a, dict) else a for a in ans.values()]))

        with ThreadPoolExecutor(self.threads) as ex:
            return np.array(list(ex.map(one, jobs))).reshape(X.shape[0], W.shape[1])


class SimE:
    """Offline stand-in for the arm E neuron: stage 6c's CurveNeuron form (graded p squeezed into [floor, 1 - floor],
    blur k(n) / lam) at the sparse term count, with a constant threshold shift instead of the yes/no drift m0(n), since
    arm E's two questions lean opposite ways. Hidden and output layers get their own (lam, floor, shift): output inputs
    are graded p values, not pixels. Deterministic for a given state, like Jev's decisions, which repeat 86-88% of the
    time. Parameters fitted to stage 7's arm E answers by `local` (runs/stage8a/local.json)."""

    def __init__(self, hidden, output):
        self.layers = (hidden, output)

    def __call__(self, X, W, b, tag=None):
        lam, floor, shift = self.layers[(tag or {}).get("layer", 0)]
        z = X @ W + b
        n, spread = terms_and_spread(X, W, b)
        k, _ = curve(n)
        return floor + (1 - 2 * floor) * Phi(k / lam * (z / spread - shift))


SIM_E = ([4.5, 0.04, 0.5], [2.0, 0.04, 0.8])  # (lam, floor, shift) for hidden, output: `local`'s fit (runs/stage8a/local.json)


def call_terms(params, X):
    """Numbers per call: hidden (image-dependent), and output (every hidden p, since graded Jev answers are never 0)."""
    n1 = terms_and_spread(X, params[0][0], params[0][1])[0]
    return n1, np.full((len(X), params[1][0].shape[1]), params[1][0].shape[0] + 1.0)


def usd_per_image(params, X):
    n1, n2 = call_terms(params, X)
    tok = (TOKENS_FIXED + TOKENS_PER_TERM * n1).sum(1) + (TOKENS_FIXED + TOKENS_PER_TERM * n2).sum(1)
    return float(tok.mean() * USD_PER_INPUT_TOKEN)


def data(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST):
    xtr, ytr, xte, yte = mnist()
    tr, va = split(ytr, range(10), n_train, n_val)
    te = test_subset(yte, n_test)
    return xtr[tr], np.eye(10)[ytr[tr]], xtr[va], ytr[va], xte[te], yte[te]


def e_calls():
    """Stage 7's arm E test calls: (layer, terms, m = z / spread, averaged p). Cached in runs/stage8a/e-calls.npz."""
    cache = OUT / "e-calls.npz"
    if not cache.exists():
        rows = []
        with (ROOT / "runs/stage7/journal.jsonl").open() as f:
            for line in f:
                if '"split": "test"' not in line:
                    continue
                r = json.loads(line)
                pr, a = r["state"]["products"], r["answers"]
                rows.append((r["tag"]["layer"], len(pr), sum(pr) / (np.sqrt(sum(v * v for v in pr)) + 1e-6),
                             (a["v1"] + a["choice"]["above"]) / 2))
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(cache, calls=np.array(rows))
    return np.load(cache)["calls"]


def fit_sim(c):
    """Grid-fit SimE per layer to arm E's answers: count-weighted squared error of mean p in m bins."""
    bins = np.array([-9, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3, 9])
    table, fitted = {}, []
    for layer in (0, 1):
        s = c[c[:, 0] == layer]
        b = np.digitize(s[:, 2], bins)
        rows = np.array([[s[b == i, 1].mean(), s[b == i, 2].mean(), s[b == i, 3].mean(), np.mean(s[b == i, 3] >= 0.5), (b == i).sum()]
                         for i in range(1, len(bins)) if (b == i).sum() > 50])
        table[layer] = rows.round(3).tolist()
        k = curve(s[:, 1])[0]
        best = None
        for lam in np.arange(1, 8.01, 0.5):
            for floor in (0, 0.02, 0.04, 0.06, 0.08, 0.1):
                for shift in np.arange(-0.4, 1.01, 0.1):
                    q = floor + (1 - 2 * floor) * Phi(k / lam * (s[:, 2] - shift))
                    err = float(np.mean((q - s[:, 3]) ** 2))  # per call, so every term count counts
                    if best is None or err < best[0]:
                        best = (err, [float(lam), floor, round(float(shift), 2)])
        fitted.append(best[1])
        print(f"layer {layer}: lam, floor, shift = {best[1]}, rms {np.sqrt(best[0]):.3f}; "
              f"measured (terms, m, mean p, fire rate, count): {table[layer]}", flush=True)
    return table, fitted


def local():
    OUT.mkdir(parents=True, exist_ok=True)
    table, sim = fit_sim(e_calls())
    X, Y, Xv, yv, _, _ = data()
    act = active_pixels(X)
    res = {"sim_fit": sim, "e_table": table, "active_pixels": int(act.sum()), "arms": {}}
    configs = [("dense", 784, 32)] + [(kind, k, h) for kind in ("random", "local") for k in (64, 96, 128) for h in (32, 64)] + [("random", 64, 128)]
    for kind, k, h in configs:
        mask = np.ones((784, h)) if kind == "dense" else connections(kind, k, h, act)
        start = init(SEED, mask)  # dense: identical to stage6.init(0)
        r = {}
        ex = fit(start, X, Y, StepNeuron(), 10, BATCH, LR, TAU, SEED, mask=mask)
        r["exact_trained_exact"] = exact_accuracy(ex, Xv, yv)
        r["exact_trained_on_sim"] = float(np.mean(np.argmax(forward(ex, Xv, SimE(*sim))[0][-1], 1) == yv))
        sm = fit(start, X, Y, SimE(*sim), 10, BATCH, LR, TAU, SEED, mask=mask, slope=slope)
        r["sim_trained_on_sim"] = float(np.mean(np.argmax(forward(sm, Xv, SimE(*sim))[0][-1], 1) == yv))
        r["sim_trained_exact"] = exact_accuracy(sm, Xv, yv)
        n1 = call_terms(start, Xv)[0]
        n1t = call_terms(sm, Xv)[0]
        r["hidden_terms_init"] = {"mean": float(n1.mean()), "p10": float(np.percentile(n1, 10)), "p90": float(np.percentile(n1, 90)),
                                  "share_below_10": float(np.mean(n1 < 10))}
        r["hidden_terms_trained"] = float(n1t.mean())
        r["usd_per_image"] = usd_per_image(sm, Xv)
        res["arms"][f"{kind} k={k} h={h}"] = r
        print(f"{kind:6s} k={k:3d} h={h:3d}: exact {r['exact_trained_exact']:.3f}, exact on sim {r['exact_trained_on_sim']:.3f}, "
              f"sim on sim {r['sim_trained_on_sim']:.3f}, sim exact {r['sim_trained_exact']:.3f}, terms {n1.mean():.1f} "
              f"(p10 {r['hidden_terms_init']['p10']:.0f}, p90 {r['hidden_terms_init']['p90']:.0f}), ${r['usd_per_image'] * 1000:.2f}/1k images",
              flush=True)
    (OUT / "local.json").write_text(json.dumps(res, indent=1))


def spent():
    """Jev dollars spent by stages 8a and 8c so far (every paid call is journaled)."""
    usd = 0.0
    for journal in list(OUT.glob("journal*.jsonl")) + list((ROOT / "runs/stage8c").glob("journal*.jsonl")):
        if "failures" in journal.name:
            continue
        with journal.open() as f:
            usd += sum(json.loads(line)["usage"].get("input_tokens", 0) for line in f) * USD_PER_INPUT_TOKEN
    return usd


def pilot():
    """Stage 10's exact-trained sparse 3 vs 8 net (runs/stage10/payoff-truth-n64) on the arm E Jev neuron, 500 val images."""
    from jevrons.stage8g import data as data38, evaluate as evaluate38
    _, _, Xv, yv = data38()
    w = np.load(ROOT / "runs/stage10/payoff-truth-n64/weights-swap.npz")
    params = [(w["arr_0"], w["arr_1"]), (w["arr_2"], w["arr_3"])]
    n1 = terms_and_spread(Xv, *params[0])[0]
    est = ((TOKENS_FIXED + TOKENS_PER_TERM * n1).sum() + len(Xv) * (TOKENS_FIXED + TOKENS_PER_TERM * 33)) * USD_PER_INPUT_TOKEN
    print(f"hidden terms mean {n1.mean():.1f}, max {n1.max():.0f}; estimated ${est:.2f}", flush=True)
    jev = Jev(OUT / "journal-pilot.jsonl", max_usd=1.0)
    res = evaluate38(params, Xv, yv, SparseENeuron(jev), {"arm": "pilot-swap", "split": "val"}, passes=1)
    res["exact_step"] = float(np.mean((forward(params, Xv, StepNeuron())[0][-1][:, 0] >= 0.5) == (yv >= 0.5)))
    res["hidden_terms"] = float(n1.mean())
    res["spend"] = {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4)}
    (OUT / "result-pilot.json").write_text(json.dumps(res, indent=1))
    print(res)


def summarize(params, X, y, neuron, tag):
    r = evaluate(params, X, y, neuron, tag)
    r["exact_step"] = exact_accuracy(params, X, y)
    n1 = call_terms(params, X)[0]
    r["hidden_terms"] = {"mean": float(n1.mean()), "p10": float(np.percentile(n1, 10)), "p90": float(np.percentile(n1, 90)),
                         "max": float(n1.max())}
    return r


TOTAL_USD = 35.0  # hard cap across every stage 8a and 8c run, pilot included


def budget():
    """What this process may spend: the overall cap minus what the journals say is already spent."""
    left = TOTAL_USD - spent()
    print(f"spent so far ${TOTAL_USD - left:.2f}; this process may spend ${left:.2f}", flush=True)
    return left


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, Xv, yv, Xt, yt = data()
    mask = connections(KIND, K, H, active_pixels(X))
    start = init(SEED, mask)
    jev = Jev(OUT / "journal.jsonl", max_usd=budget())
    neuron = SparseENeuron(jev)
    for arm in ("swap", "jev"):
        done = OUT / f"result-{arm}.json"
        if done.exists():
            continue
        tag = {"seed": SEED, "arm": arm}
        partial = OUT / f"partial-{arm}.json"
        res = json.loads(partial.read_text()) if partial.exists() else {"arm": arm, "val": {}}

        def on_epoch(epoch, params):
            print(f"{arm} epoch {epoch + 1} done  (${jev.usd:.2f} this process)", flush=True)
            if arm == "jev" and epoch in VAL_EPOCHS and str(epoch + 1) not in res["val"]:
                r = summarize(params, Xv, yv, neuron, {**tag, "split": "val", "epoch": epoch})
                r["usd_so_far"] = jev.usd
                res["val"][str(epoch + 1)] = r
                partial.write_text(json.dumps(res))
                print(f"jev epoch {epoch + 1}: val on Jev {r['accuracy']:.3f}, exact {r['exact_step']:.3f}  (${jev.usd:.2f})", flush=True)

        log = []
        params = fit(start, X, Y, neuron if arm == "jev" else StepNeuron(), EPOCHS, BATCH, LR, TAU, SEED, log,
                     {**tag, "split": "train"}, on_epoch=on_epoch, checkpoint=OUT / f"checkpoint-{arm}.pkl",
                     slope=slope if arm == "jev" else None, mask=mask)
        if arm == "swap":
            res["val"][str(EPOCHS)] = summarize(params, Xv, yv, neuron, {**tag, "split": "val"})
        res["test"] = summarize(params, Xt, yt, neuron, {**tag, "split": "test"})
        res["spend_process"] = {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4)}
        np.savez(OUT / f"weights-{arm}.npz", *[a for layer in params for a in layer])
        (OUT / f"curve-{arm}.json").write_text(json.dumps(log))
        done.write_text(json.dumps(res))
        print(f"{arm}: val {res['val'][str(EPOCHS)]['accuracy']:.3f}, test on Jev {res['test']['accuracy']:.3f}, "
              f"test exact {res['test']['exact_step']:.3f}  (${jev.usd:.2f})", flush=True)
    print({"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4), "by_backend": jev.by_backend})


def by_terms():
    """Hidden-neuron misfires on the test calls by numbers sent and by margin m = z / spread, per arm (8a and 8c journals),
    against stage 7's dense arm E hidden calls (both arms pooled) at the same margins."""
    nb, mb = (0, 20, 30, 40, 50, 80), (0, 0.5, 1, 1.5, 2, 3, 99)

    def table(n, m, wrong):
        return {"by_terms": {f"{lo}-{hi}": [int(s.sum()), float(wrong[s].mean())] for lo, hi in zip(nb, nb[1:]) if (s := (n >= lo) & (n < hi)).sum()},
                "by_abs_m": {f"{lo}-{hi}": [int(s.sum()), float(wrong[s].mean())] for lo, hi in zip(mb, mb[1:]) if (s := (abs(m) >= lo) & (abs(m) < hi)).sum()},
                "misfire": float(wrong.mean()), "mean_terms": float(n.mean()), "median_abs_m": float(np.median(abs(m)))}

    out = {}
    for journal in (OUT / "journal.jsonl", ROOT / "runs/stage8c/journal.jsonl"):
        if not journal.exists():
            continue
        rows = {}
        with journal.open() as f:
            for line in f:
                if '"split": "test"' not in line or '"layer": 0' not in line:
                    continue
                r = json.loads(line)
                pr, a = r["state"]["products"], r["answers"]
                rows.setdefault(r["tag"]["arm"], []).append((len(pr), sum(pr) / (np.sqrt(sum(v * v for v in pr)) + 1e-6),
                                                             (a["v1"] + a["choice"]["above"]) / 2))
        for arm, v in rows.items():
            v = np.array(v)
            out[arm] = table(v[:, 0], v[:, 1], (v[:, 2] >= 0.5) != (v[:, 1] > 0))
    c = e_calls()
    c = c[c[:, 0] == 0]
    out["stage 7 dense (both arms)"] = table(c[:, 1], c[:, 2], (c[:, 3] >= 0.5) != (c[:, 2] > 0))
    (OUT / "by-terms.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"local": local, "pilot": pilot, "run": run, "by_terms": by_terms}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
