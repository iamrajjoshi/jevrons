"""Stage 1 probe. Usage: uv run python -m jevrons.probe {scalar,full,full2,all}"""

import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevrons.jev import Jev
from jevrons.states import (SCALAR_Q, SCALAR_Q_ALT, neuron_question, neuron_state,
                            packed_questions, sample_neuron)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "probe"
THREADS = 16  # ~20 req/s allowed; latency ~0.2-1 s per call
FORMATS = ("paired", "parallel", "products")
TERMS = (10, 30, 75, 150, 250)
PACKABLE = (10, 30, 75, 150)  # 8 x 250-term paired states exceed the 32k state limit


def pmap(fn, items):
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(fn, items))


def scalar_grid():
    inner = np.round(np.arange(-1, 1.0001, 0.01), 2)
    outer = np.round(np.concatenate([np.arange(-100, -1, 0.1), np.arange(1.1, 100.0001, 0.1)]), 1)
    return sorted(set(inner.tolist()) | set(outer.tolist()))


def run_scalar(jev: Jev):
    grid = scalar_grid()
    jobs = [(z, rep) for rep in range(5) for z in grid]
    ps = pmap(lambda j: jev.ask({"z": j[0]}, SCALAR_Q, {"part": "scalar", "z": j[0], "rep": j[1]})["pos"], jobs)
    coarse = [round(v, 1) for v in np.arange(-10, 10.0001, 0.1)]
    alt = pmap(lambda z: jev.ask({"z": z}, SCALAR_Q_ALT, {"part": "scalar_alt", "z": z})["pos"], coarse)
    # keyed batch of 50 on the coarse grid, to compare with isolated calls
    batched = {}
    for i in range(0, len(coarse), 50):
        chunk = coarse[i:i + 50]
        keys = [f"n{j:02d}" for j in range(len(chunk))]
        state = dict(zip(keys, chunk))
        qs = {k: {**SCALAR_Q["pos"], "instructions": f"Look only at the value of {k}.",
                  "criteria": {"true": f"{k} is greater than zero.", "false": f"{k} is zero or less."}}
              for k in keys}
        ans = jev.ask(state, qs, {"part": "scalar_batched", "zs": chunk})
        batched.update({z: ans[k] for k, z in zip(keys, chunk)})

    by_z = {}
    for (z, _), p in zip(jobs, ps):
        by_z.setdefault(z, []).append(p)
    zs = np.array(grid)
    mean = np.array([np.mean(by_z[z]) for z in grid])
    spread = np.array([np.ptp(by_z[z]) for z in grid])
    iso_coarse = {z: np.mean(by_z[z]) for z in coarse if z in by_z}
    summary = {
        "points": len(grid), "repeats": 5,
        "repeat_identical_frac": float(np.mean(spread == 0)),
        "repeat_max_range": float(spread.max()),
        "sign_accuracy": float(np.mean((mean >= 0.5) == (zs > 0))),
        "sign_accuracy_neg": float(np.mean(mean[zs < 0] < 0.5)),
        "sign_accuracy_pos": float(np.mean(mean[zs > 0] >= 0.5)),
        "sign_accuracy_abs_le_1": float(np.mean(((mean >= 0.5) == (zs > 0))[np.abs(zs) <= 1])),
        "p_at_zero": float(mean[zs == 0][0]),
        "alt_wording_sign_accuracy": float(np.mean([(p >= 0.5) == (z > 0) for z, p in zip(coarse, alt)])),
        "alt_vs_main_mean_abs_diff": float(np.mean([abs(p - iso_coarse[z]) for z, p in zip(coarse, alt)])),
        "batched_vs_isolated_mean_abs_diff": float(np.mean([abs(batched[z] - iso_coarse[z]) for z in coarse])),
        "batched_sign_accuracy": float(np.mean([(batched[z] >= 0.5) == (z > 0) for z in coarse])),
    }
    curve = {"z": grid, "p_mean": mean.tolist(), "p_range": spread.tolist(),
             "alt": dict(zip(map(str, coarse), alt)), "batched": {str(k): v for k, v in batched.items()}}
    (OUT / "scalar_curve.json").write_text(json.dumps(curve))
    return summary


def run_full(jev: Jev):
    imgs = np.frombuffer(gzip.open(ROOT / "data/train-images-idx3-ubyte.gz").read(), np.uint8, offset=16)
    pool = imgs[imgs > 0] / 255.0
    rng = np.random.default_rng(1)
    neurons = []  # (fmt, k, idx, margin, x, w, b)
    for k in TERMS:
        for i in range(100):
            margin = rng.uniform(-2, 2)
            x, w, b, _ = sample_neuron(rng, pool, k, margin)
            for fmt in FORMATS:  # same neuron in all three formats
                neurons.append((fmt, k, i, margin, x, w, b))

    def one(job):
        (fmt, k, i, margin, x, w, b), rep = job
        state, z = neuron_state(fmt, x, w, b)
        p = jev.ask(state, neuron_question(fmt), {"part": "full", "fmt": fmt, "k": k, "i": i,
                                                   "rep": rep, "z": z, "margin": margin})["fires"]
        return fmt, k, i, rep, margin, z, p

    jobs = [(n, 0) for n in neurons] + [(n, rep) for n in neurons if n[2] < 20 for rep in (1, 2, 3, 4)]
    rows = pmap(one, jobs)

    # packing: paired neurons, 8 per request
    packed = {}
    for k in PACKABLE:
        group = [n for n in neurons if n[0] == "paired" and n[1] == k]
        for g in range(0, 96, 8):
            chunk = group[g:g + 8]
            keys = [f"n{j}" for j in range(8)]
            state = {key: neuron_state("paired", n[4], n[5], n[6])[0] for key, n in zip(keys, chunk)}
            ans = jev.ask(state, packed_questions("paired", keys), {"part": "packed", "k": k, "i0": chunk[0][2]})
            packed.update({(k, n[2]): ans[key] for key, n in zip(keys, chunk)})

    base = {(f, k, i): (m, z, p) for f, k, i, rep, m, z, p in rows if rep == 0}
    reps = {}
    for f, k, i, rep, m, z, p in rows:
        reps.setdefault((f, k, i), []).append(p)
    cells = []
    for fmt in FORMATS:
        for k in TERMS:
            items = [base[(fmt, k, i)] for i in range(100)]
            correct = np.array([(p >= 0.5) == (z > 0) for m, z, p in items])
            margins = np.abs([m for m, z, p in items])
            rr = [np.ptp(reps[(fmt, k, i)]) for i in range(20)]
            cell = {"fmt": fmt, "k": k, "sign_accuracy": float(correct.mean()),
                    "acc_margin_lt_0.5": float(correct[margins < 0.5].mean()),
                    "acc_margin_gt_1": float(correct[margins > 1].mean()),
                    "frac_yes": float(np.mean([p >= 0.5 for m, z, p in items])),
                    "repeat_identical_frac": float(np.mean(np.array(rr) == 0)),
                    "repeat_max_range": float(max(rr))}
            if fmt == "paired" and k in PACKABLE:
                diffs = [abs(packed[(k, i)] - base[("paired", k, i)][2]) for i in range(96)]
                pc = [(packed[(k, i)] >= 0.5) == (base[("paired", k, i)][1] > 0) for i in range(96)]
                cell["packed_mean_abs_diff"] = float(np.mean(diffs))
                cell["packed_sign_accuracy"] = float(np.mean(pc))
            cells.append(cell)
    (OUT / "full_rows.json").write_text(json.dumps(rows))
    return cells


def run_full_unconfounded(jev: Jev):
    """Bias either 0 or independent of the terms, so the bias can't stand in for the sum."""
    imgs = np.frombuffer(gzip.open(ROOT / "data/train-images-idx3-ubyte.gz").read(), np.uint8, offset=16)
    pool = imgs[imgs > 0] / 255.0
    rng = np.random.default_rng(2)
    jobs = []
    for mode in ("bias0", "bias_indep"):
        for k in TERMS:
            for i in range(60):
                x = rng.choice(pool, k)
                w = np.clip(rng.normal(0, 1, k), -9.99, 9.99)
                spread = np.sqrt(np.sum((x * w) ** 2))
                b = 0.0 if mode == "bias0" else rng.normal(0, spread)
                for fmt in FORMATS:
                    jobs.append((mode, fmt, k, i, x, w, b, spread))

    def one(job):
        mode, fmt, k, i, x, w, b, spread = job
        state, z = neuron_state(fmt, x, w, b)
        p = jev.ask(state, neuron_question(fmt), {"part": mode, "fmt": fmt, "k": k, "i": i, "z": z,
                                                   "margin": z / spread, "bias": state["bias"]})["fires"]
        return mode, fmt, k, z / spread, z, state["bias"], p

    rows = pmap(one, jobs)
    cells = []
    for mode in ("bias0", "bias_indep"):
        for fmt in FORMATS:
            for k in TERMS:
                sel = [r for r in rows if r[0] == mode and r[1] == fmt and r[2] == k]
                acc = np.mean([(p >= 0.5) == (z > 0) for *_, z, b, p in sel])
                cell = {"mode": mode, "fmt": fmt, "k": k, "sign_accuracy": float(acc),
                        "acc_margin_gt_1": float(np.mean([(p >= 0.5) == (z > 0) for _, _, _, m, z, b, p in sel if abs(m) > 1] or [np.nan])),
                        "frac_yes": float(np.mean([p >= 0.5 for *_, p in sel]))}
                if mode == "bias_indep":
                    cell["follows_bias"] = float(np.mean([(p >= 0.5) == (b > 0) for *_, z, b, p in sel]))
                cells.append(cell)
    (OUT / "full2_rows.json").write_text(json.dumps(rows))
    return cells


def run_wording(jev: Jev):
    """v1 vs v2 question wording on bias-0 and independent-bias neurons, products and folded formats."""
    imgs = np.frombuffer(gzip.open(ROOT / "data/train-images-idx3-ubyte.gz").read(), np.uint8, offset=16)
    pool = imgs[imgs > 0] / 255.0
    rng = np.random.default_rng(3)
    jobs = []
    for k in (10, 30, 75, 150):
        for i in range(40):
            x = rng.choice(pool, k)
            w = np.clip(rng.normal(0, 1, k), -9.99, 9.99)
            b = rng.normal(0, np.sqrt(np.sum((x * w) ** 2)))
            for fmt in ("products", "folded"):
                for wording in ("v1", "v2"):
                    jobs.append((fmt, wording, k, i, x, w, b))

    def one(job):
        fmt, wording, k, i, x, w, b = job
        state, z = neuron_state(fmt, x, w, b)
        p = jev.ask(state, neuron_question(fmt, wording),
                    {"part": "wording", "fmt": fmt, "wording": wording, "k": k, "i": i, "z": z, "bias": b})["fires"]
        return fmt, wording, k, z, round(b, 2), p

    rows = pmap(one, jobs)
    cells = []
    for fmt in ("products", "folded"):
        for wording in ("v1", "v2"):
            for k in (10, 30, 75, 150):
                sel = [r for r in rows if r[:3] == (fmt, wording, k)]
                cells.append({"fmt": fmt, "wording": wording, "k": k,
                              "sign_accuracy": float(np.mean([(p >= 0.5) == (z > 0) for *_, z, b, p in sel])),
                              "follows_bias": float(np.mean([(p >= 0.5) == (b > 0) for *_, z, b, p in sel]))})
    (OUT / "wording_rows.json").write_text(json.dumps(rows))
    return cells


def token_stats(journal: Path):
    stats = {}
    for line in journal.open():
        r = json.loads(line)
        tag = r["tag"] or {}
        key = tag.get("part", "?") + (f"/{tag['fmt']}/{tag['k']}" if "fmt" in tag else
                                      f"/{tag['k']}" if "k" in tag else "")
        s = stats.setdefault(key, {"calls": 0, "tokens": 0, "latency": 0.0})
        s["calls"] += 1
        s["tokens"] += r["usage"].get("input_tokens", 0)
        s["latency"] += r["latency_s"]
    return {k: {"calls": v["calls"], "tokens_per_call": round(v["tokens"] / v["calls"]),
                "latency_s": round(v["latency"] / v["calls"], 3)} for k, v in sorted(stats.items())}


def main(part: str):
    OUT.mkdir(parents=True, exist_ok=True)
    journal = OUT / f"journal-{part}.jsonl"
    jev = Jev(journal)
    result = {}
    if part in ("scalar", "all"):
        result["scalar"] = run_scalar(jev)
        print(json.dumps(result["scalar"], indent=1), flush=True)
    if part in ("full", "all"):
        result["full"] = run_full(jev)
        for c in result["full"]:
            print(c, flush=True)
    if part == "wording":
        result["wording"] = run_wording(jev)
        for c in result["wording"]:
            print(c, flush=True)
    if part in ("full2", "all"):
        result["full2"] = run_full_unconfounded(jev)
        for c in result["full2"]:
            print(c, flush=True)
    result["tokens"] = token_stats(journal)
    result["spend"] = {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4)}
    (OUT / f"summary-{part}.json").write_text(json.dumps(result, indent=1))
    print(result["spend"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
