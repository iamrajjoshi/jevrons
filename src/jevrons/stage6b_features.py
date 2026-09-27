"""Stage 6b, step 0: stream every full-state Jev journal into one table of compact numeric features.

Reads journals line by line (stage 6's is >1 GB and still growing, so it is read up to its size at
start), keeps only numbers, and writes runs/stage6b/features.npz. z is recomputed from the state
exactly as states.neuron_state does, so every full-state call has a known true total.
Usage: uv run python -m jevrons.stage6b_features
"""

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage6b"
SOURCES = {  # name -> journal; stage 4 is scalar and the probe-laya/ollaya runs are other models
    "s1_full": "runs/probe/journal-full.jsonl",
    "s1_full2": "runs/probe/journal-full2.jsonl",
    "s1_wording": "runs/probe/journal-wording.jsonl",
    "s2": "runs/stage2/journal.jsonl",
    "s3": "runs/stage3/journal.jsonl",
    "s5": "runs/stage5/journal.jsonl",
    "s6": "runs/stage6/journal.jsonl",
    "margin": "runs/margin/journal.jsonl",
    "backend": "runs/backend_check/journal.jsonl",
    "backend6000": "runs/backend_check/journal-6000.jsonl",
    "backend12000": "runs/backend_check/journal-12000.jsonl",
}
FMTS = {"paired": 0, "parallel": 1, "products": 2, "folded": 3}
HEUR_N = (5, 10, 20, 40)
CODES = {"arm": {"jev": 0, "swap": 1}, "split": {"train": 0, "val": 1, "test": 2},
         "pair": {"0v1": 0, "3v8": 1}, "backend": {"typesafe": 0, "vercel": 1}}
TAG_INTS = ("seed", "epoch", "layer", "rep", "eval", "i", "j", "k")
COLS = (["src", "fmt", "q2", "backend", "arm", "split", "pair", *[f"tag_{t}" for t in TAG_INTS],
         "k", "p", "z", "spread", "m", "bias", "frac_zero", "frac_tiny", "frac_pos", "pos_mass", "neg_mass",
         "sq_first", "sq_mid", "sq_last", "sum_first", "sum_mid", "sum_last", "dom", "dom_sign", "dom_pos",
         "tokens"]
        + [f"{h}{n}" for h in ("pre", "suf", "rnd") for n in HEUR_N] + ["pre_half", "suf_half"])


def parse(state):
    """(terms as Jev sees them in order, bias, format) or None for non-neuron states."""
    if "products" in state and "bias" not in state:
        v = np.asarray(state["products"], float)
        return (v[:-1], float(v[-1]), "folded") if len(v) else (v, 0.0, "folded")
    if "products" in state:
        return np.asarray(state["products"], float), float(state["bias"]), "products"
    if "terms" in state:
        return np.array([t["x"] * t["w"] for t in state["terms"]]), float(state["bias"]), "paired"
    if "x" in state and "w" in state:
        return np.asarray(state["x"], float) * np.asarray(state["w"], float), float(state["bias"]), "parallel"
    return None


def features(t, b, rng):
    k = len(t)
    z = round(float(t.sum()) + b, 6)
    spread = float(np.sqrt(np.sum(t * t) + b * b))
    s = spread or 1.0
    third = [t[: k // 3], t[k // 3: k - k // 3], t[k - k // 3:]] if k >= 3 else [t, t[:0], t[:0]]
    sq = np.sum(t * t) or 1.0
    d = int(np.argmax(np.abs(t))) if k else 0
    f = {"k": k, "z": z, "spread": spread, "m": z / s, "bias": b,
         "frac_zero": float(np.mean(t == 0)) if k else 0.0, "frac_tiny": float(np.mean(np.abs(t) <= 0.02)) if k else 0.0,
         "frac_pos": float(np.mean(t > 0)) if k else 0.0,
         "pos_mass": float(t[t > 0].sum()) / s, "neg_mass": float(-t[t < 0].sum()) / s,
         "sq_first": float(np.sum(third[0] ** 2) / sq), "sq_mid": float(np.sum(third[1] ** 2) / sq),
         "sq_last": float(np.sum(third[2] ** 2) / sq),
         "sum_first": float(third[0].sum()) / s, "sum_mid": float(third[1].sum()) / s, "sum_last": float(third[2].sum()) / s,
         "dom": float(np.abs(t[d])) / s if k else 0.0, "dom_sign": float(np.sign(t[d])) if k else 0.0,
         "dom_pos": d / max(k - 1, 1),
         "pre_half": (float(t[: k // 2].sum()) + b) / s, "suf_half": (float(t[k - k // 2:].sum()) + b) / s}
    for n in HEUR_N:  # partial sums + bias, normalized by spread; random subset is the position-free null
        f[f"pre{n}"] = (float(t[:n].sum()) + b) / s
        f[f"suf{n}"] = (float(t[-n:].sum()) + b) / s
        f[f"rnd{n}"] = (float(t[rng.choice(k, min(n, k), replace=False)].sum()) + b) / s if k else b / s
    return f


def chunk(job):
    src, path, start, end = job
    rows, hashes = [], []
    rng = np.random.default_rng(start)
    with open(path, "rb") as fh:
        if start:  # the line straddling start belongs to the previous chunk
            fh.seek(start - 1)
            fh.readline()
        while fh.tell() <= end:
            line = fh.readline()
            if not line:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line still being written
                continue
            ans = r.get("answers") or {}
            if len(ans) != 1 or "fires" not in ans:  # scalar or packed calls
                continue
            parsed = parse(r["state"])
            if parsed is None:
                continue
            t, b, fmt = parsed
            tag = r.get("tag") or {}
            f = features(t, b, rng)
            f.update(src=list(SOURCES).index(src), fmt=FMTS[fmt], p=ans["fires"],
                     q2=float("Is the total positive" in r["questions"]["fires"]["instructions"]),
                     tokens=(r.get("usage") or {}).get("input_tokens", 0),
                     **{c: CODES[c].get(tag.get(c) if c != "backend" else r.get("backend"), -1) for c in CODES},
                     **{f"tag_{c}": tag[c] if isinstance(tag.get(c), int) else -1 for c in TAG_INTS})
            rows.append([f[c] for c in COLS])
            h = hashlib.blake2b(t.tobytes() + np.float64(b).tobytes() + bytes([FMTS[fmt]]), digest_size=8).digest()
            hashes.append(int.from_bytes(h, "little", signed=True))
    return src, np.array(rows, np.float64).reshape(-1, len(COLS)), np.array(hashes, np.int64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for src, rel in SOURCES.items():
        path = ROOT / rel
        size = os.path.getsize(path)  # snapshot: stage 6 keeps appending
        n = max(1, size // 50_000_000)
        edges = np.linspace(0, size, n + 1).astype(int)
        jobs += [(src, path, int(a), int(b) - 1 if i < n - 1 else size) for i, (a, b) in enumerate(zip(edges, edges[1:]))]
    parts, hashes = [], []
    with ProcessPoolExecutor(8) as ex:
        for src, arr, h in ex.map(chunk, jobs):
            parts.append(arr)
            hashes.append(h)
    data = np.concatenate(parts)
    cols = {c: data[:, i].astype(np.float32 if c not in ("z", "spread", "m", "p") else np.float64) for i, c in enumerate(COLS)}
    np.savez(OUT / "features.npz", hash=np.concatenate(hashes), sources=np.array(list(SOURCES)), **cols)
    counts = np.bincount(data[:, 0].astype(int), minlength=len(SOURCES))
    print({s: int(c) for s, c in zip(SOURCES, counts)}, "total", len(data))


if __name__ == "__main__":
    main()
