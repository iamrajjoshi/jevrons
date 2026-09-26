"""Which rule do Jev's full-state answers follow? Compares Jev's yes/no against candidate shortcuts."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def features(state):
    b = state["bias"]
    if "terms" in state:
        xw = np.array([t["x"] * t["w"] for t in state["terms"]])
        w = np.array([t["w"] for t in state["terms"]])
    elif "products" in state:
        xw = np.array(state["products"])
        w = None
    else:
        xw = np.array(state["x"]) * np.array(state["w"])
        w = np.array(state["w"])
    return b, xw, w


def main(journal):
    rows = []
    for line in open(journal):
        r = json.loads(line)
        t = r["tag"]
        if t.get("part") != "full" or t["rep"] != 0:
            continue
        b, xw, w = features(r["state"])
        rows.append({"fmt": t["fmt"], "k": t["k"], "p": r["answers"]["fires"], "z": t["z"],
                     "margin": t["margin"], "b": b, "sum_xw": xw.sum(),
                     "sum_w": None if w is None else w.sum(),
                     "npos": np.mean(xw > 0), "first10": xw[:10].sum() + b, "last10": xw[-10:].sum() + b})
    print(f"{len(rows)} neurons\n")
    rules = {
        "true sign of total": lambda r: r["z"] > 0,
        "sign of bias alone": lambda r: r["b"] > 0,
        "sign of products, ignoring bias": lambda r: r["sum_xw"] > 0,
        "majority of products positive": lambda r: r["npos"] > 0.5,
        "first 10 terms + bias": lambda r: r["first10"] > 0,
        "last 10 terms + bias": lambda r: r["last10"] > 0,
        "always yes": lambda r: True,
    }
    fmts = sorted({r["fmt"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    print("Agreement between Jev's answer (p >= 0.5) and each rule, all neurons:")
    for name, rule in rules.items():
        per = [np.mean([(r["p"] >= 0.5) == rule(r) for r in rows if r["k"] == k]) for k in ks]
        print(f"  {name:<34} {np.mean([(r['p'] >= 0.5) == rule(r) for r in rows]):.2f}   by k {ks}: "
              + " ".join(f"{a:.2f}" for a in per))
    print("\nJev accuracy vs |margin| (true total / spread of terms):")
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 1.5), (1.5, 2)):
        sel = [r for r in rows if lo <= abs(r["margin"]) < hi]
        print(f"  {lo:.2f}-{hi:.2f}: acc {np.mean([(r['p'] >= 0.5) == (r['z'] > 0) for r in sel]):.2f}  n={len(sel)}")
    print("\nMean p by sign of true total, per format:")
    for f in fmts:
        pos = [r["p"] for r in rows if r["fmt"] == f and r["z"] > 0]
        neg = [r["p"] for r in rows if r["fmt"] == f and r["z"] <= 0]
        print(f"  {f:<9} total>0: {np.mean(pos):.2f}   total<=0: {np.mean(neg):.2f}")
    corr = np.corrcoef([r["p"] for r in rows], [np.tanh(r["margin"]) for r in rows])[0, 1]
    print(f"\ncorr(p, tanh(margin)) = {corr:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/probe/journal-full.jsonl")
