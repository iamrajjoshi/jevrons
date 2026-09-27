"""Stage 6c, step 0: stream the stage 5, 6 and 8g journals and keep every validation and test call
as numbers (which run, which image and neuron, the intended sum z, and the p the neuron returned).

Training calls are skipped by a substring check before parsing, so the 2.2 GB stage 6 journal is read
line by line without holding it. Stage 8g's p is what its neuron returned: the mean over its questions
(the choice question contributes its "above" probability). Output: runs/stage6c/calls.npz.
Usage: uv run python -m jevrons.stage6c_calls
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage6c"
SOURCES = {"s5": "runs/stage5/journal.jsonl", "s6": "runs/stage6/journal.jsonl", "s8g": "runs/stage8g/journal.jsonl"}
ARMS = {"jev": 0, "swap": 1, "B": 2, "C": 3, "D": 4, "E": 5}
SPLITS = {"val": 1, "test": 2}
PAIRS = {"0v1": 0, "3v8": 1}
COLS = ("src", "arm", "swap", "split", "pair", "seed", "epoch", "eval", "layer", "i", "j", "z", "p")


def p_of(ans):
    return float(np.mean([a["above"] if isinstance(a, dict) else a for a in ans.values()]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, (src, rel) in enumerate(SOURCES.items()):
        n = 0
        with open(ROOT / rel, "rb") as fh:
            for line in fh:
                if b'"split": "train"' in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("tag") or {}
                if t.get("split") not in SPLITS or not r.get("answers"):
                    continue
                rows.append((s, ARMS[t["arm"]], int(bool(t.get("swap"))), SPLITS[t["split"]], PAIRS.get(t.get("pair"), -1),
                             t.get("seed", -1), t.get("epoch", -1), t.get("eval", -1), t["layer"], t["i"], t["j"],
                             t["z"], p_of(r["answers"])))
                n += 1
        print(src, n, flush=True)
    a = np.array(rows, np.float64)
    np.savez_compressed(OUT / "calls.npz", **{c: a[:, k] for k, c in enumerate(COLS)})
    print("total", len(a))


if __name__ == "__main__":
    main()
