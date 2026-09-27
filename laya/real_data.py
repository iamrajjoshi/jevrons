"""Collect the real neuron states Jev has answered: folded-bias products lists asked with the standard
folded question, from every journal whose states are Jev's (not laya's or other Ollaya models').
Writes runs/stage10/real.jsonl.gz: one line per distinct state with its true total and Jev's mean p.
Stage 6 is read from the 20:46 backup, since its live journal is still being written.
Usage: uv run python real_data.py
"""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = Path.home() / "jevrons-journal-backup/20260926-2046/runs"
SOURCES = {
    "s1_wording": ROOT / "runs/probe/journal-wording.jsonl",
    "s1_full2": ROOT / "runs/probe/journal-full2.jsonl",
    "s2": ROOT / "runs/stage2/journal.jsonl",
    "s3": ROOT / "runs/stage3/journal.jsonl",
    "s5": ROOT / "runs/stage5/journal.jsonl",
    "s6": BACKUP / "stage6/journal.jsonl.gz",
    "margin": ROOT / "runs/margin/journal.jsonl",  # held out entirely: it is the Jev reference curve
    "backend": ROOT / "runs/backend_check/journal.jsonl",
    "backend6000": ROOT / "runs/backend_check/journal-6000.jsonl",
    "backend12000": ROOT / "runs/backend_check/journal-12000.jsonl",
}
INSTR = "Add all the numbers in products together."
MAX_TERMS = 97  # nothing longer can fit laya's 512-token context


def scan(item):
    src, path = item
    states = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if '"products"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            st, qs = r.get("state") or {}, r.get("questions") or {}
            if set(st) != {"products"} or set(qs) != {"fires"} or qs["fires"]["instructions"] != INSTR:
                continue
            v = st["products"]
            if not 2 <= len(v) <= MAX_TERMS:
                continue
            key = json.dumps(v)
            e = states.setdefault(key, [0.0, 0])
            e[0] += r["answers"]["fires"]
            e[1] += 1
    return src, states


def main():
    out = ROOT / "runs/stage10/real.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged, counts = {}, {}  # a state seen in several journals keeps its first source and pools its answers
    with ProcessPoolExecutor(6) as ex:
        for src, states in ex.map(scan, SOURCES.items()):
            for key, (ps, c) in states.items():
                e = merged.setdefault(key, [src, 0.0, 0])
                e[1] += ps
                e[2] += c
    with gzip.open(out, "wt") as f:
        for key, (src, ps, c) in merged.items():
            v = json.loads(key)
            f.write(json.dumps({"src": src, "products": v, "z": round(sum(v), 6), "p_jev": ps / c, "n": c}) + "\n")
            counts[src] = counts.get(src, 0) + 1
    print(counts, "total", sum(counts.values()))


if __name__ == "__main__":
    main()
