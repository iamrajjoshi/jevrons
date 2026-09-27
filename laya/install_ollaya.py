"""Install a fine-tuned checkpoint as an Ollaya model, e.g. laya-neuron:latest.

Ollaya 0.7.2's Modelfile only takes FROM (a registry name), PARAMETER, QUESTIONS, CALIBRATION and LICENSE,
so `ollaya create` can't swap weights. This does what create would: it copies laya:en's manifest, points
its weights layer at the new safetensors (same keys and dtypes, so the same arch layer, tokenizer,
decision and calibration layers apply) and writes the blob and manifest into Ollaya's store. laya:en
is untouched; `ollaya rm <name>` undoes it. The running server picks the model up without a restart.
Usage: uv run python install_ollaya.py truth laya-neuron
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import data

STORE = Path.home() / ".ollaya/models"


def put(path: Path | None = None, blob: bytes | None = None):
    h = hashlib.sha256()
    if blob is None:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    else:
        h.update(blob)
    digest = h.hexdigest()
    dst = STORE / "blobs" / f"sha256-{digest}"
    if not dst.exists():
        shutil.copyfile(path, dst) if blob is None else dst.write_bytes(blob)
    return f"sha256:{digest}", dst.stat().st_size


def main(run, name):
    weights = data.ROOT / f"runs/stage10/{run}/model.safetensors"
    man = json.loads((STORE / "manifests/ollaya.dev/library/laya/en").read_text())
    cfg = json.loads((STORE / "blobs" / man["config"]["digest"].replace(":", "-")).read_text())
    cfg.update(description=f"laya:en fine-tuned by Jevrons stage 10 (runs/stage10/{run}): is the sum of a list of numbers positive?",
               source=f"jevrons runs/stage10/{run}")
    digest, size = put(blob=json.dumps(cfg, indent=2).encode())
    man["config"] = {"mediaType": man["config"]["mediaType"], "digest": digest, "size": size}
    for layer in man["layers"]:
        if layer["mediaType"] == "application/vnd.ollaya.weights":
            digest, size = put(weights)
            layer.update(digest=digest, size=size)
            layer.pop("urls", None)
    out = STORE / f"manifests/ollaya.dev/library/{name}/latest"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2))
    print(f"installed {name}:latest from {weights} (weights {digest})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
