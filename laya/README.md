# laya/: teaching laya to be a Jevrons neuron (stage 10)

A self-contained sub-project with its own uv environment, so torch and transformers never enter the
main project. Results: [docs/results/stage10-teaching-laya.md](../docs/results/stage10-teaching-laya.md).

`hf/` holds laya's own inference code and configs from
[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) at revision `aa8c91ca`
(Apache-2.0). `hf/model.safetensors` and `hf/tokenizer/tokenizer.json` are symlinks to the identical
blobs Ollaya already stores for `laya:en` (same sha256 as the Hugging Face LFS objects), so nothing large
was downloaded twice. On a machine without Ollaya, download those two files from the same revision.

## Commands

Run from this directory. Each step writes to `../runs/stage10/`.

```bash
uv sync                                   # torch + transformers, this directory only
uv run python real_data.py                # real Jev-answered states from the journals -> real.jsonl.gz
uv run python data.py                     # self-check of the synthetic generator
uv run python train.py truth --name truth --steps 3000   # true-math arm
uv run python evaluate.py base            # -> eval-base.json (unmodified laya)
uv run python evaluate.py truth           # -> eval-truth.json
uv run python payoff.py truth             # sparse 3 vs 8 through the laya neuron, plus the swap control
uv run python install_ollaya.py truth laya-neuron   # serve it: adds laya-neuron:latest to Ollaya's store
uv run python client_check.py ollaya-laya-neuron truth   # probes through Jev(..., backends=["ollaya-laya-neuron"])
```

`real_data.py` reads the Jev journals, which aren't published. The figures come from the main project:
`uv run python -m jevrons.figures stage10 stage10_accuracy stage10_training stage10_traps` from the repo root.

Remove the served model with `ollaya rm laya-neuron`. From the main project, the backend is
`Jev(journal, backends=["ollaya-laya-neuron"])` (512-token context, so about 99 terms at most).

Training takes about 2.4 s per step (batch 16) on an M5 Pro's GPU and peaks near 20 GB of MPS memory.

## Files

- `laya_model.py`: loads laya and scores noul questions exactly as laya's `rl_agent_api.py` does
  (reproduces Ollaya's `laya:en` answers to within its 4-decimal rounding).
- `data.py`: the synthetic state generator and the real-state train/test split.
- `real_data.py`: collects real states from the journals.
- `train.py`: full fine-tune, cross-entropy on laya's calibrated noul probability.
- `evaluate.py`: margin curves, real states, traps, scalar sweep, speed.
- `payoff.py`: the sparse 3 vs 8 network through the fine-tuned neuron.
- `install_ollaya.py`: installs a checkpoint as an Ollaya model (Ollaya's Modelfile can't swap weights).
- `client_check.py`: checks the served model through the normal Jev client against the offline answers.
