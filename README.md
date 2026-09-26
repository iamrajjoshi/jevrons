# Jevrons

Training a network around neurons it can't see inside: an MNIST classifier whose neurons are frozen Jev evaluations, with only the outer weights learned.

Inspired by [Mustafa Akın's NAND-gate ALU built from Jev calls](https://x.com/mustafaakin/status/2103574428475154635) (7 + 5 = 12 in 116 gates, 7.6 s, $0.0018). That had fixed wiring; Jevrons adds learning. The question is whether learned connections can compensate for an imperfect model acting as a neuron. The plan is in [docs/proposal.md](docs/proposal.md); the pre-audit version is [docs/original-proposal.md](docs/original-proposal.md).

## Status

Stage 0 (local capacity check) is done. No Jev calls have been made yet.

## Reproduce the local spikes

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
./scripts/fetch_mnist.sh
uv run spikes/baseline.py
uv run spikes/mock_b.py
```

`baseline.py` compares 784-H-10 networks (H = 32, 64, 128) with sigmoid and hard-step hidden units on 1,000 training images, plus a 50,000-image reference. `mock_b.py` simulates a noisy full-state neuron and compares swap-after-training against training through the noise. Both are deterministic; saved output is in `spikes/results/`. The baseline takes a few minutes on a laptop.
