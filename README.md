# Jevrons

A handwritten-digit classifier where every neuron is a live API call to [Jev](https://typesafe.ai), trained with backprop anyway. [Try the demo](https://jevrons.rajjoshi.me).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](.python-version)

![The Jevrons demo replaying a live run on the sparse network: a hand-drawn 3 on the left; 64 hidden Jev neurons, each tile showing its 96-pixel receptive field, with three answers that disagree with the arithmetic outlined in pink; output 3 firing at 0.89; and a timeline of the 74 calls below.](demo/screenshots/04-done.png)

[Jev](https://docs.typesafe.ai) is [TypeSafe AI](https://typesafe.ai)'s System-1 model. You send it a JSON state and a yes/no question, and it sends back a probability. There's no gradient to take and no weights to tune. So each neuron here is one question: the code multiplies the pixels by the weights, sends Jev the list of products, and asks whether they add up to more than zero. Jev's probability is the neuron's output.

Jev is [not great at arithmetic](https://docs.typesafe.ai/model-jaggedness/jev-1.13#math-and-numbers). Training uses a straight-through estimator: the forward pass uses Jev's real answer, and the backward pass pretends Jev was a smooth function of the sum it was asked about. The weights end up working around Jev's mistakes. The test that shows it is a swap: take a network trained with perfect arithmetic and drop Jev in as its neuron. It loses 7 to 31 points, depending on how long the sums are.

The best network gives each of its 64 hidden neurons its own 96 pixels, so every call is a short sum (about 37 numbers) that Jev handles well. Trained on 1,000 images, it reads **85.3%** of 1,000 test digits with every neuron a Jev call. A dense 784-32-10 network trained on 5,000 images gets 84.6%, and its perfect-math twin falls from 83.9% to 52.6% on Jev.

| Network | Exact neuron | Live Jev |
| --- | --- | --- |
| Sparse 784-64-10, trained through Jev, 1,000 images (stage 8a) | 75.7% | **85.3%** |
| Sparse, trained with exact math (stage 8a) | 72.6% | 65.8% |
| Dense 784-32-10, trained through Jev, 5,000 images (stage 7) | 70.0% | 84.6% |
| Dense, trained with exact math (stage 7) | 83.9% | 52.6% |

The rest of the repo is how it got there, one experiment at a time: probing a single Jev neuron, logic gates, a circle, two digits, ten digits, what the trained weights reveal about Jev, and teaching a small open model ([laya](https://huggingface.co/convaiinnovations/laya)) to be a better neuron than Jev. All the Jev calls together cost about $230.

## The neuron

```text
z = Σ w_i x_i + b                       computed locally, never sent
state = {"products": [x_1 w_1, ..., x_k w_k, b]}
p = Jev(state)                          P("the total is greater than zero"), the activation
∂L/∂z ≈ ∂L/∂p · σ'(z/τ)/τ               straight-through: backprop as if p = σ(z/τ)
```

The stage 7 and 8a networks ask each neuron two questions in one call (the yes/no above and a neutral above/below choice) and averages them, because the two lean in opposite directions ([stage 8g](docs/results/stage8g-neuron-fixes.md)). Its backward pass uses the slope of Jev's measured response curve instead of the sigmoid. The client is [`src/jevrons/jev.py`](src/jevrons/jev.py) and the training loop is [`src/jevrons/net.py`](src/jevrons/net.py).

## Quickstart

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
./scripts/fetch_mnist.sh        # MNIST into data/, sha256-checked
```

The demo runs without an API key. It serves the draw-a-digit page on http://127.0.0.1:8765 with exact math as the neurons, and can replay recorded Jev answers (`?source=replay`).

```bash
uv run python demo/server.py
uv run python demo/server.py --check    # self-test: exact == training code, replay == journal
```

Offline analysis and figures also cost nothing. `jevrons.figures` redraws every figure in the blog post from the saved run files, and stage 3 has a local mode.

```bash
uv run python -m jevrons.figures stage7     # docs/figures/stage7-swap.png; no argument redraws all
uv run python -m jevrons.stage3 local
```

### Live runs

Live runs call Jev and are billed, at about $0.042 per million input tokens. The client reads `TYPESAFE_API_KEY` ([TypeSafe](https://typesafe.ai) direct) and `VERCEL_API_KEY` ([Vercel AI Gateway](https://vercel.com/ai-gateway), pinned to the TypeSafe upstream) from the environment, or from `~/.config/jev-research/credentials.env` as `KEY=value` lines. `JEVRONS_BACKENDS=vercel` (or `typesafe`) restricts the router to one backend.

```bash
export VERCEL_API_KEY=...
JEVRONS_BACKENDS=vercel uv run python demo/server.py --live --daily-usd 0.05   # one live draw is ~$0.002
uv run python -m jevrons.stage2                                              # XOR, ~33k calls, ~$0.46
```

Each stage script resumes from the result files already in `runs/stageN/`, so a fresh run of a finished stage needs that directory moved aside first. Check the cost column below before you start one. Stage 7 alone was 1.94M calls and $84.71, and only the stage 6b traps, 7, 8a and 8c scripts enforce a spend cap in code.

## Experiments

| Stage | Question | Result | Cost |
| --- | --- | --- | --- |
| 0. [Local capacity](spikes/) | Can a hard-step 784-32-10 net learn MNIST from 1,000 images? | 83.4% validation with step units; swapping sigmoid for step costs under 1 point | $0 |
| 1. [Probe](docs/results/stage1-probe.md) | Does one Jev call add up a weighted sum? | 82-93% sign accuracy up to 75 terms; not deterministic; over-trusts a separate bias | $0.48 |
| 2. [Logic](docs/results/stage2-logic.md) | Can a Jev network learn XOR? | 3 of 3 seeds, only with the bias folded into the list | $0.46 |
| 3. [Circle](docs/results/stage3-circle.md) | Does training through Jev beat swapping Jev in? | 84-88% trained through Jev, 47-70% swapped | $1.16 |
| 4. [Scalar digits](docs/results/stage4-scalar-digits.md) | Is Jev a good step function when the sum is done in code? | 79.1% on Jev vs 78.5% exact; fires as intended 99.9% of the time | $1.10 |
| 5. [Two digits](docs/results/stage5-two-digits.md) | Do full-state Jev neurons learn digits? | 0 vs 1: 99.4%, 3 vs 8: 94.2%; swap drops 3 vs 8 to 67.8% | $13.88 |
| 6. [Ten digits](docs/results/stage6-ten-digits.md) | And on all ten digits? | 68.5% trained through Jev vs 48.0% swapped (3 seeds) | ~$70 |
| 6b. [Jev's quirks](docs/results/stage6b-jev-quirks.md) | What does training find out about Jev? | Sign voting, a yes-lean on long lists, compressed p | $0.08 |
| 6c. [Weights as a probe](docs/results/stage6c-weights-probe.md) | Do the weights read out the neuron's bias? | Only at the extremes; Jev repeats its own misfires | $0 |
| 7. [More data](docs/results/stage7-more-data.md) | What do 5,000 images and the two-question neuron give? | 84.6% on Jev; swap 52.6% | $84.71 |
| 8g. [Neuron fixes](docs/results/stage8g-neuron-fixes.md) | Can a Jev neuron behave like the neuron we meant? | Two averaged questions cut the 3 vs 8 swap gap from 25.0 to 4.6 points | $28.99 |
| 8a / 8c. Sparse neurons, mock pretraining | Do shorter sums or local pretraining cut cost without losing accuracy? | Each of 64 hidden neurons sees 96 pixels (about 37 numbers per call): 85.3% on Jev with 1,000 training images, swap gap 6.8 points. Pretraining on a simulated Jev alone gets 83.8% with no Jev training calls ([8a](docs/results/stage8a-sparse.md), [8c](docs/results/stage8c-mock-pretrain.md)) | $29 |
| 9. [Open-weight neurons](docs/PLAN.md#stage-9-open-weight-neurons-planned) | Can a local System-1 model via [Ollaya](https://ollaya.dev) stand in for Jev? | Probed laya, von and decider locally; decider learns XOR | $0 |
| 9c. [Laya as a scalar](docs/results/stage9c-laya-scalar.md) | Can a neuron that answers wrong still be trained through? | 52% sign accuracy, yet 96.0% on 3 vs 8 trained through it; swap 50.0% | $0 |
| 10. [Teaching laya](docs/results/stage10-teaching-laya.md) | Can a 421M open model be fine-tuned into a better neuron than Jev? | 97-99% sign accuracy; swap gap 1 point | $0 (local GPU) |

The design and staged gates are in [docs/proposal.md](docs/proposal.md), and decisions and status in [docs/PLAN.md](docs/PLAN.md).

## Repo layout

```text
src/jevrons/      Jev client (jev.py), network and straight-through training (net.py), one module per stage, figures.py
scripts/          fetch_mnist.sh
spikes/           stage 0: local capacity checks and a noisy-neuron mock, with saved output
runs/             per-stage results, weights and curves (raw Jev call journals aren't published)
docs/results/     one write-up per stage
docs/figures/     figures for the write-ups and the blog post (PNG and SVG)
demo/             draw-a-digit web demo, replay data, Fly.io deploy files
laya/             stage 10 subproject (own uv env with torch): fine-tuning laya into a neuron
```

## Demo

Live at [jevrons.rajjoshi.me](https://jevrons.rajjoshi.me). `demo/server.py` is a single-file server with a static page: draw a digit and it runs on its own, through the sparse network or the stage 7 one, on live Jev or exact math, optionally next to its perfect-math twin. Click any neuron to see the exact call it sent. It caps live spend per day and keeps nothing on disk. [demo/README.md](demo/README.md) covers the flags and film mode; pushes to `main` deploy it to one Fly.io machine (`.github/workflows/demo.yml`).

## Blog post

The full story, with the derivations and dead ends: [Training a Neural Network Made of Jev Calls](https://rajjoshi.me/blog/jevrons).

## Credit

This project exists because of [Mustafa Akın's 4-bit ALU made of Jev NAND gates](https://x.com/mustafaakin/status/2103574428475154635): 7 + 5 = 12 in 116 gates, 7.6 seconds and $0.0018. His circuit had fixed wiring; Jevrons asks what happens when the wiring can learn.

MNIST is by Yann LeCun, Corinna Cortes and Christopher Burges. It isn't redistributed here; `scripts/fetch_mnist.sh` downloads it from Google's CVDF mirror. Stage 10 fine-tunes [laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0) and serves it through [Ollaya](https://github.com/ollaya-dev/ollaya) (Apache-2.0); neither's code nor weights are included, and `laya/README.md` says how to fetch them.

## Citation

If you build on this, cite it with [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button reads it).

```bibtex
@software{joshi2026jevrons,
  author = {Joshi, Raj},
  title  = {Jevrons: training a neural network whose neurons are Jev calls},
  year   = {2026},
  url    = {https://github.com/iamrajjoshi/jevrons}
}
```

## License

Code is [MIT](LICENSE).
