# Pretraining on a fake Jev: pretrain on a simulated neuron, fine-tune on Jev

2026-09-27 to 28 · 407,000 calls · $10.20 · Vercel gateway only

A network trained only on a simulated Jev neuron, with no Jev calls at all in training, gets 83.8% on Jev on 1,000 test images. The sparse network with the same architecture, data and start, trained six epochs through Jev for $11.05 of training calls, gets 85.3%. Two epochs of fine-tuning on Jev ($3.69) raised validation from 81.0% to 83.4% but left test where it was, at 83.8%. So most of what Jev training buys here can be had offline, and the fine-tuning's contribution is within noise on one seed.

The simulated neuron is the reason. An exact-trained network with the same start keeps only 65.8% on Jev (the sparse network's swap arm). Training through a neuron that answers like Jev's arm E (graded, blurred, leaning slightly toward "no") gets 18 points of that back for free.

Code: `src/jevrons/stage8c.py` (`SimE` and the shared pieces are in `stage8a.py`). Results, weights, curves: `runs/stage8c/`. The journals aren't published.

## Setup

[The sparse network](stage8a-sparse.md) exactly: 784-64-10, 96 random connections per hidden neuron (about 37 numbers per call), arm E neuron, measured-curve backward, init N(0, 0.3), the first ten-digit run's seed-0 split (1,000 train, 500 val) and 1,000 test images. Combining with the sparse network was the obvious design. Its Jev-trained run is then the matching Jev-only baseline at no extra cost, and every call in fine-tuning is a cheap sparse one.

1. Pretrain 6 epochs through `SimE`, Adam 0.02, batch 32, same batches as the sparse network. Free, about a minute.
2. Fine-tune 2 epochs through Jev with a fresh Adam at 0.01 (a local check below picked it).
3. Validate on Jev before fine-tuning and after each epoch, then test once. The pretrained network was also tested on Jev directly.

`SimE` is the `CurveNeuron` form from reading the weights, p = floor + (1 − 2·floor)·Φ(k(n)/λ · (m − shift)), at the sparse term count n, with a constant threshold shift in place of the yes/no neuron's drift m0(n), since arm E averages two questions that lean opposite ways. It returns the graded p, so it is deterministic for a given state, as Jev's decisions mostly are (86-88% of misfires repeat). λ, floor and shift were grid-fitted per layer to the dense network's 192,000 arm E test calls (152,000 hidden, 40,000 output) (mean squared error of p):

| Layer | λ | floor | shift | rms error per call |
| --- | --- | --- | --- | --- |
| Hidden (pixels, ~150 numbers in the dense network) | 4.5 | 0.04 | +0.5 | 0.087 |
| Output (hidden p values, 33 numbers) | 2.0 | 0.04 | +0.8 | 0.040 |

Arm E's graded p is much softer than the margin probe's fire curve (λ = 4.5 means a quarter of the sharpness), and it crosses 0.5 about half a spread above zero on hidden sums and 0.8 above on output sums. The fit is to dense-network states. It wasn't refitted on sparse calls, which only existed after the sparse network ran.

## Results

| | Jev training spend | Val on Jev (500) | Test on Jev (1,000) | Same weights, exact neuron (test) | Hidden / output fire as intended (test) | No output fires (test) |
| --- | --- | --- | --- | --- | --- | --- |
| Trained exact (sparse network swap) | $0 | 66.4% | 65.8% | 72.6% | 91.6% / 98.3% | 81.0% |
| Pretrained on SimE only | $0 | 81.0% | **83.8%** | 57.1% | 90.9% / 95.6% | 9.0% |
| + 1 epoch on Jev | $1.84 | 82.4% | | | | |
| + 2 epochs on Jev | $3.69 | 83.4% | **83.8%** | 79.0% | 90.3% / 97.9% | 57.3% |
| Jev only (sparse network), 2 epochs | $3.67 | 78.4% | | | | |
| Jev only (sparse network), 4 epochs | $7.36 | 82.6% | | | | |
| Jev only (sparse network), 6 epochs | $11.05 | 83.8% | **85.3%** | 75.7% | 90.7% / 97.7% | 54.6% |

Training spend counts only training calls. Each validation pass cost $0.93 and each test pass $1.86. With evaluations, this run cost $8.34 and the pretrained-only test $1.86. At equal Jev training spend ($3.7, two epochs), pretrain plus fine-tune leads Jev-only training by 5 points on validation (83.4% against 78.4%). Jev-only training needed two and a half to three times the spend to reach the same validation accuracy.

SimE was a good predictor of Jev for SimE-trained weights: 84.8% on SimE against 83.8% on Jev (test). It was a bad one for exact-trained weights. The local check put the sparse network's configuration trained exact at 35.0% on SimE, and on Jev it got 66.4%. SimE is harsher than Jev on a network that never saw a graded neuron.

## Per digit (test)

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained on SimE only | 97% | 97% | 84% | 89% | 85% | 80% | 89% | 90% | 49% | 78% |
| + 2 epochs on Jev | 97% | 93% | 86% | 83% | 80% | 66% | 88% | 88% | 74% | 83% |
| Jev only (sparse network) | 96% | 94% | 82% | 84% | 87% | 74% | 90% | 91% | 82% | 73% |
| Trained exact (sparse network swap) | 99% | 70% | 93% | 61% | 62% | 51% | 38% | 70% | 29% | 85% |

Fine-tuning didn't move the average, but it moved the digits. 8 went from 49% to 74% and 5 from 80% to 66%. The pretrained network has an output firing on 91% of test images. After fine-tuning, only 43% do, about the same as the Jev-only network (45%). On Jev the output layer settles into the same regime whichever way it starts: a firing output on under half the images and a ranking of sub-threshold p values on the rest.

## Local check of the fine-tuning ($0)

`stage8c local` pretrained on SimE and then fine-tuned on two differently tuned SimE neurons standing in for Jev (validation, 500 images):

| Stand-in | Trained on it directly, 6 epochs | Pretrained, not fine-tuned | Fine-tune lr 0.02: after 1 / 2 epochs | lr 0.01 | lr 0.005 |
| --- | --- | --- | --- | --- | --- |
| Less lean, sharper | 84.2% | 84.0% | 84.8 / 86.2% | 86.2 / 87.2% | 86.0 / 86.8% |
| More lean, softer | 80.0% | 80.4% | 83.0 / 84.2% | 84.4 / 85.0% | 84.8 / 84.8% |

This checked the plumbing and picked lr 0.01. It over-promised: every stand-in was a SimE, and on real Jev the pretrained start was already where direct training ended up.

## Caveats

- One seed and one split. The pretrained-only and fine-tuned networks tie on test (83.8%) while validation says fine-tuning added 2.4 points. With 500 validation images (standard error about 1.7 points) and 1,000 test images (about 1.2), neither difference is established, and neither is the 1.5-point gap to Jev-only training.
- SimE was fitted to Jev answers from the dense network. That's Jev data paid for earlier (part of the dense network's $84.71), not free. The fit transfers to sparse calls well enough here, but a different neuron or question set would need its own fitted curve.
- Two epochs of fine-tuning is a budget choice. The validation curve (81.0, 82.4, 83.4%) was still rising.
- Pretraining ran six epochs to match the sparse network. More offline epochs cost nothing and weren't tried.
