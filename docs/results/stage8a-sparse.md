# The sparse network: sparse Jev neurons

2026-09-27 · 756,500 calls · $18.80 · Vercel gateway only (TypeSafe direct out of credits)

Giving each hidden neuron 96 random pixels instead of all 784 cuts its sums from about 150 numbers to 37, and it nearly removes the swap gap on ten digits. A 784-64-10 network trained through Jev gets 85.3% on 1,000 test images from 1,000 training images, about what [the dense network](stage7-more-data.md) reached from 5,000 (84.6%, on a different test sample). The swap control is the bigger change. The same architecture trained with an exact neuron keeps 65.8% on Jev against 72.6% on an exact neuron, a 6.8-point gap. Dense ten-digit networks lost 29 (the first ten-digit run) and 31 points (the dense network).

Calls cost 43% less (593 input tokens against the dense network's 1,037), but the network has twice as many hidden neurons, so an image costs the same as on the dense network: $1.85 per 1,000 forward passes.

Code: `src/jevrons/stage8a.py` (and a one-line `mask` option in `net.fit`). Results, weights, curves: `runs/stage8a/`. Journals are gitignored and backed up outside git (`~/jevrons-journal-backup/20260927-2158-stage8a/`).

## Setup

784-64-10. Each hidden neuron connects to 96 pixels drawn at random from the 379 that are nonzero in at least 5% of training images; the other weights are zero and `net.fit` holds them there after every step. A neuron's folded state holds only connected pixels that are nonzero in the image, plus the bias, so hidden calls carried 36.8 numbers on average (10th-90th percentile 23-50, max 76). Output neurons see all 64 hidden answers plus the bias, 64.2 numbers per call on average, since Jev's graded p rarely rounds to 0.00.

The neuron is arm E from [the two-question neuron](stage8g-neuron-fixes.md) (yes/no and a neutral choice question averaged in one call). Backprop goes through Jev's measured curve, symmetric, at each call's own term count. The first ten-digit run's recipe otherwise: Adam 0.02, batch 32, τ = 3, init N(0, 0.3) on the connected weights, random tie-breaking. 1,000 training and 500 validation images (the first ten-digit run's seed-0 split), 1,000 test images (its test subset), 6 epochs, seed 0. The swap arm has the same init, mask and batches and an exact step neuron in training.

Settings were chosen with the local check below. Six epochs, not the first ten-digit run's ten, is a budget choice.

## Results (test, 1,000 images)

| | Test on Jev | Same weights, exact neuron | Hidden fire as intended | Output fire as intended | No output fires | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| Trained through Jev | **85.3%** | 75.7% | 90.7% | 97.7% | 54.6% | $15.69 |
| Trained exact (swap) | 65.8% | 72.6% | 91.6% | 98.3% | 81.0% | $2.74 |
| Dense network, 5,000 images: Jev / swap | 84.6% / 52.6% | 70.0% / 83.9% | 90.8% / 93.5% | 94.9% / 98.2% | 59.5% / 97.8% | $84.71 both |
| First ten-digit run, dense, 1,000 images, plain neuron (3 seeds): Jev / swap | 68.5% / 48.0% | 67.2% / 76.8% | 91.1% / 84.9% | 99.5% / 99.8% | | ~$70 all |

Validation on Jev during training (500 images): 78.4% after 2 epochs ($3.67 of training calls), 82.6% after 4 ($7.36), 83.8% after 6 ($11.05). Each validation pass cost $0.93 and the test pass $1.86. The swap arm gets 66.4% on validation. [Pretraining on a fake Jev](stage8c-mock-pretrain.md) trains the same network offline and reaches 83.8% on test with no Jev training at all.

The 3 vs 8 pilot ($0.37) ran the exact-trained sparse network from teaching laya to add (64 weights per hidden neuron, 28 numbers per call) on the arm E neuron: 86.6% on Jev against 91.2% on an exact neuron, a 4.6-point gap. That's the same as the two-question neuron's dense E network on 3 vs 8 (4.6), so on two digits sparsity didn't change the gap. On ten digits, where the E neuron alone didn't help in the dense network, it did.

## Per digit (test)

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Trained through Jev | 96% | 94% | 82% | 84% | 87% | 74% | 90% | 91% | 82% | 73% |
| Swap on Jev | 99% | 70% | 93% | 61% | 62% | 51% | 38% | 70% | 29% | 85% |
| Dense network swap on Jev | 6% | 61% | 29% | 17% | 51% | 92% | 58% | 92% | 72% | 50% |

The dense swap networks lost 0 and 3, the digits with the longest lists. With a fixed 96-pixel field, list length depends much less on the image, and 0 is now the swap arm's best digit. It loses 8 and 6 instead.

## Shorter sums are sharper, but the networks didn't use it the same way

Every hidden test call, binned by margin m = z / spread (misfire = Jev's answer disagrees with the sign of the true sum):

| \|m\| | 0-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | Overall | Median \|m\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sparse, swap arm (37 numbers) | 29.5% | 4.9% | 0.2% | 0.0% | 0.0% | 8.4% | 1.05 |
| Sparse, trained through Jev | 33.7% | 9.1% | 1.2% | 0.0% | 0.0% | 9.3% | 1.14 |
| Dense network, both arms (153 numbers) | 39.3% | 20.1% | 9.7% | 3.6% | 0.8% | 7.9% | 2.11 |

At the same margin, a 37-number call misfires a quarter as often as a 153-number one (0.5-1: 5% against 20%). The overall rates end up the same because the sparse networks' sums sit closer to zero, half the median margin. So sparsity doesn't make hidden neurons fire as intended more often (90.7-91.6% against 90.8-93.5% on the dense network). What it changes is where the misfires are: nearly all within one spread of zero, where a flip costs the next layer little, instead of spread out to two spreads.

In the swap arm, misfires don't depend on term count at all (8.3-8.8% from under 20 to over 50 numbers). In the Jev-trained arm they rise with it, 5.3% under 20 numbers to 10.7% over 50. Dense Jev training pushed sums away from zero to 2+ spreads ([reading the weights](stage6c-weights-probe.md)). Sparse Jev training barely moved the median margin (1.14 against the twin's 1.05).

The output layer is the other difference. The dense network's swap twin had no output firing on 97.8% of test images, so it predicted from the ranking of ten sub-threshold p values, which Jev doesn't preserve. The sparse swap network has an output firing on 19% of images, and the Jev-trained one on 45%. Output neurons here see about 64 numbers instead of 33. Which change, the fields or the doubled hidden layer, moved the output layer wasn't separated.

## Choosing the connectivity (local, $0)

Before spending, `stage8a local` compared random and local receptive fields at K = 64, 96 and 128 connections and H = 32 and 64 hidden neurons (plus K = 64, H = 128), trained 10 epochs on the same 1,000 images. A local field is the K active pixels nearest a random active centre. Each configuration was trained with an exact neuron and with a simulated arm E neuron (SimE, from pretraining on a fake Jev), and costs use a token model fitted to the dense network's journal: 379 + 5.33 tokens per number sent.

| Configuration | Exact neuron, val | Trained on SimE, val on SimE | Hidden numbers per call: mean (p10-p90) | Calls under 10 numbers | $ per 1,000 images |
| --- | --- | --- | --- | --- | --- |
| Dense, H = 32 | 76.8% | 85.4% | 152 (95-206) | | 1.83 |
| Random, K = 64, H = 64 | 64.4% | 87.0% | 25 (15-34) | | 1.68 |
| **Random, K = 96, H = 64** | 69.2% | 85.6% | 37 (23-51) | 0.03% | 1.85 |
| Random, K = 128, H = 64 | 72.2% | 86.4% | 49 (31-66) | | 2.03 |
| Local, K = 96, H = 64 | 68.2% | 86.8% | 41 (21-61) | 2.6% | 1.91 |
| Local, K = 128, H = 64 | 74.8% | 85.2% | 55 (34-77) | | 2.11 |
| Random, K = 96, H = 32 | 63.2% | 83.4% | 37 (22-51) | | 1.01 |
| Random, K = 64, H = 128 | 73.8% | 84.4% | 25 (15-34) | | 3.20 |

Random and local fields were within a point or two of each other on accuracy at every K. Random fields won on the thing this stage is about, the length of each sum: every neuron sees a similar share of the ink, so the spread of term counts is narrow, while a local field near the edge of the digit sends almost nothing and one at the centre sends 60+. K = 96 puts the whole 10th-90th percentile range inside 20-60. H = 64 over 32 gained 6 points on the exact neuron for less than twice the cost. SimE scored every configuration within 83-87%, so it didn't discriminate, and the choice rests on the exact-neuron numbers and term counts.

## Caveats

- One seed per arm. The first ten-digit run's three seeds spread 6.6 points on Jev, so a single number isn't a mean.
- Three things changed against the first ten-digit run at once: sparse fields, the arm E neuron and 64 hidden neurons instead of 32. The run doesn't say which one bought the 17 points over it. A dense 784-64-10 E network at 1,000 images wasn't run. The swap-gap comparison with the dense network is closer (same neuron and backward pass), but it also has twice the hidden neurons and a fifth of the data.
- The dense network used a different split and test subset (seed 7, 2,000 test images). Comparisons with it are across test samples.
- The exact-neuron ceiling is lower when sparse: 72.6% on test against 76.8% for the first run's dense exact networks. The Jev-trained network beats its own weights on an exact neuron by 9.6 points, as the dense network did (14.6).
- Vercel returned bursts of 429s at the paced 3,000 calls a minute (32 calls exhausted their retries and waited out the cool-off). None were billed or lost.
