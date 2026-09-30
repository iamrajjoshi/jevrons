# The dense network: five times the data, two questions per neuron

2026-09-27 · 1.94M calls · $84.71 · Vercel gateway only (TypeSafe direct out of credits)

A 784-32-10 network of Jev neurons trained through Jev gets 84.6% on 2,000 test images. That's up from 68.5% in the first ten-digit run, and it beats the ~77-79% an exact-neuron network reached on 1,000 images when the sum was done in code. Two things changed at once: 5,000 training images instead of 1,000, and arm E from [the two-question neuron](stage8g-neuron-fixes.md) (yes/no and a neutral choice question averaged in one call). One seed, so the two effects aren't separated. The swap control still collapses: the same architecture trained with an exact neuron gets 83.9% on an exact neuron and 52.6% on Jev, a 31.3-point gap (first ten-digit run: 28.8).

Code: `src/jevrons/stage7.py`. Results, weights, curves: `runs/stage7/`. The journal isn't published.

## Setup

The architecture and recipe of the [first ten-digit run](stage6-ten-digits.md) (folded-bias products, zero pixels dropped, straight-through, Adam 0.02, batch 32, init N(0, 0.3), random tie-breaking) with 5,000 stratified training images, 500 validation, 2,000 stratified test images, 8 epochs, seed 0. The neuron is the two-question neuron's arm E; backprop goes through Jev's measured response curve, symmetric. The swap arm has the same init and batches and an exact step neuron in training.

The first run crashed in the jev arm's test pass (about 30 s of Vercel 429s with TypeSafe returning 402, so every backend was down). It resumed from the epoch-8 checkpoint; about $1 of test calls were repeated. The client now waits out rate limits instead of failing (`jev.py`).

## Results

| | Val, epoch 4 | Val, epoch 8 | Test on Jev | Test, same weights on an exact neuron |
| --- | --- | --- | --- | --- |
| Trained through Jev | 84.6% | 85.2% | **84.6%** | 70.0% |
| Trained exact (swap) | | 54.2% | 52.6% | 83.9% |

| Test | Hidden fire as intended | Output fire as intended | No output fires |
| --- | --- | --- | --- |
| Trained through Jev | 90.8% | 94.9% | 59.5% |
| Swap on Jev | 93.5% | 98.2% | 97.8% |

## Per digit (test)

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Trained through Jev | 96% | 97% | 78% | 83% | 85% | 70% | 95% | 85% | 81% | 72% |
| Swap on Jev | 6% | 61% | 29% | 17% | 51% | 92% | 58% | 92% | 72% | 50% |
| First ten-digit run, trained through Jev (3-seed mean) | 82% | 85% | 79% | 66% | 58% | 58% | 80% | 75% | 43% | 59% |

## Reading it

- The Jev-trained weights are now better on Jev than on a perfect neuron (84.6% vs 70.0%). In the first ten-digit run the two were about equal (68.5% vs 67.2%). With more data, training specialised further to the neuron it had.
- The swap arm's hidden neurons are more faithful than the Jev-trained arm's (93.5% vs 90.8% fire as intended), and it still loses 31 points. On 3 vs 8 the E neuron cut the gap from 25 to 4.6. On ten digits it didn't help, so hidden misfires aren't what sinks the exact network here.
- The output layer is. The swap network on Jev has no output firing on 97.8% of test images, so every prediction is a ranking of ten below-threshold p values. An exact network separates classes by small differences in output sums, and Jev's graded "no" doesn't preserve their order. The Jev-trained network learned to make an output actually fire on 40% of images, and to rank the rest with gaps Jev keeps.
- 0 and 3, the digits with the longest pixel lists after 8, collapse in the swap arm again (6%, 17%), as in the first ten-digit run. Trained through Jev, 8 gained the most over that run (43% → 81%), then 4 (+27) and 3 (+17). 5 and 9, as weak in the first run, gained only 12-13 and are now the worst digits.

One seed. The val-test agreement (85.2 vs 84.6) says the test number isn't a fluke of the split, not that another seed would match.
