# Stage 9c: a neuron that looks broken, trained through anyway

2026-09-26 · ~150,000 local calls to laya (Ollaya, laya:en) · $0

Laya fails the obvious test: asked "is z greater than zero?" it gets the sign right 52% of the time. Used as the activation of 3 vs 8's hidden layer anyway, with the sum computed in code, a network trained through it reaches 96.0% on live laya. The same architecture trained with an exact neuron and then run on laya is a coin flip (50.0%). A useful neuron doesn't need to answer the question correctly, only to respond consistently to its input, and training learns the response.

Code: `src/jevrons/stage9c.py`. Results, weights, curves: `runs/stage9c/`.

## Laya's curve

Laya is deterministic, so its measured response is its activation function. Across 2,181 values of z (runs/probe-laya/scalar_curve.json) it is jagged and roughly V-shaped: about 0.19 at z = 0, 0.32 at z = ±1, 0.75 at z = 0.5, and settling near 0.72 for z below −2 and 0.92 above +2. A finer check between 1 and 5 in steps of 0.01 found it even more jagged than the 0.1-step table suggested: neighbouring values differ by 0.027 on average (the table implied 0.004), and live answers differ from the table by up to 0.48.

## Setup

Stage 5's 3 vs 8 (784-32-1, 500 train and 500 validation images, same seed and initialization), scalar Variant A: code computes z = w·x + b, laya answers `states.SCALAR_Q` on {"z": z} at two decimals. Backprop goes through a smoothed copy of the measured curve (Gaussian, σ = 0.5 in units of z) instead of a guessed sigmoid. Two arms:

- laya hidden: all 32 hidden neurons are laya; the single output neuron is an ordinary sigmoid in code.
- all-laya: the output neuron is laya too, calibrated in code to [0, 1] by its two measured levels, (p − 0.718) / (0.919 − 0.718), because raw laya p only drops below 0.5 in narrow pockets near zero.

The swap control is stage 5's exact-trained 3 vs 8 network, run through the same laya neurons.

## Results

| Arm | Trained through laya, on laya | Same weights, exact neuron | Swap (exact-trained), on laya | Swap, exact neuron |
| --- | --- | --- | --- | --- |
| laya hidden | 96.0% | 95.6% | 50.0% | 92.8% |
| all-laya | 72.6% | 88.0% | 50.0% | 92.8% |

For comparison, Jev on the same task: 94.2% trained, 67.8% swapped (stage 5, full-state neurons where Jev also does the sum, so not a like-for-like comparison).

Locally, against the 0.1-step table, the all-laya arm reached 92.2% and scored 89.6% on the table after live training; live it scored 72.6%. The difference is the fine-scale jitter above: 32 hidden neurons average it out, a single output neuron can't. The laya-hidden arm agrees live vs tabulated (96.0% vs 96.2%).

## What it shows

The swap control is the result. Weights trained for an exact step are worthless on laya (50.0% in both arms, the output collapses to one class), while weights trained through laya's real curve work, 96.0% with laya in every hidden neuron. Laya was later found to answer the same question correctly with a neutral choice format (100% sign accuracy on z in [−10, 10], see stage 10), so this experiment is best read as a stress test: training through a deliberately bad-looking activation.
