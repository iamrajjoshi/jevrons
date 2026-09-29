# Ten digits: all ten with full-state Jev neurons

2026-09-26 to 27 · ~1.7M calls · ~$70 · TypeSafe direct and Vercel gateway (one Vercel credit top-up mid-run; seed 2 resumed from its epoch-2 checkpoint)

On all ten digits, a 784-32-10 network of full-state Jev neurons trained through Jev gets 68.5% on 1,000 test images (three seeds: 71.0%, 70.2%, 64.4%). The same architecture from the same starts, trained with an exact neuron and then run on Jev, gets 48.0% (43.7%, 52.9%, 47.5%), although it's the better network on paper: 76.8% on an exact neuron. Training through Jev wins by 17-27 points on every seed.

Code: `src/jevrons/stage6.py`. Per-seed results, weights and curves: `runs/stage6/`. Journals are gitignored and backed up outside git (`~/jevrons-journal-backup/`).

## Setup

784-32-10, folded-bias products format, zero pixels dropped (about 150 terms per hidden neuron, 33 per output neuron). 1,000 stratified training images and 500 validation images from the training partition; 1,000 stratified test images from the official test set. Straight-through with τ = 3, Adam at 0.02, batch 32, 10 epochs, weights initialized N(0, 0.3), random tie-breaking between outputs. The swap arm uses the same initialization and batches with an exact step neuron in training. Checkpoints after every epoch.

## Results (test, 1,000 images)

| Seed | Trained through Jev, on Jev | Same weights, exact neuron | Trained exact, on Jev | Trained exact, exact neuron |
| --- | --- | --- | --- | --- |
| 0 | 71.0% | 61.5% | 43.7% | 78.7% |
| 1 | 70.2% | 71.4% | 52.9% | 78.1% |
| 2 | 64.4% | 68.7% | 47.5% | 73.6% |
| Mean ± sd | 68.5% ± 2.9 | 67.2% ± 4.2 | 48.0% ± 3.8 | 76.8% ± 2.3 |

Validation after 5 and 10 epochs, trained through Jev: 69.0 / 70.0%, 68.4 / 70.4%, 68.6 / 66.2%.

Hidden neurons fired as intended 91.1% of the time in Jev-trained networks and 84.9% in exact-trained ones run on Jev (output neurons: 99.5% and 99.8%).

## Per digit (mean over seeds)

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Trained through Jev | 82% | 85% | 79% | 66% | 58% | 58% | 80% | 75% | 43% | 59% |
| Trained exact, on Jev | 7% | 82% | 47% | 17% | 34% | 67% | 72% | 82% | 5% | 67% |

The swapped networks don't degrade evenly: they nearly stop predicting 0, 3 and 8, the digits with the most ink and therefore the longest sums, where Jev's yes-lean is strongest.

## Notes

In 97-99% of test images no output neuron's p reaches 0.5; the prediction comes from the ranking of graded outputs below 0.5. That's why graded Jev outputs matter at ten classes and why the tie-breaking correction to the local capacity check mattered. The ceiling for an exact-neuron 784-32-10 at 1,000 images with random tie-breaking is about 77-79% (from [the scalar-neuron run](stage4-scalar-digits.md), where the sum was done in code), which the exact arm reaches; the Jev-trained arm is about 8 points below it. [The dense network](stage7-more-data.md) tests whether more data narrows that.
