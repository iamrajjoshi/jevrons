# The single-neuron probe: how well one Jev neuron adds

2026-09-26 · 16,299 calls · 11.5M input tokens · $0.48 · model `jev-1.13.0`

A Jev neuron is a real but imperfect adder. With the bias out of the way, it gets the sign of a weighted sum right 82-93% of the time up to 75 terms, falling to 65-82% at 150-250 terms. That's the range where the local mock showed training can compensate, so the gate to the logic-gates experiment passes. Three findings overturned the audited proposal: Jev isn't deterministic, packing neurons into one request changes their answers, and the neuron over-trusts a standalone bias.

Code: `src/jevrons/probe.py`, `src/jevrons/analyze_probe.py`. Raw journals: `runs/probe/*.jsonl.gz` (every state sent and answer received).

## The full-state neuron sums, roughly

Bias fixed at 0, weights N(0, 1), inputs drawn from nonzero MNIST pixel values, 60 neurons per cell. Accuracy is Jev's yes/no (p >= 0.5) against the true sign of the total.

| Format | 10 terms | 30 | 75 | 150 | 250 |
| --- | --- | --- | --- | --- | --- |
| paired `{x, w}` records | 82% | 87% | 90% | 80% | 77% |
| parallel `x`, `w` arrays | 82% | 85% | 82% | 68% | 65% |
| precomputed products | 93% | 88% | 92% | 82% | 78% |

Errors concentrate near zero. When the true total is more than one term-spread from zero, accuracy is 100% up to 75 terms in every format and 88-100% beyond. The products format is the most accurate and the cheapest (378 tokens at 10 terms, 1,673 at 250, versus 544 and 5,395 for paired).

## It over-trusts the bias

The first probe set the bias so the total landed at a chosen margin, which made the bias a near copy of the answer. In the 300 neurons where the bias sign disagreed with the true total, Jev followed the bias 64% of the time. With the bias drawn independently of the terms, Jev still followed the bias sign 72-97% of the time.

Folding the bias into the products list as one more number fixes most of it (40 neurons per cell, independent bias, same wording):

| Format | 10 terms | 30 | 75 | 150 |
| --- | --- | --- | --- | --- |
| products + separate bias | 90% | 85% | 83% | 75% |
| products with bias folded in | 95% | 85% | 85% | 90% |

The folded format is the new default candidate; the [logic-gates experiment](stage2-logic.md) compares both.

## It isn't deterministic

Only 1% of full-state neurons returned the same p across 5 identical calls (mean range 0.04, max 0.15). Scalar calls repeat more often (46% identical, max range 0.09). The audit's plan to tabulate Variant A once and replay it is out: answers must come from live calls, and caching changes the experiment. It also means every Jev neuron is a little stochastic, which fits the Bernoulli-neuron framing.

## Packing changes answers

Eight paired neurons under separate keys in one request, versus the same neurons sent alone:

| Terms | Answers flipped | Accuracy packed | Accuracy alone | Tokens per request |
| --- | --- | --- | --- | --- |
| 10 | 14% | 80% | 85% | 2,658 |
| 30 | 16% | 85% | 86% | 5,894 |
| 75 | 18% | 80% | 88% | 13,174 |
| 150 | 11% | 79% | 86% | 25,314 |

Eight 250-term neurons exceed the 32k state limit (`max_tokens_exceeded`). Neurons go one per request. At the 1,200 requests/min limit, that makes the ten-digit experiment about 36 hours of calls, unless its design changes.

## Scalar neuron (Variant A): wording decides everything

2,181 values of z from -100 to 100, 5 repeats each. With the first wording ("z is a number." / "z is greater than zero." / "z is zero or less."), Jev answered p = 0.66-0.74 for every negative z and 0.97-0.98 for every positive one: a clean step, but with the "no" side above 0.5, so thresholded accuracy on negatives is 0%. Rewording to "Is the number z positive?" gives p = 0.01-0.05 for z <= 0 and 0.99 for z >= 0.1, 100% correct over ±10. Keyed batches of 50 matched that.

So Variant A is a near-perfect step, as the audit predicted, but only with the right wording. The same rewording didn't help the full-state neuron (within noise at 40 neurons per cell), so full-state keeps the first wording.

## Tokens and latency

Median latency is 0.13-0.16 s per call regardless of state size. Scalar calls bill about 307 tokens, which is the fixed overhead per call. The per-image estimates in the proposal were too low for paired records (about 20 tokens per term, not 9) and about right for products (about 5 per term).
