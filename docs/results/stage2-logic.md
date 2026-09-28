# Stage 2: logic gates from Jev neurons

2026-09-26 · 33,183 calls · $0.46 · model `jev-1.13.0`

A two-layer network of Jev neurons learns XOR, on 3 of 3 seeds, but only when the bias travels inside the products list. With the bias as a separate field, the same training fails on every seed. At an equal call budget, SPSA (which used the separate-bias format) lost to straight-through with the same format on AND, tied on OR and beat it on XOR (3/15 fully correct passes vs 0/15). Folded-bias straight-through beat both.

Code: `src/jevrons/stage2.py`, `src/jevrons/net.py`. Per-run results, weights and loss curves: `runs/stage2/`. Journal: `runs/stage2/journal.jsonl.gz`.

## Setup

Inputs are the four rows of a truth table. AND and OR use one Jev neuron; XOR uses 2 inputs, 4 hidden Jev neurons and 1 output Jev neuron (20 calls per forward pass). Hidden neurons pass Jev's probability p to the next layer. Zero inputs are dropped from the state. Loss is BCE on the output p.

Three arms get the same Jev-call budget: straight-through with products plus a separate bias, straight-through with the bias folded into the products, and SPSA (half the steps, two passes per step). Evaluation runs the trained net 5 more times through live Jev, since Jev isn't deterministic. "Pass accuracy" is the mean fraction of the 4 rows correct per pass.

## Results

| Task | Arm | Seed 0 | Seed 1 | Seed 2 | Fully correct passes |
| --- | --- | --- | --- | --- | --- |
| AND | straight-through, separate bias | 100% | 100% | 100% | 15/15 |
| AND | straight-through, folded bias | 100% | 95% | 95% | 13/15 |
| AND | SPSA | 75% | 100% | 75% | 5/15 |
| OR | straight-through, separate bias | 100% | 100% | 100% | 15/15 |
| OR | straight-through, folded bias | 100% | 100% | 100% | 15/15 |
| OR | SPSA | 100% | 100% | 100% | 15/15 |
| XOR | straight-through, separate bias | 75% | 50% | 70% | 0/15 |
| XOR | straight-through, folded bias | 95% | 100% | 100% | 14/15 |
| XOR | SPSA | 75% | 65% | 80% | 3/15 |

Cells are pass accuracy over 5 live evaluations. Gate for stage 3: XOR correct on 3 of 3 seeds. The folded arm passes, with one of fifteen evaluation passes missing a row.

## Jev's quirks show up in the learned weights

Every run also checks whether its learned weights solve the task under an exact step neuron. Two XOR runs split the two ways that matter for the blog and demo:

- Seed 0, folded bias: correct on live Jev (95%, 4 of 5 passes perfect), but the same weights fail XOR under an exact step neuron. The network learned a solution that works with how Jev actually behaves, not the ideal arithmetic.
- Seed 2, separate bias: the weights solve XOR under an exact step neuron, but Jev running them gets 70%. A correct design, broken by the neuron's handling of the bias.

These are single runs, not yet a measured effect. Stage 6's swap control is where the claim gets tested at scale.

## What changes going forward

The folded-bias products format is the default neuron from here on. SPSA stays in the comparison for stage 3, which is small enough, and is dropped for digits as planned. Stage 2 took about 35 minutes of wall-clock because each training step waits for the previous layer; later stages should run several seeds concurrently to use the rate limit.
