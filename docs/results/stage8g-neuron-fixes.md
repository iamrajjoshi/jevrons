# Stage 8g: making a Jevron behave like the neuron we meant

2026-09-27 · 561,000 calls · $28.99 · TypeSafe direct (27%) and Vercel gateway (73%)

Asking each neuron two questions at once, the usual yes/no and a neutral two-option choice, and averaging them nearly closes the swap gap. A 3 vs 8 network trained with an exact neuron keeps 88.2% when run on that Jev neuron, against 67.8% on the plain yes/no neuron (the exact-neuron score is 92.8%, so the gap falls from 25.0 to 4.6 points). The two formats lean in opposite directions (yes/no toward "fire", the choice toward "below"), and the average cancels most of both. Networks trained through the new neuron are as good as before or slightly better (95.2% vs 94.2%).

Code: `src/jevrons/stage8g.py`. Results, weights, curves: `runs/stage8g/`. The format comparison that motivated arm E: `runs/choice-jev/`.

## Setup

Stage 5's 3 vs 8 exactly (784-32-1, 500 train and 500 validation images, same seed and initialization, 5 epochs), so stage 5 is the baseline arm. Jev's measured response curve per term count, P(fire) = Φ(k(n)·(z/spread − m0(n))) from `runs/margin`, drives the fixes:

| Arm | Backward pass | Neuron |
| --- | --- | --- |
| Baseline (stage 5) | sigmoid(z/τ) slope, τ = 3 | yes/no, folded products |
| B | slope of the measured curve for that neuron's term count | same |
| C | measured curve, symmetric | bias sent to Jev shifted by m0 × spread, to cancel the threshold drift |
| D | as C | as C, plus three yes/no phrasings averaged in one call |
| E | measured curve, symmetric | yes/no and a neutral choice ("above zero" / "below zero or equal") averaged in one call, no shift |

The swap control, stage 5's exact-trained 3 vs 8 weights, is re-run on the C, D and E neurons, since those change the neuron itself.

## Results (validation, 500 images)

| Arm | Trained through this neuron | Exact-trained network on this neuron | Swap gap | Hidden fires as intended (trained) | Cost |
| --- | --- | --- | --- | --- | --- |
| Baseline | 94.2% | 67.8% | 25.0 pts | 93.0% | (stage 5) |
| B | 94.5% | 67.8% | 25.0 pts | 95.3% | $5.78 |
| C | 95.2% | 75.2% | 17.6 pts | 89.0% | $7.40 |
| D | 95.3% | 74.9% | 17.9 pts | 89.0% | $7.96 |
| E | 95.2% | **88.2%** | **4.6 pts** | **96.0%** | $7.85 |

Swap gap = 92.8% (the exact-trained network on an exact neuron) minus its accuracy on the arm's neuron.

## Reading it

- Backprop through the measured curve (B) doesn't move accuracy, but trained hidden neurons misfire less (95.3% vs 93.0% fire as intended).
- Pre-compensation (C) closes about 30% of the swap gap. Its hidden neurons match the true sign less often (89%), which suggests the drift measured on probe neurons over-corrects some trained ones.
- Three yes/no phrasings (D) add nothing over C: they share the same lean, so averaging doesn't cancel it.
- Averaging two formats with opposite leans (E) does, without any shift, and gives the most faithful neuron: 96.0% of hidden neurons fire as intended in the trained network and 91.4% in the swapped one (baseline swapped: 86.9%).

One seed per arm; the differences between B, C, D and E trained accuracies (94.5-95.3%) are within noise. The swap-gap differences are large enough to matter.
