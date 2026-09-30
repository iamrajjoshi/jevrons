# Jevrons: proposal (audited)

2026-09-26 · Raj Joshi

Snapshot of the audited proposal. The live copy is the Claude doc "Jev Neuron Classifier Proposal (Audited)". The pre-audit text is in [original-proposal.md](original-proposal.md).

The experiment is worth running, but the original plan spends most of its budget on the least informative variant. The scalar-activation network (Variant A) can be characterized for under $1 and then simulated exactly; the real research question lives in Variant B, where Jev does the weighted sum itself.

## Audit verdict

Every parameter count, evaluation count and cost cell in the original checks out arithmetically. The problems are in what the numbers are spent on. Six changes, ranked by how much they move the result:

| # | Finding | Change |
| --- | --- | --- |
| 1 | Variant A is a lookup table. Jev sees one number at fixed precision, and TypeSafe's own repeat test found most answers identical across 5 runs (std 0.0). A grid of ~2,000 z values x 5 repeats characterizes it completely for about $0.17. The planned 420,000 live evaluations per run then add no information. | Tabulate Variant A once, train against the table, and verify live only on the test set. |
| 2 | Variant A will almost certainly look like a perfect step function. In a trained 784-32-10 net, only 0.1% of hidden pre-activations fall within 0.05 of zero; the 1st-99th percentile range is -87 to +97. Judging the sign of "47.3" is trivial. | Treat Variant A as a smoke test. Move the research budget to Variant B. |
| 3 | The flagship demo comparison (fails after swap, succeeds after training) won't appear in Variant A. Locally, swapping a trained sigmoid net to a hard step costs under 1 point of accuracy. It does appear once the neuron is imperfect: a mock noisy full-state neuron cost 10 points on swap and training recovered most of it. | Make that comparison in Variant B. |
| 4 | SPSA over 25,450 parameters with ~300 updates will not train MNIST. Gradient-estimate variance scales with dimension; the nine-run pilot spends a third of its budget on an arm that is expected to fail. | Keep SPSA for XOR and the circle task only. Replace it on digits with a straight-through rule on the intended arithmetic. |
| 5 | Variant B at 784 dense inputs is ~7,400 tokens per hidden neuron, five times the table's "verbose" T = 1,500 column. MNIST averages 150 nonzero pixels (95th percentile 219), and zero pixels contribute nothing to the sum. | Send only nonzero pixel/weight pairs. It is mathematically identical and cuts tokens ~4x. |
| 6 | Runtime is unaddressed and the request limit binds first. At 1,200 requests/min, one request per neuron is 42 requests per image, so 10 epochs on 1,000 images takes ~5.8 hours per run. | Measure layer-batched requests (2 per image) against isolated ones in stage 1; budget wall-clock alongside dollars. |

Smaller corrections: with 1,000 training images even a conventional net tops out near 84% at width 32, so the accuracy target must be set against that ceiling, not the ~98% TensorFlow reference. The NAND bill implies ~370 billed tokens per gate, which suggests a fixed per-call overhead; T = 100 is optimistic for unbatched scalar calls. Jev's jaggedness page notes that P(yes) and 1 - P(no) for a reworded question aren't interchangeable, so the neuron question wording must be frozen, not just the number format.

## Local evidence

Width 32 is enough for a demo, and a hard-threshold neuron costs almost nothing. These runs used NumPy on the proposal's exact setup: 1,000 stratified training images, 1,000 held-out validation images from the training partition, ten sigmoid outputs, mean BCE, Adam at 0.01, batch 32, 3 seeds. The hard-threshold rows use a sigmoid-slope straight-through backward pass. Reproduce with `spikes/baseline.py`; raw output is in `spikes/results/`.

| Setup | Val accuracy, 10 epochs | Val accuracy, 50 epochs | Swapped to step (10 / 50 epochs) |
| --- | --- | --- | --- |
| 784-32-10, sigmoid hidden | 83.7% +/- 1.1 | 84.3% +/- 0.3 | 82.9% / 83.1% |
| 784-32-10, hard step hidden and output | 83.4% +/- 0.3 | 83.6% +/- 0.4 | - |
| 784-64-10, sigmoid hidden | 85.7% +/- 0.3 | 85.7% +/- 0.1 | 85.7% / 85.3% |
| 784-128-10, sigmoid hidden | 86.5% +/- 0.3 | 86.0% +/- 0.3 | 86.6% / 86.3% |
| 784-32-10, sigmoid, 50,000 images | 95.2% (test 95.3%) | - | - |
| 784-32-10, hard step, 50,000 images | 94.2% (test 94.5%) | - | - |

Ten epochs already converges on 1,000 images, so the proposal's epoch count is right. Doubling width buys ~2 points for twice the Jev calls; 32 is the right first width. Data volume, not width, is the ceiling.

### A mock imperfect neuron shows the adaptation effect

To preview Variant B, each hidden neuron was replaced with one that fires on z + sigma x ||w * x|| x noise, so its error grows with the size of the terms it has to add. Training ran forward through the noisy neuron and backward through the intended sum (784-32-10, 1,000 images, 50 epochs, 3 seeds, 5 noisy evaluations averaged). Reproduce with `spikes/mock_b.py`.

| Neuron noise sigma | Neuron agrees with true sign | Swap: trained clean, run noisy | Trained through noisy neuron | Recovered |
| --- | --- | --- | --- | --- |
| 0.5 | 96% | 81.8% | 82.6% | +0.8 pts |
| 1.0 | 91% | 77.0% | 79.6% | +2.6 pts |
| 2.0 | 80-85% | 60.7% | 70.6% | +9.9 pts |

The network did not make the neuron more accurate (agreement slightly fell). It learned weights whose downstream use tolerates the errors. That's the mechanism the demo should show, and it needs a neuron that is wrong 5-20% of the time. Variant A won't be; a full-state Jev neuron summing 150 terms plausibly will. This mock is Gaussian and independent; Jev's errors will be systematic, which could help (learnable) or hurt (correlated).

## Revised neuron design

Keep both variants, but give them different jobs. Variant A validates the plumbing and the backward rule for under $10. Variant B is the experiment.

### Variant A: calibrate once, then simulate

The question stays "Is z greater than zero?" with z at one fixed format (two decimals). Sweep z from -100 to +100: step 0.01 inside [-1, 1], step 0.1 elsewhere, about 2,200 points, 5 repeats each. If repeats agree, that table is the neuron. Train against it locally, then run the chosen checkpoint live on the 10,000 test images and report agreement between live and tabulated activations. If repeats disagree, the table stores the per-z distribution and training samples from it.

Batch hidden neurons per image by naming them (`{"n00": 47.31, "n01": -3.02, ...}` with one question per key). Don't use positional arrays: the jaggedness page says Jev doesn't count list items reliably.

### Variant B: Jev does the arithmetic

Each hidden neuron's state is its bias plus the nonzero pixels as paired records, keyed by pixel index:

```json
{"bias": 0.15, "terms": [{"pixel": 152, "x": 0.53, "w": -1.27}, {"pixel": 153, "x": 0.99, "w": 0.40}]}
```

The question: "Is the sum of x times w over all terms, plus bias, greater than zero?" Output neurons get the same shape with 32 hidden activations as x. Pixel index is carried for the demo and diagnostics; drop it in an ablation if it costs too many tokens.

Two constraints come with this. Weights need a bounded, fixed format (clip to [-9.99, 9.99], two decimals) or training will grow them and the state will grow with them; the unregularized local runs reached |z| around 90. Add weight decay and record the transmitted-precision histogram. And keep the precomputed-contributions variant (send each w x x product, Jev only sums) as the middle rung, so a failure can be pinned on multiplication or on addition.

If 150 terms turns out to be hopeless, the fallback is a fixed 32-dimensional PCA projection in code before the hidden layer. That shortens every state to 32 terms. It changes the input, so the scalar control must use the same projection.

## Learning rules

The default for both variants is the same rule: forward through real Jev, backward as if the neuron computed the intended sum. This is the straight-through estimator used in quantization-aware training, and it's the rule the mock above used.

```text
z_intended = dot(w, x) + b          # computed in code, never sent in Variant B
a = Jev(state)                       # real forward value
dL/dw_i = dL/da * sigmoid'(z_intended / tau) / tau * x_i
```

It needs no extra calls and no model of Jev's full-state function, and it backpropagates to earlier layers through the intended weights. `tau` sets how wide the surrogate slope is; tune it on validation data in the local hard-step control first.

The proposal's other rules still have a place, just a narrower one:

| Rule | Where to use it | Why not everywhere |
| --- | --- | --- |
| Fitted surrogate of Variant A's measured curve | Variant A | The tabulated curve is exact, so this is free. |
| Central differences, 3 calls per neuron | Nowhere by default | A near-step curve gives zero slope almost everywhere; only try it if the sweep shows a real transition band. |
| Fitted full-state surrogate | Variant B, second arm | Worth trying only if straight-through stalls; it must predict slopes for 150+ inputs. |
| SPSA | XOR, circle, and 32-dimensional PCA Variant B | Its update noise grows with parameter count; 25,450 parameters in ~300 updates won't converge. |

Whatever the rule, log the fraction of neurons whose Jev output disagrees with sign(z_intended) at every epoch. If straight-through works, this number explains why; if it fails, it says whether the neuron or the rule is to blame.

## Staged plan

Each stage has a gate. Don't start the next one until it passes.

| Stage | Work | Gate to advance | Est. cost |
| --- | --- | --- | --- |
| 0. Local | Done above. Freeze splits, seeds, preprocessing, and the hard-step control. | 784-32-10 hard step reaches ~83% on validation | $0 |
| 1. Probe | Variant A sweep (2,200 z x 5). Variant B probe: 500 random neuron states at 10, 30, 75, 150 and 250 terms, 5 repeats, each format (paired, parallel arrays, precomputed products). Measure tokens, latency, and isolated vs layer-batched requests. | Know repeat variance, sign accuracy vs term count, and real T | ~$5 |
| 2. Logic | AND, OR, XOR with Variant B; SPSA vs straight-through at equal call budgets | XOR truth table correct on 3 of 3 seeds | ~$2 |
| 3. Circle | 2-D inside/outside, 500 points, plot boundary | Beats a linear classifier on held-out points | ~$5 |
| 4. Variant A digits | Train on the tabulated neuron; verify live on 10,000 test images | Live/tabulated agreement above 99%, accuracy within 1 pt of local | ~$7 |
| 5. Two digits | Variant B, 0 vs 1, then 3 vs 8; 500 train / 500 val, 1 seed each | Held-out accuracy above 95% on 0 vs 1 | ~$50 |
| 6. Ten digits | Variant B, 1,000 train / 1,000 val, 3 seeds, straight-through, plus swap control | Trained beats swap on validation across seeds | ~$160 |
| 7. Scale | 10,000 training images, best config, 1 seed; final test once | Validation improves over stage 6 | ~$310 |

If stage 1 shows Jev's sign accuracy on 150-term sums is near 50%, stop and switch Variant B to the 32-dimensional PCA input before stage 5. If it's near 100%, Variant B collapses to Variant A and the headline result is a clean negative, which is worth writing up too.

## Cost and runtime

The full plan through stage 7 is about $540, or roughly $650 with a 20% retry and re-run reserve. Wall-clock matters more than dollars here: stage 7 alone is 8 hours at the rate limits, and 3 days if every neuron is its own request.

### Tokens per image pass

These are estimates at ~9 tokens per paired record and ~370 tokens of fixed overhead per call (the overhead the NAND bill implies). Stage 1 replaces them with measured values.

| Design | Tokens per hidden neuron | Tokens per image (42 neurons) | $ per image pass |
| --- | --- | --- | --- |
| Variant A, keyed batch per layer | ~35 | ~1,600 | $0.00007 |
| Variant B, 32-dim PCA input | ~660 | ~27,000 | $0.0011 |
| Variant B, sparse nonzero pixels (plan default) | ~1,700 | ~61,000 | $0.0026 |
| Variant B, dense 784 pixels | ~7,400 | ~243,000 | $0.0102 |

### Rate limits set the schedule

Jev 1.13.0 allows 250,000 tokens/s and 1,200 requests/min, with 32,000 tokens for state plus the longest question. Sparse Variant B at one request per neuron is 42 requests per image, which caps throughput at 0.48 images/s. Packing 8 hidden neurons per request under separate keys (~14,000 tokens each) gives 5 requests per image and ~4 images/s, where the token and request limits roughly meet.

| Stage | Image passes | One request per neuron | 8 neurons per request |
| --- | --- | --- | --- |
| 6. Ten digits, 3 seeds | ~63,000 | ~36 h | ~4.4 h |
| 7. Scale, 10,000 images | ~120,000 | ~69 h | ~8.3 h |

Packing is only safe if neighbouring neuron states don't change each other's answers. Stage 1 tests that directly: the same neuron state alone vs packed with 7 others, 5 repeats each.

### Budget caps

Set a hard cap per stage at 2x its estimate, enforced in code from the `usage` field, and a $700 total cap. The optional dense-784 ablation (one seed of stage 6) adds ~$200. The original $25 feasibility cap still fits stages 1-3 with room to spare.

## Demo and success criteria

The claim to show is "training through Jev recovers accuracy that swapping Jev in loses." It holds if, on stage 6, the trained Variant B net beats the swap control on validation for all 3 seeds and the gap survives on the untouched test set. The local mock suggests a 3-10 point gap is plausible if Jev's full-state sign accuracy lands between 80% and 95%.

Absolute targets, set against the 1,000-image ceiling of ~84%: 75% on validation is a working classifier; above 80% is a strong result. Anything under 20% is a failure to learn, not a weak classifier.

The draw-a-digit demo costs ~$0.003 and 1-3 seconds per live inference in sparse Variant B, so it can run live. Show per neuron: the intended z, Jev's answer, and whether they disagree. The disagreement highlights are the point of the demo; without them it looks like any small MLP.

## What stays from the original

Most of the original holds and should carry over unchanged: the checkpoint and manifest layout, exact-match caching rules, ten independent sigmoid outputs with mean BCE and argmax, the stratified 1,000/1,000 split with the official test set held back, the four-run comparison table, the fallback table for saturated or noisy neurons, and the separation of semantic neurons and prompt optimization into later work. The soft/signed/hard/sampled activation comparison is still worth running, but only in Variant B; in Variant A the outputs are already 0 or 1.

### Open questions

- [ ] Does Jev return a Noul deterministically for identical input, or is there sampling noise? The docs don't say; stage 1 answers it.
- [ ] Is there a per-account request limit above 1,200/min available on request? It would cut stage 7 from 8 hours to about 2.
- [ ] Does packing 8 neuron states in one request change individual answers?
- [ ] Should the ALU's 84% compound figure be re-measured end to end before citing it next to these results?

## Sources

TypeSafe pages opened for this audit: [Models](https://docs.typesafe.ai/models) (price, 64k context, 32k state limit, 250k tokens/s, 1,200 requests/min), [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) (counting, numeric formats, question-wording invariants), [API reference](https://docs.typesafe.ai/api) (no gradient or logit output, 429/529 backoff), [Noul](https://docs.typesafe.ai/primitives/noul) (determinism not documented), and [Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions) (5-repeat variance mostly 0.0, batching adds no variance).

Method references carried over from the original: [Bengio et al. 2013](https://arxiv.org/abs/1308.3432) on straight-through estimators, [Spall 1992](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf) on SPSA, [Courbariaux et al. 2016](https://arxiv.org/abs/1602.02830) on binarized networks, and [the MNIST database](http://yann.lecun.com/exdb/mnist/). Local runs used the MNIST files from Google's CVDF mirror (`scripts/fetch_mnist.sh`, checksums pinned).
