# Stage 4: Jev as a scalar activation on ten digits

2026-09-26 · 84,000 calls (+13,437 in a crashed first attempt) · $1.10 · TypeSafe direct and Vercel gateway

With the sum computed in code, Jev is an almost perfect step function. A 784-32-10 network trained locally with an exact step neuron gets 78.5% on 1,000 test images; the same weights with every neuron replaced by a live Jev call get 79.1%. Jev fired exactly as intended on 99.96% of hidden and 99.92% of output evaluations. Gate passed. Variant A is the control, not the experiment.

Code: `src/jevrons/stage4.py`. Weights and summary: `runs/stage4/`.

| Split | Exact step | Jev activation | Hidden fires as intended | Output fires as intended | No output fires (Jev) |
| --- | --- | --- | --- | --- | --- |
| Validation (1,000, from the training partition) | 77.5% | 77.0% | 99.95% | 99.89% | 14.1% |
| Test (1,000, stratified subset of the official test set) | 78.5% | 79.1% | 99.96% | 99.92% | 11.0% |

The question was "Is the number z positive?" with z at two decimals; the stage 1 wording would have answered yes to every negative z.

## Correction to stage 0

The stage 0 spike reported ~83-84% for a hard-step 784-32-10 net on 1,000 images. That figure broke ties between output neurons using the exact output sums. With near-binary outputs, 11-22% of images have no output neuron firing at all, so the tie rule matters: with random tie-breaking, the honest ceiling is about 77-79%. All results from stage 4 on use random tie-breaking (`net.decide`). Graded outputs, such as full-state Jev neurons produce, should be less affected.

## Infrastructure notes

This was the first stage on both backends. The first attempt paced the gateway at 5,000/min: about 20% of gateway calls needed retries, p99 latency reached 16 s, and one call exhausted its retries and crashed the run. The client now cools off a failing backend for 60 s and sends its calls to the other one, logs failures to `*-failures.jsonl`, and paces the gateway at 3,000/min. The rerun sustained about 4,100 calls/min with about 1% retries; 73% of calls went through the gateway.
