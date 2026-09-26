# Plan

Resume point for the project. The experiment design and staged gates are in [proposal.md](proposal.md); this file records owner decisions and where the work stands.

## Decisions (2026-09-26, Raj)

- Deliverables: a live draw-a-digit demo, a short screen-recorded video of it for Twitter, and a long-form post for Raj's engineering blog. Blog figures and the video need saved artifacts from every stage, so every paid call is journaled.
- Budget: no cap for now. Spend is still tracked per call and per stage; a stage can set `max_usd` if needed.
- Model: `jev-1.13.0`, pinned. A model change starts a new baseline.
- Credits: use the existing Jev credits; no top-ups and no rate-limit increase request.
- Backends: calls are split between TypeSafe direct (1,100/min) and Vercel AI Gateway (3,000/min, key `VERCEL_API_KEY`). The gateway is pinned to the `typesafe-ai` upstream and only exposes the unversioned `typesafe-ai/jev` alias; `backend_check` measured it as the same neuron (88.0% vs 87.3% sign accuracy, per-state difference 0.018 below repeat noise 0.022). Every journal line records `backend` and `served`.

## Design changes from measured results

- Jev answers vary between identical calls, so there's no tabulation or caching; every forward pass is live.
- Neurons go one per request. Packing 8 into a request flipped 11-18% of answers.
- The neuron state is the folded-bias products list: `{"products": [x1*w1, ..., xk*wk, bias]}` with "Add all the numbers in products together." Separate-bias formats over-weight the bias and fail XOR.
- Variant A (scalar) needs the question "Is the number z positive?"; the first wording answered yes for every z.
- Spend so far: $17.40 over ~532,000 calls (stages 1-5 plus backend check).

## Stages

| Stage | Status | Output |
| --- | --- | --- |
| 0. Local capacity | Done | `spikes/results/` |
| 1. Probe | Done, gate passed | [stage1-probe.md](results/stage1-probe.md) |
| 2. Logic (AND/OR/XOR) | Done, gate passed | [stage2-logic.md](results/stage2-logic.md) |
| 3. Circle | Done, gate passed | [stage3-circle.md](results/stage3-circle.md) |
| 4. Variant A digits | Done, gate passed | [stage4-scalar-digits.md](results/stage4-scalar-digits.md) |
| 5. Two digits | Done, gate passed | [stage5-two-digits.md](results/stage5-two-digits.md) |
| 6. Ten digits | Not started | |
| 7. Scale | Not started | |
| Demo + video | Not started | |
| Blog post | Not started | |

## Stage 1 probe

Answers: is Jev deterministic on identical states; how sign accuracy falls with term count and margin; which state format works best; real tokens and latency per neuron; whether packing 8 neurons in one request changes answers.

- Scalar sweep: z in [-100, 100], 0.01 steps in [-1, 1] and 0.1 elsewhere, keyed batches of 50, 5 repeats, plus an isolated-call subset to compare against batching.
- Full-state: 100 random neurons per (format, term count), formats paired / parallel / products, term counts 10, 30, 75, 150, 250. Inputs are nonzero MNIST pixel values, weights N(0, 1) at two decimals, bias set so the true sum spans a normalized margin of -2 to +2. 20 neurons per cell repeated 5 times. Paired-format neurons also sent packed 8 per request.

Gate: the go/no-go rules in proposal.md stage 1.
