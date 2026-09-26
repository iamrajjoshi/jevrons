# Plan

Resume point for the project. The experiment design and staged gates are in [proposal.md](proposal.md); this file records owner decisions and where the work stands.

## Decisions (2026-09-26, Raj)

- Deliverables: a live draw-a-digit demo, a short screen-recorded video of it for Twitter, and a long-form post for Raj's engineering blog. Blog figures and the video need saved artifacts from every stage, so every paid call is journaled.
- Budget: no cap for now. Spend is still tracked per call and per stage; a stage can set `max_usd` if needed.
- Model: `jev-1.13.0`, pinned. A model change starts a new baseline.

## Stages

| Stage | Status | Output |
| --- | --- | --- |
| 0. Local capacity | Done | `spikes/results/` |
| 1. Probe | In progress | `runs/probe/` |
| 2. Logic (AND/OR/XOR) | Not started | |
| 3. Circle | Not started | |
| 4. Variant A digits | Not started | |
| 5. Two digits | Not started | |
| 6. Ten digits | Not started | |
| 7. Scale | Not started | |
| Demo + video | Not started | |
| Blog post | Not started | |

## Stage 1 probe

Answers: is Jev deterministic on identical states; how sign accuracy falls with term count and margin; which state format works best; real tokens and latency per neuron; whether packing 8 neurons in one request changes answers.

- Scalar sweep: z in [-100, 100], 0.01 steps in [-1, 1] and 0.1 elsewhere, keyed batches of 50, 5 repeats, plus an isolated-call subset to compare against batching.
- Full-state: 100 random neurons per (format, term count), formats paired / parallel / products, term counts 10, 30, 75, 150, 250. Inputs are nonzero MNIST pixel values, weights N(0, 1) at two decimals, bias set so the true sum spans a normalized margin of -2 to +2. 20 neurons per cell repeated 5 times. Paired-format neurons also sent packed 8 per request.

Gate: the go/no-go rules in proposal.md stage 1.
