# Plan

Resume point for the project. The experiment design and staged gates are in [proposal.md](proposal.md); this file records owner decisions and where the work stands.

Inspiration: [Mustafa Akın's NAND-gate ALU made of Jev calls](https://x.com/mustafaakin/status/2103574428475154635). Credit him wherever the ALU numbers appear.

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
| 6. Ten digits | Done | [stage6-ten-digits.md](results/stage6-ten-digits.md) |
| 6b. What Jev's behaviour reveals | Done | [stage6b-jev-quirks.md](results/stage6b-jev-quirks.md) |
| 7. Scale | Not started | |
| 8g. A Jevron that behaves like a neuron | Queued after stage 6 | `runs/stage8g/` |
| 8. Improvements | Planned | |
| 9. Open-weight neurons (Ollaya) | Planned, needs install approval | |
| 9c. Laya as a scalar neuron | Done | [stage9c-laya-scalar.md](results/stage9c-laya-scalar.md) |
| 10. Teaching laya to add | Done | [stage10-teaching-laya.md](results/stage10-teaching-laya.md) |
| Demo + video | Demo built (`demo/`), video not started | |
| Blog post | Not started | |

## Stage 1 probe

Answers: is Jev deterministic on identical states; how sign accuracy falls with term count and margin; which state format works best; real tokens and latency per neuron; whether packing 8 neurons in one request changes answers.

- Scalar sweep: z in [-100, 100], 0.01 steps in [-1, 1] and 0.1 elsewhere, keyed batches of 50, 5 repeats, plus an isolated-call subset to compare against batching.
- Full-state: 100 random neurons per (format, term count), formats paired / parallel / products, term counts 10, 30, 75, 150, 250. Inputs are nonzero MNIST pixel values, weights N(0, 1) at two decimals, bias set so the true sum spans a normalized margin of -2 to +2. 20 neurons per cell repeated 5 times. Paired-format neurons also sent packed 8 per request.

Gate: the go/no-go rules in proposal.md stage 1.

## Stage 6b: what Jev's behaviour reveals

Mostly offline, over journals already paid for (backed up to `~/jevrons-journal-backup/`). Error model of misfires (margin, term count, sign mix, where the big terms sit, whether Jev skims the list); calibration of p; margin distributions by epoch for Jev-trained vs swap runs; about 5 adversarial trap families with matched controls (live, $1 cap); weight maps for Jev-trained vs exact-trained networks from the same start; repeat noise and backend equivalence at scale. Framing: optimization finds quirks you didn't know to look for, led by the bias-trust story. Output: `docs/results/stage6b-jev-quirks.md`.

## Stage 8g: make a Jevron behave like a real neuron (queued)

Stage 5's 3 vs 8 setup, so stage 5 is the baseline. Uses the measured response curve P(fire) = Φ(k(n)(z/spread − m0(n))). Arms: B backprops through the measured curve; C also pre-compensates the threshold by shifting the sent bias by m0 × spread; D also averages three phrasings asked about the same state in one call. The swap gap (exact-trained network: accuracy on an exact neuron minus accuracy on Jev) is re-measured on the C and D neurons; if Jev now behaves like the intended neuron, it shrinks. A quirk-matched local mock checked the code (swap 86.1% → 88.4% with compensation); real Jev is harsher than the mock. ~$17, ~2 h.

## Stage 8: improvements (planned)

Each arm is compared against the stage 6 baseline on the same split, seeds and test set, and reports accuracy, misfire rate, tokens per inference and training cost.

- 8a Sparse neurons. L1 or top-k on first-layer weights so each hidden neuron sums ~30 terms instead of ~150. Stage 1 says Jev is more accurate on shorter sums (90% at 75 terms vs 80% at 150), and tokens drop ~4x. Target: equal or better accuracy at a quarter of the cost per call.
- 8b Margin training. Add a hinge term that pushes each neuron's intended sum at least m spreads away from zero. Stage 5 found training does this implicitly; doing it explicitly should cut misfires further.
- 8c Pretrain on the stand-in, fine-tune on Jev. Train most epochs locally against the graded mock neuron (free), then 1-2 epochs through live Jev. Target: stage 6 accuracy at ~20% of its training cost.
- 8e Learn the neuron from the journals (Raj's idea). Every paid call is a labeled example of Jev's behavior: (products list, true total, term count) in, p out. ~2M such pairs exist after stage 6. Fit a small local model of the Jev neuron on them (features such as margin = total/spread, term count, bias share, largest terms) and validate it on held-out calls. A good fit replaces the hand-tuned stand-in: train networks against it for free, fine-tune briefly on live Jev (8c), and use it to predict which neurons Jev will get wrong. Stretch goal: fine-tune an open-weight System-1 model (stage 9) on the same pairs to get a local, free, Jev-like neuron.
- 8d Test-time voting. Jev is stochastic, so average k live passes per image. Measure accuracy against cost for k = 1, 3, 5.

## Stage 9: open-weight neurons (planned)

[Ollaya](https://ollaya.dev/) ([source](https://github.com/ollaya-dev/ollaya), Apache-2.0) runs open-weight System-1 models locally (laya 322-421M, nli, decider 0.75-4.2B, others) behind the same `/v1/systemone` API, so it is one more `Backend` entry (`http://127.0.0.1:11435`, no key). Runs on this Mac's CPU via ONNX Runtime; free per call.

- 9a Probe each model with the stage 1 neuron probe (sign accuracy vs term count, determinism, latency).
- 9b XOR and 3 vs 8 with the best model, with the swap control.
- 9c If a local model is a usable adder, run the stage 8 sweeps on it for free, then confirm the winners on Jev.
- Question for the blog: which System-1 model is the best neuron, and does training compensate for a worse one?

## Stage 10: teaching laya to add (done)

Hypothesis: laya (421M, open weights, Apache-2.0) fails as a neuron because it was never taught, not because a model its size can't learn it. Fine-tune it on Mac MPS in an isolated `laya/` sub-project: a "true math" arm (unlimited synthetic folded-format sums up to the 512-token limit plus real journal states, labelled with the true sign) and a "copy Jev" arm (journal states labelled with Jev's p). Evaluate with Jev's margin probe (k, m0 per term count, held-out lengths), the stage 6b traps and the scalar sweep. Serve the winner through Ollaya or a local /v1/systemone server as a new backend, then train a sparse 3 vs 8 network through it with the swap control. No Jev calls.
