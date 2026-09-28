# Jevrons demo

Draw a digit and watch a 784-32-10 network answer it, one Jev call per neuron.

```bash
uv run python demo/server.py                                    # exact, mock, replay; http://127.0.0.1:8765
JEVRONS_BACKENDS=vercel uv run python demo/server.py --live --max-usd 0.01   # also live
uv run python demo/server.py --check                            # exact inference == jevrons.net.forward; replay == journal
uv run python demo/make_replay.py                               # re-cut demo/runs/stage7-* from runs/stage7/journal.jsonl
```

The default network is stage 7 (84.6% on Jev). Each neuron is stage 8g's arm E: one call carrying the
folded-bias products and two questions (the v1 yes/no and the above/below choice), averaged into the activation.

Sources:

- `exact` is a local step neuron. Free.
- `mock` resamples a recorded stage 7 call with the same layer and a similar z / spread. Free.
- `replay` plays back recorded answers. Stage 7 test digits from the "Test digit" button were recorded
  call for call, and so is anything already run live (demo/runs/journal.jsonl). Other inputs fall back
  to the mock and are labeled "not recorded".
- `live` makes real calls and is billed. A live draw is 42 calls and about 41,000 input tokens, roughly
  $0.0017 at $0.042 per 1M. The page shows its estimate before you press Run.

## Film mode

`/?film` hides the copy and controls, enlarges the pad and tiles, fixes the seed (7) and keeps the frame
still while calls stream in. Add:

| flag | effect |
| --- | --- |
| `&stroke=3` | draws a stored stroke (0-9) after `delay` ms, then runs |
| `&digit=5110` | loads an MNIST test digit instead |
| `&source=replay` | `exact`, `mock`, `replay` or `live` (live also needs `--live`, and runs on its own in film mode) |
| `&theme=dark` | `light` or `dark` |
| `&model=s7-jev`, `&compare=training`, `&seed=7`, `&delay=900` | as named |

The takes worth recording:

- `/?film&source=replay&stroke=3` replays the live Jev draw of stroke 3 from 2026-09-27: 42 real answers,
  predicted 3, with rate-limit waits squeezed.
- `/?film&source=replay&digit=5110` replays a stage 7 test 7 from the training journal.

A recording script can call `await window.jevrons.play("3")` (it resolves when the answer is in) and
`window.jevrons.clear()`. It can also watch `document.body.dataset.state` (`idle`, `drawing`, `running`,
`done`) or listen for `jevrons:done` on `window`.
