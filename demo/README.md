# Jevrons demo

Draw a digit and watch a network read it, one Jev call per neuron.

```bash
uv run python demo/server.py                                        # exact math only; http://127.0.0.1:8765
JEVRONS_BACKENDS=vercel uv run python demo/server.py --live         # visitors can also run live Jev
uv run python demo/server.py --check                                # parity, replay and live-limit self-checks
uv run python demo/make_replay.py                                   # re-cut demo/runs/stage7-* and stage8a-* from the training journals
```

## Networks

The default is stage 8a (`s8a-jev`, 85.3% on 1,000 test digits, on Jev): 784-64-10, where each hidden neuron is wired
to 96 fixed pixels picked at random from the ones that are ever inked. Its tiles show those receptive fields, 96 dots
on the 28 × 28 grid. A hidden call sends only the connected pixels that are nonzero in the image and whose weight
doesn't round to zero, plus the bias, about 37 numbers instead of stage 7's 150. The mask isn't stored anywhere:
training holds absent weights at exactly 0, so it's `W1 != 0`, and `--check` confirms that matches `stage8a.connections()`.

The other network is stage 7 (`s7-jev`, 84.6% on 2,000 test digits): 784-32-10, every pixel connected. Both have an
exact-trained twin (`s8a-swap`, `s7-swap`): same start, trained on a perfect step neuron, then run on Jev.

Every neuron is stage 8g's arm E: one call carrying the folded-bias products and two questions (the v1 yes/no and
the above/below choice), averaged into the activation. Stage 4-6, 3 vs 8 and 0 vs 1 are out of `models.json`; their
weights stay in `runs/`.

## Sources

Visitors pick between two:

- `live` makes real Jev calls and is billed: 74 calls, about 43,000 input tokens, $0.0018 a draw for stage 8a
  ($0.0017 for stage 7). It's the default when the server runs with `--live`. The page prices each draw before Run.
- `exact` is a local step neuron, free. It's the default without `--live`, and the page says live is off.

"Compare with the network trained on perfect math" runs the twin next to it, so a live compare is two draws (about
$0.0035).

Two more stay in the code, for URL flags and as fallbacks:

- `replay` plays back recorded answers keyed by the exact request. The first 12 test digits of each class from the
  stage 7 and 8a test passes were recorded call for call ("Test digit" picks one of them), and so are the live draws
  in `demo/runs/journal.jsonl`.
- `mock` resamples a recorded test call from the same network family with the same layer and a similar z / spread.

When a live draw is refused (daily budget, per-visitor or concurrency limit) or its calls fail (rate limit, payment,
outage), it finishes on recorded answers where that exact request was recorded and simulated ones elsewhere. The
panel says why, marks those tiles R or S, and doesn't bill them. Rate limits are named; payment and other failures
get a generic message.

## Live limits

| flag | default | |
| --- | --- | --- |
| `--daily-usd` | 9 | live spend cap per UTC day, about 5,000 draws, counted in `demo/runs/spend.json` (last 7 days kept) |
| `--per-ip-hour` | 0 (off) | live draws per visitor per hour |
| `--max-live` | 0 (off) | live draws running at once; the Dockerfile sets 16 |
| `--journal` | off | keep live calls in `demo/runs/journal.jsonl`; they replay after a restart |

A draw that stops early is charged a full draw, since calls in flight can still bill.

## Film mode

`/?film` hides the copy and controls, enlarges the pad and tiles, fixes the seed (7) and keeps the frame still while
calls stream in. Film mode defaults to stage 7 and the mock, which is what the video's takes were recorded on. Add:

| flag | effect |
| --- | --- |
| `&stroke=3` | draws a stored stroke (0-9) after `delay` ms, then runs |
| `&digit=5110` | loads an MNIST test digit instead |
| `&source=replay` | `exact`, `mock`, `replay` or `live` (live also needs `--live`, and runs on its own in film mode) |
| `&model=s8a-jev` | any name in `models.json` |
| `&theme=dark` | `light` or `dark` |
| `&compare=training`, `&seed=7`, `&delay=900` | as named (`compare=neuron` puts exact and Jev side by side) |

The takes the video uses:

- `/?film&source=replay&stroke=3` replays the live stage 7 draw of stroke 3 from 2026-09-27: 42 real answers,
  predicted 3, with rate-limit waits squeezed.
- `/?film&source=replay&digit=5110` replays a stage 7 test 7 from the training journal.
- `/?film&source=replay&stroke=3&model=s8a-jev` replays a live stage 8a draw of the same stroke from 2026-09-28.

A recording script can call `await window.jevrons.play("3")` (it resolves when the answer is in) and
`window.jevrons.clear()`. It can also watch `document.body.dataset.state` (`idle`, `drawing`, `running`, `done`) or
listen for `jevrons:done` on `window`.

## Deploy

`demo/Dockerfile` builds from the repo root and needs a local checkout with the gitignored files it copies: the MNIST
test set, `runs/stage8a/weights-*.npz`, `runs/stage7/weights-*.npz` and `runs/margin/summary.json` (`.dockerignore`
lets only those in). It runs `--live --daily-usd 9 --max-live 16` on Vercel and writes nothing but the spend counter.

```bash
fly launch --no-deploy --copy-config --config demo/fly.toml --dockerfile demo/Dockerfile
fly secrets set VERCEL_API_KEY=... --config demo/fly.toml
fly deploy --config demo/fly.toml --dockerfile demo/Dockerfile
```

On a 512 MB machine the server idles at about 24 MB, holds about 90 MB once the test set is loaded, and peaked at
220 MB over 16 concurrent draws on recorded answers (the fallback path, which also loads the replay files).

Screenshots are in `demo/screenshots/`.
