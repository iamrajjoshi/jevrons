# Jevrons demo

Draw a digit and watch a network read it, one Jev call per neuron.

```bash
uv run python demo/server.py                                        # exact math only; http://127.0.0.1:8765
JEVRONS_BACKENDS=vercel uv run python demo/server.py --live         # visitors can also run live Jev
uv run python demo/server.py --check                                # parity, replay and live-limit self-checks
uv run python demo/make_replay.py                                   # re-cut demo/runs/stage7-* and stage8a-* from the training journals
```

## Networks

The default is the sparse network (`s8a-jev`, 85.3% on 1,000 test digits, on Jev): 784-64-10, where each hidden neuron is wired
to 96 fixed pixels picked at random from the ones that are ever inked. Its tiles show those receptive fields, 96 dots
on the 28 × 28 grid. A hidden call sends only the connected pixels that are nonzero in the image and whose weight
doesn't round to zero, plus the bias, about 37 numbers instead of the dense network's 150. The mask isn't stored anywhere:
training holds absent weights at exactly 0, so it's `W1 != 0`, and `--check` confirms that matches `stage8a.connections()`.

The other network is the dense network (`s7-jev`, 84.6% on 2,000 test digits): 784-32-10, every pixel connected. Both have an
exact-trained twin (`s8a-swap`, `s7-swap`): same start, trained on a perfect step neuron, then run on Jev.

Every neuron is arm E from the two-question neuron experiment: one call carrying the folded-bias products and two questions (the v1 yes/no and
the above/below choice), averaged into the neuron's output. The earlier digit networks (the scalar neuron, 0 vs 1, 3 vs 8 and
the first ten-digit runs) are out of `models.json`; their weights stay in `runs/`.

## Sources

Visitors pick between two:

- `live` makes real Jev calls and is billed: 74 calls, about 43,000 input tokens, $0.0018 a draw for the sparse network
  ($0.0017 for the dense one). It's the default when the server runs with `--live`. There's no Run button: a draw starts 800 ms after the
  last stroke (Enter on the pad starts it now), and drawing again mid-draw cancels it. The "this draw" box shows what it cost.
- `exact` is a local step neuron, free. It's the default without `--live`, and the page says live is off.

"Also run a network trained with perfect math" runs the twin next to it, so a live compare is two draws (about
$0.0035). The verdict above the panels names them A and B. "Show a test digit they read differently" loads a recorded
test digit the twins disagree on (`GET /api/disagree?model=s8a-jev` lists them; the dense network's twin has no recorded answers,
so it has no list).

Two more stay in the code for URL flags (film mode and the video), never as a stand-in for live:

- `replay` plays back recorded answers keyed by the exact request. The first 12 test digits of each class from the
  dense and sparse networks' test passes were recorded call for call ("Test digit" picks one of them), and so are the live draws
  in `demo/runs/journal.jsonl`.
- `mock` resamples a recorded test call from the same network family with the same layer and a similar z / spread.

Both read recorded Jev answers (`demo/runs/*-replay.jsonl`, `*-mock.json`, `journal.jsonl`), which aren't published;
`demo/make_replay.py` rebuilds them from the training journals. The site only needs `demo/runs/recorded.json`, the
digit indices behind "Test digit" and the compare view.

A replayed request that was never recorded falls back to the mock, and its tile is marked S.

## When Jev is busy

A live draw is only ever answered by live Jev. When it can't be answered right away, it waits:

- Past `--max-live` draws, a new one waits in line, first come first served. It holds no worker threads until its turn.
- All live draws share one call rate (`--calls-per-min`, with a burst of 128 so an idle demo sends a layer at once).
  A 429 halves it, at most once every 2 s and never below a sixteenth, and it climbs back over a minute.
- A call that gets a 429, a 5xx or a timeout tries again after 1, 2, 4, 8 s, then every 10 s, with jitter. Slow calls
  are still hedged at 1.5, 4 and 10 s when the shared rate has room.
- The stream says what it's waiting on (`{"type": "waiting", "reason": "queue" | "busy" | "pace", "ahead": N}`) and
  sends a keepalive comment every 10 s, so Cloudflare and Fly keep it open. The panel footer reads "waiting on
  TypeSafe · 3 draws ahead" or "waiting on TypeSafe · retrying", and a toast says so once per draw.
- A visitor who leaves (or draws again) gives up their place or slot, and the draw's remaining calls are never sent.

A live draw ends early only on the daily cap ("Today's live budget is used up. Exact math still works, or try again
tomorrow.", before any call), a key or billing failure ("Live Jev isn't available right now. Try exact math.", never
the code), or 5 minutes after it arrived ("TypeSafe is still under heavy load. Try again in a few minutes.").

A banner across the top says when live draws are slow. The server reads TypeSafe's status page
(`status.typesafe.ai/index.json`, at most once a minute, off the request path; the page never calls it) and watches its
own calls: 3 transient failures within a minute mark it degraded, and 20 successes in a row or 2 minutes without one
clear it. If TypeSafe reports an issue, the banner quotes it; if only our calls are failing, it says TypeSafe is under
heavy load. `GET /api/status` returns `{"degraded": bool, "typesafe": {"state", "report"} | null}`, and open pages poll
it every 30 s. `JEVRONS_STATUS_URL` points it elsewhere (empty turns it off).

### Trying it offline

Two hidden flags run the whole live path against a fake Jev, for $0 and with no real key sent anywhere:

```bash
uv run python demo/server.py --stub busy=30,delay=0.4 --port 8766   # 429 for 30 s, then answers after 0.4 s (code=402 for a billing failure)
uv run python demo/server.py --live --upstream http://127.0.0.1:8766/ --max-live 1   # the demo, keyless, spend kept off demo/runs/spend.json
```

`--check` runs the same stub in-process: 429s then success, a 402, the daily cap, the queue, a visitor leaving while
queued and while running, and the status page (operational, an open report, unreachable).

## Live limits

| flag | default | |
| --- | --- | --- |
| `--daily-usd` | 9 | live spend cap per UTC day, about 5,000 draws, counted in `demo/runs/spend.json` (last 7 days kept) |
| `--per-ip-hour` | 0 (off) | live draws per visitor per hour; past it the draw is refused, not simulated |
| `--max-live` | 0 (off) | live draws running at once, the rest wait in line; the Dockerfile sets 16 |
| `--calls-per-min` | 2200 | Jev calls a minute across all live draws, halved after a 429 (0 = no limit) |
| `--journal` | off | keep live calls in `demo/runs/journal.jsonl`; they replay after a restart |

A draw that stops early is charged a full draw, since calls in flight can still bill.

## Film mode

`/?film` hides the copy and controls, enlarges the pad and tiles, fixes the seed (7) and keeps the frame still while
calls stream in. Film mode defaults to the dense network and the mock. Add:

| flag | effect |
| --- | --- |
| `&stroke=3` | draws a stored stroke (0-9) after `delay` ms, then runs |
| `&digit=5110` | loads an MNIST test digit instead |
| `&source=replay` | `exact`, `mock`, `replay` or `live` (live also needs `--live`, and runs on its own in film mode) |
| `&model=s8a-jev` | any name in `models.json` |
| `&theme=dark` | `light` or `dark` |
| `&compare=training`, `&seed=7`, `&delay=900` | as named (`compare=neuron` puts exact and Jev side by side) |

A recording script can call `await window.jevrons.play("3")` (it resolves when the answer is in) and
`window.jevrons.clear()`. It can also watch `document.body.dataset.state` (`idle`, `drawing`, `running`, `done`) or
listen for `jevrons:done` on `window`.

## Deploy

`demo/Dockerfile` builds from the repo root and needs a local checkout with the gitignored files it copies: the MNIST
test set, `runs/stage8a/weights-*.npz`, `runs/stage7/weights-*.npz` and `runs/margin/summary.json` (`.dockerignore`
lets only those in). It runs `--live --daily-usd 30 --max-live 16 --calls-per-min 3000` on Vercel and writes nothing
but the spend counter.

```bash
fly apps create jevrons-demo
fly secrets set VERCEL_API_KEY="$(pbpaste)" -a jevrons-demo   # key copied to the clipboard, kept out of shell history
fly deploy . --config demo/fly.toml --dockerfile demo/Dockerfile --ha=false
```

`--ha=false` keeps it to one machine (Fly otherwise starts two on the first deploy). If an app already has two,
`fly scale count 1 -a jevrons-demo` brings it back to one.

The server idles at about 40 MB and holds about 115 MB once the test set is loaded. Against a local stub, 16 running
draws plus 50 in line peaked at 182 MB (1,221 threads), both when every call took 3 s and when Jev returned 429 for
20 s first; 16 running plus 100 in line peaked at 181 MB (measured on macOS, 2026-09-29).

Screenshots are in `demo/screenshots/`.
