"use strict";
/* Jevrons demo. No framework: state lives in `S`, every render reads from it.

   URL flags (all optional, combine freely):
     ?film              presentation state for recording: no copy or controls, bigger pad, fixed seed
     &stroke=3          draw a stored stroke (0-9) after `delay` ms, then run; see STROKES
     &digit=4203        load MNIST test digit #4203 instead (under stage 7 + replay, use a recorded one:
                        GET /api/digit?recorded=s7-jev lists one at random)
     &source=mock       exact | mock | replay | live   (live also needs `server.py --live`)
     &model=s7-jev      any name in demo/models.json
     &compare=training  one | training | neuron
     &seed=7            mock/replay randomness; film defaults to 7 so a take repeats exactly
     &delay=900         ms before the stroke starts (default 900)
     &theme=dark        light | dark (default follows the system)
   From a recording script: window.jevrons.play("3") draws and runs, resolving when the answer is in;
   window.jevrons.clear(); document.body.dataset.state is idle | drawing | running | done, and a
   "jevrons:done" event fires on window when every panel has its answer. */

const $ = (s, el = document) => el.querySelector(s);
const USD_PER_TOKEN = 0.042 / 1e6;
const Q = new URLSearchParams(location.search);
const FILM = Q.has("film");
const TOKENS = { e: [381, 5.37], folded: [310, 5.3], scalar: [310, 0] };  // input tokens ≈ a + b · numbers sent
const SOURCE_NAME = { exact: "exact arithmetic", mock: "mock Jev", replay: "recorded Jev", live: "live Jev" };

const S = {
  models: [], live: false, weights: {},
  input: null, real: null,            // processed 784 input; {index, label} when it is an unmodified test digit
  compare: "one", trained: "jev", source: "mock", task: null,
  panels: [], t: Infinity, sel: null, // t: scrub position in ms (Infinity = now); sel: {panel, layer, j}
};

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const hid = (j) => "h" + String(j + 1).padStart(2, "0");
const usd = (v) => "$" + v.toFixed(v < 0.01 ? 4 : 2);
const setState = (s) => { document.body.dataset.state = s; };

// ---------- drawing pad and MNIST preprocessing ----------

const pad = $("#pad"), pctx = pad.getContext("2d", { willReadFrequently: true });
let drawing = false, last = null, playing = false;

function padPoint(e) {
  const r = pad.getBoundingClientRect();
  return [(e.clientX - r.left) * pad.width / r.width, (e.clientY - r.top) * pad.height / r.height];
}
pad.addEventListener("pointerdown", (e) => {
  if (playing) return;
  pad.setPointerCapture(e.pointerId);
  if (S.real) { clearPad(); S.real = null; }
  drawing = true; last = padPoint(e); stroke(last, last); $("#pad-empty").style.opacity = 0;
});
pad.addEventListener("pointermove", (e) => { if (drawing) { const p = padPoint(e); stroke(last, p); last = p; } });
const end = () => { if (!drawing) return; drawing = false; updateInput(); if (autorun()) runAll(); };
pad.addEventListener("pointerup", end);
pad.addEventListener("pointercancel", end);
const autorun = () => S.source !== "live" || FILM;  // live spends money: only on Run, or a scripted film take

function stroke(a, b) {
  pctx.strokeStyle = css("--ink"); pctx.lineWidth = 42; pctx.lineCap = pctx.lineJoin = "round";
  pctx.beginPath(); pctx.moveTo(...a); pctx.lineTo(...b); pctx.stroke();
}
function clearPad() { pctx.clearRect(0, 0, pad.width, pad.height); }

// MNIST: bounding box scaled to fit 20x20 (area-averaged, so anti-aliased), then placed in 28x28 so the
// center of mass sits at the center.
function preprocess(alpha, W, H) {
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if (alpha[y * W + x] > 12) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (x1 < 0) return null;
  const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h);
  const tw = Math.max(1, Math.round(w * s)), th = Math.max(1, Math.round(h * s));
  const sum = new Float32Array(tw * th), cnt = new Float32Array(tw * th);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = Math.min(th - 1, Math.floor(y * th / h)) * tw + Math.min(tw - 1, Math.floor(x * tw / w));
    sum[k] += alpha[(y + y0) * W + x + x0] / 255; cnt[k]++;
  }
  let m = 0, mx = 0, my = 0;
  const small = sum.map((v, k) => v / cnt[k]);
  const peak = Math.max(...small);
  for (let k = 0; k < small.length; k++) { small[k] /= peak; m += small[k]; mx += small[k] * (k % tw + 0.5); my += small[k] * (Math.floor(k / tw) + 0.5); }
  const ox = Math.min(28 - tw, Math.max(0, Math.round(14 - mx / m)));
  const oy = Math.min(28 - th, Math.max(0, Math.round(14 - my / m)));
  const out = new Float32Array(784);
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) out[(y + oy) * 28 + x + ox] = small[y * tw + x];
  return out;
}

function updateInput() {
  const d = pctx.getImageData(0, 0, pad.width, pad.height).data;
  const a = new Uint8ClampedArray(pad.width * pad.height);
  for (let i = 0; i < a.length; i++) a[i] = d[i * 4 + 3];
  S.input = preprocess(a, pad.width, pad.height);
  renderInput();
}

async function loadReal(index) {
  const m = S.panels[0].model, labels = $("#real-label").value || m.labels.join(",");
  const url = Number.isInteger(index) ? `/api/digit?index=${index}`
    : `/api/digit?labels=${labels}` + (m.replay ? `&recorded=${m.name}` : "");
  const r = await (await fetch(url)).json();
  S.input = Float32Array.from(r.pixels); S.real = { index: r.index, label: r.label };
  const c = document.createElement("canvas"); c.width = c.height = 28;
  paintGray(c, S.input);
  clearPad(); pctx.imageSmoothingEnabled = false; pctx.drawImage(c, 0, 0, pad.width, pad.height);
  renderInput();
  if (autorun()) return runAll();
}

// ---------- stored strokes, for reproducible film takes ----------
// Pen paths in pad units (0-1), one array per pen-down. Smoothed with Catmull-Rom and drawn at hand speed.
const STROKES = {
  0: [[[.52, .2], [.38, .26], [.31, .44], [.33, .64], [.44, .78], [.58, .77], [.67, .6], [.67, .38], [.6, .24], [.5, .2], [.44, .23]]],
  1: [[[.44, .3], [.53, .2], [.53, .5], [.52, .8]]],
  2: [[[.32, .33], [.4, .22], [.54, .19], [.65, .27], [.64, .42], [.52, .56], [.36, .72], [.31, .79], [.5, .77], [.7, .79]]],
  3: [[[.33, .26], [.45, .19], [.6, .21], [.65, .32], [.57, .43], [.46, .47], [.58, .5], [.67, .6], [.64, .74], [.5, .81], [.34, .77]]],
  4: [[[.52, .2], [.36, .5], [.33, .57], [.5, .57], [.7, .56]], [[.6, .38], [.59, .6], [.58, .82]]],
  5: [[[.66, .2], [.4, .21], [.37, .45], [.5, .42], [.64, .5], [.67, .64], [.6, .77], [.46, .81], [.33, .75]]],
  6: [[[.62, .2], [.48, .3], [.37, .48], [.34, .64], [.42, .78], [.56, .8], [.65, .68], [.61, .54], [.49, .5], [.38, .58]]],
  7: [[[.3, .22], [.5, .21], [.7, .21], [.6, .38], [.51, .57], [.45, .8]]],
  8: [[[.6, .26], [.5, .19], [.38, .25], [.4, .39], [.55, .49], [.65, .62], [.6, .77], [.47, .8], [.36, .7], [.4, .56], [.55, .47], [.63, .35], [.6, .25]]],
  9: [[[.63, .3], [.53, .2], [.38, .25], [.36, .4], [.48, .47], [.62, .4], [.63, .28], [.62, .55], [.58, .81]]],
};

function smooth(pts, per = 12) {
  const out = [], P = (i) => pts[Math.max(0, Math.min(pts.length - 1, i))];
  for (let i = 0; i < pts.length - 1; i++) for (let s = 0; s < per; s++) {
    const t = s / per, [a, b, c, d] = [P(i - 1), P(i), P(i + 1), P(i + 2)];
    out.push([0, 1].map((k) => 0.5 * (2 * b[k] + (c[k] - a[k]) * t + (2 * a[k] - 5 * b[k] + 4 * c[k] - d[k]) * t * t +
      (3 * b[k] - a[k] - 3 * c[k] + d[k]) * t ** 3)));
  }
  out.push(pts.at(-1));
  return out.map(([x, y]) => [x * pad.width, y * pad.height]);
}

async function draw(name, speed = 0.8) {           // speed in pad px per ms
  const paths = STROKES[name];
  if (!paths) throw new Error(`no stored stroke ${name}; have ${Object.keys(STROKES)}`);
  playing = true; setState("drawing"); clear(); $("#pad-empty").style.opacity = 0;
  for (const path of paths) {
    const pts = smooth(path);
    let i = 1, carry = 0, t = performance.now();
    stroke(pts[0], pts[0]);
    await new Promise((done) => {
      const step = (now) => {
        carry += (now - t) * speed; t = now;
        while (i < pts.length && carry > 0) { carry -= Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); stroke(pts[i - 1], pts[i]); i++; }
        i < pts.length ? requestAnimationFrame(step) : done();
      };
      requestAnimationFrame(step);
    });
    await new Promise((r) => setTimeout(r, 140));  // pen lift
  }
  playing = false; updateInput();
}

async function play(name) { await draw(name); if (autorun()) await runAll(); }

// ---------- pixel maps ----------

function rgb(hex) {
  let h = hex.replace("#", ""); if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255];
}

function paintGray(canvas, v) {           // intensity 0..1 as ink on paper
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(28, 28), ink = rgb(css("--ink")), paper = rgb(css("--paper"));
  for (let i = 0; i < 784; i++) { for (let c = 0; c < 3; c++) img.data[i * 4 + c] = paper[c] + (ink[c] - paper[c]) * v[i]; img.data[i * 4 + 3] = 255; }
  ctx.putImageData(img, 0, 0);
}

// Signed map: positive (pushes the sum up) in ink, negative in mid gray. `faint` weights sit under products.
function paintSigned(canvas, vals, faint = null) {
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(28, 28);
  const ink = rgb(css("--ink")), neg = rgb(css("--mute")), paper = rgb(css("--paper"));
  const mx = Math.max(1e-6, ...vals.map(Math.abs)), fm = faint ? Math.max(1e-6, ...faint.map(Math.abs)) : 1;
  for (let i = 0; i < 784; i++) {
    let v = vals[i] / mx, a = Math.min(1, Math.abs(v) ** 0.7);
    if (faint && vals[i] === 0) { v = faint[i] / fm; a = Math.min(1, Math.abs(v)) * 0.16; }
    const col = v >= 0 ? ink : neg;
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = paper[c] + (col[c] - paper[c]) * a;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const kept = (x) => Math.round(x * 100) / 100 !== 0;  // the server drops inputs that round to zero
const products = (w) => S.input ? w.map((wi, i) => kept(S.input[i]) ? S.input[i] * wi : 0) : w.map(() => 0);

function renderInput() {
  paintGray($("#seen"), S.input || new Float32Array(784));
  $("#pad-empty").style.opacity = S.input ? 0 : 1;
  $("#pad-note").textContent = S.real ? `MNIST test #${S.real.index}` : S.input ? "your drawing" : "28 × 28 after preprocessing";
  $("#seen-note").textContent = S.real
    ? `Test digit #${S.real.index}, labeled ${S.real.label}. Already 28 × 28, sent as is.`
    : "What the network sees: your strokes fit to 20 × 20 and centered by mass, as MNIST was made.";
  for (const p of S.panels) paintTiles(p);
  renderBill();
}

// ---------- configuration ----------

const pick = (task, trained) => S.models.find((m) => m.title === task && m.trained === trained && m.available);

function plan() {
  const one = pick(S.task, S.trained) || pick(S.task, "jev") || pick(S.task, "exact");
  if (S.compare === "training" && pick(S.task, "jev") && pick(S.task, "exact"))
    return [{ model: pick(S.task, "jev"), source: S.source }, { model: pick(S.task, "exact"), source: S.source }];
  if (S.compare === "neuron")
    return [{ model: one, source: "exact" }, { model: one, source: S.source === "exact" ? "mock" : S.source }];
  return [{ model: one, source: S.source }];
}

function setSeg(id, v) { for (const b of $(id).children) b.setAttribute("aria-pressed", String(b.dataset.v === v)); }
function bindSeg(id, key) {
  $(id).addEventListener("click", (e) => {
    const v = e.target.dataset.v; if (!v || e.target.disabled) return;
    S[key] = v; setSeg(id, v);
    configure().then(() => S.input && autorun() && runAll());
  });
}

function sourceHelp(fmt) {
  return {
    exact: "Local step neurons that fire exactly when the sum is positive: what these weights do on a perfect neuron. Free.",
    mock: fmt === "e" ? "Each call draws a recorded stage 7 answer from a neuron with a similar sum, so Jev's quirks come along. Local and free."
      : "A local stand-in fitted to Jev's measured error curve, with made-up latency. Free.",
    replay: "Jev's recorded answers, in their recorded order (waits over 1.5 s, which are rate-limit retries, are squeezed). Stage 7's test digits were recorded call for call; anything else falls back to the mock and is labeled so.",
    live: "Real calls to Jev from this server, journaled to demo/runs/journal.jsonl. Nothing is sent until you press Run.",
  }[S.source];
}

async function configure() {
  const hasTwin = pick(S.task, "jev") && pick(S.task, "exact");
  $("#compare [data-v=training]").disabled = !hasTwin;
  if (!hasTwin && S.compare === "training") { S.compare = "one"; setSeg("#compare", "one"); }
  $("#trained-field").hidden = S.compare === "training" || !hasTwin;
  $("#source [data-v=live]").disabled = !S.live;
  $("#source [data-v=live]").title = S.live ? "" : "Start the server with --live";
  const cfg = plan(), m = cfg[0].model;
  const COMPARE_HELP = {
    training: "Top: trained through Jev. Bottom: same start, trained on exact neurons, then run on the same neurons.",
    neuron: "Same weights twice. Top: exact arithmetic. Bottom: Jev.", one: "" };
  $("#source-help").textContent = [COMPARE_HELP[S.compare], sourceHelp(m.format)].filter(Boolean).join(" ");
  $("#n-neurons").textContent = 32 + (m.labels.length > 2 ? m.labels.length : 1);
  const labels = `<option value="">any</option>` + m.labels.map((d) => `<option>${d}</option>`).join("");
  if ($("#real-label").innerHTML !== labels) $("#real-label").innerHTML = labels;
  await buildPanels(cfg);
}

// A live draw's cost before it happens, from the token fit for this neuron format and the pixels drawn.
// Counts the panels that would call Jev (all of them when the source is exact).
function estimate() {
  const nnz = S.input ? S.input.filter(kept).length : 150;
  const calling = S.panels.filter((p) => p.source !== "exact");
  return (calling.length ? calling : S.panels).reduce((acc, p) => {
    const [a, b] = TOKENS[p.model.format], outs = p.model.labels.length > 2 ? p.model.labels.length : 1;
    const hiddenT = p.model.format === "scalar" ? a : a + b * (nnz + 1), outT = p.model.format === "scalar" ? a : a + b * 33;
    return { calls: acc.calls + 32 + outs, tokens: acc.tokens + 32 * hiddenT + outs * outT };
  }, { calls: 0, tokens: 0 });
}

function renderStatus() {
  const live = S.panels.some((p) => p.source === "live"), el = $("#status");
  const e = estimate();
  el.className = "status" + (live ? " live" : "");
  el.innerHTML = `<i></i>` + (live ? `live Jev · billed · ≈ ${usd(e.tokens * USD_PER_TOKEN)} a draw`
    : { exact: "exact arithmetic · no calls", mock: "mock Jev · local, not billed", replay: "recorded Jev · not billed" }[S.source]);
  $("#b-live").textContent = `Live, this draw would be ${e.calls} calls, ≈ ${(Math.round(e.tokens / 100) * 100).toLocaleString()} input tokens, ≈ ${usd(e.tokens * USD_PER_TOKEN)}.`;
  $("#run").textContent = live ? `Run live · ≈ ${usd(e.tokens * USD_PER_TOKEN)}` : "Run";
}

// ---------- panels ----------

async function weightsFor(name) {
  if (!S.weights[name]) S.weights[name] = await (await fetch(`/api/weights?name=${name}`)).json();
  return S.weights[name];
}

function panelTitle(p) {
  const trained = p.model.trained === "jev" ? "trained through Jev" : "trained exact";
  return `${p.model.title} · ${trained} · ${p.source === "exact" ? "run exact" : "on " + SOURCE_NAME[p.source]}`;
}

async function buildPanels(cfg) {
  for (const p of S.panels) p.ctrl?.abort();
  const root = $("#panels");
  S.panels = cfg.map((c, k) => ({ id: k, ...c, events: [], layers: [], result: null, error: null }));
  closeInspector();
  const els = [];
  for (const p of S.panels) {
    const el = $("#panel-tpl").content.firstElementChild.cloneNode(true);
    p.el = el; els.push(el);
    const binary = p.model.labels.length === 2;
    $(".p-title", el).textContent = panelTitle(p);
    $(".p-arch", el).textContent = `784 → 32 → ${binary ? 1 : p.model.labels.length}`;
    $(".p-note", el).textContent = p.model.note;
    p.w = await weightsFor(p.model.name);
    const grid = $(".hidden-grid", el);
    p.tiles = p.w.W1.map((_, j) => {
      const t = document.createElement("button");
      t.className = "tile idle"; t.style.setProperty("--d", `${(-Math.random() * 1.1).toFixed(2)}s`);
      t.innerHTML = `<canvas width="28" height="28"></canvas><span class="act"><i></i></span><span class="lbl">${hid(j)}</span>`;
      t.addEventListener("click", () => select(p, 0, j));
      grid.append(t); return t;
    });
    const outs = $(".outs", el);
    outs.classList.toggle("binary", binary);
    p.outs = (binary ? [p.model.labels[1]] : p.model.labels).map((lab, k) => {
      const o = document.createElement("button");
      o.className = "out"; o.style.setProperty("--d", `${(-Math.random() * 1.1).toFixed(2)}s`);
      o.innerHTML = `<span class="lab">${lab}</span><span class="bar-track"><i></i><b></b></span><span class="v"></span>`;
      o.title = `output ${lab}; the tick marks 0.5, where it fires`;
      o.addEventListener("click", () => select(p, 1, k));
      outs.append(o); return o;
    });
    paintTiles(p);
  }
  root.replaceChildren(...els);
  for (const p of S.panels) { new ResizeObserver(() => layoutWires(p)).observe(p.el); renderPanel(p); }
  renderCalls(); renderBill(); renderStatus();
}

function paintTiles(p) {
  if (p.tiles) p.tiles.forEach((t, j) => paintSigned($("canvas", t), products(p.w.W1[j]), p.w.W1[j]));
}

// One line per hidden→output weight, built when the layout changes; renders only change their opacity.
function layoutWires(p) {
  const svg = $(".wires", p.el), box = $(".net", p.el).getBoundingClientRect();
  const W2 = p.w.W2, mx = Math.max(...W2.flat().map(Math.abs));
  const r = (el) => { const b = el.getBoundingClientRect(); return { l: b.left - box.left, r: b.right - box.left, t: b.top - box.top, b: b.bottom - box.top }; };
  const g = r($(".hidden-grid", p.el)), outs = p.outs.map((o) => r($(".bar-track", o)));
  const across = outs[0].l > g.r;  // outputs to the right (desktop) or below (phone)
  // Wires leave the grid's edge in neuron order (a bus), so they never cross the tiles.
  const n = p.tiles.length, at = (j) => (j + 0.5) / n;
  let html = "";
  p.tiles.forEach((_, j) => p.outs.forEach((_, k) => {
    const o = outs[k], w = W2[j][k];
    const [x1, y1] = across ? [g.r + 6, g.t + (g.b - g.t) * at(j)] : [g.l + (g.r - g.l) * at(j), g.b + 6];
    const [x2, y2] = across ? [o.l - 3, (o.t + o.b) / 2] : [(o.l + o.r) / 2, o.t - 3];
    html += `<line class="${w < 0 ? "neg" : ""}" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" ` +
      `stroke-width="${(0.5 + 1.3 * Math.abs(w) / mx).toFixed(2)}" style="stroke-opacity:0"/>`;
  }));
  svg.innerHTML = html;
  p.wires = [...svg.children];
  p.wmax = mx;
  renderPanel(p);
}

// ---------- rendering a panel at time S.t ----------

const now = (p) => Math.min(S.t, p.t0 == null ? 0 : p.result ? Infinity : performance.now() - p.t0);
const visible = (p) => p.events.filter((e) => e.t_ms <= S.t);
const sent = (p, layer) => p.layers.some((l) => l.layer === layer && l.t_ms <= S.t);

function renderPanel(p) {
  if (!p.tiles) return;
  const ev = visible(p), byKey = new Map(ev.map((e) => [`${e.layer}:${e.j}`, e]));
  const binary = p.model.labels.length === 2;
  p.tiles.forEach((t, j) => {
    const e = byKey.get(`0:${j}`), flight = !e && sent(p, 0) && !p.error;
    t.className = "tile" + (e ? " done" : flight ? " flight" : " idle") + (e?.disagree ? " dis" : "") +
      (isSel(p, 0, j) ? " sel" : "") + (t._land ? " land" : "");
    t.style.setProperty("--p", e ? e.p : 0);
    $(".lbl", t).textContent = e ? e.p.toFixed(2) : hid(j);
  });
  const outEv = p.outs.map((_, k) => byKey.get(`1:${k}`));
  const done = !!p.result && ev.length === p.events.length;
  const top = done ? p.result.prediction : null;
  p.outs.forEach((o, k) => {
    const e = outEv[k], flight = !e && sent(p, 1) && !p.error;
    o.className = "out" + (e ? " done" : flight ? " flight" : "") + (e?.disagree ? " dis" : "") +
      (!binary && top === p.model.labels[k] ? " top1" : "") + (isSel(p, 1, k) ? " sel" : "");
    o.style.setProperty("--p", e ? e.p : 0);
    $(".v", o).textContent = e ? e.p.toFixed(2) : "";
  });
  if (p.wires) p.tiles.forEach((_, j) => {
    const e = byKey.get(`0:${j}`);
    p.outs.forEach((_, k) => {
      const w = p.w.W2[j][k], line = p.wires[j * p.outs.length + k];
      line.style.strokeOpacity = e ? (0.04 + 0.6 * e.p * Math.abs(w) / p.wmax).toFixed(3) : 0;
    });
  });

  const pred = $(".pred", p.el), big = $(".pred-big", p.el);
  const hiddenEv = ev.filter((e) => e.layer === 0), outs = ev.filter((e) => e.layer === 1);
  const inFlight = (sent(p, 0) ? p.tiles.length - hiddenEv.length : 0) + (sent(p, 1) ? p.outs.length - outs.length : 0);
  pred.classList.toggle("pending", !done);
  const text = done ? p.result.prediction : "";
  if (big.textContent !== text) { big.textContent = text; big.classList.remove("in"); if (text) { void big.offsetWidth; big.classList.add("in"); } }
  const fired = outs.filter((e) => e.p >= 0.5).length;
  $(".pred-cap", p.el).textContent = p.error ? `Stopped: ${p.error}` :
    done ? (binary ? `p ${p.result.outputs[0].toFixed(2)}; ${p.model.labels[1]} at 0.5 or more, else ${p.model.labels[0]}`
      : fired === 0 ? "No output reached 0.5, so the highest one wins."
      : fired === 1 ? `Output ${p.result.prediction} fired.` : `${fired} outputs fired; the highest wins.`)
    : inFlight ? `waiting on ${inFlight} call${inFlight > 1 ? "s" : ""}` : S.input ? "" : "Draw a digit, or load a test digit.";
  const dis = ev.filter((e) => e.disagree).length, unrec = ev.filter((e) => e.backend === "mock (not recorded)").length;
  $(".p-stats", p.el).innerHTML = !ev.length && !inFlight ? "no calls yet" :
    `hidden ${hiddenEv.length}/${p.tiles.length} · output ${outs.length}/${p.outs.length} · ` +
    `<span class="${dis ? "dis" : ""}">${dis} disagree</span>` + (unrec ? ` · ${unrec} not recorded, mocked` : "");
}

// ---------- running ----------

let runSeq = 0;
async function runAll() {
  if (!S.input || !S.panels.length) return;
  const seed = Q.has("seed") || FILM ? Number(Q.get("seed") ?? 7) : Math.floor(Math.random() * 2 ** 31), seq = ++runSeq;
  S.t = Infinity; $("#calls-range").value = 1000; setState("running");
  const runs = S.panels.map((p) => runPanel(p, seed));  // sets each panel's ctrl first, which tick() watches
  tick();
  await Promise.all(runs);
  if (seq === runSeq && S.panels.every((p) => p.result || p.error)) { setState("done"); dispatchEvent(new Event("jevrons:done")); }
}

async function runPanel(p, seed) {
  p.ctrl?.abort(); p.ctrl = new AbortController();
  Object.assign(p, { events: [], layers: [], result: null, error: null, t0: performance.now() });
  renderPanel(p); renderBill();
  const body = { model: p.model.name, source: p.source, seed,
    ...(S.real ? { mnist: S.real.index } : { pixels: Array.from(S.input, (v) => Math.round(v * 1000) / 1000) }) };
  try {
    const res = await fetch("/api/run", { method: "POST", signal: p.ctrl.signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error((await res.json()).error);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (line.startsWith("data: ")) onEvent(p, JSON.parse(line.slice(6)));
      }
    }
  } catch (e) {
    if (e.name === "AbortError") return;
    p.error = String(e.message || e);
  }
  p.ctrl = null;
  renderPanel(p); renderCalls(); renderBill();
}

function onEvent(p, e) {
  if (e.type === "start") p.t0 = performance.now();
  else if (e.type === "layer") p.layers.push(e);
  else if (e.type === "neuron") {
    p.events.push(e);
    const el = e.layer === 0 ? p.tiles[e.j] : null;
    if (el && S.t === Infinity) { el._land = true; setTimeout(() => { el._land = false; el.classList.remove("land"); }, 80); }
    if (S.sel && S.sel.panel === p && S.sel.layer === e.layer && S.sel.j === e.j) renderInspector();
  } else if (e.type === "result") p.result = e;
  else if (e.type === "error") p.error = e.message;
  renderPanel(p); renderBill();
}

function tick() {  // keeps in-flight bars and the clock moving while any panel waits
  renderCalls(); renderBill();
  if (S.panels.some((p) => p.ctrl && !p.result)) requestAnimationFrame(tick);
}

function renderBill() {
  const ps = S.panels.filter((p) => p.source !== "exact"), ev = ps.flatMap((p) => p.events), tokens = ev.reduce((a, e) => a + e.tokens, 0);
  const live = ps.some((p) => p.source === "live");
  const elapsed = Math.max(0, ...S.panels.map((p) => p.result ? p.result.elapsed_ms : p.ctrl ? now(p) : p.events.at(-1)?.t_ms || 0));
  $("#b-calls").textContent = ev.length.toLocaleString();
  $("#b-tokens").textContent = tokens.toLocaleString();
  $("#b-usd").textContent = usd(tokens * USD_PER_TOKEN);
  $("#b-time").textContent = (elapsed / 1000).toFixed(1) + " s";
  $("#bill-kind").textContent = !S.panels.some((p) => p.events.length) ? "nothing run" : live ? "billed" : ps.length ? "simulated, not billed" : "local, free";
  renderStatus();
}

// ---------- calls: one bar per call, from when it was sent to when Jev answered ----------

function renderCalls() {
  const ran = S.panels.filter((p) => p.layers.length);
  $("#calls").hidden = !ran.length;
  if (!ran.length) return;
  const svg = $("#calls-svg"), w = svg.clientWidth || 800, rowH = FILM ? 4 : 3, gap = 1, layerGap = 6, labelH = S.panels.length > 1 ? 16 : 4;
  const end = Math.max(1000, ...ran.map((p) => p.result ? p.result.elapsed_ms : now(p)), ...ran.flatMap((p) => p.events.map((e) => e.t_ms)));
  const X = (t) => 1 + (w - 2) * Math.min(t, end) / end;
  let y = 0, html = "";
  for (const p of ran) {
    if (S.panels.length > 1) { html += `<text x="0" y="${y + 11}">${panelTitle(p)}</text>`; }
    y += labelH;
    for (const l of p.layers) {
      const arrived = p.events.filter((e) => e.layer === l.layer).sort((a, b) => a.t_ms - b.t_ms);
      const tNow = now(p);
      for (let r = 0; r < l.n; r++) {
        const e = arrived[r], fut = e && e.t_ms > S.t, t1 = e ? (fut ? Math.max(Math.min(S.t, e.t_ms), l.t_ms) : e.t_ms) : tNow;
        if (e && fut) html += `<rect class="fut" x="${X(l.t_ms)}" y="${y}" width="${Math.max(1, X(e.t_ms) - X(l.t_ms))}" height="${rowH}"/>`;
        if (l.t_ms <= S.t) html += `<rect class="${!e || fut ? "wait" : e.disagree ? "dis" : ""}" x="${X(l.t_ms)}" y="${y}" width="${Math.max(1, X(t1) - X(l.t_ms))}" height="${rowH}"/>`;
        y += rowH + gap;
      }
      y += layerGap;
    }
  }
  if (S.t < Infinity) html += `<line class="head" x1="${X(S.t)}" x2="${X(S.t)}" y1="0" y2="${y}"/>`;
  svg.setAttribute("viewBox", `0 0 ${w} ${y}`); svg.setAttribute("height", y); svg.innerHTML = html;
  $("#calls-end").textContent = (end / 1000).toFixed(2) + " s";
  $("#calls-t").textContent = S.t < Infinity ? `at ${(S.t / 1000).toFixed(2)} s` : ran.some((p) => p.ctrl) ? "" : "drag to replay";
}

const runEnd = () => Math.max(1, ...S.panels.map((p) => p.result?.elapsed_ms || 0), ...S.panels.flatMap((p) => p.events.map((e) => e.t_ms)));
function seek(t) {
  S.t = t >= runEnd() ? Infinity : t;
  S.panels.forEach(renderPanel); renderCalls();
  if (S.sel) renderInspector();
}
$("#calls-range").addEventListener("input", (e) => seek(runEnd() * e.target.value / 1000));
let anim = null;
$("#replay-anim").addEventListener("click", () => {
  cancelAnimationFrame(anim);
  const end = runEnd(), dur = Math.max(1500, end), t0 = performance.now();
  const step = (tNow) => {
    const f = Math.min(1, (tNow - t0) / dur);
    $("#calls-range").value = Math.round(f * 1000); seek(f * end);
    if (f < 1) anim = requestAnimationFrame(step);
  };
  anim = requestAnimationFrame(step);
});

// ---------- inspector ----------

const isSel = (p, layer, j) => S.sel && S.sel.panel === p && S.sel.layer === layer && S.sel.j === j;
function select(p, layer, j) { S.sel = { panel: p, layer, j }; S.panels.forEach(renderPanel); renderInspector(); }
function closeInspector() { S.sel = null; $("#inspector").hidden = true; S.panels.forEach(renderPanel); }
$("#i-close").addEventListener("click", closeInspector);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeInspector(); });

function answerRows(e) {
  if (e.backend === "local step") return "";
  return Object.entries(e.answers).map(([k, a]) => typeof a === "object"
    ? `<dt>choice: above zero</dt><dd>${a.above.toFixed(2)}</dd>`
    : `<dt>${k === "v1" || k === "fires" ? "yes/no: greater than zero" : "yes/no"}</dt><dd>${a.toFixed(2)}</dd>`).join("") +
    (Object.keys(e.answers).length > 1 ? `<dt>activation, their mean</dt><dd class="${e.disagree ? "dis" : ""}">${e.p.toFixed(2)}</dd>` : "");
}

function renderInspector() {
  const { panel: p, layer, j } = S.sel;
  const e = visible(p).find((x) => x.layer === layer && x.j === j);
  const binary = p.model.labels.length === 2;
  const name = layer === 0 ? `hidden ${hid(j)}` : `output ${binary ? p.model.labels[1] : p.model.labels[j]}`;
  $("#inspector").hidden = false;
  $("#i-title").textContent = `${name} · ${panelTitle(p)}`;
  const body = $("#i-body");
  if (!e) {
    body.innerHTML = `<p class="i-verdict">${sent(p, layer) ? "Asked. Waiting for Jev." : "Not asked yet."}</p>` +
      (layer === 0 ? mapsHTML() : "") + `<p class="cap">Run the network to see what this neuron sends and what comes back.</p>`;
    if (layer === 0) paintMaps(p, j);
    return;
  }
  const fires = e.p >= 0.5, jevish = e.backend !== "local step";
  const verdict = !jevish ? `${fires ? "Fires" : "Silent"}: the sum is ${fmt(e.z)}.`
    : e.disagree ? `Jev says ${fires ? "fire" : "stay silent"}, but the sum is ${fmt(e.z)}.`
    : `Jev ${fires ? "fires" : "stays silent"}, and the sum is ${fmt(e.z)}. They agree.`;
  const sw = e.terms.length > 1;
  const request = `{"state": ${JSON.stringify(sw ? { products: e.terms } : { z: e.terms[0] }).replaceAll(",", ", ")},\n "questions": ` +
    JSON.stringify(e.questions, null, 2).replaceAll("\n", "\n ") + "}";
  body.innerHTML = `
    <p class="i-verdict ${e.disagree ? "dis" : ""}">${verdict}</p>
    ${layer === 0 ? mapsHTML() : ""}
    <dl class="facts">
      <dt>numbers sent</dt><dd>${e.n_terms}</dd>
      <dt>their true sum</dt><dd>${fmt(e.z)}</dd>
      ${answerRows(e)}
      <dt>answered after</dt><dd>${(e.latency_ms / 1000).toFixed(2)} s</dd>
      <dt>by</dt><dd>${e.backend}</dd>
      <dt>input tokens</dt><dd>${e.tokens.toLocaleString()}</dd>
      <dt>cost</dt><dd>$${(e.tokens * USD_PER_TOKEN).toFixed(6)}${p.source === "live" ? "" : ", not billed"}</dd>
    </dl>
    ${sw ? `<div><svg class="walk ${e.disagree ? "dis" : ""}" viewBox="0 0 400 110" preserveAspectRatio="none"></svg>
      <p class="cap">Running total over the ${e.terms.length} numbers, in the order Jev reads them. Jev gets only the list and has to add it up itself.</p></div>` : ""}
    <div><pre class="state">${request}</pre><p class="cap">The request, as sent.</p></div>`;
  if (layer === 0) paintMaps(p, j);
  if (sw) drawWalk($(".walk", body), e.terms);
}

const mapsHTML = () => `<div class="i-maps">
  <figure><canvas id="i-w" width="28" height="28"></canvas><figcaption>weights: ink +, gray −</figcaption></figure>
  <figure><canvas id="i-x" width="28" height="28"></canvas><figcaption>pixels × weights, sent to Jev</figcaption></figure></div>`;
function paintMaps(p, j) { paintSigned($("#i-w"), p.w.W1[j]); paintSigned($("#i-x"), products(p.w.W1[j])); }

function drawWalk(svg, terms) {
  let s = 0; const pts = [0, ...terms.map((v) => (s += v))];
  const lo = Math.min(0, ...pts), hi = Math.max(0, ...pts), span = hi - lo || 1;
  const X = (i) => 4 + 392 * i / (pts.length - 1), Y = (v) => 6 + 98 * (hi - v) / span;
  svg.innerHTML = `<line class="zero" x1="0" x2="400" y1="${Y(0)}" y2="${Y(0)}"/>` +
    `<polyline class="path" points="${pts.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")}"/>` +
    `<circle class="end" cx="${X(pts.length - 1)}" cy="${Y(pts.at(-1))}" r="3.5"/>`;
}

// ---------- boot ----------

function clear() {
  clearPad(); S.real = null; S.input = null; renderInput(); setState("idle");
  for (const p of S.panels) { p.ctrl?.abort(); Object.assign(p, { ctrl: null, events: [], layers: [], result: null, error: null }); renderPanel(p); }
  renderCalls(); renderBill();
}

async function boot() {
  if (Q.get("theme")) document.documentElement.dataset.theme = Q.get("theme");
  if (FILM) document.body.classList.add("film");
  const r = await (await fetch("/api/models")).json();
  S.models = r.models; S.live = r.live;
  const avail = S.models.filter((m) => m.available), tasks = [...new Set(avail.map((m) => m.title))];
  const want = avail.find((m) => m.name === Q.get("model"));
  $("#task").innerHTML = tasks.map((t) => `<option>${t}</option>`).join("");
  S.task = want ? want.title : tasks[0]; $("#task").value = S.task;
  if (want) S.trained = want.trained;
  if (["exact", "mock", "replay", "live"].includes(Q.get("source"))) S.source = Q.get("source");
  if (["one", "training", "neuron"].includes(Q.get("compare"))) S.compare = Q.get("compare");
  setSeg("#source", S.source); setSeg("#compare", S.compare); setSeg("#trained", S.trained);
  $("#task").addEventListener("change", (e) => { S.task = e.target.value; configure().then(() => S.input && autorun() && runAll()); });
  bindSeg("#compare", "compare"); bindSeg("#trained", "trained"); bindSeg("#source", "source");
  $("#clear").addEventListener("click", clear);
  $("#real").addEventListener("click", () => loadReal());
  $("#run").addEventListener("click", runAll);
  renderInput(); await configure();
  window.jevrons = { play, draw, clear, run: runAll, loadDigit: loadReal, strokes: Object.keys(STROKES) };
  const delay = Number(Q.get("delay") ?? 900);
  if (Q.has("digit")) setTimeout(() => loadReal(Number(Q.get("digit"))), delay);
  else if (Q.has("stroke")) setTimeout(() => play(Q.get("stroke")), delay);
}
boot();
