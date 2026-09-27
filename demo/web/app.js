"use strict";
// Jevrons demo. No framework: state lives in `S`, every render reads from it.

const $ = (s, el = document) => el.querySelector(s);
const USD_PER_TOKEN = 0.042 / 1e6;
const SOURCE_NAME = { exact: "exact arithmetic", mock: "mock Jev", replay: "recorded Jev", live: "live Jev" };
const SOURCE_HELP = {
  exact: "A local step neuron: fires iff the true sum is positive. Free, instant.",
  mock: "A local stand-in fitted to Jev's stage 1 error curve, with fake latency. Free.",
  replay: "Answers recorded from earlier live runs of this exact state. Unrecorded neurons fall back to the mock. Free.",
  live: "Real Jev calls, journaled to demo/runs/journal.jsonl. About 33 calls and $0.002 per network.",
};

const S = {
  models: [], live: false, weights: {},
  input: null, real: null,           // processed 784 input; {index,label} when it is an unmodified test digit
  compare: "training", trained: "jev", source: "mock", task: null,
  panels: [], t: Infinity, session: { calls: 0, usd: 0 },
  sel: null,                          // {panel, layer, j}
  ink: "A",                           // pad effect variant
};

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const hid = (j) => "h" + String(j + 1).padStart(2, "0");

// ---------- drawing pad and MNIST preprocessing ----------

// Strokes live on an offscreen full-resolution layer (`ink`), which is also what preprocessing reads, so the
// visible pad is free to animate. The layer is only ever drawn, never redrawn, so it is the same raster as before.
const pad = $("#pad"), pctx = pad.getContext("2d");
const N = pad.width, CELL = N / 28;
const ink = document.createElement("canvas"); ink.width = ink.height = N;
const ictx = ink.getContext("2d", { willReadFrequently: true });
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const clock = () => window.__inkT ?? performance.now();  // tests pin time here to capture mid-animation frames
const QUANT_MS = 350, SETTLE_MS = 300, POP_MS = 120;
let drawing = false, last = null, penTimer = null, frame = null, live = null;

function padPoint(e) {
  const r = pad.getBoundingClientRect();
  return [(e.clientX - r.left) * N / r.width, (e.clientY - r.top) * N / r.height];
}
pad.addEventListener("pointerdown", (e) => {
  pad.setPointerCapture(e.pointerId);
  clearTimeout(penTimer); cancelAnimationFrame(frame);
  if (S.real) { clearPad(); S.real = null; }
  if (S.ink === "B" && !live) live = { prev: softGrid(cellMeans(alpha(), CELL)), at: new Float64Array(784) };
  drawing = true; last = padPoint(e); stroke(last, last); showDrawing();
});
pad.addEventListener("pointermove", (e) => { if (drawing) { const p = padPoint(e); stroke(last, p); last = p; showDrawing(); } });
const end = () => { if (!drawing) return; drawing = false; penTimer = setTimeout(penUp, 150); };
pad.addEventListener("pointerup", end);
pad.addEventListener("pointercancel", end);

function stroke(a, b) {
  ictx.strokeStyle = css("--ink"); ictx.lineWidth = 42; ictx.lineCap = ictx.lineJoin = "round";
  ictx.beginPath(); ictx.moveTo(...a); ictx.lineTo(...b); ictx.stroke();
}
function clearPad() {
  clearTimeout(penTimer); cancelAnimationFrame(frame); live = null;
  ictx.clearRect(0, 0, N, N); pctx.clearRect(0, 0, N, N); pad.classList.remove("gridded");
}

function penUp() {
  const a = alpha();
  S.input = preprocess(a, N, N);
  $("#pad-note").textContent = S.input ? "your drawing" : "draw a digit";
  renderInput();
  if (S.input && S.source !== "live") runAll();
  animateInk(a);
}

// ---------- ink display ----------

function alpha() {
  const d = ictx.getImageData(0, 0, N, N).data, a = new Uint8ClampedArray(N * N);
  for (let i = 0; i < a.length; i++) a[i] = d[i * 4 + 3];
  return a;
}
function cellMeans(a, b) {             // mean ink per b x b block, 0..1
  const n = N / b, out = new Float32Array(n * n);
  for (let y = 0; y < N; y++) { const row = (y / b | 0) * n; for (let x = 0; x < N; x++) out[row + (x / b | 0)] += a[y * N + x]; }
  return out.map((v) => v / (255 * b * b));
}
function softGrid(v) {                 // B's brush: a touched cell also lifts its four neighbors a little
  return v.map((x, k) => { const i = k % 28, j = k / 28 | 0;
    const nb = Math.max(i ? v[k - 1] : 0, i < 27 ? v[k + 1] : 0, j ? v[k - 28] : 0, j < 27 ? v[k + 28] : 0);
    return Math.max(x, 0.35 * nb); });
}
function cells(v, n, mul = 1) {
  const c = N / n; pctx.fillStyle = css("--ink");
  for (let k = 0; k < v.length; k++) if (v[k] > 0.004) { pctx.globalAlpha = Math.min(1, v[k] * mul); pctx.fillRect((k % n) * c, (k / n | 0) * c, c, c); }
  pctx.globalAlpha = 1;
}
function gridLines() {
  pctx.strokeStyle = css("--rule"); pctx.lineWidth = 1; pctx.beginPath();
  for (let i = 1; i < 28; i++) { const x = i * CELL + 0.5; pctx.moveTo(x, 0); pctx.lineTo(x, N); pctx.moveTo(0, x); pctx.lineTo(N, x); }
  pctx.stroke();
}
function smooth() { pctx.clearRect(0, 0, N, N); pctx.drawImage(ink, 0, 0); pad.classList.remove("gridded"); }
function finalGrid(v) {                // the exact 28x28 input sent to the model
  pctx.clearRect(0, 0, N, N); gridLines(); cells(v, 28); pad.classList.add("gridded"); S.shown = v;
}

function showDrawing() {
  if (S.ink !== "B" || !live) return smooth();
  cancelAnimationFrame(frame);
  const v = softGrid(cellMeans(alpha(), CELL)), t = clock(); let popping = false;
  pctx.clearRect(0, 0, N, N); gridLines(); pad.classList.add("gridded");
  const shown = v.map((x, k) => {
    if (live.prev[k] < 0.03 && x >= 0.03 && !live.at[k]) live.at[k] = t;
    const age = t - live.at[k];
    if (reduced.matches || !live.at[k] || age >= POP_MS) return x;
    popping = true; return x + (1 - x) * 0.8 * (1 - age / POP_MS);  // overshoot, then settle to the value
  });
  cells(shown, 28);
  if (popping) frame = requestAnimationFrame(() => { if (drawing || penTimer) showDrawing(); });
}

const easeOut = (u) => 1 - (1 - u) ** 3;
function animateInk(a) {
  cancelAnimationFrame(frame); penTimer = null;
  const fin = S.input, raw = S.ink === "B" && live ? softGrid(cellMeans(a, CELL)) : cellMeans(a, CELL);
  live = null;
  if (!fin) return smooth();
  if (reduced.matches) return finalGrid(fin);
  const g = fin.box, from = [g.x0, g.y0, g.w, g.h], to = [g.ox * CELL, g.oy * CELL, g.tw * CELL, g.th * CELL];
  const levels = [2, 4, 5, 7, 10, 14, 20].map((b) => [N / b, b === CELL ? raw : cellMeans(a, b)]);
  const order = Float32Array.from({ length: 784 }, () => Math.random() * QUANT_MS);
  const quant = S.ink === "B" ? 0 : QUANT_MS, t0 = clock();
  const step = () => {
    const t = clock() - t0;
    pctx.clearRect(0, 0, N, N);
    if (t < quant && S.ink === "A") {          // smooth -> 2px -> 4px ... -> 28x28 blocks, all at once
      pad.classList.remove("gridded");
      const [n, v] = levels[Math.min(levels.length - 1, Math.floor(t / quant * levels.length))]; cells(v, n);
    } else if (t < quant) {                    // C: each 28x28 cell snaps at its own random moment
      pad.classList.remove("gridded"); pctx.drawImage(ink, 0, 0);
      for (let k = 0; k < 784; k++) if (order[k] <= t && raw[k] > 0.004) pctx.clearRect((k % 28) * CELL, (k / 28 | 0) * CELL, CELL, CELL);
      cells(raw.map((v, k) => order[k] <= t ? v : 0), 28);
    } else if (t < quant + SETTLE_MS) {         // settle: the model's 28x28 slides and scales into place
      const u = (t - quant) / SETTLE_MS, k = easeOut(u), fade = Math.min(1, u / 0.2);
      pad.classList.add("gridded"); gridLines();
      if (fade < 1) cells(raw, 28, 1 - fade);
      const cur = from.map((f, i) => f + (to[i] - f) * k), sx = cur[2] / to[2], sy = cur[3] / to[3];
      pctx.save(); pctx.setTransform(sx, 0, 0, sy, cur[0] - to[0] * sx, cur[1] - to[1] * sy);
      cells(fin, 28, fade); pctx.restore();
    } else return finalGrid(fin);
    frame = requestAnimationFrame(step);
  };
  step();
}

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
  out.box = { x0, y0, w, h, ox, oy, tw, th };  // where the digit was drawn and where it lands, for the settle
  return out;
}

async function loadReal(index) {
  const labels = $("#real-label").value || S.panels[0].model.labels.join(",");
  const r = await (await fetch(Number.isInteger(index) ? `/api/digit?index=${index}` : `/api/digit?labels=${labels}`)).json();
  S.input = Float32Array.from(r.pixels); S.real = { index: r.index, label: r.label };
  clearPad(); finalGrid(S.input);
  $("#pad-note").textContent = `MNIST test #${r.index}, label ${r.label}`;
  renderInput();
  if (S.source !== "live") runAll();  // live runs only on an explicit click
}

// ---------- pixel maps ----------

function rgb(hex) {
  let h = hex.replace("#", ""); if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }

function paintGray(canvas, v) {           // intensity 0..1 as ink on paper
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(28, 28), ink = rgb(css("--ink")), paper = rgb(css("--paper"));
  for (let i = 0; i < 784; i++) for (let c = 0; c < 3; c++) img.data[i * 4 + c] = paper[c] + (ink[c] - paper[c]) * v[i];
  for (let i = 0; i < 784; i++) img.data[i * 4 + 3] = 255;
  ctx.putImageData(img, 0, 0);
}

// Signed map: positive (pushes the sum up) in ink, negative in mid gray. `faint` weights sit under products.
function paintSigned(canvas, vals, faint = null) {
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(28, 28);
  const ink = rgb(css("--ink")), neg = rgb(css("--mute")), paper = rgb(css("--paper"));
  const mx = Math.max(1e-6, ...vals.map(Math.abs)), fm = faint ? Math.max(1e-6, ...faint.map(Math.abs)) : 1;
  for (let i = 0; i < 784; i++) {
    let v = vals[i] / mx, a = Math.min(1, Math.abs(v) ** 0.7);
    if (faint && vals[i] === 0) { v = faint[i] / fm; a = Math.min(1, Math.abs(v)) * 0.18; }
    const col = v >= 0 ? ink : neg;
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = paper[c] + (col[c] - paper[c]) * a;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const products = (w) => S.input ? w.map((wi, i) => Math.round(S.input[i] * 100) / 100 === 0 ? 0 : S.input[i] * wi) : w.map(() => 0);

function renderInput() {
  paintGray($("#seen"), S.input || new Float32Array(784));
  $("#seen-note").textContent = S.real
    ? `MNIST test digit #${S.real.index}, label ${S.real.label}. Already 28×28, sent as is.`
    : "What the network sees: 28×28, digit fit into a 20×20 box and centered by mass, as in MNIST.";
  for (const p of S.panels) paintPanelMaps(p);
}

// ---------- configuration ----------

function pick(task, trained) { return S.models.find((m) => m.title === task && m.trained === trained && m.available); }

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
    configure().then(() => S.input && S.source !== "live" && runAll());
  });
}

async function configure() {
  const hasTwin = pick(S.task, "jev") && pick(S.task, "exact");
  $("#compare [data-v=training]").disabled = !hasTwin;
  if (!hasTwin && S.compare === "training") { S.compare = "one"; setSeg("#compare", "one"); }
  $("#trained-field").hidden = S.compare === "training" || !hasTwin;
  $("#source [data-v=live]").disabled = !S.live;
  const COMPARE_HELP = {
    training: "Left: trained through Jev. Right: same architecture and start, trained with exact arithmetic, then run on the same neurons.",
    neuron: "Same weights on both sides. Left: exact arithmetic. Right: Jev.", one: "" };
  $("#source-help").textContent = [COMPARE_HELP[S.compare], SOURCE_HELP[S.source],
    S.source !== "live" && !S.live ? "Live is off; start the server with --live." : ""].filter(Boolean).join(" ");
  const cfg = plan();
  $("#run").textContent = S.source === "live" ? `Run live · ${cfg.filter((c) => c.source === "live").length * (32 + (cfg[0].model.labels.length > 2 ? cfg[0].model.labels.length : 1))} calls` : "Run";
  await buildPanels(cfg);
}

// ---------- panels ----------

async function weightsFor(name) {
  if (!S.weights[name]) S.weights[name] = await (await fetch(`/api/weights?name=${name}`)).json();
  return S.weights[name];
}

function panelTitle(p) {
  const trained = p.model.trained === "jev" ? "trained through Jev" : "trained exact";
  const run = p.source === "exact" ? "run exact" : `run on ${SOURCE_NAME[p.source]}`;
  return `${trained}, ${run} · ${p.model.title}`;
}

async function buildPanels(cfg) {
  for (const p of S.panels) p.ctrl?.abort();
  const root = $("#panels"); root.replaceChildren();
  S.panels = cfg.map((c, k) => ({ id: k, ...c, events: [], result: null, error: null }));
  closeInspector();
  for (const p of S.panels) {
    const el = $("#panel-tpl").content.firstElementChild.cloneNode(true);
    p.el = el; root.append(el);
    $(".p-title", el).textContent = panelTitle(p);
    $(".p-source", el).textContent = p.model.labels.length > 2 ? "784-32-10" : "784-32-1";
    $(".p-note", el).textContent = p.model.note;
    p.w = await weightsFor(p.model.name);
    const grid = $(".hidden-grid", el);
    p.tiles = p.w.W1.map((_, j) => {
      const t = document.createElement("button");
      t.className = "tile"; t.innerHTML = `<canvas width="28" height="28"></canvas><span class="act"><i></i></span><span class="lbl">${hid(j)}</span>`;
      t.addEventListener("click", () => select(p, 0, j));
      grid.append(t); return t;
    });
    const outs = $(".outs", el), labels = p.model.labels, binary = labels.length === 2;
    outs.classList.toggle("binary", binary);
    p.outs = (binary ? ["1"] : labels).map((lab, j) => {
      const o = document.createElement("button");
      o.className = "out"; o.innerHTML = `<span class="node">${binary ? labels[1] : lab}</span><span class="v"></span>`;
      o.addEventListener("click", () => select(p, 1, j));
      outs.append(o); return o;
    });
    if (binary) outs.insertAdjacentHTML("beforeend", `<p class="mono small mute bin-cap">fires: ${labels[1]}<br>silent: ${labels[0]}</p>`);
    paintPanelMaps(p);
    new ResizeObserver(() => drawWires(p)).observe(el);
    renderPanel(p);
  }
  renderTimeline(); renderBill();
}

function paintPanelMaps(p) {
  if (!p.tiles) return;
  paintGray($(".in-img", p.el), S.input || new Float32Array(784));
  p.tiles.forEach((t, j) => paintSigned($("canvas", t), products(p.w.W1[j]), p.w.W1[j]));
}

function drawWires(p) {
  const svg = $(".wires", p.el), box = $(".net", p.el).getBoundingClientRect();
  const c = (el, side) => { const r = el.getBoundingClientRect(); return [(r.left + r.right) / 2 - box.left, (side === "b" ? r.bottom : r.top) - box.top]; };
  const img = $(".in-img", p.el), fired = p.acts || [];
  let html = "";
  if (img.offsetParent) {
    const [ix, iy] = c(img, "b");
    p.tiles.forEach((t) => { const [x, y] = c($("canvas", t), "t"); html += `<line x1="${ix}" y1="${iy}" x2="${x}" y2="${y}" stroke-opacity="0.07"/>`; });
  }
  const W2 = p.w.W2, mx = Math.max(...W2.flat().map(Math.abs));
  p.outs.forEach((o, k) => {
    const [ox, oy] = c($(".node", o), "t");
    p.tiles.forEach((t, j) => {
      const w = W2[j][k], [x, y] = c($(".act", t), "b");
      const on = fired[j] >= 0.5, base = Math.abs(w) / mx;
      const op = p.acts ? (on ? 0.1 + 0.6 * base : 0.03) : 0.04 + 0.2 * base;
      html += `<line class="${w < 0 ? "neg" : ""}" x1="${x}" y1="${y}" x2="${ox}" y2="${oy}" stroke-opacity="${op.toFixed(3)}" stroke-width="${(0.5 + 1.5 * base).toFixed(2)}"/>`;
    });
  });
  svg.innerHTML = html;
}

function visible(p) { return p.events.filter((e) => e.t_ms <= S.t); }

function renderPanel(p) {
  if (!p.tiles) return;
  const ev = visible(p), running = !!p.ctrl && !p.result;
  const byKey = new Map(ev.map((e) => [`${e.layer}:${e.j}`, e]));
  p.tiles.forEach((t, j) => {
    const e = byKey.get(`0:${j}`);
    t.className = "tile" + (e ? " done" : (running || p.events.length) ? " pending" : "") + (e?.disagree ? " dis" : "") + (isSel(p, 0, j) ? " sel" : "");
    t.style.setProperty("--p", e ? e.p : 0);
    $(".lbl", t).textContent = e ? e.p.toFixed(2) : hid(j);
    t.title = e ? `${hid(j)}: true sum ${fmt(e.z)}, Jev p ${e.p.toFixed(2)}${e.disagree ? " (disagrees)" : ""}` : hid(j);
  });
  const outEv = p.outs.map((_, k) => byKey.get(`1:${k}`));
  const done = outEv.every(Boolean) && ev.length === p.events.length && p.result;
  const winner = done ? p.result.prediction : null;
  p.outs.forEach((o, k) => {
    const e = outEv[k], lab = p.model.labels.length === 2 ? p.model.labels[1] : p.model.labels[k];
    o.className = "out" + (e ? "" : " pending") + (e && e.p >= 0.5 ? " lit" : "") + (e?.disagree ? " dis" : "") +
      (winner === lab && p.model.labels.length > 2 ? " top1" : "") + (isSel(p, 1, k) ? " sel" : "");
    o.style.setProperty("--p", e ? e.p : 0);
    $(".v", o).textContent = e ? `p ${e.p.toFixed(2)}` : "";
  });
  const hiddenEv = ev.filter((e) => e.layer === 0);
  p.acts = hiddenEv.length === p.tiles.length ? p.tiles.map((_, j) => byKey.get(`0:${j}`).p) : null;
  drawWires(p);

  const pred = $(".pred", p.el);
  pred.classList.toggle("pending", !done);
  $(".pred-big", p.el).textContent = done ? p.result.prediction : p.events.length || running ? "…" : "·";
  $(".pred-cap", p.el).textContent = p.error || (done && p.result.no_fire ? "no output fired; tie broken at random" : "prediction");
  const dh = hiddenEv.filter((e) => e.disagree).length, oe = ev.filter((e) => e.layer === 1), dO = oe.filter((e) => e.disagree).length;
  $(".p-stats", p.el).innerHTML = !ev.length ? (p.events.length || running ? "waiting for Jev" : "no run yet") :
    `hidden <b>${hiddenEv.length}/${p.tiles.length}</b> answered, <span class="${dh ? "dis" : ""}">${dh} disagree</span> with the true sum` +
    (oe.length ? `<br>output <b>${oe.length}/${p.outs.length}</b> answered, <span class="${dO ? "dis" : ""}">${dO} disagree</span>` : "");
}

// ---------- running ----------

let runSeq = 0;
async function runAll() {
  if (!S.input || !S.panels.length) return;
  const seed = Math.floor(Math.random() * 2 ** 31), seq = ++runSeq;
  S.t = Infinity; $("#tl-range").value = 1000;
  await Promise.all(S.panels.map((p) => runPanel(p, seed, seq)));
}

async function runPanel(p, seed, seq) {
  p.ctrl?.abort(); p.ctrl = new AbortController();
  p.events = []; p.result = null; p.error = null; p.acts = null;
  renderPanel(p); renderBill();
  try {
    const res = await fetch("/api/run", { method: "POST", signal: p.ctrl.signal, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: p.model.name, source: p.source, pixels: Array.from(S.input, (v) => Math.round(v * 1000) / 1000), seed }) });
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
    if (e.name !== "AbortError") { p.error = String(e.message || e); renderPanel(p); }
  }
  if (seq === runSeq) p.ctrl = null;
  renderPanel(p);
}

function onEvent(p, e) {
  if (e.type === "neuron") {
    p.events.push(e);
    if (p.source === "live") { S.session.calls++; S.session.usd += e.tokens * USD_PER_TOKEN; }
    if (S.sel && S.sel.panel === p && S.sel.layer === e.layer && S.sel.j === e.j) renderInspector();
  } else if (e.type === "result") p.result = e;
  else if (e.type === "error") p.error = e.message;
  renderPanel(p); renderTimeline(); renderBill();
}

function renderBill() {
  const ev = S.panels.filter((p) => p.source !== "exact").flatMap((p) => p.events), tokens = ev.reduce((a, e) => a + e.tokens, 0);
  const live = S.panels.some((p) => p.source === "live"), anyJev = S.panels.some((p) => p.source !== "exact");
  const elapsed = Math.max(0, ...S.panels.map((p) => p.result?.elapsed_ms ?? (p.events.at(-1)?.t_ms || 0)));
  $("#b-calls").textContent = ev.length.toLocaleString();
  $("#b-tokens").textContent = tokens.toLocaleString();
  $("#b-usd").textContent = "$" + (tokens * USD_PER_TOKEN).toFixed(6);
  $("#b-time").textContent = (elapsed / 1000).toFixed(2) + " s";
  $("#bill-kind").textContent = !ev.length ? "no run yet" : live ? "billed by Jev" : anyJev ? "simulated, not billed" : "local, free";
  $("#b-session").textContent = S.session.calls ? `This session: ${S.session.calls} live calls, $${S.session.usd.toFixed(6)}.` : "";
}

// ---------- timeline ----------

function renderTimeline() {
  const all = S.panels.flatMap((p) => p.events);
  $("#timeline").hidden = !all.length;
  if (!all.length) return;
  const svg = $("#tl-svg"), w = svg.clientWidth || 800, h = 64, end = Math.max(1, ...all.map((e) => e.t_ms));
  const lanes = S.panels.length, lh = h / lanes;
  let html = "";
  S.panels.forEach((p, k) => {
    if (lanes > 1) html += `<text class="lane" x="2" y="${k * lh + 10}">${p.model.trained === "jev" && S.compare === "training" ? "through Jev" : S.compare === "training" ? "exact" : SOURCE_NAME[p.source]}</text>`;
    for (const e of p.events) {
      const x = 2 + (w - 4) * e.t_ms / end, y0 = k * lh + (e.layer ? 3 : lh * 0.35), y1 = (k + 1) * lh - 3;
      html += `<line class="tick${e.disagree ? " dis" : ""}" x1="${x}" x2="${x}" y1="${y0}" y2="${y1}" stroke-opacity="${e.t_ms <= S.t ? 1 : 0.2}"/>`;
    }
  });
  if (S.t < Infinity) { const x = 2 + (w - 4) * Math.min(1, S.t / end); html += `<line class="head" x1="${x}" x2="${x}" y1="0" y2="${h}"/>`; }
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.innerHTML = html;
  $("#tl-end").textContent = (end / 1000).toFixed(2) + " s";
  $("#tl-t").textContent = S.t < Infinity ? `showing ${(S.t / 1000).toFixed(2)} s` : "drag to replay how it resolved";
}

function timelineEnd() { return Math.max(1, ...S.panels.flatMap((p) => p.events.map((e) => e.t_ms))); }
function seek(t) {
  S.t = t >= timelineEnd() ? Infinity : t;
  S.panels.forEach(renderPanel); renderTimeline();
  if (S.sel) renderInspector();
}
$("#tl-range").addEventListener("input", (e) => seek(timelineEnd() * e.target.value / 1000));
let anim = null;
$("#replay-anim").addEventListener("click", () => {
  cancelAnimationFrame(anim);
  const end = timelineEnd(), dur = Math.max(1500, end), t0 = performance.now();
  const step = (now) => {
    const f = Math.min(1, (now - t0) / dur);
    $("#tl-range").value = Math.round(f * 1000); seek(f * end);
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

function renderInspector() {
  const { panel: p, layer, j } = S.sel;
  const e = visible(p).find((x) => x.layer === layer && x.j === j);
  const binary = p.model.labels.length === 2;
  const name = layer === 0 ? `hidden ${hid(j)}` : `output ${binary ? p.model.labels[1] : p.model.labels[j]}`;
  $("#inspector").hidden = false;
  $("#i-title").textContent = `${name} · ${panelTitle(p)}`;
  const body = $("#i-body");
  if (!e) {
    body.innerHTML = `<p class="i-verdict">${p.events.length ? "Waiting for Jev." : "Not asked yet."}</p>` +
      (layer === 0 ? mapsHTML() : "") + `<p class="cap">Run the network to see this neuron's state and answer.</p>`;
    if (layer === 0) paintMaps(p, j);
    return;
  }
  const fires = e.p >= 0.5, pos = e.z > 0;
  const verdict = e.disagree
    ? `Jev says it ${fires ? "fires" : "stays silent"}, but the sum is ${fmt(e.z)}.`
    : `Jev agrees: ${fires ? "fires" : "stays silent"}, sum ${fmt(e.z)}.`;
  const sw = e.terms.length > 1;
  const stateJSON = sw ? JSON.stringify({ products: e.terms }) : JSON.stringify({ z: e.terms[0] });
  body.innerHTML = `
    <p class="i-verdict ${e.disagree ? "dis" : ""}">${verdict}</p>
    ${layer === 0 ? mapsHTML() : ""}
    <dl class="facts">
      <dt>terms sent</dt><dd>${e.n_terms}</dd>
      <dt>true total</dt><dd>${fmt(e.z)}</dd>
      <dt>should fire</dt><dd>${pos ? "yes" : "no"}</dd>
      <dt>Jev's p</dt><dd>${e.p.toFixed(2)}</dd>
      <dt>latency</dt><dd>${e.latency_ms} ms</dd>
      <dt>backend</dt><dd>${e.backend}</dd>
      <dt>tokens</dt><dd>${e.tokens.toLocaleString()}</dd>
      <dt>cost</dt><dd>$${(e.tokens * USD_PER_TOKEN).toFixed(6)}${p.source === "live" ? "" : " (not billed)"}</dd>
    </dl>
    ${sw ? `<div><svg class="walk ${e.disagree ? "dis" : ""}" viewBox="0 0 400 120" preserveAspectRatio="none"></svg>
      <p class="cap">Running total across the ${e.terms.length} numbers, in the order Jev reads them. Jev only sees the list; it has to do this sum itself.</p></div>` : ""}
    <div><pre class="state">${stateJSON}</pre>
      <p class="cap">The exact state sent. Question: ${sw ? "“Add all the numbers in products together.” True if the final total is greater than zero." : "“Is the number z positive?”"}</p></div>`;
  if (layer === 0) paintMaps(p, j);
  if (sw) drawWalk($(".walk", body), e.terms);
}

const mapsHTML = () => `<div class="i-maps">
  <figure><canvas id="i-w" width="28" height="28"></canvas><figcaption>weights (ink +, gray −)</figcaption></figure>
  <figure><canvas id="i-x" width="28" height="28"></canvas><figcaption>products x·w sent to Jev</figcaption></figure></div>`;
function paintMaps(p, j) { paintSigned($("#i-w"), p.w.W1[j]); paintSigned($("#i-x"), products(p.w.W1[j])); }

function drawWalk(svg, terms) {
  let s = 0; const pts = [0, ...terms.map((v) => (s += v))];
  const lo = Math.min(0, ...pts), hi = Math.max(0, ...pts), span = hi - lo || 1;
  const X = (i) => 4 + 392 * i / (pts.length - 1), Y = (v) => 6 + 108 * (hi - v) / span;
  svg.innerHTML = `<line class="zero" x1="0" x2="400" y1="${Y(0)}" y2="${Y(0)}"/>` +
    `<polyline class="path" points="${pts.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")}"/>` +
    `<circle class="end" cx="${X(pts.length - 1)}" cy="${Y(pts.at(-1))}" r="3.5"/>`;
}

// ---------- boot ----------

async function boot() {
  const r = await (await fetch("/api/models")).json();
  S.models = r.models; S.live = r.live;
  const tasks = [...new Set(S.models.filter((m) => m.available).map((m) => m.title))];
  $("#task").innerHTML = tasks.map((t) => `<option>${t}</option>`).join("");
  S.task = tasks[0];
  $("#real-label").innerHTML = `<option value="">any in task</option>` + [...Array(10).keys()].map((d) => `<option>${d}</option>`).join("");
  $("#task").addEventListener("change", (e) => { S.task = e.target.value; configure().then(() => S.input && S.source !== "live" && runAll()); });
  bindSeg("#compare", "compare"); bindSeg("#trained", "trained"); bindSeg("#source", "source");
  $("#ink").addEventListener("click", (e) => {
    const v = e.target.dataset.v; if (!v) return;
    S.ink = v; setSeg("#ink", v);
    if (S.input && !S.real) animateInk(alpha());  // replay on the same drawing, so variants compare directly
  });
  $("#clear").addEventListener("click", () => { clearPad(); S.real = null; S.input = null; renderInput(); configure(); $("#pad-note").textContent = "draw a digit"; });
  $("#real").addEventListener("click", () => loadReal());
  $("#run").addEventListener("click", runAll);
  renderInput(); await configure();
}
boot();
