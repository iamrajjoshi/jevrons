"use strict";
/* Jevrons demo. No framework: state lives in `S`, every render reads from it.

   URL flags (all optional, combine freely):
     ?film              presentation state for recording: no copy or controls, bigger pad, fixed seed
     &stroke=3          draw a stored stroke (0-9) after `delay` ms, then run; see STROKES
     &digit=4203        load MNIST test digit #4203 instead (with replay, use a recorded one:
                        GET /api/digit?recorded=s8a-jev picks one at random)
     &source=replay     live | exact (the two visitors see) | replay | mock (URL only; not in the UI).
                        Default: live when the server runs with --live, else exact; mock in film mode
     &model=s7-jev      any name in demo/models.json (default s8a-jev; s7-jev in film mode)
     &compare=training  one | training | neuron (neuron: URL only)
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
const SOURCE_NAME = { exact: "exact math", mock: "simulated Jev", replay: "recorded Jev", live: "live Jev" };
// what answered a neuron, as the visitor should read it
const answeredBy = (e) => e.backend === "simulated" ? "simulated answer" : e.backend.startsWith("recorded") ? "recorded answer"
  : e.backend === "local step" ? "exact math" : "live Jev";
const NATIVE = { exact: "exact math", mock: "simulated answer", replay: "recorded answer", live: "live Jev" };
const fallbackOf = (p, e) => answeredBy(e) !== NATIVE[p.source] ? answeredBy(e) : null;  // "recorded answer" | "simulated answer" | null
const MARK = { "recorded answer": "R", "simulated answer": "S" };

const S = {
  models: [], live: false, weights: {},
  input: null, real: null,            // processed 784 input; {index, label} when it is an unmodified test digit
  compare: "one", trained: "jev", source: "exact", task: null,
  panels: [], t: Infinity, sel: null, // t: scrub position in ms (Infinity = now); sel: {panel, layer, j}
};

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const hid = (j) => "h" + String(j + 1).padStart(2, "0");
// Activations and answers are probabilities in [0, 1], so they drop the leading zero (".27", "1.00"): one character
// less keeps ten outputs readable across a phone. Sums, seconds and dollars keep theirs (they can be negative or > 1).
const pr = (v) => { const s = v.toFixed(2); return s[0] === "0" ? s.slice(1) : s; };
const usd = (v) => "$" + v.toFixed(v < 0.01 ? 4 : 2);
const setState = (s) => { document.body.dataset.state = s; };

// ---------- drawing pad and MNIST preprocessing ----------

const pad = $("#pad"), pctx = pad.getContext("2d", { willReadFrequently: true });
const PEN = 42;  // pen width in pad px (the pad is 560 wide): about MNIST's stroke once scaled to 20 × 20
let drawing = false, last = null, lastMid = null, playing = false;

function padPoint(e) {
  const r = pad.getBoundingClientRect();
  return [(e.clientX - r.left) * pad.width / r.width, (e.clientY - r.top) * pad.height / r.height];
}
pad.addEventListener("pointerdown", (e) => {
  if (playing || e.button > 0) return;
  pad.setPointerCapture(e.pointerId);
  clearTimeout(pending); stopRuns();  // drawing again: drop the draw in flight, it reruns after the next pause
  if (S.real) { clearPad(); S.real = null; }
  drawing = true; last = lastMid = padPoint(e); stroke(last, last); $("#pad-empty").style.opacity = 0;
  setState("drawing");
});
// Pressure-free and smooth: every pointer sample (coalesced ones too) becomes a quadratic through the midpoints.
pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  for (const c of e.getCoalescedEvents?.() || [e]) {
    const p = padPoint(c), mid = [(last[0] + p[0]) / 2, (last[1] + p[1]) / 2];
    curve(lastMid, last, mid); last = p; lastMid = mid;
  }
});
// No Run button: a draw starts PAUSE ms after the last stroke ends, so a two-stroke 4 or 7 is one draw, not two.
const PAUSE = 800;
let pending = null;
const end = () => {
  if (!drawing) return;
  drawing = false; stroke(lastMid, last); updateInput(); setState("idle");
  clearTimeout(pending); pending = setTimeout(runAll, PAUSE);
};
pad.addEventListener("pointerup", end);
pad.addEventListener("pointercancel", end);
// Enter on the focused pad runs now, without waiting for the pause
pad.addEventListener("keydown", (e) => { if (e.key === "Enter" && S.input) { e.preventDefault(); clearTimeout(pending); runAll(); } });
function stopRuns() { for (const p of S.panels) if (p.ctrl) { p.ctrl.abort(); p.ctrl = null; } }

function pen() { pctx.strokeStyle = css("--ink"); pctx.lineWidth = PEN; pctx.lineCap = pctx.lineJoin = "round"; pctx.beginPath(); }
function stroke(a, b) { pen(); pctx.moveTo(...a); pctx.lineTo(...b); pctx.stroke(); }
function curve(a, c, b) { pen(); pctx.moveTo(...a); pctx.quadraticCurveTo(...c, ...b); pctx.stroke(); }
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
  clearTimeout(pending);
  return runAll();
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

async function play(name) { await draw(name); await runAll(); }

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

// A sparse neuron's receptive field: each of its connected pixels is a square (ink +, gray −), the rest is paper.
// With a digit in, the pixels it actually sends light up at full strength, the rest of the field recedes, and the
// digit shows as a faint ghost so you can see which dots it hits.
function paintField(canvas, w) {
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(28, 28);
  const ink = rgb(css("--ink")), neg = rgb(css("--mute")), paper = rgb(css("--paper")), ghost = rgb(css("--faint"));
  const pr = products(w), mw = Math.max(1e-6, ...w.map(Math.abs)), mp = Math.max(1e-6, ...pr.map(Math.abs));
  for (let i = 0; i < 784; i++) {
    let col = paper, a = 1;
    if (w[i] !== 0) {
      col = w[i] > 0 ? ink : neg;
      a = !S.input ? 0.3 + 0.7 * Math.abs(w[i] / mw) ** 0.6 : pr[i] !== 0 ? 0.45 + 0.55 * Math.abs(pr[i] / mp) ** 0.6 : 0.16;
    } else if (S.input && kept(S.input[i])) { col = ghost; a = 0.7 * S.input[i]; }
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = paper[c] + (col[c] - paper[c]) * a;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const kept = (x) => Math.round(x * 100) / 100 !== 0;  // the server drops inputs that round to zero
const products = (w) => S.input ? w.map((wi, i) => kept(S.input[i]) ? S.input[i] * wi : 0) : w.map(() => 0);
const sparse = (m) => m.format === "sparse-e";

function renderInput() {
  paintGray($("#seen"), S.input || new Float32Array(784));
  $("#pad-empty").style.opacity = S.input ? 0 : 1;
  $("#pad-note").textContent = S.real ? `MNIST test #${S.real.index}` : S.input ? "your drawing" : "28 × 28 after preprocessing";
  $("#seen-note").textContent = S.real
    ? `MNIST test #${S.real.index} · label ${S.real.label}` : "28 × 28 input";
  pad.setAttribute("aria-label", S.real ? `Drawing pad showing test digit ${S.real.index}, labeled ${S.real.label}`
    : S.input ? "Drawing pad with your drawing" : "Drawing pad, empty. Draw a digit with a mouse, pen or finger, or press Test digit.");
  $("#clear").disabled = !S.input;
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

// A radiogroup: one tab stop (the checked option, or the first enabled one), arrow keys move and select.
function setSeg(id, v) {
  const opts = [...$(id).children], on = opts.find((b) => b.dataset.v === v && !b.disabled) || opts.find((b) => !b.disabled);
  for (const b of opts) { b.setAttribute("aria-checked", String(b.dataset.v === v)); b.tabIndex = b === on ? 0 : -1; }
}
function bindSeg(id, key) {
  const pickV = (v) => { if (v === "url" || S[key] === v) return; S[key] = v; setSeg(id, v); configure().then(() => S.input && runAll()); };
  $(id).addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) pickV(b.dataset.v); });
  $(id).addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]; if (!step) return;
    e.preventDefault();
    const opts = [...$(id).children].filter((b) => !b.disabled), i = opts.indexOf(document.activeElement);
    const next = opts[(i + step + opts.length) % opts.length]; next.focus(); pickV(next.dataset.v);
  });
}

// Arrow keys (and Home/End) inside a group of buttons laid out `cols` wide; one tab stop per group.
function rove(items, cols, onFocus) {
  items.forEach((el, i) => {
    el.tabIndex = i ? -1 : 0;
    el.addEventListener("keydown", (e) => {
      const d = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols, Home: -i, End: items.length - 1 - i }[e.key];
      if (d === undefined) return;
      e.preventDefault();
      const n = items[Math.max(0, Math.min(items.length - 1, i + d))];
      items.forEach((x) => { x.tabIndex = x === n ? 0 : -1; }); n.focus(); onFocus(items.indexOf(n));
    });
  });
}

async function configure() {
  const hasTwin = pick(S.task, "jev") && pick(S.task, "exact");
  if (!hasTwin && S.compare === "training") S.compare = "one";
  $("#compare").disabled = !hasTwin;
  $("#compare").setAttribute("aria-checked", String(S.compare === "training"));
  $("#source [data-v=live]").disabled = !S.live;
  $("#source [data-v=live]").textContent = S.live ? "live Jev" : "live Jev · off";
  $("#source [data-v=live]").title = S.live ? "" : "This server was started without --live";
  const viaUrl = !["live", "exact"].includes(S.source), seg = $("#source");
  let u = $("[data-v=url]", seg);
  if (viaUrl && !u) {
    u = document.createElement("button"); u.type = "button"; u.dataset.v = "url"; u.setAttribute("role", "radio");
    u.title = "Set by ?source= in the URL"; seg.append(u);
  }
  if (!viaUrl && u) u.remove();
  if (u) u.textContent = `${SOURCE_NAME[S.source].split(" ")[0]} · URL`;  // "recorded · URL", "simulated · URL"
  setSeg("#source", viaUrl ? "url" : S.source);
  const cfg = plan(), m = cfg[0].model;
  // which recorded test digits this network and its twin read differently, for "Show a test digit they read differently"
  if (S.compare === "training" && !(m.name in diffs)) {
    diffs[m.name] = [];
    fetch(`/api/disagree?model=${m.name}`).then((r) => r.ok ? r.json() : { digits: [] })
      .then((r) => { diffs[m.name] = r.digits; renderSummary(); }).catch(() => {});
  }
  const labels = `<option value="">any</option>` + m.labels.map((d) => `<option>${d}</option>`).join("");
  if ($("#real-label").innerHTML !== labels) $("#real-label").innerHTML = labels;
  await buildPanels(cfg);
}

// ---------- panels ----------

async function weightsFor(name) {
  if (!S.weights[name]) S.weights[name] = await (await fetch(`/api/weights?name=${name}`)).json();
  return S.weights[name];
}

// In a comparison each network has one name everywhere (verdict, panel titles, calls, inspector): "A · trained
// through Jev", "B · trained with perfect math". The part that differs leads, so they're easy to tell apart.
const trainedName = (p) => p.model.trained === "jev" ? "trained through Jev" : "trained with perfect math";
const runName = (p) => p.source === "exact" ? "run exact" : "on " + SOURCE_NAME[p.source];
const panelKey = (p) => (S.panels.length > 1 ? "AB"[p.id] + " · " : "") + (S.compare === "neuron" ? SOURCE_NAME[p.source] : trainedName(p));
function panelTitle(p) {
  if (S.panels.length < 2) return [p.model.title, trainedName(p), runName(p)].join(" · ");
  return [panelKey(p), p.model.title, S.compare === "neuron" ? trainedName(p) : runName(p)].join(" · ");
}

async function buildPanels(cfg) {
  for (const p of S.panels) p.ctrl?.abort();
  const root = $("#panels");
  S.panels = cfg.map((c, k) => ({ id: k, ...c, events: [], layers: [], result: null, error: null, notice: null }));
  closeInspector();
  const els = [];
  for (const p of S.panels) {
    const el = $("#panel-tpl").content.firstElementChild.cloneNode(true);
    p.el = el; els.push(el);
    const binary = p.model.labels.length === 2;
    p.w = await weightsFor(p.model.name);
    $(".p-title", el).textContent = panelTitle(p);
    $(".p-arch", el).textContent = `784 → ${p.w.W1.length} → ${binary ? 1 : p.model.labels.length}` +
      (sparse(p.model) ? `, ${p.model.fields} pixels each` : "");
    el.classList.toggle("sparse", sparse(p.model));
    $(".p-note", el).textContent = p.model.note;
    const grid = $(".hidden-grid", el);
    p.tiles = p.w.W1.map((_, j) => {
      const t = document.createElement("button");
      t.className = "tile idle"; t.style.setProperty("--d", `${(-Math.random() * 1.1).toFixed(2)}s`);
      t.type = "button";
      t.innerHTML = `<canvas width="28" height="28" aria-hidden="true"></canvas><span class="act"><i></i></span><span class="lbl" aria-hidden="true">${hid(j)}</span>`;
      t.addEventListener("click", () => select(p, 0, j));
      grid.append(t); return t;
    });
    const outs = $(".outs", el);
    outs.classList.toggle("binary", binary);
    p.outs = (binary ? [p.model.labels[1]] : p.model.labels).map((lab, k) => {
      const o = document.createElement("button");
      o.type = "button"; o.className = "out"; o.style.setProperty("--d", `${(-Math.random() * 1.1).toFixed(2)}s`);
      o.innerHTML = `<span class="lab" aria-hidden="true">${lab}</span><span class="bar-track"><i></i><b></b></span><span class="v" aria-hidden="true"></span>`;
      o.title = `output ${lab} · tick at 0.5 · click to see its call`;
      o.addEventListener("click", () => select(p, 1, k));
      outs.append(o); return o;
    });
    // an axis under the bars, so the tick reads as 0.5
    const axis = document.createElement("div");
    axis.className = "out-axis"; axis.setAttribute("aria-hidden", "true");
    axis.innerHTML = `<span></span><span><i>0</i><i>.5</i><i>1</i></span><span></span>`;
    outs.append(axis);
    const follow = (layer) => (i) => { if (S.sel) select(p, layer, i); };  // with the inspector open, it follows the focus
    rove(p.tiles, 8, follow(0));
    rove(p.outs, 1, follow(1));
    paintTiles(p);
  }
  root.replaceChildren(...els);
  for (const p of S.panels) { new ResizeObserver(() => layoutWires(p)).observe(p.el); renderPanel(p); }
  renderCalls(); renderBill();
}

function paintTiles(p) {
  if (!p.tiles) return;
  p.tiles.forEach((t, j) => sparse(p.model) ? paintField($("canvas", t), p.w.W1[j])
    : paintSigned($("canvas", t), products(p.w.W1[j]), p.w.W1[j]));
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
  // Badges only mark the odd ones out: when every answer came the same way (all simulated, all recorded), the notice
  // and the footer say so once, and the tiles stay clean.
  const kinds = new Set(p.events.map((e) => fallbackOf(p, e))), mixed = kinds.size > 1;
  const badge = (e) => mixed && e && fallbackOf(p, e);
  p.tiles.forEach((t, j) => {
    const e = byKey.get(`0:${j}`), flight = !e && sent(p, 0) && !p.error;
    const fb = e && fallbackOf(p, e), mark = badge(e);
    t.className = "tile" + (e ? " done" : flight ? " flight" : " idle") + (e?.disagree ? " dis" : "") + (mark ? " fb" : "") +
      (isSel(p, 0, j) ? " sel" : "") + (t._land ? " land" : "");
    t.dataset.fb = mark ? MARK[mark] : "";
    t.title = `${hid(j)}${fb ? `: ${fb}, not live` : ""} · click to see its call`;
    t.style.setProperty("--p", e ? e.p : 0);
    $(".lbl", t).textContent = e ? pr(e.p) : hid(j);
    t.setAttribute("aria-label", neuronLabel(`hidden ${hid(j)}`, e, flight, fb));
  });
  const outEv = p.outs.map((_, k) => byKey.get(`1:${k}`));
  const done = !!p.result && ev.length === p.events.length;
  const top = done ? p.result.prediction : null;
  p.outs.forEach((o, k) => {
    const e = outEv[k], flight = !e && sent(p, 1) && !p.error;
    const fb = e && fallbackOf(p, e), mark = badge(e);
    o.className = "out" + (e ? " done" : flight ? " flight" : "") + (e?.disagree ? " dis" : "") + (mark ? " fb" : "") +
      (!binary && top === p.model.labels[k] ? " top1" : "") + (isSel(p, 1, k) ? " sel" : "");
    o.dataset.fb = mark ? MARK[mark] : "";
    o.style.setProperty("--p", e ? e.p : 0);
    $(".v", o).textContent = e ? pr(e.p) : "";
    o.setAttribute("aria-label", neuronLabel(`output ${binary ? p.model.labels[1] : p.model.labels[k]}`, e, flight, fb) +
      (!binary && top === p.model.labels[k] ? ", the prediction" : ""));
  });
  // wire strength: how much each answered hidden neuron pushes each output. Once the answer is in, the wires into
  // the predicted digit stay and the rest step back.
  const win = done && !binary ? p.model.labels.indexOf(top) : -1;
  if (p.wires) p.tiles.forEach((_, j) => {
    const e = byKey.get(`0:${j}`);
    p.outs.forEach((_, k) => {
      const w = p.w.W2[j][k], line = p.wires[j * p.outs.length + k];
      const op = e ? (0.03 + 0.45 * e.p * Math.abs(w) / p.wmax) * (win < 0 ? 1 : k === win ? 1.5 : 0.4) : 0;
      line.style.strokeOpacity = Math.min(1, op).toFixed(3);
    });
  });

  const pred = $(".pred", p.el), big = $(".pred-big", p.el);
  const hiddenEv = ev.filter((e) => e.layer === 0), outs = ev.filter((e) => e.layer === 1);
  const inFlight = (sent(p, 0) ? p.tiles.length - hiddenEv.length : 0) + (sent(p, 1) ? p.outs.length - outs.length : 0);
  pred.classList.toggle("pending", !done);
  const text = done ? p.result.prediction : "";
  if (big.textContent !== text) { big.textContent = text; big.classList.remove("in"); if (text) { void big.offsetWidth; big.classList.add("in"); } }
  $(".pred-cap", p.el).classList.toggle("err", !!p.error);
  $(".pred-cap", p.el).textContent = p.error ? "Stopped." :
    done ? (binary ? `p ${pr(p.result.outputs[0])}; ${p.model.labels[1]} at 0.5 or more, else ${p.model.labels[0]}` : "")
    : inFlight || S.input ? "" : "Waiting for a digit.";
  const dis = ev.filter((e) => e.disagree).length;
  const fbs = ev.map((e) => fallbackOf(p, e)), rec = fbs.filter((f) => f === "recorded answer").length,
    sim = fbs.filter((f) => f === "simulated answer").length;
  // progress lives here now that there's no Run button: "running · 41 / 74"
  const busy = !!p.ctrl && !p.result;
  p.el.setAttribute("aria-busy", String(busy));
  $(".p-stats", p.el).innerHTML = !ev.length && !inFlight ? (busy ? "running · starting" : "no calls yet") :
    (busy ? `running · ${ev.length} / ${p.tiles.length + p.outs.length} · ` : "") + `hidden ${hiddenEv.length}/${p.tiles.length} · output ${outs.length}/${p.outs.length} · ` +
    `<span class="${dis ? "dis" : ""}">${dis} disagree</span>` +
    (rec ? ` · <span class="fbk">R</span> ${rec} recorded` : "") + (sim ? ` · <span class="fbk">S</span> ${sim} simulated` : "") +
    "";
  const note = $(".p-notice", p.el);
  note.hidden = !p.error && !p.notice && !(p.source === "replay" && sim);
  const only = !mixed && [...kinds][0];  // "recorded answer" | "simulated answer" when every answer is one fallback
  const key = mixed ? ` Tiles marked <span class="fbk">R</span> show a recorded Jev answer to this exact request, <span class="fbk">S</span> a simulated one.`
    : only === "recorded answer" ? ` All ${p.events.length} answers are recorded.`
    : only === "simulated answer" ? ` All ${p.events.length} answers are simulated.` : "";
  note.innerHTML = p.error ? `Stopped: ${p.error} Draw again to retry.`
    : p.notice ? p.notice + key
    : only ? `Never recorded: all ${p.events.length} answers are simulated.`
    : `<span class="fbk">S</span> ${sim} ${sim === 1 ? "answer" : "answers"} simulated: never recorded.`;
  // the one-time hint that a neuron opens its call: first panel, once there are answers, until the inspector is used
  $(".p-hint", p.el).hidden = FILM || p.id > 0 || !ev.length || inspected();
  renderSummary();
}
const inspected = () => { try { return localStorage.getItem("jevrons:inspected") === "1"; } catch { return false; } };

// Plain words for the one question each call asks ("do these numbers add up to more than zero?"): yes when p ≥ 0.5.
const yes = (p) => p >= 0.5 ? "yes" : "no";
const neuronLabel = (name, e, flight, fb) => name + (!e ? (flight ? ", waiting on Jev" : "")
  : e.backend === "local step" ? `, outputs ${e.p >= 0.5 ? 1 : 0}`
  : `, Jev said ${yes(e.p)}, ${pr(e.p)}` + (e.disagree ? `, the sum says ${yes(e.z > 0 ? 1 : 0)}` : "") + (fb ? `, ${fb}` : ""));

// Two networks: the verdict leads, big and plain, one column per network above its panel.
let diffs = {};  // model name -> recorded test digits it and its twin read differently (from /api/disagree)
function renderSummary() {
  const el = $("#summary"), more = $("#summary-more"), on = S.panels.length > 1;
  el.hidden = more.hidden = !on;
  document.body.classList.toggle("cmp", on && !FILM);  // film keeps its own layout
  if (!on) return;
  const [a, b] = S.panels, done = a.result && b.result;
  const dis = (p) => p.events.filter((e) => e.disagree).length;
  const col = (p) => `<div class="v-col"><span class="v-tag">${panelKey(p)}</span>
    <span class="v-digit ${p.result ? "" : "pending"}">${p.result ? p.result.prediction : "·"}</span>
    <span class="v-dis">${p.events.length ? `<span class="${dis(p) ? "dis" : ""}">${dis(p)} <span class="v-of">of ${p.events.length} answers </span>disagree</span>` : p.error ? "stopped" : "not run"}</span></div>`;
  const line = !done ? (S.input ? "Both networks are reading the same drawing…" : "Draw a digit to compare the two.")
    : a.result.prediction === b.result.prediction ? `Both read ${a.result.prediction}.`
    : `They disagree: A reads ${a.result.prediction}, B reads ${b.result.prediction}.`;
  const d = diffs[a.model.name];
  const why = S.compare === "neuron" ? "Same weights: A on exact math, B on Jev."
    : `B was trained without Jev. On the test set, running on Jev: A ${a.model.on_jev}, B ${b.model.on_jev}.`;
  const html = `<p class="v-line">${line}</p>${col(a)}${col(b)}`;
  if (el.innerHTML !== html) el.innerHTML = html;
  const extra = `<p class="v-why">${why}</p>` + (d?.length && S.source !== "exact" ? `<button class="btn small" id="find-diff" type="button">Show a test digit they read differently</button>` : "");
  if (more.dataset.html !== extra) { more.innerHTML = extra; more.dataset.html = extra; }  // don't steal focus from the button mid-run
}
 $("#summary-more").addEventListener("click", (e) => {
  if (!e.target.closest("#find-diff")) return;
  const d = diffs[S.panels[0].model.name].filter((i) => i !== S.real?.index);
  loadReal(d[Math.floor(Math.random() * d.length)]);
});

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
  Object.assign(p, { events: [], layers: [], result: null, error: null, notice: null, t0: performance.now() });
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
    // a dropped connection reads as a TypeError; say what it means rather than "Failed to fetch"
    p.error = e instanceof TypeError && /fetch|network|load/i.test(e.message) ? "lost the connection to the demo server." : String(e.message || e).replace(/\.?$/, ".");
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
  else if (e.type === "notice") p.notice = e.message;
  renderPanel(p); renderBill();
}

function tick() {  // keeps in-flight bars and the clock moving while any panel waits
  renderCalls(); renderBill();
  if (S.panels.some((p) => p.ctrl && !p.result)) requestAnimationFrame(tick);
}

function renderBill() {
  const ps = S.panels.filter((p) => p.source !== "exact"), ev = ps.flatMap((p) => p.events);
  const tokens = ps.reduce((a, p) => a + p.events.reduce((b, e) => b + (p.source !== "live" || e.billed ? e.tokens : 0), 0), 0);
  const live = ps.some((p) => p.source === "live" && p.events.some((e) => e.billed));
  const elapsed = Math.max(0, ...S.panels.map((p) => p.result ? p.result.elapsed_ms : p.ctrl ? now(p) : p.events.at(-1)?.t_ms || 0));
  $("#b-calls").textContent = ev.length.toLocaleString();
  $("#b-tokens").textContent = tokens.toLocaleString();
  // only live answers are billed; recorded and simulated ones show their tokens but cost nothing
  const billed = ps.reduce((a, p) => a + p.events.reduce((b, e) => b + (e.billed ? e.tokens : 0), 0), 0);
  $("#b-usd").textContent = usd(billed * USD_PER_TOKEN);
  $("#b-time").textContent = (elapsed / 1000).toFixed(1) + " s";
  const fell = ps.some((p) => p.source === "live" && p.events.some((e) => !e.billed));
  $("#bill-kind").textContent = !S.panels.some((p) => p.events.length) ? "nothing run" : live ? (fell ? "live Jev, part recorded or simulated" : "live Jev")
    : ps.length ? (ps.some((p) => p.source === "live") ? "recorded or simulated, free"
      : ps.every((p) => p.source === "replay") ? "recorded, free" : "simulated, free") : "local, free";
}

// ---------- calls: one bar per call, from when it was sent to when Jev answered ----------

function renderCalls() {
  // exact answers are instant and local, so there's nothing to show on a timeline
  const ran = S.panels.filter((p) => p.layers.length && p.source !== "exact");
  $("#calls").hidden = !ran.length;
  if (!ran.length) return;
  const svg = $("#calls-svg"), w = svg.clientWidth || 800, rowH = FILM ? 4 : 3, gap = 1, layerGap = 6, labelH = ran.length > 1 ? 16 : 4;
  const end = Math.max(1000, ...ran.map((p) => p.result ? p.result.elapsed_ms : now(p)), ...ran.flatMap((p) => p.events.map((e) => e.t_ms)));
  const X = (t) => 1 + (w - 2) * Math.min(t, end) / end;
  let y = 0, html = "";
  for (const p of ran) {
    if (ran.length > 1) { html += `<text x="0" y="${y + 11}">${panelTitle(p)}</text>`; }
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
  // time axis: a gridline and label every `step` seconds, at most about 8 of them
  const step = [0.25, 0.5, 1, 2, 5, 10, 20, 30, 60].find((s) => end / 1000 / s <= 8) || 60;
  let grid = "";
  for (let s = 0; s * 1000 <= end + 1; s += step) {
    const x = X(s * 1000), anchor = s === 0 ? "start" : x > w - 24 ? "end" : "middle";
    grid += `<line class="grid" x1="${x}" x2="${x}" y1="0" y2="${y}"/><text x="${x}" y="${y + 13}" text-anchor="${anchor}">${+s.toFixed(2)} s</text>`;
  }
  svg.setAttribute("viewBox", `0 0 ${w} ${y + 16}`); svg.setAttribute("height", y + 16); svg.innerHTML = grid + html;
  $("#calls-end").textContent = `${(end / 1000).toFixed(2)} s in all`;
  $("#calls-t").textContent = S.t < Infinity ? `at ${(S.t / 1000).toFixed(2)} s` : ran.some((p) => p.ctrl) ? "calls in flight"
    : FILM ? "" : "drag to scrub, or focus and use ← →";
  $("#replay-anim").disabled = ran.some((p) => p.ctrl);
  $("#calls-range").setAttribute("aria-valuetext", S.t < Infinity ? `${(S.t / 1000).toFixed(2)} seconds` : "end of the run");
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
// Non-modal: focus stays on the tile, so arrow keys keep exploring and the inspector follows. Escape or close
// puts focus back on the tile that opened it.
let opener = null;
function select(p, layer, j) {
  if (!S.sel) opener = document.activeElement;
  try { localStorage.setItem("jevrons:inspected", "1"); } catch {}
  S.sel = { panel: p, layer, j }; S.panels.forEach(renderPanel); renderInspector();
}
function closeInspector(restore = false) {
  if (!S.sel) return;
  S.sel = null; $("#inspector").hidden = true; S.panels.forEach(renderPanel);
  if (restore && opener?.isConnected) opener.focus();
  opener = null;
}
$("#i-close").addEventListener("click", () => closeInspector(true));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && S.sel) { e.preventDefault(); closeInspector(true); } });

function answerRows(e) {
  if (e.backend === "local step") return "";
  return Object.entries(e.answers).map(([k, a]) => typeof a === "object"
    ? `<dt>above or below zero: above</dt><dd>${pr(a.above)}</dd>`
    : `<dt>more than zero? yes</dt><dd>${pr(a)}</dd>`).join("") +
    (Object.keys(e.answers).length > 1 ? `<dt>Jev's answer, the mean of both</dt><dd class="${e.disagree ? "dis" : ""}">${pr(e.p)}</dd>` : "");
}

function renderInspector() {
  const { panel: p, layer, j } = S.sel;
  const e = visible(p).find((x) => x.layer === layer && x.j === j);
  const binary = p.model.labels.length === 2;
  const name = layer === 0 ? `hidden ${hid(j)}` : `output ${binary ? p.model.labels[1] : p.model.labels[j]}`;
  $("#inspector").hidden = false;
  $("#i-title").textContent = name;
  const body = $("#i-body"), sub = `<p class="i-sub">${panelTitle(p)}</p>`;
  if (!e) {
    body.innerHTML = sub + `<p class="i-verdict">${sent(p, layer) ? "Asked. Waiting for Jev." : "Not asked yet."}</p>` +
      (layer === 0 ? mapsHTML(p) : "");
    if (layer === 0) paintMaps(p, j);
    return;
  }
  const jevish = e.backend !== "local step", sum = `The numbers add up to ${fmt(e.z)}`;
  const verdict = !jevish ? `${sum}, so this neuron outputs ${e.p >= 0.5 ? 1 : 0}.`
    : e.disagree ? `${sum}, so the answer should be ${e.z > 0 ? "yes" : "no"}. Jev said ${yes(e.p)} (${pr(e.p)}).`
    : `${sum}. Jev said ${yes(e.p)} (${pr(e.p)}), which is right.`;
  const sw = e.terms.length > 1;
  const request = `{"state": ${JSON.stringify(sw ? { products: e.terms } : { z: e.terms[0] }).replaceAll(",", ", ")},\n "questions": ` +
    JSON.stringify(e.questions, null, 2).replaceAll("\n", "\n ") + "}";
  body.innerHTML = sub + `
    <p class="i-verdict ${e.disagree ? "dis" : ""}">${verdict}</p>
    ${layer === 0 ? mapsHTML(p) : ""}
    <dl class="facts">
      <dt>numbers sent</dt><dd>${e.n_terms}</dd>
      <dt>their true sum</dt><dd>${fmt(e.z)}</dd>
      ${answerRows(e)}
      <dt>answered after</dt><dd>${(e.latency_ms / 1000).toFixed(2)} s</dd>
      <dt>answered by</dt><dd>${answeredBy(e)}${fallbackOf(p, e) && p.source === "live" ? ", not live" : ""}</dd>
      <dt>source</dt><dd>${e.backend}</dd>
      <dt>input tokens</dt><dd>${e.tokens.toLocaleString()}</dd>
      <dt>cost</dt><dd>${e.billed ? "$" + (e.tokens * USD_PER_TOKEN).toFixed(6) : "free"}</dd>
    </dl>
    ${sw ? `<div><svg class="walk ${e.disagree ? "dis" : ""}" viewBox="0 0 400 110" preserveAspectRatio="none"></svg>
      <p class="cap">Running total of the ${e.terms.length} numbers sent, in order.</p></div>` : ""}
    <div><div class="req-head"><p class="cap">The request, as sent</p><button class="btn small" id="i-copy" type="button">Copy</button></div>
      <pre class="state" id="i-req">${request}</pre></div>`;
  if (layer === 0) paintMaps(p, j);
  if (sw) drawWalk($(".walk", body), e.terms);
}

$("#i-body").addEventListener("click", async (ev) => {
  const b = ev.target.closest("#i-copy"); if (!b) return;
  try { await navigator.clipboard.writeText($("#i-req").textContent); b.textContent = "Copied"; b.classList.add("done"); }
  catch { b.textContent = "Couldn't copy"; }
  setTimeout(() => { if (b.isConnected) { b.textContent = "Copy"; b.classList.remove("done"); } }, 1200);
});

// Dense neurons: the weight map and pixels × weights. Sparse neurons: the receptive field as its tile draws it, and
// which of its pixels this draw actually sent (filled) against the ones it didn't (outlined), over the digit's ghost.
const mapsHTML = (p) => sparse(p.model) ? `<div class="i-maps">
  <figure><canvas id="i-w" width="28" height="28"></canvas><figcaption>its ${p.model.fields} pixels: ink +, gray −</figcaption></figure>
  <figure><canvas id="i-x" class="fine" width="560" height="560"></canvas><figcaption id="i-x-cap"></figcaption></figure></div>`
  : `<div class="i-maps">
  <figure><canvas id="i-w" width="28" height="28"></canvas><figcaption>weights: ink +, gray −</figcaption></figure>
  <figure><canvas id="i-x" width="28" height="28"></canvas><figcaption>pixels × weights, sent to Jev</figcaption></figure></div>`;
function paintMaps(p, j) {
  const w = p.w.W1[j];
  if (!sparse(p.model)) { paintSigned($("#i-w"), w); paintSigned($("#i-x"), products(w)); return; }
  paintField($("#i-w"), w);
  const n = paintSent($("#i-x"), w), total = w.filter((v) => v !== 0).length;
  $("#i-x-cap").textContent = S.input ? `sent: ${n} of ${total} (filled)` : "sent: none yet";
}
// the pixels a sparse call sends are exactly those whose input and weight are both nonzero at two decimals
function paintSent(canvas, w) {
  const ctx = canvas.getContext("2d"), c = canvas.width / 28, px = products(w);
  const mp = Math.max(1e-6, ...px.map(Math.abs)), ink = css("--ink"), neg = css("--mute-2"), ghost = css("--faint");
  ctx.globalAlpha = 1; ctx.fillStyle = css("--paper"); ctx.fillRect(0, 0, canvas.width, canvas.height);
  let n = 0;
  for (let i = 0; i < 784; i++) {
    const x = (i % 28) * c, y = Math.floor(i / 28) * c, inked = S.input && kept(S.input[i]);
    if (inked) { ctx.globalAlpha = 0.7 * S.input[i]; ctx.fillStyle = ghost; ctx.fillRect(x, y, c, c); }
    if (w[i] === 0) continue;
    const col = w[i] > 0 ? ink : neg;
    if (inked && kept(w[i])) {
      n++; ctx.globalAlpha = 0.45 + 0.55 * Math.abs(px[i] / mp) ** 0.6; ctx.fillStyle = col; ctx.fillRect(x + 1, y + 1, c - 2, c - 2);
    } else {
      ctx.globalAlpha = 1; ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.strokeRect(x + 2.5, y + 2.5, c - 5, c - 5);
    }
  }
  ctx.globalAlpha = 1;
  return n;
}

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
  for (const p of S.panels) { p.ctrl?.abort(); Object.assign(p, { ctrl: null, events: [], layers: [], result: null, error: null, notice: null }); renderPanel(p); }
  renderCalls(); renderBill();
}

async function boot() {
  { const m = document.createElementNS("http://www.w3.org/1998/Math/MathML", "math"), sp = document.createElementNS(m.namespaceURI, "mspace");
    sp.setAttribute("width", "40px"); m.append(sp); m.style.position = "absolute"; document.body.append(m);
    if (Math.abs(m.getBoundingClientRect().width - 40) > 2) document.documentElement.classList.add("no-mathml"); m.remove(); }
  if (Q.get("theme")) document.documentElement.dataset.theme = Q.get("theme");
  if (FILM) document.body.classList.add("film");
  const r = await (await fetch("/api/models")).json();
  S.models = r.models; S.live = r.live;
  const avail = S.models.filter((m) => m.available), tasks = [...new Set(avail.map((m) => m.title))];
  const want = avail.find((m) => m.name === (Q.get("model") ?? (FILM ? "s7-jev" : "s8a-jev")));
  $("#task").innerHTML = tasks.map((t) => `<option>${t}</option>`).join("");
  S.task = want ? want.title : tasks[0]; $("#task").value = S.task;
  if (want) S.trained = want.trained;
  S.source = ["exact", "mock", "replay", "live"].includes(Q.get("source")) ? Q.get("source") : FILM ? "mock" : S.live ? "live" : "exact";
  if (["one", "training", "neuron"].includes(Q.get("compare"))) S.compare = Q.get("compare");
  const rerun = () => configure().then(() => S.input && runAll());
  $("#task").addEventListener("change", (e) => { S.task = e.target.value; rerun(); });
  $("#compare").addEventListener("click", () => { S.compare = S.compare === "training" ? "one" : "training"; rerun(); });
  bindSeg("#source", "source");
  $("#clear").addEventListener("click", clear);
  $("#real").addEventListener("click", () => loadReal());
  renderInput(); await configure();
  window.jevrons = { play, draw, clear, run: runAll, loadDigit: loadReal, strokes: Object.keys(STROKES) };
  const delay = Number(Q.get("delay") ?? 900);
  if (Q.has("digit")) setTimeout(() => loadReal(Number(Q.get("digit"))), delay);
  else if (Q.has("stroke")) setTimeout(() => play(Q.get("stroke")), delay);
}
boot();
