# Jevrons demo: design checklist

The rules `demo/web/` follows. They're drawn from impeccable.style (its anti-pattern list and audit), shadcn/ui's
component anatomy and transitions.dev's motion tokens, cut down to what a one-bit paper-and-ink page needs. Every
value below is a CSS variable in `web/style.css`, so a number that isn't on this page shouldn't be in the CSS either.

## Palette

| token | light | dark | used for |
| --- | --- | --- | --- |
| `--paper` | `#f6f6f3` | `#0f0f0f` | page and window background |
| `--ink` | `#141414` | `#ededea` | text, borders, fills, primary button |
| `--ink-2` | `#4d4d4a` | `#b3b3ae` | secondary text (7.8:1 / 9.1:1) |
| `--mute` | `#65655f` | `#8c8c86` | captions and labels; the lightest *text* color (5.4:1 / 5.7:1, 4.8:1 on `--faint`) |
| `--mute-2` | `#8f8f8a` | `#75756f` | graphics only: negative weights, idle marks. Never text |
| `--rule` | `#d9d9d4` | `#2d2d2b` | hairlines, empty tracks, pending digit |
| `--faint` | `#e9e9e4` | `#1c1c1b` | hover fill, notice and code backgrounds |
| `--accent` | `#e0467a` | `#f2769c` | outlines, bars and wires where Jev disagrees with the arithmetic (3.6:1, passes 3:1 for graphics) |
| `--accent-text` | `#c92f64` | `#f2769c` | the same meaning in text, and paper-on-pink fills (4.8:1 / 7.2:1) |

- Pink means one thing: Jev's answer disagrees with the sum. It's not used for focus, hover, errors, live or brand.
- Errors and fallbacks are ink on `--faint`. Being calm is the point: a refusal isn't an alarm.
- No gradients, glows, blur, shadows or tinted neutrals. There's one elevation (the inspector), and a 1px ink border
  carries it.

## Type

- Display serif (`Instrument Serif`, 40px) for the "Jevrons" title only. Nothing else uses it.
- Sans (`Inter Tight`, then system) for prose: the intro, verdicts, the compare switch label. Mono (`JetBrains Mono`)
  for labels, data and controls. Both are self-hosted from `web/fonts/` (OFL, licences alongside): the variable latin
  cuts, with Google Fonts' unicode-range, so → ≈ ← still fall back to the system font as before.
- Scale: `--fs-cap` 11px (mono captions and badges), `--fs-ui` 13px (controls and data), `--fs-body` 15px (prose),
  `--fs-lead` 20px (the compare verdict line), the verdict digits (`--fs-v`, 96px, 56px phone) and the prediction
  digit (`--fs-digit`, 176px, 112px narrow, 96px phone). Nothing smaller than 11px except the 9px
  R/S badge letters.
- `font-variant-numeric: tabular-nums` on the body, so every number lines up.
- Probabilities (Jev's answers, neuron outputs, the 0 / .5 / 1 axis) drop the leading zero: ".27", "1.00". Sums, seconds and
  dollars keep it, since they can be negative or above 1. Prose keeps "0.5".
- Prose stops at 64ch. Line height 1.5 for prose, 1.4 for mono.

## Space and shape

- Spacing steps: 4, 8, 12, 16, 24, 32, 40 (`--s1` … `--s7`). Gaps between sibling controls are 8 or 16. Sections
  sit 24 apart and columns 40.
- Radius is 0 everywhere (`--radius`). The square one-bit windows are the look. Borders are 1px ink for windows and
  controls, 1px `--rule` for hairlines inside them.
- Control height is one token (`--h`, 36px). Small controls (`--h-sm`, 28px) are only used inside window bars and
  the calls header.

## Components

All controls are native elements (`button`, `select`, `input[type=range]`), with shadcn's state anatomy:

| state | treatment |
| --- | --- |
| rest | 1px ink border on paper; primary is ink filled |
| hover | `--faint` fill; primary goes to `--ink-2` |
| active | fill held, 1px down (`translateY(1px)`), no scale |
| focus-visible | 2px ink outline, 2px offset, on every focusable thing, the range and tiles included. Never pink |
| disabled | 40% opacity, `not-allowed` cursor, and a reason in text nearby |
| loading | the panel footer says what it's waiting on (`running · 41 / 74`) and the panel gets `aria-busy` |

- The network select is a native `<select>` (appearance off) with a drawn chevron. It has a visible label.
- The source toggle group is a `radiogroup`: one tab stop, arrow keys move and select, and a disabled option is
  skipped. It uses `aria-checked`, not colour alone.
- The theme switch is the byline's last item ("by Raj Joshi · GitHub · dark"), a `button` styled like the byline
  links. Its word is the theme a click switches to, and its `aria-label` says so ("Switch to dark theme"). Until the
  first click the page follows the system; picking the system's own theme clears the stored choice, so there's no
  visible "auto". The choice lives in `localStorage` (the page works without it), and `?theme=` in the URL wins. An
  inline script in `<head>` sets `data-theme` on `<html>` before the first paint, so nothing flashes. A change
  repaints the canvases, which read the colour tokens when they draw. Hidden in film mode with the byline.
- The compare switch is `role="switch"` with `aria-checked`. Its price is in the field label and tied to it by
  `aria-describedby`.
- Neuron tiles and output bars are buttons in two roving-tabindex grids (arrow keys, Home/End). Enter opens the
  inspector.
- The inspector is a non-modal `role="dialog"`. Focus stays on the tile that opened it, so the arrow keys keep
  exploring and the inspector follows. Escape or close shuts it and puts focus back on that tile. The request JSON
  has a Copy button that confirms for 1.2 s.
- Badges (R, S) are 9px mono on an ink square: a single letter, explained in the panel notice and the tile's tooltip.

## Header and "How it works"

- The header is one left-aligned block on the page's grid: the serif title, the byline, a 24px lead sentence (`--fs-hero`),
  and one quiet supporting line with the two links and the pink key. It stops at 62ch however wide the screen is.
- The detail and the math live at the bottom in "How it works", in the same frame: a serif heading over a 1px rule, mono
  labels, sans prose, and equations in native MathML on a `--faint` box that scrolls sideways on its own if it has to.
  If the browser can't lay out MathML, a plain-text version of each equation shows instead (`.no-mathml`).

## Compare

- Off by default. The switch says what it does ("Also run a network trained with perfect math"). No prices up front.
- The two networks have one name everywhere: "A · trained through Jev", "B · trained with perfect math" (verdict,
  panel titles, calls, inspector).
- The verdict leads: one plain line ("Both read 7." / "They disagree: A reads 4, B reads 5."), then each network's
  digit and disagreement count in a column above its panel. On a phone it's pinned while the panels scroll.
- The panels stay, but quieter: side by side from 901px, no wires, no big digit, smaller tiles.
- "Show a test digit they read differently" loads one of the recorded test digits the twins disagree on
  (`/api/disagree`), for networks where both twins were recorded.

## Motion

From transitions.dev's scale: `--dur-1` 120ms (hover, press, colour), `--dur-2` 200ms (enter, bar fills, drawer),
`--dur-3` 300ms (wires, the prediction settling). The easing is `--ease` `cubic-bezier(0.22, 1, 0.36, 1)` (smooth out)
for anything entering or changing value, and plain `ease-out` for exits. No bounce or elastic.

- Animate only `transform` and `opacity` (and colour). Nothing animates width, height or layout.
- The loops are the in-flight sweep on a tile or bar that's waiting on a call, which shows real work. There are no
  pulsing status dots.
- Exits are faster than enters, and the inspector closes instantly.
- `prefers-reduced-motion: reduce` turns every transition and animation into an instant change. The in-flight sweep
  becomes a static half-tone bar, so "waiting" still reads.

## Accessibility

- Text meets AA (4.5:1) in both themes, pink included (`--accent-text`). Non-text marks meet 3:1.
- Every control has a visible label or an `aria-label`. Status changes (the bill, the summary, the panel's progress) sit in
  `aria-live="polite"` regions.
- The drawing canvas is `role="img"`, and its label says what's on it ("empty", "your drawing", "test digit 3490,
  labelled 4"). Test digit is the keyboard route to input.
- The phone layout runs pad, controls, result, bill, calls. The pad takes the full width for a finger, uses
  `touch-action: none`, and nothing scrolls sideways.

## Anti-patterns avoided

The ones I checked for: pulsing dots, a glow or gradient anywhere, card-in-card nesting, status-chip soup (R/S are
the only badges), text below 11px, grey text on a coloured fill, a line length over 75ch, content hidden behind an
entrance animation, layout-shifting animation, em dashes in the copy, and vague buttons (each one names its action;
there's no Run button, since drawing is the action).
