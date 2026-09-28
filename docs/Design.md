# Design

*The visual system for the web app and the paper figures: colours, theme, type,
components and chart rules. Tokens live in `frontend/css/app.css` and are mirrored in
`src/flutter_ai/plots.py`.*

---

## 1. Principles

1. **Quiet chrome, loud data.** Neutral warm-gray surfaces and hairline borders; colour
   is reserved for data identity and status.
2. **Colour follows the entity.** Each controller has one colour everywhere: every chart,
   the animation, the legend, the table and the paper figures. A filter never repaints
   survivors.
3. **Honest geometry.** The wing animation uses one scale for chord and plunge (no
   exaggeration). Slow motion is labelled (0.25× by default).
4. **Never colour alone.** Status always comes with an icon and a label. Every chart has a
   legend (for ≥ 2 series) and a table view (the results table and CSV export).

## 2. Colour tokens

### Surfaces and ink

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `--page` | `#f9f9f7` | `#0d0d0d` | Page background, viewport |
| `--surface` | `#fcfcfb` | `#1a1a19` | Cards, top bar, controls |
| `--surface-2` | `#f0efec` | `#242422` | Chips, hover wash |
| `--ink` | `#0b0b0b` | `#ffffff` | Primary text, primary button |
| `--ink-2` | `#52514e` | `#c3c2b7` | Secondary text, labels |
| `--muted` | `#898781` | `#898781` | Axis ticks, hints |
| `--grid` | `#e1e0d9` | `#2c2c2a` | Gridlines (solid hairline, never dashed) |
| `--axis` | `#c3c2b7` | `#383835` | Baselines, zero lines |
| `--border` | `rgba(11,11,11,.10)` | `rgba(255,255,255,.10)` | Card and input borders |

### Controller identity (categorical, validated)

Validated with the dataviz palette checker. Adjacent colour-blind ΔE ≥ 9.1 (light) and
≥ 8.4 (dark); normal-vision ΔE ≥ 19. In light mode, aqua, yellow and pink are below 3:1
contrast, so the rule is: always show a legend and the results table.

| Case | Token | Light | Dark |
|------|-------|-------|------|
| Open loop (baseline, deliberately recessive) | `--c-open_loop` | `#898781` | `#898781` |
| LQR | `--c-lqr` | `#2a78d6` | `#3987e5` |
| Neural (physics-trained) | `--c-neural` | `#eb6834` | `#d95926` |
| LQR + rule supervisor | `--c-lqr-rules` | `#1baf7a` | `#199e70` |
| Neural + rule supervisor | `--c-neural-rules` | `#eda100` | `#c98500` |
| + Claude supervisor | `--c-llm` | `#e87ba4` | `#d55181` |

A new controller takes the **next** palette slot (violet `#4a3aa7` / `#9085e9`, then red
`#e34948` / `#e66767`). Re-run the validator for the new set. Never generate new hues.
At most one Claude case is shown at a time.

### Other roles

| Role | Token | Light | Dark |
|------|-------|-------|------|
| Structural mode 1 / 2 / 3 (flutter chart only) | `--c-mode1…3` | `#4a3aa7` / `#008300` / `#52514e` | `#9085e9` / `#008300` / `#c3c2b7` |
| Status good | `--good` | `#0ca30c` | `#0ca30c` |
| Status critical (limit exceeded, structural limit lines) | `--critical` | `#d03b3b` | `#d03b3b` |
| Supervisor mode meter (ordinal, efficient → emergency) | `--seq-1…4` | `#86b6ef → #0d366b` | `#184f95 → #86b6ef` |

Status colours are **reserved**: never use them for a data series.

## 3. Theme

- The default follows the OS (`prefers-color-scheme`). The top-bar toggle overrides it and
  remembers the choice in `localStorage` (a per-viewer convenience only).
- Dark values are **stepped for the dark surface**, not inverted. Both scopes
  (`@media` and `[data-theme="dark"]`) define them.
- Canvases read tokens with `getComputedStyle` at draw time, so a theme switch simply
  redraws.

## 4. Typography

- One family: the system UI sans (`system-ui, -apple-system, "Segoe UI", Roboto`). No
  display or serif faces.
- Sizes: 17 px app title; 13 px card titles (h3); **13 px uppercase, tracked +0.06em**
  for section labels (h2); 14 px body; 12 px hints and axis labels.
- `font-variant-numeric: tabular-nums` only where numbers align (tables, readouts, time
  display, axis ticks).

## 5. Layout and components

- Desktop: a 300 px sticky control panel plus a fluid content column (max 1600 px). Below
  900 px everything stacks, with 16 px gutters and no horizontal scroll.
- **Cards:** 10 px radius, 1 px hairline border, 14–16 px padding, no heavy shadows.
- **Buttons:** primary = ink background (not a series colour); ghost = hairline border;
  icon buttons are 32 × 32 px.
- **Chips** (top-bar status): pill, `--surface-2`, with a 12 px icon (check / circle /
  warning).
- **Scenario presets:** pill toggle buttons with `aria-pressed`.
- **Supervisor mode:** text label plus a 4-bar ordinal meter.

## 6. Chart rules (web and paper)

- Thin marks: 2 px lines (1.5 px for the open-loop baseline); solid hairline grid;
  recessive axes.
- **One y-axis per chart.** Quantities with different units go in separate charts (for
  example, gust and aggressiveness are separate).
- Hover: a vertical crosshair snaps to the nearest sample; the tooltip lists **every**
  series at that time, value first, label second, with a line key. All text is inserted
  with `textContent`.
- An animation playhead is shown on the time charts.
- Reference lines are labelled directly ("structural limit", "saturation",
  "U_F = 49.2 m/s").
- A failure is marked with a red ✕ at the end of the truncated series, plus a label.
- Loading keeps the previous render at 45 % opacity (no skeleton flash).

## 7. The wing animation

- One lane per case. The label column shows the colour key, name, mode (or "Fixed
  gains" / "No control") and live h/α readouts.
- NACA 0012 outline, drawn at true scale; a faint ghost shows the rest position; a spring
  runs from the mount to the elastic axis; the elastic axis is a dot.
- **3-DOF:** the flap is drawn rotated about its hinge by the *actual* β (solid). The
  *commanded* β_cmd is a thin outline, so the gap shows servo lag.
- **2-DOF:** the idealised plunge force is an arrow; the pitch moment is an arc (nose-up is
  clockwise on screen, with air flowing left to right).
- Gusts are drawn as a vertical arrow plus a label; flow streaks move at a speed that is
  capped for visibility.
- Failure: a critical wash over the lane, a red outline, and "⚠ Limit exceeded at t s".

## 8. Paper figures (`plots.py`)

- White background, 200 dpi, sans-serif 9 pt, left-aligned bold titles, top/right spines
  removed.
- The same entity colours as the app (light-mode values).
- Figure list: V-g/V-f with both flutter speeds annotated; time histories per scenario;
  the effort-vs-RMS trade-off (the key RQ2 figure); the velocity sweep; training curves.

## 9. Don'ts

Dual y-axes · rainbow colour maps · dashed gridlines · a value label on every point ·
status colours on data · colour as the only cue · exaggerated deflections without a label.
