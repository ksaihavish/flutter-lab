# Memory

*The project's running state, so a new chat or AI tool can continue without re-reading the
whole codebase. **Update this file at the end of every working session.***

Last updated: **2026-09-28** · Version **0.3.0** · Phase: 3-DOF upgrade done; **next = Phase 4 (paper)**

---

## 1. Start here (60-second orientation)

- **What it is:** a flutter-suppression research platform. It has a validated typical-section
  model (2-DOF, or 3-DOF with a trailing-edge flap), LQR, a physics-trained neural policy, a
  rule-based or Claude supervisor, a benchmark runner, and a web simulator.
- **Default preset:** `uav_flap`: 3-DOF, 0.3 m chord, 20 %-chord flap, 30 Hz servo, ±15° command,
  1-step (5 ms) loop delay.
- **Run:** `start.bat` (web app) · `python -m pytest` (65 tests, ~70 s) ·
  `python -m flutter_ai evaluate --preset uav_flap` (~20 s).
- **Environment:** on Windows the venv is at `%LOCALAPPDATA%\flutter-ai\venv` (kept out of OneDrive).
  Python 3.14, torch 2.14 CPU. The CLI and tests use that interpreter.
- **Read next:** `Rules.md` (integrity rules), then `Phases.md` Phase 4.

## 2. Current state of each part

| Part | State | Notes |
|---|---|---|
| Physics (2-DOF + 3-DOF flap) | ✅ validated | See §3 |
| Discrete, delay-aware LQR | ✅ | Designed for the loop delay (augmented state) |
| Neural policy `uav_flap` | ✅ trained 2026-09-28 | 64×64 tanh MLP; trained for delays 0–2, speeds 0.8–1.3 U_F, turbulence ≤ 1.5 m/s RMS; ~5 min |
| Neural policy `uav_section` | ✅ trained 2026-09-28 | Speeds 0.8–1.4 U_F, turbulence ≤ 2 m/s |
| Rule supervisor | ✅ | Works; **adds no measurable value** (see §4) |
| Claude supervisor | ✅ code + tests (mocked) | **Never run against the real API yet.** Needs `ANTHROPIC_API_KEY` |
| Benchmark + figures | ✅ | `outputs/<preset>/results.json`, fig1–fig9 |
| Web app | ✅ | Vanilla JS; flap drawing; delay input; per-preset policies |
| Dataset intake | ✅ format + `check-data` | **No real dataset yet.** User offered to supply them; spec in `data/README.md` |
| Git/GitHub | 🔄 | Branch `v0.3-rebuild` pushed; **PR #1 open** (not merged). Local repo lives in OneDrive: move it to e.g. `C:\devlutter-ai` |

## 3. Validation evidence (all automated in `tests/test_physics.py`)

- Theodorsen C(k) matches tabulated values; the Jones approximation is within 0.02 of it.
- Hodges & Pierce benchmark: divergence = √8·bω_α (exact); flutter = 2.170·bω_α (p-k: 2.184).
- State-space vs independent p-k (exact Theodorsen): −0.6 % (H&P), −0.75 % (2-DOF), −0.99 % (3-DOF).
- 3-DOF flap aerodynamics: steady lift, moment and hinge moment match **independent thin-airfoil
  theory** (Glauert series) to 2e-4 relative, for three geometries.
- Flap apparent-mass matrix is symmetric positive definite; a stiff hinge recovers the 2-DOF flutter speed.
- Delay-margin analysis agrees with the simulator (a test checks this).

## 4. Latest results (copied from `outputs/*/results.json`, generated 2026-09-28T17:48Z)

**uav_flap (3-DOF), nominal 5 ms delay.** Flutter 46.50 m/s @ 10.65 Hz (p-k 46.96).

| Scenario | LQR RMS h (mm) | Neural RMS h (mm) | Notes |
|---|---|---|---|
| Turbulence patch, 1.1 U_F (5 seeds) | 6.22 ± 0.34 | **4.86 ± 0.28** | Neural also uses less effort (1.51 vs 1.74) |
| Discrete gust 4 m/s, 1.1 U_F | 3.87 | **2.74** | Effort 0.048 vs 0.109 |
| High-speed gust, 1.3 U_F | fail | fail | Beyond the flap's authority; stress test |
| Below flutter, 0.9 U_F | 3.87 (worse than open loop, 2.31) | 2.32 | LQR adds response below flutter: investigate |

- **Delay robustness** (discrete gust, 1.1 U_F), RMS in mm by delay of 0/5/10/15/20/30 ms:
  - neural 2.79 / 2.74 / 2.70 / 5.20 / fail / fail;
  - LQR fixed for 5 ms 3.92 / 3.87 / 3.71 / 4.18 / fail / fail;
  - LQR re-designed for the true delay 2.14 / 3.87 / 4.77 / 4.95 / 5.05 / 5.11.
- **Envelope** (velocity sweep with gust): open loop survives to 1.0 U_F, LQR to 1.20, neural to 1.25.
- **Supervisor:** the rule supervisor lands *on* the fixed-aggressiveness trade-off curve
  (turbulence: neural 4.86 vs neural+rules 4.94). **No evidence yet that supervision adds
  performance.** This is the key open question for RQ2 and the Claude experiments.

The 2-DOF `uav_section` results show the same pattern (neural better than LQR; supervisor neutral).

## 5. Decisions log (why things are the way they are)

| Date | Decision | Why |
|---|---|---|
| 2026-09-28 | Rebuilt everything; legacy moved to `legacy/` | Original results invalid (NaN network, masked NaNs, no-op LLM): `research/audit-2026-09.md` |
| 2026-09-28 | Theodorsen/Wagner (Jones) state space + exact ZOH | Validated, exact discretisation, fast |
| 2026-09-28 | LQR designed in **discrete time** | Continuous LQR chattered at 200 Hz |
| 2026-09-28 | Neural output `u_max·tanh(g(x,u_prev) − g(0,0))` | Removed a steady force offset at rest |
| 2026-09-28 | Control-rate penalty 1.0 | At 0 / 0.2 the policy chattered at the sampling rate (pole near z = −1) |
| 2026-09-28 | **1-step loop delay in the plant; neural trained on delays 0–2 with u_prev as an input** | The perfect-timing policy went unstable with +5 ms of delay |
| 2026-09-28 | Delay-margin analysis includes the **aero-lag observer** | Without it the analysis contradicted the simulator |
| 2026-09-28 | **3-DOF with flap servo** is the default; ±15° command | The user asked for ≥ 3 DOF; realistic actuation. ±10° could not handle 1.5 m/s turbulence |
| 2026-09-28 | Flap training distribution ≤ 1.3 U_F, turbulence ≤ 1.5 m/s | ~20 % of episodes were unrecoverable for any controller on the old distribution |
| 2026-09-28 | LLM model `claude-opus-5`, structured JSON, `fallbacks: "default"`, effort low, 40-call budget | Safety and cost; configurable via `FLUTTER_AI_LLM_MODEL` |
| 2026-09-28 | Frontend = vanilla JS, no build step | Zero-toolchain start; React migration is Phase 5.3 |

## 6. Known issues and limitations (be honest about these in the paper)

- Gust entry uses the Wagner lag (not Küssner); the model is linear (no freeplay or LCO yet).
- LQR's optimism: in benchmarks it is re-designed for the true delay; the fixed-design variant is in the delay sweep.
- The neural policy is evaluated on scenarios drawn from its own training distribution (turbulence
  statistics); an LQG with a turbulence model is the fair baseline (Phase 4.2).
- LQR increases the response below flutter on `uav_flap` (3.87 vs 2.31 mm open loop); its Q/R weights may need retuning.
- The Claude supervisor is untested against the live API (only with mocks).
- The observer's aero-lag estimate lags about half a step; that is fine, but it is part of the loop.
- `pytest` takes ~70 s (flutter sweeps); make the slow ones optional if needed.

## 7. Next steps (in order)

1. **Review and merge PR #1** on GitHub; update the repo description (the current one overclaims).
2. **Get one real dataset** (see `data/README.md`, ★★★ rows) → `check-data` → model error %.
3. Phase 4.1: run the **Claude supervisor** benchmark (needs an API key and a budget).
4. Phase 4.2: add the **LQG + turbulence model** baseline; re-tune LQR weights.
5. Phase 4.3–4.5: Monte Carlo robustness, statistics (20+ seeds), ablations.
6. Write the paper from `research/paper-outline.md`.

## 8. How to update this file

At the end of a session: bump "Last updated", change §2 states, paste new numbers into §4
**from results.json only**, add rows to §5 for any decision (with *why*), and edit §6/§7.
