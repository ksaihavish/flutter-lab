# Memory

*The project's running state, so a new chat or AI tool can continue without re-reading the
whole codebase. **Update this file at the end of every working session.***

Last updated: **2026-09-29** · Version **0.3.0** · Phase: 4 in progress (4.2 LQG ✅, 4.4 statistics ✅, local LLM ✅); **next = the LLM's job (RQ2 redesign)**

---

## 1. Start here (60-second orientation)

- **What it is:** a flutter-suppression research platform. It has a validated typical-section
  model (2-DOF, or 3-DOF with a trailing-edge flap), LQR, LQG, a physics-trained neural policy, a
  rule-based or LLM supervisor (local Ollama model by default, or Claude), a benchmark runner, and a web simulator.
- **Repo:** https://github.com/ksaihavish/flutter-lab (renamed 2026-09-29 from Hybrid-PINN-LLM-...; old URL redirects).
- **Validated rig:** `bact` preset = NASA BACT wind-tunnel model (TP-1998-206270). `python -m flutter_ai validate`.
- **Default preset:** `uav_flap`: 3-DOF, 0.3 m chord, 20 %-chord flap, 30 Hz servo (damping 0.56), ±15° command,
  1-step (5 ms) loop delay.
- **Run:** `start.bat` (web app) · `python -m pytest` (82 tests, ~2 min) ·
  `python -m flutter_ai evaluate --preset uav_flap` (20 seeds, ~2 min).
- **Location:** `C:\Users\lio48\dev\flutter-lab` (moved out of OneDrive on 2026-09-29; the old OneDrive copy was
  deleted; a zip backup of the *original* project remains in `OneDrive\Desktop\RESEARCH PAPER PROJECT BACKUP`).
- **Environment:** on Windows the venv is at `%LOCALAPPDATA%\flutter-ai\venv` (outside the project).
  Python 3.14, torch 2.14 CPU. The CLI and tests use that interpreter.
- **Read next:** `Rules.md` (integrity rules), then `Phases.md` Phase 4.
- **Owner's learning path:** `Study-Guide.md` (8 tracks mapped to the code, exercises, 16-week plan).

## 2. Current state of each part

| Part | State | Notes |
|---|---|---|
| Physics (2-DOF + 3-DOF flap) | ✅ validated | See §3 |
| Discrete, delay-aware LQR | ✅ | Designed for the loop delay (augmented state). Optimal for initial conditions, but amplifies gusts (no disturbance model) |
| LQG (turbulence model) | ✅ 2026-09-29 | `controllers/lqg.py`: LQR on plant + AR(1) gust state, steady-state Kalman filter for aero-lag states and gust. **Best controller on both UAV wings** |
| Neural policy `uav_flap` | ✅ **retrained 2026-09-29** (servo damping 0.05 → 0.56) | 64×64 tanh MLP; trained for delays 0–2, speeds 0.8–1.3 U_F, turbulence ≤ 1.5 m/s RMS; ~5 min |
| Neural policy `uav_section` | ✅ trained 2026-09-28 | Speeds 0.8–1.4 U_F, turbulence ≤ 2 m/s |
| Rule supervisor | ✅ | Works; **adds no measurable value** (see §4) |
| LLM supervisor | ✅ two backends, one safety wrapper | `FLUTTER_AI_LLM_BACKEND=ollama` (default, local `llama3.1:8b`) or `claude`. Local tested live (§6); Claude only with mocks |
| Benchmark + figures | ✅ | `outputs/<preset>/results.json`, fig1–fig9. 20 seeds; deterministic gusts randomised per seed; cost `J`; paired Wilcoxon + bootstrap CIs |
| Web app | ✅ | Vanilla JS; flap drawing; delay input; per-preset policies |
| Dataset intake | ✅ format + `check-data` | No time-history dataset yet; spec in `data/README.md` |
| BACT preset + validation | ✅ 2026-09-29 | Flutter freq 3.99 Hz vs ~4 Hz measured (−0.2 %); flutter q 183 vs ~148 psf (+24 %); NASA's model 150.8 psf. Neural policy `neural_policy_bact.npz` trained |
| Git/GitHub | ✅ | PR #1 (rebuild) merged 2026-09-28; repo description/topics updated. PR #2 adds CI/pre-commit/Docker |
| Automation | ✅ | GitHub Actions CI (lint, tests Linux+Windows, benchmark artifact, Docker build), pre-commit, Dependabot |

## 3. Validation evidence (all automated in `tests/test_physics.py`)

- Theodorsen C(k) matches tabulated values; the Jones approximation is within 0.02 of it.
- Hodges & Pierce benchmark: divergence = √8·bω_α (exact); flutter = 2.170·bω_α (p-k: 2.184).
- State-space vs independent p-k (exact Theodorsen): −0.6 % (H&P), −0.75 % (2-DOF), −0.99 % (3-DOF).
- 3-DOF flap aerodynamics: steady lift, moment and hinge moment match **independent thin-airfoil
  theory** (Glauert series) to 2e-4 relative, for three geometries.
- Flap apparent-mass matrix is symmetric positive definite; a stiff hinge recovers the 2-DOF flutter speed.
- Delay-margin analysis agrees with the simulator (a test checks this).

## 4. Latest results (generated from `outputs/*/results.json`, 2026-09-29T05:00Z, 20 seeds)

Cells: RMS plunge h in mm (mean ± std over seeds) · mean cost J (the objective every controller minimises;
lower is better; omitted when any run failed) · failures. Last column: neural relative to LQG over the seeds
where both survived (positive = neural worse), with the larger of the two Wilcoxon p-values.

**uav_flap** (generated 2026-09-29T05:00Z, 20 seeds). Flutter 46.40 m/s @ 10.43 Hz.

| Scenario | Open loop | LQR | LQG | Neural | Neural vs LQG (RMS h, J) |
|---|---|---|---|---|---|
| Turbulence, 1.1 U_F | 25.18 ± 0.00 · **20/20 fail** | 7.10 ± 0.81 · **4/20 fail** | 3.71 ± 0.93 · **2/20 fail** | 6.10 ± 0.91 · **3/20 fail** | +72 % / +27 %, p ≤ 2e-05 |
| Discrete gust, 1.1 U_F | 25.18 ± 2.08 · **20/20 fail** | 4.37 ± 0.73 · J 2.25 | 1.56 ± 0.24 · J 1.25 | 3.38 ± 0.58 · J 1.52 | +117 % / +21 %, p ≤ 2e-06 |
| Gust, 1.3 U_F | 30.15 ± 4.17 · **20/20 fail** | 9.40 ± 1.81 · **19/20 fail** | 3.33 ± 0.55 · J 2.53 | 3.83 ± 2.29 · **2/20 fail** | -4 % / +48 %, p ≤ 1e-02 |
| Gust, 0.9 U_F | 2.41 ± 0.29 · J 0.98 | 3.98 ± 0.57 · J 1.76 | 1.58 ± 0.22 · J 0.55 | 2.45 ± 0.38 · J 0.76 | +55 % / +38 %, p ≤ 2e-06 |

Delay tolerance (ms survived, discrete gust): lqr 30; lqg 30; neural 10; lqr_nominal 15; lqg_nominal 10
Envelope (max U/U_F survived with gust): open_loop 1.00; lqr 1.25; lqg 1.50; neural 1.30
Supervisor (turbulence, seed 7): lqr+rules 6.81 mm @ effort 2.43; neural+rules 6.21 mm @ effort 1.94 | fixed neural a=0: 6.04 @ 1.60; a=0.5: 6.14 @ 1.87

**uav_section** (generated 2026-09-29T05:00Z, 20 seeds). Flutter 49.22 m/s @ 10.91 Hz.

| Scenario | Open loop | LQR | LQG | Neural | Neural vs LQG (RMS h, J) |
|---|---|---|---|---|---|
| Turbulence, 1.1 U_F | 18.45 ± 0.00 · **20/20 fail** | 8.11 ± 0.78 · J 14.23 | 3.95 ± 0.42 · J 5.04 | 6.36 ± 0.66 · J 8.42 | +61 % / +67 %, p ≤ 2e-06 |
| Discrete gust, 1.1 U_F | 21.40 ± 2.69 · **20/20 fail** | 4.45 ± 0.68 · J 2.54 | 1.99 ± 0.32 · J 0.64 | 3.42 ± 0.55 · J 1.36 | +72 % / +113 %, p ≤ 2e-06 |
| Gust, 1.3 U_F | 32.20 ± 4.87 · **20/20 fail** | 6.86 ± 1.97 · **1/20 fail** | 2.52 ± 0.65 · J 0.96 | 5.02 ± 0.85 · J 2.78 | +99 % / +189 %, p ≤ 2e-06 |
| Gust, 0.9 U_F | 2.56 ± 0.35 · J 0.98 | 3.10 ± 0.47 · J 1.38 | 1.64 ± 0.23 · J 0.44 | 2.43 ± 0.37 · J 0.70 | +48 % / +59 %, p ≤ 2e-06 |

Delay tolerance (ms survived, discrete gust): lqr 30; lqg 30; neural 15; lqr_nominal 15; lqg_nominal 15
Envelope (max U/U_F survived with gust): open_loop 1.00; lqr 1.35; lqg 1.50; neural 1.50
Supervisor (turbulence, seed 7): lqr+rules 8.35 mm @ effort 3.79; neural+rules 6.74 mm @ effort 2.03 | fixed neural a=0: 7.45 @ 1.35; a=0.5: 6.90 @ 1.85

**bact** (generated 2026-09-29T05:01Z, 20 seeds). Flutter 135.67 m/s @ 3.99 Hz.

| Scenario | Open loop | LQR | LQG | Neural | Neural vs LQG (RMS h, J) |
|---|---|---|---|---|---|
| Turbulence, 1.1 U_F | 18.57 ± 0.00 · **20/20 fail** | 0.71 ± 0.15 · J 1.48 | 0.69 ± 0.15 · J 1.44 | 0.54 ± 0.13 · J 1.81 | -22 % / +26 %, p ≤ 3e-05 |
| Discrete gust, 1.1 U_F | 18.38 ± 0.96 · **20/20 fail** | 0.49 ± 0.09 · J 0.48 | 0.48 ± 0.09 · J 0.48 | 0.40 ± 0.07 · J 0.59 | -17 % / +23 %, p ≤ 2e-06 |
| Gust, 1.3 U_F | 18.55 ± 1.36 · **20/20 fail** | 0.65 ± 0.12 · J 1.02 | 0.65 ± 0.12 · J 1.02 | 0.43 ± 0.07 · J 1.42 | -34 % / +38 %, p ≤ 1e-03 |
| Gust, 0.9 U_F | 1.79 ± 0.27 · J 2.66 | 0.47 ± 0.07 · J 0.36 | 0.42 ± 0.08 · J 0.35 | 0.38 ± 0.06 · J 0.39 | -9 % / +12 %, p ≤ 4e-05 |

Delay tolerance (ms survived, discrete gust): lqr 30; lqg 30; neural 30; lqr_nominal 30; lqg_nominal 30
Envelope (max U/U_F survived with gust): open_loop 1.00; lqr 1.50; lqg 1.50; neural 1.50
Supervisor (turbulence, seed 7): lqr+rules 1.20 mm @ effort 0.77; neural+rules 0.97 mm @ effort 1.21 | fixed neural a=0: 1.41 @ 0.49; a=0.5: 0.96 @ 1.24


**What the numbers say**
- **RQ1 is answered *no* against a fair baseline.** LQG beats the neural policy on J in every scenario on all
  three wings (by 12–189 %). Neural beats plain LQR only because LQR has no disturbance model. On BACT, neural has
  lower RMS h but pays with more flap effort; at equal effort (trade-off sweep) it ties LQG.
- Why: the neural policy is memoryless (state estimate + last command) and cannot infer the gust; LQG's Kalman
  filter estimates it (correlation with the true gust 0.92) and feeds it forward.
- Stress tests (scratch script, uav_flap turbulence): LQG stays best with sensor noise up to 10 % of the reference
  scales (3.56 → 4.03 mm vs neural 6.14 → 6.29) and under model mismatch (f_alpha +10 %, f_h −10 %, mass +15 %,
  x_alpha +25 %); every controller fails for f_alpha −10 % and for a 30 % slower servo.
- **Supervisor (RQ2): still no value.** Rule-supervised points sit on the fixed-setting curve; for neural,
  fixed "efficient" beats supervision. Paired effects are ≤ ±4 % on the UAV wings (significant but tiny) and
  +9 to +25 % *worse* on BACT.
- uav_flap turbulence at 1.1 U_F is at the edge of the flap's authority: with 20 seeds every controller fails some
  realisations (LQG 2, neural 3, LQR 4 of 20). From 1.2 U_F, 1.5 m/s turbulence defeats every controller.

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
| 2026-09-29 | Project moved to `C:\Users\lio48\dev\flutter-lab`; venv always in `%LOCALAPPDATA%` | OneDrive syncing conflicts with git and 1 GB venvs |
| 2026-09-29 | CI: ruff + pytest (Linux py3.11/3.13, Windows py3.13) + benchmark artifact + Docker health check; pre-commit; Dependabot | Every change is verified automatically |
| 2026-09-29 | Docker image serves only (no PyTorch); `FLUTTER_AI_HOME=/app` | Small image; training stays local |
| 2026-09-29 | Modified strip theory params (`cl_alpha`, `x_ac`, `flap_span_fraction`); defaults = Theodorsen | Needed to model a real finite wing (BACT, aspect ratio 2) from measured data |
| 2026-09-29 | BACT aerodynamic centre from Table 2 CM_alpha (17.5 % chord), not the text's "~30 %" | 30 % gives 283 psf (+91 %); Table 2 is what NASA's validated model used |
| 2026-09-29 | BACT scenarios: disturbances ×0.25; training range 0.8–1.2 U_F, turbulence ≤ 0.5 m/s | Weak 30 %-span flap; NASA demonstrated suppression to ~1.16 U_F; LQR fails only from the standard 2°/10 mm start |
| 2026-09-29 | Flutter sweep range adapts (6 → 30 → 150 b·ω_α) | BACT (mass ratio ~880) flutters at ~20 b·ω_α |
| 2026-09-29 | Repo renamed to `flutter-lab` | The method is no longer a PINN; name matches the project |
| 2026-09-29 | `uav_flap` servo damping 0.05 → 0.56 (BACT's measured actuator); neural policy retrained, benchmark re-run, data template regenerated | At 0.05 the real flap overshot to ~29° with a ±15° command, giving controllers unrealistic authority. Old policy is in git history |
| 2026-09-29 | **LQG baseline** = LQR on plant + AR(1) gust state (same τ = 0.05 s and RMS as the benchmark turbulence) + steady-state Kalman filter; plain LQR kept | Plain LQR is optimal for initial conditions (J 0.075 vs open loop 0.314) but triples the gust cost below flutter: a missing disturbance model, not a tuning error. Re-weighting Q would change the objective both controllers share, so the fair fix is giving LQR the turbulence knowledge the neural policy trains with |
| 2026-09-29 | Cost `J` (fixed effort weight a = 0.5) added to metrics; None for failed runs | It is the objective all controllers minimise; RMS h alone rewarded spending more flap |
| 2026-09-29 | 20 seeds by default; deterministic gusts randomised per seed (amplitude ×0.75–1.25, length 0.2–0.5 s, start 2.5–3.5 s, initial condition ×0.5–1.5 with random sign); paired Wilcoxon + bootstrap 95 % CI | Single hand-picked gust runs cannot support claims; seed 0 stays the nominal case for figures |
| 2026-09-29 | LLM backends: local Ollama (default, `llama3.1:8b`) or Claude, one safety wrapper; Rules.md §2 updated | User wants a local on-board LLM now, with Claude switchable. Ollama's native REST API via `urllib` (no new dependency) |
| 2026-09-29 | LLM answers slower than `FLUTTER_AI_LLM_TIMEOUT` (30 s) in total count as timeouts (fallback, latency = timeout); the local model is preloaded at reset with identical options | Socket timeouts only bound silences (a live call was recorded at 3989 s, likely laptop sleep); a preload with a different num_ctx made Ollama reload the model (14 s first answers) |
| 2026-09-29 | Supervisor asks one question at a time (ticks skipped while an answer is pending) | Old code assumed many parallel LLM calls and let stale answers overwrite newer ones |
| 2026-09-29 | `hodges_pierce` is validation-only (`VALIDATION_ONLY_PRESETS`): hidden from the simulator, 422 on /api/simulate | 0.1 Hz flutter at 2.2 m/s makes the gust scenarios meaningless |

## 6. Known issues and limitations (be honest about these in the paper)

- Gust entry uses the Wagner lag (not Küssner); the model is linear (no freeplay or LCO yet).
- LQR's and LQG's optimism: in benchmarks they are re-designed for the true delay; fixed-design variants
  (`lqr_nominal`, `lqg_nominal`) are in the delay sweep. LQG designed for 5 ms tolerates 10 ms, like neural.
- LQG and the neural policy both use a model identical to the simulator (except in the scratch mismatch check).
  A robustness study with model error (Phase 4.3) is still needed before claiming anything about real hardware.
- The neural policy has no memory or gust estimate. Giving it the Kalman estimate (neural + filter) is the obvious
  way to test whether learning can beat LQG (e.g. under saturation); only worth doing if RQ1 stays in the paper.
- Since the servo fix, the neural policy tolerates only 0–10 ms of delay on uav_flap (its training range).
- The neural policy's aggressiveness input barely moves its trade-off, and "aggressive" is worse than "efficient"
  on both axes, so an effort-mode supervisor has almost nothing to choose between. RQ2 needs a different action space.
- Local LLM (2026-09-29, Ollama `llama3.1:8b` Q4 on the RTX 4050, 6 GB, 19 % of the model on CPU): 72/72 valid schema
  answers, latency median 3.75 s / p95 5.7 s, cold load 16 s; agrees with the rules 56/72 and over-escalates to
  "emergency" at 30–50 % of limit. Its decisions arrive ~4 s late, after the gust: an LLM cannot react to gusts.
  `llama3.2:3b` (100 % GPU): median 2.4 s / p95 2.9 s, but agrees with the rules only 15/72 (says "aggressive" 63/72,
  even in calm air): faster, much worse judgement. Live web-app run (8B, turbulence): answers at 3.7–4.6 s, three per
  10 s run; the "aggressive" asked at 4.0 s landed at 8.5 s, after the turbulence ended at 7 s.
- The Claude backend is untested against the live API (only with mocks).
- The observer's aero-lag estimate lags about half a step; that is fine, but it is part of the loop.
- `pytest` takes ~2 min (flutter sweeps); make the slow ones optional if needed.
- Novelty (quick search 2026-09-29, not a full review): neural flutter control under input constraints with
  wind-tunnel tests exists (Tang et al., J. Vib. Control 2020); NASA designed and tunnel-tested classical, **LQG**
  and minimax flutter controllers on BACT (Mukhopadhyay, IFA 1999, NTRS 19990050943); "LLM as slow supervisor over a
  fast certified inner loop" is surveyed (Du & Chan, arXiv 2609.16599); LLM fault-tolerant reconfiguration exists
  for process plants (Vyas et al., arXiv 2606.31635). No LLM + flutter/aeroelastic work found. Not a PINN any more.
- Real data (search 2026-09-29): no openly downloadable flutter time histories found. BACT's test database and
  NASA's MATLAB simulation were distributed on request only (TM-97-206252). Usable: measured flutter points (in use),
  NASA's closed-loop boundary extension to the 200 psf tunnel limit (IFA 1999), measured tunnel turbulence spectra
  (TM-107734, not yet opened).
- BACT: after the turbulence patch the neural policy leaves a slowly decaying pitch oscillation (LQR settles
  faster), so neural's settling time reads ~10 s / "–" in the web app.

## 7. Next steps (in order)

1. **Decide the LLM's job (RQ2).** Effort modes cannot add value. Candidates: choosing among pre-verified inner-loop
   controllers, fault handling (degrading flap/servo, sensor faults, stiffness loss), and airspeed / envelope advice,
   evaluated against rules and an oracle, with the measured latency.
2. **Stability guarantee** for the supervised switched system (each mode stable + dwell time), the open problem the
   Du & Chan survey highlights.
3. Improve BACT accuracy (compressibility; second flutter point, air Mach 0.38) and use TM-107734 for tunnel turbulence.
4. Phase 4.3 robustness (model mismatch) and 4.5 ablations; decide whether RQ1 stays (neural + Kalman filter).
5. Write the paper from `research/paper-outline.md`.

## 8. How to update this file

At the end of a session: bump "Last updated", change §2 states, paste new numbers into §4
**from results.json only**, add rows to §5 for any decision (with *why*), and edit §6/§7.
