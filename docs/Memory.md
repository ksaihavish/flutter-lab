# Memory

*The project's running state, so a new chat or AI tool can continue without re-reading the
whole codebase. **Update this file at the end of every working session.***

Last updated: **2026-09-30 (evening)** · Version **0.3.0** · Phase: 4 done (fair baselines, realistic actuator, robustness, ablations, LLM benchmark, identification, fault-tolerant supervision, paper generator) + physics advisor, pre-registered physics test, Guardian supervisor (final test done), LLM explainer / post-flight reviewer; **experiments finished; next = the owner's paper items** (`research/submission-checklist.md`)

---

## 1. Start here (60-second orientation)

- **What it is:** a flutter-suppression research platform. It has a validated typical-section
  model (2-DOF, or 3-DOF with a trailing-edge flap), LQR, LQG, a physics-trained neural policy, a
  rule-based or LLM supervisor (local Ollama model by default, or Claude), a benchmark runner, and a web simulator.
- **Repo:** https://github.com/ksaihavish/flutter-lab (renamed 2026-09-29 from Hybrid-PINN-LLM-...; old URL redirects).
- **Validated rig:** `bact` preset = NASA BACT wind-tunnel model (TP-1998-206270). `python -m flutter_ai validate`.
- **Default preset:** `uav_flap`: 3-DOF, 0.3 m chord, 20 %-chord flap, 30 Hz servo (damping 0.56), ±15° command,
  545 °/s slew-rate limit, 1-step (5 ms) loop delay; benchmark disturbances ×0.33.
- **Run:** `start.bat` (web app) · `python -m pytest` (178 tests, ~2 min) ·
  `python -m flutter_ai evaluate --preset uav_flap` (20 seeds, ~2 min).
- **Location:** `C:\Users\lio48\dev\flutter-lab` (moved out of OneDrive on 2026-09-29; the old OneDrive copy was
  deleted; a zip backup of the *original* project remains in `OneDrive\Desktop\RESEARCH PAPER PROJECT BACKUP`).
- **Environment:** on Windows the venv is at `%LOCALAPPDATA%\flutter-ai\venv` (outside the project).
  Python 3.14, torch 2.14 CPU. The CLI and tests use that interpreter.
- **Read next:** `Rules.md` (integrity rules), then `Phases.md` Phase 4.
- **Owner's learning path:** `Study-Guide.md` (8 tracks mapped to the code, exercises, 16-week plan).
- **Complete project guide** (theory, every component, what makes the software special, results traced to files,
  novelty and limits, talk outline, demo script, reviewer Q&A): Claude Doc
  https://claude.ai/code/artifact/deb247ab-a97f-462c-91f5-814deac16edc (2026-09-30).
- **The course** (lessons teaching every topic in Study-Guide.md's tracks, plus Track 9 on fault tolerance; each lesson:
  idea, maths, where it is in the code with real numbers, a check question with answer): Claude Doc
  https://claude.ai/code/artifact/3e3770fd-cff5-4e44-9659-3f16cf23ba0b (2026-10-01).
- **Housekeeping 2026-09-30:** all agent worktrees and `flutter-lab-exp` removed; branches
  `worktree-agent-a20cf8ee09288bbd4` and `worktree-agent-a689a0e68a75a3835` kept (unmerged WIP drafts, superseded by
  commit 1c59694). `physics-advisor` pushed to GitHub as **PR #9** (supersedes PR #8, closed 2026-09-30).

## 2. Current state of each part

| Part | State | Notes |
|---|---|---|
| Physics (2-DOF + 3-DOF flap) | ✅ validated | See §3 |
| Discrete, delay-aware LQR | ✅ | Designed for the loop delay (augmented state). Optimal for initial conditions, but amplifies gusts (no disturbance model) |
| LQG (turbulence model) | ✅ 2026-09-29 | `controllers/lqg.py`: LQR on plant + AR(1) gust state + shared rate penalty; Kalman filter for aero-lag states and gust, with the actuator's known slew limit in its input model. Knobs tuned per wing (`tune`). Best on J in most scenarios |
| Neural policy `uav_flap` | ✅ **retrained 2026-09-29** (servo damping 0.56, 545 °/s slew limit, disturbances ×0.33) | 64×64 tanh MLP; delays 0–2, speeds 0.8–1.3 U_F, turbulence ≤ 0.5 m/s RMS; ~40 min on a busy CPU. Ablation variants in `models/ablations/` |
| Neural policy `uav_section` | ✅ trained 2026-09-28 | Speeds 0.8–1.4 U_F, turbulence ≤ 2 m/s |
| Rule supervisor | ✅ | Works; **adds no measurable value** (see §4) |
| Fault-tolerant supervision | ✅ 2026-09-29 | `fault_tolerance/`: faults, diagnostics, bank, rules/random/oracle/LLM, `ft-benchmark`, `ft-stability`. Results in §4 (`outputs/uav_flap/ft_benchmark.json`, fig12) |
| Physics-informed advisor | ✅ 2026-09-30 | `research/physics-advisor-plan.md`: `fault_tolerance/margin.py` (v1 flutter-margin estimator: pitch stiffness + flap effectiveness, flutter speed, per-mode predicted stability; `tests/test_margin.py`). **v2** (authority, turbulence/noise, measured delay, robust margins; `rules+physics_v2`; `margin-eval` on dev seeds → `margin_eval.json`, fig15; `tests/test_margin_v2.py`). **Pre-registered test done** (seeds 100-119, snapshot 3a9d325; §4) |
| Guardian supervisor | ✅ 2026-09-30 | `GuardianSupervisor` (learned diagnosis + physics v2 admissibility + symptom-rule fallback, no LLM). Dev `guardian_dev1000`, **final test once on seeds 300-319** (`ft_benchmark_guardian_test300.json`, fig12_..._guardian_test300); §4 |
| LLM off the control path | ✅ built 2026-09-30 | Explainer (`explainer-eval` → `explainer_eval.json`) and post-flight reviewer + verification gate (`review-loop` → `review_loop.json`, fig17); §4 |
| Paper prep | ✅ 2026-09-30 | `python -m flutter_ai paper` → `research/paper-draft.md` (full draft; numbers and comparisons computed; figures copied to `research/figures/`); `research/references.bib` (read status per entry); `paper-outline.md`, `submission-checklist.md`; PRD research questions with status. §7.5-§7.8 (FT study, pre-registered physics test, Guardian, LLM explainer/reviewer), the abstract's and conclusion's later sentences, §8 and §10 generated from the results; the owner edits the wording and writes the closing sentence |
| LLM supervisor | ✅ two backends, one safety wrapper | `FLUTTER_AI_LLM_BACKEND=ollama` (default, local `llama3.1:8b`) or `claude`. Local tested live (§6); Claude only with mocks |
| Benchmark + figures | ✅ | `outputs/<preset>/results.json`, fig1–fig9. 20 seeds; deterministic gusts randomised per seed; cost `J`; paired Wilcoxon + bootstrap CIs |
| Web app | ✅ | Vanilla JS; flap drawing; delay input; per-preset policies. **Guardian view** (`/guardian`, 2026-09-29): one recorded fault-tolerant flight (uav_flap; none / rules / rules + physics / **guardian** (default since 2026-09-30: estimator v2 + diagnoser, same flight as ft-benchmark, per-mode admitted/vetoed from its physics verdict); no LLM), played back with the on-board flutter-margin estimate, true flutter speed (simulation-only toggle), per-mode predicted stability, report text and decision log; `api/guardian.py`, `fault_tolerance/recording.py`, `tests/test_guardian_api.py` |
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

## 4. Latest results (generated from `outputs/*/results.json` and `robustness.json`, 2026-09-29, 20 seeds)

Final Phase 4 setup: shared cost J with the rate penalty, realistic actuator (uav_flap 545 °/s, BACT 250 °/s),
classical design knobs tuned on seeds disjoint from the test set, rate-aware LQG filter.
Cells: RMS plunge h in mm (mean ± std) · mean cost J (lower is better; omitted if any run failed) · failures.
Last column: neural relative to LQG over the seeds where both survived (positive = neural worse).

**uav_flap** (generated 2026-09-29T07:56Z, 20 seeds). Flutter 46.40 m/s @ 10.43 Hz.

| Scenario | Open loop | LQR | LQG | Neural | Neural vs LQG (RMS h / J) |
|---|---|---|---|---|---|
| Turbulence, 1.1 U_F | 21.09 ± 0.00 · **20/20 fail** | 2.74 ± 0.32 · J 3.77 | 1.12 ± 0.10 · J 2.65 | 2.29 ± 0.23 · J 3.82 | +104 % / +44 % |
| Discrete gust, 1.1 U_F | 20.31 ± 1.52 · **20/20 fail** | 1.70 ± 0.30 · J 0.343 | 0.63 ± 0.10 · J 0.129 | 1.30 ± 0.22 · J 0.203 | +106 % / +57 % |
| Gust, 1.3 U_F | 26.67 ± 2.95 · **20/20 fail** | 0.48 ± 0.07 · J 0.852 | 0.99 ± 0.16 · J 0.32 | 0.72 ± 0.14 · J 0.513 | -28 % / +60 % |
| Gust, 0.9 U_F | 0.79 ± 0.10 · J 0.107 | 1.48 ± 0.25 · J 0.25 | 0.62 ± 0.09 · J 0.0654 | 0.81 ± 0.12 · J 0.0827 | +31 % / +27 % |

Delay tolerance (ms, discrete gust): lqr 30; lqg 30; neural 15; lqr_nominal 10; lqg_nominal 10

Robustness, turbulence (failures of 20): | condition | lqr | lqg | neural |
|---|---|---|---|
| model error ±5 % | 2 | 1 | 1 |
| model error ±10 % | 1 | 2 | 1 |
| model error ±20 % | 9 | 0 | 7 |
| sensor noise 0 % | 0 | 0 | 0 |
| sensor noise 2 % | 1 | 2 | 0 |
| sensor noise 5 % | 12 | 7 | 6 |
| sensor noise 10 % | 20 | 18 | 20 |
| slew limit none | 0 | 0 | 0 |
| slew limit 500 | 3 | 1 | 0 |
| slew limit 250 | 20 | 20 | 20 |
| slew limit 125 | 20 | 20 | 20 |

**uav_section** (generated 2026-09-29T07:32Z, 20 seeds). Flutter 49.22 m/s @ 10.91 Hz.

| Scenario | Open loop | LQR | LQG | Neural | Neural vs LQG (RMS h / J) |
|---|---|---|---|---|---|
| Turbulence, 1.1 U_F | 18.45 ± 0.00 · **20/20 fail** | 8.78 ± 0.81 · J 17 | 4.21 ± 0.42 · J 5.96 | 6.36 ± 0.66 · J 8.81 | +51 % / +48 % |
| Discrete gust, 1.1 U_F | 21.40 ± 2.69 · **20/20 fail** | 4.77 ± 0.73 · J 2.83 | 2.11 ± 0.32 · J 0.664 | 3.42 ± 0.55 · J 1.37 | +63 % / +105 % |
| Gust, 1.3 U_F | 32.20 ± 4.87 · **20/20 fail** | 6.95 ± 1.10 · J 5.66 | 2.58 ± 0.57 · J 0.973 | 5.02 ± 0.85 · J 2.82 | +94 % / +190 % |
| Gust, 0.9 U_F | 2.56 ± 0.35 · J 0.985 | 3.33 ± 0.50 · J 1.57 | 1.70 ± 0.24 · J 0.455 | 2.43 ± 0.37 · J 0.708 | +43 % / +56 % |

Delay tolerance (ms, discrete gust): lqr 30; lqg 30; neural 15; lqr_nominal 15; lqg_nominal 15

Robustness, turbulence (failures of 20): | condition | lqr | lqg | neural |
|---|---|---|---|
| model error ±5 % | 0 | 0 | 0 |
| model error ±10 % | 0 | 0 | 0 |
| model error ±20 % | 7 | 0 | 1 |
| sensor noise 0 % | 0 | 0 | 0 |
| sensor noise 2 % | 0 | 0 | 0 |
| sensor noise 5 % | 0 | 0 | 0 |
| sensor noise 10 % | 0 | 0 | 0 |

**bact** (generated 2026-09-29T08:00Z, 20 seeds). Flutter 135.67 m/s @ 3.99 Hz.

| Scenario | Open loop | LQR | LQG | Neural | Neural vs LQG (RMS h / J) |
|---|---|---|---|---|---|
| Turbulence, 1.1 U_F | 18.57 ± 0.00 · **20/20 fail** | 1.11 ± 0.13 · J 1.95 | 0.78 ± 0.15 · J 1.53 | 0.89 ± 1.38 · **1/20 fail** | -24 % / +37 % |
| Discrete gust, 1.1 U_F | 18.38 ± 0.96 · **20/20 fail** | 0.79 ± 0.09 · J 0.626 | 0.59 ± 0.11 · J 0.523 | 0.45 ± 0.07 · J 0.72 | -24 % / +37 % |
| Gust, 1.3 U_F | 18.55 ± 1.36 · **20/20 fail** | 0.79 ± 0.11 · J 1.13 | 0.74 ± 0.12 · J 1.11 | 0.46 ± 0.08 · J 1.84 | -38 % / +66 % |
| Gust, 0.9 U_F | 1.79 ± 0.27 · J 2.66 | 0.75 ± 0.08 · J 0.496 | 0.52 ± 0.09 · J 0.377 | 0.42 ± 0.06 · J 0.428 | -20 % / +13 % |

Delay tolerance (ms, discrete gust): lqr 30; lqg 30; neural 15; lqr_nominal 20; lqg_nominal 20

Robustness, turbulence (failures of 20): | condition | lqr | lqg | neural |
|---|---|---|---|
| model error ±5 % | 0 | 0 | 1 |
| model error ±10 % | 0 | 0 | 0 |
| model error ±20 % | 1 | 1 | 2 |
| sensor noise 0 % | 0 | 0 | 1 |
| sensor noise 2 % | 0 | 0 | 2 |
| sensor noise 5 % | 0 | 0 | 2 |
| sensor noise 10 % | 1 | 2 | 2 |
| slew limit none | 0 | 0 | 0 |
| slew limit 500 | 0 | 0 | 0 |
| slew limit 250 | 0 | 0 | 1 |
| slew limit 125 | 20 | 20 | 20 |

**What the numbers say**
- **Neural vs LQG (RQ1): no.** Against the tuned, turbulence-aware LQG, the neural policy has a higher cost J in
  every scenario on every wing. On BACT and in uav_flap's 1.3 U_F gust it has lower RMS h, bought with more effort.
- **Neural vs plain LQR: yes, mostly** (uav_flap: lower J in 3 of 4 scenarios). The claim that survives is
  "a physics-trained policy beats plain LQR", not "beats classical control".
- **Each controller has a regime:** LQG in nominal conditions and under model error (0/20 failures at ±20 %);
  neural under loop delay (fixed designs: neural 15 ms, LQG and LQR 10 ms) and under a 500 °/s slew limit
  (0 failures vs 1 and 3); at 5 % sensor noise neural and LQG are close (6 vs 7 failures of 20). At ≤ 250 °/s no
  controller survives uav_flap turbulence: only slowing down helps. This motivates fault-tolerant supervision.
- **Supervisor with effort modes (RQ2): no value** (rule-supervised points on the fixed-setting curve).
- LQR still amplifies the gust response below flutter on uav_flap (J 0.25 vs open loop 0.107): its weakness is
  the missing disturbance model, which LQG fixes.

### Fault-tolerant supervision (uav_flap)

**Fault-tolerant supervision, uav_flap** (generated 2026-09-29T10:39:45+00:00, 20 seeds, 30 s flights at 1.1 U_F cruise, fault onset 6 s, 5 s ramp; LLM `llama3.1:8b`).
Cells: survival (share of seeds without flutter) · mean time spent slowed down over surviving flights (s). Held-out faults (†) are in neither the rules nor the LLM's guidance.

| Fault | Oracle mode | oracle | none | rules | random | llm |
|---|---|---|---|---|---|---|
| none | nominal | 100 % · 0.0 | 100 % · 0.0 | 100 % · 2.8 | 100 % · 11.7 | 100 % · 0.0 |
| servo_slew | cautious, robust | 100 % · 21.6 | 0 % · – | 5 % · 15.5 | 15 % · 13.1 | 0 % · – |
| stiffness_loss | nominal | 100 % · 0.0 | 100 % · 0.0 | 0 % · – | 0 % · – | 100 % · 0.0 |
| sensor_noise | cautious, nominal | 100 % · 19.3 | 15 % · 0.0 | 100 % · 22.3 | 0 % · – | 50 % · 11.3 |
| turbulence_rise | cautious, robust | 100 % · 14.8 | 0 % · – | 70 % · 18.2 | 55 % · 11.7 | 10 % · 9.6 |
| loop_delay † | cautious, robust | 100 % · 20.5 | 0 % · – | 100 % · 21.8 | 0 % · – | 0 % · – |
| flap_effectiveness † | cautious | 100 % · 22.8 | 0 % · – | 10 % · 17.9 | 15 % · 12.9 | 0 % · – |
| **all faulty** | | **100 %** · 16.5 | **19 %** · 0.0 | **48 %** · 20.8 | **14 %** · 12.2 | **27 %** · 4.1 |

Paired tests over all faulty flights (exact McNemar on survival; Wilcoxon on time slowed):

| Pair | Survived only first | Survived only second | McNemar p | Δ slowed, both survived (s) | Wilcoxon p |
|---|---|---|---|---|---|
| llm vs rules | 20 | 45 | 0.003 | -10.7 | < 0.001 |
| llm vs none | 9 | 0 | 0.004 | +1.4 | 0.109 |
| rules vs none | 54 | 20 | < 0.001 | +22.3 | 0.250 |
| llm vs random | 31 | 16 | 0.040 | -3.8 | 1.000 |
| rules vs random | 45 | 5 | < 0.001 | +5.5 | < 0.001 |

Fixed modes (no supervisor; which response each fault needs), survival:

| Fault | nominal | robust | cautious | safe |
|---|---|---|---|---|
| none | 100 % | 100 % | 100 % | 100 % |
| servo_slew | 0 % | 5 % | 100 % | 100 % |
| stiffness_loss | 100 % | 0 % | 0 % | 70 % |
| sensor_noise | 15 % | 0 % | 100 % | 100 % |
| turbulence_rise | 0 % | 35 % | 100 % | 100 % |
| loop_delay | 0 % | 10 % | 100 % | 100 % |
| flap_effectiveness | 0 % | 0 % | 100 % | 100 % |

LLM answers: 536, latency median 4.4 s, p95 5.9 s; fallbacks to the rules: 1.

**What the numbers say** (generated by `paper._ft_discussion` from `ft_benchmark.json`)
- **LLM vs rules.** Pooled over faulty flights the LLM survived 27 % and the rules 48 % (significant, exact McNemar p = 0.0026). The LLM did better on stiffness_loss and worse on servo_slew, sensor_noise, turbulence_rise, loop_delay, flap_effectiveness.
- **Held-out faults** (in neither the rules nor the guidance): loop_delay: LLM 0 %, rules 100 %, oracle 100 %; flap_effectiveness: LLM 0 %, rules 10 %, oracle 100 %. They test different things: the loop delay appears as a number in the report (a flight computer times its own loop), so it tests acting on an explicit indicator nobody wrote a rule for; flap damage shows only through its symptoms.
- **False alarms** (time slowed down in fault-free flights): oracle 0.0 s, none 0.0 s, rules 2.8 s, random 11.7 s, llm 0.0 s.
- **Gap to the oracle** (share of faulty flights the oracle saves but the supervisor loses): none 81 %, rules 52 %, random 86 %, llm 73 %.
- **The structural-damage trap.** After stiffness loss, holding speed survived 100 % but a moderate slow-down only 0 %: the controllers are gain-scheduled for the healthy wing, which is below flutter at the lower speed while the damaged wing is not. A reflex of "anomaly, slow down a little" is wrong here.
- **In one sentence:** Given slow faults and a menu of pre-verified modes, the same model did worse than hand-written rules (27 % vs 48 % of faulty flights survived), and both stayed below an oracle that knows the fault (100 %).

### Physics estimator v2 (uav_flap, development seeds only)

**margin-eval** (generated 2026-09-29T19:30:46+00:00, seeds 1000-1004, `outputs/uav_flap/margin_eval.json`, fig15). Development data: the design was made on these seeds, so this is not a test.

Estimator accuracy after fault onset, per fault: median |flutter-speed error| v1 / v2; share of estimates whose v2 worst case is at or below the true flutter speed; turbulence estimate / truth (median); predicted / measured nominal flap-rate RMS, windows without saturation (on board; with the true inputs):

| Fault | U_F error v1 / v2 | worst case ≤ truth | turbulence est./true | rate RMS pred./meas. (on board; truth model) |
|---|---|---|---|---|
| none | 1 % / 1 % | 100 % | 1.03 | 1.03; 0.99 |
| servo_slew | 1 % / 1 % | 93 % | 1.08 | 1.02; 0.97 |
| stiffness_loss | 4 % / 4 % | 94 % | 0.87 | 0.83; 0.99 |
| sensor_noise | 16 % / 16 % | 20 % | 1.40 | 1.04; 1.72 |
| turbulence_rise | 1 % / 1 % | 100 % | 0.96 | 1.04; 1.10 |
| loop_delay | 1 % / 1 % | 100 % | 1.06 | 1.00; 0.98 |
| flap_effectiveness | 2 % / 2 % | 71 % | 1.11 | 1.01; 1.10 |

Per-mode predictions at the fully developed fault vs the fixed-mode outcomes (pooled faulty (fault, seed, mode) cases): accuracy · predicted fine but the mode failed · flagged but the mode survived:

- v1 stable: 58 % of 120 · 48 missed · 2 conservative
- v2 admissible (robust): 67 % of 120 · 22 missed · 18 conservative
- v2 admissible (point): 66 % of 120 · 25 missed · 16 conservative

Warning before the nominal flight fails (flights warned / failed, median lead time):

- servo_slew: v1 0/5, v2 2/5 (6.0 s)
- sensor_noise: v1 0/4, v2 4/4 (11.6 s)
- turbulence_rise: v1 0/5, v2 5/5 (6.6 s)
- loop_delay: v1 0/5, v2 5/5 (2.8 s)
- flap_effectiveness: v1 0/5, v2 4/5 (4.3 s)
- v2 flags the nominal mode where it survived (share of estimates): none 11 %, stiffness_loss 75 %, sensor_noise 12 %

Dev comparison (5 seeds; survival · mean time slowed over survivors, s):

| Fault | oracle | rules | rules+physics | rules+physics_v2 | rules+physics_v2_strict |
|---|---|---|---|---|---|
| none | 100 % · 0.0 | 100 % · 2.6 | 100 % · 2.6 | 100 % · 6.8 | 100 % · 6.8 |
| servo_slew | 100 % · 22.8 | 20 % · 15.7 | 20 % · 15.7 | 20 % · 22.2 | 20 % · 22.2 |
| stiffness_loss | 100 % · 0.0 | 0 % · – | 80 % · 0.9 | 20 % · 19.7 | 0 % · – |
| sensor_noise | 100 % · 18.2 | 100 % · 22.4 | 60 % · 22.5 | 100 % · 22.4 | 100 % · 22.4 |
| turbulence_rise | 100 % · 18.2 | 60 % · 17.5 | 60 % · 17.5 | 80 % · 17.4 | 80 % · 17.4 |
| loop_delay | 100 % · 22.8 | 80 % · 21.8 | 80 % · 21.8 | 20 % · 22.2 | 20 % · 22.2 |
| flap_effectiveness | 100 % · 22.8 | 20 % · 17.5 | 20 % · 17.5 | 40 % · 20.2 | 40 % · 20.2 |
| **all faulty** | **100 %** · 17.4 | **47 %** · 20.4 | **53 %** · 15.2 | **47 %** · 20.5 | **43 %** · 20.5 |

rules+physics_v2 vs rules+physics (v1), pooled: survived only with v2 7, only with v1 9, exact McNemar p = 0.80.

**What it says (dev seeds; interpretation of the tables above)**
- **As a predictor, v2 sees what v1 cannot:** v1 never warned before a nominal flight failed; v2 warned before most failures
  (turbulence, delay, noise, flap damage) and halved the "predicted fine but failed" cases, at the price of more
  conservative calls. Its authority prediction matches the measured flap rate when the inputs are right (truth model
  close to 1 except under sensor noise, where the linear noise model overpredicts).
- **As a supervisor, v2 did not beat v1** (pooled survival equal to the rules, below v1; not significant). Where it
  loses: after **stiffness loss** the identified wing is too soft (flutter-speed error negative), so the nominal mode
  is not robustly stable on the softer wings of the confidence region and its point-estimate authority ratio is near
  or above 1 (`at_evaluation` in the JSON), although nominal survives: the structural-damage trap is then answered by
  a slow-down, which fails; under a
  **growing loop delay** v2 correctly finds LQG unstable at 10 ms and picks the neural mode, which then saturates at
  15 ms while the 7 s dwell time locks it in. Where it gains: turbulence rise and flap damage (the faults v1 is blind
  to). False alarms in healthy flight rose (more time slowed). **Sensor noise** biases the identified stiffness upward
  (errors in the regressors), so the estimated flutter speed is far too high there and its worst case rarely covers
  the truth; the symptom rules carry that fault.
- Next if pursued: an authority test robust to the stiffness bias (e.g. authority over the confidence region with a
  better-identified flap factor), a fault-growth-aware choice (avoid controller switches that lock in under a growing
  fault), and a shorter or weighted window to reduce the lag behind ramping faults. Evaluate only on fresh seeds.

### Pre-registered physics test (uav_flap, fresh seeds 100-119)

Supervisors, seeds and the code were fixed before the run (`research/physics-advisor-plan.md` Next 2; snapshot
commit 3a9d325, checked out as a worktree that was removed after the run (recreate with
`git worktree add ../flutter-lab-exp 3a9d325`), run one fault per process by
`docs/research/runs/run_pieces_physics_test100.py` after the first attempt was killed by a low-memory reaper; pieces
merged with the snapshot's `_save`; its first-attempt log is `outputs/uav_flap/physics_test100_pieces_log_first_attempts.txt`). LLM `llama3.1:8b`, answers applied after a
fixed latency (the median of the main benchmark), so the result does not depend on machine load. Files:
`outputs/uav_flap/ft_benchmark_physics_test100.json` (+ the seven `_<fault>` pieces), fig12_fault_tolerance_physics_test100.
Cells: survival · mean time slowed over surviving flights (s). † held out.

**physics_test100** (generated 2026-09-29T19:31:05+00:00, seeds 100-119, complete True, fixed LLM latency 4.41 s)

| Fault | oracle | none | rules | rules+physics | llm | llm+physics |
|---|---|---|---|---|---|---|
| none | 100 % · 0.0 | 100 % · 0.0 | 100 % · 2.6 | 100 % · 2.6 | 100 % · 0.0 | 100 % · 0.0 |
| servo_slew | 100 % · 21.6 | 0 % · – | 15 % · 20.2 | 15 % · 20.2 | 0 % · – | 0 % · – |
| stiffness_loss | 100 % · 0.0 | 100 % · 0.0 | 5 % · 25.0 | 70 % · 4.0 | 100 % · 0.0 | 5 % · 2.8 |
| sensor_noise | 100 % · 21.6 | 5 % · 0.0 | 100 % · 22.5 | 85 % · 22.5 | 50 % · 14.0 | 10 % · 5.7 |
| turbulence_rise | 100 % · 14.8 | 0 % · – | 85 % · 18.2 | 85 % · 18.2 | 20 % · 8.3 | 0 % · – |
| loop_delay † | 100 % · 21.6 | 0 % · – | 100 % · 21.9 | 100 % · 21.9 | 0 % · – | 0 % · – |
| flap_effectiveness † | 100 % · 22.8 | 0 % · – | 20 % · 18.9 | 25 % · 19.2 | 0 % · – | 0 % · – |
| **all faulty** | **100 %** · 17.1 | **18 %** · 0.0 | **54 %** · 20.9 | **63 %** · 17.7 | **28 %** · 5.1 | **2 %** · 4.7 |

| Pair | Survived only first | Survived only second | McNemar p | Δ slowed, both survived (s) | Wilcoxon p |
|---|---|---|---|---|---|
| llm vs rules | 19 | 50 | < 0.001 | -10.0 | < 0.001 |
| llm vs none | 13 | 0 | < 0.001 | +0.8 | 0.32 |
| rules vs none | 63 | 19 | < 0.001 | +23.7 | 0.50 |
| rules+physics vs rules | 14 | 3 | 0.013 | +0.0 | 1.00 |
| llm+physics vs llm | 1 | 32 | < 0.001 | -6.5 | 1.00 |
| llm+physics vs rules+physics | 2 | 75 | < 0.001 | -13.7 | 1.00 |

LLM answers: 1046, measured latency median 3.6 s, p95 4.9 s; fallbacks: 0.

**What it says**
- **The physics estimate helps the rules** (rules+physics vs rules significant, mostly through the structural-damage
  trap: stiffness_loss 70 % vs 5 %; it costs some sensor-noise flights, 85 % vs 100 %, because noise biases the
  identified stiffness).
- **The LLM lost to the rules again** on fresh seeds (replicates §4's main study).
- **The same estimate made the LLM much worse** (llm+physics vs llm: 1 vs 32 flights). After stiffness loss the LLM
  alone kept `nominal` (right) in all 20 flights; with the estimate it slowed to `cautious` (the trap) in all 20, citing
  the estimate. It also acted less on sensor noise and turbulence. Mode changes after onset over the 120 faulty flights:
  llm 16, llm+physics 23 (paper §7.6 prints these from the file). Information is not what the LLM lacks.

### Guardian supervisor (uav_flap; dev seeds 1000-1009, then the final test once on seeds 300-319)

Design frozen after the dev run (not changed after seeing it); final test run once. No LLM in either run.

**guardian_dev1000** (generated 2026-09-30T13:56:51+00:00, seeds 1000-1009, complete True)

| Fault | oracle | none | rules | rules+physics | rules+physics_v2 | guardian |
|---|---|---|---|---|---|---|
| none | 100 % · 0.0 | 100 % · 0.0 | 100 % · 1.3 | 100 % · 1.3 | 100 % · 3.4 | 100 % · 0.0 |
| servo_slew | 100 % · 22.8 | 0 % · – | 10 % · 15.7 | 10 % · 15.7 | 20 % · 19.7 | 90 % · 18.2 |
| stiffness_loss | 100 % · 0.0 | 100 % · 0.0 | 0 % · – | 70 % · 1.5 | 10 % · 19.7 | 90 % · 0.0 |
| sensor_noise | 100 % · 20.5 | 10 % · 0.0 | 100 % · 22.4 | 70 % · 22.4 | 100 % · 22.4 | 100 % · 22.4 |
| turbulence_rise | 100 % · 15.9 | 0 % · – | 80 % · 16.9 | 80 % · 16.9 | 90 % · 16.8 | 100 % · 21.4 |
| loop_delay † | 100 % · 20.5 | 0 % · – | 90 % · 21.8 | 90 % · 21.8 | 10 % · 22.2 | 100 % · 22.2 |
| flap_effectiveness † | 100 % · 22.8 | 0 % · – | 10 % · 17.5 | 10 % · 17.5 | 50 % · 19.8 | 100 % · 19.7 |
| **all faulty** | **100 %** · 17.1 | **18 %** · 0.0 | **48 %** · 20.3 | **55 %** · 16.1 | **47 %** · 19.9 | **97 %** · 17.6 |

| Pair | Survived only first | Survived only second | McNemar p | Δ slowed, both survived (s) | Wilcoxon p |
|---|---|---|---|---|---|
| rules vs none | 28 | 10 | 0.005 | +22.5 | 1.00 |
| rules+physics vs rules | 7 | 3 | 0.34 | +0.0 | 1.00 |
| guardian vs rules | 29 | 0 | < 0.001 | +1.6 | < 0.001 |
| guardian vs rules+physics | 25 | 0 | < 0.001 | +1.1 | 0.011 |
| guardian vs none | 48 | 1 | < 0.001 | +2.2 | 1.00 |
| guardian vs rules+physics_v2 | 30 | 0 | < 0.001 | +0.7 | 0.18 |

**guardian_test300** (generated 2026-09-30T14:20:17+00:00, seeds 300-319, complete True)

| Fault | oracle | none | rules | rules+physics | rules+physics_v2 | guardian |
|---|---|---|---|---|---|---|
| none | 100 % · 0.0 | 100 % · 0.0 | 100 % · 3.0 | 100 % · 3.0 | 100 % · 8.3 | 100 % · 0.7 |
| servo_slew | 100 % · 20.5 | 0 % · – | 20 % · 17.8 | 20 % · 17.8 | 30 % · 19.4 | 80 % · 18.7 |
| stiffness_loss | 100 % · 0.0 | 100 % · 0.0 | 0 % · – | 55 % · 1.5 | 0 % · – | 80 % · 0.2 |
| sensor_noise | 100 % · 20.5 | 10 % · 0.0 | 100 % · 22.4 | 90 % · 22.4 | 100 % · 22.4 | 100 % · 22.4 |
| turbulence_rise | 100 % · 14.8 | 0 % · – | 70 % · 18.1 | 70 % · 18.1 | 75 % · 18.1 | 100 % · 21.3 |
| loop_delay † | 100 % · 20.5 | 0 % · – | 100 % · 21.8 | 100 % · 21.8 | 0 % · – | 100 % · 22.2 |
| flap_effectiveness † | 100 % · 21.6 | 0 % · – | 15 % · 19.7 | 15 % · 19.7 | 50 % · 20.6 | 100 % · 19.9 |
| **all faulty** | **100 %** · 16.3 | **18 %** · 0.0 | **51 %** · 20.8 | **58 %** · 17.7 | **42 %** · 20.4 | **93 %** · 18.0 |

| Pair | Survived only first | Survived only second | McNemar p | Δ slowed, both survived (s) | Wilcoxon p |
|---|---|---|---|---|---|
| rules vs none | 59 | 20 | < 0.001 | +22.4 | 0.50 |
| rules+physics vs rules | 11 | 2 | 0.022 | -0.0 | 0.32 |
| guardian vs rules | 51 | 0 | < 0.001 | +0.9 | < 0.001 |
| guardian vs rules+physics | 43 | 1 | < 0.001 | +0.7 | 0.003 |
| guardian vs none | 94 | 4 | < 0.001 | +2.7 | 0.11 |
| guardian vs rules+physics_v2 | 61 | 0 | < 0.001 | +0.7 | 0.47 |

**What it says**
- **Final test: guardian 93 % of faulty flights survived vs rules 51 %, rules+physics 58 %, none 18 %, oracle 100 %**;
  51 flights saved that the rules lost and none the other way; both held-out faults 100 %; 0.7 s slowed on average in
  healthy flights (rules 3.0 s). The dev run (97 %) did not overstate it much.
- **Misses (test):** 4 servo_slew flights (diagnosed or flagged 4.8-5.5 s after onset, slowed to `cautious`, failed
  within 2 s: the slow servo is latent until the flap must move fast) and 4 stiffness_loss flights (physics v2 read the
  damaged wing as softer than it is, predicted `nominal` unstable, chose `safe`, which failed; `none` survives all 20).
  Same two causes as the dev misses (seeds 1004 servo, 1009 stiffness).
- rules+physics_v2 on its own is worse than v1 here (loop_delay 0 %: it switches to the neural mode that the growing
  delay then saturates, as found in margin-eval); the guardian avoids this by putting the neural mode last.


### LLM off the control path (uav_flap, development seeds)

**Explainer** (`explainer_eval.json`, generated 2026-09-30T15:28:17+00:00, llama3.1:8b, 7 flights, seeds 34000-34006, rules + physics supervising): 34 reports, 34 valid answers, 29 faithful (85 %), 5 unfaithful, 0 failed calls; template shown for 15 % of reports; latency median 4.2 s, p95 6.1 s. Rejection reasons: fault 5.
Faults the rejected answers named: servo_slew 4, none 1.

**Post-flight reviewer + verification gate** (`review_loop.json`, generated 2026-09-30T15:35:16+00:00, llama3.1:8b, wall time 7 min): anomaly search loop_delay: 8 of 8 flights raised v1's unknown alarm; flap_effectiveness: 2 of 8 flights raised v1's unknown alarm. Reviews 5, valid hypotheses 4, naming the true fault 1; latency median 7.8 s.
- review seed 45000 (truth loop_delay): flap_effectiveness, severity 0.5-1.2, growth 10-15 s
- review seed 45001 (truth loop_delay): flap_effectiveness, severity 0.5-1.2, growth 8-12 s
- review seed 45002 (truth loop_delay): flap_effectiveness, severity 0.8-1.2, growth 10-12 s
- review seed 45103 (truth flap_effectiveness): no valid hypothesis (ValueError: growth_min_s/growth_max_s = 10/20 must satisfy 0.5 <= min <= max <= 15.0 s)
- review seed 45106 (truth flap_effectiveness): flap_effectiveness, severity 0.8-1.2, growth 10-15 s

| Candidate | Source | Anomaly (truth) | Hypothesis | Right fault | Gate | Reasons | Fresh flights: new class detected / v1 unknown alarm |
|---|---|---|---|---|---|---|---|
| 0 | llm | loop_delay | flap_effectiveness | no | rejected | healthy-flight false alarms 0.133 -> 0.2 (tolerance 0.05); hypothesised class confirmed in 0% of the anomaly's windows (needs >= 50%); unknown alarm still in 97% of the anomaly's windows (allowed <= 10%) | – |
| 1 | forced_wrong | loop_delay | sensor_noise | no | rejected | F1 of servo_slew 0.581 -> 0.525 (tolerance 0.05); hypothesised class confirmed in 0% of the anomaly's windows (needs >= 50%); unknown alarm still in 99% of the anomaly's windows (allowed <= 10%) | – |
| 2 | llm | flap_effectiveness | flap_effectiveness | yes | rejected | healthy-flight false alarms 0.133 -> 0.2 (tolerance 0.05) | – |
| 3 | forced_wrong | flap_effectiveness | servo_slew | no | rejected | healthy-flight false alarms 0.133 -> 0.3 (tolerance 0.05); hypothesised class confirmed in 0% of the anomaly's windows (needs >= 50%); unknown alarm still in 43% of the anomaly's windows (allowed <= 10%) | – |

**What it says**
- **Explaining works, mostly.** Every call returned a valid answer; 85 % were faithful by the check (mode = decision,
  fault = diagnosis, no unflagged fault named). All rejections were a wrong `fault` field, mostly `servo_slew`; the
  operator then sees the template. The check covers fields and literal fault names, not every claim of the free text.
- **Reviewing did not produce an accepted model.** The LLM named the right fault for flap damage (1 of 1 valid review)
  but called every loop-delay anomaly flap damage (3 of 3). The gate rejected the wrong hypothesis and both forced-wrong
  controls on several criteria each, and **also rejected the right one**, only because healthy-flight alarms rose from
  4 to 6 of 30 flights (tolerance 0.05 = at most 1 extra flight); that candidate explained the anomaly fully and
  improved every v1 class. The gate is safe but conservative at this sample size (thresholds fixed in advance, not
  changed). No `models/diagnoser_uav_flap_v2_*` file was written.
- The flap-damage anomaly was rare: only 2 of 8 searched flap-damage flights raised v1's unknown alarm (loop delay 8 of 8).

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
| 2026-09-29 | **Shared rate penalty**: the (Δu/u_max)² term (weight 1.0) is now in the LQR/LQG design cost and in the reported J, not only in neural training (`costs.RATE_WEIGHT`, `lqr.dlqr_with_rate`) | costs.py claimed one shared objective, but only the neural policy was penalised for fast commands: an unfair comparison |
| 2026-09-29 | **Actuator slew-rate limit** in plant and training: uav_flap 545 °/s (DS-W001A UAV servo datasheet, no-load ≤0.11 s/60°, optimistic), BACT 250 °/s (NASA TM-97-206252 Table 3, stated to be an educated guess) | Unlimited, the controllers commanded ~5000 °/s; with any realistic limit every controller failed the old uav_flap scenarios |
| 2026-09-29 | uav_flap disturbances ×0.33 (`DISTURBANCE_SCALE`), training turbulence ≤ 0.5 m/s, ic_scale 0.4; neural policy retrained | Largest tested scale at which a classical controller survives every scenario with the 545 °/s servo (same approach as BACT ×0.25) |
| 2026-09-29 | Classical **design knobs tuned for the real actuator** (`tune`: LQR rate multiplier; LQG rate multiplier + assumed sensor noise), on seeds 1000–1009 only; fewest failures, then lowest J; stored in `models/classical_tuning_<preset>.json` (fingerprint-checked) | The neural policy is optimised through the rate-limited plant; the classical designs deserve the same chance. Untuned LQG failed 20–100 % of runs with the rate limit (no guaranteed margins, Doyle 1978) |
| 2026-09-29 | `simulate(..., design_params=)` separates the controllers' on-board model from the true wing | Needed for the robustness study (model error) |
| 2026-09-29 | Identification: output error with **multiple shooting** (0.25 s segments from the measured state; lag states rebuilt with first-order hold, gust held per sample) | Replaying a whole run through a model above flutter amplifies CSV rounding exponentially |
| 2026-09-29 | mypy on `src/` in CI (numpy stubs need python_version 3.12 for analysis; torch/scipy/matplotlib/anthropic not followed; pydantic plugin) | Phase 6.3; it caught ~80 issues, mostly unchecked `None` flutter speeds and flap hinges |
| 2026-09-29 | Paper draft is generated (`python -m flutter_ai paper`), prose in `paper.py`, numbers from result files | Rules §1: no hand-typed numbers, even in drafts |
| 2026-09-29 | Fault-tolerant study in its own package (`fault_tolerance/`) and simulator | The validated Phase 4 benchmark (`simulation.py`) stays untouched |
| 2026-09-29 | Controller **bank with bumpless transfer**: all controllers run every step on the same measurements and applied-command history | Switching without transients; makes all modes share one state, which the stability proof needs |
| 2026-09-29 | FT modes: nominal (LQG, cruise), robust (neural, cruise), cautious (LQG, 0.9 U_F), safe (LQG, 0.7 U_F) | Calibration: slowing down with neural/LQR failed after structural damage; 0.9 U_F is still above a damaged wing's flutter speed; neural was never uniquely right |
| 2026-09-29 | Oracle = least disruptive surviving mode with hindsight (hold speed if possible, then the smallest slow-down) | Otherwise "always slow down" wins trivially; mission time matters |
| 2026-09-29 | Diagnostics: nominal-model one-step residual per channel and high-frequency content vs a 4 s healthy baseline; pitch frequency from a 4 s Welch spectrum | Single 2 s FFTs wandered ±1.5 Hz and caused false alarms |
| 2026-09-29 | Rules are written from the same principles as the LLM guidance and are **not** tuned on benchmark outcomes; held-out faults (loop delay, flap damage) are in neither | A fair rules-vs-LLM comparison |
| 2026-09-29 | Minimum dwell between mode switches 7 s ≥ dwell-time bound 6.5 s (Hespanha & Morse; `ft-stability`, scaled state); a test enforces the inequality and checks the mode matrices reproduce the simulator | Guarantee: no sequence of supervisor answers destabilises the nominal closed loop (small-signal) |
| 2026-09-29 | LLM supervisor is task-configurable (`LLMSupervisor.configure`: prompt, mode enum, report, decision, fallback) | One safety wrapper for effort modes and FT modes |
| 2026-09-29 | After a code review, before any complete `ft-benchmark`: the 7 s dwell applies only to **controller** switches (LQG ↔ neural); an answer blocked by it is dropped and the supervisor asked again (never applied late) | The stability bound only concerns controller switches; the old rule delayed LQG-only escalation (cautious → safe) by 7 s and then applied a stale answer. The first run was stopped after 3 of 7 faults and is superseded (only its log lines were seen) |
| 2026-09-29 | FT metrics: reaction = first mode change *decided after onset* that took effect; time slowed and cost J averaged over surviving flights; Wilcoxon on flights both survived; `ft_benchmark.json` rewritten after every fault | Old reaction counted pre-fault false alarms and "keep" answers; old time slowed rewarded failing early; a stopped run lost everything |
| 2026-09-29 | Diagnostics use the plant's quantised airspeed, the previous applied command at the window start, and skip the first 0.5 s of residuals; random supervisor seeded independently of the flight | Review findings (< 0.25 % model mismatch; start-of-window transient); cleanliness |
| 2026-09-29 | Related work corrected from full texts: Papacharalampopoulos et al. 2026 **do** let the LLM choose the mode (with rule fallback and a dwell-time bound); Scott & Pado 2000 scheduled gains on the BACT wing | The draft said their LLM only wrote audit records; the novelty claim is now the plant, on-board latency and the evaluation, not the architecture pattern |
| 2026-09-29 | Next research direction (owner): physics-informed advisor (on-board flutter-margin estimator) + time-scale map; ablations first (no latency, stronger model, history) | The LLM lost to rules; find out why before redesigning, and give the supervisor physics instead of symptoms |
| 2026-09-29 | Margin estimator identifies pitch stiffness and flap effectiveness only; plunge stiffness held at its healthy value; estimates self-calibrated against the take-off fit; developed on seed 1000 | Plunge stiffness is not identifiable in closed loop (drifted +40 % on an unchanged wing); the unmeasured gust biases the fit; test seeds must stay untouched |
| 2026-09-29 | **Guardian live view** = a background job (one daemon worker, POST → poll → full recording), then playback in the browser; identical requests reuse the recording; kept in memory only; uav_flap only; supervisors none / rules / rules + physics, **no LLM**; default seed 1000 (dev), onset ≥ 4.5 s; ground truth = `flutter_analysis`'s flutter speed computed root-only (a test pins equality), per fault level | A 30 s flight with the estimator costs ~0.5–2 min of CPU (12 s flight: 40 s on the busy machine), too slow for one request; streaming would need hooks inside `ft_simulate`, and recording through a supervisor wrapper leaves the benchmark code untouched and the flight identical. The GPU is busy and the view must never call a model. The FT study is calibrated for uav_flap (fault sizes, dwell time). The diagnostics calibrate on 0.5–4.5 s |
| 2026-09-29 | Guardian colours: wing = active controller's entity colour (LQG / neural); on-board estimate = new token `--c-estimate` (slot 3 aqua) | Validator (all pairs, with LQG + neural): aqua passes light and dark; blue failed in dark next to the LQG violet (CVD ΔE 1.9); magenta, yellow, red and green failed next to orange |
| 2026-09-29 | **Learned fault diagnoser** (`fault_tolerance/diagnoser*.py`, `train-diagnoser` / `eval-diagnoser`, `models/diagnoser_uav_flap.npz` + `.json`, results `outputs/uav_flap/diagnoser_eval.json`, fig14): physics features per 2.5 s window -> encoder -> multi-label head for the 4 known faults + decoder; unknown-fault score = reconstruction error (fixed before evaluation; Mahalanobis only secondary); thresholds from validation flights; loop timing not an input; loop_delay / flap_effectiveness never in training or calibration; `CompositeFault` for 2-3 simultaneous faults; seeds 10000+ / 20000+ / 30000-32055 | Owner: the LLM is too slow and passive to supervise; a fast learned model should analyse the data and an LLM only explain it (`format_diagnosis` report block). Features, not raw signals: the fault signatures are known physics and the model stays tiny (NumPy, sub-millisecond). Weak spots are in the JSON and `research/physics-advisor-plan.md`: slow servo (latent until it limits the flap), false servo alarms in `safe`-mode flights, flap damage rarely flagged as unknown |
| 2026-09-30 | **Estimator v2** (`margin.estimate_v2`, `FTPhysicsRuleSupervisorV2`, `ft_simulate(margin_version=2)`, `margin-eval`); v1 `estimate`, `FTPhysicsRuleSupervisor` and the `mode_matrices` defaults unchanged. Authority = 3σ flap command/rate (Lyapunov, AR(1) gust + sensor noise) / limit, required only where the identified wing needs control; turbulence and noise from the residual's gust component and a 4th-difference noise level, calibrated on take-off; measured per-sample loop delay in the fit and `mode_matrices(delay=d)`; robust = worst case over the 95 % confidence ellipse (Gauss-Newton covariance inflated to contain a HAC sandwich, plus the calibration's uncertainty); supervisor falls back to point-estimate admissibility when nothing is robustly admissible (`strict` variant without it); flap factor bounds 0.05-2 in v2. Developed on seeds 1000-1004 only | v1 cannot see saturation (flap damage), delay, turbulence or noise. Details and limits: `research/physics-advisor-plan.md` Step 2; results only in `outputs/uav_flap/margin_eval.json`. v1's flap floor 0.3 on a raw healthy fit of ~0.75 capped the identifiable damage at about −60 %; the current airspeed is left out of a mode's transit because it cannot tell apart modes sharing a controller; the ellipse (not a box) keeps the correlation of the two factors; the strict 95 % region after stiffness loss certified no mode at all |

| 2026-09-30 | **Pre-registered physics test run one fault per process** (`docs/research/runs/run_pieces_physics_test100.py`, run from a worktree of snapshot 3a9d325 that was removed afterwards, seeds 100-119, `--llm-latency` = the main benchmark's median); a piece with any LLM fallback would be retried once (none had any) | The first single-process attempt was killed after the `none` fault by a low-memory reaper, and the PC froze once with several experiments running side by side; the pieces use the same function and arguments, only the process boundary differs |
| 2026-09-30 | **Guardian design frozen after the dev run** (seeds 1000-1009; its two dev misses have known causes: latent slow servo, physics v2 soft-wing bias); final test **once** on seeds 300-319 with none, rules, rules+physics, rules+physics_v2, guardian; **no LLM in the guardian test** | Rules §1: evaluate once on fresh seeds. The LLM-vs-rules comparison is already replicated on two test sets (0-19, 100-119); adding it would have cost ~1 h of LLM time for no new question |
| 2026-09-30 | Experiments run **one LLM job at a time**; Ollama restarted between the benchmark and the LLM studies | The llama-server working set grew from ~1.4 to ~4.9 GB (model file mapped as it is used) and free RAM dipped to ~0.2 GB with Chrome/VS Code open; queued LLM calls would also hit the 30 s timeout |
| 2026-09-30 | `plots.ft_figure` draws **every** supervisor in the file (it drew only none/random/rules/llm/oracle, so guardian and the physics supervisors were missing); legend moved above the axes; new supervisor colours from existing palette slots, validated (Design.md) | The Guardian figure did not show the guardian. The main fig12 (seeds 0-19) is unchanged until it is regenerated |
| 2026-09-30 | Paper generator: §7.6-7.8, abstract/conclusion sentences and discussion bullets for the new studies, computed from the result files (`_ft_supervisor_table`, `_pair_sentence`, `_acted`, `_verdict`, `_later_findings`, `_later_discussion`); fixed a missing f-string that printed `{cite('doyle1978')}` literally | Rules §1: no hand-typed numbers in the draft |

## 6. Known issues and limitations (be honest about these in the paper)

- Gust entry uses the Wagner lag (not Küssner); the model is linear (no freeplay or LCO yet).
- LQR's and LQG's optimism: in benchmarks they are re-designed for the true delay; fixed-design variants
  (`lqr_nominal`, `lqg_nominal`) are in the delay sweep. LQG designed for 5 ms tolerates 10 ms, like neural.
- LQG and the neural policy are designed on a model identical to the nominal simulator. The robustness study
  (`robustness`, §4) perturbs parameters by ±5/10/20 %, adds sensor noise and tightens the slew limit; that is
  parametric uncertainty only (no unmodelled dynamics), so it still says nothing definite about real hardware.
- The neural policy has no memory or gust estimate. Giving it the Kalman estimate (neural + filter) is the obvious
  way to test whether learning can beat LQG (e.g. under saturation); only worth doing if RQ1 stays in the paper.
- Delay tolerance of fixed designs on uav_flap: neural 15 ms, LQR and LQG 10 ms (re-designed LQR/LQG: 30 ms).
- The neural policy's aggressiveness input barely moves its trade-off, and "aggressive" is worse than "efficient"
  on both axes, so an effort-mode supervisor has almost nothing to choose between. RQ2 needs a different action space.
- Local LLM (2026-09-29, Ollama `llama3.1:8b` Q4 on the RTX 4050, 6 GB, 19 % of the model on CPU): 72/72 valid schema
  answers, latency median 3.75 s / p95 5.7 s, cold load 16 s; agrees with the rules 56/72 and over-escalates to
  "emergency" at 30–50 % of limit. Its decisions arrive ~4 s late, after the gust: an LLM cannot react to gusts.
  `llama3.2:3b` (100 % GPU): median 2.4 s / p95 2.9 s, but agrees with the rules only 15/72 (says "aggressive" 63/72,
  even in calm air): faster, much worse judgement. Live web-app run (8B, turbulence): answers at 3.7–4.6 s, three per
  10 s run; the "aggressive" asked at 4.0 s landed at 8.5 s, after the turbulence ended at 7 s. The `llm-benchmark`
  run (5 seeds, 79 decisions, busier machine) measured median 5.4 s / p95 9.7 s, 0 fallbacks, 68 % agreement with
  the rules (`outputs/uav_flap/llm_benchmark.json`).
- Final `ft-benchmark` (2026-09-29): one LLM call (stiffness_loss, seed 9, t = 10.75 s) returned after 7165 s of wall
  time; the whole process paused ~2 h (most likely the PC slept). The wrapper treated it as a timeout (fallback,
  latency capped at 30 s, so the answer never landed) and that flight kept `nominal` and survived. The LLM kept
  `nominal` in all other stiffness_loss flights, so the outcome is very probably unchanged; it is the only LLM
  fallback in 140 flights. Re-run `ft-benchmark` on a machine that cannot sleep to remove it entirely.
- The Claude backend is untested against the live API (only with mocks).
- Guardian: its misses are a slow servo recognised too late (latent until the flap must move fast) and physics v2
  reading a damaged wing as softer than it is (then `safe` is chosen and fails). Its thresholds (0.70, 0.92) and fault
  responses come from dev data and the fixed-mode table of this benchmark: the same simulator family, fault catalogue and
  severities; nothing says how it transfers to other fault sizes or a real wing. The guardian test has no LLM
  supervisor (the LLM was tested on seeds 0-19 and 100-119).
- Pre-registered test: its `none` piece is from the first attempt (2026-09-29, same snapshot and arguments); the other
  six pieces ran on 2026-09-30. The metadata's `generated_at` is the first piece's; `pieces` lists each.
- Memory: with Chrome and VS Code open, llama-server (~5 GB mapped) and experiments leave < 1 GB free on this 16 GB
  laptop. Run LLM experiments one at a time and close other apps for long runs.
- Guardian view: checked in Chrome on 2026-09-30 with a guardian flight (stiffness loss, seed 1000: records in ~11 s,
  plays back, per-mode admitted/vetoed cards, v2 worst case and turbulence, no console errors). Two bugs found and
  fixed there: the progress block ignored `hidden` (CSS `display: flex`), and the note under the mode cards described
  estimator v1 on v2 flights. Recordings live in server memory only (lost on restart). No LLM supervisor there by
  design; adding one would need the call budget and a busy-GPU guard. A 30 s physics-supervised flight takes tens of
  seconds of CPU, longer while experiments run.
- The observer's aero-lag estimate lags about half a step; that is fine, but it is part of the loop.
- `pytest` takes ~2 min (flutter sweeps); make the slow ones optional if needed.
- Novelty (search 2026-09-29, full texts checked where openly available; `research/related-work.md`): neural
  flutter control under input constraints with wind-tunnel tests exists (Tang et al., J. Vib. Control 2021, online
  2020); NASA tunnel-tested classical, **LQG** and minimax flutter controllers on BACT (Mukhopadhyay, IFA 1999) and a
  neural gain scheduler on the same wing (Scott & Pado 2000); "LLM as slow supervisor over a fast inner loop" is
  surveyed (Du & Chan, arXiv 2609.16599). **The architecture pattern is not new:** Papacharalampopoulos et al.
  (Processes 2026) have an LLM select among pre-verified controllers with a rule fallback and a dwell-time bound, for
  process systems (cloud model, once per fault, no latency model, no random/oracle baseline). No LLM + flutter /
  aeroelastic work found. The claim is the plant, the on-board model with measured latency, and the evaluation.
  Not a PINN any more.
- Real data (search 2026-09-29): no openly downloadable flutter time histories found. BACT's test database and
  NASA's MATLAB simulation were distributed on request only (TM-97-206252). Usable: measured flutter points (in use),
  NASA's closed-loop boundary extension to the 200 psf tunnel limit (IFA 1999), measured tunnel turbulence spectra
  (TM-107734, not yet opened).
- BACT: after the turbulence patch the neural policy leaves a slowly decaying pitch oscillation (LQR settles
  faster), so neural's settling time reads ~10 s / "–" in the web app.

## 7. Next steps (in order)

1. The owner's items in `research/submission-checklist.md`: advisor review, venue, author list, library copies of
   the four paywalled papers, read Papacharalampopoulos et al. 2026, edit the generated prose and write the closing
   sentence of §10. **Decide the paper's framing**: the evidence now supports "fast learned + physics models decide,
   the LLM explains" (Guardian 93 % vs rules 51 % on fresh seeds; LLM below rules twice; physics hurt the LLM) more
   than the current title ("On-board language-model supervision ..."). The title and contributions list in `paper.py`
   are the owner's call.
2. Optional, strengthens the paper: run `ft-benchmark` with a second LLM (`FLUTTER_AI_LLM_BACKEND=claude`, or another
   local model) to show the result is not specific to one model. Before claiming the Guardian's novelty, search
   fault detection and isolation (FDI) and flutter-margin monitoring literature (`related-work.md` §7).
   The Guardian view now offers the `guardian` supervisor (2026-09-30).
3. Qualitative analysis of the LLM's reasoning (RQ4) from `ft_benchmark.json` → `decisions`.
4. Improve BACT accuracy (compressibility; second flutter point) and use TM-107734 for tunnel turbulence.
5. Measured time histories from a rig, then `identify` (parameters with uncertainty).

## 8. How to update this file

At the end of a session: bump "Last updated", change §2 states, paste new numbers into §4
**from results.json only**, add rows to §5 for any decision (with *why*), and edit §6/§7.
