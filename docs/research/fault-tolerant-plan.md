# Fault-tolerant LLM supervision: study design

*Written 2026-09-29, after Phase 4. Approved by the owner ("build the fault-tolerant plan"). Built in
`src/flutter_ai/fault_tolerance/`; commands `ft-benchmark` and `ft-stability`. Memory.md tracks results.*

## 1. Why this study

Phase 4 showed that an LLM choosing *effort modes* cannot add value: the knob barely moves the trade-off, and a
local 8B model answers in ~4 s, far slower than a gust. The same Phase 4 results show that **no single inner-loop
controller is best everywhere** on the realistic `uav_flap` wing (545 °/s servo):

| Condition | Best choice (Phase 4 evidence) |
|---|---|
| Nominal, turbulence or gusts | LQG (tuned) |
| Sensor noise ≥ 5 % of reference scale | Neural (fewest failures) |
| Loop delay 10–15 ms (fixed designs) | Neural (tolerates 15 ms; LQG designed for 5 ms only 5 ms) |
| Actuator slew rate ≤ 250 °/s, or stiffness loss (lower flutter speed) | No controller survives above flutter: **slow down** |

So a supervisor has a real decision to make, the decision depends on a *diagnosis*, and the conditions change over
seconds to minutes: a time scale an on-board LLM can meet. That is the study.

## 2. Research question

**RQ2′.** Under slowly developing faults, does an on-board LLM supervisor that chooses among pre-verified
(controller, airspeed) modes improve survival and cost compared with no supervision, hand-written rules and random
choice, how close does it come to an oracle that knows the fault, and does that hold for faults the rules were not
written for? All with the LLM's measured latency, one question at a time.

## 3. Faults (injected mid-flight, with onset time and ramp)

| # | Fault | Model | Right response (hypothesis) | In rules / prompt? |
|---|---|---|---|---|
| F1 | Servo slew rate degrades (545 → 200 °/s) | `u_rate_max` ramps down | Slow down below flutter | Yes |
| F2 | Stiffness loss (f_alpha −8 %) | plant parameters ramp; flutter speed drops | Slow down | Yes |
| F3 | Pitch-sensor noise rises (to 5–10 % of reference) | measurement noise ramps up | Switch to neural | Yes |
| F4 | Turbulence intensifies (×2) | gust RMS ramps up | LQG, or slow down if saturating | Yes |
| F5 | Computation overload: loop delay 1 → 3 samples | delay steps up | Switch to neural | **Held out** |
| F6 | Flap effectiveness loss (−50 %) | flap aerodynamic coupling scaled | Slow down | **Held out** |

Held-out faults appear neither in the prompt's guidance nor in the rules: they test whether the LLM generalises,
which is the only way an LLM could beat good rules. They test different things: the loop delay is printed in the
report as a number (a flight computer can time its own loop), so F5 tests acting on an explicit indicator that no
rule was written for; F6 shows only through symptoms (code review 2026-09-29; the paper states this).

## 4. Action space (fixed, pre-verified), revised after calibration

| Mode | Inner loop | Airspeed |
|---|---|---|
| `nominal` | LQG | hold cruise (1.1 × nominal U_F) |
| `robust` | Neural | hold cruise |
| `cautious` | LQG | slow to 0.9 × nominal U_F (autopilot, 2 m/s²) |
| `safe` | LQG | slow to 0.7 × nominal U_F |

The LLM chooses a mode name from this enum. It never outputs gains, forces or speeds.

**What calibration changed (6 seeds per cell, `ft_benchmark.json → fixed_modes` has the final table):**
- The first menu paired slowing down with *other* controllers (neural, LQR). After structural damage those
  controllers failed during the deceleration, while LQG survived even at cruise. Slowing down now keeps LQG.
- Slowing down only to 0.9 U_F after structural damage *fails*: the controllers are gain-scheduled for the healthy
  wing, which is below flutter there, but the damaged wing (flutter at 0.83 U_F) is not. `safe` therefore goes to
  0.7 U_F. The least disruptive correct answer to that fault is to hold speed, a deliberate trap for "anomaly →
  slow down" reflexes.
- The neural controller was never the uniquely correct choice; `robust` stays in the menu as a possible mistake.
- Severities were set so that doing nothing fails (except for structural damage, see above); see `faults.FAULTS`.
- Diagnostics: pitch frequency from a Welch spectrum over 4 s and a 4 s healthy baseline (single FFTs over 2 s
  wandered by ±1.5 Hz and caused false alarms); faults start at 6 s.
- Minimum dwell between **controller** switches (LQG ↔ neural): 7 s ≥ the computed dwell-time bound (6.5 s worst
  case, `ft-stability`). Mode changes that keep LQG (nominal ↔ cautious ↔ safe) only move the airspeed target and
  are not delayed. An answer that would switch controllers inside the dwell time is dropped and the supervisor is
  asked again (outcome `blocked`); it is never applied later.
- Changed after the code review of 2026-09-29, before any complete benchmark: the dwell used to hold *every* switch
  and then apply the waiting answer up to 7 s late without re-asking, which penalised step-by-step escalation
  (cautious, then safe); the reaction time counted answers decided before the fault or never applied; the time
  slowed down stopped counting when a flight failed, so crashing early looked cheap. Now the reaction is the first
  mode change decided after the onset that took effect, and time slowed (and cost J) is averaged over surviving
  flights, with the paired Wilcoxon test on the flights both supervisors survived.

## 5. Simulator extensions

1. **Time-varying plant**: piecewise-constant parameter and airspeed schedules (fault ramps and autopilot
   deceleration quantised into small steps; discrete models cached per step).
2. **Controller bank with bumpless transfer**: all candidate controllers run every step on the same measurements
   and the same *applied* command history, so their estimators stay warm; the active one's command is applied.
3. **Gain scheduling in airspeed**: controllers re-design (cached) when the airspeed changes.
4. **Fault injection** for sensors (noise, bias), actuator (slew rate, effectiveness), structure (stiffness) and
   loop delay.
5. **Diagnostic features** for the situation report, all computable on board: dominant oscillation frequency and
   its change, damping estimate (log decrement), flap tracking error (command vs measured flap angle), estimator
   innovation level, measurement noise level, saturation, loop-timing health.

## 6. Supervisors compared (same seeds, same faults)

- **None** (stays `nominal`), **random**, **rules** (hand-written symptom → mode table for F1–F4),
  **LLM** (local `llama3.1:8b` by default; Claude switchable), **oracle** (knows the true fault; picks the best
  mode with hindsight, found by exhaustive evaluation): the upper bound.

Metrics: survival; reaction (time from fault onset to the first mode change decided after the onset that took
effect) and the mode chosen; mission cost (time spent slowed down) and cost J over surviving flights; switches, LLM
latency and fallbacks. Paired statistics over 20 seeds per fault (exact McNemar on survival; Wilcoxon on time slowed
over flights both supervisors survived); held-out faults reported separately.

## 7. Stability guarantee

For each mode, the closed loop is checked stable across the speed schedule. Using per-mode quadratic Lyapunov
functions, compute the average dwell time τ* above which any switching sequence is stable (Hespanha & Morse 1999),
and enforce a minimum dwell of max(τ*, the LLM's own decision interval) between controller switches. The theorem
is stated in continuous time; the discrete-time counterpart used here follows in one line (V ≤ μ^N λ^k V(0) after
k steps with N switches). Claim: *no sequence of LLM answers can destabilise the nominal closed loop* (small-signal,
healthy wing). Faults, saturation, the rate limit and airspeed transitions are outside this guarantee; the paper
says so. Tests check that the analysed matrices reproduce both simulators (LQG exactly; the neural policy to its
linearisation error, which shrinks with amplitude).

## 8. Risks (stated up front)

- The LLM may not beat good rules on F1–F4. The held-out faults and the oracle gap are then the interesting result.
- Diagnosis may be too hard from the report; the rules may get the same features, so the comparison stays fair.
- Airspeed changes make the plant time-varying; the dwell-time argument covers controller switching at a fixed
  speed, and slow speed changes are an assumption (gain scheduling).

## 9. Build order

1. Simulator: schedules, controller bank, gain scheduling, fault injection (+ tests).
2. Diagnostics and the extended situation report (+ tests).
3. Rules, random and oracle supervisors for the new modes; LLM prompt for the new modes.
4. `ft-benchmark` command, figures, paired statistics.
5. Dwell-time analysis command; paper-draft section generated from the results.

## 10. Results (2026-09-29)

Generated from `outputs/uav_flap/ft_benchmark.json`; full tables in Memory.md §4 and the paper draft §7.5.

- **LLM vs rules.** Pooled over faulty flights the LLM survived 27 % and the rules 48 % (significant, exact McNemar p = 0.0026). The LLM did better on stiffness_loss and worse on servo_slew, sensor_noise, turbulence_rise, loop_delay, flap_effectiveness.
- **Held-out faults** (in neither the rules nor the guidance): loop_delay: LLM 0 %, rules 100 %, oracle 100 %; flap_effectiveness: LLM 0 %, rules 10 %, oracle 100 %. They test different things: the loop delay appears as a number in the report (a flight computer times its own loop), so it tests acting on an explicit indicator nobody wrote a rule for; flap damage shows only through its symptoms.
- **False alarms** (time slowed down in fault-free flights): oracle 0.0 s, none 0.0 s, rules 2.8 s, random 11.7 s, llm 0.0 s.
- **Gap to the oracle** (share of faulty flights the oracle saves but the supervisor loses): none 81 %, rules 52 %, random 86 %, llm 73 %.
- **The structural-damage trap.** After stiffness loss, holding speed survived 100 % but a moderate slow-down only 0 %: the controllers are gain-scheduled for the healthy wing, which is below flutter at the lower speed while the damaged wing is not. A reflex of "anomaly, slow down a little" is wrong here.
