# Fault-tolerant LLM supervision: study design

*Written 2026-09-29, after Phase 4. Approved by the owner ("build the fault-tolerant plan"). This is the design;
Memory.md tracks progress.*

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

Held-out faults appear neither in the prompt's guidance nor in the rules: they test whether the LLM generalises
from symptoms, which is the only way an LLM could beat good rules.

## 4. Action space (fixed, pre-verified)

| Mode | Inner loop | Airspeed |
|---|---|---|
| `nominal` | LQG | hold |
| `robust` | Neural | hold |
| `cautious` | Neural | slow to 0.9 × nominal U_F (autopilot, limited deceleration) |
| `safe` | LQR (high rate penalty) | slow to 0.8 × nominal U_F |

The LLM chooses a mode name from this enum, as today. It never outputs gains, forces or speeds.

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

Metrics: survival, cost J, time from fault onset to the first correct mode, mission cost (time spent slowed down),
switches, LLM latency and fallbacks. Paired statistics over ≥ 20 seeds per fault; held-out faults reported separately.

## 7. Stability guarantee

For each mode, the closed loop is checked stable across the speed schedule. Using per-mode quadratic Lyapunov
functions, compute the average dwell time τ* above which any switching sequence is stable (Hespanha & Morse 1999),
and enforce a minimum dwell of max(τ*, the LLM's own decision interval). Claim: *no sequence of LLM answers can
destabilise the nominal closed loop*. Faults are outside this guarantee; that is what the supervisor is for, and
the paper must say so.

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
