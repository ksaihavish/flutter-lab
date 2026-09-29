# Physics-informed advisor: plan (started 2026-09-29)

*Owner's direction after the fault-tolerant results (LLM 27 % vs rules 48 % of faulty flights survived): give the
supervisor physics, not symptoms (idea 1), and map when a slow supervisor can protect a fast, unstable system
(idea 2). Owner's framing: automatically keep an actively controlled wing out of flutter when the aircraft itself
changes, before it becomes catastrophic, with a guarantee that the automation cannot make things worse.*

## Why the LLM lost (to be tested, not assumed)

It rarely acted (changed mode in 2 of 6 fault types; median reaction 10-13 s vs rules 1.5-7 s). Candidate causes:
latency (≈ 4.4 s per answer), the prompt's bias to stay nominal, memoryless snapshots, model size, and symptom-only
reports. Ablations: `ft-benchmark --no-latency --interval 2 --tag no_latency_2s` (judgement without the delay;
running 2026-09-29), then a stronger model (Claude backend), then a report with history.

## Step 1: on-board flutter-margin estimator (`fault_tolerance/margin.py`, v1 built)

- Grey-box prediction-error identification over the last 4 s: pitch stiffness and flap effectiveness as factors of
  the healthy values; lag states rebuilt from measured motion; commands through the known delay and slew limit.
- Self-calibrated against the same fit on the healthy wing at take-off (the unmeasured turbulence biases the fit).
- Outputs: flutter speed of the identified wing (+1σ), margin to the current airspeed, and for every mode the
  spectral radius of that mode's controller (designed for the healthy wing) on the identified wing.
- Plunge stiffness is held at its healthy value: in closed loop it trades off against pitch stiffness (both mainly
  set the coupled mode's frequency), and fitting it drifted by up to +40 % on an unchanged wing.
- Development on seed 1000 (never the test seeds 0-19). `tests/test_margin.py` checks: healthy wing recognised
  (flutter speed within 3 %, all modes stable); after structural damage the pitch stiffness is identified and the
  "slow-down trap" is predicted (nominal stable, cautious and robust unstable, safe stable).
- **Limits of v1:** linear prediction. Flap damage kills the wing through saturation (loss of authority), which it
  cannot see; servo, sensor-noise, delay and turbulence faults do not change the identified physics.

## Step 2: estimator v2 (`margin.estimate_v2`, built; v1 and its pre-registered users unchanged)

v1 is linear and blind to authority: flap damage kills the wing through saturation, and loop delay, turbulence and
sensor noise do not change the identified physics. v2 adds, from the same 4 s window:

- **Control authority.** For every mode, the stationary RMS of the flap command and of its rate at the mode's target
  airspeed, from the discrete Lyapunov equation of the closed loop (bank state of `stability.mode_systems`) augmented
  with the simulator's AR(1) gust (correlation time 0.05 s, entering through the plant's gust column) and white sensor
  noise on every measurement. Authority ratio = 3 sigma / limit, the larger of command (`u_max`) and rate
  (`u_rate_max`); >= 1 means saturation expected. `tests/test_margin_v2.py` checks the Lyapunov statistics against a
  Monte Carlo run of the same loop.
- **Turbulence and sensor-noise estimates.** Two statistics of the window, both relative to the take-off window
  (assumed at the design turbulence, 0.25 m/s, without sensor noise): the RMS of the residual's gust component
  (per-sample least squares of the identified model's one-step residual on its gust column, jointly with an actuator
  column so that a flap which does not follow its commands is not read as turbulence), and a noise-equivalent level
  of h and alpha (mean square of the 4th difference / C(8,4), exactly s^2 for white noise; the diagnostics' 2nd
  difference kept a motion share that wandered by about ±20 % between healthy windows). Both rise with turbulence;
  only the second rises strongly with noise; solving the two relations separates them (the residual alone reads
  sensor noise as a turbulence rise).
- **Measured loop delay.** The flight computer knows its own timing: the fit rebuilds the applied commands with the
  per-sample delay history (a single value over a window that spans a delay change biased the fit), and every
  closed-loop prediction uses `mode_matrices(..., delay=d)` (d-sample delayed plant input, controllers keep their
  1-sample design). `tests/test_margin_v2.py` checks that the delayed matrices reproduce `ft_simulate` step by step for
  d = 2 and 3 (LQG exactly, neural as a linearisation).
- **Robust margins, in the spirit of NASA's Flutterometer** (Lind & Brenner, J. Aircraft 37(6), 2000,
  doi:10.2514/2.2719; NASA/TM-1997-207056, NTRS 19980017615: worst-case flutter margins with respect to an uncertainty
  description kept consistent with flight data). Here: the worst-case flutter speed and per-mode spectral radius over
  the identification's joint 95 % confidence ellipse (12 points on its boundary plus the point estimate). The
  covariance is the Gauss-Newton one inflated by the smallest factor that contains a sandwich (HAC, Bartlett weights,
  50 lags) covariance, because the gust makes the residuals coloured; the take-off calibration's own uncertainty is
  added (delta method for the ratio). **Not mu-analysis:** parametric uncertainty in two physical factors, worst case
  by sampling, no unmodelled dynamics.
- **Admissibility** (`MarginEstimateV2` docstring): a mode is stable on a wing if its loop is stable at every transit
  airspeed where that wing needs control (at or above its open-loop flutter speed; the current airspeed is excluded
  because every mode starts there, a mode that holds speed is judged where it holds); authority is required only
  where the identified wing needs control (below flutter a saturating controller chatters, the wing does not
  flutter). Admissible = stable on every sampled wing and enough authority (point estimate); also reported on the
  point estimate alone.
- **Supervisor `FTPhysicsRuleSupervisorV2`** (`rules+physics_v2`): symptom rules propose; the proposal is kept if
  admissible, otherwise the least disruptive admissible mode; if none is robustly admissible, the same choice on the
  point estimate (the 95 % region after structural damage contains wings on which no mode is stable, and 'safe' then
  is not certified either); if none at all, 'safe'. `strict=True` (`rules+physics_v2_strict`) skips the point-estimate
  tier, as a first design stated.
- `format_physics` shows the v2 lines only for v2 estimates; v1 reports are byte-identical.

**Validation:** `python -m flutter_ai margin-eval --preset uav_flap` (development seeds 1000-1004 only; reserved seeds
are refused) → `outputs/uav_flap/margin_eval.json` + `fig15_margin.png`: estimated vs true flutter speed over time
for every fault (truth from `faults.plant_at` + `flutter_analysis`), turbulence, noise and delay estimates vs truth,
predicted vs measured flap command statistics (on-board prediction and truth-model prediction), per-mode predictions
of v1 and v2 vs the oracle's fixed-mode outcomes, warning time before the nominal flight fails and false alarms, and a
small comparison of rules, rules + physics (v1) and rules + physics v2 (tiered and strict) on the same flights. The
numbers are in that file only; Memory.md §4 quotes them.

**Known limits of v2** (see the JSON for how often they matter): the 4 s window lags a ramping fault, so flap damage
is flagged only near full severity; sensor noise biases the identified pitch stiffness upward (errors in the
regressors), which raises the estimated flutter speed; the prediction is linear (it says saturation is coming, not
what the saturated loop then does); a growing delay can make a mode admissible that the grown fault then breaks
(the neural mode at 10 ms, which saturates at 15 ms), and a controller switch then locks in for the 7 s dwell time.

**Findings on the development seeds** (`margin_eval.json`; tables in Memory.md §4): as a *predictor*, v2 warns before
the nominal flight fails for the faults v1 is blind to and misses fewer mode failures than v1, but it is more
conservative. As a *supervisor*, `rules+physics_v2` did not beat `rules+physics` (v1): it gains on turbulence rise and
flap damage, and loses on the structural-damage trap (the identified wing is too soft, so holding speed is no longer
certified) and on the growing loop delay (it picks the neural mode, which the grown delay then saturates while the
dwell time locks it in). This is a negative result for the supervisor as designed, not for the estimator's
measurements; the next iteration should be judged on fresh seeds.

## Next

1. Authority margin (v2) and the estimator benchmark command: built (Step 2).
2. **Report integration:** add the margin block to the supervisor report; compare, on **fresh test seeds**
   (e.g. 100-119, untouched so far): rules, rules + margin (a physics rule), LLM, LLM + margin, oracle. v2
   (`rules+physics_v2`, `margin_version=2`) is not part of that pre-registered comparison; if it is to be tested, it
   needs its own fresh seeds.
3. **Time-scale map (idea 2):** sweep fault growth time and supervisor latency; where does each supervisor keep up?
4. Between flights (owner's continual-learning idea): the LLM reviews logs and proposes training scenarios; a
   retrained policy is accepted only if it passes the full benchmark and the stability checks (verification gate).
