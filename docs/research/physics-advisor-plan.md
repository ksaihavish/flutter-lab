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

## Next

1. **Authority margin (v2):** predicted command and command-rate RMS of each mode under the estimated turbulence
   (discrete Lyapunov on the closed loop with the AR(1) gust), relative to the flap's limits; turbulence intensity
   from the fit residual relative to the healthy baseline; include the measured loop delay in the closed loop.
2. **Estimator benchmark command:** estimate vs truth over time for every fault and dev seed → JSON + figure.
3. **Report integration:** add the margin block to the supervisor report; compare, on **fresh test seeds**
   (e.g. 100-119, untouched so far): rules, rules + margin (a physics rule), LLM, LLM + margin, oracle.
4. **Time-scale map (idea 2):** sweep fault growth time and supervisor latency; where does each supervisor keep up?
5. Between flights (owner's continual-learning idea): the LLM reviews logs and proposes training scenarios; a
   retrained policy is accepted only if it passes the full benchmark and the stability checks (verification gate).
