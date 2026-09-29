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

## Learned fault diagnoser (fast encoder, built 2026-09-29)

Owner's direction: the on-board LLM is too slow and too passive to supervise faults; a fast learned model should
analyse the on-board data, and an LLM should only explain. Built in `fault_tolerance/diagnoser*.py`; commands
`train-diagnoser` and `eval-diagnoser`. Every number lives in `models/diagnoser_uav_flap.json` (training, validation,
thresholds) and `outputs/uav_flap/diagnoser_eval.json` (evaluation); figure `outputs/uav_flap/fig14_diagnoser.png`.

- **Input** (every 0.5 s, last 2.5 s of data, only on-board information): one feature vector: the nominal model's
  one-step residual per rate channel (with each sample's airspeed and the servo's known slew limit), high-frequency
  content and RMS per displacement, command rate, one Gauss-Newton step of margin.py's pitch-stiffness fit, pitch
  frequency shift, saturation, peak motion, energy trend, and context (airspeed, speed change, neural share), scaled
  by the take-off baseline. Features rather than raw signals: the signatures are known physics and the model stays
  tiny. The measured loop timing is deliberately *not* an input (a one-line check catches it), so the held-out loop
  delay tests its dynamic symptoms.
- **Model** (pure NumPy inference; torch only in training): encoder to an 8-d embedding, a multi-label head (sigmoid
  per known fault: servo_slew, stiffness_loss, sensor_noise, turbulence_rise; combinations allowed) and a decoder.
  Unknown-fault score = reconstruction error, chosen before any evaluation; a Mahalanobis distance in the embedding
  is reported as a secondary score only. A detection counts when it holds at two consecutive diagnoses.
- **Data** (`diagnoser_data.py`): randomised 30 s flights with `ft_simulate`: healthy, one, two or three simultaneous
  faults (`faults.CompositeFault`), random onset, growth time and size, three ways of flying (no supervisor, the
  rules, a forced mode change) so slowed-down and neural-controller flights occur with and without faults. Disjoint
  seed ranges >= 1000 for train / validation / evaluation. **loop_delay and flap_effectiveness never appear in
  training or calibration** (checked in code and tests); they are only in the evaluation split `eval_held_out`.
  `eval_benchmark` repeats the fault-tolerant benchmark's conditions.
- **Calibration** (validation flights only): probability threshold 0.5 (not tuned); unknown-fault threshold such that
  at most 5 % of validation flights (healthy or known faults) raise a confirmed unknown alarm.
- **Integration hooks** (no supervisor uses them yet): `ft_simulate(..., diagnose_every_s=0.5)` fills
  `FTFeatures.diagnosis`; `diagnostics.format_diagnosis` appends a "LEARNED DIAGNOSIS" block to the report, so an
  LLM (or a person) can explain the model's output. A test checks that the on-board path computes exactly the
  features the model was trained on.
- **What the evaluation shows** (numbers in `diagnoser_eval.json`): sensor noise, stiffness loss and rising
  turbulence are recognised reliably within a few seconds of onset, also in combinations; a slower servo is the weak
  class, because it is physically latent until the flap is asked to move faster than the degraded limit (after a
  slow-down it can stay invisible); the known-fault alarms raise false alarms in healthy flights forced into the `safe`
  mode (a rare regime in training); the held-out loop delay mostly triggers the unknown alarm, while flap damage
  mostly does not (it resembles rising turbulence plus a slow servo), although nearly every held-out flight raises
  *some* alarm. The secondary Mahalanobis score does better on flap damage and worse on the loop delay, and it
  saturates far from the data (the tanh encoder bounds the embedding).
- **Next:** a supervisor that acts on the diagnosis (known fault -> the pre-verified response, unknown -> the most
  conservative mode) compared on fresh test seeds with rules and the LLM; more `safe`-mode and slow-servo training
  data; an LLM that only explains the diagnosis block.
