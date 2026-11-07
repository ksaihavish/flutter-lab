# PRD: Flutter Lab

*Product requirements: what we are building, for whom, and what "done" means.*
Last updated: 2026-09-28 · Version 0.3.0

---

## 1. Summary

**Flutter Lab** is a research platform and interactive simulator for **active flutter
suppression** on an aircraft wing section. It combines:

1. **A validated physics model.** A typical wing section with unsteady Theodorsen/Wagner
   aerodynamics: 3-DOF (plunge, pitch and a servo-driven trailing-edge flap) by default, or
   2-DOF. It includes a realistic 5 ms control-loop delay.
2. **Feedback controllers.** A classical LQR baseline and a **neural policy trained
   through the differentiable physics model**.
3. **A slow supervisory layer.** Rule-based or **LLM-based (Claude)**. It reads a
   plain-language situation report and chooses the controller's operating mode
   (efficient ↔ emergency). It never commands forces directly.
4. **A web app.** It animates the wing in real time and compares controllers
   side by side.

The immediate goal is a **publishable, reproducible research paper**. The longer-term
goal is a **commercial-grade engineering tool** (see `Phases.md`, Phases 6–8).

## 2. Problem

Flutter is a self-excited aeroelastic instability that can destroy a wing within
seconds once airspeed passes the flutter speed. Active flutter suppression can extend
the safe flight envelope, but:

- classical controllers (LQR, gain scheduling) are designed for one condition and
  handle actuator saturation poorly;
- learned controllers are often unvalidated and opaque;
- operators and engineers cannot easily see *why* a control system changed behaviour.

**Research questions**

| ID | Question | How we answer it | Status (2026-09-29) |
|----|----------|------------------|---------------------|
| RQ1 | Can a neural policy trained through a validated physics model outperform LQR under actuator saturation, turbulence and loop delay? | Benchmark scenarios, 20 seeds, paired statistics, robustness and ablations, against LQR **and a turbulence-aware LQG** | Answered: beats plain LQR, not LQG; wins only in some regimes (delay, some actuator limits) |
| RQ2 | Does an LLM supervisor add value *beyond* choosing a good fixed trade-off? | A supervisor only adds value if its point lies **below** the fixed-aggressiveness curve | Answered for effort modes: **no** (latency ≫ gust time scale) |
| RQ2′ | Under slowly developing faults, does an on-board LLM choosing among pre-verified modes beat no, random and rule-based supervision, and how close is it to an oracle, also on held-out faults? | `ft-benchmark`: 6 faults (2 held out), 20 seeds, exact McNemar and Wilcoxon tests | Answered 2026-09-29: LLM 27 %, rules 48 %, oracle 100 % of faulty flights survived (LLM vs rules: exact McNemar p = 0.0026); Memory.md §4 |
| RQ3 | What do LLM latency and failures cost, and are the safety mechanisms sufficient? | Measured latency applied in simulation time; fallback rate; one question at a time; dwell-time stability bound | Latency measured (local 8B model: seconds); dwell-time guarantee proven (small-signal) |
| RQ4 | Is the supervisor's reasoning useful to a human (interpretability)? | Qualitative analysis of the decision log; optional user study | Open: decision logs are saved (`ft_benchmark.json → decisions`) |

## 3. Target users

| User | Need | Priority |
|------|------|----------|
| **The researcher (you)** | Reproducible experiments, honest figures, fast iteration | Primary, now |
| **Paper reviewers / readers** | Re-run everything with one command; check the physics | Primary, now |
| **Students and educators** in aeroelasticity and controls | See flutter happen and see control fix it; play with parameters | Secondary |
| **Aerospace / UAV engineers** (future) | Evaluate control strategies on their own wing models before wind-tunnel tests | Future, commercial |

## 4. Goals and non-goals

**Goals**
- Physics that matches published references (checked by automated tests).
- Every number in the paper comes from `python -m flutter_ai evaluate`.
- A controller comparison that is fair: same plant, same actuator limits, same sensors.
- An LLM integration that is safe by construction and whose value is measured, not
  assumed.
- One-click local start-up, plus a web app that shows the simulation.

**Non-goals (for now)**
- Flight-certified software, or real-time embedded deployment.
- Full-aircraft / CFD aeroelasticity (the 2-DOF section is the research vehicle).
- Using the LLM in the fast (200 Hz) control loop.

## 5. Features

### P0 (done in v0.3.0)

| Feature | Acceptance criteria |
|---------|---------------------|
| Validated aeroelastic model (2-DOF and 3-DOF flap) | Flutter speed within 1.5 % of an independent p-k analysis; Hodges & Pierce divergence speed equals √8·bω_α; flap loads match independent thin-airfoil theory; tests pass |
| Realistic loop | Flap servo actuation (±15°); 1-step (5 ms) input delay applied to every controller; delay-margin analysis |
| Dataset intake | `data/README.md` spec; `check-data` validates files and compares predicted vs measured flutter speed |
| Honest closed-loop simulator | Exact ZOH plant; actuator saturation; no state clipping; runs that exceed structural limits are flagged with a failure time |
| LQR baseline | Discrete-time, delay-aware LQR (no chatter at 200 Hz) |
| Neural policy | Trained through the physics (imitation, then differentiable-simulation fine-tuning with delay randomisation); zero force at equilibrium; no chatter; stable for delays 0–2 steps at the design points; NumPy inference |
| Supervisor | Four modes; rule-based and Claude backends; enum-constrained JSON; fallback, call budget, latency applied |
| Benchmark | Four scenarios × five cases × seeds → `outputs/results.json` + figures, in about 15 s |
| Web app | Animated true-scale wing lanes, synchronised charts with crosshair, results table, V-g chart, supervisor log, CSV export, light/dark mode |
| One-click start | `start.bat` (Windows) / `start.sh` (macOS/Linux) |

### P1 (next, needed for the paper): see `Phases.md` Phase 4
- Run the Claude supervisor experiments; report latency and fallback statistics.
- Stronger classical baselines: LQG with a turbulence model, H∞, gain-scheduled LQR.
- Robustness: parameter uncertainty (Monte Carlo), sensor noise and delay.
- Validation against a real wind-tunnel dataset (the user is supplying data).
- Nonlinear structure (cubic or freeplay pitch stiffness → limit-cycle oscillations).

### P2 (product): see `Phases.md` Phases 5–8
- Live streaming simulation (WebSocket) with interactive gust injection.
- A user-supplied wing model (import modal data); saved and shareable scenarios.
- Accounts, job queue, cloud deployment.

## 6. Success metrics

- **Scientific:** all tests green; results regenerate bit-for-bit from a clean checkout;
  every figure and table traceable to `results.json`.
- **Paper:** RQ1–RQ3 answered with statistics (mean ± std over seeds, significance where
  claimed); limitations stated.
- **Usability:** a new user goes from download to animated simulation in under
  5 minutes with no manual steps.

## 7. Constraints and principles

- **Integrity first.** No hand-typed results. Negative results (for example, "the rule
  supervisor does not beat a fixed trade-off") are reported, not hidden.
- **Safety architecture.** The LLM chooses among bounded modes; it never produces
  actuator commands.
- **Reproducibility.** Seeds, model metadata (training config, parameter fingerprint)
  and software versions are stored with results.
- **Small dependency footprint.** Serving does not need PyTorch.

## 8. Open questions (decide with your advisor)

1. **Naming.** The neural controller is *physics-trained* (differentiable simulation),
   not a classical PINN. Rename the paper's method, or add a genuine PINN component
   (for example, PINN-based system identification of the wing from sensor data)?
2. **Which LLM claims to make?** The current evidence suggests value in
   interpretability, not raw performance. Frame the paper accordingly unless the Claude
   experiments show otherwise.
3. **Target venue.** This decides the page budget and how much of the product story
   belongs in the paper.
