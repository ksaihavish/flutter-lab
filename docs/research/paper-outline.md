# Paper outline (corrected)

This replaces `legacy/docs/researchpaper_original.md`. That file's numbers were never
produced by any code (see `audit-2026-09.md`).

**Rule:** every number marked `⟨results.json: …⟩` must be copied from
`outputs/results.json` after running `python -m flutter_ai evaluate` (or the Phase 4 LLM
benchmark). Never type an estimate.

---

## Working titles (pick after the Phase 4 results)

- "Physics-Trained Neural Flutter Suppression with Language-Model Supervision"
- "Can a Language Model Supervise an Active Flutter Suppression System? A Pareto Analysis"
- "Differentiable-Simulation Neural Control and LLM Supervision for Aeroelastic Flutter"

Avoid "PINN" in the title unless a genuine physics-residual component is added (PRD §8).

## Abstract skeleton (150–250 words)

Problem (flutter, envelope extension) → gap (fixed-gain controllers, saturation, opaque
adaptation) → method (validated 2-DOF Theodorsen/Wagner model; neural policy trained by
differentiable simulation; LLM supervisor choosing among bounded modes from text
situation reports) → results (neural vs LQR ⟨results.json⟩; supervisor on/under the
Pareto front ⟨results.json⟩; LLM latency and fallback ⟨LLM benchmark⟩) → significance
(safe architecture for LLMs in control: slow loop, bounded action space, measured value).

## 1. Introduction
1. Flutter and active flutter suppression. Cite classic sources and wind-tunnel
   programmes you have actually read.
2. Limits of fixed-gain designs under saturation and changing conditions.
3. Learned controllers: promise and trust problems.
4. LLMs as supervisory agents: interpretability versus latency and reliability.
5. Contributions (only what the results support):
   - a validated, open benchmark (physics tests, p-k cross-check);
   - a physics-trained neural policy with zero-force equilibrium and rate penalty;
     comparison vs LQR (and vs LQG/H∞ after Phase 4.2);
   - a safe supervisory architecture plus a **Pareto test** for whether adaptation adds value;
   - an open-source implementation and interactive simulator.

## 2. Background and related work
- Theodorsen / Wagner / Jones aerodynamics; flutter mechanism (mode coalescence).
- Active flutter suppression (LQR/LQG, H∞, gain scheduling, adaptive control).
- Learning-based control and differentiable simulation.
- LLMs for control and robotics: do the literature search (Phase 4.7) before writing
  this section.

## 3. System model
- 3-DOF typical section with a servo-driven trailing-edge flap: EOM and Theodorsen flap
  aerodynamics (Architecture.md §2); state space; ZOH discretisation; 5 ms loop delay.
- Parameters: table from `results.json → params`.
- Validation: Hodges & Pierce benchmark; flap loads vs thin-airfoil theory; state-space vs p-k flutter speed
  ⟨results.json: flutter.state_space, flutter.pk_exact_theodorsen⟩. Figure 1 =
  `fig1_flutter_vg.png`.

## 4. Controllers
- Actuators and limits; sensors; aero-lag observer.
- LQR: discrete DARE; Bryson weights; aggressiveness → effort weight 10^(1−2a).
- Neural policy: architecture u = u_max·tanh(g(x) − g(0)); training (imitation →
  differentiable-physics fine-tuning with log(1+J), rate penalty, domain randomisation,
  best-checkpoint on held-out episodes); validation vs LQR ⟨models/neural_policy.npz
  meta.validation, meta.lqr_validation⟩. Figure = `fig*_training.png`.

## 5. Supervisory layer
- Situation features and report text (show one example report).
- Action space (four modes) and why forces are never exposed to the LLM.
- Rule baseline; Claude backend (structured output, fallback, call budget, latency).

## 6. Experiments
- Scenarios (`scenarios.py`) with the rationale for each.
- Metrics (`metrics.py`): peak, RMS, settling, effort, control rate, saturation, failures.
- Seeds and statistics.

## 7. Results
- Table: all cases × scenarios ⟨results.json: scenarios.*.cases⟩ (mean ± std).
- Velocity sweep (stability envelope) ⟨results.json: velocity_sweep⟩.
- **Delay robustness** ⟨results.json: delay_sweep⟩: neural vs LQR fixed for 5 ms vs LQR told the true delay.
- **Pareto trade-off (key figure)** ⟨results.json: tradeoff⟩: fixed-aggressiveness curves
  versus supervised points. State plainly whether each supervisor lies above, on or below
  the curve.
- LLM: latency distribution, fallback rate, cost, examples of reasoning.

## 8. Discussion
- Why the neural policy wins where it wins (saturation, turbulence statistics seen in
  training). Fairness caveats: LQR had no disturbance model, which is why LQG is needed.
- What the supervisor is good for (interpretability, operator trust) versus what it is
  not (raw performance, if the Pareto test is negative).
- Limitations: 2-DOF linear model, generalised-force actuation (until Phase 4.6),
  simulation only, Wagner gust approximation.

## 9. Conclusion and future work
Phase 7 items: experimental validation, nonlinear LCO, stability certificates, HIL.

## Reproducibility statement
Commit hash, `python -m flutter_ai evaluate`, the `results.json` metadata block, and the
policy metadata.

## References (verified so far; add only papers you have opened)
- Theodorsen, T. (1935). *General theory of aerodynamic instability and the mechanism of
  flutter.* NACA Report 496.
- Fung, Y.C. (1955/2008). *An Introduction to the Theory of Aeroelasticity.* Dover.
- Hodges, D.H., Pierce, G.A. (2011). *Introduction to Structural Dynamics and
  Aeroelasticity*, 2nd ed. Cambridge University Press.
- Jones, R.T. (1938). *Operational treatment of the nonuniform-lift theory in airplane
  dynamics.* NACA Report 667. Source of the two-lag Wagner approximation; **check the
  report number and year against the NASA Technical Reports Server before citing.**
- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks.
  *J. Comput. Phys.* 378, 686–707. (Cite only if you keep a PINN discussion.)
