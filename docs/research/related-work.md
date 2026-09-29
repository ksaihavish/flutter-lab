# Related work and novelty check (Phase 4.7)

*Compiled 2026-09-29 from searches of OpenAlex, arXiv, NASA NTRS and the web. Every entry below was opened;
the "Read" column says how far. **Abstract** means only the abstract was read: read the full paper before
citing anything beyond what its abstract says (Rules.md §1.7). Full texts were checked the same day for every
entry with a legal open copy, metadata confirmed against Crossref / arXiv, and the novelty search re-run (arXiv
API, OpenAlex from 2025, web): still no LLM supervision or decision-making in aeroelastic or flutter control.*

## 1. Summary for the paper

| Claim | Status |
|---|---|
| "First neural-network flutter suppression" | **False.** Wind-tunnel-tested neural flutter controllers exist since 2000 (NASA/Boeing; Politecnico di Milano) |
| "First learned controller that beats classical control under actuator limits" | **False.** Tang et al. (2020) did it with wind-tunnel validation |
| "First AI that chooses between flutter-suppression control laws" | **Misleading.** Scott & Pado (2000) used a neural network to schedule (interpolate) the coefficients of flutter-suppression control laws over Mach number and dynamic pressure, on the BACT wing in a wind tunnel; it did not choose among discrete laws. Do not claim "first AI in the loop of flutter control laws" |
| "First LLM supervisor over a bank of pre-verified controllers with dwell-time switching" | **False.** Papacharalampopoulos et al. (2026) did this for process systems: their LLM (cloud, called once per fault event) selects one of four bounded actions, with a risk-score rule as fallback and an average dwell-time bound (it overrode the rule once). Their text also calls the LLM "advisory and audit", which contradicts their own algorithm; cite the algorithm |
| "First LLM that *decides* supervisory mode / controller switches in active flutter suppression, evaluated with its measured on-board latency against rules, random and an oracle, on held-out faults" | **Not found** in these searches (re-run 2026-09-29 with full texts). This is the defensible, narrow novelty: the plant (fast, unstable without control), the on-board model with measured latency asked repeatedly, and the evaluation design. Still say "to our knowledge" |
| Stability of LLM-supervised switching | Not a new theorem: the average dwell-time argument (Hespanha & Morse 1999, continuous time; we use its discrete-time counterpart) applies, and Papacharalampopoulos et al. also compute such a bound. The contribution is computing it for the implemented controllers (checked against the simulator) and enforcing it |

## 2. Active flutter suppression: classical and learned

| Reference | What they did | How we differ | Read |
|---|---|---|---|
| Waszak, NASA/TP-1998-206270, *Modeling the BACT wind-tunnel model* | Equations of motion and measured parameters of the BACT pitch-and-plunge wing with a TE flap | Our `bact` preset and flutter validation are built on it | Full (earlier session) |
| Waszak, NASA/TM-97-206252, *BACT Simulation User Guide* | NASA's MATLAB/Simulink BACT model; TDT Dryden turbulence model; actuator nonlinearities (TE rate limit ±250 °/s, stated to be "educated guesses") | Source of our BACT rate limit | Full |
| Mukhopadhyay, IFA 1999 (NTRS 19990050943), *Transonic flutter suppression control law design, analysis and wind tunnel results* | Classical, **LQG** and minimax flutter suppression on BACT; tunnel tests raised the closed-loop flutter boundary to the 200 psf tunnel limit | LQG is the established baseline for this rig; our LQG follows the same idea | Full |
| Scott & Pado, J. Guid. Control Dyn. 23(6) 2000, doi:10.2514/2.4661 | Three neural-network flutter controllers on the **BACT** wing in the NASA TDT (1995, 200 Hz): one **scheduled the coefficients** of 56 classical point designs over Mach and dynamic pressure, one predictive, one inverse-model; the classical systems were significantly more robust than the neural predictive and inverse-model systems | Same wing as our `bact` preset. Their scheduler interpolates gains; ours is a language model choosing among bounded modes, with measured latency and a comparison with rules/random/oracle | Full (SPIE 3991 version, same title and abstract, NTRS 20040086574) |
| Bernelli-Zazzera et al., J. Guid. Control Dyn. 23(6) 2000, doi:10.2514/2.4671 | Adaptive recurrent neural networks (online identification + control) on an unstable wind-tunnel wing with LE and TE surfaces (per the abstract) | Online adaptation; we train offline through the physics and test against LQG. The draft only cites it at title level until the full text is read | Abstract (no open copy) |
| Mattaboni et al., J. Guid. Control Dyn. 32(4) 2009, doi:10.2514/1.40774 | Recurrent neural networks for flutter suppression of a three-surface transport aircraft (simulation) | Full aircraft; no LLM layer | Abstract (no open copy) |
| Tang et al., J. Vib. Control 24(22) 2018, doi:10.1177/1077546317750504 | Adaptive optimal neural-network control of airfoil flutter (online HJB); wind-tunnel comparison with an LPV optimal controller | They beat an LPV design in a tunnel; we find our memoryless policy does **not** beat a turbulence-aware LQG | Abstract (no open copy) |
| Tang et al., J. Vib. Control 27(3–4) 2021 (online 2020), doi:10.1177/1077546320929153 | Neural optimal control **under control-input constraints**, Lyapunov analysis, wind-tunnel validation vs LPV | Directly overlaps our old RQ1; cite as prior art | Abstract (no open copy) |
| Chen et al., AIP Advances 13(1) 2023, doi:10.1063/5.0130370 | Model-free deep RL (TD3) trained **in a wind tunnel** with circulation control; flutter amplitude −92 % | Hardware RL; we use a validated model and differentiable simulation | Full |

Also found but **not opened** (no abstract available): Mu et al., J. Sound Vib. 2022, doi:10.1016/j.jsv.2022.116916
(ML-based flutter suppression of a flying wing). Open it before citing.

## 3. Learning control through differentiable simulation

| Reference | What they did | Relevance | Read |
|---|---|---|---|
| Xu et al., ICLR 2022 (arXiv 2204.07137), SHAC | Policy learning with parallel differentiable simulation and a smooth critic | Method family of our neural policy training | Full |
| Heeg et al., ICRA 2025 (arXiv 2410.15979) | Quadrotor control learned in differentiable simulation; beats PPO in sample efficiency and training time | Same method family, aerial vehicle | Full |

## 4. LLMs in or around control loops

| Reference | What they did | How we differ | Read |
|---|---|---|---|
| Du & Chan, arXiv 2609.16599 (2026), survey | LLMs in networked control / CPS; the LLM as a slow supervisor over a fast, certified inner loop; latency acts as delay; names the lack of formal stability proofs as the field's central open problem and says stability across LLM-triggered mode switches "remain[s] largely unanalyzed" | Our architecture is an instance; we add a flutter application and a dwell-time bound for the implemented controllers | Full |
| Papacharalampopoulos et al., *Processes* 14(14) 2298 (2026), doi:10.3390/pr14142298 | Three-layer supervisory fault-tolerant control for process systems (4th-order coupled oscillator, 1-D thermal process; simulation): digital twin, controller bank pre-verified with CARE (LQI nominal / degraded / degraded with effort cap), average dwell-time bound 11.03 s (continuous time; tested dwell 0.2–6 s). A cloud LLM (Claude Sonnet, temperature 0), called once per fault event 0.4 s after detection, **selects** keep_nominal / switch_degraded / energy_save_mode / request_human_review; out-of-set answers keep the risk-score rule's choice; a failed call keeps the nominal controller. Compared with a rule-based supervisor; no random or oracle baseline; latency not modelled ("would require optimization for high-bandwidth mechanical systems") | **Closest prior work: the same architecture pattern.** Ours differs in the plant (unstable without control, faster than the LLM), an on-board model asked repeatedly with measured latency applied, and the evaluation (random / rules / oracle, 20 seeds with paired tests, held-out faults). Their abstract calls the LLM advisory; cite their Algorithm A1, where it sets the mode | Full |
| Vyas et al., arXiv 2606.28011 (2026) | Agentic LLM fault-tolerant control for process plants (GPT-4o-mini / 4.1-mini; mixing module and CSTR, simulated): monitoring, planning, simulation-validated recovery actions | Process plants; recovery actions validated before execution | Full |
| Vyas et al., arXiv 2606.31635 (2026), tutorial | LLM as a "constrained supervisory planner without execution authority"; every proposal checked by an external validator before actuation | Same safety idea (bounded, validated actions) | Full |
| Ahn et al., arXiv 2204.01691 (2022), SayCan | Skill selection by LLM score × value-function affordance (not the LLM alone) among **pre-trained skills** | Architectural analogue of "LLM chooses among controllers" (robotics) | Full |
| Chen et al., arXiv 2312.14950 (2023), TypeFly | LLM-driven drone control; tackles the latency of sequential token generation (up to 62 % faster), one of the key obstacles | Supports our finding that LLM latency rules out gust-level decisions | Full |
| Shibu et al., DAUS 2026 (arXiv 2602.01226), SkySim | LLM (cloud Gemini, planning latency 34–50 s) plans drone-swarm waypoints; a heuristic potential-field filter at 20 Hz enforces limits (not certified) | Same "LLM high level, safety layer low level" split | Full |
| Shou et al., arXiv 2607.26594 (2026) | LLM / small-LM agents that tune PID loops of chemical processes | LLMs acting on controllers offline; ours is online supervision (not cited in the draft) | Full |
| Yildirim et al., arXiv 2405.13547 (2024), HighwayLLM | LLM + RL planner for highway driving decisions | Another LLM-over-learned-controller example | Abstract |

## 5. Control-theory foundations

| Reference | Use in the paper | Read |
|---|---|---|
| Hespanha & Morse, IEEE CDC 1999, vol. 3, pp. 2655–2660, doi:10.1109/cdc.1999.831330 | Switching among stable linear systems is stable if switching is slow on average (average dwell time); hysteresis-based supervisors satisfy it. Stated in continuous time; the paper uses the discrete-time counterpart (one-line derivation in its Section 5) | Full (author copy) |
| Doyle, IEEE TAC 23(4) 1978, doi:10.1109/TAC.1978.1101812 | "Guaranteed margins for LQG regulators: there are none" (gain margin can be arbitrarily small). Explains the LQG fragility we observe with an actuator rate limit | Full |
| Theodorsen, NACA Report 496 (dated 1934; NACA 20th Annual Report, 1935) | Unsteady aerodynamics of the typical section with a flap (3 DOF; our model) | Full (NTRS scans 19800006788, 19930090935) |
| Hodges & Pierce, *Introduction to Structural Dynamics and Aeroelasticity* | Textbook benchmark reproduced in `tests/test_physics.py` | Owner to confirm the edition used |

## 6. Where the project fits

- The neural-policy part is **not novel** and, against a fair turbulence-aware LQG with an ideal actuator, it
  loses; with a realistic actuator rate limit the picture changes (see Memory.md §4). Present it as one of the
  inner-loop candidates, not as the contribution.
- The LLM part is novel **only in a narrow sense**: the decision-making LLM in flutter suppression, run on board
  (local model), with measured latency, a fair comparison against rules / random / oracle supervisors, and a
  dwell-time bound on what it can do. The architecture pattern itself (an LLM choosing among pre-verified
  controllers with dwell-time switching and a rule fallback) is Papacharalampopoulos et al.'s; the paper must say
  so and claim the plant, the on-board latency and the evaluation instead. Scott & Pado must be discussed as the
  earlier neural scheduler on the same BACT wing.

## 7. Still to search (before submission)

- Full texts of the four paywalled "Abstract" rows (Bernelli-Zazzera 2000, Mattaboni 2009, Tang 2018 and 2021) through a
  library; until then the draft cites them only for what their abstracts say.
- Gain-scheduled / LPV flutter suppression (the usual industrial baseline).
- Envelope protection and flutter-margin monitoring in flight test (for the airspeed-advice part of the fault-tolerant plan).
- Any 2026 LLM + aeroelasticity preprints (re-run the arXiv queries just before submission).
