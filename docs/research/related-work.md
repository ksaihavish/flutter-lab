# Related work and novelty check (Phase 4.7)

*Compiled 2026-09-29 from searches of OpenAlex, arXiv, NASA NTRS and the web. Every entry below was opened;
the "Read" column says how far. **Abstract** means only the abstract was read: read the full paper before
citing anything beyond what its abstract says (Rules.md §1.7).*

## 1. Summary for the paper

| Claim | Status |
|---|---|
| "First neural-network flutter suppression" | **False.** Wind-tunnel-tested neural flutter controllers exist since 2000 (NASA/Boeing; Politecnico di Milano) |
| "First learned controller that beats classical control under actuator limits" | **False.** Tang et al. (2020) did it with wind-tunnel validation |
| "First AI that chooses between flutter-suppression control laws" | **False.** Scott & Pado (2000) scheduled flutter-suppression control laws with a neural network in a wind tunnel |
| "First LLM supervisor over a bank of pre-verified controllers with dwell-time switching" | **False in general.** Papacharalampopoulos et al. (2026) did this for process plants, but their LLM only writes audit records; switching is decided by a risk score |
| "First LLM that *decides* supervisory mode / controller switches in active flutter suppression, evaluated with its measured on-board latency against rules and an oracle" | **Not found** in these searches. This is the defensible, narrow novelty. Still say "to our knowledge" |
| Stability of LLM-supervised switching | Not a new theorem: dwell-time stability (Hespanha & Morse 1999) applies. The contribution is applying it to bound what any LLM answer can do |

## 2. Active flutter suppression: classical and learned

| Reference | What they did | How we differ | Read |
|---|---|---|---|
| Waszak, NASA/TP-1998-206270, *Modeling the BACT wind-tunnel model* | Equations of motion and measured parameters of the BACT pitch-and-plunge wing with a TE flap | Our `bact` preset and flutter validation are built on it | Full (earlier session) |
| Waszak, NASA/TM-97-206252, *BACT Simulation User Guide* | NASA's MATLAB/Simulink BACT model; TDT Dryden turbulence model; actuator nonlinearities (TE rate limit ±250 °/s, stated to be "educated guesses") | Source of our BACT rate limit | Full |
| Mukhopadhyay, IFA 1999 (NTRS 19990050943), *Transonic flutter suppression control law design, analysis and wind tunnel results* | Classical, **LQG** and minimax flutter suppression on BACT; tunnel tests raised the closed-loop flutter boundary to the 200 psf tunnel limit | LQG is the established baseline for this rig; our LQG follows the same idea | Full |
| Scott & Pado, J. Guid. Control Dyn. 2000, doi:10.2514/2.4661 | Three neural-network flutter controllers tested in the NASA TDT: one **scheduled flutter-suppression control laws**, one predictive, one inverse-model; "first experimental applications of neural networks to flutter suppression" | Their scheduler is a trained network; ours is a language model with explicit reasoning, a bounded mode set, measured latency and a fair comparison with rules/oracle | Abstract |
| Bernelli Zazzera et al., J. Guid. Control Dyn. 2000, doi:10.2514/2.4671 | Adaptive recurrent neural networks (online identification + control) on an unstable wind-tunnel wing with LE and TE surfaces | Online adaptation; we train offline through the physics and test against LQG | Abstract |
| Mattaboni et al., J. Guid. Control Dyn. 2009, doi:10.2514/1.40774 | Recurrent neural networks for flutter suppression of a three-surface transport aircraft (simulation) | Full aircraft; no LLM layer | Abstract |
| Tang et al., J. Vib. Control 2018, doi:10.1177/1077546317750504 | Adaptive optimal neural-network control of airfoil flutter (online HJB); wind-tunnel comparison with an LPV optimal controller | They beat an LPV design in a tunnel; we find our memoryless policy does **not** beat a turbulence-aware LQG | Abstract |
| Tang et al., J. Vib. Control 2020, doi:10.1177/1077546320929153 | Neural optimal control **under control-input constraints**, Lyapunov analysis, wind-tunnel validation vs LPV | Directly overlaps our old RQ1; cite as prior art | Abstract |
| Chen et al., AIP Advances 2023, doi:10.1063/5.0130370 | Model-free deep RL trained **in a wind tunnel** with circulation control; flutter amplitude −92 % | Hardware RL; we use a validated model and differentiable simulation | Abstract |

Also found but **not opened** (no abstract available): Mu et al., J. Sound Vib. 2022, doi:10.1016/j.jsv.2022.116916
(ML-based flutter suppression of a flying wing). Open it before citing.

## 3. Learning control through differentiable simulation

| Reference | What they did | Relevance | Read |
|---|---|---|---|
| Xu et al., arXiv 2204.07137 (2022), SHAC | Policy learning with parallel differentiable simulation and a smooth critic | Method family of our neural policy training | Abstract |
| Heeg et al., arXiv 2410.15979 (2024) | Quadrotor control learned in differentiable simulation; outperforms model-free RL | Same method family, aerial vehicle | Abstract |

## 4. LLMs in or around control loops

| Reference | What they did | How we differ | Read |
|---|---|---|---|
| Du & Chan, arXiv 2609.16599 (2026), survey | LLMs in networked control / CPS; the LLM as a slow supervisor over a fast certified inner loop; missing stability proofs are the central open problem; latency acts as delay | Our architecture is an instance; we add a flutter application and a dwell-time argument | Abstract |
| Papacharalampopoulos et al., *Processes* 2026, doi:10.3390/pr14142298 | Three-layer supervisory fault-tolerant control: digital twin, **dwell-time-constrained switching over pre-verified stable controllers**, optional bounded LLM producing audit records | **Closest prior work.** Their LLM explains; ours decides (and is compared with rules and an oracle). Different domain (process plants vs aeroelastic) | Abstract |
| Vyas et al., arXiv 2606.28011 (2026) | Agentic LLM fault-tolerant control for process plants: monitoring, planning, simulation-validated recovery actions | Process plants; recovery actions validated before execution | Abstract |
| Vyas et al., arXiv 2606.31635 (2026), tutorial | LLM as a constrained supervisory planner; every proposal checked by an external validator before actuation | Same safety idea (bounded, validated actions) | Abstract |
| Ahn et al., arXiv 2204.01691 (2022), SayCan | LLM chooses among **pre-trained skills** grounded by value functions | Architectural analogue of "LLM chooses among controllers" (robotics) | Abstract |
| Chen et al., arXiv 2312.14950 (2023), TypeFly | LLM-driven drone control; **latency** of token generation is the main obstacle | Supports our finding that LLM latency rules out gust-level decisions | Abstract |
| Shibu et al., arXiv 2602.01226 (2026), SkySim | LLM plans drone-swarm waypoints; a potential-field safety filter enforces limits | Same "LLM high level, safety layer low level" split | Abstract |
| Shou et al., arXiv 2607.26594 (2026) | LLM / small-LM agents that tune PID loops of chemical processes | LLMs acting on controllers offline; ours is online supervision | Abstract |
| Yildirim et al., arXiv 2405.13547 (2024), HighwayLLM | LLM + RL planner for highway driving decisions | Another LLM-over-learned-controller example | Abstract |

## 5. Control-theory foundations

| Reference | Use in the paper | Read |
|---|---|---|
| Hespanha & Morse, IEEE CDC 1999, doi:10.1109/cdc.1999.831330 | Switching among stable linear systems is stable if switching is slow on average (average dwell time); hysteresis-based supervisors satisfy it. Basis of our stability argument for LLM-selected modes | Abstract |
| Doyle, IEEE TAC 1978, doi:10.1109/TAC.1978.1101812 | "Guaranteed margins for LQG regulators: there are none." Explains the LQG fragility we observe with an actuator rate limit | Abstract (the whole abstract) |
| Theodorsen, NACA Report 496 (1935) | Unsteady aerodynamics of the typical section (our model) | **Not opened**: open the NTRS scan before citing |
| Hodges & Pierce, *Introduction to Structural Dynamics and Aeroelasticity* | Textbook benchmark reproduced in `tests/test_physics.py` | Owner to confirm the edition used |

## 6. Where the project fits

- The neural-policy part is **not novel** and, against a fair turbulence-aware LQG with an ideal actuator, it
  loses; with a realistic actuator rate limit the picture changes (see Memory.md §4). Present it as one of the
  inner-loop candidates, not as the contribution.
- The LLM part is novel **only in a narrow sense**: the decision-making LLM in flutter suppression, run on board
  (local model), with measured latency, a fair comparison against rules / random / oracle supervisors, and a
  dwell-time bound on what it can do. The Papacharalampopoulos et al. and Scott & Pado papers must be discussed
  explicitly as the closest prior work.

## 7. Still to search (before submission)

- Full texts of the "Abstract" rows above that the paper leans on (Scott & Pado; Papacharalampopoulos et al.; Tang 2020).
- Gain-scheduled / LPV flutter suppression (the usual industrial baseline).
- Envelope protection and flutter-margin monitoring in flight test (for the airspeed-advice part of the fault-tolerant plan).
- Any 2026 LLM + aeroelasticity preprints (re-run the arXiv queries just before submission).
