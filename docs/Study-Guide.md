# Study Guide: everything you need to understand and own Flutter Lab

*The goal of this guide: by the end, you can explain every file in this repository, defend every
result in the paper, and change any part of it yourself.*

Last updated: 2026-09-29

---

## 0. How to use this guide

- **Time:** about 16 weeks at 8–10 hours a week (§11 has a week-by-week plan). Faster if you already know some of it.
- **Method, in each track:**
  1. Read the *Why it matters here* part.
  2. Study the concepts with the listed resources.
  3. Open the listed code and read it line by line.
  4. Do the **exercises**. They use this repo, so you learn by changing real code.
  5. Answer the **self-check questions** out loud, without notes. If you can't, go back.
- **Rule:** never move on from a track until you pass its self-check. Understanding beats speed.
- **Keep a learning log** (`notes/` in your own fork, or a notebook). Write what you learned and what confused you.
- **Setup first:** run `start.bat -Test` (all tests should pass) and `start.bat` (the web app opens).

### Self-assessment: where are you now?

Rate yourself 0 (never heard of it), 1 (heard of it), 2 (can explain it), 3 (can implement it):

| Topic | 0–3 | Track |
|---|---|---|
| Git: branch, commit, pull request, merge | | 1 |
| Python functions, classes, dataclasses, NumPy arrays | | 1 |
| Matrices, eigenvalues and eigenvectors | | 2 |
| Ordinary differential equations, state-space form | | 2 |
| Complex numbers, e^(iωt) | | 2 |
| Mass-spring-damper vibration, natural frequency, damping ratio | | 3 |
| Lift, angle of attack, thin-airfoil theory | | 3 |
| What flutter is and why it happens | | 3 |
| Stability of a linear system (poles, eigenvalues) | | 4 |
| Continuous vs discrete time, sampling, delays | | 4 |
| LQR / Riccati equation | | 4 |
| Neural networks, backpropagation, PyTorch | | 5 |
| Training through a simulator (differentiable simulation) | | 5 |
| LLM APIs, structured outputs | | 6 |
| Unit tests, CI, Docker | | 7 |
| JavaScript, HTML canvas, HTTP APIs | | 7 |
| Mean ± std, confidence intervals, significance | | 8 |
| Writing and structuring a research paper | | 8 |

Everything rated 0–1 is your priority.

---

## 1. Map: which file needs which knowledge

| File | What it does | Tracks |
|---|---|---|
| `src/flutter_ai/params.py` | Wing parameters, presets (incl. NASA BACT), unit conversions | 2, 3 |
| `src/flutter_ai/aero.py` | Theodorsen function C(k), Jones approximation, flap coefficients T1–T14 | 2, 3 |
| `src/flutter_ai/model.py` | Equations of motion → state-space matrices A, B, G; exact discretisation | 2, 3, 4 |
| `src/flutter_ai/flutter.py` | Flutter/divergence speed from eigenvalues; independent p-k check | 2, 3 |
| `src/flutter_ai/gusts.py` | Gust and turbulence signals | 3, 8 |
| `src/flutter_ai/controllers/lqr.py` | Delay-aware discrete LQR; delay-margin analysis | 4 |
| `src/flutter_ai/controllers/base.py` | Controller interface, aero-lag observer, command history | 4 |
| `src/flutter_ai/controllers/neural.py` | Neural policy inference (NumPy), linearisation | 5 |
| `src/flutter_ai/training.py` | PyTorch training through the simulator | 5 |
| `src/flutter_ai/simulation.py` | The closed-loop simulator (delay, saturation, failure detection) | 4, 7 |
| `src/flutter_ai/metrics.py` | RMS, peak, settling, effort, command rate | 4, 8 |
| `src/flutter_ai/supervisor/*` | Situation reports, rule supervisor, Claude supervisor | 6 |
| `src/flutter_ai/experiments.py`, `plots.py` | Benchmark, sweeps, figures | 8 |
| `src/flutter_ai/validation.py`, `data.py` | Wind-tunnel comparison, dataset checks | 3, 8 |
| `src/flutter_ai/api/*` | FastAPI web API | 7 |
| `frontend/*` | Web app: animation, charts | 7 |
| `tests/*` | 70+ automated tests | 7 |
| `.github/workflows/ci.yml`, `Dockerfile` | CI and container | 7 |

---

## Track 1: Tools and Python (weeks 1–2)

**Why it matters here:** everything is Python + git. If the tools feel foreign, everything else is harder.

**Concepts**
- Terminal basics: paths, running commands, environment variables (`ANTHROPIC_API_KEY`).
- Virtual environments and `pip install -e ".[dev]"`; what `pyproject.toml` declares.
- Git: commit, branch, push, pull request, merge, merge conflicts. You saw a real conflict (Dependabot PR #4).
- Python: functions, classes, `@dataclass(frozen=True)`, type hints, modules/packages, `pathlib`.
- NumPy: arrays, shapes, slicing, `@` (matrix multiply), broadcasting, `np.linalg.eigvals`.

**Resources**
- *The Missing Semester of Your CS Education* (MIT, free online): shell, git, editors.
- *Pro Git* (free book, git-scm.com): chapters 1–3.
- Official Python tutorial (docs.python.org); NumPy "absolute beginners" guide (numpy.org).

**Read in the repo:** `params.py` (dataclasses, properties), `paths.py`, `__main__.py` (argparse CLI).

**Exercises**
1. Create a branch, add a preset `my_wing` to `PRESETS` with a different `f_alpha`, run
   `python -m flutter_ai analyze --preset my_wing`, then open a PR and let CI run.
2. In a Python shell: `from flutter_ai.params import get_preset; p = get_preset(); print(p.k_h, p.I_alpha)`.
   Derive these two numbers by hand from `mass`, `f_h`, `r_alpha_sq` and `b`.
3. Break a test on purpose (change a number in `aero.py`), run `python -m pytest`, read the failure, then revert.

**Self-check**
- What happens between `git push` and a green tick on GitHub?
- Why is `WingParams` frozen, and why does that matter for `lru_cache` in `model.py`?
- What is the shape of `continuous_state_space(get_preset(), 40.0).A`, and why?

---

## Track 2: Mathematics (weeks 2–4)

**Why it matters here:** flutter *is* an eigenvalue crossing into the right half-plane. Control design is linear algebra.

**Concepts**
- **Linear algebra:** matrix multiplication, inverse, eigenvalues/eigenvectors, what a complex eigenvalue pair means
  (an oscillation), spectral radius, positive-definite and symmetric matrices.
- **ODEs:** second-order systems `M q̈ + C q̇ + K q = f`; rewriting as first-order state space `ẋ = A x + B u`.
- **Linear systems:** the solution `x(t) = e^{At} x(0)`; stability ⇔ all eigenvalues have negative real part.
- **Matrix exponential and discretisation:** zero-order hold, `A_d = e^{A dt}` (used in `model.discretize`).
- **Discrete time:** stability ⇔ all eigenvalues inside the unit circle (|z| < 1); why a pole at z ≈ −1 alternates sign every step.
- **Complex numbers and frequency response:** `e^{iωt}`, reduced frequency `k = ωb/U`.
- **Convolution / Duhamel integral:** how a step response (the Wagner function) builds the response to any input.
- **Special functions, conceptually only:** Hankel/Bessel functions appear in the exact Theodorsen C(k).
  You don't need to derive them; know what C(k) represents.
- **Root finding:** bisection/Brent's method (`scipy.optimize.brentq` in `flutter.py`).

**Resources**
- Gilbert Strang, *Introduction to Linear Algebra* + MIT 18.06 lectures (OCW, free). Especially eigenvalues and the chapter on differential equations.
- 3Blue1Brown, *Essence of Linear Algebra* (YouTube, free): the intuition first.
- Gilbert Strang, *Differential Equations and Linear Algebra* (or the MIT 18.03 OCW course).

**Read in the repo:** `model.continuous_state_space`, `model.discretize`, `flutter._first_crossing`.

**Exercises**
1. By hand, write the 2×2 mass-spring system without aerodynamics as a 4×4 state matrix. Check with
   `continuous_state_space(get_preset("uav_section").with_overrides(rho=1e-12), 1.0).A`.
2. Compute `np.linalg.eigvals(A)` at U = 0.9 U_F and 1.1 U_F; explain the sign change of the real parts.
3. Show numerically that `discretize(ss, dt).Ad` equals `scipy.linalg.expm(ss.A * dt)` (a test already does this; read it).
4. Build a 2×2 discrete system with an eigenvalue at −0.9 and simulate 20 steps. Watch the sign alternate: this is the chatter bug from Memory.md.

**Self-check**
- Why does a complex pair of eigenvalues λ = σ ± iω mean an oscillation at ω rad/s growing or decaying at rate σ?
- What is the difference between |z| < 1 (discrete) and Re(λ) < 0 (continuous)? How are they related (z = e^{λ dt})?
- What does "positive definite" mean for a mass matrix, physically?

---

## Track 3: Structural dynamics, aerodynamics and aeroelasticity (weeks 4–7)

**Why it matters here:** this is the physics of the paper. Reviewers from aerospace will test you on it.

**Concepts**
1. **Vibration basics:** natural frequency ω = √(k/m), damping ratio ζ, modes, coupling through static unbalance `S_α`.
2. **Lagrange's equations:** deriving equations of motion from kinetic and potential energy (how NASA derived BACT's equations).
3. **Aerodynamics basics:** lift, angle of attack, lift-curve slope (2π per radian for a thin airfoil), aerodynamic centre (quarter chord), pitching moment.
4. **Thin-airfoil theory:** camber line, Glauert series, flap effectiveness. It is used in `tests/test_physics.py` to validate the flap loads.
5. **Unsteady aerodynamics:**
   - apparent mass (non-circulatory) vs circulatory loads;
   - Theodorsen's function C(k) (frequency domain);
   - the Wagner function (step response, time domain);
   - R.T. Jones' two-lag approximation, which creates the two aero-lag states z₁ and z₂;
   - the Küssner function for gusts (our documented simplification uses Wagner instead).
6. **The typical section:** 2-DOF (plunge h, pitch α) and 3-DOF with a trailing-edge flap β and Theodorsen's T-functions.
7. **Flutter:**
   - mode coalescence;
   - V-g and V-f diagrams (`fig1_flutter_vg.png`);
   - flutter speed and frequency;
   - divergence (a static instability);
   - the difference between the k, p and p-k methods (`flutter.pk_flutter_speed`).
8. **Real wings:**
   - strip theory and **modified strip theory** (measured `cl_alpha`, `x_ac`);
   - finite-span (3D) effects and aspect ratio;
   - **compressibility** (Mach number, Prandtl–Glauert, the transonic dip). This is why BACT's flutter pressure is 24 % high.
9. **Wind-tunnel testing:** pitch-and-plunge apparatus, dynamic pressure q = ½ρU², heavy-gas testing (R-12), why tunnels vary density.

**Resources**
- **Hodges & Pierce**, *Introduction to Structural Dynamics and Aeroelasticity* (2nd ed., Cambridge). Main textbook; chapters 1, 4 and 5 (the typical section, flutter, the example we reproduce).
- **Wright & Cooper**, *Introduction to Aircraft Aeroelasticity and Loads* (Wiley). Very readable, including flutter and control surfaces.
- Bisplinghoff, Ashley & Halfman, *Aeroelasticity* (Dover), and Fung, *An Introduction to the Theory of Aeroelasticity* (Dover). Classic references.
- J.D. Anderson, *Fundamentals of Aerodynamics*: thin-airfoil theory, compressibility (Prandtl–Glauert).
- Theodorsen (1935), NACA Report 496: the original paper (free on NASA NTRS). Read the introduction and look at the equations you implemented.
- Waszak, NASA/TP-1998-206270 (BACT). Links are in `data/README.md`. You will cite it.

**Read in the repo:** `aero.py`, `model.py` (the whole docstring, then each matrix), `flutter.py`, `validation.py`, and the `_bact_preset()` function in `params.py`.

**Exercises**
1. Derive the 2-DOF typical-section equations of motion yourself with Lagrange's equations (no aerodynamics), and compare with `structural_matrices`.
2. Reproduce the Hodges & Pierce example: `python -m flutter_ai analyze --preset hodges_pierce`. Find the corresponding figure in the book and compare.
3. **Parameter study:** vary `x_alpha` (CG position) from 0.0 to 0.4 for `uav_section` and plot flutter speed vs `x_alpha`. Explain the trend physically (mass balancing).
4. Vary `f_beta` (flap servo stiffness) for `uav_flap` from 10 to 300 Hz; explain why it approaches the 2-DOF result (a test checks this).
5. Run `python -m flutter_ai validate`; explain each of the four rows. Why does moving the aerodynamic centre from 17.5 % to 32.5 % chord change the flutter pressure so much?
6. Hard: add a Prandtl–Glauert factor 1/√(1−M²) to `cl_alpha` for BACT and see how the flutter-pressure error changes. This is a real next research step (Memory.md §7).

**Self-check**
- Explain flutter to a friend with no engineering background in 60 seconds.
- Why can two stable modes combine into an unstable one? What role does the aerodynamic coupling play?
- What do the aero-lag states z₁ and z₂ represent physically?
- Why does the model predict BACT's flutter *frequency* almost exactly but its flutter *pressure* 24 % high?
- What is the difference between flutter and divergence?

---

## Track 4: Control engineering (weeks 7–10)

**Why it matters here:** the controllers are the core contribution. The bugs we found (chatter, delay instability) are control-theory lessons.

**Concepts**
- Feedback, open loop vs closed loop, stability, poles.
- State-space control: state feedback `u = −Kx`, controllability, observability.
- **Observers:** estimating unmeasured states. The aero-lag observer in `controllers/base.py` is a simple exact one.
- **LQR:** the cost J = Σ(xᵀQx + uᵀRu), the Riccati equation, Bryson's rule for choosing Q and R (`controllers/costs.py`).
- **Digital control:**
  - sampling and zero-order hold;
  - why a continuous-time design chatters at 200 Hz (we saw this);
  - the discrete Riccati equation (`solve_discrete_are`).
- **Time delay:**
  - augmenting the state with past commands (`lqr.delay_augmented`);
  - delay margin;
  - why a controller tuned for perfect timing can go unstable with +5 ms (`closed_loop_radius`).
- **Actuator limits:** saturation, rate limits, servo dynamics (the flap servo; BACT's 26 Hz actuator).
- **Robustness:** gain and phase margins, uncertainty. Next steps: LQG (LQR + Kalman filter), H∞, gain scheduling.
- **Performance metrics:** RMS, peak, settling time, control effort, command rate (`metrics.py`).

**Resources**
- Steve Brunton, **"Control Bootcamp"** (YouTube, free). Watch the whole series; it matches this project almost exactly.
- Åström & Murray, **Feedback Systems** (free online book): chapters on state feedback, observers, robustness.
- Franklin, Powell & Emami-Naeini, *Feedback Control of Dynamic Systems*: the standard textbook.
- Franklin, Powell & Workman, *Digital Control of Dynamic Systems*: sampling, discrete LQR, delay.
- Anderson & Moore, *Optimal Control: Linear Quadratic Methods* (Dover): deeper LQR/LQG.
- Skogestad & Postlethwaite, *Multivariable Feedback Control*: robustness (later).
- Mukhopadhyay (1999), BACT flutter-suppression wind-tunnel results (link in `data/README.md`): real control laws on the real rig.

**Read in the repo:** `controllers/costs.py`, `controllers/lqr.py`, `controllers/base.py`, `simulation.py` (the main loop, especially the `in_flight` delay queue), `metrics.py`.

**Exercises**
1. Compute the LQR gain for `uav_flap` at 1.1 U_F with `lqr_gain(p, U, 0.005, 0.5, delay=1)`. Compute the closed-loop eigenvalues and check |z| < 1.
2. Change Q in `costs.py` (e.g. weight pitch 10× more). Re-run `python -m flutter_ai evaluate --preset uav_flap` and explain what changed.
3. Reproduce the delay study: for delays 0–6 steps, compute `closed_loop_radius` for LQR designed for 1 step. Plot it. Where does it cross 1?
4. **Implement LQG** (Kalman filter + LQR) as a new controller (`Phases.md` 4.2). This is the most valuable exercise in the guide. Follow `Architecture.md` §9 to register it.
5. Simulate LQR without the observer (use the true lag states) and compare with the observer version. Why did the old delay analysis disagree with the simulator? (Memory.md §5.)

**Self-check**
- Why does LQR need a model, and what happens if the model is wrong (BACT: 24 % off)?
- What does the aggressiveness knob change in the LQR cost, and why does that trade effort against response?
- Explain the chatter bug and the delay bug, and how each was detected and fixed.
- Why does the benchmark include both `lqr` and `lqr_nominal` in the delay sweep?

---

## Track 5: Machine learning and differentiable simulation (weeks 10–12)

**Why it matters here:** this is the "AI model" in the project. You must know exactly what kind of AI it is.

**Concepts**
- Neural networks: layers, activation functions (tanh), parameters, forward pass.
- **Backpropagation** and gradient descent (Adam); learning rate; gradient clipping.
- PyTorch: tensors, autograd, `nn.Sequential`, optimisers, schedulers.
- **Imitation learning (behaviour cloning):** stage 1 of `training.py` copies LQR.
- **Differentiable simulation / backpropagation through time:** stage 2 improves the policy by differentiating the whole closed-loop simulation. This is what makes it "physics-trained".
- Training design:
  - curriculum (growing horizon);
  - domain randomisation (airspeed, turbulence, delay);
  - log(1+J) to limit the influence of hopeless episodes;
  - validation sets and best-checkpoint selection.
- **Generalisation:** in-distribution vs out-of-distribution. Why the neural policy warns outside its training range.
- **What a PINN really is** (physics-informed neural network): a network whose loss includes the *residual of the governing
  equations*. It is used to solve equations or identify parameters. Our controller is *not* a PINN. A real PINN use here
  would be identifying a wing's parameters from wind-tunnel data (Phase 4.0).
- Related methods to know by name: reinforcement learning (PPO, SAC), model predictive control (MPC), system identification (SINDy, DMD).

**Resources**
- Andrej Karpathy, **"Neural Networks: Zero to Hero"** (YouTube, free): builds backprop from scratch. Watch at least the first 3 videos.
- Official PyTorch tutorials ("Learn the Basics").
- *Dive into Deep Learning* (d2l.ai, free) and Goodfellow, Bengio & Courville, *Deep Learning* (free online): reference.
- **Brunton & Kutz, *Data-Driven Science and Engineering*** (Cambridge): ML + control + dynamical systems. Very relevant.
- Raissi, Perdikaris & Karniadakis (2019), "Physics-informed neural networks", J. Comput. Phys. 378: the PINN paper, to understand the difference.
- NASA, "Active Control of Wind-Tunnel Model Aeroelastic Response Using Neural Networks" (link in `data/README.md`): prior work you must cite.

**Read in the repo:** `training.py` (slowly; it is the most complex file), `controllers/neural.py`, `outputs/training_log_*.txt`.

**Exercises**
1. Train with `python -m flutter_ai train --preset uav_flap --quick` and read the log line by line.
2. Set `rate_weight=0` in `TrainConfig`, retrain, and run the chatter test. Watch it fail: that is the bug from Memory.md.
3. Train without delay randomisation (`delay_range=(1, 1)`) and compare delay robustness with the default policy.
4. Linearise the policy (`NeuralPolicy.linearize`) and compare its gain matrix with LQR's. How similar are they, and why?
5. Hard: write a tiny PINN that identifies `k_alpha` from a simulated time history (use `data.export_run_csv` to create data).

**Self-check**
- Is this project "AI"? What exactly is learned, from what, and how?
- Why does training start by imitating LQR?
- Why did the neural policy beat LQR on the UAV wing but spend more effort on BACT?
- What would convince a sceptical reviewer that the neural policy is safe? (Delay margins, linearisation, tests.)
- What is the difference between a PINN and our physics-trained policy?

---

## Track 6: LLMs in the control loop (week 12)

**Why it matters here:** the LLM supervisor is the most novel and most questioned part of the paper.

**Concepts**
- How LLM APIs work: messages, tokens, latency, cost, model choice.
- **Structured outputs** (JSON schema): why the LLM may only choose a mode from an enum.
- **Safety architecture:**
  - the slow loop (0.25 s) vs the fast loop (200 Hz);
  - fallback to rules on any failure;
  - the call budget;
  - applying the measured latency in simulation time.
- **Evaluation:** the Pareto (trade-off) test. A supervisor only adds value if it lands *below* the fixed-gain curve (`fig6_tradeoff.png`). So far the rule supervisor does not.
- Prompt design and the risk of prompt injection (why situation reports contain only numbers the code computed).

**Resources**
- Anthropic documentation (docs.anthropic.com / platform docs): messages API, structured outputs, prompt engineering.
- The supervisor code and its tests: `supervisor/llm.py`, `tests/test_supervisor.py` (it shows how to test an LLM integration without calling the API).

**Exercises**
1. Read `format_report()` output for a turbulence run; judge whether *you* would pick the same mode as the rules.
2. Change a threshold in `supervisor/rules.py` and check whether the supervisor's point on the trade-off figure moves below the curve.
3. With an API key: run one Claude-supervised simulation in the web app and read the decision log. Record latency and cost.

**Self-check**
- Why must the LLM never output actuator commands?
- What happens if the API is down, slow, refuses, or returns invalid JSON? Where is each handled?
- How would you prove, with numbers, that an LLM supervisor helps? What if it doesn't?

---

## Track 7: Software engineering, web and DevOps (weeks 13–14)

**Why it matters here:** this is what makes the project reproducible and presentable, and it's what employers look at on GitHub.

**Concepts**
- **Testing:** unit tests, regression tests (every bug gets one), `pytest`, fixtures, `skipif`.
- **CI:** GitHub Actions workflow syntax; jobs, matrices (Linux/Windows, Python versions), artifacts.
- **pre-commit** hooks and linters (ruff).
- **Docker:** images, layers, `Dockerfile`, `docker compose`, health checks, non-root users.
- **Web APIs:** HTTP, JSON, REST endpoints, FastAPI, Pydantic validation, status codes (404/409/422).
- **Frontend:** HTML, CSS (custom properties, dark mode), JavaScript modules, `fetch`, the Canvas 2D API, `requestAnimationFrame`.
- **Dependency management:** `pyproject.toml` extras, Dependabot.

**Resources**
- Brian Okken, *Python Testing with pytest*; pytest documentation.
- GitHub Actions documentation ("Understanding GitHub Actions").
- Docker documentation ("Get started").
- FastAPI official tutorial (fastapi.tiangolo.com): the "Tutorial – User Guide" section.
- MDN Web Docs (developer.mozilla.org): JavaScript guide, Canvas tutorial, Fetch API.

**Read in the repo:** `tests/` (all), `.github/workflows/ci.yml`, `Dockerfile`, `api/app.py`, `frontend/js/main.js`, `frontend/js/viewport.js`, `frontend/js/chart.js`.

**Exercises**
1. Add a new API endpoint `/api/validation` returning `outputs/validation/bact.json`, with a test in `tests/test_api.py`.
2. Add a "BACT validation" card to the web app that shows the measured vs predicted table.
3. Build and run the container: `docker compose up --build`.
4. Make CI fail on purpose in a branch (a failing test), look at the logs on GitHub, then fix it.

**Self-check**
- What exactly does CI check on every push, and why does that protect the paper's numbers?
- Trace one click on "Run simulation": browser → `api.js` → FastAPI → `simulate()` → JSON → canvas.
- Why doesn't the Docker image include PyTorch?

---

## Track 8: Research methods and writing (weeks 15–16, then ongoing)

**Why it matters here:** a paper is judged on honest evidence and clear writing, not on code size.

**Concepts**
- **Literature review:** search (Google Scholar, NASA NTRS, AIAA ARC), read abstracts → figures → conclusions, track citations.
- **Novelty claims:** never write "first" unless you searched thoroughly. Neural flutter control is *not* new (NASA did it).
- **Statistics:**
  - mean ± standard deviation over seeds;
  - confidence intervals (bootstrap);
  - paired tests (Wilcoxon);
  - effect size vs significance.
- **Reproducibility:** fixed seeds, versioned code, `results.json` metadata, "every number from a script".
- **Validation vs verification:**
  - verification = the code solves the equations correctly (p-k cross-check, tests);
  - validation = the equations match reality (BACT wind-tunnel comparison).
- **Honest reporting:** negative results (supervisor neutral), limitations (compressibility, strip theory), threats to validity.
- **Scientific integrity:** the audit (`docs/research/audit-2026-09.md`) is your case study. Understand every bug in it.
- **Writing:** paper structure, figures that make one point each, abstracts, responding to reviewers.

**Resources**
- Booth, Colomb & Williams, *The Craft of Research*.
- Justin Zobel, *Writing for Computer Science*.
- Allen Downey, *Think Stats* (free online): statistics with Python.
- `docs/research/paper-outline.md` and `docs/research/audit-2026-09.md` in this repo.

**Exercises**
1. Read the three NASA BACT reports and write a one-page summary of each in your own words.
2. Find and read 10 papers on active flutter suppression; build a table (method, test article, result, year).
3. Run the turbulence scenario with 20 seeds and compute a 95 % bootstrap confidence interval for the neural-vs-LQR RMS difference.
4. Write the "System model" and "Validation" sections of the paper using only numbers from `results.json` and `bact.json`.

**Self-check**
- What exactly is the contribution of your paper, in two sentences, without the word "first"?
- What are the three biggest limitations, and how does each affect the conclusions?
- Which results would change if the model were 24 % off in flutter speed, as it is for BACT?

---

## 9. Case studies: the bugs in this project (learn from them)

| Bug | Lesson | Track |
|---|---|---|
| Legacy PINN weights all NaN (explicit Euler, ω·dt ≈ 4.7) | Numerical stability of integrators; check your trained model | 2, 5 |
| `np.nanmax` hid the NaNs | Never mask failures; test for them | 7, 8 |
| LLM strategy never reached the controller | Test that a component actually has an effect | 6, 7 |
| Aero matrices dimensionally wrong | Unit/dimension checks; validate against independent methods | 3 |
| Continuous LQR chattered at 200 Hz | Design in the domain you implement (discrete time) | 4 |
| Neural policy chattered (pole near z = −1) | Penalise control rate; check closed-loop poles | 4, 5 |
| Neural policy unstable with +5 ms delay | Always model latency; test delay margins | 4, 5 |
| Delay analysis disagreed with the simulator | Your analysis must model what you implemented (the observer) | 4 |
| V-g plot mode swaps | Track modes by continuity, not sorting | 2 |
| BACT aerodynamic-centre ambiguity (17.5 % vs 30 %) | When a source contradicts itself, test both and report | 3, 8 |

For each bug: find it in `docs/Memory.md` §5 or the audit, reproduce it (most have a regression test), and explain the fix.

---

## 10. Explaining the project (interviews, viva, conferences)

**30-second pitch.** "Flutter is an aeroelastic instability that can destroy a wing in seconds. I built a validated simulation
of a wing section with a trailing-edge flap, compared a classical optimal controller (LQR) with a neural-network
controller trained through the physics simulator, and tested whether an LLM supervisor can safely tune the controller.
The model predicts NASA's wind-tunnel flutter frequency within 0.2 %. The neural controller outperforms LQR on
response. So far the LLM supervisor doesn't add measurable value, and the paper reports that honestly."

**Questions you should be able to answer** (answers are in the tracks above):
1. What is flutter and how do you predict it? (Track 3)
2. How did you validate the model? What is the remaining error and why? (Tracks 3, 8)
3. Why is your neural network not a PINN? Then what is it? (Track 5)
4. How do you know the neural controller is safe? (Tracks 4, 5)
5. What happens with sensor or actuator delay? (Track 4)
6. Why use an LLM at all, and why can't it crash the wing? (Track 6)
7. What would you do differently, or next? (Memory.md §7, Phases.md)
8. Walk me through your CI pipeline. (Track 7)
9. What did you learn from the original project's mistakes? (§9)

---

## 11. 16-week plan (8–10 h/week)

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Track 1: git, terminal, Python | Your first PR (new preset) merged with green CI |
| 2 | Track 1 + NumPy; start Track 2 | Track 1 self-check passed |
| 3 | Track 2: linear algebra, eigenvalues (3Blue1Brown, Strang) | Exercise 2.1–2.2 |
| 4 | Track 2: ODEs, state space, discretisation | Track 2 self-check passed |
| 5 | Track 3: vibrations, Lagrange, typical section (Hodges & Pierce ch. 1, 4) | Hand derivation of the EOM |
| 6 | Track 3: unsteady aero, Theodorsen, flutter (H&P ch. 5; Wright & Cooper) | Exercises 3.2–3.4 |
| 7 | Track 3: real wings, BACT report, compressibility | `validate` explained in writing; Track 3 self-check |
| 8 | Track 4: Control Bootcamp (first half), state feedback | Exercise 4.1–4.2 |
| 9 | Track 4: LQR, digital control, delay | Exercise 4.3 |
| 10 | Track 4: observers, **implement LQG** (exercise 4.4) | LQG controller PR |
| 11 | Track 5: Karpathy videos 1–3, PyTorch basics | Exercise 5.1 |
| 12 | Track 5: `training.py` in depth; Track 6 | Exercises 5.2–5.4, 6.1–6.2 |
| 13 | Track 7: testing, CI, Docker | Exercise 7.1, 7.3 |
| 14 | Track 7: frontend | Exercise 7.2 (BACT card in the app) |
| 15 | Track 8: literature review, statistics | Related-work table (10 papers); bootstrap CI |
| 16 | Track 8: writing | Draft "System model" + "Validation" sections |

After week 16: the Phase 4 research tasks in `Phases.md` (compressibility correction, Claude benchmark, stronger baselines), then write the paper.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| Aeroelasticity | Interaction of aerodynamic, elastic and inertial forces |
| Flutter | Self-excited oscillation that grows when aerodynamic forces feed energy into structural modes |
| Divergence | Static aeroelastic instability: aerodynamic twisting moment exceeds structural stiffness |
| Typical section | A 2D wing slice on springs (plunge, pitch, optionally flap) |
| Plunge h / pitch α / flap β | Vertical displacement (down +) / nose-up rotation / trailing-edge-down flap rotation |
| Elastic axis / shear centre | Point about which the section rotates under load (`a` in semichords) |
| Semichord b | Half the chord length |
| Reduced frequency k | ωb/U: how unsteady the flow is (BACT at flutter: ~0.04, nearly steady) |
| Theodorsen function C(k) | Lag and attenuation of circulatory lift in harmonic motion |
| Wagner function | Lift build-up after a sudden change in angle of attack |
| Mass ratio μ | m/(πρb²): wing mass relative to surrounding air (UAV ~58, BACT ~880) |
| Dynamic pressure q | ½ρU²: the aerodynamic load scale (psf = lb/ft²) |
| V-g diagram | Damping of each mode vs airspeed; flutter where one crosses zero |
| p-k method | Classical frequency-domain flutter method (our independent check) |
| Strip theory / modified strip theory | 2D sections summed along the span / with measured slopes and aerodynamic centre |
| State space | ẋ = Ax + Bu form of a dynamic system |
| Zero-order hold (ZOH) | Input held constant between samples; exact discretisation |
| Spectral radius | Largest |eigenvalue| of a discrete system; < 1 means stable |
| LQR / LQG | Optimal state feedback / LQR + Kalman filter |
| Riccati equation (DARE) | Equation whose solution gives the LQR gain |
| Observer | Algorithm estimating unmeasured states from measurements |
| Delay margin | How much extra latency the loop tolerates before instability |
| Saturation | Actuator at its limit (±15° flap; ±12° on BACT) |
| Imitation learning | Training a network to copy an expert (here LQR) |
| Differentiable simulation / BPTT | Training by differentiating through a simulated rollout |
| Domain randomisation | Training on randomised conditions for robustness |
| PINN | Network trained with a physics-residual loss (not what our controller is) |
| Structured output | LLM response constrained to a JSON schema |
| Pareto front | The best achievable trade-off curve between two objectives |
| CI | Continuous integration: automatic checks on every change |
| Regression test | A test that ensures a fixed bug never returns |
