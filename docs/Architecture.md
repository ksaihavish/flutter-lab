# Architecture

*How the system is built: flow, components, folder structure, tech stack.*
Last updated: 2026-09-28 · Version 0.3.0

---

## 1. Big picture

```
                          ┌──────────────────── slow loop (every 0.25 s) ────────────────────┐
                          │                                                                   │
   measurements ──► Situation report ──► Supervisor (rules | Claude) ──► mode ──► aggressiveness a ∈ [0,1]
        ▲            (features + text)     enum JSON, fallback,                          │
        │                                  call budget, latency                          ▼
        │                  ┌──────────────── fast loop (200 Hz) ─────────────────────────────────┐
        │                  │  y = [q, q̇] ──► aero-lag observer ──► x̂ ──► Controller(x̂, u_prev, U, a)  │
        │                  │  (q = h, α, β)                               (LQR | Neural) ──► u_k     │
        │                  └──────────────────────────────────────────────────────┬───────────────┘
        │                                           input delay d steps (default 1 = 5 ms) │ saturate
        └──────────── Plant: x_{k+1} = A_d x_k + B_d u_{k−d} + G_d w_gust,k   (exact ZOH) ◄──┘
```

- **Plant.** A linear model of the typical section at airspeed U.
  - State `x = [q, q̇, z₁, z₂]` with `q = [h, α]` (2-DOF) or `[h, α, β]` (3-DOF with a trailing-edge flap).
  - `z` are the Wagner/Jones aerodynamic-lag states.
- **Actuation.**
  - 3-DOF (`flap_servo`): the controller commands a flap angle `β_cmd`, and a servo spring `k_β`
    pulls the flap towards it. This is realistic control-surface actuation, with ±15° command
    saturation.
  - 2-DOF (`force_moment`): idealised plunge force plus pitch moment.
- **Loop delay.** A command computed at step k reaches the actuator at step k + d (computation
  and transport latency). Every controller sees the same delay.
- **Fast loop.** Controllers see measurements only. The aero-lag states come from an on-board
  observer, which lags the true states by about half a step (this matters for stability
  margins, see §7).
- **Slow loop.** The supervisor picks one of four modes. The mode sets the aggressiveness knob
  (the effort-vs-damping trade-off) that both controllers honour.

## 2. Physics (see `src/flutter_ai/model.py` for the full equations)

```
M_s q̈ + C_s q̇ + K_s q = f_aero + B_q u           (per unit span; h down, α nose-up, β TE-down)
```

- **Structure:** mass matrix with flap coupling `I_β + b(c − a)S_β`; diagonal stiffness and damping.
- **Aerodynamics:** Theodorsen (1935) with the geometric flap coefficients T1…T14 (`aero.theodorsen_T`):
  - non-circulatory apparent mass, damping **and stiffness** terms (the flap adds a V²β stiffness);
  - circulatory loads through the ¾-chord downwash
    `Q = ḣ + Uα + b(½−a)α̇ + (U/π)T10 β + (b/2π)T11 β̇`, filtered by the Wagner function
    (Jones two-lag approximation) → 2 lag states;
  - gusts enter the downwash. Simplification: the Wagner lag is used, not Küssner.

**Validation** (`tests/test_physics.py`):
- Theodorsen C(k) against tabulated values.
- Hodges & Pierce benchmark: divergence √8·bω_α exact, flutter ≈ 2.17·bω_α.
- State-space vs an **independent p-k analysis with the exact Theodorsen function** (< 1.5 %, 2-DOF and 3-DOF).
- Flap loads vs **independent thin-airfoil theory** (Glauert series).
- Flap apparent mass symmetric positive definite.
- A stiff flap hinge recovers the 2-DOF result.

## 3. Components

| Module | Responsibility |
|--------|----------------|
| `params.py` | `WingParams` (frozen dataclass, SI): geometry, inertia, frequencies, damping, optional flap, actuation, limits; presets `uav_flap` (default, 3-DOF), `uav_section` (2-DOF), `hodges_pierce` (validation) |
| `aero.py` | Exact Theodorsen C(k), Jones approximation, flap coefficients T1…T14 |
| `model.py` | Generic N-DOF assembly (A, B, G), exact ZOH discretisation (cached), energy, state names |
| `flutter.py` | V-g/V-f sweep with continuous mode tracking, flutter and divergence speeds, independent p-k validation |
| `gusts.py` | 1-cosine, sine, AR(1) turbulence patch |
| `controllers/base.py` | `Controller` interface, command history, `AeroLagObserver`, aggressiveness mapping |
| `controllers/costs.py` | Shared Q and R weights (Bryson), so LQR and neural optimise the same objective |
| `controllers/lqr.py` | Delay-augmented discrete LQR (DARE); `closed_loop_radius()` for delay-margin analysis (includes the observer) |
| `controllers/neural.py` | Policy file format, NumPy inference, `linearize()`; one policy per preset |
| `supervisor/` | `report.py` (features + text), `modes.py`, `rules.py`, `llm.py` (Claude) |
| `simulation.py` | Closed-loop runner (`Scenario` → `SimResult`): delay queue, saturation, failure detection, supervisor scheduling |
| `metrics.py` | Peak/RMS h, α, β; settling; gust recovery; effort; command rate; saturation; mode switches |
| `scenarios.py` | Named benchmark scenarios |
| `training.py` | PyTorch: imitation of the delay-aware LQR → differentiable-physics fine-tuning with delay/speed/turbulence randomisation → best checkpoint |
| `experiments.py` | Benchmark: scenarios × cases × seeds; trade-off, velocity and delay sweeps → `outputs/<preset>/results.json` |
| `plots.py` | Publication figures (same entity colours as the app) |
| `data.py` | Dataset intake: format checks, dimensional parameters → `WingParams`, predicted vs measured flutter, export of simulated runs |
| `api/` | FastAPI app + Pydantic schemas; serves `frontend/` |
| `__main__.py` | CLI: `analyze`, `train`, `evaluate`, `serve`, `check-data`, `all` |

## 4. Folder structure

```
.
├── start.bat / start.ps1 / start.sh   one-click setup and launch
├── Dockerfile, docker-compose.yml     container for the web app (serving only, no PyTorch)
├── .github/                           CI workflow (lint, tests Linux+Windows, benchmark, Docker) + Dependabot
├── .pre-commit-config.yaml            ruff + hygiene hooks run before each commit
├── pyproject.toml                     package, dependencies, extras (train, llm, dev)
├── CLAUDE.md                          entry point for AI assistants
├── .env.example                       ANTHROPIC_API_KEY etc. (copy to .env; git-ignored)
├── src/flutter_ai/                    the Python package (§3)
├── frontend/                          static web app, no build step (index.html, css/app.css, js/*.js)
├── models/
│   ├── neural_policy_<preset>.npz     trained policy per preset (weights + metadata)
│   └── training_history_<preset>.json
├── data/                              datasets (README.md = what to collect + format; template_synthetic/)
├── outputs/<preset>/                  generated: results.json, fig1…fig9 (git-ignored)
├── tests/                             pytest: physics, control, supervisor, data, API
├── docs/                              PRD, Architecture, Rules, Phases, Design, Memory, research/
└── legacy/                            original scripts, outputs and docs (read-only history)
```

## 5. Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python ≥ 3.10 (developed on 3.14) | Scientific ecosystem |
| Numerics | NumPy, SciPy (`expm`, `solve_discrete_are`, Hankel functions, `linear_sum_assignment`) | Exact, well tested |
| Learning | PyTorch CPU, **training only** | Differentiable simulation; serving needs only NumPy |
| LLM | Anthropic SDK, `claude-opus-5` (via `FLUTTER_AI_LLM_MODEL`), structured JSON output, server-side refusal fallback | Enum-constrained decisions |
| API | FastAPI + Pydantic v2 + Uvicorn | Typed validation; docs at `/docs` |
| Frontend | Vanilla ES modules + Canvas 2D | Zero toolchain |
| Figures | Matplotlib (Agg) | Publication PNGs |
| Tests | pytest + FastAPI TestClient | 65 tests |
| Quality gates | ruff, pre-commit, GitHub Actions CI, Dependabot | Every PR is linted, tested on Linux + Windows, benchmarked, containerised |
| Deployment | Docker (`python:3.12-slim`, non-root, health check) | `docker compose up --build` |

## 6. Request flow: `POST /api/simulate`

1. Pydantic validates the body (preset, overrides, scenario incl. `input_delay_steps`, 1–6 cases).
2. `WingParams` = preset + overrides (a bad combination returns 422). Flutter analysis is cached.
3. For each case, `run_case` → `simulate` → `compute_metrics`. The neural controller loads
   `models/neural_policy_<preset>.npz` and warns if the parameters, speed, dt or delay are
   outside its training range.
4. The time series are downsampled for the browser (`u` is a list of input channels; `beta` is
   included for 3-DOF).

## 7. Timing, numerics and stability analysis

- `dt = 5 ms` (200 Hz); exact ZOH discretisation, so there is no integrator error.
- Default loop delay: 1 step. The delay-aware LQR includes past commands in its state; the
  neural policy receives the previous command.
- The run stops when |h| > `h_limit` or |α| > `alpha_limit_deg` (the linear model is invalid
  beyond them) and is reported as a failure.
- `closed_loop_radius(p, U, dt, K, delay)` gives the exact spectral radius of the implemented
  loop, including the observer. Use it for delay margins (a test checks it against the simulator).

## 8. Neural policy file (`models/neural_policy_<preset>.npz`, format 3)

- Contents: layer weights (hidden layers tanh, last layer linear), state scales, `u_max`, and
  `meta` JSON.
- `meta` includes: preset, parameter fingerprint, flutter speed, speed range, **delay range**,
  dt, training config, and validation vs LQR per delay.
- Older formats are rejected with a "retrain" message.

## 9. Extension points

- **New wing / rig:** add to `PRESETS` (or build one from a dataset with
  `data.params_from_metadata`), then `analyze` and `train --preset <name>`.
- **More DOF (flexible wing):** `model.py` assembles M, C, K, e and the downwash rows
  generically. A modal N-DOF wing needs strip-theory versions of those five matrices (Phase 7.3).
- **New controller:** subclass `Controller`, implement `compute(t, x_hat)` (use `self.history`
  for past commands), honour `self.aggressiveness`, then register it in
  `experiments.make_controller`, `api/schemas.CaseIn` and `frontend/js/main.js` (`CASES`, with a
  validated colour).
- **New scenario:** add it to `scenarios.SCENARIOS`; it appears in the benchmark and the app.
