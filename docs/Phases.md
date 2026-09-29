# Phases

*The project broken into steps an AI assistant (or you) can finish one at a time.
Each phase has a goal, tasks, a "done when" test, and a starter prompt.*

Status legend: ✅ done · 🔄 in progress · ⬜ not started

---

## Phase 0: Audit and rescue ✅
Found why the original results were invalid (`docs/research/audit-2026-09.md`), moved the
originals to `legacy/`, and set up the new package structure.

## Phase 1: Validated physics core ✅
6-state Theodorsen/Wagner model, exact ZOH discretisation, flutter and divergence
analysis, independent p-k validation, Hodges & Pierce benchmark test.

## Phase 2: Controllers, supervisor, benchmark ✅
Discrete LQR, a physics-trained neural policy (zero-centred, rate-penalised), rule and
Claude supervisors with safety rails, four scenarios, the benchmark runner, figures and
31+ tests.

## Phase 3: Web simulation lab v1 ✅
FastAPI + vanilla JS app: animated true-scale wing lanes, synchronised charts, V-g chart,
results table, supervisor log, CSV export, light/dark theme, and one-click start scripts.

## Phase 3.5: 3-DOF upgrade, loop delay, dataset intake ✅
Trailing-edge flap DOF (Theodorsen T-functions, validated against thin-airfoil theory), flap servo
actuation, a 5 ms loop delay for every controller, delay-aware LQR, a delay-randomised neural
policy, a delay-robustness sweep, dataset format + `check-data`.

---

## Phase 4: Make the paper publishable 🟡 (all evidence done 2026-09-29, incl. fault-tolerant supervision; remaining: the owner's writing and review, 4.8)

**Goal:** answer RQ1–RQ4 (`PRD.md`) with evidence a reviewer will accept.

| # | Task | Done when |
|---|------|-----------|
| 4.0a | ✅ **BACT preset** from NASA/TP-1998-206270 + `validate` command: flutter frequency −0.2 %, flutter q +24 % vs wind tunnel | `outputs/validation/bact.json` |
| 4.0 | 🟡 No open time-history data found (search 2026-09-29, `related-work.md`). **`identify` pipeline done** (multiple-shooting output error, tested on synthetic data). **Real data.** Collect one ★★★ dataset (`data/README.md`) → `check-data` → report the model error; then physics-informed parameter identification from time histories (the legitimate PINN contribution) | Identified vs nominal parameters table; benchmark re-run on the identified preset |
| 4.1 | ✅ `llm-benchmark` (local Ollama by default; Claude switchable): latency, fallbacks, agreement with rules, paired statistics, figure. **Run the LLM supervisor experiments.** Set `ANTHROPIC_API_KEY`, add `("neural","llm")` and `("lqr","llm")` to a separate LLM benchmark (it costs money; keep it out of the default `evaluate`), use 5+ seeds, and record latency, fallback rate and API cost | LLM points are on the trade-off figure; latency histogram; cost per run |
| 4.2 | ✅ **LQG** + shared rate penalty + design knobs tuned for the real actuator (`tune`). Remaining, optional: **Stronger baselines.** LQG with a Dryden turbulence model (so LQR also "knows" the turbulence), gain-scheduled LQR, and optionally H∞ (`python-control` / `slycot`) | The neural advantage is re-measured against fair baselines |
| 4.3 | ✅ `robustness`: model error ±5/10/20 %, sensor noise, actuator rate limits. **Robustness study.** Monte Carlo over ±10–20 % mass, stiffness and damping; sensor noise; 1–3 sample delay; actuator bandwidth (first-order lag) | Failure rate and RMS vs uncertainty, per controller |
| 4.4 | ✅ 2026-09-29: 20 seeds, randomised gusts, cost J, paired Wilcoxon + bootstrap CIs in `results.json`. **Statistics.** 20+ seeds for stochastic scenarios; paired tests (Wilcoxon) or bootstrap confidence intervals | Every claimed improvement has a CI or p-value |
| 4.5 | ✅ `ablations` + `train --variant` (imitation only, no rate penalty, no airspeed; random supervisor). **Ablations.** Neural without fine-tuning (imitation only), without the rate penalty, without airspeed input; supervisor with random mode choice | Table showing what each component contributes |
| 4.6 | ✅ Flap servo (3.5) + **slew-rate limit in plant and training** (uav_flap 545 °/s from a UAV servo datasheet; BACT 250 °/s NASA nominal). Remaining: values identified from data | Rate limit in the plant and in training |
| 4.7 | ✅ `docs/research/related-work.md` (every entry opened; closest prior work identified). **Literature review and novelty check.** Search: "LLM controller tuning", "language model supervisory control", "differentiable simulation flutter", "neural network flutter suppression", "BACT / NATA wind tunnel" | Related-work section written; the "first" claim is removed or verified |
| 4.9 | ✅ 2026-09-29 **Fault-tolerant LLM supervision** (`research/fault-tolerant-plan.md`): faults, diagnostics, controller bank with bumpless transfer, rules / random / LLM / oracle supervisors, `ft-benchmark` (20 seeds, paired statistics), `ft-stability` (dwell-time bound 6.5 s ≤ enforced 7 s) | Survival and mission cost per fault, per supervisor; held-out faults reported separately; paper section generated |
| 4.8 | 🟡 `python -m flutter_ai paper` generates `paper-draft.md` from the result files; every section generated, including §7.5, §8 and §10; the owner edits the wording and adds the closing sentence. **Write the paper** from `docs/research/paper-outline.md`; every number cites `results.json` | Draft reviewed by your advisor |

**Starter prompt:** *"Read CLAUDE.md, docs/Memory.md and docs/Rules.md. Do Phase 4.2: add an
LQG controller with a first-order turbulence model as a new Controller subclass, register
it everywhere Architecture.md §9 lists, add tests, re-run evaluate, and update Memory.md."*

## Phase 5: Web app v2 (the simulation showcase) ⬜

**Goal:** a demo you can show at conferences, interviews and to investors.

| # | Task | Done when |
|---|------|-----------|
| 5.1 | **Live mode:** stream the simulation over WebSocket, step by step, so the user can **inject gusts ("poke the wing")** and change airspeed while it runs | Flutter visibly grows when the controller is switched off live |
| 5.2 | **3D view** with three.js: a wing with span-wise mode shapes, flow particles, and a colour map of structural stress | Runs at 60 fps on a laptop |
| 5.3 | Migrate to **React + TypeScript + Vite** once the UI grows beyond ~5 views. Keep the API contract; add generated TS types from the OpenAPI schema | Parity with v1 plus type-checked build |
| 5.4 | A parameter editor with live flutter-speed readout; shareable scenario links (URL state) | A link reproduces a run |
| 5.5 | Save/compare runs; export a PDF report | — |
| 5.6 | Accessibility pass: keyboard control of playback, screen-reader table, texture option for colour-blind printing | axe-core audit clean |

## Phase 6: Engineering hardening 🔄 (6.1–6.4 done)

| # | Task |
|---|------|
| 6.1 | ✅ Git + GitHub; project moved out of OneDrive to `C:\Users\lio48\dev\flutter-lab` |
| 6.2 | ✅ CI (GitHub Actions): ruff, pytest on Linux (py3.11, 3.13) + Windows, benchmark smoke run with figure artifacts, Docker build + health check; Dependabot |
| 6.3 | ✅ `pre-commit` hooks (ruff, whitespace, large files, private keys); ✅ `mypy` clean on `src/`, run in CI |
| 6.4 | ✅ Docker image (`python:3.12-slim`, non-root, health check) + `docker compose up --build` |
| 6.5 | Config files for experiments (YAML/TOML scenarios) instead of code edits; experiment tracking (MLflow or Weights & Biases) for training runs |
| 6.6 | Versioned models and datasets (git-LFS or DVC), with a model card per policy |
| 6.7 | ✅ `/api/health` includes the version and model/tuning file hashes. Remaining: structured (JSON) logging |

## Phase 7: Engineering credibility (what makes it trustworthy to industry) ⬜

| # | Task |
|---|------|
| 7.1 | **Validate against experimental data.** Reproduce published wind-tunnel flutter results (e.g. the Texas A&M NATA apparatus or the NASA BACT wing). This is the single biggest credibility step |
| 7.2 | **Nonlinear aeroelasticity:** freeplay and cubic stiffness → limit-cycle oscillations. Here the neural policy's advantage over linear control should grow |
| 7.3 | **Import real structures:** read modal data (frequencies, mode shapes, generalised masses) from NASTRAN/FEM output and build reduced-order models beyond 2-DOF |
| 7.4 | **Stability guarantees** for the neural policy: Lyapunov / region-of-attraction certificates, or a runtime safety filter (fall back to LQR when outside a verified set) |
| 7.5 | Hardware-in-the-loop: export the controller (ONNX / C code) and run it on a microcontroller against the simulated plant in real time |
| 7.6 | Interoperability: FMI/FMU export so the model runs in Simulink/Modelica toolchains |

## Phase 8: Commercial product ⬜

| # | Task |
|---|------|
| 8.1 | **Decide the business model first.** Examples: a SaaS "flutter what-if" tool for UAV makers; a teaching platform licensed to universities; consulting and custom controller design. Interview 10 potential users before building 8.2–8.6 |
| 8.2 | Accounts and auth (e.g. Auth0/Clerk), organisations, per-user saved projects |
| 8.3 | A job queue for long runs (training, Monte Carlo): Redis + RQ/Celery/Arq; progress over WebSocket |
| 8.4 | Persistence: PostgreSQL for projects and runs; object storage (S3) for results |
| 8.5 | Cloud deployment (Fly.io / Render / AWS), HTTPS, rate limiting, observability (Sentry, OpenTelemetry), backups |
| 8.6 | Security review (OWASP top 10, dependency scanning, secrets management); an LLM cost dashboard per organisation |
| 8.7 | Legal: pick a licence (open-core?), terms of use, and an export-control check (flight-control technology can be dual-use; check ITAR/EAR/your country's rules before selling abroad) |
| 8.8 | Documentation site (MkDocs Material), tutorials, and an example gallery |
| 8.9 | **Positioning:** this is a *design and analysis tool*, not certified flight software (DO-178C/DO-331 would be a separate, very large effort) |
