# Flutter Lab

[![CI](https://github.com/ksaihavish/Hybrid-PINN-LLM-Framework-for-Supercritical-Aeroelastic-Flutter-Control/actions/workflows/ci.yml/badge.svg)](https://github.com/ksaihavish/Hybrid-PINN-LLM-Framework-for-Supercritical-Aeroelastic-Flutter-Control/actions/workflows/ci.yml)

**Active flutter suppression on a wing section (plunge, pitch and trailing-edge flap).
A validated physics model, a physics-trained neural controller, and an LLM (Claude)
supervisor, with an interactive web simulator.**

![Flutter Lab web app](docs/screenshot.png)

## Quick start

**Windows:** double-click **`start.bat`**. The first run creates a Python environment and
installs dependencies (a few minutes); later runs start in seconds. Your browser opens at
<http://127.0.0.1:8000>.

**macOS / Linux:** `./start.sh`

**Manual:**
```bash
python -m venv .venv && .venv/Scripts/activate    # macOS/Linux: source .venv/bin/activate
pip install -e ".[llm,dev]"          # add ,train to retrain neural policies (installs PyTorch)
python -m flutter_ai serve
```

**Docker:** `docker compose up --build`, then open <http://127.0.0.1:8000>.

**Optional: the Claude supervisor.** Copy `.env.example` to `.env`, set
`ANTHROPIC_API_KEY`, and restart.

## What you can do

| Command | What it does |
|---------|--------------|
| `start.bat` / `python -m flutter_ai serve` | Web app: animated wing with a moving flap, controller comparison, charts, flutter diagram, supervisor log, CSV export |
| `start.bat -Test` / `python -m pytest` | 65 tests, including physics validation against textbook values, an independent p-k method and thin-airfoil theory |
| `python -m flutter_ai evaluate --preset uav_flap` | Full benchmark → `outputs/uav_flap/results.json` + 9 publication figures (~20 s) |
| `python -m flutter_ai analyze --preset uav_flap` | Flutter speed, p-k cross-validation, V-g diagram |
| `python -m flutter_ai train --preset uav_flap` | Retrain the neural policy through the differentiable physics (~5 min, CPU) |
| `python -m flutter_ai check-data data/<folder>` | Validate an experimental dataset and compare predicted vs measured flutter speed ([data/README.md](data/README.md)) |

Presets: `uav_flap` (3-DOF, default), `uav_section` (2-DOF), `hodges_pierce` (textbook validation case).

## How it works

- **The wing.** A typical section (plunge h, pitch α, trailing-edge flap β) with unsteady
  Theodorsen aerodynamics (Wagner function, Jones approximation), written as an exact
  discrete-time state-space model.
- **Validation.** The flutter speed matches an independent p-k analysis with the exact
  Theodorsen function to within 1 %. The flap loads match classical thin-airfoil theory.
- **Controllers (200 Hz, 5 ms loop delay).** They command the flap through a ±15° servo:
  - a **delay-aware discrete LQR**;
  - a **neural policy** trained by back-propagating through the physics, with randomised delay,
    airspeed and turbulence.
- **Supervisor (every 0.25 s).** A supervisor reads a plain-language situation report and picks
  a mode (efficient / balanced / aggressive / emergency). It is either rule-based, or Claude via
  structured output with automatic fallback. The LLM never commands the flap directly.

**Current findings** (from `outputs/uav_flap/results.json`, 2026-09-28; details in [docs/Memory.md](docs/Memory.md)):
- The neural policy beats LQR in every scenario both survive (e.g. turbulence RMS plunge
  4.86 ± 0.28 vs 6.22 ± 0.34 mm), and it uses less actuator effort.
- Both tolerate the same loop delay (up to 15 ms).
- The rule-based supervisor does **not** yet improve on a well-chosen fixed setting.

## Development workflow

- `pip install -e ".[llm,dev]"`, then `pre-commit install`: lint and hygiene hooks run before every commit.
- Every push and pull request runs **CI** (`.github/workflows/ci.yml`): ruff; tests on Linux and Windows;
  a benchmark smoke run whose figures are uploaded as a downloadable artifact; and a Docker build with a health check.
- Dependabot opens weekly dependency-update PRs.

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/PRD.md](docs/PRD.md) | What we're building, for whom, research questions |
| [docs/Architecture.md](docs/Architecture.md) | System flow, physics, modules, folder structure, tech stack |
| [docs/Rules.md](docs/Rules.md) | Integrity rules, libraries, error handling, LLM safety, AI-assistant rules |
| [docs/Phases.md](docs/Phases.md) | Roadmap: paper → showcase app → hardening → commercial |
| [docs/Design.md](docs/Design.md) | Colours, theme, typography, chart and animation rules |
| [docs/Memory.md](docs/Memory.md) | Current state, latest results, decisions log, next steps |
| [data/README.md](data/README.md) | Which datasets to collect and the file format |
| [docs/research/](docs/research/) | Audit of the original code; corrected paper outline |

## Project history

The original research scripts are preserved in [`legacy/`](legacy/). They contained
critical bugs; see [docs/research/audit-2026-09.md](docs/research/audit-2026-09.md). **Do
not cite results from the legacy files.**
