# Flutter Lab: notes for AI assistants

Read these before changing anything:
1. `docs/Memory.md`: current state, latest numbers, decisions, open issues. **Update it at the end of every session.**
2. `docs/Rules.md`: integrity rules, allowed libraries, error handling, LLM safety.
3. `docs/Phases.md`: what to work on next (one phase at a time).

Reference: `docs/PRD.md` (what and why), `docs/Architecture.md` (how), `docs/Design.md` (visual system),
`data/README.md` (dataset format), `docs/Study-Guide.md` (the owner's learning path).

## Commands
```
python -m pytest                                    # 178 tests (~2 min), must pass before committing
python -m flutter_ai analyze  --preset uav_flap     # flutter speed + p-k validation + V-g figure
python -m flutter_ai evaluate --preset uav_flap     # benchmark -> outputs/uav_flap/results.json + figures
python -m flutter_ai train    --preset uav_flap     # retrain neural policy (needs torch, ~5 min)
python -m flutter_ai validate                       # BACT wind-tunnel comparison -> outputs/validation/bact.json
python -m flutter_ai tune|robustness|ablations      # fair baselines, robustness, ablations (--preset uav_flap)
python -m flutter_ai llm-benchmark|ft-benchmark     # LLM supervisor studies (slow; local Ollama by default)
python -m flutter_ai ft-stability                   # dwell-time stability bound for supervisor switching
python -m flutter_ai paper                          # regenerate docs/research/paper-draft.md from results
python -m flutter_ai check-data data/<folder>       # validate a dataset, predicted vs measured flutter
python -m flutter_ai serve                          # web app at http://127.0.0.1:8000 (API docs at /docs)
```
Project location: `C:\Users\lio48\dev\flutter-lab` (moved out of OneDrive on 2026-09-29).
On Windows the virtual environment lives in `%LOCALAPPDATA%\flutter-ai\venv`; `start.bat` / `start.ps1` create and use it.
`pre-commit` hooks (ruff + hygiene) run on commit; CI (`.github/workflows/ci.yml`) must be green before merging.

## Hard rules (short version)
- Never hand-type results; they come from `outputs/<preset>/results.json`.
- Never mask NaNs, clip states, or hide failed runs. Every controller sees the same loop delay.
- The LLM only chooses a mode from an enum; never actuator commands.
- Don't edit `legacy/`. Don't overwrite `models/neural_policy_<preset>.npz` without recording it in Memory.md.
- No Claude co-author trailers or "Generated with Claude Code" lines in commits or PRs (owner's request).
