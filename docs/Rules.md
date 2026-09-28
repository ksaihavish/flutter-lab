# Rules

*Boundaries for anyone, human or AI, who changes this project.*
**AI assistants must read this file and `Memory.md` before making changes.**

---

## 1. Scientific integrity (non-negotiable)

1. **Never type a result by hand.** Every number in the paper, README or slides must come
   from `outputs/results.json` (produced by `python -m flutter_ai evaluate`) or from a
   script committed to the repo.
2. **Never hide failure.** No `np.nanmax`/`nanmean` over trajectories, no clipping of states,
   no `try/except: pass` around simulation steps. A run that exceeds structural limits is
   reported as a failure, with its failure time.
3. **Report negative results.** If a method does not help (for example, the rule supervisor
   sits on the fixed-trade-off curve), say so.
4. **Physics changes need validation.** Any change to `model.py`, `aero.py` or `flutter.py`
   must keep `tests/test_physics.py` green. New physics needs a new test against an
   independent method or a published value.
5. **Compare fairly.** All controllers share the same plant, `dt`, **loop delay**, actuator
   limits, sensors and scenarios. When one controller is told the true delay (the benchmark's
   `lqr`), also report the fixed-design variant (`lqr_nominal`). If one controller gets extra information (for example, turbulence
   statistics during training), state it.
6. **Do not claim "first".** Verify novelty in the literature first.
7. **Do not invent citations.** Only cite papers you have actually opened.

## 2. Libraries

| Use | Library |
|-----|---------|
| Numerics | `numpy`, `scipy` |
| Learning (training only) | `torch`, CPU wheels. Never import torch in serving code (`api/`, `controllers/`, `simulation.py`) |
| LLM | the official `anthropic` SDK only. No OpenAI-compatible shims and no LangChain |
| Web API | `fastapi`, `pydantic` v2, `uvicorn` |
| Figures | `matplotlib` (Agg backend) |
| Tests | `pytest` |
| Frontend | Vanilla JavaScript ES modules plus Canvas. No frameworks, bundlers or CDN dependencies until Phase 5 (React migration) |

Adding a dependency requires a one-line reason in `Memory.md`. Put it in the right extra
in `pyproject.toml` (`train`, `llm`, `dev`) so the base install stays small.

## 3. Code conventions

- **SI units internally** (m, rad, s, N/m, N·m/m). Convert to mm and degrees only at the
  presentation boundary (reports, UI, figures).
- **Sign conventions:** h positive down; α positive nose-up; `a` is semichords aft of
  mid-chord. Document units and signs in every docstring that takes physical quantities.
- Match the surrounding style: type hints, dataclasses, small pure functions, and
  comments that explain *why*.
- Keep one source of truth: cost weights in `controllers/costs.py`, modes in
  `supervisor/modes.py`, scenarios in `scenarios.py`, entity colours in `docs/Design.md`
  (mirrored in `frontend/css/app.css` and `plots.py`).

## 4. Error handling

- **Fail loudly and early** with a clear message that says how to fix the problem (for
  example, the missing policy error says to run `python -m flutter_ai train`).
- API: return `4xx` with a `detail` message for client errors (unknown preset → 404;
  invalid combination → 422; missing model → 409). Never return 200 with an error inside.
- Catch **specific** exceptions. The only broad fallback allowed is the LLM supervisor,
  which must fall back to rules and **record** `source="llm_fallback"` plus the reason.
- The frontend shows every API error and every controller warning in the banner. Nothing
  is swallowed.

## 5. LLM safety rules

1. The LLM chooses a **mode from a fixed enum**. It never outputs forces, gains or code.
2. The response is enforced with a JSON schema, then validated again in Python.
3. Every failure mode falls back to the rule supervisor: timeout, API error, refusal,
   invalid JSON, unknown mode, or call budget exceeded.
4. Cost guard: at most `max_calls` (default 40) per simulation run.
5. Measured latency is applied in simulation time by default. Do not report LLM results
   with latency disabled unless you say so explicitly.
6. API keys come only from environment variables or `.env`, which is git-ignored. Never
   hard-code, log or send a key to the frontend.

## 6. Testing

- `python -m pytest` must pass before any commit (about 10 s).
- Every bug fix gets a regression test (see `test_supervisor_decisions_change_the_control`
  and `test_neural_policy_is_quiet_at_rest_and_does_not_chatter`).
- After retraining a policy, run `pytest` **and** `python -m flutter_ai evaluate --preset <p>`, then
  update `Memory.md` with the new validation numbers. Policies are per preset
  (`models/neural_policy_<preset>.npz`); changing a preset's physics means retraining it.

## 7. What AI assistants should and shouldn't do

**Do**
- Read `Memory.md` first. Update it at the end of every session: what changed, the new
  numbers, open issues.
- Work phase by phase (`Phases.md`); keep changes small and tested.
- Run the code and look at the output (figures, the app) before saying something works.
- Ask before changing research framing, method names or physical parameters.

**Don't**
- Don't fabricate or "estimate" results, citations or benchmark values.
- Don't silently overwrite `models/neural_policy.npz` or change presets.
- Don't edit anything in `legacy/`. It is the historical record.
- Don't disable tests, loosen tolerances or add `skip` to make a failing test pass.
- Don't commit `.env`, `outputs/` or large binaries.
- Don't put the LLM in the fast control loop.
