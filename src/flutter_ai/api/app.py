"""
HTTP API + static frontend.

    python -m flutter_ai serve        # http://127.0.0.1:8000
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..controllers import policy_path
from ..controllers.neural import NeuralPolicy
from ..experiments import neural_available, run_case, survival_check
from ..flutter import flutter_analysis, pk_flutter_speed
from ..gusts import GustSpec
from ..params import PRESETS, VALIDATION_ONLY_PRESETS, WingParams, get_preset
from ..paths import FRONTEND_DIR, MODELS_DIR
from ..scenarios import scenarios_for
from ..simulation import Scenario
from ..supervisor import MODES, llm_status
from . import guardian
from .schemas import FlutterIn, ParamOverrides, SimulateIn, SurvivalIn

log = logging.getLogger("flutter_ai.api")

app = FastAPI(title="Flutter AI", version=__version__,
              description="Physics-based flutter simulation with neural and LLM-supervised control.")
app.include_router(guardian.router)  # /api/guardian/*: fault-tolerant flights for the Guardian live view


def _params(preset: str, overrides: ParamOverrides | None) -> WingParams:
    try:
        return get_preset(preset, **(overrides.model_dump(exclude_none=True) if overrides else {}))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:  # inconsistent overrides, e.g. wrong number of actuator limits
        raise HTTPException(422, str(exc)) from exc


@lru_cache(maxsize=64)
def _flutter(p: WingParams):
    return flutter_analysis(p)


def _flutter_speed(p: WingParams) -> float:
    Uf = _flutter(p).flutter_speed
    if Uf is None:
        raise HTTPException(422, "No flutter found for these parameters in the analysed speed range.")
    return Uf


def _file_hashes() -> dict[str, str]:
    """Short SHA-256 of every trained policy and tuning file, so a result can be traced to the exact models."""
    files = sorted([*MODELS_DIR.glob("neural_policy_*.npz"), *MODELS_DIR.glob("classical_tuning_*.json")])
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:12] for f in files}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "model_hashes": _file_hashes(),
        "neural_policy_available": {name: neural_available(p) for name, p in PRESETS.items()},
        "llm": llm_status(),
    }


def _neural_training_range(p: WingParams) -> list[float] | None:
    """Airspeed range (x flutter speed) the preset's neural policy was trained on; outside it, it extrapolates."""
    if not neural_available(p):
        return None
    rng = NeuralPolicy.load(policy_path(p.name)).meta.get("velocity_ratio_range")
    return None if rng is None else [float(v) for v in rng]


@app.get("/api/presets")
def presets():
    return [{**p.to_dict(), "flutter_speed": _flutter(p).flutter_speed, "neural_policy_available": neural_available(p),
             "divergence_speed": _flutter(p).divergence_speed, "neural_training_range": _neural_training_range(p)}
            for p in PRESETS.values() if p.name not in VALIDATION_ONLY_PRESETS]


@app.get("/api/scenarios")
def scenarios(preset: str = "uav_flap"):
    return {name: {"description": s["description"], **s["scenario"].to_dict()}
            for name, s in scenarios_for(preset).items()}


@app.get("/api/modes")
def modes():
    return MODES


@app.post("/api/flutter")
def flutter(req: FlutterIn):
    p = _params(req.preset, req.overrides)
    out = _flutter(p).to_dict()
    if req.validate_pk:
        out["pk_flutter_speed"] = pk_flutter_speed(p)
    return out


@app.post("/api/simulate")
def simulate_endpoint(req: SimulateIn):
    if req.preset in VALIDATION_ONLY_PRESETS:
        raise HTTPException(422, f"'{req.preset}' is a validation-only preset (flutter analysis only); "
                                 "use it with /api/flutter or 'python -m flutter_ai analyze'.")
    p = _params(req.preset, req.overrides)
    Uf = _flutter_speed(p)
    scenario = _scenario(req)
    results = []
    for case in req.cases:
        if case.controller == "open_loop" and case.supervisor != "none":
            raise HTTPException(422, "A supervisor needs a feedback controller (lqr, lqg or neural).")
        try:
            res = run_case(p, Uf, scenario, case.controller, case.supervisor, case.aggressiveness)
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc)) from exc
        out = res.to_dict(max_points=req.max_points)
        out["case"] = case.model_dump()
        results.append(out)
    return {
        "preset": p.name,
        "params": p.to_dict(),
        "flutter_speed": Uf,
        "velocity": scenario.velocity_ratio * Uf,
        "scenario": scenario.to_dict(),
        "results": results,
    }


def _scenario(req: SimulateIn | SurvivalIn) -> Scenario:
    s = req.scenario
    return Scenario(
        velocity_ratio=s.velocity_ratio,
        duration=s.duration,
        h0=s.h0_mm / 1000.0,
        alpha0_deg=s.alpha0_deg,
        input_delay_steps=s.input_delay_steps,
        gust=GustSpec(**s.gust.model_dump()),
        sensor_noise=s.sensor_noise,
        supervisor_interval=s.supervisor_interval,
    )


@app.post("/api/survival")
def survival_endpoint(req: SurvivalIn):
    """
    The current flight over n_seeds realisations of its disturbance (turbulence: new random sequences; discrete
    gusts: amplitude, length, start and initial condition drawn as in the benchmark). Seed 0 is the flight
    /api/simulate shows. A single run cannot tell a stable controller from a lucky one.
    """
    if req.preset in VALIDATION_ONLY_PRESETS:
        raise HTTPException(422, f"'{req.preset}' is a validation-only preset.")
    if any(c.supervisor == "llm" for c in req.cases):
        raise HTTPException(422, "The multi-seed check does not run the LLM supervisor (too many slow calls).")
    if any(c.controller == "open_loop" and c.supervisor != "none" for c in req.cases):
        raise HTTPException(422, "A supervisor needs a feedback controller (lqr, lqg or neural).")
    p = _params(req.preset, req.overrides)
    Uf = _flutter_speed(p)
    workers = int(os.environ.get("FLUTTER_AI_WORKERS", str(max(1, min(4, (os.cpu_count() or 2) // 2)))))
    try:
        rows = survival_check(p, Uf, _scenario(req), [(c.controller, c.supervisor) for c in req.cases],
                              n_seeds=req.n_seeds, workers=workers)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"preset": p.name, "flutter_speed": Uf, "velocity": req.scenario.velocity_ratio * Uf,
            "n_seeds": req.n_seeds, "results": rows}


# ---- static frontend (mounted last so /api routes win) ----------------------
if FRONTEND_DIR.exists():
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/guardian", include_in_schema=False)
    def guardian_page():
        return FileResponse(FRONTEND_DIR / "guardian.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
