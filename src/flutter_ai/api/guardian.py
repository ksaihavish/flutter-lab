"""
Guardian live view: fly one fault-tolerant mission in the background, poll its progress, then play it back.

    GET  /api/guardian/options        faults, supervisors, modes and defaults for the form
    POST /api/guardian/flights        start (or reuse) a flight -> 202 {id, status, progress}
    GET  /api/guardian/flights/{id}   status and progress; the full recording once status == "done"

A 30 s flight with the on-board estimator costs tens of seconds to minutes of CPU, so it runs in one background
worker thread (one flight at a time, others queue) instead of inside the request. Flights are deterministic, so an
identical request returns the finished recording at once. Recordings live in memory only (the last MAX_KEPT).
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException

from ..experiments import neural_available
from ..fault_tolerance.benchmark import MARGIN_EVERY_S
from ..fault_tolerance.faults import FAULTS, Fault, benchmark_fault
from ..fault_tolerance.recording import SUPERVISORS, healthy_flutter, record_flight
from ..fault_tolerance.simulate import FT_MODES, MIN_DWELL_S, MODE_ORDER
from ..params import PRESETS, WingParams, get_preset
from .schemas import GuardianIn

log = logging.getLogger("flutter_ai.api.guardian")
router = APIRouter(prefix="/api/guardian", tags=["guardian"])

# The fault sizes, mode speeds and the dwell-time bound of the fault-tolerant study are calibrated for this wing.
GUARDIAN_PRESETS = ("uav_flap",)
MAX_KEPT = 12  # finished recordings kept in memory
MAX_PENDING = 3  # queued + running flights before new ones are refused (429)


@dataclass
class _Job:
    id: str
    request: GuardianIn
    params: WingParams
    fault: Fault
    status: str = "queued"  # queued | running | done | failed
    stage: str = "queued"  # queued | flying | ground_truth | done
    progress: float = 0.0  # of the current stage
    sim_time: float = 0.0
    result: dict | None = None
    error: BaseException | None = None


_JOBS: OrderedDict[str, _Job] = OrderedDict()
_LOCK = threading.Lock()
_QUEUE: queue.Queue[_Job] = queue.Queue()
_worker: threading.Thread | None = None


def _run(job: _Job) -> None:
    def progress(stage: str, fraction: float, sim_time: float) -> None:
        job.stage, job.progress, job.sim_time = stage, fraction, sim_time

    job.status = job.stage = "running"
    req = job.request
    try:
        job.result = record_flight(job.params, job.fault, req.supervisor, estimator=req.estimator,
                                   duration=req.duration, seed=req.seed, max_points=req.max_points,
                                   progress=progress)
    except Exception as exc:
        # Not swallowed: logged, stored and returned to the client as an HTTP error by the poll endpoint. Without it
        # an unexpected error would kill the worker and leave the job "running" forever.
        log.exception("Guardian flight %s failed", job.id)
        job.error, job.status = exc, "failed"
        return
    job.status, job.stage, job.progress = "done", "done", 1.0


def _work() -> None:
    while True:
        job = _QUEUE.get()
        try:
            _run(job)
        finally:
            _QUEUE.task_done()


def _ensure_worker() -> None:
    global _worker
    with _LOCK:
        if _worker is None or not _worker.is_alive():
            # daemon: stopping the server never waits for a flight in progress
            _worker = threading.Thread(target=_work, name="guardian-flights", daemon=True)
            _worker.start()


def _evict() -> None:
    """Drop the oldest finished recordings beyond MAX_KEPT (call with _LOCK held)."""
    finished = [k for k, j in _JOBS.items() if j.status in ("done", "failed")]
    while len(_JOBS) > MAX_KEPT and finished:
        _JOBS.pop(finished.pop(0))


def _validate(req: GuardianIn) -> tuple[WingParams, Fault]:
    try:
        p = get_preset(req.preset)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if p.name not in GUARDIAN_PRESETS:
        raise HTTPException(422, f"The fault-tolerant study is calibrated for {', '.join(GUARDIAN_PRESETS)} only "
                                 "(fault sizes, mode speeds, dwell-time bound).")
    if req.fault not in FAULTS:
        raise HTTPException(422, f"Unknown fault '{req.fault}'. Choose one of: {', '.join(FAULTS)}.")
    if SUPERVISORS[req.supervisor]["needs_estimator"] and not req.estimator:
        raise HTTPException(422, f"The '{req.supervisor}' supervisor needs the on-board physics estimator.")
    base = benchmark_fault(req.fault)
    fault = Fault(req.fault, onset=base.onset if req.onset is None else req.onset,
                  ramp=base.ramp if req.ramp is None else req.ramp,
                  severity=base.severity if req.severity is None else req.severity)
    if fault.kind != "none" and fault.onset >= req.duration:
        raise HTTPException(422, f"The fault onset ({fault.onset:g} s) must be before the end of the flight "
                                 f"({req.duration:g} s).")
    if not neural_available(p):
        raise HTTPException(409, "The controller bank's 'robust' mode needs the neural policy: "
                                 f"run python -m flutter_ai train --preset {p.name}")
    if req.supervisor == "guardian":
        from ..fault_tolerance.diagnoser import model_path

        if not model_path(p.name).exists():
            raise HTTPException(409, "The guardian needs the learned fault diagnoser: "
                                     f"run python -m flutter_ai train-diagnoser --preset {p.name}")
    return p, fault


def _status(job: _Job, with_result: bool) -> dict:
    out = {"id": job.id, "status": job.status, "stage": job.stage, "progress": job.progress,
           "sim_time": job.sim_time, "duration": job.request.duration,
           "queued_ahead": sum(1 for j in _JOBS.values() if j.status == "queued" and j is not job)
           if job.status == "queued" else 0}
    if with_result and job.status == "done":
        out["result"] = job.result
    return out


def _http_error(exc: BaseException) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, FileNotFoundError):  # e.g. a missing policy or tuning file
        return HTTPException(409, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(422, str(exc))
    return HTTPException(500, f"The flight simulation failed: {type(exc).__name__}: {exc}")


@router.get("/options")
def options():
    presets = []
    for name in GUARDIAN_PRESETS:
        p = PRESETS[name]
        Uf, f_hz = healthy_flutter(p)
        presets.append({"name": name, "description": p.description, "flutter_speed": Uf,
                        "flutter_frequency_hz": f_hz, "neural_policy_available": neural_available(p)})
    faults = {}
    for kind, spec in FAULTS.items():
        b = benchmark_fault(kind)
        faults[kind] = {**spec, "onset": b.onset, "ramp": b.ramp}
    return {"presets": presets, "faults": faults,
            "supervisors": {k: {"label": v["label"], "description": v["description"],
                                "needs_estimator": v["needs_estimator"]} for k, v in SUPERVISORS.items()},
            "modes": {m: FT_MODES[m] for m in MODE_ORDER}, "mode_order": list(MODE_ORDER),
            "min_dwell_s": MIN_DWELL_S, "estimator_every_s": MARGIN_EVERY_S, "defaults": GuardianIn().model_dump()}


@router.post("/flights", status_code=202)
def start_flight(req: GuardianIn):
    p, fault = _validate(req)
    key = hashlib.sha256(req.model_dump_json().encode()).hexdigest()[:16]
    with _LOCK:
        job = _JOBS.get(key)
        if job is not None and job.status != "failed":  # deterministic: reuse the running or finished flight
            _JOBS.move_to_end(key)
            return _status(job, with_result=False)
        if sum(1 for j in _JOBS.values() if j.status in ("queued", "running")) >= MAX_PENDING:
            raise HTTPException(429, f"{MAX_PENDING} flights are already queued or running; wait for one to finish.")
        job = _Job(id=key, request=req, params=p, fault=fault)
        _JOBS[key] = job
        _JOBS.move_to_end(key)
        _evict()
        _QUEUE.put(job)
        status = _status(job, with_result=False)
    _ensure_worker()
    return status


@router.get("/flights/{flight_id}")
def flight(flight_id: str):
    with _LOCK:
        job = _JOBS.get(flight_id)
        status = None if job is None or job.error is not None else _status(job, with_result=True)
    if job is None:
        raise HTTPException(404, "Unknown flight (recordings are kept in memory only; the server may have "
                                 "restarted). Start the flight again.")
    if job.error is not None:
        raise _http_error(job.error)
    return status
