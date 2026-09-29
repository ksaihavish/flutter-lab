"""
Flight data for the learned diagnoser (no torch): randomised flights simulated with ft_simulate and cut into exactly
the windows the on-board diagnoser sees (diagnoser.window_features at the times ft_simulate would call it), with a
label per window and known fault.

Flights: healthy (20 %), one (45 %), two (25 %) or three (10 %) simultaneous faults, each with a random onset
(6-18 s), growth time (1-12 s) and size (SEVERITY_RANGE x the benchmark severity); cruise 1.0-1.15 U_F; turbulence
0.20-0.30 m/s RMS; flown with no supervisor (40 %), the symptom rules (35 %) or a forced mode change at a random time
(25 %), so that slowed-down, neural-controller and decelerating flights occur with and without faults.
A slower servo only matters (and only shows) once its limit falls below the flap rates the controller uses (~330
deg/s in healthy flight, reached at ~65 % of the benchmark severity; checked on development seed 1002): below that the
fault changes nothing in the flight, so it is sampled from 0.7x and labelled present from 0.65 of its size.
Training and validation use only the four known fault kinds. The held-out kinds (loop_delay, flap_effectiveness) are
only in the evaluation split `eval_held_out`; `eval_benchmark` repeats the fault-tolerant benchmark's conditions
(benchmark severity, onset 6 s, growth 5 s, cruise 1.1 U_F, no supervisor).

Seeds (all >= 1000, disjoint, never the benchmark's 0-19, 100-119 or 200-204): see SPLITS.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..flutter import flutter_analysis
from ..params import WingParams, get_preset
from ..paths import OUTPUTS_DIR
from .diagnoser import BASELINE_S, DIAGNOSE_EVERY_S, HELD_OUT, KNOWN, baseline_stats, feature_names, window_features
from .faults import FAULTS, AnyFault, CompositeFault, Fault, benchmark_fault, components
from .simulate import FTResult, ft_simulate
from .supervisors import FTRuleSupervisor

ALL_KINDS = tuple(k for k in FAULTS if k != "none")  # label columns: known kinds first in FAULTS order, then held out
# A fault counts as present (target 1) from this fraction of its benchmark size; a started but smaller fault is
# excluded from the loss and from the window metrics (see the module docstring for the servo)
MIN_LEVEL: dict[str, float] = {"servo_slew": 0.65}
MIN_LEVEL_DEFAULT = 0.3
SEVERITY_RANGE: dict[str, tuple[float, float]] = {"servo_slew": (0.7, 1.3)}
SEVERITY_RANGE_DEFAULT = (0.5, 1.3)
DURATION = 30.0
DT = 0.005
N_FAULTS_P = (0.2, 0.45, 0.25, 0.1)  # healthy, one, two, three simultaneous faults
SUPERVISION_P = {"none": 0.4, "rules": 0.35, "forced": 0.25}
DATA_VERSION = 2  # bump when the flights or the features change: invalidates the cache
# Sizes chosen for the compute budget (well under an hour for all splits with 4 workers on a shared machine)
SPLITS: dict[str, dict] = {
    "train": {"first_seed": 10000, "n": 480},
    "val": {"first_seed": 20000, "n": 120},  # early stopping and every threshold
    "eval": {"first_seed": 30000, "n": 160},  # in-distribution test
    "eval_held_out": {"first_seed": 31000, "n": 100},  # loop_delay / flap_effectiveness, alternating
    "eval_benchmark": {"first_seed": 32000, "n": 56},  # every FAULTS kind under the benchmark's conditions, 8 each
}


def min_level(kind: str) -> float:
    return MIN_LEVEL.get(kind, MIN_LEVEL_DEFAULT)


def _bank() -> dict:
    """The simulator's bank without LQR: no FT mode uses it, so the flight is identical (checked) and a third faster."""
    from ..controllers import LQGController, NeuralController

    return {"lqg": LQGController(), "neural": NeuralController()}


def flight_spec(split: str, i: int) -> dict:
    """The i-th flight of a split (JSON-serialisable; the draw uses its own stream, not the flight's turbulence)."""
    seed = SPLITS[split]["first_seed"] + i
    if split == "eval_benchmark":
        kind = tuple(FAULTS)[i % len(FAULTS)]
        faults = [] if kind == "none" else [asdict(benchmark_fault(kind))]
        return {"split": split, "seed": seed, "faults": faults, "supervision": "none", "forced_mode": None,
                "cruise_ratio": 1.1, "turbulence_rms": 0.25}
    g = np.random.default_rng([seed, 7])
    kinds: tuple[str, ...]
    if split == "eval_held_out":
        kinds, n_f = (HELD_OUT[i % len(HELD_OUT)],), 1
    else:
        kinds, n_f = KNOWN, int(g.choice(len(N_FAULTS_P), p=N_FAULTS_P))
    faults = [asdict(Fault(k, onset=float(g.uniform(6.0, 18.0)), ramp=float(g.uniform(1.0, 12.0)),
                           severity=float(FAULTS[k]["severity"] * g.uniform(*SEVERITY_RANGE.get(k, SEVERITY_RANGE_DEFAULT)))))
              for k in (str(x) for x in g.choice(kinds, n_f, replace=False))]
    sup = str(g.choice(list(SUPERVISION_P), p=list(SUPERVISION_P.values())))
    forced = [float(g.uniform(5.0, 25.0)), str(g.choice(["robust", "cautious", "safe"]))] if sup == "forced" else None
    return {"split": split, "seed": seed, "faults": faults, "supervision": sup, "forced_mode": forced,
            "cruise_ratio": float(g.uniform(1.0, 1.15)), "turbulence_rms": float(g.uniform(0.20, 0.30))}


def fault_of(spec: dict) -> AnyFault:
    parts = tuple(Fault(**f) for f in spec["faults"])
    return Fault("none") if not parts else parts[0] if len(parts) == 1 else CompositeFault(parts)


def diagnosis_indices(t: np.ndarray, k_base: int, every: float = DIAGNOSE_EVERY_S) -> list[int]:
    """The samples at which ft_simulate(diagnose_every_s=every) diagnoses (the first one at the baseline instant)."""
    ks, nxt = [], float(t[k_base])
    for k in range(k_base, len(t)):
        if t[k] >= nxt - 1e-12:
            ks.append(k)
            nxt += every
    return ks


def featurise(p: WingParams, flutter_speed: float, r: FTResult, fault: AnyFault, dt: float = DT) -> dict:
    """Window features and labels of a finished flight, identical to what the on-board diagnoser computes."""
    d = len(feature_names(p))
    empty = {"t": np.zeros(0), "X": np.zeros((0, d)), "level": np.zeros((0, len(ALL_KINDS))),
             "present": np.zeros((0, len(ALL_KINDS)), bool)}
    if r.t[-1] < BASELINE_S[1]:
        return empty  # failed before the take-off baseline was complete
    k_base = int(np.argmax(r.t >= BASELINE_S[1]))
    base = baseline_stats(p, dt, r.y, r.u, r.airspeed, r.mode, k_base)
    ks = diagnosis_indices(r.t, k_base)
    X = np.array([window_features(p, dt, flutter_speed, r.y, r.u, r.airspeed, r.mode, k, base) for k in ks])
    times = r.t[ks]
    level, present = np.zeros((len(ks), len(ALL_KINDS))), np.zeros((len(ks), len(ALL_KINDS)), bool)
    for f in components(fault):
        if f.kind in ALL_KINDS:
            j = ALL_KINDS.index(f.kind)
            level[:, j] = [f.level(float(tt)) / FAULTS[f.kind]["severity"] for tt in times]
            present[:, j] = times >= f.onset
    return {"t": times, "X": X, "level": level, "present": present}


def run_flight(job: tuple) -> dict:
    """Simulate and featurise one flight (runs in a worker process)."""
    preset, spec, flutter_speed, flutter_freq_hz = job
    t0 = time.time()
    p = get_preset(preset)
    fault = fault_of(spec)
    r = ft_simulate(p, flutter_speed, flutter_freq_hz, fault, FTRuleSupervisor() if spec["supervision"] == "rules"
                    else None, duration=DURATION, dt=DT, seed=spec["seed"], cruise_ratio=spec["cruise_ratio"],
                    turbulence_rms=spec["turbulence_rms"], bank_factory=_bank,
                    forced_mode=None if spec["forced_mode"] is None else tuple(spec["forced_mode"]))
    out = featurise(p, flutter_speed, r, fault)
    out["flight"] = {"spec": spec, "failed": r.failed, "failure_time_s": r.failure_time,
                     "slowed": bool(np.any(r.airspeed < r.airspeed[0] - 0.5)), "wall_s": round(time.time() - t0, 2)}
    return out


@dataclass
class Dataset:
    """All windows of a split. flight[i] indexes `flights`; level / present have one column per ALL_KINDS."""

    X: np.ndarray
    t: np.ndarray
    flight: np.ndarray
    level: np.ndarray
    present: np.ndarray
    flights: list[dict]

    def targets(self, kinds: tuple[str, ...] = KNOWN) -> tuple[np.ndarray, np.ndarray]:
        """(Y, defined): Y = fault at >= min_level(kind) of its benchmark size; a fault that has started but is
        smaller is neither positive nor negative (defined = False): nothing can see a fault at 5 % of its size."""
        idx = [ALL_KINDS.index(k) for k in kinds]
        Y = self.level[:, idx] >= np.array([min_level(k) for k in kinds])
        return Y, ~(self.present[:, idx] & ~Y)

    def healthy_window(self) -> np.ndarray:
        """Windows with no fault started (healthy flights and the time before any onset)."""
        return ~np.any(self.present, axis=1)


def _cache_path(p: WingParams, split: str) -> Path:
    return OUTPUTS_DIR / p.name / "diagnoser_data" / f"{split}.npz"


def _cache_key(p: WingParams, split: str) -> dict:
    return {"data_version": DATA_VERSION, "split": split, "config": SPLITS[split], "fingerprint": p.fingerprint(),
            "features": list(feature_names(p)), "duration": DURATION, "dt": DT}


def build_split(p: WingParams, split: str, workers: int = 4, log=print, n: int | None = None) -> Dataset:
    """Simulate (or load from the cache in outputs/<preset>/diagnoser_data/) every flight of a split."""
    path = _cache_path(p, split)
    key = _cache_key(p, split) | ({"n": n} if n is not None else {})
    if path.exists():
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta["key"] == key:
            log(f"  {split}: {len(meta['flights'])} flights from the cache {path}")
            return Dataset(data["X"], data["t"], data["flight"], data["level"], data["present"], meta["flights"])
    fa = flutter_analysis(p)
    if fa.flutter_speed is None or fa.flutter_frequency_hz is None:
        raise ValueError(f"No flutter found for '{p.name}'")
    n_flights = SPLITS[split]["n"] if n is None else n
    jobs = [(p.name, flight_spec(split, i), fa.flutter_speed, fa.flutter_frequency_hz) for i in range(n_flights)]
    t0 = time.time()
    outs: list[dict | None] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, os.cpu_count() or 1))) as ex:
        futures = {ex.submit(run_flight, job): i for i, job in enumerate(jobs)}
        for done, fut in enumerate(as_completed(futures), 1):
            outs[futures[fut]] = fut.result()
            if done % 50 == 0 or done == len(jobs):
                log(f"  {split}: {done}/{len(jobs)} flights ({time.time() - t0:.0f} s)")
    flights: list[dict] = []
    parts: list[tuple[np.ndarray, ...]] = []
    for i, o in enumerate(outs):
        assert o is not None
        flights.append(o["flight"] | {"n_windows": len(o["t"])})
        parts.append((o["X"], o["t"], np.full(len(o["t"]), i), o["level"], o["present"]))
    X, t, fl, level, present = (np.concatenate([q[j] for q in parts]) for j in range(5))
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Non-finite diagnoser features in split '{split}': a feature is broken, fix it (never mask)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, t=t, flight=fl, level=level, present=present,
                        meta=np.array(json.dumps({"key": key, "flights": flights, "wall_s": round(time.time() - t0, 1)})))
    log(f"  {split}: saved {len(flights)} flights, {len(t)} windows to {path} ({time.time() - t0:.0f} s)")
    return Dataset(X, t, fl.astype(int), level, present.astype(bool), flights)
