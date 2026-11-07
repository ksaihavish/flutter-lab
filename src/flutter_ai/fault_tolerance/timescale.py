"""
Time-scale study: how fast must a supervisor answer, relative to how fast a fault develops?

    python -m flutter_ai ft-timescale --preset uav_flap      # -> outputs/<preset>/ft_timescale.json + fig13

For every fault, fault growth time (ramp from onset to full severity) and supervisor latency, the survival of two
supervisors with fixed decision logic, the symptom rules and rules + physics, whose answers are delayed by the given
latency (supervisors.Delayed). Fixed logic isolates the time scale from judgement; the LLM's measured latency
(ft_benchmark.json) is one point on the latency axis. Reference: the oracle (best fixed mode applied 1 s after
onset) for the same fault and growth time. Seeds 200-204: an analysis set, disjoint from the test seeds.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..flutter import flutter_analysis
from ..params import WingParams, get_preset
from ..paths import OUTPUTS_DIR
from .benchmark import MARGIN_EVERY_S
from .faults import FAULTS, benchmark_fault
from .simulate import ft_simulate
from .supervisors import Delayed, FTPhysicsRuleSupervisor, FTRuleSupervisor, oracle_mode

RAMPS_S = (1.0, 4.0, 16.0)
LATENCIES_S = (0.0, 1.0, 3.0, 9.0)
SEEDS = tuple(range(200, 205))
DURATION = 40.0
SUPERVISORS = ("rules", "rules+physics")


def _cell(job: tuple) -> dict:
    """One flight (or one oracle evaluation); runs in a worker process."""
    preset, kind, ramp, seed, sup, latency = job
    p = get_preset(preset)
    fa = flutter_analysis(p)
    Uf, f_hz = fa.flutter_speed, fa.flutter_frequency_hz
    if Uf is None or f_hz is None:
        raise ValueError(f"No flutter found for '{preset}'")
    fault = benchmark_fault(kind, ramp=ramp)
    if sup == "oracle":
        best, _ = oracle_mode(p, Uf, f_hz, fault, seed, duration=DURATION)
        return {"job": job, "survived": best != "none survives", "mode": best}
    inner = FTPhysicsRuleSupervisor() if sup == "rules+physics" else FTRuleSupervisor()
    r = ft_simulate(p, Uf, f_hz, fault, Delayed(inner, latency), duration=DURATION,
                    seed=seed, margin_every_s=MARGIN_EVERY_S if sup == "rules+physics" else None)
    return {"job": job, "survived": not r.failed, "reaction_s": r.metrics["reaction_s"]}


def run_timescale(p: WingParams, faults=None, ramps=RAMPS_S, latencies=LATENCIES_S, seeds=SEEDS,
                  supervisors=SUPERVISORS, out_dir: Path | None = None, workers: int | None = None,
                  make_figure: bool = True, log=print) -> dict:
    faults = tuple(faults or [k for k in FAULTS if k != "none"])
    jobs = [(p.name, k, r, s, "oracle", 0.0) for k in faults for r in ramps for s in seeds]
    jobs += [(p.name, k, r, s, sup, lat) for k in faults for r in ramps for s in seeds for sup in supervisors
             for lat in latencies]
    t0 = time.time()
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_cell, jobs, chunksize=1))
    grid: dict = {}
    for o in out:
        _, kind, ramp, seed, sup, lat = o["job"]
        cell = grid.setdefault(kind, {}).setdefault(f"{ramp:g}", {})
        key = "oracle" if sup == "oracle" else f"{sup}@{lat:g}"
        cell.setdefault(key, []).append(bool(o["survived"]))
    summary = {k: {r: {key: float(np.mean(v)) for key, v in cell.items()} for r, cell in rows.items()}
               for k, rows in grid.items()}
    res: dict = {"metadata": {"version": __version__, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "preset": p.name, "faults": list(faults), "ramps_s": list(ramps), "latencies_s": list(latencies),
                        "seeds": list(seeds), "supervisors": list(supervisors), "duration_s": DURATION,
                        "wall_time_s": round(time.time() - t0, 1),
                        "note": "Every value in this file is produced by flutter_ai.fault_tolerance.timescale."},
           "survival": summary}
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ft_timescale.json"
    path.write_text(json.dumps(res, indent=2))
    log(f"Saved {path} ({res['metadata']['wall_time_s']:.0f} s, {len(jobs)} flights)")
    if make_figure:
        log(f"Saved {timescale_figure(res, out_dir / 'fig13_timescale.png')}")
    return res


def timescale_figure(res: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    md = res["metadata"]
    faults, sups, lats, ramps = md["faults"], md["supervisors"], md["latencies_s"], md["ramps_s"]
    fig, axes = plt.subplots(len(sups), len(faults), figsize=(2.3 * len(faults), 2.4 * len(sups) + 0.6),
                             squeeze=False, sharex=True, sharey=True)
    for i, sup in enumerate(sups):
        for j, kind in enumerate(faults):
            ax = axes[i][j]
            rows = res["survival"][kind]
            M = np.array([[100 * rows[f"{r:g}"][f"{sup}@{lat:g}"] for lat in lats] for r in ramps])
            ax.imshow(M, cmap="Greens", vmin=0, vmax=100, aspect="auto", origin="lower")
            for a in range(len(ramps)):
                for b in range(len(lats)):
                    ax.text(b, a, f"{M[a, b]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if M[a, b] > 60 else "black")
            if i == 0:
                ax.set_title(kind, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{sup}\nfault growth (s)", fontsize=8)
            ax.set_xticks(range(len(lats)), [f"{x:g}" for x in lats], fontsize=8)
            ax.set_yticks(range(len(ramps)), [f"{x:g}" for x in ramps], fontsize=8)
            orc = [100 * rows[f"{r:g}"]["oracle"] for r in ramps]
            ax.set_xlabel("latency (s)" if i == len(sups) - 1 else "", fontsize=8)
            ax.text(1.02, 0.5, "oracle " + "/".join(f"{x:.0f}" for x in orc), transform=ax.transAxes, rotation=90,
                    va="center", fontsize=6, color="#555555")
    fig.suptitle("Survival (%) by supervisor latency and fault growth time (oracle per growth time, bottom to top)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
