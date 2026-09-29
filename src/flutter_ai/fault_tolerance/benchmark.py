"""
Fault-tolerant supervision benchmark (docs/research/fault-tolerant-plan.md).

    python -m flutter_ai ft-benchmark --preset uav_flap --seeds 20     # -> outputs/<preset>/ft_benchmark.json

For every fault (including "none", where any slow-down is a false alarm) and seed, each supervisor flies the same
flight: none (stays nominal), rules, random, the LLM (local by default; skipped if unavailable) and the oracle
(knows the fault; the least disruptive surviving mode, applied 1 s after onset). Saved: the oracle's per-mode
survival table (which responses work), per-seed outcomes, summaries, agreement with the oracle, and paired
statistics (exact McNemar test on survival; Wilcoxon on time slowed down, over the flights both supervisors survived)
per fault and pooled over faults. Time slowed down and cost J are averaged over surviving flights: a flight that fails
early would otherwise score as "not slowed". The file is rewritten after every fault, so a stopped run keeps its
finished faults.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..flutter import flutter_analysis
from ..params import WingParams
from ..paths import OUTPUTS_DIR
from .faults import FAULTS, benchmark_fault
from .simulate import MODE_ORDER, ft_simulate
from .supervisors import FTPhysicsRuleSupervisor, FTRandomSupervisor, FTRuleSupervisor, oracle_mode

DURATION = 30.0
PAIRS = (("llm", "rules"), ("llm", "none"), ("rules", "none"), ("llm", "random"), ("rules", "random"),
         ("rules+physics", "rules"), ("llm+physics", "llm"), ("llm+physics", "rules+physics"))
MARGIN_EVERY_S = 1.0  # on-board estimator period for the physics-informed supervisors
KEEP = ("failed", "failure_time_s", "cost_J", "slowed_s", "reaction_s", "first_response", "final_mode",
        "switches", "decisions", "llm_fallbacks")


def _make(kind: str, seed: int):
    if kind == "none":
        return None
    if kind == "rules":
        return FTRuleSupervisor()
    if kind == "rules+physics":
        return FTPhysicsRuleSupervisor()
    if kind == "llm+physics":
        from .llm import make_ft_llm_supervisor

        return make_ft_llm_supervisor(physics=True)
    if kind == "random":
        return FTRandomSupervisor(seed=seed)
    if kind == "llm":
        from .llm import make_ft_llm_supervisor

        return make_ft_llm_supervisor()
    raise ValueError(f"Unknown supervisor '{kind}'")


def _summary(ms: list[dict], oracle: list[str]) -> dict:
    ok = [m for m in ms if not m["failed"]]
    J = [m["cost_J"] for m in ok if m["cost_J"] is not None]
    react = [m["reaction_s"] for m in ms if m["reaction_s"] is not None]
    agree = [m["final_mode"] == o for m, o in zip(ms, oracle) if o in MODE_ORDER]
    return {"n": len(ms), "failures": len(ms) - len(ok), "survival": len(ok) / len(ms),
            "cost_J_mean": float(np.mean(J)) if J else None,  # over survivors (not paired across supervisors)
            "slowed_s_mean": float(np.mean([m["slowed_s"] for m in ok])) if ok else None,  # over survivors
            "reaction_s_median": float(np.median(react)) if react else None,
            "oracle_agreement": float(np.mean(agree)) if agree else None,
            "llm_fallbacks": sum(m["llm_fallbacks"] for m in ms)}


def paired(a: list[dict], b: list[dict]) -> dict:
    """Candidate a vs reference b over the same flights: exact McNemar on survival; Wilcoxon on time slowed over
    the flights both survived (a failed flight's time slowed is cut short by the failure, so it is not comparable)."""
    from scipy.stats import binomtest, wilcoxon

    a_only = sum(1 for x, y in zip(a, b) if not x["failed"] and y["failed"])
    b_only = sum(1 for x, y in zip(a, b) if x["failed"] and not y["failed"])
    k = a_only + b_only
    p_surv = 1.0 if k == 0 else float(binomtest(a_only, k, 0.5).pvalue)
    d = np.array([x["slowed_s"] - y["slowed_s"] for x, y in zip(a, b) if not x["failed"] and not y["failed"]])
    p_slow = None if d.size == 0 else 1.0 if np.allclose(d, 0.0) else float(wilcoxon(d).pvalue)
    return {"n": len(a), "survived_only_candidate": a_only, "survived_only_reference": b_only,
            "survival_diff": (a_only - b_only) / len(a), "mcnemar_p": p_surv, "n_both_survived": int(d.size),
            "slowed_s_mean_diff": float(d.mean()) if d.size else None, "wilcoxon_p_slowed": p_slow}


def _slow(x: float | None) -> str:
    return "  –  " if x is None else f"{x:4.1f}s"


def run_ft_benchmark(p: WingParams, seeds=tuple(range(20)), supervisors=("none", "rules", "random", "llm"),
                     out_dir: Path | None = None, make_figure: bool = True, log=print, *, apply_latency: bool = True,
                     supervisor_interval: float = 0.25, tag: str = "", faults: tuple[str, ...] | None = None,
                     llm_latency_s: float | None = None) -> dict:
    """
    Ablations: apply_latency=False applies every answer at once (simulation time stops while the LLM thinks), which
    separates judgement from speed; supervisor_interval sets how often every supervisor is asked. A tag writes
    ft_benchmark_<tag>.json / fig12_fault_tolerance_<tag>.png, so the main result file is never overwritten.
    """
    from ..supervisor import llm_status

    def say(msg: str) -> None:
        log(msg, flush=True) if log is print else log(msg)

    fa = flutter_analysis(p)
    Uf, f_hz = fa.flutter_speed, fa.flutter_frequency_hz
    if Uf is None or f_hz is None:
        raise ValueError(f"No flutter found for '{p.name}'")
    status = llm_status()
    if any(s.startswith("llm") for s in supervisors) and not status["available"]:
        say(f"LLM unavailable ({status['detail']}); skipping the LLM supervisors")
        supervisors = tuple(s for s in supervisors if not s.startswith("llm"))
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"metadata": {"version": __version__, "seeds": list(seeds), "duration_s": DURATION,
                              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "llm": status if any(s.startswith("llm") for s in supervisors) else None,
                              "margin_every_s": MARGIN_EVERY_S,
                              "apply_latency": apply_latency, "supervisor_interval_s": supervisor_interval,
                              "llm_latency_fixed_s": llm_latency_s,
                              "tag": tag,
                              "note": "Every value in this file is produced by flutter_ai.fault_tolerance.benchmark."},
                 "faults": {}}
    t0 = time.time()
    everything: dict[str, list[dict]] = {}
    suffix = f"_{tag}" if tag else ""
    res["metadata"]["file"] = f"ft_benchmark{suffix}.json"
    for kind, spec in FAULTS.items():
        if faults is not None and kind not in faults:
            continue
        fault = benchmark_fault(kind)
        per: dict[str, list[dict]] = {}
        decisions: list[dict] = []
        oracle_modes: list[str] = []
        mode_table: dict[str, list[dict]] = {m: [] for m in MODE_ORDER}
        for s in seeds:
            best, runs = oracle_mode(p, Uf, f_hz, fault, s, duration=DURATION)
            oracle_modes.append(best)
            for m in MODE_ORDER:
                mode_table[m].append(runs[m])
            per.setdefault("oracle", []).append({k: runs[best if best in runs else "safe"][k] for k in KEEP})
            for sup in supervisors:
                r = ft_simulate(p, Uf, f_hz, fault, _make(sup, s), duration=DURATION, seed=s,
                                apply_latency=apply_latency, supervisor_interval=supervisor_interval,
                                margin_every_s=MARGIN_EVERY_S if sup.endswith("+physics") else None,
                                fixed_llm_latency_s=llm_latency_s)
                per.setdefault(sup, []).append({k: r.metrics[k] for k in KEEP})
                if sup != "random":
                    decisions += [{"supervisor": sup, "seed": s, "time": d.time, "mode": d.mode, "source": d.source,
                                   "latency_s": d.latency_s, "applied_at": d.applied_at, "outcome": d.outcome,
                                   "reasoning": d.reasoning} for d in r.decisions]
        if kind != "none":
            for sup, ms in per.items():
                everything.setdefault(sup, []).extend(ms)
        summary = {k: _summary(v, oracle_modes) for k, v in per.items()}
        res["faults"][kind] = {
            "description": spec["description"], "severity": fault.severity, "held_out": spec["held_out"],
            "oracle_modes": oracle_modes, "fixed_modes": {m: _summary(v, oracle_modes) for m, v in mode_table.items()},
            "supervisors": summary, "per_seed": per, "decisions": decisions,
            "paired": {f"{a} vs {b}": paired(per[a], per[b]) for a, b in PAIRS if a in per and b in per},
        }
        say(f"  {kind:18s} oracle {max(set(oracle_modes), key=oracle_modes.count):9s} | " + "  ".join(
            f"{k} {v['survival']:.0%} slow {_slow(v['slowed_s_mean'])}" for k, v in summary.items())
            + f"   ({time.time() - t0:.0f} s)")
        _save(res, everything, out_dir, t0)
    for label, st in res["pooled_faulty"]["paired"].items():
        say(f"  pooled {label:16s}: survived only by candidate {st['survived_only_candidate']:3d}, only by reference "
            f"{st['survived_only_reference']:3d} (McNemar p = {st['mcnemar_p']:.2g}); slowed diff over "
            f"{st['n_both_survived']} flights both survived: {_slow(st['slowed_s_mean_diff'])}")
    path = out_dir / res["metadata"]["file"]
    say(f"Saved {path}")
    if make_figure:
        from .. import plots

        say(f"Saved {plots.ft_figure(res, out_dir / f'fig12_fault_tolerance{suffix}.png')}")
    return res


def _save(res: dict, everything: dict[str, list[dict]], out_dir: Path, t0: float) -> None:
    """Pooled statistics over the faulty flights finished so far; (re)write the result file."""
    res["pooled_faulty"] = {
        "supervisors": {k: _summary(v, [""] * len(v)) for k, v in everything.items()},
        "paired": {f"{a} vs {b}": paired(everything[a], everything[b]) for a, b in PAIRS
                   if a in everything and b in everything},
    }
    llm_dec = [d for f in res["faults"].values() for d in f["decisions"]
               if d["supervisor"].startswith("llm") and d["source"] == "llm"]
    if llm_dec:
        lat = np.array([d["latency_s"] for d in llm_dec])
        res["llm_latency_s"] = {"n": int(lat.size), "median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))}
    res["metadata"]["wall_time_s"] = round(time.time() - t0, 1)
    res["metadata"]["complete"] = set(res["faults"]) == set(FAULTS)
    (out_dir / res["metadata"]["file"]).write_text(json.dumps(res, indent=2))
