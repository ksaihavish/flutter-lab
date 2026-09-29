"""
LLM supervisor benchmark (Phase 4.1), kept out of `evaluate` because it is slow (and, with Claude, costs money).

    python -m flutter_ai llm-benchmark --preset uav_flap --seeds 5     # -> outputs/<preset>/llm_benchmark.json

The LLM (backend from FLUTTER_AI_LLM_BACKEND; local Ollama by default) supervises the neural policy and LQG on
the standard scenarios with its measured latency applied in simulation time and one question at a time. Next
to it, over the same seeds: no supervisor, the rule supervisor and a random supervisor. Recorded per decision:
latency, source (llm / llm_fallback) and the mode the rules would have chosen for the same report.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .experiments import _aggregate, draw_scenario, make_controller, paired_statistics
from .flutter import flutter_speed
from .params import WingParams
from .paths import OUTPUTS_DIR
from .scenarios import scenarios_for
from .simulation import simulate
from .supervisor import RandomSupervisor, RuleSupervisor, llm_status, make_llm_supervisor
from .supervisor.modes import MODE_ORDER

CONTROLLERS = ("neural", "lqg")
SUPERVISORS = ("none", "rules", "random", "llm")


class ShadowedLLM:
    """The LLM supervisor, plus a record of what the rules would have proposed for the very same report."""

    name = "llm"

    def __init__(self, llm):
        self.llm, self.rules = llm, RuleSupervisor()
        self.log: list[dict] = []

    def reset(self) -> None:
        self.llm.reset()
        self.rules.reset()

    def decide(self, f):
        d = self.llm.decide(f)
        rule_mode, _ = self.rules._propose(f)
        self.log.append({"time": d.time, "mode": d.mode, "rules_mode": rule_mode, "source": d.source,
                         "latency_s": d.latency_s, "risk": f.risk, "energy_growth": f.energy_growth})
        return d


def run_llm_benchmark(p: WingParams, seeds=tuple(range(5)), out_dir: Path | None = None, make_figure: bool = True,
                      log=print) -> dict:
    status = llm_status()
    if not status["available"]:
        raise RuntimeError(f"LLM supervisor unavailable: {status['detail']}")
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    Uf = flutter_speed(p)
    res: dict = {"metadata": {"version": __version__, "backend": status["backend"], "model": status["model"],
                              "seeds": list(seeds),
                              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "note": "Every value in this file is produced by flutter_ai.llm_benchmark."},
                 "scenarios": {}, "decisions": []}
    log(f"LLM benchmark on '{p.name}' with {status['label']}, seeds {list(seeds)}")
    t0 = time.time()
    for sname, spec in scenarios_for(p.name).items():
        base = spec["scenario"]
        raw, examples = {}, {}
        for ctrl in CONTROLLERS:
            for sup in SUPERVISORS:
                key = ctrl if sup == "none" else f"{ctrl}+{sup}"
                per_seed = []
                for s in seeds:
                    shadow = ShadowedLLM(make_llm_supervisor()) if sup == "llm" else None
                    supervisor = {"none": None, "rules": RuleSupervisor(), "random": RandomSupervisor(seed=s),
                                  "llm": shadow}[sup]
                    r = simulate(p, make_controller(ctrl), draw_scenario(base, s), Uf, supervisor)
                    per_seed.append(r.metrics)
                    if s == seeds[0]:
                        examples[key] = r.metrics
                    if shadow is not None:
                        res["decisions"] += [{"scenario": sname, "controller": ctrl, "seed": s, **d}
                                             for d in shadow.log]
                raw[key] = per_seed
        table = {k: _aggregate(v) for k, v in raw.items()}
        stats = {}
        for ctrl in CONTROLLERS:
            for sup in ("rules", "random", "llm"):
                stats[f"{ctrl}+{sup} vs {ctrl}"] = paired_statistics(raw[f"{ctrl}+{sup}"], raw[ctrl])
            stats[f"{ctrl}+llm vs {ctrl}+rules"] = paired_statistics(raw[f"{ctrl}+llm"], raw[f"{ctrl}+rules"])
        res["scenarios"][sname] = {"cases": table, "statistics": stats, "seed0": examples}
        log(f"  {sname}: " + "  ".join(
            f"{k} J {v['cost_J']['mean']:.3f}" if v.get("cost_J") else f"{k} {v['failures']} fail"
            for k, v in table.items()))

    dec = [d for d in res["decisions"] if d["source"] == "llm"]
    lat = np.array([d["latency_s"] for d in dec]) if dec else np.array([np.nan])
    res["summary"] = {
        "decisions": len(res["decisions"]),
        "fallbacks": sum(d["source"] == "llm_fallback" for d in res["decisions"]),
        "latency_s": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "max": float(np.max(lat))},
        "agreement_with_rules": float(np.mean([d["mode"] == d["rules_mode"] for d in dec])) if dec else None,
        "mean_mode_step_vs_rules": float(np.mean([MODE_ORDER.index(d["mode"]) - MODE_ORDER.index(d["rules_mode"])
                                                  for d in dec])) if dec else None,
        "wall_time_s": round(time.time() - t0, 1),
    }
    s = res["summary"]
    log(f"  decisions {s['decisions']}, fallbacks {s['fallbacks']}, latency median {s['latency_s']['median']:.2f} s "
        f"(p95 {s['latency_s']['p95']:.2f} s), agreement with rules {s['agreement_with_rules']}, "
        f"mean mode step vs rules {s['mean_mode_step_vs_rules']}")
    path = out_dir / "llm_benchmark.json"
    path.write_text(json.dumps(res, indent=2))
    log(f"Saved {path}")
    if make_figure:
        from . import plots

        bench = out_dir / "results.json"
        tradeoff = json.loads(bench.read_text())["tradeoff"] if bench.exists() else None
        log(f"Saved {plots.llm_figure(res, tradeoff, out_dir / 'fig11_llm_supervisor.png')}")
    return res
