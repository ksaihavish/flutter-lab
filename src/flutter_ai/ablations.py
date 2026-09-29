"""
Ablations (Phase 4.5): what each part of the neural policy and of the supervisor contributes.

    python -m flutter_ai train --preset uav_flap --variant imitation_only    # also no_rate_penalty, no_airspeed
    python -m flutter_ai ablations --preset uav_flap                         # -> outputs/<preset>/ablations.json

Neural-policy variants, each trained with exactly one component removed (same seed and budget otherwise):
  imitation_only    no differentiable-physics fine-tuning (the policy only imitates the delay-aware LQR)
  no_rate_penalty   fine-tuning without the control-rate penalty
  no_airspeed       the policy never sees the airspeed
Supervisor ablation: neural + random mode choice, next to neural + rules and neural alone.
Every case is compared with the full neural policy over the same seeds (paired statistics).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .controllers import NeuralController
from .experiments import _aggregate, draw_scenario, paired_statistics
from .flutter import flutter_speed
from .params import WingParams
from .paths import MODELS_DIR, OUTPUTS_DIR
from .scenarios import scenarios_for
from .simulation import simulate
from .supervisor import RandomSupervisor, RuleSupervisor

ABLATION_VARIANTS: dict[str, dict] = {
    "full": {},
    "imitation_only": {"finetune_iters": 0},
    "no_rate_penalty": {"rate_weight": 0.0},
    "no_airspeed": {"use_airspeed": False},
}
DELAYS = (0, 1, 2, 3, 4, 6)


def ablation_policy_path(preset: str, variant: str) -> Path:
    return MODELS_DIR / "ablations" / f"neural_policy_{preset}_{variant}.npz"


def _cases(preset: str, log) -> dict:
    """name -> factory(seed) returning (controller, supervisor or None); variants without a policy file are skipped."""
    cases: dict[str, Callable[[int], tuple[NeuralController, Any]]] = {"neural": lambda s: (NeuralController(), None)}
    for variant in ABLATION_VARIANTS:
        if variant == "full":
            continue
        path = ablation_policy_path(preset, variant)
        if path.exists():
            def variant_case(s: int, path: Path = path) -> tuple[NeuralController, Any]:
                return NeuralController(model_path=path), None

            cases[f"neural:{variant}"] = variant_case
        else:
            log(f"  (skipping {variant}: train it with  python -m flutter_ai train --preset {preset} --variant {variant})")
    cases["neural+rules"] = lambda s: (NeuralController(), RuleSupervisor())
    cases["neural+random"] = lambda s: (NeuralController(), RandomSupervisor(seed=s))
    return cases


def run_ablations(p: WingParams, seeds=tuple(range(20)), out_dir: Path | None = None, log=print) -> dict:
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    Uf = flutter_speed(p)
    cases = _cases(p.name, log)
    res: dict = {"metadata": {"version": __version__, "seeds": list(seeds),
                              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "note": "Every value in this file is produced by flutter_ai.ablations."},
                 "cases": list(cases), "scenarios": {}, "delay_sweep": {}}
    log(f"Ablations for '{p.name}': {', '.join(cases)}")
    for sname, spec in scenarios_for(p.name).items():
        base = spec["scenario"]
        raw = {}
        for name, factory in cases.items():
            per_seed = []
            for s in seeds:
                ctrl, sup = factory(s)
                per_seed.append(simulate(p, ctrl, draw_scenario(base, s), Uf, sup).metrics)
            raw[name] = per_seed
        table = {name: _aggregate(m) for name, m in raw.items()}
        stats = {f"{name} vs neural": paired_statistics(raw[name], raw["neural"]) for name in raw if name != "neural"}
        res["scenarios"][sname] = {"cases": table, "statistics": stats}
        log(f"\n  {sname}")
        for name, agg in table.items():
            j = agg.get("cost_J")
            st = stats.get(f"{name} vs neural", {}).get("cost_J", {})
            cmp = (f"  J vs full {100 * st['rel_diff']:+6.1f} % (p = {st['wilcoxon_p']:.2g})" if "rel_diff" in st else "")
            log(f"    {name:24s} failures {agg['failures']:2d}/{agg['n']}  RMS h {agg['rms_h_mm']['mean']:6.2f} mm  "
                f"J {j['mean'] if j else float('nan'):7.3f}  rate {agg['control_rate_rms']['mean']:.3f}{cmp}")

    gust = scenarios_for(p.name)["discrete_gust"]["scenario"]
    for name, factory in cases.items():
        if "+" in name:
            continue  # supervisors do not change the delay tolerance of the inner loop
        rows = []
        for d in DELAYS:
            ctrl, _ = factory(0)
            m = simulate(p, ctrl, replace(gust, input_delay_steps=d), Uf).metrics
            rows.append({"delay_ms": 1000 * d * gust.dt, "exceeded_limits": m["exceeded_limits"],
                         "rms_h_mm": m["rms_h_mm"], "control_rate_rms": m["control_rate_rms"]})
        res["delay_sweep"][name] = rows
        log(f"  delay tolerance {name:24s} survives: {[r['delay_ms'] for r in rows if not r['exceeded_limits']]} ms")

    path = out_dir / "ablations.json"
    path.write_text(json.dumps(res, indent=2))
    log(f"Saved {path}")
    return res
