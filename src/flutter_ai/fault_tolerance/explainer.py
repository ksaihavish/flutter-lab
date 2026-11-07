"""
LLM explainer: after the supervisor has decided, a language model turns the report (incl. the PHYSICS ESTIMATE and
LEARNED DIAGNOSIS blocks) and the decision actually taken into one or two plain sentences for a pilot or operator.

    python -m flutter_ai explainer-eval --preset uav_flap     # -> outputs/<preset>/explainer_eval.json

Why: the owner's architecture lets fast models act and a small local LLM explain. The explanation is off the control
path: it is requested after the decision, its output is never fed back to any supervisor or controller, and the LLM
cannot change the mode (it only restates it).

Guard rails (never silently accepted):
  - schema-constrained JSON through the same safety wrapper as the supervisor (supervisor/llm.py `ask`): an
    explanation plus two enum fields, `fault` (the diagnoser's classes, "unknown", "none") and `mode` (the modes);
  - a faithfulness check against the report and the decision: `mode` must equal the mode taken; `fault` must be the
    diagnosis's top class if that class is at or above its alarm threshold, or "unknown" if the unknown-fault alarm
    is confirmed, or "none" if neither; the text must be 20-400 characters and may not name (by its identifier) a
    fault the diagnosis does not flag;
  - a failed or unfaithful answer is replaced by a deterministic template explanation built from the same report,
    and the reason is recorded.
The check covers the fields and literal fault names only; it cannot verify every claim of the free text (the
evaluation keeps every text for reading).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..params import WingParams
from ..paths import OUTPUTS_DIR
from .diagnostics import FTFeatures, format_ft_report
from .faults import FAULTS, benchmark_fault
from .simulate import FT_MODES, MODE_ORDER, FTDecision

FAULT_WORDS = {"servo_slew": "a slower flap servo", "stiffness_loss": "structural damage (lost pitch stiffness)",
               "sensor_noise": "noisy sensors", "turbulence_rise": "rising turbulence",
               "loop_delay": "a longer control-loop delay", "flap_effectiveness": "flap damage (lost flap effectiveness)"}
TEXT_CHARS = (20, 400)
EVAL_FIRST_SEED = 34000  # development seeds, disjoint from the diagnoser's splits (10000-33000)
EVAL_PER_FLIGHT = 5
EVAL_TIMES = (5.0, 9.0, 13.0, 18.0, 25.0)  # report times sampled per flight (plus mode changes, first)

EXPLAINER_PROMPT = """You explain, after the fact, what the flutter-protection system of a small UAV wing decided.
The wing cruises above its flutter speed; a fast controller keeps it stable. A fast fault classifier (LEARNED
DIAGNOSIS) and an on-board physics estimate (PHYSICS ESTIMATE, if present) analyse the data, and a supervisor has
already chosen the operating mode (DECISION TAKEN). You cannot change that decision.

Write one or two short, plain sentences for the pilot: what the data show and why the chosen mode follows. Use only
facts stated in the report. Then fill in:
- fault: the classifier's most probable fault if its probability is at or above the alarm threshold; "unknown" if an
  unknown fault is confirmed; otherwise "none".
- mode: the mode in DECISION TAKEN."""


def explainer_schema(classes) -> dict:
    """Sent to the backend (enums only: range and length limits are checked in Python, which every backend's
    constrained decoding supports)."""
    return {"type": "object",
            "properties": {"explanation": {"type": "string"},
                           "fault": {"type": "string", "enum": [*classes, "unknown", "none"]},
                           "mode": {"type": "string", "enum": list(MODE_ORDER)}},
            "required": ["explanation", "fault", "mode"], "additionalProperties": False}


def decision_block(f: FTFeatures, d: FTDecision) -> str:
    change = "keep the current mode" if d.mode == f.mode else f"change from '{f.mode}'"
    return f"""

DECISION TAKEN (by the {d.source} supervisor at {d.time:.1f} s, before this explanation was requested):
- Mode: {d.mode} ({change}): {FT_MODES[d.mode]['description']}
- Supervisor's reason: {d.reasoning}"""


def explainer_report(f: FTFeatures, d: FTDecision) -> str:
    return format_ft_report(f) + decision_block(f, d)


def _top(diagnosis: dict) -> tuple[str, float, bool]:
    probs = diagnosis["probabilities"]
    top = max(probs, key=lambda c: probs[c])
    return top, probs[top], probs[top] >= diagnosis["prob_threshold"][top]


def expected_faults(diagnosis: dict) -> set[str]:
    """The `fault` answers faithful to the diagnosis: its top class if flagged, "unknown" if the unknown alarm is
    confirmed; "none" only if neither."""
    top, _, flagged = _top(diagnosis)
    out = {top} if flagged else set()
    if diagnosis["unknown"]:
        out.add("unknown")
    return out or {"none"}


def faithfulness(answer: dict, f: FTFeatures, d: FTDecision) -> list[str]:
    """Why the answer is not faithful to the report and the decision (empty list: faithful)."""
    dg = f.diagnosis
    if dg is None:
        raise ValueError("The explainer explains the learned diagnosis; run the diagnoser (diagnose_every_s)")
    reasons = []
    if answer["mode"] != d.mode:
        reasons.append(f"mode '{answer['mode']}' is not the mode taken '{d.mode}'")
    expected = expected_faults(dg)
    if answer["fault"] not in expected:
        reasons.append(f"fault '{answer['fault']}' does not match the diagnosis (expected one of {sorted(expected)})")
    text = answer["explanation"].strip()
    if not TEXT_CHARS[0] <= len(text) <= TEXT_CHARS[1]:
        reasons.append(f"explanation has {len(text)} characters (allowed {TEXT_CHARS[0]}-{TEXT_CHARS[1]})")
    flagged = expected | set(dg["confirmed"]) | {c for c, v in dg["probabilities"].items()
                                                  if v >= dg["prob_threshold"][c]}
    named = {k for k in FAULT_WORDS if re.search(rf"\b{k}\b", text)}
    if named - flagged:
        reasons.append(f"explanation names {sorted(named - flagged)}, which the diagnosis does not flag")
    return reasons


def template_explanation(f: FTFeatures, d: FTDecision) -> tuple[str, str]:
    """Deterministic explanation from the same report: (text, fault field)."""
    dg = f.diagnosis
    if dg is None:
        raise ValueError("The explainer explains the learned diagnosis; run the diagnoser (diagnose_every_s)")
    top, p_top, flagged = _top(dg)
    word = FAULT_WORDS.get(top, top.replace("_", " "))
    if dg["unknown"]:
        what = (f"The fault classifier finds data that match none of its known fault patterns (unknown-fault score "
                f"{dg['ood_score']:.2f}, alarm threshold {dg['ood_threshold']:.2f})")
        what += f"; it also flags {word} ({100 * p_top:.0f} %)" if flagged else ""
        fault = "unknown"
    elif flagged:
        what, fault = f"The fault classifier indicates {word} ({100 * p_top:.0f} %)", top
    else:
        what, fault = f"The fault classifier sees no fault (most likely: {word}, {100 * p_top:.0f} %)", "none"
    m = f.margin
    if m and m.get("flutter_speed") is not None:
        what += f"; the on-board physics estimate puts the flutter speed at {m['flutter_speed']:.1f} m/s"
    act = "keeps" if d.mode == f.mode else "switches to"
    return f"{what}. The supervisor {act} '{d.mode}': {FT_MODES[d.mode]['description']}", fault


@dataclass
class Explanation:
    time: float
    text: str  # what the operator sees
    source: str  # "llm" (passed the faithfulness check) or "template" (LLM failed or was unfaithful)
    fault: str
    mode: str
    llm: dict | None = None  # the LLM's answer as received (None if the call failed)
    rejected: list[str] = field(default_factory=list)  # why the LLM answer was not used
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Explainer:
    """Asks the LLM (through the supervisor's safety wrapper) to explain one decision; never on the control path."""

    def __init__(self, llm=None, backend: str | None = None, max_calls: int = 100):
        from ..supervisor.llm import make_llm_supervisor

        self.llm = llm if llm is not None else make_llm_supervisor(backend, max_calls=max_calls)

    def explain(self, f: FTFeatures, d: FTDecision) -> Explanation:
        if f.diagnosis is None:
            raise ValueError("The explainer explains the learned diagnosis; run the diagnoser (diagnose_every_s)")
        ans = self.llm.ask(EXPLAINER_PROMPT, explainer_report(f, d), explainer_schema(f.diagnosis["probabilities"]))
        text, fault = template_explanation(f, d)
        if ans.data is None:
            return Explanation(f.time, text, "template", fault, d.mode, None, [f"LLM answer failed: {ans.error}"],
                               ans.latency_s, ans.meta | {"raw": ans.raw})
        reasons = faithfulness(ans.data, f, d)
        if reasons:
            return Explanation(f.time, text, "template", fault, d.mode, ans.data, reasons, ans.latency_s, ans.meta)
        return Explanation(f.time, ans.data["explanation"].strip(), "llm", ans.data["fault"], ans.data["mode"],
                           ans.data, [], ans.latency_s, ans.meta)


# ---------------------------------------------------------------------------------------------------------------
# Evaluation on recorded reports from development flights


def eval_specs(first_seed: int = EVAL_FIRST_SEED) -> list[dict]:
    """One flight per fault kind (incl. none) under the fault-tolerant benchmark's conditions, development seeds."""
    return [{"split": "explainer_eval", "seed": first_seed + i,
             "faults": [] if k == "none" else [asdict(benchmark_fault(k))], "supervision": "rules+physics",
             "forced_mode": None, "cruise_ratio": 1.1, "turbulence_rms": 0.25} for i, k in enumerate(FAULTS)]


def select_ticks(ticks: list[dict], n: int = EVAL_PER_FLIGHT, times=EVAL_TIMES) -> list[int]:
    """Deterministic sample of a flight's ticks: its first two mode changes, then the ticks nearest `times`."""
    ok = [i for i, tk in enumerate(ticks) if tk["decision"] is not None and tk["features"]["diagnosis"] is not None]
    chosen = [i for i in ok if ticks[i]["decision"]["mode"] != ticks[i]["features"]["mode"]][:2]
    t = np.array([ticks[i]["features"]["time"] for i in ok])
    for target in times:
        if len(chosen) >= n or not len(t) or target > t[-1] + 1e-9:
            continue
        i = ok[int(np.argmin(np.abs(t - target)))]
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)[:n]


def _stats(x: list[float]) -> dict:
    if not x:
        return {"n": 0, "median": None, "p95": None, "max": None}
    a = np.asarray(x, float)
    return {"n": len(a), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def run_explainer_eval(p: WingParams, workers: int = 3, out_dir: Path | None = None, explainer: Explainer | None = None,
                       specs: list[dict] | None = None, log=print) -> dict:
    """Record development flights, sample a few dozen reports, explain each decision, score the answers."""
    from ..supervisor.llm import llm_status
    from .recording import fly_recorded

    if explainer is None:
        status = llm_status()
        if not status["available"]:
            raise RuntimeError(f"The explainer evaluation needs the LLM: {status['detail']}")
        explainer = Explainer()
    specs = specs if specs is not None else eval_specs()
    t0 = time.time()
    flights: list[dict | None] = [None] * len(specs)
    log(f"Explainer evaluation: flying {len(specs)} development flights (rules + physics, on-board diagnoser)")
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fly_recorded, (p.name, s, "rules+physics", True)): i for i, s in enumerate(specs)}
        for fut in as_completed(futs):
            flights[futs[fut]] = fut.result()
    llm = explainer.llm
    warm = llm.ask(EXPLAINER_PROMPT, "Warm-up: reply with any valid answer.", explainer_schema(["none"]))
    log(f"  warm-up call ({warm.latency_s:.1f} s, {'ok' if warm.data else warm.error}); not counted")
    rows = []
    for fl in flights:
        assert fl is not None
        kind = fl["spec"]["faults"][0]["kind"] if fl["spec"]["faults"] else "none"
        for i in select_ticks(fl["ticks"]):
            tk = fl["ticks"][i]
            f, d = FTFeatures(**tk["features"]), FTDecision(**tk["decision"])
            e = explainer.explain(f, d)
            rows.append({"seed": fl["spec"]["seed"], "true_fault": kind, "time": f.time, "mode_before": f.mode,
                         "decision": {"mode": d.mode, "source": d.source, "reasoning": d.reasoning},
                         "expected_faults": sorted(expected_faults(tk["features"]["diagnosis"])),
                         "explanation": e.to_dict(), "report": explainer_report(f, d)})
            log(f"  seed {fl['spec']['seed']} {kind:18s} t={f.time:5.1f} s  {e.source:8s} {e.latency_s:5.1f} s  "
                f"{'; '.join(e.rejected)[:90]}")
    n = len(rows)
    ex_ = [r["explanation"] for r in rows]
    answered = [e for e in ex_ if e["llm"] is not None]
    faithful = [e for e in answered if not e["rejected"]]
    reasons: dict[str, int] = {}
    for e in ex_:
        for r in e["rejected"]:
            key = r.split(":")[0] if r.startswith("LLM answer failed") else r.split(" ")[0]
            reasons[key] = reasons.get(key, 0) + 1
    res: dict = {
        "metadata": {"version": __version__, "preset": p.name,
                     "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "backend": llm.backend, "model": llm.model, "seeds": [s["seed"] for s in specs],
                     "flights": [{"seed": fl["spec"]["seed"], "fault": fl["spec"]["faults"], "failed": fl["failed"],
                                  "failure_time_s": fl["failure_time_s"]} for fl in flights if fl is not None],
                     "reports_per_flight": EVAL_PER_FLIGHT, "sampled_times_s": list(EVAL_TIMES),
                     "wall_time_s": round(time.time() - t0, 1),
                     "note": "Every value in this file is produced by flutter_ai.fault_tolerance.explainer. Latency "
                             "was measured on a shared development machine (another LLM experiment may run on the "
                             "same local model at the same time)."},
        "summary": {
            "reports": n, "llm_answers_valid": len(answered), "llm_call_failures": n - len(answered),
            "faithful": len(faithful), "unfaithful": len(answered) - len(faithful),
            "faithfulness_rate": len(faithful) / len(answered) if answered else None,
            "fallback_rate": (n - len(faithful)) / n if n else None,
            "rejection_reasons": reasons,
            "latency_s": _stats([e["latency_s"] for e in ex_]),
            "by_true_fault": {k: {"reports": sum(r["true_fault"] == k for r in rows),
                                  "llm_used": sum(r["true_fault"] == k and r["explanation"]["source"] == "llm"
                                                  for r in rows)} for k in FAULTS},
        },
        "reports": rows,
    }
    out = Path(out_dir or OUTPUTS_DIR / p.name)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "explainer_eval.json"
    path.write_text(json.dumps(res, indent=2))
    s = res["summary"]
    log(f"  {n} reports: {s['faithful']} faithful LLM explanations, {s['unfaithful']} unfaithful, "
        f"{s['llm_call_failures']} failed calls; fallback rate {s['fallback_rate']:.0%}; latency median "
        f"{s['latency_s']['median']:.1f} s")
    log(f"Saved {path}")
    return res
