"""
Post-flight reviewer with a verification gate: the owner's loop "a fast model reacts; the LLM thinks about the past
flight and catches a new scenario; the fast model is then trained for it; guard rails make sure the LLM cannot break
the fast model".

    python -m flutter_ai review-loop --preset uav_flap   # -> outputs/<preset>/review_loop.json + fig17_review_loop.png
                                                         #    accepted candidates: models/diagnoser_<preset>_v2_<kind>.npz

1. Trigger. A development flight (rules + v1 physics estimate, the deployed v1 diagnoser on board) in which the
   diagnoser raised a confirmed UNKNOWN-fault alarm after the fault's onset. Anomalies are searched among flights of
   the two faults v1 never saw (loop delay, flap damage); a real system would not know the kind, the loop does not use it.
2. Reviewer (the local LLM, after the flight, off the control path). It reads a compact log (symptom features over
   time, the diagnosis history, the physics estimate, the outcome) and proposes a hypothesis: a `kind` from the
   simulator's whole fault catalogue (incl. kinds v1 never learned, and the known ones), a severity range (x the
   catalogue's size) and a growth-time range inside fixed bounds, with its evidence. Schema-constrained JSON through
   the supervisor's safety wrapper (supervisor/llm.py `ask`); ranges are checked in Python; a failed or invalid
   answer yields no hypothesis. The LLM never sees or changes weights, thresholds or flights: it names a scenario.
3. Gate (deterministic Python; everything below was fixed before any candidate was scored).
   a. N_TRAIN + N_VAL flights of the hypothesis (seeds from HYP_FIRST_SEED; onset, growth, size, cruise, turbulence
      and way of flying drawn like the diagnoser's own data, the size and growth within the proposed ranges);
   b. a candidate diagnoser trained on v1's train split + the new training flights, the hypothesis as an extra class
      (for a kind v1 already knows: the same classes, more data), calibrated on v1's val split + the new validation
      flights (diagnoser_training.fit_diagnoser; the anomalous flights are never training or calibration data);
   c. ACCEPT only if
      (1) no regression on the regression set fixed in advance: v1's `eval` split (160 flights, known faults and
          healthy flights, never used for training): every v1 class's window F1 drops by at most F1_TOL, and the share
          of healthy flights with any confirmed alarm rises by at most FALSE_ALARM_TOL;
      (2) the hypothesis explains the anomaly: over the windows of the anomalous flights from v1's first unknown
          alarm on, the hypothesised class is confirmed in at least EXPLAIN_CLASS_MIN of them and a confirmed unknown
          alarm remains in at most EXPLAIN_UNKNOWN_MAX.
      Otherwise REJECT and keep v1.
4. Controls. The same gate with a hypothesis forced to be wrong (WRONG_HYPOTHESIS), to see whether it rejects it.
5. For an accepted candidate: fresh flights of the anomaly's kind (FRESH_FIRST_SEED; used nowhere else): detection
   before (v1: unknown alarm, any alarm) and after (candidate: the new class).
The gate checks consistency in the diagnoser's feature space, not physical truth: a wrong hypothesis whose flights
look like the anomaly can pass. The simulator knows the truth, so the result file reports whether each accepted
hypothesis was right.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..params import WingParams
from ..paths import MODELS_DIR, OUTPUTS_DIR
from .diagnoser import KNOWN, DiagnoserModel, confirmed, model_path
from .diagnoser_data import (SUPERVISION_P, Dataset, build_split, merge_datasets, simulate_flights)
from .faults import FAULTS, Fault

CATALOGUE = tuple(k for k in FAULTS if k != "none")
HELD_OUT_KINDS = tuple(k for k in CATALOGUE if FAULTS[k]["held_out"])
SEVERITY_BOUNDS = (0.3, 1.5)  # x the catalogue's (benchmark) size
GROWTH_BOUNDS_S = (0.5, 15.0)
# Gate (fixed before any candidate was scored)
F1_TOL = 0.05
FALSE_ALARM_TOL = 0.05
EXPLAIN_CLASS_MIN = 0.5
EXPLAIN_UNKNOWN_MAX = 0.1
N_TRAIN, N_VAL = 100, 25
# Seeds (development, disjoint from the diagnoser's splits 10000-32055 and the explainer's 34000+)
ANOMALY_FIRST_SEED = 45000  # candidate anomalous flights, ANOMALY_SEARCH per held-out kind
ANOMALY_SEARCH = 8
ANOMALIES_PER_KIND = 3  # reviewed flights per kind (the first one is also the gate's anomaly set together with the rest)
HYP_FIRST_SEED = 50000  # + 1000 x candidate index
FRESH_FIRST_SEED = 47000  # + 100 x kind index
N_FRESH = 20
WRONG_HYPOTHESIS = {"flap_effectiveness": "servo_slew", "loop_delay": "sensor_noise"}

REVIEWER_PROMPT = """You review a flight of a small UAV wing after it ended. The wing cruises above its flutter speed;
a fast controller keeps it stable. During the flight a fast fault classifier, trained on four known faults
(servo_slew, stiffness_loss, sensor_noise, turbulence_rise), raised an UNKNOWN-FAULT alarm: the data matched none of
the patterns it learned.

From the flight log, propose which fault in the simulator's fault catalogue most likely caused the alarm, so that
engineers can generate training flights for it. Give a severity range as a fraction of the catalogue size
(0.3 to 1.5) and a growth-time range in seconds (0.5 to 15), and cite the log values that support your choice.
Your proposal is only a hypothesis: a separate automatic check retrains the classifier and accepts the result only
if nothing else gets worse and the new classifier explains this flight.

Fault catalogue (kind: description at size 1.0):
{catalogue}"""


def reviewer_schema() -> dict:
    """Sent to the backend (types and the enum; ranges are checked in Python)."""
    return {"type": "object",
            "properties": {"kind": {"type": "string", "enum": list(CATALOGUE)},
                           "severity_min": {"type": "number"}, "severity_max": {"type": "number"},
                           "growth_min_s": {"type": "number"}, "growth_max_s": {"type": "number"},
                           "evidence": {"type": "string"}},
            "required": ["kind", "severity_min", "severity_max", "growth_min_s", "growth_max_s", "evidence"],
            "additionalProperties": False}


def check_hypothesis(h: dict) -> None:
    """Raise ValueError unless the ranges are ordered and inside the bounds (the answer is then not used)."""
    for lo, hi, (a, b), unit in (("severity_min", "severity_max", SEVERITY_BOUNDS, ""),
                                 ("growth_min_s", "growth_max_s", GROWTH_BOUNDS_S, " s")):
        if not a <= h[lo] <= h[hi] <= b:
            raise ValueError(f"{lo}/{hi} = {h[lo]}/{h[hi]} must satisfy {a} <= min <= max <= {b}{unit}")
    if not 10 <= len(h["evidence"].strip()) <= 600:
        raise ValueError(f"evidence has {len(h['evidence'].strip())} characters (allowed 10-600)")


def reviewer_prompt() -> str:
    return REVIEWER_PROMPT.format(catalogue="\n".join(f"- {k}: {FAULTS[k]['description']}" for k in CATALOGUE))


# ------------------------------------------------------------------------------------------------ anomalies and logs


def anomaly_spec(kind: str, i: int) -> dict:
    """The i-th candidate anomalous flight of `kind`: one fault near its catalogue size, rules + physics on board."""
    seed = ANOMALY_FIRST_SEED + 100 * HELD_OUT_KINDS.index(kind) + i
    g = np.random.default_rng([seed, 11])
    f = Fault(kind, onset=float(g.uniform(6.0, 12.0)), ramp=float(g.uniform(2.0, 8.0)),
              severity=float(FAULTS[kind]["severity"] * g.uniform(0.8, 1.2)))
    return {"split": "review_anomaly", "seed": seed, "faults": [asdict(f)], "supervision": "rules+physics",
            "forced_mode": None, "cruise_ratio": 1.1, "turbulence_rms": 0.25}


def hypothesis_specs(h: dict, first_seed: int, n: int, split: str) -> list[dict]:
    """Training flights for a hypothesis: drawn like diagnoser_data.flight_spec, one fault of the hypothesised kind with
    size and growth time inside the proposed ranges; a fifth of the flights healthy (the classifier must not learn
    'any flight of this batch is the new fault')."""
    out = []
    for i in range(n):
        seed = first_seed + i
        g = np.random.default_rng([seed, 13])
        faults = [] if g.uniform() < 0.2 else [asdict(Fault(
            h["kind"], onset=float(g.uniform(6.0, 18.0)), ramp=float(g.uniform(h["growth_min_s"], h["growth_max_s"])),
            severity=float(FAULTS[h["kind"]]["severity"] * g.uniform(h["severity_min"], h["severity_max"]))))]
        sup = str(g.choice(list(SUPERVISION_P), p=list(SUPERVISION_P.values())))
        forced = [float(g.uniform(5.0, 25.0)), str(g.choice(["robust", "cautious", "safe"]))] if sup == "forced" else None
        out.append({"split": split, "seed": seed, "faults": faults, "supervision": sup, "forced_mode": forced,
                    "cruise_ratio": float(g.uniform(1.0, 1.15)), "turbulence_rms": float(g.uniform(0.20, 0.30))})
    return out


def fresh_specs(kind: str) -> list[dict]:
    """Fresh evaluation flights of `kind` (as the diagnoser's eval_held_out split draws them), new seeds."""
    from .diagnoser_data import SEVERITY_RANGE, SEVERITY_RANGE_DEFAULT

    out = []
    for i in range(N_FRESH):
        seed = FRESH_FIRST_SEED + 100 * CATALOGUE.index(kind) + i
        g = np.random.default_rng([seed, 7])
        f = Fault(kind, onset=float(g.uniform(6.0, 18.0)), ramp=float(g.uniform(1.0, 12.0)),
                  severity=float(FAULTS[kind]["severity"] * g.uniform(*SEVERITY_RANGE.get(kind, SEVERITY_RANGE_DEFAULT))))
        sup = str(g.choice(list(SUPERVISION_P), p=list(SUPERVISION_P.values())))
        forced = [float(g.uniform(5.0, 25.0)), str(g.choice(["robust", "cautious", "safe"]))] if sup == "forced" else None
        out.append({"split": "review_fresh", "seed": seed, "faults": [asdict(f)], "supervision": sup,
                    "forced_mode": forced, "cruise_ratio": float(g.uniform(1.0, 1.15)),
                    "turbulence_rms": float(g.uniform(0.20, 0.30))})
    return out


def alarms(model: DiagnoserModel, X: np.ndarray) -> np.ndarray:
    """Confirmed alarms per window of one flight: one column per model class, then the unknown alarm."""
    prob, rec, _ = model.predict(X)
    return confirmed(np.column_stack([prob >= model.prob_threshold, rec > model.ood_threshold]))


def first_unknown(model: DiagnoserModel, X: np.ndarray, t: np.ndarray, onset: float) -> float | None:
    if len(t) == 0:
        return None
    hit = alarms(model, X)[:, -1] & (t >= onset)
    return float(t[np.argmax(hit)]) if hit.any() else None


def flight_log(fl: dict, every_s: float = 2.0) -> str:
    """Compact post-flight log for the reviewer (fits the local model's context with the prompt)."""
    ticks = [tk["features"] for tk in fl["ticks"]]
    uf = fl["flutter_speed"]
    lines = [f"FLIGHT LOG. Cruise {fl['cruise']:.1f} m/s = {fl['cruise'] / uf:.2f} x the healthy wing's flutter speed "
             f"({uf:.1f} m/s). Supervisor: symptom rules with an on-board physics estimate.",
             "Outcome: " + (f"flutter, structural limit exceeded at {fl['failure_time_s']:.1f} s."
                            if fl["failed"] else f"survived the {fl['t_end']:.0f} s flight."),
             "Mode changes: " + (", ".join(f"{s['mode']} at {s['time']:.1f} s" for s in fl["switches"]) or "none") + ".",
             f"First unknown-fault alarm: {fl['first_unknown']:.1f} s.",
             "Every 2 s: t [s] | mode | airspeed [m/s] | response [% of limit] | model residual h/pitch/flap [x healthy]"
             " | sensor noise [x healthy] | flap saturated [%] | pitch frequency shift [%] | loop time [ms] |"
             " classifier: most likely known faults (probability) | unknown score/threshold |"
             " physics: flutter speed [m/s], pitch stiffness [x], flap effectiveness [x]"]
    nxt = 4.5
    for f in ticks:
        if f["time"] < nxt - 1e-9:
            continue
        nxt += every_s
        r = f["residual_ratio"]
        shift = ("–" if f["pitch_freq_hz"] is None or f["nominal_freq_hz"] is None
                 else f"{100 * (f['pitch_freq_hz'] / f['nominal_freq_hz'] - 1):+.0f}")
        d = f.get("diagnosis") or {}
        probs = sorted((d.get("probabilities") or {}).items(), key=lambda kv: -kv[1])[:2]
        cls = ", ".join(f"{k} {v:.2f}" for k, v in probs) or "–"
        unk = f"{d['ood_score']:.2f}/{d['ood_threshold']:.2f}" if d else "–"
        m = f.get("margin") or {}
        par = m.get("params") or {}
        phys = ("–" if not m else f"{m['flutter_speed']:.1f}" if m.get("flutter_speed") is not None else "none") + \
               (f", {par['f_alpha']:.2f}, {par['flap_span_fraction']:.2f}" if par else "")
        lines.append(f"{f['time']:4.1f} | {f['mode']} | {f['airspeed']:.1f} | {100 * f['risk']:.0f} | "
                     f"{r.get('h', 1):.1f}/{r.get('alpha', 1):.1f}/{r.get('beta', 1):.1f} | {f['noise_ratio']:.1f} | "
                     f"{100 * f['saturation_fraction']:.0f} | {shift} | {f['loop_delay_ms']:.0f} | {cls} | {unk} | {phys}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------ gate


def regression_scores(model: DiagnoserModel, ds: Dataset) -> dict:
    """Per v1-class window F1 (at the probability threshold) and the healthy-flight any-alarm rate on a split."""
    prob, _, _ = model.predict(ds.X)
    Y, M = ds.targets(KNOWN)
    f1 = {}
    for j, c in enumerate(KNOWN):
        k = model.classes.index(c)
        y, pr, m = Y[:, j], prob[:, k] >= model.prob_threshold[k], M[:, j]
        tp, fp, fn = int(np.sum(y & pr & m)), int(np.sum(~y & pr & m)), int(np.sum(y & ~pr & m))
        f1[c] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    healthy = [i for i, fl in enumerate(ds.flights) if not fl["spec"]["faults"] and fl["n_windows"]]
    alarmed = sum(bool(alarms(model, ds.X[ds.flight == i]).any()) for i in healthy)
    return {"f1": f1, "healthy_flights": len(healthy), "healthy_any_alarm_rate": alarmed / len(healthy) if healthy else None}


def explanation_scores(model: DiagnoserModel, kind: str, anomalies: list[dict]) -> dict:
    """Over the anomalous flights' windows from v1's first unknown alarm on: share with the hypothesised class
    confirmed, and share with a confirmed unknown alarm."""
    j = model.classes.index(kind) if kind in model.classes else None
    cls, unk = [], []
    for a in anomalies:
        sel = np.asarray(a["t"]) >= a["first_unknown"] - 1e-9
        al = alarms(model, np.asarray(a["X"]))[sel]
        cls += list(al[:, j]) if j is not None else [False] * len(al)
        unk += list(al[:, -1])
    return {"windows": len(cls), "class_share": float(np.mean(cls)) if cls else 0.0,
            "unknown_share": float(np.mean(unk)) if unk else 1.0}


def gate_decision(v1: dict, cand: dict, explain: dict) -> dict:
    """ACCEPT iff no regression (F1 of every v1 class within F1_TOL, healthy false alarms within FALSE_ALARM_TOL) and
    the anomaly is explained (class share >= EXPLAIN_CLASS_MIN, unknown share <= EXPLAIN_UNKNOWN_MAX)."""
    reasons = []
    for c, f in v1["f1"].items():
        g = cand["f1"].get(c)
        if f is not None and (g is None or g < f - F1_TOL):
            reasons.append(f"F1 of {c} {f:.3f} -> {g if g is None else round(g, 3)} (tolerance {F1_TOL})")
    fa0, fa1 = v1["healthy_any_alarm_rate"], cand["healthy_any_alarm_rate"]
    if fa0 is not None and (fa1 is None or fa1 > fa0 + FALSE_ALARM_TOL):
        reasons.append(f"healthy-flight false alarms {fa0:.3f} -> {fa1} (tolerance {FALSE_ALARM_TOL})")
    if explain["class_share"] < EXPLAIN_CLASS_MIN:
        reasons.append(f"hypothesised class confirmed in {explain['class_share']:.0%} of the anomaly's windows "
                       f"(needs >= {EXPLAIN_CLASS_MIN:.0%})")
    if explain["unknown_share"] > EXPLAIN_UNKNOWN_MAX:
        reasons.append(f"unknown alarm still in {explain['unknown_share']:.0%} of the anomaly's windows "
                       f"(allowed <= {EXPLAIN_UNKNOWN_MAX:.0%})")
    return {"accepted": not reasons, "reasons": reasons}


def fresh_detection(model: DiagnoserModel, kind: str, flights: Dataset) -> dict:
    """Per fresh flight of `kind`: confirmed detection after onset of the class (if the model has it), of the unknown
    alarm, and of any alarm."""
    j = model.classes.index(kind) if kind in model.classes else None
    cls, unk, anyal, delay = 0, 0, 0, []
    n = 0
    for i, fl in enumerate(flights.flights):
        if not fl["n_windows"]:
            continue
        n += 1
        sel = flights.flight == i
        onset = fl["spec"]["faults"][0]["onset"]
        after = flights.t[sel] >= onset
        al = alarms(model, flights.X[sel])[after]
        hit = al[:, j] if j is not None else np.zeros(len(al), bool)
        cls += bool(hit.any())
        unk += bool(al[:, -1].any())
        anyal += bool(al.any())
        if hit.any():
            delay.append(float(flights.t[sel][after][np.argmax(hit)] - onset))
    return {"flights": n, "class_detected": cls / n if n and j is not None else None, "unknown_alarm": unk / n if n else None,
            "any_alarm": anyal / n if n else None, "class_delay_median_s": float(np.median(delay)) if delay else None}


# ------------------------------------------------------------------------------------------------------------ loop


def review_flight(llm, fl: dict) -> dict:
    """Ask the reviewer about one anomalous flight; returns the hypothesis (or the failure) with the log it read."""
    log = flight_log(fl)
    ans = llm.ask(reviewer_prompt(), log, reviewer_schema(), check_hypothesis)
    true = fl["spec"]["faults"][0]["kind"]
    return {"seed": fl["spec"]["seed"], "true_kind": true, "log": log, "latency_s": ans.latency_s,
            "hypothesis": ans.data, "error": ans.error, "raw": ans.raw,
            "right_kind": None if ans.data is None else ans.data["kind"] == true}


def train_candidate(p: WingParams, h: dict, index: int, v1_train: Dataset, v1_val: Dataset, workers: int, log) -> tuple:
    from .diagnoser_training import fit_diagnoser

    first = HYP_FIRST_SEED + 1000 * index
    tr = simulate_flights(p, hypothesis_specs(h, first, N_TRAIN, "review_train"), workers, log, f"hypothesis {index} train")
    va = simulate_flights(p, hypothesis_specs(h, first + 500, N_VAL, "review_val"), workers, log, f"hypothesis {index} val")
    classes = KNOWN if h["kind"] in KNOWN else (*KNOWN, h["kind"])
    model, info = fit_diagnoser(p, merge_datasets(v1_train, tr), merge_datasets(v1_val, va), classes, log=log)
    return model, info, first


def run_review_loop(p: WingParams, workers: int = 3, llm=None, out_dir: Path | None = None, log=print) -> dict:
    from ..supervisor.llm import llm_status, make_llm_supervisor
    from .recording import fly_recorded

    t0 = time.time()
    v1 = DiagnoserModel.load(model_path(p.name))
    if llm is None:
        status = llm_status()
        if not status["available"]:
            raise RuntimeError(f"The review loop needs the LLM: {status['detail']}")
        llm = make_llm_supervisor(max_calls=40)
    log(f"Review loop for '{p.name}': searching anomalous development flights ({ANOMALY_SEARCH} per held-out kind)")
    from concurrent.futures import ProcessPoolExecutor

    specs = [anomaly_spec(k, i) for k in HELD_OUT_KINDS for i in range(ANOMALY_SEARCH)]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        flights = list(ex.map(fly_recorded, [(p.name, s, "rules+physics", True) for s in specs]))
    anomalies: dict[str, list[dict]] = {k: [] for k in HELD_OUT_KINDS}
    searched = []
    for fl in flights:
        kind, onset = fl["spec"]["faults"][0]["kind"], fl["spec"]["faults"][0]["onset"]
        fu = first_unknown(v1, np.asarray(fl["X"]), np.asarray(fl["t"]), onset)
        searched.append({"seed": fl["spec"]["seed"], "kind": kind, "failed": bool(fl["failed"]), "first_unknown_s": fu})
        if fu is not None and len(anomalies[kind]) < ANOMALIES_PER_KIND:
            anomalies[kind].append(fl | {"first_unknown": fu})
    for k, a in anomalies.items():
        log(f"  {k}: {len(a)} anomalous flights (v1 unknown alarm) among {ANOMALY_SEARCH}")

    warm = llm.ask(reviewer_prompt(), "Warm-up: reply with any valid answer.", reviewer_schema())
    log(f"  warm-up call ({warm.latency_s:.1f} s, {'ok' if warm.data else warm.error}); not counted")
    reviews = []
    for k, flist in anomalies.items():
        for fl in flist:
            rv = review_flight(llm, fl)
            reviews.append(rv)
            h = rv["hypothesis"]
            log(f"  review seed {rv['seed']} (truth {k}): " + (f"{h['kind']} severity {h['severity_min']}-"
                f"{h['severity_max']}, growth {h['growth_min_s']}-{h['growth_max_s']} s ({rv['latency_s']:.1f} s)"
                if h else f"no hypothesis ({rv['error']})"))

    log("  regression set: v1's eval split; v1 training data: its train / val splits (cached)")
    v1_train, v1_val = build_split(p, "train", workers, log), build_split(p, "val", workers, log)
    reg = build_split(p, "eval", workers, log)
    v1_reg = regression_scores(v1, reg)
    candidates = []
    plan = []  # (label, kind of the anomaly, hypothesis, forced?)
    for k in HELD_OUT_KINDS:
        first = next((rv for rv in reviews if rv["true_kind"] == k and rv["hypothesis"] is not None), None)
        if first is not None:
            plan.append(("llm", k, first["hypothesis"], first["seed"]))
        if anomalies[k]:
            wrong = {"kind": WRONG_HYPOTHESIS[k], "severity_min": 0.8, "severity_max": 1.2, "growth_min_s": 2.0,
                     "growth_max_s": 8.0, "evidence": "control: hypothesis forced to be wrong"}
            plan.append(("forced_wrong", k, wrong, None))
    for idx, (source, k, h, seed) in enumerate(plan):
        log(f"  candidate {idx}: {source} hypothesis '{h['kind']}' for the {k} anomalies")
        model, info, first_seed = train_candidate(p, h, idx, v1_train, v1_val, workers, log)
        cand_reg = regression_scores(model, reg)
        expl = explanation_scores(model, h["kind"], anomalies[k])
        v1_expl = explanation_scores(v1, h["kind"], anomalies[k])
        gate = gate_decision(v1_reg, cand_reg, expl)
        row = {"index": idx, "source": source, "anomaly_kind": k, "reviewed_seed": seed, "hypothesis": h,
               "right_kind": h["kind"] == k, "anomaly_seeds": [a["spec"]["seed"] for a in anomalies[k]],
               "training_seeds": [first_seed, first_seed + N_TRAIN - 1], "validation_seeds":
               [first_seed + 500, first_seed + 500 + N_VAL - 1], "classes": list(model.classes),
               "regression": cand_reg, "explanation": expl, "explanation_v1": v1_expl, "gate": gate,
               "candidate_validation_f1": {c: v["f1"] for c, v in info["validation_window_scores"].items()},
               "ood_threshold": model.ood_threshold}
        log(f"    gate: {'ACCEPT' if gate['accepted'] else 'REJECT'}" + ("" if gate["accepted"] else
            " (" + "; ".join(gate["reasons"]) + ")"))
        if gate["accepted"]:
            fresh = simulate_flights(p, fresh_specs(k), workers, log, f"fresh {k}")
            row["fresh"] = {"v1": fresh_detection(v1, k, fresh), "candidate": fresh_detection(model, h["kind"], fresh),
                            "seeds": [fresh.flights[0]["spec"]["seed"], fresh.flights[-1]["spec"]["seed"]]}
            path = MODELS_DIR / f"diagnoser_{p.name}_v2_{h['kind']}_{idx}.npz"
            model.save(path)
            path.with_suffix(".json").write_text(json.dumps(info | {"review": {k2: v for k2, v in row.items()
                                                                               if k2 != "fresh"} | {"fresh": row["fresh"]}},
                                                            indent=2, default=float) + "\n")
            row["model_file"] = path.name
            log(f"    saved {path.name}; fresh {k} flights: v1 unknown alarm {row['fresh']['v1']['unknown_alarm']}, "
                f"candidate class detected {row['fresh']['candidate']['class_detected']}")
        candidates.append(row)

    llm_reviews = [rv for rv in reviews if rv["hypothesis"] is not None]
    res = {
        "metadata": {"version": __version__, "preset": p.name,
                     "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "backend": llm.backend, "model": llm.model, "v1_model": model_path(p.name).name,
                     "gate": {"f1_tolerance": F1_TOL, "false_alarm_tolerance": FALSE_ALARM_TOL,
                              "explain_class_min": EXPLAIN_CLASS_MIN, "explain_unknown_max": EXPLAIN_UNKNOWN_MAX,
                              "regression_set": "v1 eval split (seeds 30000-30159)", "n_train": N_TRAIN, "n_val": N_VAL},
                     "wall_time_s": round(time.time() - t0, 1),
                     "note": "Every value in this file is produced by flutter_ai.fault_tolerance.review. Development "
                             "seeds only (>= 45000 for this study)."},
        "anomaly_search": searched,
        "reviews": [{k: v for k, v in rv.items()} for rv in reviews],
        "summary": {"reviews": len(reviews), "valid_hypotheses": len(llm_reviews),
                    "right_kind": sum(bool(rv["right_kind"]) for rv in llm_reviews),
                    "latency_s_median": float(np.median([rv["latency_s"] for rv in reviews])) if reviews else None,
                    "gate": [{"source": c["source"], "anomaly": c["anomaly_kind"], "hypothesis": c["hypothesis"]["kind"],
                              "accepted": c["gate"]["accepted"], "right_kind": c["right_kind"]} for c in candidates]},
        "v1_regression": v1_reg,
        "candidates": candidates,
    }
    out = Path(out_dir or OUTPUTS_DIR / p.name)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "review_loop.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    log(f"Saved {path}")
    from .. import plots

    log(f"Saved {plots.review_figure(res, out / 'fig17_review_loop.png')}")
    return res
