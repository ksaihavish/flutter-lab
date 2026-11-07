"""
Evaluate the learned fault diagnoser on evaluation flights (seeds disjoint from training and validation).

    python -m flutter_ai eval-diagnoser --preset uav_flap   # -> outputs/<preset>/diagnoser_eval.json + fig14_diagnoser.png

Splits (diagnoser_data): `eval` (known faults, in distribution), `eval_held_out` (loop_delay, flap_effectiveness:
never seen in training or calibration), `eval_benchmark` (every fault under the fault-tolerant benchmark's conditions).
Reported:
  - window level (raw per-window predictions): precision, recall, F1 and AUROC per known fault, exact match of the
    whole label set, recall by the number of simultaneous faults;
  - flight level (confirmed alarms, the deployed rule): detection rate and delay after onset per known fault, lead
    time before a flutter failure, false alarms on healthy flights and before any onset, alarms of a fault that is not
    there (misattribution);
  - unknown faults: detection rate and delay of the confirmed unknown alarm, flight- and window-level AUROC, which known
    faults the classifier names instead, the in-distribution false-alarm rate; the Mahalanobis score as a secondary.
  - timing: model inference and feature extraction per diagnosis.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..params import WingParams
from ..paths import OUTPUTS_DIR
from .diagnoser import HELD_OUT, KNOWN, DiagnoserModel, baseline_stats, confirmed, model_path, window_features
from .diagnoser_data import ALL_KINDS, MIN_LEVEL_DEFAULT, SPLITS, Dataset, build_split, diagnosis_indices, min_level
from .diagnoser_training import flight_max_confirmed
from .faults import FAULTS

FAULT_COLORS = {"servo_slew": "#2a78d6", "stiffness_loss": "#eb6834", "sensor_noise": "#1baf7a",
                "turbulence_rise": "#eda100", "loop_delay": "#4a3aa7", "flap_effectiveness": "#008300"}
REFERENCE_COLOR = "#898781"  # in-distribution flights: a recessive baseline
SEVERITY_BINS = ((0.5, 0.8), (0.8, 1.0), (1.0, 1.31))  # fault size / benchmark severity


def auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)."""
    from scipy.stats import rankdata

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _stats(x: list[float]) -> dict:
    if not x:
        return {"n": 0, "median": None, "p25": None, "p75": None, "p90": None}
    a = np.asarray(x, float)
    return {"n": len(a), "median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90))}


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


class Scored:
    """A split with the model's outputs and the confirmed (deployed) alarms per window."""

    def __init__(self, ds: Dataset, model: DiagnoserModel):
        self.ds = ds
        self.prob, self.rec, self.maha = model.predict(ds.X)
        raw = np.column_stack([self.prob >= model.prob_threshold, self.rec > model.ood_threshold,
                               self.maha > model.meta["mahalanobis_threshold"]])
        self.conf = np.zeros_like(raw)
        for i in range(len(ds.flights)):
            sel = ds.flight == i
            self.conf[sel] = confirmed(raw[sel])
        self.known, self.unknown, self.unknown_maha = self.conf[:, :len(KNOWN)], self.conf[:, -2], self.conf[:, -1]

    def windows(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.ds.flight == i)


def _first(times: np.ndarray, flags: np.ndarray, after: float) -> float | None:
    hit = flags & (times >= after)
    return float(times[np.argmax(hit)]) if hit.any() else None


def window_level(s: Scored, model: DiagnoserModel) -> dict:
    Y, M = s.ds.targets()
    pred = s.prob >= model.prob_threshold
    per: dict = {}
    for j, c in enumerate(KNOWN):
        y, p, m = Y[:, j], pred[:, j], M[:, j]
        tp, fp, fn = int(np.sum(y & p & m)), int(np.sum(~y & p & m)), int(np.sum(y & ~p & m))
        per[c] = {"precision": _rate(tp, tp + fp), "recall": _rate(tp, tp + fn), "f1": _rate(2 * tp, 2 * tp + fp + fn),
                  "auroc": auroc(s.prob[m & y, j], s.prob[m & ~y, j]), "positive_windows": int(np.sum(y & m)),
                  "windows": int(m.sum())}
    full = np.all(M, axis=1)
    n_present = np.sum(Y, axis=1)
    by_n = {}
    for n in (1, 2, 3):
        rec = {}
        for j, c in enumerate(KNOWN):
            sel = M[:, j] & Y[:, j] & (n_present == n)
            rec[c] = {"recall": _rate(int(np.sum(pred[sel, j])), int(sel.sum())), "positive_windows": int(sel.sum())}
        by_n[str(n)] = rec
    tp = sum(int(np.sum(Y[:, j] & pred[:, j] & M[:, j])) for j in range(len(KNOWN)))
    fp = sum(int(np.sum(~Y[:, j] & pred[:, j] & M[:, j])) for j in range(len(KNOWN)))
    fn = sum(int(np.sum(Y[:, j] & ~pred[:, j] & M[:, j])) for j in range(len(KNOWN)))
    f1s = [v["f1"] for v in per.values() if v["f1"] is not None]
    return {"per_class": per, "macro_f1": float(np.mean(f1s)) if f1s else None, "micro_f1": _rate(2 * tp, 2 * tp + fp + fn),
            "exact_match": _rate(int(np.sum(np.all(pred[full] == Y[full], axis=1))), int(full.sum())),
            "exact_match_windows": int(full.sum()), "recall_by_simultaneous_faults": by_n,
            "note": "Raw per-window predictions at the probability threshold; windows where a fault has started but is "
                    "below its label level are excluded for that fault."}


def detection(s: Scored) -> dict:
    """Confirmed detection of each known fault after its onset (flight level)."""
    ds = s.ds
    rows: dict[str, list[dict]] = {c: [] for c in KNOWN}
    for i, fl in enumerate(ds.flights):
        w = s.windows(i)
        t = ds.t[w]
        for f in fl["spec"]["faults"]:
            if f["kind"] not in KNOWN:
                continue
            j, col = KNOWN.index(f["kind"]), ALL_KINDS.index(f["kind"])
            reached = w[ds.level[w, col] >= min_level(f["kind"])]
            t_det = _first(t, s.known[w, j], f["onset"])
            rows[f["kind"]].append({
                "eligible": len(reached) > 0, "t_reach": float(ds.t[reached[0]]) if len(reached) else None,
                "onset": f["onset"], "t_det": t_det, "failed": fl["failed"], "failure_time": fl["failure_time_s"],
                "size": f["severity"] / FAULTS[f["kind"]]["severity"]})
    out = {}
    for c, rs in rows.items():
        el = [r for r in rs if r["eligible"]]
        det = [r for r in el if r["t_det"] is not None]
        by_sev = {}
        for lo, hi in SEVERITY_BINS:
            b = [r for r in el if lo <= r["size"] < hi]
            by_sev[f"{lo:g}-{min(hi, 1.3):g}"] = {"faults": len(b),
                                                   "rate": _rate(sum(r["t_det"] is not None for r in b), len(b))}
        out[c] = {"faults": len(rs), "eligible": len(el), "detected": len(det), "rate": _rate(len(det), len(el)),
                  "delay_from_onset_s": _stats([r["t_det"] - r["onset"] for r in det]),
                  "delay_from_label_level_s": _stats([max(0.0, r["t_det"] - r["t_reach"]) for r in det]),
                  "lead_before_failure_s": _stats([r["failure_time"] - r["t_det"] for r in det if r["failed"]]),
                  "missed_then_failed": sum(1 for r in el if r["t_det"] is None and r["failed"]),
                  "by_size": by_sev, "delays_from_onset_s": [round(r["t_det"] - r["onset"], 3) for r in det]}
    return out


def false_alarms(s: Scored) -> dict:
    ds = s.ds
    healthy_fl = [i for i, fl in enumerate(ds.flights) if not fl["spec"]["faults"] and fl["n_windows"]]
    known_any = np.any(s.known, axis=1)
    any_alarm = known_any | s.unknown
    k_fl = [i for i in healthy_fl if known_any[s.windows(i)].any()]
    u_fl = [i for i in healthy_fl if s.unknown[s.windows(i)].any()]
    a_fl = [i for i in healthy_fl if any_alarm[s.windows(i)].any()]
    healthy_w = ds.healthy_window()
    in_healthy_flight = np.isin(ds.flight, healthy_fl)
    pre = healthy_w & ~in_healthy_flight
    wrong = {}
    for j, c in enumerate(KNOWN):
        without = [i for i, fl in enumerate(ds.flights) if fl["n_windows"]
                   and c not in {f["kind"] for f in fl["spec"]["faults"]}]
        alarm = [i for i in without if s.known[s.windows(i), j].any()]
        other = [i for i in without if ds.flights[i]["spec"]["faults"]]
        wrong[c] = {"flights_without": len(without), "rate": _rate(len(alarm), len(without)),
                    "healthy_flights_rate": _rate(len(set(alarm) - set(other)), len(without) - len(other)),
                    "other_fault_flights_rate": _rate(len(set(alarm) & set(other)), len(other))}
    return {"healthy_flights": len(healthy_fl), "known_alarm_flights": len(k_fl), "unknown_alarm_flights": len(u_fl),
            "any_alarm_flights": len(a_fl), "any_alarm_rate": _rate(len(a_fl), len(healthy_fl)),
            "any_alarm_flight_details": [
                {"seed": ds.flights[i]["spec"]["seed"], "flown": ds.flights[i]["spec"]["supervision"],
                 "forced_mode": ds.flights[i]["spec"]["forced_mode"], "slowed": ds.flights[i]["slowed"],
                 "alarms": sorted({c for j, c in enumerate(KNOWN) if s.known[s.windows(i), j].any()}
                                  | ({"unknown"} if s.unknown[s.windows(i)].any() else set())),
                 "first_alarm_s": _first(ds.t[s.windows(i)], any_alarm[s.windows(i)], 0.0)} for i in a_fl],
            "healthy_flight_window_alarm_share": float(np.mean(any_alarm[in_healthy_flight])) if in_healthy_flight.any() else None,
            "pre_onset_window_alarm_share": float(np.mean(any_alarm[pre])) if pre.any() else None,
            "wrong_class_alarms": wrong}


def held_out(s: Scored, ref: Scored, model: DiagnoserModel) -> dict:
    """Unknown-fault detection for the held-out kinds; `ref` = in-distribution evaluation flights."""
    ds = s.ds
    ref_ok = [i for i, fl in enumerate(ref.ds.flights) if fl["n_windows"]]
    ref_max = flight_max_confirmed(ref.rec, ref.ds.flight, len(ref.ds.flights))[ref_ok]
    ref_max_m = flight_max_confirmed(ref.maha, ref.ds.flight, len(ref.ds.flights))[ref_ok]
    fmax = flight_max_confirmed(s.rec, ds.flight, len(ds.flights))
    fmax_m = flight_max_confirmed(s.maha, ds.flight, len(ds.flights))
    out: dict = {"score": "reconstruction error (primary, chosen before evaluation)",
                 "threshold": model.ood_threshold, "mahalanobis_threshold": model.meta["mahalanobis_threshold"],
                 "in_distribution_flights": len(ref_ok),
                 "in_distribution_unknown_alarm_rate": float(np.mean(ref_max > model.ood_threshold)),
                 "in_distribution_unknown_alarm_rate_mahalanobis":
                     float(np.mean(ref_max_m > model.meta["mahalanobis_threshold"])),
                 "kinds": {}}
    ref_w = np.ones(len(ref.ds.t), bool)
    for kind in HELD_OUT:
        col = ALL_KINDS.index(kind)
        fl_idx = [i for i, fl in enumerate(ds.flights) if fl["n_windows"]
                  and any(f["kind"] == kind for f in fl["spec"]["faults"])]
        det, det_m, anyal, lead, names = [], [], [], [], {c: 0 for c in KNOWN}
        for i in fl_idx:
            w = s.windows(i)
            t, onset = ds.t[w], ds.flights[i]["spec"]["faults"][0]["onset"]
            tu = _first(t, s.unknown[w], onset)
            tm = _first(t, s.unknown_maha[w], onset)
            ta = _first(t, s.unknown[w] | np.any(s.known[w], axis=1), onset)
            det.append(None if tu is None else tu - onset)
            det_m.append(None if tm is None else tm - onset)
            anyal.append(None if ta is None else ta - onset)
            if ds.flights[i]["failed"] and ta is not None:
                lead.append(ds.flights[i]["failure_time_s"] - ta)
            for j, c in enumerate(KNOWN):
                names[c] += bool(_first(t, s.known[w, j], onset) is not None)
        n = len(fl_idx)
        win = (ds.level[:, col] >= MIN_LEVEL_DEFAULT)
        out["kinds"][kind] = {
            "flights": n, "failed_flights": sum(ds.flights[i]["failed"] for i in fl_idx),
            "unknown_detection_rate": _rate(sum(d is not None for d in det), n),
            "unknown_delay_from_onset_s": _stats([d for d in det if d is not None]),
            "auroc_flight": auroc(fmax[fl_idx], ref_max),
            "auroc_window": auroc(s.rec[win], ref.rec[ref_w]),
            "any_alarm_rate": _rate(sum(d is not None for d in anyal), n),
            "any_alarm_delay_from_onset_s": _stats([d for d in anyal if d is not None]),
            "any_alarm_lead_before_failure_s": _stats(lead),
            "named_as_known_fault_rate": {c: _rate(v, n) for c, v in names.items()},
            "mahalanobis": {"detection_rate": _rate(sum(d is not None for d in det_m), n),
                            "delay_from_onset_s": _stats([d for d in det_m if d is not None]),
                            "auroc_flight": auroc(fmax_m[fl_idx], ref_max_m),
                            "auroc_window": auroc(s.maha[win], ref.maha[ref_w])},
        }
    return out


def benchmark_conditions(s: Scored) -> dict:
    ds = s.ds
    out = {}
    for kind in FAULTS:
        idx = [i for i, fl in enumerate(ds.flights) if [f["kind"] for f in fl["spec"]["faults"]] == ([] if kind == "none" else [kind])]
        rows = []
        for i in idx:
            w, fl = s.windows(i), ds.flights[i]
            t = ds.t[w]
            onset = fl["spec"]["faults"][0]["onset"] if fl["spec"]["faults"] else 0.0
            any_al = np.any(s.known[w], axis=1) | s.unknown[w]
            right = s.known[w, KNOWN.index(kind)] if kind in KNOWN else s.unknown[w]
            rows.append({"any": _first(t, any_al, onset), "right": _first(t, right, onset), "failed": fl["failed"],
                         "failure": fl["failure_time_s"], "onset": onset})
        n = len(rows)
        out[kind] = {
            "flights": n, "failed": sum(r["failed"] for r in rows),
            "response_checked": "none" if kind == "none" else kind if kind in KNOWN else "unknown alarm",
            "alarm_rate": _rate(sum(r["any"] is not None for r in rows), n),
            "right_alarm_rate": None if kind == "none" else _rate(sum(r["right"] is not None for r in rows), n),
            "right_alarm_delay_from_onset_s": _stats([r["right"] - r["onset"] for r in rows if r["right"] is not None]),
            "any_alarm_delay_from_onset_s": _stats([r["any"] - r["onset"] for r in rows if r["any"] is not None]),
            "lead_before_failure_s": _stats([r["failure"] - r["any"] for r in rows if r["failed"] and r["any"] is not None]),
        }
    return out


def timing(p: WingParams, model: DiagnoserModel, flutter_speed: float, flutter_freq_hz: float) -> dict:
    """Wall time per diagnosis on this machine: the model alone, and the feature extraction (seed 33000)."""
    from .faults import Fault
    from .simulate import ft_simulate

    x = np.zeros((1, len(model.mean)))
    for _ in range(50):
        model.predict(x)
    ts = []
    for _ in range(2000):
        t0 = time.perf_counter()
        model.predict(x)
        ts.append(time.perf_counter() - t0)
    r = ft_simulate(p, flutter_speed, flutter_freq_hz, Fault("none"), None, duration=15.0, seed=33000)
    k_base = int(np.argmax(r.t >= 4.5))
    base = baseline_stats(p, 0.005, r.y, r.u, r.airspeed, r.mode, k_base)
    fs = []
    for k in diagnosis_indices(r.t, k_base)[5:]:
        t0 = time.perf_counter()
        window_features(p, 0.005, flutter_speed, r.y, r.u, r.airspeed, r.mode, k, base)
        fs.append(time.perf_counter() - t0)
    return {"model_inference_us_median": 1e6 * float(np.median(ts)), "model_inference_us_p99": 1e6 * float(np.percentile(ts, 99)),
            "feature_extraction_ms_median": 1e3 * float(np.median(fs)), "feature_extraction_windows": len(fs),
            "note": "Single window, NumPy, on the (shared, busy) development machine."}


def evaluate_diagnoser(p: WingParams, model: DiagnoserModel | None = None, workers: int = 4, out_dir: Path | None = None,
                       make_figure: bool = True, log=print) -> dict:
    from ..flutter import flutter_analysis

    t0 = time.time()
    model = model or DiagnoserModel.load(model_path(p.name))
    splits = {s: build_split(p, s, workers, log) for s in ("eval", "eval_held_out", "eval_benchmark")}
    sc = {s: Scored(ds, model) for s, ds in splits.items()}
    fa = flutter_analysis(p)
    assert fa.flutter_speed is not None and fa.flutter_frequency_hz is not None
    ev = sc["eval"]
    res: dict = {
        "metadata": {"version": __version__, "preset": p.name,
                     "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "model": str(model_path(p.name).name), "model_trained_at": model.meta.get("trained_at"),
                     "splits": {s: {"first_seed": SPLITS[s]["first_seed"], "flights": len(d.flights),
                                    "windows": int(len(d.t)),
                                    "flights_without_windows": sum(f["n_windows"] == 0 for f in d.flights)}
                                for s, d in splits.items()},
                     "training_splits": {"train": SPLITS["train"]["first_seed"], "val": SPLITS["val"]["first_seed"]},
                     "prob_threshold": model.prob_threshold.tolist(), "ood_threshold": model.ood_threshold,
                     "confirm": model.meta["confirm"], "window_s": model.meta["window_s"],
                     "diagnose_every_s": model.meta["diagnose_every_s"],
                     "note": "Every value in this file is produced by flutter_ai.fault_tolerance.diagnoser_eval."},
        "in_distribution": {"flights": len(ev.ds.flights), "window_level": window_level(ev, model),
                            "detection": detection(ev), "false_alarms": false_alarms(ev)},
        "held_out": held_out(sc["eval_held_out"], ev, model),
        "benchmark_conditions": benchmark_conditions(sc["eval_benchmark"]),
        "timing": timing(p, model, fa.flutter_speed, fa.flutter_frequency_hz),
    }
    res["metadata"]["wall_time_s"] = round(time.time() - t0, 1)
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "diagnoser_eval.json"
    path.write_text(json.dumps(res, indent=2))
    wl, det, fa_ = res["in_distribution"]["window_level"], res["in_distribution"]["detection"], res["in_distribution"]["false_alarms"]
    log("  window F1: " + ", ".join(f"{c} {v['f1']:.2f}" for c, v in wl["per_class"].items() if v["f1"] is not None))
    log("  detected: " + ", ".join(f"{c} {v['rate']:.0%} (median {v['delay_from_onset_s']['median']:.1f} s)"
                                   for c, v in det.items() if v["rate"] is not None and v["delay_from_onset_s"]["median"] is not None))
    log(f"  healthy flights with any alarm: {fa_['any_alarm_flights']}/{fa_['healthy_flights']}")
    for k, v in res["held_out"]["kinds"].items():
        log(f"  held out {k}: unknown alarm {v['unknown_detection_rate']:.0%}, any alarm {v['any_alarm_rate']:.0%}, "
            f"AUROC flight {v['auroc_flight']:.2f}")
    log(f"Saved {path}")
    if make_figure:
        log(f"Saved {diagnoser_figure(res, sc, model, out_dir / 'fig14_diagnoser.png')}")
    return res


def diagnoser_figure(res: dict, sc: dict[str, Scored], model: DiagnoserModel, path: Path) -> Path:
    """(a) window precision/recall per known fault, (b) detection delay, (c) unknown-fault score per flight,
    (d) one multi-fault evaluation flight."""
    from .. import plots  # first: selects the Agg backend and the house style (rcParams); ink colours

    import matplotlib.pyplot as plt

    INK, INK2, MUTED = plots.INK, plots.INK2, plots.MUTED
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    (ax1, ax2), (ax3, ax4) = axes
    wl = res["in_distribution"]["window_level"]["per_class"]
    y = np.arange(len(KNOWN))[::-1]
    for yy, c in zip(y, KNOWN):
        v = wl[c]
        ax1.plot([v["precision"]], [yy + 0.12], marker="o", ms=8, color=FAULT_COLORS[c], linestyle="none")
        ax1.plot([v["recall"]], [yy - 0.12], marker="^", ms=8, color=FAULT_COLORS[c], linestyle="none")
        ax1.annotate(f"F1 {v['f1']:.2f}", (1.0, yy), xycoords=("axes fraction", "data"), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8, color=INK2, annotation_clip=False)
    ax1.plot([], [], marker="o", color=MUTED, linestyle="none", label="precision")
    ax1.plot([], [], marker="^", color=MUTED, linestyle="none", label="recall")
    ax1.set_yticks(y, KNOWN)
    ax1.set_xlim(0, 1.04)
    ax1.set_ylim(-0.6, len(KNOWN) - 0.4)
    ax1.set_xlabel("Per-window score (in-distribution evaluation flights)")
    ax1.set_title("(a) Which known fault: precision and recall per window")
    ax1.legend(loc="lower left")

    det = res["in_distribution"]["detection"]
    ds = sc["eval"]
    for yy, c in zip(y, KNOWN):
        d = det[c]["delays_from_onset_s"]
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(d))
        ax2.scatter(d, yy + jitter, s=12, color=FAULT_COLORS[c], alpha=0.55, linewidths=0)
        if d:
            ax2.plot([np.median(d)] * 2, [yy - 0.3, yy + 0.3], color=INK, lw=2)
        rate = det[c]["rate"]
        ax2.annotate(f"{100 * rate:.0f} % detected" if rate is not None else "", (1.0, yy), xycoords=("axes fraction", "data"),
                     xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2, annotation_clip=False)
    ax2.set_yticks(y, KNOWN)
    ax2.set_ylim(-0.6, len(KNOWN) - 0.4)
    ax2.set_xlabel("Confirmed detection after fault onset (s); black bar = median")
    ax2.set_title("(b) How fast: detection delay per known fault")

    ho = sc["eval_held_out"]
    groups = []
    ref_max = flight_max_confirmed(ds.rec, ds.ds.flight, len(ds.ds.flights))
    healthy = np.array([not fl["spec"]["faults"] for fl in ds.ds.flights])
    ok = np.isfinite(ref_max)
    groups.append(("healthy (in distribution)", ref_max[ok & healthy], REFERENCE_COLOR))
    groups.append(("known faults (in distribution)", ref_max[ok & ~healthy], REFERENCE_COLOR))
    ho_max = flight_max_confirmed(ho.rec, ho.ds.flight, len(ho.ds.flights))
    for kind in HELD_OUT:
        sel = np.array([bool(fl["n_windows"]) and fl["spec"]["faults"][0]["kind"] == kind for fl in ho.ds.flights])
        a = res["held_out"]["kinds"][kind]["auroc_flight"]
        groups.append((f"{kind} (held out)\nAUROC {a:.2f}", ho_max[sel], FAULT_COLORS[kind]))
    for g, (_label, vals, color) in enumerate(groups):
        yy = len(groups) - 1 - g
        jitter = np.random.default_rng(1).uniform(-0.2, 0.2, len(vals))
        ax3.scatter(vals, yy + jitter, s=12, color=color, alpha=0.6, linewidths=0)
    ax3.axvline(model.ood_threshold, color=INK, lw=1.0)
    ax3.annotate("alarm threshold (set on validation flights)", (model.ood_threshold, len(groups) - 0.45),
                 xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax3.set_xscale("log")
    ax3.set_ylim(-0.5, len(groups) - 0.25)
    ax3.set_yticks(range(len(groups))[::-1], [g[0] for g in groups], fontsize=8)
    ax3.set_xlabel("Unknown-fault score per flight (largest confirmed reconstruction error)")
    ax3.set_title("(c) Unknown faults: score of each evaluation flight")

    cand = [i for i, fl in enumerate(ds.ds.flights) if len(fl["spec"]["faults"]) >= 2 and not fl["failed"]]
    if cand:
        i = cand[0]
        w = ds.windows(i)
        t = ds.ds.t[w]
        onsets = {f["kind"]: f["onset"] for f in ds.ds.flights[i]["spec"]["faults"]}
        for j, c in enumerate(KNOWN):
            label = f"{c} (starts {onsets[c]:.1f} s)" if c in onsets else f"{c} (absent)"
            ax4.plot(t, ds.prob[w, j], color=FAULT_COLORS[c], lw=1.6, label=label)
        for kind, onset in onsets.items():  # thin marker of each onset in the fault's colour (named in the legend)
            ax4.axvline(onset, color=FAULT_COLORS[kind], lw=0.9)
        ax4.axhline(float(model.prob_threshold[0]), color=MUTED, lw=0.8)
        ax4.annotate("alarm threshold", (t[0], float(model.prob_threshold[0])), xytext=(0, 3),
                     textcoords="offset points", fontsize=7.5, color=INK2)
        ax4.set_ylim(0, 1.1)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Probability")
        ax4.legend(loc="lower left", bbox_to_anchor=(0.0, 0.1), fontsize=7.5)
        ax4.set_title(f"(d) One evaluation flight with {len(ds.ds.flights[i]['spec']['faults'])} faults "
                      f"(seed {ds.ds.flights[i]['spec']['seed']})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
