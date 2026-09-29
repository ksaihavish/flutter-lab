"""
Learned fault diagnoser: a small encoder that reads on-board data every DIAGNOSE_EVERY_S and says which of the known
faults are present (multi-label; combinations possible) and whether the data match none of them (an unknown-fault
score). Inference is pure NumPy (microseconds); training (torch) is in diagnoser_training.py, the flight data in
diagnoser_data.py, the evaluation in diagnoser_eval.py.

Why: the on-board LLM needed ~4.4 s per answer and rarely acted (docs/Memory.md §4). A diagnoser analyses the data
fast; a language model can still explain its output (diagnostics.format_diagnosis).

Input: one feature vector per window (the last WINDOW_S of data), all available on board:
  - the nominal model's one-step prediction residual per rate channel (as diagnostics.py, but with each sample's
    airspeed and the servo's known slew limit, so that a speed change or a fast command in healthy turbulence is not
    a model error), the high-frequency content and RMS of each measured displacement, the RMS command rate;
  - one Gauss-Newton step of the pitch-stiffness fit of margin.py: the residual projected on the model's sensitivity
    to f_alpha (the physics of the known structural fault; the held-out faults get no feature of their own);
  - the dominant pitch-frequency shift, saturation, peak motion, energy trend;
  - context: airspeed, speed change over the window, share of time on the neural controller.
Scale-type features are log10 ratios to the same statistics on the healthy wing at take-off (self-calibration, like
the rules' baseline). Features rather than raw time series: at 200 Hz the raw signals are dominated by turbulence,
while the fault signatures are known physics (which channel the model stops predicting); a feature vector keeps the
model tiny, fast and its inputs interpretable.
The measured loop timing is deliberately not an input: a one-line check catches it, and leaving it out makes the
held-out loop-delay fault a test of whether its dynamic symptoms are recognised as unknown.

Model: encoder (features -> EMBED-d embedding), a multi-label head (sigmoid per known fault) and a decoder that
reconstructs the features from the embedding. Unknown-fault score = reconstruction error (the decoder has only seen
healthy and known-fault data), chosen before any evaluation; the Mahalanobis distance of the embedding to the nearest
known fault combination is computed as a secondary score (reported, never used for a decision). Thresholds are
calibrated on validation flights only; a detection counts once it holds at CONFIRM consecutive diagnoses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..aero import JONES_EPS
from ..controllers.costs import reference_scales
from ..model import discrete_state_space, downwash_rows, structural_energy
from ..params import WingParams
from ..paths import MODELS_DIR
from .diagnostics import RESIDUAL_WARMUP_S, healthy_freq
from .faults import FAULTS
from .simulate import FT_MODES, SPEED_STEP, controller_of

KNOWN = ("servo_slew", "stiffness_loss", "sensor_noise", "turbulence_rise")  # the only faults it is trained on
HELD_OUT = tuple(k for k, v in FAULTS.items() if v["held_out"])  # never in training or calibration
WINDOW_S = 2.5  # data per diagnosis: RESIDUAL_WARMUP_S for the rebuilt aero-lag states + 2 s of statistics
DIAGNOSE_EVERY_S = 0.5
BASELINE_S = (0.5, 4.5)  # healthy take-off window (the simulator's diagnostics baseline)
CONFIRM = 2  # consecutive diagnoses a detection must hold (a single noisy window is not an alarm)
MODEL_FORMAT = 1
_EPS = 1e-12


def model_path(preset: str) -> Path:
    return MODELS_DIR / f"diagnoser_{preset}.npz"


def feature_names(p: WingParams) -> tuple[str, ...]:
    d = p.dof_names
    return (*(f"residual_{n}" for n in d), *(f"hf_{n}" for n in d), *(f"rms_{n}" for n in d), "command_rate",
            "stiffness_shift", "freq_shift", "freq_valid", "saturation", "risk", "energy_trend", "airspeed",
            "speed_change", "neural_share")


@dataclass
class WindowStats:
    """Raw statistics of one window (features are these relative to the take-off baseline)."""

    residual: np.ndarray  # RMS one-step residual per rate channel / reference rate
    hf: np.ndarray  # RMS second difference per displacement / reference
    rms: np.ndarray  # RMS displacement / reference
    command_rate: float  # RMS command change per sample / u_max
    stiffness: float  # one-step estimate of the relative pitch-frequency change (uncalibrated)
    freq_hz: float | None  # dominant pitch frequency (None: motion too small)
    saturation: float
    risk: float  # peak |h|, |alpha| as a fraction of the structural limits
    energy_trend: float  # log10 mean structural energy, second half / first half
    airspeed: float  # [m/s] at the window end
    speed_change: float  # [m/s] over the window
    neural_share: float  # share of samples flown on the neural controller


@lru_cache(maxsize=512)
def _stiffness_sensitivity(p: WingParams, U: float, dt: float, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """d(Ad, Bd) / d(relative change of f_alpha), rate rows only (finite difference, cached per airspeed level)."""
    n = p.n_dof
    s0, s1 = discrete_state_space(p, U, dt), discrete_state_space(p.with_overrides(f_alpha=p.f_alpha * (1 + eps)), U, dt)
    return (s1.Ad - s0.Ad)[n:2 * n] / eps, (s1.Bd - s0.Bd)[n:2 * n] / eps


def window_stats(p: WingParams, dt: float, y: np.ndarray, u_cmd: np.ndarray, u_before: np.ndarray,
                 levels: np.ndarray, neural: np.ndarray) -> WindowStats:
    """
    y (T, 2n) measurements, u_cmd (T, m) commands sent at those samples, u_before the command sent just before the
    window, levels (T,) the quantised airspeed of each sample as the simulator uses it, neural (T,) bool.
    The nominal model: 1-sample loop delay and the healthy servo's slew limit; the transition from sample j to j+1
    uses the airspeed level of sample j+1 (as ft_simulate advances the plant).
    """
    n, T = p.n_dof, len(y)
    skip = int(round(RESIDUAL_WARMUP_S / dt))
    if T < skip + int(round(1.0 / dt)):
        raise ValueError(f"A diagnosis window needs at least {RESIDUAL_WARMUP_S + 1.0:g} s of data, got {T * dt:g} s")
    ref = reference_scales(p)
    u_app = np.vstack([np.atleast_2d(u_before), u_cmd[:-1]]).astype(float)
    for j, rate in enumerate(p.u_rate_max or ()):  # started where the command stood; the warm-up absorbs the rest
        step, col = rate * dt, u_app[:, j].tolist()
        act = col[0]
        for i, c in enumerate(col):
            act += min(max(c - act, -step), step)
            col[i] = act
        u_app[:, j] = col
    lv, inv = np.unique(levels, return_inverse=True)
    systems = [discrete_state_space(p, float(v), dt) for v in lv]
    rows = [downwash_rows(p, float(v)) for v in lv]
    Q = (np.einsum("ti,ti->t", y[:, :n], np.array([r[0] for r in rows])[inv])
         + np.einsum("ti,ti->t", y[:, n:2 * n], np.array([r[1] for r in rows])[inv]))
    lam = np.outer(levels, JONES_EPS) / p.b
    decay = np.exp(-lam * dt)
    b0 = (1.0 - decay) / lam
    b1 = (dt / lam - b0 / lam) / dt
    c0, c1 = (b0 - b1).tolist(), b1.tolist()
    dl, q = decay.tolist(), Q.tolist()
    z1, z2 = [0.0], [0.0]
    for k in range(T - 1):  # aero-lag states rebuilt from the measured downwash (as the on-board observer does)
        d, a, b = dl[k + 1], c0[k + 1], c1[k + 1]
        z1.append(d[0] * z1[-1] + a[0] * q[k] + b[0] * q[k + 1])
        z2.append(d[1] * z2[-1] + a[1] * q[k] + b[1] * q[k + 1])
    x = np.hstack([y, np.column_stack([z1, z2])])
    tr = inv[1:]
    Ad = np.array([s.Ad[n:2 * n] for s in systems])[tr]
    Bd = np.array([s.Bd[n:2 * n] for s in systems])[tr]
    pred = np.einsum("tij,tj->ti", Ad, x[:-1]) + np.einsum("tij,tj->ti", Bd, u_app[:-1])
    rate_ref = ref[n:2 * n]
    r = (y[1:, n:2 * n] - pred) / rate_ref
    sens = [_stiffness_sensitivity(p, float(v), dt) for v in lv]
    dA = np.array([s[0] for s in sens])[tr]
    dB = np.array([s[1] for s in sens])[tr]
    s = (np.einsum("tij,tj->ti", dA, x[:-1]) + np.einsum("tij,tj->ti", dB, u_app[:-1])) / rate_ref
    r, s = r[skip:], s[skip:]
    ss = float(np.sum(s * s))
    ys, us = y[skip:], u_cmd[skip:]
    u_max = np.asarray(p.u_max, dtype=float)
    e = structural_energy(p, ys)
    half = len(e) // 2
    return WindowStats(
        residual=np.sqrt(np.mean(r**2, axis=0)),
        hf=np.sqrt(np.mean((np.diff(ys[:, :n], n=2, axis=0) / ref[:n]) ** 2, axis=0)),
        rms=np.sqrt(np.mean((ys[:, :n] / ref[:n]) ** 2, axis=0)),
        command_rate=float(np.sqrt(np.mean((np.diff(u_cmd[skip - 1:], axis=0) / u_max) ** 2))),
        stiffness=float(np.sum(r * s) / ss) if ss > 0 else 0.0,
        freq_hz=healthy_freq(p, y, dt),  # whole window: the Welch estimate needs every 1 s segment it can get
        saturation=float(np.mean(np.any(np.abs(us) >= 0.999 * u_max, axis=1))),
        risk=float(max(np.max(np.abs(ys[:, 0])) / p.h_limit, np.max(np.abs(ys[:, 1])) / np.radians(p.alpha_limit_deg))),
        energy_trend=float(np.log10(max(np.mean(e[half:]), _EPS) / max(np.mean(e[:half]), _EPS))),
        airspeed=float(levels[-1]), speed_change=float(levels[-1] - levels[0]),
        neural_share=float(np.mean(neural[skip:])))


def _log_ratio(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.log10(np.maximum(a, _EPS) / np.maximum(b, _EPS))


def features(w: WindowStats, base: WindowStats, flutter_speed: float) -> np.ndarray:
    """Feature vector (feature_names order): scale statistics relative to the healthy take-off baseline."""
    valid = w.freq_hz is not None and base.freq_hz is not None
    shift = (w.freq_hz - base.freq_hz) / base.freq_hz if w.freq_hz is not None and base.freq_hz is not None else 0.0
    return np.concatenate([
        _log_ratio(w.residual, base.residual), _log_ratio(w.hf, base.hf), _log_ratio(w.rms, base.rms),
        [float(_log_ratio(w.command_rate, base.command_rate)), w.stiffness - base.stiffness, shift, float(valid),
         w.saturation, w.risk, w.energy_trend, w.airspeed / flutter_speed - 1.0, w.speed_change / flutter_speed,
         w.neural_share]])


def _levels(speeds: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(speeds, dtype=float) / SPEED_STEP) * SPEED_STEP


_NEURAL_MODES = frozenset(m for m in FT_MODES if controller_of(m) == "neural")


def _neural(modes) -> np.ndarray:
    return np.fromiter((m in _NEURAL_MODES for m in modes), dtype=bool, count=len(modes))


def stats_between(p: WingParams, dt: float, ys: np.ndarray, us: np.ndarray, speeds: np.ndarray, modes, a: int,
                  k: int) -> WindowStats:
    """Statistics of samples a..k-1 of a flight (full-flight arrays as ft_simulate / FTResult hold them)."""
    u_before = us[a - 1] if a > 0 else np.zeros(us.shape[1])
    return window_stats(p, dt, ys[a:k], us[a:k], u_before, _levels(speeds[a:k]), _neural(modes[a:k]))


def baseline_stats(p: WingParams, dt: float, ys: np.ndarray, us: np.ndarray, speeds: np.ndarray, modes,
                   k: int, start_s: float = BASELINE_S[0]) -> WindowStats:
    """The healthy take-off baseline, from start_s to sample k (the simulator's baseline instant)."""
    return stats_between(p, dt, ys, us, speeds, modes, int(start_s / dt), k)


def window_features(p: WingParams, dt: float, flutter_speed: float, ys: np.ndarray, us: np.ndarray,
                    speeds: np.ndarray, modes, k: int, base: WindowStats) -> np.ndarray:
    """Features of the WINDOW_S before sample k (exclusive): what the flight computer has at sample k."""
    a = max(0, k - int(round(WINDOW_S / dt)))
    return features(stats_between(p, dt, ys, us, speeds, modes, a, k), base, flutter_speed)


def confirmed(flags: np.ndarray, n: int = CONFIRM) -> np.ndarray:
    """(T, K) bool -> True where the flag has held for the last n diagnoses (False before n are available)."""
    flags = np.asarray(flags, dtype=bool)
    out = flags.copy()
    for j in range(1, n):
        out[j:] &= flags[:-j]
        out[:j] = False
    return out


Layers = list[tuple[np.ndarray, np.ndarray]]


def _mlp(layers: Layers, x: np.ndarray) -> np.ndarray:
    """tanh hidden layers, linear output (x: (B, d))."""
    for W, b in layers[:-1]:
        x = np.tanh(x @ W.T + b)
    W, b = layers[-1]
    return x @ W.T + b


class DiagnoserModel:
    """Weights, input normalisation and calibration of a trained diagnoser (pure NumPy)."""

    def __init__(self, encoder: Layers, classifier: Layers, decoder: Layers, mean: np.ndarray, std: np.ndarray,
                 centroids: np.ndarray, precision: np.ndarray, meta: dict):
        self.encoder, self.classifier, self.decoder = encoder, classifier, decoder
        self.mean, self.std = mean, std
        self.centroids, self.precision = centroids, precision  # known fault combinations in the embedding
        self.meta = meta

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.meta.get("classes", KNOWN))

    @property
    def prob_threshold(self) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.meta.get("prob_threshold", 0.5), dtype=float), (len(self.classes),))

    @property
    def ood_threshold(self) -> float:
        return float(self.meta.get("ood_threshold", np.inf))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X (B, d) raw features -> probabilities (B, K), reconstruction score (B,), Mahalanobis score (B,)."""
        xn = (np.atleast_2d(X) - self.mean) / self.std
        z = _mlp(self.encoder, xn)
        prob = 1.0 / (1.0 + np.exp(-_mlp(self.classifier, z)))
        rec = np.mean((_mlp(self.decoder, z) - xn) ** 2, axis=1)
        d = z[:, None, :] - self.centroids[None]
        maha = np.min(np.einsum("bci,ij,bcj->bc", d, self.precision, d), axis=1)
        return prob, rec, maha

    @classmethod
    def load(cls, path: str | Path) -> "DiagnoserModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No trained diagnoser at {path}. Train one with:  python -m flutter_ai "
                                    f"train-diagnoser --preset {path.stem.removeprefix('diagnoser_')}")
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != MODEL_FORMAT:
            raise FileNotFoundError(f"Diagnoser file {path} has format {meta.get('format')}, expected {MODEL_FORMAT}. "
                                    f"Retrain with:  python -m flutter_ai train-diagnoser --preset {meta.get('preset')}")

        def layers(name: str) -> Layers:
            return [(data[f"{name}_W{i}"], data[f"{name}_b{i}"]) for i in range(int(data[f"{name}_n"]))]

        return cls(layers("encoder"), layers("classifier"), layers("decoder"), data["mean"], data["std"],
                   data["centroids"], data["precision"], meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"mean": self.mean, "std": self.std, "centroids": self.centroids,
                                  "precision": self.precision, "meta": np.array(json.dumps(self.meta))}
        for name, layers in (("encoder", self.encoder), ("classifier", self.classifier), ("decoder", self.decoder)):
            arrays[f"{name}_n"] = np.array(len(layers))
            for i, (W, b) in enumerate(layers):
                arrays[f"{name}_W{i}"], arrays[f"{name}_b{i}"] = W, b
        np.savez(path, **arrays)


class FlightDiagnoser:
    """
    The diagnoser on board one flight: the take-off baseline, the model, and the last CONFIRM raw outputs (a
    detection is confirmed when it holds at CONFIRM consecutive diagnoses). Call it every DIAGNOSE_EVERY_S.
    """

    def __init__(self, model: DiagnoserModel, p: WingParams, flutter_speed: float, dt: float, base: WindowStats):
        if model.meta.get("params_fingerprint") != p.fingerprint():
            raise ValueError(f"The diagnoser was trained on different '{model.meta.get('preset')}' parameters than "
                             f"'{p.name}'. Retrain with:  python -m flutter_ai train-diagnoser --preset {p.name}")
        self.model, self.p, self.flutter_speed, self.dt, self.base = model, p, flutter_speed, dt, base
        self._flags: list[np.ndarray] = []

    def __call__(self, t: float, ys: np.ndarray, us: np.ndarray, speeds: np.ndarray, modes, k: int) -> dict:
        x = window_features(self.p, self.dt, self.flutter_speed, ys, us, speeds, modes, k, self.base)
        prob, rec, maha = self.model.predict(x)
        m = self.model
        self._flags = (self._flags + [np.append(prob[0] >= m.prob_threshold, rec[0] > m.ood_threshold)])[-CONFIRM:]
        conf = confirmed(np.array(self._flags))[-1] if len(self._flags) == CONFIRM else np.zeros(len(m.classes) + 1, bool)
        return {"time": float(t), "window_s": WINDOW_S,
                "probabilities": {c: float(v) for c, v in zip(m.classes, prob[0])},
                "prob_threshold": {c: float(v) for c, v in zip(m.classes, m.prob_threshold)},
                "confirmed": [c for c, f in zip(m.classes, conf[:-1]) if f],
                "ood_score": float(rec[0]), "ood_threshold": m.ood_threshold, "unknown": bool(conf[-1]),
                "ood_mahalanobis": float(maha[0])}
