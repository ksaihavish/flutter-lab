"""
Neural feedback policy trained through the differentiable physics model
(see flutter_ai.training). Inference is pure NumPy, so serving the app does not
need PyTorch - only training does.

Policy inputs  : normalised state estimate x_hat (q, q', z1, z2), the previous command
                 u_{k-1} / u_max (the policy knows what is "in flight" through the delay),
                 airspeed ratio U/U_f, aggressiveness a in [0, 1]
Policy outputs : u = u_max * tanh(g(x, u_prev) - g(0, 0))
                 (saturation built in; exactly zero command at the equilibrium)

One policy file per preset: models/neural_policy_<preset>.npz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..aero import JONES_EPS
from ..params import WingParams
from ..paths import MODELS_DIR
from .base import Controller
from .costs import reference_scales

U_RATIO_SCALE = 0.3
POLICY_FORMAT = 3  # bump when the network structure changes; old files are rejected


def policy_path(preset: str) -> Path:
    return MODELS_DIR / f"neural_policy_{preset}.npz"


def state_scales(p: WingParams) -> np.ndarray:
    """Normalisation for the full state (aero-lag states scale with b / eps_i)."""
    ref = reference_scales(p)
    z_ref = ref[1] * p.b / np.array(JONES_EPS)
    return np.concatenate([ref, z_ref])


def build_features(x_hat: np.ndarray, u_prev: np.ndarray, scales: np.ndarray, u_max: np.ndarray,
                   u_ratio: float, aggressiveness: float) -> np.ndarray:
    return np.concatenate([x_hat / scales, u_prev / u_max,
                           [(u_ratio - 1.0) / U_RATIO_SCALE, 2.0 * aggressiveness - 1.0]])


class NeuralPolicy:
    """Weights + metadata of a trained MLP policy."""

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]], scales: np.ndarray, u_max: np.ndarray, meta: dict):
        self.weights = weights
        self.scales = scales
        self.u_max = u_max
        self.meta = meta

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPolicy":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained neural policy at {path}. Train one with:  "
                f"python -m flutter_ai train --preset {path.stem.removeprefix('neural_policy_')}"
            )
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != POLICY_FORMAT:
            raise FileNotFoundError(
                f"Policy file {path} uses an old format ({meta.get('format')}); expected {POLICY_FORMAT}. "
                f"Retrain with:  python -m flutter_ai train --preset {meta.get('preset', '<preset>')}"
            )
        n_layers = int(data["n_layers"])
        weights = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)]
        return cls(weights, data["scales"], data["u_max"], meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"n_layers": np.array(len(self.weights)), "scales": self.scales, "u_max": self.u_max,
                  "meta": np.array(json.dumps(self.meta))}
        for i, (W, b) in enumerate(self.weights):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        np.savez(path, **arrays)

    def _pre(self, features: np.ndarray) -> np.ndarray:
        """Network output before the saturating tanh (hidden layers tanh, last layer linear)."""
        h = features
        for W, b in self.weights[:-1]:
            h = np.tanh(W @ h + b)
        W, b = self.weights[-1]
        return W @ h + b

    def __call__(self, features: np.ndarray, features_at_zero: np.ndarray) -> np.ndarray:
        return self.u_max * np.tanh(self._pre(features) - self._pre(features_at_zero))

    def features(self, x_hat, u_prev, u_ratio, aggressiveness) -> np.ndarray:
        if not self.meta.get("config", {}).get("use_airspeed", True):
            u_ratio = 1.0  # ablation policy trained speed-blind: its airspeed feature is always zero
        return build_features(x_hat, u_prev, self.scales, self.u_max, u_ratio, aggressiveness)

    def linearize(self, u_ratio: float, aggressiveness: float, eps: float = 1e-7) -> np.ndarray:
        """Gain K of the linearisation at equilibrium: u ~ -K [x, u_prev] (for delay-margin analysis)."""
        nx, m = len(self.scales), len(self.u_max)
        f0 = self.features(np.zeros(nx), np.zeros(m), u_ratio, aggressiveness)
        K = np.zeros((m, nx + m))
        for i in range(nx + m):
            z = np.zeros(nx + m)
            z[i] = eps
            K[:, i] = -self(self.features(z[:nx], z[nx:], u_ratio, aggressiveness), f0) / eps
        return K


class NeuralController(Controller):
    name = "neural"
    label = "Neural (physics-trained)"

    def __init__(self, policy: NeuralPolicy | None = None, model_path: str | Path | None = None):
        super().__init__()
        self._policy = policy
        self._model_path = model_path

    def reset(self, params: WingParams, U: float, dt: float, delay: int = 1) -> None:
        super().reset(params, U, dt, delay)
        if self._policy is None or self._policy.meta.get("preset") != params.name:
            self._policy = NeuralPolicy.load(self._model_path or policy_path(params.name))
        meta = self._policy.meta
        self.flutter_speed = float(meta["flutter_speed"])
        if meta.get("params_fingerprint") != params.fingerprint():
            self.warnings.append(
                f"The neural policy was trained on different '{meta.get('preset')}' parameters; results are "
                "out of distribution. Retrain for this configuration."
            )
        lo, hi = meta.get("velocity_ratio_range", [0.0, np.inf])
        ratio = U / self.flutter_speed
        if not lo <= ratio <= hi:
            self.warnings.append(f"Airspeed ratio {ratio:.2f} is outside the policy's training range [{lo:.2f}, {hi:.2f}].")
        if abs(dt - float(meta.get("dt", dt))) > 1e-12:
            self.warnings.append(f"Control period {dt} s differs from training ({meta.get('dt')} s).")
        dlo, dhi = meta.get("delay_range", [delay, delay])
        if not dlo <= delay <= dhi:
            self.warnings.append(f"Input delay of {delay} steps is outside the training range [{dlo}, {dhi}].")

    def compute(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        pol = self._policy
        if pol is None:
            raise RuntimeError("NeuralController used before reset()")
        ratio = self.U / self.flutter_speed
        f = pol.features(x_hat, self.history[0], ratio, self.aggressiveness)
        f0 = pol.features(np.zeros_like(x_hat), np.zeros_like(self.u_max), ratio, self.aggressiveness)
        return pol(f, f0)
