"""Vertical gust profiles w_g(t) [m/s, positive up]."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

GUST_KINDS = ("none", "one_minus_cosine", "sine", "turbulence")


@dataclass(frozen=True)
class GustSpec:
    kind: str = "one_minus_cosine"
    amplitude: float = 3.0  # peak vertical velocity [m/s] (turbulence: RMS)
    start: float = 2.0  # [s]
    duration: float = 0.25  # [s]
    frequency: float = 5.0  # [Hz], sine only
    seed: int = 0  # turbulence only
    correlation_time: float = 0.05  # [s], turbulence only

    def __post_init__(self):
        if self.kind not in GUST_KINDS:
            raise ValueError(f"Unknown gust kind '{self.kind}'. Use one of {GUST_KINDS}")

    def to_dict(self) -> dict:
        return asdict(self)

    def sample(self, t: np.ndarray) -> np.ndarray:
        """Gust velocity at each time in t."""
        t = np.asarray(t, dtype=float)
        w = np.zeros_like(t)
        if self.kind == "none" or self.amplitude == 0.0:
            return w
        active = (t >= self.start) & (t <= self.start + self.duration)
        tau = t - self.start
        if self.kind == "one_minus_cosine":
            w[active] = 0.5 * self.amplitude * (1.0 - np.cos(2.0 * np.pi * tau[active] / self.duration))
        elif self.kind == "sine":
            w[active] = self.amplitude * np.sin(2.0 * np.pi * self.frequency * tau[active])
        elif self.kind == "turbulence":
            # First-order (Dryden-like) filtered white noise, active over the window.
            rng = np.random.default_rng(self.seed)
            dt = float(t[1] - t[0]) if len(t) > 1 else 1e-3
            a = np.exp(-dt / self.correlation_time)
            noise = rng.standard_normal(len(t)) * self.amplitude * np.sqrt(1.0 - a**2)
            state = 0.0
            for i in range(len(t)):
                state = a * state + noise[i]
                w[i] = state if active[i] else 0.0
        return w
