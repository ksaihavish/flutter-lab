"""
Tuned design knobs of the classical baselines (see flutter_ai.tuning). The objective J is shared by every
controller (costs.py); these knobs only change how LQR / LQG are *designed* for the real actuator:

    rate_multiplier     extra weight on command changes in the design cost (smoother, slower commands)
    measurement_noise   (LQG) assumed sensor noise, as a fraction of the reference scales (filter bandwidth)

Tuned values live in models/classical_tuning_<preset>.json and apply only to the parameters they were tuned for.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..params import WingParams
from ..paths import MODELS_DIR

DEFAULTS = {"lqr": {"rate_multiplier": 1.0}, "lqg": {"rate_multiplier": 1.0, "measurement_noise": 1e-3}}


def tuning_path(preset: str) -> Path:
    return MODELS_DIR / f"classical_tuning_{preset}.json"


@lru_cache(maxsize=32)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def design_settings(p: WingParams, kind: str) -> tuple[dict, str | None]:
    """(settings, warning): tuned knobs for this wing, or the defaults (with a warning if the file is stale)."""
    path = tuning_path(p.name)
    if not path.exists():
        return dict(DEFAULTS[kind]), None
    data = _load(str(path), path.stat().st_mtime)
    if data.get("params_fingerprint") != p.fingerprint():
        return dict(DEFAULTS[kind]), (f"{kind.upper()} design knobs in {path.name} were tuned for other '{p.name}' "
                                      "parameters; using defaults. Re-run: python -m flutter_ai tune")
    return {**DEFAULTS[kind], **data.get(kind, {})}, None
