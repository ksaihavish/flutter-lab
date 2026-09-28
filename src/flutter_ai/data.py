"""
Experimental / external dataset intake (format spec: data/README.md).

    python -m flutter_ai check-data data/<dataset>      validate files, build the wing model from the
                                                         rig parameters, compare predicted vs measured
                                                         flutter speed

A dataset is a folder:
    metadata.json          rig parameters (SI), source, licence, sensors, measured flutter point
    runs/<run_id>.csv      time histories: t, U, h, alpha[, beta, beta_cmd, w_gust, ...]
    runs/<run_id>.json     optional per-run notes (controller on/off, airspeed, ...)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .params import WingParams

REQUIRED_RIG = ("b", "a", "mass", "k_h", "k_alpha")
REQUIRED_COLUMNS = ("t", "U", "h", "alpha")
KNOWN_COLUMNS = REQUIRED_COLUMNS + ("h_dot", "alpha_dot", "beta", "beta_dot", "beta_cmd", "F_h", "M_alpha", "w_gust")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = [f"[ERROR] {e}" for e in self.errors] + [f"[WARN]  {w}" for w in self.warnings] + \
                [f"[INFO]  {i}" for i in self.info]
        return "\n".join(lines) + f"\n=> {'OK' if self.ok else 'FAILED'}"


def read_run_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(line for line in f if not line.startswith("#")))
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(v) if v.strip() != "" else np.nan for v in r] for r in rows[1:] if r], dtype=float)
    return {name: data[:, j] for j, name in enumerate(header)}


def params_from_metadata(meta: dict, name: str = "dataset") -> WingParams:
    """Build WingParams from dimensional rig parameters (SI, per unit span)."""
    rig = meta["rig"]
    m, b = float(rig["mass"]), float(rig["b"])
    S_a = float(rig.get("S_alpha", rig.get("x_alpha", 0.0) * m * b))
    I_a = float(rig.get("I_alpha", rig.get("r_alpha_sq", 0.0) * m * b * b))
    if I_a <= 0:
        raise ValueError("rig needs I_alpha (kg m^2/m) or r_alpha_sq")
    k_h, k_a = float(rig["k_h"]), float(rig["k_alpha"])
    zeta = lambda c, k, mm: float(c) / (2.0 * math.sqrt(k * mm)) if c is not None else 0.0  # noqa: E731
    kw = dict(
        name=name, description=meta.get("description", ""), b=b, a=float(rig["a"]), mass=m,
        x_alpha=S_a / (m * b), r_alpha_sq=I_a / (m * b * b),
        f_h=math.sqrt(k_h / m) / (2 * math.pi), f_alpha=math.sqrt(k_a / I_a) / (2 * math.pi),
        zeta_h=zeta(rig.get("c_h"), k_h, m), zeta_alpha=zeta(rig.get("c_alpha"), k_a, I_a),
        rho=float(rig.get("rho", 1.225)),
    )
    if rig.get("flap_hinge") is not None:
        S_b = float(rig.get("S_beta", rig.get("x_beta", 0.0) * m * b))
        I_b = float(rig.get("I_beta", rig.get("r_beta_sq", 0.0) * m * b * b))
        k_b = float(rig["k_beta"])
        kw.update(flap_hinge=float(rig["flap_hinge"]), x_beta=S_b / (m * b), r_beta_sq=I_b / (m * b * b),
                  f_beta=math.sqrt(k_b / I_b) / (2 * math.pi), zeta_beta=zeta(rig.get("c_beta"), k_b, I_b),
                  actuation="flap_servo", u_max=(math.radians(float(rig.get("beta_cmd_max_deg", 15.0))),))
    return WingParams(**kw)


def check_dataset(folder: str | Path) -> Report:
    folder = Path(folder)
    rep = Report()
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        rep.errors.append(f"missing {meta_path}")
        return rep
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        rep.errors.append(f"metadata.json is not valid JSON: {exc}")
        return rep

    for key in ("name", "source", "licence"):
        if not meta.get(key):
            rep.warnings.append(f"metadata.{key} is empty - needed to cite/redistribute the data")
    rig = meta.get("rig", {})
    missing = [k for k in REQUIRED_RIG if k not in rig]
    if "I_alpha" not in rig and "r_alpha_sq" not in rig:
        missing.append("I_alpha (or r_alpha_sq)")
    if missing:
        rep.errors.append(f"rig parameters missing: {missing} (SI units, per unit span - see data/README.md)")

    params = None
    if not missing:
        try:
            params = params_from_metadata(meta, name=meta.get("name", folder.name))
            rep.info.append(f"model: {params.n_dof}-DOF, mu = {params.mass_ratio:.1f}, f_h = {params.f_h:.2f} Hz, "
                            f"f_alpha = {params.f_alpha:.2f} Hz" + (f", f_beta = {params.f_beta:.2f} Hz" if params.n_dof == 3 else ""))
        except (ValueError, KeyError, ZeroDivisionError) as exc:
            rep.errors.append(f"could not build a wing model from the rig parameters: {exc}")

    if params is not None:
        from .flutter import flutter_analysis

        fa = flutter_analysis(params)
        measured = (meta.get("measured") or {}).get("flutter_speed_mps")
        if fa.flutter_speed is None:
            rep.warnings.append("model predicts no flutter in the analysed speed range")
        else:
            msg = f"predicted flutter speed {fa.flutter_speed:.2f} m/s at {fa.flutter_frequency_hz:.2f} Hz"
            if measured:
                err = 100 * (fa.flutter_speed - measured) / measured
                msg += f"; measured {measured:.2f} m/s -> model error {err:+.1f} %"
                (rep.info if abs(err) <= 10 else rep.warnings).append(msg)
            else:
                rep.info.append(msg + " (no measured flutter speed in metadata.measured)")

    runs = sorted((folder / "runs").glob("*.csv")) if (folder / "runs").exists() else []
    if not runs:
        rep.warnings.append("no time histories in runs/*.csv (optional, but needed for identification and time-domain validation)")
    for run in runs:
        try:
            cols = read_run_csv(run)
        except (ValueError, IndexError, OSError) as exc:
            rep.errors.append(f"{run.name}: unreadable ({exc})")
            continue
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing_cols:
            rep.errors.append(f"{run.name}: missing columns {missing_cols}")
            continue
        unknown = [c for c in cols if c not in KNOWN_COLUMNS]
        if unknown:
            rep.info.append(f"{run.name}: extra columns kept as-is: {unknown}")
        t = cols["t"]
        dt = np.diff(t)
        if np.any(dt <= 0):
            rep.errors.append(f"{run.name}: time is not strictly increasing")
            continue
        fs = 1.0 / np.median(dt)
        if np.max(np.abs(dt - np.median(dt))) > 0.01 * np.median(dt):
            rep.warnings.append(f"{run.name}: non-uniform sampling (will be resampled)")
        nan_cols = [c for c, v in cols.items() if np.isnan(v).any()]
        if nan_cols:
            rep.warnings.append(f"{run.name}: missing values in {nan_cols}")
        if np.nanmax(np.abs(cols["alpha"])) > 1.0:
            rep.warnings.append(f"{run.name}: |alpha| > 1 rad - is alpha in degrees? (must be radians)")
        if np.nanmax(np.abs(cols["h"])) > 1.0:
            rep.warnings.append(f"{run.name}: |h| > 1 m - is h in mm? (must be metres)")
        if fs < 200:
            rep.warnings.append(f"{run.name}: sampling rate {fs:.0f} Hz is low; >= 500 Hz recommended for 10-20 Hz flutter")
        rep.info.append(f"{run.name}: {len(t)} samples, {t[-1] - t[0]:.2f} s at {fs:.0f} Hz, "
                        f"U = {np.nanmean(cols['U']):.1f} m/s")
    return rep


def export_run_csv(result, path: str | Path, note: str = "") -> Path:
    """Write a simulation result in the dataset run format (useful as a template / for synthetic data)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = result.params
    n = p.n_dof
    cols = {"t": result.t, "U": np.full(len(result.t), result.velocity), "h": result.x[:, 0], "alpha": result.x[:, 1],
            "h_dot": result.x[:, n], "alpha_dot": result.x[:, n + 1]}
    if n == 3:
        cols["beta"], cols["beta_dot"] = result.x[:, 2], result.x[:, 5]
    for j, name in enumerate(p.input_names):
        cols[name] = result.u[:, j]
    cols["w_gust"] = result.w_gust
    with open(path, "w", newline="", encoding="utf-8") as f:
        if note:
            f.write(f"# {note}\n")
        w = csv.writer(f)
        w.writerow(cols.keys())
        for row in zip(*cols.values()):
            w.writerow([f"{v:.8g}" for v in row])
    return path
