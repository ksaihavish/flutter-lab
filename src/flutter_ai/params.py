"""
Wing-section parameters for the typical section: 2-DOF (plunge, pitch) or 3-DOF
(plunge, pitch, trailing-edge flap).

Parameters are given in the form used in the aeroelasticity literature
(semichord-normalised geometry, uncoupled frequencies, damping ratios); stiffness,
inertia and damping coefficients are derived. All values are SI, per unit span.

Sign conventions (Theodorsen / Hodges & Pierce):
    h      plunge, positive DOWN                          [m]
    alpha  pitch, positive NOSE-UP                        [rad]
    beta   flap deflection, positive TRAILING-EDGE DOWN   [rad]
    a      elastic-axis position, semichords aft of mid-chord
    c      flap-hinge position, semichords aft of mid-chord
    x_alpha  wing CG offset aft of elastic axis, semichords
    x_beta   flap CG offset aft of hinge, semichords

Actuation
    "force_moment" : inputs [F_h, M_alpha] (generalised force/moment; 2-DOF idealisation)
    "flap_servo"   : input [beta_cmd]; a servo with stiffness k_beta drives the flap
                     towards the commanded angle (realistic control-surface actuation)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class WingParams:
    name: str
    description: str
    b: float  # semichord [m]
    a: float  # elastic axis, semichords aft of mid-chord
    mass: float  # total mass per unit span (incl. flap) [kg/m]
    x_alpha: float  # static unbalance about elastic axis, semichords
    r_alpha_sq: float  # (radius of gyration about EA / b)^2
    f_h: float  # uncoupled plunge frequency [Hz]
    f_alpha: float  # uncoupled pitch frequency [Hz]
    zeta_h: float = 0.01
    zeta_alpha: float = 0.01
    rho: float = 1.225  # air density [kg/m^3]
    # ---- optional trailing-edge flap (3rd DOF) ------------------------------
    flap_hinge: float | None = None  # c; None = no flap DOF (2-DOF section)
    x_beta: float = 0.0  # flap static unbalance about hinge, semichords
    r_beta_sq: float = 0.0  # (flap radius of gyration about hinge / b)^2
    f_beta: float = 0.0  # uncoupled flap (servo) frequency [Hz]
    zeta_beta: float = 0.0
    # ---- actuation and limits ---------------------------------------------
    actuation: str = "force_moment"  # "force_moment" | "flap_servo"
    u_max: tuple[float, ...] = field(default=(60.0, 6.0))  # per input, SI
    h_limit: float = 0.10  # structural limit [m]
    alpha_limit_deg: float = 20.0  # structural limit [deg]

    def __post_init__(self):
        if self.actuation not in ("force_moment", "flap_servo"):
            raise ValueError(f"unknown actuation '{self.actuation}'")
        if self.actuation == "flap_servo" and self.flap_hinge is None:
            raise ValueError("flap_servo actuation needs a flap (set flap_hinge)")
        if len(self.u_max) != len(self.input_names):
            raise ValueError(f"u_max needs {len(self.input_names)} values for actuation '{self.actuation}'")

    # ---- structure ------------------------------------------------------------
    @property
    def n_dof(self) -> int:
        return 2 if self.flap_hinge is None else 3

    @property
    def dof_names(self) -> tuple[str, ...]:
        return ("h", "alpha") if self.n_dof == 2 else ("h", "alpha", "beta")

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("F_h", "M_alpha") if self.actuation == "force_moment" else ("beta_cmd",)

    @property
    def input_units(self) -> tuple[str, ...]:
        return ("N/m", "N·m/m") if self.actuation == "force_moment" else ("rad",)

    @property
    def primary_input(self) -> int:
        """Index of the input shown in single-channel plots."""
        return 1 if self.actuation == "force_moment" else 0

    # ---- derived quantities -------------------------------------------------
    @property
    def chord(self) -> float:
        return 2.0 * self.b

    @property
    def omega_h(self) -> float:
        return 2.0 * math.pi * self.f_h

    @property
    def omega_alpha(self) -> float:
        return 2.0 * math.pi * self.f_alpha

    @property
    def omega_beta(self) -> float:
        return 2.0 * math.pi * self.f_beta

    @property
    def S_alpha(self) -> float:
        return self.mass * self.x_alpha * self.b

    @property
    def I_alpha(self) -> float:
        return self.mass * self.r_alpha_sq * self.b**2

    @property
    def S_beta(self) -> float:
        return self.mass * self.x_beta * self.b

    @property
    def I_beta(self) -> float:
        return self.mass * self.r_beta_sq * self.b**2

    @property
    def k_h(self) -> float:
        return self.mass * self.omega_h**2

    @property
    def k_alpha(self) -> float:
        return self.I_alpha * self.omega_alpha**2

    @property
    def k_beta(self) -> float:
        return self.I_beta * self.omega_beta**2

    @property
    def c_h(self) -> float:
        return 2.0 * self.zeta_h * self.mass * self.omega_h

    @property
    def c_alpha(self) -> float:
        return 2.0 * self.zeta_alpha * self.I_alpha * self.omega_alpha

    @property
    def c_beta(self) -> float:
        return 2.0 * self.zeta_beta * self.I_beta * self.omega_beta

    @property
    def mass_ratio(self) -> float:
        """mu = m / (pi rho b^2)"""
        return self.mass / (math.pi * self.rho * self.b**2)

    def with_overrides(self, **overrides) -> "WingParams":
        clean = {k: v for k, v in overrides.items() if v is not None}
        if "u_max" in clean:
            clean["u_max"] = tuple(clean["u_max"])
        return replace(self, **clean)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["u_max"] = list(self.u_max)
        d["n_dof"] = self.n_dof
        d["dof_names"] = list(self.dof_names)
        d["inputs"] = [{"name": n, "unit": u, "max": m}
                       for n, u, m in zip(self.input_names, self.input_units, self.u_max)]
        d["primary_input"] = self.primary_input
        d["derived"] = {
            "chord": self.chord,
            "S_alpha": self.S_alpha,
            "I_alpha": self.I_alpha,
            "k_h": self.k_h,
            "k_alpha": self.k_alpha,
            "c_h": self.c_h,
            "c_alpha": self.c_alpha,
            "mass_ratio": self.mass_ratio,
        }
        if self.n_dof == 3:
            d["derived"].update({"S_beta": self.S_beta, "I_beta": self.I_beta, "k_beta": self.k_beta,
                                 "c_beta": self.c_beta})
        return d

    def fingerprint(self) -> str:
        """Stable string identifying the physics (used to match trained policies to parameters)."""
        d = asdict(self)
        d.pop("description", None)
        return json.dumps(d, sort_keys=True)


_UAV = dict(b=0.15, a=-0.3, mass=5.0, x_alpha=0.2, r_alpha_sq=0.25, f_h=8.0, f_alpha=16.0,
            zeta_h=0.01, zeta_alpha=0.01)

PRESETS: dict[str, WingParams] = {
    "uav_flap": WingParams(
        name="uav_flap",
        description="3-DOF small-UAV wing section (0.3 m chord) with a 20%-chord trailing-edge flap "
        "driven by a 30 Hz servo, +/-15 deg command. Default research case.",
        **_UAV,
        flap_hinge=0.6,
        x_beta=0.01,
        r_beta_sq=0.003,
        f_beta=30.0,
        zeta_beta=0.05,
        actuation="flap_servo",
        u_max=(math.radians(15.0),),
    ),
    "uav_section": WingParams(
        name="uav_section",
        description="2-DOF version of the same section with idealised force/moment actuators "
        "(the v0.2 research case).",
        **_UAV,
    ),
    "hodges_pierce": WingParams(
        name="hodges_pierce",
        description=(
            "Nondimensional textbook case (Hodges & Pierce, Ch. 5): a=-1/5, x_alpha=1/10, "
            "r^2=6/25, mu=20, sigma=2/5, b=1 m, omega_alpha=1 rad/s. Validation only: "
            "report flutter speed as U/(b*omega_alpha)."
        ),
        b=1.0,
        a=-0.2,
        mass=20.0 * math.pi * 1.225,  # mu = 20
        x_alpha=0.1,
        r_alpha_sq=0.24,
        f_h=0.4 / (2.0 * math.pi),  # sigma = omega_h/omega_alpha = 0.4
        f_alpha=1.0 / (2.0 * math.pi),
        zeta_h=0.0,
        zeta_alpha=0.0,
        u_max=(1e3, 1e3),
        h_limit=1.0,
        alpha_limit_deg=20.0,
    ),
}

DEFAULT_PRESET = "uav_flap"


def get_preset(name: str | None = None, **overrides) -> WingParams:
    key = name or DEFAULT_PRESET
    if key not in PRESETS:
        raise KeyError(f"Unknown preset '{key}'. Available: {sorted(PRESETS)}")
    params = PRESETS[key]
    return params.with_overrides(**overrides) if overrides else params
