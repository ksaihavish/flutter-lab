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
    # ---- modified strip theory (Yates 1966); defaults = 2D thin-airfoil Theodorsen ----------
    cl_alpha: float = 2.0 * math.pi  # section lift-curve slope [1/rad] (e.g. measured, incl. 3D/compressibility)
    x_ac: float = -0.5  # aerodynamic centre, semichords aft of mid-chord (quarter chord = -0.5)
    # ---- optional trailing-edge flap (3rd DOF) ------------------------------
    flap_hinge: float | None = None  # c; None = no flap DOF (2-DOF section)
    x_beta: float = 0.0  # flap static unbalance about hinge, semichords
    r_beta_sq: float = 0.0  # (flap radius of gyration about hinge / b)^2
    f_beta: float = 0.0  # uncoupled flap (servo) frequency [Hz]
    zeta_beta: float = 0.0
    flap_span_fraction: float = 1.0  # fraction of the span the flap covers (1 = full span)
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
        for key, default in _OPTIONAL_DEFAULTS.items():  # omitted at default: keeps older fingerprints valid
            if d.get(key) == default:
                d.pop(key)
        return json.dumps(d, sort_keys=True)


_OPTIONAL_DEFAULTS = {"cl_alpha": 2.0 * math.pi, "x_ac": -0.5, "flap_span_fraction": 1.0}

# Unit conversions (exact)
SLUG, FT, LBF, INCH = 14.59390294, 0.3048, 4.4482216153, 0.0254
PSF = LBF / FT**2  # Pa per lb/ft^2


def _bact_preset() -> WingParams:
    """
    NASA Benchmark Active Controls Technology (BACT) wind-tunnel model on the Pitch And Plunge
    Apparatus: rigid NACA 0012 wing, 16 in chord x 32 in span, 25 %-chord / 30 %-span trailing-edge
    control surface. Sources: Waszak, NASA/TP-1998-206270 (Tables 1-3); Mukhopadhyay (1999).

    The wing is rigid on springs, so total properties are divided by the span to get the per-unit-span
    values the strip model uses. Aerodynamics: modified strip theory with the MEASURED lift-curve slope
    and moment slope at Mach 0.77 (Table 2). Density: the test medium (R-12 heavy gas) at the measured
    flutter point, q = 148 psf, assuming U = 400 ft/s (the report's reference speed "close to test
    conditions"). Estimated (not in the report): flap inertia about its hinge; structural limits.
    """
    span, chord = 32 * INCH, 16 * INCH
    b = chord / 2
    m = 6.08 * SLUG / span  # Table 1: total mass 6.08 slug
    I_theta = 2.80 * SLUG * FT**2 / span  # Table 1: pitch inertia about the shear centre (mid-chord)
    S_theta = 0.0142 * SLUG * FT / span  # Table 1: pitch-plunge inertial coupling
    S_delta = 0.00288 * SLUG * FT / span  # Table 1: plunge-TE coupling = flap mass x arm
    flap_chord = 0.25 * chord
    m_flap = 0.00288 * SLUG * FT / (flap_chord / 2)  # estimate: flap CG at half its chord
    I_delta = m_flap * flap_chord**2 / 3 / span  # estimate: thin plate about its hinge
    cl_alpha, cm_alpha = 4.584, 1.490  # Table 2 (Mach 0.77); moments about mid-chord
    rho = 2 * 148 * PSF / (400 * FT) ** 2
    return WingParams(
        name="bact",
        description="NASA BACT wind-tunnel model (pitch-and-plunge apparatus, 25% chord TE flap over 30% span, "
        "26 Hz hydraulic actuator, +/-12 deg). Parameters from NASA/TP-1998-206270; measured flutter "
        "~148 psf at ~4 Hz (Mach 0.77, R-12 heavy gas). Modified strip theory with measured lift/moment slopes.",
        b=b,
        a=0.0,  # shear centre and CG at mid-chord
        mass=m,
        x_alpha=S_theta / (m * b),
        r_alpha_sq=I_theta / (m * b * b),
        f_h=21.01 / (2 * math.pi),  # Table 1: in-vacuo plunge 21.01 rad/s
        f_alpha=32.72 / (2 * math.pi),  # Table 1: in-vacuo pitch 32.72 rad/s
        zeta_h=0.0014,
        zeta_alpha=0.0010,
        rho=rho,
        cl_alpha=cl_alpha,
        x_ac=0.0 - 2 * cm_alpha / cl_alpha,  # from CM_alpha = CL_alpha (a - x_ac) / 2  ->  AC at 17.5 % chord
        flap_hinge=0.5,  # 25 % chord flap -> hinge at 75 % chord
        x_beta=S_delta / (m * b),
        r_beta_sq=I_delta / (m * b * b),
        f_beta=165.3 / (2 * math.pi),  # Table 3: TE actuator 165.3 rad/s ...
        zeta_beta=0.56,  # ... damping 0.56
        flap_span_fraction=0.30,
        actuation="flap_servo",
        u_max=(math.radians(12.0),),  # TE position limit ~12 deg
        h_limit=0.05,  # assumed PAPA travel
        alpha_limit_deg=10.0,  # assumed
    )

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
    "bact": _bact_preset(),
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
