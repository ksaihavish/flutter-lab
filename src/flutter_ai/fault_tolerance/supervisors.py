"""
Non-LLM supervisors for the fault-tolerant study: hand-written rules (cover faults F1-F4 only), random choice, and an
oracle that knows the fault. Rules and the LLM see the same on-board features (diagnostics.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .diagnostics import FTFeatures
from .faults import Fault
from .simulate import FT_MODES, MODE_ORDER, FTDecision, ft_simulate


class FTRuleSupervisor:
    """
    Hand-written symptom -> mode table for the in-distribution faults (servo, structure, sensor noise, turbulence),
    written from the same engineering principles as the LLM's guidance (not tuned on benchmark outcomes).
    It keeps the current mode when no rule fires, so it never de-escalates on its own.
    """

    name = "rules"

    def decide(self, f: FTFeatures) -> FTDecision:
        res = f.residual_ratio
        shift = (None if f.pitch_freq_hz is None or f.nominal_freq_hz is None
                 else (f.pitch_freq_hz - f.nominal_freq_hz) / f.nominal_freq_hz)
        rank = {m: i for i, m in enumerate(MODE_ORDER)}
        if f.risk > 0.6:
            mode, why = "safe", "Response above 60 % of the structural limit."
        elif shift is not None and shift < -0.20:
            mode, why = "safe", f"Pitch frequency {100 * shift:+.0f} %: structural damage lowers the flutter speed."
        elif res.get("beta", 1.0) > 5.0:
            mode, why = "cautious", "The flap does not follow its commands (servo degradation): slow down."
        elif f.noise_ratio > 3.0:
            mode, why = "cautious", f"Measurement noise x{f.noise_ratio:.1f}: control quality degraded, slow down."
        elif f.saturation_fraction > 0.3 or (f.energy_growth > 1.5 and f.risk > 0.3):
            mode, why = "cautious", "Actuator saturating or response growing: slow down."
        else:
            return FTDecision(time=f.time, mode=f.mode, reasoning="No rule fired: keep the current mode.",
                              source=self.name)
        if rank[mode] < rank.get(f.mode, 0):  # never de-escalate automatically
            mode, why = f.mode, f"{why} (keeping the more cautious current mode)"
        return FTDecision(time=f.time, mode=mode, reasoning=why, source=self.name)


class FTPhysicsRuleSupervisor(FTRuleSupervisor):
    """
    The same symptom rules decide *whether* to act; the on-board physics estimate decides *which modes are safe*:
    a mode predicted unstable on the identified wing is never chosen, and if the current mode is predicted unstable
    the least disruptive predicted-stable mode is taken (hold speed first, then the smallest slow-down).
    Written before any test on the benchmark seeds; not tuned on outcomes.
    """

    name = "rules+physics"

    def decide(self, f: FTFeatures) -> FTDecision:
        d = super().decide(f)
        d.source = self.name
        rad = (f.margin or {}).get("mode_spectral_radius")
        if not rad:
            return d
        stable = [m for m in MODE_ORDER if rad.get(m, 0.0) < 1.0]
        if d.mode in stable:
            return d
        if not stable:
            return FTDecision(time=f.time, mode="safe", source=self.name,
                              reasoning="Physics estimate: no mode predicted stable; take the largest margin.")
        return FTDecision(time=f.time, mode=stable[0], source=self.name,
                          reasoning=f"{d.reasoning} Physics estimate predicts '{d.mode}' unstable on the identified "
                                    f"wing; least disruptive predicted-stable mode: '{stable[0]}'.")


class Delayed:
    """Wraps a supervisor so that its answers take `latency_s` to arrive (time-scale study)."""

    def __init__(self, inner, latency_s: float):
        self.inner, self.latency_s = inner, float(latency_s)
        self.name = f"{inner.name}@{latency_s:g}s"

    def decide(self, f: FTFeatures) -> FTDecision:
        d = self.inner.decide(f)
        d.latency_s = self.latency_s
        return d


class FTRandomSupervisor:
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng([seed, 1])  # independent of the flight's own stream (default_rng(seed))

    def decide(self, f: FTFeatures) -> FTDecision:
        mode = MODE_ORDER[int(self._rng.integers(len(MODE_ORDER)))]
        return FTDecision(time=f.time, mode=mode, reasoning="Random mode (baseline).", source=self.name)


def oracle_mode(p, flutter_speed: float, flutter_freq_hz: float, fault: Fault, seed: int, react_s: float = 1.0,
                **kwargs) -> tuple[str, dict[str, dict]]:
    """
    The least disruptive mode that survives, chosen with hindsight: every mode is tried from onset + react_s.
    Order of preference: survive; then hold cruise speed (nominal/robust, lower J first); then the smallest
    slow-down (cautious before safe).
    """
    runs = {m: ft_simulate(p, flutter_speed, flutter_freq_hz, fault, None, seed=seed,
                           forced_mode=(fault.onset + react_s, m), **kwargs).metrics for m in MODE_ORDER}
    alive = [m for m in MODE_ORDER if not runs[m]["failed"]]
    hold = sorted([m for m in ("nominal", "robust") if m in alive], key=lambda m: runs[m]["cost_J"])
    if hold:
        return hold[0], runs
    slow = [m for m in ("cautious", "safe") if m in alive]  # least slow-down first
    return (slow[0] if slow else "none survives"), runs


class FTPhysicsRuleSupervisorV2(FTRuleSupervisor):
    """
    Symptom rules + the v2 physics estimate (margin.estimate_v2: robust stability, control authority, measured loop
    delay, turbulence and sensor noise). As in FTPhysicsRuleSupervisor, the symptom rules decide *whether* to act and
    propose a mode; the physics decides which modes may be chosen:
      1. admissible = stable over the mode's transit on every wing of the identification's confidence region AND
         enough control authority (predicted 3-sigma flap command and rate below the limits) where the identified wing
         needs control at the mode's airspeed (margin.MarginEstimateV2 defines both);
      2. the rules' proposal is kept if admissible; otherwise the least disruptive admissible mode (MODE_ORDER: hold
         speed first, then the smallest slow-down). This also acts when no rule fires but the current mode has become
         inadmissible (e.g. a turbulence rise that will saturate the flap);
      3. if no mode is robustly admissible, the same choice among the modes admissible on the point estimate, unless
         `strict`. Reason: the confidence region is conservative (after structural damage it contains wings on which
         no mode is stable), and 'safe' is then not certified either (its slow-down crosses the band where the damaged
         wing needs control);
      4. if none is admissible even on the point estimate, 'safe' (the largest open-loop margin at the end).
    Without a v2 estimate in the features it is exactly the symptom rules. Written from the estimator's definitions,
    developed on seeds >= 1000 only, never tuned on the reserved seeds.
    """

    name = "rules+physics_v2"

    def __init__(self, strict: bool = False):
        self.strict = strict
        if strict:
            self.name = "rules+physics_v2_strict"

    def decide(self, f: FTFeatures) -> FTDecision:
        d = super().decide(f)
        d.source = self.name
        modes = (f.margin or {}).get("modes")
        if not modes:
            return d
        tiers = [("robustly admissible", "admissible")]
        if not self.strict:
            tiers.append(("admissible on the point estimate (none robustly)", "admissible_point"))
        for label, key in tiers:
            ok = [m for m in MODE_ORDER if modes.get(m, {}).get(key, False)]
            if not ok:
                continue
            if d.mode in ok:
                return d
            why = ", ".join(f"'{m}' {_why_not(modes[m])}" for m in MODE_ORDER if m not in ok and m in modes)
            return FTDecision(time=f.time, mode=ok[0], source=self.name,
                              reasoning=f"{d.reasoning} Physics v2: {why}; least disruptive mode {label}: '{ok[0]}'.")
        return FTDecision(time=f.time, mode="safe", source=self.name,
                          reasoning="Physics v2: no mode admissible; take the largest margin.")


def _why_not(m: dict) -> str:
    if not m.get("authority_ok", True):
        return "would saturate the flap" if m.get("authority_ratio") is not None else "unstable at its airspeed"
    return "predicted unstable" if not m.get("stable", True) else "not robustly stable"


# --------------------------------------------------------------------------------------------------------------------
# Guardian: the learned diagnosis says what is wrong, the physics estimate says which modes are safe, the symptom rules
# are the fallback (docs/research/physics-advisor-plan.md, "Guardian supervisor").
# --------------------------------------------------------------------------------------------------------------------
# Pre-verified response to each known fault (the benchmark's fixed-mode table: which response each fault needs).
# "hold" = keep the airspeed, "slow" = the least disruptive slow-down, "conservative" = the largest slow-down.
FAULT_RESPONSE = {"servo_slew": "slow", "sensor_noise": "slow", "turbulence_rise": "slow", "stiffness_loss": "hold"}
RESPONSE_RANK = {"none": 0, "hold": 1, "slow": 2, "conservative": 3}
# Modes in order of preference per response; the first one the physics admits is chosen. The neural mode comes last
# everywhere: it is never the known right answer (fixed-mode table), a controller switch locks it in for MIN_DWELL_S,
# and a growing loop delay that it tolerates for a while then saturates it (outputs/uav_flap/margin_eval.json).
PREFERENCE = {"hold": ("nominal", "cautious", "safe", "robust"),
              "slow": ("cautious", "safe", "nominal", "robust"),
              "conservative": ("safe", "cautious", "nominal", "robust")}
SPEED_RANK = {"nominal": 0, "robust": 0, "cautious": 1, "safe": 2}  # "never de-escalate" compares these
SLOW_MODES = tuple(m for m in MODE_ORDER if FT_MODES[m]["speed_ratio"] is not None)
STRUCTURAL_SHIFT = -0.20  # pitch-frequency drop that FTRuleSupervisor reads as structural damage
# Identified factors outside every healthy development flight (seeds 1000-1004, all faults, margin v2: pitch stiffness
# 0.97-1.03 without structural damage, flap effectiveness >= 0.78 without flap damage before a failure or slow-down)
STIFFNESS_ALARM = 0.92  # identified pitch stiffness / healthy below this: structural evidence
FLAP_ALARM = 0.70  # identified flap effectiveness / healthy below this: control authority lost, slow down


@dataclass
class PhysicsVerdict:
    """What the on-board physics estimate says about each mode (GuardianSupervisor's pluggable check).
    time None: no estimate yet (or no estimator); then every mode is admissible and none is known unstable."""

    time: float | None
    admissible: dict[str, bool]  # may the supervisor choose this mode?
    unstable: dict[str, bool]  # predicted linearly unstable where the mode holds or goes (point estimate)
    why: dict[str, str]  # reason per inadmissible mode
    stiffness: float | None = None  # identified pitch stiffness / healthy
    flap: float | None = None  # identified flap effectiveness / healthy


def physics_verdict(margin: dict | None, flutter_speed_nominal: float) -> PhysicsVerdict:
    """
    Admissibility of each mode from the latest margin estimate (v1 `MarginEstimate` or v2 `MarginEstimateV2` dict).
      unstable    v2: not `stable` (point estimate, every transit airspeed where the identified wing needs control);
                  v1: spectral radius >= 1 (worst case on the way to the mode's airspeed).
      authority   a mode whose predicted authority ratio is >= 1 where control is needed saturates the flap. Read from
                  v2's per-mode `authority_ok` / `authority_ratio`, or from a top-level `mode_authority_ratio` dict;
                  without such fields the check is skipped (v1).
      admissible  stable and enough authority; or, for a slow-down mode, a destination below the identified wing's
                  open-loop flutter speed (the wing needs no control where it ends up, so only the transit is at risk:
                  under a growing loop delay the transit band is predicted unstable for LQG and only the neural mode
                  would pass, which the grown delay then breaks). If no mode passes, the authority test is dropped
                  (it flickers above 1 for holding speed after structural damage, which survives); if still none,
                  every mode is admitted (the physics gives no guidance).
    The robust (95 % region) admissibility of v2 is not used: after structural damage it certifies no mode, not even
    holding speed, which survives (margin_eval.json).
    """
    if not margin:
        return PhysicsVerdict(time=None, admissible={m: True for m in MODE_ORDER},
                              unstable={m: False for m in MODE_ORDER}, why={})
    v2 = margin.get("modes") or {}
    top = margin.get("mode_authority_ratio") or {}
    radius = margin.get("mode_spectral_radius") or {}
    uf = margin.get("flutter_speed")
    unstable, authority_ok, free = {}, {}, {}
    for m in MODE_ORDER:
        info = v2.get(m, {})
        unstable[m] = (not info["stable"]) if "stable" in info else radius.get(m, 0.0) >= 1.0
        ratio = info.get("authority_ratio", top.get(m))
        authority_ok[m] = bool(info["authority_ok"]) if "authority_ok" in info else (ratio is None or ratio < 1.0)
        if m in SLOW_MODES:
            needed = info.get("control_needed")
            if needed is None:
                needed = uf is None or FT_MODES[m]["speed_ratio"] * flutter_speed_nominal >= uf
            free[m] = not needed
        else:
            free[m] = False
    why: dict[str, str] = {}
    admissible = {m: free[m] or (not unstable[m] and authority_ok[m]) for m in MODE_ORDER}
    if not any(admissible.values()):
        admissible = {m: free[m] or not unstable[m] for m in MODE_ORDER}
    if not any(admissible.values()):
        admissible = {m: True for m in MODE_ORDER}
        why["all"] = "no mode predicted admissible: the physics gives no guidance"
    for m in MODE_ORDER:
        if not admissible[m]:
            why[m] = "predicted unstable" if unstable[m] else "would saturate the flap"
    params = margin.get("params") or {}
    return PhysicsVerdict(time=margin.get("time"), admissible=admissible, unstable=unstable, why=why,
                          stiffness=params.get("f_alpha"), flap=params.get("flap_span_fraction"))


class GuardianSupervisor:
    """
    The decision layer over the pre-verified modes. Every tick it combines
      (a) the learned diagnosis (FTFeatures.diagnosis): a confirmed known fault -> its pre-verified response
          (FAULT_RESPONSE; servo, noise, turbulence -> the least disruptive slow-down; structural damage -> hold speed);
          a confirmed unknown-fault alarm -> the most conservative mode;
      (b) the physics estimate (FTFeatures.margin, through the pluggable `verdict`): never a mode it finds
          inadmissible; the current mode predicted unstable is a reason to act by itself; an identified flap
          effectiveness below FLAP_ALARM means lost authority (slow down); an identified pitch stiffness below
          STIFFNESS_ALARM is structural evidence;
      (c) the symptom rules (FTRuleSupervisor) as a fallback, with their structural rule (pitch-frequency drop) read
          as structural evidence instead of as "slow down to safe".
    Structural damage (diagnosis, frequency drop or identified stiffness) changes the logic: hold the current mode
    unless the physics predicts it unstable, because slowing down can enter the band in which the damaged wing flutters
    while the controllers are scheduled for the healthy one. Timing fix: no action on structural evidence until a
    physics estimate made after the evidence first appeared exists (the estimate lags the symptom rules; acting
    earlier slowed the aircraft into that band, ft_timescale.json). Otherwise the strongest response wins, the first
    admissible mode of its PREFERENCE list is chosen, and the supervisor never de-escalates on its own unless the
    physics finds the current mode inadmissible.
    `single_window`: act on a known fault the diagnoser flags in one window (probability >= its threshold) instead of
    waiting for its confirmation (CONFIRM consecutive windows); the unknown alarm always needs confirmation.
    Without a diagnosis it is rules + physics (with the timing fix); without a margin estimate, diagnosis + rules.
    Developed on seeds >= 1000 only.
    """

    name = "guardian"

    def __init__(self, verdict: Callable[[dict | None, float], PhysicsVerdict] = physics_verdict,
                 rules: FTRuleSupervisor | None = None, single_window: bool = False):
        self.verdict, self.rules, self.single_window = verdict, rules or FTRuleSupervisor(), single_window
        self.structural_since: float | None = None

    def reset(self) -> None:
        self.structural_since = None

    def _diagnosed(self, d: dict) -> list[str]:
        if not self.single_window:
            return list(d.get("confirmed", ()))
        thr = d.get("prob_threshold", {})
        flagged = [c for c, p in d.get("probabilities", {}).items() if p >= thr.get(c, 0.5)]
        return sorted(set(flagged) | set(d.get("confirmed", ())))

    def decide(self, f: FTFeatures) -> FTDecision:
        phys = self.verdict(f.margin, f.airspeed / f.airspeed_ratio)
        diag = f.diagnosis or {}
        faults = self._diagnosed(diag)

        structural = []
        if "stiffness_loss" in faults:
            structural.append("learned diagnosis: stiffness loss")
        if f.pitch_freq_hz is not None and f.nominal_freq_hz is not None:
            shift = (f.pitch_freq_hz - f.nominal_freq_hz) / f.nominal_freq_hz
            if shift < STRUCTURAL_SHIFT:
                structural.append(f"pitch frequency {100 * shift:+.0f} %")
        if phys.stiffness is not None and phys.stiffness < STIFFNESS_ALARM:
            structural.append(f"identified pitch stiffness x{phys.stiffness:.2f}")
        if structural and self.structural_since is None:
            self.structural_since = f.time

        evidence: list[tuple[str, str]] = []  # (response, why)
        if diag.get("unknown"):
            evidence.append(("conservative", "learned diagnosis: unknown fault (the data match no known fault)"))
        evidence += [(FAULT_RESPONSE[c], f"learned diagnosis: {c}") for c in faults if FAULT_RESPONSE.get(c) != "hold"]
        # the symptom rules without their structural rule (structural evidence is handled below)
        r = self.rules.decide(replace(f, mode="nominal", pitch_freq_hz=f.nominal_freq_hz))
        if r.mode in SLOW_MODES:
            evidence.append(("slow" if r.mode == "cautious" else "conservative", f"symptom rules: {r.reasoning}"))
        if phys.flap is not None and phys.flap < FLAP_ALARM:
            evidence.append(("slow", f"identified flap effectiveness x{phys.flap:.2f}: control authority lost"))
        if phys.unstable.get(f.mode):
            evidence.append(("hold", f"physics: '{f.mode}' predicted unstable"))
        response = max((e[0] for e in evidence), key=RESPONSE_RANK.__getitem__, default="none")
        why = "; ".join(e[1] for e in evidence)

        def decision(mode: str, reasoning: str) -> FTDecision:
            return FTDecision(time=f.time, mode=mode, reasoning=reasoning, source=self.name)

        def first_admissible(resp: str) -> str:
            return next(m for m in PREFERENCE[resp] if phys.admissible[m])

        if structural:
            signs = ", ".join(structural)
            since = self.structural_since if self.structural_since is not None else f.time
            if f.margin is not None and (phys.time is None or phys.time < since - 1e-9):
                return decision(f.mode, f"Structural evidence ({signs}) since {since:.2f} s: holding '{f.mode}' until "
                                        f"a physics estimate made after it exists.")
            if not phys.unstable.get(f.mode):
                return decision(f.mode, f"Structural evidence ({signs}): hold '{f.mode}' (slowing down can enter the "
                                        f"band where the damaged wing flutters); the physics does not predict it "
                                        f"unstable." + (f" Other evidence: {why}." if why else ""))
            mode = first_admissible(response if RESPONSE_RANK[response] >= RESPONSE_RANK["slow"] else "hold")
            return decision(mode, f"Structural evidence ({signs}) and '{f.mode}' predicted unstable: '{mode}', the "
                                  f"first mode the physics admits. {why}.")
        if response == "none":
            return decision(f.mode, "No fault diagnosed, no rule fired, no physics alarm: keep the current mode.")
        mode = first_admissible(response)
        if SPEED_RANK[mode] < SPEED_RANK[f.mode] and phys.admissible[f.mode]:
            return decision(f.mode, f"{why}. Keeping the more cautious current mode '{f.mode}'.")
        vetoed = [f"'{m}' {phys.why[m]}" for m in PREFERENCE[response][:PREFERENCE[response].index(mode)]]
        return decision(mode, f"{why}. Response: {response} -> '{mode}'"
                              + (f" (physics: {', '.join(vetoed)})." if vetoed else "."))
