"""
Publication figures (matplotlib). Colour follows the entity (see docs/Design.md):
the same controller always has the same colour, in the app and in the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .paths import MODELS_DIR  # noqa: E402

# Validated categorical palette (light mode); open loop is a recessive gray baseline.
CASE_COLORS = {
    "open_loop": "#898781",
    "lqr": "#2a78d6",
    "lqg": "#4a3aa7",
    "neural": "#eb6834",
    "lqr+rules": "#1baf7a",
    "neural+rules": "#eda100",
    "lqr+llm": "#e87ba4",
    "neural+llm": "#e87ba4",
}
CASE_LABELS = {
    "open_loop": "Open loop",
    "lqr": "LQR",
    "lqg": "LQG (turbulence model)",
    "neural": "Neural",
    "lqr+rules": "LQR + rule supervisor",
    "neural+rules": "Neural + rule supervisor",
    "lqr+llm": "LQR + LLM supervisor",
    "neural+llm": "Neural + LLM supervisor",
}
MODE_COLORS = ("#4a3aa7", "#008300", "#52514e")  # structural modes: kept distinct from controller colours
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def case_key(res) -> str:
    return res.controller if res.supervisor is None else f"{res.controller}+{res.supervisor}"


def flutter_figure(p, fa, Uf_pk, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    U = np.asarray(fa.velocities)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    for m in range(len(fa.damping)):
        ax1.plot(U, fa.damping[m], color=MODE_COLORS[m], label=f"Mode {m + 1}")
        ax2.plot(U, fa.frequency_hz[m], color=MODE_COLORS[m], label=f"Mode {m + 1}")
    ax1.axhline(0, color=AXIS, linewidth=0.8)
    for ax in (ax1, ax2):
        ax.axvline(fa.flutter_speed, color=INK2, linewidth=0.8)
    ax1.annotate(f"$U_F$ = {fa.flutter_speed:.1f} m/s (state-space)\n$U_F$ = {Uf_pk:.1f} m/s (p-k, exact Theodorsen)",
                 xy=(fa.flutter_speed, 0), xytext=(8, 12), textcoords="offset points", fontsize=8, color=INK)
    ax1.set_title("Damping ratio vs airspeed (V-g)")
    ax1.set_ylabel("Damping ratio ζ")
    ax2.set_title("Frequency vs airspeed (V-f)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Airspeed U (m/s)")
    ax1.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def time_history_figure(results, p, title: str, path: Path) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2, 1.4]})
    ax_h, ax_a, ax_u, ax_g = axes
    for res in results:
        key = case_key(res)
        color = CASE_COLORS.get(key, INK2)
        lw = 1.1 if key == "open_loop" else 1.5
        label = CASE_LABELS.get(key, key)
        ax_h.plot(res.t, res.x[:, 0] * 1000, color=color, lw=lw, label=label)
        ax_a.plot(res.t, np.degrees(res.x[:, 1]), color=color, lw=lw)
        if key != "open_loop":
            j = p.primary_input
            ax_u.plot(res.t, res.u[:, j] / p.u_max[j], color=color, lw=1.0)
        if res.exceeded_limits:
            ax_h.plot(res.t[-1], res.x[-1, 0] * 1000, marker="x", color=CRITICAL, ms=7, mew=2)
            ax_h.annotate(f"limit exceeded, t = {res.failure_time:.2f} s", (res.t[-1], res.x[-1, 0] * 1000),
                          xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=CRITICAL, va="center")
    g = max(results, key=lambda r: len(r.t))  # a failed run stops early; use the longest
    ax_g.plot(g.t, g.w_gust, color=INK2, lw=1.0)
    ax_h.set_ylabel("Plunge h (mm)")
    ax_a.set_ylabel("Pitch α (deg)")
    ax_u.set_ylabel("β_cmd / β_max" if p.actuation == "flap_servo" else "M_α / M_max")
    ax_u.set_ylim(-1.1, 1.1)
    ax_g.set_ylabel("Gust (m/s)")
    ax_g.set_xlabel("Time (s)")
    ax_h.set_title(title)
    ax_h.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def tradeoff_figure(tradeoff: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ctrl in [c for c in ("lqr", "lqg", "neural") if c in tradeoff]:
        pts = tradeoff[ctrl]
        e = [q["control_energy"] for q in pts]
        r = [q["rms_h_mm"] for q in pts]
        ax.plot(e, r, color=CASE_COLORS[ctrl], marker="o", ms=4, label=f"{CASE_LABELS[ctrl]}, fixed aggressiveness 0→1")
        ax.annotate("a = 0", (e[0], r[0]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
        ax.annotate("a = 1", (e[-1], r[-1]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    for key, m in tradeoff["supervised"].items():
        ax.plot(m["control_energy"], m["rms_h_mm"], marker="D", ms=8, color=CASE_COLORS.get(key, INK),
                mec="white", mew=1.5, linestyle="none", label=CASE_LABELS.get(key, key))
    ax.set_xlabel("Actuator effort  ∫ Σ(u/u_max)² dt  (s)")
    ax.set_ylabel("RMS plunge (mm)")
    ax.set_title("Effort vs response: a supervisor adds value only below the fixed-gain curve")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def velocity_sweep_figure(sweep: dict, p, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    lim = p.h_limit * 1000
    for ctrl in [c for c in ("open_loop", "lqr", "lqg", "neural") if c in sweep]:
        rows = sweep[ctrl]
        r = np.array([q["ratio"] for q in rows])
        pk = np.array([min(q["peak_h_mm"], lim) for q in rows])
        fail = np.array([q["exceeded_limits"] for q in rows])
        ax.plot(r, pk, color=CASE_COLORS[ctrl], marker="o", ms=3.5, label=CASE_LABELS[ctrl])
        if fail.any():
            ax.plot(r[fail], pk[fail], linestyle="none", marker="x", color=CRITICAL, ms=7, mew=2)
    ax.axhline(lim, color=CRITICAL, lw=0.8)
    ax.annotate("structural limit (× = exceeded)", (sweep["ratios"][0], lim), xytext=(0, -12),
                textcoords="offset points", fontsize=7.5, color=CRITICAL)
    ax.axvline(1.0, color=INK2, lw=0.8)
    ax.set_xlabel("Airspeed / open-loop flutter speed")
    ax.set_ylabel("Peak plunge (mm)")
    ax.set_title("Stability envelope with a 4 m/s discrete gust")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def training_figure(preset: str, path: Path) -> Path | None:
    hist_path = MODELS_DIR / f"training_history_{preset}.json"
    if not hist_path.exists():
        return None
    h = json.loads(hist_path.read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    ax1.semilogy(h["imitation"], color=CASE_COLORS["neural"], lw=1.0)
    ax1.set_title("1. Imitation warm start")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("MSE (normalised)")
    ax2.plot(h["finetune"], color=CASE_COLORS["neural"], lw=1.0)
    ax2.set_title("2. Differentiable-physics fine-tuning")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("mean log(1 + J), training batch")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def delay_sweep_figure(sweep: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    dt_ms = sweep["dt"] * 1000
    labels = {"lqr": "LQR re-designed for the true delay (optimistic)", "lqr_nominal": "LQR designed for 5 ms, fixed",
              "lqg": "LQG re-designed for the true delay (optimistic)", "lqg_nominal": "LQG designed for 5 ms, fixed",
              "neural": "Neural (one policy)"}
    styles: dict[str, dict] = {"lqr": dict(color=CASE_COLORS["lqr"], alpha=0.45), "lqr_nominal": dict(color=CASE_COLORS["lqr"]),
              "lqg": dict(color=CASE_COLORS["lqg"], alpha=0.45), "lqg_nominal": dict(color=CASE_COLORS["lqg"]),
              "neural": dict(color=CASE_COLORS["neural"])}
    for ctrl in [c for c in ("lqr", "lqr_nominal", "lqg", "lqg_nominal", "neural") if c in sweep]:
        rows = sweep[ctrl]
        d = np.array([r["delay_steps"] for r in rows]) * dt_ms
        rms = np.array([r["rms_h_mm"] for r in rows])
        fail = np.array([r["exceeded_limits"] for r in rows])
        ok = ~fail
        ax.plot(d[ok], rms[ok], marker="o", ms=4, label=labels[ctrl], **styles[ctrl])
        if fail.any():
            ax.plot(d[fail], np.full(fail.sum(), max(rms[ok].max() if ok.any() else 1, 1) * 1.15), linestyle="none",
                    marker="x", ms=8, mew=2, **styles[ctrl])
    ax.axvspan(0, 2 * dt_ms, color=GRID, alpha=0.5, lw=0)
    ax.annotate("neural training range", (0.5, 0.02), xycoords=("data", "axes fraction"), fontsize=7.5, color=INK2)
    ax.set_xlabel("Input delay (ms)   (× = structural limit exceeded)")
    ax.set_ylabel("RMS plunge (mm)")
    ax.set_title("Robustness to computation / actuation delay (discrete gust, 1.1 U_F)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(p, fa, Uf_pk, examples: dict, results: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    figs = [flutter_figure(p, fa, Uf_pk, out_dir / "fig1_flutter_vg.png")]
    for i, (name, res_list) in enumerate(examples.items(), start=2):
        figs.append(time_history_figure(res_list, p, f"Scenario: {name.replace('_', ' ')}",
                                        out_dir / f"fig{i}_time_{name}.png"))
    n = len(figs) + 1
    figs.append(tradeoff_figure(results["tradeoff"], out_dir / f"fig{n}_tradeoff.png"))
    figs.append(velocity_sweep_figure(results["velocity_sweep"], p, out_dir / f"fig{n + 1}_velocity_sweep.png"))
    figs.append(delay_sweep_figure(results["delay_sweep"], out_dir / f"fig{n + 2}_delay_robustness.png"))
    tf = training_figure(p.name, out_dir / f"fig{n + 3}_training.png")
    if tf:
        figs.append(tf)
    return figs


def robustness_figure(res: dict, path: Path) -> Path:
    """Failure rate and cost of the surviving runs vs model error, sensor noise and actuator rate limit."""
    ctrls = res["controllers"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    levels = [100 * lv for lv in res["levels"]]
    par = res["parameters"]["turbulence_patch"]
    rows = [par[f"{lv:.2f}"] for lv in res["levels"]]
    for c in ctrls:
        color = CASE_COLORS.get(c, INK2)
        axes[0, 0].plot(levels, [100 * r[c]["failure_rate"] for r in rows], color=color, marker="o", ms=4,
                        label=CASE_LABELS.get(c, c))
        j = [r[c]["cost_J"]["median"] if r[c]["cost_J"] else np.nan for r in rows]
        axes[0, 1].plot(levels, j, color=color, marker="o", ms=4)
    axes[0, 0].set_xlabel("Model error: mass, stiffness, damping, servo (± %)")
    axes[0, 0].set_ylabel("Failed runs (%)")
    axes[0, 0].set_title("Turbulence at 1.1 × nominal flutter speed")
    axes[0, 1].set_xlabel("Model error (± %)")
    axes[0, 1].set_ylabel("Cost J of surviving runs (median)")
    axes[0, 1].set_title("Lower is better")

    noise = res["sensor_noise"]
    nx = [100 * float(k) for k in noise]
    for c in ctrls:
        j = [noise[k][c]["cost_J"]["median"] if noise[k][c]["cost_J"] else np.nan for k in noise]
        axes[1, 0].plot(nx, j, color=CASE_COLORS.get(c, INK2), marker="o", ms=4)
    axes[1, 0].set_xlabel("Sensor noise (% of reference scale)")
    axes[1, 0].set_ylabel("Cost J of surviving runs (median)")
    axes[1, 0].set_title("Sensor noise (turbulence)")

    rate = res.get("rate_limit") or {}
    if rate:
        keys = list(rate)
        labels = ["none" if k == "none" else f"{k}°/s" for k in keys]
        x = np.arange(len(keys))
        for c in ctrls:
            axes[1, 1].plot(x, [100 * rate[k][c]["failure_rate"] for k in keys], color=CASE_COLORS.get(c, INK2),
                            marker="o", ms=4)
        axes[1, 1].set_xticks(x, labels)
        axes[1, 1].set_xlabel("Flap slew-rate limit (unknown to the designs)")
        axes[1, 1].set_ylabel("Failed runs (%)")
        axes[1, 1].set_title("Actuator rate limit (turbulence)")
    else:
        axes[1, 1].axis("off")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"Robustness: controllers designed for the nominal wing ({res['metadata']['draws']} draws per point)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path



def envelope_figure(res: dict, path: Path) -> Path:
    """Survival vs airspeed per controller (real servo solid, slew limit removed dashed), the linear loop's growth
    rate, and how often the commanded flap rate exceeds the real servo's limit (envelope.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ctrls, ratios, uf = res["controllers"], res["ratios"], res["flutter_speed"]
    band = res.get("neural_training_range")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    titles = {"turbulence_patch": "Turbulence patch", "discrete_gust": "Discrete (1 - cos) gust"}
    for ax, (sc, by_servo) in zip(axes[0], res["scenarios"].items()):
        for c in ctrls:
            for sv, style in (("actual", "-"), ("unlimited", "--")):
                rows = by_servo.get(sv, {}).get(c)
                if rows:
                    ax.plot([r["ratio"] for r in rows], [100 * (1 - r["failures"] / r["n"]) for r in rows], style,
                            color=CASE_COLORS.get(c, INK2), marker="o" if sv == "actual" else None, ms=4,
                            label=CASE_LABELS.get(c, c) if sv == "actual" else None)
        ax.set_ylim(-4, 104)
        ax.set_ylabel(f"Flights within limits (%, {len(res['metadata']['seeds'])} seeds)")
        ax.set_title(titles.get(sc, sc))
    lim = res.get("servo_rate_limit_deg_s")
    axes[0, 0].plot([], [], "-", color=INK2, label=f"servo {lim[0]:.0f}°/s" if lim else "actuator as designed")
    axes[0, 0].plot([], [], "--", color=INK2, label="slew limit removed")
    axes[0, 0].legend(loc="lower left")

    ss = res["small_signal"]
    ax = axes[1, 0]
    for c in ["open_loop", *ctrls]:
        ys = [ss[f"{r:.2f}"].get(c, {}).get("growth_per_s", np.nan) for r in ratios]
        ax.plot(ratios, ys, color=CASE_COLORS.get(c, INK2), marker="o", ms=4, label=CASE_LABELS.get(c, c),
                lw=1.2 if c == "open_loop" else 1.6)
    ax.axhline(0.0, color=AXIS, lw=1.0)
    ax.set_ylabel("Growth rate of the slowest-decaying mode (1/s)")
    ax.set_title("Linear loop (no saturation): above 0 = unstable")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    turb = res["scenarios"].get("turbulence_patch", {}).get("unlimited", {})
    for c in ctrls:
        rows = turb.get(c) or []
        ax.plot([r["ratio"] for r in rows], [100 * (r["rate_above_limit"] or 0.0) for r in rows],
                color=CASE_COLORS.get(c, INK2), marker="o", ms=4)
    ax.set_ylabel("Commanded flap rate above the servo limit (% of steps)")
    ax.set_title("Turbulence, slew limit removed: rate the servo would need")
    for ax in axes.flat:
        if band:
            ax.axvspan(band[0], band[1], color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel(f"Airspeed / flutter speed (U_F = {uf:.1f} m/s)")
    if band:
        axes[0, 1].text(band[0] + 0.01, 2, "neural training range", color=INK2, fontsize=7.5, va="bottom")
    fig.suptitle(f"Operating envelope, {res['preset']}: controllers gain-scheduled at the flown airspeed")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

def llm_figure(res: dict, tradeoff: dict | None, path: Path) -> Path:
    """LLM latency, and where supervised runs land relative to the fixed-aggressiveness trade-off curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    lat = [d["latency_s"] for d in res["decisions"] if d["source"] == "llm"]
    ax1.hist(lat, bins=20, color=CASE_COLORS["neural+llm"], edgecolor="white")
    ax1.axvline(0.25, color=INK2, lw=0.8)
    ax1.annotate("supervisor interval 0.25 s", (0.25, 0.95), xycoords=("data", "axes fraction"), fontsize=7.5,
                 color=INK2, xytext=(4, 0), textcoords="offset points", va="top")
    ax1.set_xlabel("Answer latency (s)")
    ax1.set_ylabel("Decisions")
    ax1.set_title(f"{res['metadata']['model']} ({res['metadata']['backend']}): latency per decision")

    seed0 = res["scenarios"]["turbulence_patch"]["seed0"]
    for ctrl in ("neural", "lqg"):
        if tradeoff and ctrl in tradeoff:
            pts = tradeoff[ctrl]
            ax2.plot([q["control_energy"] for q in pts], [q["rms_h_mm"] for q in pts], color=CASE_COLORS[ctrl],
                     marker="o", ms=3, lw=1.2, label=f"{CASE_LABELS[ctrl]}, fixed aggressiveness 0→1")
        for sup, marker in (("rules", "D"), ("random", "s"), ("llm", "*")):
            m = seed0.get(f"{ctrl}+{sup}")
            if m and not m["exceeded_limits"]:
                ax2.plot(m["control_energy"], m["rms_h_mm"], marker=marker, ms=10 if marker == "*" else 7,
                         color=CASE_COLORS[ctrl], mec="white", mew=1.2, linestyle="none",
                         label=f"{CASE_LABELS[ctrl]} + {sup} supervisor")
    ax2.set_xlabel("Actuator effort  ∫ Σ(u/u_max)² dt  (s)")
    ax2.set_ylabel("RMS plunge (mm)")
    ax2.set_title("Turbulence: a supervisor adds value only below its curve")
    ax2.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


FT_SUP_COLORS = {"none": "#898781", "random": "#c3c2b7", "rules": "#1baf7a", "llm": "#e87ba4", "oracle": "#0b0b0b"}


def ft_figure(res: dict, path: Path) -> Path:
    """Survival and mission cost (time slowed down) per fault and supervisor."""
    faults = list(res["faults"])
    sups = [s for s in ("none", "random", "rules", "llm", "oracle") if s in res["faults"][faults[0]]["supervisors"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    x = np.arange(len(faults))
    w = 0.8 / len(sups)
    for i, s in enumerate(sups):
        surv = [100 * res["faults"][f]["supervisors"][s]["survival"] for f in faults]
        slow = [np.nan if (v := res["faults"][f]["supervisors"][s]["slowed_s_mean"]) is None else v
                for f in faults]  # surviving flights only; no bar if none survived
        ax1.bar(x + (i - (len(sups) - 1) / 2) * w, surv, w * 0.9, color=FT_SUP_COLORS.get(s, INK2), label=s)
        ax2.bar(x + (i - (len(sups) - 1) / 2) * w, slow, w * 0.9, color=FT_SUP_COLORS.get(s, INK2))
    labels = [f + ("\n(held out)" if res["faults"][f]["held_out"] else "") for f in faults]
    ax2.set_xticks(x, labels, fontsize=8)
    ax1.set_ylabel("Flights survived (%)")
    ax2.set_ylabel("Time slowed down (s)\n(surviving flights)")
    ax1.set_title("Fault-tolerant supervision: survival (higher is better) and mission cost (lower is better)")
    ax1.legend(ncol=len(sups), loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
