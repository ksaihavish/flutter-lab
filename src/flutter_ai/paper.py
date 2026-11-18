"""
Paper draft generator (Phase 4.8): every number and every comparative statement in the draft is computed from the
result files or read from the code's parameters, so nothing is typed by hand (Rules.md §1). Citations come from
docs/research/references.bib; entries not read in full are listed in a citation-check appendix.

    python -m flutter_ai paper        # -> docs/research/paper-draft.md (+ docs/research/figures/)
    python -m flutter_ai paper --springer   # + single-column LaTeX for Springer journals (paper_latex.py)

Inputs (run these first; missing ones become a visible "not run yet" note):
    outputs/<preset>/results.json (evaluate), robustness.json (robustness), ablations.json (ablations),
    llm_benchmark.json (llm-benchmark), ft_benchmark.json (ft-benchmark), ft_stability.json (ft-stability),
    outputs/validation/bact.json (validate), models/classical_tuning_<preset>.json (tune).
Edit the prose here; never edit numbers in the generated file.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .params import PRESETS
from .paths import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT

PRESET_ORDER = ("uav_flap", "uav_section", "bact")
SCEN_LABEL = {"turbulence_patch": "Turbulence, 1.1 U_F", "discrete_gust": "Discrete gust, 1.1 U_F",
              "high_speed_gust": "Gust, 1.3 U_F", "below_flutter": "Gust, 0.9 U_F"}
CTRL_LABEL = {"open_loop": "Open loop", "lqr": "LQR", "lqg": "LQG", "neural": "Neural",
              "lqr+rules": "LQR + rules", "neural+rules": "Neural + rules"}
RESEARCH = PROJECT_ROOT / "docs" / "research"
AUTHORS = ("Kukkadapu Sai Havish", "Rammohan B")  # as printed on the paper, first author first (owner, 2026-11-16)
AFFILIATIONS = ("Student, RV University, Bengaluru, India", "Professor, RV University, Bengaluru, India")
AFFILIATION = "RV University, Bengaluru, India"  # one line for the IEEE author block
CORRESPONDING = ("Rammohan B", "rammohanb@rvu.edu.in")  # as in his revision of the manuscript
REPOSITORY = "https://github.com/ksaihavish/flutter-lab"
KEYWORDS = ("Active flutter suppression, Fault-tolerant control, Supervisory control, Fault diagnosis, "
            "Large language models, Switched systems")
TITLE = ("Language-Model versus Learned and Physics-Based Supervision for Fault-Tolerant Active Flutter "
         "Suppression")
FIGURES: dict[str, tuple[list[tuple[str, str]], str]] = {  # key: ([(preset, file), ...], caption); numbered in this order
    "section": ([], r"Typical section with a trailing-edge flap: geometry in semichords $b$ and the positive "
                r"directions of plunge $h$, pitch $\alpha$ and flap deflection $\beta$."),
    "vg": ([("uav_flap", "fig1_flutter_vg.png"), ("bact", "fig1_flutter_vg.png")],
           "Open-loop damping ratio and frequency of the aeroelastic modes against airspeed for `uav_flap` (left) and "
           "the NASA BACT model (right). Flutter occurs where a mode's damping crosses zero; the vertical line marks "
           "U_F."),
    "training": ([("uav_flap", "fig9_training.png")], "Training of the neural policy: imitation warm start (left) and "
                 "differentiable-physics fine-tuning (right)."),
    "time": ([("uav_flap", "fig2_time_turbulence_patch.png")], "Time histories in turbulence at 1.1 U_F on `uav_flap` "
             "(seed 0): plunge, pitch, flap command and gust velocity."),
    "delay": ([("uav_flap", "fig8_delay_robustness.png")], "Tolerance to loop delay on `uav_flap`: designs re-computed "
              "for the true delay against designs fixed at one sample of delay, and the single neural policy "
              "(crosses: structural limit exceeded)."),
    "robust": ([("uav_flap", "fig10_robustness.png")], "Robustness of designs made for the nominal `uav_flap` wing to "
               "model error, sensor noise and actuator slew-rate limits."),
    "llm": ([("uav_flap", "fig11_llm_supervisor.png")], "Local LLM supervising effort modes: distribution of the "
            "latency per decision (left) and the position of the supervised runs relative to the fixed-aggressiveness "
            "curves (right)."),
    "tradeoff": ([("uav_flap", "fig6_tradeoff.png")], r"Trade-off between effort and response at fixed aggressiveness "
                 r"$a \in [0, 1]$ in turbulence on `uav_flap`. A supervisor adds value only if its point lies below "
                 "the curve of its controller."),
    "ft": ([("uav_flap", "fig12_fault_tolerance.png")], "Fault-tolerant supervision on set A: flights survived (top) "
           "and time slowed down over the surviving flights (bottom), per fault and supervisor."),
    "physics": ([("uav_flap", "fig12_fault_tolerance_physics_test100.png")], "Physics-informed supervision on set B: "
                "rules and LLM with and without the on-board flutter-margin estimate."),
    "diagnoser": ([("uav_flap", "fig14_diagnoser.png")], "Learned fault diagnoser on held-out evaluation flights: (a) "
                  "precision and recall per known fault, (b) detection delay, (c) unknown-fault score of each flight, "
                  "including the two fault types never seen in training, and (d) one evaluation flight with three "
                  "faults."),
    "guardian": ([("uav_flap", "fig12_fault_tolerance_guardian_test300.png")], "The guardian (learned diagnosis with "
                 "physics admissibility, no language model) against the other supervisors on set C."),
    "review": ([("uav_flap", "fig17_review_loop.png")], "Post-flight LLM reviewer: each hypothesis, and each "
               "forced-wrong control, through the checks of the verification gate against the thresholds fixed in "
               "advance."),
}
TABLES = ("valid", "tuning", "faults", "bench", "delay", "robust", "ablation", "fixed", "survival", "paired", "gate")
SUP_SHORT = {"oracle": "Orc", "none": "None", "rules": "Rules", "random": "Rand", "llm": "LLM", "rules+physics": "R+P",
             "llm+physics": "L+P", "rules+physics_v2": "R+P2", "guardian": "Guard"}
ABL_LABEL = {"neural": "Full neural policy", "neural:imitation_only": "Imitation only",
             "neural:no_rate_penalty": "No rate penalty", "neural:no_airspeed": "No airspeed input",
             "neural+rules": "Neural + rule supervisor", "neural+random": "Neural + random supervisor"}


# ------------------------------------------------------------------------------------------------ helpers
def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _pct(x: float) -> str:
    return f"{100 * x:+.0f} %"


def _sec(x: float | None) -> str:
    """Seconds, or a dash when there is nothing to average (no surviving flight)."""
    return "–" if x is None else f"{x:.1f} s"


def _p(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else f"p = {p:.2g}"


def _bib(path: Path) -> dict[str, dict[str, str]]:
    """Minimal parser for our own references.bib (one field per line)."""
    entries: dict[str, dict[str, str]] = {}
    key = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"@(\w+)\{([^,]+),", line.strip())
        if m:
            key = m.group(2)
            entries[key] = {"type": m.group(1)}
            continue
        m = re.match(r"\s*(\w+)\s*=\s*\{(.*)\},?\s*$", line)
        if key and m:
            entries[key][m.group(1)] = m.group(2).replace("{", "").replace("}", "")
    return entries


class Refs:
    """Collects citations in order of first use and renders them as numbers [n] (IJASS style)."""

    def __init__(self, bib: dict[str, dict[str, str]], marker: bool = False):
        self.bib = bib
        self.marker = marker  # True: emit ⟦cite:k1,k2⟧ placeholders for the LaTeX export (paper_latex.py)
        self.used: list[str] = []
    def __call__(self, *keys: str) -> str:
        for k in keys:
            if k not in self.used:
                self.used.append(k)
        if self.marker:
            return "⟦cite:" + ",".join(keys) + "⟧"
        return "[" + ", ".join(str(self.used.index(k) + 1) for k in keys) + "]"

    @staticmethod
    def _plain(text: str) -> str:
        """BibTeX/LaTeX markup -> plain text for Markdown."""
        for a, b in ((r'\"o', "ö"), (r'\"a', "ä"), (r"\~a", "ã"), (r"$\le$", "≤"), (r"$^\circ$", "°"), ("$", "")):
            text = text.replace(a, b)
        return text

    def listing(self) -> list[str]:
        return [f"- [{n}] {springer_ref(self.bib[k])}" for n, k in enumerate(self.used, start=1)]

    def to_check(self) -> list[str]:
        return [f"- [{k}] read status: **{self.bib[k].get('readstatus', '?')}**" for k in self.used
                if self.bib[k].get("readstatus") != "full"]


def _initials(name: str) -> str:
    """'Hespanha, João P.' -> 'Hespanha JP' (Springer basic style); a name without a comma is kept."""
    if "," not in name:
        return name
    last, first = (x.strip() for x in name.split(",", 1))
    return f"{last} " + "".join(w[0] for w in re.split(r"[\s.\-]+", first) if w)


def springer_ref(entry: dict[str, str]) -> str:
    """One reference in the Springer basic style that IJASS asks for, e.g. 'Hespanha JP, Morse AS (1999) Title.
    In: Booktitle, vol 3, pp 2655–2660. https://doi.org/...'. Full journal titles (allowed by the guidelines)."""
    e = {f: Refs._plain(v) for f, v in entry.items()}
    names = [a.strip() for a in e.get("author", "").split(" and ")]
    who = ", ".join(_initials(a) for a in names if a != "others") + (" et al" if "others" in names else "")
    head = f"{who} ({e.get('year', '')}) {e['title'].rstrip('.')}"
    pages = e.get("pages", "").replace("--", "–")
    kind = e["type"]
    if kind == "article":
        body = e["journal"] + (f" {e['volume']}" if "volume" in e else "") + (f":{pages}" if pages else "")
    elif kind == "book":
        ed = e.get("edition", "")
        ed = ed + {"1": "st", "2": "nd", "3": "rd"}.get(ed, "th") if ed.isdigit() else ed
        head += f", {ed} edn" if ed else ""
        body = e.get("publisher", "")
    elif kind == "inproceedings":
        body = "In: " + ", ".join(x for x in (e.get("booktitle", ""), f"vol {e['volume']}" if "volume" in e else "",
                                              f"pp {pages}" if pages else "") if x)
    elif kind == "techreport":
        body = ", ".join(x for x in (e.get("number", ""), e.get("institution", "")) if x)
    else:  # misc: preprint or web page
        body = f"Preprint at https://arxiv.org/abs/{e['eprint']}" if "eprint" in e else e.get("howpublished", "")
    tail = f". https://doi.org/{e['doi']}" if "doi" in e else ""
    return f"{head}. {body}{tail}" if body else f"{head}{tail}"


def _cell(agg: dict) -> str:
    s = f"{agg['rms_h_mm']['mean']:.2f} ± {agg['rms_h_mm']['std']:.2f}"
    if agg.get("cost_J"):
        s += f" · J {agg['cost_J']['mean']:.3g}"
    if agg["failures"]:
        s += f" · **{agg['failures']}/{agg['n']} failed**"
    return s


def _results_table(r: dict) -> list[str]:
    cases = [c for c in ("open_loop", "lqr", "lqg", "neural") if c in next(iter(r["scenarios"].values()))["cases"]]
    lines = ["| Scenario | " + " | ".join(CTRL_LABEL[c] for c in cases) + " |", "|---" * (len(cases) + 1) + "|"]
    for s, d in r["scenarios"].items():
        lines.append(f"| {SCEN_LABEL.get(s, s)} | " + " | ".join(_cell(d["cases"][c]) for c in cases) + " |")
    return lines


def _best_by_J(d: dict) -> tuple[str, int]:
    ok = {c: a["cost_J"]["mean"] for c, a in d["cases"].items() if "+" not in c and a.get("cost_J") and not a["failures"]}
    return (min(ok, key=lambda c: ok[c]), len(ok)) if ok else ("none", 0)


def _stat_sentence(d: dict, key: str, a: str, b: str) -> str:
    st = d["statistics"].get(key)
    if not st or "rel_diff" not in st.get("cost_J", {}):
        return f"{a} vs {b}: not comparable (too few runs where both survived)."
    j = st["cost_J"]
    word = "lower" if j["mean_diff"] < 0 else "higher"
    return (f"{a} has {abs(100 * j['rel_diff']):.0f} % {word} cost J than {b} "
            f"(95 % CI of the difference [{j['ci95'][0]:+.3g}, {j['ci95'][1]:+.3g}], Wilcoxon {_p(j['wilcoxon_p'])}, "
            f"n = {j['n']}; failures {st['failures']['candidate']} vs {st['failures']['reference']} of {st['n_seeds']}).")


def _neural_vs(r: dict, ref: str) -> tuple[int, int]:
    """(scenarios where neural has lower J than `ref` with p < 0.05, scenarios compared)."""
    wins = n = 0
    for d in r["scenarios"].values():
        st = d["statistics"].get(f"neural vs {ref}", {}).get("cost_J", {})
        if "rel_diff" in st:
            n += 1
            wins += st["mean_diff"] < 0 and st["wilcoxon_p"] < 0.05
    return wins, n


def _llm_model(ft: dict | None, llm: dict | None) -> str:
    if ft and ft["metadata"].get("llm"):
        return str(ft["metadata"]["llm"]["model"])
    if llm:
        return str(llm["metadata"]["model"])
    return "llama3.1:8b"


def _mode_text(spec: dict) -> str:
    speed = "cruise" if spec["speed_ratio"] is None else f"{spec['speed_ratio']:.1f} U_F"
    return f"{spec['controller'].upper()} at {speed}"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _surv(ft: dict, fault: str, sup: str) -> float:
    return float(ft["faults"][fault]["supervisors"][sup]["survival"])


def _ft_supervisor_table(ft: dict) -> list[str]:
    """Survival per fault and supervisor, pooled summary and paired tests of one ft-benchmark result file."""
    faults = list(ft["faults"])
    sups = list(ft["faults"][faults[0]]["supervisors"])
    out = [f"\nSupervisors ({len(ft['metadata']['seeds'])} seeds per fault, seeds {ft['metadata']['seeds'][0]}-"
           f"{ft['metadata']['seeds'][-1]}; survival · mean time slowed down over the surviving flights · final mode = "
           "oracle's):\n",
           "| Fault | " + " | ".join(sups) + " |", "|---" * (len(sups) + 1) + "|"]
    for f in faults:
        d = ft["faults"][f]["supervisors"]
        out.append(f"| {f} | " + " | ".join(
            f"{d[s]['survival']:.0%} · {_sec(d[s]['slowed_s_mean'])}"
            + (f" · {d[s]['oracle_agreement']:.0%}" if d[s].get("oracle_agreement") is not None else "")
            for s in sups) + " |")
    out.append("")
    pooled = ft["pooled_faulty"]
    held = [f for f in faults if ft["faults"][f]["held_out"]]
    for s in sups:
        surv_h = float(np.mean([ft["faults"][f]["supervisors"][s]["survival"] for f in held])) if held else float("nan")
        fa = ft["faults"]["none"]["supervisors"][s]["slowed_s_mean"]
        out.append(f"- **{s}**: {100 * pooled['supervisors'][s]['survival']:.0f} % of faulty flights survived "
                   f"({100 * surv_h:.0f} % on held-out faults); {_sec(fa)} slowed down on average without a fault.")
    for label, st in pooled["paired"].items():
        out.append(f"- {label}: survived only with the first {st['survived_only_candidate']}, only with the second "
                   f"{st['survived_only_reference']} (exact McNemar {_p(st['mcnemar_p'])})"
                   + (f"; time slowed {st['slowed_s_mean_diff']:+.1f} s over the {st['n_both_survived']} flights both "
                      f"survived (Wilcoxon {_p(st['wilcoxon_p_slowed'])})." if st.get("n_both_survived") else "."))
    if ft.get("llm_latency_s"):
        lat = ft["llm_latency_s"]
        out.append(f"- LLM latency: median {lat['median']:.1f} s, 95th percentile {lat['p95']:.1f} s over {lat['n']} "
                   "decisions" + (f" (applied as a fixed {ft['metadata']['llm_latency_fixed_s']:.2f} s in simulation "
                                  "time)" if ft["metadata"].get("llm_latency_fixed_s") else "") + ".")
    out.append("")
    return out


def _pair_sentence(ft: dict, a: str, b: str) -> str | None:
    """'a X % vs b Y % of faulty flights survived (...)' from the pooled paired test, or None if not in the file."""
    st = ft["pooled_faulty"]["paired"].get(f"{a} vs {b}")
    if st is None:
        return None
    sa, sb = (ft["pooled_faulty"]["supervisors"][s]["survival"] for s in (a, b))
    sig = "significant" if st["mcnemar_p"] < 0.05 else "not significant"
    return (f"{a} {100 * sa:.0f} % vs {b} {100 * sb:.0f} % of faulty flights survived ({sig}: survived only with "
            f"{a} {st['survived_only_candidate']}, only with {b} {st['survived_only_reference']}, exact McNemar "
            f"{_p(st['mcnemar_p'])})")


def _acted(ft: dict) -> str:
    """Per supervisor, faulty flights with a mode change decided after the fault began (first_response set)."""
    out = []
    for s in ft["faults"]["none"]["per_seed"]:
        if s == "oracle":
            continue
        runs = [r for f, v in ft["faults"].items() if f != "none" for r in v["per_seed"][s]]
        out.append(f"{s} {sum(r['first_response'] is not None for r in runs)} of {len(runs)}")
    return ", ".join(out)


def _changed_faults(ft: dict, a: str, b: str) -> str:
    """Per fault, where supervisor a's survival differs from b's: 'fault a % vs b %'."""
    diff = [(f, _surv(ft, f, a), _surv(ft, f, b)) for f in ft["faults"] if f != "none"
            and _surv(ft, f, a) != _surv(ft, f, b)]
    return "; ".join(f"{f} {100 * x:.0f} % vs {100 * y:.0f} %" for f, x, y in diff) or "no fault"


def _ft_discussion(ft: dict | None) -> list[str]:
    """Data-driven discussion bullets for the fault-tolerant study (the owner refines the wording)."""
    if not ft:
        return ["- **Fault tolerance:** *[results pending: `python -m flutter_ai ft-benchmark`]*"]
    faults = [f for f in ft["faults"] if f != "none"]
    sups = list(ft["faults"][faults[0]]["supervisors"])
    out = []
    pooled = ft["pooled_faulty"]
    if "llm" in sups and "rules" in sups:
        st = pooled["paired"]["llm vs rules"]
        better = [f for f in faults if _surv(ft, f, "llm") > _surv(ft, f, "rules")]
        worse = [f for f in faults if _surv(ft, f, "llm") < _surv(ft, f, "rules")]
        sig = "significant" if st["mcnemar_p"] < 0.05 else "not significant"
        out.append(f"- **LLM vs rules.** Pooled over faulty flights the LLM survived "
                   f"{100 * pooled['supervisors']['llm']['survival']:.0f} % and the rules "
                   f"{100 * pooled['supervisors']['rules']['survival']:.0f} % ({sig}, exact McNemar {_p(st['mcnemar_p'])}). "
                   f"The LLM did better on {', '.join(better) or 'no fault'} and worse on {', '.join(worse) or 'no fault'}.")
        held = [f for f in faults if ft["faults"][f]["held_out"]]
        if held:
            out.append("- **Held-out faults** (in neither the rules nor the guidance): " + "; ".join(
                f"{f}: LLM {100 * _surv(ft, f, 'llm'):.0f} %, rules {100 * _surv(ft, f, 'rules'):.0f} %, "
                f"oracle {100 * _surv(ft, f, 'oracle'):.0f} %" for f in held) + ". They test different things: the "
                "loop delay appears as a number in the report (a flight computer times its own loop), so it tests "
                "acting on an explicit indicator nobody wrote a rule for; flap damage shows only through its symptoms.")
        none = ft["faults"].get("none", {}).get("supervisors", {})
        if none:
            out.append("- **False alarms** (time slowed down in fault-free flights): " + ", ".join(
                f"{s} {_sec(none[s]['slowed_s_mean'])}" for s in sups if s in none) + ".")
    if "oracle" in sups:
        gap = {s: pooled["supervisors"]["oracle"]["survival"] - pooled["supervisors"][s]["survival"]
               for s in sups if s != "oracle"}
        out.append("- **Gap to the oracle** (share of faulty flights the oracle saves but the supervisor loses): "
                   + ", ".join(f"{s} {100 * g:.0f} %" for s, g in gap.items()) + ".")
    fm = ft["faults"].get("stiffness_loss", {}).get("fixed_modes")
    if fm and fm["cautious"]["survival"] < fm["nominal"]["survival"]:
        out.append(f"- **The structural-damage trap.** After stiffness loss, holding speed survived "
                   f"{100 * fm['nominal']['survival']:.0f} % but a moderate slow-down only "
                   f"{100 * fm['cautious']['survival']:.0f} %: the controllers are gain-scheduled for the healthy wing, "
                   "which is below flutter at the lower speed while the damaged wing is not. A reflex of \"anomaly, "
                   "slow down a little\" is wrong here.")
    undetected = [f for f in faults if all(_surv(ft, f, s) == 0.0 for s in ("rules", "llm") if s in sups)
                  and _surv(ft, f, "oracle") > 0.5]
    if undetected:
        out.append(f"- **Faults no supervisor caught in time:** {', '.join(undetected)}. A correct response existed "
                   "(the oracle survives) but the on-board symptoms appeared too late for both the rules and the LLM; "
                   "better diagnostics, not a better decision-maker, is the bottleneck there.")
    return out


def _verdict(ft: dict, a: str, b: str) -> str:
    """'helped' / 'hurt' / 'did not measurably change' from the pooled paired test of a against b."""
    st = ft["pooled_faulty"]["paired"][f"{a} vs {b}"]
    sa, sb = (ft["pooled_faulty"]["supervisors"][s]["survival"] for s in (a, b))
    if st["mcnemar_p"] >= 0.05:
        return "did not measurably change"
    return "helped" if sa > sb else "hurt"


def _later_discussion(phys: dict | None, guard: dict | None) -> list[str]:
    """Discussion bullets for Sections 7.6-7.7, computed from the result files."""
    out = []
    if phys:
        pool = phys["pooled_faulty"]["supervisors"]
        v = _verdict(phys, "llm+physics", "llm")
        dmg = phys["faults"].get("stiffness_loss", {}).get("per_seed", {})
        first = {}
        for s in ("llm", "llm+physics", "rules+physics"):
            c: dict[str, int] = {}
            for r in dmg.get(s, []):
                k = r["first_response"] or "no change"
                c[k] = c.get(k, 0) + 1
            first[s] = ", ".join(f"{k} {n}" for k, n in sorted(c.items(), key=lambda x: -x[1]))
        out.append(f"- **Is information the LLM's bottleneck?** The same physics estimate {_verdict(phys, 'rules+physics', 'rules')} "
                   f"the rules and {v} the LLM ({100 * pool['llm+physics']['survival']:.0f} % vs "
                   f"{100 * pool['llm']['survival']:.0f} %). After structural damage the first responses were: llm "
                   f"({first['llm']}); llm+physics ({first['llm+physics']}); rules+physics ({first['rules+physics']}). "
                   "The model used the estimate as a reason to slow down one step rather than as the per-mode prediction "
                   "it is, which is the trap described above.")
    if guard:
        pool = guard["pooled_faulty"]["supervisors"]
        rx = ", ".join(f"{s} {_sec(pool[s]['reaction_s_median'])}" for s in pool if s not in ("oracle", "none"))
        out.append(f"- **What made the guardian work.** It decides every 0.25 s from a diagnosis that names the fault, and "
                   "maps a named fault to a response verified in advance; the physics vetoes modes and adds alarms of its "
                   "own (lost control authority, current mode predicted unstable). Median reaction "
                   f"after fault onset (s): {rx}. Its misses are faults whose on-board signature appears late (a slow servo "
                   "is invisible until the flap is asked to move fast) and physics estimates that read a damaged wing as "
                   "softer than it is.")
    return out


def _later_findings(phys: dict | None, guard: dict | None, expl: dict | None, rev: dict | None = None) -> list[str]:
    """Headline sentences of Sections 7.6-7.8 for the abstract and the conclusion (computed, never typed)."""
    out = []
    if phys:
        pool = phys["pooled_faulty"]["supervisors"]
        out.append(f"In a pre-registered test on fresh seeds, an on-board flutter-margin estimate "
                   f"{_verdict(phys, 'rules+physics', 'rules')} the rules ({100 * pool['rules+physics']['survival']:.0f} % "
                   f"vs {100 * pool['rules']['survival']:.0f} % of faulty flights survived) and "
                   f"{_verdict(phys, 'llm+physics', 'llm')} the LLM ({100 * pool['llm+physics']['survival']:.0f} % vs "
                   f"{100 * pool['llm']['survival']:.0f} %).")
    if guard:
        pool = guard["pooled_faulty"]["supervisors"]
        st = guard["pooled_faulty"]["paired"]["guardian vs rules"]
        out.append(f"A supervisor built from fast models instead (a learned fault diagnoser and the physics estimate "
                   f"extended with control authority, no language model) survived "
                   f"{100 * pool['guardian']['survival']:.0f} % of faulty flights on fresh seeds, against "
                   f"{100 * pool['rules']['survival']:.0f} % for the rules (exact McNemar {_p(st['mcnemar_p'])}) and "
                   f"{100 * pool['oracle']['survival']:.0f} % for the oracle.")
    if expl and expl["summary"]["faithfulness_rate"] is not None:
        s_ = expl["summary"]
        out.append(f"Off the control path, the same language model's explanations of a physics-informed supervisor's "
                   f"decisions passed a faithfulness check in {100 * s_['faithfulness_rate']:.0f} % of its valid answers.")
    if rev:
        s_ = rev["summary"]
        acc = sum(c["accepted"] for c in s_["gate"])
        right_acc = sum(c["accepted"] and c["right_kind"] for c in s_["gate"])
        out.append(f"As a post-flight reviewer it named the true fault in {s_['right_kind']} of {s_['valid_hypotheses']} "
                   f"valid hypotheses, and a verification gate accepted {acc} of {len(s_['gate'])} retrained diagnosers"
                   + (f" ({right_acc} with the right fault)" if acc else "") + ", so no unverified change reached the "
                   "fast models.")
    return out


def _conclusion(r: dict | None, llm: dict | None, ft: dict | None, stab: dict | None,
                later: list[str] | None = None) -> str:
    parts = []
    if r:
        wl, nl = _neural_vs(r, "lqr")
        wg, ng = _neural_vs(r, "lqg")
        parts.append(f"On a validated, realistic flutter-suppression benchmark with fair baselines, a neural policy "
                     f"trained through differentiable simulation beat plain LQR in {wl} of {nl} scenarios and a "
                     f"turbulence-aware LQG in {wg} of {ng}.")
    if llm:
        parts.append(f"An on-board LLM supervising controller aggressiveness answered in "
                     f"{llm['summary']['latency_s']['median']:.1f} s at the median and could not act on gusts.")
    if ft:
        pooled = ft["pooled_faulty"]
        if "llm" in pooled["supervisors"] and "rules" in pooled["supervisors"]:
            llm_s, rules_s = pooled["supervisors"]["llm"]["survival"], pooled["supervisors"]["rules"]["survival"]
            p_ = pooled["paired"]["llm vs rules"]["mcnemar_p"]
            verdict = ("did better than" if llm_s > rules_s and p_ < 0.05 else
                       "did worse than" if llm_s < rules_s and p_ < 0.05 else "was statistically indistinguishable from")
            parts.append(f"Given slow faults and a menu of pre-verified modes, the same model {verdict} hand-written "
                         f"rules ({100 * llm_s:.0f} % vs {100 * rules_s:.0f} % of faulty flights survived), and both "
                         f"stayed below an oracle that knows the fault ({100 * pooled['supervisors']['oracle']['survival']:.0f} %).")
    if stab:
        parts.append(f"Because every supervisor only selects among modes that are each stable, and controller switches are spaced "
                     f"by at least the computed dwell time ({stab['max_dwell_s']:.1f} s), no sequence of its answers can "
                     "destabilise the nominal closed loop in the small-signal sense (faults, actuator limits and airspeed "
                     "transitions are outside this guarantee); its value depends on the diagnostics it is given and on "
                     "the time scale of the decision.")
    parts.extend(later or [])
    parts.append("*[Owner: closing sentence in your words. Suggestion: on-board supervision of a fast, unstable plant "
                 "should be built from fast models whose decisions can be verified, with the language model kept off "
                 "the control path where its explanations help the operator; the next step is a wind-tunnel test on "
                 "measured data.]*")
    return " ".join(parts)


# ------------------------------------------------------------------------------------------------ the draft
def _holm(ps: list[float]) -> list[float]:
    """Holm's step-down adjusted p values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def _pt(p: float) -> str:
    """p value for a table cell: two significant digits, or an order-of-magnitude bound below 0.001."""
    return f"{p:.2g}" if p >= 1e-3 else f"$<10^{{{math.floor(math.log10(p)) + 1}}}$"


def _pc(x: float) -> str:
    return f"{math.floor(100 * x + 0.5)}%"  # half up: 2.5 % -> 3 %


def _ci(d: dict, s: str) -> str:
    """Wilson 95 % interval of a pooled survival rate, e.g. '95% CI 87–97%'."""
    n, p, z = d["pooled_faulty"]["supervisors"][s]["n"], d["pooled_faulty"]["supervisors"][s]["survival"], 1.959964
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"95% CI {_pc(c - h)[:-1]}–{_pc(c + h)}"


def _scalar(x: Any) -> float:
    return float(max(x.values())) if isinstance(x, dict) else float(x)


def _signed(x: float, digits: int = 0) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def _of(k: int, n: int) -> str:
    return f"all {n}" if k == n else f"{k} of {n}"


def _rng(vals: list[float], fmt: str = ".0f") -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:{fmt}}" if f"{lo:{fmt}}" == f"{hi:{fmt}}" else f"{lo:{fmt}} to {hi:{fmt}}"


def _need(cond: bool, claim: str) -> None:
    """The fixed prose asserts `claim`. Stop if the result files no longer support it (Rules.md §1)."""
    if not cond:
        raise ValueError(f"paper.py prose is out of date: the results no longer support '{claim}'. Edit the prose.")


def _acted_n(ft: dict, sup: str, fault: str | None = None) -> tuple[int, int]:
    """(faulty flights with a mode change decided after onset, faulty flights) for one supervisor."""
    runs = [r for f, v in ft["faults"].items() if f != "none" and fault in (None, f) for r in v["per_seed"][sup]]
    return sum(r["first_response"] is not None for r in runs), len(runs)


def _delay_tol(ds: dict) -> dict[str, float]:
    """Largest loop delay (ms) each design survives in the delay sweep."""
    return {c: max([x["delay_steps"] * ds["dt"] * 1000 for x in ds[c] if not x["exceeded_limits"]] or [float("nan")])
            for c in ds if c not in ("delays", "dt")}


def _schematic(path: Path) -> None:
    """Sketch of the typical section with the sign conventions of Section 3 (a drawing, not a result)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ea, cg, hinge, beta = -0.3, -0.1, 0.6, math.radians(18)  # semichords aft of mid-chord (uav_flap layout)
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.plot([-1, hinge], [0, 0], lw=6, color="0.3", solid_capstyle="round")
    tip = (hinge + (1 - hinge) * math.cos(beta), -(1 - hinge) * math.sin(beta))
    ax.plot([hinge, tip[0]], [0, tip[1]], lw=6, color="0.3", solid_capstyle="round")
    ax.plot([hinge, 1.05], [0, 0], ls="--", lw=0.8, color="0.4")
    ax.plot(hinge, 0, "o", ms=5, mfc="white", mec="k", zorder=5)
    ax.text(1.08, -0.07, r"$\beta$", fontsize=11)
    zz = np.linspace(0.1, 0.72, 13)  # plunge spring
    ax.plot(ea + 0.05 * np.array([0, *[(-1) ** i for i in range(11)], 0]), zz, color="k", lw=1)
    ax.plot([ea - 0.18, ea + 0.18], [0.72, 0.72], color="k", lw=1.5)
    for x0 in np.linspace(ea - 0.16, ea + 0.16, 6):
        ax.plot([x0, x0 + 0.05], [0.72, 0.79], color="k", lw=0.7)
    ax.text(ea + 0.1, 0.42, r"$K_h$", fontsize=11)
    th = np.linspace(0.5 * math.pi, 3.4 * math.pi, 120)  # torsional spring
    r = 0.04 + 0.014 * th
    ax.plot(ea + r * np.cos(th), 0.0 + r * np.sin(th), color="k", lw=1)
    ax.text(ea - 0.42, 0.14, r"$K_\alpha$", fontsize=11)
    ax.plot(ea, 0, "o", ms=8, mfc="white", mec="k", zorder=6)
    ax.plot(cg, 0, "o", ms=6, color="k", zorder=6)
    ax.plot([0, 0], [-0.07, 0.07], color="k", lw=1)
    ax.annotate("elastic axis", (ea, -0.02), (ea - 0.75, -0.42), fontsize=9, arrowprops={"arrowstyle": "-", "lw": 0.6})
    ax.annotate("c.g.", (cg, -0.02), (cg - 0.12, -0.42), fontsize=9, arrowprops={"arrowstyle": "-", "lw": 0.6})
    ax.annotate("hinge", (hinge, -0.03), (hinge - 0.05, -0.42), fontsize=9, arrowprops={"arrowstyle": "-", "lw": 0.6})

    def span(x0: float, x1: float, y: float, label: str) -> None:
        ax.annotate("", (x0, y), (x1, y), arrowprops={"arrowstyle": "<->", "lw": 0.7})
        ax.text((x0 + x1) / 2, y - 0.02, label, ha="center", va="top", fontsize=10)

    span(-1, 0, -0.62, r"$b$")
    span(0, 1, -0.62, r"$b$")
    span(ea, 0, 0.2, "")
    ax.text(ea / 2 + 0.03, 0.23, r"$a_h b$", ha="center", fontsize=9)
    span(0, hinge, 0.2, "")
    ax.text(hinge / 2, 0.23, r"$c\,b$", ha="center", fontsize=9)
    ax.annotate("", (ea, -0.2), (cg, -0.2), arrowprops={"arrowstyle": "<->", "lw": 0.7})
    ax.text((ea + cg) / 2, -0.23, r"$x_\alpha b$", ha="center", va="top", fontsize=9)
    ax.annotate("", (-1.5, 0), (-1.95, 0), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(-1.95, 0.06, r"$U$", fontsize=11)
    ax.annotate("", (-1.12, -0.32), (-1.12, 0), arrowprops={"arrowstyle": "->", "lw": 1})
    ax.text(-1.3, -0.24, r"$h$", fontsize=11)
    ax.annotate("", (-0.97, 0.3), (-1.05, 0.04), arrowprops={"arrowstyle": "->", "lw": 1,
                                                             "connectionstyle": "arc3,rad=-0.35"})
    ax.text(-1.2, 0.24, r"$\alpha$", fontsize=11)
    ax.set_xlim(-2.05, 1.3)
    ax.set_ylim(-0.82, 0.85)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _copy_figures() -> dict[str, list[str]]:
    """Copy the paper's figures to docs/research/figures; key -> relative paths of the panels that exist."""
    out_dir = RESEARCH / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _schematic(out_dir / "typical_section.png")
    done: dict[str, list[str]] = {"section": ["figures/typical_section.png"]}
    for key, (panels, _) in FIGURES.items():
        for preset, name in panels:
            src = OUTPUTS_DIR / preset / name
            if src.exists():
                dst = out_dir / f"{preset}_{name}"
                shutil.copyfile(src, dst)
                done.setdefault(key, []).append(f"figures/{dst.name}")
    return done


def _front(w: Any, cite: Refs) -> None:
    """Sections 1 and 2: prose that needs no result file."""
    w("## 1. Introduction\n")
    w("Flutter is a dynamic aeroelastic instability. Above a critical airspeed the unsteady aerodynamic forces feed "
      "energy into coupled structural modes, and the oscillation grows until the structure fails. Active flutter "
      f"suppression (AFS) uses control surfaces to extend the stable flight envelope {cite('mukhopadhyay2003', 'livne2018')}. "
      f"It has been demonstrated in wind tunnels with classical, LQG and minimax control laws {cite('mukhopadhyay1999')} "
      f"and with neural networks {cite('scott2000', 'bernellizazzera2000')}, and learned controllers are still being "
      f"studied {cite('mattaboni2009', 'tang2018', 'tang2020', 'chen2023rl')}.\n")
    w("A system certified for the healthy aircraft must also cope with an unhealthy one. An actuator may slow down, "
      "the structure may lose stiffness, sensors may degrade, or the flight computer may become overloaded. Each case "
      "calls for a response: carry on, switch to a more conservative controller, or slow down. The response has to "
      "come in time, and it must not destabilise the loop. Choosing it is the job of the supervisor in a "
      f"fault-tolerant control architecture {cite('zhang2008', 'blanke2016')}.\n")
    w(f"Large language models (LLMs) are now proposed for this supervisory role above fast feedback loops "
      f"{cite('du2026survey')}. Examples include fault-tolerant control of process plants "
      f"{cite('vyas2026ftc', 'vyas2026tutorial', 'papacharalampopoulos2026')}, robot skill selection "
      f"{cite('ahn2022saycan')} and drones, where the latency of sequential token generation is a known obstacle "
      f"{cite('chen2023typefly')}. The survey of Du and Chan {cite('du2026survey')} names the lack of formal stability "
      "guarantees as the central open problem of the field and notes that stability across LLM-triggered mode "
      "switches is largely unanalysed. To the authors' knowledge, no study has evaluated an LLM supervisor for "
      "flutter suppression. None has run the model on board, with its measured latency applied in simulation, "
      "against rule-based and oracle supervisors and against faults the rules were not written for.\n")
    w("This paper asks two questions. Can an on-board LLM choose among pre-verified control modes better than simple "
      "rules when faults develop in flight? If it cannot, what should make that choice? The contributions are:\n")
    w("- An open flutter-suppression benchmark with realistic actuation and fair baselines, checked against "
      "independent analyses and a wind-tunnel flutter point. LQR, LQG and a "
      "differentiable-simulation neural policy minimise one shared cost, the classical designs are tuned for the "
      "actual actuator, and robustness and ablation studies are reported.")
    w("- A supervisory architecture above a fast, open-loop-unstable plant: a bounded mode set, a controller bank "
      "with bumpless transfer, and a discrete-time average dwell-time bound computed for the implemented "
      "controllers. For the healthy wing at a fixed airspeed, they keep the loop small-signal stable under any sequence of "
      "controller switches that the supervisor (rules, LLM or "
      "learned) requests.")
    w("- An evaluation of an on-board LLM as the decision-maker, with its measured latency, against no, random, "
      "rule-based and oracle supervision on two disjoint test sets, with and without an on-board flutter-margin "
      "estimate.")
    w("- A guardian supervisor that combines a learned fault diagnoser, which includes an unknown-fault score, with "
      "a physics-based admissibility check on flutter margin and control authority. It was designed on development "
      "seeds and tested once on fresh seeds, including two fault types that its diagnoser and the rules never saw.")
    w("- An evaluation of the LLM off the control path: explaining decisions to the operator under a faithfulness "
      "check, and proposing post-flight model updates behind a verification gate.\n")
    w("Negative results are reported as such.\n")

    w("## 2. Related work\n")
    w(f"*Neural flutter suppression.* Scott and Pado {cite('scott2000')} used a neural network to schedule the "
      "coefficients of flutter-suppression control laws over Mach number and dynamic pressure on the NASA Benchmark "
      "Active Controls Technology (BACT) wing in the Transonic Dynamics Tunnel, where the classical systems proved "
      "more robust than the neural ones. Recurrent neural networks have been applied to AFS of a typical section "
      f"{cite('bernellizazzera2000')} and of a three-surface transport aircraft {cite('mattaboni2009')}. Neural control "
      f"has handled control-input constraints {cite('tang2018', 'tang2020')}, and deep reinforcement learning has been "
      f"trained directly in a wind tunnel {cite('chen2023rl')}. The neural policy used here is trained offline through "
      f"a differentiable model {cite('xu2022shac', 'heeg2024')}. It is a learned baseline and no novelty is claimed "
      "for it.\n")
    w(f"*LLM supervision.* The closest work is a three-layer fault-tolerant scheme for process systems "
      f"{cite('papacharalampopoulos2026')}. A cloud LLM is called once per fault event and selects one of four bounded "
      "actions over pre-verified controllers. The scheme has a rule-based fallback and an average dwell-time bound "
      "on switching, and it is compared with a rule-based supervisor. "
      "The present architecture shares these elements and differs in three "
      "respects. The plant is an aeroelastic system that is unstable without control and faster than the LLM's "
      "latency. The model runs on board, is queried repeatedly, and has its measured latency applied in simulation "
      "time. The evaluation adds random, rule-based and oracle supervisors, paired statistics over seeds, and faults "
      f"described neither in the rules nor in the prompt. LLM agents that propose validated recovery actions "
      f"{cite('vyas2026ftc')} and LLM planning above a safety filter {cite('shibu2026skysim')} use the same separation "
      "between a slow language layer and a fast deterministic layer.\n")
    w(f"*Switching stability.* Switching among stable linear systems preserves stability if the switches are slow "
      f"enough on average {cite('hespanha1999', 'liberzon2003')}. The original result is in continuous time, and "
      "Section 5 gives the discrete-time counterpart used here. LQG designs have no guaranteed robustness margins "
      f"{cite('doyle1978')}, as the results of Section 8.2 show.\n")
    w(f"*Fault diagnosis and flutter-margin monitoring.* Model-based fault detection and isolation "
      f"{cite('blanke2016', 'isermann2006')} and reconfigurable fault-tolerant control {cite('zhang2008')} are mature "
      "fields. In flight testing, the flutter margin is predicted from subcritical measurements, for example with "
      f"the Zimmerman and Weissenburger criterion {cite('zimmerman1964')}. The guardian proposed here combines a "
      "learned diagnoser with an on-line grey-box estimate of flutter margin and control authority. Diagnosis that "
      "triggers a switch among pre-designed configurations has also been applied to launch vehicles, where a "
      "multi-model observer identifies actuator and thrust-drop faults and selects an actuator group while the "
      f"baseline control law is kept {cite('yu2026ijass')}. Only the combination used here, and its evaluation "
      "as a mode supervisor of an aeroelastic system, is claimed as new.\n")


def build(out_path: Path | None = None, copy_figures: bool = True, cite_markers: bool = False) -> Path:
    out_path = Path(out_path or RESEARCH / "paper-draft.md")
    up = OUTPUTS_DIR / "uav_flap"
    res = {k: _load(OUTPUTS_DIR / k / "results.json") for k in PRESET_ORDER}
    rob = {k: _load(OUTPUTS_DIR / k / "robustness.json") for k in PRESET_ORDER}
    tun = {k: _load(MODELS_DIR / f"classical_tuning_{k}.json") for k in PRESET_ORDER}
    abl, llm, ft, stab, phys, guard, expl, rev, diag, nolat = (_load(up / f"{n}.json") for n in (
        "ablations", "llm_benchmark", "ft_benchmark", "ft_stability", "ft_benchmark_physics_test100",
        "ft_benchmark_guardian_test300", "explainer_eval", "review_loop", "diagnoser_eval",
        "ft_benchmark_no_latency_2s"))
    bact = _load(OUTPUTS_DIR / "validation" / "bact.json")
    cite = Refs(_bib(RESEARCH / "references.bib"), marker=cite_markers)
    L: list[str] = []
    w = L.append
    fig_no = {k: i for i, k in enumerate(FIGURES, start=1)}
    tab_no = {k: i for i, k in enumerate(TABLES, start=1)}
    figs = _copy_figures() if copy_figures else {}

    def F(k: str) -> str:
        return f"Fig. {fig_no[k]}"

    def T(k: str) -> str:
        return f"Table {tab_no[k]}"

    def figure(k: str) -> None:
        if k in figs:
            w(f"**Figure {fig_no[k]}.** {FIGURES[k][1]}\n")
            w("\n".join(f"![Figure {fig_no[k]}]({path})" for path in figs[k]) + "\n")

    def table(k: str, caption: str, header: list[str], rows: list[list[str]]) -> None:
        w(f"**Table {tab_no[k]}.** {caption}\n")
        w("| " + " | ".join(header) + " |")
        w("|---" * len(header) + "|")
        for row in rows:
            w("| " + " | ".join(row) + " |")
        w("")

    w(f"# {TITLE}\n")
    w(", ".join(f"{a} ({aff})" for a, aff in zip(AUTHORS, AFFILIATIONS, strict=True))
      + f". Corresponding author: {CORRESPONDING[0]}, {CORRESPONDING[1]}\n")
    note_at = len(L)
    w("")

    required = {"evaluate": all(res.values()), "robustness": all(rob.values()), "tune": all(tun.values()),
                "ablations": abl, "llm-benchmark": llm, "ft-benchmark (sets A, B and C)": ft and phys and guard,
                "ft-stability": stab, "explainer-eval": expl, "review-loop": rev, "validate": bact,
                "eval-diagnoser": diag}
    missing = [k for k, v in required.items() if not v]
    if missing:  # e.g. a fresh clone: outputs/ is not in git
        L[note_at] = (f"*Results pending. Run these `python -m flutter_ai` commands first: {', '.join(missing)}.*\n")
        w("## Abstract\n")
        w("*Results pending.*\n")
        _front(w, cite)
        w("## 8. Results\n")
        w("### 8.4 Fault-tolerant supervision\n")
        w("*Results pending.*\n")
        w("## References\n")
        L.extend(cite.listing())
        out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
        return out_path

    from .fault_tolerance.faults import FAULTS
    from .fault_tolerance.simulate import FT_MODES, MIN_DWELL_S
    from .flutter import flutter_analysis, pk_flutter_speed

    p = PRESETS["uav_flap"]
    assert p.u_rate_max is not None
    rate = math.degrees(p.u_rate_max[0])
    dt_ms = 1000 * res["uav_flap"]["delay_sweep"]["dt"]
    seeds = len(res["uav_flap"]["metadata"]["seeds"])
    sets = {"A": ft, "B": phys, "C": guard}

    def pool(d: dict, s: str) -> float:
        return float(d["pooled_faulty"]["supervisors"][s]["survival"])

    def pair(d: dict, label: str) -> dict:
        return d["pooled_faulty"]["paired"][label]

    def seed_span(d: dict) -> str:
        return f"{d['metadata']['seeds'][0]} to {d['metadata']['seeds'][-1]}"

    family = [(k, lab) for k, d in sets.items() for lab in d["pooled_faulty"]["paired"]]
    holm = dict(zip(family, _holm([pair(sets[k], lab)["mcnemar_p"] for k, lab in family]), strict=True))
    combos = [(k, s, d) for k in PRESET_ORDER for s, d in res[k]["scenarios"].items()]
    lqg_best = sum(_best_by_J(d)[0] == "lqg" for _, _, d in combos)
    lqr_wins = {k: _neural_vs(res[k], "lqr") for k in PRESET_ORDER}
    lqg_wins = sum(_neural_vs(res[k], "lqg")[0] for k in PRESET_ORDER)
    lat_meds = [llm["summary"]["latency_s"]["median"], ft["llm_latency_s"]["median"], phys["llm_latency_s"]["median"]]
    llm_rel = max(abs(d["statistics"]["neural+llm vs neural"]["cost_J"]["rel_diff"]) for d in llm["scenarios"].values())
    held = [f for f in guard["faults"] if guard["faults"][f]["held_out"]]
    faith = expl["summary"]["faithfulness_rate"]
    _need(llm_rel < 0.01, "LLM effort supervision changes the cost by less than 1%")
    _need(_verdict(phys, "rules+physics", "rules") == "helped" and _verdict(phys, "llm+physics", "llm") == "hurt",
          "the flutter-margin estimate helped the rules and hurt the LLM")
    _need(pool(ft, "llm") < pool(ft, "rules") and pair(ft, "llm vs rules")["mcnemar_p"] < 0.05,
          "the LLM kept fewer faulty flights than the rules")
    _need(all(_surv(guard, f, "guardian") == 1.0 for f in held), "the guardian saved every held-out flight")

    # ------------------------------------------------------------ abstract
    w("## Abstract\n")
    w("**Purpose:** Large language models (LLMs) are proposed as supervisors of fast feedback loops but have not been "
      "tested choosing a flutter-suppressed wing's response to an in-flight fault. This paper tests an "
      "on-board LLM in that role against rule-based, oracle, and fast learned and physics-based supervisors. "
      f"**Methods:** Three typical-section wings with a servo-driven flap, a {rate:.0f}°/s slew-rate "
      f"limit and a {dt_ms:.0f} ms loop delay were checked against independent flutter analyses, a textbook case and "
      "wind-tunnel data. A linear-quadratic regulator (LQR), a turbulence-aware linear-quadratic-Gaussian "
      "(LQG) controller and a neural policy trained through differentiable simulation stabilise them under one shared cost. Supervisors choose only pre-verified modes, and "
      "a dwell-time bound keeps the healthy wing's switched loop small-signal stable. "
      f"**Results:** LQG had the lowest cost in {_of(lqg_best, len(combos))} combinations. A local "
      f"8-billion-parameter LLM answered in {_rng(lat_meds, '.1f')} s, too slow for gusts, and changed the cost by less "
      f"than 1%. Under six slow faults it saved {_pc(pool(ft, 'llm'))} of faulty flights and "
      f"hand-written rules {_pc(pool(ft, 'rules'))} (Holm-adjusted p = {holm['A', 'llm vs rules']:.2g}). "
      f"An on-board flutter-margin estimate helped the rules ({_pc(pool(phys, 'rules+physics'))} against "
      f"{_pc(pool(phys, 'rules'))}) and hurt the LLM ({_pc(pool(phys, 'llm+physics'))} against {_pc(pool(phys, 'llm'))}). "
      "A guardian combining a learned fault diagnoser with a physics-based admissibility check kept "
      f"{_pc(pool(guard, 'guardian'))} on fresh seeds, including both fault types unseen by its diagnoser, while the rules kept {_pc(pool(guard, 'rules'))} (Holm-adjusted {_p(holm['C', 'guardian vs rules'])}). "
      "**Conclusion:** For this plant and the local model tested, mode decisions belong to fast, verifiable models. The "
      f"LLM is better used off the control path, where {_pc(faith)} of its operator explanations passed a faithfulness check.\n")
    w(f"**Keywords:** {KEYWORDS}\n")

    _front(w, cite)

    # ------------------------------------------------------------ model
    hp = PRESETS["hodges_pierce"]
    fa = flutter_analysis(hp)
    u_ref = hp.b * hp.omega_alpha
    mass_ratio = hp.mass / (math.pi * hp.rho * hp.b ** 2)
    ud_exact = math.sqrt(mass_ratio * hp.r_alpha_sq / (1 + 2 * hp.a))
    hp_pk = pk_flutter_speed(hp)
    assert hp_pk and fa.flutter_speed and fa.divergence_speed and fa.flutter_frequency_hz
    _need(PRESETS["uav_section"].actuation == "force_moment", "uav_section is the plain 2-DOF section")
    w("## 3. Aeroelastic models and validation\n")
    w("### 3.1 Two-degree-of-freedom typical section\n")
    w("The structural model is the two-degree-of-freedom typical section of the standard aeroelasticity texts "
      f"{cite('fung1955', 'bisplinghoff1955', 'hodges2011')}: a rigid airfoil of semichord $b$ and mass $m$ per unit "
      r"span, held at its elastic axis by a plunge spring $K_h$ and a torsional spring $K_\alpha$ "
      f"({F('section')}). "
      r"The elastic axis lies $a_h b$ aft of mid-chord and the centre of gravity $x_\alpha b$ aft of the elastic "
      r"axis. With the plunge $h$ positive downward and the pitch $\alpha$ positive nose-up, the equations of "
      "motion are\n")
    w(r"$$m\ddot h + S_\alpha\ddot\alpha + C_h\dot h + K_h h = -L, \tag{1}$$" + "\n")
    w(r"$$S_\alpha\ddot h + I_\alpha\ddot\alpha + C_\alpha\dot\alpha + K_\alpha\alpha = M_\alpha, \tag{2}$$" + "\n")
    w(r"where $S_\alpha = m x_\alpha b$ is the static unbalance, $I_\alpha = m r_\alpha^2 b^2$ the moment of inertia "
      r"about the elastic axis, $K_h = m\omega_h^2$ and $K_\alpha = I_\alpha\omega_\alpha^2$. The damping terms "
      rf"$C_h$ and $C_\alpha$ give each uncoupled mode a damping ratio of {p.zeta_h:.2f} on the UAV wings. "
      r"For incompressible flow at airspeed $U$ and density $\rho$, Theodorsen's lift and pitching moment about the "
      rf"elastic axis for harmonic motion at the reduced frequency $k = \omega b/U$ are {cite('theodorsen1935')}" + "\n")
    w(r"$$L = \pi\rho b^2\left(\ddot h + U\dot\alpha - b a_h\ddot\alpha\right) + 2\pi\rho U b\,C(k)\,w, \tag{3}$$" + "\n")
    w(r"$$M_\alpha = \pi\rho b^2\left[b a_h\ddot h - U b\left(\tfrac12 - a_h\right)\dot\alpha - b^2\left(\tfrac18 + "
      r"a_h^2\right)\ddot\alpha\right] + 2\pi\rho U b^2\left(a_h + \tfrac12\right)C(k)\,w, \tag{4}$$" + "\n")
    w(r"where $w = \dot h + U\alpha + b\left(\tfrac12 - a_h\right)\dot\alpha$ is the downwash at the three-quarter "
      r"chord and $C(k)$ is Theodorsen's function. The first term of each load is the apparent-mass part and the "
      r"second the circulatory part. In the time domain the circulatory loads use Wagner's indicial function, the counterpart of $C(k)$, in the "
      r"two-term exponential "
      rf"approximation due to Jones {cite('jones1940')}, which adds two aerodynamic lag "
      f"states. The model is discretised exactly with a zero-order hold at {1000 / dt_ms:.0f} Hz.\n")
    figure("section")
    w("### 3.2 Flap, actuator and loop delay\n")
    w(r"Control acts through a trailing-edge flap. It adds a third degree of freedom, the deflection $\beta$ "
      r"(positive trailing edge down) about a hinge $c\,b$ aft of mid-chord. Theodorsen's flap terms "
      f"{cite('theodorsen1935')} enter the lift and both moments, and a hinge-moment equation joins Eqs. (1) and (2). "
      f"The flap is driven by a servo modelled as a second-order system with a natural frequency of {p.f_beta:.0f} Hz "
      f"and a damping ratio of {p.zeta_beta:.2f}, the damping measured on the BACT actuator {cite('waszak1998bact')}. "
      f"The servo has a ±{math.degrees(p.u_max[0]):.0f}° command limit and a {rate:.0f}°/s slew-rate limit taken from "
      f"a UAV servo datasheet {cite('dsw001a')}. Every command reaches the actuator one sample ({dt_ms:.0f} ms) later.\n")
    w("### 3.3 Wings and validation\n")
    m = bact["reference"]["measured"]
    pr = next(iter(bact["predictions"].values()))
    p2 = next(v for k, v in bact["predictions"].items() if k.startswith("plain 2D"))
    nasa = bact["reference"]["nasa_model"]
    bact_alt = (f"Plain two-dimensional Theodorsen aerodynamics give {p2['flutter_q_psf']:.0f} psf "
                f"({_signed(100 * (p2['flutter_q_psf'] - m['flutter_q_psf']) / m['flutter_q_psf'])}%) at "
                f"{p2['flutter_freq_hz']:.2f} Hz "
                f"({_signed(100 * (p2['flutter_freq_hz'] - m['flutter_freq_hz']) / m['flutter_freq_hz'])}%), and "
                f"NASA's model with measured and computed aerodynamic derivatives gives {nasa['flutter_q_psf']:g} psf "
                f"{cite('waszak1998bact')}. ")
    rows: list[list[str]] = []
    diffs: list[float] = []
    for key in PRESET_ORDER:
        f = res[key]["flutter"]
        diffs.append(100 * (f["state_space"] - f["pk_exact_theodorsen"]) / f["pk_exact_theodorsen"])
        rows.append([f"`{key}`", f"{f['state_space']:.2f} m/s", f"{f['frequency_hz']:.2f} Hz",
                     f"{f['pk_exact_theodorsen']:.2f} m/s", _signed(diffs[-1], 2) + "%"])
    diffs.append(100 * (fa.flutter_speed - hp_pk) / hp_pk)
    rows.append([r"Textbook section, $U/b\omega_\alpha$", f"{fa.flutter_speed / u_ref:.3f}",
                 rf"{2 * math.pi * fa.flutter_frequency_hz / hp.omega_alpha:.3f} $\omega_\alpha$",
                 f"{hp_pk / u_ref:.3f}", _signed(diffs[-1], 2) + "%"])
    w("Three wings are studied: two UAV-scale sections, `uav_flap` and `uav_section`, and the NASA BACT wind-tunnel "
      f"model, `bact` {cite('waszak1998bact')}. `uav_section` is the plain section of Section 3.1 with ideal force "
      "and moment inputs, and the other two carry the flap. BACT has a finite span, so its circulatory terms use "
      "the measured lift-curve slope and aerodynamic centre in place of the thin-airfoil values.\n")
    w(f"The two-degree-of-freedom model was first run on the textbook section of Hodges and Pierce {cite('hodges2011')} "
      rf"($a_h = {hp.a:g}$, $x_\alpha = {hp.x_alpha:g}$, $r_\alpha^2 = {hp.r_alpha_sq:g}$, mass ratio "
      rf"$m/\pi\rho b^2 = {mass_ratio:.0f}$, $\omega_h/\omega_\alpha = {2 * math.pi * hp.f_h / hp.omega_alpha:g}$). "
      rf"It diverges at $U/b\omega_\alpha = {fa.divergence_speed / u_ref:.3f}$, the closed-form static value being "
      rf"{ud_exact:.3f}, and it flutters at {fa.flutter_speed / u_ref:.3f}. "
      f"{T('valid')} compares the flutter points of all state-space models with an independent p-k analysis that "
      f"uses the exact Theodorsen function. The flutter speeds agree within {max(abs(x) for x in diffs):.1f}%. For "
      f"BACT, the predicted flutter dynamic pressure is {pr['flutter_q_psf']:.0f} psf at {pr['flutter_freq_hz']:.2f} Hz "
      f"and the tunnel measured about {m['flutter_q_psf']:.0f} psf at {m['flutter_freq_hz']:.0f} Hz: "
      f"{100 * (pr['flutter_q_psf'] - m['flutter_q_psf']) / m['flutter_q_psf']:.0f}% high in pressure and within "
      f"{abs(100 * (pr['flutter_freq_hz'] - m['flutter_freq_hz']) / m['flutter_freq_hz']):.1f}% in frequency. " + bact_alt + "The "
      f"pressure error is not resolved: the test was run at Mach {m['mach']} and the model is "
      f"incompressible, and the finite-span corrections that bring the frequency close raise the pressure. {F('vg')} shows the open-loop damping and frequency of `uav_flap` and `bact` against airspeed.\n")
    table("valid", "Flutter points of the state-space models against an independent p-k analysis with the exact "
          "Theodorsen function", ["Wing", "U_F, state space", "Flutter frequency", "U_F, p-k", "Difference"], rows)
    figure("vg")

    # ------------------------------------------------------------ controllers
    w("## 4. Inner-loop controllers and fair tuning\n")
    w("All controllers minimise the same cost,\n")
    w(r"$$J = \sum_k \left( x_k^{\mathsf T} Q x_k + c(a)\,u_k^{\mathsf T} R u_k + \Delta u_k^{\mathsf T} W \Delta u_k "
      r"\right), \tag{5}$$" + "\n")
    w(r"where $Q$ and $R$ are Bryson-normalised, $W$ penalises changes of the command, and the effort weight $c(a)$ "
      r"is set by an aggressiveness parameter $a \in [0, 1]$. All controllers see the same measurements and are "
      "subject to the same loop delay and actuator limits.\n")
    w("*LQR.* A discrete-time regulator is designed for the loop delay, with the command history in the state, and "
      "for the rate penalty through a cross term. An on-board observer supplies the aerodynamic lag states.\n")
    w("*LQG.* The same regulator is designed on the plant augmented with a first-order autoregressive (AR(1)) "
      "turbulence state. A steady-state Kalman filter estimates the lag states and the gust, and its input model "
      "includes the actuator's known slew-rate limit. Its gust model has the correlation time and intensity of the "
      "simulated turbulence, so LQG knows the disturbance statistics; a mismatched spectrum was not tested.\n")
    w(r"*Neural policy.* A 64 × 64 tanh network receives the state estimate, the previous command, the airspeed "
      r"ratio and $a$. Its output is $u_{\max}\tanh\left(g(x) - g(0)\right)$, so the command is zero at equilibrium. "
      "The network is first trained to imitate LQR and then fine-tuned through the differentiable, rate-limited "
      f"plant with randomised loop delay, airspeed and turbulence {cite('xu2022shac', 'heeg2024')} ({F('training')}).\n")
    w("*Fairness.* The free parameters of the classical designs are tuned for the actual actuator on episodes whose "
      "seeds never appear in the test set. They are an extra rate-penalty multiplier and, for LQG, the sensor noise "
      f"assumed by the Kalman filter ({T('tuning')}).\n")
    table("tuning", "Tuned parameters of the classical controllers (tuning seeds disjoint from the test seeds)",
          ["Wing", "LQR rate multiplier", "LQG rate multiplier", "LQG assumed sensor noise"],
          [[f"`{k}`", f"{tun[k]['lqr']['rate_multiplier']:g}", f"{tun[k]['lqg']['rate_multiplier']:g}",
            f"{tun[k]['lqg']['measurement_noise']:g}"] for k in PRESET_ORDER])
    figure("training")

    # ------------------------------------------------------------ supervision
    ctrl_name = {"lqg": "LQG", "lqr": "LQR", "neural": "the neural policy"}
    modes = "; ".join(f"`{name}`, {ctrl_name[spec['controller']]} at "
                      + ("cruise" if spec["speed_ratio"] is None else f"{spec['speed_ratio']:.1f} U_F")
                      for name, spec in FT_MODES.items())
    rhos = max(max(r_["spectral_radius"].values()) for r_ in stab["speeds"])
    worst = max(stab["speeds"], key=lambda r_: r_["dwell_s"])
    _need(MIN_DWELL_S > stab["max_dwell_s"], "the enforced dwell time exceeds the computed bound")
    w("## 5. Supervisory architecture\n")
    w("At a fixed supervision interval, a supervisor (rules, the LLM, or the guardian of Section 6) returns one mode "
      "name from a fixed enumeration, using only on-board information. It never outputs gains, forces or speeds. For "
      f"the fault-tolerant studies on `uav_flap`, where cruise is 1.1 U_F, the menu is: {modes}.\n")
    w("*LLM interface.* The LLM receives a plain-language status report. Its answer is constrained by a JSON schema "
      "and validated again in code. On any failure (a timeout, an invalid answer, a refusal or an exhausted call "
      f"budget) the rule-based decision replaces it. The model (`{_llm_model(ft, llm)}`, quantised, served by Ollama) "
      "runs locally, so it needs no data link. It is asked one question at a time, and its measured latency is "
      "applied in simulation time.\n")
    w("*Controller bank.* All candidate controllers run at every step on the same measurements and the same "
      "applied-command history. Switching between them therefore causes no transient (bumpless transfer), and all "
      "modes share one state.\n")
    w(r"*Stability of switching.* On the shared state $\xi$, each inner-loop controller closes a linear loop "
      r"$\xi_{k+1} = A_i\xi_k$. Every $A_i$ is Schur stable at every airspeed of the schedule, and the largest "
      f"spectral radius is {rhos:.4f}.\n")
    w(r"**Proposition 1.** Let $P_i \succ 0$ solve the discrete Lyapunov equation of $A_i$ and let "
      r"$V_i(\xi) = \xi^{\mathsf T} P_i \xi$. Define $\lambda < 1$ and $\mu \ge 1$ such that "
      r"$V_i(A_i\xi) \le \lambda V_i(\xi)$ and $V_j(\xi) \le \mu V_i(\xi)$ for all modes $i$, $j$ and all $\xi$. "
      r"If, over any interval of $k$ steps, the number of switches $N$ satisfies $N \le N_0 + k/\tau_a$ with the "
      "average dwell time\n")
    w(r"$$\tau_a > \tau_a^* = \frac{\ln\mu}{-\ln\lambda}\ \text{steps}, \tag{6}$$" + "\n")
    w("then the switched nominal closed loop is exponentially stable for every switching sequence.\n")
    w(r"*Proof.* Between switches $V$ falls by at least the factor $\lambda$ per step, and at a switch it grows by "
      r"at most the factor $\mu$. After $k$ steps with $N$ switches, "
      r"$V(k) \le \mu^N\lambda^k V(0) \le \mu^{N_0}\left(\mu^{1/\tau_a}\lambda\right)^k V(0)$, and "
      r"$\mu^{1/\tau_a}\lambda < 1$ exactly when Eq. (6) holds. This is the discrete-time counterpart of the "
      rf"continuous-time result of {cite('hespanha1999')}. $\square$" + "\n")
    w(rf"In scaled coordinates, the bound computed for the implemented controllers is $\tau_a^* \le$ "
      f"{stab['max_dwell_s']:.2f} s. The architecture enforces at least {MIN_DWELL_S:.0f} s between controller "
      "switches, so no sequence of controller switches can destabilise the nominal closed loop. The guarantee is "
      "small-signal and applies to the healthy wing at a fixed airspeed. It excludes faults, saturation and the "
      f"slew-rate limit. At the worst airspeed of the schedule, λ = {_scalar(worst['lambda']):.4f} and "
      f"μ = {_scalar(worst['mu']):.2f}. Three of the four modes use the same LQG controller and differ only in "
      "their airspeed target, so in practice the bound constrains switches between LQG and the neural policy. "
      "Airspeed changes are slow, deceleration-limited, gain-scheduled transitions, so mode changes that keep the "
      "controller and only move the airspeed target are not delayed. Automated tests verify that the analysis "
      "matrices reproduce the simulators step by step.\n")

    # ------------------------------------------------------------ supervisors
    g_ = rev["metadata"]["gate"]
    w("## 6. Supervisors\n")
    w("*Baselines.* `none` never changes mode and `random` picks a mode at random. `rules` maps symptoms in the "
      "status report, such as oscillation growth, saturation and noise level, to modes through hand-written "
      "thresholds. `oracle` knows the injected fault and, with hindsight, selects the least disruptive mode that "
      "survives it (Section 8.4). It is an upper bound and cannot be implemented. `llm` receives the same symptom "
      "report with written guidance on what the modes mean. The rules and the guidance were written from the same "
      "principles and were not tuned on benchmark outcomes.\n")
    w("*On-board flutter-margin estimate.* Every second, a grey-box fit over the last 4 s of flight identifies the "
      "wing's pitch stiffness and flap effectiveness relative to the same fit at take-off, which absorbs the bias "
      "caused by the unmeasured turbulence. From the identified wing the estimator predicts the flutter speed and "
      "whether each mode's controller keeps the wing stable. In `rules+physics` the symptom rules decide whether to "
      "act and the estimate decides which modes are safe. A mode predicted to be unstable on the identified wing is "
      "never chosen. If the current mode is predicted unstable, the least disruptive predicted-stable mode is "
      "taken: hold speed if possible, otherwise the smallest slow-down. In `llm+physics` the LLM receives the same "
      "estimate as a \"physics estimate\" block in its report.\n")
    w("*Guardian.* The guardian keeps the same mode menu, dwell time and fallback, and it uses no language model. "
      "Every 0.25 s it reads three sources. The first is a learned fault diagnoser: a small encoder over physics "
      "features of the last 2.5 s, with a multi-label head for the four known faults and a reconstruction-error "
      "score for unknown faults. It is trained on simulated flights that never contain the two held-out faults. The "
      "second is the flutter-margin estimate extended with control authority. The predicted flap command and flap "
      "rate are compared with the actuator limits, using estimates of the turbulence, sensor noise and loop delay "
      "and worst-case margins over the confidence region of the identification. The third is the set of symptom "
      "rules, kept as a fallback.\n")
    w("A confirmed known fault receives its pre-verified response and an unknown fault the most conservative mode. "
      "The chosen mode is the first of that response's preferences that the physics check admits. A mode is never "
      "chosen if it is predicted to be unstable, or to saturate where the identified wing needs control. Structural "
      "evidence holds speed unless the physics predicts the current mode to be unstable, and the guardian never "
      "de-escalates on its own. Its thresholds were set on development seeds (1000 to 1009), and it was evaluated "
      "once on fresh seeds after its design was frozen. For comparison, `rules+physics_v2` combines the symptom "
      "rules with the same extended physics estimate and no learned diagnoser.\n")
    w("*LLM off the control path.* Two further roles use the same safety wrapper: a schema-constrained JSON answer "
      "re-checked in code, a call budget and a timeout. In neither can the LLM change a decision.\n")
    w("As an explainer, the LLM reads the report (including the diagnosis and physics blocks) and the decision "
      "taken, then writes one or two sentences for the operator together with the fault and mode it refers to. The "
      "answer is shown only if it passes a faithfulness check. Its mode must be the decision taken, its fault must "
      "be the diagnosis's top flagged class (or \"unknown\" or \"none\"), its length must be 20 to 400 characters, "
      "and it may name no fault that the diagnosis does not flag. Otherwise a deterministic template is shown. The "
      "check covers these fields and literal fault names. Other claims in the free text are not checked.\n")
    w("As a post-flight reviewer, the LLM reads a compact log of a development flight in which the diagnoser raised "
      "a confirmed unknown-fault alarm, and proposes a fault from the catalogue with bounded severity and growth "
      "ranges. Flights are simulated for the hypothesis and a candidate diagnoser is trained with it as an extra "
      f"class. The candidate is accepted only if two conditions hold. First, no existing class loses more than "
      f"{g_['f1_tolerance']} in window F1 on the original evaluation split, and the healthy-flight alarm rate rises "
      f"by at most {g_['false_alarm_tolerance']}. Second, on the anomalous flights the hypothesised class is "
      f"confirmed in at least {_pc(g_['explain_class_min'])} of the windows after the first unknown alarm, and the "
      f"unknown alarm appears in at most {_pc(g_['explain_unknown_max'])} of them. The thresholds were fixed before "
      "any candidate was scored, and hypotheses forced to be wrong serve as controls.\n")

    # ------------------------------------------------------------ experimental design
    sc = res["uav_flap"]["scenarios"]
    lat_fixed = phys["metadata"]["llm_latency_fixed_s"]
    held_txt = " and ".join(f"`{f}`" for f in held)
    w("## 7. Experimental design\n")
    w("*Controller benchmark.* Four scenarios were run on each wing: a 4 s turbulence patch at 1.1 U_F, a randomised "
      "1-cosine (discrete) gust at 1.1 U_F, the same gust at 1.3 U_F, and the same gust at 0.9 U_F, below flutter. "
      f"Gust amplitude, length and onset and the initial condition are randomised per seed, with {seeds} seeds per "
      f"scenario. Disturbance sizes for `uav_flap` are those a {rate:.0f}°/s servo can handle "
      f"({sc['turbulence_patch']['scenario']['gust']['amplitude']:.2f} m/s RMS turbulence). A run fails if a "
      "structural limit is exceeded.\n")
    w(f"*Fault-tolerant flights.* Each flight lasts {ft['metadata']['duration_s']:.0f} s at 1.1 U_F in light "
      f"turbulence on `uav_flap`. A fault starts 6 s into the flight and grows over 5 s ({T('faults')}). Two faults, "
      f"{held_txt}, are held out: they appear in neither the rules, the LLM's guidance nor "
      "the diagnoser's training data, and no response was designed for them. The physics estimate is not blind "
      "to them, since it identifies flap effectiveness and reads the measured loop delay, and a conservative "
      "response survives both (Section 8.4). A flight survives if it never exceeds a structural limit. "
      "Fault-free flights measure false alarms. Every design choice was made on "
      "development seeds numbered from 1000 upwards. Three disjoint test sets were each used once. Set A (seeds "
      f"{seed_span(ft)}) compares the rules and the LLM. Set B (seeds {seed_span(phys)}) tests the physics-informed "
      "supervisors, with the comparison, supervisors and seeds fixed before the run. Set C (seeds "
      f"{seed_span(guard)}) tests the guardian against the non-LLM supervisors; the LLM was not run on it.\n")
    table("faults", "Fault catalogue (onset at 6 s, growth over 5 s)", ["Fault", "Description"],
          [[f"`{f}`" + (" (held out)" if spec["held_out"] else ""), spec["description"]]
           for f, spec in FAULTS.items() if f != "none"])
    w("*LLM timing.* In set A the measured latency of each LLM answer was applied in simulation time. In set B the "
      f"answers take effect after a fixed {lat_fixed:.2f} s of simulation time, the median latency measured in set "
      "A, so that the outcome does not depend on the load of the machine.\n")
    w(f"*Statistics.* Controller costs are compared with paired Wilcoxon signed-rank tests {cite('wilcoxon1945')} "
      f"over seeds, with bootstrap 95% confidence intervals of the paired differences {cite('efron1979')}. Survival "
      f"is compared with the exact McNemar test {cite('mcnemar1947')} on the discordant pairs of faulty flights (the "
      f"same seed and fault under two supervisors). Because {len(family)} survival comparisons are reported, "
      f"Holm-adjusted p values {cite('holm1979')} over the whole family are given as well.\n")
    w(f"*Use of AI tools.* Apart from the language model under study (`{_llm_model(ft, llm)}`, Sections 5 and 6), "
      "an AI coding assistant (Claude, Anthropic) was used under the authors' direction throughout the project: to "
      "write and test much of the simulation and analysis code, including the scripts that compute every number "
      "and table in this paper from the result files, and to draft and edit the text. The first author built the "
      "initial model and code, defined the research questions and the studies, and reviewed all code and "
      "results. Both authors revised the text, checked all content and take full responsibility for it.\n")

    # ------------------------------------------------------------ results: controllers
    w("## 8. Results\n")
    w("### 8.1 Inner-loop controllers\n")
    nl = [d["statistics"]["neural vs lqg"]["cost_J"] for _, _, d in combos]
    _need(all(x["rel_diff"] > 0 and x["wilcoxon_p"] < 1e-3 for x in nl), "the neural policy costs more than LQG, p < 0.001")
    for k in ("uav_flap", "uav_section"):
        c_ = res[k]["scenarios"]["below_flutter"]["cases"]
        _need(c_["lqr"]["cost_J"]["mean"] > c_["open_loop"]["cost_J"]["mean"], f"LQR amplifies gusts below flutter on {k}")
    bs = res["bact"]["scenarios"]
    bact_rms = sum(min(("lqr", "lqg", "neural"), key=lambda c: bs[s]["cases"][c]["rms_h_mm"]["mean"]) == "neural"
                   for s in bs if s != "turbulence_patch")
    bt = bs["turbulence_patch"]["cases"]["neural"]
    tol = {k: _delay_tol(res[k]["delay_sweep"]) for k in PRESET_ORDER}
    _need(tol["uav_flap"]["neural"] > tol["uav_flap"]["lqg_nominal"], "neural outlasts the fixed designs on uav_flap")
    cap = res["uav_flap"]["delay_sweep"]["delays"][-1] * dt_ms
    _need(all(tol[k][c] >= cap for k in tol for c in ("lqr", "lqg")), "the re-designed classical controllers reach the end of the sweep")
    w(f"{T('bench')} summarises the controller benchmark and {F('time')} shows one turbulence run. All open-loop runs "
      f"above flutter fail. LQG has the lowest cost $J$ without failures in {_of(lqg_best, len(combos))} wing and "
      "scenario combinations, and its difference from the neural policy is significant in every case (Wilcoxon "
      f"p < 0.001 over the seeds both survived). The neural policy costs {_rng([100 * x['rel_diff'] for x in nl])}% "
      f"more than LQG. It beats plain LQR in {sum(a for a, _ in lqr_wins.values())} of the {len(combos)} "
      "combinations: " + ", ".join(f"{a} of {n} on `{k}`" for k, (a, n) in lqr_wins.items()) + ". Plain LQR has no "
      "disturbance model and amplifies the response to gusts below flutter on both UAV wings. On BACT the neural "
      f"policy reaches the lowest RMS plunge in {bact_rms} gust scenarios, at a higher cost through control effort, "
      f"and it failed {bt['failures']} of {bt['n']} turbulence runs.\n")
    rows = []
    for k, s, d in combos:
        best = _best_by_J(d)[0]
        cells: list[str] = []
        for c in ("open_loop", "lqr", "lqg", "neural"):
            a = d["cases"][c]
            rms = f"{a['rms_h_mm']['mean']:.2f} ± {a['rms_h_mm']['std']:.2f} / " if a["failures"] < a["n"] else ""
            j = f"{a['cost_J']['mean']:.3g}" if a.get("cost_J") and not a["failures"] else f"{a['failures']}/{a['n']} failed"
            cells.append(rms + (f"**{j}**" if c == best else j))
        rows.append([f"`{k}`", SCEN_LABEL[s], *cells])
    table("bench", f"Controller benchmark: RMS plunge (mm, mean ± std over {seeds} seeds) / mean cost J of the "
          "surviving runs. Lowest J in bold", ["Wing", "Scenario", "Open loop", "LQR", "LQG", "Neural"], rows)
    figure("time")
    w(f"The neural policy's advantage lies elsewhere. {T('delay')} gives the largest loop delay under which each "
      f"design survives the discrete gust. Classical designs re-computed for the true delay tolerate "
      f"at least {cap:.0f} ms on all wings, the end of the sweep. Designs fixed at {dt_ms:.0f} ms "
      f"tolerate only {_rng([tol[k][c] for k in tol for c in ('lqr_nominal', 'lqg_nominal')])} ms, and the single "
      f"neural policy tolerates {_rng([tol[k]['neural'] for k in tol])} ms ({F('delay')}). On `uav_flap` the neural "
      "policy therefore outlasts the fixed classical designs, though not the designs that know the true delay.\n")
    table("delay", "Largest tolerated loop delay (ms), discrete gust at 1.1 U_F (≥: survived up to the end of the sweep)",
          ["Wing", "LQR", "LQG", "Neural", f"LQR, {dt_ms:.0f} ms design", f"LQG, {dt_ms:.0f} ms design"],
          [[f"`{k}`", *((f"≥ {tol[k][c]:.0f}" if tol[k][c] >= cap else f"{tol[k][c]:.0f}") for c in ("lqr", "lqg", "neural", "lqr_nominal", "lqg_nominal"))] for k in tol])
    figure("delay")

    # ------------------------------------------------------------ results: robustness and ablations
    w("### 8.2 Robustness and ablations\n")
    rf_ = rob["uav_flap"]
    draws = rf_["metadata"]["draws"]
    par = rf_["parameters"]["turbulence_patch"]
    worst = max(par, key=float)

    def fails(row: dict) -> dict[str, int]:
        return {c: row[c]["failures"] for c in ("lqr", "lqg", "neural")}

    e_, n5, n10 = fails(par[worst]), fails(rf_["sensor_noise"]["0.05"]), fails(rf_["sensor_noise"]["0.10"])
    _need(e_["lqg"] < min(e_["lqr"], e_["neural"]), "LQG is the most robust to model error on uav_flap")
    _need(all(v == draws for lim in ("250", "125") for v in fails(rf_["rate_limit"][lim]).values()),
          "every uav_flap run fails at a slew limit of 250 deg/s or less")
    _need(all(v == 0 for row in rob["uav_section"]["sensor_noise"].values() for v in fails(row).values()),
          "uav_section is insensitive to sensor noise")
    w(f"{T('robust')} counts failed turbulence runs when designs made for the nominal wing fly perturbed conditions "
      f"(see also {F('robust')}). On `uav_flap`, LQG is the most robust to model error: at ±{100 * float(worst):.0f}% "
      f"it fails {e_['lqg']} of {draws} runs, against {e_['lqr']} for LQR and {e_['neural']} for the neural policy. "
      f"All three designs become fragile at 5% sensor noise ({n5['lqr']}, {n5['lqg']} and {n5['neural']} failures) "
      f"and fail almost always at 10% ({n10['lqr']}, {n10['lqg']} and {n10['neural']}). With a slew-rate limit of "
      "250°/s or less, every run fails. BACT is far more tolerant and fails consistently only at 125°/s, and "
      "`uav_section` is insensitive to sensor noise. The neural policy was trained with randomised conditions, yet "
      "it is not consistently more robust than LQG.\n")
    conds = ([(f"Model error ±{100 * float(lv):.0f}%", "parameters", lv) for lv in par]
             + [(f"Sensor noise {100 * float(nz):.0f}%", "sensor_noise", nz) for nz in rf_["sensor_noise"]]
             + [("Slew limit " + (lim if lim == "none" else f"{lim}°/s"), "rate_limit", lim) for lim in rf_["rate_limit"]])
    rows = []
    for label, group, lv in conds:
        row = [label]
        for k in PRESET_ORDER:
            src = rob[k]["parameters"]["turbulence_patch"] if group == "parameters" else rob[k].get(group, {})
            row += [str(v) for v in fails(src[lv]).values()] if lv in src else ["-"] * 3
        rows.append(row)
    table("robust", f"Failed turbulence runs out of {draws} for designs made for the nominal wing (L: LQR, G: LQG, "
          "N: neural)", ["Condition", *(f"`{k}` {c}" for k in PRESET_ORDER for c in "LGN")], rows)
    figure("robust")
    scen = abl["scenarios"]

    def rel(case: str) -> list[float]:
        return [100 * d["statistics"][f"{case} vs neural"]["cost_J"]["rel_diff"] for d in scen.values()
                if "rel_diff" in d["statistics"].get(f"{case} vs neural", {}).get("cost_J", {})]

    na_t = scen["turbulence_patch"]["cases"]["neural:no_airspeed"]
    na_b = scen["below_flutter"]["cases"]
    sup_rel = rel("neural+rules") + rel("neural+random")
    w(f"The ablations ({T('ablation')}) show that differentiable-physics fine-tuning is what makes the neural policy "
      f"work: imitation alone raises the cost by {_rng(rel('neural:imitation_only'))}%. The airspeed input is needed "
      "below flutter, where removing it multiplies the cost by "
      f"{na_b['neural:no_airspeed']['cost_J']['mean'] / na_b['neural']['cost_J']['mean']:.0f}, and in turbulence, "
      f"where {na_t['failures']} of {na_t['n']} runs fail without it. The rate penalty has little effect. Adding a "
      "rule-based or random effort supervisor on top of the neural policy changes the cost by "
      f"{_signed(min(sup_rel))} to {_signed(max(sup_rel))}%.\n")
    rows = []
    for case in abl["cases"]:
        cells = []
        for d in scen.values():
            a = d["cases"][case]
            st = d["statistics"].get(f"{case} vs neural", {}).get("cost_J", {})
            cells.append((f"{a['cost_J']['mean']:.3g}" if a.get("cost_J") else f"{a['failures']}/{a['n']} failed")
                         + (f" ({_signed(100 * st['rel_diff'])}%)" if "rel_diff" in st and a.get("cost_J") else ""))
        ds_ = abl["delay_sweep"].get(case)
        rows.append([ABL_LABEL.get(case, case), *cells,
                     f"{max([x['delay_ms'] for x in ds_ if not x['exceeded_limits']] or [float('nan')]):.0f} ms" if ds_ else "-"])
    table("ablation", "Ablations of the neural policy on `uav_flap`: cost J (change relative to the full policy) and "
          "delay tolerance", ["Case", *(SCEN_LABEL[s] for s in scen), "Delay tolerance"], rows)

    # ------------------------------------------------------------ results: LLM effort modes
    w("### 8.3 LLM supervising controller aggressiveness\n")
    s_ = llm["summary"]
    lst = [d["statistics"]["neural+llm vs neural"]["cost_J"] for d in llm["scenarios"].values()]
    _need(s_["fallbacks"] == 0, "the LLM never needed the fallback in the effort study")
    w(f"In a first test, the LLM supervised only the aggressiveness $a$ of the neural policy. Over {s_['decisions']} "
      f"decisions it never needed the fallback. Its median latency was {s_['latency_s']['median']:.2f} s (95th "
      f"percentile {s_['latency_s']['p95']:.2f} s), and it chose the same mode as the rules for "
      f"{_pc(s_['agreement_with_rules'])} of reports ({F('llm')}). In all {len(lst)} scenarios the cost with LLM "
      "supervision differed from that of the neural policy alone by less than 1% (Wilcoxon "
      f"{_p(min(x['wilcoxon_p'] for x in lst))}, n = {lst[0]['n']} seeds, too few to detect small effects). {F('tradeoff')} explains why. The "
      "fixed-aggressiveness curves show that the effort setting moves the trade-off between response and effort only "
      "slightly, and a supervisor adds value only if its operating point lies below the curve of its own controller. "
      "Effort supervision was therefore not pursued further.\n")
    figure("llm")
    figure("tradeoff")

    # ------------------------------------------------------------ results: fault-tolerant supervision
    w("### 8.4 Fault-tolerant supervision\n")
    faults = [f for f in ft["faults"] if f != "none"]
    fixed = {f: ft["faults"][f]["fixed_modes"] for f in ft["faults"]}
    sl = fixed["stiffness_loss"]
    n_caut = sum(fixed[f]["cautious"]["survival"] == 1.0 for f in faults)
    better = [f for f in faults if _surv(ft, f, "llm") > _surv(ft, f, "rules")]
    _need(sl["cautious"]["survival"] < sl["nominal"]["survival"], "slowing down after stiffness loss is worse than holding speed")
    _need(better == ["stiffness_loss"] and _acted_n(ft, "llm", "stiffness_loss")[0] == 0,
          "the LLM beat the rules only on stiffness_loss, where it never acted")
    _need(all(_surv(ft, f, "llm") < _surv(ft, f, "rules") for f in faults if f not in better),
          "the LLM was worse than the rules on every other fault")
    _need(all(_surv(ft, f, "llm") == 0.0 for f in held), "the LLM saved no held-out flight")
    none_a = ft["faults"]["none"]["supervisors"]
    la, lb = ft["llm_latency_s"], phys["llm_latency_s"]
    w(f"{T('fixed')} shows which fixed responses, applied 1 s after fault onset, survive each fault. No single mode "
      f"is right for all faults. The `cautious` mode (slowing to 0.9 U_F) saves the wing from {n_caut} of the "
      f"{len(faults)} faults. After a loss of pitch stiffness, however, slowing down is fatal: holding speed "
      f"survives in {_pc(sl['nominal']['survival'])} of flights, the `cautious` mode in {_pc(sl['cautious']['survival'])} "
      f"and the `safe` mode in {_pc(sl['safe']['survival'])}. The controllers are gain-scheduled for the healthy "
      "wing, which is below flutter at the lower speed while the damaged wing is still above it. A reflex of "
      "\"anomaly, slow down a little\" is wrong here, and a supervisor has to recognise this trap.\n")
    mode_names = list(fixed[faults[0]])
    table("fixed", "Survival of fixed responses applied 1 s after fault onset, and the oracle's least disruptive "
          "surviving choice (set A)", ["Fault", *(f"`{x}`" for x in mode_names), "Oracle"],
          [[f"`{f}`", *(_pc(fixed[f][x]["survival"]) for x in mode_names),
            f"`{max(set(ft['faults'][f]['oracle_modes']), key=ft['faults'][f]['oracle_modes'].count)}`"]
           for f in ft["faults"]])
    w(f"{T('survival')} gives the survival of each supervisor per fault on the three test sets and {T('paired')} the "
      f"paired comparisons. On set A the LLM kept {_pc(pool(ft, 'llm'))} ({_ci(ft, 'llm')}) of faulty flights from flutter, against "
      f"{_pc(pool(ft, 'rules'))} ({_ci(ft, 'rules')}) for the rules (exact McNemar {_p(pair(ft, 'llm vs rules')['mcnemar_p'])}, "
      f"Holm-adjusted {holm['A', 'llm vs rules']:.2g}), {_pc(pool(ft, 'none'))} without supervision and "
      f"{_pc(pool(ft, 'oracle'))} for the oracle ({F('ft')}). The LLM was better than the rules only on "
      "`stiffness_loss`, where it never acted. It was worse on every other fault, and on both held-out faults it "
      f"saved no flight. It rarely acted at all: in set B it changed mode after onset in {_acted_n(phys, 'llm')[0]} "
      f"of {_acted_n(phys, 'llm')[1]} faulty flights, against {_acted_n(phys, 'rules')[0]} for the rules. Its "
      f"advantage was the absence of false alarms. In fault-free flights it slowed down for "
      f"{none_a['llm']['slowed_s_mean']:.1f} s on average, against {none_a['rules']['slowed_s_mean']:.1f} s for the "
      f"rules and {none_a['random']['slowed_s_mean']:.1f} s for random choices. The LLM's median latency was "
      f"{la['median']:.1f} s (95th percentile {la['p95']:.1f} s) over {la['n']} decisions.\n")
    if nolat:  # ablation: the LLM without latency
        pn = nolat["pooled_faulty"]
        _need(pn["supervisors"]["llm"]["survival"] < pn["supervisors"]["rules"]["survival"],
              "the LLM lost to the rules even without latency")
        w("Latency alone does not explain the gap. In an ablation on the same seeds, every supervisor was asked "
          f"every {nolat['metadata']['supervisor_interval_s']:.0f} s and the LLM's answers took effect without delay. "
          f"The LLM then kept {_pc(pn['supervisors']['llm']['survival'])} of faulty flights and the rules "
          f"{_pc(pn['supervisors']['rules']['survival'])} (exact McNemar {_p(pn['paired']['llm vs rules']['mcnemar_p'])}; "
          "this test reuses the seeds of set A and is not part of the Holm family).\n")
    w("The two held-out faults test different capabilities. A loop delay appears as an explicit number in the "
      "report, because the flight computer times its own loop, so it tests whether the supervisor acts on an "
      f"indicator for which nobody wrote a rule. The rules saved {_pc(_surv(ft, 'loop_delay', 'rules'))} of these "
      f"flights through generic symptoms and the LLM {_pc(_surv(ft, 'loop_delay', 'llm'))}. Flap damage shows only "
      f"through its symptoms, and here both the rules ({_pc(_surv(ft, 'flap_effectiveness', 'rules'))}) and the LLM "
      f"({_pc(_surv(ft, 'flap_effectiveness', 'llm'))}) failed.\n")
    cols = [(k, s) for k, d in sets.items() for s in d["faults"]["none"]["supervisors"]]
    rows = [[f"`{f}`", *(f"{100 * _surv(sets[k], f, s):.0f}" for k, s in cols)] for f in faults]
    rows.append(["All faults", *(f"**{100 * pool(sets[k], s):.0f}**" for k, s in cols)])
    rows.append(["Held-out faults", *(f"{100 * np.mean([_surv(sets[k], f, s) for f in held]):.0f}" for k, s in cols)])
    rows.append(["No fault, time slowed (s)",
                 *(f"{sets[k]['faults']['none']['supervisors'][s]['slowed_s_mean']:.1f}" for k, s in cols)])
    table("survival", f"Survival (%) of faulty flights per supervisor and fault ({len(ft['metadata']['seeds'])} seeds "
          "per fault) on sets A, B and C, and mean time slowed down in fault-free flights. Orc: oracle; Rand: random; "
          "R+P: rules+physics; L+P: llm+physics; R+P2: rules+physics_v2; Guard: guardian",
          ["Fault", *(f"{k}: {SUP_SHORT[s]}" for k, s in cols)], rows)
    figure("ft")
    table("paired", f"Paired survival comparisons over the faulty flights of each test set: discordant pairs (b: only "
          f"the first survived; c: only the second) and exact McNemar p values, raw and Holm-adjusted over all "
          f"{len(family)} tests", ["Set", "Comparison", "b", "c", "p", "p (Holm)"],
          [[k, lab, str(pair(sets[k], lab)["survived_only_candidate"]), str(pair(sets[k], lab)["survived_only_reference"]),
            _pt(pair(sets[k], lab)["mcnemar_p"]), _pt(holm[k, lab])] for k, lab in family])

    # ------------------------------------------------------------ results: physics-informed supervision
    w("### 8.5 Physics-informed supervision\n")
    rp, lp = pair(phys, "rules+physics vs rules"), pair(phys, "llm+physics vs llm")
    first: dict[str, str] = {}
    for s in ("llm", "llm+physics", "rules+physics"):
        cnt: dict[str, int] = {}
        for r_ in phys["faults"]["stiffness_loss"]["per_seed"][s]:
            key = f"`{r_['first_response']}`" if r_["first_response"] else "no change"
            cnt[key] = cnt.get(key, 0) + 1
        first[s] = " and ".join(f"{k} in {n}" for k, n in sorted(cnt.items(), key=lambda x: -x[1]))
    _need(first["llm+physics"].startswith("`cautious`"), "with the estimate the LLM slowed to cautious after stiffness loss")
    w(f"The supervisors of set A see only symptoms. On set B the flutter-margin estimate was added to the report "
      f"({F('physics')}). It helped the rules, raising survival from {_pc(pool(phys, 'rules'))} to "
      f"{_pc(pool(phys, 'rules+physics'))} ({_p(rp['mcnemar_p'])}, Holm-adjusted {holm['B', 'rules+physics vs rules']:.2g}), "
      f"mainly by fixing the structural-damage trap, where survival rose from {_pc(_surv(phys, 'stiffness_loss', 'rules'))} "
      f"to {_pc(_surv(phys, 'stiffness_loss', 'rules+physics'))}. For the LLM the effect was reversed: survival fell "
      f"from {_pc(pool(phys, 'llm'))} to {_pc(pool(phys, 'llm+physics'))} ({_p(lp['mcnemar_p'])}). The first responses "
      f"to structural damage show why. Out of {len(phys['metadata']['seeds'])} flights, the LLM without the estimate "
      f"made {first['llm']}, the LLM with it chose {first['llm+physics']}, and rules+physics chose "
      f"{first['rules+physics']}. These choices suggest that the LLM read the estimate as a reason to slow down one step rather than as a "
      "stability prediction for each mode, and so fell into the trap the estimate was meant to prevent; its "
      "reasoning was not analysed further. More "
      "information changed which mistake the LLM made. Over "
      f"{lb['n']} decisions its median latency was {lb['median']:.1f} s (95th percentile {lb['p95']:.1f} s).\n")
    figure("physics")

    # ------------------------------------------------------------ results: guardian
    w("### 8.6 The guardian: learned diagnosis with physics admissibility\n")
    pcl, det, hk = (diag["in_distribution"]["window_level"]["per_class"], diag["in_distribution"]["detection"],
                    diag["held_out"]["kinds"])
    easy = [c for c in pcl if c != "servo_slew"]
    gp = guard["pooled_faulty"]["supervisors"]
    lost = {s: pair(guard, f"guardian vs {s}")["survived_only_reference"] for s in ("rules", "rules+physics_v2", "rules+physics")}
    acting = [s for s in gp if s not in ("oracle", "none")]
    none_c = guard["faults"]["none"]["supervisors"]
    _need(min(acting, key=lambda s: none_c[s]["slowed_s_mean"]) == "guardian", "the guardian raised the fewest false alarms")
    _need(gp["guardian"]["reaction_s_median"] < min(gp[s]["reaction_s_median"] for s in ("rules", "rules+physics")),
          "the guardian reacted faster than the rules")
    w(f"On held-out evaluation flights ({F('diagnoser')}), the learned diagnoser identifies "
      + ", ".join(f"`{c}`" for c in easy) + f" with window F1 scores of {_rng([pcl[c]['f1'] for c in easy], '.2f')}. "
      f"Detection is confirmed in {_rng([100 * det[c]['rate'] for c in easy])}% of these flights, with a median delay "
      f"of {_rng([det[c]['delay_from_onset_s']['median'] for c in easy])} s after onset. A slow servo is harder to "
      f"identify: its F1 is {pcl['servo_slew']['f1']:.2f}, {_pc(det['servo_slew']['rate'])} of flights are detected, "
      f"and the median delay is about {det['servo_slew']['delay_from_onset_s']['median']:.0f} s. The unknown-fault "
      f"score separates the held-out loop delay from in-distribution flights well (AUROC {hk['loop_delay']['auroc_flight']:.2f}) "
      f"and flap damage only moderately (AUROC {hk['flap_effectiveness']['auroc_flight']:.2f}). Only "
      f"{_pc(hk['flap_effectiveness']['unknown_detection_rate'])} of flap-damage flights cross the alarm threshold.\n")
    w(f"On set C the guardian survived {_pc(pool(guard, 'guardian'))} ({_ci(guard, 'guardian')}) of faulty flights, including every flight with a "
      f"held-out fault. The rules survived {_pc(pool(guard, 'rules'))} ({_ci(guard, 'rules')}), rules+physics {_pc(pool(guard, 'rules+physics'))} "
      f"and rules+physics_v2 {_pc(pool(guard, 'rules+physics_v2'))} ({T('paired')}, {F('guardian')}). The guardian lost "
      f"{lost['rules']} flights that the rules saved, {lost['rules+physics_v2']} that rules+physics_v2 saved and "
      f"{lost['rules+physics']} that rules+physics saved. Because the diagnoser rarely flags flap damage as unknown, "
      "the guardian's result on that fault depends on the physics check as well, which detects the loss of control "
      "authority. Among the supervisors on set C that act on faults, the guardian raised the fewest false alarms "
      f"({none_c['guardian']['slowed_s_mean']:.1f} s slowed down per fault-free flight) and was among the fastest: "
      f"its median reaction after fault onset was {gp['guardian']['reaction_s_median']:.1f} s, against "
      f"{gp['rules']['reaction_s_median']:.1f} s for the rules and {gp['rules+physics']['reaction_s_median']:.1f} s for "
      f"rules+physics. Rules+physics_v2 reacts in {gp['rules+physics_v2']['reaction_s_median']:.1f} s too.\n")
    w("The guardian's remaining misses have two sources. Some faults show an on-board signature late. A slow servo, "
      f"for instance, is invisible until the flap is asked to move fast ({_pc(_surv(guard, 'servo_slew', 'guardian'))} "
      "survival on `servo_slew`). In other flights the physics estimate reads a damaged wing as softer than it is "
      f"({_pc(_surv(guard, 'stiffness_loss', 'guardian'))} on `stiffness_loss`).\n")
    figure("diagnoser")
    figure("guardian")

    # ------------------------------------------------------------ results: LLM off the control path
    w("### 8.7 The language model off the control path\n")
    es, rs = expl["summary"], rev["summary"]
    gate = rs["gate"]
    n_h = rev["v1_regression"]["healthy_flights"]
    _need(set(es["rejection_reasons"]) == {"fault"}, "every rejected explanation named the wrong fault")
    _need(sum(c["source"] == "llm" for c in gate) == len({c["anomaly"] for c in gate}),
          "one LLM candidate per anomaly type")
    _need(not any(c["accepted"] for c in gate) and any(c["right_kind"] for c in gate),
          "the gate rejected every candidate, including the one with the right fault")
    w(f"*Explainer.* Over {es['reports']} reports from {len(expl['metadata']['flights'])} development flights, the LLM "
      f"produced {es['llm_answers_valid']} valid answers, of which {es['faithful']} ({_pc(faith)}) passed the "
      f"faithfulness check. All {es['llm_answers_valid'] - es['faithful']} rejections named the wrong fault. The "
      f"template was therefore shown for {_pc(es['fallback_rate'])} of reports, {es['llm_call_failures']} calls "
      f"failed, and the median latency was {es['latency_s']['median']:.1f} s.\n")
    w(f"*Post-flight reviewer.* Over {rs['reviews']} reviews the LLM produced {rs['valid_hypotheses']} valid "
      f"hypotheses, of which {rs['right_kind']} named the true fault ({T('gate')}, {F('review')}), with a median "
      f"latency of {rs['latency_s_median']:.1f} s. The gate tested one candidate built from the LLM's hypotheses for each anomaly type, plus one "
      f"forced-wrong control for each, and rejected all {len(gate)} "
      "diagnosers, including the one built on the correct hypothesis, so no unverified change reached the fast "
      f"models. The false-alarm criterion is evaluated on {n_h} healthy flights, of which the original diagnoser "
      f"alarms on {round(rev['v1_regression']['healthy_any_alarm_rate'] * n_h)}. Its tolerance of "
      f"{g_['false_alarm_tolerance']} therefore allows at most {math.floor(g_['false_alarm_tolerance'] * n_h + 1e-9)} "
      "additional flight with an alarm, which makes the gate conservative at this sample size. The gate checks "
      "consistency in the diagnoser's feature space and cannot check physical truth, so a wrong hypothesis whose "
      "flights resemble the anomaly could pass. The forced-wrong controls and the correctness of each hypothesis "
      "are reported for that reason.\n")
    table("gate", "Post-flight reviewer with a verification gate", ["Source", "Anomaly (truth)", "Hypothesis", "Correct", "Gate"],
          [[{"llm": "LLM", "forced_wrong": "Forced wrong"}.get(c["source"], c["source"]), f"`{c['anomaly']}`",
            f"`{c['hypothesis']}`", "yes" if c["right_kind"] else "no", "accepted" if c["accepted"] else "rejected"]
           for c in gate])
    figure("review")

    # ------------------------------------------------------------ discussion, limitations, conclusions
    w("## 9. Discussion\n")
    w("*Which inner loop wins depends on what it knows and on the actuator.* Given the turbulence statistics that the "
      "neural policy saw in training, which match the simulated turbulence, LQG is the strongest inner loop in "
      + ("every condition" if lqg_best == len(combos) else "most conditions") + " tested. The neural policy keeps an "
      "edge only under loop delay on `uav_flap`, where it outlasts the fixed classical designs. Claims that learned "
      "control outperforms \"classical\" control therefore depend on which classical baseline is used, and "
      "comparisons with plain LQR overstate the benefit.\n")
    w("*Latency limits what an LLM may do, but it is not the whole story.* A decision that arrives every few "
      "seconds cannot help against a gust that acts within tenths of a second, yet even without latency the "
      "local model lost to the rules (Section 8.4).\n")
    w("*More information did not help the LLM.* The LLM was cautious about acting, which avoided false alarms and "
      "missed most faults. Given a stability prediction for each mode, it read the prediction as a general signal "
      "to slow down. The same estimate improved the rules, which used it as a veto on modes predicted to be "
      "unstable. In this test, more physics did not lead the language model to use it as the rules did.\n")
    w("*Safety by construction.* The bounded mode set and the dwell-time bound make every supervisor's worst case "
      "analysable within the scope of Proposition 1, however it reasons. "
      "What a supervisor adds is the choice among safe modes, and in "
      "this study fast models made that choice better than the language model did.\n")
    w("## 10. Limitations\n")
    w(f"- The wings are linear typical sections with no freeplay or limit cycles, and the aerodynamics are "
      f"incompressible strip theory although BACT was tested at Mach {m['mach']}. All results come from simulation. "
      "No measured time histories were used, and the identification pipeline is tested on synthetic data only.")
    w("- The authors set the disturbance sizes, fault severities and mode menu from calibration runs.")
    w("- Only one local model size (8B, quantised) was tested, on one laptop GPU, with one prompt and fixed "
      "sampling settings, and the cloud backend was not run. "
      f"The effort-mode study used {lst[0]['n']} seeds per scenario. The oracle is an upper bound defined by "
      "hindsight and by a preference for the least disruptive surviving mode.")
    w("- Each fault was tested at one severity, and a conservative response survives both held-out faults, so "
      "generalisation to unknown faults that need a non-conservative response is untested.")
    w("- The stability guarantee is small-signal and holds for the nominal wing switching at a fixed airspeed. "
      "Faults and airspeed changes are outside it.")
    w("- The guardian's thresholds, fault responses and diagnoser were designed with the same simulator, fault "
      "catalogue and fault sizes on which it was tested, on disjoint seeds. Its transfer to other fault sizes, "
      "fault types or a real wing is untested. The LLM studies off the control path used development flights only "
      "and small samples.\n")
    w("## 11. Conclusions\n")
    w("The main findings are as follows.\n")
    w(f"- A turbulence-aware LQG gave the lowest cost in {_of(lqg_best, len(combos))} wing and scenario combinations. "
      "A neural policy trained through differentiable simulation beat plain LQR in "
      f"{sum(a for a, _ in lqr_wins.values())} of {len(combos)} and LQG in {lqg_wins}.")
    w(f"- An on-board 8B LLM answered in {_rng(lat_meds, '.1f')} s at the median and could not act on gusts. As a "
      f"fault-tolerant mode supervisor it kept {_pc(pool(ft, 'llm'))} of faulty flights from flutter, against "
      f"{_pc(pool(ft, 'rules'))} for hand-written rules and {_pc(pool(ft, 'oracle'))} for an oracle.")
    w(f"- An on-board flutter-margin estimate improved the rules ({_pc(pool(phys, 'rules+physics'))} against "
      f"{_pc(pool(phys, 'rules'))}) and degraded the LLM ({_pc(pool(phys, 'llm+physics'))} against "
      f"{_pc(pool(phys, 'llm'))}), because the LLM treated a stability prediction for each mode as a general cue to "
      "slow down.")
    w("- A guardian built from a learned fault diagnoser and a physics-based admissibility check, with no language "
      f"model, kept {_pc(pool(guard, 'guardian'))} of faulty flights from flutter on fresh seeds, including both "
      f"fault types its diagnoser never saw. It reacted in {gp['guardian']['reaction_s_median']:.1f} s and raised "
      "almost no false alarms.")
    w("- Every supervisor only selects among modes that are individually stable, and controller switches are spaced "
      f"by more than the computed average dwell time of {stab['max_dwell_s']:.1f} s. No sequence of answers can "
      "therefore destabilise the healthy wing's closed loop at a fixed airspeed in the small-signal sense; faults, "
      "saturation, the slew-rate limit and airspeed changes are outside this guarantee.")
    w(f"- Off the control path, the language model's explanations of a physics-informed supervisor's decisions passed "
      f"a faithfulness check in {_pc(faith)} of valid answers, and a verification gate rejected all candidate "
      "post-flight model updates, including one built on the correct fault.\n")
    w("For the local model tested here, on-board supervision of a fast, open-loop-unstable plant is better built "
      "from fast models whose decisions can be verified, with the language model kept off the control path to "
      "explain decisions to the operator. Whether larger or faster language models change this is open. The next "
      "step is a wind-tunnel test on measured data.\n")

    # ------------------------------------------------------------ declarations, references
    w("## Statements and declarations\n")
    w("*Funding.* The authors did not receive support from any organization for the submitted work.\n")
    w("*Competing interests.* The authors have no relevant financial or non-financial interests to disclose.\n")
    w("*Ethics.* This article does not contain any studies with human participants or animals performed by any of "
      "the authors.\n")
    w(f"*Data and code availability.* The simulator, controllers, supervisors and the scripts that regenerate every "
      f"result and every number in this paper are available at {REPOSITORY} (flutter_ai {__version__}, commit "
      f"{_git_commit()}).\n")
    w(f"*Author contributions.* {AUTHORS[0]}: conceptualization, development of the aeroelastic model, "
      "methodology, software, validation, investigation, formal "
      f"analysis, visualization, writing (original draft). {AUTHORS[1]}: supervision, writing (review and editing). "
      "Both authors read and approved the final manuscript.\n")
    w("*Use of AI tools.* Described in Section 7.\n")
    w("## References\n")
    L.extend(cite.listing())
    todo = cite.to_check()
    if todo:
        w("\n## Appendix: citation check (Rules.md §1.7)\n")
        w("Open and read these before submission (only their abstracts were read, or they were not opened):\n")
        L.extend(todo)

    def words(lines: list[str]) -> int:
        prose = [x for x in lines if x.strip() and not x.lstrip().startswith(("|", "!", "$$", "#", "**Table", "**Figure"))]
        return len(re.findall(r"[A-Za-z0-9][\w'./%+-]*", " ".join(prose)))

    i_abs, i_intro, i_decl = (L.index(h) for h in ("## Abstract\n", "## 1. Introduction\n", "## Statements and declarations\n"))
    L[note_at] = (f"*Draft generated {datetime.now(timezone.utc).isoformat(timespec='minutes')} by "
                  f"`python -m flutter_ai paper` (flutter_ai {__version__}, commit {_git_commit()}). Numbers and "
                  "comparisons are computed from the result files; edit the prose in `src/flutter_ai/paper.py`, "
                  f"never the numbers here. Abstract: {words(L[i_abs + 1:i_abs + 2])} words (journal limit 250). "
                  f"Body, Sections 1 to 11 without tables and captions: {words(L[i_intro:i_decl])} words (journal "
                  "limit 6,000 words or about 20 pages). To do: confirm the statements and declarations and the AI-use paragraph; "
                  "add the flutter speed printed in the textbook for the Section 3.3 case after checking the book; "
                  "open the references listed in the citation-check appendix.*\n")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_path.parent / "cover-letter.md").write_text(
        f"{' and '.join(AUTHORS)}  \n{AFFILIATION}  \n{datetime.now(timezone.utc):%d %B %Y}\n\n"
        "Prof. Sang Joon Shin  \nEditor-in-Chief, International Journal of Aeronautical and Space Sciences\n\n"
        "Dear Prof. Shin,\n\n"
        f"Please consider our manuscript \"{TITLE}\" for publication as an Original Paper in the International "
        "Journal of Aeronautical and Space Sciences.\n\n"
        "Large language models are being proposed as supervisors for flight control. We test that proposal on "
        "active flutter suppression, where the plant is unstable without control and faster than the model can "
        "answer. On a validated typical-section model with a realistic flap actuator, an on-board language model "
        f"kept {_pc(pool(ft, 'llm'))} of faulty flights from flutter, against {_pc(pool(ft, 'rules'))} for simple "
        "rules. A supervisor built from a learned fault diagnoser and a physics-based admissibility check kept "
        f"{_pc(pool(guard, 'guardian'))} on fresh test flights, including two fault types that its diagnoser and the rules never saw. "
        "An average dwell-time bound guarantees that no sequence of controller switches can destabilise the "
        "healthy wing's loop (small-signal, fixed airspeed).\n\n"
        "The paper fits the journal's coverage of structural dynamics, aeroelasticity and flight control, and it "
        "relates to recent work in the journal on fault-tolerant control by switching among pre-designed "
        "configurations (Yu et al., 2026, https://doi.org/10.1007/s42405-026-01315-3).\n\n"
        "We confirm that the manuscript has not been published and is not under consideration elsewhere, and "
        "that both authors approved this submission. The code and the scripts that regenerate every result and "
        f"number in the paper are at {REPOSITORY}; earlier drafts of the manuscript are in its history. "
        "The use of an AI coding assistant is described in Section 7 of the manuscript. The authors declare no "
        "competing interests.\n\nSincerely,\n\n"
        f"{AUTHORS[0]} (first author), on behalf of both authors  \n"
        f"Corresponding author: {CORRESPONDING[0]}, {CORRESPONDING[1]}\n", encoding="utf-8")
    return out_path
