"""
Paper draft generator (Phase 4.8): every number and every comparative statement in the draft is computed from the
result files or read from the code's parameters, so nothing is typed by hand (Rules.md §1). Citations come from
docs/research/references.bib; entries not read in full are listed in a citation-check appendix.

    python -m flutter_ai paper        # -> docs/research/paper-draft.md (+ docs/research/figures/)

Inputs (run these first; missing ones become a visible "not run yet" note):
    outputs/<preset>/results.json (evaluate), robustness.json (robustness), ablations.json (ablations),
    llm_benchmark.json (llm-benchmark), ft_benchmark.json (ft-benchmark), ft_stability.json (ft-stability),
    outputs/validation/bact.json (validate), models/classical_tuning_<preset>.json (tune).
Edit the prose here; never edit numbers in the generated file.
"""

from __future__ import annotations

import hashlib
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
FIGURES = [  # (preset, file, caption)
    ("uav_flap", "fig1_flutter_vg.png", "Damping and frequency of the structural modes vs airspeed (default wing, open "
     "loop). Flutter where a mode's damping crosses zero."),
    ("bact", "fig1_flutter_vg.png", "The same analysis for the NASA BACT wind-tunnel model."),
    ("uav_flap", "fig2_time_turbulence_patch.png", "Time histories in turbulence at 1.1 U_F (seed 0): plunge, pitch, "
     "flap command and gust."),
    ("uav_flap", "fig6_tradeoff.png", "Effort–response trade-off at fixed aggressiveness (turbulence). A supervisor "
     "adds value only if its point lies below the curve of its controller."),
    ("uav_flap", "fig8_delay_robustness.png", "Tolerance to loop delay: designs re-computed for the true delay vs "
     "fixed designs."),
    ("uav_flap", "fig10_robustness.png", "Robustness of designs made for the nominal wing: model error, sensor noise "
     "and actuator slew-rate limits."),
    ("uav_flap", "fig9_training.png", "Neural policy training: imitation loss and differentiable-physics fine-tuning."),
    ("uav_flap", "fig11_llm_supervisor.png", "Local LLM supervisor choosing effort modes: latency per decision, and "
     "where supervised runs land relative to the fixed-aggressiveness curves."),
    ("uav_flap", "fig12_fault_tolerance.png", "Fault-tolerant supervision: flights survived and time slowed down, per "
     "fault and supervisor."),
    ("uav_flap", "fig12_fault_tolerance_physics_test100.png", "Pre-registered test of physics-informed supervision "
     "(seeds 100-119): rules and LLM with and without the on-board flutter-margin estimate."),
    ("uav_flap", "fig14_diagnoser.png", "Learned fault diagnoser on held-out evaluation flights: detection of the "
     "known faults and the unknown-fault score on the two faults it never saw."),
    ("uav_flap", "fig12_fault_tolerance_guardian_test300.png", "The guardian (learned diagnosis + physics "
     "admissibility, no language model) against the other supervisors on fresh seeds 300-319."),
    ("uav_flap", "fig17_review_loop.png", "Post-flight LLM reviewer: each hypothesis (and each forced-wrong control) "
     "through the verification gate's checks, against the thresholds fixed in advance."),
]


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
    """Collects citations in order of first use and renders author-year text."""

    def __init__(self, bib: dict[str, dict[str, str]]):
        self.bib = bib
        self.used: list[str] = []
        base = {k: self._base(e) for k, e in bib.items()}
        self.label = dict(base)
        for lab in set(base.values()):
            same = [k for k in bib if base[k] == lab]  # bib-file order
            if len(same) > 1:  # author-year collisions get a/b suffixes
                for i, k in enumerate(same):
                    self.label[k] = lab + "abcdefghij"[i]

    @staticmethod
    def _base(e: dict[str, str]) -> str:
        authors = [a.strip() for a in e.get("author", "Anon.").split(" and ")]
        first = authors[0].split(",")[0]
        who = first if len(authors) == 1 else (f"{first} & {authors[1].split(',')[0]}" if len(authors) == 2
                                               and authors[1] != "others" else f"{first} et al.")
        return f"{who} {e.get('year', '')}"

    def __call__(self, *keys: str) -> str:
        for k in keys:
            if k not in self.used:
                self.used.append(k)
        return "(" + "; ".join(self.label[k] for k in keys) + ")"

    @staticmethod
    def _plain(text: str) -> str:
        """BibTeX/LaTeX markup -> plain text for Markdown."""
        for a, b in ((r'\"o', "ö"), (r"\~a", "ã"), (r"$\le$", "≤"), (r"$^\circ$", "°"), ("$", "")):
            text = text.replace(a, b)
        return text

    def listing(self) -> list[str]:
        out = []
        for k in self.used:
            e = {f: self._plain(v) for f, v in self.bib[k].items()}
            venue = e.get("journal") or e.get("booktitle") or e.get("institution") or e.get("publisher") or \
                e.get("howpublished") or ""
            ident = f"doi:{e['doi']}" if "doi" in e else (f"arXiv:{e['eprint']}" if "eprint" in e else "")
            if "volume" in e:  # journal-style: volume(issue), pages
                where = [e["volume"] + (f"({e['number'].replace('--', '–')})" if "number" in e else ""),
                         e.get("pages", "").replace("--", "–")]
            else:  # report number
                where = [e.get("number", "")]
            parts = [x for x in (venue, *where, ident) if x]
            authors = [a.strip() for a in e.get("author", "").split(" and ")]
            who = "; ".join(authors if len(authors) <= 6 else authors[:3] + ["et al."])
            out.append(f"- **{self.label[k]}** [{k}]: {who.rstrip('.')}. *{e.get('title', '')}*. "
                       + ", ".join(parts) + ".")
        return out

    def to_check(self) -> list[str]:
        return [f"- [{k}] read status: **{self.bib[k].get('readstatus', '?')}**" for k in self.used
                if self.bib[k].get("readstatus") != "full"]


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


def _copy_figures() -> list[tuple[str, str]]:
    out_dir = RESEARCH / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for preset, name, caption in FIGURES:
        src = OUTPUTS_DIR / preset / name
        if src.exists():
            dst = out_dir / f"{preset}_{name}"
            shutil.copyfile(src, dst)
            done.append((f"figures/{dst.name}", caption))
    return done


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
        out.append(f"Off the control path, the same language model's explanations of these decisions passed a "
                   f"faithfulness check in {100 * s_['faithfulness_rate']:.0f} % of its valid answers.")
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
        parts.append(f"Because the LLM only selects among modes that are each stable, and controller switches are spaced "
                     f"by at least the computed dwell time ({stab['max_dwell_s']:.1f} s), no sequence of its answers can "
                     "destabilise the nominal closed loop in the small-signal sense (faults, actuator limits and airspeed "
                     "transitions are outside this guarantee); its value depends on the diagnostics it is given and on "
                     "the time scale of the decision.")
    parts.extend(later or [])
    parts.append("*[Owner: add one closing sentence on what this means for practice and the next step.]*")
    return " ".join(parts)


# ------------------------------------------------------------------------------------------------ the draft
def build(out_path: Path | None = None, copy_figures: bool = True) -> Path:
    out_path = Path(out_path or RESEARCH / "paper-draft.md")
    res = {p: _load(OUTPUTS_DIR / p / "results.json") for p in PRESET_ORDER}
    rob = {p: _load(OUTPUTS_DIR / p / "robustness.json") for p in PRESET_ORDER}
    tun = {p: _load(MODELS_DIR / f"classical_tuning_{p}.json") for p in PRESET_ORDER}
    abl = _load(OUTPUTS_DIR / "uav_flap" / "ablations.json")
    llm = _load(OUTPUTS_DIR / "uav_flap" / "llm_benchmark.json")
    ft = _load(OUTPUTS_DIR / "uav_flap" / "ft_benchmark.json")
    stab = _load(OUTPUTS_DIR / "uav_flap" / "ft_stability.json")
    phys = _load(OUTPUTS_DIR / "uav_flap" / "ft_benchmark_physics_test100.json")
    guard = _load(OUTPUTS_DIR / "uav_flap" / "ft_benchmark_guardian_test300.json")
    expl = _load(OUTPUTS_DIR / "uav_flap" / "explainer_eval.json")
    rev = _load(OUTPUTS_DIR / "uav_flap" / "review_loop.json")
    bact = _load(OUTPUTS_DIR / "validation" / "bact.json")
    cite = Refs(_bib(RESEARCH / "references.bib"))
    p = PRESETS["uav_flap"]
    rate = math.degrees(p.u_rate_max[0]) if p.u_rate_max else float("nan")
    L: list[str] = []
    w = L.append

    w("# On-board language-model supervision for active flutter suppression: a fair benchmark with fault "
      "tolerance and a dwell-time guarantee\n")
    w(f"*Draft generated {datetime.now(timezone.utc).isoformat(timespec='minutes')} by `python -m flutter_ai paper` "
      f"(flutter_ai {__version__}, commit {_git_commit()}). Numbers and comparisons are computed from the result "
      "files; edit the prose in `src/flutter_ai/paper.py`, never the numbers here. Author list, affiliations and "
      "target venue: to be decided by the owner (see `submission-checklist.md`).*\n")

    # ------------------------------------------------------------ abstract
    w("## Abstract\n")
    r = res["uav_flap"]
    abstract = ("Language models are increasingly proposed as supervisors of control systems, but their latency and "
                "reliability sit uneasily with fast, safety-critical loops. We study this tension on active flutter "
                "suppression. A validated typical-section model with a servo-driven trailing-edge flap, a realistic "
                f"actuator slew-rate limit ({rate:.0f} °/s) and a {1000 * 0.005:.0f} ms loop delay is controlled by "
                "LQR, a turbulence-aware LQG and a neural policy trained through differentiable simulation, all "
                "minimising the same cost, with the classical designs tuned for the actual actuator on data disjoint "
                "from the test set.")
    if r:
        wins_lqr, n_lqr = _neural_vs(r, "lqr")
        wins_lqg, n_lqg = _neural_vs(r, "lqg")
        abstract += (f" Over {len(r['metadata']['seeds'])} randomised runs per scenario the neural policy beats LQR "
                     f"in {wins_lqr} of {n_lqr} scenarios but LQG in {wins_lqg} of {n_lqg}.")
    if llm:
        diffs = [abs(d["statistics"]["neural+llm vs neural"]["cost_J"].get("rel_diff", float("nan")))
                 for d in llm["scenarios"].values()]
        effect = ("and changes the cost by less than 1 % in every scenario" if all(x < 0.01 for x in diffs) else
                  f"and changes the cost by up to {100 * np.nanmax(diffs):.0f} %")
        abstract += (f" A local 8-billion-parameter model supervising controller aggressiveness answers in "
                     f"{llm['summary']['latency_s']['median']:.1f} s (median), far slower than the gusts it would have "
                     f"to react to, {effect}.")
    if ft:
        pooled = ft["pooled_faulty"]["supervisors"]
        abstract += (" Moved to a slower task, fault-tolerant supervision with pre-verified modes, the flights "
                     "survived under six mid-flight faults were: " + ", ".join(
                         f"{s} {100 * v['survival']:.0f} %" for s, v in pooled.items()) + ".")
    if stab:
        abstract += (f" Switching between the inner-loop controllers is provably stable (small-signal, nominal wing) "
                     f"when switches are at least {stab['max_dwell_s']:.1f} s apart on average, a condition the "
                     "architecture enforces.")
    later = _later_findings(phys, guard, expl, rev)
    if later:
        abstract += " " + " ".join(later)
    w(abstract + "\n")

    # ------------------------------------------------------------ introduction
    w("## 1. Introduction\n")
    w("Flutter is a dynamic aeroelastic instability: above a critical airspeed the aerodynamic forces feed energy "
      "into coupled structural modes and the oscillation grows until the structure fails. Active flutter "
      "suppression moves control surfaces to extend the stable envelope. It has been demonstrated in wind tunnels "
      f"with classical, LQG and minimax control laws {cite('mukhopadhyay1999')} and with neural networks "
      f"{cite('scott2000', 'bernellizazzera2000')}, and learned controllers remain an active topic "
      f"{cite('mattaboni2009', 'tang2018', 'tang2020', 'chen2023rl')}.\n")
    w("Large language models (LLMs) are now proposed as supervisory layers above fast feedback loops "
      f"{cite('du2026survey')}: for fault-tolerant control of process plants {cite('vyas2026ftc', 'vyas2026tutorial', 'papacharalampopoulos2026')}, "
      f"for robot skill selection {cite('ahn2022saycan')} and for drones, where the latency of sequential token "
      f"generation is a key obstacle {cite('chen2023typefly')}. The survey by Du & Chan names the lack of formal "
      "stability proofs as the field's central open problem and notes that stability across LLM-triggered mode "
      "switches is largely unanalysed. We found no evaluation of an LLM supervisor for flutter suppression, and none that "
      "runs the model on board with its measured latency against rule-based and oracle supervisors, including "
      "faults the rules were not written for.\n")
    w("**Contributions.** (i) An open, validated flutter-suppression benchmark with realistic actuation and fair "
      "baselines (one shared cost; classical designs tuned for the real actuator). (ii) A controlled comparison of "
      "LQR, LQG and a differentiable-simulation neural policy, with robustness studies and ablations. (iii) A "
      "supervisory architecture for an on-board LLM above a fast, open-loop-unstable plant, building on dwell-time "
      f"switching among pre-verified controllers {cite('papacharalampopoulos2026')}: a bounded mode set, a controller "
      "bank with bumpless transfer, one question at a time with measured latency, and a discrete-time dwell-time bound "
      "computed for the implemented controllers. "
      "(iv) An evaluation of an on-board LLM supervisor for fault-tolerant flutter suppression against no, random, "
      "rule-based and oracle supervision. (v) Negative results reported as such.\n")

    # ------------------------------------------------------------ related work
    w("## 2. Related work\n")
    w(f"*Neural flutter suppression.* A neural network scheduled the coefficients of flutter-suppression control laws "
      f"over Mach number and dynamic pressure on the BACT wing (our second benchmark) in the NASA Transonic Dynamics "
      f"Tunnel, where the classical systems proved more robust than the neural ones {cite('scott2000')}. Recurrent "
      f"neural networks have been used for active flutter suppression {cite('bernellizazzera2000')}, and neural "
      f"control has handled control-input constraints with wind-tunnel validation {cite('tang2020')}; deep reinforcement "
      f"learning has been trained directly in a wind tunnel {cite('chen2023rl')}. Our neural policy is trained "
      f"offline through a differentiable model {cite('xu2022shac', 'heeg2024')}; its novelty is not claimed.\n")
    w(f"*LLM supervision.* The closest work is a three-layer fault-tolerant scheme for process systems "
      f"{cite('papacharalampopoulos2026')}: a cloud LLM, called once per fault event, selects one of four bounded "
      "actions over pre-verified controllers, with a rule-based fallback and an average dwell-time bound on "
      "switching, and is compared with a rule-based supervisor; its authors describe the LLM's role as advisory and "
      "as an audit record. Our architecture shares these elements. It differs in the plant (an aeroelastic system "
      "that is unstable without control, with dynamics faster than the LLM's latency), in querying an on-board model "
      "repeatedly with its measured latency applied in simulation time, and in the evaluation: random, rule-based and "
      "oracle supervisors, paired statistics over seeds, and faults described neither in the rules nor in the prompt. "
      f"LLM agents that propose validated recovery actions {cite('vyas2026ftc')} and LLM planning above a safety "
      f"filter {cite('shibu2026skysim')} share our separation of a slow language layer from a fast, deterministic "
      "layer.\n")
    w(f"*Switching stability.* Switching among stable linear systems is stable if switches are slow on average "
      f"{cite('hespanha1999')} (stated in continuous time; Section 5 gives the discrete-time counterpart we use); "
      f"LQG designs have no guaranteed robustness margins {cite('doyle1978')}, which our "
      "robustness results illustrate.\n")

    # ------------------------------------------------------------ model
    w("## 3. Model and validation\n")
    w(f"The typical section has plunge h, pitch α and trailing-edge flap β (3 degrees of freedom). Unsteady "
      f"aerodynamics use Theodorsen's theory {cite('theodorsen1935')} with a two-lag Wagner approximation in state "
      f"space; the model is discretised exactly (zero-order hold) at {1 / 0.005:.0f} Hz. The flap is driven by a "
      f"servo modelled as a second-order system ({p.f_beta:.0f} Hz, damping {p.zeta_beta:.2f}, the value measured on "
      f"the BACT actuator {cite('waszak1998bact')}), a ±{math.degrees(p.u_max[0]):.0f}° command limit and a "
      f"{rate:.0f} °/s slew-rate limit from a UAV servo datasheet {cite('dsw001a')}. Every command reaches the "
      "actuator one sample later.\n")
    for key in PRESET_ORDER:
        rr = res[key]
        if rr:
            f = rr["flutter"]
            pk = f.get("pk_exact_theodorsen")
            agree = (f"; an independent p-k analysis with the exact Theodorsen function gives {pk:.2f} m/s "
                     f"({100 * (f['state_space'] - pk) / pk:+.2f} %)") if pk else ""
            w(f"- **{key}**: flutter at {f['state_space']:.2f} m/s, {f['frequency_hz']:.2f} Hz{agree}.")
    if bact:
        m = bact["reference"]["measured"]
        pr = next(iter(bact["predictions"].values()))
        if pr.get("flutter_q_psf"):
            w(f"- **Wind-tunnel check:** for the NASA BACT model {cite('waszak1998bact')} the predicted flutter point is "
              f"{pr['flutter_q_psf']:.0f} psf at {pr['flutter_freq_hz']:.2f} Hz against ≈{m['flutter_q_psf']:.0f} psf at "
              f"≈{m['flutter_freq_hz']:.0f} Hz measured ({100 * (pr['flutter_q_psf'] - m['flutter_q_psf']) / m['flutter_q_psf']:+.0f} % "
              f"in dynamic pressure, {100 * (pr['flutter_freq_hz'] - m['flutter_freq_hz']) / m['flutter_freq_hz']:+.1f} % "
              f"in frequency); the pressure error is attributed to compressibility (Mach {m['mach']}) in an "
              "incompressible model.")
    w("")

    # ------------------------------------------------------------ controllers
    w("## 4. Controllers and fairness\n")
    w("All controllers minimise J = Σ (xᵀQx + c(a)·uᵀRu + ΔuᵀWΔu), with Bryson-normalised Q and R, an effort weight "
      "c(a) set by the aggressiveness a ∈ [0, 1], and a penalty W on command changes. They see the same "
      "measurements, loop delay and actuator limits.\n")
    w("- **LQR:** discrete-time, designed for the loop delay (command history in the state) and the rate penalty "
      "(cross term); aerodynamic lag states from an on-board observer.")
    w("- **LQG:** the same regulator on the plant augmented with a first-order (AR(1)) turbulence state, and a "
      "steady-state Kalman filter for the lag states and the gust whose input model includes the actuator's known "
      "slew-rate limit.")
    w("- **Neural policy:** a 64×64 tanh network of the state estimate, the previous command, the airspeed ratio "
      "and a; output u_max·tanh(g(x) − g(0)) so the command is zero at equilibrium. Trained by imitation of LQR, "
      "then fine-tuned through the differentiable, rate-limited plant with randomised loop delay, airspeed and "
      "turbulence.")
    w("- **Fairness:** the classical designs' free knobs (an extra rate-penalty multiplier; the Kalman filter's "
      "assumed sensor noise) are tuned for the actual actuator on episodes whose seeds never appear in the test set:")
    for key in PRESET_ORDER:
        t = tun[key]
        if t:
            w(f"  - {key}: LQR rate multiplier {t['lqr']['rate_multiplier']:g}; LQG rate multiplier "
              f"{t['lqg']['rate_multiplier']:g}, assumed sensor noise {t['lqg']['measurement_noise']:g} of the reference scale.")
    w("")

    # ------------------------------------------------------------ supervision
    w("## 5. Supervisory architecture\n")
    w("Every few seconds the supervisor receives a plain-language status report built only from on-board "
      "information and returns one mode name from a fixed enum; it never outputs gains, forces or speeds. The "
      "answer is constrained by a JSON schema, validated again, and replaced by the rule-based decision on any "
      f"failure (timeout, invalid answer, refusal, call budget). The model runs locally (Ollama, {_llm_model(ft, llm)}) "
      "so it works without a data link; a cloud model can be selected instead. The supervisor is asked one "
      "question at a time and its measured latency is applied in simulation time. All candidate controllers run "
      "every step on the same measurements and applied-command history (a controller bank with bumpless transfer), "
      "so switching causes no transient and all modes share one state.\n")
    if stab:
        from .fault_tolerance.simulate import MIN_DWELL_S

        rhos = max(max(r_["spectral_radius"].values()) for r_ in stab["speeds"])
        w(f"**Stability of switching.** On that shared state each inner-loop controller gives a linear map "
          f"ξ_(k+1) = A_i ξ_k. All are Schur-stable at every airspeed of the schedule (largest spectral radius "
          f"{rhos:.4f}). With V_i = ξᵀ P_i ξ from the discrete Lyapunov equation, V_i(A_i ξ) ≤ λ V_i(ξ) and "
          "V_j ≤ μ V_i for all i, j; after k steps containing N switches V ≤ μ^N λ^k V(0), which decays whenever "
          "switches are on average more than ln μ / (−ln λ) steps apart: the discrete-time counterpart of the average "
          f"dwell-time argument {cite('hespanha1999')}. In scaled coordinates the bound is at most "
          f"{stab['max_dwell_s']:.2f} s; the architecture waits at least {MIN_DWELL_S:.0f} s between controller "
          "switches, so no sequence of supervisor answers can destabilise the nominal closed loop. The guarantee is "
          "small-signal and for the healthy wing: it excludes faults, saturation and the rate limit, and airspeed "
          "changes, which are slow (limited deceleration), gain-scheduled transitions; mode changes that keep the "
          "controller and only move the airspeed target are therefore not delayed. Tests check that the analysis "
          "matrices reproduce the simulators step by step.\n")

    # ------------------------------------------------------------ experiments
    w("## 6. Experiments\n")
    if r:
        sc = r["scenarios"]
        w(f"Four scenarios per wing: turbulence (a 4 s patch), a 1-cosine gust, the same gust at 1.3 U_F and "
          f"below flutter. Deterministic gusts are randomised per seed (amplitude, length, start, initial condition). "
          f"{len(r['metadata']['seeds'])} seeds per scenario; paired Wilcoxon signed-rank tests and bootstrap 95 % "
          f"confidence intervals on the paired differences. Disturbance sizes for uav_flap are those a {rate:.0f} °/s "
          f"servo can handle (turbulence {sc['turbulence_patch']['scenario']['gust']['amplitude']:.2f} m/s RMS).\n")
    w("Further studies: robustness (designs for the nominal wing flying perturbed wings, sensor noise, slower "
      "actuators), ablations of the neural policy, an LLM supervising effort modes, the fault-tolerant study "
      "(Section 7.5), a pre-registered test of physics-informed supervision (7.6), a supervisor built from fast "
      "learned and physics models (7.7), and the LLM in roles off the control path (7.8). Fault-tolerant studies use "
      "disjoint seed sets: 0-19 (7.5), 100-119 (7.6), 300-319 (7.7); every design choice was made on seeds from 1000 "
      "up.\n")

    # ------------------------------------------------------------ results
    w("## 7. Results\n")
    w("### 7.1 Controllers\n")
    w("Cells: RMS plunge (mm, mean ± std over seeds) · mean cost J of surviving runs (lower is better) · failures.\n")
    for key in PRESET_ORDER:
        rr = res[key]
        if not rr:
            w(f"*{key}: not run yet.*\n")
            continue
        w(f"**{key}**\n")
        L.extend(_results_table(rr))
        w("")
        wl, nl = _neural_vs(rr, "lqr")
        wg, ng = _neural_vs(rr, "lqg")
        w(f"- The neural policy has significantly lower J than LQR in {wl} of {nl} scenarios and than LQG in {wg} of "
          f"{ng}.")
        for s, d in rr["scenarios"].items():
            best, n_ok = _best_by_J(d)
            w(f"- *{SCEN_LABEL.get(s, s)}*: lowest J without failures: **{CTRL_LABEL.get(best, best)}**. "
              f"{_stat_sentence(d, 'neural vs lqg', 'Neural', 'LQG')}")
        ds = rr["delay_sweep"]
        tol = {c: max([x["delay_steps"] * ds["dt"] * 1000 for x in ds[c] if not x["exceeded_limits"]] or [float("nan")])
               for c in ds if c not in ("delays", "dt")}
        w("- Delay tolerance (ms, discrete gust): " + ", ".join(f"{c} {v:.0f}" for c, v in tol.items()
                                                                if not math.isnan(v)) + ".\n")

    w("### 7.2 Robustness\n")
    for key in PRESET_ORDER:
        rb = rob[key]
        if not rb:
            continue
        ctrls = rb["controllers"]
        w(f"**{key}**, failed runs out of {rb['metadata']['draws']} in turbulence:\n")
        w("| Condition | " + " | ".join(CTRL_LABEL.get(c, c) for c in ctrls) + " |")
        w("|---" * (len(ctrls) + 1) + "|")
        for lv, row in rb["parameters"]["turbulence_patch"].items():
            w(f"| model error ±{100 * float(lv):.0f} % | " + " | ".join(str(row[c]["failures"]) for c in ctrls) + " |")
        for nz, row in rb["sensor_noise"].items():
            w(f"| sensor noise {100 * float(nz):.0f} % | " + " | ".join(str(row[c]["failures"]) for c in ctrls) + " |")
        for lim, row in rb.get("rate_limit", {}).items():
            w(f"| slew limit {lim}{'' if lim == 'none' else ' °/s'} | " + " | ".join(str(row[c]["failures"]) for c in ctrls) + " |")
        w("")

    w("### 7.3 Ablations (uav_flap)\n")
    if abl:
        w("| Case | " + " | ".join(SCEN_LABEL[s] for s in abl["scenarios"]) + " | Delay tolerance |")
        w("|---" * (len(abl["scenarios"]) + 2) + "|")
        for case in abl["cases"]:
            cells = []
            for d in abl["scenarios"].values():
                a = d["cases"][case]
                st = d["statistics"].get(f"{case} vs neural", {}).get("cost_J", {})
                rel = f" ({_pct(st['rel_diff'])} vs full)" if "rel_diff" in st else ""
                cells.append((f"J {a['cost_J']['mean']:.3g}" if a.get("cost_J") else f"{a['failures']}/{a['n']} failed") + rel)
            ds_ = abl["delay_sweep"].get(case)
            tol_ = f"{max([x['delay_ms'] for x in ds_ if not x['exceeded_limits']] or [float('nan')]):.0f} ms" if ds_ else "–"
            w(f"| {case} | " + " | ".join(cells) + f" | {tol_} |")
        w("")
    else:
        w("*Not run yet.*\n")

    w("### 7.4 LLM supervising effort modes\n")
    if llm:
        s_ = llm["summary"]
        w(f"{llm['metadata']['model']} ({llm['metadata']['backend']}): {s_['decisions']} decisions, {s_['fallbacks']} "
          f"fallbacks, latency median {s_['latency_s']['median']:.2f} s (95th percentile {s_['latency_s']['p95']:.2f} s); "
          f"same mode as the rules for {100 * s_['agreement_with_rules']:.0f} % of reports.\n")
        for sc_, d in llm["scenarios"].items():
            w(f"- *{SCEN_LABEL.get(sc_, sc_)}*: {_stat_sentence(d, 'neural+llm vs neural', 'Neural + LLM', 'neural alone')}")
        w("")
    else:
        w("*Not run yet.*\n")

    w("### 7.5 Fault-tolerant supervision (uav_flap)\n")
    if ft:
        from .fault_tolerance.faults import FAULTS
        from .fault_tolerance.simulate import FT_MODES

        w("Modes: " + "; ".join(f"**{m}**: {_mode_text(spec)}" for m, spec in FT_MODES.items()) + ". Faults start 6 s into a "
          f"{ft['metadata']['duration_s']:.0f} s flight at 1.1 U_F in light turbulence and grow over 5 s:\n")
        for f, spec in FAULTS.items():
            if f != "none":
                w(f"- **{f}**{' (held out)' if spec['held_out'] else ''}: {spec['description']}.")
        faults = list(ft["faults"])
        modes = list(ft["faults"][faults[0]]["fixed_modes"])
        w("\nWhich fixed responses survive (applied 1 s after onset) and the oracle's least disruptive choice:\n")
        w("| Fault | " + " | ".join(modes) + " | Oracle |")
        w("|---" * (len(modes) + 2) + "|")
        for f in faults:
            d = ft["faults"][f]
            om = max(set(d["oracle_modes"]), key=d["oracle_modes"].count)
            w(f"| {f} | " + " | ".join(f"{d['fixed_modes'][m]['survival']:.0%}" for m in modes) + f" | {om} |")
        L.extend(_ft_supervisor_table(ft))
    else:
        w("*Running (`python -m flutter_ai ft-benchmark`); this section fills in automatically.*\n")

    w("### 7.6 Physics-informed supervision (pre-registered test)\n")
    w("The supervisors above see symptoms. Here the report also carries an on-board flutter-margin estimate: every "
      "second a grey-box fit of the last 4 s of flight identifies the wing's pitch stiffness and flap effectiveness "
      "(relative to the same fit at take-off, which absorbs the bias of the unmeasured turbulence) and predicts the "
      "identified wing's flutter speed and whether each mode's controller keeps it stable. In rules + physics the "
      "symptom rules decide whether to act and the estimate decides which modes are safe: a mode predicted unstable on "
      "the identified wing is never chosen, and if the current mode is predicted unstable the least disruptive "
      "predicted-stable mode is taken (hold speed first, then the smallest slow-down). The LLM receives the same "
      "estimate as a PHYSICS ESTIMATE block (llm + physics). The "
      "estimator was developed on seeds from 1000 up; the comparison, its supervisors and its seeds were fixed before "
      "the run, and the seeds had not been used before.\n")
    if phys:
        lat = phys["metadata"].get("llm_latency_fixed_s")
        w(f"Seeds {phys['metadata']['seeds'][0]}-{phys['metadata']['seeds'][-1]}"
          + (f"; the LLM's answers take effect after a fixed {lat:.2f} s of simulation time (the median latency "
             "measured in Section 7.5), so the outcome does not depend on the load of the machine" if lat else "")
          + (f"; run one fault per process ({len(phys['metadata']['pieces'])} pieces, merged with the benchmark's own "
             "statistics)" if phys["metadata"].get("pieces") else "") + ".")
        L.extend(_ft_supervisor_table(phys))
        w(f"- **How often each supervisor acted** (faulty flights with a mode change decided after onset): "
          f"{_acted(phys)}.")
        for a, b in (("rules+physics", "rules"), ("llm+physics", "llm"), ("llm+physics", "rules+physics"),
                     ("llm", "rules")):
            s_ = _pair_sentence(phys, a, b)
            if s_:
                w(f"- **{a} vs {b}:** {s_}. Per fault where they differ: {_changed_faults(phys, a, b)}.")
        w("")
    else:
        w("*Not run yet (seeds 100-119, supervisors none, rules, rules+physics, llm, llm+physics).*\n")

    w("### 7.7 A fast guardian: learned diagnosis with physics admissibility\n")
    w("In Section 7.5 the LLM answered in seconds and rarely changed mode. The guardian asks whether fast models can "
      "make that decision instead; it keeps the same mode menu, dwell time and fallback. Every 0.25 s it reads (a) "
      "a learned fault diagnoser (a small encoder over physics features of the last 2.5 s with a multi-label head for "
      "the four known faults and a reconstruction-error score for unknown faults, trained on simulated flights that "
      "never contain the two held-out faults), (b) the flutter-margin estimate extended with "
      "control authority (predicted flap command and rate against the actuator limits), turbulence, sensor-noise and "
      "loop-delay estimates and worst-case margins over the identification's confidence region, and (c) the symptom "
      "rules as a fallback. A confirmed known fault receives its pre-verified response, an unknown fault the most "
      "conservative mode; the chosen mode is the first of that response's preferences the physics admits (never one "
      "predicted unstable or saturating where the identified wing needs control); structural evidence holds speed unless "
      "the physics predicts the current mode unstable; it never de-escalates on its own. It uses no language model. "
      "Its thresholds were set on development seeds (1000-1009), and it was evaluated once on fresh seeds after the "
      "design was frozen.\n")
    if guard:
        L.extend(_ft_supervisor_table(guard))
        w(f"- **How often each supervisor acted** (faulty flights with a mode change decided after onset; holding "
          f"speed is the right response to stiffness loss): {_acted(guard)}.")
        for a, b in (("guardian", "rules"), ("guardian", "rules+physics"), ("guardian", "rules+physics_v2"),
                     ("guardian", "none")):
            s_ = _pair_sentence(guard, a, b)
            if s_:
                w(f"- **{a} vs {b}:** {s_}. Per fault where they differ: {_changed_faults(guard, a, b)}.")
        w("")
    else:
        w("*Not run yet (`ft-benchmark --first-seed 300 --seeds 20 --tag guardian_test300`).*\n")

    w("### 7.8 The language model off the control path\n")
    w("If the LLM should not choose modes, it may still help where time is not critical. Both roles below use the "
      "same safety wrapper (schema-constrained JSON re-checked in Python, a call budget, a timeout) and neither can "
      "change a decision.\n")
    if expl:
        s_ = expl["summary"]
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(s_["rejection_reasons"].items(), key=lambda x: -x[1]))
        w(f"**Explainer.** After a decision, {expl['metadata']['model']} reads the report (with the diagnosis and "
          "physics blocks) and the decision taken, and writes one or two sentences for the operator plus the fault and "
          "mode it refers to. An answer is shown only if it is faithful: its mode is the decision taken, its fault is "
          "the diagnosis's top flagged class (or unknown / none), its length is 20-400 characters, and it names no fault "
          "the diagnosis does not flag; otherwise a deterministic template is shown. The check covers these fields and "
          f"literal fault names, not every claim of the free text. Over {s_['reports']} reports from "
          f"{len(expl['metadata']['flights'])} development flights: {s_['llm_answers_valid']} valid answers, "
          f"{s_['faithful']} faithful"
          + (f" ({100 * s_['faithfulness_rate']:.0f} % of valid answers)" if s_["faithfulness_rate"] is not None else "")
          + f", {s_['llm_call_failures']} failed calls; the template was shown for {100 * s_['fallback_rate']:.0f} % "
          f"of reports; latency median {s_['latency_s']['median']:.1f} s"
          + (f". Rejection reasons: {reasons}." if reasons else ".") + "\n")
    else:
        w("*Explainer: not run yet (`python -m flutter_ai explainer-eval`).*\n")
    if rev:
        s_ = rev["summary"]
        g_ = rev["metadata"]["gate"]
        w(f"**Post-flight reviewer with a verification gate.** After a development flight in which the diagnoser "
          "raised a confirmed unknown-fault alarm, the LLM reads a compact log and proposes a fault from the catalogue "
          "with bounded severity and growth ranges. Flights are simulated for the hypothesis, a candidate diagnoser is "
          "trained with it as an extra class, and the candidate is accepted only if (1) no existing class loses more "
          f"than {g_['f1_tolerance']} window F1 on the original evaluation split and the healthy-flight alarm rate rises "
          f"by at most {g_['false_alarm_tolerance']}, and (2) on the anomalous flights the hypothesised class is "
          f"confirmed in at least {100 * g_['explain_class_min']:.0f} % of the windows after the first unknown alarm "
          f"and the unknown alarm in at most {100 * g_['explain_unknown_max']:.0f} %. Thresholds were fixed before any "
          "candidate was scored; hypotheses forced to be wrong serve as controls. "
          f"{s_['reviews']} reviews, {s_['valid_hypotheses']} valid hypotheses, {s_['right_kind']} naming the true "
          f"fault; latency median {s_['latency_s_median']:.1f} s.\n")
        w("| Hypothesis from | Anomaly (truth) | Hypothesis | Right fault | Gate |")
        w("|---|---|---|---|---|")
        for c in s_["gate"]:
            w(f"| {c['source']} | {c['anomaly']} | {c['hypothesis']} | {'yes' if c['right_kind'] else 'no'} | "
              f"{'accepted' if c['accepted'] else 'rejected'} |")
        for c in rev["candidates"]:
            if c.get("fresh"):
                fr = c["fresh"]
                cd, ua = fr["candidate"]["class_detected"], fr["v1"]["unknown_alarm"]
                w(f"- Accepted candidate {c['index']} ({c['hypothesis']['kind']} for {c['anomaly_kind']}), on "
                  f"{fr['candidate']['flights']} fresh {c['anomaly_kind']} flights: the new class confirmed in "
                  + (f"{100 * cd:.0f} %" if cd is not None else "–") + " of them; the original diagnoser's unknown "
                  "alarm in " + (f"{100 * ua:.0f} %" if ua is not None else "–") + ".")
        n_h = rev["v1_regression"]["healthy_flights"]
        w(f"\nThe false-alarm check counts healthy flights with any alarm on {n_h} flights (the original diagnoser: "
          f"{round(rev['v1_regression']['healthy_any_alarm_rate'] * n_h)}), so its tolerance of "
          f"{g_['false_alarm_tolerance']} allows at most {math.floor(g_['false_alarm_tolerance'] * n_h + 1e-9)} additional "
          "flight: at this sample size the gate is conservative. The gate checks consistency in the diagnoser's feature "
          "space, not physical truth: a wrong hypothesis whose flights look like the anomaly can pass, which is why the "
          "controls and the right-fault column are reported.\n")
    else:
        w("*Post-flight reviewer: not run yet (`python -m flutter_ai review-loop`).*\n")

    # ------------------------------------------------------------ discussion etc.
    w("## 8. Discussion\n")
    w("- **Which controller wins depends on what it knows and on the actuator.** Plain LQR has no disturbance model "
      "and amplifies gusts below flutter; given the same turbulence statistics the neural policy saw in training, "
      "LQG is the strongest inner loop in most conditions, and the neural policy keeps an edge only under loop delay "
      "and some actuator limits. Claims that learned control beats \"classical\" control depend on which classical "
      "baseline is used.")
    w("- **Latency decides what an LLM may do.** A decision every few seconds cannot help against a gust that acts "
      "within tenths of a second; an effort knob that barely moves the trade-off gives it nothing to choose. The "
      "same model is only useful for slow, persistent changes such as faults, and only through modes that were "
      "verified in advance.")
    for line in _ft_discussion(ft) + _later_discussion(phys, guard):
        w(line)
    w("- **Safety by construction.** The bounded mode set, schema-constrained answers, rule fallback, one question "
      "at a time and the dwell-time bound make the LLM's worst case analysable, independent of its reasoning.\n")
    w("## 9. Limitations\n")
    w("- Linear typical section (no freeplay or limit cycles); incompressible strip-theory aerodynamics (BACT runs "
      "at Mach 0.77); simulation only; no measured time histories (the identification pipeline is tested on "
      "synthetic data).")
    w("- Disturbance sizes, fault severities and the mode menu were set by the authors from calibration runs; the "
      "rules were written from the same principles as the LLM's guidance and not tuned on benchmark outcomes.")
    w("- One local model size (8B, quantised) on one laptop GPU; the cloud backend was not run. The oracle is an "
      "upper bound defined by hindsight and a preference for the least disruptive surviving mode.")
    w("- The stability guarantee is small-signal, for the nominal wing and switching at a fixed airspeed; faults and "
      "airspeed changes are outside it.")
    w("- The guardian's thresholds, fault responses and diagnoser were designed with the same simulator, fault "
      "catalogue and fault sizes it was tested on (on disjoint seeds); its transfer to other fault sizes, fault types "
      "or a real wing is untested. The LLM studies off the control path used development flights only.\n")
    w("## 10. Conclusion\n")
    w(_conclusion(res.get("uav_flap"), llm, ft, stab, later) + "\n")

    # ------------------------------------------------------------ reproducibility, figures, references
    w("## Reproducibility\n")
    w(f"Code: flutter_ai {__version__}, commit {_git_commit()}. Commands: `tune`, `evaluate`, `robustness`, "
      "`ablations`, `llm-benchmark`, `ft-benchmark`, `ft-stability`, `validate`, then `paper`. Model files:")
    for f in sorted([*MODELS_DIR.glob("neural_policy_*.npz"), *MODELS_DIR.glob("classical_tuning_*.json")]):
        w(f"- `{f.name}` sha256 {hashlib.sha256(f.read_bytes()).hexdigest()[:12]}")
    w("")
    if copy_figures:
        figs = _copy_figures()
        if figs:
            w("## Figures\n")
            for i, (path, caption) in enumerate(figs, start=1):
                w(f"**Figure {i}.** {caption}\n\n![Figure {i}]({path})\n")
    w("## References\n")
    L.extend(cite.listing())
    todo = cite.to_check()
    if todo:
        w("\n## Appendix: citation check (Rules.md §1.7)\n")
        w("Read these in full before submission (only their abstracts were read, or they were not opened):\n")
        L.extend(todo)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path
