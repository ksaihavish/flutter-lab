# Record of how outputs/uav_flap/ft_benchmark_physics_test100.json was produced (2026-09-30). Run from a checkout of
# commit 3a9d325 (the pre-registered snapshot) with PYTHONPATH=src; it is not meant to run against later code.
"""
Crash-resilient driver for the pre-registered five-way comparison (snapshot 3a9d325, seeds 100-119).

The first attempt (2026-09-30 01:01 IST, one process, all faults) was killed by Claude Code's low-memory reaper after
the 'none' fault; that fault's file part is complete. This driver runs the same function with the same arguments one
fault at a time (`faults=(kind,)`, tag physics_test100_<kind>), skips pieces that are already complete, and then merges
the pieces with the snapshot's own `_save` (same pooled statistics). No flight, supervisor or statistic differs from a
single run; only the process boundary does.

    python run_pieces.py run      # resumable
    python run_pieces.py merge    # -> outputs/uav_flap/ft_benchmark_physics_test100.json (+ figure)
"""

import json
import sys
import time
from pathlib import Path

from flutter_ai.fault_tolerance.benchmark import _save, run_ft_benchmark
from flutter_ai.fault_tolerance.faults import FAULTS
from flutter_ai.params import get_preset
from flutter_ai.paths import OUTPUTS_DIR

OUT = OUTPUTS_DIR / "uav_flap"
TAG = "physics_test100"
SUPERVISORS = ("none", "rules", "rules+physics", "llm", "llm+physics")
SEEDS = tuple(range(100, 120))
MAIN = Path("C:/Users/lio48/dev/flutter-lab/outputs/uav_flap/ft_benchmark.json")


def piece(kind: str) -> Path:
    return OUT / f"ft_benchmark_{TAG}_{kind}.json"


def first_attempt_none() -> None:
    """The killed first attempt finished 'none' completely: keep it as that fault's piece."""
    first = OUT / f"ft_benchmark_{TAG}.json"
    if piece("none").exists() or not first.exists():
        return
    d = json.loads(first.read_text(encoding="utf-8"))
    if list(d["faults"]) == ["none"] and d["metadata"]["seeds"] == list(SEEDS):
        d["metadata"]["note_pieces"] = "first attempt (killed after this fault by a low-memory reaper), kept as is"
        piece("none").write_text(json.dumps(d, indent=2))


def warm() -> None:
    """Load the model with the benchmark's options before a piece (a cold load under memory pressure timed out)."""
    import urllib.request
    body = json.dumps({"model": "llama3.1:8b", "prompt": "ok", "stream": False, "keep_alive": "10m",
                       "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 1}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    print(f"  model warm ({time.time() - t:.1f} s)", flush=True)


def fallbacks(path: Path) -> int:
    """LLM answers replaced by the rule fallback (timeouts, invalid answers) in a finished piece."""
    d = json.loads(path.read_text(encoding="utf-8"))
    return sum(1 for f in d["faults"].values() for x in f["decisions"]
               if x["supervisor"].startswith("llm") and x["source"] != "llm")


def run() -> None:
    lat = json.loads(MAIN.read_text(encoding="utf-8"))["llm_latency_s"]["median"]
    print(f"fixed LLM latency {lat} s (median of the main benchmark)", flush=True)
    first_attempt_none()
    p = get_preset("uav_flap")
    for kind in FAULTS:
        if piece(kind).exists():
            print(f"  {kind}: piece exists, skipped", flush=True)
            continue
        for attempt in (1, 2):
            warm()
            run_ft_benchmark(p, seeds=SEEDS, supervisors=SUPERVISORS, tag=f"{TAG}_{kind}", faults=(kind,),
                             llm_latency_s=lat, make_figure=False)
            n = fallbacks(piece(kind))
            print(f"  {kind}: LLM fallbacks {n} (attempt {attempt})", flush=True)
            if n == 0 or attempt == 2:
                break
            # an infrastructure failure (not the model's judgement): keep the file, retry the piece once
            piece(kind).rename(piece(kind).with_name(piece(kind).stem + f"_attempt{attempt}_fallbacks{n}.json"))


def merge() -> None:
    parts = {k: json.loads(piece(k).read_text(encoding="utf-8")) for k in FAULTS if piece(k).exists()}
    missing = [k for k in FAULTS if k not in parts]
    if missing:
        sys.exit(f"missing pieces: {missing}")
    first = parts["none"]
    res = {"metadata": dict(first["metadata"]), "faults": {}}
    res["metadata"].update(tag=TAG, file=f"ft_benchmark_{TAG}.json", pieces={
        k: {"generated_at": v["metadata"]["generated_at"], "wall_time_s": v["metadata"]["wall_time_s"]}
        for k, v in parts.items()},
        note_pieces="run one fault per process (run_pieces.py) after the first attempt was killed; merged with "
                    "the snapshot's _save")
    everything: dict[str, list[dict]] = {}
    for k in FAULTS:
        res["faults"][k] = parts[k]["faults"][k]
        if k != "none":
            for sup, ms in res["faults"][k]["per_seed"].items():
                everything.setdefault(sup, []).extend(ms)
    wall = sum(v["metadata"]["wall_time_s"] for v in parts.values())
    _save(res, everything, OUT, time.time() - wall)
    from flutter_ai import plots
    print("Saved", OUT / res["metadata"]["file"], "and", plots.ft_figure(res, OUT / f"fig12_fault_tolerance_{TAG}.png"))


if __name__ == "__main__":
    {"run": run, "merge": merge}[sys.argv[1]]()
