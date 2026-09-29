"""
Command-line interface.

    python -m flutter_ai analyze   [--preset uav_flap]      flutter speed, p-k validation, V-g figure
    python -m flutter_ai train     [--preset uav_flap]      train the neural policy (needs PyTorch)
    python -m flutter_ai evaluate  [--preset uav_flap]      benchmark -> outputs/<preset>/results.json + figures
    python -m flutter_ai tune      [--preset uav_flap]      tune LQR/LQG design knobs -> models/classical_tuning_*.json
    python -m flutter_ai robustness [--preset uav_flap]     model error, sensor noise, rate limit -> robustness.json
    python -m flutter_ai ablations [--preset uav_flap]      ablation study (train variants with train --variant)
    python -m flutter_ai llm-benchmark [--preset uav_flap]  LLM supervisor experiments (slow) -> llm_benchmark.json
    python -m flutter_ai ft-benchmark [--preset uav_flap]   fault-tolerant supervision study -> ft_benchmark.json
    python -m flutter_ai ft-stability [--preset uav_flap]   dwell-time bound for switching -> ft_stability.json
    python -m flutter_ai serve     [--port 8000]            web app at http://127.0.0.1:8000
    python -m flutter_ai validate                           compare with NASA BACT wind-tunnel flutter data
    python -m flutter_ai check-data <folder>                validate an experimental dataset
    python -m flutter_ai identify <folder>                  identify physical parameters from its time histories
    python -m flutter_ai paper                              paper draft from the result files -> docs/research
    python -m flutter_ai all                                analyze + train (if missing) + evaluate
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser


def load_env_file() -> None:
    """Load KEY=VALUE lines from <project>/.env into the environment (existing variables win)."""
    from .paths import PROJECT_ROOT

    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _analyze(args) -> None:
    from .flutter import flutter_analysis, flutter_speed, pk_flutter_speed
    from .params import get_preset
    from .paths import OUTPUTS_DIR

    p = get_preset(args.preset)
    fa = flutter_analysis(p)
    Uf, Uf_pk = flutter_speed(p), pk_flutter_speed(p)
    u_ref = p.b * p.omega_alpha
    print(f"Preset: {p.name} - {p.description}")
    print(f"  mass ratio mu = {p.mass_ratio:.1f}, f_h = {p.f_h:.2f} Hz, f_alpha = {p.f_alpha:.2f} Hz")
    print(f"  Flutter speed (state-space, Jones/Wagner): {Uf:.3f} m/s  (U/(b w_a) = {Uf / u_ref:.4f})")
    if Uf_pk is None:
        print("  Flutter speed (p-k, exact Theodorsen)    : none found in the p-k sweep range")
    else:
        print(f"  Flutter speed (p-k, exact Theodorsen)    : {Uf_pk:.3f} m/s  (U/(b w_a) = {Uf_pk / u_ref:.4f})")
        print(f"  Agreement: {100 * (Uf - Uf_pk) / Uf_pk:+.2f}%")
    print(f"  Flutter frequency: {fa.flutter_frequency_hz:.2f} Hz")
    print(f"  Divergence speed : {fa.divergence_speed if fa.divergence_speed else 'none in range'}")
    from . import plots

    path = plots.flutter_figure(p, fa, Uf_pk, OUTPUTS_DIR / p.name / "fig1_flutter_vg.png")
    print(f"Saved {path}")


def _train(args) -> None:
    from .params import get_preset
    from .training import TrainConfig, train_policy

    from .ablations import ABLATION_VARIANTS, ablation_policy_path

    p = get_preset(args.preset)
    cfg = TrainConfig.for_preset(p)
    variant = getattr(args, "variant", "full")
    for key, value in ABLATION_VARIANTS[variant].items():
        setattr(cfg, key, value)
    if args.quick:
        cfg.imitation_iters, cfg.finetune_iters = 300, 60
    train_policy(p, cfg, out_path=None if variant == "full" else ablation_policy_path(p.name, variant))


def _evaluate(args) -> None:
    from .experiments import run_benchmark
    from .params import get_preset

    run_benchmark(get_preset(args.preset), seeds=tuple(range(args.seeds)))


def _tune(args) -> None:
    from .params import get_preset
    from .tuning import tune_classical

    tune_classical(get_preset(args.preset))


def _robustness(args) -> None:
    from .params import get_preset
    from .robustness import robustness_study

    robustness_study(get_preset(args.preset), n_draws=args.draws)


def _ablations(args) -> None:
    from .ablations import run_ablations
    from .params import get_preset

    run_ablations(get_preset(args.preset), seeds=tuple(range(args.seeds)))


def _llm_benchmark(args) -> None:
    from .llm_benchmark import run_llm_benchmark
    from .params import get_preset

    run_llm_benchmark(get_preset(args.preset), seeds=tuple(range(args.seeds)))


def _ft_benchmark(args) -> None:
    from .fault_tolerance.benchmark import run_ft_benchmark
    from .params import get_preset

    sups = tuple(args.supervisors.split(",")) if args.supervisors else ("none", "rules", "random", "llm")
    if args.no_llm:
        sups = tuple(s for s in sups if s != "llm")
    run_ft_benchmark(get_preset(args.preset), seeds=tuple(range(args.seeds)), supervisors=sups,
                     apply_latency=not args.no_latency, supervisor_interval=args.interval, tag=args.tag)


def _ft_stability(args) -> None:
    from .fault_tolerance.stability import run_stability
    from .params import get_preset

    run_stability(get_preset(args.preset))


def _serve(args) -> None:
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Flutter AI running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("flutter_ai.api.app:app", host=args.host, port=args.port, reload=args.reload, log_level="warning")


def _validate(args) -> None:
    from .validation import validate_bact

    validate_bact()


def _check_data(args) -> None:
    from .data import check_dataset

    report = check_dataset(args.folder)
    print(report)
    if not report.ok:
        sys.exit(1)


def _identify(args) -> None:
    from .identification import identify_dataset

    identify_dataset(args.folder, tuple(args.params.split(",")), args.delay)


def _paper(args) -> None:
    from .paper import build

    print(f"Saved {build()}")


def _all(args) -> None:
    from .controllers import policy_path

    _analyze(args)
    if not policy_path(args.preset).exists() or args.retrain:
        _train(args)
    _evaluate(args)


def main(argv: list[str] | None = None) -> int:
    from .params import DEFAULT_PRESET, PRESETS

    parser = argparse.ArgumentParser(prog="flutter_ai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS))
        sp.set_defaults(fn=fn)
        return sp

    add("analyze", _analyze, "flutter analysis and validation")
    sp = add("train", _train, "train the neural policy (needs PyTorch)")
    sp.add_argument("--quick", action="store_true", help="short training run for testing")
    sp.add_argument("--variant", default="full", choices=["full", "imitation_only", "no_rate_penalty", "no_airspeed"],
                    help="ablation variant (saved under models/ablations/)")
    sp = add("evaluate", _evaluate, "run the benchmark and write results + figures")
    sp.add_argument("--seeds", type=int, default=20)
    add("tune", _tune, "tune the classical baselines' design knobs for the real actuator")
    sp = add("robustness", _robustness, "robustness study: model error, sensor noise, actuator rate limit")
    sp.add_argument("--draws", type=int, default=20)
    sp = add("ablations", _ablations, "ablation study of the neural policy and the supervisor")
    sp.add_argument("--seeds", type=int, default=20)
    sp = add("llm-benchmark", _llm_benchmark, "LLM supervisor experiments (slow; uses FLUTTER_AI_LLM_BACKEND)")
    sp.add_argument("--seeds", type=int, default=5)
    sp = add("ft-benchmark", _ft_benchmark, "fault-tolerant supervision study (slow with the LLM)")
    sp.add_argument("--seeds", type=int, default=20)
    sp.add_argument("--no-llm", action="store_true", help="skip the LLM supervisor")
    sp.add_argument("--supervisors", default="", help="comma-separated subset, e.g. none,rules,llm")
    sp.add_argument("--no-latency", action="store_true",
                    help="ablation: apply answers at once (simulation time stops while the LLM thinks)")
    sp.add_argument("--interval", type=float, default=0.25, help="seconds between supervisor questions")
    sp.add_argument("--tag", default="", help="write ft_benchmark_<tag>.json instead of ft_benchmark.json")
    add("ft-stability", _ft_stability, "dwell-time stability bound for supervisor switching")
    sp = add("serve", _serve, "start the web app")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    sp.add_argument("--no-browser", action="store_true")
    sp = sub.add_parser("validate", help="compare the model with NASA BACT wind-tunnel flutter data")
    sp.set_defaults(fn=_validate)
    sp = sub.add_parser("check-data", help="validate an experimental dataset folder (see data/README.md)")
    sp.add_argument("folder")
    sp.set_defaults(fn=_check_data)
    sp = sub.add_parser("identify", help="identify physical parameters from a dataset's time histories")
    sp.add_argument("folder")
    sp.add_argument("--params", default="f_h,f_alpha,zeta_h,zeta_alpha", help="comma-separated parameters")
    sp.add_argument("--delay", type=int, default=1, help="loop delay (samples) between recorded command and actuator")
    sp.set_defaults(fn=_identify)
    sp = sub.add_parser("paper", help="generate docs/research/paper-draft.md from the result files")
    sp.set_defaults(fn=_paper)
    sp = add("all", _all, "analyze + train (if needed) + evaluate")
    sp.add_argument("--seeds", type=int, default=20)
    sp.add_argument("--retrain", action="store_true")
    sp.add_argument("--quick", action="store_true")

    args = parser.parse_args(argv)
    load_env_file()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
