"""
Command-line interface.

    python -m flutter_ai analyze   [--preset uav_flap]      flutter speed, p-k validation, V-g figure
    python -m flutter_ai train     [--preset uav_flap]      train the neural policy (needs PyTorch)
    python -m flutter_ai evaluate  [--preset uav_flap]      benchmark -> outputs/<preset>/results.json + figures
    python -m flutter_ai serve     [--port 8000]            web app at http://127.0.0.1:8000
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
    from .flutter import flutter_analysis, pk_flutter_speed
    from .params import get_preset
    from .paths import OUTPUTS_DIR

    p = get_preset(args.preset)
    fa = flutter_analysis(p)
    Uf_pk = pk_flutter_speed(p)
    u_ref = p.b * p.omega_alpha
    print(f"Preset: {p.name} - {p.description}")
    print(f"  mass ratio mu = {p.mass_ratio:.1f}, f_h = {p.f_h:.2f} Hz, f_alpha = {p.f_alpha:.2f} Hz")
    print(f"  Flutter speed (state-space, Jones/Wagner): {fa.flutter_speed:.3f} m/s  (U/(b w_a) = {fa.flutter_speed / u_ref:.4f})")
    print(f"  Flutter speed (p-k, exact Theodorsen)    : {Uf_pk:.3f} m/s  (U/(b w_a) = {Uf_pk / u_ref:.4f})")
    print(f"  Agreement: {100 * (fa.flutter_speed - Uf_pk) / Uf_pk:+.2f}%")
    print(f"  Flutter frequency: {fa.flutter_frequency_hz:.2f} Hz")
    print(f"  Divergence speed : {fa.divergence_speed if fa.divergence_speed else 'none in range'}")
    from . import plots

    path = plots.flutter_figure(p, fa, Uf_pk, OUTPUTS_DIR / p.name / "fig1_flutter_vg.png")
    print(f"Saved {path}")


def _train(args) -> None:
    from .params import get_preset
    from .training import TrainConfig, train_policy

    p = get_preset(args.preset)
    cfg = TrainConfig.for_preset(p)
    if args.quick:
        cfg.imitation_iters, cfg.finetune_iters = 300, 60
    train_policy(p, cfg)


def _evaluate(args) -> None:
    from .experiments import run_benchmark
    from .params import get_preset

    run_benchmark(get_preset(args.preset), seeds=tuple(range(args.seeds)))


def _serve(args) -> None:
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Flutter AI running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("flutter_ai.api.app:app", host=args.host, port=args.port, reload=args.reload, log_level="warning")


def _check_data(args) -> None:
    from .data import check_dataset

    report = check_dataset(args.folder)
    print(report)
    if not report.ok:
        sys.exit(1)


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
    sp = add("evaluate", _evaluate, "run the benchmark and write results + figures")
    sp.add_argument("--seeds", type=int, default=5)
    sp = add("serve", _serve, "start the web app")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    sp.add_argument("--no-browser", action="store_true")
    sp = sub.add_parser("check-data", help="validate an experimental dataset folder (see data/README.md)")
    sp.add_argument("folder")
    sp.set_defaults(fn=_check_data)
    sp = add("all", _all, "analyze + train (if needed) + evaluate")
    sp.add_argument("--seeds", type=int, default=5)
    sp.add_argument("--retrain", action="store_true")
    sp.add_argument("--quick", action="store_true")

    args = parser.parse_args(argv)
    load_env_file()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
