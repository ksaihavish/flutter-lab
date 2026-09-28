"""
Train the neural feedback policy through the differentiable physics model.

Method ("physics-trained" / differentiable-simulation policy optimisation):
  1. Warm start by imitating the discrete, delay-aware LQR across airspeeds and
     aggressiveness levels (fast, stable initialisation).
  2. Fine-tune by back-propagating the closed-loop cost through rollouts of the exact
     ZOH-discretised aeroelastic model, with actuator saturation, the on-board aero-lag
     observer, and domain randomisation over initial conditions, airspeed, turbulence
     intensity and **input delay (0-2 steps)**. The cost is the LQR's quadratic cost plus
     a control-rate penalty; the best checkpoint on held-out episodes is kept.

History of what went wrong (see docs/Memory.md): without the rate penalty the policy
chattered at the sampling rate; with a perfect-timing plant it learned high-gain
behaviour that went unstable with one extra sample of delay. Hence rate penalty +
delay randomisation + the previous command as an input.

Requires PyTorch (pip install -e ".[train]").
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .aero import JONES_EPS
from .controllers.base import effort_cost_multiplier
from .controllers.costs import input_weight, reference_scales, state_weight
from .controllers.lqr import lqr_gain
from .controllers.neural import POLICY_FORMAT, U_RATIO_SCALE, NeuralPolicy, policy_path, state_scales
from .flutter import flutter_speed
from .model import discrete_state_space, downwash_rows, n_inputs, n_states
from .params import WingParams
from .paths import MODELS_DIR


@dataclass
class TrainConfig:
    velocity_ratio_range: tuple[float, float] = (0.8, 1.4)
    n_velocities: int = 13
    dt: float = 0.005
    delay_range: tuple[int, int] = (0, 2)  # input delay steps randomised during fine-tuning
    nominal_delay: int = 1
    hidden: tuple[int, ...] = (64, 64)
    batch_size: int = 256
    imitation_iters: int = 1500
    finetune_iters: int = 500
    horizon_start: int = 100  # rollout steps; grows linearly to horizon_end
    horizon_end: int = 300
    lr_imitation: float = 3e-3
    lr_finetune: float = 5e-4
    turbulence_rms_max: float = 2.0  # [m/s]
    rate_weight: float = 1.0  # penalty on (du / u_max)^2 per step (actuator bandwidth)
    ic_scale: float = 1.0  # scales the random initial conditions of fine-tuning episodes
    seed: int = 0

    @classmethod
    def for_preset(cls, p: WingParams) -> "TrainConfig":
        """
        Train on conditions the actuator can physically recover from. A +/-15 deg flap has far
        less authority than the idealised force actuators; on the default distribution ~20 % of
        episodes were unrecoverable even for LQR, which wastes the gradient.
        """
        if p.name == "bact":  # NASA demonstrated suppression to ~1.16 U_F; weak 30 %-span flap, quiet tunnel
            return cls(velocity_ratio_range=(0.8, 1.2), n_velocities=9, turbulence_rms_max=0.5, ic_scale=0.3,
                       finetune_iters=800)
        if p.actuation == "flap_servo":
            return cls(velocity_ratio_range=(0.8, 1.3), n_velocities=11, turbulence_rms_max=1.5, finetune_iters=800)
        return cls()


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError('Training needs PyTorch. Install with:  pip install -e ".[train]"') from exc
    return torch


def history_path(preset: str) -> Path:
    return MODELS_DIR / f"training_history_{preset}.json"


def train_policy(p: WingParams, cfg: TrainConfig | None = None, out_path: Path | None = None, log=print) -> NeuralPolicy:
    torch = _torch()
    cfg = cfg or TrainConfig.for_preset(p)
    out_path = Path(out_path or policy_path(p.name))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    t_start = time.time()

    n, nx, m = p.n_dof, n_states(p), n_inputs(p)
    Uf = flutter_speed(p)
    ratios = np.linspace(*cfg.velocity_ratio_range, cfg.n_velocities)
    velocities = ratios * Uf
    log(f"Preset '{p.name}' ({n}-DOF, inputs {list(p.input_names)}); flutter speed {Uf:.2f} m/s; "
        f"training on {cfg.n_velocities} airspeeds {velocities[0]:.1f}-{velocities[-1]:.1f} m/s, "
        f"delays {cfg.delay_range[0]}-{cfg.delay_range[1]} steps")

    # ---- plant, observer and LQR data per airspeed -------------------------
    Ad, Bd, Gd, obs_decay, obs_gain, w_rows = [], [], [], [], [], []
    for U in velocities:
        d = discrete_state_space(p, U, cfg.dt)
        Ad.append(d.Ad), Bd.append(d.Bd), Gd.append(d.Gd[:, 0])
        lam = np.array(JONES_EPS) * U / p.b
        decay = np.exp(-lam * cfg.dt)
        obs_decay.append(decay), obs_gain.append((1 - decay) / lam)
        dq, dqd = downwash_rows(p, U)
        w_rows.append(np.concatenate([dq, dqd]))
    agg_grid = np.linspace(0.0, 1.0, 11)
    # Delay-1 LQR: u_k = -K [x_k, u_{k-1}]  -> exactly the policy's inputs
    K = np.array([[lqr_gain(p, U, cfg.dt, a, delay=1) for a in agg_grid] for U in velocities])  # (V, A, m, nx+m)

    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    Ad_t, Bd_t, Gd_t = T(Ad), T(Bd), T(Gd)
    obs_decay_t, obs_gain_t, w_rows_t = T(obs_decay), T(obs_gain), T(w_rows)
    K_t = T(K)
    scales = state_scales(p)
    scales_t = T(scales)
    u_max = np.asarray(p.u_max, dtype=float)
    u_max_t = T(u_max)
    Q_t = T(state_weight(p))
    R_base_t = T(input_weight(p, 0.5) / effort_cost_multiplier(0.5))
    ratios_t = T(ratios)
    ref = reference_scales(p)
    lim_t = T([p.h_limit, np.radians(p.alpha_limit_deg)])

    # ---- network ------------------------------------------------------------
    layers: list = []
    prev = nx + m + 2
    for h in cfg.hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.Tanh()]
        prev = h
    layers += [torch.nn.Linear(prev, m)]
    net = torch.nn.Sequential(*layers)

    def features(x, u_prev, v_idx, a):
        return torch.cat([x / scales_t, u_prev / u_max_t, ((ratios_t[v_idx] - 1.0) / U_RATIO_SCALE)[:, None],
                          (2 * a - 1)[:, None]], dim=1)

    def policy_offset(v_idx, a):
        """g(0, 0) per sample; constant during a rollout, so it is computed once."""
        B = len(v_idx)
        return net(features(torch.zeros(B, nx), torch.zeros(B, m), v_idx, a))

    def policy(x, u_prev, v_idx, a, g0=None):
        # zero-centred: exactly zero command at the equilibrium, saturation via tanh
        if g0 is None:
            g0 = policy_offset(v_idx, a)
        return u_max_t * torch.tanh(net(features(x, u_prev, v_idx, a)) - g0)

    history: dict = {"imitation": [], "finetune": [], "horizon": []}

    # ---- 1. imitation of the delay-aware LQR (saturated) --------------------
    q_mult = np.concatenate([np.full(n, 3.0), np.full(n, 2.0)])
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr_imitation)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.imitation_iters)
    log(f"[1/2] Imitation warm start: {cfg.imitation_iters} iterations")
    for it in range(cfg.imitation_iters):
        B = cfg.batch_size * 4
        v_idx = T(rng.integers(0, len(velocities), B)).long()
        a_idx = T(rng.integers(0, len(agg_grid), B)).long()
        a = T(agg_grid)[a_idx]
        x = np.zeros((B, nx))
        x[:, :2 * n] = rng.uniform(-1, 1, (B, 2 * n)) * ref * q_mult
        x[:, 2 * n:] = rng.uniform(-1, 1, (B, 2)) * scales[2 * n:] * 2.0
        x = T(x)
        u_prev = T(rng.uniform(-1, 1, (B, m)) * u_max)
        target = torch.clamp(-(K_t[v_idx, a_idx] @ torch.cat([x, u_prev], 1)[:, :, None])[:, :, 0], -u_max_t, u_max_t)
        loss = torch.mean(((policy(x, u_prev, v_idx, a) - target) / u_max_t) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history["imitation"].append(loss.item())
        if (it + 1) % 250 == 0:
            log(f"   iter {it + 1:4d}  imitation MSE {loss.item():.5f}")

    # ---- 2. differentiable-simulation fine-tuning -------------------------
    phi = float(np.exp(-cfg.dt / 0.05))  # AR(1) turbulence correlation per step
    d_max = cfg.delay_range[1]
    x0_mult = np.concatenate([[1.5, 1.5], np.full(n - 2, 0.5), np.full(n, 0.5)])

    def sample_episodes(B, horizon, gen, delay=None):
        """Initial conditions within +/-1.5 reference scales (recoverable with saturated actuators)."""
        x0 = np.zeros((B, nx))
        x0[:, :2 * n] = gen.uniform(-1, 1, (B, 2 * n)) * ref * x0_mult * cfg.ic_scale
        rms = gen.uniform(0, cfg.turbulence_rms_max, B)
        noise = gen.standard_normal((horizon, B)) * rms * np.sqrt(1 - phi**2)
        d = np.full(B, delay) if delay is not None else gen.integers(cfg.delay_range[0], d_max + 1, B)
        return (T(gen.integers(0, len(velocities), B)).long(), T(gen.uniform(0, 1, B)), T(x0), T(noise),
                torch.tensor(d, dtype=torch.long))

    def rollout(episodes, horizon, control_fn, return_rate=False):
        """Per-episode cost J_i (mean over time): LQ cost + rate penalty + soft structural-limit barrier."""
        v_idx, a, x, noise, d_idx = episodes
        B = x.shape[0]
        z_hat = torch.zeros(B, 2)
        w = torch.zeros(B)
        H = torch.zeros(B, d_max + 1, m)  # H[:, 0] newest command; applied input = H[:, delay]
        R_eff = R_base_t[None] * (10.0 ** (1.0 - 2.0 * a))[:, None, None]
        J = torch.zeros(B)
        rate_sq = torch.zeros(B)
        rows = torch.arange(B)
        g0 = policy_offset(v_idx, a) if control_fn is policy else None
        for k in range(horizon):
            x_hat = torch.cat([x[:, :2 * n], z_hat], dim=1)
            u_prev = H[:, 0]
            u = control_fn(x_hat, u_prev, v_idx, a, g0)
            du = torch.sum(((u - u_prev) / u_max_t) ** 2, dim=1) if k else torch.zeros(B)
            rate_sq = rate_sq + du
            step = (torch.einsum("bi,ij,bj->b", x, Q_t, x) + torch.einsum("bi,bij,bj->b", u, R_eff, u)
                    + cfg.rate_weight * du)
            over = torch.relu(torch.abs(x[:, :2]) / lim_t - 0.8)
            J = J + step + 100.0 * torch.sum(over**2, dim=1)
            w_meas = torch.sum(w_rows_t[v_idx] * x[:, :2 * n], dim=1)  # observer sees measurements only
            z_hat = obs_decay_t[v_idx] * z_hat + obs_gain_t[v_idx] * w_meas[:, None]
            H = torch.cat([u[:, None], H[:, :-1]], dim=1)
            u_applied = H[rows, d_idx]
            w = phi * w + noise[k]
            x = (Ad_t[v_idx] @ x[:, :, None])[:, :, 0] + (Bd_t[v_idx] @ u_applied[:, :, None])[:, :, 0] \
                + Gd_t[v_idx] * w[:, None]
        if return_rate:
            return J / horizon, torch.sqrt(rate_sq / horizon)
        return J / horizon

    def lqr_control(x_hat, u_prev, v_idx, a, g0=None):
        a_idx = torch.round(a * (len(agg_grid) - 1)).long()
        z = torch.cat([x_hat, u_prev], 1)[:, :, None]
        return torch.clamp(-(K_t[v_idx, a_idx] @ z)[:, :, 0], -u_max_t, u_max_t)

    delays = list(range(cfg.delay_range[0], d_max + 1))
    val = {d: sample_episodes(256, cfg.horizon_end, np.random.default_rng(cfg.seed + 1000), delay=d) for d in delays}

    def validate(control_fn) -> dict:
        out = {}
        with torch.no_grad():
            for d in delays:
                J, rate = rollout(val[d], cfg.horizon_end, control_fn, return_rate=True)
                out[f"delay{d}"] = {"mean_log_cost": float(torch.mean(torch.log1p(J))),
                                    "median_cost": float(torch.median(J)),
                                    "unrecovered": float(torch.mean((J > 1e3).float())),
                                    "median_control_rate": float(torch.median(rate))}
        out["score"] = float(np.mean([out[f"delay{d}"]["mean_log_cost"] for d in delays]))
        return out

    def fmt(v: dict) -> str:
        return "  ".join(f"d{d}: logJ {v[f'delay{d}']['mean_log_cost']:.3f} unrec {100 * v[f'delay{d}']['unrecovered']:.1f}%"
                         for d in delays)

    lqr_val = validate(lqr_control)
    best = validate(policy)
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    log(f"[2/2] Differentiable-physics fine-tuning: {cfg.finetune_iters} iterations")
    log(f"   validation  LQR (delay-1 design): {fmt(lqr_val)}")
    log(f"   validation  after imitation     : {fmt(best)}")
    history["validation"] = [{"iter": 0, **best}]
    history["lqr_validation"] = lqr_val

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr_finetune)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, cfg.finetune_iters))
    for it in range(cfg.finetune_iters):
        frac = it / max(1, cfg.finetune_iters - 1)
        horizon = int(cfg.horizon_start + frac * (cfg.horizon_end - cfg.horizon_start))
        J = rollout(sample_episodes(cfg.batch_size, horizon, rng), horizon, policy)
        loss = torch.mean(torch.log1p(J))  # an unrecoverable episode cannot dominate the gradient
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        history["finetune"].append(loss.item())
        history["horizon"].append(horizon)
        if (it + 1) % 25 == 0 or it + 1 == cfg.finetune_iters:
            v = validate(policy)
            history["validation"].append({"iter": it + 1, **v})
            improved = v["score"] < best["score"]
            if improved:
                best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
            log(f"   iter {it + 1:4d}  horizon {horizon:3d}  train {loss.item():.3f}  val {fmt(v)}"
                f"{'  (best)' if improved else ''}")
    net.load_state_dict(best_state)
    log(f"   kept best checkpoint: {fmt(best)}")

    # ---- export (pure NumPy weights) --------------------------------------
    linears = [mod for mod in net if isinstance(mod, torch.nn.Linear)]
    weights = [(mod.weight.detach().numpy().astype(np.float64), mod.bias.detach().numpy().astype(np.float64))
               for mod in linears]
    meta = {
        "format": POLICY_FORMAT,
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset": p.name,
        "params_fingerprint": p.fingerprint(),
        "flutter_speed": Uf,
        "velocity_ratio_range": list(cfg.velocity_ratio_range),
        "delay_range": list(cfg.delay_range),
        "dt": cfg.dt,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "final_imitation_mse": history["imitation"][-1],
        "validation": best,
        "lqr_validation": lqr_val,
        "training_seconds": round(time.time() - t_start, 1),
    }
    policy_np = NeuralPolicy(weights, scales, u_max, meta)
    policy_np.save(out_path)
    history_path(p.name).write_text(json.dumps(history))
    log(f"Saved policy to {out_path} ({meta['training_seconds']} s)")
    return policy_np


def training_history(preset: str) -> dict | None:
    path = history_path(preset)
    return json.loads(path.read_text()) if path.exists() else None
