"""
Train the learned fault diagnoser (diagnoser.py) with PyTorch; the result is exported as NumPy weights.

    python -m flutter_ai train-diagnoser --preset uav_flap     # -> models/diagnoser_<preset>.npz + .json

Data (diagnoser_data): the `train` split for the weights, the `val` split for early stopping and every threshold. Only
the four known fault kinds; the held-out kinds never appear here (checked).

Loss: masked binary cross-entropy per known fault (windows where a fault has started but is still below its label
level are left out) + RECON_WEIGHT x the reconstruction error of the standardised features, through the shared
embedding. The reconstruction error is the unknown-fault score: the decoder only learns to reproduce healthy and
known-fault data.

Calibration, on validation flights only (all healthy or known faults):
  - probability threshold 0.5 per class (the natural threshold of a cross-entropy model; not tuned);
  - unknown-fault threshold: the smallest value at which at most OOD_FALSE_ALARM of the validation flights raise a
    confirmed unknown alarm (score above it at CONFIRM consecutive diagnoses); the same for the Mahalanobis score.
Requires PyTorch (pip install -e ".[train]").
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..params import WingParams
from .diagnoser import (
    CONFIRM, DIAGNOSE_EVERY_S, HELD_OUT, KNOWN, MODEL_FORMAT, WINDOW_S, DiagnoserModel, feature_names, model_path,
)
from .diagnoser_data import ALL_KINDS, SPLITS, Dataset, build_split, min_level

OOD_FALSE_ALARM = 0.05  # share of validation flights (no unknown fault) allowed to raise a confirmed unknown alarm


@dataclass
class DiagnoserConfig:
    encoder_hidden: tuple[int, ...] = (64, 32)
    embed: int = 8
    classifier_hidden: tuple[int, ...] = (32,)
    decoder_hidden: tuple[int, ...] = (32, 64)
    recon_weight: float = 0.5
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 25  # epochs without a better validation loss before stopping
    prob_threshold: float = 0.5
    seed: int = 0


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError('Training the diagnoser needs PyTorch. Install with:  pip install -e ".[train]"') from exc
    return torch


def metadata_path(preset: str) -> Path:
    return model_path(preset).with_suffix(".json")


def flight_max_confirmed(score: np.ndarray, flight: np.ndarray, n_flights: int, n: int = CONFIRM) -> np.ndarray:
    """Per flight, the largest score that held at n consecutive diagnoses (-inf for flights with fewer windows).
    An alarm with threshold tau is confirmed in a flight exactly when this exceeds tau."""
    out = np.full(n_flights, -np.inf)
    for i in range(n_flights):
        s = score[flight == i]
        if len(s) >= n:
            out[i] = float(np.max(np.min(np.lib.stride_tricks.sliding_window_view(s, n), axis=1)))
    return out


def calibrate_threshold(flight_max: np.ndarray, false_alarm: float = OOD_FALSE_ALARM) -> float:
    """Smallest threshold with at most `false_alarm` of the flights above it (flights without windows excluded)."""
    m = np.sort(flight_max[np.isfinite(flight_max)])
    allowed = int(np.floor(false_alarm * len(m)))
    return float(m[len(m) - 1 - allowed])


def _check_known_only(ds: Dataset, split: str) -> None:
    held = [ALL_KINDS.index(k) for k in HELD_OUT]
    kinds = {f["kind"] for fl in ds.flights for f in fl["spec"]["faults"]}
    if ds.present[:, held].any() or not kinds <= set(KNOWN):
        raise ValueError(f"Split '{split}' contains held-out faults {sorted(kinds - set(KNOWN))}: they must never be "
                         "used for training or calibration")


def window_scores(Y: np.ndarray, M: np.ndarray, pred: np.ndarray) -> dict:
    """Per-class precision / recall / F1 over the windows where the label is defined."""
    out = {}
    for j, c in enumerate(KNOWN):
        y, p = Y[M[:, j], j], pred[M[:, j], j]
        tp, fp, fn = int(np.sum(y & p)), int(np.sum(~y & p)), int(np.sum(y & ~p))
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "positives": int(y.sum()), "windows": int(len(y))}
    return out


def train_diagnoser(p: WingParams, cfg: DiagnoserConfig | None = None, workers: int = 4, log=print,
                    out_path: Path | None = None) -> DiagnoserModel:
    torch = _torch()
    cfg = cfg or DiagnoserConfig()
    t_start = time.time()
    log(f"Diagnoser for '{p.name}': flight data (seeds >= 1000; cached in outputs/{p.name}/diagnoser_data)")
    train, val = build_split(p, "train", workers, log), build_split(p, "val", workers, log)
    _check_known_only(train, "train")
    _check_known_only(val, "val")
    names = feature_names(p)
    Ytr, Mtr = train.targets()
    Yva, Mva = val.targets()
    mean, std = train.X.mean(axis=0), train.X.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)  # a feature that never varies in training passes through unscaled
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    Xtr_t, Ytr_t, Mtr_t = T((train.X - mean) / std), T(Ytr), T(Mtr)
    Xva_t, Yva_t, Mva_t = T((val.X - mean) / std), T(Yva), T(Mva)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    d, K = len(names), len(KNOWN)

    def mlp(sizes: list[int]):
        layers: list = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.Tanh()]
        return torch.nn.Sequential(*layers[:-1])  # linear output

    enc = mlp([d, *cfg.encoder_hidden, cfg.embed])
    cls = mlp([cfg.embed, *cfg.classifier_hidden, K])
    dec = mlp([cfg.embed, *cfg.decoder_hidden, d])
    nets = (enc, cls, dec)
    params = [q for net in nets for q in net.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def losses(X, Y, M):
        z = enc(X)
        bce = torch.sum(bce_fn(cls(z), Y) * M) / torch.clamp(torch.sum(M), min=1.0)
        return bce, torch.mean((dec(z) - X) ** 2)

    log(f"Training: {len(train.t)} windows from {len(train.flights)} flights; validation {len(val.t)} windows from "
        f"{len(val.flights)} flights; {d} features -> {cfg.embed}-d embedding -> {K} faults + reconstruction")
    history: list[dict] = []
    best, best_state, since = np.inf, None, 0
    for epoch in range(cfg.max_epochs):
        for net in nets:
            net.train()
        perm = rng.permutation(len(Xtr_t))
        tot = []
        for i in range(0, len(perm), cfg.batch_size):
            b = torch.as_tensor(perm[i:i + cfg.batch_size])
            bce, mse = losses(Xtr_t[b], Ytr_t[b], Mtr_t[b])
            loss = bce + cfg.recon_weight * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot.append(loss.item())
        with torch.no_grad():
            bce_v, mse_v = losses(Xva_t, Yva_t, Mva_t)
        score = bce_v.item() + cfg.recon_weight * mse_v.item()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(tot)), "val_bce": bce_v.item(),
                        "val_recon": mse_v.item(), "val_loss": score})
        if score < best - 1e-5:
            best, since = score, 0
            best_state = [{k: v.clone() for k, v in net.state_dict().items()} for net in nets]
        else:
            since += 1
        if (epoch + 1) % 10 == 0:
            log(f"   epoch {epoch + 1:3d}  train {np.mean(tot):.4f}  val bce {bce_v.item():.4f}  "
                f"recon {mse_v.item():.4f}{'  (best)' if since == 0 else ''}")
        if since >= cfg.patience:
            break
    assert best_state is not None
    for net, state in zip(nets, best_state):
        net.load_state_dict(state)
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    log(f"   kept epoch {best_epoch} (validation loss {best:.4f})")

    def export(net) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64))
                for m in net if isinstance(m, torch.nn.Linear)]

    # Mahalanobis reference: embedding means of the known fault combinations (windows with every label defined),
    # shared within-combination covariance
    with torch.no_grad():
        Z = enc(Xtr_t).numpy().astype(np.float64)
        np_check = torch.sigmoid(cls(enc(Xva_t))).numpy()
    full = np.all(Mtr, axis=1)
    combos = {tuple(row) for row in Ytr[full]}
    cents, resid = [], []
    for cmb in sorted(combos):
        sel = full & np.all(Ytr == np.array(cmb), axis=1)
        if sel.sum() >= 20:
            mu = Z[sel].mean(axis=0)
            cents.append(mu)
            resid.append(Z[sel] - mu)
    R = np.vstack(resid)
    precision = np.linalg.inv(R.T @ R / len(R) + 1e-6 * np.eye(cfg.embed))
    meta: dict = {
        "format": MODEL_FORMAT, "version": __version__, "preset": p.name, "params_fingerprint": p.fingerprint(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "classes": list(KNOWN),
        "held_out_never_seen": list(HELD_OUT), "feature_names": list(names), "window_s": WINDOW_S,
        "diagnose_every_s": DIAGNOSE_EVERY_S, "confirm": CONFIRM, "prob_threshold": cfg.prob_threshold,
        "label_min_level": {k: min_level(k) for k in KNOWN}, "combinations_in_embedding": len(cents),
    }
    model = DiagnoserModel(export(enc), export(cls), export(dec), mean, std, np.array(cents), precision, meta)
    prob_va, rec_va, maha_va = model.predict(val.X)
    if not np.allclose(prob_va, np_check, atol=1e-4):  # the NumPy export must be the trained network
        raise RuntimeError("NumPy diagnoser disagrees with the torch network; the export is broken")

    # thresholds from validation flights only
    n_va = len(val.flights)
    meta["ood_threshold"] = calibrate_threshold(flight_max_confirmed(rec_va, val.flight, n_va))
    meta["mahalanobis_threshold"] = calibrate_threshold(flight_max_confirmed(maha_va, val.flight, n_va))
    val_scores = window_scores(Yva, Mva, prob_va >= cfg.prob_threshold)
    out_path = Path(out_path or model_path(p.name))
    model.save(out_path)
    info = meta | {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "ood_false_alarm_target": OOD_FALSE_ALARM,
        "data": {s: {"first_seed": SPLITS[s]["first_seed"], "flights": len(ds.flights), "windows": int(len(ds.t)),
                     "failed_flights": sum(f["failed"] for f in ds.flights),
                     "flights_without_windows": sum(f["n_windows"] == 0 for f in ds.flights),
                     "positive_windows": {c: int(Y[:, j].sum()) for j, c in enumerate(KNOWN)}}
                 for s, ds, Y in (("train", train, Ytr), ("val", val, Yva))},
        "best_epoch": best_epoch, "validation_window_scores": val_scores, "history": history,
        "parameters": int(sum(W.size + b.size for net in (model.encoder, model.classifier, model.decoder)
                              for W, b in net)),
        "training_seconds": round(time.time() - t_start, 1),
        "note": "Every value in this file is produced by flutter_ai.fault_tolerance.diagnoser_training.",
    }
    meta_file = metadata_path(p.name) if out_path == model_path(p.name) else out_path.with_suffix(".json")
    meta_file.write_text(json.dumps(info, indent=2) + "\n")
    log("   validation F1: " + ", ".join(f"{c} {v['f1']:.2f}" for c, v in val_scores.items() if v["f1"] is not None))
    log(f"   unknown-fault threshold {meta['ood_threshold']:.3f} (reconstruction), "
        f"{meta['mahalanobis_threshold']:.2f} (Mahalanobis, secondary)")
    log(f"Saved {out_path} and {meta_file} ({info['parameters']} parameters, {info['training_seconds']:.0f} s)")
    return model
