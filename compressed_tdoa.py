"""Learnable compressed-domain TDOA likelihood models.

V5-A intentionally leaves the waveform-reconstruction DAE route.  Each UAV
transmits a strict CR=16 budget of 64 complex compressed features.  Pairwise
TDOA is estimated from a physics-constrained likelihood over lag samples:

    S(tau) = Re sum_k alpha_k z_i,k conj(z_j,k) exp(j 2*pi*f_k*tau)

The learnable part is the frequency-domain filterbank that produces z and the
per-bin likelihood weights alpha.  The downstream localization chain remains
the same all-pair WLS evaluator used by the existing direct baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from task_baselines import (
    _diagnostics_from_estimator,
    _lag_mask,
    _peak_from_score,
    _quality_from_score,
)


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0, :], x[:, 1, :])


def _shifted_freq_grid(signal_len: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(int(signal_len))).astype(np.float32)


def _initial_bins(signal_len: int, num_features: int, mode: str = "wide") -> np.ndarray:
    """Deterministic initialization for the learnable 64-filter bank."""
    signal_len = int(signal_len)
    num_features = int(num_features)
    mode = str(mode).lower()
    if num_features < 1 or num_features > signal_len:
        raise ValueError("num_features must be in [1, signal_len]")

    if mode in ("wide", "uniform", "fullband"):
        bins = np.linspace(0, signal_len - 1, num_features + 2)[1:-1]
        return np.unique(np.round(bins).astype(int))[:num_features]
    if mode in ("center", "band"):
        half = num_features // 2
        start = signal_len // 2 - half
        return np.arange(start, start + num_features, dtype=int)
    if mode in ("high", "edge"):
        freqs = _shifted_freq_grid(signal_len)
        return np.argsort(-np.abs(freqs))[:num_features].astype(int)
    raise ValueError(f"Unknown V5-A init mode: {mode}")


def _gaussian_filter_logits(signal_len: int, centers: np.ndarray,
                            width: float = 3.0) -> torch.Tensor:
    idx = torch.arange(int(signal_len), dtype=torch.float32)
    rows = []
    for center in np.asarray(centers, dtype=float).tolist():
        dist = torch.abs(idx - float(center))
        rows.append(-0.5 * (dist / max(float(width), 1e-3)).pow(2))
    return torch.stack(rows, dim=0)


class LearnableCompressedTDOALikelihood(nn.Module):
    """CR16-only compressed-domain TDOA posterior estimator."""

    def __init__(
        self,
        signal_len: int = 1024,
        num_features: int = 64,
        lag_limit_samples: int = 48,
        init_mode: str = "wide",
        filter_width: float = 3.0,
        filter_temperature: float = 1.0,
        score_temperature: float = 0.35,
    ):
        super().__init__()
        self.signal_len = int(signal_len)
        self.num_features = int(num_features)
        self.lag_limit_samples = int(lag_limit_samples)
        self.filter_temperature = float(filter_temperature)
        self.score_temperature = float(score_temperature)
        if self.num_features != self.signal_len // 16:
            raise ValueError(
                "V5-A is CR16-only: num_features must equal signal_len // 16"
            )

        centers = _initial_bins(self.signal_len, self.num_features, init_mode)
        if centers.size != self.num_features:
            raise ValueError("initial bin selection did not produce num_features bins")
        logits = _gaussian_filter_logits(self.signal_len, centers, width=filter_width)
        self.filter_logits = nn.Parameter(logits)
        self.log_alpha = nn.Parameter(torch.zeros(self.num_features))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        freqs = torch.tensor(_shifted_freq_grid(self.signal_len), dtype=torch.float32)
        lags = torch.arange(
            -self.lag_limit_samples, self.lag_limit_samples + 1,
            dtype=torch.float32,
        )
        self.register_buffer("freq_grid", freqs)
        self.register_buffer("lag_grid", lags)
        self.register_buffer("initial_bins", torch.as_tensor(centers, dtype=torch.long))

    def filter_weights(self) -> torch.Tensor:
        temp = max(float(self.filter_temperature), 1e-3)
        return torch.softmax(self.filter_logits / temp, dim=-1)

    def effective_freqs(self) -> torch.Tensor:
        weights = self.filter_weights()
        return weights @ self.freq_grid

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = _to_complex(x)
        spec = torch.fft.fftshift(torch.fft.fft(xc, norm="ortho"), dim=-1)
        weights = self.filter_weights().to(spec.device, dtype=spec.real.dtype)
        weights_c = torch.complex(weights, torch.zeros_like(weights))
        return torch.einsum("bn,kn->bk", spec, weights_c)

    def pair_logits(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        z_i = self.encode(x_i)
        z_j = self.encode(x_j)
        return self.pair_logits_from_encoded(z_i, z_j)

    def pair_logits_from_encoded(self, z_i: torch.Tensor,
                                 z_j: torch.Tensor) -> torch.Tensor:
        cross = z_i * torch.conj(z_j)
        phase = cross / (torch.abs(cross) + 1e-8)
        amp = torch.log1p(torch.abs(cross))
        amp = amp / (torch.median(amp, dim=-1, keepdim=True).values + 1e-8)
        amp = torch.clamp(amp, 0.05, 12.0)
        alpha = torch.nn.functional.softplus(self.log_alpha).to(amp.device) + 1e-4
        weighted_phase = phase * amp * alpha[None, :]

        freqs = self.effective_freqs().to(z_i.device)
        steering = torch.exp(
            2j * np.pi * self.lag_grid[:, None].to(z_i.device) * freqs[None, :]
        )
        logits = torch.real(torch.einsum("lk,bk->bl", steering, weighted_phase))
        logits = logits / np.sqrt(float(self.num_features))
        scale = torch.exp(torch.clamp(self.logit_scale, -3.0, 3.0))
        return logits * scale / max(float(self.score_temperature), 1e-3)

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.pair_logits(x_i, x_j)

    def posterior_stats(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prob = torch.softmax(logits, dim=-1)
        lags = self.lag_grid.to(logits.device, dtype=logits.dtype)
        mean = torch.sum(prob * lags[None, :], dim=-1)
        var = torch.sum(prob * (lags[None, :] - mean[:, None]).pow(2), dim=-1)
        return mean, torch.clamp(var, min=0.25)

    def filterbank_regularization(self) -> torch.Tensor:
        weights = self.filter_weights()
        entropy = -torch.sum(weights * torch.log(weights + 1e-9), dim=-1)
        entropy = entropy / np.log(float(self.signal_len))
        gram = weights @ weights.T
        eye = torch.eye(self.num_features, device=weights.device, dtype=weights.dtype)
        overlap = torch.mean((gram - eye * torch.diag(gram)) ** 2)
        return torch.mean(entropy) + 0.25 * overlap

    def selected_bin_summary(self) -> Dict[str, object]:
        with torch.no_grad():
            weights = self.filter_weights().detach().cpu()
            top_bins = torch.argmax(weights, dim=-1).numpy().astype(int)
            eff_freqs = self.effective_freqs().detach().cpu().numpy().astype(float)
            entropy = -torch.sum(weights * torch.log(weights + 1e-9), dim=-1)
            entropy = entropy / np.log(float(self.signal_len))
        return {
            "num_features_complex": int(self.num_features),
            "top_bins": [int(v) for v in top_bins.tolist()],
            "effective_freq_min": float(np.min(eff_freqs)),
            "effective_freq_max": float(np.max(eff_freqs)),
            "mean_filter_entropy": float(torch.mean(entropy).item()),
            "initial_bins": [int(v) for v in self.initial_bins.detach().cpu().tolist()],
        }


def soft_lag_ce(logits: torch.Tensor, lag_grid: torch.Tensor,
                target_tdoa: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    target = target_tdoa.to(logits.device, dtype=logits.dtype)
    lags = lag_grid.to(logits.device, dtype=logits.dtype)
    sigma = max(float(sigma), 1e-3)
    weights = torch.exp(-0.5 * ((lags[None, :] - target[:, None]) / sigma) ** 2)
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-9)
    logp = torch.log_softmax(logits, dim=-1)
    return torch.mean(-torch.sum(weights * logp, dim=-1))


def posterior_aux_losses(model: LearnableCompressedTDOALikelihood,
                         logits: torch.Tensor,
                         target_tdoa: torch.Tensor,
                         width_radius: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prob = torch.softmax(logits, dim=-1)
    target = target_tdoa.to(logits.device, dtype=logits.dtype)
    lags = model.lag_grid.to(logits.device, dtype=logits.dtype)
    distance = torch.abs(lags[None, :] - target[:, None])
    outside = distance > float(width_radius)
    outside_mass = torch.sum(torch.where(outside, prob, torch.zeros_like(prob)), dim=-1)
    width = torch.sum(prob * torch.relu(distance - float(width_radius)).pow(2), dim=-1)
    width = width / max(float(model.lag_limit_samples) ** 2, 1.0)

    mean, var = model.posterior_stats(logits)
    err2 = (mean - target).pow(2)
    unc = 0.5 * (err2 / var + torch.log(var))
    return torch.mean(outside_mass), torch.mean(width), torch.mean(unc)


def _batch_from_loader(batch):
    if len(batch) >= 5:
        return batch[0], batch[2], batch[4]
    return batch[0], batch[1], batch[2]


class _RefreshableUrbanPairDataset(Dataset):
    def __init__(self, simulator, snapshot_indices, uav_i_indices, uav_j_indices,
                 base_seed: int = 42, refresh_interval: int = 1):
        self.sim = simulator
        self.snapshot_indices = np.asarray(snapshot_indices, dtype=int)
        self.uav_i_indices = np.asarray(uav_i_indices, dtype=int)
        self.uav_j_indices = np.asarray(uav_j_indices, dtype=int)
        self.base_seed = int(base_seed)
        self.refresh_interval = max(1, int(refresh_interval))
        self._refresh_id = None
        self.X1_noisy = None
        self.X2_noisy = None
        self.tdoa = None
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        refresh_id = int(epoch) // self.refresh_interval
        if refresh_id == self._refresh_id:
            return
        seed = self.base_seed + refresh_id
        X1_n, _, X2_n, _, tdoa = self.sim.generate_urban_pair_training_dataset_from_plan(
            self.snapshot_indices,
            self.uav_i_indices,
            self.uav_j_indices,
            seed=seed,
            return_groups=False,
        )
        self.X1_noisy = X1_n
        self.X2_noisy = X2_n
        self.tdoa = tdoa
        self._refresh_id = refresh_id

    def __len__(self):
        return len(self.snapshot_indices)

    def __getitem__(self, idx):
        return self.X1_noisy[idx], self.X2_noisy[idx], self.tdoa[idx]


@dataclass
class V5ATrainConfig:
    n_pairs: int = 10000
    epochs: int = 40
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    sigma: float = 1.0
    ambiguity_weight: float = 0.05
    width_weight: float = 0.05
    uncertainty_weight: float = 0.01
    filter_reg_weight: float = 0.005
    resample_train_each_epoch: bool = True
    resample_interval: int = 2
    init_mode: str = "wide"


def train_v5a_compressed_tdoa(simulator, device, seed: int = 42,
                              lag_limit_samples: Optional[int] = None,
                              config: Optional[V5ATrainConfig] = None):
    """Train the CR16 V5-A compressed-domain TDOA likelihood model."""
    cfg = config or V5ATrainConfig()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    lag_limit = int(lag_limit_samples) if lag_limit_samples is not None else 48
    model = LearnableCompressedTDOALikelihood(
        signal_len=simulator.signal_len,
        num_features=simulator.signal_len // 16,
        lag_limit_samples=lag_limit,
        init_mode=cfg.init_mode,
    ).to(device)

    snapshot_idx, uav_i_idx, uav_j_idx = simulator.build_urban_pair_training_plan(
        int(cfg.n_pairs), seed=int(seed)
    )
    X1_n, _, X2_n, _, tdoa, groups = simulator.generate_urban_pair_training_dataset_from_plan(
        snapshot_idx, uav_i_idx, uav_j_idx, seed=int(seed), return_groups=True
    )
    dataset = TensorDataset(X1_n, X2_n, tdoa)
    n_total = len(dataset)
    groups_np = np.asarray(groups, dtype=int)
    unique_groups = np.unique(groups_np)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_groups)
    val_group_count = max(1, int(round(unique_groups.size * float(cfg.val_fraction))))
    val_groups = set(int(v) for v in unique_groups[:val_group_count].tolist())
    val_mask = np.asarray([int(g) in val_groups for g in groups_np], dtype=bool)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    if train_idx.size == 0 or val_idx.size == 0:
        val_count = max(1, int(round(n_total * float(cfg.val_fraction))))
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n_total, generator=gen)
        val_idx = perm[:val_count].numpy()
        train_idx = perm[val_count:].numpy()

    if cfg.resample_train_each_epoch:
        train_dataset = _RefreshableUrbanPairDataset(
            simulator,
            snapshot_idx[train_idx],
            uav_i_idx[train_idx],
            uav_j_idx[train_idx],
            base_seed=int(seed) + 10000,
            refresh_interval=int(cfg.resample_interval),
        )
    else:
        train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False
    )

    opt = optim.AdamW(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    history = {
        "train_loss": [], "val_loss": [],
        "train_ce": [], "val_ce": [],
        "train_mae": [], "val_mae": [],
        "train_within1": [], "val_within1": [],
        "train_width": [], "val_width": [],
        "train_ambiguity": [], "val_ambiguity": [],
        "filter_reg": [],
    }

    def run_epoch(loader, train: bool, epoch: int):
        if train and hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        model.train(train)
        totals = {
            "loss": 0.0, "ce": 0.0, "mae": 0.0, "within1": 0.0,
            "width": 0.0, "ambiguity": 0.0, "filter_reg": 0.0,
        }
        count = 0
        for batch in loader:
            x1, x2, target = _batch_from_loader(batch)
            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device).float()
            if train:
                opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            ce = soft_lag_ce(logits, model.lag_grid, target, sigma=cfg.sigma)
            amb, width, unc = posterior_aux_losses(model, logits, target)
            reg = model.filterbank_regularization()
            loss = (
                ce
                + float(cfg.ambiguity_weight) * amb
                + float(cfg.width_weight) * width
                + float(cfg.uncertainty_weight) * unc
                + float(cfg.filter_reg_weight) * reg
            )
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

            with torch.no_grad():
                pred = model.lag_grid.to(device)[torch.argmax(logits, dim=-1)]
                abs_err = torch.abs(pred.float() - target)
                bs = int(target.numel())
                totals["loss"] += float(loss.item()) * bs
                totals["ce"] += float(ce.item()) * bs
                totals["mae"] += float(torch.mean(abs_err).item()) * bs
                totals["within1"] += float(torch.mean((abs_err <= 1.0).float()).item()) * bs
                totals["width"] += float(width.item()) * bs
                totals["ambiguity"] += float(amb.item()) * bs
                totals["filter_reg"] += float(reg.item()) * bs
                count += bs
        return {k: v / max(count, 1) for k, v in totals.items()}

    print("\n" + "=" * 40)
    print("[V5-A] Training CR16 learnable compressed-domain TDOA likelihood")
    print(
        f"[V5-A] pairs={cfg.n_pairs} | epochs={cfg.epochs} | "
        f"batch={cfg.batch_size} | lag_limit={lag_limit} | "
        f"features={simulator.signal_len // 16} complex"
    )
    for ep in range(int(cfg.epochs)):
        tr = run_epoch(train_loader, train=True, epoch=ep)
        va = run_epoch(val_loader, train=False, epoch=ep)
        for split, stats in [("train", tr), ("val", va)]:
            history[f"{split}_loss"].append(stats["loss"])
            history[f"{split}_ce"].append(stats["ce"])
            history[f"{split}_mae"].append(stats["mae"])
            history[f"{split}_within1"].append(stats["within1"])
            history[f"{split}_width"].append(stats["width"])
            history[f"{split}_ambiguity"].append(stats["ambiguity"])
        history["filter_reg"].append(tr["filter_reg"])
        if ep == 0 or (ep + 1) % max(1, min(10, int(cfg.epochs))) == 0 or ep + 1 == int(cfg.epochs):
            print(
                f"[V5-A][Epoch {ep + 1:03d}/{cfg.epochs}] "
                f"TrainLoss={tr['loss']:.4f} ValLoss={va['loss']:.4f} | "
                f"ValMAE={va['mae']:.3f} samples | Val<=1={va['within1']:.3f} | "
                f"ValAmb={va['ambiguity']:.3f}"
            )

    history["config"] = {
        "n_pairs": int(cfg.n_pairs),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "val_fraction": float(cfg.val_fraction),
        "sigma": float(cfg.sigma),
        "ambiguity_weight": float(cfg.ambiguity_weight),
        "width_weight": float(cfg.width_weight),
        "uncertainty_weight": float(cfg.uncertainty_weight),
        "filter_reg_weight": float(cfg.filter_reg_weight),
        "resample_train_each_epoch": bool(cfg.resample_train_each_epoch),
        "resample_interval": int(cfg.resample_interval),
        "lag_limit_samples": int(lag_limit),
        "cr": 16,
        "num_features_complex": int(simulator.signal_len // 16),
        "groups": [int(v) for v in np.unique(groups).tolist()],
        "val_groups": [int(v) for v in sorted(val_groups)],
        "train_pair_count": int(train_idx.size),
        "val_pair_count": int(val_idx.size),
    }
    history["filterbank"] = model.selected_bin_summary()
    return model, history


class LearnedCompressedTDOAEstimator:
    """Adapter so the V5-A model can enter the existing direct-estimator API."""

    def __init__(self, model: LearnableCompressedTDOALikelihood, device,
                 label: str = "V5A-CR16"):
        self.model = model.to(device).eval()
        self.device = device
        self.label = str(label)
        self.lags = model.lag_grid.detach().cpu().numpy().astype(float)

    def _tensor_from_sig(self, sig: np.ndarray) -> torch.Tensor:
        sig = np.asarray(sig)
        arr = np.stack((sig.real, sig.imag), axis=0).astype(np.float32)
        return torch.from_numpy(arr[None, :, :]).to(self.device)

    def score_pair(self, sig_i, sig_j):
        x_i = self._tensor_from_sig(sig_i)
        x_j = self._tensor_from_sig(sig_j)
        with torch.no_grad():
            logits = self.model(x_i, x_j)[0]
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(float)
        return prob

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        score = self.score_pair(sig_i, sig_j)
        search_mask = _lag_mask(self.lags, lag_limit_samples)
        idx, lag, search_mask = _peak_from_score(
            self.lags, score, search_mask, sub_sample=sub_sample
        )
        if not return_quality:
            return lag
        weight, peak, sidelobe_ratio = _quality_from_score(score, idx, search_mask)
        return lag, weight, peak, sidelobe_ratio

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        return _diagnostics_from_estimator(
            self, batch_np, meta, fs, c, use_los_only=use_los_only,
            sub_sample=sub_sample, lag_limit_samples=lag_limit_samples,
        )

    def alias_diagnostics(self, lag_limit_samples=None, threshold=0.8):
        del threshold
        summary = self.model.selected_bin_summary()
        summary["direct_lag_min"] = float(np.min(self.lags))
        summary["direct_lag_max"] = float(np.max(self.lags))
        summary["direct_lag_limit_samples"] = (
            float(lag_limit_samples) if lag_limit_samples is not None else float("nan")
        )
        return summary
