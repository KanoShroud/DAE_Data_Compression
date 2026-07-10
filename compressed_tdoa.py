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
import torch.nn.functional as F
from scipy import signal
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from task_baselines import (
    _diagnostics_from_estimator,
    _diagnostics_from_quality_estimator,
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
        """Convert complex [N] or two-channel [2,N] to batched float [1,2,N]."""
        sig = np.asarray(sig)
        if np.iscomplexobj(sig):
            arr = np.stack((sig.real, sig.imag), axis=0).astype(np.float32)
        elif sig.ndim == 2 and sig.shape[0] == 2 and sig.dtype in (np.float32, np.float64):
            arr = sig.astype(np.float32)
        else:
            raise ValueError(
                f"_tensor_from_sig expects complex [N] or two-channel [2,N], "
                f"got shape={sig.shape} dtype={sig.dtype}"
            )
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


class HardBinCompressedTDOALikelihood(nn.Module):
    """
    V5-A.1 exact-bin compressed-domain likelihood.

    Unlike V5-A, every transmitted feature is exactly one selected FFT bin.  The
    delay steering therefore uses the true bin frequency, avoiding the
    soft-filter/effective-frequency approximation that made V5-A hard to
    interpret.
    """

    def __init__(
        self,
        selected_bins,
        signal_len: int = 1024,
        lag_limit_samples: int = 48,
        score_temperature: float = 0.35,
        amp_power: float = 1.0,
    ):
        super().__init__()
        selected = np.asarray(selected_bins, dtype=int)
        if selected.ndim != 1 or selected.size < 1:
            raise ValueError("HardBinCompressedTDOALikelihood needs selected bins")
        if np.any(selected < 0) or np.any(selected >= int(signal_len)):
            raise ValueError("selected bins out of FFT range")
        self.signal_len = int(signal_len)
        self.num_features = int(selected.size)
        self.lag_limit_samples = int(lag_limit_samples)
        self.score_temperature = float(score_temperature)
        self.amp_power = float(amp_power)

        freqs = torch.tensor(_shifted_freq_grid(self.signal_len), dtype=torch.float32)
        lags = torch.arange(
            -self.lag_limit_samples, self.lag_limit_samples + 1,
            dtype=torch.float32,
        )
        self.register_buffer("freq_grid", freqs)
        self.register_buffer("lag_grid", lags)
        self.register_buffer(
            "selected_bins", torch.as_tensor(selected, dtype=torch.long)
        )
        self.log_alpha = nn.Parameter(torch.zeros(self.num_features))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def selected_freqs(self) -> torch.Tensor:
        return self.freq_grid[self.selected_bins]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(
                f"encode expects [B,2,N], got shape={tuple(x.shape)}"
            )
        if x.shape[-1] != self.signal_len:
            raise ValueError(
                f"encode: signal length mismatch, "
                f"expected {self.signal_len} got {x.shape[-1]}"
            )
        if self.selected_bins.min() < 0 or self.selected_bins.max() >= self.signal_len:
            raise IndexError(
                f"encode: selected_bins out of [0, {self.signal_len}), "
                f"range=[{self.selected_bins.min()}, {self.selected_bins.max()}]"
            )
        xc = _to_complex(x)
        spec = torch.fft.fftshift(torch.fft.fft(xc, norm="ortho"), dim=-1)
        return spec.index_select(
            dim=-1, index=self.selected_bins.to(spec.device)
        )

    def pair_logits(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        z_i = self.encode(x_i)
        z_j = self.encode(x_j)
        return self.pair_logits_from_encoded(z_i, z_j)

    def pair_logits_from_encoded(self, z_i: torch.Tensor,
                                 z_j: torch.Tensor) -> torch.Tensor:
        cross = z_i * torch.conj(z_j)
        phase = cross / (torch.abs(cross) + 1e-8)
        amp = torch.abs(cross)
        amp = amp / (torch.median(amp, dim=-1, keepdim=True).values + 1e-8)
        amp = torch.clamp(amp, 0.05, 12.0).pow(float(self.amp_power))
        alpha = F.softplus(self.log_alpha).to(amp.device) + 1e-4
        weighted_phase = phase * amp * alpha[None, :]

        freqs = self.selected_freqs().to(z_i.device)
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

    def selected_bin_summary(self) -> Dict[str, object]:
        bins = self.selected_bins.detach().cpu().numpy().astype(int)
        freqs = self.selected_freqs().detach().cpu().numpy().astype(float)
        alpha = F.softplus(self.log_alpha).detach().cpu().numpy().astype(float)
        return {
            "num_features_complex": int(self.num_features),
            "selected_bins": [int(v) for v in bins.tolist()],
            "selected_bin_min": int(np.min(bins)),
            "selected_bin_max": int(np.max(bins)),
            "selected_freq_min": float(np.min(freqs)),
            "selected_freq_max": float(np.max(freqs)),
            "alpha_min": float(np.min(alpha)),
            "alpha_max": float(np.max(alpha)),
            "alpha_mean": float(np.mean(alpha)),
            "score_temperature": float(self.score_temperature),
            "amp_power": float(self.amp_power),
        }


def _snr_bucket_name(snr_value: float) -> str:
    value = float(snr_value)
    if value <= 0.0:
        return "low"
    if 2.0 <= value <= 6.0:
        return "mid"
    if value >= 8.0:
        return "high"
    return "other"


def _generate_urban_pair_dataset_from_plan_with_snr(
    simulator,
    snapshot_indices,
    uav_i_indices,
    uav_j_indices,
    snr_values,
    seed: int = 42,
    return_groups: bool = False,
):
    """Generate pair data with explicit per-sample SNR values."""
    if simulator.scenario_mode != "urban8":
        raise RuntimeError("V5-A.1 hard-bin training requires scenario_mode='urban8'")
    snapshot_indices = np.asarray(snapshot_indices, dtype=int)
    uav_i_indices = np.asarray(uav_i_indices, dtype=int)
    uav_j_indices = np.asarray(uav_j_indices, dtype=int)
    snr_values = np.asarray(snr_values, dtype=float)
    if not (
        len(snapshot_indices) == len(uav_i_indices)
        == len(uav_j_indices) == len(snr_values)
    ):
        raise ValueError("pair plan arrays and snr_values must have the same length")

    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    n_items = int(len(snapshot_indices))
    x1n = torch.empty((n_items, 2, simulator.signal_len), dtype=torch.float32)
    x2n = torch.empty_like(x1n)
    tdoa = torch.empty((n_items,), dtype=torch.float32)

    for snr in np.unique(snr_values):
        group_idx = np.where(np.isclose(snr_values, snr))[0]
        for start in range(0, len(group_idx), 500):
            idx = group_idx[start:start + 500]
            snap_chunk = snapshot_indices[idx]
            ui_chunk = uav_i_indices[idx]
            uj_chunk = uav_j_indices[idx]
            Xn, _, meta = simulator.generate_urban_batch(
                len(idx), snr_db=float(snr), snapshot_indices=snap_chunk
            )
            row_idx = torch.arange(len(idx), dtype=torch.long)
            ui_idx = torch.tensor(ui_chunk, dtype=torch.long)
            uj_idx = torch.tensor(uj_chunk, dtype=torch.long)
            x1n[idx] = Xn[row_idx, ui_idx]
            x2n[idx] = Xn[row_idx, uj_idx]
            distances = np.asarray(meta["distances"], dtype=float)
            rows = np.arange(len(idx))
            tau = (
                distances[rows, ui_chunk] - distances[rows, uj_chunk]
            ) / (simulator.c / simulator.fs)
            tdoa[idx] = torch.tensor(tau, dtype=torch.float32)

    np.random.set_state(rng_state)
    snr_tensor = torch.tensor(snr_values, dtype=torch.float32)
    groups = torch.tensor(snapshot_indices, dtype=torch.long)
    if return_groups:
        return x1n, x2n, tdoa, snr_tensor, groups
    return x1n, x2n, tdoa, snr_tensor


def prepare_v5a1_hardbin_dataset(simulator, config, seed: int = 42):
    """Build one reusable explicit-SNR pair dataset for all V5-A.1 bin sets."""
    cfg = config or V5A1TrainConfig()
    rng = np.random.default_rng(int(seed))
    snapshot_idx, uav_i_idx, uav_j_idx = simulator.build_urban_pair_training_plan(
        int(cfg.n_pairs), seed=int(seed)
    )
    snr_grid = np.asarray(cfg.snr_grid, dtype=float)
    if snr_grid.ndim != 1 or snr_grid.size < 1:
        raise ValueError("V5-A.1 snr_grid must contain at least one SNR value")
    snr_values = rng.choice(snr_grid, size=int(cfg.n_pairs), replace=True)
    X1_n, X2_n, tdoa, snr_tensor, groups = (
        _generate_urban_pair_dataset_from_plan_with_snr(
            simulator, snapshot_idx, uav_i_idx, uav_j_idx, snr_values,
            seed=int(seed), return_groups=True,
        )
    )
    dataset = TensorDataset(X1_n, X2_n, tdoa, snr_tensor)
    groups_np = np.asarray(groups, dtype=int)
    unique_groups = np.unique(groups_np)
    rng.shuffle(unique_groups)
    val_group_count = max(1, int(round(unique_groups.size * float(cfg.val_fraction))))
    val_groups = set(int(v) for v in unique_groups[:val_group_count].tolist())
    val_mask = np.asarray([int(g) in val_groups for g in groups_np], dtype=bool)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    if train_idx.size == 0 or val_idx.size == 0:
        val_count = max(1, int(round(len(dataset) * float(cfg.val_fraction))))
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(seed)))
        val_idx = perm[:val_count].numpy()
        train_idx = perm[val_count:].numpy()
    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "groups": groups_np,
        "val_groups": sorted(val_groups),
        "snr_values": snr_values,
        "snapshot_indices": snapshot_idx,
        "uav_i_indices": uav_i_idx,
        "uav_j_indices": uav_j_idx,
    }


@dataclass
class V5A1TrainConfig:
    n_pairs: int = 6000
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    sigma: float = 1.0
    point_weight: float = 0.05
    ambiguity_weight: float = 0.10
    width_weight: float = 0.05
    uncertainty_weight: float = 0.01
    amp_power: float = 1.0
    score_temperature: float = 0.35
    restore_best_by_val_mae: bool = True
    best_low_snr_weight: float = 0.50
    best_mid_snr_weight: float = 0.25
    best_high_snr_weight: float = 0.25
    snr_grid: Tuple[float, ...] = (
        -10.0, -8.0, -6.0, -4.0, -2.0, 0.0,
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
        14.0, 16.0, 18.0, 20.0,
    )


def _empty_bucket_totals():
    return {
        "loss": 0.0, "ce": 0.0, "point": 0.0, "mae": 0.0,
        "within1": 0.0, "within2": 0.0, "width": 0.0,
        "ambiguity": 0.0, "entropy": 0.0, "count": 0,
    }


def _bucket_finalize(totals):
    count = max(int(totals.get("count", 0)), 1)
    return {
        key: (value / count)
        for key, value in totals.items()
        if key != "count"
    }


def train_v5a1_hardbin_tdoa(
    simulator,
    device,
    selected_bins,
    label: str,
    seed: int = 42,
    lag_limit_samples: Optional[int] = None,
    config: Optional[V5A1TrainConfig] = None,
    dataset_bundle: Optional[Dict[str, object]] = None,
):
    """Train a V5-A.1 exact-bin likelihood for one fixed bin set."""
    cfg = config or V5A1TrainConfig()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    lag_limit = int(lag_limit_samples) if lag_limit_samples is not None else 48
    model = HardBinCompressedTDOALikelihood(
        selected_bins=selected_bins,
        signal_len=simulator.signal_len,
        lag_limit_samples=lag_limit,
        score_temperature=float(cfg.score_temperature),
        amp_power=float(cfg.amp_power),
    ).to(device)

    bundle = dataset_bundle or prepare_v5a1_hardbin_dataset(
        simulator, cfg, seed=int(seed)
    )
    train_dataset = Subset(bundle["dataset"], bundle["train_idx"])
    val_dataset = Subset(bundle["dataset"], bundle["val_idx"])
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
        "label": str(label),
        "train_loss": [], "val_loss": [],
        "train_ce": [], "val_ce": [],
        "train_point": [], "val_point": [],
        "train_mae": [], "val_mae": [],
        "train_within1": [], "val_within1": [],
        "train_within2": [], "val_within2": [],
        "train_width": [], "val_width": [],
        "train_ambiguity": [], "val_ambiguity": [],
        "train_entropy": [], "val_entropy": [],
        "train_buckets": [], "val_buckets": [],
    }

    def run_epoch(loader, train: bool):
        model.train(train)
        totals = _empty_bucket_totals()
        buckets = {name: _empty_bucket_totals() for name in ("low", "mid", "high", "other")}
        for x1, x2, target, snr_batch in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device).float()
            snr_batch = snr_batch.float()
            if train:
                opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            ce = soft_lag_ce(logits, model.lag_grid, target, sigma=cfg.sigma)
            amb, width, unc = posterior_aux_losses(model, logits, target)
            mean, _ = model.posterior_stats(logits)
            point = F.smooth_l1_loss(mean, target, beta=1.0)
            loss = (
                ce
                + float(cfg.point_weight) * point
                + float(cfg.ambiguity_weight) * amb
                + float(cfg.width_weight) * width
                + float(cfg.uncertainty_weight) * unc
            )
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

            with torch.no_grad():
                prob = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=-1)
                entropy = entropy / np.log(float(prob.shape[-1]))
                pred = model.lag_grid.to(device)[torch.argmax(logits, dim=-1)]
                abs_err = torch.abs(pred.float() - target)
                bs = int(target.numel())
                batch_stats = {
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "point": float(point.item()),
                    "mae": float(torch.mean(abs_err).item()),
                    "within1": float(torch.mean((abs_err <= 1.0).float()).item()),
                    "within2": float(torch.mean((abs_err <= 2.0).float()).item()),
                    "width": float(width.item()),
                    "ambiguity": float(amb.item()),
                    "entropy": float(torch.mean(entropy).item()),
                }
                for key, value in batch_stats.items():
                    totals[key] += value * bs
                totals["count"] += bs

                snr_np = snr_batch.detach().cpu().numpy()
                abs_np = abs_err.detach().cpu().numpy()
                entropy_np = entropy.detach().cpu().numpy()
                for bucket_name in buckets:
                    mask = np.asarray([
                        _snr_bucket_name(v) == bucket_name for v in snr_np
                    ], dtype=bool)
                    if not np.any(mask):
                        continue
                    count = int(np.sum(mask))
                    b = buckets[bucket_name]
                    b["loss"] += batch_stats["loss"] * count
                    b["ce"] += batch_stats["ce"] * count
                    b["point"] += batch_stats["point"] * count
                    b["mae"] += float(np.mean(abs_np[mask])) * count
                    b["within1"] += float(np.mean(abs_np[mask] <= 1.0)) * count
                    b["within2"] += float(np.mean(abs_np[mask] <= 2.0)) * count
                    b["width"] += batch_stats["width"] * count
                    b["ambiguity"] += batch_stats["ambiguity"] * count
                    b["entropy"] += float(np.mean(entropy_np[mask])) * count
                    b["count"] += count
        stats = _bucket_finalize(totals)
        bucket_stats = {
            key: _bucket_finalize(value) | {"count": int(value["count"])}
            for key, value in buckets.items()
        }
        return stats, bucket_stats

    def bucket_weighted_val_mae(stats, bucket_stats):
        low = bucket_stats.get("low", {})
        mid = bucket_stats.get("mid", {})
        high = bucket_stats.get("high", {})
        weights = [
            (float(cfg.best_low_snr_weight), low),
            (float(cfg.best_mid_snr_weight), mid),
            (float(cfg.best_high_snr_weight), high),
        ]
        total_weight = 0.0
        value = 0.0
        for weight, bucket in weights:
            count = int(bucket.get("count", 0))
            mae = bucket.get("mae", float("nan"))
            if count <= 0 or not np.isfinite(mae) or weight <= 0:
                continue
            total_weight += weight
            value += weight * float(mae)
        if total_weight <= 0:
            return float(stats.get("mae", float("inf")))
        return float(value / total_weight)

    print("\n" + "=" * 40)
    print(f"[V5-A.1] Training hard-bin exact likelihood: {label}")
    print(
        f"[V5-A.1] pairs={cfg.n_pairs} | epochs={cfg.epochs} | "
        f"batch={cfg.batch_size} | lag_limit={lag_limit} | "
        f"features={model.num_features} complex"
    )
    best_state = None
    best_epoch = -1
    best_val_mae = float("inf")
    best_val_weighted_mae = float("inf")
    best_val_loss = float("inf")
    for ep in range(int(cfg.epochs)):
        tr, tr_buckets = run_epoch(train_loader, train=True)
        va, va_buckets = run_epoch(val_loader, train=False)
        weighted_mae = bucket_weighted_val_mae(va, va_buckets)
        for split, stats, bucket_stats in [
            ("train", tr, tr_buckets), ("val", va, va_buckets)
        ]:
            for key in ("loss", "ce", "point", "mae", "within1",
                        "within2", "width", "ambiguity", "entropy"):
                history[f"{split}_{key}"].append(stats[key])
            history[f"{split}_buckets"].append(bucket_stats)
        history.setdefault("val_weighted_mae", []).append(weighted_mae)
        selection_value = weighted_mae if bool(cfg.restore_best_by_val_mae) else va["loss"]
        best_value = (
            best_val_weighted_mae if bool(cfg.restore_best_by_val_mae)
            else best_val_loss
        )
        if selection_value < best_value:
            best_epoch = int(ep)
            best_val_mae = float(va["mae"])
            best_val_weighted_mae = float(weighted_mae)
            best_val_loss = float(va["loss"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if ep == 0 or (ep + 1) % max(1, min(5, int(cfg.epochs))) == 0 or ep + 1 == int(cfg.epochs):
            low = va_buckets["low"]
            high = va_buckets["high"]
            print(
                f"[V5-A.1][{label}][Epoch {ep + 1:03d}/{cfg.epochs}] "
                f"ValLoss={va['loss']:.4f} ValMAE={va['mae']:.3f} | "
                f"ValWMAE={weighted_mae:.3f} | "
                f"lowMAE={low.get('mae', float('nan')):.3f} "
                f"highMAE={high.get('mae', float('nan')):.3f}"
            )

    if best_state is not None:
        model.load_state_dict({
            key: value.to(device) for key, value in best_state.items()
        })
        print(
            f"[V5-A.1][{label}] Restored best "
            f"{'weighted-ValMAE' if cfg.restore_best_by_val_mae else 'ValLoss'} "
            f"epoch={best_epoch + 1} | ValMAE={best_val_mae:.3f} | "
            f"ValWMAE={best_val_weighted_mae:.3f} | ValLoss={best_val_loss:.4f}"
        )

    history["config"] = {
        "label": str(label),
        "n_pairs": int(cfg.n_pairs),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "val_fraction": float(cfg.val_fraction),
        "sigma": float(cfg.sigma),
        "point_weight": float(cfg.point_weight),
        "ambiguity_weight": float(cfg.ambiguity_weight),
        "width_weight": float(cfg.width_weight),
        "uncertainty_weight": float(cfg.uncertainty_weight),
        "amp_power": float(cfg.amp_power),
        "score_temperature": float(cfg.score_temperature),
        "restore_best_by_val_mae": bool(cfg.restore_best_by_val_mae),
        "best_low_snr_weight": float(cfg.best_low_snr_weight),
        "best_mid_snr_weight": float(cfg.best_mid_snr_weight),
        "best_high_snr_weight": float(cfg.best_high_snr_weight),
        "lag_limit_samples": int(lag_limit),
        "cr": 16,
        "snr_grid": [float(v) for v in cfg.snr_grid],
        "train_pair_count": int(len(bundle["train_idx"])),
        "val_pair_count": int(len(bundle["val_idx"])),
        "val_groups": [int(v) for v in bundle["val_groups"]],
    }
    history["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
    history["best_val_mae"] = float(best_val_mae)
    history["best_val_weighted_mae"] = float(best_val_weighted_mae)
    history["best_val_loss"] = float(best_val_loss)
    history["restored_best_state"] = bool(best_state is not None)
    history["bin_summary"] = model.selected_bin_summary()
    return model, history


class LearnedHardBinCompressedTDOAEstimator:
    """Adapter for V5-A.1 hard-bin likelihood estimators."""

    def __init__(self, model: HardBinCompressedTDOALikelihood, device,
                 label: str = "V5A1-HardBin64",
                 weight_mode: str = "combined"):
        self.model = model.to(device).eval()
        self.device = device
        self.label = str(label)
        self.weight_mode = str(weight_mode).lower()
        self.lags = model.lag_grid.detach().cpu().numpy().astype(float)
        self.last_quality = {}

    def _tensor_from_sig(self, sig: np.ndarray) -> torch.Tensor:
        """Convert complex [N] or two-channel [2,N] to batched float [1,2,N]."""
        sig = np.asarray(sig)
        if np.iscomplexobj(sig):
            arr = np.stack((sig.real, sig.imag), axis=0).astype(np.float32)
        elif sig.ndim == 2 and sig.shape[0] == 2 and sig.dtype in (np.float32, np.float64):
            arr = sig.astype(np.float32)
        else:
            raise ValueError(
                f"_tensor_from_sig expects complex [N] or two-channel [2,N], "
                f"got shape={sig.shape} dtype={sig.dtype}"
            )
        return torch.from_numpy(arr[None, :, :]).to(self.device)

    def score_pair(self, sig_i, sig_j):
        x_i = self._tensor_from_sig(sig_i)
        x_j = self._tensor_from_sig(sig_j)
        with torch.no_grad():
            logits = self.model(x_i, x_j)[0]
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(float)
        return prob

    @staticmethod
    def _posterior_quality(prob, lags):
        prob = np.asarray(prob, dtype=float)
        lags = np.asarray(lags, dtype=float)
        order = np.argsort(prob)
        top1 = float(prob[order[-1]])
        top2 = float(prob[order[-2]]) if prob.size > 1 else 0.0
        margin = max(0.0, top1 - top2)
        mean = float(np.sum(prob * lags))
        var = float(np.sum(prob * (lags - mean) ** 2))
        entropy = float(-np.sum(prob * np.log(prob + 1e-12)) / np.log(prob.size))
        return top1, top2, margin, mean, max(var, 0.25), entropy

    def _calibrated_weight(self, prob, idx, search_mask):
        sidelobe_weight, peak, sidelobe_ratio = _quality_from_score(
            prob, idx, search_mask
        )
        top1, top2, margin, post_mean, post_var, entropy = self._posterior_quality(
            prob, self.lags
        )
        entropy_conf = float(np.clip(1.0 - entropy, 0.0, 1.0))
        var_scale = np.sqrt(post_var) + 0.5
        margin_weight = float(np.clip(12.0 * margin, 0.02, 10.0))
        variance_weight = float(np.clip(8.0 / var_scale, 0.02, 10.0))
        entropy_weight = float(np.clip(10.0 * entropy_conf, 0.02, 10.0))
        if self.weight_mode in ("sidelobe", "score"):
            weight = sidelobe_weight
        elif self.weight_mode in ("variance", "var"):
            weight = variance_weight
        elif self.weight_mode in ("entropy",):
            weight = entropy_weight
        else:
            weight = float(np.clip(
                0.5 * sidelobe_weight
                + 0.3 * variance_weight
                + 0.2 * margin_weight,
                0.02, 10.0,
            ))
            weight *= float(np.clip(0.25 + 0.75 * entropy_conf, 0.02, 1.0))
            weight = float(np.clip(weight, 0.02, 10.0))
        self.last_quality = {
            "top1_prob": top1,
            "top2_prob": top2,
            "peak_margin": margin,
            "posterior_mean": post_mean,
            "posterior_var": post_var,
            "posterior_std": float(np.sqrt(post_var)),
            "posterior_entropy": entropy,
            "entropy_confidence": entropy_conf,
            "sidelobe_weight": float(sidelobe_weight),
            "variance_weight": variance_weight,
            "margin_weight": margin_weight,
            "pair_weight": float(weight),
        }
        return weight, peak, sidelobe_ratio

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        score = self.score_pair(sig_i, sig_j)
        search_mask = _lag_mask(self.lags, lag_limit_samples)
        idx, lag, search_mask = _peak_from_score(
            self.lags, score, search_mask, sub_sample=sub_sample
        )
        if not return_quality:
            return lag
        weight, peak, sidelobe_ratio = self._calibrated_weight(
            score, idx, search_mask
        )
        return lag, weight, peak, sidelobe_ratio

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        return _diagnostics_from_quality_estimator(
            self, batch_np, meta, fs, c, use_los_only=use_los_only,
            sub_sample=sub_sample, lag_limit_samples=lag_limit_samples,
        )

    def alias_diagnostics(self, lag_limit_samples=None, threshold=0.8):
        selected = self.model.selected_bins.detach().cpu().numpy().astype(int)
        freqs = np.fft.fftshift(np.fft.fftfreq(self.model.signal_len))[selected]
        lags = self.lags
        response = np.abs(
            np.exp(2j * np.pi * np.outer(lags, freqs))
            @ np.ones(selected.size, dtype=np.complex64)
        )
        response = response / (np.max(response) + 1e-12)
        physical = _lag_mask(lags, lag_limit_samples)
        side = physical & (np.abs(lags) > 2)
        peaks, props = signal.find_peaks(response, height=float(threshold))
        summary = self.model.selected_bin_summary()
        summary.update({
            "hardbin_estimator": True,
            "weight_mode": self.weight_mode,
            "direct_lag_min": float(np.min(lags)),
            "direct_lag_max": float(np.max(lags)),
            "direct_lag_limit_samples": (
                float(lag_limit_samples) if lag_limit_samples is not None else float("nan")
            ),
            "alias_threshold": float(threshold),
            "alias_peak_lags": [int(v) for v in lags[peaks].astype(int).tolist()],
            "alias_peak_values": [
                float(v) for v in props.get("peak_heights", np.asarray([])).tolist()
            ],
            "max_sidelobe_inside_physical_lag": (
                float(np.max(response[side])) if np.any(side) else None
            ),
        })
        return summary


class V5BExpertGatedTDOAEstimator:
    """
    V5-B lightweight expert mixture for hard-bin compressed-domain TDOA.

    This is not a waveform-denoising neural network. It combines two trained
    hard-bin likelihood experts with non-oracle posterior-quality gates:
    Power64 is the low-confidence/low-SNR-safe expert, while GeoHybrid64 is the
    higher-resolution expert used when its posterior is sufficiently reliable.
    """

    def __init__(self, experts: Dict[str, LearnedHardBinCompressedTDOAEstimator],
                 label: str = "V5B-Expert64",
                 low_confidence_power_prior: float = 0.65):
        if "power" not in experts or "geohybrid" not in experts:
            raise ValueError("V5BExpertGatedTDOAEstimator requires power and geohybrid experts")
        self.experts = dict(experts)
        self.label = str(label)
        self.low_confidence_power_prior = float(low_confidence_power_prior)
        self.lags = self.experts["power"].lags.astype(float)
        self.last_quality = {}

    @staticmethod
    def _expert_quality(prob, lags):
        top1, top2, margin, mean, var, entropy = (
            LearnedHardBinCompressedTDOAEstimator._posterior_quality(prob, lags)
        )
        entropy_conf = float(np.clip(1.0 - entropy, 0.0, 1.0))
        quality = (0.05 + 10.0 * margin) * (0.10 + entropy_conf)
        quality = quality / (np.sqrt(max(var, 0.25)) + 0.5)
        return {
            "top1": top1,
            "top2": top2,
            "margin": margin,
            "mean": mean,
            "var": var,
            "entropy": entropy,
            "entropy_conf": entropy_conf,
            "quality": float(max(quality, 1e-6)),
        }

    def _mixture_weights(self, power_q, geo_q):
        q_power = float(power_q["quality"])
        q_geo = float(geo_q["quality"])
        max_q = max(q_power, q_geo)
        if max_q < 0.08:
            wp = float(np.clip(self.low_confidence_power_prior, 0.05, 0.95))
            return wp, 1.0 - wp, True
        # GeoHybrid is allowed to take over only when its posterior quality is
        # clearly competitive; otherwise the more robust power bins keep weight.
        q_geo *= float(np.clip(0.75 + geo_q["entropy_conf"], 0.35, 1.25))
        q_power *= float(np.clip(1.10 - 0.35 * power_q["entropy"], 0.50, 1.20))
        total = q_power + q_geo + 1e-12
        wp = q_power / total
        wg = q_geo / total
        return float(wp), float(wg), False

    def score_pair(self, sig_i, sig_j):
        power_prob = self.experts["power"].score_pair(sig_i, sig_j)
        geo_prob = self.experts["geohybrid"].score_pair(sig_i, sig_j)
        power_q = self._expert_quality(power_prob, self.lags)
        geo_q = self._expert_quality(geo_prob, self.lags)
        wp, wg, low_conf_prior = self._mixture_weights(power_q, geo_q)
        prob = wp * power_prob + wg * geo_prob
        prob = prob / (np.sum(prob) + 1e-12)
        self.last_quality = {
            "expert_power_weight": float(wp),
            "expert_geohybrid_weight": float(wg),
            "expert_low_confidence_prior": float(1.0 if low_conf_prior else 0.0),
            "power_quality": float(power_q["quality"]),
            "geohybrid_quality": float(geo_q["quality"]),
            "power_entropy": float(power_q["entropy"]),
            "geohybrid_entropy": float(geo_q["entropy"]),
            "power_margin": float(power_q["margin"]),
            "geohybrid_margin": float(geo_q["margin"]),
        }
        return prob

    def _topk_from_prob(self, prob, top_k: int = 3, lag_limit_samples=None):
        search_mask = _lag_mask(self.lags, lag_limit_samples)
        search_idx = np.where(search_mask)[0]
        if search_idx.size == 0:
            search_idx = np.arange(prob.size)
        top_k = max(1, int(top_k))
        ranked = search_idx[np.argsort(prob[search_idx])[-top_k:]][::-1]
        hypotheses = []
        for idx in ranked:
            hypotheses.append({
                "lag": float(self.lags[int(idx)]),
                "prob": float(prob[int(idx)]),
            })
        if hypotheses:
            self.last_quality["topk_lag_spread"] = float(
                max(h["lag"] for h in hypotheses) - min(h["lag"] for h in hypotheses)
            )
            self.last_quality["topk_prob_mass"] = float(
                sum(h["prob"] for h in hypotheses)
            )
        return hypotheses

    def estimate_pair_hypotheses(self, sig_i, sig_j, top_k: int = 3,
                                 lag_limit_samples=None):
        prob = self.score_pair(sig_i, sig_j)
        return self._topk_from_prob(
            prob, top_k=top_k, lag_limit_samples=lag_limit_samples
        )

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        prob = self.score_pair(sig_i, sig_j)
        search_mask = _lag_mask(self.lags, lag_limit_samples)
        idx, lag, search_mask = _peak_from_score(
            self.lags, prob, search_mask, sub_sample=sub_sample
        )
        if not return_quality:
            return lag
        weight, peak, sidelobe_ratio = _quality_from_score(prob, idx, search_mask)
        top1, top2, margin, post_mean, post_var, entropy = (
            LearnedHardBinCompressedTDOAEstimator._posterior_quality(prob, self.lags)
        )
        entropy_conf = float(np.clip(1.0 - entropy, 0.0, 1.0))
        calibrated = float(np.clip(
            (0.45 * weight)
            + (0.35 * np.clip(10.0 * margin, 0.02, 10.0))
            + (0.20 * np.clip(8.0 / (np.sqrt(post_var) + 0.5), 0.02, 10.0)),
            0.02, 10.0,
        ))
        calibrated *= float(np.clip(0.25 + 0.75 * entropy_conf, 0.02, 1.0))
        calibrated = float(np.clip(calibrated, 0.02, 10.0))
        self.last_quality.update({
            "top1_prob": top1,
            "top2_prob": top2,
            "peak_margin": margin,
            "posterior_mean": post_mean,
            "posterior_var": post_var,
            "posterior_std": float(np.sqrt(post_var)),
            "posterior_entropy": entropy,
            "entropy_confidence": entropy_conf,
            "sidelobe_weight": float(weight),
            "pair_weight": calibrated,
        })
        # Populate top-K diagnostics for later pair-level debugging. The current
        # all-pair WLS still consumes the top-1 lag, keeping localization chain
        # comparable with existing direct baselines.
        self._topk_from_prob(prob, top_k=3, lag_limit_samples=lag_limit_samples)
        return lag, calibrated, peak, sidelobe_ratio

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        return _diagnostics_from_quality_estimator(
            self, batch_np, meta, fs, c, use_los_only=use_los_only,
            sub_sample=sub_sample, lag_limit_samples=lag_limit_samples,
        )

    def alias_diagnostics(self, lag_limit_samples=None, threshold=0.8):
        diag = {
            "v5b_expert_gated": True,
            "low_confidence_power_prior": float(self.low_confidence_power_prior),
        }
        for key, est in self.experts.items():
            if hasattr(est, "alias_diagnostics"):
                sub = est.alias_diagnostics(
                    lag_limit_samples=lag_limit_samples, threshold=threshold
                )
                for sub_key, value in sub.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        diag[f"{key}_{sub_key}"] = float(value)
        return diag

    def pair_diagnostic(self, sig_i, sig_j, true_tdoa, top_k=5):
        """Comprehensive per-pair diagnostic for top-K oracle analysis.

        Returns a dict with fused and individual expert top-K lags/probs,
        hit/miss flags, expert quality metrics, and gating weights.
        This is eval-only and does not modify any model state.
        """
        power_est = self.experts["power"]
        geo_est = self.experts["geohybrid"]
        power_prob = power_est.score_pair(sig_i, sig_j)
        geo_prob = geo_est.score_pair(sig_i, sig_j)
        power_q = self._expert_quality(power_prob, self.lags)
        geo_q = self._expert_quality(geo_prob, self.lags)
        wp, wg, low_conf = self._mixture_weights(power_q, geo_q)
        fused_prob = wp * power_prob + wg * geo_prob
        fused_prob = fused_prob / (np.sum(fused_prob) + 1e-12)

        ranked = np.argsort(fused_prob)[::-1]
        topk_lags = [float(self.lags[int(idx)]) for idx in ranked[:top_k]]
        topk_probs = [float(fused_prob[int(idx)]) for idx in ranked[:top_k]]

        def _hit(lag, tol):
            return abs(lag - float(true_tdoa)) <= float(tol)

        topk_hit1 = [_hit(l, 1.0) for l in topk_lags]
        topk_hit2 = [_hit(l, 2.0) for l in topk_lags]

        power_ranked = np.argsort(power_prob)[::-1]
        p_top1_lag = float(self.lags[int(power_ranked[0])])
        p_top1_hit1 = _hit(p_top1_lag, 1.0)

        geo_ranked = np.argsort(geo_prob)[::-1]
        g_top1_lag = float(self.lags[int(geo_ranked[0])])
        g_top1_hit1 = _hit(g_top1_lag, 1.0)

        fused_top1, fused_top2, fused_margin, fused_mean, fused_var, fused_entropy = (
            LearnedHardBinCompressedTDOAEstimator._posterior_quality(fused_prob, self.lags)
        )

        oracle_hit1 = bool(p_top1_hit1 or g_top1_hit1)

        # Compute pair_weight the same way as estimate_pair()
        search_mask = _lag_mask(self.lags, None)
        idx, _, search_mask = _peak_from_score(
            self.lags, fused_prob, search_mask, sub_sample=True
        )
        sidelobe_weight, peak, sidelobe_ratio = _quality_from_score(
            fused_prob, idx, search_mask
        )
        entropy_conf = float(np.clip(1.0 - fused_entropy, 0.0, 1.0))
        pair_weight = float(np.clip(
            (0.45 * sidelobe_weight)
            + (0.35 * np.clip(10.0 * fused_margin, 0.02, 10.0))
            + (0.20 * np.clip(8.0 / (np.sqrt(fused_var) + 0.5), 0.02, 10.0)),
            0.02, 10.0,
        ))
        pair_weight *= float(np.clip(0.25 + 0.75 * entropy_conf, 0.02, 1.0))
        pair_weight = float(np.clip(pair_weight, 0.02, 10.0))

        return {
            "true_tdoa": float(true_tdoa),
            "top1_lag": topk_lags[0] if topk_lags else np.nan,
            "top1_prob": topk_probs[0] if topk_probs else np.nan,
            "top2_lag": topk_lags[1] if len(topk_lags) > 1 else np.nan,
            "top2_prob": topk_probs[1] if len(topk_probs) > 1 else np.nan,
            "top3_lag": topk_lags[2] if len(topk_lags) > 2 else np.nan,
            "top3_prob": topk_probs[2] if len(topk_probs) > 2 else np.nan,
            "top4_lag": topk_lags[3] if len(topk_lags) > 3 else np.nan,
            "top4_prob": topk_probs[3] if len(topk_probs) > 3 else np.nan,
            "top5_lag": topk_lags[4] if len(topk_lags) > 4 else np.nan,
            "top5_prob": topk_probs[4] if len(topk_probs) > 4 else np.nan,
            "top1_hit1": bool(topk_hit1[0]) if topk_hit1 else False,
            "top1_hit2": bool(topk_hit2[0]) if topk_hit2 else False,
            "top3_hit1": any(topk_hit1[:3]) if len(topk_hit1) >= 3 else any(topk_hit1),
            "top3_hit2": any(topk_hit2[:3]) if len(topk_hit2) >= 3 else any(topk_hit2),
            "top5_hit1": any(topk_hit1[:5]) if len(topk_hit1) >= 5 else any(topk_hit1),
            "top5_hit2": any(topk_hit2[:5]) if len(topk_hit2) >= 5 else any(topk_hit2),
            "power_top1_lag": p_top1_lag,
            "power_top1_prob": float(power_prob[int(power_ranked[0])]),
            "power_top1_hit1": bool(p_top1_hit1),
            "power_entropy": float(power_q["entropy"]),
            "power_margin": float(power_q["margin"]),
            "power_quality": float(power_q["quality"]),
            "geo_top1_lag": g_top1_lag,
            "geo_top1_prob": float(geo_prob[int(geo_ranked[0])]),
            "geo_top1_hit1": bool(g_top1_hit1),
            "geo_entropy": float(geo_q["entropy"]),
            "geo_margin": float(geo_q["margin"]),
            "geo_quality": float(geo_q["quality"]),
            "oracle_expert_hit1": oracle_hit1,
            "expert_power_weight": float(wp),
            "expert_geohybrid_weight": float(wg),
            "low_confidence_prior": bool(low_conf),
            "posterior_entropy": float(fused_entropy),
            "posterior_variance": float(fused_var),
            "peak_margin": float(fused_margin),
            "pair_weight": pair_weight,
        }


def run_v5b_topk_diagnostics(simulator, device, v5b_estimator, snr_range,
                              num_trials=200, snapshot_indices=None, seed=42):
    """Run pair-level top-K diagnostics across the eval SNR range.

    Returns a pandas DataFrame with per-pair top-K hit/miss, expert
    posteriors, gating weights, and oracle upper bounds.  Requires
    ``pandas`` (already available in the project environment).
    """
    import pandas as pd

    # Expand snapshot_indices to match num_trials if necessary
    if snapshot_indices is not None:
        snapshot_indices = np.asarray(snapshot_indices, dtype=int)
        if len(snapshot_indices) < num_trials:
            rng = np.random.RandomState(seed + 10000)
            snapshot_indices = rng.choice(snapshot_indices, size=num_trials, replace=True)

    rows = []
    for snr_db in snr_range:
        snr_db = float(snr_db)
        X_noisy, X_clean, meta = simulator.generate_urban_batch(
            num_trials, snr_db=snr_db, snapshot_indices=snapshot_indices, seed=seed
        )
        for trial_idx in range(num_trials):
            los_arr = meta["los"][trial_idx]
            los_indices = np.where(los_arr)[0]
            if len(los_indices) < 2:
                continue
            delays_float = meta["delay_float"][trial_idx]
            for pi, i in enumerate(los_indices):
                for pj, j in enumerate(los_indices):
                    if i >= j:
                        continue
                    sig_i_ch = X_noisy[trial_idx, i]
                    sig_j_ch = X_noisy[trial_idx, j]
                    sig_i = sig_i_ch[0, :] + 1j * sig_i_ch[1, :]
                    sig_j = sig_j_ch[0, :] + 1j * sig_j_ch[1, :]
                    true_tdoa = float(delays_float[i] - delays_float[j])
                    diag = v5b_estimator.pair_diagnostic(sig_i, sig_j, true_tdoa)
                    diag["SNR_dB"] = snr_db
                    diag["trial_idx"] = int(trial_idx)
                    if "snapshot_id" in meta:
                        diag["snapshot_id"] = int(meta["snapshot_id"][trial_idx])
                    diag["uav_i"] = int(i)
                    diag["uav_j"] = int(j)
                    diag["pair_id"] = f"t{trial_idx}_u{i}_u{j}"
                    rows.append(diag)

    df = pd.DataFrame(rows)
    # Add SNR bucket labels for convenient grouping
    snr_arr = df["SNR_dB"].values
    buckets = np.full(len(df), "other", dtype=object)
    buckets[snr_arr <= 0.0] = "low"
    buckets[(snr_arr >= 2.0) & (snr_arr <= 6.0)] = "mid"
    buckets[snr_arr >= 8.0] = "high"
    df["snr_bucket"] = buckets
    return df
