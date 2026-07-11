# train.py
import time
import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold, GroupKFold

from model import DAE
from signal_gen import SignalSimulator


class RefreshableUrbanWaveformDataset(Dataset):
    """
    固定 snapshot/UAV 计划、按 epoch 周期性重采样 waveform 的 urban8 训练集。

    验证集仍使用固定 TensorDataset；只有训练集刷新基带符号、SNR 与噪声，
    用来缓解 fixed dataset 在 200 epoch 训练下的记忆化。
    """
    def __init__(self, sim, snapshot_indices, uav_indices, base_seed=42,
                 refresh_interval=1):
        self.sim = sim
        self.snapshot_indices = np.asarray(snapshot_indices, dtype=int)
        self.uav_indices = np.asarray(uav_indices, dtype=int)
        self.base_seed = int(base_seed)
        self.refresh_interval = max(1, int(refresh_interval))
        self._refresh_id = None
        self.X_noisy = None
        self.X_clean = None
        self.set_epoch(0)

    def set_epoch(self, epoch):
        refresh_id = int(epoch) // self.refresh_interval
        if refresh_id == self._refresh_id:
            return
        seed = self.base_seed + refresh_id
        self.X_noisy, self.X_clean = self.sim.generate_urban_training_dataset_from_plan(
            self.snapshot_indices, self.uav_indices, seed=seed, return_groups=False
        )
        self._refresh_id = refresh_id

    def __len__(self):
        return len(self.snapshot_indices)

    def __getitem__(self, idx):
        return self.X_noisy[idx], self.X_clean[idx]


class RefreshableUrbanPairDataset(Dataset):
    """
    固定 snapshot/UAV-pair 计划、按 epoch 周期性重采样 waveform 的 urban8 pair 训练集。

    用于 FreqDAE v2：训练时显式看到同一 observation 内的 LOS pair，目标
    TDOA 使用几何距离差的浮点采样值，与最终 all-pair WLS 评估保持一致。
    """
    def __init__(self, sim, snapshot_indices, uav_i_indices, uav_j_indices,
                 base_seed=42, refresh_interval=1):
        self.sim = sim
        self.snapshot_indices = np.asarray(snapshot_indices, dtype=int)
        self.uav_i_indices = np.asarray(uav_i_indices, dtype=int)
        self.uav_j_indices = np.asarray(uav_j_indices, dtype=int)
        self.base_seed = int(base_seed)
        self.refresh_interval = max(1, int(refresh_interval))
        self._refresh_id = None
        self.X1_noisy = None
        self.X1_clean = None
        self.X2_noisy = None
        self.X2_clean = None
        self.tdoa = None
        self.set_epoch(0)

    def set_epoch(self, epoch):
        refresh_id = int(epoch) // self.refresh_interval
        if refresh_id == self._refresh_id:
            return
        seed = self.base_seed + refresh_id
        self.X1_noisy, self.X1_clean, self.X2_noisy, self.X2_clean, self.tdoa = (
            self.sim.generate_urban_pair_training_dataset_from_plan(
                self.snapshot_indices,
                self.uav_i_indices,
                self.uav_j_indices,
                seed=seed,
                return_groups=False,
            )
        )
        self._refresh_id = refresh_id

    def __len__(self):
        return len(self.snapshot_indices)

    def __getitem__(self, idx):
        return (
            self.X1_noisy[idx], self.X1_clean[idx],
            self.X2_noisy[idx], self.X2_clean[idx],
            self.tdoa[idx],
        )


def _refresh_epoch_dataset(dataset, epoch):
    """支持 DataLoader.dataset 或 Subset.dataset 上的 set_epoch(epoch)。"""
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, "set_epoch"):
        dataset.dataset.set_epoch(epoch)


def estimate_batch_snr(noisy, clean):
    """
    估计 batch 的平均 SNR

    参数:
        noisy: (batch, 2, signal_len) 含噪信号
        clean: (batch, 2, signal_len) 纯净信号
    返回:
        snr_db: 平均 SNR (dB)
    """
    clean_power = torch.mean(clean[:, 0, :] ** 2 + clean[:, 1, :] ** 2, dim=1)
    noise_power = torch.mean((noisy[:, 0, :] - clean[:, 0, :]) ** 2 +
                             (noisy[:, 1, :] - clean[:, 1, :]) ** 2, dim=1)
    snr_linear = clean_power / (noise_power + 1e-10)
    snr_db = 10 * torch.log10(snr_linear + 1e-10)
    return torch.mean(snr_db)


def estimate_per_sample_snr(noisy, clean):
    """
    逐样本估计 SNR

    参数:
        noisy: (batch, 2, signal_len) 含噪信号
        clean: (batch, 2, signal_len) 纯净信号
    返回:
        snr_db: (batch,) 每个样本的 SNR (dB)
    """
    clean_power = torch.mean(clean[:, 0, :] ** 2 + clean[:, 1, :] ** 2, dim=1)
    noise_power = torch.mean((noisy[:, 0, :] - clean[:, 0, :]) ** 2 +
                             (noisy[:, 1, :] - clean[:, 1, :]) ** 2, dim=1)
    snr_linear = clean_power / (noise_power + 1e-10)
    snr_db = 10 * torch.log10(snr_linear + 1e-10)
    return snr_db


def gcc_torch(sig1_real, sig1_imag, sig2_real, sig2_imag, fft_len=None):
    """
    可微分 GCC 计算（PyTorch 版本）

    返回互相关函数的**幅度**（而非实部）：
    - 与 gcc_peak_loss（使用 |cc|）保持定义一致
    - 保留复数互相关的全部幅度信息

    参数:
        sig1_real, sig1_imag: (batch, signal_len) 实部和虚部
        sig2_real, sig2_imag: (batch, signal_len)
        fft_len: FFT 长度
    返回:
        cc_mag: (batch, fft_len) 互相关幅度 |cc|
    """
    n = sig1_real.shape[-1]
    if fft_len is None:
        # 与 evaluate.py 的 gcc_standard 保持一致，使用完整线性相关长度 2N-1。
        # 旧实现使用 next power-of-two(2048)，会让训练和最终评估的 lag 域不一致。
        fft_len = 2 * n - 1

    sig1 = torch.complex(sig1_real, sig1_imag)
    sig2 = torch.complex(sig2_real, sig2_imag)

    SIG1 = torch.fft.fft(sig1, n=fft_len)
    SIG2 = torch.fft.fft(sig2, n=fft_len)

    R = SIG1 * torch.conj(SIG2)
    cc = torch.fft.ifft(R)
    cc = torch.fft.fftshift(cc, dim=-1)

    return torch.abs(cc)


def crop_gcc_same(cc, signal_len):
    """
    将完整 GCC 序列裁剪到 scipy.signal.correlation_lags(..., mode='same') 对应窗口。

    对 N=1024，完整 lag 为 [-1023, 1023]，same 窗口为 [-512, 511]，
    零延迟索引为 N//2。训练、Peak loss 和 evaluate.py 使用同一 lag 域。
    """
    if cc.shape[-1] == signal_len:
        return cc
    start = (cc.shape[-1] - signal_len) // 2
    return cc[..., start:start + signal_len]


def gcc_peak_loss(y1_real, y1_imag, y2_real, y2_imag, true_tdoa, fft_len=None):
    """
    PNCC 启发的峰值损失：最大化真实 TDOA 位置的 GCC 幅度

    原理：DAE 去噪后，两个接收信号的互相关在真实 TDOA 位置应有强峰。
    损失 = 1 - |GCC(true_tdoa)| / max(|GCC|)，需要最小化。

    参数:
        y1_real, y1_imag: (batch, signal_len) DAE 输出信号 1
        y2_real, y2_imag: (batch, signal_len) DAE 输出信号 2
        true_tdoa: (batch,) 真实 TDOA（采样点）
    返回:
        loss: 标量损失
    """
    n = y1_real.shape[-1]
    if fft_len is None:
        fft_len = 2 * n - 1

    # 计算复数 GCC
    sig1 = torch.complex(y1_real, y1_imag)
    sig2 = torch.complex(y2_real, y2_imag)
    SIG1 = torch.fft.fft(sig1, n=fft_len)
    SIG2 = torch.fft.fft(sig2, n=fft_len)
    R = SIG1 * torch.conj(SIG2)
    cc = torch.fft.ifft(R)
    cc = torch.fft.fftshift(cc, dim=-1)

    # 计算 GCC 幅度，并裁剪到 evaluate.py 使用的 same lag 窗口
    cc_mag = crop_gcc_same(torch.abs(cc), n)  # (batch, n)

    # 真实 TDOA 对应的索引（same 窗口中零延迟在 n//2）
    zero_idx = n // 2
    true_idx = zero_idx + torch.round(true_tdoa).long()
    true_idx = torch.clamp(true_idx, 0, n - 1)

    # 取真实 TDOA 位置的 GCC 幅度
    batch_idx = torch.arange(cc_mag.shape[0], device=cc_mag.device)
    peak_at_true = cc_mag[batch_idx, true_idx]

    # 归一化：除以最大 GCC 幅度
    peak_max = torch.max(cc_mag, dim=-1)[0]
    peak_norm = peak_at_true / (peak_max + 1e-9)

    # 损失：1 - 归一化峰值（最小化 = 最大化真实 TDOA 处的峰）
    return torch.mean(1.0 - peak_norm)


def gcc_peak_loss_per_sample(y1_real, y1_imag, y2_real, y2_imag, true_tdoa, fft_len=None):
    """
    逐样本 PNCC 峰值损失（不取均值），配合自适应峰值权重使用。

    返回:
        loss_per_sample: (batch,) 每个样本的 1 - |GCC(true_tdoa)| / max(|GCC|)
    """
    n = y1_real.shape[-1]
    if fft_len is None:
        fft_len = 2 * n - 1

    sig1 = torch.complex(y1_real, y1_imag)
    sig2 = torch.complex(y2_real, y2_imag)
    SIG1 = torch.fft.fft(sig1, n=fft_len)
    SIG2 = torch.fft.fft(sig2, n=fft_len)
    R = SIG1 * torch.conj(SIG2)
    cc = torch.fft.ifft(R)
    cc = torch.fft.fftshift(cc, dim=-1)

    cc_mag = crop_gcc_same(torch.abs(cc), n)
    zero_idx = n // 2
    true_idx = zero_idx + torch.round(true_tdoa).long()
    true_idx = torch.clamp(true_idx, 0, n - 1)

    batch_idx = torch.arange(cc_mag.shape[0], device=cc_mag.device)
    peak_at_true = cc_mag[batch_idx, true_idx]
    peak_max = torch.max(cc_mag, dim=-1)[0]
    peak_norm = peak_at_true / (peak_max + 1e-9)

    return 1.0 - peak_norm  # (batch,)


def adaptive_peak_lambda(snr, peak_max=0.15, snr_center=0.0, temperature=5.0):
    """
    峰值损失权重：低 SNR 时 GCC 主峰更易被噪声/多径破坏 → Peak 物理先验更强。

    λ_peak(SNR) = peak_max · σ((snr_center - SNR) / temperature)

    设计动机：
      - 低 SNR：GCC 形状被噪声破坏，真实 TDOA 处需要更强先验拉起梯度
      - 高 SNR：Corr 形状匹配已较可靠，Peak 权重自然减弱，避免过度尖峰化

    参数:
        snr:          (batch,) 每样本 SNR (dB)
        peak_max:     峰值损失最大权重（默认 0.15）
        snr_center:   sigmoid 中心点 SNR (dB)
        temperature:  sigmoid 温度参数

    返回:
        lambda_peak: (batch,) 每样本的峰值损失权重
    """
    return peak_max * torch.sigmoid((snr_center - snr) / temperature)


def fisher_tdoa_loss(y1_r, y1_i, y2_r, y2_i, c1_r, c1_i, c2_r, c2_i, fs=40e6):
    """
    Fisher信息引导的RMS带宽保持损失。

    TDOA的Cramér-Rao下界与信号RMS带宽成反比（Fowler 2000a, Weiss 2026）。
    保持DAE输出信号的RMS带宽 = 保持TDOA估计精度的物理潜力。

    rms_bw = sqrt( sum_f (2*pi*f)^2 * |S(f)|^2 / sum_f |S(f)|^2 )

    L_FI = |rms_bw_DAE - rms_bw_Clean| / (rms_bw_Clean + eps)

    这是标量损失，与GCC形状匹配完全互补：
    - GCC相关性损失: 匹配互相关结构的形状
    - RMS带宽损失:   匹配TDOA精度的物理上界
    """
    n = y1_r.shape[-1]

    freqs = torch.fft.fftfreq(n, d=1.0 / fs).to(y1_r.device)
    w = (2.0 * torch.pi * freqs) ** 2  # (n,)

    def relative_bandwidth_error(y_r, y_i, c_r, c_i):
        y = torch.complex(y_r, y_i)
        clean = torch.complex(c_r, c_i)
        psd_y = torch.abs(torch.fft.fft(y, dim=-1)) ** 2
        psd_clean = torch.abs(torch.fft.fft(clean, dim=-1)) ** 2
        rms_y = torch.sqrt(
            torch.sum(w[None, :] * psd_y, dim=-1)
            / (torch.sum(psd_y, dim=-1) + 1e-9)
        )
        rms_clean = torch.sqrt(
            torch.sum(w[None, :] * psd_clean, dim=-1)
            / (torch.sum(psd_clean, dim=-1) + 1e-9)
        )
        return torch.abs(rms_y - rms_clean) / (rms_clean + 1e-9)

    rel1 = relative_bandwidth_error(y1_r, y1_i, c1_r, c1_i)
    rel2 = relative_bandwidth_error(y2_r, y2_i, c2_r, c2_i)
    return torch.mean(0.5 * (rel1 + rel2))


def frequency_task_reconstruction_loss(y, target, spectral_blend=0.25,
                                       fisher_power=2.0):
    """
    Reconstruction loss with a TDOA-aware spectral term.

    The time-domain NMSE preserves waveform fidelity. The spectral NMSE
    upweights high-RMS-bandwidth components, reflecting the delay-estimation
    Fisher-information intuition without hard-selecting bins or changing the
    evaluation chain.
    """
    spectral_blend = float(max(0.0, min(1.0, spectral_blend)))
    fisher_power = float(max(0.0, fisher_power))

    sig_power = torch.mean(target[:, 0, :] ** 2 + target[:, 1, :] ** 2, dim=1)
    # Complex NMSE must use I/Q power as a sum. Averaging over both channel and
    # time would make a zero output look like NMSE=0.5 instead of NMSE=1.0.
    time_mse = torch.mean(
        (y[:, 0, :] - target[:, 0, :]) ** 2
        + (y[:, 1, :] - target[:, 1, :]) ** 2,
        dim=1,
    )
    time_nmse = time_mse / (sig_power + 1e-9)

    yc = torch.complex(y[:, 0, :], y[:, 1, :])
    tc = torch.complex(target[:, 0, :], target[:, 1, :])
    Y = torch.fft.fft(yc, dim=-1)
    T = torch.fft.fft(tc, dim=-1)

    freq = torch.fft.fftfreq(y.shape[-1], device=y.device, dtype=y.dtype)
    freq_norm = torch.abs(freq) / (torch.max(torch.abs(freq)) + 1e-12)
    weight = 1.0 + freq_norm.pow(fisher_power)
    spec_err = torch.mean(weight[None, :] * torch.abs(Y - T) ** 2, dim=1)
    spec_ref = torch.mean(weight[None, :] * torch.abs(T) ** 2, dim=1)
    spec_nmse = spec_err / (spec_ref + 1e-9)

    combined = (1.0 - spectral_blend) * time_nmse + spectral_blend * spec_nmse
    return torch.mean(combined), torch.mean(time_nmse), torch.mean(spec_nmse)


def pairwise_cross_spectrum_phase_loss(y1, y2, c1, c2, fisher_power=1.0):
    """
    Pairwise cross-spectrum phase consistency loss.

    TDOA is encoded in the relative phase slope between two receivers. This loss
    compares the normalized cross spectra of the DAE outputs and clean targets,
    so it directly penalizes systematic pairwise delay bias without caring about
    trivial output amplitude scaling.
    """
    yc1 = torch.complex(y1[:, 0, :], y1[:, 1, :])
    yc2 = torch.complex(y2[:, 0, :], y2[:, 1, :])
    cc1 = torch.complex(c1[:, 0, :], c1[:, 1, :])
    cc2 = torch.complex(c2[:, 0, :], c2[:, 1, :])

    Y_raw = torch.fft.fft(yc1, dim=-1) * torch.conj(torch.fft.fft(yc2, dim=-1))
    C_raw = torch.fft.fft(cc1, dim=-1) * torch.conj(torch.fft.fft(cc2, dim=-1))
    c_mag = torch.abs(C_raw)
    Y = Y_raw / (torch.abs(Y_raw) + 1e-9)
    C = C_raw / (c_mag + 1e-9)

    freq = torch.fft.fftfreq(y1.shape[-1], device=y1.device, dtype=y1.dtype)
    freq_norm = torch.abs(freq) / (torch.max(torch.abs(freq)) + 1e-12)
    phase_weight = 1.0 + freq_norm.pow(float(max(0.0, fisher_power)))
    # Reliability must come from the unnormalized clean cross-spectrum magnitude.
    # If it is computed after phase normalization, it is nearly constant and
    # low-energy/noisy bins receive the same phase weight as active bins.
    reliability = torch.sqrt(
        c_mag / (torch.max(c_mag, dim=-1, keepdim=True)[0] + 1e-9)
    )
    weight = phase_weight[None, :] * reliability

    phase_agreement = torch.real(Y * torch.conj(C))
    loss_per_sample = torch.sum(weight * (1.0 - phase_agreement), dim=-1) / (
        torch.sum(weight, dim=-1) + 1e-9
    )
    return torch.mean(loss_per_sample)


def soft_gcc_peak_loss_per_sample(gcc_mag_same, true_tdoa, sigma=1.5,
                                  sharpness_weight=0.0):
    """
    Float-TDOA soft peak loss on a same-window GCC magnitude sequence.

    Unlike the legacy integer-index peak loss, this uses a Gaussian target
    around the physical float TDOA. A small optional sharpness term compares the
    true-lag neighborhood against the strongest outside-lag response, preventing
    the broad-peak shortcut observed in FreqDAE v2.
    """
    n = gcc_mag_same.shape[-1]
    lags = torch.arange(n, device=gcc_mag_same.device, dtype=gcc_mag_same.dtype) - (n // 2)
    target = true_tdoa.to(gcc_mag_same.device, dtype=gcc_mag_same.dtype)
    sigma = max(float(sigma), 1e-3)
    weights = torch.exp(-0.5 * ((lags[None, :] - target[:, None]) / sigma) ** 2)
    peak_at_true = torch.sum(weights * gcc_mag_same, dim=-1) / (torch.sum(weights, dim=-1) + 1e-9)
    peak_max = torch.max(gcc_mag_same, dim=-1)[0]
    peak_loss = 1.0 - peak_at_true / (peak_max + 1e-9)
    sharpness_weight = float(max(0.0, sharpness_weight))
    if sharpness_weight <= 0.0:
        return peak_loss

    guard = max(2.0, 2.0 * sigma)
    outside = torch.abs(lags[None, :] - target[:, None]) > guard
    side_values = torch.where(
        outside,
        gcc_mag_same,
        torch.full_like(gcc_mag_same, -1e9),
    )
    side_max = torch.max(side_values, dim=-1)[0].clamp_min(0.0)
    sharpness_loss = side_max / (peak_at_true + 1e-9)
    return peak_loss + sharpness_weight * sharpness_loss


def soft_tdoa_distribution_loss(logits, true_tdoa, sigma=1.5):
    """
    Cross-entropy from a soft Gaussian TDOA target to predicted lag logits.

    The lag axis is the same centered "same" GCC window used elsewhere:
    index n//2 corresponds to zero delay, negative/positive indices correspond
    to negative/positive TDOA samples.
    """
    n = logits.shape[-1]
    lags = torch.arange(n, device=logits.device, dtype=logits.dtype) - (n // 2)
    target = true_tdoa.to(logits.device, dtype=logits.dtype)
    sigma = max(float(sigma), 1e-3)
    weights = torch.exp(-0.5 * ((lags[None, :] - target[:, None]) / sigma) ** 2)
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-9)
    logp = torch.log_softmax(logits, dim=-1)
    return torch.mean(-torch.sum(weights * logp, dim=-1))


def soft_tdoa_distribution_loss_per_sample(logits, true_tdoa, sigma=1.5):
    n = logits.shape[-1]
    lags = torch.arange(n, device=logits.device, dtype=logits.dtype) - (n // 2)
    target = true_tdoa.to(logits.device, dtype=logits.dtype)
    sigma = max(float(sigma), 1e-3)
    weights = torch.exp(-0.5 * ((lags[None, :] - target[:, None]) / sigma) ** 2)
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-9)
    logp = torch.log_softmax(logits, dim=-1)
    return -torch.sum(weights * logp, dim=-1)


def expected_tdoa_from_logits(logits):
    n = logits.shape[-1]
    lags = torch.arange(n, device=logits.device, dtype=logits.dtype) - (n // 2)
    prob = torch.softmax(logits, dim=-1)
    return torch.sum(prob * lags[None, :], dim=-1)


def tdoa_uncertainty_nll(logits, log_var, true_tdoa):
    pred = expected_tdoa_from_logits(logits)
    target = true_tdoa.to(logits.device, dtype=logits.dtype)
    err2 = (pred - target).pow(2)
    log_var = torch.clamp(log_var.to(logits.device, dtype=logits.dtype), -6.0, 6.0)
    return torch.mean(0.5 * (torch.exp(-log_var) * err2 + log_var))


def nested_task_sufficient_losses(model, z_full1, z_full2, true_tdoa,
                                  nested_crs=(4, 8, 16), sigma=1.5):
    """
    Evaluate task-head losses for all nested prefixes from one full latent pair.

    Returns:
        task_loss: mean soft-label CE across CR4/8/16 prefixes.
        uncertainty_loss: Gaussian NLL from predicted TDOA uncertainty.
        monotonic_loss: penalizes CR4 worse than CR8 and CR8 worse than CR16.
    """
    ce_by_cr = {}
    uncertainty_terms = []
    for cr in nested_crs:
        z1 = model._mask_latent(z_full1, cr=cr)
        z2 = model._mask_latent(z_full2, cr=cr)
        if hasattr(model, "pair_task_outputs_from_latent"):
            logits, log_var = model.pair_task_outputs_from_latent(z1, z2)
            uncertainty_terms.append(tdoa_uncertainty_nll(logits, log_var, true_tdoa))
        else:
            logits = model.pair_task_logits_from_latent(z1, z2)
        ce_by_cr[int(cr)] = soft_tdoa_distribution_loss_per_sample(
            logits, true_tdoa.float(), sigma=sigma
        )

    task_loss = torch.stack([v.mean() for v in ce_by_cr.values()]).mean()
    uncertainty_loss = (
        torch.stack(uncertainty_terms).mean()
        if uncertainty_terms
        else torch.tensor(0.0, device=z_full1.device)
    )

    ordered = [int(v) for v in sorted(nested_crs)]
    monotonic_terms = []
    for better, worse in zip(ordered[:-1], ordered[1:]):
        # Smaller CR means larger latent budget; its task CE should not be worse.
        monotonic_terms.append(torch.relu(ce_by_cr[better] - ce_by_cr[worse]).mean())
    monotonic_loss = (
        torch.stack(monotonic_terms).mean()
        if monotonic_terms
        else torch.tensor(0.0, device=z_full1.device)
    )
    return task_loss, uncertainty_loss, monotonic_loss


def train_one_fold(device, model, train_loader, val_loader, epochs, lr,
                   fold_idx, patience=20, weight_decay=1e-4, corr_weight=1.0,
                   lambda_peak=0.0, beta_fi=0.0, epsilon_mse=1.0,
                   mse_weight_max=None, phase_mix=None, selection_start_epoch=None,
                   use_adaptive_peak=False, snr_threshold=0.0, lambda_temperature=5.0,
                   loss_mode="task", early_stopping=True, restore_best=True,
                   spectral_blend=0.25, spectral_power=2.0,
                   pair_phase_weight=0.0, soft_peak_sigma=1.5,
                   soft_peak_sharpness_weight=0.0,
                   pair_task_weight=0.0, pair_uncertainty_weight=0.0,
                   cr_monotonic_weight=0.0, nested_crs=(4, 8, 16)):
    """
    在单个 fold 上训练模型，含早停机制。

    统一损失（R20固定比例混合NMSE）:
        NMSE_mix = (1-ε)·NMSE_mag + ε·NMSE_complex
        L = corr_weight·L_corr + λp(SNR)·L_peak + ε_current·NMSE_mix
    ε_current 仅控制NMSE总权重warmup；混合比例固定为 α=ε。

    参数:
        epsilon_mse:    NMSE正则项权重（统一公式，所有CR使用同一路径）
        lambda_peak:    PNCC 峰值损失最大权重
        corr_weight:    GCC形状损失显式权重（R20固定为1.0）
        use_adaptive_peak: 是否启用逐样本SNR自适应Peak权重
    """
    # R20.1 decouples total reconstruction pressure from complex-phase mixing.
    # epsilon_mse remains a legacy alias when mse_weight_max/phase_mix are omitted.
    if mse_weight_max is None:
        mse_weight_max = epsilon_mse
    if phase_mix is None:
        phase_mix = epsilon_mse
    phase_mix = float(max(0.0, min(1.0, phase_mix)))

    freq_task_recon_enabled = loss_mode == "freq_task_mse"
    nested_pair_task_enabled = loss_mode in ("freq_task_nested_pair", "freq_task_v4_min_pair")
    v4_task_sufficient_enabled = loss_mode == "freq_task_v4_min_pair"
    pair_freq_task_enabled = loss_mode in (
        "freq_task_pair", "freq_task_nested_pair", "freq_task_v4_min_pair"
    )
    task_loss_enabled = pair_freq_task_enabled or (
        (loss_mode != "paper_mse")
        and (not freq_task_recon_enabled)
        and (
            corr_weight > 0 or lambda_peak > 0 or beta_fi > 0
            or pair_phase_weight > 0 or pair_task_weight > 0
            or pair_uncertainty_weight > 0 or cr_monotonic_weight > 0
        )
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    train_loss_hist = []
    val_loss_hist = []
    train_nmse_mag_hist = []
    train_nmse_complex_hist = []
    val_nmse_mag_hist = []
    val_nmse_complex_hist = []
    corr_hist = []
    peak_hist = []
    snr_hist = []
    val_corr_hist = []
    val_peak_hist = []
    val_total_hist = []
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    stopped_epoch = epochs
    warmup_ep = 100 if task_loss_enabled and not pair_freq_task_enabled else 0
    if selection_start_epoch is None:
        selection_start_epoch = warmup_ep if task_loss_enabled else 1

    for ep in range(epochs):
        _refresh_epoch_dataset(train_loader.dataset, ep)

        # 任务驱动分支保留 warmup；论文复现分支使用标准 MSE，不做课程学习。
        progress = min((ep + 1) / warmup_ep, 1.0) if warmup_ep > 0 else 1.0
        eps_factor = 0.5 * (1.0 - math.cos(math.pi * progress)) if warmup_ep > 0 else 1.0
        mse_weight_current = mse_weight_max * eps_factor
        t_ep = time.time()

        model.train()
        ep_loss = 0.0
        ep_nmse_mag = 0.0
        ep_nmse_complex = 0.0
        ep_corr = 0.0
        ep_peak = 0.0
        ep_fi = 0.0
        ep_snr = 0.0
        n_batches = 0
        for batch in train_loader:
            if task_loss_enabled and len(batch) >= 4:
                # 配对数据：(X1_noisy, X1_clean, X2_noisy, X2_clean[, tdoa])
                bx1, by1, bx2, by2 = [b.to(device) for b in batch[:4]]
                true_tdoa = batch[4].to(device) if len(batch) >= 5 else None
                optimizer.zero_grad()

                # 合并 X1/X2 为单次前向传播
                bx_all = torch.cat([bx1, bx2], dim=0)
                active_cr = (
                    nested_crs[(ep + n_batches) % len(nested_crs)]
                    if nested_pair_task_enabled else None
                )
                if nested_pair_task_enabled and hasattr(model, "set_active_cr"):
                    model.set_active_cr(active_cr)
                if v4_task_sufficient_enabled and hasattr(model, "encode_full_latent"):
                    if active_cr is None:
                        raise RuntimeError("nested V4 training requires an active CR")
                    z_full_all = model.encode_full_latent(bx_all)
                    z_all = model._mask_latent(z_full_all, cr=active_cr)
                    y_all = model.decode_latent(z_all)
                    z_full1, z_full2 = z_full_all.chunk(2, dim=0)
                    z1, z2 = z_all.chunk(2, dim=0)
                elif nested_pair_task_enabled:
                    y_all, z_all = model(bx_all, return_latent=True)
                    z1, z2 = z_all.chunk(2, dim=0)
                    z_full1 = z_full2 = None
                else:
                    y_all = model(bx_all)
                    z1 = z2 = None
                    z_full1 = z_full2 = None
                y1, y2 = y_all.chunk(2, dim=0)

                if pair_freq_task_enabled:
                    # FreqDAE v2: reconstruction remains single-waveform, while
                    # pairwise phase/GCC terms directly constrain TDOA behavior.
                    y_pair = torch.cat([y1, y2], dim=0)
                    clean_pair = torch.cat([by1, by2], dim=0)
                    loss_mse_norm, loss_mse_mag, loss_mse_complex = (
                        frequency_task_reconstruction_loss(
                            y_pair, clean_pair,
                            spectral_blend=spectral_blend,
                            fisher_power=spectral_power,
                        )
                    )
                else:
                    # === 固定比例混合 NMSE（α=ε，无ramp） ===
                    # 包络NMSE：约束幅度轮廓，不与Corr冲突，但遇相位地板(~0.38)
                    # 复NMSE：微量加入打破相位地板，比例=ε，ε小则几乎不进
                    mag_y1 = torch.sqrt(y1[:,0,:]**2 + y1[:,1,:]**2 + 1e-10)
                    mag_y2 = torch.sqrt(y2[:,0,:]**2 + y2[:,1,:]**2 + 1e-10)
                    mag_s1 = torch.sqrt(by1[:,0,:]**2 + by1[:,1,:]**2 + 1e-10)
                    mag_s2 = torch.sqrt(by2[:,0,:]**2 + by2[:,1,:]**2 + 1e-10)
                    mag_mse1 = torch.mean((mag_y1 - mag_s1)**2, dim=1)
                    mag_mse2 = torch.mean((mag_y2 - mag_s2)**2, dim=1)
                    mag_mse = mag_mse1 + mag_mse2
                    complex_mse = torch.mean((y1-by1)**2 + (y2-by2)**2, dim=[1,2])
                    sig_power = torch.mean(by1**2 + by2**2, dim=[1,2])
                    nmse_mag = mag_mse / (sig_power + 1e-9)
                    nmse_complex = complex_mse / (sig_power + 1e-9)
                    env_ratio = 1.0 - phase_mix
                    nmse = env_ratio * nmse_mag + phase_mix * nmse_complex
                    loss_mse_norm = torch.mean(nmse)
                    loss_mse_mag = torch.mean(nmse_mag)
                    loss_mse_complex = torch.mean(nmse_complex)

                # Clean 信号 GCC 不需要梯度
                with torch.no_grad():
                    gcc_clean = gcc_torch(by1[:, 0, :], by1[:, 1, :],
                                          by2[:, 0, :], by2[:, 1, :])
                    gcc_clean = crop_gcc_same(gcc_clean, by1.shape[-1])
                    gcc_clean_n = gcc_clean / (torch.max(torch.abs(gcc_clean), dim=-1, keepdim=True)[0] + 1e-9)

                # DAE 输出 GCC 需要梯度
                gcc_dae = gcc_torch(y1[:, 0, :], y1[:, 1, :],
                                    y2[:, 0, :], y2[:, 1, :])
                gcc_dae = crop_gcc_same(gcc_dae, y1.shape[-1])
                gcc_dae_n = gcc_dae / (torch.max(torch.abs(gcc_dae), dim=-1, keepdim=True)[0] + 1e-9)

                # 逐样本 GCC 相关性损失
                corr_per_sample = torch.mean((gcc_dae_n - gcc_clean_n) ** 2, dim=1)  # (batch,)

                # 逐样本 SNR 估计（用于峰值损失的自适应权重 + 日志）
                if use_adaptive_peak and lambda_peak > 0:
                    snr_per_sample = estimate_per_sample_snr(bx1, by1)  # (batch,)
                    avg_snr_batch = torch.mean(snr_per_sample).item()   # 用于日志
                else:
                    snr_per_sample = None
                    avg_snr_batch = 0.0

                # PNCC 峰值损失
                loss_peak = torch.tensor(0.0, device=device)
                if lambda_peak > 0 and true_tdoa is not None:
                    if pair_freq_task_enabled:
                        peak_per_sample = soft_gcc_peak_loss_per_sample(
                            gcc_dae_n, true_tdoa.float(), sigma=soft_peak_sigma,
                            sharpness_weight=soft_peak_sharpness_weight,
                        )
                        loss_peak = lambda_peak * torch.mean(peak_per_sample)
                    elif snr_per_sample is not None:
                        peak_per_sample = gcc_peak_loss_per_sample(
                            y1[:, 0, :], y1[:, 1, :],
                            y2[:, 0, :], y2[:, 1, :],
                            true_tdoa.float()
                        )
                        peak_lambda = adaptive_peak_lambda(
                            snr_per_sample, peak_max=lambda_peak,
                            snr_center=snr_threshold, temperature=lambda_temperature)
                        loss_peak = torch.mean(peak_lambda * peak_per_sample)
                    else:
                        loss_peak = lambda_peak * gcc_peak_loss(
                            y1[:, 0, :], y1[:, 1, :],
                            y2[:, 0, :], y2[:, 1, :],
                            true_tdoa.float()
                        )

                # Fisher 信息损失（可选）
                loss_fi = torch.tensor(0.0, device=device)
                if beta_fi > 0:
                    loss_fi = fisher_tdoa_loss(
                        y1[:, 0, :], y1[:, 1, :], y2[:, 0, :], y2[:, 1, :],
                        by1[:, 0, :], by1[:, 1, :], by2[:, 0, :], by2[:, 1, :]
                    )

                loss_pair_phase = torch.tensor(0.0, device=device)
                if pair_freq_task_enabled and pair_phase_weight > 0:
                    loss_pair_phase = pairwise_cross_spectrum_phase_loss(
                        y1, y2, by1, by2, fisher_power=spectral_power
                    )
                loss_pair_task = torch.tensor(0.0, device=device)
                loss_pair_uncertainty = torch.tensor(0.0, device=device)
                loss_cr_monotonic = torch.tensor(0.0, device=device)
                if (v4_task_sufficient_enabled and true_tdoa is not None
                        and z_full1 is not None):
                    loss_pair_task, loss_pair_uncertainty, loss_cr_monotonic = (
                        nested_task_sufficient_losses(
                            model, z_full1, z_full2, true_tdoa.float(),
                            nested_crs=nested_crs, sigma=soft_peak_sigma
                        )
                    )
                elif (nested_pair_task_enabled and pair_task_weight > 0
                        and true_tdoa is not None
                        and hasattr(model, "pair_task_logits_from_latent")):
                    logits = model.pair_task_logits_from_latent(z1, z2)
                    loss_pair_task = soft_tdoa_distribution_loss(
                        logits, true_tdoa.float(), sigma=soft_peak_sigma
                    )

                loss_corr = torch.mean(corr_per_sample)
                loss = (
                    corr_weight * loss_corr
                    + loss_peak
                    + mse_weight_current * loss_mse_norm
                    + pair_phase_weight * loss_pair_phase
                    + pair_task_weight * loss_pair_task
                    + pair_uncertainty_weight * loss_pair_uncertainty
                    + cr_monotonic_weight * loss_cr_monotonic
                    + beta_fi * loss_fi
                )

                loss.backward()
                optimizer.step()
                ep_loss += loss_mse_norm.item() * bx1.size(0)
                ep_nmse_mag += loss_mse_mag.item() * bx1.size(0)
                ep_nmse_complex += loss_mse_complex.item() * bx1.size(0)
                ep_corr += loss_corr.item() * bx1.size(0)
                ep_peak += loss_peak.item() * bx1.size(0)
                ep_fi += (
                    (loss_pair_phase.item() + loss_pair_task.item()
                     + loss_pair_uncertainty.item() + loss_cr_monotonic.item())
                    if pair_freq_task_enabled else loss_fi.item()
                ) * bx1.size(0)
                ep_snr += avg_snr_batch * bx1.size(0)
                n_batches += 1
            else:
                # 单信号数据：(noisy, clean)。论文复现分支使用标准 MSE；
                # 频域任务感知分支使用 time NMSE + Fisher-weighted spectral NMSE。
                bx, by = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                y = model(bx)
                if freq_task_recon_enabled:
                    loss, time_nmse, spec_nmse = frequency_task_reconstruction_loss(
                        y, by, spectral_blend=spectral_blend,
                        fisher_power=spectral_power
                    )
                else:
                    loss = criterion(y, by)
                    time_nmse = loss
                    spec_nmse = loss
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * bx.size(0)
                ep_nmse_mag += time_nmse.item() * bx.size(0)
                ep_nmse_complex += spec_nmse.item() * bx.size(0)
        train_loss_hist.append(ep_loss / len(train_loader.dataset))
        train_nmse_mag_hist.append(ep_nmse_mag / len(train_loader.dataset))
        train_nmse_complex_hist.append(ep_nmse_complex / len(train_loader.dataset))
        corr_hist.append(ep_corr / len(train_loader.dataset) if n_batches > 0 else 0.0)
        peak_hist.append(ep_peak / len(train_loader.dataset) if n_batches > 0 else 0.0)
        snr_hist.append(ep_snr / len(train_loader.dataset) if n_batches > 0 else 0.0)

        scheduler.step()

        model.eval()
        val_mse = 0.0
        val_mse_mag = 0.0
        val_mse_complex = 0.0
        val_corr = 0.0
        val_peak = 0.0
        val_total = 0.0
        n_val = 0
        val_batch_index = 0
        with torch.no_grad():
            for batch in val_loader:
                if task_loss_enabled and len(batch) >= 4:
                    # 配对数据：计算复合损失（与训练目标完全一致，含peak loss）
                    bx1, by1, bx2, by2 = [b.to(device) for b in batch[:4]]
                    true_tdoa_v = batch[4].to(device) if len(batch) >= 5 else None
                    bx_all = torch.cat([bx1, bx2], dim=0)
                    val_active_cr = (
                        nested_crs[val_batch_index % len(nested_crs)]
                        if nested_pair_task_enabled else None
                    )
                    if nested_pair_task_enabled and hasattr(model, "set_active_cr"):
                        model.set_active_cr(val_active_cr)
                    if v4_task_sufficient_enabled and hasattr(model, "encode_full_latent"):
                        z_full_all = model.encode_full_latent(bx_all)
                        z_all = model._mask_latent(z_full_all, cr=val_active_cr)
                        y_all = model.decode_latent(z_all)
                        z_full1_v, z_full2_v = z_full_all.chunk(2, dim=0)
                        z1_v, z2_v = z_all.chunk(2, dim=0)
                    elif nested_pair_task_enabled:
                        y_all, z_all = model(bx_all, return_latent=True)
                        z1_v, z2_v = z_all.chunk(2, dim=0)
                        z_full1_v = z_full2_v = None
                    else:
                        y_all = model(bx_all)
                        z1_v = z2_v = None
                        z_full1_v = z_full2_v = None
                    y1, y2 = y_all.chunk(2, dim=0)

                    if pair_freq_task_enabled:
                        y_pair = torch.cat([y1, y2], dim=0)
                        clean_pair = torch.cat([by1, by2], dim=0)
                        mse_v_norm, mse_v_mag, mse_v_complex = (
                            frequency_task_reconstruction_loss(
                                y_pair, clean_pair,
                                spectral_blend=spectral_blend,
                                fisher_power=spectral_power,
                            )
                        )
                    else:
                        # 验证NMSE：与训练同公式（固定比例混合，α=ε）
                        v_sig_power = torch.mean(by1**2 + by2**2, dim=[1,2])
                        vm_y1 = torch.sqrt(y1[:,0,:]**2 + y1[:,1,:]**2 + 1e-10)
                        vm_y2 = torch.sqrt(y2[:,0,:]**2 + y2[:,1,:]**2 + 1e-10)
                        vm_s1 = torch.sqrt(by1[:,0,:]**2 + by1[:,1,:]**2 + 1e-10)
                        vm_s2 = torch.sqrt(by2[:,0,:]**2 + by2[:,1,:]**2 + 1e-10)
                        vm_mag_mse = torch.mean((vm_y1-vm_s1)**2 + (vm_y2-vm_s2)**2, dim=1)
                        v_nmse_mag = vm_mag_mse / (v_sig_power + 1e-9)
                        v_complex_mse = torch.mean((y1-by1)**2 + (y2-by2)**2, dim=[1,2])
                        v_nmse_complex = v_complex_mse / (v_sig_power + 1e-9)
                        v_env_ratio = 1.0 - phase_mix
                        mse_v_mag = torch.mean(v_nmse_mag)
                        mse_v_complex = torch.mean(v_nmse_complex)
                        mse_v_norm = torch.mean(
                            v_env_ratio * v_nmse_mag + phase_mix * v_nmse_complex
                        )

                    # GCC 分量（验证时也计算，确保早停反映真实目标）
                    gcc_clean = gcc_torch(by1[:, 0, :], by1[:, 1, :],
                                          by2[:, 0, :], by2[:, 1, :])
                    gcc_clean = crop_gcc_same(gcc_clean, by1.shape[-1])
                    gcc_clean_n = gcc_clean / (torch.max(torch.abs(gcc_clean), dim=-1, keepdim=True)[0] + 1e-9)
                    gcc_dae = gcc_torch(y1[:, 0, :], y1[:, 1, :],
                                        y2[:, 0, :], y2[:, 1, :])
                    gcc_dae = crop_gcc_same(gcc_dae, y1.shape[-1])
                    gcc_dae_n = gcc_dae / (torch.max(torch.abs(gcc_dae), dim=-1, keepdim=True)[0] + 1e-9)
                    corr_v = torch.mean((gcc_dae_n - gcc_clean_n) ** 2)

                    # Peak 分量：验证时使用与训练相同的逐样本SNR自适应权重
                    peak_v = torch.tensor(0.0, device=device)
                    if lambda_peak > 0 and true_tdoa_v is not None:
                        if pair_freq_task_enabled:
                            peak_per_sample_v = soft_gcc_peak_loss_per_sample(
                                gcc_dae_n, true_tdoa_v.float(), sigma=soft_peak_sigma,
                                sharpness_weight=soft_peak_sharpness_weight,
                            )
                            peak_v = lambda_peak * torch.mean(peak_per_sample_v)
                        elif use_adaptive_peak:
                            snr_per_sample_v = estimate_per_sample_snr(bx1, by1)
                            peak_per_sample_v = gcc_peak_loss_per_sample(
                                y1[:, 0, :], y1[:, 1, :],
                                y2[:, 0, :], y2[:, 1, :],
                                true_tdoa_v.float())
                            peak_lambda_v = adaptive_peak_lambda(
                                snr_per_sample_v, peak_max=lambda_peak,
                                snr_center=snr_threshold, temperature=lambda_temperature)
                            peak_v = torch.mean(peak_lambda_v * peak_per_sample_v)
                        else:
                            peak_v = lambda_peak * gcc_peak_loss(
                                y1[:, 0, :], y1[:, 1, :],
                                y2[:, 0, :], y2[:, 1, :],
                                true_tdoa_v.float())

                    pair_phase_v = torch.tensor(0.0, device=device)
                    if pair_freq_task_enabled and pair_phase_weight > 0:
                        pair_phase_v = pairwise_cross_spectrum_phase_loss(
                            y1, y2, by1, by2, fisher_power=spectral_power
                        )
                    pair_task_v = torch.tensor(0.0, device=device)
                    pair_uncertainty_v = torch.tensor(0.0, device=device)
                    cr_monotonic_v = torch.tensor(0.0, device=device)
                    if (v4_task_sufficient_enabled and true_tdoa_v is not None
                            and z_full1_v is not None):
                        pair_task_v, pair_uncertainty_v, cr_monotonic_v = (
                            nested_task_sufficient_losses(
                                model, z_full1_v, z_full2_v, true_tdoa_v.float(),
                                nested_crs=nested_crs, sigma=soft_peak_sigma
                            )
                        )
                    elif (nested_pair_task_enabled and pair_task_weight > 0
                            and true_tdoa_v is not None
                            and hasattr(model, "pair_task_logits_from_latent")):
                        logits_v = model.pair_task_logits_from_latent(z1_v, z2_v)
                        pair_task_v = soft_tdoa_distribution_loss(
                            logits_v, true_tdoa_v.float(), sigma=soft_peak_sigma
                        )

                    fi_v = torch.tensor(0.0, device=device)
                    if beta_fi > 0:
                        fi_v = fisher_tdoa_loss(
                            y1[:, 0, :], y1[:, 1, :],
                            y2[:, 0, :], y2[:, 1, :],
                            by1[:, 0, :], by1[:, 1, :],
                            by2[:, 0, :], by2[:, 1, :],
                        )

                    # Validation uses the final reconstruction weight for comparable selection.
                    loss_v = (
                        corr_weight * corr_v
                        + peak_v
                        + mse_weight_max * mse_v_norm
                        + pair_phase_weight * pair_phase_v
                        + pair_task_weight * pair_task_v
                        + pair_uncertainty_weight * pair_uncertainty_v
                        + cr_monotonic_weight * cr_monotonic_v
                        + beta_fi * fi_v
                    )

                    val_mse += mse_v_norm.item() * bx1.size(0)
                    val_mse_mag += mse_v_mag.item() * bx1.size(0)
                    val_mse_complex += mse_v_complex.item() * bx1.size(0)
                    val_corr += corr_v.item() * bx1.size(0)
                    val_peak += peak_v.item() * bx1.size(0)
                    val_total += loss_v.item() * bx1.size(0)
                    n_val += bx1.size(0)
                    val_batch_index += 1
                else:
                    bx, by = batch[0].to(device), batch[1].to(device)
                    y = model(bx)
                    if freq_task_recon_enabled:
                        loss_v, time_nmse_v, spec_nmse_v = frequency_task_reconstruction_loss(
                            y, by, spectral_blend=spectral_blend,
                            fisher_power=spectral_power
                        )
                        mse_v = loss_v.item()
                        mse_mag_v = time_nmse_v.item()
                        mse_complex_v = spec_nmse_v.item()
                    else:
                        loss_v = criterion(y, by)
                        mse_v = loss_v.item()
                        mse_mag_v = mse_v
                        mse_complex_v = mse_v
                    val_mse += mse_v * bx.size(0)
                    val_mse_mag += mse_mag_v * bx.size(0)
                    val_mse_complex += mse_complex_v * bx.size(0)
                    val_total += mse_v * bx.size(0)
                    n_val += bx.size(0)

        val_mse /= n_val
        val_mse_mag /= n_val
        val_mse_complex /= n_val
        val_corr /= n_val
        val_peak /= n_val
        val_loss = val_total / n_val  # 用于早停的复合损失 (corr + peak + ε·NMSE)
        val_loss_hist.append(val_mse)  # Mixed NMSE trace for plotting.
        val_nmse_mag_hist.append(val_mse_mag)
        val_nmse_complex_hist.append(val_mse_complex)
        val_corr_hist.append(val_corr)
        val_peak_hist.append(val_peak)
        val_total_hist.append(val_loss)

        can_update_best = (ep + 1) >= selection_start_epoch
        if can_update_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            epochs_no_improve = 0
        elif can_update_best:
            epochs_no_improve += 1
        else:
            epochs_no_improve = 0

        # Grace period 仅用于任务驱动分支；论文复现 MSE 不需要锁死早停。
        if task_loss_enabled and ep < warmup_ep + 30:
            epochs_no_improve = 0

        ep_time = time.time() - t_ep

        if (ep + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            early_mark = " [EARLY STOP]" if early_stopping and epochs_no_improve >= patience else ""
            corr_str = f" | Corr {ep_corr / len(train_loader.dataset):.5f}" if corr_weight > 0 else ""
            peak_str = f" | Peak {ep_peak / len(train_loader.dataset):.5f}" if lambda_peak > 0 else ""
            if pair_freq_task_enabled and pair_phase_weight > 0:
                fi_str = f" | XPhase {ep_fi / len(train_loader.dataset):.5f}"
            else:
                fi_str = f" | FI {ep_fi / len(train_loader.dataset):.5f}" if beta_fi > 0 else ""
            val_corr_str = f" | ValCorr {val_corr:.5f}" if corr_weight > 0 else ""
            val_peak_str = f" | ValPeak {val_peak:.5f}" if lambda_peak > 0 else ""
            snr_str = f" | SNR={ep_snr / len(train_loader.dataset):.1f}dB" if use_adaptive_peak and n_batches > 0 else ""
            if pair_freq_task_enabled:
                metric_name = (
                    "TaskSufficientFreqNMSE"
                    if v4_task_sufficient_enabled
                    else ("NestedPairFreqNMSE" if nested_pair_task_enabled else "PairFreqNMSE")
                )
                loss_state_str = (
                    f"{loss_mode} | recon_w={mse_weight_current:.3f} | "
                    f"corr_w={corr_weight:.3f} | xphase_w={pair_phase_weight:.3f} | "
                    f"peak_w={lambda_peak:.3f} | peak_sharp={soft_peak_sharpness_weight:.3f} | "
                    f"task_w={pair_task_weight:.3f} | "
                    f"unc_w={pair_uncertainty_weight:.3f} | "
                    f"mono_w={cr_monotonic_weight:.3f} | "
                    f"spectral_blend={spectral_blend:.3f}"
                )
            elif task_loss_enabled:
                metric_name = "NMSE"
                loss_state_str = (f"mse_w={mse_weight_current:.3f} | phase_mix={phase_mix:.3f} | "
                                  f"NetComplex={mse_weight_current * phase_mix:.4f}")
            elif freq_task_recon_enabled:
                metric_name = "FreqNMSE"
                loss_state_str = (f"freq_task_mse | spectral_blend={spectral_blend:.3f} | "
                                  f"spectral_power={spectral_power:.2f}")
            else:
                metric_name = "MSE"
                loss_state_str = "paper_mse"
            print(f"  Fold {fold_idx} Epoch {ep + 1}/{epochs}: "
                  f"Train{metric_name} {train_loss_hist[-1]:.5f} | "
                  f"Val{metric_name} {val_mse:.5f}{val_corr_str}{val_peak_str} | "
                  f"ValTotal {val_loss:.5f}{corr_str}{peak_str}{fi_str}{snr_str} | "
                  f"{loss_state_str} | "
                  f"LR: {current_lr:.6f} | {ep_time:.1f}s{early_mark}")

        if early_stopping and epochs_no_improve >= patience:
            stopped_epoch = ep + 1
            print(f"  Fold {fold_idx} Early stopping at Epoch {stopped_epoch} "
                  f"(no improvement for {patience} epochs, epoch耗时 {ep_time:.1f}s)")
            break
    else:
        stopped_epoch = epochs

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = len(val_total_hist)
        best_val_loss = val_total_hist[-1] if val_total_hist else float('inf')
    if restore_best:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(val_total_hist)
        best_val_loss = val_total_hist[-1] if val_total_hist else float('inf')
    return model, train_loss_hist, val_loss_hist, best_val_loss, stopped_epoch, \
        corr_hist, peak_hist, snr_hist, val_corr_hist, val_peak_hist, val_total_hist, \
        train_nmse_mag_hist, train_nmse_complex_hist, val_nmse_mag_hist, val_nmse_complex_hist, best_epoch


def train_with_cv(device, cr, k=5, n_samples=10000, epochs=100,
                  batch_size=64, lr=0.0005, seed=42,
                  patience=20, weight_decay=1e-4, use_split=False,
                  channel_mode="random", n_fixed_channels=50, channel_pool_seed=42,
                  sim=None, corr_weight=1.0, lambda_peak=0.0, beta_fi=0.0,
                  epsilon_mse=1.0,
                  mse_weight_max=None, phase_mix=None, selection_start_epoch=None,
                  use_adaptive_peak=False, snr_threshold=0.0, lambda_temperature=5.0,
                  loss_config=None, loss_mode="task",
                  nlos_prob=0.2, delay_label_mode="strongest",
                  snr_train_range=(-10, 10), multipath_scale=0.3,
                  scenario_mode="sv_pair", normalization_mode="none",
                  cv_group_mode="sample", urban_base_delay=16, urban_min_los=4,
                  urban_train_los_only=False, early_stopping=True,
                  restore_best=True, final_retrain=False, final_epochs=None,
                  training_protocol=None, resample_train_each_epoch=False,
                  resample_interval=1, model_factory=DAE, model_name=None,
                  spectral_blend=0.25, spectral_power=2.0,
                  pair_phase_weight=0.0, soft_peak_sigma=1.5,
                  soft_peak_sharpness_weight=0.0,
                  pair_task_weight=0.0, pair_uncertainty_weight=0.0,
                  cr_monotonic_weight=0.0, nested_crs=(4, 8, 16)):
    """
    训练 DAE 模型，支持 K-fold CV 或单次 train/val 划分。

    参数:
        corr_weight:          GCC 互相关损失显式权重
        lambda_peak:          PNCC 峰值损失最大权重
        beta_fi:              Fisher 信息损失权重（0=不启用）
        use_adaptive_peak:    是否使用SNR自适应Peak权重
        snr_threshold:        Peak权重SNR中心点（dB）
        lambda_temperature:   Peak权重sigmoid温度参数
        loss_config:           Per-CR损失配置字典 {epsilon_mse, lambda_peak}
                               覆盖同名的单个参数。
                               统一公式: L = corr_weight·L_corr + λp·L_peak + ε·NMSE。
                               R20中ε为温和重建/相位引导强度。
        loss_mode:             "task" 使用R20.1任务驱动损失；"paper_mse" 使用论文式MSE复现基线。
                               "freq_task_mse" 使用频域任务感知重构损失。
        cv_group_mode:         "sample" 随机样本划分；"snapshot" 按 urban snapshot 分组划分。
        resample_train_each_epoch:
                               urban8 MSE 复现轨道下，训练集按固定 snapshot/UAV
                               计划周期性重采样 waveform，验证集保持固定。
    """
    # === loss_config 覆盖 ===
    # 每个CR的loss配置覆盖同名参数。统一公式:
    #   L = corr_weight·L_corr + λp·L_peak + ε·[(1-ε)·NMSE_mag + ε·NMSE_complex]
    # R20中ε不再取大容量占比，而是作为温和重建/相位引导强度:
    #   CR=4: ε=0.15, λp=0.15
    #   CR=8: ε=0.08, λp=0.20
    #   CR=16: ε=0.03, λp=0.25
    # Corr=1.0恒定的原因：瓶颈独立性（128维足够GCC），TDOA是所有CR的最终目标。
    if mse_weight_max is None:
        mse_weight_max = epsilon_mse
    if phase_mix is None:
        phase_mix = epsilon_mse
    if loss_config is not None:
        legacy_eps = loss_config.get('epsilon_mse', epsilon_mse)
        mse_weight_max = loss_config.get('mse_weight_max', legacy_eps)
        phase_mix = loss_config.get('phase_mix', legacy_eps)
        selection_start_epoch = loss_config.get('selection_start_epoch', selection_start_epoch)
        loss_mode = loss_config.get('loss_mode', loss_mode)
        epsilon_mse = legacy_eps
        corr_weight = loss_config.get('corr_weight', corr_weight)
        lambda_peak = loss_config.get('lambda_peak', lambda_peak)
        beta_fi = loss_config.get('beta_fi', beta_fi)
        spectral_blend = loss_config.get('spectral_blend', spectral_blend)
        spectral_power = loss_config.get('spectral_power', spectral_power)
        pair_phase_weight = loss_config.get('pair_phase_weight', pair_phase_weight)
        pair_task_weight = loss_config.get('pair_task_weight', pair_task_weight)
        pair_uncertainty_weight = loss_config.get(
            'pair_uncertainty_weight', pair_uncertainty_weight
        )
        cr_monotonic_weight = loss_config.get(
            'cr_monotonic_weight', cr_monotonic_weight
        )
        nested_crs = tuple(loss_config.get('nested_crs', nested_crs))
        soft_peak_sigma = loss_config.get('soft_peak_sigma', soft_peak_sigma)
        soft_peak_sharpness_weight = loss_config.get(
            'soft_peak_sharpness_weight', soft_peak_sharpness_weight
        )

    if loss_mode == "paper_mse":
        corr_weight = 0.0
        lambda_peak = 0.0
        beta_fi = 0.0
        use_adaptive_peak = False
        mse_weight_max = 1.0
        phase_mix = 1.0
        if selection_start_epoch is None:
            selection_start_epoch = 1
    elif loss_mode == "freq_task_mse":
        corr_weight = 0.0
        lambda_peak = 0.0
        beta_fi = 0.0
        use_adaptive_peak = False
        if selection_start_epoch is None:
            selection_start_epoch = 1
    elif loss_mode in ("freq_task_pair", "freq_task_nested_pair", "freq_task_v4_min_pair"):
        use_adaptive_peak = False
        if selection_start_epoch is None:
            selection_start_epoch = 1

    # 固定全局随机种子，确保数据集生成、模型初始化和数据划分完全可复现
    torch.manual_seed(seed)
    np.random.seed(seed)

    if sim is None:
        sim = SignalSimulator(channel_mode=channel_mode,
                              n_fixed_channels=n_fixed_channels,
                              channel_pool_seed=channel_pool_seed,
                              nlos_prob=nlos_prob,
                              delay_label_mode=delay_label_mode,
                              snr_train_range=snr_train_range,
                              multipath_scale=multipath_scale,
                              scenario_mode=scenario_mode,
                              normalization_mode=normalization_mode,
                              urban_base_delay=urban_base_delay,
                              urban_min_los=urban_min_los,
                              urban_train_los_only=urban_train_los_only)

    pair_freq_task_train = loss_mode in (
        "freq_task_pair", "freq_task_nested_pair", "freq_task_v4_min_pair"
    )
    dynamic_urban_train = (
        bool(resample_train_each_epoch)
        and loss_mode in ("paper_mse", "freq_task_mse")
        and scenario_mode == "urban8"
        and corr_weight == 0.0
        and lambda_peak == 0.0
    )
    dynamic_urban_pair_train = (
        bool(resample_train_each_epoch)
        and pair_freq_task_train
        and scenario_mode == "urban8"
    )

    urban_snapshot_plan = None
    urban_uav_plan = None
    urban_pair_snapshot_plan = None
    urban_pair_uav_i_plan = None
    urban_pair_uav_j_plan = None

    if pair_freq_task_train and scenario_mode == "urban8":
        urban_pair_snapshot_plan, urban_pair_uav_i_plan, urban_pair_uav_j_plan = (
            sim.build_urban_pair_training_plan(n_samples, seed=seed)
        )
        X1_n, X1_c, X2_n, X2_c, tdoa, cv_groups = (
            sim.generate_urban_pair_training_dataset_from_plan(
                urban_pair_snapshot_plan,
                urban_pair_uav_i_plan,
                urban_pair_uav_j_plan,
                seed=seed,
                return_groups=True,
            )
        )
        dataset = TensorDataset(X1_n, X1_c, X2_n, X2_c, tdoa)
    elif corr_weight > 0 or lambda_peak > 0:
        # 配对数据：X1 和 X2 来自同信道同噪声，用于 GCC 损失和峰值损失
        X1_n, X1_c, X2_n, X2_c, tdoa = sim.generate_paired_training_dataset(n_samples, seed=seed)
        dataset = TensorDataset(X1_n, X1_c, X2_n, X2_c, tdoa)
    else:
        need_groups = cv_group_mode == "snapshot"
        if dynamic_urban_train:
            urban_snapshot_plan, urban_uav_plan = sim.build_urban_training_plan(n_samples, seed=seed)
            X_noisy, X_clean, cv_groups = sim.generate_urban_training_dataset_from_plan(
                urban_snapshot_plan, urban_uav_plan, seed=seed, return_groups=True
            )
        elif need_groups:
            X_noisy, X_clean, cv_groups = sim.generate_training_dataset(
                n_samples, seed=seed, return_groups=True
            )
        else:
            X_noisy, X_clean = sim.generate_training_dataset(n_samples, seed=seed)
            cv_groups = None
        dataset = TensorDataset(X_noisy, X_clean)

    if training_protocol == "fixed_epoch_final_only":
        suffix = "Fast Final-Only Training"
        n_folds_actual = 0
    elif use_split:
        # 单次 80/20 train/val 划分 (快速模式)
        n_train = int(n_samples * 0.8)
        indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:].tolist()

        suffix = f"Quick Split ({n_train}/{n_samples - n_train})"
        n_folds_actual = 1
    else:
        suffix = f"{k}-Fold Cross-Validation"
        n_folds_actual = k

    print(f"\n{'='*60}")
    resolved_model_name = model_name or getattr(model_factory, "__name__", str(model_factory))
    print(f"Training {resolved_model_name} (CR={cr}) with {suffix}")
    print(f"  Samples: {n_samples} | Max Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    early_stop_text = f"patience={patience}" if early_stopping else "disabled"
    print(f"  Early Stopping: {early_stop_text} | Restore Best: {restore_best} | "
          f"Weight Decay: {weight_decay}")
    print(f"  Model: {resolved_model_name}")
    print(f"  Loss Mode: {loss_mode}")
    if loss_mode == "paper_mse":
        print("  Paper Repro Loss: real/imag MSE only (Corr/Peak/FI disabled)")
    elif loss_mode == "freq_task_mse":
        print("  Frequency-task Loss: time-domain NMSE + Fisher-weighted spectral NMSE "
              f"(blend={spectral_blend}, spectral_power={spectral_power})")
    elif loss_mode in ("freq_task_pair", "freq_task_nested_pair", "freq_task_v4_min_pair"):
        if loss_mode == "freq_task_v4_min_pair":
            print("  Nested task-sufficient Pair Loss v4-min: shared nested latent + "
                  "multi-prefix task-head TDOA + uncertainty calibration + "
                  "CR monotonicity + weak waveform reconstruction")
            print(f"  Nested CR masks: {tuple(nested_crs)}")
        elif loss_mode == "freq_task_nested_pair":
            print("  Nested Frequency-task Pair Loss: shared nested latent + reconstruction + "
                  "GCC shape + soft peak + cross-spectrum phase + task-head TDOA")
            print(f"  Nested CR masks: {tuple(nested_crs)}")
        else:
            print("  Frequency-task Pair Loss: reconstruction + GCC shape + soft peak + "
                  "cross-spectrum phase consistency")
        print(f"  Pair Loss Weights: recon={mse_weight_max}, corr={corr_weight}, "
              f"xphase={pair_phase_weight}, peak={lambda_peak}, "
              f"soft_peak_sigma={soft_peak_sigma}, "
              f"soft_peak_sharpness={soft_peak_sharpness_weight}, "
              f"task_head={pair_task_weight}, "
              f"uncertainty={pair_uncertainty_weight}, "
              f"cr_mono={cr_monotonic_weight}")
    else:
        print(f"  Unified Loss (R20.1 fixed blend): L = {corr_weight}*Corr + "
              f"lambda_peak(SNR)*Peak + mse_weight_current*"
              f"[(1-{phase_mix})*NMSE_mag + {phase_mix}*NMSE_complex]")
        print(f"  mse_weight_max={mse_weight_max} | phase_mix={phase_mix} | "
              f"selection_start_epoch={selection_start_epoch if selection_start_epoch is not None else 100}")
    if lambda_peak > 0:
        if loss_mode in ("freq_task_pair", "freq_task_nested_pair", "freq_task_v4_min_pair"):
            print(f"  Soft Peak Weight: peak_weight={lambda_peak}, "
                  f"sigma={soft_peak_sigma}, "
                  f"sharpness={soft_peak_sharpness_weight}")
        else:
            print(f"  Adaptive Peak Weight: lambda_peak(SNR) = "
                  f"{lambda_peak}*sigmoid(({snr_threshold}-SNR)/{lambda_temperature}), "
                  f"range ~[0, {lambda_peak}]")
    print(f"  Training SNR: [{snr_train_range[0]}, {snr_train_range[1]}] dB uniform")
    print(f"  Scenario: {sim.scenario_mode} | normalization={sim.normalization_mode} | CV group={cv_group_mode}")
    if sim.scenario_mode == "urban8":
        print(f"  Urban: base_delay={sim.urban_base_delay:g} samples | "
              f"min_los={sim.urban_min_los} | train_los_only={sim.urban_train_los_only}")
        sample_unit = "LOS UAV pair" if pair_freq_task_train else "single-UAV waveform"
        print(f"  Urban sample unit: {sample_unit} | planned samples={n_samples}")
        if dynamic_urban_train:
            n_groups = len(np.unique(urban_snapshot_plan))
            print(f"  Train waveform resampling: enabled every {max(1, int(resample_interval))} epoch(s) "
                  f"on fixed snapshot/UAV plan ({n_groups} snapshots)")
        if dynamic_urban_pair_train:
            n_groups = len(np.unique(urban_pair_snapshot_plan))
            print(f"  Pair train resampling: enabled every {max(1, int(resample_interval))} epoch(s) "
                  f"on fixed snapshot/UAV-pair plan ({n_groups} snapshots)")
    if beta_fi > 0:
        print(f"  Fisher Loss: β_fi={beta_fi}")
    model_params = sum(p.numel() for p in model_factory(cr=cr).parameters())
    print(f"  Model Parameters: {model_params:,}")
    print(f"{'='*60}")

    t_train_start = time.time()

    cv_results = {
        'cr': cr,
        'fold_train_loss': [],
        'fold_val_loss': [],
        'fold_best_val': [],
        'fold_stopped_epoch': [],
        'fold_corr_loss': [],
        'fold_peak_loss': [],
        'fold_avg_snr': [],
        'fold_val_corr_loss': [],
        'fold_val_peak_loss': [],
        'fold_val_total_loss': [],
        'fold_train_nmse_mag': [],
        'fold_train_nmse_complex': [],
        'fold_val_nmse_mag': [],
        'fold_val_nmse_complex': [],
        'fold_best_epoch': [],
        'mse_weight_max': mse_weight_max,
        'phase_mix': phase_mix,
        'selection_start_epoch': selection_start_epoch if selection_start_epoch is not None else 100,
        'loss_mode': loss_mode,
        'model_name': resolved_model_name,
        'spectral_blend': float(spectral_blend),
        'spectral_power': float(spectral_power),
        'pair_phase_weight': float(pair_phase_weight),
        'pair_uncertainty_weight': float(pair_uncertainty_weight),
        'cr_monotonic_weight': float(cr_monotonic_weight),
        'soft_peak_sigma': float(soft_peak_sigma),
        'soft_peak_sharpness_weight': float(soft_peak_sharpness_weight),
        'scenario_mode': sim.scenario_mode,
        'normalization_mode': sim.normalization_mode,
        'cv_group_mode': cv_group_mode,
        'early_stopping': early_stopping,
        'restore_best': restore_best,
        'final_retrain': final_retrain,
        'final_epochs': final_epochs if final_epochs is not None else epochs,
        'final_epoch_policy': 'configured',
        'final_monitor_split': (
            'full_training_dataset_eval_mode' if final_retrain else None
        ),
        'resample_train_each_epoch': bool(resample_train_each_epoch),
        'resample_interval': max(1, int(resample_interval)),
        'training_protocol': training_protocol or (
            'cv_fixed_best_fold' if not final_retrain else 'cv_then_final_retrain'
        ),
        'diagnostics_version': (
            ('paper_repro_v3_1_epoch_final_train'
             if training_protocol == "cv_best_epoch_final_train"
             else ('paper_repro_v3_1_fast_final'
                   if training_protocol == "fixed_epoch_final_only"
                   else 'paper_repro_v3_urban8_fair_eval'))
            if loss_mode == "paper_mse" and sim.scenario_mode == "urban8"
            else ('freq_task_dae_v4_min_pair' if loss_mode == "freq_task_v4_min_pair"
                  else ('freq_task_dae_v3_nested_pair' if loss_mode == "freq_task_nested_pair"
                  else ('freq_task_dae_v2_pair' if loss_mode == "freq_task_pair"
                  else ('freq_task_dae_v1' if loss_mode == "freq_task_mse"
                  else ('paper_repro_v1' if loss_mode == "paper_mse" else 'R20.1'))
                  )))
        ),
    }

    if training_protocol == "fixed_epoch_final_only":
        final_epochs_resolved = int(final_epochs if final_epochs is not None else epochs)
        final_epochs_resolved = max(1, final_epochs_resolved)
        cv_results['final_epoch_policy'] = 'configured_fast_final'
        cv_results['final_epochs'] = final_epochs_resolved

        print(f"\n  Fast final-only training on all {n_samples} samples for "
              f"{final_epochs_resolved} fixed epochs...")
        torch.manual_seed(seed + 10000 + int(cr))
        np.random.seed(seed + 10000 + int(cr))
        final_model = model_factory(cr=cr).to(device)
        if dynamic_urban_pair_train:
            final_train_dataset = RefreshableUrbanPairDataset(
                sim,
                urban_pair_snapshot_plan,
                urban_pair_uav_i_plan,
                urban_pair_uav_j_plan,
                base_seed=seed + 200000 + int(cr) * 1000,
                refresh_interval=resample_interval,
            )
        elif dynamic_urban_train:
            final_train_dataset = RefreshableUrbanWaveformDataset(
                sim, urban_snapshot_plan, urban_uav_plan,
                base_seed=seed + 200000 + int(cr) * 1000,
                refresh_interval=resample_interval,
            )
        else:
            final_train_dataset = dataset
        full_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
        full_val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        final_model, final_train_loss, final_val_loss, final_best_val, final_stopped, \
            final_corr_loss, final_peak_loss, final_avg_snr, final_val_corr_loss, \
            final_val_peak_loss, final_val_total_loss, final_train_nmse_mag, \
            final_train_nmse_complex, final_val_nmse_mag, final_val_nmse_complex, \
            final_best_epoch = train_one_fold(
                device, final_model, full_train_loader, full_val_loader, final_epochs_resolved, lr,
                "Final", patience=final_epochs_resolved + 1, weight_decay=weight_decay,
                corr_weight=corr_weight, lambda_peak=lambda_peak, beta_fi=beta_fi,
                epsilon_mse=epsilon_mse, mse_weight_max=mse_weight_max,
                phase_mix=phase_mix, selection_start_epoch=1,
                use_adaptive_peak=use_adaptive_peak,
                snr_threshold=snr_threshold, lambda_temperature=lambda_temperature,
                loss_mode=loss_mode, early_stopping=False, restore_best=False,
                spectral_blend=spectral_blend, spectral_power=spectral_power,
                pair_phase_weight=pair_phase_weight,
                soft_peak_sigma=soft_peak_sigma,
                soft_peak_sharpness_weight=soft_peak_sharpness_weight,
                pair_task_weight=pair_task_weight,
                pair_uncertainty_weight=pair_uncertainty_weight,
                cr_monotonic_weight=cr_monotonic_weight,
                nested_crs=nested_crs
            )
        cv_results.update({
            'fold_train_loss': [final_train_loss],
            'fold_val_loss': [final_val_loss],
            'fold_best_val': [final_best_val],
            'fold_stopped_epoch': [final_stopped],
            'fold_corr_loss': [final_corr_loss],
            'fold_peak_loss': [final_peak_loss],
            'fold_avg_snr': [final_avg_snr],
            'fold_val_corr_loss': [final_val_corr_loss],
            'fold_val_peak_loss': [final_val_peak_loss],
            'fold_val_total_loss': [final_val_total_loss],
            'fold_train_nmse_mag': [final_train_nmse_mag],
            'fold_train_nmse_complex': [final_train_nmse_complex],
            'fold_val_nmse_mag': [final_val_nmse_mag],
            'fold_val_nmse_complex': [final_val_nmse_complex],
            'fold_best_epoch': [final_best_epoch],
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_val_total_loss': final_val_total_loss,
            'final_best_val': final_best_val,
            'final_stopped_epoch': final_stopped,
            'final_best_epoch': final_best_epoch,
            'final_corr_loss': final_corr_loss,
            'final_peak_loss': final_peak_loss,
            'final_val_corr_loss': final_val_corr_loss,
            'final_val_peak_loss': final_val_peak_loss,
            'final_train_nmse_mag': final_train_nmse_mag,
            'final_train_nmse_complex': final_train_nmse_complex,
            'final_val_nmse_mag': final_val_nmse_mag,
            'final_val_nmse_complex': final_val_nmse_complex,
        })
        t_train_total = time.time() - t_train_start
        print(f"  Fast final-only complete: ValTotal={final_best_val:.6f} "
              f"(epoch {final_best_epoch}), time {t_train_total:.1f}s "
              f"({t_train_total/60:.1f}min)")
        return final_model, cv_results

    best_model = None
    best_overall_val = float('inf')

    if use_split:
        split_iterator = [(train_idx, val_idx)]
    else:
        if cv_group_mode == "snapshot" and cv_groups is not None:
            groups_np = cv_groups.cpu().numpy() if hasattr(cv_groups, "cpu") else np.asarray(cv_groups)
            n_groups = len(np.unique(groups_np))
            if n_groups < k:
                raise ValueError(f"snapshot GroupKFold needs at least {k} groups, got {n_groups}")
            kf = GroupKFold(n_splits=k)
            split_iterator = list(kf.split(range(n_samples), groups=groups_np))
        else:
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)
            split_iterator = list(kf.split(range(n_samples)))

    for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
        if dynamic_urban_pair_train:
            idx_arr = np.asarray(train_idx, dtype=int)
            train_dataset = RefreshableUrbanPairDataset(
                sim,
                urban_pair_snapshot_plan[idx_arr],
                urban_pair_uav_i_plan[idx_arr],
                urban_pair_uav_j_plan[idx_arr],
                base_seed=seed + 100000 + fold_idx * 1000,
                refresh_interval=resample_interval,
            )
        elif dynamic_urban_train:
            train_dataset = RefreshableUrbanWaveformDataset(
                sim,
                urban_snapshot_plan[np.asarray(train_idx, dtype=int)],
                urban_uav_plan[np.asarray(train_idx, dtype=int)],
                base_seed=seed + 100000 + fold_idx * 1000,
                refresh_interval=resample_interval,
            )
        else:
            train_dataset = Subset(dataset, train_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx),
                                batch_size=batch_size, shuffle=False)

        model = model_factory(cr=cr).to(device)
        t_fold = time.time()

        model, train_loss, val_loss, best_val, stopped, corr_loss, peak_loss, avg_snr, \
            val_corr_loss, val_peak_loss, val_total_loss, \
            train_nmse_mag, train_nmse_complex, val_nmse_mag, val_nmse_complex, best_epoch = \
            train_one_fold(
                device, model, train_loader, val_loader, epochs, lr,
                fold_idx + 1, patience=patience, weight_decay=weight_decay,
                corr_weight=corr_weight, lambda_peak=lambda_peak, beta_fi=beta_fi,
                epsilon_mse=epsilon_mse, mse_weight_max=mse_weight_max,
                phase_mix=phase_mix, selection_start_epoch=selection_start_epoch,
                use_adaptive_peak=use_adaptive_peak,
                snr_threshold=snr_threshold, lambda_temperature=lambda_temperature,
                loss_mode=loss_mode, early_stopping=early_stopping,
                restore_best=restore_best,
                spectral_blend=spectral_blend, spectral_power=spectral_power,
                pair_phase_weight=pair_phase_weight,
                soft_peak_sigma=soft_peak_sigma,
                soft_peak_sharpness_weight=soft_peak_sharpness_weight,
                pair_task_weight=pair_task_weight,
                pair_uncertainty_weight=pair_uncertainty_weight,
                cr_monotonic_weight=cr_monotonic_weight,
                nested_crs=nested_crs
            )
        fold_time = time.time() - t_fold

        cv_results['fold_train_loss'].append(train_loss)
        cv_results['fold_val_loss'].append(val_loss)
        cv_results['fold_best_val'].append(best_val)
        cv_results['fold_stopped_epoch'].append(stopped)
        cv_results['fold_corr_loss'].append(corr_loss)
        cv_results['fold_peak_loss'].append(peak_loss)
        cv_results['fold_avg_snr'].append(avg_snr)
        cv_results['fold_val_corr_loss'].append(val_corr_loss)
        cv_results['fold_val_peak_loss'].append(val_peak_loss)
        cv_results['fold_val_total_loss'].append(val_total_loss)
        cv_results['fold_train_nmse_mag'].append(train_nmse_mag)
        cv_results['fold_train_nmse_complex'].append(train_nmse_complex)
        cv_results['fold_val_nmse_mag'].append(val_nmse_mag)
        cv_results['fold_val_nmse_complex'].append(val_nmse_complex)
        cv_results['fold_best_epoch'].append(best_epoch)

        print(f"  Fold {fold_idx + 1} Best Val Loss: {best_val:.6f} "
              f"(best epoch {best_epoch}, stopped at epoch {stopped}, time {fold_time:.1f}s)")

        if best_val < best_overall_val:
            best_overall_val = best_val
            best_model = model

    t_train_total = time.time() - t_train_start
    avg_val = sum(cv_results['fold_best_val']) / n_folds_actual
    avg_stop = sum(cv_results['fold_stopped_epoch']) / n_folds_actual
    print(f"\n  Average Best Val Loss over {n_folds_actual} fold(s): {avg_val:.6f}")
    print(f"  Average Stop Epoch: {avg_stop:.1f}")
    print(f"  Selected model with Val Loss: {best_overall_val:.6f}")
    print(f"  Training Time: {t_train_total:.1f}s ({t_train_total/60:.1f}min)")

    if final_retrain:
        if final_epochs is None:
            if training_protocol == "cv_best_epoch_final_train":
                final_epochs_resolved = int(round(float(np.median(cv_results['fold_best_epoch']))))
                cv_results['final_epoch_policy'] = 'median_cv_best_epoch'
            else:
                final_epochs_resolved = int(epochs)
                cv_results['final_epoch_policy'] = 'max_epochs_default'
        else:
            final_epochs_resolved = int(final_epochs)
            cv_results['final_epoch_policy'] = 'configured'
        final_epochs_resolved = max(1, final_epochs_resolved)
        cv_results['final_epochs'] = final_epochs_resolved

        print(f"\n  Final retrain on all {n_samples} samples for {final_epochs_resolved} fixed epochs "
              f"(policy={cv_results['final_epoch_policy']})...")
        torch.manual_seed(seed + 10000 + int(cr))
        np.random.seed(seed + 10000 + int(cr))
        final_model = model_factory(cr=cr).to(device)
        if dynamic_urban_pair_train:
            final_train_dataset = RefreshableUrbanPairDataset(
                sim,
                urban_pair_snapshot_plan,
                urban_pair_uav_i_plan,
                urban_pair_uav_j_plan,
                base_seed=seed + 200000 + int(cr) * 1000,
                refresh_interval=resample_interval,
            )
        elif dynamic_urban_train:
            final_train_dataset = RefreshableUrbanWaveformDataset(
                sim, urban_snapshot_plan, urban_uav_plan,
                base_seed=seed + 200000 + int(cr) * 1000,
                refresh_interval=resample_interval,
            )
        else:
            final_train_dataset = dataset
        full_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
        full_val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        final_model, final_train_loss, final_val_loss, final_best_val, final_stopped, \
            final_corr_loss, final_peak_loss, final_avg_snr, final_val_corr_loss, \
            final_val_peak_loss, final_val_total_loss, final_train_nmse_mag, \
            final_train_nmse_complex, final_val_nmse_mag, final_val_nmse_complex, \
            final_best_epoch = train_one_fold(
                device, final_model, full_train_loader, full_val_loader, final_epochs_resolved, lr,
                "Final", patience=final_epochs_resolved + 1, weight_decay=weight_decay,
                corr_weight=corr_weight, lambda_peak=lambda_peak, beta_fi=beta_fi,
                epsilon_mse=epsilon_mse, mse_weight_max=mse_weight_max,
                phase_mix=phase_mix, selection_start_epoch=1,
                use_adaptive_peak=use_adaptive_peak,
                snr_threshold=snr_threshold, lambda_temperature=lambda_temperature,
                loss_mode=loss_mode, early_stopping=False, restore_best=False,
                spectral_blend=spectral_blend, spectral_power=spectral_power,
                pair_phase_weight=pair_phase_weight,
                soft_peak_sigma=soft_peak_sigma,
                soft_peak_sharpness_weight=soft_peak_sharpness_weight,
                pair_task_weight=pair_task_weight,
                pair_uncertainty_weight=pair_uncertainty_weight,
                cr_monotonic_weight=cr_monotonic_weight,
                nested_crs=nested_crs
            )
        cv_results.update({
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_val_total_loss': final_val_total_loss,
            'final_best_val': final_best_val,
            'final_stopped_epoch': final_stopped,
            'final_best_epoch': final_best_epoch,
            'final_corr_loss': final_corr_loss,
            'final_peak_loss': final_peak_loss,
            'final_val_corr_loss': final_val_corr_loss,
            'final_val_peak_loss': final_val_peak_loss,
            'final_train_nmse_mag': final_train_nmse_mag,
            'final_train_nmse_complex': final_train_nmse_complex,
            'final_val_nmse_mag': final_val_nmse_mag,
            'final_val_nmse_complex': final_val_nmse_complex,
        })
        best_model = final_model
        print(f"  Final retrain complete: final ValTotal={final_best_val:.6f} "
              f"(epoch {final_best_epoch})")

    return best_model, cv_results


def train_model(device, epochs, cr, batch_size=64, steps_per_epoch=100,
                lr=0.0005, weight_decay=1e-4,
                channel_mode="random", n_fixed_channels=50, channel_pool_seed=42):
    """
    流式训练引擎（保留向后兼容）

    若需要在 CV 模式下训练，请使用 train_with_cv()。
    """
    print(f"--- Training DAE (CR={cr}) on {device} with Infinite Dynamic Stream ---")

    sim = SignalSimulator(channel_mode=channel_mode,
                          n_fixed_channels=n_fixed_channels,
                          channel_pool_seed=channel_pool_seed)
    model = DAE(cr=cr).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()
    loss_hist = []

    for ep in range(epochs):
        model.train()
        ep_loss = 0

        for step in range(steps_per_epoch):
            X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2 = \
                sim.generate_pair_batch(batch_size, snr_db=None)

            bx = X1_noisy.to(device)
            by = X1_clean.to(device)

            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()

        avg_loss = ep_loss / steps_per_epoch
        loss_hist.append(avg_loss)

        if (ep + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {ep + 1}/{epochs}: Loss {avg_loss:.5f} | LR: {current_lr:.6f}")

    return model, loss_hist
