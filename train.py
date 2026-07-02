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
    true_idx = zero_idx + true_tdoa.long()
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
    true_idx = zero_idx + true_tdoa.long()
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

    y1 = torch.complex(y1_r, y1_i)
    c1 = torch.complex(c1_r, c1_i)

    S_dae = torch.fft.fft(y1, dim=-1)
    S_clean = torch.fft.fft(c1, dim=-1)

    freqs = torch.fft.fftfreq(n, d=1.0 / fs).to(y1_r.device)
    w = (2.0 * torch.pi * freqs) ** 2  # (n,)

    psd_dae = torch.abs(S_dae) ** 2
    psd_clean = torch.abs(S_clean) ** 2

    num_dae = torch.sum(w[None, :] * psd_dae, dim=-1)
    den_dae = torch.sum(psd_dae, dim=-1) + 1e-9
    rms_dae = torch.sqrt(num_dae / den_dae)

    num_clean = torch.sum(w[None, :] * psd_clean, dim=-1)
    den_clean = torch.sum(psd_clean, dim=-1) + 1e-9
    rms_clean = torch.sqrt(num_clean / den_clean)

    rel_err = torch.abs(rms_dae - rms_clean) / (rms_clean + 1e-9)
    return torch.mean(rel_err)


def train_one_fold(device, model, train_loader, val_loader, epochs, lr,
                   fold_idx, patience=20, weight_decay=1e-4, corr_weight=1.0,
                   lambda_peak=0.0, beta_fi=0.0, epsilon_mse=1.0,
                   mse_weight_max=None, phase_mix=None, selection_start_epoch=None,
                   use_adaptive_peak=False, snr_threshold=0.0, lambda_temperature=5.0,
                   loss_mode="task", early_stopping=True, restore_best=True):
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

    task_loss_enabled = (loss_mode != "paper_mse") and (corr_weight > 0 or lambda_peak > 0 or beta_fi > 0)

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
    warmup_ep = 100 if task_loss_enabled else 0
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
                y_all = model(bx_all)
                y1, y2 = y_all.chunk(2, dim=0)

                # === 固定比例混合 NMSE（α=ε，无ramp） ===
                # 包络NMSE：约束幅度轮廓，不与Corr冲突，但遇相位地板(~0.38)
                # 复NMSE：微量加入打破相位地板，比例=ε，ε小则几乎不进
                # CR4(ε=0.15): 15%复+85%包 → 温和相位引导中CR区分最大
                # CR16(ε=0.03): 3%复+97%包 → 近乎纯包络，无内战
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
                    if snr_per_sample is not None:
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

                # R20.1: decouple total reconstruction weight and mag/complex mixture.
                loss_mse_norm = torch.mean(nmse)
                loss_mse_mag = torch.mean(nmse_mag)
                loss_mse_complex = torch.mean(nmse_complex)
                loss_corr = torch.mean(corr_per_sample)
                loss = corr_weight * loss_corr + loss_peak + mse_weight_current * loss_mse_norm + beta_fi * loss_fi

                loss.backward()
                optimizer.step()
                ep_loss += loss_mse_norm.item() * bx1.size(0)
                ep_nmse_mag += loss_mse_mag.item() * bx1.size(0)
                ep_nmse_complex += loss_mse_complex.item() * bx1.size(0)
                ep_corr += loss_corr.item() * bx1.size(0)
                ep_peak += loss_peak.item() * bx1.size(0)
                ep_fi += loss_fi.item() * bx1.size(0)
                ep_snr += avg_snr_batch * bx1.size(0)
                n_batches += 1
            else:
                # 单信号数据：(noisy, clean) — 论文复现基线：实部/虚部标准 MSE
                bx, by = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * bx.size(0)
                ep_nmse_mag += loss.item() * bx.size(0)
                ep_nmse_complex += loss.item() * bx.size(0)
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
        with torch.no_grad():
            for batch in val_loader:
                if task_loss_enabled and len(batch) >= 4:
                    # 配对数据：计算复合损失（与训练目标完全一致，含peak loss）
                    bx1, by1, bx2, by2 = [b.to(device) for b in batch[:4]]
                    true_tdoa_v = batch[4].to(device) if len(batch) >= 5 else None
                    bx_all = torch.cat([bx1, bx2], dim=0)
                    y_all = model(bx_all)
                    y1, y2 = y_all.chunk(2, dim=0)

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
                    mse_v_norm = torch.mean(v_env_ratio * v_nmse_mag + phase_mix * v_nmse_complex)

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
                        if use_adaptive_peak:
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

                    # Validation uses the final reconstruction weight for comparable selection.
                    loss_v = corr_weight * corr_v + peak_v + mse_weight_max * mse_v_norm

                    val_mse += mse_v_norm.item() * bx1.size(0)
                    val_mse_mag += mse_v_mag.item() * bx1.size(0)
                    val_mse_complex += mse_v_complex.item() * bx1.size(0)
                    val_corr += corr_v.item() * bx1.size(0)
                    val_peak += peak_v.item() * bx1.size(0)
                    val_total += loss_v.item() * bx1.size(0)
                    n_val += bx1.size(0)
                else:
                    bx, by = batch[0].to(device), batch[1].to(device)
                    mse_v = criterion(model(bx), by).item()
                    val_mse += mse_v * bx.size(0)
                    val_mse_mag += mse_v * bx.size(0)
                    val_mse_complex += mse_v * bx.size(0)
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
            fi_str = f" | FI {ep_fi / len(train_loader.dataset):.5f}" if beta_fi > 0 else ""
            val_corr_str = f" | ValCorr {val_corr:.5f}" if corr_weight > 0 else ""
            val_peak_str = f" | ValPeak {val_peak:.5f}" if lambda_peak > 0 else ""
            snr_str = f" | SNR={ep_snr / len(train_loader.dataset):.1f}dB" if use_adaptive_peak and n_batches > 0 else ""
            metric_name = "NMSE" if task_loss_enabled else "MSE"
            loss_state_str = (f"mse_w={mse_weight_current:.3f} | phase_mix={phase_mix:.3f} | "
                              f"NetComplex={mse_weight_current * phase_mix:.4f}"
                              if task_loss_enabled else "paper_mse")
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
                  resample_interval=1):
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
        lambda_peak = loss_config.get('lambda_peak', lambda_peak)

    if loss_mode == "paper_mse":
        corr_weight = 0.0
        lambda_peak = 0.0
        beta_fi = 0.0
        use_adaptive_peak = False
        mse_weight_max = 1.0
        phase_mix = 1.0
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

    dynamic_urban_train = (
        bool(resample_train_each_epoch)
        and loss_mode == "paper_mse"
        and scenario_mode == "urban8"
        and corr_weight == 0.0
        and lambda_peak == 0.0
    )

    urban_snapshot_plan = None
    urban_uav_plan = None

    if corr_weight > 0 or lambda_peak > 0:
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
    print(f"Training DAE (CR={cr}) with {suffix}")
    print(f"  Samples: {n_samples} | Max Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    early_stop_text = f"patience={patience}" if early_stopping else "disabled"
    print(f"  Early Stopping: {early_stop_text} | Restore Best: {restore_best} | "
          f"Weight Decay: {weight_decay}")
    print(f"  Loss Mode: {loss_mode}")
    if loss_mode == "paper_mse":
        print("  Paper Repro Loss: real/imag MSE only (Corr/Peak/FI disabled)")
    else:
        print(f"  Unified Loss (R20.1 fixed blend): L = {corr_weight}*Corr + "
              f"lambda_peak(SNR)*Peak + mse_weight_current*"
              f"[(1-{phase_mix})*NMSE_mag + {phase_mix}*NMSE_complex]")
        print(f"  mse_weight_max={mse_weight_max} | phase_mix={phase_mix} | "
              f"selection_start_epoch={selection_start_epoch if selection_start_epoch is not None else 100}")
    if lambda_peak > 0:
        print(f"  Adaptive Peak Weight: λ_peak(SNR) = {lambda_peak}·σ(({snr_threshold}-SNR)/{lambda_temperature}), "
              f"range ~[0, {lambda_peak}]")
    print(f"  Training SNR: [{snr_train_range[0]}, {snr_train_range[1]}] dB uniform")
    print(f"  Scenario: {sim.scenario_mode} | normalization={sim.normalization_mode} | CV group={cv_group_mode}")
    if sim.scenario_mode == "urban8":
        print(f"  Urban: base_delay={sim.urban_base_delay:g} samples | "
              f"min_los={sim.urban_min_los} | train_los_only={sim.urban_train_los_only}")
        print(f"  Urban sample unit: single-UAV waveform | planned waveforms={n_samples}")
        if dynamic_urban_train:
            n_groups = len(np.unique(urban_snapshot_plan))
            print(f"  Train waveform resampling: enabled every {max(1, int(resample_interval))} epoch(s) "
                  f"on fixed snapshot/UAV plan ({n_groups} snapshots)")
    if beta_fi > 0:
        print(f"  Fisher Loss: β_fi={beta_fi}")
    model_params = sum(p.numel() for p in DAE(cr=cr).parameters())
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
        'scenario_mode': sim.scenario_mode,
        'normalization_mode': sim.normalization_mode,
        'cv_group_mode': cv_group_mode,
        'early_stopping': early_stopping,
        'restore_best': restore_best,
        'final_retrain': final_retrain,
        'final_epochs': final_epochs if final_epochs is not None else epochs,
        'final_epoch_policy': 'configured',
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
            else ('paper_repro_v1' if loss_mode == "paper_mse" else 'R20.1')
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
        final_model = DAE(cr=cr).to(device)
        if dynamic_urban_train:
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
                loss_mode=loss_mode, early_stopping=False, restore_best=False
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
        if dynamic_urban_train:
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

        model = DAE(cr=cr).to(device)
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
                restore_best=restore_best
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
        final_model = DAE(cr=cr).to(device)
        if dynamic_urban_train:
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
                loss_mode=loss_mode, early_stopping=False, restore_best=False
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
