# train.py
import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold

from model import DAE
from signal_gen import SignalSimulator


class AdaptiveLambda(nn.Module):
    """
    自适应 λ 模块：根据 SNR 动态调整 MSE 和相关性损失的权重

    λ(SNR) = λ_min + (λ_max - λ_min) · σ((SNR_threshold - SNR) / temperature)

    低 SNR → λ 接近 λ_max（更重视 GCC 结构保留）
    高 SNR → λ 接近 λ_min（更重视波形重建）

    λ 始终限制在 [λ_min, λ_max] 范围内（默认 [0.20, 0.50]，以 λ=0.30 为中心），
    平衡 GCC 结构保留与 MSE 波形重建。
    """

    def __init__(self, lambda_min=0.20, lambda_max=0.50,
                 snr_threshold=0.0, temperature=5.0):
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.snr_threshold = snr_threshold
        self.temperature = temperature

    def forward(self, snr):
        """
        参数:
            snr: SNR 值 (dB)，标量或张量
        返回:
            λ 值，范围 [lambda_min, lambda_max]
        """
        raw = torch.sigmoid(
            (self.snr_threshold - snr) / self.temperature
        )
        return self.lambda_min + (self.lambda_max - self.lambda_min) * raw


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

    参数:
        sig1_real, sig1_imag: (batch, signal_len) 实部和虚部
        sig2_real, sig2_imag: (batch, signal_len)
        fft_len: FFT 长度（默认 2 的幂次，比 2*signal_len-1 更快）
    返回:
        cc_real: (batch, fft_len) 互相关函数（实部）
    """
    n = sig1_real.shape[-1]
    if fft_len is None:
        fft_len = 1
        while fft_len < 2 * n - 1:
            fft_len <<= 1

    sig1 = torch.complex(sig1_real, sig1_imag)
    sig2 = torch.complex(sig2_real, sig2_imag)

    SIG1 = torch.fft.fft(sig1, n=fft_len)
    SIG2 = torch.fft.fft(sig2, n=fft_len)

    R = SIG1 * torch.conj(SIG2)
    cc = torch.fft.ifft(R)
    cc = torch.fft.fftshift(cc, dim=-1)

    return cc.real


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
        fft_len = 1
        while fft_len < 2 * n - 1:
            fft_len <<= 1

    # 计算复数 GCC
    sig1 = torch.complex(y1_real, y1_imag)
    sig2 = torch.complex(y2_real, y2_imag)
    SIG1 = torch.fft.fft(sig1, n=fft_len)
    SIG2 = torch.fft.fft(sig2, n=fft_len)
    R = SIG1 * torch.conj(SIG2)
    cc = torch.fft.ifft(R)
    cc = torch.fft.fftshift(cc, dim=-1)

    # 计算 GCC 幅度
    cc_mag = torch.abs(cc)  # (batch, fft_len)

    # 真实 TDOA 对应的索引（fftshift 后零延迟在 fft_len//2）
    zero_idx = fft_len // 2
    true_idx = zero_idx + true_tdoa.long()
    true_idx = torch.clamp(true_idx, 0, fft_len - 1)

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
        fft_len = 1
        while fft_len < 2 * n - 1:
            fft_len <<= 1

    sig1 = torch.complex(y1_real, y1_imag)
    sig2 = torch.complex(y2_real, y2_imag)
    SIG1 = torch.fft.fft(sig1, n=fft_len)
    SIG2 = torch.fft.fft(sig2, n=fft_len)
    R = SIG1 * torch.conj(SIG2)
    cc = torch.fft.ifft(R)
    cc = torch.fft.fftshift(cc, dim=-1)

    cc_mag = torch.abs(cc)
    zero_idx = fft_len // 2
    true_idx = zero_idx + true_tdoa.long()
    true_idx = torch.clamp(true_idx, 0, fft_len - 1)

    batch_idx = torch.arange(cc_mag.shape[0], device=cc_mag.device)
    peak_at_true = cc_mag[batch_idx, true_idx]
    peak_max = torch.max(cc_mag, dim=-1)[0]
    peak_norm = peak_at_true / (peak_max + 1e-9)

    return 1.0 - peak_norm  # (batch,)


def adaptive_peak_lambda(snr, peak_max=0.15, snr_center=0.0, temperature=5.0):
    """
    峰值损失权重：高 SNR 时峰可靠 → 权重大；低 SNR 时峰噪声主导 → 权重小。

    λ_peak(SNR) = peak_max · σ((SNR - snr_center) / temperature)

    与 AdaptiveLambda 形成互补：
      - 低 SNR：λ_corr 温和增强 + λ_peak 接近零（避免噪声梯度）
      - 高 SNR：λ_corr 温和减弱 + λ_peak 接近 peak_max（峰值可靠，精调 TDOA）

    参数:
        snr:          (batch,) 每样本 SNR (dB)
        peak_max:     峰值损失最大权重（默认 0.15）
        snr_center:   sigmoid 中心点 SNR (dB)
        temperature:  sigmoid 温度参数

    返回:
        lambda_peak: (batch,) 每样本的峰值损失权重
    """
    return peak_max * torch.sigmoid((snr - snr_center) / temperature)


def train_one_fold(device, model, train_loader, val_loader, epochs, lr,
                   fold_idx, patience=20, weight_decay=1e-4, lambda_corr=0.0,
                   lambda_peak=0.0, adaptive_lambda=None):
    """
    在单个 fold 上训练模型，含早停机制。

    参数:
        patience:       早停耐心值（连续 N 个 epoch val loss 不改善即停止）
        weight_decay:   Adam 优化器的 L2 正则化系数
        lambda_peak:    PNCC 峰值损失权重
        adaptive_lambda: AdaptiveLambda 实例，None 则使用固定 lambda_corr
    返回:
        model, train_loss_hist, val_loss_hist, best_val_loss, stopped_epoch,
        corr_hist, peak_hist, lambda_hist
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    train_loss_hist = []
    val_loss_hist = []
    corr_hist = []
    peak_hist = []
    lambda_hist = []
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    stopped_epoch = epochs

    for ep in range(epochs):
        t_ep = time.time()

        model.train()
        ep_loss = 0.0
        ep_corr = 0.0
        ep_peak = 0.0
        ep_lambda = 0.0
        n_batches = 0
        for batch in train_loader:
            if lambda_corr > 0 and len(batch) >= 4:
                # 配对数据：(X1_noisy, X1_clean, X2_noisy, X2_clean[, tdoa])
                bx1, by1, bx2, by2 = [b.to(device) for b in batch[:4]]
                true_tdoa = batch[4].to(device) if len(batch) >= 5 else None
                optimizer.zero_grad()

                # 合并 X1/X2 为单次前向传播
                bx_all = torch.cat([bx1, bx2], dim=0)
                y_all = model(bx_all)
                y1, y2 = y_all.chunk(2, dim=0)

                # 逐样本 MSE 损失
                mse1 = torch.mean((y1 - by1) ** 2, dim=[1, 2])  # (batch,)
                mse2 = torch.mean((y2 - by2) ** 2, dim=[1, 2])  # (batch,)
                mse_per_sample = mse1 + mse2  # (batch,)

                # Clean 信号 GCC 不需要梯度
                with torch.no_grad():
                    gcc_clean = gcc_torch(by1[:, 0, :], by1[:, 1, :],
                                          by2[:, 0, :], by2[:, 1, :])
                    gcc_clean_n = gcc_clean / (torch.max(torch.abs(gcc_clean), dim=-1, keepdim=True)[0] + 1e-9)

                # DAE 输出 GCC 需要梯度
                gcc_dae = gcc_torch(y1[:, 0, :], y1[:, 1, :],
                                    y2[:, 0, :], y2[:, 1, :])
                gcc_dae_n = gcc_dae / (torch.max(torch.abs(gcc_dae), dim=-1, keepdim=True)[0] + 1e-9)

                # 逐样本 GCC 相关性损失
                corr_per_sample = torch.mean((gcc_dae_n - gcc_clean_n) ** 2, dim=1)  # (batch,)

                # 计算逐样本 SNR（自适应 λ 和峰值损失共用）
                if adaptive_lambda is not None:
                    snr_per_sample = estimate_per_sample_snr(bx1, by1)  # (batch,)
                    lambda_per_sample = adaptive_lambda(snr_per_sample)  # (batch,)
                else:
                    lambda_per_sample = torch.full((bx1.size(0),), lambda_corr, device=device)
                    snr_per_sample = None

                # PNCC 峰值损失（逐样本自适应权重：高 SNR 权重大，低 SNR 权重小）
                loss_peak = torch.tensor(0.0, device=device)
                if lambda_peak > 0 and true_tdoa is not None:
                    if snr_per_sample is not None:
                        # 逐样本峰值损失 + SNR 正向依赖权重
                        peak_per_sample = gcc_peak_loss_per_sample(
                            y1[:, 0, :], y1[:, 1, :],
                            y2[:, 0, :], y2[:, 1, :],
                            true_tdoa.float()
                        )
                        peak_lambda = adaptive_peak_lambda(snr_per_sample, peak_max=lambda_peak)
                        loss_peak = torch.mean(peak_lambda * peak_per_sample)
                    else:
                        # 无自适应 λ 时使用固定权重（向后兼容）
                        loss_peak = lambda_peak * gcc_peak_loss(
                            y1[:, 0, :], y1[:, 1, :],
                            y2[:, 0, :], y2[:, 1, :],
                            true_tdoa.float()
                        )

                # 加权总损失
                loss_mse = torch.mean(mse_per_sample)
                loss_corr = torch.mean(corr_per_sample)
                loss = torch.mean(mse_per_sample + lambda_per_sample * corr_per_sample) + loss_peak

                loss.backward()
                optimizer.step()
                ep_loss += loss_mse.item() * bx1.size(0)
                ep_corr += loss_corr.item() * bx1.size(0)
                ep_peak += loss_peak.item() * bx1.size(0)
                ep_lambda += torch.mean(lambda_per_sample).item()
                n_batches += 1
            else:
                # 单信号数据：(noisy, clean)
                bx, by = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * bx.size(0)
        train_loss_hist.append(ep_loss / len(train_loader.dataset))
        corr_hist.append(ep_corr / len(train_loader.dataset) if n_batches > 0 else 0.0)
        peak_hist.append(ep_peak / len(train_loader.dataset) if n_batches > 0 else 0.0)
        lambda_hist.append(ep_lambda / n_batches if n_batches > 0 else lambda_corr)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    bx, by = batch[0].to(device), batch[1].to(device)
                else:
                    bx, by = batch[0].to(device), batch[1].to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ep_time = time.time() - t_ep

        if (ep + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            early_mark = " [EARLY STOP]" if epochs_no_improve >= patience else ""
            corr_str = f" | Corr {ep_corr / len(train_loader.dataset):.5f}" if lambda_corr > 0 else ""
            peak_str = f" | Peak {ep_peak / len(train_loader.dataset):.5f}" if lambda_peak > 0 else ""
            lambda_str = f" | λ={ep_lambda / n_batches:.3f}" if adaptive_lambda is not None and n_batches > 0 else ""
            print(f"  Fold {fold_idx} Epoch {ep + 1}/{epochs}: "
                  f"Train {train_loss_hist[-1]:.5f} | "
                  f"Val {val_loss:.5f}{corr_str}{peak_str}{lambda_str} | "
                  f"LR: {current_lr:.6f} | {ep_time:.1f}s{early_mark}")

        if epochs_no_improve >= patience:
            stopped_epoch = ep + 1
            print(f"  Fold {fold_idx} Early stopping at Epoch {stopped_epoch} "
                  f"(no improvement for {patience} epochs, epoch耗时 {ep_time:.1f}s)")
            break

    model.load_state_dict(best_state)
    return model, train_loss_hist, val_loss_hist, best_val_loss, stopped_epoch, \
        corr_hist, peak_hist, lambda_hist


def train_with_cv(device, cr, k=5, n_samples=10000, epochs=100,
                  batch_size=64, lr=0.0005, seed=42,
                  patience=20, weight_decay=1e-4, use_split=False,
                  channel_mode="random", n_fixed_channels=50, channel_pool_seed=42,
                  sim=None, lambda_corr=0.0, lambda_peak=0.0,
                  use_adaptive_lambda=False, snr_threshold=0.0, lambda_temperature=5.0):
    """
    训练 DAE 模型，支持 K-fold CV 或单次 train/val 划分。

    参数:
        lambda_corr:          GCC 互相关损失权重（0=纯 MSE，>0 启用相关性正则化）
        lambda_peak:          PNCC 峰值损失权重（0=不启用，>0 启用）
        use_adaptive_lambda:  是否使用自适应 λ（根据 SNR 动态调整）
        snr_threshold:        自适应 λ 的 SNR 阈值（dB）
        lambda_temperature:   自适应 λ 的温度参数
    """
    # 固定全局随机种子，确保数据集生成、模型初始化和数据划分完全可复现
    torch.manual_seed(seed)
    np.random.seed(seed)

    if sim is None:
        sim = SignalSimulator(channel_mode=channel_mode,
                              n_fixed_channels=n_fixed_channels,
                              channel_pool_seed=channel_pool_seed)

    if lambda_corr > 0 or lambda_peak > 0:
        # 配对数据：X1 和 X2 来自同信道同噪声，用于 GCC 损失和峰值损失
        X1_n, X1_c, X2_n, X2_c, tdoa = sim.generate_paired_training_dataset(n_samples, seed=seed)
        dataset = TensorDataset(X1_n, X1_c, X2_n, X2_c, tdoa)
    else:
        X_noisy, X_clean = sim.generate_training_dataset(n_samples, seed=seed)
        dataset = TensorDataset(X_noisy, X_clean)

    if use_split:
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
    print(f"  Early Stopping Patience: {patience} | Weight Decay: {weight_decay}")
    if lambda_corr > 0:
        if use_adaptive_lambda:
            print(f"  Correlation Loss: Adaptive λ ∈ [0.20, 0.50] "
                  f"(center={snr_threshold}dB, temp={lambda_temperature})")
        else:
            print(f"  Correlation Loss: Fixed λ={lambda_corr}")
    if lambda_peak > 0:
        if use_adaptive_lambda:
            print(f"  Peak Loss: Adaptive λ_peak ∈ [~0, {lambda_peak}] "
                  f"(SNR-positive sigmoid, center={snr_threshold}dB, temp={lambda_temperature})")
        else:
            print(f"  Peak Loss: Fixed λ_peak={lambda_peak}")
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
        'fold_avg_lambda': [],
    }

    best_model = None
    best_overall_val = float('inf')

    if use_split:
        split_iterator = [(train_idx, val_idx)]
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        split_iterator = list(kf.split(range(n_samples)))

    for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx),
                                batch_size=batch_size, shuffle=False)

        model = DAE(cr=cr).to(device)
        t_fold = time.time()

        # 创建自适应 λ 实例（如果启用）
        adaptive_lambda = None
        if use_adaptive_lambda and lambda_corr > 0:
            adaptive_lambda = AdaptiveLambda(
                snr_threshold=snr_threshold, temperature=lambda_temperature).to(device)

        model, train_loss, val_loss, best_val, stopped, corr_loss, peak_loss, avg_lambda = \
            train_one_fold(
                device, model, train_loader, val_loader, epochs, lr,
                fold_idx + 1, patience=patience, weight_decay=weight_decay,
                lambda_corr=lambda_corr, lambda_peak=lambda_peak,
                adaptive_lambda=adaptive_lambda
            )
        fold_time = time.time() - t_fold

        cv_results['fold_train_loss'].append(train_loss)
        cv_results['fold_val_loss'].append(val_loss)
        cv_results['fold_best_val'].append(best_val)
        cv_results['fold_stopped_epoch'].append(stopped)
        cv_results['fold_corr_loss'].append(corr_loss)
        cv_results['fold_peak_loss'].append(peak_loss)
        cv_results['fold_avg_lambda'].append(avg_lambda)

        print(f"  Fold {fold_idx + 1} Best Val Loss: {best_val:.6f} "
              f"(stopped at epoch {stopped},耗时 {fold_time:.1f}s)")

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
