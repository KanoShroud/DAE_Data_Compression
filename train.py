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


def train_one_fold(device, model, train_loader, val_loader, epochs, lr,
                   fold_idx, patience=20, weight_decay=1e-4, lambda_corr=0.0):
    """
    在单个 fold 上训练模型，含早停机制。

    参数:
        patience:     早停耐心值（连续 N 个 epoch val loss 不改善即停止）
        weight_decay: Adam 优化器的 L2 正则化系数
    返回:
        model, train_loss_hist, val_loss_hist, best_val_loss, stopped_epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    train_loss_hist = []
    val_loss_hist = []
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    stopped_epoch = epochs

    for ep in range(epochs):
        t_ep = time.time()

        model.train()
        ep_loss = 0.0
        ep_corr = 0.0
        for batch in train_loader:
            if lambda_corr > 0 and len(batch) == 4:
                # 配对数据：(X1_noisy, X1_clean, X2_noisy, X2_clean)
                bx1, by1, bx2, by2 = [b.to(device) for b in batch]
                optimizer.zero_grad()

                # 合并 X1/X2 为单次前向传播
                bx_all = torch.cat([bx1, bx2], dim=0)
                y_all = model(bx_all)
                y1, y2 = y_all.chunk(2, dim=0)

                loss_mse = criterion(y1, by1) + criterion(y2, by2)

                # Clean 信号 GCC 不需要梯度
                with torch.no_grad():
                    gcc_clean = gcc_torch(by1[:, 0, :], by1[:, 1, :],
                                          by2[:, 0, :], by2[:, 1, :])
                    gcc_clean_n = gcc_clean / (torch.max(torch.abs(gcc_clean), dim=-1, keepdim=True)[0] + 1e-9)

                # DAE 输出 GCC 需要梯度
                gcc_dae = gcc_torch(y1[:, 0, :], y1[:, 1, :],
                                    y2[:, 0, :], y2[:, 1, :])
                gcc_dae_n = gcc_dae / (torch.max(torch.abs(gcc_dae), dim=-1, keepdim=True)[0] + 1e-9)
                loss_corr = criterion(gcc_dae_n, gcc_clean_n)

                loss = loss_mse + lambda_corr * loss_corr
                loss.backward()
                optimizer.step()
                ep_loss += loss_mse.item() * bx1.size(0)
                ep_corr += loss_corr.item() * bx1.size(0)
            else:
                # 单信号数据：(noisy, clean)
                bx, by = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * bx.size(0)
        train_loss_hist.append(ep_loss / len(train_loader.dataset))

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
            print(f"  Fold {fold_idx} Epoch {ep + 1}/{epochs}: "
                  f"Train {train_loss_hist[-1]:.5f} | "
                  f"Val {val_loss:.5f}{corr_str} | "
                  f"LR: {current_lr:.6f} | {ep_time:.1f}s{early_mark}")

        if epochs_no_improve >= patience:
            stopped_epoch = ep + 1
            print(f"  Fold {fold_idx} Early stopping at Epoch {stopped_epoch} "
                  f"(no improvement for {patience} epochs, epoch耗时 {ep_time:.1f}s)")
            break

    model.load_state_dict(best_state)
    return model, train_loss_hist, val_loss_hist, best_val_loss, stopped_epoch


def train_with_cv(device, cr, k=5, n_samples=10000, epochs=100,
                  batch_size=64, lr=0.0005, seed=42,
                  patience=20, weight_decay=1e-4, use_split=False,
                  channel_mode="random", n_fixed_channels=50, channel_pool_seed=42,
                  sim=None, lambda_corr=0.0):
    """
    训练 DAE 模型，支持 K-fold CV 或单次 train/val 划分。

    参数:
        lambda_corr:  GCC 互相关损失权重（0=纯 MSE，>0 启用相关性正则化）
    """
    # 固定全局随机种子，确保数据集生成、模型初始化和数据划分完全可复现
    torch.manual_seed(seed)
    np.random.seed(seed)

    if sim is None:
        sim = SignalSimulator(channel_mode=channel_mode,
                              n_fixed_channels=n_fixed_channels,
                              channel_pool_seed=channel_pool_seed)

    if lambda_corr > 0:
        # 配对数据：X1 和 X2 来自同信道同噪声，用于 GCC 损失
        X1_n, X1_c, X2_n, X2_c = sim.generate_paired_training_dataset(n_samples, seed=seed)
        dataset = TensorDataset(X1_n, X1_c, X2_n, X2_c)
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
        print(f"  Correlation Loss: λ={lambda_corr}")
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
        model, train_loss, val_loss, best_val, stopped = train_one_fold(
            device, model, train_loader, val_loader, epochs, lr,
            fold_idx + 1, patience=patience, weight_decay=weight_decay,
            lambda_corr=lambda_corr
        )
        fold_time = time.time() - t_fold

        cv_results['fold_train_loss'].append(train_loss)
        cv_results['fold_val_loss'].append(val_loss)
        cv_results['fold_best_val'].append(best_val)
        cv_results['fold_stopped_epoch'].append(stopped)

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
