# train.py
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold

from model import DAE
from signal_gen import SignalSimulator


def train_one_fold(device, model, train_loader, val_loader, epochs, lr,
                   fold_idx, patience=20, weight_decay=1e-4):
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
        model.train()
        ep_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
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
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (ep + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            early_mark = " [EARLY STOP]" if epochs_no_improve >= patience else ""
            print(f"  Fold {fold_idx} Epoch {ep + 1}/{epochs}: "
                  f"Train Loss {train_loss_hist[-1]:.5f} | "
                  f"Val Loss {val_loss:.5f} | LR: {current_lr:.6f}{early_mark}")

        if epochs_no_improve >= patience:
            stopped_epoch = ep + 1
            print(f"  Fold {fold_idx} Early stopping at Epoch {stopped_epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return model, train_loss_hist, val_loss_hist, best_val_loss, stopped_epoch


def train_with_cv(device, cr, k=5, n_samples=10000, epochs=100,
                  batch_size=64, lr=0.0005, seed=42,
                  patience=20, weight_decay=1e-4, use_split=False):
    """
    训练 DAE 模型，支持 K-fold CV 或单次 train/val 划分。

    1. 用固定种子生成含 n_samples 条样本的数据集
    2a. use_split=False: 按 k-fold 划分训练/验证集（论文方案）
    2b. use_split=True:  单次 80/20 随机划分（快速验证用）
    3. 早停 & L2 正则化
    4. 返回验证 loss 最低的模型

    参数:
        device:       torch.device
        cr:           压缩率 (4/8/16)
        k:            fold 数 (use_split=True 时忽略)
        n_samples:    总样本数
        epochs:       最大训练轮数（早停可提前结束）
        batch_size:   批量大小
        lr:           初始学习率
        seed:         数据集随机种子
        patience:     早停耐心值
        weight_decay: L2 正则化系数
        use_split:    True 则使用单次 80/20 划分替代 K-fold CV
    """
    # 固定全局随机种子，确保数据集生成、模型初始化和数据划分完全可复现
    torch.manual_seed(seed)
    np.random.seed(seed)

    sim = SignalSimulator()
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
    model_params = sum(p.numel() for p in DAE(cr=cr).parameters())
    print(f"  Model Parameters: {model_params:,}")
    print(f"{'='*60}")

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
        model, train_loss, val_loss, best_val, stopped = train_one_fold(
            device, model, train_loader, val_loader, epochs, lr,
            fold_idx + 1, patience=patience, weight_decay=weight_decay
        )

        cv_results['fold_train_loss'].append(train_loss)
        cv_results['fold_val_loss'].append(val_loss)
        cv_results['fold_best_val'].append(best_val)
        cv_results['fold_stopped_epoch'].append(stopped)

        print(f"  Fold {fold_idx + 1} Best Val Loss: {best_val:.6f} "
              f"(stopped at epoch {stopped})")

        if best_val < best_overall_val:
            best_overall_val = best_val
            best_model = model

    avg_val = sum(cv_results['fold_best_val']) / n_folds_actual
    avg_stop = sum(cv_results['fold_stopped_epoch']) / n_folds_actual
    print(f"\n  Average Best Val Loss over {n_folds_actual} fold(s): {avg_val:.6f}")
    print(f"  Average Stop Epoch: {avg_stop:.1f}")
    print(f"  Selected model with Val Loss: {best_overall_val:.6f}")

    return best_model, cv_results


def train_model(device, epochs, cr, batch_size=64, steps_per_epoch=100,
                lr=0.0005, weight_decay=1e-4):
    """
    流式训练引擎（保留向后兼容）

    若需要在 CV 模式下训练，请使用 train_with_cv()。
    """
    print(f"--- Training DAE (CR={cr}) on {device} with Infinite Dynamic Stream ---")

    sim = SignalSimulator()
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
