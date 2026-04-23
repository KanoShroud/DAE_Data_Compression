# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import DAE
from signal_gen import SignalSimulator


def train_model(device, epochs, cr, batch_size=64, steps_per_epoch=100, lr=0.0005):
    """
    流式训练引擎
    不依赖静态数据集，通过实时调用发生器，提供无限的高熵样本流。
    """
    print(f"--- Training DAE (CR={cr}) on {device} with Infinite Dynamic Stream ---")

    sim = SignalSimulator()
    model = DAE(cr=cr).to(device)

    # 面对高熵动态数据，略微降低学习率以确保梯度下降稳定
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_hist = []

    for ep in range(epochs):
        model.train()
        ep_loss = 0

        # 舍弃传统的 batch loader，按 step 实时生成数据
        for step in range(steps_per_epoch):
            # 实时生成全新数据。snr_db=None 表示使用混合 SNR [-5, 15]dB
            X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2 = sim.generate_pair_batch(batch_size, snr_db=None)

            # 【核心逻辑】：
            # 我们训练模型时，只需使用链路 1 的数据。
            # 网络输入：UAV 1 接收到的含噪、带限信号
            # 网络目标：UAV 1 链路中不含噪声、但包含多径畸变的带限信号
            bx = X1_noisy.to(device)
            by = X1_clean.to(device)

            optimizer.zero_grad()
            output = model(bx)

            # 计算重构损失并更新权重
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        # ================= [新增：步进调度器] =================
        scheduler.step()  # 每个 Epoch 结束后更新一次学习率
        # ========================================================

        avg_loss = ep_loss / steps_per_epoch
        loss_hist.append(avg_loss)

        if (ep + 1) % 10 == 0:
            # 顺便打印出当前的学习率，观察衰减过程
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {ep + 1}/{epochs}: Loss {avg_loss:.5f} | LR: {current_lr:.6f}")

    return model, loss_hist