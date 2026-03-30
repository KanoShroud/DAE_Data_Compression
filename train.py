# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import DAE
from signal_gen import SignalSimulator


def train_model(device, epochs, batch_size=64, lr=0.001):
    """
    训练 DAE 模型。

    参数:
        device: 'cuda' 或 'cpu'
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率

    返回:
        model: 训练好的模型实例
        sim: 信号仿真器实例 (以便后续测试使用相同的参数)
        loss_hist: 训练损失历史列表
    """
    print(f"--- Phase 1: Training Model on {device} ---")

    # 1. 准备数据
    sim = SignalSimulator()
    # 生成 5000 个混合 SNR 的样本用于训练
    print("Generating training data...")
    X, Y, _, _ = sim.generate_batch(5000)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型与优化器
    model = DAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 论文使用 MSE Loss
    loss_hist = []

    # 3. 训练循环
    print("Start Training...")
    for ep in range(epochs):
        model.train()
        ep_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            output = model(bx)

            # Loss 计算：重构信号 vs 无噪多径信号 (Clean Target)
            loss = criterion(output, by)

            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / len(loader)
        loss_hist.append(avg_loss)

        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep + 1}/{epochs}: Loss {avg_loss:.5f}")

    print("Training Finished.")
    return model, sim, loss_hist