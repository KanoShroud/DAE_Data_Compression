# model.py
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)
    结构：Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    论文依据：对应论文 Fig. 1 中的 Residual Block 结构。
    """

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, 1), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, 1, 1), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, 1, 1), nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class DAE(nn.Module):
    """
    深度去噪自编码器 (Deep Denoising Autoencoder)
    结构：Encoder (Conv + ResBlock) -> Latent Space -> Decoder (ResBlock + DeConv)
    论文依据：对应论文 Section III 的网络架构。
    """

    def __init__(self):
        super().__init__()
        # --- 编码器 (Encoder) ---
        self.enc = nn.Sequential(
            # 下采样层，提取特征并压缩长度
            nn.Conv1d(2, 32, 10, 6, 2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(),
            # 残差块，加深网络
            ResidualBlock(64), ResidualBlock(64)
        )

        # --- 潜在空间 (Latent Space) ---
        # 经过卷积后长度约为 43，通道为 64。
        # 论文设定压缩率 CR=16。原始数据 1024*2 (复数)，压缩后应为 128。
        self.feature_len = 43
        self.fc_enc = nn.Linear(64 * 43, 128)  # 压缩到 128 维
        self.fc_dec = nn.Linear(128, 64 * 43)  # 解压回特征图大小

        # --- 解码器 (Decoder) ---
        self.dec_res = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.dec_conv = nn.Sequential(
            # 上采样层，恢复信号长度
            nn.ConvTranspose1d(64, 32, 5, 2, 2, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 5, 2, 2, 1), nn.BatchNorm1d(32), nn.ReLU(),
            # 最后一层恢复到 2 通道 (I/Q)
            nn.ConvTranspose1d(32, 2, 10, 6, 2, 0)
        )

    def forward(self, x):
        b = x.size(0)
        # 编码 + 展平 + 线性压缩
        z = self.fc_enc(self.enc(x).view(b, -1))
        # 线性恢复 + 重塑 + 解码
        y = self.dec_conv(self.dec_res(self.fc_dec(z).view(b, 64, -1)))

        # 强制插值到 1024 长度 (解决 Padding 可能带来的微小尺寸偏差)
        if y.size(2) != 1024:
            y = nn.functional.interpolate(y, size=1024)
        return y