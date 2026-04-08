# model.py
import torch
import torch.nn as nn

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        c_in = x.shape[1] // 2
        x_r = x[:, :c_in, :]
        x_i = x[:, c_in:, :]
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return torch.cat([out_r, out_i], dim=1)

class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.conv_t_r = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_t_i = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        c_in = x.shape[1] // 2
        x_r = x[:, :c_in, :]
        x_i = x[:, c_in:, :]
        out_r = self.conv_t_r(x_r) - self.conv_t_i(x_i)
        out_i = self.conv_t_r(x_i) + self.conv_t_i(x_r)
        return torch.cat([out_r, out_i], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, complex_channels):
        super().__init__()
        tensor_channels = complex_channels * 2
        self.net = nn.Sequential(
            ComplexConv1d(complex_channels, complex_channels, 3, 1, 1),
            nn.GroupNorm(1, tensor_channels), # 安全的归一化，等效 LayerNorm
            nn.LeakyReLU(0.2),
            ComplexConv1d(complex_channels, complex_channels, 3, 1, 1),
            nn.GroupNorm(1, tensor_channels),
            nn.LeakyReLU(0.2),
            ComplexConv1d(complex_channels, complex_channels, 3, 1, 1),
            nn.GroupNorm(1, tensor_channels)
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.net(x))


class DAE(nn.Module):
    def __init__(self, cr=16):  # 增加 cr 参数，默认 16
        super().__init__()
        # --- 编码器 (Encoder) ---
        self.enc = nn.Sequential(
            ComplexConv1d(1, 32, 10, 6, 2), nn.GroupNorm(1, 64), nn.LeakyReLU(0.2),
            ComplexConv1d(32, 32, 5, 2, 2), nn.GroupNorm(1, 64), nn.LeakyReLU(0.2),
            ComplexConv1d(32, 64, 5, 2, 2), nn.GroupNorm(1, 128), nn.LeakyReLU(0.2),
            ResidualBlock(64), ResidualBlock(64)
        )

        # --- 潜在空间 (Latent Space) ---
        self.feature_len = 43
        # 动态计算压缩后的特征维度 (源数据双通道总长度2048)
        self.latent_dim = 2048 // cr

        self.fc_enc = nn.Linear(128 * 43, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, 128 * 43)

        # --- 解码器 (Decoder) ---
        self.dec_res = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.dec_conv = nn.Sequential(
            ComplexConvTranspose1d(64, 32, 5, 2, 2, output_padding=0), nn.GroupNorm(1, 64), nn.LeakyReLU(0.2),
            ComplexConvTranspose1d(32, 32, 5, 2, 2, output_padding=1), nn.GroupNorm(1, 64), nn.LeakyReLU(0.2),
            ComplexConvTranspose1d(32, 1, 10, 6, 2, output_padding=4)
        )

    # forward 函数保持不变
    def forward(self, x):
        b = x.size(0)
        z = self.fc_enc(self.enc(x).view(b, -1))
        y = self.dec_conv(self.dec_res(self.fc_dec(z).view(b, 128, -1)))
        return y