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


class ComplexLinear(nn.Module):
    """
    复数全连接层 (Complex-valued Fully Connected Layer)

    对复数特征 z = x_r + j·x_i 执行复数矩阵乘法 W·z + b:
      W = W_r + j·W_i
      out_r = W_r·x_r - W_i·x_i + b_r
      out_i = W_r·x_i + W_i·x_r + b_i

    Parameters:
        in_features:  输入复数特征数（不含实/虚拼接翻倍）
        out_features: 输出复数特征数（不含实/虚拼接翻倍）
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, x):
        c_in = x.shape[1] // 2
        x_r = x[:, :c_in]
        x_i = x[:, c_in:]
        out_r = self.fc_r(x_r) - self.fc_i(x_i)
        out_i = self.fc_r(x_i) + self.fc_i(x_r)
        return torch.cat([out_r, out_i], dim=1)


class ComplexBatchNorm1d(nn.Module):
    """
    复数批归一化 (Complex Batch Normalization)

    对复数特征 z = x + iy 执行 2×2 协方差白化归一化：
      z_hat = V^{-1/2} (z - E[z])
      z_out = γ · z_hat + β

    其中 V = [[Vrr, Vri], [Vri, Vii]] 是实部/虚部之间的 2×2 协方差矩阵，
    γ 是 2×2 可学习缩放矩阵，β 是复数可学习偏置。
    """

    def __init__(self, complex_channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.complex_channels = complex_channels
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_Vrr', torch.ones(complex_channels))
        self.register_buffer('running_Vii', torch.ones(complex_channels))
        self.register_buffer('running_Vri', torch.zeros(complex_channels))
        self.register_buffer('running_mean_r', torch.zeros(complex_channels))
        self.register_buffer('running_mean_i', torch.zeros(complex_channels))

        self.gamma_rr = nn.Parameter(torch.ones(complex_channels) / (2 ** 0.5))
        self.gamma_ii = nn.Parameter(torch.ones(complex_channels) / (2 ** 0.5))
        self.gamma_ri = nn.Parameter(torch.zeros(complex_channels))
        self.beta_r = nn.Parameter(torch.zeros(complex_channels))
        self.beta_i = nn.Parameter(torch.zeros(complex_channels))

    def forward(self, x):
        C = self.complex_channels
        x_r = x[:, :C, :]
        x_i = x[:, C:, :]

        if self.training:
            mu_r = x_r.mean(dim=[0, 2])
            mu_i = x_i.mean(dim=[0, 2])
            x_r_ctr = x_r - mu_r[None, :, None]
            x_i_ctr = x_i - mu_i[None, :, None]

            N = x_r.size(0) * x_r.size(2)
            Vrr = (x_r_ctr ** 2).sum(dim=[0, 2]) / N + self.eps
            Vii = (x_i_ctr ** 2).sum(dim=[0, 2]) / N + self.eps
            Vri = (x_r_ctr * x_i_ctr).sum(dim=[0, 2]) / N

            with torch.no_grad():
                self.running_mean_r.mul_(1 - self.momentum).add_(self.momentum * mu_r)
                self.running_mean_i.mul_(1 - self.momentum).add_(self.momentum * mu_i)
                self.running_Vrr.mul_(1 - self.momentum).add_(self.momentum * Vrr)
                self.running_Vii.mul_(1 - self.momentum).add_(self.momentum * Vii)
                self.running_Vri.mul_(1 - self.momentum).add_(self.momentum * Vri)
        else:
            mu_r = self.running_mean_r
            mu_i = self.running_mean_i
            Vrr = self.running_Vrr
            Vii = self.running_Vii
            Vri = self.running_Vri
            x_r_ctr = x_r - mu_r[None, :, None]
            x_i_ctr = x_i - mu_i[None, :, None]

        det = Vrr * Vii - Vri * Vri
        s = torch.sqrt(torch.clamp(det, min=self.eps))
        t = torch.sqrt(torch.clamp(Vrr + Vii + 2 * s, min=self.eps))
        Wrr = (Vii + s) / (s * t)
        Wii = (Vrr + s) / (s * t)
        Wri = -Vri / (s * t)

        x_r_hat = Wrr[None, :, None] * x_r_ctr + Wri[None, :, None] * x_i_ctr
        x_i_hat = Wri[None, :, None] * x_r_ctr + Wii[None, :, None] * x_i_ctr

        out_r = (self.gamma_rr[None, :, None] * x_r_hat
                 + self.gamma_ri[None, :, None] * x_i_hat
                 + self.beta_r[None, :, None])
        out_i = (self.gamma_ri[None, :, None] * x_r_hat
                 + self.gamma_ii[None, :, None] * x_i_hat
                 + self.beta_i[None, :, None])

        return torch.cat([out_r, out_i], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, complex_channels):
        super().__init__()
        self.net = nn.Sequential(
            ComplexConv1d(complex_channels, complex_channels, 3, 1, 1),
            ComplexBatchNorm1d(complex_channels),
            nn.ReLU(),
            ComplexConv1d(complex_channels, complex_channels, 3, 1, 1),
            ComplexBatchNorm1d(complex_channels),
            nn.ReLU(),
            ComplexConv1d(complex_channels, complex_channels, 3, 1, 1),
            ComplexBatchNorm1d(complex_channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DAE(nn.Module):
    def __init__(self, cr=16):
        super().__init__()
        # --- 编码器 (Encoder) ---
        # 维度链: 1024 -> 170 -> 85 -> 43
        self.enc = nn.Sequential(
            ComplexConv1d(1, 32, 10, 6, 2), ComplexBatchNorm1d(32), nn.ReLU(),
            ComplexConv1d(32, 32, 5, 2, 2), ComplexBatchNorm1d(32), nn.ReLU(),
            ComplexConv1d(32, 64, 5, 2, 2), ComplexBatchNorm1d(64), nn.ReLU(),
            ResidualBlock(64), ResidualBlock(64)
        )

        self.feature_len = 43
        self.latent_dim = 2048 // cr

        # 复数全连接层
        self.fc_enc = ComplexLinear(128 * 43 // 2, self.latent_dim // 2)
        self.fc_dec = ComplexLinear(self.latent_dim // 2, 128 * 43 // 2)

        # --- 解码器 (Decoder) ---
        # 维度链: 43 -> 85 -> 170 -> 1024
        self.dec_res = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.dec_conv = nn.Sequential(
            ComplexConvTranspose1d(64, 32, 5, 2, 2, output_padding=0),
            ComplexBatchNorm1d(32), nn.ReLU(),
            ComplexConvTranspose1d(32, 32, 5, 2, 2, output_padding=1),
            ComplexBatchNorm1d(32), nn.ReLU(),
            ComplexConvTranspose1d(32, 1, 10, 6, 2, output_padding=4)
        )

    def forward(self, x):
        b = x.size(0)
        z = self.fc_enc(self.enc(x).view(b, -1))
        y = self.dec_conv(self.dec_res(self.fc_dec(z).view(b, 128, -1)))
        return y
