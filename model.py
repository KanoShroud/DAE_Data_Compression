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


class FrequencySelectiveDAE(nn.Module):
    """
    Frequency-task-aware DAE v1.

    This model keeps the original Chen-style DAE backbone unchanged and adds a
    lightweight, learnable frequency pre-emphasis layer in front of it. The
    prior is based on the delay-estimation intuition that higher RMS bandwidth
    carries more timing information, but the mask is soft and trainable instead
    of hard-selecting frequency bins. That keeps this innovation branch
    isolated from the paper reproduction baseline while giving the encoder a
    better inductive bias for TDOA-sensitive details.
    """

    def __init__(self, cr=16, signal_len=1024, prior_strength=1.5,
                 min_gain=0.35, max_gain=1.65):
        super().__init__()
        self.cr = int(cr)
        self.signal_len = int(signal_len)
        self.prior_strength = float(prior_strength)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)

        # The backbone preserves the original latent dimensionality convention:
        # transmitted real scalars = 2 * signal_len / CR.
        self.backbone = DAE(cr=cr)

        freq = torch.fft.fftfreq(self.signal_len)
        freq_norm = torch.abs(freq) / (torch.max(torch.abs(freq)) + 1e-12)
        fisher_prior = freq_norm.pow(2)
        fisher_prior = fisher_prior / (torch.max(fisher_prior) + 1e-12)
        self.register_buffer("fisher_prior", fisher_prior.float())

        init_mask = 0.50 + 0.35 * fisher_prior
        init_mask = torch.clamp(init_mask, 1e-4, 1.0 - 1e-4)
        init_logits = torch.log(init_mask / (1.0 - init_mask))
        self.freq_logits = nn.Parameter(init_logits.float())
        self.emphasis_logit = nn.Parameter(torch.tensor(0.0))

    def _frequency_emphasis(self, x):
        if x.shape[-1] != self.signal_len:
            raise ValueError(
                f"FrequencySelectiveDAE expects signal_len={self.signal_len}, "
                f"got {x.shape[-1]}"
            )

        xc = torch.complex(x[:, 0, :], x[:, 1, :])
        X = torch.fft.fft(xc, dim=-1)

        learned = torch.sigmoid(self.freq_logits)
        prior_boost = 1.0 + self.prior_strength * torch.sigmoid(self.emphasis_logit) * self.fisher_prior
        gain = torch.clamp(learned * prior_boost, min=self.min_gain, max=self.max_gain)

        Xg = X * gain[None, :]
        yg = torch.fft.ifft(Xg, dim=-1)

        # Preserve each sample's RMS scale so the front-end changes spectral
        # emphasis, not the trivial signal power seen by the backbone.
        in_rms = torch.sqrt(torch.mean(torch.abs(xc) ** 2, dim=-1, keepdim=True) + 1e-9)
        out_rms = torch.sqrt(torch.mean(torch.abs(yg) ** 2, dim=-1, keepdim=True) + 1e-9)
        yg = yg * (in_rms / out_rms)
        return torch.stack((yg.real, yg.imag), dim=1).float()

    def forward(self, x):
        return self.backbone(self._frequency_emphasis(x))


class FrequencyPairwiseDAE(FrequencySelectiveDAE):
    """
    Frequency-task-aware DAE v2.

    The inference interface stays identical to the Chen-style DAE: one received
    waveform in, one reconstructed waveform out. The difference from v1 is the
    training objective, which uses urban LOS waveform pairs and explicitly
    constrains cross-spectrum phase/GCC/TDOA consistency. The front-end prior is
    intentionally softer than v1 so the pairwise objective, rather than a fixed
    high-frequency bias, decides which bands are useful.
    """

    def __init__(self, cr=16, signal_len=1024, prior_strength=0.8,
                 min_gain=0.50, max_gain=1.50):
        super().__init__(
            cr=cr,
            signal_len=signal_len,
            prior_strength=prior_strength,
            min_gain=min_gain,
            max_gain=max_gain,
        )


class NestedFrequencyPairwiseDAE(nn.Module):
    """
    Frequency-task-aware DAE v3.

    v1/v2 trained one independent bottleneck per CR. This supernet keeps one
    maximum CR=4 latent space and exposes nested prefixes for CR=16/8/4:

        CR16 prefix ⊂ CR8 prefix ⊂ CR4 prefix

    The forward path still returns a reconstructed waveform, so all existing
    Chen-style waveform/GCC/WLS evaluation code remains usable. During training,
    an auxiliary pairwise task head can read two latent prefixes and predict a
    soft TDOA-lag distribution. That gives the shared encoder a task branch
    without replacing the waveform reconstruction chain.
    """

    def __init__(self, cr=16, signal_len=1024, max_cr=4,
                 prior_strength=0.6, min_gain=0.65, max_gain=1.35,
                 nested_crs=(4, 8, 16), task_hidden=512):
        super().__init__()
        self.cr = int(cr)
        self.active_cr = int(cr)
        self.signal_len = int(signal_len)
        self.max_cr = int(max_cr)
        self.nested_crs = tuple(int(v) for v in nested_crs)
        self.max_real_latent_dim = 2048 // self.max_cr
        self.max_complex_latent_dim = self.max_real_latent_dim // 2
        self.feature_len = 43
        self.prior_strength = float(prior_strength)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)

        self.enc = nn.Sequential(
            ComplexConv1d(1, 32, 10, 6, 2), ComplexBatchNorm1d(32), nn.ReLU(),
            ComplexConv1d(32, 32, 5, 2, 2), ComplexBatchNorm1d(32), nn.ReLU(),
            ComplexConv1d(32, 64, 5, 2, 2), ComplexBatchNorm1d(64), nn.ReLU(),
            ResidualBlock(64), ResidualBlock(64)
        )
        self.fc_enc = ComplexLinear(128 * 43 // 2, self.max_complex_latent_dim)
        self.fc_dec = ComplexLinear(self.max_complex_latent_dim, 128 * 43 // 2)
        self.dec_res = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.dec_conv = nn.Sequential(
            ComplexConvTranspose1d(64, 32, 5, 2, 2, output_padding=0),
            ComplexBatchNorm1d(32), nn.ReLU(),
            ComplexConvTranspose1d(32, 32, 5, 2, 2, output_padding=1),
            ComplexBatchNorm1d(32), nn.ReLU(),
            ComplexConvTranspose1d(32, 1, 10, 6, 2, output_padding=4)
        )

        freq = torch.fft.fftfreq(self.signal_len)
        freq_norm = torch.abs(freq) / (torch.max(torch.abs(freq)) + 1e-12)
        fisher_prior = freq_norm.pow(2)
        fisher_prior = fisher_prior / (torch.max(fisher_prior) + 1e-12)
        self.register_buffer("fisher_prior", fisher_prior.float())
        init_mask = 0.55 + 0.25 * fisher_prior
        init_mask = torch.clamp(init_mask, 1e-4, 1.0 - 1e-4)
        self.freq_logits = nn.Parameter(torch.log(init_mask / (1.0 - init_mask)))
        self.emphasis_logit = nn.Parameter(torch.tensor(0.0))

        pair_dim = self.max_real_latent_dim * 4
        self.task_head = nn.Sequential(
            nn.Linear(pair_dim, int(task_hidden)),
            nn.ReLU(),
            nn.Linear(int(task_hidden), self.signal_len),
        )

    def set_active_cr(self, cr):
        cr = int(cr)
        if cr not in self.nested_crs:
            raise ValueError(f"Unsupported nested CR={cr}; expected one of {self.nested_crs}")
        self.active_cr = cr

    def active_real_dim(self, cr=None):
        cr = self.active_cr if cr is None else int(cr)
        return 2048 // cr

    def _frequency_emphasis(self, x):
        if x.shape[-1] != self.signal_len:
            raise ValueError(
                f"NestedFrequencyPairwiseDAE expects signal_len={self.signal_len}, "
                f"got {x.shape[-1]}"
            )
        xc = torch.complex(x[:, 0, :], x[:, 1, :])
        X = torch.fft.fft(xc, dim=-1)
        learned = torch.sigmoid(self.freq_logits)
        prior_boost = 1.0 + self.prior_strength * torch.sigmoid(self.emphasis_logit) * self.fisher_prior
        gain = torch.clamp(learned * prior_boost, min=self.min_gain, max=self.max_gain)
        yg = torch.fft.ifft(X * gain[None, :], dim=-1)
        in_rms = torch.sqrt(torch.mean(torch.abs(xc) ** 2, dim=-1, keepdim=True) + 1e-9)
        out_rms = torch.sqrt(torch.mean(torch.abs(yg) ** 2, dim=-1, keepdim=True) + 1e-9)
        yg = yg * (in_rms / out_rms)
        return torch.stack((yg.real, yg.imag), dim=1).float()

    def _mask_latent(self, z, cr=None):
        active_dim = self.active_real_dim(cr)
        if active_dim > z.shape[1]:
            raise ValueError(f"active latent dim {active_dim} exceeds z dim {z.shape[1]}")
        mask = torch.zeros_like(z)
        half_total = z.shape[1] // 2
        half_active = active_dim // 2
        # Preserve the first active complex coordinates in both real and imag halves.
        mask[:, :half_active] = 1.0
        mask[:, half_total:half_total + half_active] = 1.0
        return z * mask

    def encode_latent(self, x, cr=None):
        z = self.encode_full_latent(x)
        return self._mask_latent(z, cr=cr)

    def encode_full_latent(self, x):
        b = x.size(0)
        return self.fc_enc(self.enc(self._frequency_emphasis(x)).view(b, -1))

    def decode_latent(self, z):
        b = z.size(0)
        return self.dec_conv(self.dec_res(self.fc_dec(z).view(b, 128, -1)))

    def forward(self, x, return_latent=False):
        z = self.encode_latent(x)
        y = self.decode_latent(z)
        if return_latent:
            return y, z
        return y

    def pair_task_logits_from_latent(self, z1, z2):
        pair_feat = torch.cat([z1, z2, z1 - z2, z1 * z2], dim=1)
        return self.task_head(pair_feat)


class NestedTaskSufficientDAE(NestedFrequencyPairwiseDAE):
    """
    FreqDAE v4-min: nested task-sufficient compression.

    Compared with v3, this model keeps the same waveform reconstruction
    interface but exposes an uncertainty head for the pairwise TDOA branch.
    Training can evaluate all nested prefixes (CR4/8/16) from one full latent
    vector, which lets the loss enforce monotonic resource use instead of
    hoping that extra dimensions become useful indirectly.
    """

    def __init__(self, cr=16, signal_len=1024, max_cr=4,
                 prior_strength=0.45, min_gain=0.70, max_gain=1.30,
                 nested_crs=(4, 8, 16), task_hidden=512):
        super().__init__(
            cr=cr,
            signal_len=signal_len,
            max_cr=max_cr,
            prior_strength=prior_strength,
            min_gain=min_gain,
            max_gain=max_gain,
            nested_crs=nested_crs,
            task_hidden=task_hidden,
        )
        pair_dim = self.max_real_latent_dim * 4
        self.uncertainty_head = nn.Sequential(
            nn.Linear(pair_dim, int(task_hidden // 2)),
            nn.ReLU(),
            nn.Linear(int(task_hidden // 2), 1),
        )

    def pair_features_from_latent(self, z1, z2):
        return torch.cat([z1, z2, z1 - z2, z1 * z2], dim=1)

    def pair_task_outputs_from_latent(self, z1, z2):
        pair_feat = self.pair_features_from_latent(z1, z2)
        logits = self.task_head(pair_feat)
        log_var = torch.clamp(self.uncertainty_head(pair_feat).squeeze(-1), -6.0, 6.0)
        return logits, log_var

    def pair_task_logits_from_latent(self, z1, z2):
        logits, _ = self.pair_task_outputs_from_latent(z1, z2)
        return logits
