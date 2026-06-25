# signal_gen.py
import numpy as np
import torch
from scipy import signal


def rrc_filter(beta, span, sps):
    """
    Root Raised Cosine (RRC) 脉冲成形滤波器

    参数:
        beta: 滚降系数 (roll-off factor)，典型值 0.22~0.35
        span: 滤波器跨度，以符号数为单位
        sps:  每符号采样点数 (samples per symbol)

    返回:
        h: RRC 滤波器系数 (归一化能量)
    """
    num_taps = span * sps
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
    h = np.zeros_like(t)

    # t == 0 的情况
    mask_zero = np.abs(t) < 1e-12
    h[mask_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)

    # t == ±1/(4β) 的情况
    mask_edge = np.abs(np.abs(4.0 * beta * t) - 1.0) < 1e-10
    mask_edge &= ~mask_zero
    if np.any(mask_edge):
        t_edge = np.abs(t[mask_edge])
        h[mask_edge] = (beta / np.sqrt(2.0)
                        * ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                           + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))))

    # 一般情况
    mask_normal = ~mask_zero & ~mask_edge
    t_norm = t[mask_normal]
    h[mask_normal] = (
        (np.sin(np.pi * t_norm * (1.0 - beta))
         + 4.0 * beta * t_norm * np.cos(np.pi * t_norm * (1.0 + beta)))
        / (np.pi * t_norm * (1.0 - (4.0 * beta * t_norm) ** 2))
    )

    # 能量归一化
    h /= np.sqrt(np.sum(h ** 2))
    return h


class SignalSimulator:
    """
    信号生成器
    模拟 BPSK 调制的非协作辐射源，经 RRC 脉冲成形、多径信道和接收匹配滤波。
    """

    def __init__(self, signal_len=1024, beta=0.2, rrc_span=8,
                 channel_mode="random", n_fixed_channels=50, channel_pool_seed=42):
        """
        参数:
            signal_len:         信号长度（采样点数）
            beta:               RRC 滚降系数 (0.2 → BW = (1+0.2)×20 = 24 MHz)
            rrc_span:           RRC 滤波器跨度（符号数）
            channel_mode:       "random" 每样本随机信道；"fixed" 从预生成信道池中随机选取
            n_fixed_channels:   固定信道池大小（仅 channel_mode="fixed" 时生效）
            channel_pool_seed:  信道池生成种子（仅 channel_mode="fixed" 时生效）
        """
        self.signal_len = signal_len
        self.samples_per_symbol = 2
        self.beta = beta
        self.rrc_span = rrc_span
        self.channel_mode = channel_mode
        # RRC 滤波器群延迟（采样点）：mode='same' 引入 (len(rrc)-1)//2 = 8 样本延迟
        # 发射端 + 接收端 RRC 各一次 → 总计 16 样本
        # TDOA 计算中两路抵消，不影响差值；仅在需要绝对延迟时需要补偿
        self._rrc_group_delay = (rrc_span * self.samples_per_symbol) // 2 * 2  # = 16

        # RRC 脉冲成形滤波器（替代原 Butterworth 滤波器）
        self.rrc = rrc_filter(beta=beta, span=rrc_span, sps=self.samples_per_symbol)

        # 固定信道池
        self._channel_pool = None
        self._channel_delays = None
        if channel_mode == "fixed":
            self._build_channel_pool(n_fixed_channels, channel_pool_seed)

    def _apply_rrc(self, x, axis=-1):
        """对信号施加 RRC 匹配滤波，输出长度与输入一致"""
        if x.ndim == 1:
            return signal.convolve(x, self.rrc, mode='same')
        else:
            return np.array([signal.convolve(x[i], self.rrc, mode='same')
                             for i in range(x.shape[0])])

    def _build_channel_pool(self, n_channels, seed):
        """
        预生成 n_channels 条固定多径信道，确保可复现。

        Saleh-Valenzuela 聚簇多径信道模型（匹配论文 Wireless InSite 城市场景）：
        - 多径按簇到达，更接近真实城市散射环境
        - 簇到达服从 Poisson 过程（率 Λ）
        - 簇内径到达服从 Poisson 过程（率 λ，λ > Λ）
        - 簇幅度指数衰减（时间常数 Γ）
        - 簇内径幅度指数衰减（时间常数 γ）
        - 20% 概率 NLOS（最强径不是第一径）

        参数说明（采样点为单位，1 采样点 = 25ns @ 40MHz）：
        - Λ = 0.1: 簇到达率，平均簇间距 10 采样点（250ns）
        - λ = 0.5: 簇内径到达率，平均径间距 2 采样点（50ns）
        - Γ = 50: 簇衰减时间常数（1.25μs）
        - γ = 10: 簇内径衰减时间常数（250ns）
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)

        # Saleh-Valenzuela 参数
        Lambda = 0.1      # 簇到达率（每采样点）
        lam = 0.5         # 簇内径到达率（每采样点）
        Gamma = 50        # 簇衰减时间常数（采样点）
        gamma = 10        # 簇内径衰减时间常数（采样点）

        self._channel_pool = []
        self._channel_delays = []
        for _ in range(n_channels):
            main_delay = np.random.randint(10, 50)
            h = np.zeros(self.signal_len, dtype=complex)
            h[main_delay] += 1.0  # LOS 分量

            # 生成簇到达时间（Poisson 过程）
            cluster_delays = []
            t = main_delay + 5
            while t < min(main_delay + 300, self.signal_len):
                cluster_delays.append(t)
                t += np.random.exponential(1.0 / Lambda)

            # 对每个簇生成径
            for cluster_delay in cluster_delays:
                # 簇幅度：指数衰减
                cluster_amp = np.exp(-(cluster_delay - main_delay) / Gamma)

                # 簇内径到达时间（Poisson 过程）
                ray_t = cluster_delay
                while ray_t < min(cluster_delay + 50, self.signal_len):
                    if int(ray_t) < self.signal_len:
                        # 径幅度 = 簇幅度 × 簇内径衰减 × Rayleigh 衰落
                        ray_amp = cluster_amp * np.exp(-(ray_t - cluster_delay) / gamma)
                        rayleigh = np.sqrt(-2 * np.log(np.random.uniform(1e-10, 1.0)))
                        gain = ray_amp * rayleigh * 0.3
                        phase = np.random.uniform(0, 2 * np.pi)
                        h[int(ray_t)] += gain * np.exp(1j * phase)
                    ray_t += np.random.exponential(1.0 / lam)

            # 20% 概率 NLOS：LOS 径被遮挡，NLOS 径成为主导
            if np.random.random() < 0.2:
                # LOS 径衰减 20 倍（模拟物理遮挡）
                h[main_delay] *= 0.05
                # 非 LOS 径中选取最强径放大为新的主导
                non_los_gains = [(i, abs(h[i])) for i in range(len(h))
                                 if i != main_delay and abs(h[i]) > 0]
                if non_los_gains:
                    strongest = max(non_los_gains, key=lambda x: x[1])
                    h[strongest[0]] *= 5.0

            self._channel_pool.append(h)
            self._channel_delays.append(main_delay)

        np.random.set_state(rng_state)
        print(f"[SignalSimulator] 固定信道池已生成: {n_channels} 条信道 (seed={seed})")

    def generate_pair_batch(self, batch_size, snr_db=None):
        """
        生成配对的 UAV 接收信号（两路独立链路）

        参数:
            batch_size: 批量大小
            snr_db:     SNR (dB)。若为 None，训练模式使用 [-10, 10] dB 混合（与评估范围一致）

        返回:
            X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2
        """
        num_symbols = self.signal_len // self.samples_per_symbol
        bits = np.random.choice([-1, 1], size=(batch_size, num_symbols))
        base_signal = np.repeat(bits, self.samples_per_symbol, axis=1)

        if base_signal.shape[1] < self.signal_len:
            pad = np.zeros((batch_size, self.signal_len - base_signal.shape[1]))
            base_signal = np.hstack([base_signal, pad])
        else:
            base_signal = base_signal[:, :self.signal_len]

        # 发射端 RRC 脉冲成形
        base_signal_shaped = self._apply_rrc(base_signal, axis=1)
        u_t = base_signal_shaped + 1j * np.zeros_like(base_signal_shaped)

        def apply_channel_and_noise(u_t_batch):
            # 注意：不保存/恢复RNG状态，确保两次调用(X1和X2)产生不同的随机采样
            X_noisy_out = np.zeros_like(u_t_batch)
            X_clean_out = np.zeros_like(u_t_batch)
            delays = []

            if snr_db is None:
                snrs = np.random.uniform(-10, 10, size=batch_size)
            else:
                snrs = np.ones(batch_size) * snr_db

            for i in range(batch_size):
                if self.channel_mode == "fixed":
                    ch_idx = np.random.randint(0, len(self._channel_pool))
                    h = self._channel_pool[ch_idx]
                    main_delay = self._channel_delays[ch_idx]
                else:
                    main_delay = np.random.randint(10, 50)
                    h = np.zeros(self.signal_len, dtype=complex)
                    h[main_delay] = 1.0
                    for _ in range(np.random.randint(2, 4)):
                        tap_delay = np.random.randint(main_delay + 5,
                                                      min(main_delay + 100, self.signal_len))
                        h[tap_delay] = (np.random.uniform(0.1, 0.4)
                                        * np.exp(1j * np.random.uniform(0, 2 * np.pi)))
                delays.append(main_delay)

                full_conv = signal.convolve(u_t_batch[i], h, mode='full')
                x_clean_raw = full_conv[:self.signal_len]

                # 接收端 RRC 匹配滤波 → 纯净目标
                clean_real = self._apply_rrc(x_clean_raw.real)
                clean_imag = self._apply_rrc(x_clean_raw.imag)
                clean_complex = clean_real + 1j * clean_imag
                X_clean_out[i] = clean_complex

                # 加噪 — 以 Rx RRC 滤波后的信号功率为 SNR 基准
                # RRC 滤波器单位能量归一化(Σ|h|²=1),白噪声通过后功率不变
                sig_p = np.mean(np.abs(clean_complex) ** 2)
                noise_p = sig_p / (10 ** (snrs[i] / 10))
                noise = (np.random.normal(0, 1, self.signal_len)
                         + 1j * np.random.normal(0, 1, self.signal_len)) * np.sqrt(noise_p / 2)
                raw_noisy = x_clean_raw + noise

                # 接收端 RRC 匹配滤波 → 含噪输入
                noisy_real = self._apply_rrc(raw_noisy.real)
                noisy_imag = self._apply_rrc(raw_noisy.imag)
                X_noisy_out[i] = noisy_real + 1j * noisy_imag

            T_noisy = torch.stack([torch.tensor(X_noisy_out.real),
                                   torch.tensor(X_noisy_out.imag)], dim=1).float()
            T_clean = torch.stack([torch.tensor(X_clean_out.real),
                                   torch.tensor(X_clean_out.imag)], dim=1).float()

            return T_noisy, T_clean, delays

        X1_noisy, X1_clean, delays1 = apply_channel_and_noise(u_t)
        X2_noisy, X2_clean, delays2 = apply_channel_and_noise(u_t)

        return X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2

    def generate_training_dataset(self, n_samples, seed=42):
        """
        生成固定训练/验证数据集，确保实验可复现。

        参数:
            n_samples: 样本总数（论文为 10,000）
            seed:      随机种子

        返回:
            X_noisy:  (n_samples, 2, 1024) 含噪输入张量
            X_clean:  (n_samples, 2, 1024) 纯净目标张量
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)

        # 分批次生成，避免内存溢出
        batch_cap = 500
        noisy_list, clean_list = [], []

        for start in range(0, n_samples, batch_cap):
            end = min(start + batch_cap, n_samples)
            bs = end - start
            X1_n, X1_c, _, _, _, _ = self.generate_pair_batch(bs, snr_db=None)
            noisy_list.append(X1_n)
            clean_list.append(X1_c)

        np.random.set_state(rng_state)

        return torch.cat(noisy_list, dim=0), torch.cat(clean_list, dim=0)

    def generate_paired_training_dataset(self, n_samples, seed=42):
        """
        生成配对训练数据集（X1, X2 同信道同噪声），用于相关性损失训练。

        返回:
            X1_noisy, X1_clean, X2_noisy, X2_clean: (n_samples, 2, signal_len)
            tdoa: (n_samples,) 真实 TDOA（采样点）
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)

        batch_cap = 500
        x1n_list, x1c_list, x2n_list, x2c_list, tdoa_list = [], [], [], [], []

        for start in range(0, n_samples, batch_cap):
            end = min(start + batch_cap, n_samples)
            bs = end - start
            X1_n, X1_c, X2_n, X2_c, delays1, delays2 = self.generate_pair_batch(bs, snr_db=None)
            x1n_list.append(X1_n)
            x1c_list.append(X1_c)
            x2n_list.append(X2_n)
            x2c_list.append(X2_c)
            tdoa_list.append(torch.tensor(np.array(delays1) - np.array(delays2)))

        np.random.set_state(rng_state)

        return (torch.cat(x1n_list, dim=0), torch.cat(x1c_list, dim=0),
                torch.cat(x2n_list, dim=0), torch.cat(x2c_list, dim=0),
                torch.cat(tdoa_list, dim=0))
