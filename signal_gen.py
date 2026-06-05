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

        增强版信道模型（匹配论文城市交叉路口场景）：
        - 4-7 条多径 tap（原 2-4 条），模拟丰富散射环境
        - 指数衰减功率延迟剖面（PDP），符合城市信道统计特性
        - 最大延迟扩展 ~200 采样点（5μs @ 40MHz），对应 ~1500m 路径差
        - 复增益随机相位，模拟各径独立衰落
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)

        self._channel_pool = []
        self._channel_delays = []
        for _ in range(n_channels):
            main_delay = np.random.randint(10, 50)
            h = np.zeros(self.signal_len, dtype=complex)
            h[main_delay] = 1.0

            # 多径 tap 数量：4-7 条（城市环境丰富散射）
            n_taps = np.random.randint(3, 7)

            # 延迟扩展：taps 均匀分布在 [main+5, main+200) 范围
            max_delay = min(main_delay + 200, self.signal_len)
            tap_delays = np.sort(np.random.randint(main_delay + 5, max_delay, size=n_taps))

            # 指数衰减功率延迟剖面（PDP）
            # 相对延迟越大，功率越小；衰减因子 tau_rms 控制衰减速度
            tau_rms = np.random.uniform(20, 60)  # 均方根延迟扩展（采样点）
            for td in tap_delays:
                rel_delay = td - main_delay
                power = np.exp(-rel_delay / tau_rms)  # 指数衰减
                gain = np.sqrt(power) * np.random.uniform(0.2, 0.6)
                phase = np.random.uniform(0, 2 * np.pi)
                h[td] = gain * np.exp(1j * phase)

            self._channel_pool.append(h)
            self._channel_delays.append(main_delay)

        np.random.set_state(rng_state)
        print(f"[SignalSimulator] 固定信道池已生成: {n_channels} 条信道 (seed={seed})")

    def generate_pair_batch(self, batch_size, snr_db=None):
        """
        生成配对的 UAV 接收信号（两路独立链路）

        参数:
            batch_size: 批量大小
            snr_db:     SNR (dB)。若为 None，训练模式使用 [-10, 20] dB 混合

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
            X_noisy_out = np.zeros_like(u_t_batch)
            X_clean_out = np.zeros_like(u_t_batch)
            delays = []

            if snr_db is None:
                snrs = np.random.uniform(-10, 20, size=batch_size)
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
