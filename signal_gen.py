# signal_gen.py
import numpy as np
import torch
from scipy import signal


class SignalSimulator:
    """
    流式信号生成器
    专为非协作被动定位场景设计，实时生成高熵随机 BPSK 序列。
    """

    def __init__(self, signal_len=1024):
        self.signal_len = signal_len
        self.samples_per_symbol = 2
        # 设计 20MHz 带宽的收发端低通滤波器
        self.b_lpf, self.a_lpf = signal.butter(4, 0.5, btype='low')

    def generate_pair_batch(self, batch_size, snr_db=None):
        """
        [核心方法] 实时孪生节点数据生成
        生成完全随机的 BPSK 序列，并模拟两个独立接收链路。
        返回：两路链路的 含噪输入、无噪目标、以及真实延迟。
        """
        # 1. 实时生成高熵发射源信号 (模拟非协作通信)
        num_symbols = self.signal_len // self.samples_per_symbol
        bits = np.random.choice([-1, 1], size=(batch_size, num_symbols))
        base_signal = np.repeat(bits, self.samples_per_symbol, axis=1)

        if base_signal.shape[1] < self.signal_len:
            pad = np.zeros((batch_size, self.signal_len - base_signal.shape[1]))
            base_signal = np.hstack([base_signal, pad])
        else:
            base_signal = base_signal[:, :self.signal_len]

        # 发射端脉冲成形 (Tx LPF)
        base_signal = signal.filtfilt(self.b_lpf, self.a_lpf, base_signal, axis=1)
        u_t = base_signal + 1j * np.zeros_like(base_signal)

        # 2. 内部闭包：模拟单路 UAV 的物理衰落、滤波与加噪过程
        def apply_channel_and_noise(u_t_batch):
            X_noisy_out = np.zeros_like(u_t_batch)
            X_clean_out = np.zeros_like(u_t_batch)
            delays = []

            # 训练时使用混合信噪比 [-15, 15]dB，提升网络在各个噪声区间的鲁棒性
            snrs = np.random.uniform(-15, 15, size=batch_size) if snr_db is None else np.ones(batch_size) * snr_db

            for i in range(batch_size):
                main_delay = np.random.randint(10, 50)
                delays.append(main_delay)

                h = np.zeros(self.signal_len, dtype=complex)
                h[main_delay] = 1.0
                for _ in range(np.random.randint(2, 4)):
                    tap_delay = np.random.randint(main_delay + 5, min(main_delay + 100, self.signal_len))
                    h[tap_delay] = np.random.uniform(0.1, 0.4) * np.exp(1j * np.random.uniform(0, 2 * np.pi))

                # 物理因果卷积 (线性卷积)
                full_conv = signal.convolve(u_t_batch[i], h, mode='full')
                x_clean_raw = full_conv[:self.signal_len]

                # ================= [关键点 1: 生成 Target] =================
                # 对齐滤波尺度：网络要拟合的目标是经过接收端滤波的纯净多径信号
                clean_real = signal.filtfilt(self.b_lpf, self.a_lpf, x_clean_raw.real)
                clean_imag = signal.filtfilt(self.b_lpf, self.a_lpf, x_clean_raw.imag)
                X_clean_out[i] = clean_real + 1j * clean_imag

                # ================= [关键点 2: 生成 Noisy Input] =================
                sig_p = np.mean(np.abs(x_clean_raw) ** 2)
                noise_p = sig_p / (10 ** (snrs[i] / 10))
                noise = (np.random.normal(0, 1, self.signal_len) + 1j * np.random.normal(0, 1,
                                                                                         self.signal_len)) * np.sqrt(
                    noise_p / 2)

                raw_noisy = x_clean_raw + noise

                # 接收端滤波器 (Rx LPF)，彻底滤除带外白噪声
                noisy_real = signal.filtfilt(self.b_lpf, self.a_lpf, raw_noisy.real)
                noisy_imag = signal.filtfilt(self.b_lpf, self.a_lpf, raw_noisy.imag)
                X_noisy_out[i] = noisy_real + 1j * noisy_imag

            # 封装为 PyTorch 张量
            T_noisy = torch.stack([torch.tensor(X_noisy_out.real), torch.tensor(X_noisy_out.imag)], dim=1).float()
            T_clean = torch.stack([torch.tensor(X_clean_out.real), torch.tensor(X_clean_out.imag)], dim=1).float()

            return T_noisy, T_clean, delays

        # 3. 生成两个相互独立的 UAV 接收链路
        X1_noisy, X1_clean, delays1 = apply_channel_and_noise(u_t)
        X2_noisy, X2_clean, delays2 = apply_channel_and_noise(u_t)

        return X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2