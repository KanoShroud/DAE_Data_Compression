# signal_gen.py
import numpy as np
import torch
from scipy import signal

class SignalSimulator:
    """
    信号生成器类
    功能：模拟生成经过多径信道传输并受AWGN影响的BPSK信号。
    论文依据：对应论文 Section II System Model 中的信号模型。
    """

    def __init__(self, signal_len=1024):
        self.signal_len = signal_len
        self.samples_per_symbol = 2  # 模拟过采样，每个符号占2个采样点

        # 预先设计发射端(Tx)和接收端(Rx)的低通滤波器 (带宽 20MHz，归一化截止频率 0.5)
        self.b_lpf, self.a_lpf = signal.butter(4, 0.5, btype='low')

    def generate_batch(self, batch_size, snr_db=None):
        """
        生成一批训练或测试数据。

        参数:
            batch_size (int): 样本数量
            snr_db (float or None): 指定信噪比(dB)。如果为None，则在[-5, 15]dB间随机生成。

        返回:
            X_in (Tensor): 含噪信号 (Batch, 2, 1024) - 用于模型输入
            Y_target (Tensor): 无噪多径信号 (Batch, 2, 1024) - 用于计算Loss
            u_t (ndarray): 原始发射源信号 (Batch, 1024) - 用于互相关验证
            true_delays (list): 真实主径延迟 - 用于验证TDOA精度
        """
        # 1. 生成 BPSK 源信号 (Source Signal)
        num_symbols = self.signal_len // self.samples_per_symbol
        bits = np.random.choice([-1, 1], size=(batch_size, num_symbols))
        base_signal = np.repeat(bits, self.samples_per_symbol, axis=1)

        # 补齐长度到 signal_len
        if base_signal.shape[1] < self.signal_len:
            pad = np.zeros((batch_size, self.signal_len - base_signal.shape[1]))
            base_signal = np.hstack([base_signal, pad])
        else:
            base_signal = base_signal[:, :self.signal_len]

        # 1. 发射端脉冲成形 (Tx LPF)
        base_signal = signal.filtfilt(self.b_lpf, self.a_lpf, base_signal, axis=1)
        u_t = base_signal + 1j * np.zeros_like(base_signal)

        # 2. 模拟多径信道 (Multipath Channel)
        x_clean = np.zeros_like(u_t)
        true_delays = []

        for i in range(batch_size):
            # 随机生成主径延迟 (10-50采样点)
            main_delay = np.random.randint(10, 50)
            true_delays.append(main_delay)

            # 构建信道冲激响应 h(t)
            h = np.zeros(self.signal_len, dtype=complex)
            h[main_delay] = 1.0  # 主径分量

            # 添加 2-3 条随机弱径
            for _ in range(np.random.randint(2, 4)):
                tap_delay = np.random.randint(main_delay + 5, min(main_delay + 100, self.signal_len))
                # 随机复数衰减
                h[tap_delay] = np.random.uniform(0.1, 0.4) * np.exp(1j * np.random.uniform(0, 2 * np.pi))

            # ================= [关键修复 1：恢复物理因果卷积] =================
            # 摒弃破坏因果律的 mode='same'。
            # 使用 mode='full' 计算完整卷积，然后截取前 signal_len 个点。
            # 这才是射频信号在时域上延迟的真实物理过程！
            full_conv = signal.convolve(u_t[i], h, mode='full')
            x_clean[i] = full_conv[:self.signal_len]
            # ==================================================================


        # 3. 添加高斯白噪声 (AWGN)
        x_noisy = np.zeros_like(x_clean)
        # 确定信噪比
        if snr_db is None:
            snrs = np.random.uniform(-5, 15, size=batch_size)  # 训练模式：混合SNR
        else:
            snrs = np.ones(batch_size) * snr_db  # 测试模式：固定SNR

        for i in range(batch_size):
            # 计算信号功率
            sig_p = np.mean(np.abs(x_clean[i]) ** 2)
            # 计算噪声功率
            noise_p = sig_p / (10 ** (snrs[i] / 10))
            # 生成复数高斯噪声
            noise = (np.random.normal(0, 1, self.signal_len) + 1j * np.random.normal(0, 1, self.signal_len)) * np.sqrt(noise_p / 2)
            x_noisy[i] = x_clean[i] + noise

            # raw_noisy = x_clean[i] + noise
            # 3. 引入接收端低通滤波器 (Rx LPF): 滤除带外噪声
            # raw_noisy_real = signal.filtfilt(self.b_lpf, self.a_lpf, raw_noisy.real)
            # raw_noisy_imag = signal.filtfilt(self.b_lpf, self.a_lpf, raw_noisy.imag)
            # x_noisy[i] = raw_noisy_real + 1j * raw_noisy_imag

            # ================= [关键修复 2：对齐 Target 滤波尺度] =================
            # 确保网络的监督信号 (Clean Target) 也经过同样的 Rx LPF。
            # 这样网络只负责信道去噪，而不负责带外重建，保证互相关波形的一致性。
            # clean_real = signal.filtfilt(self.b_lpf, self.a_lpf, x_clean[i].real)
            # clean_imag = signal.filtfilt(self.b_lpf, self.a_lpf, x_clean[i].imag)
            # x_clean[i] = clean_real + 1j * clean_imag
            # ==================================================================

        # 转换为 PyTorch 张量 (将实部虚部拆分为两个通道)
        X_in = torch.stack([torch.tensor(x_noisy.real), torch.tensor(x_noisy.imag)], dim=1).float()
        Y_target = torch.stack([torch.tensor(x_clean.real), torch.tensor(x_clean.imag)], dim=1).float()

        return X_in, Y_target, u_t, true_delays

    def generate_pair_batch(self, batch_size, snr_db=None):
        """
        [新增] 专门用于 TDOA 评估的双节点数据生成器
        生成两个 UAV 接收到的、同源但经历了独立多径和独立噪声的信号。
        """
        # 1. 生成共同的发射源信号 (同源)
        num_symbols = self.signal_len // self.samples_per_symbol
        bits = np.random.choice([-1, 1], size=(batch_size, num_symbols))
        base_signal = np.repeat(bits, self.samples_per_symbol, axis=1)

        if base_signal.shape[1] < self.signal_len:
            pad = np.zeros((batch_size, self.signal_len - base_signal.shape[1]))
            base_signal = np.hstack([base_signal, pad])
        else:
            base_signal = base_signal[:, :self.signal_len]

        base_signal = signal.filtfilt(self.b_lpf, self.a_lpf, base_signal, axis=1)
        u_t = base_signal + 1j * np.zeros_like(base_signal)

        # 内部闭包函数：模拟单一 UAV 的物理接收全过程
        def apply_channel_and_noise(u_t_batch):
            X_out = np.zeros_like(u_t_batch)
            delays = []
            snrs = np.random.uniform(-5, 15, size=batch_size) if snr_db is None else np.ones(batch_size) * snr_db

            for i in range(batch_size):
                # 确保延迟在训练分布内 [10, 50)
                main_delay = np.random.randint(10, 50)
                delays.append(main_delay)

                h = np.zeros(self.signal_len, dtype=complex)
                h[main_delay] = 1.0
                for _ in range(np.random.randint(2, 4)):
                    tap_delay = np.random.randint(main_delay + 5, min(main_delay + 100, self.signal_len))
                    h[tap_delay] = np.random.uniform(0.1, 0.4) * np.exp(1j * np.random.uniform(0, 2 * np.pi))

                # 线性卷积 + 截断 (物理传播)
                full_conv = signal.convolve(u_t_batch[i], h, mode='full')
                x_clean = full_conv[:self.signal_len]

                # 加噪声
                sig_p = np.mean(np.abs(x_clean) ** 2)
                noise_p = sig_p / (10 ** (snrs[i] / 10))
                noise = (np.random.normal(0, 1, self.signal_len) + 1j * np.random.normal(0, 1,
                                                                                         self.signal_len)) * np.sqrt(
                    noise_p / 2)

                raw_noisy = x_clean + noise

                # 接收端低通滤波 (Rx LPF) - 极其关键，保证输入不超纲！
                raw_noisy_real = signal.filtfilt(self.b_lpf, self.a_lpf, raw_noisy.real)
                raw_noisy_imag = signal.filtfilt(self.b_lpf, self.a_lpf, raw_noisy.imag)
                X_out[i] = raw_noisy_real + 1j * raw_noisy_imag

            X_tensor = torch.stack([torch.tensor(X_out.real), torch.tensor(X_out.imag)], dim=1).float()
            return X_tensor, delays

        # 2. 分别生成两个 UAV 的接收数据
        X1, delays1 = apply_channel_and_noise(u_t)
        X2, delays2 = apply_channel_and_noise(u_t)

        return X1, X2, delays1, delays2