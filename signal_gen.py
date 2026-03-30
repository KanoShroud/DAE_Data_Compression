# signal_gen.py
import numpy as np
import torch


class SignalSimulator:
    """
    信号生成器类
    功能：模拟生成经过多径信道传输并受AWGN影响的BPSK信号。
    论文依据：对应论文 Section II System Model 中的信号模型。
    """

    def __init__(self, signal_len=1024):
        self.signal_len = signal_len
        self.samples_per_symbol = 2  # 模拟过采样，每个符号占2个采样点

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

            # 信号通过信道：卷积运算 (频域乘法加速)
            x_clean[i] = np.fft.ifft(np.fft.fft(u_t[i]) * np.fft.fft(h))

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

        # 转换为 PyTorch 张量 (将实部虚部拆分为两个通道)
        X_in = torch.stack([torch.tensor(x_noisy.real), torch.tensor(x_noisy.imag)], dim=1).float()
        Y_target = torch.stack([torch.tensor(x_clean.real), torch.tensor(x_clean.imag)], dim=1).float()

        return X_in, Y_target, u_t, true_delays