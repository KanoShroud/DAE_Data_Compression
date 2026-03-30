import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import signal
# 确保 model.py 在同一目录下
from model import DAE


class ResamplingSignalSimulator:
    def __init__(self, dae_input_len=1024):
        self.dae_len = dae_input_len
        self.fs_org = 10e6  # 原始: 10 MHz
        self.bw_sig = 100e3  # 信号带宽: 100 kHz
        self.fs_sig = 150e3  # 降采样后: 150 kHz

        self.ratio = self.fs_org / self.fs_sig
        self.len_org = int(self.dae_len * self.ratio)
        if self.len_org % 2 != 0: self.len_org += 1

        self.symbol_rate = 50e3

    def generate_batch(self, batch_size, snr_db=20):
        # 1. 生成高采样率信号
        num_symbols = int((self.len_org / self.fs_org) * self.symbol_rate)
        bits = np.random.choice([-1, 1], size=(batch_size, num_symbols))
        sps_high = int(self.fs_org / self.symbol_rate)

        s_org = np.repeat(bits, sps_high, axis=1)
        curr_len = s_org.shape[1]
        if curr_len < self.len_org:
            s_org = np.hstack([s_org, np.zeros((batch_size, self.len_org - curr_len))])
        else:
            s_org = s_org[:, :self.len_org]

        s_org = s_org + 1j * np.zeros_like(s_org)

        # 多径模拟
        s_org_multipath = np.zeros_like(s_org)
        true_delays_high = []
        for i in range(batch_size):
            delay = np.random.randint(100, 500)
            true_delays_high.append(delay)
            freqs = np.fft.fftfreq(self.len_org)
            phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
            s_org_multipath[i] = np.fft.ifft(np.fft.fft(s_org[i]) * phase_shift)

        noise_p = np.mean(np.abs(s_org_multipath) ** 2) / (10 ** (snr_db / 10))
        noise = (np.random.randn(*s_org.shape) + 1j * np.random.randn(*s_org.shape)) * np.sqrt(noise_p / 2)
        s_org = s_org_multipath + noise

        # 2. 频域截取 (滤波+降采样)
        spec_org = np.fft.fft(s_org, axis=1)
        spec_org_shifted = np.fft.fftshift(spec_org, axes=1)

        center = self.len_org // 2
        half_width = self.dae_len // 2
        spec_sig_shifted = spec_org_shifted[:, center - half_width: center + half_width]

        spec_sig = np.fft.ifftshift(spec_sig_shifted, axes=1)
        s_sig = np.fft.ifft(spec_sig, axis=1)

        # 3. 归一化输入 (关键!)
        # 记录缩放因子，虽然DAE不需要知道它，但为了数值稳定性必须做
        self.scale_factor = np.std(s_sig) + 1e-8
        s_sig = s_sig / self.scale_factor

        X_in = torch.stack([torch.tensor(s_sig.real), torch.tensor(s_sig.imag)], dim=1).float()
        Y_target = X_in.clone()

        return {
            's_org': s_org,
            'spec_org': spec_org_shifted,
            's_sig': s_sig,  # 此时已归一化
            'spec_sig': spec_sig_shifted,
            'X_in': X_in,
            'Y_target': Y_target
        }

    def upsample_reconstruct(self, s_sig_rec_tensor):
        # 确保输入是 numpy 格式
        if torch.is_tensor(s_sig_rec_tensor):
            s_sig_rec = s_sig_rec_tensor[:, 0, :].numpy() + 1j * s_sig_rec_tensor[:, 1, :].numpy()
        else:
            s_sig_rec = s_sig_rec_tensor

        spec_sig_rec = np.fft.fft(s_sig_rec, axis=1)
        spec_sig_rec_shifted = np.fft.fftshift(spec_sig_rec, axes=1)

        batch_size = s_sig_rec.shape[0]
        spec_rec_shifted = np.zeros((batch_size, self.len_org), dtype=complex)

        center = self.len_org // 2
        half_width = self.dae_len // 2
        spec_rec_shifted[:, center - half_width: center + half_width] = spec_sig_rec_shifted

        spec_rec = np.fft.ifftshift(spec_rec_shifted, axes=1)
        s_rec = np.fft.ifft(spec_rec, axis=1)

        return s_rec, spec_rec_shifted, spec_sig_rec_shifted


def normalize_for_plot(sig):
    """绘图专用归一化：将信号减均值并除以最大幅值，使其范围在 [-1, 1]"""
    sig = sig - np.mean(sig)
    return sig / (np.max(np.abs(sig)) + 1e-9)


def run_resampling_experiment():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Resampling Experiment on {DEVICE}...")

    sim = ResamplingSignalSimulator()
    model = DAE().to(DEVICE)

    # 增加训练轮数，因为这是一个较难的压缩任务
    print("Training DAE (100 iterations)...")
    opt = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for i in range(100):
        data = sim.generate_batch(64, snr_db=25)  # 训练用高信噪比
        x = data['X_in'].to(DEVICE)
        target = data['Y_target'].to(DEVICE)

        opt.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        opt.step()
        if i % 10 == 0: print(f"Iter {i}, Loss: {loss.item():.6f}")

    print("Generating validation data...")
    model.eval()

    val_data = sim.generate_batch(1, snr_db=20)
    with torch.no_grad():
        x_in = val_data['X_in'].to(DEVICE)
        x_rec_tensor = model(x_in).cpu()

    s_rec, spec_rec, spec_sig_rec = sim.upsample_reconstruct(x_rec_tensor)

    # --- 数据提取与处理 (关键步骤) ---
    # 1. 提取第0个样本并压缩维度 (Squeeze)
    s_org_raw = val_data['s_org'][0]
    s_sig_raw = val_data['s_sig'][0]
    s_rec_raw = s_rec[0]
    s_sig_rec_raw = x_rec_tensor[0, 0, :].numpy() + 1j * x_rec_tensor[0, 1, :].numpy()

    # 2. 统一归一化 (为了绘图美观和频谱对比)
    s_org_norm = normalize_for_plot(s_org_raw)
    s_rec_norm = normalize_for_plot(s_rec_raw)
    s_sig_norm = normalize_for_plot(s_sig_raw)
    s_sig_rec_norm = normalize_for_plot(s_sig_rec_raw)

    # 3. 基于归一化后的信号重新计算频谱 (解决右下角幅度不一致问题)
    spec_org_plot = np.fft.fftshift(np.fft.fft(s_org_norm))
    spec_rec_plot = np.fft.fftshift(np.fft.fft(s_rec_norm))
    spec_sig_plot = np.fft.fftshift(np.fft.fft(s_sig_norm))
    spec_sig_rec_plot = np.fft.fftshift(np.fft.fft(s_sig_rec_norm))

    # --- 绘图 ---
    plt.figure(figsize=(14, 10))
    plt.suptitle("Verification: Resampling & DAE (Normalized View)", fontsize=16)

    # 子图 1: 低速时域
    plt.subplot(2, 2, 1)
    plt.plot(s_sig_norm.real, label='DAE Input', color='gray', alpha=0.5)
    plt.plot(s_sig_rec_norm.real, label='DAE Output', color='red', linestyle='--', linewidth=1.5)
    plt.title(f"1. Low Rate Time Domain (Fs={sim.fs_sig / 1e3:.0f}kHz)")
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图 2: 低速频谱
    plt.subplot(2, 2, 2)
    plt.semilogy(np.abs(spec_sig_plot), label='Input Spec', color='gray', alpha=0.6)
    plt.semilogy(np.abs(spec_sig_rec_plot), label='Output Spec', color='red', linestyle='--', linewidth=1.5)
    plt.title("2. Low Rate Spectrum")
    plt.legend()  # 这里的Legend现在正常了，因为输入是一维数组
    plt.grid(alpha=0.3)

    # 子图 3: 高速时域 (Zoom In)
    plt.subplot(2, 2, 3)
    # 只画前 200 个点，看细节
    zoom = 200
    plt.plot(s_org_norm.real[:zoom], label='Original (Normalized)', color='black', alpha=0.3, linewidth=2)
    plt.plot(s_rec_norm.real[:zoom], label='Restored (Normalized)', color='blue', linestyle='--', linewidth=1.5)
    plt.title(f"3. High Rate Time Domain (Fs={sim.fs_org / 1e6:.0f}MHz, Zoomed)")
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图 4: 高速频谱
    plt.subplot(2, 2, 4)
    f_axis = np.linspace(-sim.fs_org / 2, sim.fs_org / 2, len(spec_org_plot))
    # 降采样绘图以提速
    step = 10
    plt.semilogy(f_axis[::step] / 1e6, np.abs(spec_org_plot)[::step], label='Original', color='black', alpha=0.5)
    plt.semilogy(f_axis[::step] / 1e6, np.abs(spec_rec_plot)[::step], label='Restored', color='blue', linestyle='--')
    plt.title("4. High Rate Spectrum")
    plt.xlabel("Frequency (MHz)")
    # 限制显示范围，重点看中心部分
    plt.xlim(-1, 1)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- 计算 RMSE ---
    # 使用归一化后的信号计算波形相似度
    err_sig = np.sqrt(np.mean(np.abs(s_org_norm - s_rec_norm) ** 2))
    print(f"\n[Metrics] Signal Reconstruction RMSE (Normalized): {err_sig:.6f}")

    # TDOA 计算 (使用归一化信号)
    corr = signal.correlate(s_rec_norm.real, s_org_norm.real, mode='same')
    lags = signal.correlation_lags(len(s_rec_norm), len(s_org_norm), mode='same')
    detected_shift = lags[np.argmax(np.abs(corr))]

    print(f"[Metrics] TDOA Alignment Error: {detected_shift} samples")


if __name__ == "__main__":
    run_resampling_experiment()