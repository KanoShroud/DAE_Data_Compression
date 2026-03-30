import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal
import torch.utils.data as data_utils

# --- 复用现有的模块 ---
# 请确保 model.py 和 train.py 在同一目录下
from model import DAE
from train import train_model


# ==========================================
# 1. 高级信号生成器 (修正版：Denoising模式)
# ==========================================
def _generate_rrc_filter(fs, rs, beta, span=6):
    """生成升余弦滚降滤波器 (处理奇点)"""
    global mask_side
    T = 1.0 / rs
    t = np.arange(-span * fs / rs, span * fs / rs + 1) / fs
    h = np.zeros_like(t)

    # 1. t=0
    mask_0 = np.isclose(t, 0)
    h[mask_0] = 1.0 - beta + (4 * beta / np.pi)

    # 2. t = +/- T/4beta
    if beta != 0:
        t_singularity = T / (4 * beta)
        mask_side = np.isclose(np.abs(t), t_singularity)
        if np.any(mask_side):
            val = (beta / np.sqrt(2)) * (
                        (1 + 2 / np.pi) * np.sin(np.pi / 4 / beta) + (1 - 2 / np.pi) * np.cos(np.pi / 4 / beta))
            h[mask_side] = val

    # 3. Normal
    mask_normal = ~(mask_0 | mask_side)
    t_norm = t[mask_normal]
    num = np.sin((1 - beta) * np.pi * t_norm / T) + (4 * beta * t_norm / T) * np.cos(
        (1 + beta) * np.pi * t_norm / T)
    denom = (np.pi * t_norm / T) * (1 - (4 * beta * t_norm / T) ** 2)
    h[mask_normal] = num / denom

    return h / np.sqrt(np.sum(h ** 2))


# class RRCSignalSimulator:
#     def __init__(self, dae_input_len=1024, rolloff=0.25):
#         self.dae_len = dae_input_len
#         self.beta = rolloff
#
#         self.fs_high = 10e6  # 10 MHz
#         self.fs_low = 150e3  # 150 kHz
#
#         # Resample poly parameters: 10M -> 150k
#         # 10000 / 150 = 200 / 3
#         self.up_factor = 3
#         self.down_factor = 200
#
#         # 计算需要的 High Rate 长度
#         self.len_high = int(self.dae_len * self.down_factor / self.up_factor)
#         self.symbol_rate = 50e3
#
#         # 生成 RRC 滤波器
#         self.rrc_taps = _generate_rrc_filter(self.fs_high, self.symbol_rate, self.beta)
#
#     def generate_batch(self, batch_size, snr_db=None):
#         """
#         生成数据
#         关键修改：同时生成 Noisy 输入和 Clean 目标
#         """
#         # --- 1. 生成高采样率 RRC 信号 ---
#         num_symbols = int(self.len_high / self.fs_high * self.symbol_rate)
#         bits = np.random.choice([-1, 1], size=(batch_size, num_symbols))
#
#         sps = int(self.fs_high / self.symbol_rate)
#         symbols_upsampled = np.zeros((batch_size, num_symbols * sps), dtype=complex)
#         symbols_upsampled[:, ::sps] = bits
#
#         # 脉冲成型
#         s_high_clean = np.zeros((batch_size, self.len_high), dtype=complex)
#         for i in range(batch_size):
#             sig = signal.fftconvolve(symbols_upsampled[i], self.rrc_taps, mode='same')
#             if len(sig) < self.len_high:
#                 s_high_clean[i, :len(sig)] = sig
#             else:
#                 s_high_clean[i, :] = sig[:self.len_high]
#
#         # --- 2. 多径信道 ---
#         s_high_multipath = np.zeros_like(s_high_clean)
#         true_delays = []
#         for i in range(batch_size):
#             delay = np.random.randint(50, 200)
#             true_delays.append(delay)
#             freqs = np.fft.fftfreq(self.len_high)
#             phase = np.exp(-1j * 2 * np.pi * freqs * delay)
#             s_high_multipath[i] = np.fft.ifft(np.fft.fft(s_high_clean[i]) * phase)
#
#         # --- 3. 添加噪声 ---
#         if snr_db is None:
#             snrs = np.random.uniform(-5, 25, size=batch_size)
#         else:
#             snrs = np.ones(batch_size) * snr_db
#
#         s_high_noisy = np.zeros_like(s_high_multipath)
#         for i in range(batch_size):
#             p_sig = np.mean(np.abs(s_high_multipath[i]) ** 2)
#             p_noise = p_sig / (10 ** (snrs[i] / 10))
#             noise = (np.random.randn(self.len_high) + 1j * np.random.randn(self.len_high)) * np.sqrt(p_noise / 2)
#             s_high_noisy[i] = s_high_multipath[i] + noise
#
#         # --- 4. 降采样 (生成输入和目标) ---
#         # A. 含噪信号降采样 -> DAE 输入
#         s_low_noisy_real = signal.resample_poly(s_high_noisy.real, self.up_factor, self.down_factor, axis=1)
#         s_low_noisy_imag = signal.resample_poly(s_high_noisy.imag, self.up_factor, self.down_factor, axis=1)
#         s_low_noisy = s_low_noisy_real + 1j * s_low_noisy_imag
#
#         # B. [关键] 无噪信号降采样 -> DAE 训练目标 (Clean Target)
#         s_low_clean_real = signal.resample_poly(s_high_multipath.real, self.up_factor, self.down_factor, axis=1)
#         s_low_clean_imag = signal.resample_poly(s_high_multipath.imag, self.up_factor, self.down_factor, axis=1)
#         s_low_clean = s_low_clean_real + 1j * s_low_clean_imag
#
#         # 截断
#         if s_low_noisy.shape[1] > self.dae_len:
#             s_low_noisy = s_low_noisy[:, :self.dae_len]
#             s_low_clean = s_low_clean[:, :self.dae_len]
#
#         # --- 5. 归一化 ---
#         # 使用含噪信号的 std 进行缩放，模拟接收端只知道含噪信号的情况
#         scale = np.std(s_low_noisy, axis=1, keepdims=True) + 1e-9
#         s_low_noisy = s_low_noisy / scale
#         s_low_clean = s_low_clean / scale  # 目标也要用同样的比例缩放，保持相对幅度
#
#         # 转 Tensor
#         X_in = torch.stack([torch.tensor(s_low_noisy.real), torch.tensor(s_low_noisy.imag)], dim=1).float()
#         Y_target = torch.stack([torch.tensor(s_low_clean.real), torch.tensor(s_low_clean.imag)], dim=1).float()
#
#         return {
#             's_high_noisy': s_high_noisy,
#             's_high_clean_rx': s_high_multipath,  # [新增] 用于测试的真值
#             'X_in': X_in,  # Noisy Input
#             'Y_target': Y_target  # Clean Target
#         }
#
#     def reconstruct_high_rate(self, s_low_rec_tensor):
#         if torch.is_tensor(s_low_rec_tensor):
#             s_rec_low = s_low_rec_tensor[:, 0, :].numpy() + 1j * s_low_rec_tensor[:, 1, :].numpy()
#         else:
#             s_rec_low = s_low_rec_tensor
#
#         s_rec_high_real = signal.resample_poly(s_rec_low.real, self.down_factor, self.up_factor, axis=1)
#         s_rec_high_imag = signal.resample_poly(s_rec_low.imag, self.down_factor, self.up_factor, axis=1)
#         s_rec_high = s_rec_high_real + 1j * s_rec_high_imag
#
#         if s_rec_high.shape[1] > self.len_high:
#             s_rec_high = s_rec_high[:, :self.len_high]
#
#         return s_rec_high


class RRCSignalSimulator:
    def __init__(self, dae_input_len=1024, rolloff=0.25):
        self.dae_len = dae_input_len
        self.beta = rolloff

        self.fs_high = 10e6  # 10 MHz
        self.fs_low = 150e3  # 150 kHz

        # 计算高采样率下的长度
        # 公式: len_high / fs_high = len_low / fs_low
        # 1024 * (10M / 150k) ≈ 68266
        self.len_high = int(self.dae_len * (self.fs_high / self.fs_low))

        self.symbol_rate = 50e3

        # 生成 RRC 滤波器 (保持不变，用于生成原始波形)
        self.rrc_taps = _generate_rrc_filter(self.fs_high, self.symbol_rate, self.beta)

    def generate_batch(self, batch_size, snr_db=None):
        """
        生成数据
        修改点：降采样使用 signal.resample (FFT based) 以消除群延迟
        """
        # --- 1. 生成高采样率 RRC 信号 (保持不变) ---
        num_symbols = int(self.len_high / self.fs_high * self.symbol_rate)
        bits = np.random.choice([-1, 1], size=(batch_size, num_symbols))

        sps = int(self.fs_high / self.symbol_rate)
        symbols_upsampled = np.zeros((batch_size, num_symbols * sps), dtype=complex)
        symbols_upsampled[:, ::sps] = bits

        s_high_clean = np.zeros((batch_size, self.len_high), dtype=complex)
        for i in range(batch_size):
            sig = signal.fftconvolve(symbols_upsampled[i], self.rrc_taps, mode='same')
            if len(sig) < self.len_high:
                s_high_clean[i, :len(sig)] = sig
            else:
                s_high_clean[i, :] = sig[:self.len_high]

        # --- 2. 多径信道 (保持不变) ---
        s_high_multipath = np.zeros_like(s_high_clean)
        for i in range(batch_size):
            delay = np.random.randint(50, 200)
            freqs = np.fft.fftfreq(self.len_high)
            phase = np.exp(-1j * 2 * np.pi * freqs * delay)
            s_high_multipath[i] = np.fft.ifft(np.fft.fft(s_high_clean[i]) * phase)

        # --- 3. 添加噪声 (保持不变) ---
        if snr_db is None:
            snrs = np.random.uniform(-5, 25, size=batch_size)
        else:
            snrs = np.ones(batch_size) * snr_db

        s_high_noisy = np.zeros_like(s_high_multipath)
        for i in range(batch_size):
            p_sig = np.mean(np.abs(s_high_multipath[i]) ** 2)
            p_noise = p_sig / (10 ** (snrs[i] / 10))
            noise = (np.random.randn(self.len_high) + 1j * np.random.randn(self.len_high)) * np.sqrt(p_noise / 2)
            s_high_noisy[i] = s_high_multipath[i] + noise

        # --- 4. 降采样 (关键修改：FFT Resample) ---
        # 使用 scipy.signal.resample 进行频域降采样
        # 它可以直接处理复数，且没有相位延迟
        # 参数: (信号, 目标长度, axis)
        s_low_noisy = signal.resample(s_high_noisy, self.dae_len, axis=1)
        s_low_clean = signal.resample(s_high_multipath, self.dae_len, axis=1)

        # # --- 5. 归一化 (保持不变) ---
        # scale = np.std(s_low_noisy, axis=1, keepdims=True) + 1e-9
        # s_low_noisy = s_low_noisy / scale
        # s_low_clean = s_low_clean / scale

        # --- 5. 归一化 (核心修改！！！) ---
        # 策略：独立标准化 (Standarization)
        # 输入：减均值，除以自身的 std
        mean_noisy = np.mean(s_low_noisy, axis=1, keepdims=True)
        std_noisy = np.std(s_low_noisy, axis=1, keepdims=True) + 1e-9
        s_low_noisy_norm = (s_low_noisy - mean_noisy) / std_noisy

        # 目标：减均值，除以自身的 std (独立于含噪信号)
        # 这样无论 SNR 是多少，Target 永远是标准能量的波形，梯度永远健康。
        mean_clean = np.mean(s_low_clean, axis=1, keepdims=True)
        std_clean = np.std(s_low_clean, axis=1, keepdims=True) + 1e-9
        s_low_clean_norm = (s_low_clean - mean_clean) / std_clean

        # 转 Tensor
        X_in = torch.stack([torch.tensor(s_low_noisy_norm.real), torch.tensor(s_low_noisy_norm.imag)], dim=1).float()
        Y_target = torch.stack([torch.tensor(s_low_clean_norm.real), torch.tensor(s_low_clean_norm.imag)], dim=1).float()

        return {
            's_high_noisy': s_high_noisy,
            's_high_clean_rx': s_high_multipath,
            'X_in': X_in,
            'Y_target': Y_target
        }

    def reconstruct_high_rate(self, s_low_rec_tensor):
        """
        将 DAE 输出恢复到高采样率
        修改点：升采样使用 signal.resample (FFT based)
        """
        if torch.is_tensor(s_low_rec_tensor):
            s_rec_low = s_low_rec_tensor[:, 0, :].numpy() + 1j * s_low_rec_tensor[:, 1, :].numpy()
        else:
            s_rec_low = s_low_rec_tensor

        # 升采样：直接指定变回 len_high
        s_rec_high = signal.resample(s_rec_low, self.len_high, axis=1)

        return s_rec_high

# ==========================================
# 2. 蒙特卡洛实验类 (修正版：Reference Correction)
# ==========================================
class AdvancedMonteCarlo:
    def __init__(self, model, simulator, device):
        self.model = model
        self.sim = simulator
        self.device = device
        self.snr_range = np.arange(-10, 22, 2)
        self.num_trials = 200

    def run(self):
        print(f"Running Monte Carlo (Auto-aligned, Clean Reference)...")
        rmse_tdoa = []
        mse_sig = []

        self.model.eval()

        for snr in self.snr_range:
            acc_mse = 0.0
            lags_buffer = []

            # 生成数据
            data = self.sim.generate_batch(self.num_trials, snr_db=snr)
            x_in = data['X_in'].to(self.device)
            # [关键] 基准必须是无噪信号
            s_high_ref = data['s_high_clean_rx']

            with torch.no_grad():
                x_out = self.model(x_in).cpu()

            s_high_rec = self.sim.reconstruct_high_rate(x_out)

            for i in range(self.num_trials):
                sig_ref = s_high_ref[i]
                sig_rec = s_high_rec[i]

                # 归一化 (消除能量差异)
                sig_ref = (sig_ref - np.mean(sig_ref)) / (np.std(sig_ref) + 1e-9)
                sig_rec = (sig_rec - np.mean(sig_rec)) / (np.std(sig_rec) + 1e-9)

                # 1. 互相关对齐
                corr = signal.correlate(sig_rec.real, sig_ref.real, mode='same')
                lags = signal.correlation_lags(len(sig_rec), len(sig_ref), mode='same')
                detected_lag = lags[np.argmax(np.abs(corr))]

                lags_buffer.append(detected_lag)

                # 2. 波形对齐后计算 MSE
                # 将 rec 移回原位
                if detected_lag > 0:
                    rec_aligned = sig_rec[detected_lag:]
                    ref_aligned = sig_ref[:len(rec_aligned)]
                elif detected_lag < 0:
                    ref_aligned = sig_ref[-detected_lag:]
                    rec_aligned = sig_rec[:len(ref_aligned)]
                else:
                    rec_aligned = sig_rec
                    ref_aligned = sig_ref

                err = np.mean(np.abs(ref_aligned - rec_aligned) ** 2)
                acc_mse += err

            # 统计
            # RMSE_unbiased = Std(lags) -> 反映 Jitter
            rmse_val = np.std(lags_buffer)
            mse_val = acc_mse / self.num_trials

            rmse_tdoa.append(rmse_val)
            mse_sig.append(mse_val)

            print(f"SNR {snr:3d}dB: DelayBias={np.mean(lags_buffer):.2f}, Jitter={rmse_val:.4f}, MSE={mse_val:.4f}")

        return self.snr_range, rmse_tdoa, mse_sig


# ==========================================
# 3. 主执行流程
# ==========================================

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Advanced Experiment on {DEVICE}")

    # 初始化仿真器 (保持 FFT 重采样)
    sim = RRCSignalSimulator(dae_input_len=1024, rolloff=0.25)

    # --- Phase 1: Online Training (关键修复) ---
    print("\n--- Phase 1: Training (Online / Infinite Data) ---")
    model = DAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 策略调整：
    # 减少 Epochs (因为每轮数据都是新的，效率高)
    # 增加 Batches (保证每轮见过足够多的样本)
    EPOCHS = 30
    BATCHES_PER_EPOCH = 50
    BATCH_SIZE = 64

    model.train()

    for ep in range(EPOCHS):
        epoch_loss = 0

        # === 核心差异：在线生成 ===
        # 每一轮循环，都生成全新的数据！
        # 模型永远无法“背题”，必须学会“解题方法”。
        for _ in range(BATCHES_PER_EPOCH):
            # 实时生成一批新数据
            data = sim.generate_batch(BATCH_SIZE, snr_db=None)

            # 获取数据并送入 GPU
            bx = data['X_in'].to(DEVICE)
            by = data['Y_target'].to(DEVICE)

            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)  # Target 是 Clean Signal
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / BATCHES_PER_EPOCH

        # 每 5 轮打印一次
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep + 1}/{EPOCHS}, Loss: {avg_loss:.5f}")

    # --- Phase 2: Monte Carlo Evaluation ---
    print("\n--- Phase 2: Monte Carlo Evaluation ---")
    mc = AdvancedMonteCarlo(model, sim, DEVICE)
    snrs, rmse, mse = mc.run()

    # --- Plotting ---
    plt.figure(figsize=(12, 5))

    # 子图1: TDOA RMSE
    plt.subplot(1, 2, 1)
    plt.plot(snrs, rmse, 'r-o', label='DAE Precision')
    plt.title('TDOA RMSE vs SNR (Online Training)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (Samples)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图2: Waveform MSE
    plt.subplot(1, 2, 2)
    plt.plot(snrs, mse, 'b-s', label='Waveform MSE')
    plt.title('Reconstruction MSE')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

# def main():
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Running Advanced Experiment on {DEVICE}")
#
#     sim = RRCSignalSimulator(dae_input_len=1024, rolloff=0.25)
#
#     # --- Phase 1: Offline Training ---
#     print("\n--- Phase 1: Training (Offline Data Generation) ---")
#     model = DAE().to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()
#
#     EPOCHS = 100
#     BATCH_SIZE = 64
#
#     # 离线生成 5000 条训练数据 (混合 SNR)
#     print("Generating 5000 training samples...")
#     train_data = sim.generate_batch(5000, snr_db=None)
#     train_dataset = data_utils.TensorDataset(train_data['X_in'], train_data['Y_target'])
#     train_loader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     model.train()
#     loss_history = []
#
#     for ep in range(EPOCHS):
#         epoch_loss = 0
#         for bx, by in train_loader:
#             bx, by = bx.to(DEVICE), by.to(DEVICE)
#
#             optimizer.zero_grad()
#             output = model(bx)
#             loss = criterion(output, by)  # Target 是 Clean Signal
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#
#         avg_loss = epoch_loss / len(train_loader)
#         loss_history.append(avg_loss)
#
#         if (ep + 1) % 10 == 0:
#             print(f"Epoch {ep + 1}/{EPOCHS}, Loss: {avg_loss:.5f}")
#
#     # --- Phase 2: Monte Carlo ---
#     print("\n--- Phase 2: Monte Carlo Evaluation ---")
#     mc = AdvancedMonteCarlo(model, sim, DEVICE)
#     snrs, rmse, mse = mc.run()
#
#     # --- Plotting ---
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(snrs, rmse, 'r-o', label='DAE Precision')
#     plt.title('TDOA Jitter vs SNR')
#     plt.xlabel('SNR (dB)')
#     plt.ylabel('RMSE (Samples)')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(snrs, mse, 'b-s', label='Waveform MSE')
#     plt.title('Reconstruction MSE vs SNR')
#     plt.xlabel('SNR (dB)')
#     plt.ylabel('MSE')
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    main()
