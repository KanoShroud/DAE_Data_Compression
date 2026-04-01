# evaluate.py
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal


class MonteCarloExperiment:
    """
    蒙特卡洛实验类
    功能：在不同 SNR 下进行多次实验，统计 RMSE 和 Loss。
    """

    def __init__(self, model, simulator, device):
        self.model = model
        self.sim = simulator
        self.device = device
        # 论文设定：SNR范围 -10dB 到 20dB，步长 2dB
        self.snr_range = np.arange(-10, 22, 2)
        # 论文设定：每个点 200 次独立实验
        self.num_trials = 200

    def run(self):
        print(f"Running Monte Carlo Sweep (SNR -10 to 20dB)...")
        tdoa_rmse_dae = []
        tdoa_rmse_raw = []
        recon_mse = []

        self.model.eval()
        for snr in self.snr_range:
            mse_acc = 0.0
            se_dae = 0.0
            se_raw = 0.0

            # 生成测试数据
            X, Y, src, delays = self.sim.generate_batch(self.num_trials, snr_db=snr)
            X_dev, Y_dev = X.to(self.device), Y.to(self.device)

            with torch.no_grad():
                Y_rec = self.model(X_dev)
                # 1. 记录信号恢复误差 (Reconstruction MSE)
                mse_acc = nn.functional.mse_loss(Y_rec, Y_dev).item()

                # 转回 CPU 进行 SciPy 互相关计算
                rec_np = Y_rec.cpu().numpy()
                raw_np = X.cpu().numpy()

                # 2. 计算 TDOA 误差 (复包络互相关)
                for i in range(self.num_trials):
                    # 严谨的基带处理：将双通道张量还原为复数信号 (I + jQ)
                    sig_rec = rec_np[i, 0, :] + 1j * rec_np[i, 1, :]
                    sig_raw = raw_np[i, 0, :] + 1j * raw_np[i, 1, :]
                    sig_src = src[i]  # signal_gen 中生成的 src 已经是复数

                    true_d = delays[i]

                    # --- DAE 方法 ---
                    # 执行复数互相关
                    corr_dae = signal.correlate(sig_rec, sig_src, mode='same')
                    lags = signal.correlation_lags(len(sig_rec), len(sig_src), mode='same')
                    # TDOA 估计的物理本质：寻找复互相关输出包络（模值）的峰值
                    delay_dae = lags[np.argmax(np.abs(corr_dae))]
                    se_dae += (delay_dae - true_d) ** 2

                    # --- 原始对比组 (Baseline) ---
                    corr_raw = signal.correlate(sig_raw, sig_src, mode='same')
                    delay_raw = lags[np.argmax(np.abs(corr_raw))]
                    se_raw += (delay_raw - true_d) ** 2

            # 计算 RMSE
            rmse_dae = np.sqrt(se_dae / self.num_trials)
            rmse_raw = np.sqrt(se_raw / self.num_trials)

            tdoa_rmse_dae.append(rmse_dae)
            tdoa_rmse_raw.append(rmse_raw)
            recon_mse.append(mse_acc)

        return tdoa_rmse_dae, tdoa_rmse_raw, recon_mse


# --- 绘图函数 ---

def plot_training_loss(loss_hist):
    plt.figure(1, figsize=(6, 4))
    plt.plot(loss_hist, linewidth=2)
    plt.title("Figure 1: Training Loss Curve (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)


def plot_snr_comparison(model, sim, device, snr_list=None):
    if snr_list is None:
        snr_list = [-10, 0, 10, 20]

    # 定义新的采样率 80MHz
    # 这样：40MHz / 2(samples_per_symbol) = 20MHz 符号率(带宽)
    Fs = 40e6

    print(f"--- Phase 2: Analyzing Specific SNRs {snr_list} ---")
    model.eval()

    num_rows = len(snr_list)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 2 * num_rows), num=2)
    plt.suptitle(f"Figure 2: Signal Analysis (Bandwidth=20MHz, Fs={Fs / 1e6:.0f}MHz)", fontsize=16)

    # 修改点：更新列标题以反映物理意义的改变
    cols = ['Time Domain (Magnitude)', 'Frequency Domain (Double-Sided)', 'Cross-Correlation (Envelope)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold')

    for i, snr in enumerate(snr_list):
        X, Y, src, delays = sim.generate_batch(1, snr_db=snr)
        with torch.no_grad():
            rec = model(X.to(device)).cpu()

        # 提取完整的复数信号 (I + jQ)
        noisy_complex = X[0, 0, :].numpy() + 1j * X[0, 1, :].numpy()
        clean_complex = Y[0, 0, :].numpy() + 1j * Y[0, 1, :].numpy()  # 补充提取干净信号的复数形式
        recon_complex = rec[0, 0, :].numpy() + 1j * rec[0, 1, :].numpy()

        source_complex = src[0]  # 使用原本就是复数的源信号
        true_d = delays[0]

        # 1. 时域图 (改为绘制信号模值 Magnitude)
        ax_t = axes[i, 0]

        noisy_mag = np.abs(noisy_complex)
        clean_mag = np.abs(clean_complex)
        recon_mag = np.abs(recon_complex)

        ax_t.plot(noisy_mag, color='lightgray', label='Noisy (Mag)')
        ax_t.plot(clean_mag, 'k--', linewidth=1, label='Clean (Mag)')
        ax_t.plot(recon_mag, 'r', alpha=0.8, linewidth=1, label='DAE (Mag)')
        ax_t.set_ylabel(f"SNR = {snr} dB", fontsize=12, fontweight='bold')
        if i == 0: ax_t.legend(loc='upper right', fontsize=8)
        ax_t.set_xlim(0, 300)
        ax_t.grid(alpha=0.3)

        # 2. 频域图 (保持不变，计算双边谱)
        ax_f = axes[i, 1]

        f_n, P_n = signal.periodogram(noisy_complex, fs=Fs, return_onesided=False, scaling='density')
        f_r, P_r = signal.periodogram(recon_complex, fs=Fs, return_onesided=False, scaling='density')

        f_n = np.fft.fftshift(f_n)
        P_n = np.fft.fftshift(P_n)
        f_r = np.fft.fftshift(f_r)
        P_r = np.fft.fftshift(P_r)

        ax_f.semilogy(f_n / 1e6, P_n, color='lightgray', label='Noisy')
        ax_f.semilogy(f_r / 1e6, P_r, 'r', linewidth=1, label='DAE')

        if i == num_rows - 1: ax_f.set_xlabel("Freq (MHz)")
        ax_f.grid(alpha=0.3)
        ax_f.set_xlim(-20, 20)

        # 3. 互相关图 (改为复包络互相关)
        ax_c = axes[i, 2]

        # 对复数信号执行互相关
        corr_n = signal.correlate(noisy_complex, source_complex, mode='same')
        corr_r = signal.correlate(recon_complex, source_complex, mode='same')
        lags = signal.correlation_lags(len(noisy_complex), len(source_complex), mode='same')

        # 取互相关结果的模值（包络）并归一化
        corr_n_abs = np.abs(corr_n)
        corr_r_abs = np.abs(corr_r)

        corr_n_norm = corr_n_abs / (np.max(corr_n_abs) + 1e-9)
        corr_r_norm = corr_r_abs / (np.max(corr_r_abs) + 1e-9)

        ax_c.plot(lags, corr_n_norm, color='gray', alpha=0.5, label='Noisy')
        ax_c.plot(lags, corr_r_norm, 'r', linewidth=1.5, label='DAE')
        ax_c.axvline(true_d, color='blue', linestyle='--', label='True Delay')
        ax_c.set_xlim(true_d - 100, true_d + 100)
        ax_c.grid(alpha=0.3)
        if i == 0: ax_c.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)


def plot_monte_carlo(exp_results):
    tdoa_dae, tdoa_raw, recon_mse = exp_results
    snr_range = np.arange(-10, 22, 2)

    plt.figure(3, figsize=(12, 5))
    plt.suptitle("Figure 3: Monte Carlo Experiment Results (200 Trials)", fontsize=16)

    # 1. TDOA RMSE 曲线
    plt.subplot(1, 2, 1)
    plt.plot(snr_range, tdoa_raw, 'k--o', label='Baseline (Raw)', markersize=5)
    plt.plot(snr_range, tdoa_dae, 'r-s', label='Proposed (DAE)', markersize=5)
    plt.xlabel('SNR [dB]')
    plt.ylabel('TDOA RMSE [Samples]')
    plt.title('TDOA Estimation Error vs SNR')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.xticks(np.arange(-10, 22, 5))

    # 2. MSE 曲线
    plt.subplot(1, 2, 2)
    plt.plot(snr_range, recon_mse, 'b-^', label='Reconstruction MSE')
    plt.xlabel('SNR [dB]')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title('Signal Reconstruction Quality')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()

    plt.tight_layout()