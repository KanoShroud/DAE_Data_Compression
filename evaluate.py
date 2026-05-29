# evaluate.py
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal


def gcc_phat(sig1, sig2):
    """
    广义互相关 (相位变换加权) - GCC-PHAT

    PHAT 加权将互功率谱归一化为单位幅度，仅保留相位信息。
    优点：多径环境下锐化相关峰。
    缺点：低 SNR 下噪声频段的相位被过度放大，且与 MSE 训练的 DAE 不兼容
          （DAE 优化幅度重建，PHAT 丢弃幅度信息）。
    """
    n = len(sig1)
    SIG1 = np.fft.fft(sig1, n=2 * n - 1)
    SIG2 = np.fft.fft(sig2, n=2 * n - 1)
    R = SIG1 * np.conj(SIG2)
    epsilon = np.max(np.abs(R)) * 1e-3
    R_weighted = R / (np.abs(R) + epsilon)
    cc = np.fft.fftshift(np.fft.ifft(R_weighted))
    start = (len(cc) - n) // 2
    return cc[start:start + n]


def gcc_standard(sig1, sig2):
    """
    标准广义互相关 (无加权) - Standard GCC

    直接使用互功率谱做逆傅里叶变换，保留幅度和相位信息。
    与 MSE 训练的 DAE 兼容性更好：DAE 优化幅度重建，标准 GCC 利用幅度信息。
    """
    n = len(sig1)
    SIG1 = np.fft.fft(sig1, n=2 * n - 1)
    SIG2 = np.fft.fft(sig2, n=2 * n - 1)
    R = SIG1 * np.conj(SIG2)
    cc = np.fft.fftshift(np.fft.ifft(R))
    start = (len(cc) - n) // 2
    return cc[start:start + n].real

class MonteCarloExperiment:
    """
    蒙特卡洛实验类
    功能：在不同 SNR 下进行多次实验，统计 RMSE 和 Loss。
    """

    def __init__(self, models_dict, simulator, device, seed=None, gcc_method='standard'):
        """
        参数:
            gcc_method: 'standard' (标准 GCC，默认) 或 'phat' (GCC-PHAT)
        """
        self.models_dict = models_dict
        self.sim = simulator
        self.device = device
        self.snr_range = np.arange(-10, 21, 1)
        self.num_trials = 500
        self.seed = seed
        self.gcc_func = gcc_standard if gcc_method == 'standard' else gcc_phat
        self.gcc_method = gcc_method

    def run(self):
        print(f"Running Monte Carlo Sweep (GCC method: {self.gcc_method})...")
        if self.seed is not None:
            np.random.seed(self.seed)
            print(f"  Monte Carlo random seed set to: {self.seed}")

        results = {'raw': [], 'raw_med': []}
        for cr in self.models_dict.keys():
            results[f'dae_{cr}'] = []
            results[f'dae_{cr}_med'] = []
            self.models_dict[cr].eval()

        signal_len = self.sim.signal_len
        lags = signal.correlation_lags(signal_len, signal_len, mode='same')

        for snr in self.snr_range:
            X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2 = \
                self.sim.generate_pair_batch(self.num_trials, snr_db=snr)
            X1_dev = X1_noisy.to(self.device)
            X2_dev = X2_noisy.to(self.device)
            raw_np1 = X1_noisy.cpu().numpy()
            raw_np2 = X2_noisy.cpu().numpy()

            se_raw = 0.0
            se_list_raw = []
            for i in range(self.num_trials):
                sig_raw1 = raw_np1[i, 0, :] + 1j * raw_np1[i, 1, :]
                sig_raw2 = raw_np2[i, 0, :] + 1j * raw_np2[i, 1, :]
                corr_raw = self.gcc_func(sig_raw1, sig_raw2)
                delay_raw = lags[np.argmax(np.abs(corr_raw))]
                true_tdoa = delays1[i] - delays2[i]
                se = (delay_raw - true_tdoa) ** 2
                se_raw += se
                se_list_raw.append(se)
            results['raw'].append(np.sqrt(se_raw / self.num_trials))
            results['raw_med'].append(np.sqrt(np.median(se_list_raw)))

            for cr, model in self.models_dict.items():
                se_dae = 0.0
                se_list_dae = []
                with torch.no_grad():
                    Y1_rec = model(X1_dev).cpu().numpy()
                    Y2_rec = model(X2_dev).cpu().numpy()
                for i in range(self.num_trials):
                    sig_rec1 = Y1_rec[i, 0, :] + 1j * Y1_rec[i, 1, :]
                    sig_rec2 = Y2_rec[i, 0, :] + 1j * Y2_rec[i, 1, :]
                    corr_dae = self.gcc_func(sig_rec1, sig_rec2)
                    delay_dae = lags[np.argmax(np.abs(corr_dae))]
                    true_tdoa = delays1[i] - delays2[i]
                    se = (delay_dae - true_tdoa) ** 2
                    se_dae += se
                    se_list_dae.append(se)
                results[f'dae_{cr}'].append(np.sqrt(se_dae / self.num_trials))
                results[f'dae_{cr}_med'].append(np.sqrt(np.median(se_list_dae)))

        return results, self.snr_range


# --- 绘图函数 ---

def plot_training_loss(loss_hist):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(loss_hist, linewidth=2)
    plt.title("Figure 1: Training Loss Curve (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    return fig


def plot_snr_comparison(model, sim, device, snr_list=None, cr=None):
    if snr_list is None:
        snr_list = [-10, 0, 10, 20]

    # 采样率 40MHz，符号率 = Fs / sps = 40MHz / 2 = 20MHz
    # RRC 滚降后 RF 带宽 = (1+beta) × 符号率 = (1+0.2) × 20 = 24MHz
    Fs = 40e6
    Ts_us = 1e6 / Fs  # 采样间隔 = 0.025 μs

    print(f"--- Phase 2: Analyzing Specific SNRs {snr_list} ---")
    model.eval()

    num_rows = len(snr_list)
    fig, axes = plt.subplots(num_rows, 3, figsize=(14, 2.1 * num_rows + 0.4))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    cr_str = f" (CR={cr})" if cr is not None else ""
    plt.suptitle(f"Figure 2: Signal Analysis{cr_str}  (Symbol Rate=20 MHz, BW=24 MHz, Fs={Fs / 1e6:.0f} MHz)",
                 fontsize=13, y=0.99)

    cols = ['Time Domain', 'Power Spectral Density', 'Cross-Correlation (X₁ vs X₂)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=10, fontweight='bold')

    for i, snr in enumerate(snr_list):
        X1_n, X1_c, X2_n, X2_c, d1, d2 = sim.generate_pair_batch(1, snr_db=snr)
        X, Y = X1_n, X1_c
        with torch.no_grad():
            rec = model(X.to(device)).cpu()

        # 提取完整的复数信号 (I + jQ)
        noisy_complex = X[0, 0, :].numpy() + 1j * X[0, 1, :].numpy()
        clean_complex = Y[0, 0, :].numpy() + 1j * Y[0, 1, :].numpy()
        recon_complex = rec[0, 0, :].numpy() + 1j * rec[0, 1, :].numpy()
        n_samples = len(noisy_complex)

        # 时域 x 轴：采样点 → 微秒
        t_us = np.arange(n_samples) * Ts_us

        # --- 1. 时域图 (线性幅度) ---
        ax_t = axes[i, 0]

        noisy_mag = np.abs(noisy_complex)
        clean_mag = np.abs(clean_complex)
        recon_mag = np.abs(recon_complex)

        ax_t.plot(t_us, noisy_mag, color='lightgray', linewidth=0.5, label='Noisy')
        ax_t.plot(t_us, clean_mag, 'k--', linewidth=0.8, label='Clean')
        ax_t.plot(t_us, recon_mag, 'r', alpha=0.8, linewidth=0.8, label='DAE')
        ax_t.set_ylabel(f"SNR={snr}dB\n|s(t)|", fontsize=9, fontweight='bold')
        ax_t.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax_t.set_xlim(0, t_us[299])
        ax_t.grid(alpha=0.3)
        if i == num_rows - 1:
            ax_t.set_xlabel("Time [μs]", fontsize=10)

        # --- 2. 频域图 (PSD in dB, 双边谱) ---
        ax_f = axes[i, 1]

        f_n, P_n = signal.periodogram(noisy_complex, fs=Fs, return_onesided=False,
                                      scaling='density', detrend=False)
        f_r, P_r = signal.periodogram(recon_complex, fs=Fs, return_onesided=False,
                                      scaling='density', detrend=False)
        f_n = np.fft.fftshift(f_n)
        P_n = np.fft.fftshift(P_n)
        f_r = np.fft.fftshift(f_r)
        P_r = np.fft.fftshift(P_r)

        P_n_db = 10 * np.log10(P_n + 1e-30)
        P_r_db = 10 * np.log10(P_r + 1e-30)

        ax_f.plot(f_n / 1e6, P_n_db, color='lightgray', linewidth=0.5, label='Noisy')
        ax_f.plot(f_r / 1e6, P_r_db, 'r', linewidth=0.8, label='DAE')
        ax_f.legend(loc='upper right', fontsize=7, framealpha=0.8)
        if i == num_rows - 1:
            ax_f.set_xlabel("Frequency [MHz]", fontsize=10)
        ax_f.set_ylabel("PSD [dB/Hz]", fontsize=9)
        ax_f.grid(alpha=0.3)
        ax_f.set_xlim(-20, 20)

        # --- 3. 互相关图 — 双接收机 TDOA 估计效果 ---
        ax_c = axes[i, 2]

        x2_noisy_complex = X2_n[0, 0, :].numpy() + 1j * X2_n[0, 1, :].numpy()
        with torch.no_grad():
            x2_rec = model(X2_n.to(device)).cpu()
        x2_recon_complex = x2_rec[0, 0, :].numpy() + 1j * x2_rec[0, 1, :].numpy()

        corr_n = gcc_standard(noisy_complex, x2_noisy_complex)
        corr_r = gcc_standard(recon_complex, x2_recon_complex)
        lags = signal.correlation_lags(n_samples, n_samples, mode='same')

        corr_n_norm = np.abs(corr_n) / (np.max(np.abs(corr_n)) + 1e-9)
        corr_r_norm = np.abs(corr_r) / (np.max(np.abs(corr_r)) + 1e-9)

        true_tdoa = d1[0] - d2[0]

        ax_c.plot(lags, corr_n_norm, color='gray', alpha=0.5, linewidth=0.8, label='Raw')
        ax_c.plot(lags, corr_r_norm, 'r', linewidth=1.2, label='DAE')
        ax_c.axvline(true_tdoa, color='blue', linestyle='--', linewidth=0.8,
                     label=f'True TDOA={true_tdoa}')
        ax_c.set_xlim(true_tdoa - 100, true_tdoa + 100)
        ax_c.grid(alpha=0.3)
        ax_c.set_ylabel("Norm. GCC", fontsize=9)
        ax_c.legend(loc='upper right', fontsize=7, framealpha=0.8)
        if i == num_rows - 1:
            ax_c.set_xlabel("Lag [samples]", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_monte_carlo(mc_data):
    results, snr_range = mc_data
    has_median = 'raw_med' in results
    ncols = 2 if has_median else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    # 左图: Mean RMSE
    ax = axes[0]
    ax.plot(snr_range, results['raw'], 'b-s', label='Original data', linewidth=1.5)
    if 'dae_4' in results:
        ax.plot(snr_range, results['dae_4'], 'm-*', label='Data with CR=4', linewidth=1.5)
    if 'dae_8' in results:
        ax.plot(snr_range, results['dae_8'], 'g-o', label='Data with CR=8', markerfacecolor='none', linewidth=1.5)
    if 'dae_16' in results:
        ax.plot(snr_range, results['dae_16'], 'r-+', label='Data with CR=16', linewidth=1.5)
    ax.set_xlabel('SNR [dB]', fontsize=12)
    ax.set_ylabel('TDOA RMSE [samples]', fontsize=12)
    ax.set_title('Mean RMSE (sensitive to outliers)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.5)

    # 右图: Median RMSE (对 outlier 鲁棒，展示典型性能)
    if has_median:
        ax = axes[1]
        ax.plot(snr_range, results['raw_med'], 'b-s', label='Original data', linewidth=1.5)
        if 'dae_4_med' in results:
            ax.plot(snr_range, results['dae_4_med'], 'm-*', label='Data with CR=4', linewidth=1.5)
        if 'dae_8_med' in results:
            ax.plot(snr_range, results['dae_8_med'], 'g-o', label='Data with CR=8', markerfacecolor='none', linewidth=1.5)
        if 'dae_16_med' in results:
            ax.plot(snr_range, results['dae_16_med'], 'r-+', label='Data with CR=16', linewidth=1.5)
        ax.set_xlabel('SNR [dB]', fontsize=12)
        ax.set_ylabel('TDOA RMSE [samples]', fontsize=12)
        ax.set_title('Median RMSE (robust, typical performance)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.5)

    fig.suptitle('Figure 3: Comparison of Localization Performance at Different CRs', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig