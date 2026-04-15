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

    def __init__(self, models_dict, simulator, device):
        # models_dict: {16: model_cr16, 8: model_cr8, ...}
        self.models_dict = models_dict
        self.sim = simulator
        self.device = device
        self.snr_range = np.arange(-10, 22, 2)
        self.num_trials = 500

    def run(self):
        print("Running Monte Carlo Sweep across multiple CRs...")
        # 结果容器
        results = {'raw': []}
        for cr in self.models_dict.keys():
            results[f'dae_{cr}'] = []
            self.models_dict[cr].eval()

        for snr in self.snr_range:
            # 1. 直接获取孪生双节点数据
            X1, X2, delays1, delays2 = self.sim.generate_pair_batch(self.num_trials, snr_db=snr)

            X1_dev = X1.to(self.device)
            X2_dev = X2.to(self.device)
            raw_np1 = X1.cpu().numpy()
            raw_np2 = X2.cpu().numpy()

            # 2. 计算 Baseline (Raw) TDOA
            se_raw = 0.0
            for i in range(self.num_trials):
                sig_raw1 = raw_np1[i, 0, :] + 1j * raw_np1[i, 1, :]
                sig_raw2 = raw_np2[i, 0, :] + 1j * raw_np2[i, 1, :]

                corr_raw = signal.correlate(sig_raw1, sig_raw2, mode='same')
                lags = signal.correlation_lags(len(sig_raw1), len(sig_raw2), mode='same')
                delay_raw = lags[np.argmax(np.abs(corr_raw))]

                # 真实 TDOA 是两个接收端物理延迟的差值
                true_tdoa = delays1[i] - delays2[i]
                se_raw += (delay_raw - true_tdoa) ** 2
            results['raw'].append(np.sqrt(se_raw / self.num_trials))

            # 3. 计算各个 DAE 的重构信号 TDOA
            for cr, model in self.models_dict.items():
                se_dae = 0.0
                with torch.no_grad():
                    # DAE 对两路信道分别进行降噪和特征提取
                    Y1_rec = model(X1_dev).cpu().numpy()
                    Y2_rec = model(X2_dev).cpu().numpy()

                for i in range(self.num_trials):
                    sig_rec1 = Y1_rec[i, 0, :] + 1j * Y1_rec[i, 1, :]
                    sig_rec2 = Y2_rec[i, 0, :] + 1j * Y2_rec[i, 1, :]

                    corr_dae = signal.correlate(sig_rec1, sig_rec2, mode='same')
                    delay_dae = lags[np.argmax(np.abs(corr_dae))]

                    true_tdoa = delays1[i] - delays2[i]
                    se_dae += (delay_dae - true_tdoa) ** 2

                results[f'dae_{cr}'].append(np.sqrt(se_dae / self.num_trials))

        return results, self.snr_range


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

        # 增加 detrend=False，阻止 scipy 偷偷吃掉你的直流分量
        f_n, P_n = signal.periodogram(noisy_complex, fs=Fs, return_onesided=False, scaling='density', detrend=False)
        f_r, P_r = signal.periodogram(recon_complex, fs=Fs, return_onesided=False, scaling='density', detrend=False)

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


def plot_monte_carlo(mc_data):
    results, snr_range = mc_data
    plt.figure(figsize=(10, 6))

    # 颜色和标记配置，严格贴合论文 Fig. 5 的风格
    plt.plot(snr_range, results['raw'], 'b-s', label='Original data', linewidth=1.5)
    if 'dae_4' in results:
        plt.plot(snr_range, results['dae_4'], 'm-*', label='Data with CR=4', linewidth=1.5)
    if 'dae_8' in results:
        plt.plot(snr_range, results['dae_8'], 'g-o', label='Data with CR=8', markerfacecolor='none', linewidth=1.5)
    if 'dae_16' in results:
        plt.plot(snr_range, results['dae_16'], 'r-+', label='Data with CR=16', linewidth=1.5)

    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('TDOA RMSE [Samples]', fontsize=12)
    plt.title('Comparison of localization performance at different CRs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()