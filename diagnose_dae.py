# diagnose_dae.py
# DAE 诊断脚本：对比 DAE 前后的 GCC 互相关峰特性
# 定位 DAE 在低 SNR 下比 Raw 差的根因

import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal

from evaluate import gcc_standard
from signal_gen import SignalSimulator
from model import DAE

# ===================== 配置 =====================
RESULT_DIR = "运行结果/20260605_160032"  # 包含 plot_data.pkl 和模型的目录

# 日志重定向 —— 同时输出到控制台和文件
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_TRIALS = 200  # 每个 SNR 下的试验次数
SNR_LIST = [-10, -8, -6, -4]
CR_LIST = [4, 8, 16]
CHANNEL_MODE = "fixed"
N_FIXED_CHANNELS = 50
# ==========================================


def load_models_from_result(result_dir):
    """从结果目录加载已保存的模型权重。"""
    models = {}
    for cr in CR_LIST:
        model_path = os.path.join(result_dir, f"model_cr{cr}.pt")
        if not os.path.exists(model_path):
            print(f"[Error] 模型文件不存在: {model_path}")
            print(f"  请先运行 main.py 生成模型权重（需更新后的 main.py）")
            sys.exit(1)
        model = DAE(cr=cr).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models[cr] = model
        print(f"[Diag] 已加载 CR={cr} 模型: {model_path}")
    return models


def analyze_peak(corr, lags, true_tdoa):
    """分析 GCC 互相关峰的特性"""
    corr_abs = np.abs(corr)
    peak_idx = np.argmax(corr_abs)
    peak_pos = lags[peak_idx]
    peak_amp = corr_abs[peak_idx]
    peak_pos_error = peak_pos - true_tdoa

    # 半功率宽度（-3dB 宽度）
    half_power = peak_amp / np.sqrt(2)  # -3dB = amplitude * 1/sqrt(2)
    above_half = corr_abs >= half_power
    if np.any(above_half):
        indices = np.where(above_half)[0]
        half_width = (lags[indices[-1]] - lags[indices[0]]) / 2.0
    else:
        half_width = np.nan

    # 次高峰/主峰比值（旁瓣抑制）
    corr_copy = corr_abs.copy()
    # 排除主峰附近 ±10 个采样点
    exclude = np.abs(lags - peak_pos) <= 10
    corr_copy[exclude] = 0
    sidelobe_amp = np.max(corr_copy) if np.any(corr_copy > 0) else 0
    sidelobe_ratio = sidelobe_amp / (peak_amp + 1e-30)

    return {
        'peak_pos': peak_pos,
        'peak_amp': peak_amp,
        'peak_pos_error': peak_pos_error,
        'half_width': half_width,
        'sidelobe_ratio': sidelobe_ratio,
    }


def run_diagnosis(models, sim, device):
    """运行诊断：对比 Raw 和 DAE 的 GCC 互相关峰特性"""
    results = {}
    lags = signal.correlation_lags(sim.signal_len, sim.signal_len, mode='same')

    for snr in SNR_LIST:
        results[snr] = {}

        np.random.seed(SEED)
        X1_n, X1_c, X2_n, X2_c, delays1, delays2 = \
            sim.generate_pair_batch(N_TRIALS, snr_db=snr)

        # Raw GCC
        raw_np1 = X1_n.cpu().numpy()
        raw_np2 = X2_n.cpu().numpy()
        raw_stats = []
        for i in range(N_TRIALS):
            sig1 = raw_np1[i, 0, :] + 1j * raw_np1[i, 1, :]
            sig2 = raw_np2[i, 0, :] + 1j * raw_np2[i, 1, :]
            corr = gcc_standard(sig1, sig2)
            true_tdoa = delays1[i] - delays2[i]
            raw_stats.append(analyze_peak(corr, lags, true_tdoa))
        results[snr]['raw'] = raw_stats

        # DAE GCC (for each CR)
        for cr, model in models.items():
            model.eval()
            X1_dev = X1_n.to(device)
            X2_dev = X2_n.to(device)
            with torch.no_grad():
                Y1 = model(X1_dev).cpu().numpy()
                Y2 = model(X2_dev).cpu().numpy()
            dae_stats = []
            for i in range(N_TRIALS):
                sig1 = Y1[i, 0, :] + 1j * Y1[i, 1, :]
                sig2 = Y2[i, 0, :] + 1j * Y2[i, 1, :]
                corr = gcc_standard(sig1, sig2)
                true_tdoa = delays1[i] - delays2[i]
                dae_stats.append(analyze_peak(corr, lags, true_tdoa))
            results[snr][f'dae_{cr}'] = dae_stats

    return results


def print_summary(results):
    """打印诊断汇总表"""
    metrics = ['peak_pos_error', 'half_width', 'sidelobe_ratio', 'peak_amp']
    labels = ['峰位置误差 (samples)', '半功率宽度 (samples)', '旁瓣/主峰比', '主峰幅度']

    for metric, label in zip(metrics, labels):
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        header = f"{'SNR':>5} | {'Raw':>10}"
        for cr in CR_LIST:
            header += f" | {'CR=' + str(cr):>10}"
        print(header)
        print("-" * len(header))

        for snr in SNR_LIST:
            raw_vals = [s[metric] for s in results[snr]['raw']]
            raw_median = np.median(raw_vals)
            line = f"{snr:5d} | {raw_median:10.2f}"
            for cr in CR_LIST:
                dae_vals = [s[metric] for s in results[snr][f'dae_{cr}']]
                dae_median = np.median(dae_vals)
                line += f" | {dae_median:10.2f}"
            print(line)


def plot_diagnosis(results, save_dir):
    """绘制诊断图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 柱状图定位参数：4 个柱子（Raw + 3 CR）居中排列
    # 每组柱子总宽 = 4 * bar_w + 3 * gap = 4*0.18 + 3*0.04 = 0.84
    bar_w = 0.18
    gap = 0.04
    offsets = [-(1.5 * bar_w + 1.5 * gap),  # Raw
               -(0.5 * bar_w + 0.5 * gap),  # CR=4
                (0.5 * bar_w + 0.5 * gap),  # CR=8
                (1.5 * bar_w + 1.5 * gap)]  # CR=16
    colors = ['blue', 'C1', 'C2', 'C3']
    labels = ['Raw', 'CR=4', 'CR=8', 'CR=16']

    def draw_bars(ax, snr, values_list):
        for i, (val, color, label) in enumerate(zip(values_list, colors, labels)):
            lbl = label if snr == SNR_LIST[0] else ''
            ax.bar(snr + offsets[i], val, width=bar_w, color=color, alpha=0.7, label=lbl)

    # 1. Peak position error (absolute value)
    ax = axes[0, 0]
    for snr in SNR_LIST:
        vals = [np.median([abs(s['peak_pos_error']) for s in results[snr]['raw']])]
        vals += [np.median([abs(s['peak_pos_error']) for s in results[snr][f'dae_{cr}']]) for cr in CR_LIST]
        draw_bars(ax, snr, vals)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('|Peak Position Error| [samples]')
    ax.set_title('Peak Position Error (lower is better)')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Half-power width
    ax = axes[0, 1]
    for snr in SNR_LIST:
        raw_hw = [s['half_width'] for s in results[snr]['raw'] if not np.isnan(s['half_width'])]
        vals = [np.median(raw_hw)]
        vals += [np.median([s['half_width'] for s in results[snr][f'dae_{cr}'] if not np.isnan(s['half_width'])]) for cr in CR_LIST]
        draw_bars(ax, snr, vals)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Half-Power Width [samples]')
    ax.set_title('Peak Width (lower = sharper)')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Sidelobe ratio
    ax = axes[1, 0]
    for snr in SNR_LIST:
        vals = [np.median([s['sidelobe_ratio'] for s in results[snr]['raw']])]
        vals += [np.median([s['sidelobe_ratio'] for s in results[snr][f'dae_{cr}']]) for cr in CR_LIST]
        draw_bars(ax, snr, vals)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Sidelobe / Main Peak Ratio')
    ax.set_title('Sidelobe Suppression (lower is better)')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Peak amplitude
    ax = axes[1, 1]
    for snr in SNR_LIST:
        vals = [np.median([s['peak_amp'] for s in results[snr]['raw']])]
        vals += [np.median([s['peak_amp'] for s in results[snr][f'dae_{cr}']]) for cr in CR_LIST]
        draw_bars(ax, snr, vals)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Peak Amplitude')
    ax.set_title('Peak Amplitude (higher = stronger correlation)')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle('DAE Diagnosis: GCC Cross-Correlation Peak Analysis', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(save_dir, "Diag_GCC_Peak_Analysis.svg")
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"[Saved] {save_path}")
    plt.show(block=False)
    plt.pause(0.1)

    return fig


if __name__ == '__main__':
    # 设置日志
    log_path = os.path.join(RESULT_DIR, "diagnosis.log")
    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print("=" * 60)
    print("DAE Diagnosis: GCC Cross-Correlation Peak Analysis")
    print("=" * 60)

    # 加载模型
    models = load_models_from_result(RESULT_DIR)

    # 创建 Simulator
    sim = SignalSimulator(channel_mode=CHANNEL_MODE,
                          n_fixed_channels=N_FIXED_CHANNELS,
                          channel_pool_seed=SEED)

    # 运行诊断
    results = run_diagnosis(models, sim, DEVICE)

    # 打印汇总
    print_summary(results)

    # 绘制诊断图
    os.makedirs(RESULT_DIR, exist_ok=True)
    plot_diagnosis(results, RESULT_DIR)

    print(f"\n[Log] 诊断日志已保存至: {log_path}")
    print("诊断完成。")
    plt.show()

    # 恢复 stdout/stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()
