# main.py

import os
import sys
import time
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # 非阻塞显示后端
import matplotlib.pyplot as plt
from datetime import datetime

from train import train_with_cv
import pickle
from evaluate import MonteCarloExperiment, plot_monte_carlo, plot_snr_comparison, plot_snr_comparison_multi, generate_snr_data, generate_snr_data_all
from signal_gen import SignalSimulator

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CR_LIST = [4, 8, 16]
N_SAMPLES = 50000        # 训练样本数
K_FOLDS = 3              # 论文k=5, 但随机信道下CV方差大, 3折已足够且节省40%时间
MAX_EPOCHS = 150         # 最大训练轮数（早停可提前结束）
BATCH_SIZE = 128
LR = 0.0005
SEED = 42
PATIENCE = 10            # 早停耐心值
WEIGHT_DECAY = 1e-4      # L2 正则化系数
LAMBDA_CORR = 0.3        # GCC 互相关损失权重（0=纯 MSE，>0 启用相关性正则化）
LAMBDA_PEAK = 0.25       # PNCC 峰值损失最大权重（自适应：低SNR→0，高SNR→0.25）
BETA_FI = 0.0             # Fisher 损失已归档（GCC损失覆盖其功能）

# ========== Per-CR 损失配置 (容量分配理论 v2) ==========
# 统一GCC-centric公式: L = 1.0·L_corr + λ_peak·L_Peak + ε·NMSE
# Corr=1.0恒定的原因：TDOA是所有CR的最终目标（瓶颈独立性，128维足够）
# ε由"额外维度比例"决定: ε = (dim_latent - 128) / dim_latent × 1.0
#   CR=4 (512维):  ε=0.75, λp=0.15 → 75%额外容量去重建, Corr=1.0基准
#   CR=8 (256维):  ε=0.50, λp=0.20 → 50%额外容量去重建, Corr=1.0基准
#   CR=16 (128维): ε=0.03, λp=0.25 → 无额外容量, 纯TDOA, NMSE仅防退化
LOSS_CONFIG = {
    4:  {'epsilon_mse': 0.75, 'lambda_peak': 0.15},
    8:  {'epsilon_mse': 0.50, 'lambda_peak': 0.20},
    16: {'epsilon_mse': 0.03, 'lambda_peak': 0.25},
}

# 自适应 Peak 配置（方案1）
USE_ADAPTIVE_LAMBDA = True   # 是否启用逐样本SNR估计（用于自适应peak权重）
SNR_THRESHOLD = 0.0          # sigmoid 中心点 SNR (dB)
LAMBDA_TEMPERATURE = 5.0     # sigmoid 温度参数

# 多 seed 评估 —— 用不同 seed 训练模型，验证结果泛化性
# 设为 [SEED] 则只跑单 seed（快速）；设为 [42, 123, 456] 则跑 3 个 seed
SEED_LIST = [SEED]

# 如需快速验证（5~10分钟），可将 K_FOLDS 降至 1 并设 USE_SPLIT=True
# 此时仅做单次 80/20 train/val 划分，跳过全量 CV
USE_SPLIT = False        # True: 单次划分快速模式; False: K-fold CV

# 信道模式 —— "fixed" 从固定信道池采样（匹配论文 50 snapshots）；"random" 每样本随机信道
CHANNEL_MODE = "fixed"         # "fixed" 或 "random"
N_FIXED_CHANNELS = 50          # 固定信道快照数（论文 50 snapshots）
CHANNEL_POOL_SEED = 42         # 信道池生成种子

# 全局随机种子 —— 确保训练与评估完全可复现
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 输出目录 —— 每次运行创建独立子文件夹，避免结果互相覆盖
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "运行结果", TIMESTAMP)
os.makedirs(RESULT_DIR, exist_ok=True)

# 日志重定向 —— 同时输出到控制台和文件
class Tee:
    """将 stdout 同时输出到终端和文件"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

log_path = os.path.join(RESULT_DIR, "run.log")
log_file = open(log_path, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)
print(f"[Log] {log_path}")

# ===================== 工具函数 =====================

def save_figure(fig, filename_stem):
    """
    保存图片为 SVG 矢量图格式。

    SVG 为矢量图形，可无损缩放，适合论文插图。
    图形窗口在程序末尾统一弹出，避免重复显示。
    """
    filepath = os.path.join(RESULT_DIR, f"{filename_stem}.svg")
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"[Saved] {filepath}")

# ===================== 主流程 =====================

t_total_start = time.time()
all_seed_results = {}  # {seed: mc_results} 用于多 seed 汇总

for seed_run_idx, current_seed in enumerate(SEED_LIST):
    is_last_seed = (seed_run_idx == len(SEED_LIST) - 1)
    if len(SEED_LIST) > 1:
        print(f"\n{'#'*60}")
        print(f"# Multi-Seed Run {seed_run_idx+1}/{len(SEED_LIST)}: SEED={current_seed}")
        print(f"{'#'*60}")

    # 用当前 seed 重置随机状态
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)

    models_dict = {}
    cv_results_dict = {}

    # 1. 实例化 Simulator
    sim = SignalSimulator(channel_mode=CHANNEL_MODE,
                          n_fixed_channels=N_FIXED_CHANNELS,
                          channel_pool_seed=CHANNEL_POOL_SEED)

    # 2. 使用 k-fold 交叉验证训练各个压缩率下的网络
    for cr in CR_LIST:
        model, cv_results = train_with_cv(
            DEVICE, cr=cr, k=K_FOLDS, n_samples=N_SAMPLES,
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=current_seed,
            patience=PATIENCE, weight_decay=WEIGHT_DECAY, use_split=USE_SPLIT,
            channel_mode=CHANNEL_MODE, n_fixed_channels=N_FIXED_CHANNELS,
            channel_pool_seed=CHANNEL_POOL_SEED, sim=sim,
            lambda_corr=LAMBDA_CORR, lambda_peak=LAMBDA_PEAK, beta_fi=BETA_FI,
            epsilon_mse=0.03,
            use_adaptive_lambda=USE_ADAPTIVE_LAMBDA,
            snr_threshold=SNR_THRESHOLD, lambda_temperature=LAMBDA_TEMPERATURE,
            loss_config=LOSS_CONFIG[cr],
        )
        models_dict[cr] = model
        cv_results_dict[cr] = cv_results

    # 3. Monte Carlo 评估
    print("\n" + "=" * 40)
    exp = MonteCarloExperiment(models_dict, sim, DEVICE, seed=current_seed)
    mc_results = exp.run()
    all_seed_results[current_seed] = mc_results

    # 4. 最后一个 seed：生成图片和保存数据
    if is_last_seed:
        # ===== Figure 1: 多分量训练动态 (3行×3列面板) =====
        n_cr = len(CR_LIST)
        fig_cv, axes_cv = plt.subplots(3, n_cr, figsize=(5 * n_cr, 8.5))
        if n_cr == 1:
            axes_cv = axes_cv[:, np.newaxis]  # 保证 2D 索引

        has_corr = LAMBDA_CORR > 0
        has_peak = LAMBDA_PEAK > 0

        for col, cr in enumerate(CR_LIST):
            cr_data = cv_results_dict[cr]
            n_folds = len(cr_data['fold_train_loss'])

            # --- Row 1: MSE Loss ---
            ax_mse = axes_cv[0, col]
            for fi in range(n_folds):
                lbl_t = f'Train (F{fi+1})' if n_folds > 1 else 'Train'
                lbl_v = f'Val (F{fi+1})' if n_folds > 1 else 'Val'
                ax_mse.plot(cr_data['fold_train_loss'][fi], alpha=0.4,
                            color=f'C{fi}', linewidth=0.8, label=lbl_t)
                ax_mse.plot(cr_data['fold_val_loss'][fi], alpha=0.7,
                            color=f'C{fi}', linewidth=1.2, linestyle='--', label=lbl_v)
            ax_mse.set_title(f'CR={cr} | NMSE (ε={LOSS_CONFIG[cr]["epsilon_mse"]})', fontsize=10, fontweight='bold')
            ax_mse.set_ylabel('NMSE')
            ax_mse.grid(True, alpha=0.3)
            ax_mse.legend(fontsize=6, framealpha=0.8, ncol=2)

            # --- Row 2: GCC Correlation Loss ---
            ax_corr = axes_cv[1, col]
            if has_corr and 'fold_corr_loss' in cr_data:
                for fi in range(n_folds):
                    ax_corr.plot(cr_data['fold_corr_loss'][fi], alpha=0.5,
                                 color=f'C{fi}', linewidth=1.0,
                                 label=f'F{fi+1}' if n_folds > 1 else 'Corr')
                ax_corr.set_title(f'CR={cr} | GCC Correlation Loss', fontsize=10, fontweight='bold')
                ax_corr.legend(fontsize=6, framealpha=0.8)
            else:
                ax_corr.text(0.5, 0.5, 'N/A (pure MSE)', ha='center', va='center',
                             transform=ax_corr.transAxes, color='gray')
                ax_corr.set_title(f'CR={cr} | GCC Correlation Loss', fontsize=10, fontweight='bold')
            ax_corr.set_ylabel('Corr Loss')
            ax_corr.grid(True, alpha=0.3)

        # --- Row 3: Avg SNR estimate + Peak Loss (all CRs overlaid) ---
        ax_snr = axes_cv[2, 0]
        ax_peak = axes_cv[2, 1] if n_cr >= 2 else axes_cv[2, 0]
        for col, cr in enumerate(CR_LIST):
            cr_data = cv_results_dict[cr]
            n_folds = len(cr_data.get('fold_avg_snr', []))
            if has_corr and n_folds > 0:
                for fi in range(n_folds):
                    ax_snr.plot(cr_data['fold_avg_snr'][fi], alpha=0.5,
                                color=f'C{col}', linewidth=0.8,
                                label=f'CR={cr} F{fi+1}' if fi == 0 else f'_F{fi+1}')
            if has_peak and 'fold_peak_loss' in cr_data:
                for fi in range(n_folds):
                    ax_peak.plot(cr_data['fold_peak_loss'][fi], alpha=0.5,
                                 color=f'C{col}', linewidth=1.0,
                                 label=f'CR={cr} F{fi+1}' if fi == 0 else f'_F{fi+1}')

        ax_snr.set_title('Avg SNR Estimate [dB] (training batch)', fontsize=10, fontweight='bold')
        ax_snr.set_xlabel('Epoch'); ax_snr.set_ylabel('SNR [dB]')
        ax_snr.grid(True, alpha=0.3)
        ax_snr.axhline(y=0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        if has_corr:
            ax_snr.legend(fontsize=6, framealpha=0.8)

        if n_cr >= 2:
            ax_peak.set_title('PNCC Peak Loss', fontsize=10, fontweight='bold')
            ax_peak.set_xlabel('Epoch'); ax_peak.set_ylabel('Peak Loss')
            ax_peak.grid(True, alpha=0.3)
            if has_peak:
                ax_peak.legend(fontsize=6, framealpha=0.8)
        else:
            # n_cr=1 时 ax_peak 与 ax_lambda 共享同一轴
            pass

        # 隐藏 Row 3 中未使用的列
        # ax_lambda 使用 [2,0], ax_peak 使用 [2,1]，多余列需隐藏
        for col in range(2, axes_cv.shape[1]):
            axes_cv[2, col].set_visible(False)

        cfg_str = ", ".join([f"CR{cr}: ε={LOSS_CONFIG[cr]['epsilon_mse']}, λp={LOSS_CONFIG[cr]['lambda_peak']}" for cr in CR_LIST])
        cv_title = (f'Figure 1: {K_FOLDS}-Fold CV Training Dynamics  ({cfg_str})')
        fig_cv.suptitle(cv_title, fontsize=12, y=0.995)
        fig_cv.tight_layout()
        save_figure(fig_cv, "Fig1_CV_training_curves")

        # 生成 Figure 2 的绘图数据（所有CR叠加）
        snr_data_all = generate_snr_data_all(models_dict, sim, DEVICE)

        # 保存模型权重（供 diagnose_dae.py 使用）
        for cr, model in models_dict.items():
            model_path = os.path.join(RESULT_DIR, f"model_cr{cr}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"[Saved] {model_path}")

        # 保存绘图数据（含运行配置，供 replot.py / 离线分析）
        plot_data = {
            'cv_results_dict': cv_results_dict,
            'snr_data': snr_data_all,
            'snr_cr': CR_LIST,
            'mc_results': mc_results,
            'config': {
                'n_samples': N_SAMPLES,
                'batch_size': BATCH_SIZE,
                'k_folds': K_FOLDS,
                'max_epochs': MAX_EPOCHS,
                'lr': LR,
                'seed': current_seed,
                'seed_list': SEED_LIST,
                'patience': PATIENCE,
                'weight_decay': WEIGHT_DECAY,
                'lambda_corr': LAMBDA_CORR,
                'lambda_peak': LAMBDA_PEAK,
                'use_adaptive_lambda': USE_ADAPTIVE_LAMBDA,
                'snr_threshold': SNR_THRESHOLD,
                'lambda_temperature': LAMBDA_TEMPERATURE,
                'channel_mode': CHANNEL_MODE,
                'n_fixed_channels': N_FIXED_CHANNELS,
                'cr_list': CR_LIST,
                'loss_config': LOSS_CONFIG,
            },
        }
        pkl_path = os.path.join(RESULT_DIR, "plot_data.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f"[Saved] {pkl_path}")

        # 绘制 Figure 2 和 3
        fig_mc = plot_monte_carlo(mc_results)
        save_figure(fig_mc, "Fig3_MonteCarlo_TDOA_RMSE")

        fig_snr = plot_snr_comparison_multi(cr_list=CR_LIST, data_dict=snr_data_all)
        save_figure(fig_snr, "Fig2_SNR_Comparison")

# ===================== 多 seed 汇总 =====================
if len(SEED_LIST) > 1:
    print(f"\n{'='*60}")
    print(f"Multi-Seed Evaluation Summary ({len(SEED_LIST)} seeds)")
    print(f"{'='*60}")
    for cr in CR_LIST:
        key = f'dae_{cr}'
        key_med = f'dae_{cr}_med'
        # 取 SNR=0 dB 的结果做对比（索引 10，对应 snr=-10+10=0）
        snr_idx = 10
        vals = []
        for s in SEED_LIST:
            r = all_seed_results[s][0]
            vals.append(r[key_med][snr_idx])
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"  CR={cr} Median RMSE @ SNR=0dB: {mean_val:.2f} ± {std_val:.2f}")

# ===================== 运行总结 =====================
t_total = time.time() - t_total_start
print(f"\n{'='*60}")
print(f"总运行耗时: {t_total:.1f}s ({t_total/60:.1f}min)")
print(f"所有图片已保存至: {RESULT_DIR}")
print(f"时间戳: {TIMESTAMP}")
print(f"{'='*60}")

# 阻塞等待用户关闭所有图片窗口后退出
if plt.get_fignums():
    print("\n所有图片已显示。关闭图片窗口后程序自动退出。")
    plt.show()

print("程序运行完毕。")

# 关闭日志文件，恢复原始 stdout/stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
print(f"[Log] 日志已保存至: {log_path}")
