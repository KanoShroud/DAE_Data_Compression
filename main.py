# main.py

import os
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
from evaluate import MonteCarloExperiment, plot_monte_carlo, plot_snr_comparison, generate_snr_data
from signal_gen import SignalSimulator

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CR_LIST = [4, 8, 16]
N_SAMPLES = 40000        # 增加样本数以改善低SNR区间的CR排序
K_FOLDS = 3              # 论文k=5, 但随机信道下CV方差大, 3折已足够且节省40%时间
MAX_EPOCHS = 150         # 最大训练轮数（早停可提前结束）
BATCH_SIZE = 128
LR = 0.0005
SEED = 42
PATIENCE = 10            # 早停耐心值
WEIGHT_DECAY = 1e-4      # L2 正则化系数

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

# ===================== 工具函数 =====================

def save_figure(fig, filename_stem):
    """
    保存图片为 SVG 矢量图格式，并立即弹出显示窗口。

    SVG 为矢量图形，可无损缩放，适合论文插图。
    """
    filepath = os.path.join(RESULT_DIR, f"{filename_stem}.svg")
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"[Saved] {filepath}")
    # 非阻塞弹出窗口，后续 plt.show() 会统一阻塞等待用户关闭
    plt.show(block=False)
    plt.pause(0.1)

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
            channel_pool_seed=CHANNEL_POOL_SEED, sim=sim
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
        # 绘制交叉验证训练曲线
        fig_cv, axes_cv = plt.subplots(1, len(CR_LIST),
                                        figsize=(6 * len(CR_LIST), 4.5))
        if len(CR_LIST) == 1:
            axes_cv = [axes_cv]
        for ax, cr in zip(axes_cv, CR_LIST):
            n_folds = len(cv_results_dict[cr]['fold_train_loss'])
            for fold_idx in range(n_folds):
                lbl_train = f'Train (Fold {fold_idx+1})' if n_folds > 1 else 'Train'
                lbl_val = f'Val (Fold {fold_idx+1})' if n_folds > 1 else 'Val'
                ax.plot(cv_results_dict[cr]['fold_train_loss'][fold_idx],
                        alpha=0.4, color=f'C{fold_idx}', linewidth=0.8, label=lbl_train)
                ax.plot(cv_results_dict[cr]['fold_val_loss'][fold_idx],
                        alpha=0.7, color=f'C{fold_idx}', linewidth=1.2, linestyle='--',
                        label=lbl_val)
            ax.set_title(f'CR={cr}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, framealpha=0.8)
        cv_title = 'Figure 1: Training Curves (solid: train, dashed: val)'
        if not USE_SPLIT:
            cv_title = f'Figure 1: {K_FOLDS}-Fold Cross-Validation Training Curves'
        fig_cv.suptitle(cv_title, fontsize=13)
        fig_cv.tight_layout()
        save_figure(fig_cv, "Fig1_CV_training_curves")

        # 生成 Figure 2 的绘图数据
        snr_data = generate_snr_data(models_dict[CR_LIST[-1]], sim, DEVICE)

        # 保存绘图数据
        plot_data = {
            'cv_results_dict': cv_results_dict,
            'snr_data': snr_data,
            'snr_cr': CR_LIST[-1],
            'mc_results': mc_results,
        }
        pkl_path = os.path.join(RESULT_DIR, "plot_data.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f"[Saved] {pkl_path}")

        # 绘制 Figure 2 和 3
        fig_mc = plot_monte_carlo(mc_results)
        save_figure(fig_mc, "Fig3_MonteCarlo_TDOA_RMSE")

        fig_snr = plot_snr_comparison(cr=CR_LIST[-1], data=snr_data)
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
