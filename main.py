# main.py

import os
import torch
import matplotlib
matplotlib.use('TkAgg')  # 非阻塞显示后端
import matplotlib.pyplot as plt
from datetime import datetime

from train import train_with_cv
from evaluate import MonteCarloExperiment, plot_monte_carlo
from signal_gen import SignalSimulator

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CR_LIST = [4, 8, 16]
N_SAMPLES = 10000        # 论文: 50 snapshots × 200 trials
K_FOLDS = 5              # 论文: k=5
MAX_EPOCHS = 100         # 最大训练轮数（早停可提前结束）
BATCH_SIZE = 64
LR = 0.0005
SEED = 42
PATIENCE = 20            # 早停耐心值
WEIGHT_DECAY = 1e-4      # L2 正则化系数

# 输出目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "运行结果")
os.makedirs(RESULT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ===================== 工具函数 =====================

def save_and_show(fig, filename_stem):
    """
    保存图片到运行结果目录（含时间戳），并以非阻塞模式显示。

    参数:
        fig:           matplotlib Figure 对象
        filename_stem: 文件名主干（不含扩展名和时间戳）
    """
    filepath = os.path.join(RESULT_DIR, f"{filename_stem}_{TIMESTAMP}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[Saved] {filepath}")


def nonblocking_show():
    """非阻塞显示所有已创建的 figure，不阻塞 main 函数退出。"""
    plt.show(block=False)
    plt.pause(0.1)  # 允许 GUI 事件循环处理窗口绘制

# ===================== 主流程 =====================

models_dict = {}
cv_results_dict = {}

# 1. 实例化一个公共的 Simulator，供后续蒙特卡洛评估使用
sim = SignalSimulator()

# 2. 使用 k-fold 交叉验证训练各个压缩率下的网络
for cr in CR_LIST:
    model, cv_results = train_with_cv(
        DEVICE, cr=cr, k=K_FOLDS, n_samples=N_SAMPLES,
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=SEED,
        patience=PATIENCE, weight_decay=WEIGHT_DECAY
    )
    models_dict[cr] = model
    cv_results_dict[cr] = cv_results

# 3. 绘制交叉验证训练曲线并保存
fig_cv, axes_cv = plt.subplots(1, len(CR_LIST),
                                figsize=(6 * len(CR_LIST), 4.5))
if len(CR_LIST) == 1:
    axes_cv = [axes_cv]
for ax, cr in zip(axes_cv, CR_LIST):
    for fold_idx in range(K_FOLDS):
        ax.plot(cv_results_dict[cr]['fold_train_loss'][fold_idx],
                alpha=0.4, color=f'C{fold_idx}', linewidth=0.8)
        ax.plot(cv_results_dict[cr]['fold_val_loss'][fold_idx],
                alpha=0.7, color=f'C{fold_idx}', linewidth=1.2, linestyle='--')
    ax.set_title(f'CR={cr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.grid(True, alpha=0.3)
fig_cv.suptitle('K-Fold Cross-Validation Training Curves (solid: train, dashed: val)',
                fontsize=13)
fig_cv.tight_layout()
save_and_show(fig_cv, "CV_training_curves")

# 4. 统一蒙特卡洛评估
print("\n" + "=" * 40)
exp = MonteCarloExperiment(models_dict, sim, DEVICE)
mc_results = exp.run()

# 5. 绘制蒙特卡洛结果并保存
fig_mc = plot_monte_carlo(mc_results)
save_and_show(fig_mc, "MonteCarlo_TDOA_RMSE")

# 6. 非阻塞显示所有图片
nonblocking_show()

print(f"\n所有图片已保存至: {RESULT_DIR}")
print(f"时间戳: {TIMESTAMP}")
print("程序运行完毕。")
