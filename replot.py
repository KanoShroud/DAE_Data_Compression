# replot.py
# 从 plot_data.pkl 读取绘图数据，重新生成三张 SVG 图。
#
# PyCharm 直接运行：修改下方 RESULT_DIR 为实际路径，点击运行即可。
# 命令行运行：python replot.py "运行结果/20260529_165138"

import os
import sys
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from evaluate import plot_snr_comparison, plot_monte_carlo

# ========== 在此修改结果目录路径 ==========
RESULT_DIR = "运行结果/20260601_145854"
# ==========================================


def save_figure(fig, filepath):
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"[Saved] {filepath}")
    plt.show(block=False)
    plt.pause(0.1)


def replot(result_dir):
    pkl_path = os.path.join(result_dir, "plot_data.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    cv_results_dict = data['cv_results_dict']
    snr_data = data['snr_data']
    snr_cr = data['snr_cr']
    mc_results = data['mc_results']

    # --- Figure 1: CV 训练曲线 ---
    cr_list = list(cv_results_dict.keys())
    fig_cv, axes_cv = plt.subplots(1, len(cr_list), figsize=(6 * len(cr_list), 4.5))
    if len(cr_list) == 1:
        axes_cv = [axes_cv]
    for ax, cr in zip(axes_cv, cr_list):
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
    fig_cv.suptitle(f'Figure 1: {n_folds}-Fold Cross-Validation Training Curves', fontsize=13)
    fig_cv.tight_layout()
    save_figure(fig_cv, os.path.join(result_dir, "Fig1_CV_training_curves.svg"))

    # --- Figure 2: SNR 信号对比 ---
    fig_snr = plot_snr_comparison(cr=snr_cr, data=snr_data)
    save_figure(fig_snr, os.path.join(result_dir, "Fig2_SNR_Comparison.svg"))

    # --- Figure 3: Monte Carlo ---
    fig_mc = plot_monte_carlo(mc_results)
    save_figure(fig_mc, os.path.join(result_dir, "Fig3_MonteCarlo_TDOA_RMSE.svg"))

    if plt.get_fignums():
        print("\n所有图片已显示。关闭图片窗口后程序自动退出。")
        plt.show()

    print("重绘完成。")


if __name__ == '__main__':
    # 命令行参数优先，否则使用顶部 RESULT_DIR
    if len(sys.argv) >= 2:
        replot(sys.argv[1])
    else:
        replot(RESULT_DIR)
