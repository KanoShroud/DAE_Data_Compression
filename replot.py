# replot.py
# 从 plot_data.pkl 读取绘图数据，重新生成 SVG 图。
#
# PyCharm 直接运行：修改下方 RESULT_DIR 为实际路径，点击运行即可。
# 命令行运行：python replot.py "运行结果/20260529_165138"

import os
import sys
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from evaluate import (plot_snr_comparison, plot_snr_comparison_multi,
                      plot_monte_carlo, plot_method_comparison,
                      plot_method_zoom_pair)

# ========== 在此修改结果目录路径 ==========
RESULT_DIR = "运行结果/20260601_145854"
# ==========================================


def save_figure(fig, filepath):
    try:
        fig.canvas.manager.set_window_title(os.path.splitext(os.path.basename(filepath))[0])
    except Exception:
        pass
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"[Saved] {filepath}")


def unique_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def apply_current_method_view_config(method_results, view):
    """Apply the current paper-ready Fig6/Fig7 method lists to older pkl files."""
    if method_results is None:
        return None
    results, snr_range = method_results
    config = results.setdefault('config', {})
    baseline_cr = config.get('baseline_cr', 16)
    meta = config.get('traditional_baseline_meta', {})
    had_main = meta.get('hadamard_main_label', 'Hadamard')
    dae_label = f'DAE-CR{baseline_cr}'

    if view == 'fig6':
        main_order = unique_order(['Raw', dae_label, 'DFT', had_main, 'PCA'])
        supp_order = unique_order([
            'Raw', dae_label, 'DFT', 'DFT-SCS-lite',
            'DFT-train-power', 'DFT-bandlimited',
            had_main, 'Hadamard-block-2', 'PCA'
        ])
        config['main_method_order'] = main_order
        config['supplement_method_order'] = supp_order
        config['supplement_zoom_method_order'] = supp_order
        config['supplement_zoom_figure_filename'] = (
            f'Fig6_Supp_Chen_Baseline_Ablation_CR{baseline_cr}_SNR_Zooms'
        )
    elif view == 'fig7':
        task_order = unique_order([
            'Raw', dae_label, 'DFT', 'DFT-SCS-lite',
            'DFT-Fisher-Direct', had_main, 'PCA'
        ])
        config['taskaware_method_order'] = task_order
        config['taskaware_zoom_method_order'] = task_order
        config['taskaware_zoom_figure_filename'] = (
            f'Fig7_TaskAware_Baselines_CR{baseline_cr}_SNR_Zooms'
        )
    elif view == 'fig8':
        strong_order = unique_order([
            'Raw', dae_label, 'DFT', 'DFT-Fisher-Direct',
            'Cao2017-DFT-AML', 'Cao2020-HighFC',
            'Zhai-CRLB-Decimation', 'Zhai-Phase-Superposition',
            had_main, 'PCA'
        ])
        config['strong_method_order'] = strong_order
        config['strong_zoom_method_order'] = strong_order
        config['strong_zoom_figure_filename'] = (
            f'Fig8_Strong_TaskAware_Baselines_CR{baseline_cr}_SNR_Zooms'
        )
    return results, snr_range


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
    config = data.get('config', None)

    # --- 显示运行配置 ---
    if config:
        print(f"\n{'='*60}")
        print(f"Run Configuration:")
        if 'experiment_mode' in config:
            print(f"  Experiment mode: {config['experiment_mode']} | Loss mode: {config.get('loss_mode', 'N/A')}")
        print(f"  Samples: {config['n_samples']} | Batch: {config['batch_size']} | LR: {config['lr']}")
        print(f"  K-Folds: {config['k_folds']} | Max Epochs: {config['max_epochs']} | Patience: {config['patience']}")
        if 'early_stopping' in config:
            print(f"  Early stopping: {config['early_stopping']} | Restore best: {config.get('restore_best', 'N/A')} | "
                  f"Final retrain: {config.get('final_retrain', 'N/A')} "
                  f"({config.get('final_retrain_epochs', 'N/A')} epochs)")
        if 'training_protocol' in config:
            print(f"  Training protocol: {config['training_protocol']}")
        if 'training_sample_unit' in config:
            print(f"  Training sample unit: {config['training_sample_unit']}")
        if 'resample_train_each_epoch' in config:
            print(f"  Train resampling: {config['resample_train_each_epoch']} | "
                  f"interval: {config.get('resample_interval', 'N/A')}")
        corr_weight = config.get('corr_weight', config.get('lambda_corr', 'N/A'))
        use_adaptive_peak = config.get('use_adaptive_peak', config.get('use_adaptive_lambda', 'N/A'))
        print(f"  Corr weight: {corr_weight} | λ_peak: {config['lambda_peak']} | "
              f"Adaptive Peak: {use_adaptive_peak}")
        print(f"  Channel: {config['channel_mode']} ({config['n_fixed_channels']} fixed)")
        if 'nlos_prob' in config:
            print(f"  NLOS prob: {config['nlos_prob']} | delay label: {config.get('delay_label_mode', 'N/A')} | "
                  f"multipath scale: {config.get('multipath_scale', 'N/A')}")
        if 'scenario_mode' in config:
            print(f"  Scenario: {config['scenario_mode']} | normalization: {config.get('normalization_mode', 'N/A')} | "
                  f"CV group: {config.get('cv_group_mode', 'N/A')}")
        if 'evaluation_mode' in config:
            print(f"  Evaluation: {config['evaluation_mode']} | sub-sample TDOA: "
                  f"{config.get('tdoa_sub_sample', 'N/A')} | LOS-only: {config.get('use_los_only', 'N/A')}")
        if 'urban_base_delay' in config:
            print(f"  Urban base delay: {config.get('urban_base_delay', 'N/A')} | "
                  f"min LOS: {config.get('urban_min_los', 'N/A')} | "
                  f"train LOS-only: {config.get('urban_train_los_only', 'N/A')} | "
                  f"fixed eval: {config.get('fixed_eval_set', 'N/A')} | "
                  f"estimator: {config.get('localization_estimator', 'N/A')}")
        if 'eval_snr_range' in config:
            print(f"  Eval SNR range: {config['eval_snr_range']}")
        print(f"  CR List: {config['cr_list']}")
        if 'diagnostics_version' in config:
            print(f"  Diagnostics version: {config['diagnostics_version']}")
        if 'fig2_diagnostic_trials' in config:
            print(f"  Fig2 diagnostic trials: {config['fig2_diagnostic_trials']}")
        if 'run_traditional_baselines' in config:
            print(f"  Traditional baselines: {config['run_traditional_baselines']} | "
                  f"baseline CR: {config.get('baseline_cr', 'N/A')} | "
                  f"PCA samples: {config.get('baseline_pca_samples', 'N/A')} | "
                  f"DFT: {config.get('baseline_dft_mode', 'N/A')} | "
                  f"Hadamard: {config.get('baseline_hadamard_mode', 'N/A')} | "
                  f"PCA source: {config.get('baseline_pca_train_source', 'N/A')}")
        if 'loss_config' in config:
            print(f"  Loss config: {config['loss_config']}")
        print(f"{'='*60}\n")
    else:
        print("(No config found in plot_data.pkl — legacy data file)")

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

    # --- Figure 2: SNR 信号对比 (多CR叠加) ---
    if isinstance(snr_cr, list):
        fig_snr = plot_snr_comparison_multi(cr_list=snr_cr, data_dict=snr_data)
    else:
        # 旧格式兼容: 单CR
        fig_snr = plot_snr_comparison(cr=snr_cr, data=snr_data)
    save_figure(fig_snr, os.path.join(result_dir, "Fig2_SNR_Comparison.svg"))

    # --- Figure 3: Monte Carlo ---
    fig_mc = plot_monte_carlo(mc_results)
    save_figure(fig_mc, os.path.join(result_dir, "Fig3_MonteCarlo_TDOA_RMSE.svg"))

    fig6_results = apply_current_method_view_config(data.get('fig6_results'), 'fig6')
    if fig6_results is not None:
        baseline_cr = data.get('config', {}).get('baseline_cr', 16)
        fig6_config = fig6_results[0].get('config', {})
        fig_methods = plot_method_comparison(fig6_results, plot_kind="main")
        save_figure(fig_methods, os.path.join(
            result_dir, f"{fig6_config.get('main_figure_filename', f'Fig6_Traditional_Baselines_CR{baseline_cr}')}.svg"
        ))
        fig_methods_supp = plot_method_comparison(fig6_results, plot_kind="supplement")
        save_figure(fig_methods_supp, os.path.join(
            result_dir, f"{fig6_config.get('supplement_figure_filename', f'Fig6_Supp_Baseline_Ablation_CR{baseline_cr}')}.svg"
        ))
        if fig6_config.get('export_method_zoom_figures', True):
            zoom_order = fig6_config.get('supplement_zoom_method_order')
            fig_methods_zoom = plot_method_zoom_pair(
                fig6_results, plot_kind="supplement",
                low_snr_max=fig6_config.get('method_zoom_low_snr_max', 0.0),
                high_snr_min=fig6_config.get('method_zoom_high_snr_min', 8.0),
                method_order=zoom_order,
            )
            save_figure(fig_methods_zoom, os.path.join(
                result_dir,
                f"{fig6_config.get('supplement_zoom_figure_filename', f'Fig6_Supp_Chen_Baseline_Ablation_CR{baseline_cr}_SNR_Zooms')}.svg"
            ))

    fig7_results = apply_current_method_view_config(data.get('fig7_results'), 'fig7')
    if fig7_results is not None:
        baseline_cr = data.get('config', {}).get('baseline_cr', 16)
        fig7_config = fig7_results[0].get('config', {})
        fig_task = plot_method_comparison(fig7_results, plot_kind="taskaware")
        save_figure(fig_task, os.path.join(
            result_dir, f"{fig7_config.get('taskaware_figure_filename', f'Fig7_TaskAware_Baselines_CR{baseline_cr}')}.svg"
        ))
        if fig7_config.get('export_method_zoom_figures', True):
            zoom_order = fig7_config.get('taskaware_zoom_method_order')
            fig_task_zoom = plot_method_zoom_pair(
                fig7_results, plot_kind="taskaware",
                low_snr_max=fig7_config.get('method_zoom_low_snr_max', 0.0),
                high_snr_min=fig7_config.get('method_zoom_high_snr_min', 8.0),
                method_order=zoom_order,
            )
            save_figure(fig_task_zoom, os.path.join(
                result_dir,
                f"{fig7_config.get('taskaware_zoom_figure_filename', f'Fig7_TaskAware_Baselines_CR{baseline_cr}_SNR_Zooms')}.svg"
            ))

    fig8_results = apply_current_method_view_config(data.get('fig8_results'), 'fig8')
    if fig8_results is not None:
        baseline_cr = data.get('config', {}).get('baseline_cr', 16)
        fig8_config = fig8_results[0].get('config', {})
        fig_strong = plot_method_comparison(fig8_results, plot_kind="strong")
        save_figure(fig_strong, os.path.join(
            result_dir, f"{fig8_config.get('strong_figure_filename', f'Fig8_Strong_TaskAware_Baselines_CR{baseline_cr}')}.svg"
        ))
        if fig8_config.get('export_method_zoom_figures', True):
            zoom_order = fig8_config.get('strong_zoom_method_order')
            fig_strong_zoom = plot_method_zoom_pair(
                fig8_results, plot_kind="strong",
                low_snr_max=fig8_config.get('method_zoom_low_snr_max', 0.0),
                high_snr_min=fig8_config.get('method_zoom_high_snr_min', 8.0),
                method_order=zoom_order,
            )
            save_figure(fig_strong_zoom, os.path.join(
                result_dir,
                f"{fig8_config.get('strong_zoom_figure_filename', f'Fig8_Strong_TaskAware_Baselines_CR{baseline_cr}_SNR_Zooms')}.svg"
            ))

    if plt.get_fignums() and os.environ.get("DAE_REPLOT_NO_SHOW") != "1":
        print("\n所有图片已显示。关闭图片窗口后程序自动退出。")
        plt.show()
    else:
        plt.close('all')

    print("重绘完成。")


if __name__ == '__main__':
    # 命令行参数优先，否则使用顶部 RESULT_DIR
    if len(sys.argv) >= 2:
        replot(sys.argv[1])
    else:
        replot(RESULT_DIR)
