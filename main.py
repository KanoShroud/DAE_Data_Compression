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
from evaluate import (MonteCarloExperiment, UrbanLocalizationExperiment, plot_monte_carlo, plot_snr_comparison,
                      plot_snr_comparison_multi, generate_snr_data, generate_snr_data_all,
                      clean_peak_consistency)
from model import DAE
from signal_gen import SignalSimulator

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_MODE = os.environ.get("DAE_EXPERIMENT_MODE", "paper_repro").lower()
PAPER_REPRO_MODES = ("paper_repro", "paper_repro_fast_final", "paper_repro_eval_only")
BASELINE_RESULT_ID = "20260702_000925"
BASELINE_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "运行结果", BASELINE_RESULT_ID)
CR_LIST = [4, 8, 16]
FAST_FINAL_EPOCHS_BY_CR = {4: 197, 8: 197, 16: 197}
BATCH_SIZE = 128
LR = 0.0005
SEED = 42
PATIENCE = 10            # 早停耐心值
WEIGHT_DECAY = 1e-4      # L2 正则化系数

# R20.1 创新轨道配置：保留，必要时用 $env:DAE_EXPERIMENT_MODE='r20_1' 切回。
R20_1_LOSS_CONFIG = {
    4:  {'epsilon_mse': 0.15, 'mse_weight_max': 0.18, 'phase_mix': 0.25, 'lambda_peak': 0.15},
    8:  {'epsilon_mse': 0.08, 'mse_weight_max': 0.10, 'phase_mix': 0.12, 'lambda_peak': 0.20},
    16: {'epsilon_mse': 0.03, 'mse_weight_max': 0.03, 'phase_mix': 0.03, 'lambda_peak': 0.25},
}

# 论文复现基线轨道：关闭任务损失，仅使用复数实/虚 MSE。
PAPER_REPRO_LOSS_CONFIG = {
    cr: {'loss_mode': 'paper_mse', 'epsilon_mse': 1.0, 'mse_weight_max': 1.0,
         'phase_mix': 1.0, 'lambda_peak': 0.0, 'selection_start_epoch': 1}
    for cr in CR_LIST
}

if EXPERIMENT_MODE in PAPER_REPRO_MODES:
    N_SAMPLES = 10000        # 单 UAV waveform 训练样本数；urban snapshot 数由 N_FIXED_CHANNELS 控制
    K_FOLDS = 5              # 论文使用 k=5；fast/eval 模式只保留为配置记录
    MAX_EPOCHS = 200         # 论文报告训练 200 epochs
    EARLY_STOPPING = False
    RESTORE_BEST = True
    FINAL_RETRAIN = True
    FINAL_RETRAIN_EPOCHS = None
    TRAINING_PROTOCOL = "cv_best_epoch_final_train"
    RESAMPLE_TRAIN_EACH_EPOCH = True
    RESAMPLE_INTERVAL = 5
    RUN_TRAINING = True
    MODEL_SOURCE_DIR = os.environ.get("DAE_MODEL_DIR", BASELINE_RESULT_DIR)
    if EXPERIMENT_MODE == "paper_repro_fast_final":
        MAX_EPOCHS = 197
        K_FOLDS = 0
        TRAINING_PROTOCOL = "fixed_epoch_final_only"
        FINAL_RETRAIN_EPOCHS = None  # resolved per CR from FAST_FINAL_EPOCHS_BY_CR
    elif EXPERIMENT_MODE == "paper_repro_eval_only":
        MAX_EPOCHS = 0
        K_FOLDS = 0
        RUN_TRAINING = False
        FINAL_RETRAIN = False
        RESTORE_BEST = False
        TRAINING_PROTOCOL = "eval_only_pretrained"
    elif EXPERIMENT_MODE == "paper_repro":
        MODEL_SOURCE_DIR = None
    CORR_WEIGHT = 0.0
    LAMBDA_PEAK = 0.0
    BETA_FI = 0.0
    LOSS_CONFIG = PAPER_REPRO_LOSS_CONFIG
    LOSS_MODE = "paper_mse"
    USE_ADAPTIVE_PEAK = False
    TRAIN_SNR_RANGE = (-10, 20)
    EVAL_SNR_RANGE = np.arange(-10, 21, 2)
    MONTE_CARLO_TRIALS = 200
    FIG2_SNR_LIST = [-10, 0, 10, 20]
    DIAGNOSTICS_VERSION = {
        "paper_repro": "paper_repro_v3_1_epoch_final_train",
        "paper_repro_fast_final": "paper_repro_v3_1_fast_final",
        "paper_repro_eval_only": "paper_repro_v3_1_eval_only",
    }[EXPERIMENT_MODE]
    CHANNEL_MODE = "fixed"
    N_FIXED_CHANNELS = 50
    NLOS_PROB = 0.0
    DELAY_LABEL_MODE = "los"
    MULTIPATH_SCALE = 0.2
    SCENARIO_MODE = "urban8"
    NORMALIZATION_MODE = "per_observation_noisy_rms"
    CV_GROUP_MODE = "snapshot"
    EVALUATION_MODE = "urban_localization"
    TDOA_SUB_SAMPLE = True
    USE_LOS_ONLY = True
    URBAN_BASE_DELAY = 16
    URBAN_MIN_LOS = 5
    URBAN_TRAIN_LOS_ONLY = True
    FIXED_EVAL_SET = True
    LOCALIZATION_ESTIMATOR = "all_pair_wls"
elif EXPERIMENT_MODE in ("r20_1", "r20.1", "task"):
    N_SAMPLES = 50000
    K_FOLDS = 3
    MAX_EPOCHS = 150
    EARLY_STOPPING = True
    RESTORE_BEST = True
    FINAL_RETRAIN = False
    FINAL_RETRAIN_EPOCHS = None
    TRAINING_PROTOCOL = "cv_earlystop_best_fold"
    RESAMPLE_TRAIN_EACH_EPOCH = False
    RESAMPLE_INTERVAL = 1
    RUN_TRAINING = True
    MODEL_SOURCE_DIR = None
    CORR_WEIGHT = 1.0
    LAMBDA_PEAK = 0.25
    BETA_FI = 0.0
    LOSS_CONFIG = R20_1_LOSS_CONFIG
    LOSS_MODE = "task"
    USE_ADAPTIVE_PEAK = True
    TRAIN_SNR_RANGE = (-10, 10)
    EVAL_SNR_RANGE = np.arange(-10, 11, 1)
    MONTE_CARLO_TRIALS = 1000
    FIG2_SNR_LIST = [-10, -3, 3, 10]
    DIAGNOSTICS_VERSION = "R20.1"
    CHANNEL_MODE = "fixed"
    N_FIXED_CHANNELS = 50
    NLOS_PROB = 0.2
    DELAY_LABEL_MODE = "strongest"
    MULTIPATH_SCALE = 0.3
    SCENARIO_MODE = "sv_pair"
    NORMALIZATION_MODE = "none"
    CV_GROUP_MODE = "sample"
    EVALUATION_MODE = "pair_tdoa"
    TDOA_SUB_SAMPLE = False
    USE_LOS_ONLY = False
    URBAN_BASE_DELAY = 16
    URBAN_MIN_LOS = 4
    URBAN_TRAIN_LOS_ONLY = False
    FIXED_EVAL_SET = False
    LOCALIZATION_ESTIMATOR = "single_ref"
else:
    raise ValueError("EXPERIMENT_MODE must be 'paper_repro', 'paper_repro_fast_final', "
                     "'paper_repro_eval_only', or 'r20_1'")

SNR_THRESHOLD = 0.0          # sigmoid 中心点 SNR (dB)
LAMBDA_TEMPERATURE = 5.0     # sigmoid 温度参数
FIG2_DIAGNOSTIC_TRIALS = 128  # extra samples saved in plot_data.pkl for Fig2 diagnostics

# Lightweight smoke-test overrides. Defaults above define the formal experiment.
N_SAMPLES = int(os.environ.get("DAE_N_SAMPLES", N_SAMPLES))
BATCH_SIZE = int(os.environ.get("DAE_BATCH_SIZE", BATCH_SIZE))
K_FOLDS = int(os.environ.get("DAE_K_FOLDS", K_FOLDS))
MAX_EPOCHS = int(os.environ.get("DAE_MAX_EPOCHS", MAX_EPOCHS))
MONTE_CARLO_TRIALS = int(os.environ.get("DAE_MONTE_CARLO_TRIALS", MONTE_CARLO_TRIALS))
FIG2_DIAGNOSTIC_TRIALS = int(os.environ.get("DAE_FIG2_DIAGNOSTIC_TRIALS", FIG2_DIAGNOSTIC_TRIALS))
N_FIXED_CHANNELS = int(os.environ.get("DAE_N_FIXED_CHANNELS", N_FIXED_CHANNELS))
if "DAE_FAST_FINAL_EPOCHS" in os.environ:
    _fast_ep = int(os.environ["DAE_FAST_FINAL_EPOCHS"])
    FAST_FINAL_EPOCHS_BY_CR = {cr: _fast_ep for cr in CR_LIST}
    if TRAINING_PROTOCOL == "fixed_epoch_final_only":
        MAX_EPOCHS = _fast_ep

# 多 seed 评估 —— 用不同 seed 训练模型，验证结果泛化性
# 设为 [SEED] 则只跑单 seed（快速）；设为 [42, 123, 456] 则跑 3 个 seed
SEED_LIST = [SEED]

# 如需快速验证（5~10分钟），可将 K_FOLDS 降至 1 并设 USE_SPLIT=True
# 此时仅做单次 80/20 train/val 划分，跳过全量 CV
USE_SPLIT = False        # True: 单次划分快速模式; False: K-fold CV

# 信道模式 —— "fixed" 从固定信道池采样（匹配论文 50 snapshots）；"random" 每样本随机信道
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
print(f"[Config] EXPERIMENT_MODE={EXPERIMENT_MODE} | LOSS_MODE={LOSS_MODE}")
print(f"[Config] Samples={N_SAMPLES} | K={K_FOLDS} | Epochs={MAX_EPOCHS} | "
      f"TrainSNR={TRAIN_SNR_RANGE} | EvalSNR={list(EVAL_SNR_RANGE)}")
print(f"[Config] Channel={CHANNEL_MODE} | fixed_channels={N_FIXED_CHANNELS} | "
      f"nlos_prob={NLOS_PROB} | delay_label_mode={DELAY_LABEL_MODE} | "
      f"multipath_scale={MULTIPATH_SCALE}")
print(f"[Config] Scenario={SCENARIO_MODE} | normalization={NORMALIZATION_MODE} | "
      f"cv_group={CV_GROUP_MODE} | eval={EVALUATION_MODE} | "
      f"tdoa_sub_sample={TDOA_SUB_SAMPLE}")
print(f"[Config] early_stopping={EARLY_STOPPING} | restore_best={RESTORE_BEST} | "
      f"final_retrain={FINAL_RETRAIN} | final_epochs={FINAL_RETRAIN_EPOCHS} | "
      f"protocol={TRAINING_PROTOCOL}")
print(f"[Config] train_resample={RESAMPLE_TRAIN_EACH_EPOCH} | "
      f"resample_interval={RESAMPLE_INTERVAL}")
print(f"[Config] run_training={RUN_TRAINING} | model_source_dir={MODEL_SOURCE_DIR}")
if SCENARIO_MODE == "urban8":
    print(f"[Config] urban_base_delay={URBAN_BASE_DELAY} | urban_min_los={URBAN_MIN_LOS} | "
          f"urban_train_los_only={URBAN_TRAIN_LOS_ONLY} | fixed_eval_set={FIXED_EVAL_SET} | "
          f"localization_estimator={LOCALIZATION_ESTIMATOR}")

# ===================== 工具函数 =====================

def save_figure(fig, filename_stem):
    """
    保存图片为 SVG 矢量图格式。

    SVG 为矢量图形，可无损缩放，适合论文插图。
    图形窗口在程序末尾统一弹出，避免重复显示。
    """
    try:
        fig.canvas.manager.set_window_title(filename_stem)
    except Exception:
        pass
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
    source_plot_data = None
    source_cv_results_dict = {}
    if not RUN_TRAINING:
        source_pkl = os.path.join(MODEL_SOURCE_DIR, "plot_data.pkl")
        if not os.path.exists(source_pkl):
            raise FileNotFoundError(f"eval_only requires source plot_data.pkl: {source_pkl}")
        with open(source_pkl, 'rb') as f:
            source_plot_data = pickle.load(f)
        source_cv_results_dict = source_plot_data.get('cv_results_dict', {})
        print(f"[EvalOnly] Loaded source plot data: {source_pkl}")

    # 1. 实例化 Simulator
    sim = SignalSimulator(channel_mode=CHANNEL_MODE,
                          n_fixed_channels=N_FIXED_CHANNELS,
                          channel_pool_seed=CHANNEL_POOL_SEED,
                          nlos_prob=NLOS_PROB,
                          delay_label_mode=DELAY_LABEL_MODE,
                          snr_train_range=TRAIN_SNR_RANGE,
                          multipath_scale=MULTIPATH_SCALE,
                          scenario_mode=SCENARIO_MODE,
                          normalization_mode=NORMALIZATION_MODE,
                          urban_base_delay=URBAN_BASE_DELAY,
                          urban_min_los=URBAN_MIN_LOS,
                          urban_train_los_only=URBAN_TRAIN_LOS_ONLY)
    clean_diag = None
    if EXPERIMENT_MODE in PAPER_REPRO_MODES:
        clean_diag = clean_peak_consistency(sim, n_trials=256, snr_db=20)
        print("[PaperRepro Check] Clean GCC vs LOS label: "
              f"exact={clean_diag.get('match_rate', float('nan')):.3f}, "
              f"within1={clean_diag.get('within_1_rate', float('nan')):.3f}, "
              f"within2={clean_diag.get('within_2_rate', float('nan')):.3f}, "
              f"mean_abs_err={clean_diag.get('mean_abs_peak_error', float('nan')):.2f}, "
              f"false_peaks>0.5={clean_diag.get('mean_false_peaks_gt_05', float('nan')):.2f}")

    # 2. 使用 k-fold 交叉验证训练各个压缩率下的网络
    for cr in CR_LIST:
        if RUN_TRAINING:
            final_epochs_for_cr = (
                FAST_FINAL_EPOCHS_BY_CR[cr]
                if TRAINING_PROTOCOL == "fixed_epoch_final_only"
                else FINAL_RETRAIN_EPOCHS
            )
            model, cv_results = train_with_cv(
                DEVICE, cr=cr, k=K_FOLDS, n_samples=N_SAMPLES,
                epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=current_seed,
                patience=PATIENCE, weight_decay=WEIGHT_DECAY, use_split=USE_SPLIT,
                channel_mode=CHANNEL_MODE, n_fixed_channels=N_FIXED_CHANNELS,
                channel_pool_seed=CHANNEL_POOL_SEED, sim=sim,
                corr_weight=CORR_WEIGHT, lambda_peak=LAMBDA_PEAK, beta_fi=BETA_FI,
                epsilon_mse=0.03,
                use_adaptive_peak=USE_ADAPTIVE_PEAK,
                snr_threshold=SNR_THRESHOLD, lambda_temperature=LAMBDA_TEMPERATURE,
                loss_config=LOSS_CONFIG[cr],
                loss_mode=LOSS_MODE,
                nlos_prob=NLOS_PROB,
                delay_label_mode=DELAY_LABEL_MODE,
                snr_train_range=TRAIN_SNR_RANGE,
                multipath_scale=MULTIPATH_SCALE,
                scenario_mode=SCENARIO_MODE,
                normalization_mode=NORMALIZATION_MODE,
                cv_group_mode=CV_GROUP_MODE,
                urban_base_delay=URBAN_BASE_DELAY,
                urban_min_los=URBAN_MIN_LOS,
                urban_train_los_only=URBAN_TRAIN_LOS_ONLY,
                early_stopping=EARLY_STOPPING,
                restore_best=RESTORE_BEST,
                final_retrain=FINAL_RETRAIN,
                final_epochs=final_epochs_for_cr,
                training_protocol=TRAINING_PROTOCOL,
                resample_train_each_epoch=RESAMPLE_TRAIN_EACH_EPOCH,
                resample_interval=RESAMPLE_INTERVAL,
            )
        else:
            model_path = os.path.join(MODEL_SOURCE_DIR, f"model_cr{cr}.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"eval_only requires source model: {model_path}")
            model = DAE(cr=cr).to(DEVICE)
            state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state)
            model.eval()
            cv_results = source_cv_results_dict.get(cr, source_cv_results_dict.get(str(cr)))
            if cv_results is None:
                raise KeyError(f"source plot_data.pkl has no cv_results for CR={cr}")
            cv_results = dict(cv_results)
            cv_results['training_protocol'] = TRAINING_PROTOCOL
            cv_results['model_source_dir'] = MODEL_SOURCE_DIR
            cv_results['diagnostics_version'] = DIAGNOSTICS_VERSION
            print(f"[EvalOnly] Loaded CR={cr} model: {model_path}")
        models_dict[cr] = model
        cv_results_dict[cr] = cv_results

        # 每个CR训练/加载完立即保存模型，保持结果文件结构兼容。
        model_path = os.path.join(RESULT_DIR, f"model_cr{cr}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"[Saved] {model_path}")

    # 3. Monte Carlo 评估
    print("\n" + "=" * 40)
    if EVALUATION_MODE == "urban_localization":
        exp = UrbanLocalizationExperiment(
            models_dict, sim, DEVICE, seed=current_seed,
            snr_range=EVAL_SNR_RANGE, num_trials=MONTE_CARLO_TRIALS,
            sub_sample=TDOA_SUB_SAMPLE, use_los_only=USE_LOS_ONLY,
            batch_size=BATCH_SIZE, fixed_eval_set=FIXED_EVAL_SET,
            estimator=LOCALIZATION_ESTIMATOR,
        )
    else:
        exp = MonteCarloExperiment(models_dict, sim, DEVICE, seed=current_seed,
                                   snr_range=EVAL_SNR_RANGE,
                                   num_trials=MONTE_CARLO_TRIALS)
    mc_results = exp.run()
    all_seed_results[current_seed] = mc_results

    # 4. 最后一个 seed：生成图片和保存数据
    if is_last_seed:
        # ===== Figure 1: 多分量训练动态 (3行×3列面板) =====
        n_cr = len(CR_LIST)
        fig_cv, axes_cv = plt.subplots(3, n_cr, figsize=(5 * n_cr, 8.5))
        if n_cr == 1:
            axes_cv = axes_cv[:, np.newaxis]  # 保证 2D 索引

        has_corr = CORR_WEIGHT > 0
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
                if 'fold_val_total_loss' in cr_data:
                    ax_mse.plot(cr_data['fold_val_total_loss'][fi], alpha=0.55,
                                color=f'C{fi}', linewidth=0.9, linestyle=':',
                                label=f'ValTotal (F{fi+1})' if n_folds > 1 else 'ValTotal')
            if LOSS_MODE == "paper_mse":
                mse_title = f'CR={cr} | Paper Repro MSE'
            else:
                mse_title = (f'CR={cr} | NMSE + ValTotal '
                             f'(w={LOSS_CONFIG[cr]["mse_weight_max"]}, '
                             f'mix={LOSS_CONFIG[cr]["phase_mix"]})')
            ax_mse.set_title(mse_title, fontsize=10, fontweight='bold')
            ax_mse.set_ylabel('Value')
            ax_mse.grid(True, alpha=0.3)
            ax_mse.legend(fontsize=5.5, framealpha=0.8, ncol=3)

            # --- Row 2: GCC Correlation Loss ---
            ax_corr = axes_cv[1, col]
            if has_corr and 'fold_corr_loss' in cr_data:
                for fi in range(n_folds):
                    ax_corr.plot(cr_data['fold_corr_loss'][fi], alpha=0.5,
                                 color=f'C{fi}', linewidth=1.0,
                                 label=f'Train F{fi+1}' if n_folds > 1 else 'Train Corr')
                    if 'fold_val_corr_loss' in cr_data:
                        ax_corr.plot(cr_data['fold_val_corr_loss'][fi], alpha=0.75,
                                     color=f'C{fi}', linewidth=1.0, linestyle='--',
                                     label=f'Val F{fi+1}' if n_folds > 1 else 'Val Corr')
                ax_corr.set_title(f'CR={cr} | GCC Correlation Loss', fontsize=10, fontweight='bold')
                ax_corr.legend(fontsize=6, framealpha=0.8)
            else:
                ax_corr.text(0.5, 0.5, 'N/A (pure MSE)', ha='center', va='center',
                             transform=ax_corr.transAxes, color='gray')
                ax_corr.set_title(f'CR={cr} | GCC Correlation Loss', fontsize=10, fontweight='bold')
            ax_corr.set_ylabel('Corr Loss')
            ax_corr.grid(True, alpha=0.3)

        # --- Row 3: Peak Loss per CR (each CR its own subplot) ---
        for col, cr in enumerate(CR_LIST):
            ax_pk = axes_cv[2, col]
            cr_data = cv_results_dict[cr]
            n_folds = len(cr_data.get('fold_peak_loss', []))
            if has_peak and n_folds > 0:
                for fi in range(n_folds):
                    ax_pk.plot(cr_data['fold_peak_loss'][fi], alpha=0.5,
                               color=f'C{fi}', linewidth=1.0,
                               label=f'Train F{fi+1}' if n_folds > 1 else 'Train Peak')
                    if 'fold_val_peak_loss' in cr_data:
                        ax_pk.plot(cr_data['fold_val_peak_loss'][fi], alpha=0.75,
                                   color=f'C{fi}', linewidth=1.0, linestyle='--',
                                   label=f'Val F{fi+1}' if n_folds > 1 else 'Val Peak')
                ax_pk.set_title(f'CR={cr} | SNR-Adaptive Peak Loss (λp_max={LOSS_CONFIG[cr]["lambda_peak"]})',
                                fontsize=9, fontweight='bold')
                ax_pk.legend(fontsize=6, framealpha=0.8)
            else:
                ax_pk.text(0.5, 0.5, 'N/A', ha='center', va='center',
                           transform=ax_pk.transAxes, color='gray')
            ax_pk.set_xlabel('Epoch'); ax_pk.set_ylabel('Peak Loss')
            ax_pk.grid(True, alpha=0.3)

        if LOSS_MODE == "paper_mse":
            if TRAINING_PROTOCOL == "cv_best_epoch_final_train":
                cfg_str = ("Paper reproduction baseline: CV fixed 200 epochs + "
                           "full-data final train, real/imag MSE only")
            elif TRAINING_PROTOCOL == "fixed_epoch_final_only":
                cfg_str = ("Paper reproduction fast final-only baseline, "
                           "real/imag MSE only")
            elif TRAINING_PROTOCOL == "eval_only_pretrained":
                cfg_str = ("Paper reproduction eval-only from frozen v3.1 models, "
                           "real/imag MSE only")
            else:
                cfg_str = f"Paper reproduction baseline: {TRAINING_PROTOCOL}, real/imag MSE only"
        else:
            cfg_str = ", ".join([
                f"CR{cr}: w={LOSS_CONFIG[cr]['mse_weight_max']}, "
                f"mix={LOSS_CONFIG[cr]['phase_mix']}, λp={LOSS_CONFIG[cr]['lambda_peak']}"
                for cr in CR_LIST
            ])
        cv_title = (f'Figure 1: {K_FOLDS}-Fold CV Training Dynamics  ({cfg_str})')
        fig_cv.suptitle(cv_title, fontsize=12, y=0.995)
        fig_cv.tight_layout()
        save_figure(fig_cv, "Fig1_CV_training_curves")

        # 生成 Figure 2 的绘图数据（所有CR叠加）
        snr_data_all = generate_snr_data_all(
            models_dict, sim, DEVICE, snr_list=FIG2_SNR_LIST,
            diagnostic_trials=FIG2_DIAGNOSTIC_TRIALS
        )

        # 保存绘图数据（含运行配置，供 replot.py / 离线分析）
        plot_data = {
            'cv_results_dict': cv_results_dict,
            'snr_data': snr_data_all,
            'snr_cr': CR_LIST,
            'mc_results': mc_results,
            'config': {
                'n_samples': N_SAMPLES,
                'training_sample_unit': 'single_uav_waveform',
                'batch_size': BATCH_SIZE,
                'k_folds': K_FOLDS,
                'max_epochs': MAX_EPOCHS,
                'lr': LR,
                'seed': current_seed,
                'seed_list': SEED_LIST,
                'patience': PATIENCE,
                'early_stopping': EARLY_STOPPING,
                'restore_best': RESTORE_BEST,
                'final_retrain': FINAL_RETRAIN,
                'final_retrain_epochs': FINAL_RETRAIN_EPOCHS,
                'training_protocol': TRAINING_PROTOCOL,
                'run_training': RUN_TRAINING,
                'model_source_dir': MODEL_SOURCE_DIR,
                'fast_final_epochs_by_cr': FAST_FINAL_EPOCHS_BY_CR,
                'resample_train_each_epoch': RESAMPLE_TRAIN_EACH_EPOCH,
                'resample_interval': RESAMPLE_INTERVAL,
                'weight_decay': WEIGHT_DECAY,
                'experiment_mode': EXPERIMENT_MODE,
                'loss_mode': LOSS_MODE,
                'corr_weight': CORR_WEIGHT,
                'lambda_peak': LAMBDA_PEAK,
                'use_adaptive_peak': USE_ADAPTIVE_PEAK,
                'snr_threshold': SNR_THRESHOLD,
                'lambda_temperature': LAMBDA_TEMPERATURE,
                'train_snr_range': TRAIN_SNR_RANGE,
                'eval_snr_range': [float(x) for x in EVAL_SNR_RANGE],
                'monte_carlo_trials': MONTE_CARLO_TRIALS,
                'fig2_snr_list': FIG2_SNR_LIST,
                'fig2_diagnostic_trials': FIG2_DIAGNOSTIC_TRIALS,
                'channel_mode': CHANNEL_MODE,
                'n_fixed_channels': N_FIXED_CHANNELS,
                'nlos_prob': NLOS_PROB,
                'delay_label_mode': DELAY_LABEL_MODE,
                'multipath_scale': MULTIPATH_SCALE,
                'scenario_mode': SCENARIO_MODE,
                'normalization_mode': NORMALIZATION_MODE,
                'cv_group_mode': CV_GROUP_MODE,
                'evaluation_mode': EVALUATION_MODE,
                'tdoa_sub_sample': TDOA_SUB_SAMPLE,
                'use_los_only': USE_LOS_ONLY,
                'urban_base_delay': URBAN_BASE_DELAY,
                'urban_min_los': URBAN_MIN_LOS,
                'urban_train_los_only': URBAN_TRAIN_LOS_ONLY,
                'fixed_eval_set': FIXED_EVAL_SET,
                'localization_estimator': LOCALIZATION_ESTIMATOR,
                'paper_repro_clean_peak_check': clean_diag,
                'cr_list': CR_LIST,
                'loss_config': LOSS_CONFIG,
                'diagnostics_version': DIAGNOSTICS_VERSION,
            },
        }
        pkl_path = os.path.join(RESULT_DIR, "plot_data.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f"[Saved] {pkl_path}")

        # 绘制 Figure 2 和 3。创建顺序与本地文件序号保持一致，避免窗口标题混乱。
        fig_snr = plot_snr_comparison_multi(cr_list=CR_LIST, data_dict=snr_data_all)
        save_figure(fig_snr, "Fig2_SNR_Comparison")

        fig_mc = plot_monte_carlo(mc_results)
        save_figure(fig_mc, "Fig3_MonteCarlo_TDOA_RMSE")

# ===================== 多 seed 汇总 =====================
if len(SEED_LIST) > 1:
    print(f"\n{'='*60}")
    print(f"Multi-Seed Evaluation Summary ({len(SEED_LIST)} seeds)")
    print(f"{'='*60}")
    for cr in CR_LIST:
        key = f'dae_{cr}'
        key_med = f'dae_{cr}_med'
        # 取最接近 SNR=0 dB 的结果做对比，兼容不同 SNR 网格。
        snr_idx = int(np.argmin(np.abs(np.asarray(EVAL_SNR_RANGE, dtype=float) - 0.0)))
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
if plt.get_fignums() and os.environ.get("DAE_NO_SHOW", "0") != "1":
    print("\n所有图片已显示。关闭图片窗口后程序自动退出。")
    plt.show()
else:
    plt.close('all')

print("程序运行完毕。")

# 关闭日志文件，恢复原始 stdout/stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
print(f"[Log] 日志已保存至: {log_path}")
