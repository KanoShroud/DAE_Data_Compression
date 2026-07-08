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
                      clean_peak_consistency, run_urban_method_comparison,
                      plot_method_comparison, plot_method_zoom_pair,
                      filter_method_comparison_data)
from model import DAE, FrequencyPairwiseDAE, FrequencySelectiveDAE, NestedFrequencyPairwiseDAE
from signal_gen import SignalSimulator
from baselines import build_traditional_baselines
from task_baselines import (
    Cao2020SegmentedFCEstimator,
    Cao2017DFTAMLEstimator,
    DirectDFTTDOAEstimator,
    GeoHybridDFTTDOAEstimator,
    ZhaiPhaseSuperpositionEstimator,
)
from experiment_cache import (
    load_static_baseline_cache,
    merge_method_comparison_results,
)
from experiment_profiles import apply_experiment_profile
# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAPER_REPRO_MODES = ("paper_repro", "paper_repro_fast_final", "paper_repro_eval_only")
FREQ_TASK_TRAIN_MODES = ("freq_task", "freq_task_fast_final", "freq_task_200_final")
FREQ_TASK_EVAL_MODES = ("freq_task_eval_only", "innovation_eval_only")
FREQ_TASK_MODES = FREQ_TASK_TRAIN_MODES + FREQ_TASK_EVAL_MODES
FREQ_TASK_V2_TRAIN_MODES = (
    "freq_task_v2", "freq_task_v2_fast_final", "freq_task_v2_200_final"
)
FREQ_TASK_V2_EVAL_MODES = ("freq_task_v2_eval_only",)
FREQ_TASK_V2_MODES = FREQ_TASK_V2_TRAIN_MODES + FREQ_TASK_V2_EVAL_MODES
FREQ_TASK_V3_TRAIN_MODES = (
    "freq_task_v3", "freq_task_v3_fast_final", "freq_task_v3_200_final"
)
FREQ_TASK_V3_EVAL_MODES = ("freq_task_v3_eval_only",)
FREQ_TASK_V3_MODES = FREQ_TASK_V3_TRAIN_MODES + FREQ_TASK_V3_EVAL_MODES
INNOVATION_MODES = FREQ_TASK_MODES + FREQ_TASK_V2_MODES + FREQ_TASK_V3_MODES
BASELINE_RESULT_ID = "20260702_000925"
BASELINE_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "运行结果", BASELINE_RESULT_ID)
FREQ_TASK_RESULT_ID = "20260706_103701"
FREQ_TASK_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "运行结果", FREQ_TASK_RESULT_ID)

# ===================== 用户常用配置区 =====================
# 默认直接修改本区变量，然后在 PyCharm 点击运行 main.py。
# 环境变量覆盖默认关闭，避免外部 shell/PyCharm 配置不一致导致误运行。
ALLOW_ENV_OVERRIDES = False

# 可选: "freq_task_v3_fast_final"、"freq_task_v3_200_final"、"freq_task_v3_eval_only"、
#       "freq_task_v2_200_final"、"freq_task_v2_fast_final"、"freq_task_v2_eval_only"、
#       "freq_task_eval_only"、"innovation_eval_only"、
#       "freq_task"、"freq_task_fast_final"、"freq_task_200_final"、
#       "paper_repro_eval_only"、"paper_repro"、"paper_repro_fast_final"、"r20_1"
# 若 USER_EXPERIMENT_PROFILE 不为 None，会覆盖下方相关 USER_* 变量。
USER_EXPERIMENT_PROFILE = "geohybrid_v2_eval"
USER_EXPERIMENT_MODE = "freq_task_v3_fast_final"
USER_MODEL_SOURCE_DIR = BASELINE_RESULT_DIR
USER_FREQ_TASK_MODEL_SOURCE_DIR = FREQ_TASK_RESULT_DIR
USER_FREQ_TASK_V2_MODEL_SOURCE_DIR = None
USER_FREQ_TASK_V3_MODEL_SOURCE_DIR = None
USER_REPRO_REFERENCE_MODEL_DIR = BASELINE_RESULT_DIR
USER_REUSE_STATIC_BASELINE_CACHE = True
USER_STATIC_BASELINE_CACHE_RESULT_ID = "20260706_230643"
USER_STATIC_BASELINE_CACHE_DIR = None

# 正式实验默认值。若要快速 smoke，可直接改这些普通变量。
USER_N_SAMPLES = None
USER_BATCH_SIZE = None
USER_K_FOLDS = None
USER_MAX_EPOCHS = None
USER_MONTE_CARLO_TRIALS = None
USER_FIG2_DIAGNOSTIC_TRIALS = None
USER_N_FIXED_CHANNELS = None
USER_FAST_FINAL_EPOCHS = None
USER_FREQ_TASK_EPOCHS = None
USER_FREQ_TASK_FAST_EPOCHS = 120
USER_FREQ_TASK_FULL_EPOCHS = 200
USER_FREQ_TASK_SPECTRAL_BLEND = 0.30
USER_FREQ_TASK_SPECTRAL_POWER = 2.0
USER_FREQ_TASK_V2_FAST_EPOCHS = 120
USER_FREQ_TASK_V2_FULL_EPOCHS = 200
USER_FREQ_TASK_V2_SPECTRAL_BLEND = 0.20
USER_FREQ_TASK_V2_SPECTRAL_POWER = 1.0
USER_FREQ_TASK_V2_CORR_WEIGHT = 0.20
USER_FREQ_TASK_V2_PHASE_WEIGHT = 0.12
USER_FREQ_TASK_V2_PEAK_WEIGHT = 0.03
USER_FREQ_TASK_V2_SOFT_PEAK_SIGMA = 1.5
USER_FREQ_TASK_V2_PEAK_SHARPNESS_WEIGHT = 0.25
USER_FREQ_TASK_V3_FAST_EPOCHS = 60
USER_FREQ_TASK_V3_FULL_EPOCHS = 200
USER_FREQ_TASK_V3_SPECTRAL_BLEND = 0.20
USER_FREQ_TASK_V3_SPECTRAL_POWER = 1.0
USER_FREQ_TASK_V3_CORR_WEIGHT = 0.15
USER_FREQ_TASK_V3_PHASE_WEIGHT = 0.08
USER_FREQ_TASK_V3_PEAK_WEIGHT = 0.02
USER_FREQ_TASK_V3_TASK_HEAD_WEIGHT = 0.03
USER_FREQ_TASK_V3_SOFT_PEAK_SIGMA = 1.5
USER_FREQ_TASK_V3_PEAK_SHARPNESS_WEIGHT = 0.15
USER_FREQ_TASK_V3_NESTED_CRS = (4, 8, 16)
USER_INCLUDE_REPRO_REFERENCE = True

# Fig6 传统 baseline 配置。
USER_RUN_TRADITIONAL_BASELINES = True
USER_BASELINE_CR = 16
USER_BASELINE_DFT_MODE = "scs_lite"  # "scs_lite"、"fisher_power"、"train_band"、"train_power"、"center"、"random"
USER_BASELINE_DFT_DIRECT_SOURCE = "DFT-train-power"  # Fig6 主线 DFT direct 的 selected-bin 来源
USER_BASELINE_HADAMARD_MODE = "salari"   # "salari"、"random"、"sequency"
USER_BASELINE_PCA_TRAIN_SOURCE = "noisy" # "noisy"、"clean"、"noisy_fixed"、"clean_fixed"
USER_BASELINE_PCA_FIXED_SNR_DB = 0.0
USER_BASELINE_PCA_SAMPLES = None         # None 表示使用 N_SAMPLES
USER_BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS = True
USER_BASELINE_INCLUDE_LEGACY_VARIANTS = False
USER_BASELINE_RANDOM_VARIANT_SEEDS = 0
USER_RUN_TASK_AWARE_BASELINES = True
USER_RUN_STRONG_BASELINES = True        # 独立 Fig8：Cao/Zhai 等强 TDOA-aware baseline
USER_STRONG_INCLUDE_PHASE_SUPERPOSITION = True   # Fig8: Zhai CRLB bins + phase superposition
USER_RUN_STRONG_DIAGNOSTIC_SUPPLEMENT = True     # Fig8 supplement: proxy/diagnostic strong candidates
USER_RUN_GEOHYBRID_BASELINES = True     # GeoHybrid v2: coarse high-coherence + GeoAmbi fine refinement
USER_RUN_LOCALIZER_ABLATION = False      # Deprecated diagnostic; keep disabled to avoid bulky outputs.
USER_LOCALIZER_ABLATION_SNR_RANGE = [10, 12, 14, 16, 18, 20]
USER_LOCALIZER_ABLATION_ESTIMATORS = [
    "all_pair_wls",
]
USER_LOCALIZER_ABLATION_METHODS = [
    "Raw", "DAE-CR4", "DAE-CR16",
    "FreqDAE-v2-CR4", "FreqDAE-v2-CR16",
    "Zhai-CRLB-Decimation",
]
USER_EVAL_ONLY_COPY_MODELS = False       # False: eval_only 不把源模型权重重复复制到新结果目录
USER_FIG6_SHOW_ZOOM_INSET = True
USER_FIG6_ZOOM_SNR_MIN = 8.0
USER_EXPORT_FIG6_SVG_IN_TABLES = False   # False: 避免根目录与 tables/ 重复保存 Fig6 SVG
USER_USE_PHYSICAL_TDOA_LAG_GATE = True
USER_TDOA_LAG_LIMIT_SAMPLES = None        # None 表示按 urban8 区域尺寸自动计算
USER_TDOA_LAG_MARGIN_SAMPLES = 4
USER_EXPORT_METHOD_ZOOM_FIGURES = True
USER_METHOD_ZOOM_LOW_SNR_MAX = 0.0
USER_METHOD_ZOOM_HIGH_SNR_MIN = 8.0
USER_NO_SHOW = False                     # True: 只保存图片，不弹出 matplotlib 窗口

APPLIED_EXPERIMENT_PROFILE = apply_experiment_profile(
    globals(), USER_EXPERIMENT_PROFILE
)


def _cfg(env_name, user_value, default_value, cast=lambda x: x):
    if ALLOW_ENV_OVERRIDES and env_name in os.environ:
        return cast(os.environ[env_name])
    value = default_value if user_value is None else user_value
    return cast(value)


def _cfg_bool(env_name, user_value, default_value):
    if ALLOW_ENV_OVERRIDES and env_name in os.environ:
        return os.environ[env_name] == "1"
    return bool(default_value if user_value is None else user_value)


def _cfg_optional_int(env_name, user_value, default_value=None):
    if ALLOW_ENV_OVERRIDES and env_name in os.environ:
        raw = str(os.environ[env_name]).strip().lower()
        if raw in ("", "none", "null", "auto"):
            return None
        return int(raw)
    value = default_value if user_value is None else user_value
    return None if value is None else int(value)


EXPERIMENT_MODE = _cfg(
    "DAE_EXPERIMENT_MODE", USER_EXPERIMENT_MODE, "paper_repro_eval_only",
    lambda x: str(x).lower()
)
EXPERIMENT_MODE_SOURCE = (
    "env:DAE_EXPERIMENT_MODE"
    if ALLOW_ENV_OVERRIDES and "DAE_EXPERIMENT_MODE" in os.environ
    else "main.py:USER_EXPERIMENT_MODE"
)
CR_LIST = [4, 8, 16]
FAST_FINAL_EPOCHS_BY_CR = {4: 197, 8: 197, 16: 197}
BATCH_SIZE = 128
LR = 0.0005
SEED = 42
PATIENCE = 10            # 早停耐心值
WEIGHT_DECAY = 1e-4      # L2 正则化系数

# R20.1 创新轨道配置：保留，必要时把 USER_EXPERIMENT_MODE 改为 "r20_1" 切回。
R20_1_LOSS_CONFIG = {
    4:  {'epsilon_mse': 0.15, 'mse_weight_max': 0.18, 'phase_mix': 0.25, 'lambda_peak': 0.15},
    8:  {'epsilon_mse': 0.08, 'mse_weight_max': 0.10, 'phase_mix': 0.12, 'lambda_peak': 0.20},
    16: {'epsilon_mse': 0.03, 'mse_weight_max': 0.03, 'phase_mix': 0.03, 'lambda_peak': 0.25},
}

# 频域任务感知创新轨道：保持 urban8/WLS/fixed eval 协议与 paper_repro 一致，
# 但使用 FrequencySelectiveDAE 与 Fisher-weighted spectral reconstruction loss。
FREQ_TASK_LOSS_CONFIG = {
    cr: {'loss_mode': 'freq_task_mse', 'epsilon_mse': 1.0,
         'mse_weight_max': 1.0, 'phase_mix': 1.0, 'lambda_peak': 0.0,
         'selection_start_epoch': 1,
         'spectral_blend': USER_FREQ_TASK_SPECTRAL_BLEND,
         'spectral_power': USER_FREQ_TASK_SPECTRAL_POWER}
    for cr in CR_LIST
}

# FreqDAE v2：同一套权重用于所有 CR，避免用人为权重制造 CR 排序。
# 让瓶颈维度本身决定“多出来的容量”如何在重构和 pairwise TDOA 一致性之间分配。
FREQ_TASK_V2_LOSS_CONFIG = {
    cr: {'loss_mode': 'freq_task_pair', 'epsilon_mse': 1.0,
         'mse_weight_max': 1.0, 'phase_mix': 1.0,
         'corr_weight': USER_FREQ_TASK_V2_CORR_WEIGHT,
         'pair_phase_weight': USER_FREQ_TASK_V2_PHASE_WEIGHT,
         'lambda_peak': USER_FREQ_TASK_V2_PEAK_WEIGHT,
         'soft_peak_sigma': USER_FREQ_TASK_V2_SOFT_PEAK_SIGMA,
         'soft_peak_sharpness_weight': USER_FREQ_TASK_V2_PEAK_SHARPNESS_WEIGHT,
         'selection_start_epoch': 1,
         'spectral_blend': USER_FREQ_TASK_V2_SPECTRAL_BLEND,
         'spectral_power': USER_FREQ_TASK_V2_SPECTRAL_POWER}
    for cr in CR_LIST
}

# FreqDAE v3：共享嵌套 latent supernet。训练一次，按 CR4/8/16 前缀 mask
# 轮换优化；评估时同一权重在不同 active_cr 下输出对应压缩率的重构。
FREQ_TASK_V3_LOSS_CONFIG = {
    cr: {'loss_mode': 'freq_task_nested_pair', 'epsilon_mse': 1.0,
         'mse_weight_max': 1.0, 'phase_mix': 1.0,
         'corr_weight': USER_FREQ_TASK_V3_CORR_WEIGHT,
         'pair_phase_weight': USER_FREQ_TASK_V3_PHASE_WEIGHT,
         'lambda_peak': USER_FREQ_TASK_V3_PEAK_WEIGHT,
         'pair_task_weight': USER_FREQ_TASK_V3_TASK_HEAD_WEIGHT,
         'soft_peak_sigma': USER_FREQ_TASK_V3_SOFT_PEAK_SIGMA,
         'soft_peak_sharpness_weight': USER_FREQ_TASK_V3_PEAK_SHARPNESS_WEIGHT,
         'nested_crs': USER_FREQ_TASK_V3_NESTED_CRS,
         'selection_start_epoch': 1,
         'spectral_blend': USER_FREQ_TASK_V3_SPECTRAL_BLEND,
         'spectral_power': USER_FREQ_TASK_V3_SPECTRAL_POWER}
    for cr in CR_LIST
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
    MODEL_SOURCE_DIR = _cfg("DAE_MODEL_DIR", USER_MODEL_SOURCE_DIR, BASELINE_RESULT_DIR, str)
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
elif EXPERIMENT_MODE in FREQ_TASK_MODES:
    # 创新轨道：数据、归一化、评估和 baseline 口径与 paper_repro_v3.1 对齐；
    # 只替换模型结构与重构损失，避免污染复现基线。
    N_SAMPLES = 10000
    K_FOLDS = 0
    if EXPERIMENT_MODE in FREQ_TASK_EVAL_MODES:
        default_freq_epochs = 0
        DIAGNOSTICS_VERSION = "freq_task_dae_v1_eval_only"
    elif EXPERIMENT_MODE == "freq_task_fast_final":
        default_freq_epochs = int(USER_FREQ_TASK_FAST_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v1_fast120"
    elif EXPERIMENT_MODE == "freq_task_200_final":
        default_freq_epochs = int(USER_FREQ_TASK_FULL_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v1_200_final"
    else:
        default_freq_epochs = int(USER_FREQ_TASK_FULL_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v1"
    MAX_EPOCHS = int(default_freq_epochs if USER_FREQ_TASK_EPOCHS is None else USER_FREQ_TASK_EPOCHS)
    FAST_FINAL_EPOCHS_BY_CR = {cr: MAX_EPOCHS for cr in CR_LIST}
    EARLY_STOPPING = False
    RESTORE_BEST = False
    FINAL_RETRAIN = False
    FINAL_RETRAIN_EPOCHS = None
    TRAINING_PROTOCOL = "fixed_epoch_final_only"
    RESAMPLE_TRAIN_EACH_EPOCH = True
    RESAMPLE_INTERVAL = 5
    RUN_TRAINING = EXPERIMENT_MODE not in FREQ_TASK_EVAL_MODES
    MODEL_SOURCE_DIR = (
        _cfg("DAE_FREQ_TASK_MODEL_DIR", USER_FREQ_TASK_MODEL_SOURCE_DIR,
             FREQ_TASK_RESULT_DIR, str)
        if EXPERIMENT_MODE in FREQ_TASK_EVAL_MODES else None
    )
    if EXPERIMENT_MODE in FREQ_TASK_EVAL_MODES:
        TRAINING_PROTOCOL = "freq_task_eval_only_pretrained"
        RESAMPLE_TRAIN_EACH_EPOCH = False
    CORR_WEIGHT = 0.0
    LAMBDA_PEAK = 0.0
    BETA_FI = 0.0
    LOSS_CONFIG = FREQ_TASK_LOSS_CONFIG
    LOSS_MODE = "freq_task_mse"
    USE_ADAPTIVE_PEAK = False
    TRAIN_SNR_RANGE = (-10, 20)
    EVAL_SNR_RANGE = np.arange(-10, 21, 2)
    MONTE_CARLO_TRIALS = 200
    FIG2_SNR_LIST = [-10, 0, 10, 20]
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
elif EXPERIMENT_MODE in FREQ_TASK_V2_MODES:
    # 创新轨道 v2：数据/评估口径与 paper_repro_v3.1 对齐，但训练改为
    # urban LOS pair-aware objective，显式约束 cross-spectrum/GCC/TDOA 一致性。
    N_SAMPLES = 10000
    K_FOLDS = 0
    if EXPERIMENT_MODE in FREQ_TASK_V2_EVAL_MODES:
        default_freq_epochs = 0
        DIAGNOSTICS_VERSION = "freq_task_dae_v2_eval_only"
    elif EXPERIMENT_MODE == "freq_task_v2_fast_final":
        default_freq_epochs = int(USER_FREQ_TASK_V2_FAST_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v2_fast120"
    elif EXPERIMENT_MODE == "freq_task_v2_200_final":
        default_freq_epochs = int(USER_FREQ_TASK_V2_FULL_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v2_200_final"
    else:
        default_freq_epochs = int(USER_FREQ_TASK_V2_FULL_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v2"
    MAX_EPOCHS = int(default_freq_epochs if USER_FREQ_TASK_EPOCHS is None else USER_FREQ_TASK_EPOCHS)
    FAST_FINAL_EPOCHS_BY_CR = {cr: MAX_EPOCHS for cr in CR_LIST}
    EARLY_STOPPING = False
    RESTORE_BEST = False
    FINAL_RETRAIN = False
    FINAL_RETRAIN_EPOCHS = None
    TRAINING_PROTOCOL = "fixed_epoch_final_only"
    RESAMPLE_TRAIN_EACH_EPOCH = True
    RESAMPLE_INTERVAL = 5
    RUN_TRAINING = EXPERIMENT_MODE not in FREQ_TASK_V2_EVAL_MODES
    MODEL_SOURCE_DIR = (
        _cfg("DAE_FREQ_TASK_V2_MODEL_DIR", USER_FREQ_TASK_V2_MODEL_SOURCE_DIR,
             FREQ_TASK_RESULT_DIR, str)
        if EXPERIMENT_MODE in FREQ_TASK_V2_EVAL_MODES else None
    )
    if EXPERIMENT_MODE in FREQ_TASK_V2_EVAL_MODES:
        TRAINING_PROTOCOL = "freq_task_v2_eval_only_pretrained"
        RESAMPLE_TRAIN_EACH_EPOCH = False
    CORR_WEIGHT = USER_FREQ_TASK_V2_CORR_WEIGHT
    LAMBDA_PEAK = USER_FREQ_TASK_V2_PEAK_WEIGHT
    BETA_FI = 0.0
    LOSS_CONFIG = FREQ_TASK_V2_LOSS_CONFIG
    LOSS_MODE = "freq_task_pair"
    USE_ADAPTIVE_PEAK = False
    TRAIN_SNR_RANGE = (-10, 20)
    EVAL_SNR_RANGE = np.arange(-10, 21, 2)
    MONTE_CARLO_TRIALS = 200
    FIG2_SNR_LIST = [-10, 0, 10, 20]
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
elif EXPERIMENT_MODE in FREQ_TASK_V3_MODES:
    # 创新轨道 v3：共享 nested latent supernet + waveform reconstruction head
    # + pairwise TDOA task head。仍保持 paper_repro_v3.1 的数据/评估协议。
    N_SAMPLES = 10000
    K_FOLDS = 0
    if EXPERIMENT_MODE in FREQ_TASK_V3_EVAL_MODES:
        default_freq_epochs = 0
        DIAGNOSTICS_VERSION = "freq_task_dae_v3_nested_eval_only"
    elif EXPERIMENT_MODE == "freq_task_v3_fast_final":
        default_freq_epochs = int(USER_FREQ_TASK_V3_FAST_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v3_nested_fast"
    elif EXPERIMENT_MODE == "freq_task_v3_200_final":
        default_freq_epochs = int(USER_FREQ_TASK_V3_FULL_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v3_nested_200_final"
    else:
        default_freq_epochs = int(USER_FREQ_TASK_V3_FULL_EPOCHS)
        DIAGNOSTICS_VERSION = "freq_task_dae_v3_nested"
    MAX_EPOCHS = int(default_freq_epochs if USER_FREQ_TASK_EPOCHS is None else USER_FREQ_TASK_EPOCHS)
    FAST_FINAL_EPOCHS_BY_CR = {cr: MAX_EPOCHS for cr in CR_LIST}
    EARLY_STOPPING = False
    RESTORE_BEST = False
    FINAL_RETRAIN = False
    FINAL_RETRAIN_EPOCHS = None
    TRAINING_PROTOCOL = "fixed_epoch_final_only"
    RESAMPLE_TRAIN_EACH_EPOCH = True
    RESAMPLE_INTERVAL = 5
    RUN_TRAINING = EXPERIMENT_MODE not in FREQ_TASK_V3_EVAL_MODES
    MODEL_SOURCE_DIR = (
        _cfg("DAE_FREQ_TASK_V3_MODEL_DIR", USER_FREQ_TASK_V3_MODEL_SOURCE_DIR,
             FREQ_TASK_RESULT_DIR, str)
        if EXPERIMENT_MODE in FREQ_TASK_V3_EVAL_MODES else None
    )
    if EXPERIMENT_MODE in FREQ_TASK_V3_EVAL_MODES:
        TRAINING_PROTOCOL = "freq_task_v3_eval_only_pretrained"
        RESAMPLE_TRAIN_EACH_EPOCH = False
    CORR_WEIGHT = USER_FREQ_TASK_V3_CORR_WEIGHT
    LAMBDA_PEAK = USER_FREQ_TASK_V3_PEAK_WEIGHT
    BETA_FI = 0.0
    LOSS_CONFIG = FREQ_TASK_V3_LOSS_CONFIG
    LOSS_MODE = "freq_task_nested_pair"
    USE_ADAPTIVE_PEAK = False
    TRAIN_SNR_RANGE = (-10, 20)
    EVAL_SNR_RANGE = np.arange(-10, 21, 2)
    MONTE_CARLO_TRIALS = 200
    FIG2_SNR_LIST = [-10, 0, 10, 20]
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
                     "'paper_repro_eval_only', 'freq_task_eval_only', "
                     "'innovation_eval_only', 'freq_task', 'freq_task_fast_final', "
                     "'freq_task_200_final', 'freq_task_v2', "
                     "'freq_task_v2_fast_final', 'freq_task_v2_200_final', "
                     "'freq_task_v2_eval_only', 'freq_task_v3', "
                     "'freq_task_v3_fast_final', 'freq_task_v3_200_final', "
                     "'freq_task_v3_eval_only', "
                     "or 'r20_1'")

REFERENCE_MODEL_DIR = _cfg(
    "DAE_REFERENCE_MODEL_DIR", USER_REPRO_REFERENCE_MODEL_DIR,
    BASELINE_RESULT_DIR, str
)
STATIC_BASELINE_CACHE_DIR = _cfg(
    "DAE_STATIC_BASELINE_CACHE_DIR",
    USER_STATIC_BASELINE_CACHE_DIR,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "运行结果", USER_STATIC_BASELINE_CACHE_RESULT_ID),
    str,
)
REUSE_STATIC_BASELINE_CACHE = (
    bool(USER_REUSE_STATIC_BASELINE_CACHE)
    and EXPERIMENT_MODE in INNOVATION_MODES
)

SNR_THRESHOLD = 0.0          # sigmoid 中心点 SNR (dB)
LAMBDA_TEMPERATURE = 5.0     # sigmoid 温度参数
FIG2_DIAGNOSTIC_TRIALS = 128  # extra samples saved in plot_data.pkl for Fig2 diagnostics
RUN_TRADITIONAL_BASELINES = (
    _cfg_bool(
        "DAE_RUN_TRADITIONAL_BASELINES",
        USER_RUN_TRADITIONAL_BASELINES,
        EXPERIMENT_MODE in PAPER_REPRO_MODES
    )
)
BASELINE_CR = _cfg("DAE_BASELINE_CR", USER_BASELINE_CR, 16, int)
BASELINE_DFT_MODE = _cfg("DAE_BASELINE_DFT_MODE", USER_BASELINE_DFT_MODE, "uniform", str)
BASELINE_DFT_DIRECT_SOURCE = _cfg(
    "DAE_BASELINE_DFT_DIRECT_SOURCE", USER_BASELINE_DFT_DIRECT_SOURCE,
    "DFT-train-power", str
)
BASELINE_HADAMARD_MODE = _cfg(
    "DAE_BASELINE_HADAMARD_MODE", USER_BASELINE_HADAMARD_MODE, "salari", str
)
BASELINE_PCA_TRAIN_SOURCE = _cfg(
    "DAE_BASELINE_PCA_TRAIN_SOURCE", USER_BASELINE_PCA_TRAIN_SOURCE, "noisy", str
)
BASELINE_PCA_FIXED_SNR_DB = _cfg(
    "DAE_BASELINE_PCA_FIXED_SNR_DB", USER_BASELINE_PCA_FIXED_SNR_DB, 0.0, float
)
BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS = _cfg_bool(
    "DAE_BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS",
    USER_BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS,
    False
)
BASELINE_INCLUDE_LEGACY_VARIANTS = _cfg_bool(
    "DAE_BASELINE_INCLUDE_LEGACY_VARIANTS",
    USER_BASELINE_INCLUDE_LEGACY_VARIANTS,
    False
)
BASELINE_RANDOM_VARIANT_SEEDS = _cfg(
    "DAE_BASELINE_RANDOM_VARIANT_SEEDS", USER_BASELINE_RANDOM_VARIANT_SEEDS, 0, int
)
RUN_TASK_AWARE_BASELINES = _cfg_bool(
    "DAE_RUN_TASK_AWARE_BASELINES", USER_RUN_TASK_AWARE_BASELINES, True
)
RUN_STRONG_BASELINES = _cfg_bool(
    "DAE_RUN_STRONG_BASELINES", USER_RUN_STRONG_BASELINES, True
)
STRONG_INCLUDE_PHASE_SUPERPOSITION = _cfg_bool(
    "DAE_STRONG_INCLUDE_PHASE_SUPERPOSITION",
    USER_STRONG_INCLUDE_PHASE_SUPERPOSITION,
    False
)
RUN_STRONG_DIAGNOSTIC_SUPPLEMENT = _cfg_bool(
    "DAE_RUN_STRONG_DIAGNOSTIC_SUPPLEMENT",
    USER_RUN_STRONG_DIAGNOSTIC_SUPPLEMENT,
    True
)
RUN_GEOHYBRID_BASELINES = _cfg_bool(
    "DAE_RUN_GEOHYBRID_BASELINES",
    USER_RUN_GEOHYBRID_BASELINES,
    True
)
EVAL_ONLY_COPY_MODELS = _cfg_bool(
    "DAE_EVAL_ONLY_COPY_MODELS", USER_EVAL_ONLY_COPY_MODELS, False
)
FIG6_SHOW_ZOOM_INSET = _cfg_bool(
    "DAE_FIG6_SHOW_ZOOM_INSET", USER_FIG6_SHOW_ZOOM_INSET, True
)
FIG6_ZOOM_SNR_MIN = _cfg("DAE_FIG6_ZOOM_SNR_MIN", USER_FIG6_ZOOM_SNR_MIN, 8.0, float)
EXPORT_FIG6_SVG_IN_TABLES = _cfg_bool(
    "DAE_EXPORT_FIG6_SVG_IN_TABLES", USER_EXPORT_FIG6_SVG_IN_TABLES, False
)
USE_PHYSICAL_TDOA_LAG_GATE = _cfg_bool(
    "DAE_USE_PHYSICAL_TDOA_LAG_GATE", USER_USE_PHYSICAL_TDOA_LAG_GATE, True
)
TDOA_LAG_LIMIT_SAMPLES = _cfg_optional_int(
    "DAE_TDOA_LAG_LIMIT_SAMPLES", USER_TDOA_LAG_LIMIT_SAMPLES, None
)
TDOA_LAG_MARGIN_SAMPLES = _cfg(
    "DAE_TDOA_LAG_MARGIN_SAMPLES", USER_TDOA_LAG_MARGIN_SAMPLES, 4, int
)
EXPORT_METHOD_ZOOM_FIGURES = _cfg_bool(
    "DAE_EXPORT_METHOD_ZOOM_FIGURES", USER_EXPORT_METHOD_ZOOM_FIGURES, True
)
METHOD_ZOOM_LOW_SNR_MAX = _cfg(
    "DAE_METHOD_ZOOM_LOW_SNR_MAX", USER_METHOD_ZOOM_LOW_SNR_MAX, 0.0, float
)
METHOD_ZOOM_HIGH_SNR_MIN = _cfg(
    "DAE_METHOD_ZOOM_HIGH_SNR_MIN", USER_METHOD_ZOOM_HIGH_SNR_MIN, 8.0, float
)
RUN_LOCALIZER_ABLATION = _cfg_bool(
    "DAE_RUN_LOCALIZER_ABLATION", USER_RUN_LOCALIZER_ABLATION, False
)
LOCALIZER_ABLATION_SNR_RANGE = np.asarray(USER_LOCALIZER_ABLATION_SNR_RANGE, dtype=float)
LOCALIZER_ABLATION_ESTIMATORS = list(USER_LOCALIZER_ABLATION_ESTIMATORS)
LOCALIZER_ABLATION_METHODS = list(dict.fromkeys(USER_LOCALIZER_ABLATION_METHODS))

# Lightweight smoke-test overrides. None means keep the formal mode default.
N_SAMPLES = _cfg("DAE_N_SAMPLES", USER_N_SAMPLES, N_SAMPLES, int)
BATCH_SIZE = _cfg("DAE_BATCH_SIZE", USER_BATCH_SIZE, BATCH_SIZE, int)
K_FOLDS = _cfg("DAE_K_FOLDS", USER_K_FOLDS, K_FOLDS, int)
MAX_EPOCHS = _cfg("DAE_MAX_EPOCHS", USER_MAX_EPOCHS, MAX_EPOCHS, int)
MONTE_CARLO_TRIALS = _cfg(
    "DAE_MONTE_CARLO_TRIALS", USER_MONTE_CARLO_TRIALS, MONTE_CARLO_TRIALS, int
)
FIG2_DIAGNOSTIC_TRIALS = _cfg(
    "DAE_FIG2_DIAGNOSTIC_TRIALS", USER_FIG2_DIAGNOSTIC_TRIALS,
    FIG2_DIAGNOSTIC_TRIALS, int
)
N_FIXED_CHANNELS = _cfg("DAE_N_FIXED_CHANNELS", USER_N_FIXED_CHANNELS, N_FIXED_CHANNELS, int)
BASELINE_PCA_SAMPLES = _cfg(
    "DAE_BASELINE_PCA_SAMPLES", USER_BASELINE_PCA_SAMPLES, N_SAMPLES, int
)
if USER_FAST_FINAL_EPOCHS is not None or (
    ALLOW_ENV_OVERRIDES and "DAE_FAST_FINAL_EPOCHS" in os.environ
):
    _fast_ep = _cfg("DAE_FAST_FINAL_EPOCHS", USER_FAST_FINAL_EPOCHS, MAX_EPOCHS, int)
    FAST_FINAL_EPOCHS_BY_CR = {cr: _fast_ep for cr in CR_LIST}
    if TRAINING_PROTOCOL == "fixed_epoch_final_only":
        MAX_EPOCHS = _fast_ep
NO_SHOW = _cfg_bool("DAE_NO_SHOW", USER_NO_SHOW, False)
if EXPERIMENT_MODE in FREQ_TASK_V3_MODES:
    ACTIVE_MODEL_FACTORY = NestedFrequencyPairwiseDAE
    ACTIVE_MODEL_NAME = "NestedFrequencyPairwiseDAE"
    INNOVATION_LABEL_PREFIX = "FreqDAE-v3"
elif EXPERIMENT_MODE in FREQ_TASK_V2_MODES:
    ACTIVE_MODEL_FACTORY = FrequencyPairwiseDAE
    ACTIVE_MODEL_NAME = "FrequencyPairwiseDAE"
    INNOVATION_LABEL_PREFIX = "FreqDAE-v2"
elif EXPERIMENT_MODE in FREQ_TASK_MODES:
    ACTIVE_MODEL_FACTORY = FrequencySelectiveDAE
    ACTIVE_MODEL_NAME = "FrequencySelectiveDAE"
    INNOVATION_LABEL_PREFIX = "FreqDAE"
else:
    ACTIVE_MODEL_FACTORY = DAE
    ACTIVE_MODEL_NAME = "DAE"
    INNOVATION_LABEL_PREFIX = "DAE"
INCLUDE_REPRO_REFERENCE = (
    bool(USER_INCLUDE_REPRO_REFERENCE) and EXPERIMENT_MODE in INNOVATION_MODES
)

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
print(f"[Config] EXPERIMENT_MODE={EXPERIMENT_MODE} "
      f"(source={EXPERIMENT_MODE_SOURCE}) | LOSS_MODE={LOSS_MODE}")
print(f"[Config] experiment_profile={USER_EXPERIMENT_PROFILE} | "
      f"applied_profile_keys={list(APPLIED_EXPERIMENT_PROFILE.keys())}")
print(f"[Config] active_model={ACTIVE_MODEL_NAME} | "
      f"include_repro_reference={INCLUDE_REPRO_REFERENCE}")
print(f"[Config] allow_env_overrides={ALLOW_ENV_OVERRIDES} | no_show={NO_SHOW}")
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
if INCLUDE_REPRO_REFERENCE:
    print(f"[Config] reference_model_dir={REFERENCE_MODEL_DIR}")
print(f"[Config] traditional_baselines={RUN_TRADITIONAL_BASELINES} | "
      f"baseline_cr={BASELINE_CR} | pca_samples={BASELINE_PCA_SAMPLES} | "
      f"dft_mode={BASELINE_DFT_MODE} | hadamard_mode={BASELINE_HADAMARD_MODE} | "
      f"pca_source={BASELINE_PCA_TRAIN_SOURCE} | "
      f"diagnostic_variants={BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS} | "
      f"legacy_variants={BASELINE_INCLUDE_LEGACY_VARIANTS} | "
      f"random_variant_seeds={BASELINE_RANDOM_VARIANT_SEEDS} | "
      f"task_aware_baselines={RUN_TASK_AWARE_BASELINES} | "
      f"strong_baselines={RUN_STRONG_BASELINES} | "
      f"phase_superposition={STRONG_INCLUDE_PHASE_SUPERPOSITION} | "
      f"geohybrid={RUN_GEOHYBRID_BASELINES} | "
      f"strong_diagnostic_supplement={RUN_STRONG_DIAGNOSTIC_SUPPLEMENT} | "
      f"eval_only_copy_models={EVAL_ONLY_COPY_MODELS} | "
      f"fig6_zoom={FIG6_SHOW_ZOOM_INSET} | export_fig6_svg_in_tables={EXPORT_FIG6_SVG_IN_TABLES}")
print(f"[Config] dft_direct_source={BASELINE_DFT_DIRECT_SOURCE} | "
      f"physical_lag_gate={USE_PHYSICAL_TDOA_LAG_GATE} | "
      f"lag_limit_override={TDOA_LAG_LIMIT_SAMPLES} | lag_margin={TDOA_LAG_MARGIN_SAMPLES} | "
      f"method_zoom_figures={EXPORT_METHOD_ZOOM_FIGURES} | "
      f"low_zoom<= {METHOD_ZOOM_LOW_SNR_MAX:g} dB | high_zoom>= {METHOD_ZOOM_HIGH_SNR_MIN:g} dB")
print(f"[Config] reuse_static_baseline_cache={REUSE_STATIC_BASELINE_CACHE} | "
      f"static_baseline_cache_dir={STATIC_BASELINE_CACHE_DIR}")
print(f"[Config] localizer_ablation={RUN_LOCALIZER_ABLATION} | "
      f"localizer_estimators={LOCALIZER_ABLATION_ESTIMATORS} | "
      f"localizer_methods={LOCALIZER_ABLATION_METHODS} | "
      f"localizer_snr={list(LOCALIZER_ABLATION_SNR_RANGE)}")
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


def unique_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def load_repro_reference_models(source_dir, cr_list, device):
    """Load frozen Chen-style DAE models for innovation-mode comparison."""
    reference = {}
    for cr in cr_list:
        model_path = os.path.join(source_dir, f"model_cr{cr}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"reference DAE model not found: {model_path}")
        model = DAE(cr=cr).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        reference[cr] = model
        print(f"[Reference] Loaded Chen-DAE CR={cr}: {model_path}")
    return reference


def high_frequency_dft_bins(signal_len, cr):
    n_bins = max(1, (2 * int(signal_len) // int(cr)) // 2)
    freqs = np.fft.fftshift(np.fft.fftfreq(int(signal_len)))
    return np.argsort(-np.abs(freqs))[:n_bins].astype(int)


def compute_physical_tdoa_lag_limit(simulator):
    if not USE_PHYSICAL_TDOA_LAG_GATE:
        return None
    if TDOA_LAG_LIMIT_SAMPLES is not None:
        return int(TDOA_LAG_LIMIT_SAMPLES)
    if not hasattr(simulator, "area_size") or simulator.area_size is None:
        return None
    area = np.asarray(simulator.area_size, dtype=float)
    if area.size == 0:
        return None
    area_diag = float(np.linalg.norm(area))
    sample_distance = float(simulator.c / simulator.fs)
    return int(np.ceil(area_diag / sample_distance) + int(TDOA_LAG_MARGIN_SAMPLES))


def method_zoom_orders(baseline_cr, had_main):
    core = [
        "Raw", f"DAE-CR{baseline_cr}", "DFT", "DFT-SCS-lite",
        "DFT-Fisher-Direct", "GeoHybrid-DFT",
        "GeoAmbi-DFT-Direct", "GeoAmbi-DFT-AML",
        had_main, "PCA"
    ]
    return unique_order(core)


def localizer_estimator_label(estimator):
    labels = {
        "all_pair_wls": "WLS",
        "all_pair_wls_grid": "Grid",
        "all_pair_wls_pruned": "Pruned",
        "all_pair_wls_grid_pruned": "Grid+Pruned",
    }
    return labels.get(str(estimator), str(estimator))


def combine_localizer_ablation_results(ablation_runs, method_order, config_updates):
    """Merge per-localizer runs into one Fig10-style method-comparison result."""
    if not ablation_runs:
        return None
    combined_methods = {}
    combined_diagnostics = {}
    combined_outliers = {}
    combined_order = []
    snr_ref = None
    for estimator, method_data in ablation_runs:
        if method_data is None:
            continue
        results, snr_range = method_data
        if snr_ref is None:
            snr_ref = snr_range
        suffix = localizer_estimator_label(estimator)
        methods = results.get("methods", {})
        diagnostics = results.get("method_diagnostics", {})
        outliers = results.get("outlier_details", {})
        for method in method_order:
            if method not in methods:
                continue
            label = f"{method}|{suffix}"
            combined_methods[label] = methods[method]
            combined_diagnostics[label] = diagnostics.get(method, {})
            combined_outliers[label] = outliers.get(method, [])
            combined_order.append(label)
    if not combined_methods:
        return None
    base_results = ablation_runs[0][1][0]
    config = dict(base_results.get("config", {}))
    config.update(config_updates)
    config["localizer_ablation_estimators"] = [
        estimator for estimator, _ in ablation_runs
    ]
    config["localizer_ablation_estimator_labels"] = {
        estimator: localizer_estimator_label(estimator)
        for estimator, _ in ablation_runs
    }
    config["localizer_ablation_base_methods"] = method_order
    config["strong_method_order"] = combined_order
    config["strong_zoom_method_order"] = combined_order
    combined = {
        "metric": base_results.get("metric", "localization_m"),
        "title": config.get("strong_title", "Figure 10: Localizer Robustness Ablation"),
        "method_order": combined_order,
        "methods": combined_methods,
        "method_diagnostics": combined_diagnostics,
        "outlier_details": combined_outliers,
        "config": config,
    }
    return combined, snr_ref


def method_data_contains(method_data, required_methods):
    """Return True if a saved method-comparison tuple contains all labels."""
    if method_data is None:
        return False
    try:
        results, _ = method_data
    except Exception:
        return False
    methods = set(results.get("methods", {}).keys())
    methods.update(results.get("method_order", []))
    return all(str(name) in methods for name in required_methods)


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
    reference_models_dict = {}
    source_plot_data = None
    source_cv_results_dict = {}
    static_baseline_cache = None
    if not RUN_TRAINING:
        source_pkl = os.path.join(MODEL_SOURCE_DIR, "plot_data.pkl")
        if not os.path.exists(source_pkl):
            raise FileNotFoundError(f"eval_only requires source plot_data.pkl: {source_pkl}")
        with open(source_pkl, 'rb') as f:
            source_plot_data = pickle.load(f)
            source_cv_results_dict = source_plot_data.get('cv_results_dict', {})
        print(f"[EvalOnly] Loaded source plot data: {source_pkl}")
    if REUSE_STATIC_BASELINE_CACHE:
        try:
            static_baseline_cache = load_static_baseline_cache(STATIC_BASELINE_CACHE_DIR)
            print(f"[BaselineCache] Loaded static Fig6-Fig8 cache: "
                  f"{static_baseline_cache['source_dir']}")
            required_cache_methods = []
            if RUN_STRONG_BASELINES:
                required_cache_methods.append("GeoAmbi-DFT-AML")
                if RUN_GEOHYBRID_BASELINES:
                    required_cache_methods.append("GeoHybrid-DFT")
            if required_cache_methods and not method_data_contains(
                    static_baseline_cache.get("baseline_all_results"),
                    required_cache_methods):
                print("[BaselineCache] Cache is stale for current strong-baseline set; "
                      f"missing {required_cache_methods}. Baselines will be recomputed.")
                static_baseline_cache = None
        except Exception as exc:
            static_baseline_cache = None
            print(f"[BaselineCache] Cache unavailable, will recompute baselines: {exc}")

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
    method_tdoa_lag_limit = compute_physical_tdoa_lag_limit(sim)
    if USE_PHYSICAL_TDOA_LAG_GATE:
        print(f"[TDOA Gate] method comparison lag_limit={method_tdoa_lag_limit} samples "
              f"(sample_distance={sim.c / sim.fs:.3f} m)")
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
    shared_v3_state = None
    shared_v3_cv_results = None
    shared_v3_train_cr = min(CR_LIST)
    for cr in CR_LIST:
        if RUN_TRAINING and LOSS_MODE == "freq_task_nested_pair" and shared_v3_state is not None:
            model = ACTIVE_MODEL_FACTORY(cr=cr).to(DEVICE)
            model.load_state_dict(shared_v3_state)
            if hasattr(model, "set_active_cr"):
                model.set_active_cr(cr)
            model.eval()
            cv_results = dict(shared_v3_cv_results)
            cv_results['shared_nested_supernet'] = True
            cv_results['shared_supernet_train_cr'] = shared_v3_train_cr
            cv_results['active_eval_cr'] = cr
            print(f"[V3 Nested] Reusing shared supernet trained at CR={shared_v3_train_cr}; "
                  f"active mask set to CR={cr}.")
        elif RUN_TRAINING:
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
                model_factory=ACTIVE_MODEL_FACTORY,
                model_name=ACTIVE_MODEL_NAME,
                spectral_blend=LOSS_CONFIG[cr].get('spectral_blend', USER_FREQ_TASK_SPECTRAL_BLEND),
                spectral_power=LOSS_CONFIG[cr].get('spectral_power', USER_FREQ_TASK_SPECTRAL_POWER),
                pair_phase_weight=LOSS_CONFIG[cr].get('pair_phase_weight', 0.0),
                soft_peak_sigma=LOSS_CONFIG[cr].get('soft_peak_sigma', 1.5),
                soft_peak_sharpness_weight=LOSS_CONFIG[cr].get('soft_peak_sharpness_weight', 0.0),
                pair_task_weight=LOSS_CONFIG[cr].get('pair_task_weight', 0.0),
                nested_crs=LOSS_CONFIG[cr].get('nested_crs', USER_FREQ_TASK_V3_NESTED_CRS),
            )
            if LOSS_MODE == "freq_task_nested_pair":
                if hasattr(model, "set_active_cr"):
                    model.set_active_cr(cr)
                shared_v3_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                shared_v3_cv_results = dict(cv_results)
                shared_v3_cv_results['shared_nested_supernet'] = True
                shared_v3_cv_results['shared_supernet_train_cr'] = cr
                shared_v3_train_cr = cr
        else:
            model_path = os.path.join(MODEL_SOURCE_DIR, f"model_cr{cr}.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"eval_only requires source model: {model_path}")
            model = ACTIVE_MODEL_FACTORY(cr=cr).to(DEVICE)
            state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state)
            if hasattr(model, "set_active_cr"):
                model.set_active_cr(cr)
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

        # 每个CR训练完立即保存模型；eval_only 默认只引用源模型，避免重复复制权重。
        model_path = os.path.join(RESULT_DIR, f"model_cr{cr}.pt")
        if RUN_TRAINING or EVAL_ONLY_COPY_MODELS:
            torch.save(model.state_dict(), model_path)
            print(f"[Saved] {model_path}")
        else:
            print(f"[EvalOnly] Skip copying CR={cr} model to result dir "
                  f"(set USER_EVAL_ONLY_COPY_MODELS=True to copy).")

    # 3. Monte Carlo 评估
    print("\n" + "=" * 40)
    if EVALUATION_MODE == "urban_localization":
        exp = UrbanLocalizationExperiment(
            models_dict, sim, DEVICE, seed=current_seed,
            snr_range=EVAL_SNR_RANGE, num_trials=MONTE_CARLO_TRIALS,
            sub_sample=TDOA_SUB_SAMPLE, use_los_only=USE_LOS_ONLY,
            batch_size=BATCH_SIZE, fixed_eval_set=FIXED_EVAL_SET,
            estimator=LOCALIZATION_ESTIMATOR,
            tdoa_lag_limit_samples=method_tdoa_lag_limit,
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
            elif LOSS_MODE == "freq_task_mse":
                mse_title = (f'CR={cr} | Freq-task recon '
                             f'(blend={LOSS_CONFIG[cr]["spectral_blend"]})')
            elif LOSS_MODE == "freq_task_pair":
                mse_title = (f'CR={cr} | Freq-task pair '
                             f'(corr={LOSS_CONFIG[cr]["corr_weight"]}, '
                             f'xphase={LOSS_CONFIG[cr]["pair_phase_weight"]})')
            elif LOSS_MODE == "freq_task_nested_pair":
                mse_title = (f'CR={cr} | Nested Freq-task pair '
                             f'(task={LOSS_CONFIG[cr]["pair_task_weight"]}, '
                             f'active={cr})')
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
                if LOSS_MODE in ("freq_task_pair", "freq_task_nested_pair"):
                    pk_title = (
                        f'CR={cr} | Soft TDOA Peak Loss '
                        f'(λ={LOSS_CONFIG[cr]["lambda_peak"]})'
                    )
                else:
                    pk_title = (
                        f'CR={cr} | SNR-Adaptive Peak Loss '
                        f'(λp_max={LOSS_CONFIG[cr]["lambda_peak"]})'
                    )
                ax_pk.set_title(pk_title, fontsize=9, fontweight='bold')
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
        elif LOSS_MODE == "freq_task_mse":
            cfg_str = (f"Frequency-task DAE: {ACTIVE_MODEL_NAME}, "
                       f"spectral blend={USER_FREQ_TASK_SPECTRAL_BLEND}, "
                       f"epochs={MAX_EPOCHS}")
        elif LOSS_MODE == "freq_task_pair":
            cfg_str = (f"Frequency-task pair DAE v2: {ACTIVE_MODEL_NAME}, "
                       f"corr={USER_FREQ_TASK_V2_CORR_WEIGHT}, "
                       f"xphase={USER_FREQ_TASK_V2_PHASE_WEIGHT}, "
                       f"peak={USER_FREQ_TASK_V2_PEAK_WEIGHT}, "
                       f"epochs={MAX_EPOCHS}")
        elif LOSS_MODE == "freq_task_nested_pair":
            cfg_str = (f"Nested frequency-task DAE v3: {ACTIVE_MODEL_NAME}, "
                       f"corr={USER_FREQ_TASK_V3_CORR_WEIGHT}, "
                       f"xphase={USER_FREQ_TASK_V3_PHASE_WEIGHT}, "
                       f"peak={USER_FREQ_TASK_V3_PEAK_WEIGHT}, "
                       f"task={USER_FREQ_TASK_V3_TASK_HEAD_WEIGHT}, "
                       f"epochs={MAX_EPOCHS}")
        else:
            cfg_str = ", ".join([
                f"CR{cr}: w={LOSS_CONFIG[cr]['mse_weight_max']}, "
                f"mix={LOSS_CONFIG[cr]['phase_mix']}, λp={LOSS_CONFIG[cr]['lambda_peak']}"
                for cr in CR_LIST
            ])
        fig1_protocol_label = (
            "Final-Only Training" if TRAINING_PROTOCOL == "fixed_epoch_final_only"
            else f"{K_FOLDS}-Fold CV Training"
        )
        cv_title = (f'Figure 1: {fig1_protocol_label} Dynamics  ({cfg_str})')
        fig_cv.suptitle(cv_title, fontsize=12, y=0.995)
        fig_cv.tight_layout()
        save_figure(fig_cv, "Fig1_CV_training_curves")

        # 生成 Figure 2 的绘图数据（所有CR叠加）
        snr_data_all = generate_snr_data_all(
            models_dict, sim, DEVICE, snr_list=FIG2_SNR_LIST,
            diagnostic_trials=FIG2_DIAGNOSTIC_TRIALS
        )

        fig6_results = None
        fig7_results = None
        fig8_results = None
        fig8_supp_results = None
        fig9_results = None
        fig9_focus_results = None
        fig10_localizer_results = None
        baseline_all_results = None
        traditional_baseline_meta = None
        if (RUN_TRADITIONAL_BASELINES and static_baseline_cache is not None
                and EVALUATION_MODE == "urban_localization"):
            print("\n" + "=" * 40)
            print("[BaselineCache] Reusing static Fig6-Fig8 baseline results; "
                  "only Fig9 innovation curves will be recomputed.")
            fig6_results = static_baseline_cache.get("fig6_results")
            fig7_results = static_baseline_cache.get("fig7_results")
            fig8_results = static_baseline_cache.get("fig8_results")
            fig8_supp_results = static_baseline_cache.get("fig8_supp_results")
            baseline_all_results = static_baseline_cache.get("baseline_all_results")
            traditional_baseline_meta = (
                static_baseline_cache.get("traditional_baseline_meta") or {}
            )
            had_main = traditional_baseline_meta.get("hadamard_main_label", "Hadamard")
            cache_config = {}
            if baseline_all_results is not None:
                cache_config = dict(baseline_all_results[0].get("config", {}))
            common_config = {
                **cache_config,
                "baseline_cache_reused": True,
                "baseline_cache_dir": static_baseline_cache.get("source_dir"),
                "innovation_curves_recomputed": True,
                "table_prefix": "fig9",
                "reference_model_dir": REFERENCE_MODEL_DIR,
                "innovation_model": ACTIVE_MODEL_NAME,
            }

            if EXPERIMENT_MODE in INNOVATION_MODES:
                if INCLUDE_REPRO_REFERENCE and not reference_models_dict:
                    reference_models_dict = load_repro_reference_models(
                        REFERENCE_MODEL_DIR, CR_LIST, DEVICE
                    )
                fig9_models = {}
                for ref_cr in CR_LIST:
                    if ref_cr in reference_models_dict:
                        fig9_models[f"DAE-CR{ref_cr}"] = reference_models_dict[ref_cr]
                for cr in CR_LIST:
                    fig9_models[f"{INNOVATION_LABEL_PREFIX}-CR{cr}"] = models_dict[cr]

                fig9_fresh_results = run_urban_method_comparison(
                    fig9_models, sim, DEVICE, seed=current_seed,
                    snr_range=EVAL_SNR_RANGE, num_trials=MONTE_CARLO_TRIALS,
                    sub_sample=TDOA_SUB_SAMPLE, use_los_only=USE_LOS_ONLY,
                    batch_size=BATCH_SIZE, fixed_eval_set=FIXED_EVAL_SET,
                    estimator=LOCALIZATION_ESTIMATOR,
                    title=f"Innovation_Fresh_Comparison_{ACTIVE_MODEL_NAME}",
                    direct_estimators={},
                    tdoa_lag_limit_samples=method_tdoa_lag_limit,
                )
                merged_for_fig9 = merge_method_comparison_results(
                    baseline_all_results, fig9_fresh_results,
                    overlay_first=True,
                    title="Cached Baselines + Fresh Innovation Curves",
                    config_updates=common_config,
                )

                fig9_order = unique_order([
                    "Raw", "DAE-CR4", "DAE-CR8", "DAE-CR16",
                    f"{INNOVATION_LABEL_PREFIX}-CR4",
                    f"{INNOVATION_LABEL_PREFIX}-CR8",
                    f"{INNOVATION_LABEL_PREFIX}-CR16",
                    "DFT", "DFT-Fisher-Direct",
                    "Cao2017-DFT-AML", "GeoHybrid-DFT", "GeoAmbi-DFT-AML",
                    "Zhai-CRLB-Decimation",
                    had_main, "PCA"
                ])
                fig9_keep = unique_order(["Raw", "Clean", "Geometry"] + fig9_order)
                fig9_results = filter_method_comparison_data(
                    merged_for_fig9, fig9_keep,
                    {
                        **common_config,
                        "strong_method_order": fig9_order,
                        "strong_title": (
                            "Figure 9: Frequency-Task DAE vs Frozen Chen-DAE "
                            "References and Cached Strong Baselines"
                        ),
                        "strong_figure_filename": (
                            f"Fig9_FrequencyTaskDAE_vs_Repro_CR{BASELINE_CR}"
                        ),
                        "strong_zoom_method_order": fig9_order,
                        "strong_zoom_figure_filename": (
                            f"Fig9_FrequencyTaskDAE_vs_Repro_CR{BASELINE_CR}_SNR_Zooms"
                        ),
                        "table_prefix": "fig9",
                    },
                )
                fig9_focus_order = unique_order([
                    "Raw", "DAE-CR4", "DAE-CR8", "DAE-CR16",
                    f"{INNOVATION_LABEL_PREFIX}-CR4",
                    f"{INNOVATION_LABEL_PREFIX}-CR8",
                    f"{INNOVATION_LABEL_PREFIX}-CR16",
                ])
                fig9_focus_keep = unique_order(["Raw", "Clean", "Geometry"] + fig9_focus_order)
                fig9_focus_results = filter_method_comparison_data(
                    fig9_fresh_results, fig9_focus_keep,
                    {
                        **common_config,
                        "strong_method_order": fig9_focus_order,
                        "strong_title": (
                            "Figure 9 Focus: Frozen Chen-DAE vs Frequency-Task DAE"
                        ),
                        "strong_figure_filename": "Fig9_Focused_DAE_CR_Comparison",
                        "strong_zoom_method_order": fig9_focus_order,
                        "strong_zoom_figure_filename": (
                            "Fig9_Focused_DAE_CR_Comparison_SNR_Zooms"
                        ),
                        "table_prefix": "fig9_focus",
                    },
                )
        elif RUN_TRADITIONAL_BASELINES:
            if EVALUATION_MODE != "urban_localization":
                print("[Fig6] Skipped: traditional baseline comparison requires urban localization.")
            elif BASELINE_CR not in models_dict:
                print(f"[Fig6] Skipped: DAE model for CR={BASELINE_CR} is unavailable.")
            else:
                print("\n" + "=" * 40)
                print(f"[Fig6] Building traditional baselines at CR={BASELINE_CR}...")
                traditional_models, traditional_baseline_meta = build_traditional_baselines(
                    sim, cr=BASELINE_CR, n_pca_samples=BASELINE_PCA_SAMPLES,
                    seed=current_seed + 6000, device=DEVICE,
                    dft_mode=BASELINE_DFT_MODE,
                    hadamard_mode=BASELINE_HADAMARD_MODE,
                    pca_train_source=BASELINE_PCA_TRAIN_SOURCE,
                    pca_fixed_snr_db=BASELINE_PCA_FIXED_SNR_DB,
                    include_diagnostic_variants=BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS,
                    random_variant_seeds=BASELINE_RANDOM_VARIANT_SEEDS,
                    include_legacy_variants=BASELINE_INCLUDE_LEGACY_VARIANTS,
                    dft_lag_limit_samples=method_tdoa_lag_limit,
                    include_strong_variants=RUN_STRONG_BASELINES,
                )
                print(f"[Fig6] Baseline feature budget: {traditional_baseline_meta}")
                if INCLUDE_REPRO_REFERENCE and not reference_models_dict:
                    reference_models_dict = load_repro_reference_models(
                        REFERENCE_MODEL_DIR, CR_LIST, DEVICE
                    )
                fig6_models = dict(traditional_models)
                if reference_models_dict:
                    reference_crs = CR_LIST if EXPERIMENT_MODE in INNOVATION_MODES else [BASELINE_CR]
                    for ref_cr in reference_crs:
                        if ref_cr in reference_models_dict:
                            fig6_models[f"DAE-CR{ref_cr}"] = reference_models_dict[ref_cr]
                elif EXPERIMENT_MODE not in INNOVATION_MODES:
                    fig6_models[f"DAE-CR{BASELINE_CR}"] = models_dict[BASELINE_CR]
                else:
                    print("[Fig9] Frozen Chen-DAE reference disabled or unavailable; "
                          "innovation comparison will omit DAE-CR reference curves.")
                if EXPERIMENT_MODE in INNOVATION_MODES:
                    for cr in CR_LIST:
                        fig6_models[f"{INNOVATION_LABEL_PREFIX}-CR{cr}"] = models_dict[cr]
                direct_estimators = {}
                dft_main = traditional_baseline_meta.get("dft_main_label", "DFT-SCS-lite")
                requested_dft_direct_source = str(BASELINE_DFT_DIRECT_SOURCE)
                if requested_dft_direct_source.lower() in ("auto", "default"):
                    requested_dft_direct_source = "DFT-train-power"
                dft_direct_source_label = requested_dft_direct_source
                if dft_direct_source_label not in traditional_models:
                    fallback_order = ["DFT-train-power", dft_main, "DFT-SCS-lite", "DFT-bandlimited"]
                    dft_direct_source_label = next(
                        (name for name in fallback_order if name in traditional_models),
                        dft_main
                    )
                dft_direct_source = traditional_models.get(dft_direct_source_label)
                if dft_direct_source is not None and hasattr(dft_direct_source, "selected_bins"):
                    direct_estimators["DFT"] = DirectDFTTDOAEstimator(
                        dft_direct_source.selected_bins.detach().cpu().numpy(),
                        signal_len=sim.signal_len,
                        label="DFT",
                    )
                if RUN_TASK_AWARE_BASELINES:
                    for src_label, dst_label in [
                        ("DFT-Fisher", "DFT-Fisher-Direct"),
                        ("DFT-train-power", "DFT-train-power-Direct"),
                    ]:
                        if src_label == dft_direct_source_label:
                            continue
                        model = traditional_models.get(src_label)
                        if model is not None and hasattr(model, "selected_bins"):
                            direct_estimators[dst_label] = DirectDFTTDOAEstimator(
                                model.selected_bins.detach().cpu().numpy(),
                                signal_len=sim.signal_len,
                                label=dst_label,
                            )
                if RUN_STRONG_BASELINES:
                    if dft_direct_source is not None and hasattr(dft_direct_source, "selected_bins"):
                        direct_estimators["Cao2017-DFT-AML"] = Cao2017DFTAMLEstimator(
                            dft_direct_source.selected_bins.detach().cpu().numpy(),
                            signal_len=sim.signal_len,
                            label="Cao2017-DFT-AML",
                            weight_mode="aml",
                        )
                    geoambi_source = traditional_models.get("DFT-GeoAmbi")
                    if geoambi_source is not None and hasattr(geoambi_source, "selected_bins"):
                        geoambi_bins = geoambi_source.selected_bins.detach().cpu().numpy()
                        coarse_bins = None
                        if dft_direct_source is not None and hasattr(dft_direct_source, "selected_bins"):
                            coarse_bins = dft_direct_source.selected_bins.detach().cpu().numpy()
                        if RUN_GEOHYBRID_BASELINES and coarse_bins is not None:
                            direct_estimators["GeoHybrid-DFT"] = GeoHybridDFTTDOAEstimator(
                                coarse_bins,
                                geoambi_bins,
                                signal_len=sim.signal_len,
                                label="GeoHybrid-DFT",
                                coarse_mode="aml",
                                fine_mode="aml",
                                refinement_radius=4.0,
                                fine_gain=1.25,
                                consistency_sigma=3.0,
                            )
                            direct_estimators["GeoHybrid-DFT-PHAT-gated"] = GeoHybridDFTTDOAEstimator(
                                coarse_bins,
                                geoambi_bins,
                                signal_len=sim.signal_len,
                                label="GeoHybrid-DFT-PHAT-gated",
                                coarse_mode="aml",
                                fine_mode="phat_gated",
                                refinement_radius=4.0,
                                fine_gain=1.10,
                                consistency_sigma=3.0,
                                phat_gate=0.35,
                            )
                        direct_estimators["GeoAmbi-DFT-Direct"] = DirectDFTTDOAEstimator(
                            geoambi_bins,
                            signal_len=sim.signal_len,
                            label="GeoAmbi-DFT-Direct",
                        )
                        direct_estimators["GeoAmbi-DFT-AML"] = Cao2017DFTAMLEstimator(
                            geoambi_bins,
                            signal_len=sim.signal_len,
                            label="GeoAmbi-DFT-AML",
                            weight_mode="aml",
                        )
                        direct_estimators["GeoAmbi-DFT-PHAT"] = DirectDFTTDOAEstimator(
                            geoambi_bins,
                            signal_len=sim.signal_len,
                            label="GeoAmbi-DFT-PHAT",
                            phat=True,
                        )
                    cao2020_source = traditional_models.get("DFT-Cao2020-CRB")
                    if cao2020_source is not None and hasattr(cao2020_source, "selected_bins"):
                        direct_estimators["Cao2020-HighFC"] = Cao2020SegmentedFCEstimator(
                            cao2020_source.selected_bins.detach().cpu().numpy(),
                            signal_len=sim.signal_len,
                            label="Cao2020-HighFC",
                            n_segments=4,
                            weight_mode="sqrt_power",
                        )
                    else:
                        direct_estimators["Cao2020-HighFC"] = Cao2020SegmentedFCEstimator(
                            high_frequency_dft_bins(sim.signal_len, BASELINE_CR),
                            signal_len=sim.signal_len,
                            label="Cao2020-HighFC",
                            n_segments=4,
                            weight_mode="sqrt_power",
                        )
                    zhai_crlb_source = traditional_models.get("DFT-Zhai-CRLB")
                    if zhai_crlb_source is not None and hasattr(zhai_crlb_source, "selected_bins"):
                        direct_estimators["Zhai-CRLB-Decimation"] = Cao2017DFTAMLEstimator(
                            zhai_crlb_source.selected_bins.detach().cpu().numpy(),
                            signal_len=sim.signal_len,
                            label="Zhai-CRLB-Decimation",
                            weight_mode="aml",
                        )
                    if (STRONG_INCLUDE_PHASE_SUPERPOSITION
                            and zhai_crlb_source is not None
                            and hasattr(zhai_crlb_source, "selected_bins")):
                        direct_estimators["Zhai-Phase-Superposition"] = ZhaiPhaseSuperpositionEstimator(
                            zhai_crlb_source.selected_bins.detach().cpu().numpy(),
                            signal_len=sim.signal_len,
                            label="Zhai-Phase-Superposition",
                            weight_mode="sqrt_power",
                        )

                baseline_all_results = run_urban_method_comparison(
                    fig6_models, sim, DEVICE, seed=current_seed,
                    snr_range=EVAL_SNR_RANGE, num_trials=MONTE_CARLO_TRIALS,
                    sub_sample=TDOA_SUB_SAMPLE, use_los_only=USE_LOS_ONLY,
                    batch_size=BATCH_SIZE, fixed_eval_set=FIXED_EVAL_SET,
                    estimator=LOCALIZATION_ESTIMATOR,
                    title=f"Baseline_Method_Comparison_CR{BASELINE_CR}",
                    direct_estimators=direct_estimators,
                    tdoa_lag_limit_samples=method_tdoa_lag_limit,
                )
                had_main = traditional_baseline_meta.get("hadamard_main_label", "Hadamard")
                dft_alias_diagnostics = {
                    label: est.alias_diagnostics(lag_limit_samples=method_tdoa_lag_limit)
                    for label, est in direct_estimators.items()
                    if hasattr(est, "alias_diagnostics")
                }
                for label, diag in dft_alias_diagnostics.items():
                    outside = diag.get("max_sidelobe_outside_physical_lag")
                    if outside is not None and outside > 0.8:
                        print(f"[Warn] {label} has strong DFT alias sidelobe outside "
                              f"physical lag gate: {outside:.3f}")
                common_config = {
                    "baseline_cr": BASELINE_CR,
                    "traditional_baseline_meta": traditional_baseline_meta,
                    "fig6_show_zoom_inset": FIG6_SHOW_ZOOM_INSET,
                    "fig6_zoom_snr_min": FIG6_ZOOM_SNR_MIN,
                    "chen_fig6_dft_method": "frequency_domain_cross_spectrum_tdoa",
                    "chen_fig6_dft_direct_source": dft_direct_source_label,
                    "requested_dft_direct_source": requested_dft_direct_source,
                    "dft_waveform_reconstruction_role": "supplement_ablation_only",
                    "use_physical_tdoa_lag_gate": USE_PHYSICAL_TDOA_LAG_GATE,
                    "tdoa_lag_limit_samples": method_tdoa_lag_limit,
                    "tdoa_lag_margin_samples": TDOA_LAG_MARGIN_SAMPLES,
                    "direct_dft_alias_diagnostics": dft_alias_diagnostics,
                    "baseline_include_legacy_variants": BASELINE_INCLUDE_LEGACY_VARIANTS,
                    "run_strong_baselines": RUN_STRONG_BASELINES,
                    "strong_include_phase_superposition": STRONG_INCLUDE_PHASE_SUPERPOSITION,
                    "run_strong_diagnostic_supplement": RUN_STRONG_DIAGNOSTIC_SUPPLEMENT,
                    "cao2020_estimator": "segmented_incoherent_fc",
                    "zhai_phase_superposition": "sqrt_power_weighted_phase",
                    "run_geohybrid_baselines": RUN_GEOHYBRID_BASELINES,
                    "geohybrid_estimator": (
                        "coarse_high_coherence_plus_geoambi_fine_uncertainty_wls"
                    ),
                    "export_method_zoom_figures": EXPORT_METHOD_ZOOM_FIGURES,
                    "method_zoom_low_snr_max": METHOD_ZOOM_LOW_SNR_MAX,
                    "method_zoom_high_snr_min": METHOD_ZOOM_HIGH_SNR_MIN,
                }
                baseline_all_results[0]["config"].update(common_config)
                baseline_all_results[0]["config"]["direct_methods"] = list(direct_estimators.keys())
                baseline_all_results[0]["config"]["task_aware_direct_methods"] = [
                    name for name in [
                        "DFT", "DFT-Fisher-Direct",
                        "GeoHybrid-DFT", "GeoAmbi-DFT-Direct"
                    ]
                    if name in direct_estimators
                ]
                baseline_all_results[0]["config"]["strong_direct_methods"] = [
                    name for name in [
                        "Cao2017-DFT-AML", "GeoHybrid-DFT",
                        "GeoHybrid-DFT-PHAT-gated", "GeoAmbi-DFT-AML",
                        "GeoAmbi-DFT-Direct", "GeoAmbi-DFT-PHAT",
                        "Cao2020-HighFC",
                        "Zhai-CRLB-Decimation", "Zhai-Phase-Superposition"
                    ]
                    if name in direct_estimators
                ]

                fig6_main_order = unique_order([
                    "Raw", f"DAE-CR{BASELINE_CR}", "DFT", had_main, "PCA"
                ])
                fig6_supp_order = unique_order([
                    "Raw", f"DAE-CR{BASELINE_CR}", "DFT", "DFT-SCS-lite",
                    "DFT-train-power", "DFT-bandlimited",
                    had_main, "Hadamard-block-2", "PCA"
                ])
                fig6_keep = unique_order(
                    ["Raw", "Clean", "Geometry"] + fig6_supp_order
                )
                fig6_results = filter_method_comparison_data(
                    baseline_all_results, fig6_keep,
                    {
                        **common_config,
                        "main_method_order": fig6_main_order,
                        "supplement_method_order": fig6_supp_order,
                        "main_title": f"Figure 6: Chen-Style Traditional Baselines (CR={BASELINE_CR})",
                        "supplement_title": f"Figure 6 Supplement: Baseline Sensitivity (CR={BASELINE_CR})",
                        "main_figure_filename": f"Fig6_Chen_Traditional_Baselines_CR{BASELINE_CR}",
                        "supplement_figure_filename": f"Fig6_Supp_Chen_Baseline_Ablation_CR{BASELINE_CR}",
                        "supplement_zoom_method_order": fig6_supp_order,
                        "supplement_zoom_figure_filename": f"Fig6_Supp_Chen_Baseline_Ablation_CR{BASELINE_CR}_SNR_Zooms",
                        "table_prefix": "fig6",
                    },
                )

                if RUN_TASK_AWARE_BASELINES and direct_estimators:
                    fig7_order = unique_order([
                        "Raw", f"DAE-CR{BASELINE_CR}", "DFT",
                        "DFT-SCS-lite", "DFT-Fisher-Direct",
                        "GeoHybrid-DFT", "GeoAmbi-DFT-Direct",
                        had_main, "PCA"
                    ])
                    fig7_keep = unique_order(["Raw", "Clean", "Geometry"] + fig7_order)
                    fig7_results = filter_method_comparison_data(
                        baseline_all_results, fig7_keep,
                        {
                            **common_config,
                            "taskaware_method_order": fig7_order,
                            "taskaware_title": f"Figure 7: Task-Aware Baseline Track (CR={BASELINE_CR})",
                            "taskaware_figure_filename": f"Fig7_TaskAware_Baselines_CR{BASELINE_CR}",
                            "taskaware_zoom_method_order": fig7_order,
                            "taskaware_zoom_figure_filename": f"Fig7_TaskAware_Baselines_CR{BASELINE_CR}_SNR_Zooms",
                            "table_prefix": "fig7",
                        },
                    )

                if RUN_STRONG_BASELINES and direct_estimators:
                    # Fig8 main is intentionally reserved for stable, interpretable
                    # strong baselines. Proxy/diagnostic variants remain computed
                    # in baseline_all_results and are plotted in Fig8 supplement.
                    fig8_order = unique_order([
                        "Raw", f"DAE-CR{BASELINE_CR}", "DFT",
                        "DFT-Fisher-Direct", "Cao2017-DFT-AML",
                        "GeoHybrid-DFT",
                        "Zhai-CRLB-Decimation", had_main, "PCA"
                    ])
                    fig8_keep = unique_order(["Raw", "Clean", "Geometry"] + fig8_order)
                    fig8_results = filter_method_comparison_data(
                        baseline_all_results, fig8_keep,
                        {
                            **common_config,
                            "strong_method_order": fig8_order,
                            "strong_title": f"Figure 8: Strong Task-Aware Baselines (CR={BASELINE_CR})",
                            "strong_figure_filename": f"Fig8_Strong_TaskAware_Baselines_CR{BASELINE_CR}",
                            "strong_zoom_method_order": fig8_order,
                            "strong_zoom_figure_filename": f"Fig8_Strong_TaskAware_Baselines_CR{BASELINE_CR}_SNR_Zooms",
                            "table_prefix": "fig8",
                        },
                    )
                    if RUN_STRONG_DIAGNOSTIC_SUPPLEMENT:
                        fig8_supp_order = unique_order([
                            "Raw", f"DAE-CR{BASELINE_CR}", "DFT",
                            "DFT-Fisher-Direct", "Cao2017-DFT-AML",
                            "GeoHybrid-DFT", "GeoHybrid-DFT-PHAT-gated",
                            "GeoAmbi-DFT-Direct", "GeoAmbi-DFT-AML",
                            "GeoAmbi-DFT-PHAT", "Cao2020-HighFC",
                            "Zhai-CRLB-Decimation",
                            "Zhai-Phase-Superposition", had_main, "PCA"
                        ])
                        fig8_supp_keep = unique_order(["Raw", "Clean", "Geometry"] + fig8_supp_order)
                        fig8_supp_results = filter_method_comparison_data(
                            baseline_all_results, fig8_supp_keep,
                            {
                                **common_config,
                                "strong_method_order": fig8_supp_order,
                                "strong_title": (
                                    f"Figure 8 Supplement: Strong Baseline Diagnostics "
                                    f"(CR={BASELINE_CR})"
                                ),
                                "strong_figure_filename": (
                                    f"Fig8_Supp_Strong_Baseline_Diagnostics_CR{BASELINE_CR}"
                                ),
                                "strong_zoom_method_order": fig8_supp_order,
                                "strong_zoom_figure_filename": (
                                    f"Fig8_Supp_Strong_Baseline_Diagnostics_CR{BASELINE_CR}_SNR_Zooms"
                                ),
                                "table_prefix": "fig8_supp",
                            },
                        )

                if EXPERIMENT_MODE in INNOVATION_MODES:
                    fig9_order = unique_order([
                        "Raw", "DAE-CR4", "DAE-CR8", "DAE-CR16",
                        f"{INNOVATION_LABEL_PREFIX}-CR4",
                        f"{INNOVATION_LABEL_PREFIX}-CR8",
                        f"{INNOVATION_LABEL_PREFIX}-CR16",
                        "DFT", "DFT-Fisher-Direct",
                        "Cao2017-DFT-AML", "GeoHybrid-DFT", "GeoAmbi-DFT-AML",
                        "Zhai-CRLB-Decimation",
                        had_main, "PCA"
                    ])
                    fig9_keep = unique_order(["Raw", "Clean", "Geometry"] + fig9_order)
                    fig9_results = filter_method_comparison_data(
                        baseline_all_results, fig9_keep,
                        {
                            **common_config,
                            "strong_method_order": fig9_order,
                            "strong_title": (
                                "Figure 9: Frequency-Task DAE vs Frozen Chen-DAE "
                                "References and Strong Baselines"
                            ),
                            "strong_figure_filename": (
                                f"Fig9_FrequencyTaskDAE_vs_Repro_CR{BASELINE_CR}"
                            ),
                            "strong_zoom_method_order": fig9_order,
                            "strong_zoom_figure_filename": (
                                f"Fig9_FrequencyTaskDAE_vs_Repro_CR{BASELINE_CR}_SNR_Zooms"
                            ),
                            "table_prefix": "fig9",
                            "innovation_model": ACTIVE_MODEL_NAME,
                            "reference_model_dir": REFERENCE_MODEL_DIR,
                        },
                    )
                    fig9_focus_order = unique_order([
                        "Raw", "DAE-CR4", "DAE-CR8", "DAE-CR16",
                        f"{INNOVATION_LABEL_PREFIX}-CR4",
                        f"{INNOVATION_LABEL_PREFIX}-CR8",
                        f"{INNOVATION_LABEL_PREFIX}-CR16",
                    ])
                    fig9_focus_keep = unique_order(["Raw", "Clean", "Geometry"] + fig9_focus_order)
                    fig9_focus_results = filter_method_comparison_data(
                        baseline_all_results, fig9_focus_keep,
                        {
                            **common_config,
                            "strong_method_order": fig9_focus_order,
                            "strong_title": (
                                "Figure 9 Focus: Frozen Chen-DAE vs Frequency-Task DAE"
                            ),
                            "strong_figure_filename": (
                                f"Fig9_Focused_DAE_CR_Comparison"
                            ),
                            "strong_zoom_method_order": fig9_focus_order,
                            "strong_zoom_figure_filename": (
                                f"Fig9_Focused_DAE_CR_Comparison_SNR_Zooms"
                            ),
                            "table_prefix": "fig9_focus",
                            "innovation_model": ACTIVE_MODEL_NAME,
                            "reference_model_dir": REFERENCE_MODEL_DIR,
                        },
                    )

                if RUN_LOCALIZER_ABLATION and EVALUATION_MODE == "urban_localization":
                    ablation_models = {
                        label: fig6_models[label]
                        for label in LOCALIZER_ABLATION_METHODS
                        if label in fig6_models
                    }
                    ablation_direct = {
                        label: direct_estimators[label]
                        for label in LOCALIZER_ABLATION_METHODS
                        if label in direct_estimators
                    }
                    ablation_order = unique_order([
                        label for label in LOCALIZER_ABLATION_METHODS
                        if (label in ("Raw", "Clean", "Geometry")
                            or label in ablation_models
                            or label in ablation_direct)
                    ])
                    if not ablation_order:
                        print("[Fig10] Skipped: no requested methods available.")
                    else:
                        print("\n" + "=" * 40)
                        print("[Fig10] Running high-SNR localizer robustness ablation "
                              f"for methods={ablation_order} and "
                              f"estimators={LOCALIZER_ABLATION_ESTIMATORS}...")
                        ablation_runs = []
                        for ab_estimator in LOCALIZER_ABLATION_ESTIMATORS:
                            ablation_run = run_urban_method_comparison(
                                ablation_models, sim, DEVICE, seed=current_seed,
                                snr_range=LOCALIZER_ABLATION_SNR_RANGE,
                                num_trials=MONTE_CARLO_TRIALS,
                                sub_sample=TDOA_SUB_SAMPLE,
                                use_los_only=USE_LOS_ONLY,
                                batch_size=BATCH_SIZE,
                                fixed_eval_set=FIXED_EVAL_SET,
                                estimator=ab_estimator,
                                title=f"Localizer_Ablation_{ab_estimator}",
                                direct_estimators=ablation_direct,
                                tdoa_lag_limit_samples=method_tdoa_lag_limit,
                            )
                            ablation_runs.append((ab_estimator, ablation_run))
                        fig10_localizer_results = combine_localizer_ablation_results(
                            ablation_runs,
                            ablation_order,
                            {
                                **common_config,
                                "strong_title": (
                                    "Figure 10: Localizer Robustness Ablation "
                                    "(High-SNR Outlier Check)"
                                ),
                                "strong_figure_filename": (
                                    "Fig10_Localizer_Robustness_Ablation"
                                ),
                                "strong_zoom_figure_filename": (
                                    "Fig10_Localizer_Robustness_Ablation_SNR_Zooms"
                                ),
                                "table_prefix": "fig10_localizer",
                                "export_method_zoom_figures": False,
                                "localizer_ablation_snr_range": [
                                    float(v) for v in LOCALIZER_ABLATION_SNR_RANGE
                                ],
                                "localizer_ablation_note": (
                                    "Supplement only: main Fig3/Fig6-Fig9 retain "
                                    "the default all_pair_wls localizer for "
                                    "historical comparability."
                                ),
                            },
                        )

        # 保存绘图数据（含运行配置，供 replot.py / 离线分析）
        plot_data = {
            'cv_results_dict': cv_results_dict,
            'snr_data': snr_data_all,
            'snr_cr': CR_LIST,
            'mc_results': mc_results,
            'fig6_results': fig6_results,
            'fig7_results': fig7_results,
            'fig8_results': fig8_results,
            'fig8_supp_results': fig8_supp_results,
            'fig9_results': fig9_results,
            'fig9_focus_results': fig9_focus_results,
            'fig10_localizer_results': fig10_localizer_results,
            'baseline_all_results': baseline_all_results,
            'traditional_baseline_meta': traditional_baseline_meta,
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
                'experiment_profile': USER_EXPERIMENT_PROFILE,
                'applied_experiment_profile': APPLIED_EXPERIMENT_PROFILE,
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
                'run_traditional_baselines': RUN_TRADITIONAL_BASELINES,
                'baseline_cr': BASELINE_CR,
                'baseline_pca_samples': BASELINE_PCA_SAMPLES,
                'baseline_dft_mode': BASELINE_DFT_MODE,
                'baseline_dft_direct_source': BASELINE_DFT_DIRECT_SOURCE,
                'baseline_hadamard_mode': BASELINE_HADAMARD_MODE,
                'baseline_pca_train_source': BASELINE_PCA_TRAIN_SOURCE,
                'baseline_pca_fixed_snr_db': BASELINE_PCA_FIXED_SNR_DB,
                'baseline_include_diagnostic_variants': BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS,
                'baseline_include_legacy_variants': BASELINE_INCLUDE_LEGACY_VARIANTS,
                'baseline_random_variant_seeds': BASELINE_RANDOM_VARIANT_SEEDS,
                'reuse_static_baseline_cache': REUSE_STATIC_BASELINE_CACHE,
                'static_baseline_cache_dir': STATIC_BASELINE_CACHE_DIR,
                'static_baseline_cache_loaded': static_baseline_cache is not None,
                'run_task_aware_baselines': RUN_TASK_AWARE_BASELINES,
                'run_strong_baselines': RUN_STRONG_BASELINES,
                'strong_include_phase_superposition': STRONG_INCLUDE_PHASE_SUPERPOSITION,
                'run_strong_diagnostic_supplement': RUN_STRONG_DIAGNOSTIC_SUPPLEMENT,
                'run_geohybrid_baselines': RUN_GEOHYBRID_BASELINES,
                'run_localizer_ablation': RUN_LOCALIZER_ABLATION,
                'localizer_ablation_estimators': LOCALIZER_ABLATION_ESTIMATORS,
                'localizer_ablation_methods': LOCALIZER_ABLATION_METHODS,
                'localizer_ablation_snr_range': [
                    float(v) for v in LOCALIZER_ABLATION_SNR_RANGE
                ],
                'use_physical_tdoa_lag_gate': USE_PHYSICAL_TDOA_LAG_GATE,
                'tdoa_lag_limit_samples': method_tdoa_lag_limit,
                'tdoa_lag_margin_samples': TDOA_LAG_MARGIN_SAMPLES,
                'export_method_zoom_figures': EXPORT_METHOD_ZOOM_FIGURES,
                'method_zoom_low_snr_max': METHOD_ZOOM_LOW_SNR_MAX,
                'method_zoom_high_snr_min': METHOD_ZOOM_HIGH_SNR_MIN,
                'eval_only_copy_models': EVAL_ONLY_COPY_MODELS,
                'fig6_show_zoom_inset': FIG6_SHOW_ZOOM_INSET,
                'fig6_zoom_snr_min': FIG6_ZOOM_SNR_MIN,
                'export_fig6_svg_in_tables': EXPORT_FIG6_SVG_IN_TABLES,
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
                'active_model_name': ACTIVE_MODEL_NAME,
                'include_repro_reference': INCLUDE_REPRO_REFERENCE,
                'reference_model_dir': REFERENCE_MODEL_DIR if INCLUDE_REPRO_REFERENCE else None,
                'freq_task_epochs': MAX_EPOCHS if EXPERIMENT_MODE in INNOVATION_MODES else None,
                'freq_task_spectral_blend': (
                    USER_FREQ_TASK_V3_SPECTRAL_BLEND
                    if EXPERIMENT_MODE in FREQ_TASK_V3_MODES
                    else (
                        USER_FREQ_TASK_V2_SPECTRAL_BLEND
                        if EXPERIMENT_MODE in FREQ_TASK_V2_MODES
                        else USER_FREQ_TASK_SPECTRAL_BLEND
                    )
                ),
                'freq_task_spectral_power': (
                    USER_FREQ_TASK_V3_SPECTRAL_POWER
                    if EXPERIMENT_MODE in FREQ_TASK_V3_MODES
                    else (
                        USER_FREQ_TASK_V2_SPECTRAL_POWER
                        if EXPERIMENT_MODE in FREQ_TASK_V2_MODES
                        else USER_FREQ_TASK_SPECTRAL_POWER
                    )
                ),
                'freq_task_v2_corr_weight': (
                    USER_FREQ_TASK_V2_CORR_WEIGHT
                    if EXPERIMENT_MODE in FREQ_TASK_V2_MODES else None
                ),
                'freq_task_v2_phase_weight': (
                    USER_FREQ_TASK_V2_PHASE_WEIGHT
                    if EXPERIMENT_MODE in FREQ_TASK_V2_MODES else None
                ),
                'freq_task_v2_peak_weight': (
                    USER_FREQ_TASK_V2_PEAK_WEIGHT
                    if EXPERIMENT_MODE in FREQ_TASK_V2_MODES else None
                ),
                'freq_task_v2_soft_peak_sigma': (
                    USER_FREQ_TASK_V2_SOFT_PEAK_SIGMA
                    if EXPERIMENT_MODE in FREQ_TASK_V2_MODES else None
                ),
                'freq_task_v2_peak_sharpness_weight': (
                    USER_FREQ_TASK_V2_PEAK_SHARPNESS_WEIGHT
                    if EXPERIMENT_MODE in FREQ_TASK_V2_MODES else None
                ),
                'freq_task_v3_corr_weight': (
                    USER_FREQ_TASK_V3_CORR_WEIGHT
                    if EXPERIMENT_MODE in FREQ_TASK_V3_MODES else None
                ),
                'freq_task_v3_phase_weight': (
                    USER_FREQ_TASK_V3_PHASE_WEIGHT
                    if EXPERIMENT_MODE in FREQ_TASK_V3_MODES else None
                ),
                'freq_task_v3_peak_weight': (
                    USER_FREQ_TASK_V3_PEAK_WEIGHT
                    if EXPERIMENT_MODE in FREQ_TASK_V3_MODES else None
                ),
                'freq_task_v3_task_head_weight': (
                    USER_FREQ_TASK_V3_TASK_HEAD_WEIGHT
                    if EXPERIMENT_MODE in FREQ_TASK_V3_MODES else None
                ),
                'freq_task_v3_nested_crs': (
                    list(USER_FREQ_TASK_V3_NESTED_CRS)
                    if EXPERIMENT_MODE in FREQ_TASK_V3_MODES else None
                ),
            },
        }
        pkl_path = os.path.join(RESULT_DIR, "plot_data.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f"[Saved] {pkl_path}")

        # 绘制 Figure 2 和 3。创建顺序与本地文件序号保持一致，避免窗口标题混乱。
        fig2_suffix = f" - {ACTIVE_MODEL_NAME}" if EXPERIMENT_MODE in INNOVATION_MODES else ""
        fig_snr = plot_snr_comparison_multi(
            cr_list=CR_LIST, data_dict=snr_data_all, title_suffix=fig2_suffix
        )
        save_figure(fig_snr, "Fig2_SNR_Comparison")

        fig_mc = plot_monte_carlo(mc_results)
        save_figure(fig_mc, "Fig3_MonteCarlo_TDOA_RMSE")

        if fig6_results is not None:
            fig_methods = plot_method_comparison(fig6_results, plot_kind="main")
            fig6_name = fig6_results[0].get("config", {}).get(
                "main_figure_filename", f"Fig6_Traditional_Baselines_CR{BASELINE_CR}"
            )
            save_figure(fig_methods, fig6_name)
            fig_methods_supp = plot_method_comparison(fig6_results, plot_kind="supplement")
            fig6_supp_name = fig6_results[0].get("config", {}).get(
                "supplement_figure_filename", f"Fig6_Supp_Baseline_Ablation_CR{BASELINE_CR}"
            )
            save_figure(fig_methods_supp, fig6_supp_name)
            if EXPORT_METHOD_ZOOM_FIGURES:
                fig6_config = fig6_results[0].get("config", {})
                zoom_order = fig6_config.get("supplement_zoom_method_order")
                fig_methods_zoom = plot_method_zoom_pair(
                    fig6_results, plot_kind="supplement",
                    low_snr_max=METHOD_ZOOM_LOW_SNR_MAX,
                    high_snr_min=METHOD_ZOOM_HIGH_SNR_MIN,
                    method_order=zoom_order,
                )
                save_figure(
                    fig_methods_zoom,
                    fig6_config.get(
                        "supplement_zoom_figure_filename",
                        f"Fig6_Supp_Chen_Baseline_Ablation_CR{BASELINE_CR}_SNR_Zooms",
                    ),
                )

        if fig7_results is not None:
            fig_task = plot_method_comparison(fig7_results, plot_kind="taskaware")
            fig7_name = fig7_results[0].get("config", {}).get(
                "taskaware_figure_filename", f"Fig7_TaskAware_Baselines_CR{BASELINE_CR}"
            )
            save_figure(fig_task, fig7_name)
            if EXPORT_METHOD_ZOOM_FIGURES:
                fig7_config = fig7_results[0].get("config", {})
                zoom_order = fig7_config.get("taskaware_zoom_method_order")
                fig_task_zoom = plot_method_zoom_pair(
                    fig7_results, plot_kind="taskaware",
                    low_snr_max=METHOD_ZOOM_LOW_SNR_MAX,
                    high_snr_min=METHOD_ZOOM_HIGH_SNR_MIN,
                    method_order=zoom_order,
                )
                save_figure(
                    fig_task_zoom,
                    fig7_config.get(
                        "taskaware_zoom_figure_filename",
                        f"Fig7_TaskAware_Baselines_CR{BASELINE_CR}_SNR_Zooms",
                    ),
                )

        if fig8_results is not None:
            fig_strong = plot_method_comparison(fig8_results, plot_kind="strong")
            fig8_name = fig8_results[0].get("config", {}).get(
                "strong_figure_filename", f"Fig8_Strong_TaskAware_Baselines_CR{BASELINE_CR}"
            )
            save_figure(fig_strong, fig8_name)
            if EXPORT_METHOD_ZOOM_FIGURES:
                fig8_config = fig8_results[0].get("config", {})
                zoom_order = fig8_config.get("strong_zoom_method_order")
                fig_strong_zoom = plot_method_zoom_pair(
                    fig8_results, plot_kind="strong",
                    low_snr_max=METHOD_ZOOM_LOW_SNR_MAX,
                    high_snr_min=METHOD_ZOOM_HIGH_SNR_MIN,
                    method_order=zoom_order,
                )
                save_figure(
                    fig_strong_zoom,
                    fig8_config.get(
                        "strong_zoom_figure_filename",
                        f"Fig8_Strong_TaskAware_Baselines_CR{BASELINE_CR}_SNR_Zooms",
                    ),
                )

        if fig8_supp_results is not None:
            fig_strong_supp = plot_method_comparison(fig8_supp_results, plot_kind="strong")
            fig8_supp_name = fig8_supp_results[0].get("config", {}).get(
                "strong_figure_filename", f"Fig8_Supp_Strong_Baseline_Diagnostics_CR{BASELINE_CR}"
            )
            save_figure(fig_strong_supp, fig8_supp_name)
            if EXPORT_METHOD_ZOOM_FIGURES:
                fig8_supp_config = fig8_supp_results[0].get("config", {})
                zoom_order = fig8_supp_config.get("strong_zoom_method_order")
                fig_strong_supp_zoom = plot_method_zoom_pair(
                    fig8_supp_results, plot_kind="strong",
                    low_snr_max=METHOD_ZOOM_LOW_SNR_MAX,
                    high_snr_min=METHOD_ZOOM_HIGH_SNR_MIN,
                    method_order=zoom_order,
                )
                save_figure(
                    fig_strong_supp_zoom,
                    fig8_supp_config.get(
                        "strong_zoom_figure_filename",
                        f"Fig8_Supp_Strong_Baseline_Diagnostics_CR{BASELINE_CR}_SNR_Zooms",
                    ),
                )

        if fig9_results is not None:
            fig_innov = plot_method_comparison(fig9_results, plot_kind="strong")
            fig9_name = fig9_results[0].get("config", {}).get(
                "strong_figure_filename", f"Fig9_FrequencyTaskDAE_vs_Repro_CR{BASELINE_CR}"
            )
            save_figure(fig_innov, fig9_name)
            if EXPORT_METHOD_ZOOM_FIGURES:
                fig9_config = fig9_results[0].get("config", {})
                zoom_order = fig9_config.get("strong_zoom_method_order")
                fig_innov_zoom = plot_method_zoom_pair(
                    fig9_results, plot_kind="strong",
                    low_snr_max=METHOD_ZOOM_LOW_SNR_MAX,
                    high_snr_min=METHOD_ZOOM_HIGH_SNR_MIN,
                    method_order=zoom_order,
                )
                save_figure(
                    fig_innov_zoom,
                    fig9_config.get(
                        "strong_zoom_figure_filename",
                        f"Fig9_FrequencyTaskDAE_vs_Repro_CR{BASELINE_CR}_SNR_Zooms",
                    ),
                )

        if fig9_focus_results is not None:
            fig_innov_focus = plot_method_comparison(fig9_focus_results, plot_kind="strong")
            fig9_focus_name = fig9_focus_results[0].get("config", {}).get(
                "strong_figure_filename", "Fig9_Focused_DAE_CR_Comparison"
            )
            save_figure(fig_innov_focus, fig9_focus_name)
            if EXPORT_METHOD_ZOOM_FIGURES:
                fig9_focus_config = fig9_focus_results[0].get("config", {})
                zoom_order = fig9_focus_config.get("strong_zoom_method_order")
                fig_innov_focus_zoom = plot_method_zoom_pair(
                    fig9_focus_results, plot_kind="strong",
                    low_snr_max=METHOD_ZOOM_LOW_SNR_MAX,
                    high_snr_min=METHOD_ZOOM_HIGH_SNR_MIN,
                    method_order=zoom_order,
                )
                save_figure(
                    fig_innov_focus_zoom,
                    fig9_focus_config.get(
                        "strong_zoom_figure_filename",
                        "Fig9_Focused_DAE_CR_Comparison_SNR_Zooms",
                    ),
                )

        if fig10_localizer_results is not None:
            fig_localizer = plot_method_comparison(
                fig10_localizer_results, plot_kind="strong"
            )
            fig10_config = fig10_localizer_results[0].get("config", {})
            save_figure(
                fig_localizer,
                fig10_config.get(
                    "strong_figure_filename",
                    "Fig10_Localizer_Robustness_Ablation",
                ),
            )
            if EXPORT_METHOD_ZOOM_FIGURES and fig10_config.get("export_method_zoom_figures", True):
                zoom_order = fig10_config.get("strong_zoom_method_order")
                fig_localizer_zoom = plot_method_zoom_pair(
                    fig10_localizer_results, plot_kind="strong",
                    low_snr_max=METHOD_ZOOM_LOW_SNR_MAX,
                    high_snr_min=METHOD_ZOOM_HIGH_SNR_MIN,
                    method_order=zoom_order,
                )
                save_figure(
                    fig_localizer_zoom,
                    fig10_config.get(
                        "strong_zoom_figure_filename",
                        "Fig10_Localizer_Robustness_Ablation_SNR_Zooms",
                    ),
                )

        try:
            from export_results import export as export_result_tables
            export_result_tables(RESULT_DIR, export_fig6_figures=EXPORT_FIG6_SVG_IN_TABLES)
        except Exception as exc:
            print(f"[Warn] Automatic table export failed: {exc}")

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
if plt.get_fignums() and not NO_SHOW:
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
