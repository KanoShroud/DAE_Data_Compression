"""为插值频谱伪补全实验生成导师汇报图。

统计性能图只读取既有 ``synthetic_full`` 正式结果；波形、频谱和时延得分图
使用固定随机种子的独立机制样本，不参与正式 Monte Carlo 性能统计。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import signal

MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = MODULE_DIRECTORY.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from alignment_core import complete_spectrum, shifted_frequencies  # noqa: E402
from run_interpolation_alignment import (  # noqa: E402
    COMPLETION_METHODS,
    _add_frequency_noise,
    _controlled_masks,
    _make_source_spectrum,
)
from zhai_selection import select_zhai_thesis  # noqa: E402

DEFAULT_RESULT_DIRECTORY = (
    PROJECT_DIRECTORY
    / "运行结果"
    / "频谱插值对齐"
    / "20260714_184753_411401_synthetic_full"
)
REPORT_ASSET_DIRECTORY = (
    PROJECT_DIRECTORY / "Obsidian Vault" / "assets" / "插值频谱补全汇报_20260720"
)

# 机制图使用预先固定、非事后筛选的独立样本；每类频谱使用独立子种子。
CASE_BASE_SEED = 20260720
CASE_SIGNAL_KINDS = (
    "smooth_complex",
    "smooth_magnitude_random_phase",
    "broadband_random",
    "multiband",
)
CASE_SEEDS = {
    signal_kind: CASE_BASE_SEED + index
    for index, signal_kind in enumerate(CASE_SIGNAL_KINDS)
}
SIGNAL_KIND_LABELS = {
    "smooth_complex": "平滑复谱",
    "smooth_magnitude_random_phase": "平滑幅度随机相位",
    "broadband_random": "宽带随机复谱",
    "multiband": "多频带随机相位",
}
CASE_SNR_DB = 10.0
CASE_OVERLAP_FRACTION = 0.5

PSEUDO_COMPLETION_METHODS = (
    "zero_fill",
    "ideal_periodic",
    "polyphase_fir",
    "linear_periodic",
    "cubic_periodic",
)
METHOD_LABELS = {
    "Intersection-Continuous": "真实交集",
    "Full": "完整频谱",
    "Oracle-Union": "真实联合上界",
    "zero_fill": "零填充",
    "ideal_periodic": "理想周期插值",
    "polyphase_fir": "多相 FIR",
    "linear_periodic": "线性插值",
    "cubic_periodic": "三次样条",
}
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#777777",
}


@dataclass(frozen=True)
class MechanismCase:
    """绘制机制图所需的固定样本。"""

    signal_kind: str
    seed: int
    frequencies: np.ndarray
    true_delay_samples: float
    clean_a: np.ndarray
    clean_b: np.ndarray
    noisy_a: np.ndarray
    noisy_b: np.ndarray
    mask_a: np.ndarray
    mask_b: np.ndarray
    sparse_a: np.ndarray
    sparse_b: np.ndarray
    completions_a: dict[str, object]
    completions_b: dict[str, object]


def _configure_figure_style() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "SimHei",
    ):
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def _load_formal_results(
    result_directory: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    required = (
        "config.json",
        "trial_results.csv",
        "paired_comparisons.csv",
    )
    missing = [name for name in required if not (result_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"正式结果目录缺少文件：{missing}；当前目录为 {result_directory}"
        )
    with (result_directory / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    trial_columns = (
        "trial_id",
        "method",
        "snr_db",
        "signal_kind",
        "overlap_fraction",
        "estimated_delay_samples",
        "absolute_error_samples",
        "outlier_gt_1_sample",
    )
    trials = pd.read_csv(
        result_directory / "trial_results.csv",
        usecols=trial_columns,
    )
    paired = pd.read_csv(result_directory / "paired_comparisons.csv")
    return config, trials, paired


def _validate_formal_results(
    trials: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, object]:
    expected_methods = {
        "Full",
        "Zhai-Asymmetric",
        "Intersection-Continuous",
        "Oracle-Union",
        *PSEUDO_COMPLETION_METHODS,
    }
    actual_methods = set(trials["method"].unique())
    if actual_methods != expected_methods:
        raise ValueError(
            "方法集合与正式协议不一致："
            f"缺少 {sorted(expected_methods - actual_methods)}，"
            f"新增 {sorted(actual_methods - expected_methods)}"
        )
    if trials.duplicated(["trial_id", "method"]).any():
        raise ValueError("trial_id + method 存在重复，不能生成配对图")

    baseline = trials.loc[
        trials["method"].eq("Intersection-Continuous"),
        ["trial_id", "estimated_delay_samples", "absolute_error_samples"],
    ].rename(
        columns={
            "estimated_delay_samples": "baseline_delay",
            "absolute_error_samples": "baseline_absolute_error",
        }
    )
    maximum_delay_difference = 0.0
    maximum_error_difference = 0.0
    finite_pairs = None
    for method in PSEUDO_COMPLETION_METHODS:
        candidate = trials.loc[
            trials["method"].eq(method),
            ["trial_id", "estimated_delay_samples", "absolute_error_samples"],
        ].merge(baseline, on="trial_id", validate="one_to_one")
        finite = (
            np.isfinite(candidate["estimated_delay_samples"])
            & np.isfinite(candidate["baseline_delay"])
        )
        current_pairs = int(finite.sum())
        if finite_pairs is None:
            finite_pairs = current_pairs
        elif current_pairs != finite_pairs:
            raise ValueError("五种伪补全方法的有效配对数不一致")
        delay_difference = np.abs(
            candidate.loc[finite, "estimated_delay_samples"]
            - candidate.loc[finite, "baseline_delay"]
        )
        error_difference = np.abs(
            candidate.loc[finite, "absolute_error_samples"]
            - candidate.loc[finite, "baseline_absolute_error"]
        )
        maximum_delay_difference = max(
            maximum_delay_difference,
            float(delay_difference.max()) if not delay_difference.empty else 0.0,
        )
        maximum_error_difference = max(
            maximum_error_difference,
            float(error_difference.max()) if not error_difference.empty else 0.0,
        )

    pseudo_paired = paired.loc[paired["method"].isin(PSEUDO_COMPLETION_METHODS)]
    if len(pseudo_paired) != len(PSEUDO_COMPLETION_METHODS):
        raise ValueError("配对比较文件没有覆盖全部五种伪补全方法")
    paired_columns = (
        "mean_mae_improvement_samples",
        "ci95_low",
        "ci95_high",
    )
    maximum_paired_value = float(
        np.max(np.abs(pseudo_paired.loc[:, paired_columns].to_numpy()))
    )
    tolerance = 1e-12
    if (
        maximum_delay_difference > tolerance
        or maximum_error_difference > tolerance
        or maximum_paired_value > tolerance
    ):
        raise ValueError("正式结果不再满足伪补全与真实交集逐试验等价")

    base_trials = int(trials["trial_id"].nunique())
    return {
        "method_rows": int(len(trials)),
        "base_trials": base_trials,
        "estimable_paired_trials": int(finite_pairs or 0),
        "maximum_delay_difference_samples": maximum_delay_difference,
        "maximum_absolute_error_difference_samples": maximum_error_difference,
        "maximum_paired_statistic_abs": maximum_paired_value,
    }


def _make_mechanism_case(
    config: dict[str, object],
    signal_kind: str,
    seed: int,
) -> MechanismCase:
    n_fft = int(config["n_fft"])
    selected_bins = int(config["selected_bins"])
    interpolation_factor = int(config["interpolation_factor"])
    lag_limit = float(config["lag_limit_samples"])
    rng = np.random.default_rng(seed)
    frequencies = shifted_frequencies(n_fft)
    source = _make_source_spectrum(n_fft, signal_kind, rng)
    true_delay = float(rng.uniform(-0.8 * lag_limit, 0.8 * lag_limit))
    gain = np.exp(1j * rng.uniform(-np.pi, np.pi))
    clean_a = source
    clean_b = gain * source * np.exp(-2j * np.pi * frequencies * true_delay)
    noisy_a = _add_frequency_noise(clean_a, CASE_SNR_DB, rng)
    noisy_b = _add_frequency_noise(clean_b, CASE_SNR_DB, rng)
    mask_a, mask_b = _controlled_masks(
        n_fft,
        selected_bins,
        CASE_OVERLAP_FRACTION,
        rng,
    )
    sparse_a = np.zeros(n_fft, dtype=complex)
    sparse_b = np.zeros(n_fft, dtype=complex)
    sparse_a[mask_a] = noisy_a[mask_a]
    sparse_b[mask_b] = noisy_b[mask_b]
    completions_a = {
        method: complete_spectrum(
            sparse_a,
            mask_a,
            method,
            interpolation_factor,
        )
        for method in COMPLETION_METHODS
    }
    completions_b = {
        method: complete_spectrum(
            sparse_b,
            mask_b,
            method,
            interpolation_factor,
        )
        for method in COMPLETION_METHODS
    }
    return MechanismCase(
        signal_kind=signal_kind,
        seed=seed,
        frequencies=frequencies,
        true_delay_samples=true_delay,
        clean_a=clean_a,
        clean_b=clean_b,
        noisy_a=noisy_a,
        noisy_b=noisy_b,
        mask_a=mask_a,
        mask_b=mask_b,
        sparse_a=sparse_a,
        sparse_b=sparse_b,
        completions_a=completions_a,
        completions_b=completions_b,
    )


def _delay_score(
    spectrum_b: np.ndarray,
    spectrum_a: np.ndarray,
    valid_mask: np.ndarray,
    lag_limit_samples: float,
    grid_step_samples: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(valid_mask, dtype=bool)
    selected_indices = np.flatnonzero(mask)
    cross_spectrum = spectrum_b[mask] * np.conj(spectrum_a[mask])
    numerical_floor = (
        100.0
        * np.finfo(float).eps
        * np.linalg.norm(spectrum_b[mask])
        * np.linalg.norm(spectrum_a[mask])
    )
    active_floor = max(
        float(np.max(np.abs(cross_spectrum), initial=0.0)) * 1e-12,
        numerical_floor / max(np.sqrt(cross_spectrum.size), 1.0),
    )
    active = np.abs(cross_spectrum) > active_floor
    if np.count_nonzero(active) < 2:
        raise ValueError("机制样本的有效互谱频点不足两个")
    active_indices = selected_indices[active]
    active_cross_spectrum = cross_spectrum[active]
    lags = np.arange(
        -lag_limit_samples,
        lag_limit_samples + 0.5 * grid_step_samples,
        grid_step_samples,
    )
    steering = np.exp(
        2j * np.pi * np.outer(lags, shifted_frequencies(spectrum_a.size)[active_indices])
    )
    score = np.abs(steering @ active_cross_spectrum)
    normalised = score / max(float(np.max(score)), np.finfo(float).tiny)
    return lags, normalised


def _validate_mechanism_case(
    case: MechanismCase,
    config: dict[str, object],
) -> dict[str, object]:
    n_fft = int(config["n_fft"])
    intersection = np.intersect1d(case.mask_a, case.mask_b)
    intersection_mask = np.zeros(n_fft, dtype=bool)
    intersection_mask[intersection] = True
    all_mask = np.ones(n_fft, dtype=bool)
    lag_limit = float(config["lag_limit_samples"])
    grid_step = float(config["delay_grid_step_samples"])
    lags, baseline_score = _delay_score(
        case.sparse_b,
        case.sparse_a,
        intersection_mask,
        lag_limit,
        grid_step,
    )
    maximum_score_difference = 0.0
    maximum_generated_missing_energy = 0.0
    for method in PSEUDO_COMPLETION_METHODS:
        _, score = _delay_score(
            case.completions_b[method].spectrum,
            case.completions_a[method].spectrum,
            all_mask,
            lag_limit,
            grid_step,
        )
        maximum_score_difference = max(
            maximum_score_difference,
            float(np.max(np.abs(score - baseline_score))),
        )
        maximum_generated_missing_energy = max(
            maximum_generated_missing_energy,
            float(case.completions_a[method].generated_missing_energy)
            + float(case.completions_b[method].generated_missing_energy),
        )
    if maximum_score_difference > 1e-12:
        raise ValueError("机制样本中伪补全时延得分不再等价于真实交集")
    return {
        "case_seed": case.seed,
        "signal_kind": case.signal_kind,
        "snr_db": CASE_SNR_DB,
        "requested_overlap_fraction": CASE_OVERLAP_FRACTION,
        "actual_overlap_fraction": float(
            intersection.size / int(config["selected_bins"])
        ),
        "intersection_bins": int(intersection.size),
        "true_delay_samples": case.true_delay_samples,
        "delay_grid_points": int(lags.size),
        "maximum_score_difference": maximum_score_difference,
        "maximum_generated_missing_energy": maximum_generated_missing_energy,
    }


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _plot_interpolation_mechanism(
    cases: dict[str, MechanismCase],
    config: dict[str, object],
) -> plt.Figure:
    """按行展示四类频谱，按列分离完整频谱和机器精度缺失值。"""

    n_fft = int(config["n_fft"])
    factor = int(config["interpolation_factor"])
    fig, axes = plt.subplots(
        len(CASE_SIGNAL_KINDS),
        4,
        figsize=(16.2, 12.8),
    )
    completion_styles = {
        "zero_fill": (OKABE_ITO["black"], ":", "零填充"),
        "ideal_periodic": (
            OKABE_ITO["orange"],
            "--",
            "理想周期（与零填充重合）",
        ),
        "polyphase_fir": (OKABE_ITO["blue"], "-", "多相 FIR"),
        "linear_periodic": (OKABE_ITO["green"], "-.", "周期线性"),
        "cubic_periodic": (OKABE_ITO["purple"], (0, (5, 2)), "周期三次"),
    }

    for row, signal_kind in enumerate(CASE_SIGNAL_KINDS):
        case = cases[signal_kind]
        frequencies = case.frequencies
        common = np.intersect1d(case.mask_a, case.mask_b)
        axis_a, axis_b, axis_c, axis_d = axes[row]
        spectrum_scale = max(float(np.max(np.abs(case.noisy_a))), np.finfo(float).tiny)

        axis_a.plot(
            frequencies,
            np.abs(case.noisy_a) / spectrum_scale,
            color=OKABE_ITO["gray"],
            linewidth=0.9,
            alpha=0.85,
            label="完整 noisy 频谱",
        )
        axis_a.plot(
            frequencies,
            np.abs(case.clean_a) / spectrum_scale,
            color=OKABE_ITO["black"],
            linestyle="--",
            linewidth=1.1,
            label="无噪声真实包络",
        )
        axis_a.scatter(
            frequencies[case.mask_a],
            np.abs(case.noisy_a[case.mask_a]) / spectrum_scale,
            s=20,
            marker="o",
            facecolors="none",
            edgecolors=OKABE_ITO["blue"],
            linewidths=1.0,
            label="A 站已选",
        )
        axis_a.scatter(
            frequencies[case.mask_b],
            np.abs(case.noisy_b[case.mask_b]) / spectrum_scale,
            s=18,
            marker="x",
            color=OKABE_ITO["orange"],
            linewidths=1.0,
            label="B 站已选",
        )
        axis_a.scatter(
            frequencies[common],
            np.abs(case.noisy_a[common]) / spectrum_scale,
            s=26,
            marker="s",
            facecolors="none",
            edgecolors=OKABE_ITO["green"],
            linewidths=1.0,
            label="真实共同频点",
        )
        axis_a.set_ylabel(
            f"{SIGNAL_KIND_LABELS[signal_kind]}\n归一化幅度",
        )
        axis_a.grid(alpha=0.18)
        if row == 0:
            axis_a.set_title("无噪声包络、noisy 频谱与双站选频")
            axis_a.legend(
                frameon=True,
                facecolor="white",
                framealpha=0.95,
                edgecolor="#CCCCCC",
                fontsize=6.8,
                ncol=2,
                loc="lower center",
            )

        clean_time = np.fft.ifft(np.fft.ifftshift(case.clean_a), norm="ortho")
        full_time = np.fft.ifft(np.fft.ifftshift(case.noisy_a), norm="ortho")
        sparse_time = np.fft.ifft(np.fft.ifftshift(case.sparse_a), norm="ortho")
        output_length = n_fft * factor
        sparse_high_rate = signal.resample(sparse_time, output_length)
        high_time_grid = np.arange(output_length, dtype=float) / factor
        sample_grid = np.arange(n_fft, dtype=float)
        window = high_time_grid <= 30.0
        sample_window = sample_grid <= 30.0
        time_scale = max(
            float(np.max(np.abs(full_time[sample_window]))),
            np.finfo(float).tiny,
        )
        axis_b.plot(
            sample_grid[sample_window],
            clean_time[sample_window].real / time_scale,
            color=OKABE_ITO["gray"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label="无噪声 I",
        )
        axis_b.plot(
            sample_grid[sample_window],
            full_time[sample_window].real / time_scale,
            color=OKABE_ITO["black"],
            linewidth=0.9,
            alpha=0.85,
            label="含噪完整 I",
        )
        axis_b.plot(
            sample_grid[sample_window],
            full_time[sample_window].imag / time_scale,
            color=OKABE_ITO["gray"],
            linewidth=0.9,
            alpha=0.85,
            label="含噪完整 Q",
        )
        axis_b.plot(
            high_time_grid[window],
            sparse_high_rate[window].real / time_scale,
            color=OKABE_ITO["orange"],
            linewidth=1.1,
            label=f"稀疏 I：{factor} 倍插值",
        )
        axis_b.plot(
            high_time_grid[window],
            sparse_high_rate[window].imag / time_scale,
            color=OKABE_ITO["green"],
            linestyle="-.",
            linewidth=1.0,
            label=f"稀疏 Q：{factor} 倍插值",
        )
        axis_b.scatter(
            sample_grid[sample_window],
            sparse_time[sample_window].real / time_scale,
            s=10,
            color=OKABE_ITO["blue"],
            zorder=3,
            label="稀疏 I 原始点",
        )
        axis_b.scatter(
            sample_grid[sample_window],
            sparse_time[sample_window].imag / time_scale,
            s=11,
            marker="x",
            color=OKABE_ITO["purple"],
            zorder=3,
            label="稀疏 Q 原始点",
        )
        axis_b.set_ylabel("归一化 I/Q 幅度")
        axis_b.grid(alpha=0.18)
        if row == 0:
            axis_b.set_title("含噪 IQ 波形及稀疏波形插值")
            axis_b.legend(
                frameon=True,
                facecolor="white",
                framealpha=0.92,
                edgecolor="#CCCCCC",
                fontsize=6.1,
                ncol=2,
                loc="upper right",
            )

        axis_c.plot(
            frequencies,
            np.abs(case.noisy_a) / spectrum_scale,
            color=OKABE_ITO["gray"],
            linewidth=1.0,
            label="完整 noisy 频谱",
        )
        for method, (color, linestyle, label) in completion_styles.items():
            raw = case.completions_a[method].raw_mapped_spectrum
            axis_c.plot(
                frequencies,
                np.abs(raw) / spectrum_scale,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.9,
                label=label,
            )
        axis_c.set_ylabel("归一化幅度")
        axis_c.set_ylim(bottom=0.0)
        axis_c.grid(alpha=0.18)
        if row == 0:
            axis_c.set_title("映射回原 DFT 网格的完整频谱")
            handles, labels = axis_c.get_legend_handles_labels()
            left_legend = axis_c.legend(
                handles[:3],
                labels[:3],
                frameon=True,
                facecolor="white",
                framealpha=0.97,
                edgecolor="#CCCCCC",
                fontsize=6.4,
                loc="upper left",
            )
            axis_c.add_artist(left_legend)
            axis_c.legend(
                handles[3:],
                labels[3:],
                frameon=True,
                facecolor="white",
                framealpha=0.97,
                edgecolor="#CCCCCC",
                fontsize=6.4,
                loc="upper right",
            )

        missing = np.ones(n_fft, dtype=bool)
        missing[case.mask_a] = False
        missing_scale = max(
            float(np.max(np.abs(case.noisy_a[missing]))),
            np.finfo(float).tiny,
        )
        floor = 1e-17
        axis_d.semilogy(
            frequencies[missing],
            np.maximum(np.abs(case.noisy_a[missing]) / missing_scale, floor),
            color=OKABE_ITO["gray"],
            linewidth=1.0,
            label="真实缺失频谱",
        )
        for method, (color, linestyle, label) in completion_styles.items():
            raw = case.completions_a[method].raw_mapped_spectrum
            axis_d.semilogy(
                frequencies[missing],
                np.maximum(np.abs(raw[missing]) / missing_scale, floor),
                color=color,
                linestyle=linestyle,
                linewidth=0.9,
                alpha=0.9,
                label=label,
            )
        axis_d.set_ylim(5e-18, 2.0)
        axis_d.set_ylabel("相对幅度（对数轴）")
        axis_d.grid(alpha=0.18)
        if row == 0:
            axis_d.set_title("仅放大缺失频点")
            axis_d.legend(frameon=False, fontsize=6.8, ncol=2)

        for column, axis in enumerate((axis_a, axis_b, axis_c, axis_d)):
            _panel_label(axis, f"{chr(65 + row)}{column + 1}")
            if row == len(CASE_SIGNAL_KINDS) - 1:
                if column == 1:
                    axis.set_xlabel("时间（sample）")
                else:
                    axis.set_xlabel("归一化频率（cycles/sample）")

    seed_text = "、".join(
        f"{SIGNAL_KIND_LABELS[k]}={CASE_SEEDS[k]}" for k in CASE_SIGNAL_KINDS
    )
    fig.suptitle(
        f"四类固定机制样本：SNR={CASE_SNR_DB:g} dB，"
        f"重叠率={CASE_OVERLAP_FRACTION:g}；子种子：{seed_text}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def _plot_delay_scores(
    case: MechanismCase,
    config: dict[str, object],
) -> plt.Figure:
    n_fft = int(config["n_fft"])
    lag_limit = float(config["lag_limit_samples"])
    grid_step = float(config["delay_grid_step_samples"])
    all_mask = np.ones(n_fft, dtype=bool)
    intersection = np.intersect1d(case.mask_a, case.mask_b)
    intersection_mask = np.zeros(n_fft, dtype=bool)
    intersection_mask[intersection] = True
    union = np.union1d(case.mask_a, case.mask_b)
    union_mask = np.zeros(n_fft, dtype=bool)
    union_mask[union] = True

    zhai_selection = select_zhai_thesis(
        np.abs(case.noisy_b) ** 2,
        int(config["selected_bins"]),
        case.frequencies,
        int(config["zhai_max_sweeps"]),
        float(config["zhai_relative_tolerance"]),
    )
    zhai_mask = np.zeros(n_fft, dtype=bool)
    zhai_mask[zhai_selection.indices] = True

    lags, full_score = _delay_score(
        case.noisy_b, case.noisy_a, all_mask, lag_limit, grid_step
    )
    _, zhai_score = _delay_score(
        case.noisy_b, case.noisy_a, zhai_mask, lag_limit, grid_step
    )
    _, intersection_score = _delay_score(
        case.sparse_b, case.sparse_a, intersection_mask, lag_limit, grid_step
    )
    pseudo_scores = {}
    for method in PSEUDO_COMPLETION_METHODS:
        _, pseudo_scores[method] = _delay_score(
            case.completions_b[method].spectrum,
            case.completions_a[method].spectrum,
            all_mask,
            lag_limit,
            grid_step,
        )
    _, oracle_score = _delay_score(
        case.noisy_b, case.noisy_a, union_mask, lag_limit, grid_step
    )

    fig, (axis_a, axis_b) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.8),
        gridspec_kw={"height_ratios": (2.1, 1.4)},
    )
    axis_a.plot(
        lags,
        full_score,
        color=OKABE_ITO["blue"],
        linewidth=1.4,
        label="完整频谱",
    )
    axis_a.plot(
        lags,
        zhai_score,
        color=OKABE_ITO["sky"],
        linestyle=(0, (5, 2)),
        linewidth=1.2,
        label="Zhai 非对称",
    )
    axis_a.plot(
        lags,
        oracle_score,
        color=OKABE_ITO["green"],
        linestyle="-.",
        linewidth=1.3,
        label="真实联合上界",
    )
    axis_a.plot(
        lags,
        intersection_score,
        color=OKABE_ITO["black"],
        linewidth=1.5,
        label="真实交集",
    )
    axis_a.plot(
        lags,
        pseudo_scores["ideal_periodic"],
        color=OKABE_ITO["orange"],
        linestyle="--",
        linewidth=1.3,
        label="五种伪补全（得分完全重合）",
    )
    axis_a.axvline(
        case.true_delay_samples,
        color=OKABE_ITO["red"],
        linestyle="--",
        linewidth=1.2,
        label=f"真实时延 {case.true_delay_samples:.3f}",
    )
    axis_a.set_ylabel("归一化时延匹配得分")
    axis_a.set_title("插值前后连续时延匹配得分")
    axis_a.legend(frameon=False, ncol=3)
    axis_a.grid(alpha=0.2)
    _panel_label(axis_a, "A")

    score_by_method = {
        "Full": full_score,
        "Zhai-Asymmetric": zhai_score,
        "Intersection-Continuous": intersection_score,
        **pseudo_scores,
        "Oracle-Union": oracle_score,
    }
    method_order = (
        "Full",
        "Zhai-Asymmetric",
        "Intersection-Continuous",
        "zero_fill",
        "ideal_periodic",
        "polyphase_fir",
        "linear_periodic",
        "cubic_periodic",
        "Oracle-Union",
    )
    method_labels = {
        **METHOD_LABELS,
        "Zhai-Asymmetric": "Zhai 非对称",
    }
    method_colors = {
        "Full": OKABE_ITO["blue"],
        "Zhai-Asymmetric": OKABE_ITO["sky"],
        "Intersection-Continuous": OKABE_ITO["black"],
        "zero_fill": OKABE_ITO["gray"],
        "ideal_periodic": OKABE_ITO["orange"],
        "polyphase_fir": OKABE_ITO["blue"],
        "linear_periodic": OKABE_ITO["green"],
        "cubic_periodic": OKABE_ITO["purple"],
        "Oracle-Union": OKABE_ITO["green"],
    }
    estimates = np.array(
        [lags[int(np.argmax(score_by_method[method]))] for method in method_order]
    )
    y_positions = np.arange(len(method_order))
    axis_b.scatter(
        estimates,
        y_positions,
        s=38,
        c=[method_colors[method] for method in method_order],
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="各方法点估计",
    )
    for y_position, estimate in zip(y_positions, estimates, strict=True):
        axis_b.plot(
            [case.true_delay_samples, estimate],
            [y_position, y_position],
            color=OKABE_ITO["gray"],
            linewidth=0.7,
            alpha=0.6,
        )
    axis_b.axvline(
        case.true_delay_samples,
        color=OKABE_ITO["red"],
        linestyle="--",
        linewidth=1.2,
        label=f"真实时延 {case.true_delay_samples:.3f}",
    )
    axis_b.set_yticks(
        y_positions,
        [method_labels[method] for method in method_order],
    )
    axis_b.invert_yaxis()
    axis_b.set_xlabel("TDOA 点估计（sample）")
    axis_b.set_title("九种对照方法的点估计")
    axis_b.legend(frameon=False, ncol=2)
    axis_b.grid(axis="x", alpha=0.2)
    axis_b.text(
        0.99,
        0.04,
        "五种伪补全方法位于不同行，点估计数值完全相同",
        transform=axis_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    _panel_label(axis_b, "B")
    fig.tight_layout()
    return fig


def _aggregate_performance(
    trials: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return (
        trials.groupby(["method", *group_columns], dropna=False)
        .agg(
            mae_samples=("absolute_error_samples", "mean"),
            outlier_rate=("outlier_gt_1_sample", "mean"),
            valid_trials=("absolute_error_samples", "count"),
        )
        .reset_index()
    )


def _plot_performance(
    trials: pd.DataFrame,
    paired: pd.DataFrame,
) -> plt.Figure:
    selected_methods = (
        "Full",
        "Oracle-Union",
        "Intersection-Continuous",
        "ideal_periodic",
    )
    styles = {
        "Full": (OKABE_ITO["blue"], "-", "o", "完整频谱"),
        "Oracle-Union": (OKABE_ITO["green"], "-.", "s", "真实联合上界"),
        "Intersection-Continuous": (
            OKABE_ITO["black"],
            "-",
            "^",
            "真实交集",
        ),
        "ideal_periodic": (
            OKABE_ITO["orange"],
            "--",
            "x",
            "五种伪补全（完全重合）",
        ),
    }
    overlap_summary = _aggregate_performance(trials, ["overlap_fraction"])
    snr_summary = _aggregate_performance(
        trials.loc[np.isclose(trials["overlap_fraction"], 0.5)],
        ["snr_db"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    axis_a, axis_b, axis_c, axis_d = axes.flat
    for method in selected_methods:
        color, linestyle, marker, label = styles[method]
        rows = overlap_summary.loc[overlap_summary["method"].eq(method)].sort_values(
            "overlap_fraction"
        )
        axis_a.plot(
            rows["overlap_fraction"],
            rows["mae_samples"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.4,
            label=label,
        )
        axis_b.plot(
            rows["overlap_fraction"],
            100.0 * rows["outlier_rate"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.4,
            label=label,
        )
        snr_rows = snr_summary.loc[snr_summary["method"].eq(method)].sort_values(
            "snr_db"
        )
        axis_c.plot(
            snr_rows["snr_db"],
            snr_rows["mae_samples"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.4,
            label=label,
        )

    axis_a.set_title("平均误差随重叠率变化")
    axis_a.set_xlabel("双站频点重叠率")
    axis_a.set_ylabel("TDOA MAE（sample）")
    axis_a.grid(alpha=0.2)
    axis_a.legend(frameon=False)
    _panel_label(axis_a, "A")

    axis_b.set_title("离群率随重叠率变化")
    axis_b.set_xlabel("双站频点重叠率")
    axis_b.set_ylabel("误差大于 1 sample 的比例（%）")
    axis_b.grid(alpha=0.2)
    axis_b.legend(frameon=False)
    _panel_label(axis_b, "B")

    axis_c.set_title("50% 重叠时的 SNR 曲线")
    axis_c.set_xlabel("SNR（dB）")
    axis_c.set_ylabel("TDOA MAE（sample）")
    axis_c.grid(alpha=0.2)
    axis_c.legend(frameon=False)
    _panel_label(axis_c, "C")

    paired_rows = (
        paired.loc[paired["method"].isin(PSEUDO_COMPLETION_METHODS)]
        .set_index("method")
        .loc[list(PSEUDO_COMPLETION_METHODS)]
        .reset_index()
    )
    y_positions = np.arange(len(paired_rows))
    improvements = paired_rows["mean_mae_improvement_samples"].to_numpy()
    lower = improvements - paired_rows["ci95_low"].to_numpy()
    upper = paired_rows["ci95_high"].to_numpy() - improvements
    axis_d.errorbar(
        improvements,
        y_positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=OKABE_ITO["orange"],
        ecolor=OKABE_ITO["gray"],
        capsize=3,
        label="平均改善及 95% 置信区间",
    )
    axis_d.axvline(0.0, color=OKABE_ITO["black"], linewidth=0.9)
    axis_d.set_yticks(
        y_positions,
        [METHOD_LABELS[method] for method in paired_rows["method"]],
    )
    axis_d.set_xlim(-0.02, 0.02)
    axis_d.set_xlabel("相对真实交集的 MAE 改善（sample）")
    axis_d.set_title("配对增益及 95% 置信区间")
    axis_d.grid(axis="x", alpha=0.2)
    axis_d.legend(frameon=False, loc="lower right")
    _panel_label(axis_d, "D")

    fig.tight_layout()
    return fig


def _save_figure(
    figure: plt.Figure,
    stem: str,
    result_figure_directory: Path,
    asset_directory: Path | None,
) -> list[Path]:
    saved: list[Path] = []
    for suffix in ("svg", "png"):
        result_path = result_figure_directory / f"{stem}.{suffix}"
        save_kwargs = {"dpi": 300} if suffix == "png" else {}
        figure.savefig(result_path, **save_kwargs)
        saved.append(result_path)
        if asset_directory is not None:
            shutil.copy2(result_path, asset_directory / result_path.name)
    plt.close(figure)
    return saved


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取既有插值实验结果并生成导师汇报图",
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=DEFAULT_RESULT_DIRECTORY,
        help="synthetic_full 正式结果目录",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验正式结果和固定机制样本，不生成图片",
    )
    parser.add_argument(
        "--no-asset-copy",
        action="store_true",
        help="只写入正式结果 figures 目录，不复制到 Obsidian assets",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    result_directory = args.result_directory.resolve()
    config, trials, paired = _load_formal_results(result_directory)
    formal_validation = _validate_formal_results(trials, paired)
    cases = {
        signal_kind: _make_mechanism_case(
            config,
            signal_kind,
            CASE_SEEDS[signal_kind],
        )
        for signal_kind in CASE_SIGNAL_KINDS
    }
    case_validation = {
        signal_kind: _validate_mechanism_case(case, config)
        for signal_kind, case in cases.items()
    }
    validation = {
        "formal_results": formal_validation,
        "mechanism_cases": case_validation,
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.validate_only:
        print("校验完成：未生成或覆盖任何图片。")
        return

    _configure_figure_style()
    result_figure_directory = result_directory / "figures" / "advisor_report"
    result_figure_directory.mkdir(parents=True, exist_ok=True)
    asset_directory = None if args.no_asset_copy else REPORT_ASSET_DIRECTORY
    if asset_directory is not None:
        asset_directory.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    saved.extend(
        _save_figure(
            _plot_interpolation_mechanism(cases, config),
            "figure_01_interpolation_mechanism",
            result_figure_directory,
            asset_directory,
        )
    )
    saved.extend(
        _save_figure(
            _plot_delay_scores(cases["smooth_complex"], config),
            "figure_02_delay_score_comparison",
            result_figure_directory,
            asset_directory,
        )
    )
    saved.extend(
        _save_figure(
            _plot_performance(trials, paired),
            "figure_03_tdoa_performance",
            result_figure_directory,
            asset_directory,
        )
    )
    manifest = {
        "source_result_directory": str(result_directory),
        "report_asset_directory": (
            str(asset_directory) if asset_directory is not None else None
        ),
        "figures": [str(path) for path in saved],
        "validation": validation,
        "interpretation": (
            "四类固定机制样本只用于解释插值过程；图2用平滑复谱列出九种"
            "方法；TDOA统计图来自既有synthetic_full正式结果。"
        ),
    }
    manifest_path = result_figure_directory / "advisor_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"导师汇报图已保存：{result_figure_directory}")
    if asset_directory is not None:
        print(f"Obsidian 图片副本已保存：{asset_directory}")


if __name__ == "__main__":
    main()
