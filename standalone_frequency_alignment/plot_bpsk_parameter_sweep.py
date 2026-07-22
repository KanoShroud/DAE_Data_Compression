"""为 BPSK 扩展参数扫描生成导师汇报图和数据表。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = MODULE_DIRECTORY.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from alignment_core import complete_spectrum, shifted_frequencies  # noqa: E402
from plot_interpolation_mechanism import (  # noqa: E402
    OKABE_ITO,
    _configure_figure_style,
    _save_figure,
)
from run_bpsk_parameter_sweep import (  # noqa: E402
    METHOD_ORDER,
    PSEUDO_COMPLETION_METHODS,
    SIGNAL_KIND_LABELS,
    SIGNAL_KINDS,
    SweepConfig,
    TZP_DENSE_METHOD,
    _add_frequency_noise,
    _build_conditions,
    _controlled_masks,
    _fft_delay_estimates,
    _make_source_spectrum,
)
from tzp_dense import (  # noqa: E402
    dense_delay_estimates,
    dense_delay_score_grid,
    direct_dense_delay_estimate,
    sample_original_grid,
    time_zero_padded_dense_spectrum,
)

DEFAULT_RESULT_DIRECTORY = (
    PROJECT_DIRECTORY
    / "运行结果"
    / "频谱插值对齐"
    / "20260721_172703_080303_bpsk_parameter_sweep"
)
REPORT_ASSET_DIRECTORY = (
    PROJECT_DIRECTORY
    / "Obsidian Vault"
    / "assets"
    / "插值频谱补全汇报_BPSK扩展_20260720"
)

CASE_BASE_SEED = 20260724
CASE_SEEDS = {
    signal_kind: CASE_BASE_SEED + index
    for index, signal_kind in enumerate(SIGNAL_KINDS)
}
CASE_SNR_DB = 10.0
CASE_SELECTED_BINS = 128
CASE_OVERLAP_FRACTION = 0.5
RMSE_BOOTSTRAP_RESAMPLES = 2000
RMSE_BOOTSTRAP_BASE_SEED = 20260721
ROBUSTNESS_BOOTSTRAP_RESAMPLES = 2000
ROBUSTNESS_BOOTSTRAP_SEED = 20260722
ESTIMATOR_REPLAY_PER_SIGNAL_BUDGET = 2

METHOD_LABELS = {
    "Full": "完整频谱参考",
    "Zhai-Asymmetric": "非对称选频",
    "Intersection-Continuous": "真实交集（=五种原网格补全）",
    "Oracle-Union": "联合集合真实频谱对照",
    "zero_fill": "零填充",
    "ideal_periodic": "理想周期",
    "polyphase_fir": "多相 FIR",
    "linear_periodic": "周期线性",
    "cubic_periodic": "周期三次",
    TZP_DENSE_METHOD: "时域补零密集频谱",
}

EFFECTIVE_BIN_RELATIVE_THRESHOLD = 1e-10


def _panel_caption(
    axis: plt.Axes,
    panel_label: str,
    title: str,
    *,
    y: float,
) -> None:
    """把子图编号与简短标题统一放在坐标轴下方。"""
    axis.text(
        0.5,
        y,
        f"{panel_label}. {title}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="semibold",
        clip_on=False,
    )
KEY_METHODS = (
    "Full",
    "Zhai-Asymmetric",
    "Intersection-Continuous",
    TZP_DENSE_METHOD,
    "Oracle-Union",
)
METHOD_STYLES = {
    "Full": (OKABE_ITO["black"], "-", "o"),
    "Zhai-Asymmetric": (OKABE_ITO["blue"], "--", "s"),
    "Intersection-Continuous": (OKABE_ITO["orange"], "-.", "^"),
    TZP_DENSE_METHOD: (OKABE_ITO["purple"], (0, (5, 2)), "x"),
    "Oracle-Union": (OKABE_ITO["green"], ":", "D"),
}
COMPLETION_STYLES = {
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


@dataclass(frozen=True)
class MechanismCase:
    signal_kind: str
    seed: int
    frequencies: np.ndarray
    true_delay_samples: float
    clean_a: np.ndarray
    noisy_a: np.ndarray
    noisy_b: np.ndarray
    mask_a: np.ndarray
    mask_b: np.ndarray
    sparse_a: np.ndarray
    sparse_b: np.ndarray
    completions_a: dict[str, object]
    completions_b: dict[str, object]


def _load_results(
    result_directory: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = (
        "config.json",
        "summary.csv",
        "paired_comparisons.csv",
        "operator_diagnostics.csv",
        "COMPLETED.json",
    )
    missing = [name for name in required if not (result_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"新正式结果尚不完整：{missing}")
    config = json.loads(
        (result_directory / "config.json").read_text(encoding="utf-8")
    )
    completed = json.loads(
        (result_directory / "COMPLETED.json").read_text(encoding="utf-8")
    )
    if not completed.get("completed"):
        raise ValueError("COMPLETED.json 未确认正式运行完成")
    summary = pd.read_csv(result_directory / "summary.csv")
    paired = pd.read_csv(result_directory / "paired_comparisons.csv")
    diagnostics = pd.read_csv(result_directory / "operator_diagnostics.csv")
    return config, summary, paired, diagnostics, completed


def _read_trial_chunks(result_directory: Path) -> pd.DataFrame:
    paths = sorted((result_directory / "condition_chunks").glob("*.csv.gz"))
    if len(paths) != 500:
        raise ValueError(f"应有 500 个 trial chunk，实际为 {len(paths)}")
    columns = (
        "trial_id",
        "condition_index",
        "trial_index",
        "method",
        "snr_db",
        "signal_kind",
        "selected_bins",
        "overlap_fraction",
        "estimated_delay_samples",
        "true_delay_samples",
        "absolute_error_samples",
        "squared_error_samples",
        "outlier_gt_1_sample",
        "peak_sidelobe_ratio",
        "observed_consistency_error",
        "generated_missing_energy",
        "missing_nmse_clean_db",
        "missing_nmse_noisy_db",
        "dense_grid_nmse_clean_db",
        "dense_grid_nmse_noisy_db",
        "dense_grid_phase_rmse_rad",
        "dense_grid_complex_correlation",
        "dense_grid_common_nonzero_rate",
        "dense_grid_points",
        "time_zero_padding_factor",
    )
    return pd.concat(
        (pd.read_csv(path, usecols=columns) for path in paths),
        ignore_index=True,
    )


def _validate_results(
    config: dict[str, object],
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    completed: dict[str, object],
) -> dict[str, object]:
    expected_conditions = (
        len(config["signal_kinds"])
        * len(config["selected_bins_values"])
        * len(config["snr_values_db"])
        * len(config["overlap_values"])
    )
    expected_summary_rows = expected_conditions * len(METHOD_ORDER)
    if int(completed["conditions"]) != expected_conditions:
        raise ValueError("完成清单中的条件数与 config 不一致")
    if len(summary) != expected_summary_rows:
        raise ValueError("summary.csv 行数与完整条件网格不一致")
    if set(summary["method"]) != set(METHOD_ORDER):
        raise ValueError("summary.csv 方法集合不完整")
    if int(config["interpolation_factor"]) != 8:
        raise ValueError("TZP-DS 正式协议要求时域补零倍数固定为 8")
    pseudo_paired = paired.loc[paired["method"].isin(PSEUDO_COMPLETION_METHODS)]
    paired_maximum = float(
        np.max(
            np.abs(
                pseudo_paired[
                    [
                        "mean_mae_improvement_samples",
                        "ci95_low",
                        "ci95_high",
                        "maximum_abs_improvement_samples",
                    ]
                ].to_numpy()
            ),
            initial=0.0,
        )
    )
    if paired_maximum != 0.0:
        raise ValueError("原网格确定性补全不再与真实交集逐试验完全相同")
    if not diagnostics["passed"].astype(bool).all():
        raise ValueError("operator_diagnostics.csv 存在未通过数值一致性检验")
    tzp_gates = diagnostics.loc[diagnostics["method"].eq(TZP_DENSE_METHOD)]
    if len(tzp_gates) != len(config["selected_bins_values"]):
        raise ValueError("TZP-DS 数值一致性检验未覆盖全部 M")
    tzp_paired = paired.loc[paired["method"].eq(TZP_DENSE_METHOD)]
    if len(tzp_paired) != 1:
        raise ValueError("TZP-DS 配对统计缺失")
    pseudo_gates = diagnostics.loc[
        diagnostics["method"].isin(PSEUDO_COMPLETION_METHODS)
    ]
    return {
        "expected_conditions": expected_conditions,
        "summary_rows": int(len(summary)),
        "base_trials": int(completed["base_trials"]),
        "method_rows": int(completed["method_rows"]),
        "paired_pseudo_maximum_abs": paired_maximum,
        "operator_gate_maximum_missing_energy": float(
            diagnostics["generated_missing_energy"].max()
        ),
        "operator_gate_maximum_spectrum_difference": float(
            pseudo_gates["maximum_spectrum_difference_vs_zero"].max()
        ),
        "tzp_gate_cases": int(len(tzp_gates)),
        "tzp_gate_maximum_original_missing_energy": float(
            tzp_gates["generated_missing_energy"].max()
        ),
        "tzp_gate_maximum_consistency_error": float(
            tzp_gates["observed_consistency_error"].max()
        ),
        "tzp_gate_maximum_rank_difference": float(
            np.max(
                np.abs(
                    tzp_gates["operator_rank"].to_numpy()
                    - tzp_gates["expected_rank"].to_numpy()
                )
            )
        ),
    }


def _make_mechanism_case(
    config: SweepConfig,
    signal_kind: str,
    seed: int,
) -> MechanismCase:
    rng = np.random.default_rng(seed)
    frequencies = shifted_frequencies(config.n_fft)
    source = _make_source_spectrum(config, signal_kind, rng)
    true_delay = float(
        rng.uniform(
            config.true_delay_min_samples,
            config.true_delay_max_samples,
        )
    )
    gain = np.exp(1j * rng.uniform(-np.pi, np.pi))
    clean_a = source
    clean_b = gain * source * np.exp(-2j * np.pi * frequencies * true_delay)
    noisy_a = _add_frequency_noise(clean_a, CASE_SNR_DB, rng)
    noisy_b = _add_frequency_noise(clean_b, CASE_SNR_DB, rng)
    mask_a, mask_b = _controlled_masks(
        config.n_fft,
        CASE_SELECTED_BINS,
        CASE_OVERLAP_FRACTION,
        rng,
    )
    sparse_a = np.zeros(config.n_fft, dtype=complex)
    sparse_b = np.zeros(config.n_fft, dtype=complex)
    sparse_a[mask_a] = noisy_a[mask_a]
    sparse_b[mask_b] = noisy_b[mask_b]
    completions_a = {
        method: complete_spectrum(
            sparse_a,
            mask_a,
            method,
            config.interpolation_factor,
        )
        for method in PSEUDO_COMPLETION_METHODS
    }
    completions_b = {
        method: complete_spectrum(
            sparse_b,
            mask_b,
            method,
            config.interpolation_factor,
        )
        for method in PSEUDO_COMPLETION_METHODS
    }
    return MechanismCase(
        signal_kind=signal_kind,
        seed=seed,
        frequencies=frequencies,
        true_delay_samples=true_delay,
        clean_a=clean_a,
        noisy_a=noisy_a,
        noisy_b=noisy_b,
        mask_a=mask_a,
        mask_b=mask_b,
        sparse_a=sparse_a,
        sparse_b=sparse_b,
        completions_a=completions_a,
        completions_b=completions_b,
    )


def _effective_bin_mask(
    mapped_spectrum: np.ndarray,
    sparse_spectrum: np.ndarray,
    observed_indices: np.ndarray,
) -> np.ndarray:
    """按相对观测幅度阈值识别插值后原 DFT 网格上的有效频点。"""

    observed = np.asarray(observed_indices, dtype=int)
    observed_values = np.asarray(sparse_spectrum)[observed]
    scale = max(float(np.max(np.abs(observed_values))), np.finfo(float).tiny)
    threshold = max(
        EFFECTIVE_BIN_RELATIVE_THRESHOLD * scale,
        100.0 * np.finfo(float).eps * float(np.linalg.norm(observed_values)),
    )
    return np.abs(np.asarray(mapped_spectrum)) > threshold


def _audit_reconstructed_spectra(config: SweepConfig) -> pd.DataFrame:
    """精确重放每个正式条件的首个试验，独立审计原网格确定性补全输出与 TDOA。"""

    frequencies = shifted_frequencies(config.n_fft)
    audit_payloads: list[dict[str, object]] = []
    cross_spectra: list[np.ndarray] = []
    active_counts: list[int] = []
    for condition in _build_conditions(config):
        rng = np.random.default_rng(condition.seed)
        source = _make_source_spectrum(config, condition.signal_kind, rng)
        true_delay = float(
            rng.uniform(
                config.true_delay_min_samples,
                config.true_delay_max_samples,
            )
        )
        gain = np.exp(1j * rng.uniform(-np.pi, np.pi))
        noisy_a = _add_frequency_noise(source, condition.snr_db, rng)
        noisy_b = _add_frequency_noise(
            gain * source * np.exp(-2j * np.pi * frequencies * true_delay),
            condition.snr_db,
            rng,
        )
        mask_a, mask_b = _controlled_masks(
            config.n_fft,
            condition.selected_bins,
            condition.overlap_fraction,
            rng,
        )
        sparse_a = np.zeros(config.n_fft, dtype=complex)
        sparse_b = np.zeros(config.n_fft, dtype=complex)
        sparse_a[mask_a] = noisy_a[mask_a]
        sparse_b[mask_b] = noisy_b[mask_b]
        intersection = np.intersect1d(mask_a, mask_b)
        baseline_cross = np.zeros(config.n_fft, dtype=complex)
        baseline_cross[intersection] = (
            noisy_b[intersection] * np.conj(noisy_a[intersection])
        )
        cross_spectra.append(baseline_cross)
        active_counts.append(int(intersection.size))
        audit_payloads.append(
            {
                "trial_id": (
                    f"{condition.signal_kind}|M{condition.selected_bins}|"
                    f"{condition.snr_db:g}|{condition.overlap_fraction:g}|0"
                ),
                "condition_index": condition.index,
                "signal_kind": condition.signal_kind,
                "selected_bins": condition.selected_bins,
                "snr_db": condition.snr_db,
                "overlap_fraction": condition.overlap_fraction,
                "true_delay_samples": true_delay,
                "method": "Intersection-Continuous",
                "effective_common_bins": int(intersection.size),
                "effective_common_rate": intersection.size / condition.selected_bins,
            }
        )
        for method in PSEUDO_COMPLETION_METHODS:
            completed_a = complete_spectrum(
                sparse_a,
                mask_a,
                method,
                config.interpolation_factor,
            )
            completed_b = complete_spectrum(
                sparse_b,
                mask_b,
                method,
                config.interpolation_factor,
            )
            active_a = _effective_bin_mask(
                completed_a.raw_mapped_spectrum,
                sparse_a,
                mask_a,
            )
            active_b = _effective_bin_mask(
                completed_b.raw_mapped_spectrum,
                sparse_b,
                mask_b,
            )
            effective_common = active_a & active_b
            cross = np.zeros(config.n_fft, dtype=complex)
            cross[effective_common] = (
                completed_b.spectrum[effective_common]
                * np.conj(completed_a.spectrum[effective_common])
            )
            cross_spectra.append(cross)
            active_counts.append(int(np.count_nonzero(effective_common)))
            audit_payloads.append(
                {
                    "trial_id": (
                        f"{condition.signal_kind}|M{condition.selected_bins}|"
                        f"{condition.snr_db:g}|{condition.overlap_fraction:g}|0"
                    ),
                    "condition_index": condition.index,
                    "signal_kind": condition.signal_kind,
                    "selected_bins": condition.selected_bins,
                    "snr_db": condition.snr_db,
                    "overlap_fraction": condition.overlap_fraction,
                    "true_delay_samples": true_delay,
                    "method": method,
                    "effective_common_bins": int(
                        np.count_nonzero(effective_common)
                    ),
                    "effective_common_rate": (
                        np.count_nonzero(effective_common)
                        / condition.selected_bins
                    ),
                }
            )
    estimates, _, _ = _fft_delay_estimates(
        np.stack(cross_spectra),
        np.asarray(active_counts),
        config,
    )
    audit = pd.DataFrame(audit_payloads)
    audit["estimated_delay_samples"] = estimates
    baseline = audit.loc[
        audit["method"].eq("Intersection-Continuous"),
        ["condition_index", "estimated_delay_samples"],
    ].rename(columns={"estimated_delay_samples": "baseline_delay_samples"})
    audit = audit.merge(baseline, on="condition_index", validate="many_to_one")
    audit["delay_difference_samples"] = (
        audit["estimated_delay_samples"] - audit["baseline_delay_samples"]
    )
    audit["absolute_error_difference_samples"] = (
        np.abs(audit["estimated_delay_samples"] - audit["true_delay_samples"])
        - np.abs(audit["baseline_delay_samples"] - audit["true_delay_samples"])
    )
    return audit


def _plot_mechanism(
    cases: dict[str, MechanismCase],
    config: SweepConfig,
) -> plt.Figure:
    fig, axes = plt.subplots(5, 4, figsize=(16.2, 18.2))
    panel_titles = (
        "含噪频谱与选频",
        "IQ 波形插值",
        "原网格完整频谱",
        "缺失频点放大",
    )
    for row, signal_kind in enumerate(SIGNAL_KINDS):
        case = cases[signal_kind]
        axis_a, axis_b, axis_c, axis_d = axes[row]
        frequencies = case.frequencies
        common = np.intersect1d(case.mask_a, case.mask_b)
        spectrum_scale = max(float(np.max(np.abs(case.noisy_a))), np.finfo(float).tiny)

        axis_a.plot(
            frequencies,
            np.abs(case.noisy_a) / spectrum_scale,
            color=OKABE_ITO["gray"],
            linewidth=0.8,
            alpha=0.85,
            label="完整含噪频谱",
        )
        axis_a.plot(
            frequencies,
            np.abs(case.clean_a) / spectrum_scale,
            color=OKABE_ITO["black"],
            linestyle="--",
            linewidth=1.0,
            label="无噪声真实包络",
        )
        axis_a.scatter(
            frequencies[case.mask_a],
            np.abs(case.noisy_a[case.mask_a]) / spectrum_scale,
            s=13,
            marker="o",
            facecolors="none",
            edgecolors=OKABE_ITO["blue"],
            linewidths=0.8,
            label="A 站已选",
        )
        axis_a.scatter(
            frequencies[case.mask_b],
            np.abs(case.noisy_b[case.mask_b]) / spectrum_scale,
            s=12,
            marker="x",
            color=OKABE_ITO["orange"],
            linewidths=0.8,
            label="B 站已选",
        )
        axis_a.scatter(
            frequencies[common],
            np.abs(case.noisy_a[common]) / spectrum_scale,
            s=16,
            marker="s",
            facecolors="none",
            edgecolors=OKABE_ITO["green"],
            linewidths=0.8,
            label="真实共同频点",
        )
        axis_a.set_ylabel(f"{SIGNAL_KIND_LABELS[signal_kind]}\n归一化幅度")
        axis_a.grid(alpha=0.18)
        if row == 0:
            axis_a.legend(
                frameon=True,
                facecolor="white",
                framealpha=0.94,
                edgecolor="#CCCCCC",
                fontsize=6.2,
                ncol=2,
                loc="lower center",
            )

        clean_time = np.fft.ifft(np.fft.ifftshift(case.clean_a), norm="ortho")
        noisy_time = np.fft.ifft(np.fft.ifftshift(case.noisy_a), norm="ortho")
        sparse_time = np.fft.ifft(np.fft.ifftshift(case.sparse_a), norm="ortho")
        output_length = config.n_fft * config.interpolation_factor
        sparse_high_rate = signal.resample(sparse_time, output_length)
        high_grid = np.arange(output_length) / config.interpolation_factor
        sample_grid = np.arange(config.n_fft)
        high_window = high_grid <= 30.0
        sample_window = sample_grid <= 30.0
        time_scale = max(
            float(np.max(np.abs(noisy_time[sample_window]))),
            np.finfo(float).tiny,
        )
        axis_b.plot(
            sample_grid[sample_window],
            clean_time[sample_window].real / time_scale,
            color=OKABE_ITO["gray"],
            linestyle="--",
            linewidth=0.8,
            label="无噪声 I",
        )
        axis_b.plot(
            sample_grid[sample_window],
            noisy_time[sample_window].real / time_scale,
            color=OKABE_ITO["black"],
            linewidth=0.85,
            label="含噪完整 I",
        )
        axis_b.plot(
            sample_grid[sample_window],
            noisy_time[sample_window].imag / time_scale,
            color=OKABE_ITO["gray"],
            linewidth=0.85,
            label="含噪完整 Q",
        )
        axis_b.plot(
            high_grid[high_window],
            sparse_high_rate[high_window].real / time_scale,
            color=OKABE_ITO["orange"],
            linewidth=1.0,
            label="稀疏 I：8 倍插值",
        )
        axis_b.plot(
            high_grid[high_window],
            sparse_high_rate[high_window].imag / time_scale,
            color=OKABE_ITO["green"],
            linestyle="-.",
            linewidth=0.9,
            label="稀疏 Q：8 倍插值",
        )
        axis_b.scatter(
            sample_grid[sample_window],
            sparse_time[sample_window].real / time_scale,
            s=8,
            color=OKABE_ITO["blue"],
            zorder=3,
            label="稀疏 I 原始点",
        )
        axis_b.scatter(
            sample_grid[sample_window],
            sparse_time[sample_window].imag / time_scale,
            s=9,
            marker="x",
            color=OKABE_ITO["purple"],
            zorder=3,
            label="稀疏 Q 原始点",
        )
        axis_b.set_ylabel("归一化 I/Q 幅度")
        axis_b.grid(alpha=0.18)
        if row == 0:
            axis_b.legend(
                frameon=True,
                facecolor="white",
                framealpha=0.92,
                edgecolor="#CCCCCC",
                fontsize=5.8,
                ncol=2,
                loc="upper right",
            )

        axis_c.plot(
            frequencies,
            np.abs(case.noisy_a) / spectrum_scale,
            color=OKABE_ITO["gray"],
            linewidth=0.9,
            label="完整含噪频谱",
        )
        for method, (color, linestyle, label) in COMPLETION_STYLES.items():
            raw = case.completions_a[method].raw_mapped_spectrum
            axis_c.plot(
                frequencies,
                np.abs(raw) / spectrum_scale,
                color=color,
                linestyle=linestyle,
                linewidth=0.9,
                alpha=0.9,
                label=label,
            )
        axis_c.set_ylim(bottom=0.0)
        axis_c.set_ylabel("归一化幅度")
        axis_c.grid(alpha=0.18)
        if row == 0:
            handles, labels = axis_c.get_legend_handles_labels()
            left = axis_c.legend(
                handles[:3],
                labels[:3],
                frameon=True,
                facecolor="white",
                framealpha=0.97,
                edgecolor="#CCCCCC",
                fontsize=6.0,
                loc="upper left",
            )
            axis_c.add_artist(left)
            axis_c.legend(
                handles[3:],
                labels[3:],
                frameon=True,
                facecolor="white",
                framealpha=0.97,
                edgecolor="#CCCCCC",
                fontsize=6.0,
                loc="upper right",
            )

        missing = np.ones(config.n_fft, dtype=bool)
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
            linewidth=0.9,
            label="真实缺失频谱",
        )
        for method, (color, linestyle, label) in COMPLETION_STYLES.items():
            raw = case.completions_a[method].raw_mapped_spectrum
            axis_d.semilogy(
                frequencies[missing],
                np.maximum(np.abs(raw[missing]) / missing_scale, floor),
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.9,
                label=label,
            )
        axis_d.set_ylim(5e-18, 2.0)
        axis_d.set_ylabel("相对幅度（对数轴）")
        axis_d.grid(alpha=0.18)
        if row == 0:
            axis_d.legend(frameon=False, fontsize=6.2, ncol=2)

        for column, axis in enumerate((axis_a, axis_b, axis_c, axis_d)):
            if row == len(SIGNAL_KINDS) - 1:
                axis.set_xlabel(
                    "时间（sample）"
                    if column == 1
                    else "归一化频率（cycles/sample）"
                )
            _panel_caption(
                axis,
                f"{chr(65 + row)}{column + 1}",
                panel_titles[column],
                y=-0.31 if row == len(SIGNAL_KINDS) - 1 else -0.19,
            )
    fig.suptitle(
        (
            f"五类固定机制样本：N={config.n_fft}，M={CASE_SELECTED_BINS}，"
            f"SNR={CASE_SNR_DB:g} dB，重叠率={CASE_OVERLAP_FRACTION:g}"
        ),
        y=0.998,
        fontsize=16,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.982), h_pad=2.7)
    return fig


def _bootstrap_rmse_interval(
    squared_errors: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    """直接对 RMSE 做固定种子的非参数百分位 bootstrap。"""
    if squared_errors.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(RMSE_BOOTSTRAP_RESAMPLES, dtype=float)
    batch_size = 250
    for start in range(0, RMSE_BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, RMSE_BOOTSTRAP_RESAMPLES)
        indices = rng.integers(
            0,
            squared_errors.size,
            size=(stop - start, squared_errors.size),
        )
        estimates[start:stop] = np.sqrt(
            np.mean(squared_errors[indices], axis=1)
        )
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high)


def _rmse_bootstrap_seed(keys: tuple[object, ...]) -> int:
    signal_index = SIGNAL_KINDS.index(str(keys[0]))
    method_index = METHOD_ORDER.index(str(keys[4]))
    snr_code = int(round((float(keys[2]) + 20.0) * 10.0))
    overlap_code = int(round(float(keys[3]) * 100.0))
    return (
        RMSE_BOOTSTRAP_BASE_SEED
        + signal_index * 1_000_000
        + int(keys[1]) * 1_000
        + snr_code * 10
        + overlap_code
        + method_index
    )


def _mean_ci_summary(trials: pd.DataFrame) -> pd.DataFrame:
    grouped = trials.groupby(
        ["signal_kind", "selected_bins", "snr_db", "overlap_fraction", "method"],
        dropna=False,
    )
    rows: list[dict[str, object]] = []
    for keys, frame in grouped:
        values = frame["absolute_error_samples"].dropna().to_numpy()
        squared_values = frame["squared_error_samples"].dropna().to_numpy()
        count = values.size
        mean = float(np.mean(values)) if count else np.nan
        sem = float(np.std(values, ddof=1) / np.sqrt(count)) if count > 1 else np.nan
        mean_squared_error = (
            float(np.mean(squared_values)) if squared_values.size else np.nan
        )
        rmse = (
            float(np.sqrt(mean_squared_error))
            if np.isfinite(mean_squared_error)
            else np.nan
        )
        needs_rmse_interval = (
            np.isclose(float(keys[3]), 0.5)
            and str(keys[4]) in KEY_METHODS
            and squared_values.size > 1
        )
        if needs_rmse_interval:
            rmse_ci95_low, rmse_ci95_high = _bootstrap_rmse_interval(
                squared_values,
                _rmse_bootstrap_seed(keys),
            )
        else:
            rmse_ci95_low = np.nan
            rmse_ci95_high = np.nan
        rows.append(
            {
                "signal_kind": keys[0],
                "selected_bins": keys[1],
                "snr_db": keys[2],
                "overlap_fraction": keys[3],
                "method": keys[4],
                "valid_trials": count,
                "mae_samples": mean,
                "ci95_halfwidth_samples": 1.96 * sem if np.isfinite(sem) else np.nan,
                "rmse_samples": rmse,
                "rmse_ci95_low": rmse_ci95_low,
                "rmse_ci95_high": rmse_ci95_high,
                "rmse_ci_method": (
                    f"percentile_bootstrap_{RMSE_BOOTSTRAP_RESAMPLES}"
                    if needs_rmse_interval
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_performance_grid(
    condition_stats: pd.DataFrame,
    metric: str,
) -> plt.Figure:
    if metric not in {"mae", "rmse"}:
        raise ValueError(f"不支持的 TDOA 指标：{metric}")
    subset = condition_stats.loc[
        np.isclose(condition_stats["overlap_fraction"], 0.5)
        & condition_stats["method"].isin(KEY_METHODS)
    ]
    value_column = "mae_samples" if metric == "mae" else "rmse_samples"
    metric_label = "MAE" if metric == "mae" else "RMSE"
    selected_bins_values = (256, 128, 64, 32)
    fig, axes = plt.subplots(5, 4, figsize=(14.8, 17.5), sharex=True)
    for row, signal_kind in enumerate(SIGNAL_KINDS):
        for column, selected_bins in enumerate(selected_bins_values):
            axis = axes[row, column]
            panel = subset.loc[
                subset["signal_kind"].eq(signal_kind)
                & subset["selected_bins"].eq(selected_bins)
            ]
            for method in KEY_METHODS:
                method_rows = panel.loc[panel["method"].eq(method)].sort_values(
                    "snr_db"
                )
                color, linestyle, marker = METHOD_STYLES[method]
                if metric == "mae":
                    error = method_rows["ci95_halfwidth_samples"]
                else:
                    error = np.vstack(
                        [
                            method_rows[value_column]
                            - method_rows["rmse_ci95_low"],
                            method_rows["rmse_ci95_high"]
                            - method_rows[value_column],
                        ]
                    )
                axis.errorbar(
                    method_rows["snr_db"],
                    method_rows[value_column],
                    yerr=error,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3,
                    linewidth=1.0,
                    capsize=2,
                    label=METHOD_LABELS[method],
                )
            axis.grid(alpha=0.2)
            axis.set_yscale("log")
            if column == 0:
                axis.set_ylabel(
                    f"{SIGNAL_KIND_LABELS[signal_kind]}\nTDOA {metric_label}（sample）"
                )
            if row == len(SIGNAL_KINDS) - 1:
                axis.set_xlabel("SNR（dB）")
            _panel_caption(
                axis,
                f"{chr(65 + row)}{column + 1}",
                f"{SIGNAL_KIND_LABELS[signal_kind]}，M={selected_bins}",
                y=-0.31 if row == len(SIGNAL_KINDS) - 1 else -0.19,
            )
            if row == 0 and column == 0:
                axis.legend(
                    frameon=True,
                    facecolor="white",
                    framealpha=0.94,
                    fontsize=6.6,
                )
    fig.suptitle(
        (
            f"50% 频点重叠时的 TDOA {metric_label}"
            + (
                "（误差棒为均值的 95% t 区间）"
                if metric == "mae"
                else "（误差棒为 2,000 次非参数 bootstrap 95% 区间）"
            )
        ),
        y=0.998,
        fontsize=16,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.982), h_pad=2.7)
    return fig


def _plot_heatmaps(condition_stats: pd.DataFrame, metric: str) -> plt.Figure:
    if metric not in {"mae", "rmse"}:
        raise ValueError(f"不支持的 TDOA 指标：{metric}")
    subset = condition_stats.loc[
        np.isclose(condition_stats["overlap_fraction"], 0.5)
        & condition_stats["method"].eq("Intersection-Continuous")
    ]
    value_column = "mae_samples" if metric == "mae" else "rmse_samples"
    metric_label = "MAE" if metric == "mae" else "RMSE"
    selected_bins_values = (256, 128, 64, 32)
    snr_values = (-5.0, 0.0, 5.0, 10.0, 15.0)
    matrices = []
    for signal_kind in SIGNAL_KINDS:
        panel = subset.loc[subset["signal_kind"].eq(signal_kind)]
        matrix = (
            panel.pivot(index="selected_bins", columns="snr_db", values=value_column)
            .reindex(index=selected_bins_values, columns=snr_values)
            .to_numpy()
        )
        matrices.append(matrix)
    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    lower = max(float(np.min(finite_values)), 1e-3)
    upper = float(np.max(finite_values))
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 5, figsize=(17.2, 4.5), sharey=False)
    image = None
    for index, (axis, signal_kind, matrix) in enumerate(
        zip(axes, SIGNAL_KINDS, matrices, strict=True)
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="viridis_r",
            norm=LogNorm(vmin=lower, vmax=upper),
        )
        axis.set_xticks(np.arange(len(snr_values)), [f"{value:g}" for value in snr_values])
        axis.set_yticks(
            np.arange(len(selected_bins_values)),
            [str(value) for value in selected_bins_values],
        )
        axis.set_xlabel("SNR（dB）")
        axis.set_ylabel("单站频点数 M")
        _panel_caption(
            axis,
            chr(65 + index),
            SIGNAL_KIND_LABELS[signal_kind],
            y=-0.34,
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.1,
                    color="black" if luminance > 0.55 else "white",
                )
    fig.subplots_adjust(
        left=0.055,
        right=0.91,
        bottom=0.27,
        top=0.82,
        wspace=0.28,
    )
    if image is not None:
        colorbar_axis = fig.add_axes((0.93, 0.27, 0.012, 0.55))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(f"真实交集 TDOA {metric_label}（sample）")
    fig.suptitle(
        f"50% 重叠时真实交集的 TDOA {metric_label}",
        y=1.02,
        fontsize=16,
        fontweight="semibold",
    )
    return fig


def _plot_overlap_effect(condition_stats: pd.DataFrame) -> plt.Figure:
    subset = condition_stats.loc[
        np.isclose(condition_stats["snr_db"], 5.0)
        & condition_stats["method"].eq("Intersection-Continuous")
    ]
    selected_bins_values = (256, 128, 64, 32)
    colors = (
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
    )
    fig, axes = plt.subplots(1, 5, figsize=(17.2, 4.7), sharey=True)
    for index, (axis, signal_kind) in enumerate(zip(axes, SIGNAL_KINDS, strict=True)):
        panel = subset.loc[subset["signal_kind"].eq(signal_kind)]
        for selected_bins, color in zip(selected_bins_values, colors, strict=True):
            rows = panel.loc[panel["selected_bins"].eq(selected_bins)].sort_values(
                "overlap_fraction"
            )
            axis.errorbar(
                rows["overlap_fraction"],
                rows["mae_samples"],
                yerr=rows["ci95_halfwidth_samples"],
                marker="o",
                markersize=3,
                linewidth=1.0,
                capsize=2,
                color=color,
                label=f"M={selected_bins}",
            )
        axis.set_xlabel("双站频点重叠率")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.set_ylabel("真实交集 TDOA MAE（sample）")
        _panel_caption(
            axis,
            chr(65 + index),
            SIGNAL_KIND_LABELS[signal_kind],
            y=-0.34,
        )
        axis.legend(frameon=False, fontsize=6.4, ncol=2)
    fig.suptitle(
        "SNR=5 dB 时频点重叠率对 TDOA MAE 的影响",
        y=1.02,
        fontsize=16,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    return fig


def _plot_equivalence_and_gates(
    diagnostics: pd.DataFrame,
    reconstruction_audit: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    axis_a, axis_b, axis_c = axes

    audit_rows = (
        reconstruction_audit.loc[
            reconstruction_audit["method"].isin(PSEUDO_COMPLETION_METHODS)
        ]
        .groupby("method", sort=False)
        .agg(
            maximum_delay_difference_samples=(
                "delay_difference_samples",
                lambda values: float(np.max(np.abs(values))),
            ),
        )
        .reindex(PSEUDO_COMPLETION_METHODS)
        .reset_index()
    )
    positions = np.arange(len(audit_rows))
    axis_a.scatter(
        audit_rows["maximum_delay_difference_samples"],
        positions,
        marker="o",
        color=OKABE_ITO["orange"],
    )
    axis_a.axvline(0.0, color=OKABE_ITO["black"], linewidth=0.9)
    axis_a.set_yticks(
        positions,
        [METHOD_LABELS[method] for method in audit_rows["method"]],
    )
    axis_a.set_xlim(-1e-12, 1e-12)
    axis_a.set_xticks((-1e-12, 0.0, 1e-12), ("−1e−12", "0", "1e−12"))
    axis_a.set_xlabel("相对真实交集的最大 TDOA 差（sample）")
    axis_a.grid(axis="x", alpha=0.2)
    _panel_caption(
        axis_a,
        "A",
        "400 个可估计重构试验的 TDOA 差",
        y=-0.36,
    )

    for method, (color, linestyle, label) in COMPLETION_STYLES.items():
        rows = diagnostics.loc[diagnostics["method"].eq(method)].sort_values(
            "selected_bins"
        )
        axis_b.semilogy(
            rows["selected_bins"],
            np.maximum(rows["generated_missing_energy"], 1e-32),
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=3,
            linewidth=1.0,
            label=label,
        )
    axis_b.set_xlabel("单站频点数 M")
    axis_b.set_ylabel("生成的缺失频点能量")
    axis_b.grid(alpha=0.2)
    axis_b.legend(frameon=False, fontsize=6.2, ncol=2)
    _panel_caption(
        axis_b,
        "B",
        "原 DFT 网格的缺失频点能量",
        y=-0.36,
    )

    effective = reconstruction_audit.loc[
        reconstruction_audit["method"].isin(PSEUDO_COMPLETION_METHODS)
    ]
    styles = ("o", "s", "^", "D", "v")
    colors = (
        OKABE_ITO["black"],
        OKABE_ITO["orange"],
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
    )
    for method, marker, color in zip(
        PSEUDO_COMPLETION_METHODS,
        styles,
        colors,
        strict=True,
    ):
        rows = (
            effective.loc[effective["method"].eq(method)]
            .groupby("overlap_fraction", sort=True)["effective_common_rate"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )
        axis_c.errorbar(
            rows["overlap_fraction"],
            100.0 * rows["mean"],
            yerr=np.vstack(
                (
                    100.0 * (rows["mean"] - rows["min"]),
                    100.0 * (rows["max"] - rows["mean"]),
                )
            ),
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=0.9,
            capsize=2,
            label=METHOD_LABELS[method],
        )
    axis_c.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    axis_c.set_ylim(-3.0, 105.0)
    axis_c.set_xlabel("双站频点重叠率")
    axis_c.set_ylabel("有效共同频点率（%）")
    axis_c.grid(alpha=0.2)
    axis_c.legend(frameon=False, fontsize=5.8, loc="lower right")
    _panel_caption(
        axis_c,
        "C",
        "插值输出的有效共同频点率",
        y=-0.36,
    )
    fig.suptitle(
        "原网格补全等价性、缺失能量与共同频点检验",
        y=1.02,
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    return fig


def _information_comparator_summary(trials: pd.DataFrame) -> pd.DataFrame:
    methods = ("Zhai-Asymmetric", "Intersection-Continuous", "Oracle-Union")
    subset = trials.loc[
        trials["method"].isin(methods)
        & (trials["overlap_fraction"] > 0.0)
    ]
    return (
        subset.groupby(["method", "snr_db"], sort=False)
        .agg(
            valid_trials=("absolute_error_samples", "count"),
            mae_samples=("absolute_error_samples", "mean"),
            rmse_samples=(
                "squared_error_samples",
                lambda values: float(np.sqrt(np.nanmean(values))),
            ),
            median_ae_samples=("absolute_error_samples", "median"),
            outlier_gt_1_rate=("outlier_gt_1_sample", "mean"),
        )
        .reset_index()
    )


def _validate_reconstruction_replay(
    trials: pd.DataFrame,
    reconstruction_audit: pd.DataFrame,
) -> float:
    formal = trials.loc[
        trials["method"].eq("Intersection-Continuous")
        & trials["trial_index"].eq(0),
        ["trial_id", "true_delay_samples"],
    ]
    replayed = reconstruction_audit.loc[
        reconstruction_audit["method"].eq("Intersection-Continuous"),
        ["trial_id", "true_delay_samples"],
    ]
    merged = formal.merge(
        replayed,
        on="trial_id",
        suffixes=("_formal", "_replayed"),
        validate="one_to_one",
    )
    if len(merged) != 500:
        raise ValueError("独立重构审计未覆盖全部500个正式条件")
    maximum = float(
        np.max(
            np.abs(
                merged["true_delay_samples_formal"]
                - merged["true_delay_samples_replayed"]
            )
        )
    )
    if maximum > 1e-12:
        raise ValueError("独立重构审计未精确重放正式条件首个试验")
    return maximum


def _positive_overlap_pairs(trials: pd.DataFrame) -> pd.DataFrame:
    """配对正重叠 trial；正改善表示 TZP-DS 的绝对误差更低。"""

    common_columns = (
        "trial_id",
        "condition_index",
        "trial_index",
        "signal_kind",
        "selected_bins",
        "snr_db",
        "overlap_fraction",
        "true_delay_samples",
    )
    baseline = trials.loc[
        trials["method"].eq("Intersection-Continuous")
        & (trials["overlap_fraction"] > 0.0),
        [*common_columns, "estimated_delay_samples", "absolute_error_samples", "squared_error_samples"],
    ].rename(
        columns={
            "estimated_delay_samples": "intersection_estimate_samples",
            "absolute_error_samples": "intersection_ae_samples",
            "squared_error_samples": "intersection_se_samples",
        }
    )
    candidate = trials.loc[
        trials["method"].eq(TZP_DENSE_METHOD)
        & (trials["overlap_fraction"] > 0.0),
        ["trial_id", "estimated_delay_samples", "absolute_error_samples", "squared_error_samples"],
    ].rename(
        columns={
            "estimated_delay_samples": "tzp_estimate_samples",
            "absolute_error_samples": "tzp_ae_samples",
            "squared_error_samples": "tzp_se_samples",
        }
    )
    paired = baseline.merge(candidate, on="trial_id", validate="one_to_one")
    if len(paired) != 120_000:
        raise ValueError(f"正重叠配对 trial 数应为 120000，实际为 {len(paired)}")
    if not np.isfinite(
        paired[["intersection_estimate_samples", "tzp_estimate_samples"]].to_numpy()
    ).all():
        raise ValueError("正重叠配对中出现非有限 TDOA 估计")
    paired["mae_improvement_samples"] = (
        paired["intersection_ae_samples"] - paired["tzp_ae_samples"]
    )
    return paired


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    """返回均值及其非参数 bootstrap 95% 区间。"""

    samples = np.asarray(values, dtype=float)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("bootstrap 输入必须是非空有限数值")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        draw = rng.integers(0, samples.size, size=samples.size)
        estimates[index] = float(np.mean(samples[draw]))
    return (
        float(np.mean(samples)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _tzp_boundary_sensitivity(
    pairs: pd.DataFrame,
    config: SweepConfig,
) -> pd.DataFrame:
    """检查真实时延靠近固定搜索边界时，结论是否发生变化。"""

    work = pairs.copy()
    work["distance_to_search_boundary_samples"] = (
        config.lag_limit_samples - np.abs(work["true_delay_samples"])
    )
    work["boundary_stratum"] = np.where(
        work["distance_to_search_boundary_samples"] <= 1.0,
        "距边界≤1采样点",
        "距边界>1采样点",
    )
    rows: list[dict[str, float | int | str]] = []
    for group_index, (label, group) in enumerate(
        [("全部正重叠", work), *list(work.groupby("boundary_stratum", sort=False))]
    ):
        mean_delta, low, high = _bootstrap_mean_interval(
            group["mae_improvement_samples"].to_numpy(),
            repetitions=ROBUSTNESS_BOOTSTRAP_RESAMPLES,
            seed=ROBUSTNESS_BOOTSTRAP_SEED + group_index,
        )
        boundary_threshold = config.lag_limit_samples - 0.5 * config.delay_grid_step_samples
        rows.append(
            {
                "boundary_stratum": label,
                "paired_trials": int(len(group)),
                "intersection_mae_samples": float(group["intersection_ae_samples"].mean()),
                "intersection_rmse_samples": float(
                    np.sqrt(group["intersection_se_samples"].mean())
                ),
                "tzp_mae_samples": float(group["tzp_ae_samples"].mean()),
                "tzp_rmse_samples": float(np.sqrt(group["tzp_se_samples"].mean())),
                "mean_mae_improvement_samples": mean_delta,
                "ci95_low": low,
                "ci95_high": high,
                "intersection_boundary_hit_rate": float(
                    np.mean(
                        np.abs(group["intersection_estimate_samples"])
                        >= boundary_threshold
                    )
                ),
                "tzp_boundary_hit_rate": float(
                    np.mean(
                        np.abs(group["tzp_estimate_samples"])
                        >= boundary_threshold
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _tzp_condition_effects(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """以 400 个设计条件为等权单元，汇总 TZP-DS 的配对效应。"""

    condition_columns = (
        "condition_index",
        "signal_kind",
        "selected_bins",
        "snr_db",
        "overlap_fraction",
    )
    effects = (
        pairs.groupby(list(condition_columns), sort=True)
        .agg(
            paired_trials=("trial_id", "count"),
            intersection_mae_samples=("intersection_ae_samples", "mean"),
            tzp_mae_samples=("tzp_ae_samples", "mean"),
            intersection_rmse_samples=(
                "intersection_se_samples",
                lambda values: float(np.sqrt(np.mean(values))),
            ),
            tzp_rmse_samples=(
                "tzp_se_samples",
                lambda values: float(np.sqrt(np.mean(values))),
            ),
            mae_improvement_samples=("mae_improvement_samples", "mean"),
        )
        .reset_index()
    )
    if len(effects) != 400 or not effects["paired_trials"].eq(300).all():
        raise ValueError("条件级统计应覆盖 400 个正重叠条件且每条件 300 个 trial")
    tolerance = 1e-12
    delta = effects["mae_improvement_samples"].to_numpy()
    mean_delta, low, high = _bootstrap_mean_interval(
        delta,
        repetitions=ROBUSTNESS_BOOTSTRAP_RESAMPLES,
        seed=ROBUSTNESS_BOOTSTRAP_SEED + 20,
    )
    summary = pd.DataFrame(
        [
            {
                "conditions": int(len(effects)),
                "condition_weighting": "equal_weight_per_formal_condition",
                "mean_mae_improvement_samples": mean_delta,
                "median_mae_improvement_samples": float(np.median(delta)),
                "q25_mae_improvement_samples": float(np.quantile(delta, 0.25)),
                "q75_mae_improvement_samples": float(np.quantile(delta, 0.75)),
                "ci95_low": low,
                "ci95_high": high,
                "tzp_better_conditions": int(np.count_nonzero(delta > tolerance)),
                "equal_conditions": int(np.count_nonzero(np.abs(delta) <= tolerance)),
                "tzp_worse_conditions": int(np.count_nonzero(delta < -tolerance)),
            }
        ]
    )
    return effects, summary


def _estimator_replay_validation(config: SweepConfig) -> pd.DataFrame:
    """在预先固定的正式条件首试验上比较快速实现与定义式峰值。"""

    conditions = _build_conditions(config)
    by_key = {
        (condition.signal_kind, condition.selected_bins, condition.snr_db, condition.overlap_fraction): condition
        for condition in conditions
    }
    selected: list[object] = []
    positive_overlaps = tuple(value for value in config.overlap_values if value > 0.0)
    for signal_index, signal_kind in enumerate(config.signal_kinds):
        for budget_index, selected_bins in enumerate(config.selected_bins_values):
            for offset in range(ESTIMATOR_REPLAY_PER_SIGNAL_BUDGET):
                snr_index = (signal_index + 2 * budget_index + 2 * offset) % len(
                    config.snr_values_db
                )
                overlap_index = (2 * signal_index + budget_index + offset) % len(
                    positive_overlaps
                )
                key = (
                    signal_kind,
                    selected_bins,
                    config.snr_values_db[snr_index],
                    positive_overlaps[overlap_index],
                )
                selected.append(by_key[key])
    if len(selected) != len(config.signal_kinds) * len(config.selected_bins_values) * ESTIMATOR_REPLAY_PER_SIGNAL_BUDGET:
        raise ValueError("预注册估计器重放子集大小错误")

    frequencies = shifted_frequencies(config.n_fft)
    lag_ticks = np.arange(
        int(round(config.true_delay_min_samples / config.delay_grid_step_samples)),
        int(round(config.true_delay_max_samples / config.delay_grid_step_samples)) + 1,
        dtype=int,
    )
    lags = lag_ticks.astype(float) * config.delay_grid_step_samples
    rows: list[dict[str, float | int | str]] = []
    for condition in selected:
        rng = np.random.default_rng(condition.seed)
        source = _make_source_spectrum(config, condition.signal_kind, rng)
        true_delay = float(rng.uniform(config.true_delay_min_samples, config.true_delay_max_samples))
        gain = np.exp(1j * rng.uniform(-np.pi, np.pi))
        noisy_a = _add_frequency_noise(source, condition.snr_db, rng)
        noisy_b = _add_frequency_noise(
            gain * source * np.exp(-2j * np.pi * frequencies * true_delay),
            condition.snr_db,
            rng,
        )
        mask_a, mask_b = _controlled_masks(
            config.n_fft,
            condition.selected_bins,
            condition.overlap_fraction,
            rng,
        )
        intersection = np.intersect1d(mask_a, mask_b)
        intersection_cross = np.zeros(config.n_fft, dtype=complex)
        intersection_cross[intersection] = noisy_b[intersection] * np.conj(
            noisy_a[intersection]
        )
        fft_estimate, _, _ = _fft_delay_estimates(
            intersection_cross[None, :],
            np.asarray([intersection.size]),
            config,
        )
        direct_score = _direct_score_grid(intersection_cross, intersection, lags)
        direct_estimate = float(lags[int(np.argmax(direct_score))])

        sparse_a = np.zeros(config.n_fft, dtype=complex)
        sparse_b = np.zeros(config.n_fft, dtype=complex)
        sparse_a[mask_a] = noisy_a[mask_a]
        sparse_b[mask_b] = noisy_b[mask_b]
        dense_cross = time_zero_padded_dense_spectrum(
            sparse_b, config.interpolation_factor
        ) * np.conj(
            time_zero_padded_dense_spectrum(sparse_a, config.interpolation_factor)
        )
        zoom_estimate, _, _ = dense_delay_estimates(
            dense_cross[None, :],
            np.asarray([condition.selected_bins]),
            config.true_delay_min_samples,
            config.true_delay_max_samples,
            config.delay_grid_step_samples,
        )
        direct_dense_estimate = direct_dense_delay_estimate(
            dense_cross,
            config.true_delay_min_samples,
            config.true_delay_max_samples,
            config.delay_grid_step_samples,
        )
        common = {
            "condition_index": condition.index,
            "signal_kind": condition.signal_kind,
            "selected_bins": condition.selected_bins,
            "snr_db": condition.snr_db,
            "overlap_fraction": condition.overlap_fraction,
            "true_delay_samples": true_delay,
        }
        rows.extend(
            (
                {
                    **common,
                    "method": "Intersection-Continuous",
                    "fast_implementation": "100x_frequency_zero_padded_ifft",
                    "definition_implementation": "direct_frequency_sum",
                    "fast_estimate_samples": float(fft_estimate[0]),
                    "direct_estimate_samples": direct_estimate,
                    "absolute_difference_samples": abs(
                        float(fft_estimate[0]) - direct_estimate
                    ),
                },
                {
                    **common,
                    "method": TZP_DENSE_METHOD,
                    "fast_implementation": "zoomfft",
                    "definition_implementation": "direct_frequency_sum",
                    "fast_estimate_samples": float(zoom_estimate[0]),
                    "direct_estimate_samples": direct_dense_estimate,
                    "absolute_difference_samples": abs(
                        float(zoom_estimate[0]) - direct_dense_estimate
                    ),
                },
            )
        )
    replay = pd.DataFrame(rows)
    if len(replay) != 2 * len(selected):
        raise ValueError("估计器重放结果行数错误")
    return replay


def _plot_information_comparators(comparators: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    for axis, metric, label, panel in zip(
        axes,
        ("mae_samples", "rmse_samples"),
        ("MAE", "RMSE"),
        ("A", "B"),
        strict=True,
    ):
        for method in ("Zhai-Asymmetric", "Intersection-Continuous", "Oracle-Union"):
            rows = comparators.loc[comparators["method"].eq(method)].sort_values(
                "snr_db"
            )
            color, linestyle, marker = METHOD_STYLES[method]
            axis.plot(
                rows["snr_db"],
                rows[metric],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                linewidth=1.2,
                label=METHOD_LABELS[method],
            )
        axis.set_yscale("log")
        axis.set_xlabel("SNR（dB）")
        axis.set_ylabel(f"TDOA {label}（sample）")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
        _panel_caption(axis, panel, f"{label} 随 SNR 的变化", y=-0.31)
    fig.suptitle(
        "不同真实跨站信息结构下的 TDOA 性能",
        y=1.02,
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    return fig


def _direct_score_grid(
    shifted_cross_spectrum: np.ndarray,
    active_indices: np.ndarray,
    lags: np.ndarray,
) -> np.ndarray:
    active = np.asarray(active_indices, dtype=int)
    if active.size == 0:
        return np.zeros(lags.size, dtype=float)
    frequencies = shifted_frequencies(shifted_cross_spectrum.size)[active]
    steering = np.exp(2j * np.pi * np.outer(lags, frequencies))
    return np.abs(steering @ shifted_cross_spectrum[active])


def _normalise_curve(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values), initial=0.0))
    return np.asarray(values, dtype=float) / max(scale, np.finfo(float).tiny)


def _plot_tzp_dense_mechanism(
    cases: dict[str, MechanismCase],
    config: SweepConfig,
) -> plt.Figure:
    """展示五类信号的密集频谱曲线和对应 TDOA 得分。"""

    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.4), sharex="row")
    factor = config.interpolation_factor
    dense_length = config.n_fft * factor
    dense_frequencies = np.fft.fftshift(np.fft.fftfreq(dense_length))
    lag_ticks = np.arange(
        int(round(config.true_delay_min_samples / config.delay_grid_step_samples)),
        int(round(config.true_delay_max_samples / config.delay_grid_step_samples)) + 1,
    )
    lags = lag_ticks.astype(float) * config.delay_grid_step_samples
    for column, signal_kind in enumerate(SIGNAL_KINDS):
        case = cases[signal_kind]
        dense_a = time_zero_padded_dense_spectrum(case.sparse_a, factor)
        dense_b = time_zero_padded_dense_spectrum(case.sparse_b, factor)
        dense_oracle_a = time_zero_padded_dense_spectrum(case.noisy_a, factor)
        mapped = sample_original_grid(dense_a, config.n_fft, factor)
        spectrum_scale = max(
            float(np.max(np.abs(dense_oracle_a), initial=0.0)),
            np.finfo(float).tiny,
        )

        spectrum_axis = axes[0, column]
        spectrum_axis.plot(
            dense_frequencies,
            np.abs(dense_oracle_a) / spectrum_scale,
            color=OKABE_ITO["black"],
            linewidth=0.8,
            alpha=0.78,
            label="完整频谱密集参考",
        )
        spectrum_axis.plot(
            dense_frequencies,
            np.abs(dense_a) / spectrum_scale,
            color=OKABE_ITO["purple"],
            linewidth=0.9,
            label="TZP-DS密集曲线",
        )
        spectrum_axis.scatter(
            case.frequencies[case.mask_a],
            np.abs(mapped[case.mask_a]) / spectrum_scale,
            s=7,
            color=OKABE_ITO["orange"],
            zorder=3,
            label="A站真实保留频点",
        )
        spectrum_axis.set_ylim(-0.04, 1.08)
        spectrum_axis.set_xlabel("归一化频率")
        if column == 0:
            spectrum_axis.set_ylabel("归一化幅度")
        spectrum_axis.grid(alpha=0.18)
        spectrum_axis.legend(frameon=False, fontsize=5.4, loc="upper right")
        _panel_caption(
            spectrum_axis,
            f"A{column + 1}",
            f"{SIGNAL_KIND_LABELS[signal_kind]}密集频谱",
            y=-0.29,
        )

        intersection = np.intersect1d(case.mask_a, case.mask_b)
        full_cross = case.noisy_b * np.conj(case.noisy_a)
        intersection_score = _direct_score_grid(full_cross, intersection, lags)
        full_score = _direct_score_grid(
            full_cross,
            np.arange(config.n_fft),
            lags,
        )
        _, dense_scores = dense_delay_score_grid(
            dense_b * np.conj(dense_a),
            config.true_delay_min_samples,
            config.true_delay_max_samples,
            config.delay_grid_step_samples,
        )
        score_axis = axes[1, column]
        score_axis.plot(
            lags,
            _normalise_curve(full_score),
            color=OKABE_ITO["black"],
            linewidth=0.9,
            label="完整频谱参考",
        )
        score_axis.plot(
            lags,
            _normalise_curve(intersection_score),
            color=OKABE_ITO["orange"],
            linestyle="-.",
            linewidth=0.9,
            label="真实交集",
        )
        score_axis.plot(
            lags,
            _normalise_curve(dense_scores[0]),
            color=OKABE_ITO["purple"],
            linestyle=(0, (5, 2)),
            linewidth=0.9,
            label="TZP-DS",
        )
        score_axis.axvline(
            case.true_delay_samples,
            color=OKABE_ITO["blue"],
            linestyle=":",
            linewidth=0.9,
            label="真实TDOA",
        )
        score_axis.set_xlabel("候选TDOA（sample）")
        if column == 0:
            score_axis.set_ylabel("归一化得分")
        score_axis.set_ylim(-0.04, 1.08)
        score_axis.grid(alpha=0.18)
        score_axis.legend(frameon=False, fontsize=5.4, loc="upper right")
        _panel_caption(
            score_axis,
            f"B{column + 1}",
            f"{SIGNAL_KIND_LABELS[signal_kind]}相关峰",
            y=-0.31,
        )
    fig.suptitle(
        "8倍时域补零后的密集频谱与TDOA得分",
        y=1.01,
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.97), h_pad=3.5, w_pad=1.3)
    return fig


def _tzp_performance_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """按 SNR 汇总正重叠条件下的时域补零性能。"""

    return _tzp_performance_by_group(trials, "snr_db")


def _tzp_performance_by_group(
    trials: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """在同一正重叠试验集上，按指定条件分层比较时域补零性能。"""

    methods = ("Intersection-Continuous", TZP_DENSE_METHOD, "Oracle-Union")
    return (
        trials.loc[
            trials["method"].isin(methods)
            & (trials["overlap_fraction"] > 0.0)
        ]
        .groupby(["method", group_column], sort=False)
        .agg(
            valid_trials=("absolute_error_samples", "count"),
            mae_samples=("absolute_error_samples", "mean"),
            rmse_samples=(
                "squared_error_samples",
                lambda values: float(np.sqrt(np.nanmean(values))),
            ),
            outlier_gt_1_rate=("outlier_gt_1_sample", "mean"),
        )
        .reset_index()
    )


def _plot_tzp_dense_performance(trials: pd.DataFrame) -> plt.Figure:
    """汇总 TZP-DS 的误差、可估计率与数值非零率。"""

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4))
    summary = _tzp_performance_summary(trials)
    methods = ("Intersection-Continuous", TZP_DENSE_METHOD, "Oracle-Union")
    for axis, metric, label, panel in zip(
        axes[0],
        ("mae_samples", "rmse_samples"),
        ("MAE", "RMSE"),
        ("A", "B"),
        strict=True,
    ):
        for method in methods:
            rows = summary.loc[summary["method"].eq(method)].sort_values("snr_db")
            color, linestyle, marker = METHOD_STYLES[method]
            axis.plot(
                rows["snr_db"],
                rows[metric],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.1,
                markersize=4,
                label=METHOD_LABELS[method],
            )
        axis.set_yscale("log")
        axis.set_xlabel("SNR（dB）")
        axis.set_ylabel(f"TDOA {label}（sample）")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
        _panel_caption(axis, panel, f"{label}随SNR变化", y=-0.30)

    valid_axis = axes[1, 0]
    for method in ("Intersection-Continuous", TZP_DENSE_METHOD):
        rows = (
            trials.loc[trials["method"].eq(method)]
            .assign(valid=lambda frame: np.isfinite(frame["estimated_delay_samples"]))
            .groupby("overlap_fraction", sort=True)["valid"]
            .mean()
            .reset_index()
        )
        color, linestyle, marker = METHOD_STYLES[method]
        valid_axis.plot(
            rows["overlap_fraction"],
            100.0 * rows["valid"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.1,
            markersize=4,
            label=METHOD_LABELS[method],
        )
    valid_axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    valid_axis.set_ylim(-3.0, 103.0)
    valid_axis.set_xlabel("真实频点重叠率")
    valid_axis.set_ylabel("有限TDOA输出率（%）")
    valid_axis.grid(alpha=0.2)
    valid_axis.legend(frameon=False, fontsize=7)
    _panel_caption(valid_axis, "C", "有限估计率随重叠率变化", y=-0.30)

    density_axis = axes[1, 1]
    tzp = trials.loc[trials["method"].eq(TZP_DENSE_METHOD)]
    colors = (
        OKABE_ITO["black"],
        OKABE_ITO["orange"],
        OKABE_ITO["blue"],
        OKABE_ITO["purple"],
    )
    for selected_bins, color in zip((256, 128, 64, 32), colors, strict=True):
        rows = (
            tzp.loc[tzp["selected_bins"].eq(selected_bins)]
            .groupby("overlap_fraction", sort=True)["dense_grid_common_nonzero_rate"]
            .mean()
            .reset_index()
        )
        density_axis.plot(
            rows["overlap_fraction"],
            100.0 * rows["dense_grid_common_nonzero_rate"],
            color=color,
            marker="o",
            linewidth=1.0,
            markersize=3.5,
            label=f"TZP-DS，M={selected_bins}",
        )
    density_axis.plot(
        (0.0, 1.0),
        (0.0, 100.0),
        color=OKABE_ITO["green"],
        linestyle=":",
        linewidth=1.1,
        label="真实共同观测率",
    )
    density_axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    density_axis.set_ylim(-3.0, 103.0)
    density_axis.set_xlabel("真实频点重叠率")
    density_axis.set_ylabel("共同数值非零率（%）")
    density_axis.grid(alpha=0.2)
    density_axis.legend(frameon=False, fontsize=6.5)
    _panel_caption(density_axis, "D", "数值非零不等于真实观测", y=-0.30)
    fig.suptitle(
        "时域补零密集频谱的TDOA性能与信息边界",
        y=1.01,
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.97), h_pad=3.7, w_pad=2.1)
    return fig


def _plot_tzp_robustness_audit(
    boundary: pd.DataFrame,
    condition_effects: pd.DataFrame,
    estimator_replay: pd.DataFrame,
) -> plt.Figure:
    """展示边界、条件异质性与数值实现的补充审计。"""

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    edge_rows = boundary.loc[
        boundary["boundary_stratum"].ne("全部正重叠")
    ].copy()

    axis_a = axes[0, 0]
    positions = np.arange(len(edge_rows))
    width = 0.34
    axis_a.bar(
        positions - width / 2,
        edge_rows["intersection_mae_samples"],
        width,
        color=METHOD_STYLES["Intersection-Continuous"][0],
        label="真实交集",
    )
    axis_a.bar(
        positions + width / 2,
        edge_rows["tzp_mae_samples"],
        width,
        color=METHOD_STYLES[TZP_DENSE_METHOD][0],
        label="时域补零密集频谱",
    )
    axis_a.set_xticks(positions, edge_rows["boundary_stratum"])
    axis_a.set_ylabel("TDOA MAE（sample）")
    axis_a.grid(axis="y", alpha=0.2)
    axis_a.legend(frameon=False, fontsize=7)
    _panel_caption(axis_a, "A", "搜索边界敏感性", y=-0.27)

    axis_b = axes[0, 1]
    delta = condition_effects["mae_improvement_samples"].to_numpy()
    axis_b.hist(
        delta,
        bins=28,
        color=OKABE_ITO["purple"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    axis_b.axvline(0.0, color=OKABE_ITO["black"], linewidth=1.0)
    axis_b.axvline(
        float(np.mean(delta)),
        color=OKABE_ITO["orange"],
        linestyle="--",
        linewidth=1.1,
        label="条件等权均值",
    )
    axis_b.set_xlabel("交集 MAE − TZP-DS MAE（sample）")
    axis_b.set_ylabel("条件数")
    axis_b.grid(axis="y", alpha=0.2)
    axis_b.legend(frameon=False, fontsize=7)
    _panel_caption(axis_b, "B", "400 个条件的 MAE 改善分布", y=-0.27)

    axis_c = axes[1, 0]
    overlap_values = sorted(condition_effects["overlap_fraction"].unique())
    groups = [
        condition_effects.loc[
            np.isclose(condition_effects["overlap_fraction"], overlap),
            "mae_improvement_samples",
        ].to_numpy()
        for overlap in overlap_values
    ]
    boxplot = axis_c.boxplot(
        groups,
        tick_labels=[f"{overlap:g}" for overlap in overlap_values],
        patch_artist=True,
        showfliers=False,
    )
    for box in boxplot["boxes"]:
        box.set(facecolor=OKABE_ITO["purple"], alpha=0.65)
    axis_c.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.9)
    axis_c.set_xlabel("真实频点重叠率")
    axis_c.set_ylabel("条件 MAE 改善（sample）")
    axis_c.grid(axis="y", alpha=0.2)
    _panel_caption(axis_c, "C", "不同重叠率下的条件级效应", y=-0.27)

    axis_d = axes[1, 1]
    for method in ("Intersection-Continuous", TZP_DENSE_METHOD):
        rows = estimator_replay.loc[estimator_replay["method"].eq(method)]
        color, _, marker = METHOD_STYLES[method]
        axis_d.semilogy(
            np.arange(1, len(rows) + 1),
            np.maximum(rows["absolute_difference_samples"], 1e-16),
            linestyle="",
            marker=marker,
            markersize=3.5,
            color=color,
            label=METHOD_LABELS[method],
        )
    axis_d.set_xlabel("预先固定的正式条件重放编号")
    axis_d.set_ylabel("快速实现与定义式差（sample）")
    axis_d.grid(alpha=0.2)
    axis_d.legend(frameon=False, fontsize=7)
    _panel_caption(axis_d, "D", "正式条件重放的估计器核对", y=-0.27)

    fig.suptitle(
        "时域补零密集频谱的稳健性与实现审计",
        y=1.01,
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96), h_pad=2.7, w_pad=2.2)
    return fig


def _export_report_tables(
    trials: pd.DataFrame,
    condition_stats: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    reconstruction_audit: pd.DataFrame,
    information_comparators: pd.DataFrame,
    tzp_boundary: pd.DataFrame,
    tzp_condition_effects: pd.DataFrame,
    tzp_condition_summary: pd.DataFrame,
    estimator_replay: pd.DataFrame,
    output_directory: Path,
) -> dict[str, Path]:
    table_directory = output_directory / "report_tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    overall = (
        trials.groupby("method", sort=False)
        .agg(
            trial_rows=("trial_id", "count"),
            valid_trials=("absolute_error_samples", "count"),
            tdoa_mae_samples=("absolute_error_samples", "mean"),
            tdoa_rmse_samples=(
                "squared_error_samples",
                lambda values: float(np.sqrt(np.nanmean(values))),
            ),
            tdoa_median_ae_samples=("absolute_error_samples", "median"),
            outlier_gt_1_rate=("outlier_gt_1_sample", "mean"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    signal_summary = (
        trials.loc[trials["method"].isin(KEY_METHODS)]
        .groupby(["signal_kind", "method"], sort=False)
        .agg(
            valid_trials=("absolute_error_samples", "count"),
            tdoa_mae_samples=("absolute_error_samples", "mean"),
            tdoa_rmse_samples=(
                "squared_error_samples",
                lambda values: float(np.sqrt(np.nanmean(values))),
            ),
            tdoa_median_ae_samples=("absolute_error_samples", "median"),
            outlier_gt_1_rate=("outlier_gt_1_sample", "mean"),
        )
        .reset_index()
    )
    bpsk_overlap50 = condition_stats.loc[
        condition_stats["signal_kind"].eq("urban8_bpsk_rrc")
        & np.isclose(condition_stats["overlap_fraction"], 0.5)
        & condition_stats["method"].isin(KEY_METHODS)
    ].copy()
    zero_overlap = (
        condition_stats.loc[
            np.isclose(condition_stats["overlap_fraction"], 0.0)
            & condition_stats["method"].isin(KEY_METHODS)
        ]
        .groupby("method", sort=False)
        .agg(
            estimable_trial_rate=(
                "valid_trials",
                lambda values: float(np.mean(values > 0)),
            ),
            conditions=("valid_trials", "count"),
        )
        .reset_index()
    )
    audit_summary = (
        reconstruction_audit.loc[
            reconstruction_audit["method"].isin(PSEUDO_COMPLETION_METHODS)
        ]
        .groupby("method", sort=False)
        .agg(
            independent_reconstruction_trials=("condition_index", "count"),
            independent_valid_reconstruction_trials=(
                "delay_difference_samples",
                "count",
            ),
            maximum_effective_rate_deviation=(
                "effective_common_rate",
                lambda values: float(
                    np.max(
                        np.abs(
                            values.to_numpy()
                            - reconstruction_audit.loc[
                                values.index,
                                "overlap_fraction",
                            ].to_numpy()
                        )
                    )
                ),
            ),
            maximum_delay_difference_samples=(
                "delay_difference_samples",
                lambda values: float(np.max(np.abs(values))),
            ),
            maximum_absolute_error_difference_samples=(
                "absolute_error_difference_samples",
                lambda values: float(np.max(np.abs(values))),
            ),
            nonzero_delay_difference_trials=(
                "delay_difference_samples",
                lambda values: int(
                    np.count_nonzero(
                        values.to_numpy()[np.isfinite(values.to_numpy())]
                    )
                ),
            ),
        )
        .reindex(PSEUDO_COMPLETION_METHODS)
        .reset_index()
    )
    operator_summary = (
        diagnostics.groupby("method", sort=False)
        .agg(
            operator_cases=("selected_bins", "count"),
            operator_cases_passed=("passed", "sum"),
            maximum_spectrum_difference_vs_zero=(
                "maximum_spectrum_difference_vs_zero",
                "max",
            ),
            maximum_generated_missing_energy=("generated_missing_energy", "max"),
        )
        .reset_index()
    )
    paired_summary = paired.loc[
        paired["method"].isin(PSEUDO_COMPLETION_METHODS),
        [
            "method",
            "paired_trials",
            "mean_mae_improvement_samples",
            "ci95_low",
            "ci95_high",
            "maximum_abs_improvement_samples",
        ],
    ]
    equivalence_decision = (
        operator_summary.merge(audit_summary, on="method", validate="one_to_one")
        .merge(paired_summary, on="method", validate="one_to_one")
    )
    equivalence_decision["decision"] = np.where(
        (equivalence_decision["operator_cases_passed"] == 4)
        & (equivalence_decision["maximum_effective_rate_deviation"] == 0.0)
        & (equivalence_decision["maximum_delay_difference_samples"] == 0.0)
        & (equivalence_decision["maximum_absolute_error_difference_samples"] == 0.0),
        "equivalent_under_current_protocol",
        "not_equivalent",
    )
    tzp_rows = trials.loc[trials["method"].eq(TZP_DENSE_METHOD)].copy()
    tzp_performance = _tzp_performance_summary(trials)
    tzp_by_budget = _tzp_performance_by_group(trials, "selected_bins")
    tzp_by_signal = _tzp_performance_by_group(trials, "signal_kind")
    tzp_by_overlap = (
        tzp_rows.groupby("overlap_fraction", sort=True)
        .agg(
            trial_rows=("trial_id", "count"),
            finite_estimate_rate=(
                "estimated_delay_samples",
                lambda values: float(np.mean(np.isfinite(values))),
            ),
            tdoa_mae_samples=("absolute_error_samples", "mean"),
            tdoa_rmse_samples=(
                "squared_error_samples",
                lambda values: float(np.sqrt(np.nanmean(values))),
            ),
            outlier_gt_1_rate=("outlier_gt_1_sample", "mean"),
            dense_grid_common_nonzero_rate=(
                "dense_grid_common_nonzero_rate",
                "mean",
            ),
            dense_grid_nmse_clean_db=("dense_grid_nmse_clean_db", "mean"),
            dense_grid_nmse_noisy_db=("dense_grid_nmse_noisy_db", "mean"),
            dense_grid_complex_correlation=(
                "dense_grid_complex_correlation",
                "mean",
            ),
            original_grid_missing_energy=("generated_missing_energy", "max"),
            original_grid_consistency_error=("observed_consistency_error", "max"),
        )
        .reset_index()
    )
    tzp_dense_quality = (
        tzp_rows.groupby(["signal_kind", "selected_bins"], sort=False)
        .agg(
            trial_rows=("trial_id", "count"),
            dense_grid_nmse_clean_db=("dense_grid_nmse_clean_db", "mean"),
            dense_grid_nmse_noisy_db=("dense_grid_nmse_noisy_db", "mean"),
            dense_grid_phase_rmse_rad=("dense_grid_phase_rmse_rad", "mean"),
            dense_grid_complex_correlation=(
                "dense_grid_complex_correlation",
                "mean",
            ),
            dense_grid_common_nonzero_rate=(
                "dense_grid_common_nonzero_rate",
                "mean",
            ),
            original_grid_missing_energy=("generated_missing_energy", "max"),
            original_grid_consistency_error=("observed_consistency_error", "max"),
        )
        .reset_index()
    )
    tzp_paired = paired.loc[paired["method"].eq(TZP_DENSE_METHOD)].copy()
    tzp_operator = diagnostics.loc[
        diagnostics["method"].eq(TZP_DENSE_METHOD)
    ].copy()
    paths = {
        "overall": table_directory / "table_overall_method.csv",
        "signal": table_directory / "table_signal_method.csv",
        "bpsk": table_directory / "table_bpsk_overlap50.csv",
        "paired": table_directory / "table_pseudo_equivalence.csv",
        "operator": table_directory / "table_operator_gate.csv",
        "zero_overlap": table_directory / "table_zero_overlap_validity.csv",
        "condition_stats": table_directory / "condition_mean_ci.csv",
        "effective_common_bins": table_directory / "table_effective_common_bins.csv",
        "equivalence_decision": table_directory / "table_equivalence_decision.csv",
        "information_comparators": (
            table_directory / "table_information_comparators_by_snr.csv"
        ),
        "tzp_performance": table_directory / "table_tzp_performance_by_snr.csv",
        "tzp_by_budget": table_directory / "table_tzp_performance_by_budget.csv",
        "tzp_by_signal": table_directory / "table_tzp_performance_by_signal.csv",
        "tzp_overlap": table_directory / "table_tzp_metrics_by_overlap.csv",
        "tzp_dense_quality": table_directory / "table_tzp_dense_quality.csv",
        "tzp_paired": table_directory / "table_tzp_paired_comparison.csv",
        "tzp_operator": table_directory / "table_tzp_operator_gate.csv",
        "tzp_boundary": table_directory / "table_tzp_boundary_sensitivity.csv",
        "tzp_condition_effects": table_directory / "table_tzp_condition_effects.csv",
        "tzp_condition_summary": (
            table_directory / "table_tzp_condition_effect_summary.csv"
        ),
        "estimator_replay": table_directory / "table_estimator_replay_validation.csv",
    }
    overall.to_csv(paths["overall"], index=False, encoding="utf-8-sig")
    signal_summary.to_csv(paths["signal"], index=False, encoding="utf-8-sig")
    bpsk_overlap50.to_csv(paths["bpsk"], index=False, encoding="utf-8-sig")
    paired.to_csv(paths["paired"], index=False, encoding="utf-8-sig")
    diagnostics.to_csv(paths["operator"], index=False, encoding="utf-8-sig")
    zero_overlap.to_csv(paths["zero_overlap"], index=False, encoding="utf-8-sig")
    condition_stats.to_csv(
        paths["condition_stats"],
        index=False,
        encoding="utf-8-sig",
    )
    reconstruction_audit.to_csv(
        paths["effective_common_bins"],
        index=False,
        encoding="utf-8-sig",
    )
    equivalence_decision.to_csv(
        paths["equivalence_decision"],
        index=False,
        encoding="utf-8-sig",
    )
    information_comparators.to_csv(
        paths["information_comparators"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_performance.to_csv(
        paths["tzp_performance"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_by_budget.to_csv(
        paths["tzp_by_budget"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_by_signal.to_csv(
        paths["tzp_by_signal"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_by_overlap.to_csv(
        paths["tzp_overlap"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_dense_quality.to_csv(
        paths["tzp_dense_quality"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_paired.to_csv(
        paths["tzp_paired"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_operator.to_csv(
        paths["tzp_operator"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_boundary.to_csv(
        paths["tzp_boundary"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_condition_effects.to_csv(
        paths["tzp_condition_effects"],
        index=False,
        encoding="utf-8-sig",
    )
    tzp_condition_summary.to_csv(
        paths["tzp_condition_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    estimator_replay.to_csv(
        paths["estimator_replay"],
        index=False,
        encoding="utf-8-sig",
    )
    return paths


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=DEFAULT_RESULT_DIRECTORY,
    )
    parser.add_argument("--no-asset-copy", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    result_directory = arguments.result_directory.resolve()
    config_payload, summary, paired, diagnostics, completed = _load_results(
        result_directory
    )
    validation = _validate_results(
        config_payload,
        summary,
        paired,
        diagnostics,
        completed,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if arguments.validate_only:
        return
    trials = _read_trial_chunks(result_directory)
    if len(trials) != int(completed["method_rows"]):
        raise ValueError("trial chunks 合并行数与 COMPLETED.json 不一致")
    condition_stats = _mean_ci_summary(trials)
    config = SweepConfig(**config_payload)
    reconstruction_audit = _audit_reconstructed_spectra(config)
    replay_delay_difference = _validate_reconstruction_replay(
        trials,
        reconstruction_audit,
    )
    information_comparators = _information_comparator_summary(trials)
    positive_pairs = _positive_overlap_pairs(trials)
    tzp_boundary = _tzp_boundary_sensitivity(positive_pairs, config)
    tzp_condition_effects, tzp_condition_summary = _tzp_condition_effects(
        positive_pairs
    )
    estimator_replay = _estimator_replay_validation(config)
    report_tables = _export_report_tables(
        trials,
        condition_stats,
        paired,
        diagnostics,
        reconstruction_audit,
        information_comparators,
        tzp_boundary,
        tzp_condition_effects,
        tzp_condition_summary,
        estimator_replay,
        result_directory,
    )
    cases = {
        signal_kind: _make_mechanism_case(
            config,
            signal_kind,
            CASE_SEEDS[signal_kind],
        )
        for signal_kind in SIGNAL_KINDS
    }
    _configure_figure_style()
    figure_directory = result_directory / "figures" / "advisor_report_bpsk"
    figure_directory.mkdir(parents=True, exist_ok=True)
    asset_directory = None
    if not arguments.no_asset_copy:
        REPORT_ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
        asset_directory = REPORT_ASSET_DIRECTORY
    figures = (
        (
            "figure_01_five_signal_mechanism",
            _plot_mechanism(cases, config),
        ),
        (
            "figure_02_equivalence_and_gates",
            _plot_equivalence_and_gates(
                diagnostics,
                reconstruction_audit,
            ),
        ),
        (
            "figure_03_tdoa_mae_performance_grid",
            _plot_performance_grid(condition_stats, "mae"),
        ),
        (
            "figure_04_tdoa_rmse_performance_grid",
            _plot_performance_grid(condition_stats, "rmse"),
        ),
        (
            "figure_05_intersection_mae_heatmaps",
            _plot_heatmaps(condition_stats, "mae"),
        ),
        (
            "figure_06_intersection_rmse_heatmaps",
            _plot_heatmaps(condition_stats, "rmse"),
        ),
        (
            "figure_07_overlap_effect",
            _plot_overlap_effect(condition_stats),
        ),
        (
            "figure_08_information_comparators",
            _plot_information_comparators(information_comparators),
        ),
        (
            "figure_09_tzp_dense_mechanism",
            _plot_tzp_dense_mechanism(cases, config),
        ),
        (
            "figure_10_tzp_dense_performance",
            _plot_tzp_dense_performance(trials),
        ),
        (
            "figure_11_tzp_robustness_audit",
            _plot_tzp_robustness_audit(
                tzp_boundary,
                tzp_condition_effects,
                estimator_replay,
            ),
        ),
    )
    saved: list[Path] = []
    for stem, figure in figures:
        saved.extend(
            _save_figure(
                figure,
                stem,
                figure_directory,
                asset_directory,
            )
        )
    manifest = {
        "result_directory": str(result_directory),
        "validation": validation,
        "figures": [str(path) for path in saved],
        "asset_directory": str(asset_directory) if asset_directory else None,
        "report_tables": {key: str(path) for key, path in report_tables.items()},
        "case_seeds": CASE_SEEDS,
        "case_snr_db": CASE_SNR_DB,
        "case_selected_bins": CASE_SELECTED_BINS,
        "case_overlap_fraction": CASE_OVERLAP_FRACTION,
        "rmse_bootstrap_resamples": RMSE_BOOTSTRAP_RESAMPLES,
        "rmse_bootstrap_base_seed": RMSE_BOOTSTRAP_BASE_SEED,
        "effective_bin_relative_threshold": EFFECTIVE_BIN_RELATIVE_THRESHOLD,
        "tzp_dense_method": TZP_DENSE_METHOD,
        "tzp_time_zero_padding_factor": config.interpolation_factor,
        "tzp_dense_points": config.n_fft * config.interpolation_factor,
        "independent_reconstruction_trials": int(
            reconstruction_audit["condition_index"].nunique()
        ),
        "replay_maximum_true_delay_difference_samples": replay_delay_difference,
        "robustness_bootstrap_resamples": ROBUSTNESS_BOOTSTRAP_RESAMPLES,
        "condition_level_effect_conditions": int(len(tzp_condition_effects)),
        "estimator_replay_conditions": int(
            estimator_replay["condition_index"].nunique()
        ),
    }
    manifest_path = result_directory / "advisor_report_bpsk_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if asset_directory is not None:
        shutil.copy2(
            manifest_path,
            asset_directory / manifest_path.name,
        )
    print(f"正式汇报图：{figure_directory}")
    print(f"Obsidian 图片：{asset_directory}")


if __name__ == "__main__":
    main()
