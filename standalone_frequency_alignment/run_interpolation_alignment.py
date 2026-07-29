"""双站独立选频、频谱伪补全和 TDOA 对齐的独立运行入口。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = MODULE_DIRECTORY.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

try:  # 支持 ``python -m standalone_frequency_alignment...`` 方式导入。
    from .alignment_core import (
        complete_spectrum,
        complex_correlation,
        complex_nmse_db,
        estimate_continuous_delay,
        interpolation_operator_rank,
        phase_rmse_rad,
        shifted_frequencies,
        dense_delay_estimates,
        direct_dense_delay_estimate,
        sample_original_grid,
        time_zero_padded_dense_spectrum,
    )
    from .experiment_config import ExperimentConfig, get_profile
    from .plot_alignment_results import save_summary_figures
    from .zhai_selection import select_zhai_thesis
except ImportError:  # 仍兼容 PyCharm 中直接运行本文件。
    from alignment_core import (  # noqa: E402
        complete_spectrum,
        complex_correlation,
        complex_nmse_db,
        estimate_continuous_delay,
        interpolation_operator_rank,
        phase_rmse_rad,
        shifted_frequencies,
        dense_delay_estimates,
        direct_dense_delay_estimate,
        sample_original_grid,
        time_zero_padded_dense_spectrum,
    )
    from experiment_config import ExperimentConfig, get_profile  # noqa: E402
    from plot_alignment_results import save_summary_figures  # noqa: E402
    from zhai_selection import select_zhai_thesis  # noqa: E402

# 在 PyCharm 中只需修改这一行。正式大样本请使用 synthetic_full。
PROFILE = "synthetic_full"
# 可选："bpsk_parameter_sweep" 或 "bpsk_parameter_sweep_smoke"。
BPSK_SWEEP_WORKERS = max(1, min(2, (os.cpu_count() or 2) // 2))
BPSK_SWEEP_RESUME_DIRECTORY: Path | None = None

COMPLETION_METHODS = (
    "zero_fill",
    "ideal_periodic",
    "polyphase_fir",
    "linear_periodic",
    "cubic_periodic",
)


def _normalise_spectrum(spectrum: np.ndarray) -> np.ndarray:
    return spectrum / max(np.sqrt(np.mean(np.abs(spectrum) ** 2)), np.finfo(float).tiny)


def _make_source_spectrum(
    n_fft: int,
    signal_kind: str,
    rng: np.random.Generator,
) -> np.ndarray:
    frequencies = shifted_frequencies(n_fft)
    if signal_kind == "smooth_complex":
        magnitude = 0.2 + np.exp(-0.5 * (frequencies / 0.22) ** 2)
        phase = 0.7 * np.sin(2 * np.pi * frequencies) + 0.3 * np.cos(4 * np.pi * frequencies)
        spectrum = magnitude * np.exp(1j * phase)
    elif signal_kind == "smooth_magnitude_random_phase":
        magnitude = 0.2 + np.exp(-0.5 * (frequencies / 0.22) ** 2)
        spectrum = magnitude * np.exp(1j * rng.uniform(-np.pi, np.pi, n_fft))
    elif signal_kind == "broadband_random":
        spectrum = rng.normal(size=n_fft) + 1j * rng.normal(size=n_fft)
    elif signal_kind == "multiband":
        magnitude = (
            0.05
            + np.exp(-0.5 * ((frequencies - 0.24) / 0.055) ** 2)
            + 0.8 * np.exp(-0.5 * ((frequencies + 0.19) / 0.07) ** 2)
        )
        spectrum = magnitude * np.exp(1j * rng.uniform(-np.pi, np.pi, n_fft))
    else:
        raise ValueError(f"未知信号类型 {signal_kind!r}")
    return _normalise_spectrum(spectrum)


def _add_frequency_noise(
    clean_spectrum: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    signal_power = float(np.mean(np.abs(clean_spectrum) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=clean_spectrum.size) + 1j * rng.normal(size=clean_spectrum.size)
    )
    return clean_spectrum + noise


def _controlled_masks(
    n_fft: int,
    selected_bins: int,
    overlap_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    common_count = int(round(selected_bins * overlap_fraction))
    common_count = min(max(common_count, 0), selected_bins)
    if 2 * selected_bins - common_count > n_fft:
        raise ValueError("N、M 和目标重叠比例无法构造两个严格大小为 M 的集合")
    permutation = rng.permutation(n_fft)
    common = permutation[:common_count]
    cursor = common_count
    a_only = permutation[cursor : cursor + selected_bins - common_count]
    cursor += selected_bins - common_count
    b_only = permutation[cursor : cursor + selected_bins - common_count]
    return np.sort(np.concatenate([common, a_only])), np.sort(np.concatenate([common, b_only]))


def _select_masks(
    config: ExperimentConfig,
    spectrum_a: np.ndarray,
    spectrum_b: np.ndarray,
    overlap_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if config.selector_mode == "controlled_overlap":
        mask_a, mask_b = _controlled_masks(
            config.n_fft,
            config.selected_bins,
            overlap_fraction,
            rng,
        )
        return mask_a, mask_b, {"selector_converged": True, "selector_sweeps": 0}
    if config.selector_mode != "local_zhai_thesis":
        raise ValueError(f"未知 selector_mode {config.selector_mode!r}")
    frequencies = shifted_frequencies(config.n_fft)
    result_a = select_zhai_thesis(
        np.abs(spectrum_a) ** 2,
        config.selected_bins,
        frequencies,
        config.zhai_max_sweeps,
        config.zhai_relative_tolerance,
    )
    result_b = select_zhai_thesis(
        np.abs(spectrum_b) ** 2,
        config.selected_bins,
        frequencies,
        config.zhai_max_sweeps,
        config.zhai_relative_tolerance,
    )
    return result_a.indices, result_b.indices, {
        "selector_converged": result_a.converged and result_b.converged,
        "selector_sweeps": max(result_a.sweeps, result_b.sweeps),
        "selector_swaps": result_a.accepted_swaps + result_b.accepted_swaps,
    }


def _completion_metrics(
    estimate_a: np.ndarray,
    estimate_b: np.ndarray,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    noisy_a: np.ndarray,
    noisy_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> dict[str, float]:
    missing_a = np.ones(clean_a.size, dtype=bool)
    missing_b = np.ones(clean_b.size, dtype=bool)
    missing_a[mask_a] = False
    missing_b[mask_b] = False
    estimate = np.concatenate([estimate_a[missing_a], estimate_b[missing_b]])
    clean = np.concatenate([clean_a[missing_a], clean_b[missing_b]])
    noisy = np.concatenate([noisy_a[missing_a], noisy_b[missing_b]])
    return {
        "missing_nmse_clean_db": complex_nmse_db(estimate, clean),
        "missing_nmse_noisy_db": complex_nmse_db(estimate, noisy),
        "missing_phase_rmse_rad": phase_rmse_rad(estimate, clean),
        "missing_complex_correlation": complex_correlation(estimate, clean),
    }


def _evaluate_method(
    method_name: str,
    spectrum_a: np.ndarray,
    spectrum_b: np.ndarray,
    valid_mask: np.ndarray,
    true_delay: float,
    config: ExperimentConfig,
) -> dict[str, float | str]:
    start = time.perf_counter()
    estimate = estimate_continuous_delay(
        spectrum_b,
        spectrum_a,
        valid_mask,
        config.lag_limit_samples,
        config.delay_grid_step_samples,
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0
    error = estimate.delay_samples - true_delay
    outlier = float(abs(error) > 1.0) if np.isfinite(error) else float("nan")
    return {
        "method": method_name,
        "estimated_delay_samples": estimate.delay_samples,
        "error_samples": error,
        "absolute_error_samples": abs(error),
        "squared_error_samples": error * error,
        "outlier_gt_1_sample": outlier,
        "peak_sidelobe_ratio": estimate.peak_sidelobe_ratio,
        "runtime_ms": runtime_ms,
    }


def _evaluate_pair(
    config: ExperimentConfig,
    snr_db: float,
    overlap_fraction: float,
    signal_kind: str,
    trial_index: int,
    rng: np.random.Generator,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    noisy_a: np.ndarray,
    noisy_b: np.ndarray,
    true_delay: float,
) -> list[dict[str, object]]:
    frequencies = shifted_frequencies(config.n_fft)
    mask_a, mask_b, selector_meta = _select_masks(
        config,
        noisy_a,
        noisy_b,
        overlap_fraction,
        rng,
    )
    intersection = np.intersect1d(mask_a, mask_b)
    union = np.union1d(mask_a, mask_b)
    achieved_overlap = intersection.size / config.selected_bins
    base = {
        "trial_id": f"{signal_kind}|{snr_db:g}|{overlap_fraction:g}|{trial_index}",
        "trial_index": trial_index,
        "snr_db": snr_db,
        "signal_kind": signal_kind,
        "requested_overlap_fraction": overlap_fraction,
        "overlap_fraction": achieved_overlap,
        "intersection_bins": intersection.size,
        "union_bins": union.size,
        "station_bins": config.selected_bins,
        "transmitted_complex_values": 2 * config.selected_bins,
        "transmitted_index_count": 2 * config.selected_bins,
        "true_delay_samples": true_delay,
        **selector_meta,
    }
    rows: list[dict[str, object]] = []
    all_mask = np.ones(config.n_fft, dtype=bool)
    rows.append(
        {
            **base,
            "transmitted_complex_values": 2 * config.n_fft,
            "transmitted_index_count": 0,
            **_evaluate_method("Full", noisy_a, noisy_b, all_mask, true_delay, config),
        }
    )

    # 原论文非对称对照：B 本地按自身周期图选频，A 在中心端匹配同一组频点。
    asymmetric_selection = select_zhai_thesis(
        np.abs(noisy_b) ** 2,
        config.selected_bins,
        frequencies,
        config.zhai_max_sweeps,
        config.zhai_relative_tolerance,
    )
    asymmetric_mask = np.zeros(config.n_fft, dtype=bool)
    asymmetric_mask[asymmetric_selection.indices] = True
    rows.append(
        {
            **base,
            "transmitted_complex_values": config.selected_bins,
            "transmitted_index_count": config.selected_bins,
            **_evaluate_method(
                "Zhai-Asymmetric",
                noisy_a,
                noisy_b,
                asymmetric_mask,
                true_delay,
                config,
            ),
        }
    )

    intersection_mask = np.zeros(config.n_fft, dtype=bool)
    intersection_mask[intersection] = True
    sparse_a = np.zeros(config.n_fft, dtype=complex)
    sparse_b = np.zeros(config.n_fft, dtype=complex)
    sparse_a[mask_a] = noisy_a[mask_a]
    sparse_b[mask_b] = noisy_b[mask_b]
    rows.append(
        {
            **base,
            **_evaluate_method(
                "Intersection-Continuous",
                sparse_a,
                sparse_b,
                intersection_mask,
                true_delay,
                config,
            ),
        }
    )

    for completion_method in COMPLETION_METHODS:
        completed_a = complete_spectrum(
            sparse_a,
            mask_a,
            completion_method,
            config.interpolation_factor,
        )
        completed_b = complete_spectrum(
            sparse_b,
            mask_b,
            completion_method,
            config.interpolation_factor,
        )
        row = {
            **base,
            **_evaluate_method(
                completion_method,
                completed_a.spectrum,
                completed_b.spectrum,
                all_mask,
                true_delay,
                config,
            ),
            **_completion_metrics(
                completed_a.spectrum,
                completed_b.spectrum,
                clean_a,
                clean_b,
                noisy_a,
                noisy_b,
                mask_a,
                mask_b,
            ),
            "observed_consistency_error": max(
                completed_a.observed_consistency_error,
                completed_b.observed_consistency_error,
            ),
            "generated_missing_energy": (
                completed_a.generated_missing_energy + completed_b.generated_missing_energy
            ),
        }
        rows.append(row)

    oracle_mask = np.zeros(config.n_fft, dtype=bool)
    oracle_mask[union] = True
    rows.append(
        {
            **base,
            "transmitted_complex_values": 2 * union.size,
            "transmitted_index_count": 2 * union.size,
            **_evaluate_method(
                "Oracle-Union",
                noisy_a,
                noisy_b,
                oracle_mask,
                true_delay,
                config,
            ),
        }
    )
    return rows


def _run_one_synthetic_trial(
    config: ExperimentConfig,
    snr_db: float,
    overlap_fraction: float,
    signal_kind: str,
    trial_index: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    frequencies = shifted_frequencies(config.n_fft)
    source = _make_source_spectrum(config.n_fft, signal_kind, rng)
    true_delay = float(rng.uniform(-0.8 * config.lag_limit_samples, 0.8 * config.lag_limit_samples))
    gain = np.exp(1j * rng.uniform(-np.pi, np.pi))
    clean_a = source
    clean_b = gain * source * np.exp(-2j * np.pi * frequencies * true_delay)
    noisy_a = _add_frequency_noise(clean_a, snr_db, rng)
    noisy_b = _add_frequency_noise(clean_b, snr_db, rng)
    return _evaluate_pair(
        config,
        snr_db,
        overlap_fraction,
        signal_kind,
        trial_index,
        rng,
        clean_a,
        clean_b,
        noisy_a,
        noisy_b,
        true_delay,
    )


def run_theory_gate(config: ExperimentConfig) -> list[dict[str, object]]:
    """验证理想周期插值不产生原始网格缺失频点，也不提高算子秩。"""

    rng = np.random.default_rng(config.seed)
    diagnostics: list[dict[str, object]] = []
    for factor in (2, 4, 8):
        indices = np.sort(rng.choice(config.n_fft, config.selected_bins, replace=False))
        sparse = np.zeros(config.n_fft, dtype=complex)
        sparse[indices] = rng.normal(size=config.selected_bins) + 1j * rng.normal(
            size=config.selected_bins
        )
        zero = complete_spectrum(sparse, indices, "zero_fill", factor)
        ideal = complete_spectrum(sparse, indices, "ideal_periodic", factor)
        maximum_difference = float(np.max(np.abs(ideal.spectrum - zero.spectrum)))
        rank = interpolation_operator_rank(
            config.n_fft,
            indices,
            "ideal_periodic",
            factor,
        )
        diagnostics.append(
            {
                "interpolation_factor": factor,
                "ideal_vs_zero_max_abs": maximum_difference,
                "generated_missing_energy": ideal.generated_missing_energy,
                "operator_rank": rank,
                "observed_bins": config.selected_bins,
                "passed": bool(
                    maximum_difference <= 1e-10
                    and ideal.generated_missing_energy <= 1e-20
                    and rank <= config.selected_bins
                ),
            }
        )
    if not all(bool(row["passed"]) for row in diagnostics):
        raise RuntimeError("理论门禁失败：理想周期插值与零填充不等价")
    return diagnostics


def _summarise(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()
    trials["valid_estimate"] = np.isfinite(trials["estimated_delay_samples"]).astype(float)
    grouped = trials.groupby(["method", "snr_db", "signal_kind", "overlap_fraction"], dropna=False)
    summary = grouped.agg(
        trials=("trial_id", "count"),
        valid_rate=("valid_estimate", "mean"),
        tdoa_mae_samples=("absolute_error_samples", "mean"),
        tdoa_rmse_samples=("squared_error_samples", lambda values: float(np.sqrt(np.mean(values)))),
        tdoa_median_ae_samples=("absolute_error_samples", "median"),
        tdoa_p95_ae_samples=("absolute_error_samples", lambda values: float(np.quantile(values, 0.95))),
        outlier_gt_1_rate=("outlier_gt_1_sample", "mean"),
        mean_peak_sidelobe_ratio=("peak_sidelobe_ratio", "mean"),
        mean_runtime_ms=("runtime_ms", "mean"),
    )
    return summary.reset_index()


def _paired_comparisons(
    trials: pd.DataFrame,
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    baseline = trials[trials["method"] == "Intersection-Continuous"].set_index("trial_id")
    rng = np.random.default_rng(seed + 991)
    rows = []
    for method in sorted(set(trials["method"]) - {"Intersection-Continuous"}):
        candidate = trials[trials["method"] == method].set_index("trial_id")
        common = baseline.index.intersection(candidate.index)
        baseline_error = baseline.loc[common, "absolute_error_samples"].to_numpy()
        candidate_error = candidate.loc[common, "absolute_error_samples"].to_numpy()
        finite = np.isfinite(baseline_error) & np.isfinite(candidate_error)
        improvements = baseline_error[finite] - candidate_error[finite]
        if improvements.size == 0:
            continue
        bootstrap = np.empty(bootstrap_repetitions, dtype=float)
        for repetition in range(bootstrap_repetitions):
            sample = rng.integers(0, improvements.size, improvements.size)
            bootstrap[repetition] = float(np.mean(improvements[sample]))
        rows.append(
            {
                "method": method,
                "baseline": "Intersection-Continuous",
                "paired_trials": improvements.size,
                "mean_mae_improvement_samples": float(np.mean(improvements)),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(config: ExperimentConfig) -> Path:
    diagnostics = run_theory_gate(config)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    if config.profile.startswith("project_"):
        from project_adapter import generate_project_pairs

        for snr_index, snr_db in enumerate(config.snr_values_db):
            batch = generate_project_pairs(
                config.trials_per_point,
                snr_db,
                config.seed + 1009 * snr_index,
                config.n_fft,
            )
            for trial_index in range(config.trials_per_point):
                clean_a = np.fft.fftshift(
                    np.fft.fft(batch.station_a_clean[trial_index], norm="ortho")
                )
                clean_b = np.fft.fftshift(
                    np.fft.fft(batch.station_b_clean[trial_index], norm="ortho")
                )
                noisy_a = np.fft.fftshift(
                    np.fft.fft(batch.station_a_noisy[trial_index], norm="ortho")
                )
                noisy_b = np.fft.fftshift(
                    np.fft.fft(batch.station_b_noisy[trial_index], norm="ortho")
                )
                # 估计器返回 B-A，而项目生成器标签保存 A-B，故此处显式取负。
                true_delay = -float(batch.delay_a_minus_b[trial_index])
                rows.extend(
                    _evaluate_pair(
                        config,
                        snr_db,
                        -1.0,
                        "urban8_los",
                        trial_index,
                        rng,
                        clean_a,
                        clean_b,
                        noisy_a,
                        noisy_b,
                        true_delay,
                    )
                )
    else:
        for signal_kind in config.signal_kinds:
            for snr_db in config.snr_values_db:
                for overlap_fraction in config.overlap_values:
                    for trial_index in range(config.trials_per_point):
                        rows.extend(
                            _run_one_synthetic_trial(
                                config,
                                snr_db,
                                overlap_fraction,
                                signal_kind,
                                trial_index,
                                rng,
                            )
                        )
    trials = pd.DataFrame(rows)
    summary = _summarise(trials)
    paired = _paired_comparisons(
        trials,
        config.bootstrap_repetitions,
        config.seed,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_directory = (
        PROJECT_DIRECTORY
        / "运行结果"
        / "频谱插值对齐"
        / f"{timestamp}_{config.profile}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    with (output_directory / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
    manifest = {
        "route": "standalone_frequency_alignment",
        "independent_from_main": True,
        "selector_interpretation": (
            "双站各用本地周期图运行学位论文的严格0-1逐点交换；"
            "受控重叠配置除外"
        ),
        "communication_budget": {
            "complex_values_per_pair": 2 * config.selected_bins,
            "indices_per_pair": 2 * config.selected_bins,
        },
        "critical_interpretation": (
            "插值输出是已有频点的确定性组合，不作为新增独立观测，也不进入CRLB信息求和"
        ),
    }
    with (output_directory / "protocol_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    trials.to_csv(output_directory / "trial_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_directory / "summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output_directory / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(
        output_directory / "operator_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (output_directory / "plot_data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "config": config.to_dict(),
                "trials": trials,
                "summary": summary,
                "paired_comparisons": paired,
                "operator_diagnostics": diagnostics,
            },
            handle,
        )
    save_summary_figures(trials, output_directory)
    log_lines = [
        f"profile={config.profile}",
        f"rows={len(trials)}",
        f"theory_gate_passed={all(row['passed'] for row in diagnostics)}",
        "注意：本运行只验证独立双站选频路线，不修改或评价 V5-A.1/Shared64。",
    ]
    (output_directory / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return output_directory



# ---------------------------------------------------------------------------
# BPSK-RRC 与时域补零密集频谱扩展（统一纳入原插值实验入口）
# ---------------------------------------------------------------------------

OLD_RESULT_DIRECTORY = (
    PROJECT_DIRECTORY
    / "运行结果"
    / "频谱插值对齐"
    / "20260714_184753_411401_synthetic_full"
)
OUTPUT_ROOT = PROJECT_DIRECTORY / "运行结果" / "频谱插值对齐"

SIGNAL_KINDS = (
    "smooth_complex",
    "smooth_magnitude_random_phase",
    "broadband_random",
    "multiband",
    "urban8_bpsk_rrc",
)
SIGNAL_KIND_LABELS = {
    "smooth_complex": "平滑复谱",
    "smooth_magnitude_random_phase": "平滑幅度随机相位",
    "broadband_random": "宽带随机复谱",
    "multiband": "多频带随机相位",
    "urban8_bpsk_rrc": "Urban8 同款 BPSK-RRC",
}
PSEUDO_COMPLETION_METHODS = (
    "zero_fill",
    "ideal_periodic",
    "polyphase_fir",
    "linear_periodic",
    "cubic_periodic",
)
TZP_DENSE_METHOD = "TZP-DS"
METHOD_ORDER = (
    "Full",
    "Zhai-Asymmetric",
    "Intersection-Continuous",
    *PSEUDO_COMPLETION_METHODS,
    "Oracle-Union",
    TZP_DENSE_METHOD,
)
DIRECT_METHODS = (
    "Full",
    "Zhai-Asymmetric",
    "Intersection-Continuous",
    "Oracle-Union",
)


@dataclass(frozen=True)
class SweepConfig:
    """一次独立参数扫描所需的冻结协议。"""

    profile: str = "bpsk_parameter_sweep"
    seed: int = 20260714
    n_fft: int = 512
    selected_bins_values: tuple[int, ...] = (256, 128, 64, 32)
    interpolation_factor: int = 8
    snr_values_db: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0)
    overlap_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    signal_kinds: tuple[str, ...] = SIGNAL_KINDS
    trials_per_condition: int = 300
    true_delay_min_samples: float = -25.0
    true_delay_max_samples: float = 25.0
    lag_limit_samples: float = 25.0
    delay_grid_step_samples: float = 0.01
    selector_mode: str = "controlled_overlap"
    zhai_max_sweeps: int = 50
    zhai_relative_tolerance: float = 1e-10
    bootstrap_repetitions: int = 2000
    bpsk_samples_per_symbol: int = 2
    bpsk_rrc_rolloff: float = 0.2
    bpsk_rrc_span_symbols: int = 8
    estimator_batch_size: int = 20

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Condition:
    """单个可断点续跑的 Monte Carlo 条件。"""

    index: int
    signal_kind: str
    selected_bins: int
    snr_db: float
    overlap_fraction: float
    seed: int

    @property
    def identifier(self) -> str:
        snr_token = f"{self.snr_db:g}".replace("-", "m").replace(".", "p")
        overlap_token = f"{self.overlap_fraction:g}".replace(".", "p")
        return (
            f"{self.index:04d}_{self.signal_kind}_M{self.selected_bins}"
            f"_snr{snr_token}_ov{overlap_token}"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze_old_result_manifest() -> dict[str, object]:
    """记录旧正式结果的关键文件哈希，证明新路线未覆盖旧结果。"""

    required = (
        "config.json",
        "protocol_manifest.json",
        "operator_diagnostics.csv",
        "paired_comparisons.csv",
        "summary.csv",
        "trial_results.csv",
        "plot_data.pkl",
        "run.log",
    )
    missing = [name for name in required if not (OLD_RESULT_DIRECTORY / name).is_file()]
    if missing:
        raise FileNotFoundError(f"旧正式结果不完整，无法冻结：{missing}")
    return {
        "frozen_result_directory": str(OLD_RESULT_DIRECTORY),
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "files": {
            name: {
                "bytes": int((OLD_RESULT_DIRECTORY / name).stat().st_size),
                "sha256": _sha256(OLD_RESULT_DIRECTORY / name),
            }
            for name in required
        },
    }


def _rrc_filter(beta: float, span_symbols: int, samples_per_symbol: int) -> np.ndarray:
    """复现 ``signal_gen.rrc_filter`` 的单位能量 RRC 滤波器。"""

    num_taps = span_symbols * samples_per_symbol
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / samples_per_symbol
    taps = np.zeros_like(t, dtype=float)
    at_zero = np.abs(t) < 1e-12
    taps[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
    at_edge = np.abs(np.abs(4.0 * beta * t) - 1.0) < 1e-10
    at_edge &= ~at_zero
    if np.any(at_edge):
        taps[at_edge] = (
            beta
            / np.sqrt(2.0)
            * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        )
    regular = ~at_zero & ~at_edge
    regular_t = t[regular]
    taps[regular] = (
        np.sin(np.pi * regular_t * (1.0 - beta))
        + 4.0
        * beta
        * regular_t
        * np.cos(np.pi * regular_t * (1.0 + beta))
    ) / (
        np.pi
        * regular_t
        * (1.0 - (4.0 * beta * regular_t) ** 2)
    )
    taps /= np.sqrt(np.sum(taps**2))
    return taps


def _make_sweep_source_spectrum(
    config: SweepConfig,
    signal_kind: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成四类机制频谱或 Urban8 同参数 BPSK-RRC 波形频谱。"""

    n_fft = config.n_fft
    frequencies = shifted_frequencies(n_fft)
    if signal_kind == "smooth_complex":
        magnitude = 0.2 + np.exp(-0.5 * (frequencies / 0.22) ** 2)
        phase = 0.7 * np.sin(2.0 * np.pi * frequencies) + 0.3 * np.cos(
            4.0 * np.pi * frequencies
        )
        spectrum = magnitude * np.exp(1j * phase)
    elif signal_kind == "smooth_magnitude_random_phase":
        magnitude = 0.2 + np.exp(-0.5 * (frequencies / 0.22) ** 2)
        spectrum = magnitude * np.exp(1j * rng.uniform(-np.pi, np.pi, n_fft))
    elif signal_kind == "broadband_random":
        spectrum = rng.normal(size=n_fft) + 1j * rng.normal(size=n_fft)
    elif signal_kind == "multiband":
        magnitude = (
            0.05
            + np.exp(-0.5 * ((frequencies - 0.24) / 0.055) ** 2)
            + 0.8 * np.exp(-0.5 * ((frequencies + 0.19) / 0.07) ** 2)
        )
        spectrum = magnitude * np.exp(1j * rng.uniform(-np.pi, np.pi, n_fft))
    elif signal_kind == "urban8_bpsk_rrc":
        samples_per_symbol = config.bpsk_samples_per_symbol
        symbol_count = int(math.ceil(n_fft / samples_per_symbol))
        symbols = rng.choice((-1.0, 1.0), size=symbol_count)
        baseband = np.repeat(symbols, samples_per_symbol)[:n_fft]
        taps = _rrc_filter(
            config.bpsk_rrc_rolloff,
            config.bpsk_rrc_span_symbols,
            samples_per_symbol,
        )
        shaped = signal.convolve(baseband, taps, mode="same")
        spectrum = np.fft.fftshift(np.fft.fft(shaped, norm="ortho"))
    else:
        raise ValueError(f"未知信号类型 {signal_kind!r}")
    return _normalise_spectrum(spectrum)


def _fft_delay_estimates(
    cross_spectra: np.ndarray,
    active_counts: np.ndarray,
    config: SweepConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """在 0.01 sample 网格上用等价零填充 IFFT 批量计算连续时延得分。"""

    started = time.perf_counter()
    spectra = np.asarray(cross_spectra, dtype=np.complex128)
    counts = np.asarray(active_counts, dtype=int)
    oversampling = int(round(1.0 / config.delay_grid_step_samples))
    if not np.isclose(oversampling * config.delay_grid_step_samples, 1.0):
        raise ValueError("FFT 批量估计器要求 delay_grid_step_samples 的倒数为整数")
    transform_length = config.n_fft * oversampling
    harmonics = np.rint(shifted_frequencies(config.n_fft) * config.n_fft).astype(int)
    padded_indices = np.mod(harmonics, transform_length)
    padded = np.zeros((spectra.shape[0], transform_length), dtype=np.complex128)
    padded[:, padded_indices] = spectra
    correlations = np.fft.ifft(padded, axis=1) * transform_length
    lag_ticks = np.arange(
        int(round(config.true_delay_min_samples * oversampling)),
        int(round(config.true_delay_max_samples * oversampling)) + 1,
        dtype=int,
    )
    lag_indices = np.mod(lag_ticks, transform_length)
    scores = np.abs(correlations[:, lag_indices])
    lags = lag_ticks.astype(float) / oversampling
    peak_indices = np.argmax(scores, axis=1)
    estimates = lags[peak_indices]
    sidelobe_ratios = np.full(spectra.shape[0], np.nan, dtype=float)
    exclusion = oversampling
    norms = np.linalg.norm(spectra, axis=1)
    valid = (counts >= 2) & (norms > np.finfo(float).tiny)
    for row_index in np.flatnonzero(valid):
        peak_index = int(peak_indices[row_index])
        peak_value = float(scores[row_index, peak_index])
        keep = np.ones(scores.shape[1], dtype=bool)
        keep[
            max(0, peak_index - exclusion) : min(
                scores.shape[1],
                peak_index + exclusion + 1,
            )
        ] = False
        sidelobe = float(np.max(scores[row_index, keep])) if np.any(keep) else 0.0
        sidelobe_ratios[row_index] = sidelobe / max(
            peak_value,
            np.finfo(float).tiny,
        )
    estimates[~valid] = np.nan
    elapsed_ms_per_row = (
        (time.perf_counter() - started) * 1000.0 / max(spectra.shape[0], 1)
    )
    return estimates, sidelobe_ratios, elapsed_ms_per_row


def _nmse_db(estimate: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sum(np.abs(target) ** 2))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    ratio = float(np.sum(np.abs(estimate - target) ** 2) / denominator)
    return float(10.0 * np.log10(max(ratio, np.finfo(float).tiny)))


def _complex_correlation(estimate: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.linalg.norm(estimate) * np.linalg.norm(target))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(abs(np.vdot(target, estimate)) / denominator)


def _phase_rmse(estimate: np.ndarray, target: np.ndarray) -> float:
    target_scale = float(np.max(np.abs(target), initial=0.0))
    estimate_scale = float(np.max(np.abs(estimate), initial=0.0))
    threshold = max(target_scale, estimate_scale) * 1e-8
    active = (np.abs(target) > threshold) & (np.abs(estimate) > threshold)
    if not np.any(active):
        return float("nan")
    wrapped = np.angle(estimate[active] * np.conj(target[active]))
    return float(np.sqrt(np.mean(wrapped**2)))


def _effective_dense_mask(spectrum: np.ndarray) -> np.ndarray:
    magnitude = np.abs(spectrum)
    threshold = max(
        1e-10 * float(np.max(magnitude, initial=0.0)),
        100.0 * np.finfo(float).eps * float(np.linalg.norm(magnitude)),
    )
    return magnitude > threshold


def _tzp_dense_metrics(
    dense_a: np.ndarray,
    dense_b: np.ndarray,
    dense_clean_a: np.ndarray,
    dense_clean_b: np.ndarray,
    dense_noisy_a: np.ndarray,
    dense_noisy_b: np.ndarray,
    sparse_a: np.ndarray,
    sparse_b: np.ndarray,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    noisy_a: np.ndarray,
    noisy_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    factor: int,
) -> dict[str, float]:
    """计算 TZP-DS 的原网格门禁量和密集曲线逼近指标。"""

    n_fft = sparse_a.size
    mapped_a = sample_original_grid(dense_a, n_fft, factor)
    mapped_b = sample_original_grid(dense_b, n_fft, factor)
    observed_error = float(
        np.sqrt(
            np.sum(np.abs(mapped_a[mask_a] - sparse_a[mask_a]) ** 2)
            + np.sum(np.abs(mapped_b[mask_b] - sparse_b[mask_b]) ** 2)
        )
        / max(
            np.sqrt(
                np.sum(np.abs(sparse_a[mask_a]) ** 2)
                + np.sum(np.abs(sparse_b[mask_b]) ** 2)
            ),
            np.finfo(float).tiny,
        )
    )
    missing_a = np.ones(n_fft, dtype=bool)
    missing_b = np.ones(n_fft, dtype=bool)
    missing_a[mask_a] = False
    missing_b[mask_b] = False
    mapped_missing = np.concatenate((mapped_a[missing_a], mapped_b[missing_b]))
    clean_missing = np.concatenate((clean_a[missing_a], clean_b[missing_b]))
    noisy_missing = np.concatenate((noisy_a[missing_a], noisy_b[missing_b]))
    dense_pair = np.concatenate((dense_a, dense_b))
    dense_clean_pair = np.concatenate((dense_clean_a, dense_clean_b))
    dense_noisy_pair = np.concatenate((dense_noisy_a, dense_noisy_b))
    effective_common = _effective_dense_mask(dense_a) & _effective_dense_mask(dense_b)
    return {
        "missing_nmse_clean_db": _nmse_db(mapped_missing, clean_missing),
        "missing_nmse_noisy_db": _nmse_db(mapped_missing, noisy_missing),
        "missing_phase_rmse_rad": _phase_rmse(mapped_missing, clean_missing),
        "missing_complex_correlation": _complex_correlation(
            mapped_missing,
            clean_missing,
        ),
        "observed_consistency_error": observed_error,
        "generated_missing_energy": float(np.sum(np.abs(mapped_missing) ** 2)),
        "dense_grid_nmse_clean_db": _nmse_db(dense_pair, dense_clean_pair),
        "dense_grid_nmse_noisy_db": _nmse_db(dense_pair, dense_noisy_pair),
        "dense_grid_phase_rmse_rad": _phase_rmse(dense_pair, dense_clean_pair),
        "dense_grid_complex_correlation": _complex_correlation(
            dense_pair,
            dense_clean_pair,
        ),
        "dense_grid_common_nonzero_rate": float(np.mean(effective_common)),
        "dense_grid_points": float(dense_a.size),
        "time_zero_padding_factor": float(factor),
    }


def _method_row(
    method: str,
    estimate: float,
    sidelobe_ratio: float,
    true_delay: float,
    runtime_ms: float,
    base: dict[str, object],
    analytical_reuse: bool = False,
) -> dict[str, object]:
    error = estimate - true_delay
    valid = bool(np.isfinite(error))
    row = {
        **base,
        "method": method,
        "estimated_delay_samples": estimate,
        "error_samples": error,
        "absolute_error_samples": abs(error),
        "squared_error_samples": error * error,
        "outlier_gt_1_sample": float(abs(error) > 1.0) if valid else np.nan,
        "peak_sidelobe_ratio": sidelobe_ratio,
        "runtime_ms": runtime_ms,
        "analytical_equivalence_reuse": analytical_reuse,
        "missing_nmse_clean_db": np.nan,
        "missing_nmse_noisy_db": np.nan,
        "missing_phase_rmse_rad": np.nan,
        "missing_complex_correlation": np.nan,
        "observed_consistency_error": np.nan,
        "generated_missing_energy": np.nan,
        "dense_grid_nmse_clean_db": np.nan,
        "dense_grid_nmse_noisy_db": np.nan,
        "dense_grid_phase_rmse_rad": np.nan,
        "dense_grid_complex_correlation": np.nan,
        "dense_grid_common_nonzero_rate": np.nan,
        "dense_grid_points": np.nan,
        "time_zero_padding_factor": np.nan,
    }
    if analytical_reuse:
        row.update(
            {
                "missing_nmse_clean_db": 0.0,
                "missing_nmse_noisy_db": 0.0,
                "observed_consistency_error": 0.0,
                "generated_missing_energy": 0.0,
            }
        )
    return row


def _condition_summary(rows: pd.DataFrame) -> list[dict[str, object]]:
    work = rows.copy()
    work["valid_estimate"] = np.isfinite(work["estimated_delay_samples"]).astype(float)
    summary = (
        work.groupby("method", sort=False)
        .agg(
            trials=("trial_id", "count"),
            valid_rate=("valid_estimate", "mean"),
            tdoa_mae_samples=("absolute_error_samples", "mean"),
            tdoa_rmse_samples=(
                "squared_error_samples",
                lambda values: float(np.sqrt(np.nanmean(values)))
                if np.any(np.isfinite(values))
                else np.nan,
            ),
            tdoa_median_ae_samples=("absolute_error_samples", "median"),
            tdoa_p95_ae_samples=(
                "absolute_error_samples",
                lambda values: float(np.nanquantile(values, 0.95))
                if np.any(np.isfinite(values))
                else np.nan,
            ),
            outlier_gt_1_rate=("outlier_gt_1_sample", "mean"),
            mean_peak_sidelobe_ratio=("peak_sidelobe_ratio", "mean"),
            mean_runtime_ms=("runtime_ms", "mean"),
            mean_observed_consistency_error=("observed_consistency_error", "mean"),
            mean_generated_missing_energy=("generated_missing_energy", "mean"),
            mean_dense_grid_nmse_clean_db=("dense_grid_nmse_clean_db", "mean"),
            mean_dense_grid_nmse_noisy_db=("dense_grid_nmse_noisy_db", "mean"),
            mean_dense_grid_phase_rmse_rad=("dense_grid_phase_rmse_rad", "mean"),
            mean_dense_grid_complex_correlation=(
                "dense_grid_complex_correlation",
                "mean",
            ),
            mean_dense_grid_common_nonzero_rate=(
                "dense_grid_common_nonzero_rate",
                "mean",
            ),
        )
        .reset_index()
    )
    return summary.to_dict(orient="records")


def _run_condition(
    config_payload: dict[str, Any],
    condition_payload: dict[str, Any],
    output_directory_text: str,
) -> dict[str, object]:
    """工作进程入口：运行一个条件并原子保存 trial、summary 和 paired 数据。"""

    config = SweepConfig(**config_payload)
    condition = Condition(**condition_payload)
    output_directory = Path(output_directory_text)
    chunk_directory = output_directory / "condition_chunks"
    chunk_path = chunk_directory / f"{condition.identifier}.csv.gz"
    summary_path = chunk_directory / f"{condition.identifier}.summary.json"
    paired_path = chunk_directory / f"{condition.identifier}.paired.npz"
    metadata_path = chunk_directory / f"{condition.identifier}.done.json"
    if all(path.is_file() for path in (chunk_path, summary_path, paired_path, metadata_path)):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    rng = np.random.default_rng(condition.seed)
    frequencies = shifted_frequencies(config.n_fft)
    all_rows: list[dict[str, object]] = []
    selector_sweeps: list[int] = []
    selector_swaps: list[int] = []
    direct_method_lookup = {name: index for index, name in enumerate(DIRECT_METHODS)}

    for batch_start in range(0, config.trials_per_condition, config.estimator_batch_size):
        batch_end = min(
            config.trials_per_condition,
            batch_start + config.estimator_batch_size,
        )
        batch_payloads: list[dict[str, object]] = []
        batch_cross_spectra: list[np.ndarray] = []
        batch_active_counts: list[int] = []

        for trial_index in range(batch_start, batch_end):
            source = _make_sweep_source_spectrum(config, condition.signal_kind, rng)
            true_delay = float(
                rng.uniform(
                    config.true_delay_min_samples,
                    config.true_delay_max_samples,
                )
            )
            gain = np.exp(1j * rng.uniform(-np.pi, np.pi))
            clean_a = source
            clean_b = gain * source * np.exp(-2j * np.pi * frequencies * true_delay)
            noisy_a = _add_frequency_noise(clean_a, condition.snr_db, rng)
            noisy_b = _add_frequency_noise(clean_b, condition.snr_db, rng)
            mask_a, mask_b = _controlled_masks(
                config.n_fft,
                condition.selected_bins,
                condition.overlap_fraction,
                rng,
            )
            intersection = np.intersect1d(mask_a, mask_b)
            union = np.union1d(mask_a, mask_b)
            asymmetric = select_zhai_thesis(
                np.abs(noisy_b) ** 2,
                condition.selected_bins,
                frequencies,
                config.zhai_max_sweeps,
                config.zhai_relative_tolerance,
            )
            selector_sweeps.append(asymmetric.sweeps)
            selector_swaps.append(asymmetric.accepted_swaps)

            full_cross = noisy_b * np.conj(noisy_a)
            asymmetric_cross = np.zeros(config.n_fft, dtype=complex)
            asymmetric_cross[asymmetric.indices] = full_cross[asymmetric.indices]
            intersection_cross = np.zeros(config.n_fft, dtype=complex)
            intersection_cross[intersection] = full_cross[intersection]
            oracle_cross = np.zeros(config.n_fft, dtype=complex)
            oracle_cross[union] = full_cross[union]
            sparse_a = np.zeros(config.n_fft, dtype=complex)
            sparse_b = np.zeros(config.n_fft, dtype=complex)
            sparse_a[mask_a] = noisy_a[mask_a]
            sparse_b[mask_b] = noisy_b[mask_b]
            batch_cross_spectra.extend(
                (full_cross, asymmetric_cross, intersection_cross, oracle_cross)
            )
            batch_active_counts.extend(
                (
                    config.n_fft,
                    condition.selected_bins,
                    int(intersection.size),
                    int(union.size),
                )
            )
            batch_payloads.append(
                {
                    "trial_index": trial_index,
                    "true_delay": true_delay,
                    "intersection_bins": int(intersection.size),
                    "union_bins": int(union.size),
                    "selector_converged": asymmetric.converged,
                    "selector_sweeps": asymmetric.sweeps,
                    "selector_swaps": asymmetric.accepted_swaps,
                    "clean_a": clean_a,
                    "clean_b": clean_b,
                    "noisy_a": noisy_a,
                    "noisy_b": noisy_b,
                    "sparse_a": sparse_a,
                    "sparse_b": sparse_b,
                    "mask_a": mask_a,
                    "mask_b": mask_b,
                }
            )

        estimates, sidelobe_ratios, runtime_ms = _fft_delay_estimates(
            np.stack(batch_cross_spectra),
            np.asarray(batch_active_counts),
            config,
        )
        estimate_matrix = estimates.reshape(-1, len(DIRECT_METHODS))
        sidelobe_matrix = sidelobe_ratios.reshape(-1, len(DIRECT_METHODS))
        sparse_a_batch = np.stack([payload["sparse_a"] for payload in batch_payloads])
        sparse_b_batch = np.stack([payload["sparse_b"] for payload in batch_payloads])
        clean_a_batch = np.stack([payload["clean_a"] for payload in batch_payloads])
        clean_b_batch = np.stack([payload["clean_b"] for payload in batch_payloads])
        noisy_a_batch = np.stack([payload["noisy_a"] for payload in batch_payloads])
        noisy_b_batch = np.stack([payload["noisy_b"] for payload in batch_payloads])
        dense_a_batch = time_zero_padded_dense_spectrum(
            sparse_a_batch,
            config.interpolation_factor,
        )
        dense_b_batch = time_zero_padded_dense_spectrum(
            sparse_b_batch,
            config.interpolation_factor,
        )
        dense_clean_a_batch = time_zero_padded_dense_spectrum(
            clean_a_batch,
            config.interpolation_factor,
        )
        dense_clean_b_batch = time_zero_padded_dense_spectrum(
            clean_b_batch,
            config.interpolation_factor,
        )
        dense_noisy_a_batch = time_zero_padded_dense_spectrum(
            noisy_a_batch,
            config.interpolation_factor,
        )
        dense_noisy_b_batch = time_zero_padded_dense_spectrum(
            noisy_b_batch,
            config.interpolation_factor,
        )
        dense_cross_batch = dense_b_batch * np.conj(dense_a_batch)
        tzp_estimates, tzp_sidelobe_ratios, tzp_runtime_ms = dense_delay_estimates(
            dense_cross_batch,
            np.full(len(batch_payloads), condition.selected_bins, dtype=int),
            config.true_delay_min_samples,
            config.true_delay_max_samples,
            config.delay_grid_step_samples,
        )
        for local_index, payload in enumerate(batch_payloads):
            trial_index = int(payload["trial_index"])
            true_delay = float(payload["true_delay"])
            intersection_bins = int(payload["intersection_bins"])
            union_bins = int(payload["union_bins"])
            base = {
                "trial_id": (
                    f"{condition.signal_kind}|M{condition.selected_bins}|"
                    f"{condition.snr_db:g}|{condition.overlap_fraction:g}|{trial_index}"
                ),
                "condition_index": condition.index,
                "trial_index": trial_index,
                "snr_db": condition.snr_db,
                "signal_kind": condition.signal_kind,
                "selected_bins": condition.selected_bins,
                "requested_overlap_fraction": condition.overlap_fraction,
                "overlap_fraction": intersection_bins / condition.selected_bins,
                "intersection_bins": intersection_bins,
                "union_bins": union_bins,
                "station_bins": condition.selected_bins,
                "true_delay_samples": true_delay,
                "selector_converged": payload["selector_converged"],
                "selector_sweeps": payload["selector_sweeps"],
                "selector_swaps": payload["selector_swaps"],
            }
            direct_rows: dict[str, dict[str, object]] = {}
            for method in DIRECT_METHODS:
                method_index = direct_method_lookup[method]
                method_base = dict(base)
                if method == "Full":
                    method_base.update(
                        transmitted_complex_values=2 * config.n_fft,
                        transmitted_index_count=0,
                    )
                elif method == "Zhai-Asymmetric":
                    method_base.update(
                        transmitted_complex_values=condition.selected_bins,
                        transmitted_index_count=condition.selected_bins,
                    )
                elif method == "Oracle-Union":
                    method_base.update(
                        transmitted_complex_values=2 * union_bins,
                        transmitted_index_count=2 * union_bins,
                    )
                else:
                    method_base.update(
                        transmitted_complex_values=2 * condition.selected_bins,
                        transmitted_index_count=2 * condition.selected_bins,
                    )
                direct_rows[method] = _method_row(
                    method,
                    float(estimate_matrix[local_index, method_index]),
                    float(sidelobe_matrix[local_index, method_index]),
                    true_delay,
                    runtime_ms,
                    method_base,
                )
            all_rows.append(direct_rows["Full"])
            all_rows.append(direct_rows["Zhai-Asymmetric"])
            intersection_row = direct_rows["Intersection-Continuous"]
            all_rows.append(intersection_row)
            for method in PSEUDO_COMPLETION_METHODS:
                pseudo_base = dict(base)
                pseudo_base.update(
                    transmitted_complex_values=2 * condition.selected_bins,
                    transmitted_index_count=2 * condition.selected_bins,
                )
                all_rows.append(
                    _method_row(
                        method,
                        float(intersection_row["estimated_delay_samples"]),
                        float(intersection_row["peak_sidelobe_ratio"]),
                        true_delay,
                        0.0,
                        pseudo_base,
                        analytical_reuse=True,
                    )
                )
            tzp_base = dict(base)
            tzp_base.update(
                transmitted_complex_values=2 * condition.selected_bins,
                transmitted_index_count=2 * condition.selected_bins,
            )
            tzp_row = _method_row(
                TZP_DENSE_METHOD,
                float(tzp_estimates[local_index]),
                float(tzp_sidelobe_ratios[local_index]),
                true_delay,
                tzp_runtime_ms,
                tzp_base,
            )
            tzp_row.update(
                _tzp_dense_metrics(
                    dense_a_batch[local_index],
                    dense_b_batch[local_index],
                    dense_clean_a_batch[local_index],
                    dense_clean_b_batch[local_index],
                    dense_noisy_a_batch[local_index],
                    dense_noisy_b_batch[local_index],
                    np.asarray(payload["sparse_a"]),
                    np.asarray(payload["sparse_b"]),
                    np.asarray(payload["clean_a"]),
                    np.asarray(payload["clean_b"]),
                    np.asarray(payload["noisy_a"]),
                    np.asarray(payload["noisy_b"]),
                    np.asarray(payload["mask_a"]),
                    np.asarray(payload["mask_b"]),
                    config.interpolation_factor,
                )
            )
            all_rows.append(tzp_row)
            all_rows.append(direct_rows["Oracle-Union"])

    frame = pd.DataFrame(all_rows)
    frame["method"] = pd.Categorical(frame["method"], METHOD_ORDER, ordered=True)
    frame = frame.sort_values(["trial_index", "method"]).reset_index(drop=True)
    frame["method"] = frame["method"].astype(str)
    summary_records = _condition_summary(frame)
    for record in summary_records:
        record.update(
            {
                "condition_index": condition.index,
                "signal_kind": condition.signal_kind,
                "selected_bins": condition.selected_bins,
                "snr_db": condition.snr_db,
                "overlap_fraction": condition.overlap_fraction,
            }
        )

    baseline = frame.loc[
        frame["method"].eq("Intersection-Continuous"),
        ["trial_id", "absolute_error_samples"],
    ].rename(columns={"absolute_error_samples": "baseline_error"})
    paired_arrays: dict[str, np.ndarray] = {}
    maximum_pseudo_delay_difference = 0.0
    maximum_pseudo_error_difference = 0.0
    for method in METHOD_ORDER:
        if method == "Intersection-Continuous":
            continue
        candidate = frame.loc[
            frame["method"].eq(method),
            ["trial_id", "estimated_delay_samples", "absolute_error_samples"],
        ]
        merged = candidate.merge(baseline, on="trial_id", validate="one_to_one")
        finite = np.isfinite(merged["absolute_error_samples"]) & np.isfinite(
            merged["baseline_error"]
        )
        paired_arrays[method] = (
            merged.loc[finite, "baseline_error"].to_numpy()
            - merged.loc[finite, "absolute_error_samples"].to_numpy()
        )
        if method in PSEUDO_COMPLETION_METHODS:
            baseline_estimate = frame.loc[
                frame["method"].eq("Intersection-Continuous"),
                "estimated_delay_samples",
            ].to_numpy()
            candidate_estimate = frame.loc[
                frame["method"].eq(method),
                "estimated_delay_samples",
            ].to_numpy()
            finite_delay = np.isfinite(baseline_estimate) & np.isfinite(
                candidate_estimate
            )
            if np.any(finite_delay):
                maximum_pseudo_delay_difference = max(
                    maximum_pseudo_delay_difference,
                    float(
                        np.max(
                            np.abs(
                                baseline_estimate[finite_delay]
                                - candidate_estimate[finite_delay]
                            )
                        )
                    ),
                )
            baseline_error = frame.loc[
                frame["method"].eq("Intersection-Continuous"),
                "absolute_error_samples",
            ].to_numpy()
            candidate_error = frame.loc[
                frame["method"].eq(method),
                "absolute_error_samples",
            ].to_numpy()
            finite_error = np.isfinite(baseline_error) & np.isfinite(candidate_error)
            if np.any(finite_error):
                maximum_pseudo_error_difference = max(
                    maximum_pseudo_error_difference,
                    float(
                        np.max(
                            np.abs(
                                baseline_error[finite_error]
                                - candidate_error[finite_error]
                            )
                        )
                    ),
                )

    temporary_chunk = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    with gzip.open(temporary_chunk, "wt", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)
    temporary_chunk.replace(chunk_path)
    _atomic_write_json(summary_path, summary_records)
    temporary_paired = paired_path.with_suffix(".npz.tmp")
    with temporary_paired.open("wb") as handle:
        np.savez_compressed(handle, **paired_arrays)
    temporary_paired.replace(paired_path)
    elapsed_seconds = time.perf_counter() - started
    metadata = {
        "condition": asdict(condition),
        "condition_id": condition.identifier,
        "rows": int(len(frame)),
        "runtime_seconds": elapsed_seconds,
        "selector_sweeps_mean": float(np.mean(selector_sweeps)),
        "selector_swaps_mean": float(np.mean(selector_swaps)),
        "maximum_pseudo_delay_difference_samples": maximum_pseudo_delay_difference,
        "maximum_pseudo_absolute_error_difference_samples": maximum_pseudo_error_difference,
        "chunk_sha256": _sha256(chunk_path),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write_json(metadata_path, metadata)
    return metadata


def _build_conditions(config: SweepConfig) -> list[Condition]:
    conditions: list[Condition] = []
    index = 0
    for signal_kind in config.signal_kinds:
        for selected_bins in config.selected_bins_values:
            for snr_db in config.snr_values_db:
                for overlap_fraction in config.overlap_values:
                    seed_sequence = np.random.SeedSequence(
                        [
                            config.seed,
                            index,
                            selected_bins,
                            int(round((snr_db + 100.0) * 10.0)),
                            int(round(overlap_fraction * 1000.0)),
                        ]
                    )
                    seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
                    conditions.append(
                        Condition(
                            index=index,
                            signal_kind=signal_kind,
                            selected_bins=selected_bins,
                            snr_db=snr_db,
                            overlap_fraction=overlap_fraction,
                            seed=seed,
                        )
                    )
                    index += 1
    return conditions


def _validate_fft_estimator(config: SweepConfig) -> dict[str, object]:
    """独立核对快速 FFT 与定义式求和；不参与正式 Monte Carlo 流程。"""

    rng = np.random.default_rng(config.seed + 17)
    cross = rng.normal(size=config.n_fft) + 1j * rng.normal(size=config.n_fft)
    mask = np.zeros(config.n_fft, dtype=bool)
    mask[rng.choice(config.n_fft, 64, replace=False)] = True
    sparse_cross = np.where(mask, cross, 0.0)
    estimate, _, _ = _fft_delay_estimates(
        sparse_cross[None, :],
        np.array([np.count_nonzero(mask)]),
        config,
    )
    lags = np.arange(
        config.true_delay_min_samples,
        config.true_delay_max_samples + 0.5 * config.delay_grid_step_samples,
        config.delay_grid_step_samples,
    )
    steering = np.exp(
        2j
        * np.pi
        * np.outer(lags, shifted_frequencies(config.n_fft)[mask])
    )
    direct_score = np.abs(steering @ cross[mask])
    direct_estimate = float(lags[int(np.argmax(direct_score))])
    return {
        "scope": "implementation_only_not_formal_monte_carlo",
        "input_description": "固定种子生成的64频点稀疏复互谱测试向量",
        "fft_estimate_samples": float(estimate[0]),
        "direct_matrix_estimate_samples": direct_estimate,
        "absolute_difference_samples": abs(float(estimate[0]) - direct_estimate),
    }


def _validate_tzp_dense_estimator(config: SweepConfig) -> dict[str, object]:
    """独立核对 TZP-DS ZoomFFT 与密集频率定义式求和。"""

    rng = np.random.default_rng(config.seed + 23)
    selected_bins = min(64, config.n_fft // 2)
    sparse_a = np.zeros(config.n_fft, dtype=complex)
    sparse_b = np.zeros(config.n_fft, dtype=complex)
    indices_a = np.sort(rng.choice(config.n_fft, selected_bins, replace=False))
    indices_b = np.sort(rng.choice(config.n_fft, selected_bins, replace=False))
    sparse_a[indices_a] = rng.normal(size=selected_bins) + 1j * rng.normal(
        size=selected_bins
    )
    sparse_b[indices_b] = rng.normal(size=selected_bins) + 1j * rng.normal(
        size=selected_bins
    )
    dense_a = time_zero_padded_dense_spectrum(sparse_a, config.interpolation_factor)
    dense_b = time_zero_padded_dense_spectrum(sparse_b, config.interpolation_factor)
    dense_cross = dense_b * np.conj(dense_a)
    estimate, _, _ = dense_delay_estimates(
        dense_cross,
        np.array([selected_bins]),
        config.true_delay_min_samples,
        config.true_delay_max_samples,
        config.delay_grid_step_samples,
    )
    direct_estimate = direct_dense_delay_estimate(
        dense_cross,
        config.true_delay_min_samples,
        config.true_delay_max_samples,
        config.delay_grid_step_samples,
    )
    return {
        "scope": "implementation_only_not_formal_monte_carlo",
        "input_description": "固定种子生成的双站稀疏频谱经8倍时域补零得到的密集互谱",
        "zoomfft_estimate_samples": float(estimate[0]),
        "direct_matrix_estimate_samples": direct_estimate,
        "absolute_difference_samples": abs(float(estimate[0]) - direct_estimate),
    }


def _run_theory_gate(config: SweepConfig) -> list[dict[str, object]]:
    """分别验证五种伪补全等价性和 TZP-DS 的独立算子门禁。"""

    rng = np.random.default_rng(config.seed + 31)
    diagnostics: list[dict[str, object]] = []
    for selected_bins in config.selected_bins_values:
        indices = np.sort(rng.choice(config.n_fft, selected_bins, replace=False))
        sparse = np.zeros(config.n_fft, dtype=complex)
        sparse[indices] = rng.normal(size=selected_bins) + 1j * rng.normal(
            size=selected_bins
        )
        zero = complete_spectrum(
            sparse,
            indices,
            "zero_fill",
            config.interpolation_factor,
        )
        for method in PSEUDO_COMPLETION_METHODS:
            completed = complete_spectrum(
                sparse,
                indices,
                method,
                config.interpolation_factor,
            )
            maximum_difference = float(
                np.max(np.abs(completed.spectrum - zero.spectrum))
            )
            passed = bool(
                maximum_difference <= 1e-12
                and completed.generated_missing_energy <= 1e-20
            )
            diagnostics.append(
                {
                    "selected_bins": selected_bins,
                    "method": method,
                    "diagnostic_type": "original_grid_pseudo_equivalence",
                    "maximum_spectrum_difference_vs_zero": maximum_difference,
                    "generated_missing_energy": completed.generated_missing_energy,
                    "observed_consistency_error": completed.observed_consistency_error,
                    "operator_rank": np.nan,
                    "expected_rank": np.nan,
                    "operator_gram_error": np.nan,
                    "direct_dtft_error": np.nan,
                    "passed": passed,
                }
            )
        dense = time_zero_padded_dense_spectrum(sparse, config.interpolation_factor)
        mapped = sample_original_grid(
            dense,
            config.n_fft,
            config.interpolation_factor,
        )
        missing = np.ones(config.n_fft, dtype=bool)
        missing[indices] = False
        observed_error = float(
            np.linalg.norm(mapped[indices] - sparse[indices])
            / max(np.linalg.norm(sparse[indices]), np.finfo(float).tiny)
        )
        missing_energy = float(np.sum(np.abs(mapped[missing]) ** 2))

        basis = np.zeros((selected_bins, config.n_fft), dtype=complex)
        basis[np.arange(selected_bins), indices] = 1.0
        dense_basis = time_zero_padded_dense_spectrum(
            basis,
            config.interpolation_factor,
        )
        operator = dense_basis.T
        gram = operator.conj().T @ operator
        expected_gram = config.interpolation_factor * np.eye(selected_bins)
        gram_error = float(np.max(np.abs(gram - expected_gram)))
        operator_rank = int(np.linalg.matrix_rank(gram, tol=1e-10))

        dense_length = dense.size
        sampled_dense_indices = np.sort(
            rng.choice(dense_length, min(32, dense_length), replace=False)
        )
        time_signal = np.fft.ifft(
            np.fft.ifftshift(sparse),
            norm="ortho",
        )
        dense_frequencies = np.fft.fftshift(np.fft.fftfreq(dense_length))
        direct_dense = (
            np.sqrt(float(config.interpolation_factor) / dense_length)
            * np.exp(
                -2j
                * np.pi
                * np.outer(
                    dense_frequencies[sampled_dense_indices],
                    np.arange(config.n_fft),
                )
            )
            @ time_signal
        )
        direct_error = float(
            np.max(np.abs(dense[sampled_dense_indices] - direct_dense))
        )
        tzp_passed = bool(
            observed_error <= 1e-12
            and missing_energy <= 1e-20
            and operator_rank == selected_bins
            and gram_error <= 1e-10
            and direct_error <= 1e-11
        )
        diagnostics.append(
            {
                "selected_bins": selected_bins,
                "method": TZP_DENSE_METHOD,
                "diagnostic_type": "time_zero_padded_dense_operator",
                "maximum_spectrum_difference_vs_zero": np.nan,
                "generated_missing_energy": missing_energy,
                "observed_consistency_error": observed_error,
                "operator_rank": operator_rank,
                "expected_rank": selected_bins,
                "operator_gram_error": gram_error,
                "direct_dtft_error": direct_error,
                "passed": tzp_passed,
            }
        )
    if not all(bool(row["passed"]) for row in diagnostics):
        raise RuntimeError("理论门禁失败：伪补全或 TZP-DS 算子不满足冻结条件")
    return diagnostics


def _bootstrap_paired_comparisons(
    output_directory: Path,
    conditions: list[Condition],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 991)
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        if method == "Intersection-Continuous":
            continue
        arrays: list[np.ndarray] = []
        for condition in conditions:
            paired_path = (
                output_directory
                / "condition_chunks"
                / f"{condition.identifier}.paired.npz"
            )
            with np.load(paired_path) as data:
                arrays.append(np.asarray(data[method], dtype=float))
        improvements = np.concatenate(arrays) if arrays else np.empty(0)
        if improvements.size == 0:
            continue
        bootstrap = np.empty(repetitions, dtype=float)
        for repetition in range(repetitions):
            sample = rng.integers(0, improvements.size, improvements.size)
            bootstrap[repetition] = float(np.mean(improvements[sample]))
        rows.append(
            {
                "method": method,
                "baseline": "Intersection-Continuous",
                "paired_trials": int(improvements.size),
                "mean_mae_improvement_samples": float(np.mean(improvements)),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "maximum_abs_improvement_samples": float(
                    np.max(np.abs(improvements), initial=0.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _finalise_results(
    config: SweepConfig,
    conditions: list[Condition],
    output_directory: Path,
    diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    summary_records: list[dict[str, object]] = []
    metadata_records: list[dict[str, object]] = []
    for condition in conditions:
        prefix = output_directory / "condition_chunks" / condition.identifier
        summary_records.extend(
            json.loads(
                prefix.with_suffix(".summary.json").read_text(encoding="utf-8")
            )
        )
        metadata_records.append(
            json.loads(prefix.with_suffix(".done.json").read_text(encoding="utf-8"))
        )
    summary = pd.DataFrame(summary_records)
    method_categories = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = summary.assign(method=method_categories).sort_values(
        [
            "signal_kind",
            "selected_bins",
            "snr_db",
            "overlap_fraction",
            "method",
        ]
    )
    summary["method"] = summary["method"].astype(str)
    summary.to_csv(output_directory / "summary.csv", index=False, encoding="utf-8-sig")
    paired = _bootstrap_paired_comparisons(
        output_directory,
        conditions,
        config.bootstrap_repetitions,
        config.seed,
    )
    paired.to_csv(
        output_directory / "paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(diagnostics).to_csv(
        output_directory / "operator_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    condition_metadata = pd.json_normalize(metadata_records)
    condition_metadata.to_csv(
        output_directory / "condition_runtime.csv",
        index=False,
        encoding="utf-8-sig",
    )
    maximum_delay_difference = max(
        float(record["maximum_pseudo_delay_difference_samples"])
        for record in metadata_records
    )
    maximum_error_difference = max(
        float(record["maximum_pseudo_absolute_error_difference_samples"])
        for record in metadata_records
    )
    final_manifest = {
        "completed": True,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "conditions": len(conditions),
        "base_trials": len(conditions) * config.trials_per_condition,
        "method_rows": (
            len(conditions) * config.trials_per_condition * len(METHOD_ORDER)
        ),
        "expected_method_rows": (
            len(config.signal_kinds)
            * len(config.selected_bins_values)
            * len(config.snr_values_db)
            * len(config.overlap_values)
            * config.trials_per_condition
            * len(METHOD_ORDER)
        ),
        "maximum_pseudo_delay_difference_samples": maximum_delay_difference,
        "maximum_pseudo_absolute_error_difference_samples": maximum_error_difference,
        "all_operator_gates_passed": all(bool(row["passed"]) for row in diagnostics),
        "summary_rows": int(len(summary)),
        "paired_rows": int(len(paired)),
        "signal_kind_labels": SIGNAL_KIND_LABELS,
    }
    _atomic_write_json(output_directory / "COMPLETED.json", final_manifest)
    return final_manifest


def _prepare_output_directory(
    config: SweepConfig,
    resume_directory: Path | None,
) -> tuple[Path, list[dict[str, object]]]:
    if resume_directory is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_directory = OUTPUT_ROOT / f"{timestamp}_{config.profile}"
        output_directory.mkdir(parents=True, exist_ok=False)
        (output_directory / "condition_chunks").mkdir()
        _atomic_write_json(output_directory / "config.json", config.to_dict())
        frozen = _freeze_old_result_manifest()
        _atomic_write_json(output_directory / "frozen_previous_result.json", frozen)
        diagnostics = _run_theory_gate(config)
        estimator_validation = _validate_fft_estimator(config)
        tzp_estimator_validation = _validate_tzp_dense_estimator(config)
        protocol = {
            "route": "standalone_frequency_alignment_bpsk_parameter_sweep",
            "independent_from_main": True,
            "does_not_modify_previous_code_or_results": True,
            "previous_result_frozen_by_sha256": True,
            "source_model": {
                "four_existing_spectral_families": True,
                "urban8_bpsk_rrc": {
                    "samples_per_symbol": config.bpsk_samples_per_symbol,
                    "rrc_rolloff": config.bpsk_rrc_rolloff,
                    "rrc_span_symbols": config.bpsk_rrc_span_symbols,
                    "scope": (
                        "复现 urban8 的 BPSK、每符号2点和 RRC 参数；"
                        "不引入 urban8 几何或多径，以保持与四类单时延机制信号可比"
                    ),
                },
            },
            "delay_model": (
                "B站频谱乘 exp(-j*2*pi*f*tau)，tau 在 [-25,25] sample 均匀采样"
            ),
            "selection_model": (
                "A/B 采用受控随机集合以固定重叠率；Zhai-Asymmetric 单独按 B 站"
                " noisy 周期图运行学位论文严格0-1逐点交换"
            ),
            "analytical_reuse": (
                "五种伪补全映回原 DFT 网格后与真实交集估计完全等价；"
                "通过逐 M 数值门禁后复用交集估计，方法行仍完整保存"
            ),
            "time_zero_padded_dense_spectrum": {
                "method": TZP_DENSE_METHOD,
                "factor": config.interpolation_factor,
                "dense_points": config.n_fft * config.interpolation_factor,
                "estimator": "ZoomFFT continuous-delay score on the dense spectrum",
                "analytical_reuse": False,
                "interpretation": (
                    "只加密频率数值栅格；原始未选 DFT 频点仍为零，"
                    "密集频点具有相关性且不计为新增独立观测"
                ),
            },
            "fft_estimator_validation": estimator_validation,
            "tzp_dense_estimator_validation": tzp_estimator_validation,
            "trial_storage": "condition_chunks/*.csv.gz",
        }
        _atomic_write_json(output_directory / "protocol_manifest.json", protocol)
        pd.DataFrame(diagnostics).to_csv(
            output_directory / "operator_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _atomic_write_text(
            output_directory / "run.log",
            (
                f"started_at={datetime.now().isoformat(timespec='seconds')}\n"
                f"profile={config.profile}\n"
                "status=running\n"
            ),
        )
    else:
        output_directory = resume_directory.resolve()
        stored = json.loads(
            (output_directory / "config.json").read_text(encoding="utf-8")
        )
        normalized_config = json.loads(json.dumps(config.to_dict(), ensure_ascii=False))
        if stored != normalized_config:
            raise ValueError("恢复目录的 config.json 与当前冻结协议不一致")
        diagnostics = pd.read_csv(
            output_directory / "operator_diagnostics.csv"
        ).to_dict(orient="records")
    return output_directory, diagnostics


def _completed_condition_ids(
    output_directory: Path,
    conditions: list[Condition],
) -> set[str]:
    completed: set[str] = set()
    for condition in conditions:
        prefix = output_directory / "condition_chunks" / condition.identifier
        required = (
            prefix.with_suffix(".csv.gz"),
            prefix.with_suffix(".summary.json"),
            prefix.with_suffix(".paired.npz"),
            prefix.with_suffix(".done.json"),
        )
        if all(path.is_file() for path in required):
            completed.add(condition.identifier)
    return completed


def run_bpsk_parameter_sweep(
    config: SweepConfig,
    resume_directory: Path | None,
    workers: int,
) -> Path:
    output_directory, diagnostics = _prepare_output_directory(
        config,
        resume_directory,
    )
    conditions = _build_conditions(config)
    completed_ids = _completed_condition_ids(output_directory, conditions)
    pending = [
        condition for condition in conditions if condition.identifier not in completed_ids
    ]
    started = time.perf_counter()
    completed_count = len(completed_ids)
    print(f"RESULT_DIR={output_directory}", flush=True)
    print(
        f"CONDITIONS total={len(conditions)} completed={completed_count} "
        f"pending={len(pending)} workers={workers}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_condition,
                    config.to_dict(),
                    asdict(condition),
                    str(output_directory),
                ): condition
                for condition in pending
            }
            for future in as_completed(futures):
                condition = futures[future]
                metadata = future.result()
                completed_count += 1
                elapsed = time.perf_counter() - started
                processed_now = completed_count - len(completed_ids)
                seconds_per_condition = elapsed / max(processed_now, 1)
                eta_seconds = seconds_per_condition * (len(conditions) - completed_count)
                line = (
                    f"[{completed_count}/{len(conditions)}] "
                    f"{condition.identifier} "
                    f"condition_s={float(metadata['runtime_seconds']):.2f} "
                    f"eta_min={eta_seconds / 60.0:.1f}"
                )
                print(line, flush=True)
                with (output_directory / "run.log").open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(line + "\n")
                status = {
                    "status": "running",
                    "completed_conditions": completed_count,
                    "total_conditions": len(conditions),
                    "elapsed_seconds_this_session": elapsed,
                    "estimated_remaining_seconds": eta_seconds,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                _atomic_write_json(output_directory / "status.json", status)
    final_manifest = _finalise_results(
        config,
        conditions,
        output_directory,
        diagnostics,
    )
    with (output_directory / "run.log").open("a", encoding="utf-8") as handle:
        handle.write("status=completed\n")
        handle.write(json.dumps(final_manifest, ensure_ascii=False) + "\n")
    _atomic_write_json(
        output_directory / "status.json",
        {
            "status": "completed",
            **final_manifest,
        },
    )
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2), flush=True)
    return output_directory


def _smoke_config() -> SweepConfig:
    return SweepConfig(
        profile="bpsk_parameter_sweep_smoke",
        selected_bins_values=(32,),
        snr_values_db=(5.0,),
        overlap_values=(0.5,),
        signal_kinds=("smooth_complex", "urban8_bpsk_rrc"),
        trials_per_condition=3,
        bootstrap_repetitions=100,
        estimator_batch_size=3,
    )




def _parse_bpsk_parameter_sweep_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BPSK 与 TZP-DS 扩展参数扫描。")
    parser.add_argument(
        "--resume",
        type=Path,
        help="恢复既有未完成结果目录；省略时创建新目录",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(2, (os.cpu_count() or 2) // 2)),
        help="并行条件数；默认最多 2，避免 FFT 批处理占用过多内存",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="运行两个信号、单一 M/SNR/重叠率的 3 次轻量验证",
    )
    return parser.parse_args()


def main_bpsk_parameter_sweep() -> None:
    """兼容旧 ``run_bpsk_parameter_sweep.py`` 的命令行入口。"""

    arguments = _parse_bpsk_parameter_sweep_arguments()
    if arguments.workers < 1:
        raise ValueError("--workers 必须至少为 1")
    config = _smoke_config() if arguments.smoke else SweepConfig()
    run_bpsk_parameter_sweep(config, arguments.resume, arguments.workers)

def main() -> None:
    if PROFILE == "bpsk_parameter_sweep":
        output_directory = run_bpsk_parameter_sweep(
            SweepConfig(),
            BPSK_SWEEP_RESUME_DIRECTORY,
            BPSK_SWEEP_WORKERS,
        )
        print(f"BPSK/TZP-DS 扩展验证完成：{output_directory}")
        return
    if PROFILE == "bpsk_parameter_sweep_smoke":
        output_directory = run_bpsk_parameter_sweep(
            _smoke_config(),
            BPSK_SWEEP_RESUME_DIRECTORY,
            BPSK_SWEEP_WORKERS,
        )
        print(f"BPSK/TZP-DS 冒烟验证完成：{output_directory}")
        return
    config = get_profile(PROFILE)
    output_directory = run_experiment(config)
    print(f"独立验证完成：{output_directory}")


if __name__ == "__main__":
    main()
