"""双站独立选频、频谱伪补全和 TDOA 对齐的独立运行入口。"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = MODULE_DIRECTORY.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from alignment_core import (  # noqa: E402
    complete_spectrum,
    complex_correlation,
    complex_nmse_db,
    estimate_continuous_delay,
    interpolation_operator_rank,
    phase_rmse_rad,
    shifted_frequencies,
)
from experiment_config import ExperimentConfig, get_profile  # noqa: E402
from plot_alignment_results import save_summary_figures  # noqa: E402
from zhai_selection import select_zhai_thesis  # noqa: E402

# 在 PyCharm 中只需修改这一行。正式大样本请使用 synthetic_full。
PROFILE = "synthetic_full"

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


def main() -> None:
    config = get_profile(PROFILE)
    output_directory = run_experiment(config)
    print(f"独立验证完成：{output_directory}")


if __name__ == "__main__":
    main()
