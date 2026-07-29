"""只读回溯既有实验的研究门禁、效应量和阈值敏感性。

本脚本不训练模型、不修改历史结果目录，也不重新选择方法。所有输出写入新的
``运行结果/研究门禁审计/GATE_AUDIT_<timestamp>_readonly`` 目录，并在结束前复核输入哈希。
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest, norm


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "运行结果"
BRSR_SOURCE = (
    RESULT_ROOT / "BRSR" / "BRSR_STAGE_B_20260718_173940_development"
)
PASR_SOURCE = RESULT_ROOT / "PASR" / "PASR_20260716_193437_research"
SHARED64_SOURCE = (
    RESULT_ROOT
    / "压缩域TDOA_V5"
    / "20260711_154042_V5B_Shared64严格CR16"
    / "tables"
    / "topk_fairness"
)
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 20260723


@dataclass(frozen=True)
class PairSummary:
    n_rows: int
    n_clusters: int
    target_rmse: float
    baseline_rmse: float
    rmse_gain: float
    rmse_gain_ci_low: float
    rmse_gain_ci_high: float
    mse_gain: float
    mse_gain_ci_low: float
    mse_gain_ci_high: float
    target_mean_abs: float
    baseline_mean_abs: float
    mean_abs_gain: float
    mean_abs_gain_ci_low: float
    mean_abs_gain_ci_high: float
    paired_cluster_sd_mse_gain: float
    standardized_paired_effect: float
    mde80_mse_gain: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sources() -> dict[Path, str]:
    required = [
        BRSR_SOURCE / "stage_b_trials.csv",
        BRSR_SOURCE / "development_gate.json",
        BRSR_SOURCE / "manifest.json",
        PASR_SOURCE / "pasr_trial_results.csv.gz",
        PASR_SOURCE / "pasr_go_no_go.json",
        PASR_SOURCE / "pasr_manifest.json",
        SHARED64_SOURCE / "fair_native_dev_multi_pool_rows.csv",
        SHARED64_SOURCE / "fair_native_dev_multi_pool_manifest.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen audit inputs: {missing}")
    return {path: _sha256(path) for path in required}


def _cluster_arrays(
    rows: pd.DataFrame,
    cluster_col: str,
    target_sq_col: str,
    baseline_sq_col: str,
    target_abs_col: str,
    baseline_abs_col: str,
) -> tuple[np.ndarray, ...]:
    grouped = rows.groupby(cluster_col, sort=True)
    sums = grouped[
        [target_sq_col, baseline_sq_col, target_abs_col, baseline_abs_col]
    ].sum()
    counts = grouped.size().astype(float)
    return (
        sums[target_sq_col].to_numpy(dtype=float),
        sums[baseline_sq_col].to_numpy(dtype=float),
        sums[target_abs_col].to_numpy(dtype=float),
        sums[baseline_abs_col].to_numpy(dtype=float),
        counts.to_numpy(dtype=float),
    )


def _bootstrap_pair_summary(
    rows: pd.DataFrame,
    *,
    cluster_col: str,
    target_sq_col: str,
    baseline_sq_col: str,
    target_abs_col: str,
    baseline_abs_col: str,
    repetitions: int,
    seed: int,
) -> PairSummary:
    if rows.empty:
        raise ValueError("paired summary requires non-empty rows")
    required = {
        cluster_col,
        target_sq_col,
        baseline_sq_col,
        target_abs_col,
        baseline_abs_col,
    }
    if not required.issubset(rows.columns):
        raise ValueError(f"paired summary missing columns: {required - set(rows.columns)}")
    numeric = rows[list(required - {cluster_col})].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("paired summary received non-finite values")

    target_sq, baseline_sq, target_abs, baseline_abs, counts = _cluster_arrays(
        rows,
        cluster_col,
        target_sq_col,
        baseline_sq_col,
        target_abs_col,
        baseline_abs_col,
    )
    n_clusters = len(counts)
    if n_clusters < 2:
        raise ValueError("paired summary requires at least two independent clusters")

    total_n = float(counts.sum())
    target_mse = float(target_sq.sum() / total_n)
    baseline_mse = float(baseline_sq.sum() / total_n)
    target_mae = float(target_abs.sum() / total_n)
    baseline_mae = float(baseline_abs.sum() / total_n)

    rng = np.random.default_rng(seed)
    rmse_boot = np.empty(repetitions, dtype=float)
    mse_boot = np.empty(repetitions, dtype=float)
    mae_boot = np.empty(repetitions, dtype=float)
    batch_size = 256
    offset = 0
    while offset < repetitions:
        size = min(batch_size, repetitions - offset)
        indices = rng.integers(0, n_clusters, size=(size, n_clusters))
        n = counts[indices].sum(axis=1)
        target_mse_b = target_sq[indices].sum(axis=1) / n
        baseline_mse_b = baseline_sq[indices].sum(axis=1) / n
        target_mae_b = target_abs[indices].sum(axis=1) / n
        baseline_mae_b = baseline_abs[indices].sum(axis=1) / n
        rmse_boot[offset : offset + size] = np.sqrt(baseline_mse_b) - np.sqrt(
            target_mse_b
        )
        mse_boot[offset : offset + size] = baseline_mse_b - target_mse_b
        mae_boot[offset : offset + size] = baseline_mae_b - target_mae_b
        offset += size

    cluster_mse_gain = (baseline_sq - target_sq) / counts
    cluster_sd = float(np.std(cluster_mse_gain, ddof=1))
    mean_cluster_gain = float(np.mean(cluster_mse_gain))
    standardized = (
        mean_cluster_gain / cluster_sd if cluster_sd > 0.0 else math.nan
    )
    mde80 = float(
        (norm.ppf(0.975) + norm.ppf(0.80)) * cluster_sd / math.sqrt(n_clusters)
    )
    return PairSummary(
        n_rows=len(rows),
        n_clusters=n_clusters,
        target_rmse=math.sqrt(target_mse),
        baseline_rmse=math.sqrt(baseline_mse),
        rmse_gain=math.sqrt(baseline_mse) - math.sqrt(target_mse),
        rmse_gain_ci_low=float(np.quantile(rmse_boot, 0.025)),
        rmse_gain_ci_high=float(np.quantile(rmse_boot, 0.975)),
        mse_gain=baseline_mse - target_mse,
        mse_gain_ci_low=float(np.quantile(mse_boot, 0.025)),
        mse_gain_ci_high=float(np.quantile(mse_boot, 0.975)),
        target_mean_abs=target_mae,
        baseline_mean_abs=baseline_mae,
        mean_abs_gain=baseline_mae - target_mae,
        mean_abs_gain_ci_low=float(np.quantile(mae_boot, 0.025)),
        mean_abs_gain_ci_high=float(np.quantile(mae_boot, 0.975)),
        paired_cluster_sd_mse_gain=cluster_sd,
        standardized_paired_effect=standardized,
        mde80_mse_gain=mde80,
    )


def _summary_row(
    route: str,
    target: str,
    baseline: str,
    unit: str,
    bucket: str,
    summary: PairSummary,
) -> dict[str, object]:
    return {
        "route": route,
        "target": target,
        "baseline": baseline,
        "bucket": bucket,
        "unit": unit,
        **summary.__dict__,
    }


def _wilson_coverage(values: pd.Series, nominal: float = 0.90) -> dict[str, float]:
    successes = int(values.astype(bool).sum())
    total = int(values.notna().sum())
    if total <= 0:
        return {
            "n": 0,
            "coverage": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "nominal_inside_ci": False,
        }
    interval = binomtest(successes, total).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    return {
        "n": total,
        "coverage": successes / total,
        "ci_low": float(interval.low),
        "ci_high": float(interval.high),
        "nominal_inside_ci": bool(interval.low <= nominal <= interval.high),
    }


def _mcnemar_severe(
    target_abs: np.ndarray,
    baseline_abs: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    target_bad = np.asarray(target_abs, dtype=float) > threshold
    baseline_bad = np.asarray(baseline_abs, dtype=float) > threshold
    target_only = int(np.sum(target_bad & ~baseline_bad))
    baseline_only = int(np.sum(~target_bad & baseline_bad))
    discordant = target_only + baseline_only
    p_value = (
        float(binomtest(min(target_only, baseline_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "threshold": threshold,
        "n": len(target_bad),
        "target_rate": float(target_bad.mean()),
        "baseline_rate": float(baseline_bad.mean()),
        "target_only_bad": target_only,
        "baseline_only_bad": baseline_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def _merge_brsr(
    rows: pd.DataFrame, target: str, baseline: str
) -> pd.DataFrame:
    keys = ["seed", "trial", "snr_db"]
    columns = keys + [
        "squared_error_samples2",
        "abs_error_samples",
        "credible90_contains_true",
        "total_bits",
    ]
    left = rows.loc[rows["method"].eq(target), columns].rename(
        columns={
            "squared_error_samples2": "target_sq",
            "abs_error_samples": "target_abs",
            "credible90_contains_true": "target_covered",
            "total_bits": "target_bits",
        }
    )
    right = rows.loc[rows["method"].eq(baseline), columns].rename(
        columns={
            "squared_error_samples2": "baseline_sq",
            "abs_error_samples": "baseline_abs",
            "credible90_contains_true": "baseline_covered",
            "total_bits": "baseline_bits",
        }
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    paired["cluster"] = paired["seed"].astype(str) + ":" + paired["trial"].astype(str)
    return paired


def _audit_brsr(
    effect_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
) -> dict[str, object]:
    rows = pd.read_csv(BRSR_SOURCE / "stage_b_trials.csv")
    rows = rows.loc[rows["stage"].eq("evaluation") & ~rows["is_dp_subset"].astype(bool)]
    baseline = "SNR-only-T32"
    route_result: dict[str, object] = {}
    for index, target in enumerate(("BRSR-Rollout-T32", "BRSR-Greedy-T32")):
        paired = _merge_brsr(rows, target, baseline)
        summary = _bootstrap_pair_summary(
            paired,
            cluster_col="cluster",
            target_sq_col="target_sq",
            baseline_sq_col="baseline_sq",
            target_abs_col="target_abs",
            baseline_abs_col="baseline_abs",
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED + index,
        )
        effect_rows.append(
            _summary_row("BRSR-StageB-v1", target, baseline, "sample", "all", summary)
        )

        by_snr = paired.groupby("snr_db").agg(
            target_mse=("target_sq", "mean"),
            baseline_mse=("baseline_sq", "mean"),
        )
        by_snr["relative_degradation"] = (
            np.sqrt(by_snr["target_mse"]) / np.sqrt(by_snr["baseline_mse"]) - 1.0
        )
        worst_degradation = float(by_snr["relative_degradation"].max())
        relative_gain = summary.rmse_gain / summary.baseline_rmse
        stat_pass = summary.mse_gain_ci_low > 0.0
        for minimum_gain in (0.00, 0.02, 0.05, 0.10):
            for maximum_degradation in (0.05, 0.10, 0.20, math.inf):
                effect_pass = relative_gain >= minimum_gain
                robust_pass = worst_degradation <= maximum_degradation
                sensitivity_rows.append(
                    {
                        "route": "BRSR-StageB-v1",
                        "target": target,
                        "baseline": baseline,
                        "minimum_relative_rmse_gain": minimum_gain,
                        "maximum_per_snr_degradation": maximum_degradation,
                        "observed_relative_rmse_gain": relative_gain,
                        "observed_worst_per_snr_degradation": worst_degradation,
                        "effect_pass": effect_pass,
                        "robustness_pass": robust_pass,
                        "statistical_pass": stat_pass,
                        "full_pass": effect_pass and robust_pass and stat_pass,
                    }
                )

        target_rows = rows.loc[rows["method"].eq(target)]
        overall_coverage = _wilson_coverage(
            target_rows["credible90_contains_true"]
        )
        calibration_rows.append(
            {
                "route": "BRSR-StageB-v1",
                "method": target,
                "bucket": "all",
                **overall_coverage,
            }
        )
        for snr, group in target_rows.groupby("snr_db"):
            calibration_rows.append(
                {
                    "route": "BRSR-StageB-v1",
                    "method": target,
                    "bucket": f"{snr:g}dB",
                    **_wilson_coverage(group["credible90_contains_true"]),
                }
            )

        high = paired.loc[paired["snr_db"] >= 10.0]
        event_rows.append(
            {
                "route": "BRSR-StageB-v1",
                "target": target,
                "baseline": baseline,
                "bucket": "SNR>=10dB",
                **_mcnemar_severe(
                    high["target_abs"].to_numpy(),
                    high["baseline_abs"].to_numpy(),
                    2.0,
                ),
            }
        )
        route_result[target] = {
            "relative_rmse_gain": relative_gain,
            "worst_per_snr_degradation": worst_degradation,
            "mse_gain_ci_low": summary.mse_gain_ci_low,
            "mse_gain_ci_high": summary.mse_gain_ci_high,
        }
    return route_result


def _pasr_static_pair(rows: pd.DataFrame, gate: dict[str, object]) -> pd.DataFrame:
    outcome = next(
        item
        for item in gate["qbits_outcomes"]
        if item["analysis_role"] == "primary_pre_registered_test"
    )
    qbits = int(outcome["qbits"])
    plan = outcome["same_bit_static_plan"]
    keys = ["seed", "snr_db", "trial"]

    def select(method: str, bits: int, prefix: str) -> pd.DataFrame:
        selected = rows.loc[
            rows["stage"].eq("locked")
            & rows["method"].eq(method)
            & rows["qbits"].eq(bits),
            keys + ["squared_error_samples2", "abs_error_samples"],
        ].copy()
        return selected.rename(
            columns={
                "squared_error_samples2": f"{prefix}_sq",
                "abs_error_samples": f"{prefix}_abs",
            }
        )

    target = select("PASR-Observed", qbits, "target")
    left = select(str(plan["left_method"]), int(plan["left_qbits"]), "left")
    right = select(str(plan["right_method"]), int(plan["right_qbits"]), "right")
    paired = target.merge(left, on=keys, validate="one_to_one").merge(
        right, on=keys, validate="one_to_one"
    )
    weight = float(plan["right_weight"])
    paired["baseline_sq"] = (
        (1.0 - weight) * paired["left_sq"] + weight * paired["right_sq"]
    )
    paired["baseline_abs"] = (
        (1.0 - weight) * paired["left_abs"] + weight * paired["right_abs"]
    )
    paired["cluster"] = paired["seed"].astype(str) + ":" + paired["trial"].astype(str)
    paired.attrs["outcome"] = outcome
    return paired


def _audit_pasr(
    effect_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
) -> dict[str, object]:
    rows = pd.read_csv(PASR_SOURCE / "pasr_trial_results.csv.gz")
    gate = json.loads((PASR_SOURCE / "pasr_go_no_go.json").read_text(encoding="utf-8"))
    paired = _pasr_static_pair(rows, gate)
    outcome = paired.attrs["outcome"]
    summary = _bootstrap_pair_summary(
        paired,
        cluster_col="cluster",
        target_sq_col="target_sq",
        baseline_sq_col="baseline_sq",
        target_abs_col="target_abs",
        baseline_abs_col="baseline_abs",
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED + 100,
    )
    effect_rows.append(
        _summary_row(
            "PASR-v1",
            "PASR-Observed-q6",
            "DevelopmentFrozen-SameBitStatic",
            "sample",
            "all",
            summary,
        )
    )
    static_gross = float(
        outcome["same_bit_static_locked_evaluation"]["gross_error_rate_gt_1"]
    )
    gross_gain = float(outcome["gross_error_relative_improvement"])
    high_ratio = float(outcome["high_snr_rmse_ratio_vs_frozen_static"])
    relative_gain = summary.rmse_gain / summary.baseline_rmse
    stat_pass = summary.mse_gain_ci_low > 0.0
    for minimum_gain in (0.00, 0.02, 0.05, 0.10):
        for gross_minimum in (0.00, 0.05, 0.15):
            for maximum_high_degradation in (0.05, 0.10, 0.20, math.inf):
                effect_pass = relative_gain >= minimum_gain
                gross_pass = gross_gain >= gross_minimum
                high_pass = high_ratio - 1.0 <= maximum_high_degradation
                sensitivity_rows.append(
                    {
                        "route": "PASR-v1",
                        "target": "PASR-Observed-q6",
                        "baseline": "DevelopmentFrozen-SameBitStatic",
                        "minimum_relative_rmse_gain": minimum_gain,
                        "minimum_gross_error_improvement": gross_minimum,
                        "maximum_high_snr_degradation": maximum_high_degradation,
                        "observed_relative_rmse_gain": relative_gain,
                        "observed_gross_error_improvement": gross_gain,
                        "observed_high_snr_degradation": high_ratio - 1.0,
                        "effect_pass": effect_pass,
                        "gross_pass": gross_pass,
                        "high_snr_pass": high_pass,
                        "statistical_pass": stat_pass,
                        "full_pass": effect_pass
                        and gross_pass
                        and high_pass
                        and stat_pass,
                    }
                )
    target_gross = static_gross * (1.0 - gross_gain)
    event_rows.append(
        {
            "route": "PASR-v1",
            "target": "PASR-Observed-q6",
            "baseline": "DevelopmentFrozen-SameBitStatic",
            "bucket": "all",
            "threshold": 1.0,
            "n": int(outcome["same_bit_static_locked_evaluation"]["n"]),
            "target_rate": target_gross,
            "baseline_rate": static_gross,
            "target_only_bad": None,
            "baseline_only_bad": None,
            "discordant": None,
            "exact_two_sided_p": None,
            "note": "time-sharing comparator is an expected mixture; paired event cells are unavailable",
        }
    )
    return {
        "relative_rmse_gain": relative_gain,
        "gross_error_relative_improvement": gross_gain,
        "high_snr_degradation": high_ratio - 1.0,
        "mse_gain_ci_low": summary.mse_gain_ci_low,
        "mse_gain_ci_high": summary.mse_gain_ci_high,
    }


def _snr_bucket(snr: float) -> str:
    if snr <= 0.0:
        return "low"
    if snr <= 6.0:
        return "mid"
    return "high"


def _merge_shared64(
    rows: pd.DataFrame, target: str, baseline: str, bucket: str
) -> pd.DataFrame:
    subset = rows.copy()
    subset["bucket"] = subset["SNR_dB"].map(_snr_bucket)
    if bucket != "all":
        subset = subset.loc[subset["bucket"].eq(bucket)]
    keys = [
        "pool_seed",
        "eval_seed",
        "SNR_dB",
        "trial_idx",
        "snapshot_id",
        "cluster_id",
    ]
    values = keys + [
        "error_m",
        "solver_success",
        "tdoa_abs_error_mean_samples",
        "elapsed_s",
    ]
    left = subset.loc[subset["method"].eq(target), values].rename(
        columns={
            "error_m": "target_error",
            "solver_success": "target_success",
            "tdoa_abs_error_mean_samples": "target_tdoa",
            "elapsed_s": "target_elapsed",
        }
    )
    right = subset.loc[subset["method"].eq(baseline), values].rename(
        columns={
            "error_m": "baseline_error",
            "solver_success": "baseline_success",
            "tdoa_abs_error_mean_samples": "baseline_tdoa",
            "elapsed_s": "baseline_elapsed",
        }
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    paired = paired.loc[paired["target_success"] & paired["baseline_success"]].copy()
    paired["target_sq"] = paired["target_error"] ** 2
    paired["baseline_sq"] = paired["baseline_error"] ** 2
    paired["target_abs"] = paired["target_tdoa"]
    paired["baseline_abs"] = paired["baseline_tdoa"]
    return paired


def _audit_shared64(
    effect_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
) -> dict[str, object]:
    rows = pd.read_csv(SHARED64_SOURCE / "fair_native_dev_multi_pool_rows.csv")
    manifest = json.loads(
        (SHARED64_SOURCE / "fair_native_dev_multi_pool_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not manifest["v5b_feature_budget"]["strict_cr16_budget_passed"]:
        raise RuntimeError("Shared64 source does not pass the strict CR16 budget")
    target = "V5B-Shared64"
    baselines = (
        "V5A1-GeoShared52",
        "V5A1-PowerShared52",
        "V5A1-Power64",
        "DAE-CR16",
        "DFT-train-power",
        "Cao2017-DFT-AML",
        "Zhai-CRLB-Decimation",
    )
    route_result: dict[str, object] = {}
    for baseline_index, baseline in enumerate(baselines):
        for bucket_index, bucket in enumerate(("all", "low", "mid", "high")):
            paired = _merge_shared64(rows, target, baseline, bucket)
            summary = _bootstrap_pair_summary(
                paired,
                cluster_col="cluster_id",
                target_sq_col="target_sq",
                baseline_sq_col="baseline_sq",
                target_abs_col="target_abs",
                baseline_abs_col="baseline_abs",
                repetitions=BOOTSTRAP_REPETITIONS,
                seed=BOOTSTRAP_SEED + 200 + 10 * baseline_index + bucket_index,
            )
            effect_rows.append(
                _summary_row(
                    "V5B-Shared64",
                    target,
                    baseline,
                    "localization_m_and_tdoa_sample",
                    bucket,
                    summary,
                )
            )
            if bucket == "all":
                relative_gain = summary.rmse_gain / summary.baseline_rmse
                stat_pass = summary.mse_gain_ci_low > 0.0
                for minimum_gain in (0.00, 0.02, 0.05, 0.10):
                    sensitivity_rows.append(
                        {
                            "route": "V5B-Shared64",
                            "target": target,
                            "baseline": baseline,
                            "minimum_relative_rmse_gain": minimum_gain,
                            "observed_relative_rmse_gain": relative_gain,
                            "effect_pass": relative_gain >= minimum_gain,
                            "statistical_pass": stat_pass,
                            "full_pass": relative_gain >= minimum_gain and stat_pass,
                        }
                    )
                route_result[baseline] = {
                    "relative_rmse_gain": relative_gain,
                    "rmse_gain_ci_low": summary.rmse_gain_ci_low,
                    "rmse_gain_ci_high": summary.rmse_gain_ci_high,
                    "tdoa_mae_gain": summary.mean_abs_gain,
                    "tdoa_mae_gain_ci_low": summary.mean_abs_gain_ci_low,
                    "tdoa_mae_gain_ci_high": summary.mean_abs_gain_ci_high,
                    "target_mean_elapsed_s": float(paired["target_elapsed"].mean()),
                    "baseline_mean_elapsed_s": float(
                        paired["baseline_elapsed"].mean()
                    ),
                }
    return route_result


def _route_classification(
    brsr: dict[str, object],
    pasr: dict[str, object],
    shared64: dict[str, object],
) -> pd.DataFrame:
    rollout = brsr["BRSR-Rollout-T32"]
    v5_geo = shared64["V5A1-GeoShared52"]
    v5_power = shared64["V5A1-PowerShared52"]
    v5_legacy = shared64["V5A1-Power64"]
    return pd.DataFrame(
        [
            {
                "route": "BRSR Stage A",
                "previous_status": "STAGE_A_MECHANISM_PASS",
                "audited_status": "GO",
                "scope": "probability, quantization, ledger, and decision mechanics only",
                "reason": "mechanism invariants are not practical-effect thresholds",
                "rerun_needed": False,
            },
            {
                "route": "BRSR Stage B v1 configuration",
                "previous_status": "DEVELOPMENT_NO_GO",
                "audited_status": "NO-GO-CONFIG",
                "scope": "current likelihood, candidate pool, T32 policy, and dataset",
                "reason": (
                    f"mean RMSE gain={rollout['relative_rmse_gain']:.3%}, but "
                    f"MSE CI=[{rollout['mse_gain_ci_low']:.3f},"
                    f"{rollout['mse_gain_ci_high']:.3f}] and worst-SNR "
                    f"degradation={rollout['worst_per_snr_degradation']:.1%}"
                ),
                "rerun_needed": False,
            },
            {
                "route": "BRSR active finite-bit acquisition idea",
                "previous_status": "frozen",
                "audited_status": "HOLD",
                "scope": "route-level idea beyond the frozen Stage B configuration",
                "reason": "candidate-pool interaction and alternative valid likelihoods were not jointly tested",
                "rerun_needed": False,
            },
            {
                "route": "PASR v1 configuration",
                "previous_status": "NO_GO",
                "audited_status": "NO-GO-CONFIG",
                "scope": "modal-distance action score and early stopping",
                "reason": (
                    f"relative RMSE gain={pasr['relative_rmse_gain']:.3%}, "
                    f"MSE CI=[{pasr['mse_gain_ci_low']:.3f},"
                    f"{pasr['mse_gain_ci_high']:.3f}], high-SNR degradation="
                    f"{pasr['high_snr_degradation']:.1%}"
                ),
                "rerun_needed": False,
            },
            {
                "route": "PASR/BRSR active acquisition family",
                "previous_status": "frozen",
                "audited_status": "HOLD",
                "scope": "active finite-bit acquisition beyond PASR v1",
                "reason": "PASR oracle remained strong; configuration failure is not a route impossibility proof",
                "rerun_needed": False,
            },
            {
                "route": "V5B-Shared64",
                "previous_status": "frozen baseline/ablation",
                "audited_status": "HOLD",
                "scope": "strict CR16 three-development-pool OriginalTop1-WLS evidence",
                "reason": (
                    f"vs GeoShared52 RMSE gain={v5_geo['relative_rmse_gain']:.2%}; "
                    f"vs PowerShared52={v5_power['relative_rmse_gain']:.2%}; "
                    f"vs legacy Power64={v5_legacy['relative_rmse_gain']:.2%}. "
                    "It beats several baselines but does not establish dominance over the strongest single expert."
                ),
                "rerun_needed": False,
            },
            {
                "route": "BRSR v2 continuous-gain profile component",
                "previous_status": "STEP1_NO_GO",
                "audited_status": "NO-GO-COMPONENT",
                "scope": "frozen BestStatic-T32 continuous profile repair",
                "reason": "existing paired MSE CI supports degradation; no threshold reinterpretation is needed",
                "rerun_needed": False,
            },
            {
                "route": "PFRS fixed-mask search",
                "previous_status": "FIXED_MASK_SATURATED",
                "audited_status": "NO-GO-ROUTE",
                "scope": "only frozen N=16, M=4, eta=0.9 model and search space",
                "reason": "all 1820 masks were enumerated and the constrained space is saturated",
                "rerun_needed": False,
            },
            {
                "route": "FreqDAE v1-v4 architecture family",
                "previous_status": "frozen",
                "audited_status": "HOLD",
                "scope": "historical waveform-decoder implementations",
                "reason": "sample-level evidence is spread across incompatible protocols; deferred rather than backfilled",
                "rerun_needed": False,
            },
        ]
    )


def _plot_effects(effects: pd.DataFrame, output: Path) -> None:
    brsr_pasr = effects[
        effects["route"].isin(["BRSR-StageB-v1", "PASR-v1"])
        & effects["bucket"].eq("all")
    ].copy()
    shared = effects[
        effects["route"].eq("V5B-Shared64")
        & effects["bucket"].eq("all")
        & effects["baseline"].isin(
            ["V5A1-GeoShared52", "V5A1-PowerShared52", "V5A1-Power64", "DAE-CR16"]
        )
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for axis, frame, title, xlabel in (
        (
            axes[0],
            brsr_pasr,
            "Historical active-acquisition configurations",
            "Paired RMSE gain (sample; positive is better)",
        ),
        (
            axes[1],
            shared,
            "Strict-CR16 Shared64 comparisons",
            "Paired localization RMSE gain (m; positive is better)",
        ),
    ):
        labels = [
            f"{row.target}\\nvs {row.baseline}" for row in frame.itertuples()
        ]
        y = np.arange(len(frame))
        values = frame["rmse_gain"].to_numpy(dtype=float)
        lower = values - frame["rmse_gain_ci_low"].to_numpy(dtype=float)
        upper = frame["rmse_gain_ci_high"].to_numpy(dtype=float) - values
        axis.errorbar(
            values,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color="#2E6E9E",
            ecolor="#6F7D8C",
            capsize=4,
            linewidth=1.2,
        )
        axis.axvline(0.0, color="#B23A48", linestyle="--", linewidth=1.0)
        axis.set_yticks(y, labels)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output: Path,
    effects: pd.DataFrame,
    sensitivity: pd.DataFrame,
    classifications: pd.DataFrame,
) -> None:
    brsr = effects[
        effects["route"].eq("BRSR-StageB-v1")
        & effects["target"].eq("BRSR-Rollout-T32")
        & effects["bucket"].eq("all")
    ].iloc[0]
    pasr = effects[
        effects["route"].eq("PASR-v1") & effects["bucket"].eq("all")
    ].iloc[0]
    shared = effects[
        effects["route"].eq("V5B-Shared64") & effects["bucket"].eq("all")
    ]
    brsr_sensitivity = sensitivity[
        sensitivity["route"].eq("BRSR-StageB-v1")
        & sensitivity["target"].eq("BRSR-Rollout-T32")
    ]
    pasr_sensitivity = sensitivity[sensitivity["route"].eq("PASR-v1")]
    report = f"""# 研究门禁只读回溯统计报告

## 1. 审计边界

- 未训练模型、未重新生成信号、未覆盖历史结果。
- BRSR与PASR使用冻结逐样本结果；Shared64使用三个development snapshot池。
- 正收益定义为“基线误差减候选误差”；不同单位和协议不合并排名。
- bootstrap单位分别为独立轨迹或snapshot cluster，共{BOOTSTRAP_REPETITIONS}次。

## 2. BRSR Stage B v1

Rollout-T32相对SNR-only-T32的总体RMSE增益为
`{brsr.rmse_gain:.4f} sample`，95% CI为
`[{brsr.rmse_gain_ci_low:.4f}, {brsr.rmse_gain_ci_high:.4f}]`。
MSE增益95% CI为`[{brsr.mse_gain_ci_low:.4f}, {brsr.mse_gain_ci_high:.4f}]`，
80%功效下最小可检测MSE增益约为`{brsr.mde80_mse_gain:.4f} sample^2`。

在{len(brsr_sensitivity)}组RMSE最小收益和逐SNR退化阈值组合中，
完整门禁通过比例为`{brsr_sensitivity.full_pass.mean():.1%}`。
当前结论应表述为`NO-GO-CONFIG`；主动有限比特采集思想保持`HOLD`。

## 3. PASR v1

PASR q6相对development冻结的同bit静态时间共享方案，RMSE增益为
`{pasr.rmse_gain:.4f} sample`，95% CI为
`[{pasr.rmse_gain_ci_low:.4f}, {pasr.rmse_gain_ci_high:.4f}]`。
在{len(pasr_sensitivity)}组合理阈值组合中，完整门禁通过比例为
`{pasr_sensitivity.full_pass.mean():.1%}`。

因此PASR v1动作与停止组合为`NO-GO-CONFIG`。Oracle-Next-Bin仍有明显空间，
不能把该结论外推为整个主动采集路线失败。

## 4. V5B-Shared64

严格CR16下，V5B-Shared64的主要配对比较如下：

| 对手 | RMSE增益/m | 95% CI/m | TDOA MAE增益/sample | 95% CI/sample |
|---|---:|---:|---:|---:|
"""
    for row in shared.itertuples():
        report += (
            f"| {row.baseline} | {row.rmse_gain:.3f} | "
            f"[{row.rmse_gain_ci_low:.3f}, {row.rmse_gain_ci_high:.3f}] | "
            f"{row.mean_abs_gain:.3f} | "
            f"[{row.mean_abs_gain_ci_low:.3f}, {row.mean_abs_gain_ci_high:.3f}] |\n"
        )
    report += """

Shared64对GeoShared52有机制收益，但未稳定支配PowerShared52，且相对完整
legacy Power64不占优。其合理状态为`HOLD`或冻结的严格预算消融，而不是
`GO`主方法。

## 5. 最终重新分类

| 路线 | 回溯后状态 | 适用范围 |
|---|---|---|
"""
    for row in classifications.itertuples():
        report += f"| {row.route} | `{row.audited_status}` | {row.scope} |\n"
    report += """

## 6. 结论

此次回溯没有发现被单个人为百分比阈值明显误杀、且现有统计证据足以直接升级
为`GO`的路线。BRSR v1与PASR v1的配置失败不只来自5%或10%门槛，而同时
涉及配对不确定性和明显的条件退化。BRSR整体思想仍为`HOLD`，因为候选池与
有效似然的交互没有被当前实验完整检验。
"""
    (output / "audit_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    input_hashes = _validate_sources()
    output = RESULT_ROOT / "研究门禁审计" / (
        f"GATE_AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_readonly"
    )
    output.mkdir(parents=True, exist_ok=False)
    print(f"[Audit] output={output}", flush=True)
    print("[Audit] frozen inputs hashed; no historical result will be modified", flush=True)

    effect_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    print("[Audit] BRSR Stage B v1", flush=True)
    brsr = _audit_brsr(
        effect_rows, sensitivity_rows, calibration_rows, event_rows
    )
    print("[Audit] PASR v1", flush=True)
    pasr = _audit_pasr(effect_rows, sensitivity_rows, event_rows)
    print("[Audit] V5B-Shared64 multi-pool", flush=True)
    shared64 = _audit_shared64(effect_rows, sensitivity_rows)

    effects = pd.DataFrame(effect_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    calibration = pd.DataFrame(calibration_rows)
    events = pd.DataFrame(event_rows)
    classifications = _route_classification(brsr, pasr, shared64)
    effects.to_csv(output / "gate_effect_sizes.csv", index=False)
    sensitivity.to_csv(output / "gate_threshold_sensitivity.csv", index=False)
    calibration.to_csv(output / "gate_calibration_audit.csv", index=False)
    events.to_csv(output / "gate_severe_event_audit.csv", index=False)
    classifications.to_csv(output / "route_reclassification.csv", index=False)
    _plot_effects(effects, output / "FigGateAudit_Effect_Uncertainty.svg")
    _write_report(output, effects, sensitivity, classifications)

    current_hashes = {path: _sha256(path) for path in input_hashes}
    if current_hashes != input_hashes:
        raise RuntimeError("a frozen input changed during the read-only audit")
    manifest = {
        "audit_type": "read_only_retrospective_gate_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "routes": ["BRSR Stage B v1", "PASR v1", "V5B-Shared64"],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "input_sha256": {str(path): value for path, value in input_hashes.items()},
        "historical_inputs_unchanged": True,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": (
            "This audit reclassifies evidence strength and threshold sensitivity. "
            "It does not alter historical automatic statuses or create new locked evidence."
        ),
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[Done] rows: effects={len(effects)}, sensitivity={len(sensitivity)}, "
        f"classifications={len(classifications)}",
        flush=True,
    )
    print(f"[Done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
