"""Read-only failure certificate for the frozen BRSR Stage 2A result.

The diagnostic consumes exported CSV/JSON artifacts only. It never regenerates
signals, fits a new policy, changes a threshold, or overwrites the source result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "运行结果"
    / "BRSR"
    / "BRSR_STAGE2A_20260724_153144_development"
)
DEFAULT_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260727
FIXED_PAYLOAD_BITS = 24
PRIMARY_TARGET_BITS = 24
KEYS = ["seed", "trial", "cluster_id", "snr_db"]
SOURCE_FILES = (
    "stage2a_action_rows.csv",
    "stage2a_trial_results.csv",
    "frozen_rate_policies.csv",
    "frozen_static_plans.csv",
    "temperature_calibration.csv",
    "gate_decision.json",
    "manifest.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(source: Path) -> dict[str, str]:
    hashes = {}
    for name in SOURCE_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"required Stage 2A artifact missing: {path}")
        hashes[name] = _sha256(path)
    return hashes


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    name: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def _validate_source(
    source: Path,
    actions: pd.DataFrame,
    trials: pd.DataFrame,
    policies: pd.DataFrame,
    plans: pd.DataFrame,
) -> int:
    action_columns = {
        *KEYS,
        "true_tau",
        "position",
        "bin_index",
        "raw_predicted_voi_samples2",
        "calibrated_predicted_voi_samples2",
        "raw_estimate_tau",
        "calibrated_estimate_tau",
        "raw_posterior_risk_samples2",
        "calibrated_posterior_risk_samples2",
        "raw_squared_error_samples2",
        "calibrated_squared_error_samples2",
        "raw_realized_gain_samples2",
        "calibrated_realized_gain_samples2",
        "oracle_realized_gain_samples2",
    }
    trial_columns = {
        *KEYS,
        "true_tau",
        "method",
        "squared_error_samples2",
        "total_bits",
        "selected_position",
        "stopped",
    }
    _require_columns(actions, action_columns, name="stage2a_action_rows.csv")
    _require_columns(trials, trial_columns, name="stage2a_trial_results.csv")
    _require_columns(
        policies,
        {
            "snr_db",
            "temperature_mode",
            "target_mean_bits",
            "lambda_per_payload_bit",
        },
        name="frozen_rate_policies.csv",
    )
    _require_columns(
        plans,
        {"family", "payload_bits", "positions", "bins"},
        name="frozen_static_plans.csv",
    )
    if actions.duplicated(KEYS + ["position"]).any():
        raise RuntimeError("action rows contain duplicate sample-position keys")
    group_sizes = actions.groupby(KEYS, sort=False).size()
    if group_sizes.nunique() != 1 or int(group_sizes.iloc[0]) < 2:
        raise RuntimeError("each sample must have the same multi-action set")
    if trials.duplicated(KEYS + ["method"]).any():
        raise RuntimeError("trial rows contain duplicate sample-method keys")
    expected_groups = int(actions.groupby(KEYS, sort=False).ngroups)
    primary = trials[trials["method"].eq("BRSR-Calibrated-B24")]
    if len(primary) != expected_groups:
        raise RuntimeError("primary trial rows are incomplete")

    gate = json.loads((source / "gate_decision.json").read_text(encoding="utf-8"))
    if gate.get("mode") != "development":
        raise RuntimeError("source is not a formal development result")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "development":
        raise RuntimeError("source manifest is not development mode")
    return int(group_sizes.iloc[0])


def _fixed_position(plans: pd.DataFrame) -> tuple[int, int]:
    rows = plans[
        plans["family"].eq("BestStatic")
        & plans["payload_bits"].eq(FIXED_PAYLOAD_BITS)
    ]
    if len(rows) != 1:
        raise RuntimeError("exactly one BestStatic-B24 plan is required")
    positions = [int(value) for value in str(rows.iloc[0]["positions"]).split()]
    bins = [int(value) for value in str(rows.iloc[0]["bins"]).split()]
    if len(positions) != 1 or len(bins) != 1:
        raise RuntimeError("BestStatic-B24 must contain exactly one action")
    return positions[0], bins[0]


def _policy_thresholds(policies: pd.DataFrame) -> dict[float, float]:
    rows = policies[
        policies["temperature_mode"].eq("calibrated")
        & policies["target_mean_bits"].eq(PRIMARY_TARGET_BITS)
    ]
    thresholds = {
        float(row["snr_db"]): 8.0 * float(row["lambda_per_payload_bit"])
        for _, row in rows.iterrows()
    }
    if len(thresholds) != 6 or not np.all(np.isfinite(list(thresholds.values()))):
        raise RuntimeError("calibrated B24 thresholds are incomplete")
    return thresholds


def _method_index(trials: pd.DataFrame) -> pd.DataFrame:
    return trials.set_index(KEYS + ["method"], verify_integrity=True)


def _trial_row(indexed: pd.DataFrame, key: tuple[object, ...], method: str) -> pd.Series:
    try:
        row = indexed.loc[(*key, method)]
    except KeyError as error:
        raise RuntimeError(f"missing method row: {key}, {method}") from error
    if isinstance(row, pd.DataFrame):
        raise RuntimeError(f"duplicated method row: {key}, {method}")
    return row


def _best_row(group: pd.DataFrame, key: str) -> pd.Series:
    return group.sort_values(
        [key, "position"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]


def build_sample_diagnostics(
    actions: pd.DataFrame,
    trials: pd.DataFrame,
    policies: pd.DataFrame,
    plans: pd.DataFrame,
) -> pd.DataFrame:
    """Build one diagnostic row per frozen development sample."""

    fixed_position, fixed_bin = _fixed_position(plans)
    thresholds = _policy_thresholds(policies)
    indexed = _method_index(trials)
    rows: list[dict[str, object]] = []

    for key, group in actions.groupby(KEYS, sort=False):
        group = group.sort_values("position", kind="stable")
        snr = float(key[-1])
        fixed = group[group["position"].eq(fixed_position)]
        if len(fixed) != 1 or int(fixed.iloc[0]["bin_index"]) != fixed_bin:
            raise RuntimeError("fixed action is missing or inconsistent")
        fixed = fixed.iloc[0]
        calibrated = _best_row(group, "calibrated_predicted_voi_samples2")
        raw = _best_row(group, "raw_predicted_voi_samples2")
        oracle = _best_row(group, "oracle_realized_gain_samples2")
        ordered = group.sort_values(
            ["calibrated_predicted_voi_samples2", "position"],
            ascending=[False, True],
            kind="stable",
        )
        second = ordered.iloc[1]

        raw_selected = group[group["position"].eq(int(raw["position"]))].iloc[0]
        base_cal_values = (
            group["calibrated_squared_error_samples2"]
            + group["calibrated_realized_gain_samples2"]
        ).to_numpy(float)
        base_raw_values = (
            group["raw_squared_error_samples2"]
            + group["raw_realized_gain_samples2"]
        ).to_numpy(float)
        if not np.allclose(base_cal_values, base_cal_values[0], atol=1e-10):
            raise RuntimeError("calibrated base error cannot be reconstructed")
        if not np.allclose(base_raw_values, base_raw_values[0], atol=1e-10):
            raise RuntimeError("raw base error cannot be reconstructed")

        cal_score = float(calibrated["calibrated_predicted_voi_samples2"])
        threshold = thresholds[snr]
        should_acquire = cal_score > threshold
        b24 = _trial_row(indexed, key, "BRSR-Calibrated-B24")
        b29 = _trial_row(indexed, key, "BRSR-Calibrated-B29")
        raw_b29 = _trial_row(indexed, key, "BRSR-Raw-B29")
        selected = b24["selected_position"]
        selected_position = None if pd.isna(selected) else int(float(selected))
        expected_position = int(calibrated["position"]) if should_acquire else None
        if selected_position != expected_position:
            raise RuntimeError("B24 selected action does not match frozen policy")
        if int(float(b29["selected_position"])) != int(calibrated["position"]):
            raise RuntimeError("B29 action does not match calibrated score maximum")
        if int(float(raw_b29["selected_position"])) != int(raw["position"]):
            raise RuntimeError("raw B29 action does not match raw score maximum")

        true_tau = float(group.iloc[0]["true_tau"])
        trial_true = float(b24["true_tau"])
        if not math.isclose(true_tau, trial_true, abs_tol=1e-12):
            raise RuntimeError("true TDOA is inconsistent across source files")

        raw_pred_error = float(raw["raw_squared_error_samples2"])
        intermediate_error = float(
            raw_selected["calibrated_squared_error_samples2"]
        )
        cal_pred_error = float(calibrated["calibrated_squared_error_samples2"])
        fixed_error = float(fixed["calibrated_squared_error_samples2"])
        oracle_error = float(oracle["calibrated_squared_error_samples2"])
        b24_error = float(b24["squared_error_samples2"])

        estimation_delta = intermediate_error - raw_pred_error
        ranking_delta = cal_pred_error - intermediate_error
        total_calibration_delta = cal_pred_error - raw_pred_error
        if not math.isclose(
            estimation_delta + ranking_delta,
            total_calibration_delta,
            abs_tol=1e-10,
        ):
            raise RuntimeError("temperature decomposition is not additive")

        rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "true_tau": true_tau,
                "fixed_position": fixed_position,
                "fixed_bin": fixed_bin,
                "raw_position": int(raw["position"]),
                "raw_bin": int(raw["bin_index"]),
                "predicted_position": int(calibrated["position"]),
                "predicted_bin": int(calibrated["bin_index"]),
                "oracle_position": int(oracle["position"]),
                "oracle_bin": int(oracle["bin_index"]),
                "predicted_is_fixed": bool(
                    int(calibrated["position"]) == fixed_position
                ),
                "predicted_matches_oracle": bool(
                    int(calibrated["position"]) == int(oracle["position"])
                ),
                "raw_matches_calibrated": bool(
                    int(raw["position"]) == int(calibrated["position"])
                ),
                "b24_acquired": should_acquire,
                "policy_threshold_samples2": threshold,
                "best_predicted_voi_samples2": cal_score,
                "fixed_predicted_voi_samples2": float(
                    fixed["calibrated_predicted_voi_samples2"]
                ),
                "predicted_advantage_over_fixed_samples2": (
                    cal_score
                    - float(fixed["calibrated_predicted_voi_samples2"])
                ),
                "top1_top2_margin_samples2": (
                    cal_score
                    - float(second["calibrated_predicted_voi_samples2"])
                ),
                "acquire_margin_samples2": cal_score - threshold,
                "score_range_samples2": (
                    cal_score
                    - float(group["calibrated_predicted_voi_samples2"].min())
                ),
                "selected_posterior_risk_samples2": float(
                    calibrated["calibrated_posterior_risk_samples2"]
                ),
                "selected_temperature_shift_abs_samples": abs(
                    float(calibrated["calibrated_estimate_tau"])
                    - float(calibrated["raw_estimate_tau"])
                ),
                "base_calibrated_error_samples2": float(base_cal_values[0]),
                "base_raw_error_samples2": float(base_raw_values[0]),
                "fixed_error_samples2": fixed_error,
                "raw_predicted_error_samples2": raw_pred_error,
                "intermediate_calibrated_error_samples2": intermediate_error,
                "predicted_error_samples2": cal_pred_error,
                "oracle_error_samples2": oracle_error,
                "b24_error_samples2": b24_error,
                "temperature_estimation_delta_samples2": estimation_delta,
                "temperature_ranking_delta_samples2": ranking_delta,
                "temperature_total_delta_samples2": total_calibration_delta,
                "stopping_delta_samples2": b24_error - cal_pred_error,
                "predicted_vs_fixed_delta_samples2": (
                    cal_pred_error - fixed_error
                ),
                "predicted_vs_oracle_regret_samples2": (
                    cal_pred_error - oracle_error
                ),
                "fixed_vs_oracle_regret_samples2": fixed_error - oracle_error,
                "b24_vs_fixed_delta_samples2": b24_error - fixed_error,
                "realized_advantage_over_fixed_samples2": (
                    fixed_error - cal_pred_error
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values(KEYS, kind="stable")
    if len(frame) != actions.groupby(KEYS, sort=False).ngroups:
        raise RuntimeError("sample diagnostics are incomplete")
    return frame.reset_index(drop=True)


def _cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    unique = np.unique(clusters)
    means = np.asarray(
        [float(np.mean(values[clusters == cluster])) for cluster in unique]
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(means), size=(repetitions, len(means)))
    distribution = np.mean(means[sampled], axis=1)
    return (
        float(np.mean(values)),
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def _spearman_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[finite], y[finite], clusters[finite]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan, math.nan
    point = float(spearmanr(x, y).statistic)
    unique = np.unique(clusters)
    by_cluster = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique
    }
    rng = np.random.default_rng(seed)
    distribution = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[cluster] for cluster in sampled])
        if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
            continue
        value = float(spearmanr(x[indices], y[indices]).statistic)
        if math.isfinite(value):
            distribution.append(value)
    if not distribution:
        return point, math.nan, math.nan
    return (
        point,
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(~positive))
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = rankdata(scores, method="average")
    rank_sum = float(np.sum(ranks[positive]))
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def _scope_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [("All", frame)]
    groups.extend(
        (f"{snr:g} dB", subset)
        for snr, subset in frame.groupby("snr_db", sort=True)
    )
    return groups


def regret_decomposition(
    frame: pd.DataFrame,
    *,
    repetitions: int,
) -> pd.DataFrame:
    rows = []
    components = (
        "temperature_estimation_delta_samples2",
        "temperature_ranking_delta_samples2",
        "temperature_total_delta_samples2",
        "stopping_delta_samples2",
        "predicted_vs_fixed_delta_samples2",
        "predicted_vs_oracle_regret_samples2",
        "fixed_vs_oracle_regret_samples2",
        "b24_vs_fixed_delta_samples2",
    )
    for scope, subset in _scope_groups(frame):
        row: dict[str, object] = {
            "scope": scope,
            "snr_db": (
                math.nan if scope == "All" else float(subset["snr_db"].iloc[0])
            ),
            "n_rows": len(subset),
            "n_clusters": subset["cluster_id"].nunique(),
            "fixed_rmse_samples": math.sqrt(
                float(subset["fixed_error_samples2"].mean())
            ),
            "predicted_rmse_samples": math.sqrt(
                float(subset["predicted_error_samples2"].mean())
            ),
            "oracle_rmse_samples": math.sqrt(
                float(subset["oracle_error_samples2"].mean())
            ),
            "b24_rmse_samples": math.sqrt(
                float(subset["b24_error_samples2"].mean())
            ),
            "predicted_is_fixed_fraction": float(
                subset["predicted_is_fixed"].mean()
            ),
            "predicted_matches_oracle_fraction": float(
                subset["predicted_matches_oracle"].mean()
            ),
            "predicted_beats_fixed_fraction": float(
                (subset["realized_advantage_over_fixed_samples2"] > 0).mean()
            ),
        }
        positive_loss = subset["predicted_vs_fixed_delta_samples2"].clip(
            lower=0.0
        )
        tail_count = max(1, int(math.ceil(0.10 * len(subset))))
        tail_loss = subset["predicted_vs_fixed_delta_samples2"].nlargest(
            tail_count
        ).clip(lower=0.0)
        row["worst10_positive_loss_share"] = (
            float(tail_loss.sum() / positive_loss.sum())
            if float(positive_loss.sum()) > 0.0
            else 0.0
        )
        for offset, component in enumerate(components):
            mean, low, high = _cluster_bootstrap_mean(
                subset[component].to_numpy(float),
                subset["cluster_id"].to_numpy(str),
                repetitions=repetitions,
                seed=BOOTSTRAP_SEED + offset,
            )
            row[f"{component}_mean"] = mean
            row[f"{component}_ci_low"] = low
            row[f"{component}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_diagnostics(
    frame: pd.DataFrame,
    *,
    repetitions: int,
) -> pd.DataFrame:
    features = {
        "predicted_advantage_over_fixed": (
            "predicted_advantage_over_fixed_samples2",
            1.0,
        ),
        "top1_top2_margin": ("top1_top2_margin_samples2", 1.0),
        "score_range": ("score_range_samples2", 1.0),
        "negative_selected_posterior_risk": (
            "selected_posterior_risk_samples2",
            -1.0,
        ),
        "negative_temperature_shift": (
            "selected_temperature_shift_abs_samples",
            -1.0,
        ),
    }
    rows = []
    nonfixed = frame[~frame["predicted_is_fixed"]].copy()
    for scope, subset in _scope_groups(nonfixed):
        target = subset["realized_advantage_over_fixed_samples2"].to_numpy(
            float
        )
        positive = target > 0.0
        for offset, (name, (column, direction)) in enumerate(features.items()):
            score = direction * subset[column].to_numpy(float)
            point, low, high = _spearman_bootstrap(
                score,
                target,
                subset["cluster_id"].to_numpy(str),
                repetitions=repetitions,
                seed=BOOTSTRAP_SEED + 100 + offset,
            )
            rows.append(
                {
                    "scope": scope,
                    "snr_db": (
                        math.nan
                        if scope == "All"
                        else float(subset["snr_db"].iloc[0])
                    ),
                    "subset": "predicted_nonfixed",
                    "feature": name,
                    "n_rows": len(subset),
                    "n_clusters": subset["cluster_id"].nunique(),
                    "spearman": point,
                    "spearman_ci_low": low,
                    "spearman_ci_high": high,
                    "auc_for_predicted_beats_fixed": _auc(score, positive),
                    "realized_advantage_mean_samples2": float(
                        np.mean(target)
                    ),
                    "predicted_beats_fixed_fraction": float(
                        np.mean(positive)
                    ),
                }
            )
    return pd.DataFrame(rows)


def confidence_quantiles(
    frame: pd.DataFrame,
    *,
    repetitions: int,
) -> pd.DataFrame:
    rows = []
    subset = frame[~frame["predicted_is_fixed"]].copy()
    for scope, group in _scope_groups(subset):
        if len(group) < 10:
            continue
        ranked = group[
            "predicted_advantage_over_fixed_samples2"
        ].rank(method="first")
        quantile = pd.qcut(ranked, q=5, labels=False, duplicates="drop")
        for index, part in group.groupby(quantile, sort=True):
            mean, low, high = _cluster_bootstrap_mean(
                part["realized_advantage_over_fixed_samples2"].to_numpy(
                    float
                ),
                part["cluster_id"].to_numpy(str),
                repetitions=repetitions,
                seed=BOOTSTRAP_SEED + 300 + int(index),
            )
            rows.append(
                {
                    "scope": scope,
                    "snr_db": (
                        math.nan
                        if scope == "All"
                        else float(part["snr_db"].iloc[0])
                    ),
                    "confidence_quintile": int(index) + 1,
                    "n_rows": len(part),
                    "predicted_advantage_mean_samples2": float(
                        part[
                            "predicted_advantage_over_fixed_samples2"
                        ].mean()
                    ),
                    "realized_advantage_mean_samples2": mean,
                    "realized_advantage_ci_low": low,
                    "realized_advantage_ci_high": high,
                    "predicted_beats_fixed_fraction": float(
                        (
                            part["realized_advantage_over_fixed_samples2"] > 0
                        ).mean()
                    ),
                    "predicted_loss_mean_samples2": float(
                        part["predicted_vs_fixed_delta_samples2"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def tail_cluster_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[~frame["predicted_is_fixed"]].copy()
    subset["positive_loss_samples2"] = subset[
        "predicted_vs_fixed_delta_samples2"
    ].clip(lower=0.0)
    rows = []
    total_positive_loss = float(subset["positive_loss_samples2"].sum())
    for cluster, group in subset.groupby("cluster_id", sort=False):
        rows.append(
            {
                "cluster_id": cluster,
                "seed": int(group["seed"].iloc[0]),
                "trial": int(group["trial"].iloc[0]),
                "n_nonfixed": len(group),
                "snr_values": " ".join(
                    f"{value:g}" for value in sorted(group["snr_db"].unique())
                ),
                "predicted_advantage_mean_samples2": float(
                    group[
                        "predicted_advantage_over_fixed_samples2"
                    ].mean()
                ),
                "realized_advantage_mean_samples2": float(
                    group["realized_advantage_over_fixed_samples2"].mean()
                ),
                "net_loss_sum_samples2": float(
                    group["predicted_vs_fixed_delta_samples2"].sum()
                ),
                "positive_loss_sum_samples2": float(
                    group["positive_loss_samples2"].sum()
                ),
                "positive_loss_share": (
                    float(group["positive_loss_samples2"].sum())
                    / total_positive_loss
                    if total_positive_loss > 0.0
                    else 0.0
                ),
                "worst_loss_samples2": float(
                    group["predicted_vs_fixed_delta_samples2"].max()
                ),
                "predicted_worse_count": int(
                    np.sum(group["predicted_vs_fixed_delta_samples2"] > 0.0)
                ),
                "oracle_regret_sum_samples2": float(
                    group["predicted_vs_oracle_regret_samples2"].sum()
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        "positive_loss_sum_samples2",
        ascending=False,
        kind="stable",
    )
    result.insert(0, "positive_loss_rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def tail_sensitivity(
    frame: pd.DataFrame,
    cluster_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    subset = frame[~frame["predicted_is_fixed"]].copy()
    ranked = cluster_diagnostics["cluster_id"].tolist()
    rows = []
    for remove_count in (0, 1, 3, 5, 10):
        removed = set(ranked[:remove_count])
        kept = subset[~subset["cluster_id"].isin(removed)]
        fixed_rmse = math.sqrt(float(kept["fixed_error_samples2"].mean()))
        predicted_rmse = math.sqrt(
            float(kept["predicted_error_samples2"].mean())
        )
        rows.append(
            {
                "removed_top_positive_loss_clusters": remove_count,
                "n_rows_remaining": len(kept),
                "n_clusters_remaining": kept["cluster_id"].nunique(),
                "fixed_rmse_samples": fixed_rmse,
                "predicted_rmse_samples": predicted_rmse,
                "predicted_rmse_gain_samples": fixed_rmse - predicted_rmse,
                "interpretation": (
                    "descriptive sensitivity only; no rows are removed from "
                    "the registered Stage 2A result"
                ),
            }
        )
    return pd.DataFrame(rows)


def leave_one_cluster_spearman(frame: pd.DataFrame) -> dict[str, float | int]:
    subset = frame[~frame["predicted_is_fixed"]]
    values = []
    for cluster in subset["cluster_id"].unique():
        kept = subset[subset["cluster_id"].ne(cluster)]
        value = float(
            spearmanr(
                kept["predicted_advantage_over_fixed_samples2"],
                kept["realized_advantage_over_fixed_samples2"],
            ).statistic
        )
        values.append(value)
    array = np.asarray(values, dtype=float)
    return {
        "n_leave_one_cluster_runs": len(array),
        "leave_one_cluster_spearman_min": float(np.min(array)),
        "leave_one_cluster_spearman_median": float(np.median(array)),
        "leave_one_cluster_spearman_max": float(np.max(array)),
    }


def action_selection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, subset in _scope_groups(frame):
        for position, group in subset.groupby("predicted_position", sort=True):
            rows.append(
                {
                    "scope": scope,
                    "snr_db": (
                        math.nan
                        if scope == "All"
                        else float(subset["snr_db"].iloc[0])
                    ),
                    "predicted_position": int(position),
                    "predicted_bin": int(group["predicted_bin"].iloc[0]),
                    "count": len(group),
                    "fraction": len(group) / len(subset),
                    "realized_advantage_over_fixed_mean_samples2": float(
                        group[
                            "realized_advantage_over_fixed_samples2"
                        ].mean()
                    ),
                    "oracle_match_fraction": float(
                        group["predicted_matches_oracle"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def worst_action_errors(frame: pd.DataFrame, limit: int = 60) -> pd.DataFrame:
    columns = [
        *KEYS,
        "true_tau",
        "fixed_position",
        "predicted_position",
        "predicted_bin",
        "oracle_position",
        "oracle_bin",
        "best_predicted_voi_samples2",
        "predicted_advantage_over_fixed_samples2",
        "top1_top2_margin_samples2",
        "selected_posterior_risk_samples2",
        "fixed_error_samples2",
        "predicted_error_samples2",
        "oracle_error_samples2",
        "predicted_vs_fixed_delta_samples2",
        "predicted_vs_oracle_regret_samples2",
    ]
    return (
        frame.sort_values(
            "predicted_vs_fixed_delta_samples2",
            ascending=False,
            kind="stable",
        )
        .head(limit)[columns]
        .reset_index(drop=True)
    )


def _decision(
    confidence: pd.DataFrame,
    *,
    leave_one: dict[str, float | int] | None = None,
    cluster_diagnostics: pd.DataFrame | None = None,
) -> dict[str, object]:
    primary = confidence[
        confidence["scope"].eq("All")
        & confidence["feature"].eq("predicted_advantage_over_fixed")
    ]
    if len(primary) != 1:
        raise RuntimeError("primary confidence diagnostic is missing")
    row = primary.iloc[0]
    low = float(row["spearman_ci_low"])
    high = float(row["spearman_ci_high"])
    if math.isfinite(low) and low > 0.0:
        status = "SCORE_RELATION_PRESENT"
        implication = (
            "Only a newly preregistered Stage 2A.1 on fresh development seeds "
            "may test a non-oracle safety gate; Stage 2B remains blocked."
        )
    elif math.isfinite(high) and high <= 0.0:
        status = "NO_GO_COMPONENT_CURRENT_SCORER"
        implication = (
            "Freeze the current one-action observable scorer. Oracle action-space "
            "evidence remains valid, so this is not NO-GO-ROUTE."
        )
    else:
        status = "HOLD_SCORE_RELATION_UNCERTAIN"
        implication = (
            "Do not tune on this development set. The current scorer remains "
            "blocked and Stage 2B is not permitted."
        )
    result = {
        "status": status,
        "primary_feature": "predicted_advantage_over_fixed",
        "analysis_subset": "samples where predicted action differs from fixed B24 action",
        "n_rows": int(row["n_rows"]),
        "n_clusters": int(row["n_clusters"]),
        "spearman": float(row["spearman"]),
        "spearman_ci_low": low,
        "spearman_ci_high": high,
        "auc_for_predicted_beats_fixed": float(
            row["auc_for_predicted_beats_fixed"]
        ),
        "interpretation": implication,
        "boundary": (
            "This is a read-only mechanism certificate on the existing development "
            "set. It neither changes the Stage 2A automatic gate nor constitutes "
            "locked or cross-scene evidence."
        ),
    }
    if leave_one is not None:
        result.update(leave_one)
    if cluster_diagnostics is not None and not cluster_diagnostics.empty:
        result.update(
            {
                "largest_cluster_positive_loss_share": float(
                    cluster_diagnostics.iloc[0]["positive_loss_share"]
                ),
                "top3_cluster_positive_loss_share": float(
                    cluster_diagnostics.head(3)["positive_loss_share"].sum()
                ),
                "largest_loss_cluster": str(
                    cluster_diagnostics.iloc[0]["cluster_id"]
                ),
                "tail_interpretation": (
                    "The registered rank relation may coexist with unsafe "
                    "heavy-tail action errors; no confidence threshold is "
                    "approved on this development set."
                ),
            }
        )
    return result


def _plot_certificate(
    frame: pd.DataFrame,
    regret: pd.DataFrame,
    quantiles: pd.DataFrame,
    actions: pd.DataFrame,
    output: Path,
) -> None:
    by_snr = regret[regret["scope"].ne("All")].sort_values("snr_db")
    snr = by_snr["snr_db"].to_numpy(float)
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

    axes[0, 0].plot(
        snr,
        by_snr["fixed_rmse_samples"],
        marker="o",
        label="Fixed B24 action",
    )
    axes[0, 0].plot(
        snr,
        by_snr["predicted_rmse_samples"],
        marker="s",
        label="Predicted action",
    )
    axes[0, 0].plot(
        snr,
        by_snr["oracle_rmse_samples"],
        marker="^",
        label="Oracle action",
    )
    axes[0, 0].set_xlabel("SNR (dB)")
    axes[0, 0].set_ylabel("TDOA RMSE (samples)")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.25)

    overall_quantiles = quantiles[quantiles["scope"].eq("All")]
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[0, 1].plot(
        overall_quantiles["confidence_quintile"],
        overall_quantiles["realized_advantage_mean_samples2"],
        marker="o",
        color="#d95f02",
    )
    axes[0, 1].set_xlabel("Predicted advantage quintile (low to high)")
    axes[0, 1].set_ylabel("Realized MSE advantage over fixed")
    axes[0, 1].grid(alpha=0.25)

    components = [
        "temperature_estimation_delta_samples2_mean",
        "temperature_ranking_delta_samples2_mean",
        "stopping_delta_samples2_mean",
        "predicted_vs_fixed_delta_samples2_mean",
    ]
    labels = [
        "Temperature\nestimate",
        "Temperature\naction rank",
        "Stop decision",
        "Predicted vs\nfixed action",
    ]
    overall = regret[regret["scope"].eq("All")].iloc[0]
    values = [float(overall[column]) for column in components]
    colors = ["#4c78a8" if value <= 0 else "#e45756" for value in values]
    axes[1, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 0].bar(labels, values, color=colors)
    axes[1, 0].set_ylabel("Mean MSE delta (positive is worse)")
    axes[1, 0].grid(axis="y", alpha=0.25)

    snr_actions = actions[actions["scope"].ne("All")]
    pivot = snr_actions.pivot(
        index="predicted_position",
        columns="snr_db",
        values="fraction",
    ).fillna(0.0)
    image = axes[1, 1].imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap="Blues",
        vmin=0.0,
        vmax=max(0.01, float(pivot.to_numpy().max())),
    )
    axes[1, 1].set_xticks(range(len(pivot.columns)))
    axes[1, 1].set_xticklabels([f"{value:g}" for value in pivot.columns])
    axes[1, 1].set_yticks(range(len(pivot.index)))
    axes[1, 1].set_yticklabels([str(value) for value in pivot.index])
    axes[1, 1].set_xlabel("SNR (dB)")
    axes[1, 1].set_ylabel("Predicted action position")
    figure.colorbar(image, ax=axes[1, 1], label="Selection fraction")

    figure.tight_layout()
    figure.savefig(
        output / "Fig1_Stage2A_Failure_Certificate.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_report(
    output: Path,
    source: Path,
    decision: dict[str, object],
    regret: pd.DataFrame,
) -> None:
    overall = regret[regret["scope"].eq("All")].iloc[0]
    report = f"""# BRSR Stage 2A只读失败机理证书

- 源结果：`{source}`
- 状态：`{decision["status"]}`
- 分析样本：{decision["n_rows"]}个非固定动作条件，来自{decision["n_clusters"]}条独立轨迹
- 主指标Spearman：`{decision["spearman"]:.4f}`
- 轨迹聚类95% CI：`[{decision["spearman_ci_low"]:.4f}, {decision["spearman_ci_high"]:.4f}]`
- 预测“优于固定动作”的AUC：`{decision["auc_for_predicted_beats_fixed"]:.4f}`
- 留一轨迹Spearman范围：`[{decision["leave_one_cluster_spearman_min"]:.4f}, {decision["leave_one_cluster_spearman_max"]:.4f}]`

## 后悔值概览

- 固定动作RMSE：`{overall["fixed_rmse_samples"]:.4f} sample`
- 预测动作RMSE：`{overall["predicted_rmse_samples"]:.4f} sample`
- Oracle动作RMSE：`{overall["oracle_rmse_samples"]:.4f} sample`
- 最差10%条件占全部正向损失：`{overall["worst10_positive_loss_share"]:.2%}`
- 最大单轨迹占全部正向损失：`{decision["largest_cluster_positive_loss_share"]:.2%}`
- 最大三条轨迹占全部正向损失：`{decision["top3_cluster_positive_loss_share"]:.2%}`

## 解释边界

{decision["interpretation"]}

本诊断只读取冻结development产物，不修改方法、阈值、候选池或源结果。
"""
    (output / "diagnostic_report.md").write_text(report, encoding="utf-8")


def run_diagnostic(
    source: Path,
    output: Path,
    *,
    repetitions: int,
) -> dict[str, object]:
    started = time.perf_counter()
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents:
        raise ValueError("diagnostic output must not be inside the frozen source")
    output.mkdir(parents=True, exist_ok=False)
    hashes_before = _source_hashes(source)

    actions = pd.read_csv(source / "stage2a_action_rows.csv")
    trials = pd.read_csv(source / "stage2a_trial_results.csv")
    policies = pd.read_csv(source / "frozen_rate_policies.csv")
    plans = pd.read_csv(source / "frozen_static_plans.csv")
    action_count = _validate_source(source, actions, trials, policies, plans)
    print(
        f"[Source] {source}\n"
        f"[Data] samples={actions.groupby(KEYS).ngroups}, "
        f"actions/sample={action_count}, bootstrap={repetitions}",
        flush=True,
    )

    samples = build_sample_diagnostics(actions, trials, policies, plans)
    regret = regret_decomposition(samples, repetitions=repetitions)
    confidence = confidence_diagnostics(samples, repetitions=repetitions)
    quantiles = confidence_quantiles(samples, repetitions=repetitions)
    selection = action_selection_summary(samples)
    worst = worst_action_errors(samples)
    tail_clusters = tail_cluster_diagnostics(samples)
    tail_checks = tail_sensitivity(samples, tail_clusters)
    leave_one = leave_one_cluster_spearman(samples)
    decision = _decision(
        confidence,
        leave_one=leave_one,
        cluster_diagnostics=tail_clusters,
    )

    samples.to_csv(output / "stage2a_sample_diagnostics.csv", index=False)
    regret.to_csv(output / "stage2a_regret_decomposition.csv", index=False)
    confidence.to_csv(
        output / "stage2a_confidence_diagnostics.csv",
        index=False,
    )
    quantiles.to_csv(output / "stage2a_confidence_quantiles.csv", index=False)
    selection.to_csv(
        output / "stage2a_action_selection_by_snr.csv",
        index=False,
    )
    worst.to_csv(output / "stage2a_worst_action_errors.csv", index=False)
    tail_clusters.to_csv(
        output / "stage2a_tail_cluster_diagnostics.csv",
        index=False,
    )
    tail_checks.to_csv(
        output / "stage2a_tail_sensitivity.csv",
        index=False,
    )
    _plot_certificate(samples, regret, quantiles, selection, output)
    _write_report(output, source, decision, regret)

    hashes_after = _source_hashes(source)
    if hashes_after != hashes_before:
        raise RuntimeError("frozen source artifacts changed during diagnosis")
    elapsed = time.perf_counter() - started
    payload = {
        **decision,
        "mode": "read_only_diagnostic",
        "source_result": str(source),
        "output": str(output),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_unchanged": True,
        "elapsed_seconds": elapsed,
    }
    (output / "diagnostic_decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "mode": "read_only_diagnostic",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_result": str(source),
        "source_hashes": hashes_before,
        "configuration": {
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "fixed_payload_bits": FIXED_PAYLOAD_BITS,
            "primary_feature": decision["primary_feature"],
            "analysis_subset": decision["analysis_subset"],
        },
        "files": sorted(path.name for path in output.iterdir()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[Decision] {decision['status']}\n"
        f"[Primary] Spearman={decision['spearman']:.4f}, "
        f"95% CI=[{decision['spearman_ci_low']:.4f}, "
        f"{decision['spearman_ci_high']:.4f}], "
        f"AUC={decision['auc_for_predicted_beats_fixed']:.4f}\n"
        f"[Saved] {output}\n"
        f"[Done] elapsed={elapsed:.1f}s",
        flush=True,
    )
    return payload


def _default_output(smoke: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "smoke" if smoke else "diagnostic"
    return (
        ROOT
        / "运行结果"
        / "BRSR"
        / f"BRSR_STAGE2A_FAILURE_{stamp}_{suffix}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only BRSR Stage 2A failure certificate."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repetitions = 100 if args.smoke else int(args.bootstrap)
    if repetitions < 20:
        raise ValueError("bootstrap repetitions must be at least 20")
    output = args.output_dir or _default_output(args.smoke)
    run_diagnostic(args.source_dir, output, repetitions=repetitions)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        raise
