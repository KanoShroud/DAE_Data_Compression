"""BRSR动作价值可观测性证书的统计与学习核心。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from brsr_observability_certificate.features import feature_columns


POLICY_KEY_COLUMNS = ("seed", "trial", "cluster_id", "snr_db")
STATIC_BASELINES = (
    "BestStatic-RateMatched-B29",
    "SNR-only-RateMatched-B29",
)


def candidate_models() -> dict[str, RegressorMixin]:
    """返回预注册的小型残差模型，不进行超参数扫描。"""

    return {
        "Observable-Ridge-B29": Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        ),
        "Observable-HGB-B29": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=2026073001,
        ),
    }


def fit_candidate_models(
    rows: pd.DataFrame,
) -> tuple[dict[str, RegressorMixin], list[str]]:
    columns = feature_columns(rows)
    x = rows[columns].to_numpy(dtype=float)
    y = rows["residual_target_samples2"].to_numpy(dtype=float)
    models = candidate_models()
    for model in models.values():
        model.fit(x, y)
    return models, columns


def predict_action_loss(
    model: RegressorMixin,
    rows: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    residual = np.asarray(
        model.predict(rows[columns].to_numpy(dtype=float)),
        dtype=float,
    )
    expected = rows["action_expected_risk_samples2"].to_numpy(dtype=float)
    prediction = np.maximum(expected + residual, 0.0)
    if prediction.shape != (len(rows),) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("conditional-risk model produced invalid predictions")
    return prediction


def _selected_rows(
    rows: pd.DataFrame,
    score: np.ndarray,
    *,
    method: str,
) -> pd.DataFrame:
    if score.shape != (len(rows),):
        raise ValueError("score must align with observable rows")
    frame = rows[
        [
            *POLICY_KEY_COLUMNS,
            "action_position",
            "actual_squared_error_samples2",
            "clairvoyant_best_position",
        ]
    ].copy()
    frame["selection_score"] = score
    selected = (
        frame.sort_values(
            [*POLICY_KEY_COLUMNS, "selection_score", "action_position"],
            kind="stable",
        )
        .drop_duplicates(list(POLICY_KEY_COLUMNS), keep="first")
        .copy()
    )
    selected["method"] = method
    selected["selected_position"] = selected["action_position"].astype(int)
    selected["squared_error_samples2"] = selected[
        "actual_squared_error_samples2"
    ].astype(float)
    selected["selected_clairvoyant_action"] = (
        selected["selected_position"]
        == selected["clairvoyant_best_position"].astype(int)
    )
    return selected[
        [
            *POLICY_KEY_COLUMNS,
            "method",
            "selected_position",
            "squared_error_samples2",
            "selected_clairvoyant_action",
        ]
    ]


def learned_policy_rows(
    rows: pd.DataFrame,
    models: Mapping[str, RegressorMixin],
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        prediction = predict_action_loss(model, rows, columns)
        predictions[name] = prediction
        frames.append(_selected_rows(rows, prediction, method=name))
    return pd.concat(frames, ignore_index=True), predictions


def diagnostic_policy_rows(
    rows: pd.DataFrame,
    *,
    fixed_position: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    physics = rows["action_expected_risk_samples2"].to_numpy(dtype=float)
    clairvoyant = rows["actual_squared_error_samples2"].to_numpy(dtype=float)
    fixed_score = np.where(
        rows["action_position"].to_numpy(dtype=int) == int(fixed_position),
        0.0,
        1.0,
    )
    frames = [
        _selected_rows(rows, physics, method="RiskGreedy-B29"),
        _selected_rows(
            rows,
            clairvoyant,
            method="ClairvoyantOutcomeOracle-B29",
        ),
        _selected_rows(rows, fixed_score, method="FixedAction-B29"),
    ]
    return pd.concat(frames, ignore_index=True), {
        "RiskGreedy-B29": physics,
        "ClairvoyantOutcomeOracle-B29": clairvoyant,
    }


def choose_model_on_calibration(
    calibration_policy_rows: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    candidates = calibration_policy_rows[
        calibration_policy_rows["method"].str.startswith("Observable-")
    ]
    summary = (
        candidates.groupby("method", as_index=False)
        .agg(
            n_samples=("squared_error_samples2", "size"),
            calibration_mse_samples2=("squared_error_samples2", "mean"),
        )
        .sort_values(
            ["calibration_mse_samples2", "method"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if summary.empty:
        raise RuntimeError("no observable model is available for selection")
    selected = str(summary.iloc[0]["method"])
    summary["selected_on_calibration"] = summary["method"].eq(selected)
    return selected, summary


def summarize_policies(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in results.groupby("method", sort=True):
        errors = group["squared_error_samples2"].to_numpy(dtype=float)
        rows.append(
            {
                "scope": "all",
                "snr_db": math.nan,
                "method": method,
                "n_samples": len(group),
                "mse_samples2": float(np.mean(errors)),
                "rmse_samples": float(np.sqrt(np.mean(errors))),
                "median_squared_error_samples2": float(np.median(errors)),
                "outlier_gt_4_samples_fraction": float(np.mean(errors > 16.0)),
            }
        )
        for snr, snr_group in group.groupby("snr_db", sort=True):
            values = snr_group["squared_error_samples2"].to_numpy(dtype=float)
            rows.append(
                {
                    "scope": "snr",
                    "snr_db": float(snr),
                    "method": method,
                    "n_samples": len(snr_group),
                    "mse_samples2": float(np.mean(values)),
                    "rmse_samples": float(np.sqrt(np.mean(values))),
                    "median_squared_error_samples2": float(np.median(values)),
                    "outlier_gt_4_samples_fraction": float(
                        np.mean(values > 16.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_values(
    results: pd.DataFrame,
    target: str,
    baseline: str,
    *,
    snr_db: float | None,
) -> pd.DataFrame:
    frame = results
    if snr_db is not None:
        frame = frame[frame["snr_db"].eq(float(snr_db))]
    target_rows = frame[frame["method"].eq(target)][
        [*POLICY_KEY_COLUMNS, "squared_error_samples2"]
    ].rename(columns={"squared_error_samples2": "target_loss"})
    baseline_rows = frame[frame["method"].eq(baseline)][
        [*POLICY_KEY_COLUMNS, "squared_error_samples2"]
    ].rename(columns={"squared_error_samples2": "baseline_loss"})
    merged = target_rows.merge(
        baseline_rows,
        on=list(POLICY_KEY_COLUMNS),
        validate="one_to_one",
    )
    if len(merged) != len(target_rows) or len(merged) != len(baseline_rows):
        raise RuntimeError("paired policy comparison is not sample-aligned")
    merged["mse_improvement_samples2"] = (
        merged["baseline_loss"] - merged["target_loss"]
    )
    return merged


def _cluster_interval(
    paired: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float, float]:
    cluster_means = (
        paired.groupby("cluster_id")["mse_improvement_samples2"]
        .mean()
        .to_numpy(dtype=float)
    )
    if cluster_means.size < 2:
        return (
            float(np.mean(cluster_means)),
            math.nan,
            math.nan,
            math.nan,
        )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        cluster_means.size,
        size=(repetitions, cluster_means.size),
    )
    bootstrap = cluster_means[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    mde = z * float(np.std(cluster_means, ddof=1)) / math.sqrt(cluster_means.size)
    return float(np.mean(cluster_means)), float(lower), float(upper), float(mde)


def paired_effects(
    results: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    baselines: tuple[str, ...],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    snr_values = sorted(float(value) for value in results["snr_db"].unique())
    scopes: list[tuple[str, float | None]] = [("all", None)]
    scopes.extend(("snr", snr) for snr in snr_values)
    for target_index, target in enumerate(targets):
        for baseline_index, baseline in enumerate(baselines):
            for scope_index, (scope, snr) in enumerate(scopes):
                paired = _paired_values(results, target, baseline, snr_db=snr)
                mean, lower, upper, mde = _cluster_interval(
                    paired,
                    repetitions=repetitions,
                    seed=seed
                    + 1000 * target_index
                    + 100 * baseline_index
                    + scope_index,
                )
                target_rmse = math.sqrt(float(paired["target_loss"].mean()))
                baseline_rmse = math.sqrt(float(paired["baseline_loss"].mean()))
                per_seed = paired.groupby("seed")["mse_improvement_samples2"].mean()
                rows.append(
                    {
                        "scope": scope,
                        "snr_db": math.nan if snr is None else float(snr),
                        "target": target,
                        "baseline": baseline,
                        "n_samples": len(paired),
                        "n_clusters": paired["cluster_id"].nunique(),
                        "target_rmse_samples": target_rmse,
                        "baseline_rmse_samples": baseline_rmse,
                        "relative_rmse_improvement": (
                            (baseline_rmse - target_rmse) / baseline_rmse
                        ),
                        "paired_mse_improvement_samples2": mean,
                        "ci95_lower_samples2": lower,
                        "ci95_upper_samples2": upper,
                        "power80_mde_samples2": mde,
                        "positive_seed_fraction": float(np.mean(per_seed > 0.0)),
                        "positive_seed_count": int(np.sum(per_seed > 0.0)),
                        "seed_count": int(per_seed.size),
                    }
                )
    return pd.DataFrame(rows)


def prediction_diagnostics(
    rows: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    actual = rows["actual_squared_error_samples2"].to_numpy(dtype=float)
    output: list[dict[str, object]] = []
    for method, prediction in predictions.items():
        selected = _selected_rows(rows, prediction, method=method)
        clairvoyant = _selected_rows(
            rows,
            actual,
            method="ClairvoyantOutcomeOracle-B29",
        )
        merged = selected.merge(
            clairvoyant[
                [
                    *POLICY_KEY_COLUMNS,
                    "squared_error_samples2",
                ]
            ].rename(columns={"squared_error_samples2": "clairvoyant_loss"}),
            on=list(POLICY_KEY_COLUMNS),
            validate="one_to_one",
        )
        correlations = []
        frame = rows[list(POLICY_KEY_COLUMNS)].copy()
        frame["prediction"] = prediction
        frame["actual"] = actual
        for _, group in frame.groupby(list(POLICY_KEY_COLUMNS), sort=False):
            if (
                group["prediction"].nunique(dropna=False) < 2
                or group["actual"].nunique(dropna=False) < 2
            ):
                continue
            correlation = spearmanr(
                group["prediction"],
                group["actual"],
            ).statistic
            if math.isfinite(float(correlation)):
                correlations.append(float(correlation))
        output.append(
            {
                "method": method,
                "n_action_rows": len(rows),
                "action_loss_r2": float(r2_score(actual, prediction)),
                "mean_within_sample_spearman": (
                    float(np.mean(correlations)) if correlations else math.nan
                ),
                "top1_clairvoyant_agreement": float(
                    selected["selected_clairvoyant_action"].mean()
                ),
                "mean_regret_to_clairvoyant_samples2": float(
                    np.mean(
                        merged["squared_error_samples2"]
                        - merged["clairvoyant_loss"]
                    )
                ),
            }
        )
    return pd.DataFrame(output)


def gate_decision(
    *,
    mode: str,
    selected_method: str,
    effects: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, object]:
    overall = effects[
        effects["scope"].eq("all")
        & effects["target"].eq(selected_method)
        & effects["baseline"].isin(STATIC_BASELINES)
    ].copy()
    if len(overall) != len(STATIC_BASELINES):
        raise RuntimeError("observability gate is missing a static baseline")
    diagnostic = diagnostics[diagnostics["method"].eq(selected_method)]
    if len(diagnostic) != 1:
        raise RuntimeError("selected model diagnostic is missing")

    point_positive = bool(
        np.all(overall["paired_mse_improvement_samples2"] > 0.0)
    )
    ci_positive = bool(np.all(overall["ci95_lower_samples2"] > 0.0))
    seed_consistent = bool(np.all(overall["positive_seed_fraction"] == 1.0))
    spearman_positive = bool(
        float(diagnostic.iloc[0]["mean_within_sample_spearman"]) > 0.0
    )

    if mode == "smoke":
        status = "SMOKE_ONLY"
    elif point_positive and ci_positive and seed_consistent:
        status = "GO_OBSERVABLE"
    elif point_positive:
        status = "HOLD_POWER"
    elif spearman_positive:
        status = "HOLD_PROXY_MISMATCH"
    else:
        status = "NO_GO_OBSERVABILITY_CURRENT_FEATURES"
    return {
        "mode": mode,
        "status": status,
        "selected_method": selected_method,
        "point_improvement_positive_vs_both": point_positive,
        "ci95_lower_positive_vs_both": ci_positive,
        "positive_in_every_test_seed_vs_both": seed_consistent,
        "mean_within_sample_spearman_positive": spearman_positive,
        "clairvoyant_is_gate_evidence": False,
        "block_protocol_status": "NO_GO_PROTOCOL_IID_BLOCK",
        "route_status_after_smoke": "HOLD_OBSERVABILITY_AUDIT",
        "interpretation": (
            "Only a formal development run may change the BRSR route status. "
            "Clairvoyant outcome oracles remain descriptive lower bounds."
        ),
    }
