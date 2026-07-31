"""从动作前信息构造BRSR条件风险学习特征。"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from brsr_route_feasibility.feasibility import base_levels
from brsr_stage2_rate_adaptive.stage2a import temper_probability
from brsr_tdoa.brsr_core import (
    BayesianSpectralModel,
    NestedComplexQuantizer,
    tau_posterior,
)
from brsr_tdoa.brsr_stage_b import StageBObservation, state_from_levels


KEY_COLUMNS = ("stage", "seed", "trial", "cluster_id", "snr_db")
LABEL_COLUMNS = (
    "true_tau",
    "actual_squared_error_samples2",
    "actual_gain_vs_fixed_samples2",
    "residual_target_samples2",
    "clairvoyant_best_position",
)
BASE_FEATURE_COLUMNS = (
    "snr_db",
    "base_posterior_mean",
    "base_posterior_risk",
    "base_posterior_entropy",
    "base_top1_probability",
    "base_top2_probability",
    "base_peak_margin",
    "base_peak_distance_samples",
    "base_credible90_width_samples",
    "base_mode_count",
    "public_b_scale",
    "context_sigma_a2",
    "context_sigma_b2",
    "action_position",
    "action_bin_index",
    "action_frequency_abs_normalized",
    "action_source_power",
    "action_context_strength",
    "action_expected_risk_samples2",
    "action_predicted_voi_samples2",
    "action_risk_gap_to_min_samples2",
    "action_risk_rank",
    "physics_gain_vs_fixed_samples2",
)
FEATURE_PREFIXES = (
    "base_code_",
    "posterior_p_",
    "expected_risk_pos_",
)


def _as_observation(value: object) -> StageBObservation:
    if not isinstance(value, StageBObservation):
        raise TypeError("record observation is not a StageBObservation")
    return value


def _as_model(value: object) -> BayesianSpectralModel:
    if not isinstance(value, BayesianSpectralModel):
        raise TypeError("record model is not a BayesianSpectralModel")
    return value


def _mode_count(probability: np.ndarray) -> int:
    if probability.size < 3:
        return int(probability.size > 0)
    count = int(probability[0] > probability[1])
    count += int(probability[-1] > probability[-2])
    count += int(
        np.sum(
            (probability[1:-1] > probability[:-2])
            & (probability[1:-1] >= probability[2:])
        )
    )
    return count


def _posterior_features(
    probability: np.ndarray,
    tau_values: np.ndarray,
) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float)
    tau_values = np.asarray(tau_values, dtype=float)
    if probability.shape != tau_values.shape:
        raise ValueError("posterior and tau grid must have matching shapes")
    if np.any(probability < 0.0) or not np.all(np.isfinite(probability)):
        raise ValueError("posterior must be finite and non-negative")
    total = float(np.sum(probability))
    if total <= 0.0:
        raise ValueError("posterior has no positive mass")
    probability = probability / total

    mean = float(np.sum(tau_values * probability))
    risk = float(np.sum((tau_values - mean) ** 2 * probability))
    order = np.argsort(probability, kind="stable")[::-1]
    first = int(order[0])
    second = int(order[1]) if order.size > 1 else first
    positive = probability > 0.0
    entropy = float(-np.sum(probability[positive] * np.log(probability[positive])))
    entropy /= math.log(probability.size) if probability.size > 1 else 1.0
    cdf = np.cumsum(probability)
    lower = tau_values[int(np.searchsorted(cdf, 0.05, side="left"))]
    upper = tau_values[
        min(int(np.searchsorted(cdf, 0.95, side="left")), probability.size - 1)
    ]

    features = {
        "base_posterior_mean": mean,
        "base_posterior_risk": max(risk, 0.0),
        "base_posterior_entropy": entropy,
        "base_top1_probability": float(probability[first]),
        "base_top2_probability": float(probability[second]),
        "base_peak_margin": float(probability[first] - probability[second]),
        "base_peak_distance_samples": float(
            abs(tau_values[first] - tau_values[second])
        ),
        "base_credible90_width_samples": float(upper - lower),
        "base_mode_count": float(_mode_count(probability)),
    }
    features.update(
        {
            f"posterior_p_{index:03d}": float(value)
            for index, value in enumerate(probability)
        }
    )
    return features


def _base_code_features(
    observation: StageBObservation,
    model: BayesianSpectralModel,
    quantizer: NestedComplexQuantizer,
) -> dict[str, float]:
    levels = 2**4 - 1
    features: dict[str, float] = {}
    for ordinal, position in enumerate(model.base_positions):
        real_code = int(
            quantizer.coarsen(observation.fine_real_codes[position], 4)
        )
        imag_code = int(
            quantizer.coarsen(observation.fine_imag_codes[position], 4)
        )
        features[f"base_code_{ordinal}_real"] = 2.0 * real_code / levels - 1.0
        features[f"base_code_{ordinal}_imag"] = 2.0 * imag_code / levels - 1.0
    return features


def _record_key(record: dict[str, object]) -> tuple[object, ...]:
    return tuple(record[column] for column in KEY_COLUMNS)


def _validate_action_group(
    group: pd.DataFrame,
    model: BayesianSpectralModel,
) -> pd.DataFrame:
    ordered = group.sort_values("position", kind="stable").reset_index(drop=True)
    expected = set(range(model.n_bins)) - set(model.base_positions)
    actual = set(ordered["position"].astype(int))
    if actual != expected or len(ordered) != len(expected):
        raise RuntimeError("action rows do not contain every legal action exactly once")
    return ordered


def build_observable_action_rows(
    records: Iterable[dict[str, object]],
    action_rows: pd.DataFrame,
    *,
    quantizer: NestedComplexQuantizer,
    temperature_by_snr: dict[float, float],
    fixed_position: int,
) -> pd.DataFrame:
    """构造每个样本、每个合法动作的一行可观测特征和事后标签。"""

    indexed_actions = {
        key: group.copy()
        for key, group in action_rows.groupby(list(KEY_COLUMNS), sort=False)
    }
    output: list[dict[str, object]] = []
    for record in records:
        key = _record_key(record)
        if key not in indexed_actions:
            raise RuntimeError("record is missing its action rows")
        observation = _as_observation(record["observation"])
        model = _as_model(record["model"])
        group = _validate_action_group(indexed_actions[key], model)
        temperature = temperature_by_snr[float(record["snr_db"])]

        state = state_from_levels(
            observation,
            base_levels(model),
            quantizer=quantizer,
            model=model,
        )
        posterior = temper_probability(tau_posterior(state, model), temperature)
        sample_features = {
            **_posterior_features(posterior, model.tau_values),
            **_base_code_features(observation, model, quantizer),
            "public_b_scale": float(observation.public_b_scale),
            "context_sigma_a2": float(observation.context.sigma_a2),
            "context_sigma_b2": float(observation.context.sigma_b2),
        }

        expected_by_position = {
            int(row["position"]): float(
                row["calibrated_expected_risk_after_samples2"]
            )
            for _, row in group.iterrows()
        }
        actual_by_position = {
            int(row["position"]): float(
                row["calibrated_squared_error_samples2"]
            )
            for _, row in group.iterrows()
        }
        expected_values = np.asarray(list(expected_by_position.values()), dtype=float)
        if not np.all(np.isfinite(expected_values)):
            raise RuntimeError("predicted action risks contain non-finite values")
        fixed_expected = expected_by_position[fixed_position]
        fixed_actual = actual_by_position[fixed_position]
        clairvoyant_position = min(
            actual_by_position,
            key=lambda position: (actual_by_position[position], position),
        )
        ordered_expected = sorted(
            expected_by_position,
            key=lambda position: (expected_by_position[position], position),
        )
        rank_by_position = {
            position: rank for rank, position in enumerate(ordered_expected, start=1)
        }
        risk_vector = {
            f"expected_risk_pos_{position:02d}": value
            for position, value in expected_by_position.items()
        }

        for _, action in group.iterrows():
            position = int(action["position"])
            context_strength = float(
                np.mean(np.abs(observation.context.mean_by_bin[position]))
            )
            expected = expected_by_position[position]
            actual = actual_by_position[position]
            output.append(
                {
                    **{column: record[column] for column in KEY_COLUMNS},
                    "true_tau": float(observation.true_tau),
                    **sample_features,
                    **risk_vector,
                    "action_position": float(position),
                    "action_bin_index": float(model.bin_indices[position]),
                    "action_frequency_abs_normalized": float(
                        abs(model.omega[position]) / np.pi
                    ),
                    "action_source_power": float(model.source_power[position]),
                    "action_context_strength": context_strength,
                    "action_expected_risk_samples2": expected,
                    "action_predicted_voi_samples2": float(
                        action["calibrated_predicted_voi_samples2"]
                    ),
                    "action_risk_gap_to_min_samples2": float(
                        expected - np.min(expected_values)
                    ),
                    "action_risk_rank": float(rank_by_position[position]),
                    "physics_gain_vs_fixed_samples2": fixed_expected - expected,
                    "actual_squared_error_samples2": actual,
                    "actual_gain_vs_fixed_samples2": fixed_actual - actual,
                    "residual_target_samples2": actual - expected,
                    "clairvoyant_best_position": int(clairvoyant_position),
                }
            )
    frame = pd.DataFrame(output)
    if frame.empty:
        raise RuntimeError("observable feature table is empty")
    if frame[list(KEY_COLUMNS) + ["action_position"]].duplicated().any():
        raise RuntimeError("observable feature table contains duplicate action rows")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """返回严格白名单化的动作前特征列。"""

    columns = [
        column
        for column in frame.columns
        if column in BASE_FEATURE_COLUMNS
        or any(column.startswith(prefix) for prefix in FEATURE_PREFIXES)
    ]
    forbidden = set(columns) & set(LABEL_COLUMNS)
    if forbidden:
        raise RuntimeError(f"truth leaked into feature columns: {sorted(forbidden)}")
    if not columns:
        raise RuntimeError("no observable feature columns were found")
    values = frame[columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("observable feature matrix contains non-finite values")
    return columns
