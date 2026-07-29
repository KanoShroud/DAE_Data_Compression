"""BRSR Stage 2A core algorithms.

Stage 2A freezes the Stage 1 waveform, candidate pool, likelihood and numerical
quadrature.  It adds a single feedback-aware decision after the 16-bit base
layer: stop, or acquire one 4-bit I/Q complex frequency sample.

The posterior temperature is a calibration-only post-processing transform of
the marginal TDOA posterior.  It does not alter the quantized observation
likelihood or reuse development labels.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm, spearmanr

from brsr_route_feasibility.feasibility import (
    ACQUIRED_BIN_PAYLOAD_BITS,
    BASE_PAYLOAD_BITS,
    ONE_ACTION_FEEDBACK_BITS,
    base_levels,
    one_action_positions,
)
from brsr_tdoa.brsr_core import (
    BayesianSpectralModel,
    DecisionAction,
    DecisionState,
    NestedComplexQuantizer,
    enumerate_action_outcomes,
    posterior_statistics,
    state_after_observed_action,
    tau_posterior,
)
from brsr_tdoa.brsr_stage_b import StageBObservation, state_from_levels


TARGET_MEAN_BITS = (21, 24, 27, 29)
PRIMARY_TARGET_BITS = 24
TEMPERATURE_BOUNDS = (0.25, 4.0)
TEMPERATURE_BOUNDARY_LOG_TOL = 1e-3


@dataclass(frozen=True)
class Stage2AConfig:
    mode: str
    snr_db: tuple[float, ...]
    prior_seeds: tuple[int, ...]
    plan_seeds: tuple[int, ...]
    policy_calibration_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    prior_trials_per_seed: int
    plan_trials_per_seed: int
    policy_calibration_trials_per_seed: int
    development_trials_per_seed: int
    tau_step: float
    marginal_amplitude_nodes: int
    marginal_phase_nodes: int
    target_mean_bits: tuple[int, ...]
    bootstrap_repetitions: int
    progress_every: int

    def validate(self) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.snr_db or len(set(self.snr_db)) != len(self.snr_db):
            raise ValueError("snr_db must contain unique values")
        seed_groups = (
            self.prior_seeds,
            self.plan_seeds,
            self.policy_calibration_seeds,
            self.development_seeds,
        )
        flattened = [seed for group in seed_groups for seed in group]
        if any(not group for group in seed_groups):
            raise ValueError("all seed groups must be non-empty")
        if len(flattened) != len(set(flattened)):
            raise ValueError("all seed pools must be disjoint")
        counts = (
            self.prior_trials_per_seed,
            self.plan_trials_per_seed,
            self.policy_calibration_trials_per_seed,
            self.development_trials_per_seed,
        )
        if min(counts) < 1:
            raise ValueError("all trial counts must be positive")
        if self.tau_step <= 0.0:
            raise ValueError("tau_step must be positive")
        if (self.marginal_amplitude_nodes, self.marginal_phase_nodes) != (5, 64):
            raise ValueError("Stage 2A must use the certified 5x64 quadrature")
        if self.target_mean_bits != TARGET_MEAN_BITS:
            raise ValueError("Stage 2A target mean bits are frozen")
        if self.bootstrap_repetitions < 20 or self.progress_every < 1:
            raise ValueError("bootstrap and progress settings are invalid")


CONFIGS = {
    "smoke": Stage2AConfig(
        mode="smoke",
        snr_db=(-5.0, 10.0),
        prior_seeds=(2026072501,),
        plan_seeds=(2026072511,),
        policy_calibration_seeds=(2026072541,),
        development_seeds=(2026072521,),
        prior_trials_per_seed=4,
        plan_trials_per_seed=3,
        policy_calibration_trials_per_seed=3,
        development_trials_per_seed=4,
        tau_step=1.0,
        marginal_amplitude_nodes=5,
        marginal_phase_nodes=64,
        target_mean_bits=TARGET_MEAN_BITS,
        bootstrap_repetitions=50,
        progress_every=1,
    ),
    "development": Stage2AConfig(
        mode="development",
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
        prior_seeds=(2026072501, 2026072502, 2026072503),
        plan_seeds=(2026072511, 2026072512, 2026072513),
        policy_calibration_seeds=(2026072541, 2026072542, 2026072543),
        development_seeds=(2026072521, 2026072522, 2026072523),
        prior_trials_per_seed=32,
        plan_trials_per_seed=24,
        policy_calibration_trials_per_seed=24,
        development_trials_per_seed=40,
        tau_step=0.5,
        marginal_amplitude_nodes=5,
        marginal_phase_nodes=64,
        target_mean_bits=TARGET_MEAN_BITS,
        bootstrap_repetitions=2000,
        progress_every=10,
    ),
}


def config_for_mode(mode: str) -> Stage2AConfig:
    try:
        config = CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported mode: {mode}") from exc
    config.validate()
    return config


@dataclass(frozen=True)
class TemperatureFit:
    snr_db: float
    temperature: float
    n_rows: int
    nll_raw: float
    nll_calibrated: float
    hit_lower_bound: bool
    hit_upper_bound: bool


@dataclass(frozen=True)
class FrozenRatePolicy:
    snr_db: float
    temperature_mode: str
    target_mean_bits: int
    lambda_per_payload_bit: float
    calibration_acquire_fraction: float
    calibration_mean_total_bits: float


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("probability vector must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("probability vector has no positive mass")
    return values / total


def temper_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature to a marginal probability vector."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    probability = _normalize_probability(probability)
    log_probability = np.full(probability.shape, -math.inf, dtype=float)
    positive = probability > 0.0
    log_probability[positive] = np.log(probability[positive]) / temperature
    normalized = np.exp(log_probability - logsumexp(log_probability))
    return _normalize_probability(normalized)


def tau_metrics_from_probability(
    probability: np.ndarray,
    *,
    model: BayesianSpectralModel,
    true_tau: float,
) -> dict[str, float | bool]:
    probability = _normalize_probability(probability)
    estimate = float(np.sum(model.tau_values * probability))
    risk = float(np.sum((model.tau_values - estimate) ** 2 * probability))
    interpolated_mass = float(
        np.interp(
            true_tau,
            model.tau_values,
            probability,
            left=probability[0],
            right=probability[-1],
        )
    )
    tau_step = float(np.median(np.diff(model.tau_values)))
    nll = -math.log(max(interpolated_mass / tau_step, 1e-300))
    cdf = np.cumsum(probability)
    lower = model.tau_values[int(np.searchsorted(cdf, 0.05, side="left"))]
    upper = model.tau_values[
        min(int(np.searchsorted(cdf, 0.95, side="left")), model.n_tau - 1)
    ]
    return {
        "estimate_tau": estimate,
        "posterior_risk_samples2": max(risk, 0.0),
        "posterior_nll": nll,
        "credible90_contains_true": bool(lower <= true_tau <= upper),
    }


def tempered_state_metrics(
    state: DecisionState,
    *,
    model: BayesianSpectralModel,
    true_tau: float,
    temperature: float,
) -> dict[str, float | bool]:
    probability = temper_probability(tau_posterior(state, model), temperature)
    return tau_metrics_from_probability(
        probability,
        model=model,
        true_tau=true_tau,
    )


def _interpolated_nll(
    probability: np.ndarray,
    *,
    tau_values: np.ndarray,
    true_tau: float,
    temperature: float,
) -> float:
    calibrated = temper_probability(probability, temperature)
    mass = float(
        np.interp(
            true_tau,
            tau_values,
            calibrated,
            left=calibrated[0],
            right=calibrated[-1],
        )
    )
    tau_step = float(np.median(np.diff(tau_values)))
    return -math.log(max(mass / tau_step, 1e-300))


def fit_temperature(
    *,
    snr_db: float,
    tau_probabilities: Iterable[np.ndarray],
    true_tau: Iterable[float],
    tau_values: np.ndarray,
) -> TemperatureFit:
    probabilities = [np.asarray(value, dtype=float) for value in tau_probabilities]
    truths = [float(value) for value in true_tau]
    if not probabilities or len(probabilities) != len(truths):
        raise ValueError("temperature calibration rows are empty or misaligned")

    lower, upper = TEMPERATURE_BOUNDS

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        values = [
            _interpolated_nll(
                probability,
                tau_values=tau_values,
                true_tau=truth,
                temperature=temperature,
            )
            for probability, truth in zip(probabilities, truths, strict=True)
        ]
        return float(np.mean(values))

    result = minimize_scalar(
        objective,
        bounds=(math.log(lower), math.log(upper)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"temperature fit failed at SNR={snr_db:g} dB")
    temperature = math.exp(float(result.x))
    return TemperatureFit(
        snr_db=float(snr_db),
        temperature=temperature,
        n_rows=len(probabilities),
        nll_raw=objective(0.0),
        nll_calibrated=objective(float(result.x)),
        hit_lower_bound=bool(
            abs(float(result.x) - math.log(lower)) <= TEMPERATURE_BOUNDARY_LOG_TOL
        ),
        hit_upper_bound=bool(
            abs(float(result.x) - math.log(upper)) <= TEMPERATURE_BOUNDARY_LOG_TOL
        ),
    )


def expected_tempered_risk_after_action(
    state: DecisionState,
    action: DecisionAction,
    *,
    quantizer: NestedComplexQuantizer,
    observation: StageBObservation,
    model: BayesianSpectralModel,
    temperature: float,
) -> float:
    expected = 0.0
    probability_sum = 0.0
    for _, _, probability, posterior in enumerate_action_outcomes(
        state,
        action,
        quantizer,
        observation.context,
    ):
        child_probability = temper_probability(
            posterior.reshape(model.n_tau, model.n_rho).sum(axis=1),
            temperature,
        )
        estimate = float(np.sum(model.tau_values * child_probability))
        risk = float(
            np.sum((model.tau_values - estimate) ** 2 * child_probability)
        )
        expected += probability * max(risk, 0.0)
        probability_sum += probability
    if not math.isclose(probability_sum, 1.0, abs_tol=2e-10):
        raise RuntimeError("action outcomes lost probability mass")
    return float(expected)


def evaluate_observation_actions(
    observation: StageBObservation,
    *,
    quantizer: NestedComplexQuantizer,
    model: BayesianSpectralModel,
    temperature: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate stop and every legal one-bin acquire action."""

    state = state_from_levels(
        observation,
        base_levels(model),
        quantizer=quantizer,
        model=model,
    )
    raw_base_estimate, raw_base_risk = posterior_statistics(state, model)
    calibrated_base = tempered_state_metrics(
        state,
        model=model,
        true_tau=observation.true_tau,
        temperature=temperature,
    )
    base_error = raw_base_estimate - observation.true_tau
    calibrated_base_error = (
        float(calibrated_base["estimate_tau"]) - observation.true_tau
    )
    base = {
        "base_raw_estimate_tau": raw_base_estimate,
        "base_raw_posterior_risk_samples2": raw_base_risk,
        "base_raw_squared_error_samples2": base_error**2,
        "base_calibrated_estimate_tau": float(
            calibrated_base["estimate_tau"]
        ),
        "base_calibrated_posterior_risk_samples2": float(
            calibrated_base["posterior_risk_samples2"]
        ),
        "base_calibrated_squared_error_samples2": calibrated_base_error**2,
        "base_calibrated_posterior_nll": float(
            calibrated_base["posterior_nll"]
        ),
        "base_calibrated_coverage90": bool(
            calibrated_base["credible90_contains_true"]
        ),
    }

    rows: list[dict[str, object]] = []
    for position in one_action_positions(model):
        action = DecisionAction(
            kind="acquire",
            position=position,
            target_bits=4,
            payload_bits=ACQUIRED_BIN_PAYLOAD_BITS,
            feedback_bits=ONE_ACTION_FEEDBACK_BITS,
        )
        raw_expected = expected_tempered_risk_after_action(
            state,
            action,
            quantizer=quantizer,
            observation=observation,
            model=model,
            temperature=1.0,
        )
        calibrated_expected = expected_tempered_risk_after_action(
            state,
            action,
            quantizer=quantizer,
            observation=observation,
            model=model,
            temperature=temperature,
        )
        child = state_after_observed_action(
            state,
            action,
            observation.fine_real_codes,
            observation.fine_imag_codes,
            quantizer,
            observation.context,
        )
        raw_child = tempered_state_metrics(
            child,
            model=model,
            true_tau=observation.true_tau,
            temperature=1.0,
        )
        calibrated_child = tempered_state_metrics(
            child,
            model=model,
            true_tau=observation.true_tau,
            temperature=temperature,
        )
        raw_error = float(raw_child["estimate_tau"]) - observation.true_tau
        calibrated_error = (
            float(calibrated_child["estimate_tau"]) - observation.true_tau
        )
        rows.append(
            {
                "position": position,
                "bin_index": int(model.bin_indices[position]),
                "raw_expected_risk_after_samples2": raw_expected,
                "calibrated_expected_risk_after_samples2": calibrated_expected,
                "raw_predicted_voi_samples2": raw_base_risk - raw_expected,
                "calibrated_predicted_voi_samples2": (
                    float(calibrated_base["posterior_risk_samples2"])
                    - calibrated_expected
                ),
                "raw_estimate_tau": float(raw_child["estimate_tau"]),
                "raw_posterior_risk_samples2": float(
                    raw_child["posterior_risk_samples2"]
                ),
                "raw_posterior_nll": float(raw_child["posterior_nll"]),
                "raw_coverage90": bool(
                    raw_child["credible90_contains_true"]
                ),
                "raw_squared_error_samples2": raw_error**2,
                "calibrated_estimate_tau": float(
                    calibrated_child["estimate_tau"]
                ),
                "calibrated_posterior_risk_samples2": float(
                    calibrated_child["posterior_risk_samples2"]
                ),
                "calibrated_posterior_nll": float(
                    calibrated_child["posterior_nll"]
                ),
                "calibrated_coverage90": bool(
                    calibrated_child["credible90_contains_true"]
                ),
                "calibrated_squared_error_samples2": calibrated_error**2,
                "payload_bits": BASE_PAYLOAD_BITS + ACQUIRED_BIN_PAYLOAD_BITS,
                "feedback_bits": ONE_ACTION_FEEDBACK_BITS,
                "total_bits": (
                    BASE_PAYLOAD_BITS
                    + ACQUIRED_BIN_PAYLOAD_BITS
                    + ONE_ACTION_FEEDBACK_BITS
                ),
            }
        )
    return base, rows


def freeze_rate_policy(
    rows: list[dict[str, object]],
    *,
    snr_db: float,
    temperature_mode: str,
    target_mean_bits: int,
) -> FrozenRatePolicy:
    if temperature_mode not in {"raw", "calibrated", "oracle"}:
        raise ValueError("unknown temperature mode")
    if target_mean_bits not in TARGET_MEAN_BITS:
        raise ValueError("unknown target mean bits")
    if not rows:
        raise ValueError("rate policy calibration rows are empty")
    if temperature_mode == "oracle":
        gains = np.asarray(
            [float(row["oracle_best_realized_gain_samples2"]) for row in rows]
        )
    else:
        gains = np.asarray(
            [float(row[f"{temperature_mode}_best_predicted_voi_samples2"]) for row in rows]
        )
    if not np.all(np.isfinite(gains)):
        raise ValueError("rate policy gains must be finite")

    desired_fraction = (target_mean_bits - 21.0) / 8.0
    desired_count = int(round(desired_fraction * len(gains)))
    desired_count = min(max(desired_count, 0), len(gains))
    if desired_count == 0:
        threshold = math.inf
    elif desired_count == len(gains):
        threshold = -math.inf
    else:
        descending = np.sort(gains)[::-1]
        threshold = 0.5 * (
            float(descending[desired_count - 1])
            + float(descending[desired_count])
        )
    acquire = gains > threshold
    if np.sum(acquire) != desired_count:
        order = np.argsort(-gains, kind="stable")
        acquire = np.zeros(len(gains), dtype=bool)
        acquire[order[:desired_count]] = True
        if desired_count == 0:
            threshold = math.inf
        elif desired_count == len(gains):
            threshold = -math.inf
        else:
            threshold = float(descending[desired_count - 1])
    lambda_per_payload_bit = (
        math.inf if math.isinf(threshold) and threshold > 0.0 else threshold / 8.0
    )
    return FrozenRatePolicy(
        snr_db=float(snr_db),
        temperature_mode=temperature_mode,
        target_mean_bits=int(target_mean_bits),
        lambda_per_payload_bit=float(lambda_per_payload_bit),
        calibration_acquire_fraction=float(np.mean(acquire)),
        calibration_mean_total_bits=float(21.0 + 8.0 * np.mean(acquire)),
    )


def select_policy_action(
    rows: list[dict[str, object]],
    *,
    temperature_mode: str,
    policy: FrozenRatePolicy,
) -> int | None:
    if not rows:
        raise ValueError("candidate action rows are empty")
    if temperature_mode == "oracle":
        key = "oracle_realized_gain_samples2"
    else:
        key = f"{temperature_mode}_predicted_voi_samples2"
    ordered = sorted(
        rows,
        key=lambda row: (-float(row[key]), int(row["position"])),
    )
    best = ordered[0]
    gain = float(best[key])
    threshold = 8.0 * policy.lambda_per_payload_bit
    if math.isinf(policy.lambda_per_payload_bit) and policy.lambda_per_payload_bit > 0:
        return None
    return int(best["position"]) if gain > threshold else None


def stable_cluster_rank(cluster_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}|{cluster_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def rate_match_assignments(
    cluster_ids: Iterable[str],
    *,
    desired_mean_bits: float,
    low_bits: int,
    high_bits: int,
    salt: str,
) -> dict[str, int]:
    ids = sorted(set(str(value) for value in cluster_ids))
    if not ids:
        raise ValueError("rate matching requires at least one cluster")
    if not low_bits <= desired_mean_bits <= high_bits or high_bits <= low_bits:
        raise ValueError("desired rate lies outside the static bracket")
    fraction = (desired_mean_bits - low_bits) / (high_bits - low_bits)
    high_count = int(round(fraction * len(ids)))
    ranked = sorted(ids, key=lambda value: stable_cluster_rank(value, salt))
    high_ids = set(ranked[:high_count])
    return {
        cluster_id: high_bits if cluster_id in high_ids else low_bits
        for cluster_id in ids
    }


def within_sample_score_diagnostics(
    rows: list[dict[str, object]],
    *,
    temperature_mode: str,
) -> dict[str, float | int | bool]:
    predicted_key = f"{temperature_mode}_predicted_voi_samples2"
    actual_key = f"{temperature_mode}_realized_gain_samples2"
    predicted = np.asarray([float(row[predicted_key]) for row in rows])
    actual = np.asarray([float(row[actual_key]) for row in rows])
    correlation = float(spearmanr(predicted, actual).statistic)
    predicted_best = int(
        sorted(
            rows,
            key=lambda row: (-float(row[predicted_key]), int(row["position"])),
        )[0]["position"]
    )
    actual_best = int(
        sorted(
            rows,
            key=lambda row: (-float(row[actual_key]), int(row["position"])),
        )[0]["position"]
    )
    return {
        "n_actions": len(rows),
        "spearman_predicted_vs_actual_gain": correlation,
        "predicted_best_position": predicted_best,
        "actual_best_position": actual_best,
        "action_match": predicted_best == actual_best,
    }


def cluster_bootstrap_effect(
    target: np.ndarray,
    baseline: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=str)
    if not (
        target.shape == baseline.shape == cluster_ids.shape
        and target.ndim == 1
        and target.size > 0
    ):
        raise ValueError("paired bootstrap arrays must be aligned one-dimensional vectors")
    unique = np.unique(cluster_ids)
    by_cluster = {
        cluster: np.flatnonzero(cluster_ids == cluster)
        for cluster in unique
    }
    cluster_mse_gains = np.asarray(
        [
            float(
                np.mean(
                    baseline[by_cluster[cluster]]
                    - target[by_cluster[cluster]]
                )
            )
            for cluster in unique
        ]
    )
    rng = np.random.default_rng(seed)
    mse_gain = np.empty(repetitions, dtype=float)
    rmse_gain = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[value] for value in sampled])
        target_rmse = math.sqrt(float(np.mean(target[indices])))
        baseline_rmse = math.sqrt(float(np.mean(baseline[indices])))
        mse_gain[index] = float(np.mean(baseline[indices] - target[indices]))
        rmse_gain[index] = baseline_rmse - target_rmse
    target_rmse = math.sqrt(float(np.mean(target)))
    baseline_rmse = math.sqrt(float(np.mean(baseline)))
    if len(cluster_mse_gains) > 1:
        cluster_sd = float(np.std(cluster_mse_gains, ddof=1))
        standard_error = cluster_sd / math.sqrt(len(cluster_mse_gains))
        approximate_mde = float(
            (norm.ppf(0.975) + norm.ppf(0.80)) * standard_error
        )
    else:
        cluster_sd = math.nan
        approximate_mde = math.nan
    return {
        "n_rows": int(target.size),
        "n_clusters": int(len(unique)),
        "target_rmse": target_rmse,
        "baseline_rmse": baseline_rmse,
        "relative_rmse_gain": (baseline_rmse - target_rmse) / baseline_rmse,
        "mse_gain": float(np.mean(baseline - target)),
        "mse_gain_ci_low": float(np.quantile(mse_gain, 0.025)),
        "mse_gain_ci_high": float(np.quantile(mse_gain, 0.975)),
        "rmse_gain": baseline_rmse - target_rmse,
        "rmse_gain_ci_low": float(np.quantile(rmse_gain, 0.025)),
        "rmse_gain_ci_high": float(np.quantile(rmse_gain, 0.975)),
        "cluster_mse_gain_sd": cluster_sd,
        "approx_mde_mse_gain_80pct": approximate_mde,
    }


def cluster_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=str)
    if values.shape != cluster_ids.shape or values.ndim != 1 or values.size == 0:
        raise ValueError("cluster mean arrays must be aligned")
    unique = np.unique(cluster_ids)
    cluster_means = np.asarray(
        [float(np.mean(values[cluster_ids == cluster])) for cluster in unique]
    )
    rng = np.random.default_rng(seed)
    samples = np.mean(
        cluster_means[
            rng.integers(0, len(cluster_means), size=(repetitions, len(cluster_means)))
        ],
        axis=1,
    )
    return {
        "n_rows": int(values.size),
        "n_clusters": int(len(unique)),
        "mean": float(np.mean(cluster_means)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)
