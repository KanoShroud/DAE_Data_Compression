"""BRSR 候选池、连续复增益边缘似然和单动作可行性工具。

本模块只复用 BRSR 已验证的波形、嵌套量化与后验更新纯函数，不修改
Stage A/B 的冻结实现。正式证书固定为一次 acquire 动作：所有方法使用
24 bit前向载荷；自适应方法另付5 bit动作索引，总计29 bit。静态计划无需
反馈，实际总通信量为24 bit，因此本阶段只评价动作选择价值，不声称同总bit。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss

from brsr_tdoa.brsr_core import (
    BayesianSpectralModel,
    DecisionAction,
    LikelihoodContext,
    NestedComplexQuantizer,
    expected_risk_after_action,
)
from brsr_tdoa.brsr_stage_b import (
    StageBObservation,
    build_stage_b_model,
    evaluate_fixed,
    public_b_scale,
    state_from_levels,
)
from brsr_tdoa.brsr_waveform import (
    WaveformConfig,
    apply_linear_fractional_delay,
    candidate_fft_values,
    design_candidate_bins,
    raised_cosine_power_at_bins,
    rrc_positive_support_max_bin,
)


PoolName = Literal[
    "Current-Fisher",
    "Schur-Fisher",
    "Global-Risk",
    "Local-Global-Pareto",
]
LikelihoodName = Literal[
    "Phase-Discrete24",
    "Phase-Marginal",
    "Linear-Marginal",
    "Linear-OracleRho",
]

BASE_PAYLOAD_BITS = 16
ACQUIRED_BIN_PAYLOAD_BITS = 8
ONE_ACTION_FEEDBACK_BITS = 5


@dataclass(frozen=True)
class FeasibilityConfig:
    mode: str
    snr_db: tuple[float, ...]
    prior_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    prior_trials_per_seed: int
    calibration_trials_per_seed: int
    evaluation_trials_per_seed: int
    score_trials_per_seed: int
    tau_step: float
    marginal_amplitude_nodes: int
    marginal_phase_nodes: int
    bootstrap_repetitions: int
    progress_every: int

    def validate(self) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.snr_db or len(set(self.snr_db)) != len(self.snr_db):
            raise ValueError("snr_db must contain unique values")
        if not np.all(np.isfinite(np.asarray(self.snr_db, dtype=float))):
            raise ValueError("snr_db must be finite")
        pools = (self.prior_seeds, self.calibration_seeds, self.evaluation_seeds)
        flattened = [seed for group in pools for seed in group]
        if any(not group for group in pools) or len(flattened) != len(set(flattened)):
            raise ValueError("seed pools must be non-empty and disjoint")
        counts = (
            self.prior_trials_per_seed,
            self.calibration_trials_per_seed,
            self.evaluation_trials_per_seed,
            self.score_trials_per_seed,
        )
        if min(counts) < 1:
            raise ValueError("trial counts must be positive")
        if self.score_trials_per_seed > self.evaluation_trials_per_seed:
            raise ValueError("score subset cannot exceed evaluation trials")
        if self.tau_step <= 0.0:
            raise ValueError("tau_step must be positive")
        if self.marginal_amplitude_nodes < 2 or self.marginal_phase_nodes < 4:
            raise ValueError("continuous marginal quadrature is too small")
        if self.bootstrap_repetitions < 20 or self.progress_every < 1:
            raise ValueError("bootstrap and progress settings are invalid")


CONFIGS = {
    "smoke": FeasibilityConfig(
        mode="smoke",
        snr_db=(-5.0, 10.0),
        prior_seeds=(2026072301,),
        calibration_seeds=(2026072311,),
        evaluation_seeds=(2026072321,),
        prior_trials_per_seed=4,
        calibration_trials_per_seed=2,
        evaluation_trials_per_seed=2,
        score_trials_per_seed=1,
        tau_step=1.0,
        marginal_amplitude_nodes=5,
        marginal_phase_nodes=64,
        bootstrap_repetitions=50,
        progress_every=1,
    ),
    "development": FeasibilityConfig(
        mode="development",
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
        prior_seeds=(2026072301, 2026072302, 2026072303),
        calibration_seeds=(2026072311, 2026072312, 2026072313),
        evaluation_seeds=(2026072321, 2026072322, 2026072323),
        prior_trials_per_seed=32,
        calibration_trials_per_seed=12,
        evaluation_trials_per_seed=40,
        score_trials_per_seed=20,
        tau_step=0.5,
        marginal_amplitude_nodes=5,
        marginal_phase_nodes=64,
        bootstrap_repetitions=2000,
        progress_every=10,
    ),
}


def config_for_mode(mode: str) -> FeasibilityConfig:
    try:
        config = CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported mode: {mode}") from exc
    config.validate()
    return config


def projected_information(bin_indices: np.ndarray, power: np.ndarray, n_time: int) -> float:
    bins = np.asarray(bin_indices, dtype=int)
    weight = np.asarray(power, dtype=float)
    if bins.ndim != 1 or weight.shape != bins.shape or bins.size < 2:
        raise ValueError("projected information needs matching vectors")
    if np.any(weight <= 0.0):
        raise ValueError("power must be positive")
    omega = 2.0 * np.pi * bins / n_time
    mean = float(np.sum(weight * omega) / np.sum(weight))
    return float(np.sum(weight * (omega - mean) ** 2))


def global_ambiguity_risk(
    bin_indices: np.ndarray,
    power: np.ndarray,
    *,
    n_time: int,
    tau_width: float,
    tau_step: float,
) -> tuple[float, float]:
    """返回非零有限时延差上的平均与最大归一化歧义能量。"""

    bins = np.asarray(bin_indices, dtype=int)
    weight = np.asarray(power, dtype=float)
    if bins.size < 2 or weight.shape != bins.shape:
        raise ValueError("ambiguity risk needs at least two matching bins")
    delays = np.arange(tau_step, tau_width + 0.5 * tau_step, tau_step)
    if delays.size == 0:
        raise ValueError("delay-difference grid is empty")
    omega = 2.0 * np.pi * bins / n_time
    ambiguity = np.abs(
        np.sum(
            weight[:, None] * np.exp(-1j * omega[:, None] * delays[None, :]),
            axis=0,
        )
    ) / float(np.sum(weight))
    squared = ambiguity**2
    return float(np.mean(squared)), float(np.max(squared))


def _eligible_bins(config: WaveformConfig) -> np.ndarray:
    support = rrc_positive_support_max_bin(config)
    bins = np.arange(1, support + 1, dtype=int)
    power = raised_cosine_power_at_bins(bins, config)
    return bins[power > 1e-10]


def _valid_additions(
    selected: list[int],
    eligible: np.ndarray,
    spacing: int,
) -> np.ndarray:
    return eligible[
        np.asarray(
            [
                candidate not in selected
                and all(abs(int(candidate) - value) >= spacing for value in selected)
                for candidate in eligible
            ],
            dtype=bool,
        )
    ]


def _pool_metrics(bins: np.ndarray, config: WaveformConfig) -> dict[str, float]:
    power = np.maximum(raised_cosine_power_at_bins(bins, config), 1e-10)
    mean_risk, max_risk = global_ambiguity_risk(
        bins,
        power,
        n_time=config.n_time,
        tau_width=2.0 * max(abs(config.tau_min), abs(config.tau_max)),
        tau_step=0.25,
    )
    return {
        "projected_information": projected_information(bins, power, config.n_time),
        "mean_global_ambiguity": mean_risk,
        "max_global_ambiguity": max_risk,
        "min_theoretical_power": float(np.min(power)),
    }


def _greedy_pool(config: WaveformConfig, criterion: str) -> np.ndarray:
    eligible = _eligible_bins(config)
    selected = [int(value) for value in config.base_bins]
    while len(selected) < config.candidate_count:
        valid = _valid_additions(selected, eligible, config.min_candidate_spacing)
        if valid.size == 0:
            raise RuntimeError(f"{criterion} candidate design became infeasible")
        local_values = []
        global_values = []
        for candidate in valid:
            trial_bins = np.asarray(sorted([*selected, int(candidate)]), dtype=int)
            power = np.maximum(raised_cosine_power_at_bins(trial_bins, config), 1e-10)
            local_values.append(projected_information(trial_bins, power, config.n_time))
            global_values.append(
                global_ambiguity_risk(
                    trial_bins,
                    power,
                    n_time=config.n_time,
                    tau_width=2.0 * max(abs(config.tau_min), abs(config.tau_max)),
                    tau_step=0.25,
                )[0]
            )
        local = np.asarray(local_values)
        global_risk = np.asarray(global_values)
        if criterion == "schur":
            index = int(np.argmax(local))
        elif criterion == "global":
            order = np.lexsort((-local, global_risk))
            index = int(order[0])
        elif criterion == "pareto":
            local_span = float(np.ptp(local))
            global_span = float(np.ptp(global_risk))
            local_score = (
                np.ones_like(local)
                if local_span <= 1e-15
                else (local - np.min(local)) / local_span
            )
            global_score = (
                np.ones_like(global_risk)
                if global_span <= 1e-15
                else (np.max(global_risk) - global_risk) / global_span
            )
            distance = np.sqrt((1.0 - local_score) ** 2 + (1.0 - global_score) ** 2)
            order = np.lexsort((-local, global_risk, distance))
            index = int(order[0])
        else:
            raise ValueError(f"unknown candidate criterion: {criterion}")
        selected.append(int(valid[index]))
    result = np.asarray(sorted(selected), dtype=int)
    _validate_pool(result, config)
    return result


def _validate_pool(bins: np.ndarray, config: WaveformConfig) -> None:
    bins = np.asarray(bins, dtype=int)
    if bins.shape != (config.candidate_count,) or np.unique(bins).size != bins.size:
        raise ValueError("candidate pool must contain the requested unique bins")
    if not set(config.base_bins).issubset(set(bins.tolist())):
        raise ValueError("candidate pool lost the frozen base bins")
    support = rrc_positive_support_max_bin(config)
    if np.any(bins <= 0) or np.any(bins > support):
        raise ValueError("candidate pool lies outside the RRC support")
    differences = np.abs(bins[:, None] - bins[None, :])
    differences += np.eye(bins.size, dtype=int) * config.n_time
    if int(np.min(differences)) < config.min_candidate_spacing:
        raise ValueError("candidate pool violates the spacing constraint")


def build_candidate_pools(config: WaveformConfig) -> dict[PoolName, np.ndarray]:
    config.validate()
    pools: dict[PoolName, np.ndarray] = {
        "Current-Fisher": design_candidate_bins(config),
        "Schur-Fisher": _greedy_pool(config, "schur"),
        "Global-Risk": _greedy_pool(config, "global"),
        "Local-Global-Pareto": _greedy_pool(config, "pareto"),
    }
    for bins in pools.values():
        _validate_pool(bins, config)
    return pools


def candidate_pool_rows(
    pools: dict[PoolName, np.ndarray],
    config: WaveformConfig,
) -> list[dict[str, object]]:
    rows = []
    for name, bins in pools.items():
        metrics = _pool_metrics(bins, config)
        rows.append(
            {
                "pool": name,
                "candidate_count": len(bins),
                "base_bins": " ".join(str(value) for value in config.base_bins),
                "candidate_bins": " ".join(str(value) for value in bins),
                **metrics,
            }
        )
    return rows


def continuous_rho_quadrature(
    amplitude_nodes: int,
    phase_nodes: int,
    *,
    amplitude_min: float = 0.70,
    amplitude_max: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform amplitude and phase prior using Legendre x periodic quadrature."""

    if amplitude_nodes < 2 or phase_nodes < 4:
        raise ValueError("quadrature needs at least 2x4 nodes")
    nodes, weights = leggauss(amplitude_nodes)
    amplitudes = 0.5 * (amplitude_max - amplitude_min) * nodes + 0.5 * (
        amplitude_max + amplitude_min
    )
    amplitude_weights = weights / np.sum(weights)
    phases = np.linspace(-np.pi, np.pi, phase_nodes, endpoint=False)
    rho = (
        amplitudes[:, None] * np.exp(1j * phases[None, :])
    ).reshape(-1)
    prior = np.repeat(amplitude_weights, phase_nodes) / phase_nodes
    prior /= np.sum(prior)
    return rho.astype(np.complex128), prior.astype(float)


def _replace_rho(
    model: BayesianSpectralModel,
    rho_values: np.ndarray,
    rho_prior: np.ndarray,
) -> BayesianSpectralModel:
    updated = replace(
        model,
        rho_values=np.asarray(rho_values, dtype=np.complex128),
        rho_prior=np.asarray(rho_prior, dtype=float),
    )
    updated.validate()
    return updated


def build_likelihood_cases(
    *,
    waveform: WaveformConfig,
    bins: np.ndarray,
    source_power: np.ndarray,
    tau_step: float,
    amplitude_nodes: int,
    phase_nodes: int,
) -> dict[LikelihoodName, BayesianSpectralModel]:
    discrete = build_stage_b_model(
        n_time=waveform.n_time,
        bin_indices=bins,
        source_power=source_power,
        base_bins=waveform.base_bins,
        tau_min=waveform.tau_min,
        tau_max=waveform.tau_max,
        tau_step=tau_step,
        rho_amplitudes=(0.70, 0.80, 0.90),
        rho_phase_count=8,
    )
    rho, prior = continuous_rho_quadrature(amplitude_nodes, phase_nodes)
    marginal = _replace_rho(discrete, rho, prior)
    return {
        "Phase-Discrete24": discrete,
        "Phase-Marginal": marginal,
        "Linear-Marginal": marginal,
    }


def physical_public_scale(source_power: np.ndarray, sigma2: float) -> float:
    """Use the true public uniform-amplitude prior, independent of likelihood grid."""

    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be positive")
    amplitude_second_moment = (0.90**3 - 0.70**3) / (3.0 * (0.90 - 0.70))
    predicted = amplitude_second_moment * float(np.mean(source_power))
    return math.sqrt(predicted + sigma2)


def linear_template(
    y_a: np.ndarray,
    bins: np.ndarray,
    tau_values: np.ndarray,
    waveform: WaveformConfig,
) -> np.ndarray:
    columns = []
    for tau in tau_values:
        delayed = apply_linear_fractional_delay(
            y_a,
            float(tau),
            waveform.fractional_delay_half_len,
        )
        columns.append(candidate_fft_values(delayed, bins))
    return np.stack(columns, axis=1)


def linear_likelihood_context(
    *,
    template: np.ndarray,
    sigma2: float,
    scale: float,
    model: BayesianSpectralModel,
) -> LikelihoodContext:
    if template.shape != (model.n_bins, model.n_tau):
        raise ValueError("linear template does not match model")
    means = (
        template[:, :, None] * model.rho_values[None, None, :] / scale
    ).reshape(model.n_bins, model.n_tau * model.n_rho)
    variances = np.broadcast_to(
        (
            sigma2 * np.abs(model.rho_values)[None, :] ** 2 + sigma2
        )
        / scale**2,
        (model.n_tau, model.n_rho),
    ).reshape(1, model.n_tau * model.n_rho)
    variances = np.broadcast_to(variances, means.shape).copy()
    context = LikelihoodContext(
        mean_by_bin=means,
        variance_by_bin=variances,
        sigma_a2=sigma2,
        sigma_b2=sigma2,
        b_public_scale=scale,
    )
    context.validate(model)
    return context


def phase_likelihood_context(
    *,
    y_a_selected: np.ndarray,
    sigma2: float,
    scale: float,
    model: BayesianSpectralModel,
) -> LikelihoodContext:
    from brsr_tdoa.brsr_core import build_likelihood_context

    return build_likelihood_context(
        y_a_selected,
        model,
        sigma2,
        sigma2,
        b_public_scale=scale,
    )


def observation_from_context(
    *,
    stage: str,
    seed: int,
    trial: int,
    snr_db: float,
    true_tau: float,
    true_rho: complex,
    y_b_selected: np.ndarray,
    scale: float,
    context: LikelihoodContext,
    quantizer: NestedComplexQuantizer,
) -> StageBObservation:
    normalized = np.asarray(y_b_selected, dtype=np.complex128) / scale
    fine_real, fine_imag = quantizer.encode_fine(normalized)
    clipping = float(
        np.mean(
            (np.abs(normalized.real) >= quantizer.clip_abs)
            | (np.abs(normalized.imag) >= quantizer.clip_abs)
        )
    )
    return StageBObservation(
        stage=stage,
        seed=seed,
        trial=trial,
        snr_db=snr_db,
        true_tau=true_tau,
        true_rho=true_rho,
        fine_real_codes=fine_real,
        fine_imag_codes=fine_imag,
        context=context,
        public_b_scale=scale,
        clipping_rate=clipping,
    )


def oracle_rho_model(
    model: BayesianSpectralModel,
    true_rho: complex,
) -> BayesianSpectralModel:
    epsilon = 1e-12
    values = np.asarray(
        [true_rho * (1.0 - epsilon), true_rho * (1.0 + epsilon)],
        dtype=np.complex128,
    )
    return _replace_rho(model, values, np.asarray([0.5, 0.5]))


def base_levels(model: BayesianSpectralModel) -> tuple[int, ...]:
    return tuple(4 if position in model.base_positions else 0 for position in range(model.n_bins))


def action_levels(model: BayesianSpectralModel, position: int) -> tuple[int, ...]:
    if position in model.base_positions:
        raise ValueError("one-action certificate only acquires a non-base bin")
    levels = list(base_levels(model))
    levels[position] = 4
    return tuple(levels)


def one_action_positions(model: BayesianSpectralModel) -> tuple[int, ...]:
    return tuple(
        position for position in range(model.n_bins) if position not in model.base_positions
    )


def evaluate_actual_actions(
    observation: StageBObservation,
    *,
    quantizer: NestedComplexQuantizer,
    model: BayesianSpectralModel,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    base_result = evaluate_fixed(
        observation,
        method="Base-4bit",
        levels=base_levels(model),
        quantizer=quantizer,
        model=model,
    )
    base_error = base_result.estimate - observation.true_tau
    base_row = {
        "base_estimate": base_result.estimate,
        "base_squared_error": base_error**2,
        "base_posterior_risk": base_result.posterior_risk,
        "base_coverage": base_result.credible90_contains_true,
    }
    rows = []
    for position in one_action_positions(model):
        result = evaluate_fixed(
            observation,
            method="OneAcquire",
            levels=action_levels(model, position),
            quantizer=quantizer,
            model=model,
        )
        error = result.estimate - observation.true_tau
        rows.append(
            {
                "position": position,
                "bin_index": int(model.bin_indices[position]),
                "estimate_tau": result.estimate,
                "error_samples": error,
                "squared_error_samples2": error**2,
                "abs_error_samples": abs(error),
                "posterior_risk_samples2": result.posterior_risk,
                "posterior_nll": result.posterior_nll,
                "credible90_contains_true": result.credible90_contains_true,
                "clipping_rate": observation.clipping_rate,
                "payload_bits": BASE_PAYLOAD_BITS + ACQUIRED_BIN_PAYLOAD_BITS,
                "adaptive_feedback_bits": ONE_ACTION_FEEDBACK_BITS,
                "adaptive_total_bits": (
                    BASE_PAYLOAD_BITS
                    + ACQUIRED_BIN_PAYLOAD_BITS
                    + ONE_ACTION_FEEDBACK_BITS
                ),
            }
        )
    return base_row, rows


def expected_action_risks(
    observation: StageBObservation,
    *,
    quantizer: NestedComplexQuantizer,
    model: BayesianSpectralModel,
) -> dict[int, float]:
    state = state_from_levels(
        observation,
        base_levels(model),
        quantizer=quantizer,
        model=model,
    )
    feedback_bits = int(math.ceil(math.log2(1 + 2 * model.n_bins)))
    if feedback_bits != ONE_ACTION_FEEDBACK_BITS:
        raise RuntimeError("the 12-bin certificate requires a 5-bit action index")
    scores = {}
    for position in one_action_positions(model):
        action = DecisionAction(
            kind="acquire",
            position=position,
            target_bits=4,
            payload_bits=8,
            feedback_bits=feedback_bits,
        )
        scores[position] = expected_risk_after_action(
            state,
            action,
            quantizer,
            observation.context,
            model,
        )
    return scores


def public_scale_matches_model(
    source_power: np.ndarray,
    sigma2: float,
    model: BayesianSpectralModel,
) -> float:
    """Diagnostic difference from the model-grid dependent legacy scale."""

    return physical_public_scale(source_power, sigma2) - public_b_scale(model, sigma2)
