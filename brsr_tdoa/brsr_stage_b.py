"""BRSR-TDOA Stage B 的工作后验、4->8 bit协议和公平基线。

Stage B 使用结构化 BPSK-RRC 波形，因此逐频点复高斯分布只是由
development 冻结的工作模型，而不是严格真实后验。本模块不修改 Stage A
核心；提前停止收费，达到硬决策上限后自动终止且不再发送停止码。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Literal, Sequence

import numpy as np
from scipy.stats import truncnorm

from brsr_tdoa.brsr_core import (
    BayesianSpectralModel,
    BitLedger,
    DecisionAction,
    DecisionEvaluation,
    DecisionState,
    LikelihoodContext,
    NestedComplexQuantizer,
    apply_outcome,
    available_actions,
    build_likelihood_context,
    enumerate_action_outcomes,
    initialize_state,
    observed_action_outcome,
    posterior_statistics,
    tau_posterior,
)


@dataclass(frozen=True)
class StageBProtocol:
    coarse_bits: int = 4
    max_bits: int = 8
    max_decisions: int = 2
    quantizer_clip_abs: float = 4.0
    target_mean_bits: tuple[int, ...] = (24, 32, 40)
    rollout_samples: int = 16
    rollout_first_action_candidates: int = 3

    def validate(self, n_bins: int) -> None:
        if self.coarse_bits != 4 or self.max_bits != 8:
            raise ValueError("Stage B protocol is frozen to 4->8 bit quantization")
        if self.max_decisions != 2:
            raise ValueError("Stage B protocol is frozen to two decisions")
        if self.quantizer_clip_abs <= 0.0:
            raise ValueError("quantizer_clip_abs must be positive")
        if n_bins < 4:
            raise ValueError("at least four candidate bins are required")
        if not self.target_mean_bits or any(
            value <= 0 for value in self.target_mean_bits
        ):
            raise ValueError("target_mean_bits must contain positive values")
        if self.rollout_samples < 1:
            raise ValueError("rollout_samples must be positive")
        if self.rollout_first_action_candidates < 1:
            raise ValueError("rollout_first_action_candidates must be positive")

    @property
    def feedback_bits(self) -> int:
        raise AttributeError("feedback_bits depends on the candidate count")

    @staticmethod
    def feedback_bits_for(n_bins: int) -> int:
        return int(math.ceil(math.log2(1 + 2 * n_bins)))


@dataclass(frozen=True)
class StageBObservation:
    stage: str
    seed: int
    trial: int
    snr_db: float
    true_tau: float
    true_rho: complex
    fine_real_codes: np.ndarray
    fine_imag_codes: np.ndarray
    context: LikelihoodContext
    public_b_scale: float
    clipping_rate: float


@dataclass(frozen=True)
class StageBResult:
    method: str
    estimate: float
    posterior_risk: float
    posterior_nll: float
    credible90_contains_true: bool
    ledger: BitLedger
    selected_levels: tuple[int, ...]
    action_trace: tuple[dict[str, float | int | str], ...]
    stopped_by_policy: bool

    @property
    def total_bits(self) -> int:
        return self.ledger.total_bits


def build_stage_b_model(
    *,
    n_time: int,
    bin_indices: np.ndarray,
    source_power: np.ndarray,
    base_bins: tuple[int, int],
    tau_min: float,
    tau_max: float,
    tau_step: float,
    rho_amplitudes: tuple[float, ...] = (0.70, 0.90),
    rho_phase_count: int = 8,
) -> BayesianSpectralModel:
    if tau_step <= 0.0:
        raise ValueError("tau_step must be positive")
    intervals = int(round((tau_max - tau_min) / tau_step))
    if not math.isclose(tau_min + intervals * tau_step, tau_max, abs_tol=1e-12):
        raise ValueError("tau support must be divisible by tau_step")
    tau_values = np.linspace(tau_min, tau_max, intervals + 1)
    bins = np.asarray(bin_indices, dtype=int)
    power = np.asarray(source_power, dtype=float)
    if bins.ndim != 1 or power.shape != bins.shape:
        raise ValueError("bin_indices and source_power must have matching vectors")
    if any(value not in bins for value in base_bins):
        raise ValueError("base_bins must be members of bin_indices")
    base_positions = tuple(int(np.flatnonzero(bins == value)[0]) for value in base_bins)

    phases = np.linspace(-np.pi, np.pi, rho_phase_count, endpoint=False)
    rho_values = np.asarray(
        [
            amplitude * np.exp(1j * phase)
            for amplitude in rho_amplitudes
            for phase in phases
        ],
        dtype=np.complex128,
    )
    model = BayesianSpectralModel(
        n_time=n_time,
        bin_indices=bins,
        omega=2.0 * np.pi * bins / n_time,
        source_power=np.maximum(power, 1e-8),
        tau_values=tau_values,
        tau_prior=np.full(tau_values.size, 1.0 / tau_values.size),
        rho_values=rho_values,
        rho_prior=np.full(rho_values.size, 1.0 / rho_values.size),
        base_positions=base_positions,
    )
    model.validate()
    return model


def public_b_scale(model: BayesianSpectralModel, sigma2: float) -> float:
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be positive")
    rho_second_moment = float(
        np.sum(model.rho_prior * np.abs(model.rho_values) ** 2)
    )
    predicted_power = rho_second_moment * float(np.mean(model.source_power))
    return math.sqrt(predicted_power + sigma2)


def make_observation(
    *,
    stage: str,
    seed: int,
    trial: int,
    snr_db: float,
    true_tau: float,
    true_rho: complex,
    y_a_selected: np.ndarray,
    y_b_selected: np.ndarray,
    sigma2: float,
    model: BayesianSpectralModel,
    quantizer: NestedComplexQuantizer,
) -> StageBObservation:
    scale = public_b_scale(model, sigma2)
    normalized_b = np.asarray(y_b_selected, dtype=np.complex128) / scale
    fine_real, fine_imag = quantizer.encode_fine(normalized_b)
    clipping = float(
        np.mean(
            (np.abs(normalized_b.real) >= quantizer.clip_abs)
            | (np.abs(normalized_b.imag) >= quantizer.clip_abs)
        )
    )
    context = build_likelihood_context(
        np.asarray(y_a_selected, dtype=np.complex128),
        model,
        sigma2,
        sigma2,
        b_public_scale=scale,
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


def _tau_metrics(
    state: DecisionState,
    model: BayesianSpectralModel,
    true_tau: float,
) -> tuple[float, float, float, bool]:
    estimate, risk = posterior_statistics(state, model)
    marginal = tau_posterior(state, model)
    interpolated_mass = float(
        np.interp(
            true_tau,
            model.tau_values,
            marginal,
            left=marginal[0],
            right=marginal[-1],
        )
    )
    tau_step = float(np.median(np.diff(model.tau_values)))
    interpolated_density = interpolated_mass / tau_step
    nll = -math.log(max(interpolated_density, 1e-300))
    cdf = np.cumsum(marginal)
    lower = model.tau_values[int(np.searchsorted(cdf, 0.05, side="left"))]
    upper_index = min(
        int(np.searchsorted(cdf, 0.95, side="left")),
        model.n_tau - 1,
    )
    upper = model.tau_values[upper_index]
    return estimate, risk, nll, bool(lower <= true_tau <= upper)


def _posterior_risk(
    posterior: np.ndarray,
    model: BayesianSpectralModel,
) -> float:
    posterior = np.asarray(posterior, dtype=float)
    expected_shape = (model.n_tau * model.n_rho,)
    if posterior.shape != expected_shape:
        raise ValueError("posterior shape does not match the hypothesis grid")
    if np.any(posterior < 0.0) or not np.all(np.isfinite(posterior)):
        raise ValueError("posterior must be finite and non-negative")
    marginal = posterior.reshape(model.n_tau, model.n_rho).sum(axis=1)
    total = float(np.sum(marginal))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("posterior must contain positive probability mass")
    marginal /= total
    mean = float(np.sum(model.tau_values * marginal))
    variance = float(np.sum((model.tau_values - mean) ** 2 * marginal))
    return max(variance, 0.0)


def _initial_state(
    observation: StageBObservation,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    model: BayesianSpectralModel,
) -> DecisionState:
    return initialize_state(
        fine_real_codes=observation.fine_real_codes,
        fine_imag_codes=observation.fine_imag_codes,
        coarse_bits=protocol.coarse_bits,
        quantizer=quantizer,
        context=observation.context,
        model=model,
    )


def _evaluate_one_step(
    state: DecisionState,
    *,
    lambda_per_bit: float,
    terminal_after_action: bool,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
) -> DecisionEvaluation:
    feedback_bits = protocol.feedback_bits_for(model.n_bins)
    risk = posterior_statistics(state, model)[1]
    stop_value = risk + lambda_per_bit * feedback_bits
    best_value = stop_value
    best_action = None
    rows: list[dict[str, float | int | str]] = []
    for action in available_actions(
        state,
        coarse_bits=protocol.coarse_bits,
        max_bits=protocol.max_bits,
        feedback_bits=feedback_bits,
        model=model,
    ):
        expected_risk = 0.0
        for _, _, probability, posterior in enumerate_action_outcomes(
            state,
            action,
            quantizer,
            context,
        ):
            expected_risk += probability * _posterior_risk(posterior, model)
        future_stop_bits = 0 if terminal_after_action else feedback_bits
        value = (
            expected_risk
            + lambda_per_bit * (action.total_bits + future_stop_bits)
        )
        rows.append(
            {
                "action": action.label,
                "expected_risk_after": expected_risk,
                "action_bits": action.total_bits,
                "objective_value": value,
            }
        )
        if value < best_value - 1e-14:
            best_value = value
            best_action = action
    return DecisionEvaluation(best_action, best_value, stop_value, tuple(rows))


def _sample_truncated_normal(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    lower: float,
    upper: float,
    uniforms: np.ndarray,
) -> np.ndarray:
    alpha = (lower - mean) / std
    beta = (upper - mean) / std
    values = truncnorm.ppf(
        uniforms,
        alpha,
        beta,
        loc=mean,
        scale=std,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("truncated predictive sampling produced non-finite values")
    return np.asarray(values, dtype=float)


def _sample_action_outcomes(
    state: DecisionState,
    action: DecisionAction,
    *,
    sample_count: int,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, np.ndarray], ...]:
    """从当前工作后验采样量化结果，再用精确区间似然更新后验。"""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    hypotheses = rng.choice(
        state.posterior.size,
        size=sample_count,
        p=state.posterior,
    )
    position = action.position
    means = context.mean_by_bin[position, hypotheses]
    std = np.sqrt(context.variance_by_bin[position, hypotheses] / 2.0)
    current_bits = state.levels[position]
    if current_bits == 0:
        real_values = means.real + std * rng.standard_normal(sample_count)
        imag_values = means.imag + std * rng.standard_normal(sample_count)
    else:
        real_lower, real_upper = quantizer.interval(
            state.real_codes[position],
            current_bits,
        )
        imag_lower, imag_upper = quantizer.interval(
            state.imag_codes[position],
            current_bits,
        )
        real_values = _sample_truncated_normal(
            mean=means.real,
            std=std,
            lower=real_lower,
            upper=real_upper,
            uniforms=rng.random(sample_count),
        )
        imag_values = _sample_truncated_normal(
            mean=means.imag,
            std=std,
            lower=imag_lower,
            upper=imag_upper,
            uniforms=rng.random(sample_count),
        )

    fine_real, fine_imag = quantizer.encode_fine(real_values + 1j * imag_values)
    real_codes = quantizer.coarsen(fine_real, action.target_bits)
    imag_codes = quantizer.coarsen(fine_imag, action.target_bits)
    parent_log_likelihood = np.zeros_like(state.posterior)
    if current_bits > 0:
        parent_log_likelihood = quantizer.log_probability(
            context.mean_by_bin[position],
            context.variance_by_bin[position],
            state.real_codes[position],
            state.imag_codes[position],
            current_bits,
        )

    sampled = []
    for real_code, imag_code in zip(real_codes, imag_codes, strict=True):
        log_likelihood = quantizer.log_probability(
            context.mean_by_bin[position],
            context.variance_by_bin[position],
            int(real_code),
            int(imag_code),
            action.target_bits,
        )
        if current_bits > 0:
            log_likelihood = np.minimum(
                log_likelihood - parent_log_likelihood,
                0.0,
            )
        sampled.append(
            (
                int(real_code),
                int(imag_code),
                _log_update(state.posterior, log_likelihood),
            )
        )
    return tuple(sampled)


def _evaluate_sampled_one_step(
    state: DecisionState,
    *,
    lambda_per_bit: float,
    terminal_after_action: bool,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
    rng: np.random.Generator,
) -> DecisionEvaluation:
    feedback_bits = protocol.feedback_bits_for(model.n_bins)
    risk = posterior_statistics(state, model)[1]
    stop_value = risk + lambda_per_bit * feedback_bits
    best_value = stop_value
    best_action = None
    rows: list[dict[str, float | int | str]] = []
    for action in available_actions(
        state,
        coarse_bits=protocol.coarse_bits,
        max_bits=protocol.max_bits,
        feedback_bits=feedback_bits,
        model=model,
    ):
        sampled = _sample_action_outcomes(
            state,
            action,
            sample_count=protocol.rollout_samples,
            quantizer=quantizer,
            context=context,
            rng=rng,
        )
        expected_risk = float(
            np.mean(
                [
                    _posterior_risk(posterior, model)
                    for _, _, posterior in sampled
                ]
            )
        )
        future_stop_bits = 0 if terminal_after_action else feedback_bits
        value = expected_risk + lambda_per_bit * (
            action.total_bits + future_stop_bits
        )
        rows.append(
            {
                "action": action.label,
                "sampled_expected_risk_after": expected_risk,
                "rollout_samples": protocol.rollout_samples,
                "objective_value": value,
            }
        )
        if value < best_value - 1e-14:
            best_value = value
            best_action = action
    return DecisionEvaluation(best_action, best_value, stop_value, tuple(rows))


def _evaluate_rollout(
    state: DecisionState,
    *,
    lambda_per_bit: float,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
    rng: np.random.Generator,
) -> DecisionEvaluation:
    """用后验预测采样近似两步贝尔曼前瞻。"""

    feedback_bits = protocol.feedback_bits_for(model.n_bins)
    risk = posterior_statistics(state, model)[1]
    stop_value = risk + lambda_per_bit * feedback_bits
    actions = available_actions(
        state,
        coarse_bits=protocol.coarse_bits,
        max_bits=protocol.max_bits,
        feedback_bits=feedback_bits,
        model=model,
    )
    immediate = []
    sampled_outcomes: dict[str, tuple[tuple[int, int, np.ndarray], ...]] = {}
    for action in actions:
        sampled = _sample_action_outcomes(
            state,
            action,
            sample_count=protocol.rollout_samples,
            quantizer=quantizer,
            context=context,
            rng=rng,
        )
        sampled_outcomes[action.label] = sampled
        expected_risk = float(
            np.mean(
                [
                    _posterior_risk(posterior, model)
                    for _, _, posterior in sampled
                ]
            )
        )
        immediate.append((expected_risk, action))
    immediate.sort(key=lambda item: (item[0], item[1].label))
    candidates = [
        action
        for _, action in immediate[: protocol.rollout_first_action_candidates]
    ]

    best_value = stop_value
    best_action = None
    rows: list[dict[str, float | int | str]] = []
    for action in candidates:
        future_values = []
        for real_code, imag_code, posterior in sampled_outcomes[action.label]:
            child = apply_outcome(
                state,
                action,
                real_code,
                imag_code,
                posterior,
            )
            child_evaluation = _evaluate_sampled_one_step(
                child,
                lambda_per_bit=lambda_per_bit,
                terminal_after_action=True,
                protocol=protocol,
                quantizer=quantizer,
                context=context,
                model=model,
                rng=rng,
            )
            future_values.append(child_evaluation.value)
        value = lambda_per_bit * action.total_bits + float(np.mean(future_values))
        rows.append(
            {
                "action": action.label,
                "rollout_value": value,
                "rollout_samples": protocol.rollout_samples,
            }
        )
        if value < best_value - 1e-14:
            best_value = value
            best_action = action
    return DecisionEvaluation(best_action, best_value, stop_value, tuple(rows))


def _evaluate_exact_dp(
    state: DecisionState,
    *,
    depth: int,
    lambda_per_bit: float,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
) -> DecisionEvaluation:
    """硬上限自动终止口径下的精确有限深度贝尔曼递归。"""

    risk = posterior_statistics(state, model)[1]
    if depth == 0:
        return DecisionEvaluation(None, risk, risk, tuple())
    feedback_bits = protocol.feedback_bits_for(model.n_bins)
    stop_value = risk + lambda_per_bit * feedback_bits
    best_value = stop_value
    best_action = None
    rows: list[dict[str, float | int | str]] = []
    for action in available_actions(
        state,
        coarse_bits=protocol.coarse_bits,
        max_bits=protocol.max_bits,
        feedback_bits=feedback_bits,
        model=model,
    ):
        expected_future = 0.0
        for real_code, imag_code, probability, posterior in enumerate_action_outcomes(
            state,
            action,
            quantizer,
            context,
        ):
            child = apply_outcome(
                state,
                action,
                real_code,
                imag_code,
                posterior,
            )
            child_value = _evaluate_exact_dp(
                child,
                depth=depth - 1,
                lambda_per_bit=lambda_per_bit,
                protocol=protocol,
                quantizer=quantizer,
                context=context,
                model=model,
            ).value
            expected_future += probability * child_value
        value = lambda_per_bit * action.total_bits + expected_future
        rows.append(
            {
                "action": action.label,
                "expected_future_value": expected_future,
                "objective_value": value,
            }
        )
        if value < best_value - 1e-14:
            best_value = value
            best_action = action
    return DecisionEvaluation(best_action, best_value, stop_value, tuple(rows))


def run_policy(
    observation: StageBObservation,
    *,
    method: Literal["greedy", "rollout", "exact_dp"],
    lambda_per_bit: float,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    model: BayesianSpectralModel,
    rng: np.random.Generator,
) -> StageBResult:
    if lambda_per_bit < 0.0:
        raise ValueError("lambda_per_bit must be non-negative")
    protocol.validate(model.n_bins)
    state = _initial_state(observation, protocol, quantizer, model)
    feedback_bits = protocol.feedback_bits_for(model.n_bins)
    ledger = BitLedger(
        base_payload_bits=len(model.base_positions) * 2 * protocol.coarse_bits
    )
    trace: list[dict[str, float | int | str]] = []
    stopped = False

    for decision_index in range(protocol.max_decisions):
        remaining = protocol.max_decisions - decision_index
        if method == "rollout" and remaining == 2:
            evaluation = _evaluate_rollout(
                state,
                lambda_per_bit=lambda_per_bit,
                protocol=protocol,
                quantizer=quantizer,
                context=observation.context,
                model=model,
                rng=rng,
            )
        elif method == "exact_dp":
            evaluation = _evaluate_exact_dp(
                state,
                depth=remaining,
                lambda_per_bit=lambda_per_bit,
                protocol=protocol,
                quantizer=quantizer,
                context=observation.context,
                model=model,
            )
        else:
            evaluation = _evaluate_one_step(
                state,
                lambda_per_bit=lambda_per_bit,
                terminal_after_action=remaining == 1,
                protocol=protocol,
                quantizer=quantizer,
                context=observation.context,
                model=model,
            )

        if evaluation.action is None:
            ledger.stop_feedback_bits += feedback_bits
            stopped = True
            trace.append(
                {
                    "decision": decision_index,
                    "action": "stop",
                    "objective_value": evaluation.value,
                }
            )
            break

        action = evaluation.action
        risk_before = posterior_statistics(state, model)[1]
        selected_diagnostic = next(
            (
                item
                for item in evaluation.action_rows
                if item.get("action") == action.label
            ),
            {},
        )
        real_code, imag_code, posterior = observed_action_outcome(
            state,
            action,
            fine_real_codes=observation.fine_real_codes,
            fine_imag_codes=observation.fine_imag_codes,
            quantizer=quantizer,
            context=observation.context,
        )
        state = apply_outcome(
            state,
            action,
            real_code,
            imag_code,
            posterior,
        )
        risk_after = posterior_statistics(state, model)[1]
        ledger.refinement_payload_bits += action.payload_bits
        ledger.action_feedback_bits += action.feedback_bits
        trace.append(
            {
                "decision": decision_index,
                "action": action.label,
                "risk_before": risk_before,
                "risk_after": risk_after,
                "objective_value": evaluation.value,
                "predicted_expected_risk_after": selected_diagnostic.get(
                    "expected_risk_after",
                    selected_diagnostic.get("sampled_expected_risk_after"),
                ),
                "rollout_value": selected_diagnostic.get("rollout_value"),
            }
        )

    estimate, risk, nll, coverage = _tau_metrics(
        state,
        model,
        observation.true_tau,
    )
    return StageBResult(
        method=f"BRSR-{method}",
        estimate=estimate,
        posterior_risk=risk,
        posterior_nll=nll,
        credible90_contains_true=coverage,
        ledger=ledger,
        selected_levels=state.levels,
        action_trace=tuple(trace),
        stopped_by_policy=stopped,
    )


def _log_update(posterior: np.ndarray, log_likelihood: np.ndarray) -> np.ndarray:
    values = np.full_like(posterior, -math.inf, dtype=float)
    positive = posterior > 0.0
    values[positive] = np.log(posterior[positive]) + log_likelihood[positive]
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        raise RuntimeError("working posterior normalization failed")
    updated = np.exp(values - maximum)
    total = float(np.sum(updated))
    if total <= 0.0 or not math.isfinite(total):
        raise RuntimeError("working posterior normalization failed")
    return updated / total


def state_from_levels(
    observation: StageBObservation,
    levels: Sequence[int],
    *,
    quantizer: NestedComplexQuantizer,
    model: BayesianSpectralModel,
) -> DecisionState:
    levels = tuple(int(value) for value in levels)
    if len(levels) != model.n_bins:
        raise ValueError("levels must match the candidate count")
    posterior = model.joint_prior.copy()
    real_codes = [-1] * model.n_bins
    imag_codes = [-1] * model.n_bins
    for position, bits in enumerate(levels):
        if bits == 0:
            continue
        if bits not in (4, 8):
            raise ValueError("Stage B fixed levels must be 0, 4, or 8")
        real_code = int(quantizer.coarsen(observation.fine_real_codes[position], bits))
        imag_code = int(quantizer.coarsen(observation.fine_imag_codes[position], bits))
        posterior = _log_update(
            posterior,
            quantizer.log_probability(
                observation.context.mean_by_bin[position],
                observation.context.variance_by_bin[position],
                real_code,
                imag_code,
                bits,
            ),
        )
        real_codes[position] = real_code
        imag_codes[position] = imag_code
    state = DecisionState(
        posterior=posterior,
        levels=levels,
        real_codes=tuple(real_codes),
        imag_codes=tuple(imag_codes),
    )
    state.validate(model)
    return state


def evaluate_fixed(
    observation: StageBObservation,
    *,
    method: str,
    levels: Sequence[int],
    quantizer: NestedComplexQuantizer,
    model: BayesianSpectralModel,
) -> StageBResult:
    state = state_from_levels(
        observation,
        levels,
        quantizer=quantizer,
        model=model,
    )
    estimate, risk, nll, coverage = _tau_metrics(
        state,
        model,
        observation.true_tau,
    )
    payload = int(2 * sum(levels))
    return StageBResult(
        method=method,
        estimate=estimate,
        posterior_risk=risk,
        posterior_nll=nll,
        credible90_contains_true=coverage,
        ledger=BitLedger(base_payload_bits=payload),
        selected_levels=tuple(int(value) for value in levels),
        action_trace=tuple(),
        stopped_by_policy=False,
    )


def ambiguity_period_samples(
    levels: Sequence[int],
    model: BayesianSpectralModel,
) -> float:
    selected = model.bin_indices[np.asarray(levels, dtype=int) > 0]
    if selected.size < 2:
        return 0.0
    gcd_value = 0
    reference = int(selected[0])
    for value in selected[1:]:
        gcd_value = math.gcd(gcd_value, abs(int(value) - reference))
    return 0.0 if gcd_value == 0 else model.n_time / gcd_value


def enumerate_static_configurations(
    *,
    model: BayesianSpectralModel,
    max_payload_bits: int,
) -> list[tuple[int, ...]]:
    """枚举预算内、在TDOA支持上可辨识的0/4/8-bit静态配置。"""

    support_width = float(model.tau_values[-1] - model.tau_values[0])
    configurations = []
    for units in product((0, 1, 2), repeat=model.n_bins):
        levels = tuple(4 * value for value in units)
        payload = 2 * sum(levels)
        if payload > max_payload_bits:
            continue
        if ambiguity_period_samples(levels, model) <= support_width:
            continue
        configurations.append(levels)
    configurations.sort(key=lambda item: (2 * sum(item), item))
    return configurations


def result_row(
    observation: StageBObservation,
    result: StageBResult,
    *,
    lambda_per_bit: float | None,
    is_dp_subset: bool = False,
) -> dict[str, object]:
    error = result.estimate - observation.true_tau
    action_items = [
        item for item in result.action_trace if item.get("action") != "stop"
    ]
    actual_risk_reduction = float(
        sum(
            float(item["risk_before"]) - float(item["risk_after"])
            for item in action_items
        )
    )
    first_item = result.action_trace[0] if result.action_trace else {}
    return {
        "stage": observation.stage,
        "seed": observation.seed,
        "trial": observation.trial,
        "snr_db": observation.snr_db,
        "method": result.method,
        "lambda_per_bit": lambda_per_bit,
        "true_tau": observation.true_tau,
        "true_rho_real": observation.true_rho.real,
        "true_rho_imag": observation.true_rho.imag,
        "estimate_tau": result.estimate,
        "error_samples": error,
        "squared_error_samples2": error**2,
        "abs_error_samples": abs(error),
        "posterior_risk_samples2": result.posterior_risk,
        "posterior_nll": result.posterior_nll,
        "credible90_contains_true": result.credible90_contains_true,
        "total_bits": result.total_bits,
        "base_payload_bits": result.ledger.base_payload_bits,
        "refinement_payload_bits": result.ledger.refinement_payload_bits,
        "action_feedback_bits": result.ledger.action_feedback_bits,
        "stop_feedback_bits": result.ledger.stop_feedback_bits,
        "selected_levels": " ".join(str(value) for value in result.selected_levels),
        "n_actions": len(action_items),
        "first_action": (
            str(first_item.get("action", "none"))
        ),
        "first_predicted_expected_risk_after": first_item.get(
            "predicted_expected_risk_after"
        ),
        "first_rollout_value": first_item.get("rollout_value"),
        "actual_posterior_risk_reduction": actual_risk_reduction,
        "action_trace": " | ".join(
            str(item.get("action", "none")) for item in result.action_trace
        ),
        "stopped_by_policy": result.stopped_by_policy,
        "quantizer_clipping_rate": observation.clipping_rate,
        "public_b_scale": observation.public_b_scale,
        "is_dp_subset": is_dp_subset,
    }
