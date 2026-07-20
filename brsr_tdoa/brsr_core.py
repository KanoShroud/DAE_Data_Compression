"""BRSR-TDOA 阶段 A 的严格有限状态贝叶斯决策核心。

该模块与主项目、PASR 和 PFRS 隔离。观测模型使用随机公共频谱、
离散未知复增益和离散 TDOA 先验。B 端采用固定公共量化范围的嵌套
I/Q 量化，使粗量化码和追加比特的似然均可由高斯区间概率精确计算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
from scipy.special import log_ndtr


@dataclass(frozen=True)
class BayesianSpectralModel:
    """有限状态无源双站频域观测模型。"""

    n_time: int
    bin_indices: np.ndarray
    omega: np.ndarray
    source_power: np.ndarray
    tau_values: np.ndarray
    tau_prior: np.ndarray
    rho_values: np.ndarray
    rho_prior: np.ndarray
    base_positions: tuple[int, ...]

    def validate(self) -> None:
        n_bins = self.bin_indices.size
        if self.n_time < 8:
            raise ValueError("n_time must be at least eight")
        if self.bin_indices.shape != (n_bins,) or np.unique(self.bin_indices).size != n_bins:
            raise ValueError("bin_indices must be one-dimensional and unique")
        if np.any(np.abs(self.bin_indices) >= self.n_time / 2):
            raise ValueError("bin_indices must lie inside the centered DFT support")
        if self.omega.shape != (n_bins,) or not np.all(np.isfinite(self.omega)):
            raise ValueError("omega must match bin_indices")
        expected_omega = 2.0 * np.pi * self.bin_indices / self.n_time
        if not np.allclose(self.omega, expected_omega, rtol=0.0, atol=1e-14):
            raise ValueError("omega must equal 2*pi*bin_indices/n_time")
        if (
            self.source_power.shape != (n_bins,)
            or np.any(self.source_power <= 0.0)
            or not np.all(np.isfinite(self.source_power))
        ):
            raise ValueError("source_power must be positive and match bin_indices")
        if self.tau_values.ndim != 1 or self.tau_values.size < 3:
            raise ValueError("tau_values must contain at least three hypotheses")
        if not np.all(np.isfinite(self.tau_values)) or np.any(
            np.diff(self.tau_values) <= 0.0
        ):
            raise ValueError("tau_values must be strictly increasing")
        if (
            self.rho_values.ndim != 1
            or self.rho_values.size < 2
            or not np.all(np.isfinite(self.rho_values.real))
            or not np.all(np.isfinite(self.rho_values.imag))
            or np.any(np.abs(self.rho_values) <= 0.0)
        ):
            raise ValueError("rho_values must be a finite non-zero one-dimensional grid")
        _validate_probability(self.tau_prior, self.tau_values.size, "tau_prior")
        _validate_probability(self.rho_prior, self.rho_values.size, "rho_prior")
        if len(self.base_positions) < 2 or len(set(self.base_positions)) != len(
            self.base_positions
        ):
            raise ValueError("base_positions must contain at least two unique bins")
        if min(self.base_positions) < 0 or max(self.base_positions) >= n_bins:
            raise ValueError("base_positions lie outside the candidate spectrum")
        if ambiguity_period_samples(self.base_positions, self) <= (
            self.tau_values[-1] - self.tau_values[0]
        ):
            raise ValueError("base_positions are not identifiable over the delay support")

    @property
    def n_bins(self) -> int:
        return int(self.bin_indices.size)

    @property
    def n_tau(self) -> int:
        return int(self.tau_values.size)

    @property
    def n_rho(self) -> int:
        return int(self.rho_values.size)

    @property
    def tau_hypotheses(self) -> np.ndarray:
        return np.repeat(self.tau_values, self.n_rho)

    @property
    def rho_hypotheses(self) -> np.ndarray:
        return np.tile(self.rho_values, self.n_tau)

    @property
    def joint_prior(self) -> np.ndarray:
        prior = np.repeat(self.tau_prior, self.n_rho) * np.tile(
            self.rho_prior, self.n_tau
        )
        return prior / np.sum(prior)


@dataclass(frozen=True)
class NestedComplexQuantizer:
    """固定公共范围的嵌套复数量化器。

    较低位宽的码字严格对应较高位宽相邻量化区间的并集，因此追加
    bit 可以作为真实逐层精化消息，而不是重新量化后的另一份观测。
    """

    max_bits: int
    clip_abs: float

    def __post_init__(self) -> None:
        if self.max_bits < 2:
            raise ValueError("max_bits must be at least two")
        if not math.isfinite(self.clip_abs) or self.clip_abs <= 0.0:
            raise ValueError("clip_abs must be positive and finite")

    def encode_fine(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.complex128)
        if values.ndim != 1:
            raise ValueError("values must be a one-dimensional complex array")
        return self._encode_real(values.real), self._encode_real(values.imag)

    def coarsen(self, fine_code: np.ndarray | int, bits: int) -> np.ndarray:
        self._validate_bits(bits)
        code = np.asarray(fine_code, dtype=np.int64)
        if np.any(code < 0) or np.any(code >= 2**self.max_bits):
            raise ValueError("fine_code lies outside the quantizer alphabet")
        return code >> (self.max_bits - bits)

    def interval(self, code: int, bits: int) -> tuple[float, float]:
        self._validate_bits(bits)
        levels = 2**bits
        if code < 0 or code >= levels:
            raise ValueError("code lies outside the requested bit-depth alphabet")
        step = 2.0 * self.clip_abs / levels
        lower = -math.inf if code == 0 else -self.clip_abs + code * step
        upper = math.inf if code == levels - 1 else -self.clip_abs + (code + 1) * step
        return lower, upper

    def child_codes(self, parent_code: int, parent_bits: int, target_bits: int) -> range:
        self._validate_bits(parent_bits)
        self._validate_bits(target_bits)
        if target_bits <= parent_bits:
            raise ValueError("target_bits must exceed parent_bits")
        if parent_code < 0 or parent_code >= 2**parent_bits:
            raise ValueError("parent_code lies outside its alphabet")
        factor = 2 ** (target_bits - parent_bits)
        start = parent_code * factor
        return range(start, start + factor)

    def probability(
        self,
        mean: np.ndarray,
        complex_variance: np.ndarray,
        real_code: int,
        imag_code: int,
        bits: int,
    ) -> np.ndarray:
        return np.exp(
            self.log_probability(
                mean,
                complex_variance,
                real_code,
                imag_code,
                bits,
            )
        )

    def log_probability(
        self,
        mean: np.ndarray,
        complex_variance: np.ndarray,
        real_code: int,
        imag_code: int,
        bits: int,
    ) -> np.ndarray:
        """返回量化码字的稳定对数概率。

        高信噪比下，错误 TDOA/复增益假设的区间概率可能远小于双精度浮点
        可直接表示的范围。使用对数概率可避免人为概率地板污染后验。
        """

        mean = np.asarray(mean, dtype=np.complex128)
        variance = np.asarray(complex_variance, dtype=float)
        if mean.shape != variance.shape:
            raise ValueError("mean and complex_variance must have the same shape")
        if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
            raise ValueError("complex_variance must be positive and finite")
        std = np.sqrt(variance / 2.0)
        real_log_probability = _normal_interval_log_probability(
            mean.real, std, *self.interval(real_code, bits)
        )
        imag_log_probability = _normal_interval_log_probability(
            mean.imag, std, *self.interval(imag_code, bits)
        )
        return real_log_probability + imag_log_probability

    def reconstruction(self, code: int, bits: int) -> float:
        lower, upper = self.interval(code, bits)
        finite_lower = -self.clip_abs if not math.isfinite(lower) else lower
        finite_upper = self.clip_abs if not math.isfinite(upper) else upper
        return 0.5 * (finite_lower + finite_upper)

    def _encode_real(self, values: np.ndarray) -> np.ndarray:
        levels = 2**self.max_bits
        normalized = (values + self.clip_abs) / (2.0 * self.clip_abs)
        return np.clip(np.floor(normalized * levels), 0, levels - 1).astype(np.int64)

    def _validate_bits(self, bits: int) -> None:
        if bits < 1 or bits > self.max_bits:
            raise ValueError(f"bits must lie in [1, {self.max_bits}]")


@dataclass(frozen=True)
class LikelihoodContext:
    """给定 A 端观测后的 B 端逐频点条件预测分布。"""

    mean_by_bin: np.ndarray
    variance_by_bin: np.ndarray
    sigma_a2: float
    sigma_b2: float
    b_public_scale: float = 1.0

    def validate(self, model: BayesianSpectralModel) -> None:
        expected = (model.n_bins, model.n_tau * model.n_rho)
        if self.mean_by_bin.shape != expected or self.variance_by_bin.shape != expected:
            raise ValueError("likelihood context has an invalid shape")
        if np.any(self.variance_by_bin <= 0.0):
            raise ValueError("predictive variances must be positive")
        if not math.isfinite(self.b_public_scale) or self.b_public_scale <= 0.0:
            raise ValueError("b_public_scale must be positive and finite")


@dataclass(frozen=True)
class DecisionAction:
    kind: Literal["acquire", "refine"]
    position: int
    target_bits: int
    payload_bits: int
    feedback_bits: int

    @property
    def total_bits(self) -> int:
        return self.payload_bits + self.feedback_bits

    @property
    def label(self) -> str:
        return f"{self.kind}:p{self.position}:b{self.target_bits}"


@dataclass(frozen=True)
class DecisionState:
    posterior: np.ndarray
    levels: tuple[int, ...]
    real_codes: tuple[int, ...]
    imag_codes: tuple[int, ...]

    def validate(self, model: BayesianSpectralModel) -> None:
        n_hypotheses = model.n_tau * model.n_rho
        if self.posterior.shape != (n_hypotheses,):
            raise ValueError("posterior shape does not match the hypothesis grid")
        if not np.all(np.isfinite(self.posterior)) or np.any(self.posterior < 0.0):
            raise ValueError("posterior must be finite and non-negative")
        if not math.isclose(float(np.sum(self.posterior)), 1.0, abs_tol=1e-10):
            raise ValueError("posterior must sum to one")
        if not (
            len(self.levels)
            == len(self.real_codes)
            == len(self.imag_codes)
            == model.n_bins
        ):
            raise ValueError("state code arrays must match the candidate bin count")
        for bits, real_code, imag_code in zip(
            self.levels, self.real_codes, self.imag_codes, strict=True
        ):
            if bits < 0:
                raise ValueError("quantization levels must be non-negative")
            if bits == 0 and (real_code != -1 or imag_code != -1):
                raise ValueError("unobserved bins must use -1 sentinel codes")
            if bits > 0 and (
                real_code < 0
                or imag_code < 0
                or real_code >= 2**bits
                or imag_code >= 2**bits
            ):
                raise ValueError("observed code lies outside its bit-depth alphabet")


@dataclass
class BitLedger:
    base_payload_bits: int = 0
    refinement_payload_bits: int = 0
    action_feedback_bits: int = 0
    stop_feedback_bits: int = 0
    header_bits: int = 0
    scale_bits: int = 0

    @property
    def total_bits(self) -> int:
        return (
            self.base_payload_bits
            + self.refinement_payload_bits
            + self.action_feedback_bits
            + self.stop_feedback_bits
            + self.header_bits
            + self.scale_bits
        )


@dataclass(frozen=True)
class PolicyResult:
    method: str
    estimate: float
    posterior_risk: float
    posterior_nll: float
    credible90_contains_true: bool
    ledger: BitLedger
    selected_levels: tuple[int, ...]
    action_trace: tuple[dict[str, float | int | str], ...]
    stopped_by_policy: bool


@dataclass(frozen=True)
class DecisionEvaluation:
    action: DecisionAction | None
    value: float
    stop_value: float
    action_rows: tuple[dict[str, float | int | str], ...]


def _validate_probability(values: np.ndarray, size: int, name: str) -> None:
    values = np.asarray(values, dtype=float)
    if (
        values.shape != (size,)
        or np.any(values < 0.0)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name} must be non-negative and match its support")
    if not math.isclose(float(np.sum(values)), 1.0, abs_tol=1e-10):
        raise ValueError(f"{name} must sum to one")


def _log_difference(log_larger: np.ndarray, log_smaller: np.ndarray) -> np.ndarray:
    """稳定计算 log(exp(log_larger) - exp(log_smaller))。"""

    difference = np.minimum(log_smaller - log_larger, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = log_larger + np.log1p(-np.exp(difference))
    return result


def _normal_interval_log_probability(
    mean: np.ndarray, std: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    lower_z = -math.inf if not math.isfinite(lower) else (lower - mean) / std
    upper_z = math.inf if not math.isfinite(upper) else (upper - mean) / std
    if not math.isfinite(lower):
        return log_ndtr(upper_z)
    if not math.isfinite(upper):
        return log_ndtr(-lower_z)

    result = np.empty_like(mean, dtype=float)
    negative = upper_z <= 0.0
    positive = lower_z >= 0.0
    crossing = ~(negative | positive)
    if np.any(negative):
        result[negative] = _log_difference(
            log_ndtr(upper_z[negative]),
            log_ndtr(lower_z[negative]),
        )
    if np.any(positive):
        result[positive] = _log_difference(
            log_ndtr(-lower_z[positive]),
            log_ndtr(-upper_z[positive]),
        )
    if np.any(crossing):
        probability = np.asarray(
            1.0
            - np.exp(log_ndtr(lower_z[crossing]))
            - np.exp(log_ndtr(-upper_z[crossing])),
            dtype=float,
        )
        result[crossing] = np.log(np.maximum(probability, np.finfo(float).tiny))
    return result


def build_default_model(
    *,
    n_time: int = 1024,
    tau_min: float = -8.0,
    tau_max: float = 8.0,
    tau_step: float = 0.5,
) -> BayesianSpectralModel:
    """构造 N 远大于时延假设数的阶段 A 默认模型。"""

    if tau_step <= 0.0:
        raise ValueError("tau_step must be positive")
    intervals = int(round((tau_max - tau_min) / tau_step))
    if not math.isclose(tau_min + intervals * tau_step, tau_max, abs_tol=1e-12):
        raise ValueError("tau range must be divisible by tau_step")
    tau_values = np.linspace(tau_min, tau_max, intervals + 1)

    # 基础频差 63 对应约 16.25 sample 的无歧义周期，略大于
    # 16-sample 物理支持；其余频点用于高分辨率精化。
    bin_indices = np.asarray((-383, -127, -32, 31, 129, 382), dtype=int)
    omega = 2.0 * np.pi * bin_indices / n_time
    normalized = bin_indices / float(np.max(np.abs(bin_indices)))
    source_power = 0.35 + np.exp(-0.5 * (normalized / 0.72) ** 2)
    source_power /= np.mean(source_power)

    amplitudes = (0.70, 0.90)
    # 四个相位均匀覆盖完整相位圆，避免窄相位先验泄露绝对相位信息。
    phases = tuple(value * np.pi / 4.0 for value in (-3.0, -1.0, 1.0, 3.0))
    rho_values = np.asarray(
        [amplitude * np.exp(1j * phase) for amplitude in amplitudes for phase in phases],
        dtype=np.complex128,
    )
    model = BayesianSpectralModel(
        n_time=n_time,
        bin_indices=bin_indices,
        omega=omega,
        source_power=source_power,
        tau_values=tau_values,
        tau_prior=np.full(tau_values.size, 1.0 / tau_values.size),
        rho_values=rho_values,
        rho_prior=np.full(rho_values.size, 1.0 / rho_values.size),
        base_positions=(2, 3),
    )
    model.validate()
    return model


def ambiguity_period_samples(
    positions: Sequence[int], model: BayesianSpectralModel
) -> float:
    positions = tuple(int(value) for value in positions)
    if len(positions) < 2:
        return 0.0
    selected = model.bin_indices[np.asarray(positions, dtype=int)]
    differences = [abs(int(value - selected[0])) for value in selected[1:]]
    gcd_value = 0
    for difference in differences:
        gcd_value = math.gcd(gcd_value, difference)
    if gcd_value == 0:
        return 0.0
    return float(model.n_time / gcd_value)


def build_likelihood_context(
    y_a: np.ndarray,
    model: BayesianSpectralModel,
    sigma_a2: float,
    sigma_b2: float,
    b_public_scale: float = 1.0,
) -> LikelihoodContext:
    """解析边缘化公共频谱后构造 B 端条件高斯分布。"""

    y_a = np.asarray(y_a, dtype=np.complex128)
    if y_a.shape != (model.n_bins,):
        raise ValueError("y_a must match the candidate bin count")
    if sigma_a2 <= 0.0 or sigma_b2 <= 0.0:
        raise ValueError("noise variances must be positive")
    if not math.isfinite(b_public_scale) or b_public_scale <= 0.0:
        raise ValueError("b_public_scale must be positive and finite")

    posterior_gain = model.source_power / (model.source_power + sigma_a2)
    source_mean = posterior_gain * y_a
    source_variance = model.source_power * sigma_a2 / (
        model.source_power + sigma_a2
    )
    tau_h = model.tau_hypotheses
    rho_h = model.rho_hypotheses
    steering = np.exp(-1j * model.omega[:, None] * tau_h[None, :])
    means = source_mean[:, None] * rho_h[None, :] * steering / b_public_scale
    variances = (
        source_variance[:, None] * np.abs(rho_h[None, :]) ** 2 + sigma_b2
    ) / b_public_scale**2
    context = LikelihoodContext(
        means,
        variances,
        sigma_a2,
        sigma_b2,
        b_public_scale,
    )
    context.validate(model)
    return context


def tau_posterior(state: DecisionState, model: BayesianSpectralModel) -> np.ndarray:
    posterior = state.posterior.reshape(model.n_tau, model.n_rho).sum(axis=1)
    return posterior / np.sum(posterior)


def posterior_statistics(
    state: DecisionState, model: BayesianSpectralModel
) -> tuple[float, float]:
    marginal = tau_posterior(state, model)
    mean = float(np.sum(model.tau_values * marginal))
    variance = float(np.sum((model.tau_values - mean) ** 2 * marginal))
    return mean, max(variance, 0.0)


def credible_interval_contains(
    state: DecisionState,
    model: BayesianSpectralModel,
    true_tau: float,
    probability: float = 0.90,
) -> bool:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    marginal = tau_posterior(state, model)
    cdf = np.cumsum(marginal)
    tail = 0.5 * (1.0 - probability)
    lower = model.tau_values[int(np.searchsorted(cdf, tail, side="left"))]
    upper = model.tau_values[
        min(int(np.searchsorted(cdf, 1.0 - tail, side="left")), model.n_tau - 1)
    ]
    return bool(lower <= true_tau <= upper)


def true_tau_nll(
    state: DecisionState, model: BayesianSpectralModel, true_tau: float
) -> float:
    matches = np.flatnonzero(np.isclose(model.tau_values, true_tau, atol=1e-12))
    if matches.size != 1:
        raise ValueError("true_tau must be one of the finite delay hypotheses")
    probability = tau_posterior(state, model)[int(matches[0])]
    return float(-math.log(max(float(probability), 1e-300)))


def initialize_state(
    *,
    fine_real_codes: np.ndarray,
    fine_imag_codes: np.ndarray,
    coarse_bits: int,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
) -> DecisionState:
    fine_real_codes = np.asarray(fine_real_codes, dtype=np.int64)
    fine_imag_codes = np.asarray(fine_imag_codes, dtype=np.int64)
    if fine_real_codes.shape != (model.n_bins,) or fine_imag_codes.shape != (
        model.n_bins,
    ):
        raise ValueError("fine code arrays must match the candidate bin count")
    levels = [0] * model.n_bins
    real_codes = [-1] * model.n_bins
    imag_codes = [-1] * model.n_bins
    posterior = model.joint_prior.copy()
    for position in model.base_positions:
        real_code = int(quantizer.coarsen(fine_real_codes[position], coarse_bits))
        imag_code = int(quantizer.coarsen(fine_imag_codes[position], coarse_bits))
        log_likelihood = quantizer.log_probability(
            context.mean_by_bin[position],
            context.variance_by_bin[position],
            real_code,
            imag_code,
            coarse_bits,
        )
        posterior = _bayes_update_log(posterior, log_likelihood)
        levels[position] = coarse_bits
        real_codes[position] = real_code
        imag_codes[position] = imag_code
    state = DecisionState(
        posterior=posterior,
        levels=tuple(levels),
        real_codes=tuple(real_codes),
        imag_codes=tuple(imag_codes),
    )
    state.validate(model)
    return state


def available_actions(
    state: DecisionState,
    *,
    coarse_bits: int,
    max_bits: int,
    feedback_bits: int,
    model: BayesianSpectralModel,
) -> tuple[DecisionAction, ...]:
    actions = []
    for position, current_bits in enumerate(state.levels):
        if current_bits == 0:
            actions.append(
                DecisionAction(
                    kind="acquire",
                    position=position,
                    target_bits=coarse_bits,
                    payload_bits=2 * coarse_bits,
                    feedback_bits=feedback_bits,
                )
            )
        elif current_bits < max_bits:
            actions.append(
                DecisionAction(
                    kind="refine",
                    position=position,
                    target_bits=max_bits,
                    payload_bits=2 * (max_bits - current_bits),
                    feedback_bits=feedback_bits,
                )
            )
    return tuple(actions)


def enumerate_action_outcomes(
    state: DecisionState,
    action: DecisionAction,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
) -> tuple[tuple[int, int, float, np.ndarray], ...]:
    position = action.position
    current_bits = state.levels[position]
    if action.kind == "acquire":
        if current_bits != 0:
            raise ValueError("acquire action requires an unobserved bin")
        real_codes = range(2**action.target_bits)
        imag_codes = range(2**action.target_bits)
        parent_log_likelihood = np.zeros_like(state.posterior)
    else:
        if current_bits <= 0 or action.target_bits <= current_bits:
            raise ValueError("refine action requires a coarser observed code")
        real_codes = quantizer.child_codes(
            state.real_codes[position], current_bits, action.target_bits
        )
        imag_codes = quantizer.child_codes(
            state.imag_codes[position], current_bits, action.target_bits
        )
        parent_log_likelihood = quantizer.log_probability(
            context.mean_by_bin[position],
            context.variance_by_bin[position],
            state.real_codes[position],
            state.imag_codes[position],
            current_bits,
        )

    outcomes = []
    probability_sum = 0.0
    for real_code in real_codes:
        for imag_code in imag_codes:
            log_likelihood = quantizer.log_probability(
                context.mean_by_bin[position],
                context.variance_by_bin[position],
                int(real_code),
                int(imag_code),
                action.target_bits,
            )
            log_incremental = np.minimum(
                log_likelihood - parent_log_likelihood,
                0.0,
            )
            incremental = np.exp(log_incremental)
            predictive_probability = float(np.dot(state.posterior, incremental))
            if predictive_probability <= 0.0:
                continue
            posterior = _bayes_update_log(state.posterior, log_incremental)
            outcomes.append(
                (int(real_code), int(imag_code), predictive_probability, posterior)
            )
            probability_sum += predictive_probability
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=2e-10):
        raise RuntimeError(
            f"quantized outcome probabilities sum to {probability_sum:.12g}, not one"
        )
    return tuple(outcomes)


def apply_outcome(
    state: DecisionState,
    action: DecisionAction,
    real_code: int,
    imag_code: int,
    posterior: np.ndarray,
) -> DecisionState:
    levels = list(state.levels)
    real_codes = list(state.real_codes)
    imag_codes = list(state.imag_codes)
    levels[action.position] = action.target_bits
    real_codes[action.position] = real_code
    imag_codes[action.position] = imag_code
    return DecisionState(
        posterior=posterior,
        levels=tuple(levels),
        real_codes=tuple(real_codes),
        imag_codes=tuple(imag_codes),
    )


def expected_risk_after_action(
    state: DecisionState,
    action: DecisionAction,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
) -> float:
    expected = 0.0
    for real_code, imag_code, probability, posterior in enumerate_action_outcomes(
        state, action, quantizer, context
    ):
        child = apply_outcome(state, action, real_code, imag_code, posterior)
        expected += probability * posterior_statistics(child, model)[1]
    return float(expected)


def evaluate_one_step(
    state: DecisionState,
    *,
    lambda_per_bit: float,
    stop_feedback_bits: int,
    coarse_bits: int,
    max_bits: int,
    action_feedback_bits: int,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
) -> DecisionEvaluation:
    risk = posterior_statistics(state, model)[1]
    stop_value = risk + lambda_per_bit * stop_feedback_bits
    best_value = stop_value
    best_action = None
    rows = []
    for action in available_actions(
        state,
        coarse_bits=coarse_bits,
        max_bits=max_bits,
        feedback_bits=action_feedback_bits,
        model=model,
    ):
        expected_risk = expected_risk_after_action(
            state, action, quantizer, context, model
        )
        value = (
            lambda_per_bit * action.total_bits
            + expected_risk
            + lambda_per_bit * stop_feedback_bits
        )
        rows.append(
            {
                "action": action.label,
                "expected_risk_after": expected_risk,
                "expected_risk_reduction": risk - expected_risk,
                "action_bits": action.total_bits,
                "objective_value": value,
            }
        )
        if value < best_value - 1e-14:
            best_value = value
            best_action = action
    return DecisionEvaluation(best_action, best_value, stop_value, tuple(rows))


def evaluate_dynamic_program(
    state: DecisionState,
    *,
    depth: int,
    lambda_per_bit: float,
    stop_feedback_bits: int,
    coarse_bits: int,
    max_bits: int,
    action_feedback_bits: int,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
) -> DecisionEvaluation:
    """有限深度精确贝尔曼递归，作为同决策深度的一步策略基准。"""

    if depth < 0:
        raise ValueError("depth must be non-negative")
    risk = posterior_statistics(state, model)[1]
    stop_value = risk + lambda_per_bit * stop_feedback_bits
    if depth == 0:
        return DecisionEvaluation(None, stop_value, stop_value, tuple())

    best_value = stop_value
    best_action = None
    rows = []
    for action in available_actions(
        state,
        coarse_bits=coarse_bits,
        max_bits=max_bits,
        feedback_bits=action_feedback_bits,
        model=model,
    ):
        expected_future = 0.0
        for real_code, imag_code, probability, posterior in enumerate_action_outcomes(
            state, action, quantizer, context
        ):
            child = apply_outcome(state, action, real_code, imag_code, posterior)
            child_value = evaluate_dynamic_program(
                child,
                depth=depth - 1,
                lambda_per_bit=lambda_per_bit,
                stop_feedback_bits=stop_feedback_bits,
                coarse_bits=coarse_bits,
                max_bits=max_bits,
                action_feedback_bits=action_feedback_bits,
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
                "action_bits": action.total_bits,
                "objective_value": value,
            }
        )
        if value < best_value - 1e-14:
            best_value = value
            best_action = action
    return DecisionEvaluation(best_action, best_value, stop_value, tuple(rows))


def observed_action_outcome(
    state: DecisionState,
    action: DecisionAction,
    *,
    fine_real_codes: np.ndarray,
    fine_imag_codes: np.ndarray,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
) -> tuple[int, int, np.ndarray]:
    real_code = int(
        quantizer.coarsen(fine_real_codes[action.position], action.target_bits)
    )
    imag_code = int(
        quantizer.coarsen(fine_imag_codes[action.position], action.target_bits)
    )
    current_bits = state.levels[action.position]
    log_likelihood = quantizer.log_probability(
        context.mean_by_bin[action.position],
        context.variance_by_bin[action.position],
        real_code,
        imag_code,
        action.target_bits,
    )
    if current_bits > 0:
        parent_log_likelihood = quantizer.log_probability(
            context.mean_by_bin[action.position],
            context.variance_by_bin[action.position],
            state.real_codes[action.position],
            state.imag_codes[action.position],
            current_bits,
        )
        log_likelihood = np.minimum(log_likelihood - parent_log_likelihood, 0.0)
    return real_code, imag_code, _bayes_update_log(state.posterior, log_likelihood)


def run_dynamic_policy(
    *,
    method: Literal["greedy", "dynamic_program"],
    initial_state: DecisionState,
    fine_real_codes: np.ndarray,
    fine_imag_codes: np.ndarray,
    true_tau: float,
    lambda_per_bit: float,
    max_decisions: int,
    dp_horizon: int,
    coarse_bits: int,
    max_bits: int,
    action_feedback_bits: int,
    stop_feedback_bits: int,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
) -> PolicyResult:
    if method not in {"greedy", "dynamic_program"}:
        raise ValueError(f"unsupported dynamic policy: {method}")
    if max_decisions < 0 or dp_horizon < 1:
        raise ValueError("max_decisions must be non-negative and dp_horizon positive")
    state = initial_state
    ledger = BitLedger(
        base_payload_bits=len(model.base_positions) * 2 * coarse_bits
    )
    trace = []
    stopped_by_policy = False

    for decision_index in range(max_decisions):
        if method == "greedy":
            evaluation = evaluate_one_step(
                state,
                lambda_per_bit=lambda_per_bit,
                stop_feedback_bits=stop_feedback_bits,
                coarse_bits=coarse_bits,
                max_bits=max_bits,
                action_feedback_bits=action_feedback_bits,
                quantizer=quantizer,
                context=context,
                model=model,
            )
        else:
            evaluation = evaluate_dynamic_program(
                state,
                depth=min(dp_horizon, max_decisions - decision_index),
                lambda_per_bit=lambda_per_bit,
                stop_feedback_bits=stop_feedback_bits,
                coarse_bits=coarse_bits,
                max_bits=max_bits,
                action_feedback_bits=action_feedback_bits,
                quantizer=quantizer,
                context=context,
                model=model,
            )
        if evaluation.action is None:
            ledger.stop_feedback_bits += stop_feedback_bits
            stopped_by_policy = True
            trace.append(
                {
                    "decision": decision_index,
                    "action": "stop",
                    "value": evaluation.value,
                    "stop_value": evaluation.stop_value,
                }
            )
            break

        action = evaluation.action
        real_code, imag_code, posterior = observed_action_outcome(
            state,
            action,
            fine_real_codes=fine_real_codes,
            fine_imag_codes=fine_imag_codes,
            quantizer=quantizer,
            context=context,
        )
        risk_before = posterior_statistics(state, model)[1]
        state = apply_outcome(state, action, real_code, imag_code, posterior)
        risk_after = posterior_statistics(state, model)[1]
        ledger.refinement_payload_bits += action.payload_bits
        ledger.action_feedback_bits += action.feedback_bits
        trace.append(
            {
                "decision": decision_index,
                "action": action.label,
                "risk_before": risk_before,
                "risk_after": risk_after,
                "realized_posterior_risk_reduction": risk_before - risk_after,
                "action_bits": action.total_bits,
                "value": evaluation.value,
            }
        )
    else:
        ledger.stop_feedback_bits += stop_feedback_bits

    estimate, risk = posterior_statistics(state, model)
    return PolicyResult(
        method=method,
        estimate=estimate,
        posterior_risk=risk,
        posterior_nll=true_tau_nll(state, model, true_tau),
        credible90_contains_true=credible_interval_contains(
            state, model, true_tau, 0.90
        ),
        ledger=ledger,
        selected_levels=state.levels,
        action_trace=tuple(trace),
        stopped_by_policy=stopped_by_policy,
    )


def evaluate_fixed_levels(
    *,
    name: str,
    levels: Sequence[int],
    fine_real_codes: np.ndarray,
    fine_imag_codes: np.ndarray,
    true_tau: float,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
    model: BayesianSpectralModel,
) -> PolicyResult:
    levels = tuple(int(value) for value in levels)
    if len(levels) != model.n_bins:
        raise ValueError("fixed levels must match the candidate bin count")
    posterior = model.joint_prior.copy()
    real_codes = [-1] * model.n_bins
    imag_codes = [-1] * model.n_bins
    payload_bits = 0
    for position, bits in enumerate(levels):
        if bits == 0:
            continue
        real_code = int(quantizer.coarsen(fine_real_codes[position], bits))
        imag_code = int(quantizer.coarsen(fine_imag_codes[position], bits))
        log_likelihood = quantizer.log_probability(
            context.mean_by_bin[position],
            context.variance_by_bin[position],
            real_code,
            imag_code,
            bits,
        )
        posterior = _bayes_update_log(posterior, log_likelihood)
        real_codes[position] = real_code
        imag_codes[position] = imag_code
        payload_bits += 2 * bits
    state = DecisionState(posterior, levels, tuple(real_codes), tuple(imag_codes))
    state.validate(model)
    estimate, risk = posterior_statistics(state, model)
    return PolicyResult(
        method=name,
        estimate=estimate,
        posterior_risk=risk,
        posterior_nll=true_tau_nll(state, model, true_tau),
        credible90_contains_true=credible_interval_contains(
            state, model, true_tau, 0.90
        ),
        ledger=BitLedger(base_payload_bits=payload_bits),
        selected_levels=levels,
        action_trace=tuple(),
        stopped_by_policy=False,
    )


def state_after_observed_action(
    state: DecisionState,
    action: DecisionAction,
    fine_real_codes: np.ndarray,
    fine_imag_codes: np.ndarray,
    quantizer: NestedComplexQuantizer,
    context: LikelihoodContext,
) -> DecisionState:
    real_code, imag_code, posterior = observed_action_outcome(
        state,
        action,
        fine_real_codes=fine_real_codes,
        fine_imag_codes=fine_imag_codes,
        quantizer=quantizer,
        context=context,
    )
    return apply_outcome(state, action, real_code, imag_code, posterior)


def _bayes_update_log(
    posterior: np.ndarray, log_likelihood: np.ndarray
) -> np.ndarray:
    posterior = np.asarray(posterior, dtype=float)
    log_likelihood = np.asarray(log_likelihood, dtype=float)
    if posterior.shape != log_likelihood.shape:
        raise ValueError("posterior and log_likelihood must have the same shape")
    log_updated = np.full(posterior.shape, -math.inf, dtype=float)
    positive = posterior > 0.0
    log_updated[positive] = np.log(posterior[positive]) + log_likelihood[positive]
    maximum = float(np.max(log_updated))
    if not math.isfinite(maximum):
        raise RuntimeError("Bayesian posterior normalization failed")
    updated = np.exp(log_updated - maximum)
    total = float(np.sum(updated))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("Bayesian posterior normalization failed")
    return updated / total


def replace_state_posterior(
    state: DecisionState, posterior: np.ndarray
) -> DecisionState:
    """测试和诊断使用的显式后验替换辅助函数。"""

    return replace(state, posterior=np.asarray(posterior, dtype=float))
