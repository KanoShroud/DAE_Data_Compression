"""Core model for SITQ-TDOA Stage A.

Stage A isolates the finite-bit quantizer mechanism.  The reference receiver is
available at the fusion center without compression, while the remote receiver
sends fixed-rate quantized DFT coefficients.  The source and nuisance priors
used to generate data are identical to those used by the decoder, so this stage
tests the quantizer objective without waveform-model mismatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.special import logsumexp, ndtr


Objective = Literal["quantization_mse", "posterior_entropy", "bayes_risk"]


@dataclass(frozen=True)
class ModelConfig:
    """Frozen physical and statistical assumptions for Stage A."""

    n_time: int = 1024
    fs_hz: float = 40e6
    tau_min: float = -8.0
    tau_max: float = 8.0
    tau_step: float = 0.25
    selected_bins: tuple[int, ...] = (31, 94, 145, 196, 247, 298)
    rrc_rolloff: float = 0.2
    samples_per_symbol: int = 2
    rho_amplitudes: tuple[float, ...] = (0.75, 0.90)
    rho_phase_count: int = 8
    bits_per_real: int = 2

    def validate(self) -> None:
        if self.n_time < 64:
            raise ValueError("n_time must be at least 64")
        if not math.isfinite(self.fs_hz) or self.fs_hz <= 0.0:
            raise ValueError("fs_hz must be positive and finite")
        if not self.tau_min < self.tau_max:
            raise ValueError("tau_min must be smaller than tau_max")
        if not math.isfinite(self.tau_step) or self.tau_step <= 0.0:
            raise ValueError("tau_step must be positive and finite")
        intervals = (self.tau_max - self.tau_min) / self.tau_step
        if not math.isclose(intervals, round(intervals), abs_tol=1e-10):
            raise ValueError("tau support must be divisible by tau_step")
        bins = np.asarray(self.selected_bins, dtype=int)
        if bins.ndim != 1 or bins.size < 2 or np.unique(bins).size != bins.size:
            raise ValueError("selected_bins must contain unique values")
        if np.any(bins <= 0) or np.any(bins >= self.n_time // 2):
            raise ValueError("Stage A uses positive non-Nyquist DFT bins")
        if np.any(bins > self.rrc_support_max_bin):
            raise ValueError("selected_bins must lie inside the RRC support")
        # An unknown common complex gain absorbs absolute phase.  Global delay
        # identifiability is therefore governed by pairwise frequency spacing,
        # not by the absolute-bin gcd.
        bin_differences = [int(value - bins[0]) for value in bins[1:]]
        spacing_gcd = math.gcd(*bin_differences)
        if self.n_time / spacing_gcd <= (self.tau_max - self.tau_min):
            raise ValueError("selected bins are ambiguous over the TDOA support")
        if not 0.0 < self.rrc_rolloff <= 1.0:
            raise ValueError("rrc_rolloff must lie in (0, 1]")
        if self.samples_per_symbol < 2:
            raise ValueError("samples_per_symbol must be at least two")
        if not self.rho_amplitudes or any(
            value <= 0.0 or not math.isfinite(value)
            for value in self.rho_amplitudes
        ):
            raise ValueError("rho amplitudes must be positive and finite")
        if self.rho_phase_count < 2:
            raise ValueError("rho_phase_count must be at least two")
        if self.bits_per_real != 2:
            raise ValueError("Stage A is frozen to a symmetric 2-bit real quantizer")

    @property
    def symbol_rate_hz(self) -> float:
        return self.fs_hz / self.samples_per_symbol

    @property
    def rrc_support_max_bin(self) -> int:
        bandwidth = (
            (1.0 + self.rrc_rolloff) * self.symbol_rate_hz / 2.0
        )
        return int(math.floor(bandwidth * self.n_time / self.fs_hz))

    @property
    def total_payload_bits(self) -> int:
        return 2 * self.bits_per_real * len(self.selected_bins)


@dataclass(frozen=True)
class TrialLatents:
    """One trajectory reused across SNR values."""

    seed: int
    trial: int
    true_tau: float
    true_rho_index: int
    source: np.ndarray
    noise_a_standard: np.ndarray
    noise_b_standard: np.ndarray


@dataclass(frozen=True)
class ObservationContext:
    """All action-independent arrays required by an exact posterior."""

    seed: int
    trial: int
    snr_db: float
    true_tau: float
    y_b_normalized: np.ndarray
    conditional_mean: np.ndarray
    conditional_variance: np.ndarray


@dataclass(frozen=True)
class PosteriorResult:
    estimate: float
    posterior_risk: float
    entropy: float
    nll: float
    credible90_contains_true: bool
    squared_error: float


@dataclass(frozen=True)
class ThresholdSearchResult:
    objective: Objective
    threshold: float
    objective_value: float
    at_search_boundary: bool
    rows: tuple[dict[str, float], ...]


def _complex_standard_normal(
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    return (
        rng.standard_normal(size) + 1j * rng.standard_normal(size)
    ) / math.sqrt(2.0)


def raised_cosine_power_at_bins(
    bins: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    """RRC-shaped source power, normalized over the selected coefficients."""

    indices = np.asarray(bins, dtype=int)
    frequencies = np.abs(indices) * config.fs_hz / config.n_time
    symbol_rate = config.symbol_rate_hz
    low = (1.0 - config.rrc_rolloff) * symbol_rate / 2.0
    high = (1.0 + config.rrc_rolloff) * symbol_rate / 2.0
    power = np.zeros(indices.size, dtype=float)
    power[frequencies <= low] = 1.0
    transition = (frequencies > low) & (frequencies < high)
    power[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (frequencies[transition] - low)
            / (config.rrc_rolloff * symbol_rate)
        )
    )
    if np.any(power <= 0.0):
        raise ValueError("selected bins contain a zero-power RRC coefficient")
    return power / float(np.mean(power))


class SITQModel:
    """Matched Bayesian frequency-domain model and exact quantized likelihood."""

    def __init__(self, config: ModelConfig) -> None:
        config.validate()
        self.config = config
        self.bins = np.asarray(config.selected_bins, dtype=int)
        self.omega = 2.0 * np.pi * self.bins / config.n_time
        self.source_power = raised_cosine_power_at_bins(self.bins, config)
        intervals = int(round((config.tau_max - config.tau_min) / config.tau_step))
        self.tau_values = np.linspace(config.tau_min, config.tau_max, intervals + 1)
        self.tau_prior = np.full(self.tau_values.size, 1.0 / self.tau_values.size)
        phases = np.linspace(-np.pi, np.pi, config.rho_phase_count, endpoint=False)
        self.rho_values = np.asarray(
            [
                amplitude * np.exp(1j * phase)
                for amplitude in config.rho_amplitudes
                for phase in phases
            ],
            dtype=np.complex128,
        )
        self.rho_prior = np.full(self.rho_values.size, 1.0 / self.rho_values.size)
        self.rho_second_moment = float(
            np.sum(self.rho_prior * np.abs(self.rho_values) ** 2)
        )
        self._phase_grid = np.exp(
            -1j * self.tau_values[:, None] * self.omega[None, :]
        )

    def generate_trials(self, seed: int, n_trials: int) -> list[TrialLatents]:
        if n_trials < 1:
            raise ValueError("n_trials must be positive")
        rng = np.random.default_rng(seed)
        trials: list[TrialLatents] = []
        for trial_index in range(n_trials):
            source = _complex_standard_normal(rng, self.bins.size) * np.sqrt(
                self.source_power
            )
            trials.append(
                TrialLatents(
                    seed=seed,
                    trial=trial_index,
                    # Stage A is deliberately model-matched: the generator uses
                    # the same finite-delay prior as the exact decoder.  A later
                    # model-mismatch stage will introduce continuous sub-grid TDOA.
                    true_tau=float(rng.choice(self.tau_values, p=self.tau_prior)),
                    true_rho_index=int(rng.integers(self.rho_values.size)),
                    source=source,
                    noise_a_standard=_complex_standard_normal(rng, self.bins.size),
                    noise_b_standard=_complex_standard_normal(rng, self.bins.size),
                )
            )
        return trials

    def build_context(
        self,
        trial: TrialLatents,
        snr_db: float,
    ) -> ObservationContext:
        if not math.isfinite(snr_db):
            raise ValueError("snr_db must be finite")
        sigma2 = 10.0 ** (-snr_db / 10.0)
        rho = self.rho_values[trial.true_rho_index]
        true_phase = np.exp(-1j * self.omega * trial.true_tau)
        y_a = trial.source + math.sqrt(sigma2) * trial.noise_a_standard
        y_b = (
            rho * true_phase * trial.source
            + math.sqrt(sigma2) * trial.noise_b_standard
        )

        public_scale = np.sqrt(
            self.rho_second_moment * self.source_power + sigma2
        )
        kappa = self.source_power / (self.source_power + sigma2)
        source_conditional_variance = (
            self.source_power * sigma2 / (self.source_power + sigma2)
        )
        conditional_mean = (
            self._phase_grid[:, None, :]
            * self.rho_values[None, :, None]
            * (kappa * y_a / public_scale)[None, None, :]
        )
        conditional_variance = (
            np.abs(self.rho_values[:, None]) ** 2
            * source_conditional_variance[None, :]
            + sigma2
        ) / public_scale[None, :] ** 2
        if np.any(conditional_variance <= 0.0):
            raise RuntimeError("conditional variance must be positive")
        return ObservationContext(
            seed=trial.seed,
            trial=trial.trial,
            snr_db=float(snr_db),
            true_tau=trial.true_tau,
            y_b_normalized=y_b / public_scale,
            conditional_mean=conditional_mean,
            conditional_variance=conditional_variance,
        )

    @staticmethod
    def quantizer_edges(threshold: float) -> np.ndarray:
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be positive and finite")
        return np.asarray([-np.inf, -threshold, 0.0, threshold, np.inf])

    @classmethod
    def encode_real(cls, values: np.ndarray, threshold: float) -> np.ndarray:
        edges = cls.quantizer_edges(threshold)
        return np.searchsorted(edges[1:-1], np.asarray(values, dtype=float), side="right")

    @staticmethod
    def _interval_probability(
        mean: np.ndarray,
        standard_deviation: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        if np.any(standard_deviation <= 0.0):
            raise ValueError("standard deviation must be positive")
        upper_z = (upper - mean) / standard_deviation
        lower_z = (lower - mean) / standard_deviation
        probability = ndtr(upper_z) - ndtr(lower_z)
        return np.maximum(probability, np.finfo(float).tiny)

    def posterior_quantized(
        self,
        context: ObservationContext,
        threshold: float,
    ) -> np.ndarray:
        edges = self.quantizer_edges(threshold)
        real_codes = self.encode_real(context.y_b_normalized.real, threshold)
        imag_codes = self.encode_real(context.y_b_normalized.imag, threshold)
        variance = context.conditional_variance[None, :, :]
        std = np.sqrt(variance / 2.0)
        log_likelihood = np.zeros(
            (self.tau_values.size, self.rho_values.size), dtype=float
        )
        for position in range(self.bins.size):
            mean = context.conditional_mean[:, :, position]
            real_probability = self._interval_probability(
                mean.real,
                std[:, :, position],
                edges[real_codes[position]],
                edges[real_codes[position] + 1],
            )
            imag_probability = self._interval_probability(
                mean.imag,
                std[:, :, position],
                edges[imag_codes[position]],
                edges[imag_codes[position] + 1],
            )
            log_likelihood += np.log(real_probability) + np.log(imag_probability)
        return self._marginal_posterior(log_likelihood)

    def posterior_unquantized(self, context: ObservationContext) -> np.ndarray:
        residual = (
            context.y_b_normalized[None, None, :]
            - context.conditional_mean
        )
        variance = context.conditional_variance[None, :, :]
        log_likelihood = -np.sum(
            np.abs(residual) ** 2 / variance + np.log(np.pi * variance),
            axis=2,
        )
        return self._marginal_posterior(log_likelihood)

    def _marginal_posterior(self, log_likelihood: np.ndarray) -> np.ndarray:
        expected_shape = (self.tau_values.size, self.rho_values.size)
        if log_likelihood.shape != expected_shape:
            raise ValueError(f"log_likelihood must have shape {expected_shape}")
        log_tau_likelihood = logsumexp(
            log_likelihood + np.log(self.rho_prior)[None, :], axis=1
        )
        log_posterior = log_tau_likelihood + np.log(self.tau_prior)
        log_posterior -= logsumexp(log_posterior)
        posterior = np.exp(log_posterior)
        if not np.all(np.isfinite(posterior)):
            raise RuntimeError("posterior contains non-finite values")
        return posterior

    def summarize_posterior(
        self,
        posterior: np.ndarray,
        true_tau: float,
    ) -> PosteriorResult:
        posterior = np.asarray(posterior, dtype=float)
        if posterior.shape != self.tau_values.shape:
            raise ValueError("posterior and tau grid must have matching shapes")
        if not math.isclose(float(np.sum(posterior)), 1.0, abs_tol=1e-10):
            raise ValueError("posterior must sum to one")
        estimate = float(np.sum(posterior * self.tau_values))
        risk = float(np.sum(posterior * (self.tau_values - estimate) ** 2))
        entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-300))))
        nearest = int(np.argmin(np.abs(self.tau_values - true_tau)))
        nll = float(-math.log(max(float(posterior[nearest]), 1e-300)))
        cumulative = np.cumsum(posterior)
        lower_index = int(np.searchsorted(cumulative, 0.05, side="left"))
        upper_index = int(np.searchsorted(cumulative, 0.95, side="left"))
        upper_index = min(upper_index, self.tau_values.size - 1)
        contains = bool(
            self.tau_values[lower_index] <= true_tau <= self.tau_values[upper_index]
        )
        return PosteriorResult(
            estimate=estimate,
            posterior_risk=risk,
            entropy=entropy,
            nll=nll,
            credible90_contains_true=contains,
            squared_error=(estimate - true_tau) ** 2,
        )


def quantization_mse(
    contexts: Sequence[ObservationContext],
    threshold: float,
) -> float:
    """Empirical Lloyd reconstruction MSE for the same symmetric partitions."""

    if not contexts:
        raise ValueError("contexts must not be empty")
    values = np.concatenate(
        [
            np.concatenate(
                (context.y_b_normalized.real, context.y_b_normalized.imag)
            )
            for context in contexts
        ]
    )
    codes = SITQModel.encode_real(values, threshold)
    reconstruction = np.zeros_like(values)
    global_mean = float(np.mean(values))
    for code in range(4):
        mask = codes == code
        centroid = float(np.mean(values[mask])) if np.any(mask) else global_mean
        reconstruction[mask] = centroid
    return float(np.mean((values - reconstruction) ** 2))


def evaluate_threshold_objective(
    model: SITQModel,
    contexts: Sequence[ObservationContext],
    threshold: float,
    objective: Objective,
) -> float:
    if objective == "quantization_mse":
        return quantization_mse(contexts, threshold)
    values: list[float] = []
    for context in contexts:
        posterior = model.posterior_quantized(context, threshold)
        summary = model.summarize_posterior(posterior, context.true_tau)
        if objective == "posterior_entropy":
            values.append(summary.entropy)
        elif objective == "bayes_risk":
            # Under squared-error loss, the conditional Bayes risk is the
            # posterior variance.  Its expectation equals the estimator MSE in
            # this matched model, but has lower Monte Carlo design variance.
            values.append(summary.posterior_risk)
        else:
            raise ValueError(f"unsupported objective: {objective}")
    return float(np.mean(values))


def search_threshold(
    model: SITQModel,
    contexts: Sequence[ObservationContext],
    candidates: Sequence[float],
    objective: Objective,
) -> ThresholdSearchResult:
    candidate_array = np.asarray(candidates, dtype=float)
    if candidate_array.ndim != 1 or candidate_array.size < 3:
        raise ValueError("at least three threshold candidates are required")
    if np.any(~np.isfinite(candidate_array)) or np.any(candidate_array <= 0.0):
        raise ValueError("threshold candidates must be positive and finite")
    if np.any(np.diff(candidate_array) <= 0.0):
        raise ValueError("threshold candidates must be strictly increasing")
    rows: list[dict[str, float]] = []
    for threshold in candidate_array:
        value = evaluate_threshold_objective(
            model, contexts, float(threshold), objective
        )
        rows.append({"threshold": float(threshold), "objective_value": value})
    values = np.asarray([row["objective_value"] for row in rows])
    best_index = int(np.argmin(values))
    return ThresholdSearchResult(
        objective=objective,
        threshold=float(candidate_array[best_index]),
        objective_value=float(values[best_index]),
        at_search_boundary=best_index in {0, candidate_array.size - 1},
        rows=tuple(rows),
    )


def evaluate_context(
    model: SITQModel,
    context: ObservationContext,
    method: str,
    threshold: float | None,
) -> PosteriorResult:
    if method == "Unquantized-SelectedBins":
        posterior = model.posterior_unquantized(context)
    else:
        if threshold is None:
            raise ValueError("quantized methods require a threshold")
        posterior = model.posterior_quantized(context, threshold)
    return model.summarize_posterior(posterior, context.true_tau)


def quantizer_probability_sum(
    mean: float,
    standard_deviation: float,
    threshold: float,
) -> float:
    """Mechanical helper used by unit tests and the embedded chain audit."""

    edges = SITQModel.quantizer_edges(threshold)
    total = 0.0
    for code in range(4):
        total += float(
            SITQModel._interval_probability(
                np.asarray(mean),
                np.asarray(standard_deviation),
                np.asarray(edges[code]),
                np.asarray(edges[code + 1]),
            )
        )
    return total
