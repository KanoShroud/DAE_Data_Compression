"""Finite-rate likelihoods for the RAPTQ RelativePhase feasibility audit.

The module is deliberately independent from the historical SITQ implementation.
Both finite-rate methods use the continuous common-gain-phase model in
``raptq_tdoa.core``.  ``CondMI-Matched24`` quantizes six complex coefficients
with 2 bits per real component.  ``RAPTQ-RelativePhase24`` quantizes the five
relative phases with a fixed 4/5/5/5/5-bit allocation and marginalizes the
discarded magnitudes in the likelihood domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import qmc

from raptq_tdoa.core import BayesianRAPTQModel, ObservationContext


PhaseProposal = Literal["model_radial", "defensive_mixture"]


@dataclass(frozen=True)
class RelativePhaseCodeConfig:
    """Frozen 24-bit code on the five phase differences."""

    bits_by_relative_phase: tuple[int, ...] = (5, 5, 5, 5, 4)

    def validate(self, n_bins: int) -> None:
        if len(self.bits_by_relative_phase) != n_bins - 1:
            raise ValueError("one bit width is required for each relative phase")
        if any(bits < 1 for bits in self.bits_by_relative_phase):
            raise ValueError("relative-phase bit widths must be positive")
        if self.total_payload_bits != 24:
            raise ValueError("RelativePhase code is frozen to exactly 24 payload bits")

    @property
    def total_payload_bits(self) -> int:
        return int(sum(self.bits_by_relative_phase))


@dataclass(frozen=True)
class RelativePhaseQMCConfig:
    power: int
    repeats: int
    proposal: PhaseProposal = "defensive_mixture"

    def validate(self) -> None:
        if self.power < 4:
            raise ValueError("QMC power must be at least four")
        if self.repeats < 1:
            raise ValueError("QMC repeats must be positive")
        if self.proposal not in {"model_radial", "defensive_mixture"}:
            raise ValueError("unsupported RelativePhase QMC proposal")


@dataclass(frozen=True)
class CondMIThresholdSearch:
    threshold: float
    objective_value: float
    at_boundary: bool
    rows: tuple[dict[str, float], ...]


def _log_interval_probability(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if np.any(standard_deviation <= 0.0):
        raise ValueError("standard deviation must be positive")
    probability = ndtr((upper - mean) / standard_deviation) - ndtr(
        (lower - mean) / standard_deviation
    )
    return np.log(np.maximum(probability, np.finfo(float).tiny))


class ContinuousPhaseCondMI24:
    """Model-matched 24-bit I/Q quantizer with continuous phase marginalization."""

    bits_per_real = 2

    def __init__(self, model: BayesianRAPTQModel, phase_nodes: int) -> None:
        if phase_nodes < 8 or phase_nodes % 2:
            raise ValueError("phase_nodes must be an even integer of at least eight")
        self.model = model
        self.phase_nodes = int(phase_nodes)
        self.phase_values = np.linspace(-np.pi, np.pi, phase_nodes, endpoint=False)

    @property
    def total_payload_bits(self) -> int:
        return 2 * self.bits_per_real * self.model.n_bins

    @staticmethod
    def quantizer_edges(threshold: float) -> np.ndarray:
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be positive and finite")
        return np.asarray([-np.inf, -threshold, 0.0, threshold, np.inf])

    @classmethod
    def encode_real(cls, values: np.ndarray, threshold: float) -> np.ndarray:
        edges = cls.quantizer_edges(threshold)
        return np.searchsorted(
            edges[1:-1], np.asarray(values, dtype=float), side="right"
        )

    def log_likelihood(
        self,
        context: ObservationContext,
        threshold: float,
    ) -> np.ndarray:
        edges = self.quantizer_edges(threshold)
        real_codes = self.encode_real(context.y_b_normalized.real, threshold)
        imag_codes = self.encode_real(context.y_b_normalized.imag, threshold)
        phase = np.exp(1j * self.phase_values)
        delay = self.model._phase_grid[:, None, None, :]
        amplitude = self.model.gain_amplitudes[None, :, None, None]
        source_mean = context.source_conditional_mean_normalized[None, None, None, :]
        means = delay * amplitude * phase[None, None, :, None] * source_mean
        std = np.sqrt(context.conditional_variance[None, :, None, :] / 2.0)
        log_probability = np.zeros(means.shape[:-1], dtype=float)
        for position in range(self.model.n_bins):
            mean = means[..., position]
            position_std = std[..., position]
            log_probability += _log_interval_probability(
                mean.real,
                position_std,
                edges[real_codes[position]],
                edges[real_codes[position] + 1],
            )
            log_probability += _log_interval_probability(
                mean.imag,
                position_std,
                edges[imag_codes[position]],
                edges[imag_codes[position] + 1],
            )
        log_probability += np.log(self.model.gain_amplitude_prior)[None, :, None]
        return logsumexp(log_probability, axis=(1, 2)) - math.log(self.phase_nodes)

    def posterior(
        self,
        context: ObservationContext,
        threshold: float,
    ) -> np.ndarray:
        return self.model._posterior_from_log_likelihood(
            self.log_likelihood(context, threshold)
        )

    def mean_entropy(
        self,
        contexts: Sequence[ObservationContext],
        threshold: float,
    ) -> float:
        if not contexts:
            raise ValueError("contexts must not be empty")
        entropy = []
        for context in contexts:
            posterior = self.posterior(context, threshold)
            entropy.append(
                -float(np.sum(posterior * np.log(np.maximum(posterior, 1e-300))))
            )
        return float(np.mean(entropy))

    def search_threshold(
        self,
        contexts: Sequence[ObservationContext],
        candidates: Sequence[float],
        progress_callback: Callable[[int, int, float, float], None] | None = None,
    ) -> CondMIThresholdSearch:
        values = np.asarray(candidates, dtype=float)
        if values.ndim != 1 or values.size < 3:
            raise ValueError("at least three threshold candidates are required")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("thresholds must be positive and finite")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("thresholds must be strictly increasing")
        rows_list: list[dict[str, float]] = []
        for index, threshold in enumerate(values, start=1):
            objective_value = self.mean_entropy(contexts, float(threshold))
            rows_list.append(
                {
                    "threshold": float(threshold),
                    "objective_value": objective_value,
                }
            )
            if progress_callback is not None:
                progress_callback(index, values.size, float(threshold), objective_value)
        rows = tuple(rows_list)
        objective = np.asarray([row["objective_value"] for row in rows])
        selected = int(np.argmin(objective))
        return CondMIThresholdSearch(
            threshold=float(values[selected]),
            objective_value=float(objective[selected]),
            at_boundary=selected in {0, values.size - 1},
            rows=rows,
        )


class QuantizedRelativePhase24:
    """Exact-code likelihood estimated by QMC over discarded magnitudes."""

    def __init__(
        self,
        model: BayesianRAPTQModel,
        code_config: RelativePhaseCodeConfig | None = None,
    ) -> None:
        self.model = model
        self.code_config = code_config or RelativePhaseCodeConfig()
        self.code_config.validate(model.n_bins)
        self.bits = np.asarray(self.code_config.bits_by_relative_phase, dtype=int)
        self.levels = 2**self.bits
        self.widths = 2.0 * np.pi / self.levels

    @property
    def total_payload_bits(self) -> int:
        return self.code_config.total_payload_bits

    def encode(self, observation: np.ndarray) -> np.ndarray:
        _, _, phases = self.model.observed_coordinates(observation)
        relative = phases[1:]
        scaled = (relative + np.pi) / (2.0 * np.pi)
        codes = np.floor(np.clip(scaled, 0.0, 1.0 - np.finfo(float).eps) * self.levels)
        return codes.astype(int)

    def phase_bounds(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(codes, dtype=int)
        if values.shape != self.levels.shape:
            raise ValueError("codes must match the relative-phase dimensions")
        if np.any(values < 0) or np.any(values >= self.levels):
            raise ValueError("relative-phase code is out of range")
        lower = -np.pi + values * self.widths
        return lower, lower + self.widths

    def _power_proposal(
        self,
        context: ObservationContext,
        unit: np.ndarray,
        proposal: PhaseProposal,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        cursor = 0
        model_scale = self.model._predictive_power_by_bin(context)
        observed_scale = np.maximum(
            np.abs(context.y_b_normalized) ** 2,
            np.finfo(float).eps * model_scale,
        )
        if proposal == "defensive_mixture":
            use_observed = unit[:, cursor] >= 0.5
            cursor += 1
            exponential = -np.log(unit[:, cursor : cursor + self.model.n_bins])
            cursor += self.model.n_bins
            selected_scale = np.where(
                use_observed[:, None],
                observed_scale[None, :],
                model_scale[None, :],
            )
            power = selected_scale * exponential
            log_q_model = np.sum(
                -np.log(model_scale)[None, :] - power / model_scale[None, :],
                axis=1,
            )
            log_q_observed = np.sum(
                -np.log(observed_scale)[None, :] - power / observed_scale[None, :],
                axis=1,
            )
            log_q = np.logaddexp(log_q_model, log_q_observed) - math.log(2.0)
        else:
            exponential = -np.log(unit[:, cursor : cursor + self.model.n_bins])
            cursor += self.model.n_bins
            power = model_scale[None, :] * exponential
            log_q = np.sum(
                -np.log(model_scale)[None, :] - power / model_scale[None, :],
                axis=1,
            )
        return power, -log_q, cursor

    def log_likelihood(
        self,
        context: ObservationContext,
        config: RelativePhaseQMCConfig,
        seed: int,
    ) -> np.ndarray:
        config.validate()
        codes = self.encode(context.y_b_normalized)
        lower, upper = self.phase_bounds(codes)
        repeat_log_likelihoods: list[np.ndarray] = []
        selector_dimensions = 1 if config.proposal == "defensive_mixture" else 0
        dimension = selector_dimensions + self.model.n_bins + self.model.n_bins - 1
        for repeat in range(config.repeats):
            unit = qmc.Sobol(
                dimension,
                scramble=True,
                seed=(seed + 104729 * repeat) % (2**32 - 1),
            ).random_base2(config.power)
            unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
            power, log_weights, cursor = self._power_proposal(
                context, unit, config.proposal
            )
            phase_unit = unit[:, cursor : cursor + self.model.n_bins - 1]
            cursor += self.model.n_bins - 1
            if cursor != dimension:
                raise RuntimeError("RelativePhase QMC coordinate count mismatch")
            phases = np.zeros((unit.shape[0], self.model.n_bins), dtype=float)
            phases[:, 1:] = lower[None, :] + phase_unit * (upper - lower)[None, :]
            points = np.sqrt(power) * np.exp(1j * phases)
            log_density = self.model.log_density_by_tau(points, context)
            repeat_log_likelihoods.append(
                logsumexp(log_density + log_weights[:, None], axis=0)
                - math.log(points.shape[0])
            )
        return logsumexp(np.stack(repeat_log_likelihoods), axis=0) - math.log(
            len(repeat_log_likelihoods)
        )

    def posterior(
        self,
        context: ObservationContext,
        config: RelativePhaseQMCConfig,
        seed: int,
    ) -> np.ndarray:
        return self.model._posterior_from_log_likelihood(
            self.log_likelihood(context, config, seed)
        )


def relative_phase_quantization_max_error(bits: Sequence[int]) -> float:
    values = np.asarray(bits, dtype=int)
    if values.ndim != 1 or np.any(values < 1):
        raise ValueError("bits must be a positive one-dimensional sequence")
    return float(np.max(np.pi / (2**values)))
