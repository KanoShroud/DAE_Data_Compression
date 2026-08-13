"""Matched Bayesian model for the RAPTQ Stage 0 audit.

The module is intentionally independent from SITQ.  It retains SITQ's frozen
frequency, source-power, TDOA, and gain-amplitude assumptions, but replaces the
eight-point gain-phase grid with its intended continuous uniform phase prior.
The phase is marginalized analytically, so global-phase invariance is not an
artifact of a finite quadrature grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammaincinv, gammaln, i0e, logsumexp
from scipy.stats import qmc


Representation = Literal[
    "projector_only",
    "relative_phase",
    "relative_power",
    "sparse_phase",
    "sparse_amp_phase",
]
QMCProposal = Literal["model_radial", "defensive_observed"]


@dataclass(frozen=True)
class ModelConfig:
    """Frozen matched model for Stage 0-T and Stage 0-I."""

    n_time: int = 1024
    fs_hz: float = 40e6
    tau_min: float = -8.0
    tau_max: float = 8.0
    tau_step: float = 0.25
    selected_bins: tuple[int, ...] = (31, 94, 145, 196, 247, 298)
    rrc_rolloff: float = 0.2
    samples_per_symbol: int = 2
    gain_amplitudes: tuple[float, ...] = (0.75, 0.90)

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
        if not math.isclose(intervals, round(intervals), abs_tol=1e-12):
            raise ValueError("tau support must be divisible by tau_step")
        bins = np.asarray(self.selected_bins, dtype=int)
        if bins.ndim != 1 or bins.size < 3 or np.unique(bins).size != bins.size:
            raise ValueError("selected_bins must contain at least three unique bins")
        if np.any(bins <= 0) or np.any(bins >= self.n_time // 2):
            raise ValueError("selected bins must be positive and non-Nyquist")
        if np.any(bins > self.rrc_support_max_bin):
            raise ValueError("selected bins must lie inside the RRC support")
        if not self.gain_amplitudes or any(
            value <= 0.0 or not math.isfinite(value)
            for value in self.gain_amplitudes
        ):
            raise ValueError("gain amplitudes must be positive and finite")

    @property
    def symbol_rate_hz(self) -> float:
        return self.fs_hz / self.samples_per_symbol

    @property
    def rrc_support_max_bin(self) -> int:
        bandwidth = (
            (1.0 + self.rrc_rolloff) * self.symbol_rate_hz / 2.0
        )
        return int(math.floor(bandwidth * self.n_time / self.fs_hz))


@dataclass(frozen=True)
class AuditConfig:
    """Run-size and representation settings, fixed before an audit run."""

    mode: str
    snr_db_values: tuple[float, ...]
    seeds: tuple[int, ...]
    trials_per_seed: int
    sparse_coordinate_count: int
    bootstrap_repetitions: int

    def validate(self, n_bins: int) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.snr_db_values:
            raise ValueError("snr_db_values must not be empty")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique and non-empty")
        if self.trials_per_seed < 1:
            raise ValueError("trials_per_seed must be positive")
        if not 1 <= self.sparse_coordinate_count < n_bins - 1:
            raise ValueError("sparse coordinate count must be in [1, K-2]")
        if self.bootstrap_repetitions < 20:
            raise ValueError("bootstrap_repetitions must be at least 20")


@dataclass(frozen=True)
class QMCConfig:
    """Randomized Sobol integration settings."""

    power: int
    repeats: int
    reference_power: int
    reference_context_limit: int
    proposal: QMCProposal = "model_radial"

    def validate(self) -> None:
        if self.power < 4:
            raise ValueError("QMC power must be at least four")
        if self.repeats < 1:
            raise ValueError("QMC repeats must be positive")
        if self.reference_power <= self.power:
            raise ValueError("reference_power must exceed power")
        if self.reference_context_limit < 1:
            raise ValueError("reference_context_limit must be positive")
        if self.proposal not in {"model_radial", "defensive_observed"}:
            raise ValueError("unsupported QMC proposal")


@dataclass(frozen=True)
class TrialLatents:
    seed: int
    trial: int
    true_tau: float
    true_gain_amplitude_index: int
    true_gain_phase: float
    source: np.ndarray
    noise_a_standard: np.ndarray
    noise_b_standard: np.ndarray


@dataclass(frozen=True)
class ObservationContext:
    seed: int
    trial: int
    snr_db: float
    true_tau: float
    y_a_normalized: np.ndarray
    y_b_normalized: np.ndarray
    source_conditional_mean_normalized: np.ndarray
    conditional_variance: np.ndarray


@dataclass(frozen=True)
class PosteriorSummary:
    estimate: float
    posterior_risk: float
    entropy: float
    nll: float
    credible90_contains_true: bool
    squared_error: float


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


def _log_i0(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.asarray(value, dtype=float))
    return np.log(i0e(magnitude)) + magnitude


class BayesianRAPTQModel:
    """Continuous-phase matched Bayesian model and information integrator."""

    def __init__(self, config: ModelConfig) -> None:
        config.validate()
        self.config = config
        self.bins = np.asarray(config.selected_bins, dtype=int)
        self.n_bins = self.bins.size
        self.omega = 2.0 * np.pi * self.bins / config.n_time
        self.source_power = raised_cosine_power_at_bins(self.bins, config)
        intervals = int(round((config.tau_max - config.tau_min) / config.tau_step))
        self.tau_values = np.linspace(config.tau_min, config.tau_max, intervals + 1)
        self.tau_prior = np.full(self.tau_values.size, 1.0 / self.tau_values.size)
        self.gain_amplitudes = np.asarray(config.gain_amplitudes, dtype=float)
        self.gain_amplitude_prior = np.full(
            self.gain_amplitudes.size,
            1.0 / self.gain_amplitudes.size,
        )
        self.gain_second_moment = float(
            np.sum(self.gain_amplitude_prior * self.gain_amplitudes**2)
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
            trials.append(
                TrialLatents(
                    seed=seed,
                    trial=trial_index,
                    true_tau=float(rng.choice(self.tau_values, p=self.tau_prior)),
                    true_gain_amplitude_index=int(
                        rng.integers(self.gain_amplitudes.size)
                    ),
                    true_gain_phase=float(rng.uniform(-np.pi, np.pi)),
                    source=_complex_standard_normal(rng, self.n_bins)
                    * np.sqrt(self.source_power),
                    noise_a_standard=_complex_standard_normal(rng, self.n_bins),
                    noise_b_standard=_complex_standard_normal(rng, self.n_bins),
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
        amplitude = self.gain_amplitudes[trial.true_gain_amplitude_index]
        gain = amplitude * np.exp(1j * trial.true_gain_phase)
        true_phase = np.exp(-1j * self.omega * trial.true_tau)
        y_a = trial.source + math.sqrt(sigma2) * trial.noise_a_standard
        y_b = gain * true_phase * trial.source + math.sqrt(sigma2) * trial.noise_b_standard

        public_scale = np.sqrt(
            self.gain_second_moment * self.source_power + sigma2
        )
        kappa = self.source_power / (self.source_power + sigma2)
        source_conditional_variance = (
            self.source_power * sigma2 / (self.source_power + sigma2)
        )
        conditional_variance = (
            self.gain_amplitudes[:, None] ** 2
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
            y_a_normalized=y_a / public_scale,
            y_b_normalized=y_b / public_scale,
            source_conditional_mean_normalized=kappa * y_a / public_scale,
            conditional_variance=conditional_variance,
        )

    def build_contexts(
        self,
        seeds: tuple[int, ...],
        trials_per_seed: int,
        snr_db_values: tuple[float, ...],
    ) -> list[ObservationContext]:
        contexts: list[ObservationContext] = []
        for seed in seeds:
            for trial in self.generate_trials(seed, trials_per_seed):
                contexts.extend(
                    self.build_context(trial, snr) for snr in snr_db_values
                )
        return contexts

    def _means_and_variances(
        self,
        context: ObservationContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        means = (
            self._phase_grid[:, None, :]
            * self.gain_amplitudes[None, :, None]
            * context.source_conditional_mean_normalized[None, None, :]
        )
        return means, context.conditional_variance

    def log_density_by_tau(
        self,
        points: np.ndarray,
        context: ObservationContext,
    ) -> np.ndarray:
        """Evaluate the gain-phase-marginal density at complex points.

        Returns an array with shape ``[n_points, n_tau]``.  Gain phase is
        integrated analytically with the modified Bessel function I0.
        """

        values = np.asarray(points, dtype=np.complex128)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.n_bins:
            raise ValueError("points must have shape [n_points, n_bins]")
        means, variances = self._means_and_variances(context)
        inverse_variance = 1.0 / variances
        y_energy = np.einsum(
            "qk,ak->qa",
            np.abs(values) ** 2,
            inverse_variance,
            optimize=True,
        )
        mean_energy = np.einsum(
            "tak,ak->ta",
            np.abs(means) ** 2,
            inverse_variance,
            optimize=True,
        )
        weighted_means = means * inverse_variance[None, :, :]
        coherence = np.einsum(
            "qk,tak->qta",
            np.conj(values),
            weighted_means,
            optimize=True,
        )
        log_normalizer = -np.sum(np.log(np.pi * variances), axis=1)
        component = (
            -y_energy[:, None, :]
            - mean_energy[None, :, :]
            + _log_i0(2.0 * np.abs(coherence))
            + log_normalizer[None, None, :]
            + np.log(self.gain_amplitude_prior)[None, None, :]
        )
        return logsumexp(component, axis=2)

    def posterior_full(
        self,
        context: ObservationContext,
        observation: np.ndarray | None = None,
    ) -> np.ndarray:
        point = context.y_b_normalized if observation is None else observation
        log_likelihood = self.log_density_by_tau(point, context)[0]
        return self._posterior_from_log_likelihood(log_likelihood)

    def _posterior_from_log_likelihood(self, log_likelihood: np.ndarray) -> np.ndarray:
        values = np.asarray(log_likelihood, dtype=float)
        if values.shape != self.tau_values.shape:
            raise ValueError("log likelihood must match the TDOA grid")
        log_posterior = values + np.log(self.tau_prior)
        log_posterior -= logsumexp(log_posterior)
        posterior = np.exp(log_posterior)
        if not np.all(np.isfinite(posterior)):
            raise RuntimeError("posterior contains non-finite values")
        return posterior

    @staticmethod
    def projector(observation: np.ndarray) -> tuple[float, np.ndarray]:
        vector = np.asarray(observation, dtype=np.complex128)
        if vector.ndim != 1:
            raise ValueError("observation must be one-dimensional")
        radius = float(np.linalg.norm(vector))
        if not math.isfinite(radius) or radius <= np.finfo(float).tiny:
            raise ValueError("observation norm is too small for a projective chart")
        direction = vector / radius
        return radius, np.outer(direction, np.conj(direction))

    @staticmethod
    def representative_from_projector(radius: float, projector: np.ndarray) -> np.ndarray:
        matrix = np.asarray(projector, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("projector must be square")
        diagonal = np.real(np.diag(matrix))
        anchor = int(np.argmax(diagonal))
        if diagonal[anchor] <= np.finfo(float).eps:
            raise ValueError("projector has no numerically stable chart")
        direction = matrix[:, anchor] / math.sqrt(float(diagonal[anchor]))
        return float(radius) * direction

    def summarize_posterior(
        self,
        posterior: np.ndarray,
        true_tau: float,
    ) -> PosteriorSummary:
        values = np.asarray(posterior, dtype=float)
        if values.shape != self.tau_values.shape:
            raise ValueError("posterior and TDOA grid must match")
        if not math.isclose(float(np.sum(values)), 1.0, abs_tol=1e-10):
            raise ValueError("posterior must sum to one")
        estimate = float(np.sum(values * self.tau_values))
        risk = float(np.sum(values * (self.tau_values - estimate) ** 2))
        entropy = float(-np.sum(values * np.log(np.maximum(values, 1e-300))))
        nearest = int(np.argmin(np.abs(self.tau_values - true_tau)))
        nll = float(-math.log(max(float(values[nearest]), 1e-300)))
        cumulative = np.cumsum(values)
        lower = int(np.searchsorted(cumulative, 0.05, side="left"))
        upper = min(
            int(np.searchsorted(cumulative, 0.95, side="left")),
            values.size - 1,
        )
        return PosteriorSummary(
            estimate=estimate,
            posterior_risk=risk,
            entropy=entropy,
            nll=nll,
            credible90_contains_true=bool(
                self.tau_values[lower] <= true_tau <= self.tau_values[upper]
            ),
            squared_error=(estimate - true_tau) ** 2,
        )

    def observed_coordinates(
        self,
        observation: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        vector = np.asarray(observation, dtype=np.complex128)
        if vector.shape != (self.n_bins,):
            raise ValueError("observation must match selected bins")
        energy = float(np.sum(np.abs(vector) ** 2))
        if energy <= np.finfo(float).tiny:
            raise ValueError("observation energy is too small")
        powers = np.abs(vector) ** 2 / energy
        phases = np.angle(vector) - float(np.angle(vector[0]))
        phases = np.angle(np.exp(1j * phases))
        phases[0] = 0.0
        return energy, powers, phases

    def qmc_log_likelihood(
        self,
        context: ObservationContext,
        representation: Representation,
        qmc_config: QMCConfig,
        sparse_coordinate_count: int,
        seed: int,
    ) -> np.ndarray:
        qmc_config.validate()
        if not 1 <= sparse_coordinate_count < self.n_bins - 1:
            raise ValueError("invalid sparse coordinate count")
        if representation == "relative_power":
            # Under the matched diagonal circular-shift model, delay changes only
            # the component phases.  Marginal magnitudes, their total energy, and
            # normalized relative powers are therefore exactly tau-independent.
            return np.zeros(self.tau_values.size, dtype=float)
        repeat_likelihoods: list[np.ndarray] = []
        for repeat in range(qmc_config.repeats):
            points, log_weights = self._qmc_points_for_representation(
                context,
                representation,
                qmc_config.power,
                sparse_coordinate_count,
                seed + 104729 * repeat,
                qmc_config.proposal,
            )
            log_density = self.log_density_by_tau(points, context)
            repeat_likelihoods.append(
                logsumexp(log_density + log_weights[:, None], axis=0)
                - math.log(points.shape[0])
            )
        return logsumexp(np.stack(repeat_likelihoods), axis=0) - math.log(
            len(repeat_likelihoods)
        )

    def posterior_representation(
        self,
        context: ObservationContext,
        representation: Representation,
        qmc_config: QMCConfig,
        sparse_coordinate_count: int,
        seed: int,
    ) -> np.ndarray:
        return self._posterior_from_log_likelihood(
            self.qmc_log_likelihood(
                context,
                representation,
                qmc_config,
                sparse_coordinate_count,
                seed,
            )
        )

    def _integration_scale(self, context: ObservationContext) -> float:
        means, variances = self._means_and_variances(context)
        expected_energy = float(
            np.mean(np.sum(np.abs(means) ** 2, axis=2))
            + np.mean(np.sum(variances, axis=1))
        )
        return max(expected_energy / self.n_bins, 1e-3)

    def _predictive_power_by_bin(self, context: ObservationContext) -> np.ndarray:
        means, variances = self._means_and_variances(context)
        predictive = np.mean(
            np.abs(means) ** 2 + variances[None, :, :],
            axis=(0, 1),
        )
        if np.any(~np.isfinite(predictive)) or np.any(predictive <= 0.0):
            raise RuntimeError("predictive bin powers must be positive and finite")
        return predictive

    def _qmc_relative_phase_defensive_points(
        self,
        context: ObservationContext,
        power: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Defensive MIS reference for the relative-phase induced likelihood.

        Half of the points use a public model-predictive exponential proposal;
        half use an observation-centred exponential proposal.  Every point is
        weighted by the equal whole-vector proposal mixture, so the discarded
        amplitudes affect numerical efficiency only and cannot change the exact
        induced likelihood.  Agreement with ``model_radial`` remains mandatory.
        """

        unit = qmc.Sobol(self.n_bins + 1, scramble=True, seed=seed).random_base2(
            power
        )
        unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        use_observed = unit[:, 0] >= 0.5
        exponential = -np.log(unit[:, 1:])
        model_scale = self._predictive_power_by_bin(context)
        observed_scale = np.maximum(
            np.abs(context.y_b_normalized) ** 2,
            np.finfo(float).eps * model_scale,
        )
        selected_scale = np.where(
            use_observed[:, None],
            observed_scale[None, :],
            model_scale[None, :],
        )
        component_power = selected_scale * exponential
        _, _, observed_phase = self.observed_coordinates(context.y_b_normalized)
        points = np.sqrt(component_power) * np.exp(1j * observed_phase[None, :])

        log_q_model = np.sum(
            -np.log(model_scale)[None, :] - component_power / model_scale[None, :],
            axis=1,
        )
        log_q_observed = np.sum(
            -np.log(observed_scale)[None, :]
            - component_power / observed_scale[None, :],
            axis=1,
        )
        log_half = -math.log(2.0)
        log_mixture = np.logaddexp(
            log_half + log_q_model,
            log_half + log_q_observed,
        )
        # In squared-amplitude coordinates the complex polar measure is a
        # tau-independent constant times prod_k du_k after global phase removal.
        return points, -log_mixture

    def _qmc_points_for_representation(
        self,
        context: ObservationContext,
        representation: Representation,
        power: int,
        sparse_coordinate_count: int,
        seed: int,
        proposal: QMCProposal,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, observed_power, observed_phase = self.observed_coordinates(
            context.y_b_normalized
        )
        retained = np.arange(1, sparse_coordinate_count + 1, dtype=int)
        unretained = np.asarray(
            [index for index in range(self.n_bins) if index not in retained],
            dtype=int,
        )
        if proposal == "defensive_observed":
            if representation != "relative_phase":
                raise ValueError(
                    "defensive_observed is only defined for relative_phase"
                )
            return self._qmc_relative_phase_defensive_points(
                context,
                power,
                seed,
            )
        if representation == "projector_only":
            dimension = 1
        elif representation == "relative_phase":
            dimension = self.n_bins + 1
        elif representation == "relative_power":
            raise RuntimeError("relative_power must use its analytic likelihood")
        elif representation == "sparse_phase":
            dimension = 2 * self.n_bins - sparse_coordinate_count
        elif representation == "sparse_amp_phase":
            dimension = 2 * self.n_bins - 2 * sparse_coordinate_count
        else:
            raise ValueError(f"unsupported representation: {representation}")

        unit = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(power)
        unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        cursor = 0
        scale = self._integration_scale(context)
        radius_squared = scale * gammaincinv(self.n_bins, unit[:, cursor])
        cursor += 1

        n_points = unit.shape[0]
        powers = np.empty((n_points, self.n_bins), dtype=float)
        phases = np.zeros((n_points, self.n_bins), dtype=float)

        if representation in {"projector_only", "relative_power"}:
            powers[:] = observed_power
        elif representation == "sparse_amp_phase":
            powers[:, retained] = observed_power[retained]
            remaining_mass = 1.0 - float(np.sum(observed_power[retained]))
            exponentials = -np.log(unit[:, cursor : cursor + unretained.size])
            cursor += unretained.size
            shares = exponentials / np.sum(exponentials, axis=1, keepdims=True)
            powers[:, unretained] = remaining_mass * shares
        else:
            exponentials = -np.log(unit[:, cursor : cursor + self.n_bins])
            cursor += self.n_bins
            powers[:] = exponentials / np.sum(exponentials, axis=1, keepdims=True)

        if representation in {"projector_only", "relative_phase"}:
            phases[:] = observed_phase
        elif representation == "relative_power":
            phases[:, 1:] = (
                2.0 * np.pi * unit[:, cursor : cursor + self.n_bins - 1] - np.pi
            )
            cursor += self.n_bins - 1
        else:
            phases[:, retained] = observed_phase[retained]
            phase_indices = np.asarray(
                [index for index in range(1, self.n_bins) if index not in retained],
                dtype=int,
            )
            phases[:, phase_indices] = (
                2.0 * np.pi * unit[:, cursor : cursor + phase_indices.size] - np.pi
            )
            cursor += phase_indices.size

        if cursor != dimension:
            raise RuntimeError(
                f"QMC coordinate mismatch: consumed {cursor}, expected {dimension}"
            )
        points = np.sqrt(radius_squared[:, None] * powers) * np.exp(
            1j * phases
        )
        # Complex polar volume: 2^-K t^(K-1) dt dp dphi.  The Gamma(K,
        # scale) proposal cancels t^(K-1).  Global phase was marginalized
        # analytically because the continuous gain-phase likelihood is U(1)
        # invariant; all omitted constants are independent of tau.
        log_weights = (
            gammaln(self.n_bins)
            + self.n_bins * math.log(scale)
            + radius_squared / scale
        )
        return points, log_weights


def posterior_total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(first) - np.asarray(second))))


def finite_window_delay_matrix(n_samples: int, delay: float) -> np.ndarray:
    """Truncated ideal fractional-delay operator on a finite observation window."""

    if n_samples < 4:
        raise ValueError("n_samples must be at least four")
    rows = np.arange(n_samples, dtype=float)[:, None]
    columns = np.arange(n_samples, dtype=float)[None, :]
    return np.sinc(rows - columns - float(delay))


def general_phase_marginal_log_likelihood(
    observation: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    amplitude_prior: np.ndarray,
) -> np.ndarray:
    """Continuous gain-phase marginal for full covariance matrices.

    ``means`` has shape [tau, amplitude, dimension] and excludes gain phase.
    ``covariances`` has shape [tau, amplitude, dimension, dimension].
    """

    y = np.asarray(observation, dtype=np.complex128)
    if means.ndim != 3 or covariances.ndim != 4:
        raise ValueError("invalid mean or covariance dimensions")
    n_tau, n_amp, dimension = means.shape
    if y.shape != (dimension,) or covariances.shape != (
        n_tau,
        n_amp,
        dimension,
        dimension,
    ):
        raise ValueError("incompatible general likelihood shapes")
    rows = np.empty((n_tau, n_amp), dtype=float)
    for tau_index in range(n_tau):
        for amp_index in range(n_amp):
            covariance = covariances[tau_index, amp_index]
            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0.0:
                raise ValueError("covariance must be positive definite")
            inverse_y = np.linalg.solve(covariance, y)
            inverse_mean = np.linalg.solve(
                covariance,
                means[tau_index, amp_index],
            )
            y_energy = float(np.real(np.vdot(y, inverse_y)))
            mean_energy = float(
                np.real(np.vdot(means[tau_index, amp_index], inverse_mean))
            )
            coherence = np.vdot(y, inverse_mean)
            rows[tau_index, amp_index] = (
                -y_energy
                - mean_energy
                + float(_log_i0(np.asarray(2.0 * abs(coherence))))
                - dimension * math.log(np.pi)
                - float(log_det)
                + math.log(float(amplitude_prior[amp_index]))
            )
    return logsumexp(rows, axis=1)


def finite_window_rotation_certificate(seed: int = 55109) -> dict[str, float | bool]:
    """Mechanically verify U(1) invariance for a non-unitary delay operator."""

    rng = np.random.default_rng(seed)
    dimension = 12
    tau_values = np.asarray([-1.25, 0.0, 1.25])
    amplitudes = np.asarray([0.75, 0.90])
    amplitude_prior = np.full(amplitudes.size, 1.0 / amplitudes.size)
    sigma2 = 0.3
    source_variance = 1.0
    kappa = source_variance / (source_variance + sigma2)
    conditional_variance = source_variance * sigma2 / (source_variance + sigma2)
    source = _complex_standard_normal(rng, dimension)
    y_a = source + math.sqrt(sigma2) * _complex_standard_normal(rng, dimension)
    conditional_mean = kappa * y_a
    true_tau_index = 2
    true_operator = finite_window_delay_matrix(
        dimension,
        float(tau_values[true_tau_index]),
    )
    true_gain = amplitudes[1] * np.exp(1j * 0.73)
    y_b = (
        true_gain * (true_operator @ source)
        + math.sqrt(sigma2) * _complex_standard_normal(rng, dimension)
    )
    means = np.empty((tau_values.size, amplitudes.size, dimension), dtype=np.complex128)
    covariances = np.empty(
        (tau_values.size, amplitudes.size, dimension, dimension),
        dtype=np.complex128,
    )
    radius_by_tau = []
    for tau_index, tau in enumerate(tau_values):
        operator = finite_window_delay_matrix(dimension, float(tau))
        radius_by_tau.append(float(np.linalg.norm(operator @ source)))
        for amp_index, amplitude in enumerate(amplitudes):
            means[tau_index, amp_index] = amplitude * operator @ conditional_mean
            covariances[tau_index, amp_index] = (
                sigma2 * np.eye(dimension)
                + amplitude**2
                * conditional_variance
                * operator
                @ operator.conj().T
            )
    base = general_phase_marginal_log_likelihood(
        y_b,
        means,
        covariances,
        amplitude_prior,
    )
    rotated = general_phase_marginal_log_likelihood(
        np.exp(1j * 1.137) * y_b,
        means,
        covariances,
        amplitude_prior,
    )
    base_posterior = np.exp(base - logsumexp(base))
    rotated_posterior = np.exp(rotated - logsumexp(rotated))
    rotation_tv = posterior_total_variation(base_posterior, rotated_posterior)
    radius_span = max(radius_by_tau) - min(radius_by_tau)
    return {
        "finite_window_rotation_tv": rotation_tv,
        "finite_window_rotation_invariant": rotation_tv < 1e-10,
        "finite_window_radius_span": radius_span,
        "finite_window_radius_is_delay_sensitive": radius_span > 1e-6,
    }
