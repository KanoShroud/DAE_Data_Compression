"""Finite-model core for the NP-NBQ-DPD Stage 0 structure certificate.

The model is deliberately small enough to enumerate every legal node-frequency
bit allocation.  It is not a general unknown-channel solver: source phases use
a finite prior, link amplitudes are public pilot-derived quantities, and each
non-reference receiver has a frequency-flat nuisance phase from a finite grid.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass(frozen=True)
class ModelConfig:
    frequencies_hz: tuple[float, ...] = (4.0e6, 11.0e6)
    sound_speed_m_s: float = 299_792_458.0
    x_grid_m: tuple[float, ...] = (-40.0, -20.0, 0.0, 20.0, 40.0)
    y_grid_m: tuple[float, ...] = (30.0, 50.0, 70.0, 90.0, 110.0)
    source_phase_grid_rad: tuple[float, ...] = (0.0, math.pi)
    source_amplitude_ratio_grid: tuple[float, ...] = (0.8, 1.2)
    gain_phase_grid_rad: tuple[float, ...] = (-math.pi / 4.0, math.pi / 4.0)
    gain_amplitude_grid: tuple[float, ...] = (0.85, 1.15)
    noise_variance: float = 1.0
    max_bits_per_real: int = 3
    payload_bits: int = 24
    quantizer_code_mapping: str = "natural_binary"

    @property
    def n_sensors(self) -> int:
        return 3

    @property
    def n_frequencies(self) -> int:
        return len(self.frequencies_hz)

    @property
    def n_cells(self) -> int:
        return self.n_sensors * self.n_frequencies

    @property
    def bit_units(self) -> int:
        if self.payload_bits % 2:
            raise ValueError("payload_bits must be even for complex I/Q quantization")
        return self.payload_bits // 2

    def validate(self) -> None:
        if self.n_frequencies != 2:
            raise ValueError("Stage 0 is frozen to exactly two frequency cells")
        if self.sound_speed_m_s <= 0.0 or self.noise_variance <= 0.0:
            raise ValueError("physical scale parameters must be positive")
        if self.max_bits_per_real != 3 or self.payload_bits != 24:
            raise ValueError("Stage 0 is frozen to 0-3 bits per real and 24 payload bits")
        if self.quantizer_code_mapping != "natural_binary":
            raise ValueError("Stage 0 is frozen to natural-binary quantizer labels")
        if len(self.x_grid_m) < 3 or len(self.y_grid_m) < 3:
            raise ValueError("the direct-position grid must be two-dimensional")
        if len(self.source_phase_grid_rad) < 2 or len(self.gain_phase_grid_rad) < 2:
            raise ValueError("nuisance phase priors must contain at least two states")
        if (
            len(self.source_amplitude_ratio_grid) < 2
            or len(self.gain_amplitude_grid) < 2
        ):
            raise ValueError("nuisance amplitude priors must contain at least two states")
        if any(value <= 0.0 for value in self.source_amplitude_ratio_grid):
            raise ValueError("source amplitude ratios must be positive")
        if any(value <= 0.0 for value in self.gain_amplitude_grid):
            raise ValueError("gain amplitudes must be positive")
        if len(set(self.frequencies_hz)) != self.n_frequencies:
            raise ValueError("frequency cells must be unique")


@dataclass(frozen=True)
class PublicProfile:
    profile_id: str
    sensor_positions_m: tuple[tuple[float, float], ...]
    sensor_snr_db: tuple[float, ...]
    sensor_ber: tuple[float, ...]
    sensor_capacity_depth: tuple[int, ...]

    def validate(self, config: ModelConfig) -> None:
        if len(self.sensor_positions_m) != config.n_sensors:
            raise ValueError("profile sensor count does not match the model")
        if len(self.sensor_snr_db) != config.n_sensors:
            raise ValueError("profile SNR count does not match the model")
        if len(self.sensor_ber) != config.n_sensors:
            raise ValueError("profile BER count does not match the model")
        if len(self.sensor_capacity_depth) != config.n_sensors:
            raise ValueError("profile capacity-depth count does not match the model")
        positions = np.asarray(self.sensor_positions_m, dtype=float)
        if positions.shape != (config.n_sensors, 2) or not np.all(np.isfinite(positions)):
            raise ValueError("sensor positions must be finite 2-D coordinates")
        if np.unique(positions, axis=0).shape[0] != config.n_sensors:
            raise ValueError("sensor positions must be distinct")
        ber = np.asarray(self.sensor_ber, dtype=float)
        if np.any((ber < 0.0) | (ber >= 0.5)):
            raise ValueError("BER must lie in [0, 0.5)")
        if not np.all(np.isfinite(np.asarray(self.sensor_snr_db, dtype=float))):
            raise ValueError("sensor SNR values must be finite")
        if sorted(self.sensor_capacity_depth) != [1, 2, 3]:
            raise ValueError("Stage 0 requires one exogenous 1/2/3-bit capacity class")


@dataclass(frozen=True)
class Observation:
    profile_id: str
    seed: int
    trial: int
    normalized_samples: np.ndarray
    bit_flip_uniforms: np.ndarray
    true_position_index: int


@dataclass(frozen=True)
class PosteriorSummary:
    estimate_m: np.ndarray
    posterior_risk_m2: float
    squared_error_m2: float
    posterior_entropy: float
    posterior_nll: float
    credible90_contains_true: bool


@dataclass(frozen=True)
class LikelihoodCache:
    log_cell_likelihoods: tuple[np.ndarray, ...]
    true_position_index: int


def default_public_profiles() -> tuple[PublicProfile, ...]:
    """Return frozen public geometry/reliability states for development."""

    return (
        PublicProfile(
            "balanced",
            ((-95.0, 0.0), (95.0, 0.0), (0.0, 145.0)),
            (5.0, 5.0, 5.0),
            (0.02, 0.02, 0.02),
            (1, 2, 3),
        ),
        PublicProfile(
            "left_reliable",
            ((-125.0, -10.0), (80.0, 5.0), (15.0, 150.0)),
            (11.0, -1.0, 5.0),
            (0.005, 0.10, 0.03),
            (2, 1, 3),
        ),
        PublicProfile(
            "right_reliable",
            ((-80.0, 10.0), (125.0, -15.0), (-10.0, 155.0)),
            (-2.0, 10.0, 4.0),
            (0.10, 0.01, 0.04),
            (1, 3, 2),
        ),
        PublicProfile(
            "top_degraded",
            ((-110.0, -5.0), (105.0, 15.0), (20.0, 175.0)),
            (4.0, 8.0, -3.0),
            (0.03, 0.015, 0.12),
            (2, 3, 1),
        ),
    )


def enumerate_allocations(config: ModelConfig) -> tuple[tuple[int, ...], ...]:
    config.validate()
    allocations = tuple(
        allocation
        for allocation in itertools.product(
            range(config.max_bits_per_real + 1), repeat=config.n_cells
        )
        if sum(allocation) == config.bit_units
    )
    if not allocations:
        raise RuntimeError("allocation space is empty")
    return allocations


def allocation_payload_bits(allocation: Sequence[int]) -> int:
    return 2 * int(sum(int(value) for value in allocation))


def quantizer_edges(bits_per_real: int) -> np.ndarray:
    if bits_per_real < 1:
        raise ValueError("bits_per_real must be positive")
    levels = 2**bits_per_real
    internal = norm.ppf(np.arange(1, levels, dtype=float) / levels)
    return np.concatenate(([-np.inf], internal, [np.inf]))


def bsc_transition_probability(
    transmitted_code: int,
    received_code: int,
    bits: int,
    ber: float,
) -> float:
    distance = int((transmitted_code ^ received_code).bit_count())
    return float((1.0 - ber) ** (bits - distance) * ber**distance)


def real_code_probability(
    mean: np.ndarray,
    noise_std: float,
    bits: int,
    received_code: int,
    ber: float,
) -> np.ndarray:
    edges = quantizer_edges(bits)
    upper = norm.cdf((edges[1:, None] - mean[None, :]) / noise_std)
    lower = norm.cdf((edges[:-1, None] - mean[None, :]) / noise_std)
    true_probabilities = np.maximum(upper - lower, 0.0)
    transitions = np.asarray(
        [
            bsc_transition_probability(code, received_code, bits, ber)
            for code in range(2**bits)
        ],
        dtype=float,
    )
    return transitions @ true_probabilities


def complex_code_probability(
    mean: np.ndarray,
    noise_std: float,
    bits: int,
    received_code: int,
    ber: float,
) -> np.ndarray:
    levels = 2**bits
    real_code = received_code // levels
    imag_code = received_code % levels
    p_real = real_code_probability(mean.real, noise_std, bits, real_code, ber)
    p_imag = real_code_probability(mean.imag, noise_std, bits, imag_code, ber)
    return np.maximum(p_real * p_imag, np.finfo(float).tiny)


def _quantize_real(value: float, bits: int) -> int:
    edges = quantizer_edges(bits)
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def _apply_bsc(code: int, bits: int, uniforms: np.ndarray, ber: float) -> int:
    received = int(code)
    for bit_index in range(bits):
        if float(uniforms[bit_index]) < ber:
            received ^= 1 << bit_index
    return received


def received_complex_code(
    value: complex,
    bits: int,
    uniforms: np.ndarray,
    ber: float,
) -> int:
    levels = 2**bits
    real_code = _quantize_real(float(value.real), bits)
    imag_code = _quantize_real(float(value.imag), bits)
    real_rx = _apply_bsc(real_code, bits, uniforms[0], ber)
    imag_rx = _apply_bsc(imag_code, bits, uniforms[1], ber)
    return int(real_rx * levels + imag_rx)


class NPNBQModel:
    """Matched finite Bayesian model used only for the Stage 0 certificate."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.config.validate()
        self.positions_m = np.asarray(
            list(itertools.product(self.config.x_grid_m, self.config.y_grid_m)),
            dtype=float,
        )
        self.position_prior = np.full(
            self.positions_m.shape[0], 1.0 / self.positions_m.shape[0], dtype=float
        )
        self._nuisance_source, self._nuisance_gain = self._build_nuisance_states()

    @property
    def nuisance_state_count(self) -> int:
        return int(self._nuisance_source.shape[0])

    def _build_nuisance_states(self) -> tuple[np.ndarray, np.ndarray]:
        source_phase_states = list(
            itertools.product(
                self.config.source_phase_grid_rad,
                repeat=self.config.n_frequencies,
            )
        )
        gain_phase_states = list(
            itertools.product(
                self.config.gain_phase_grid_rad,
                repeat=self.config.n_sensors - 1,
            )
        )
        source_rows: list[np.ndarray] = []
        gain_rows: list[np.ndarray] = []
        for source_phase in source_phase_states:
            for source_ratio in self.config.source_amplitude_ratio_grid:
                source_amplitude = np.ones(self.config.n_frequencies, dtype=float)
                source_amplitude[1] = float(source_ratio)
                source_state = source_amplitude * np.exp(
                    1j * np.asarray(source_phase, dtype=float)
                )
                for gain_phase in gain_phase_states:
                    for gain_amplitude in itertools.product(
                        self.config.gain_amplitude_grid,
                        repeat=self.config.n_sensors - 1,
                    ):
                        gains = np.ones(self.config.n_sensors, dtype=complex)
                        gains[1:] = np.asarray(gain_amplitude, dtype=float) * np.exp(
                            1j * np.asarray(gain_phase, dtype=float)
                        )
                        source_rows.append(source_state)
                        gain_rows.append(gains)
        return np.asarray(source_rows), np.asarray(gain_rows)

    def profile_scales(self, profile: PublicProfile) -> tuple[np.ndarray, np.ndarray]:
        profile.validate(self.config)
        amplitudes = np.sqrt(10.0 ** (np.asarray(profile.sensor_snr_db) / 10.0))
        scales = np.sqrt((amplitudes**2 + self.config.noise_variance) / 2.0)
        noise_std = math.sqrt(self.config.noise_variance / 2.0) / scales
        return amplitudes, noise_std

    def _mean_tensor(self, profile: PublicProfile) -> np.ndarray:
        """Return normalized means with shape [position, nuisance, cell]."""

        positions = self.positions_m
        sensors = np.asarray(profile.sensor_positions_m, dtype=float)
        amplitudes, _ = self.profile_scales(profile)
        scales = np.sqrt((amplitudes**2 + self.config.noise_variance) / 2.0)
        distances = np.linalg.norm(
            positions[:, None, :] - sensors[None, :, :], axis=-1
        )
        relative_distance = distances - distances[:, [0]]
        means = np.empty(
            (
                positions.shape[0],
                self.nuisance_state_count,
                self.config.n_cells,
            ),
            dtype=complex,
        )
        for sensor_index in range(self.config.n_sensors):
            for frequency_index, frequency_hz in enumerate(
                self.config.frequencies_hz
            ):
                cell = sensor_index * self.config.n_frequencies + frequency_index
                delay_phase = np.exp(
                    -1j
                    * 2.0
                    * np.pi
                    * frequency_hz
                    * relative_distance[:, sensor_index]
                    / self.config.sound_speed_m_s
                )
                nuisance = (
                    self._nuisance_source[:, frequency_index]
                    * self._nuisance_gain[:, sensor_index]
                )
                means[:, :, cell] = (
                    amplitudes[sensor_index]
                    / scales[sensor_index]
                    * delay_phase[:, None]
                    * nuisance[None, :]
                )
        return means

    def generate_observations(
        self,
        profile: PublicProfile,
        seed: int,
        n_trials: int,
    ) -> list[Observation]:
        profile.validate(self.config)
        if n_trials < 1:
            raise ValueError("n_trials must be positive")
        rng = np.random.default_rng(seed)
        amplitudes, _ = self.profile_scales(profile)
        scales = np.sqrt((amplitudes**2 + self.config.noise_variance) / 2.0)
        sensors = np.asarray(profile.sensor_positions_m, dtype=float)
        observations: list[Observation] = []
        for trial in range(n_trials):
            position_index = int(rng.integers(self.positions_m.shape[0]))
            position = self.positions_m[position_index]
            source_phase = rng.choice(
                np.asarray(self.config.source_phase_grid_rad),
                size=self.config.n_frequencies,
            )
            source_amplitude = np.ones(self.config.n_frequencies, dtype=float)
            source_amplitude[1] = float(
                rng.choice(np.asarray(self.config.source_amplitude_ratio_grid))
            )
            gain_phase = np.zeros(self.config.n_sensors, dtype=float)
            gain_phase[1:] = rng.choice(
                np.asarray(self.config.gain_phase_grid_rad),
                size=self.config.n_sensors - 1,
            )
            gain_amplitude = np.ones(self.config.n_sensors, dtype=float)
            gain_amplitude[1:] = rng.choice(
                np.asarray(self.config.gain_amplitude_grid),
                size=self.config.n_sensors - 1,
            )
            distances = np.linalg.norm(sensors - position[None, :], axis=1)
            relative_distance = distances - distances[0]
            samples = np.empty(
                (self.config.n_sensors, self.config.n_frequencies), dtype=complex
            )
            for sensor_index in range(self.config.n_sensors):
                for frequency_index, frequency_hz in enumerate(
                    self.config.frequencies_hz
                ):
                    phase = (
                        source_phase[frequency_index]
                        + gain_phase[sensor_index]
                        - 2.0
                        * np.pi
                        * frequency_hz
                        * relative_distance[sensor_index]
                        / self.config.sound_speed_m_s
                    )
                    mean = (
                        amplitudes[sensor_index]
                        * source_amplitude[frequency_index]
                        * gain_amplitude[sensor_index]
                        * np.exp(1j * phase)
                    )
                    noise = math.sqrt(self.config.noise_variance / 2.0) * (
                        rng.normal() + 1j * rng.normal()
                    )
                    samples[sensor_index, frequency_index] = (
                        mean + noise
                    ) / scales[sensor_index]
            flip_uniforms = rng.random(
                (
                    self.config.n_cells,
                    2,
                    self.config.max_bits_per_real,
                )
            )
            observations.append(
                Observation(
                    profile_id=profile.profile_id,
                    seed=seed,
                    trial=trial,
                    normalized_samples=samples,
                    bit_flip_uniforms=flip_uniforms,
                    true_position_index=position_index,
                )
            )
        return observations

    def build_likelihood_cache(
        self,
        profile: PublicProfile,
        observation: Observation,
    ) -> LikelihoodCache:
        if observation.profile_id != profile.profile_id:
            raise ValueError("observation and public profile do not match")
        mean_tensor = self._mean_tensor(profile)
        _, noise_std = self.profile_scales(profile)
        ber = np.asarray(profile.sensor_ber, dtype=float)
        per_bit: list[np.ndarray] = []
        flattened = observation.normalized_samples.reshape(-1)
        for bits in range(1, self.config.max_bits_per_real + 1):
            bit_likelihood = np.empty(
                (
                    self.config.n_cells,
                    self.positions_m.shape[0],
                    self.nuisance_state_count,
                ),
                dtype=float,
            )
            for cell in range(self.config.n_cells):
                sensor_index = cell // self.config.n_frequencies
                received_code = received_complex_code(
                    flattened[cell],
                    bits,
                    observation.bit_flip_uniforms[cell],
                    float(ber[sensor_index]),
                )
                probabilities = complex_code_probability(
                    mean_tensor[:, :, cell].reshape(-1),
                    float(noise_std[sensor_index]),
                    bits,
                    received_code,
                    float(ber[sensor_index]),
                ).reshape(self.positions_m.shape[0], self.nuisance_state_count)
                bit_likelihood[cell] = np.log(probabilities)
            per_bit.append(bit_likelihood)
        return LikelihoodCache(tuple(per_bit), observation.true_position_index)

    def posterior_from_cache(
        self,
        cache: LikelihoodCache,
        allocation: Sequence[int],
    ) -> np.ndarray:
        if len(allocation) != self.config.n_cells:
            raise ValueError("allocation length does not match the cell count")
        log_terms = np.zeros(
            (self.positions_m.shape[0], self.nuisance_state_count), dtype=float
        )
        for cell, bits in enumerate(allocation):
            bits = int(bits)
            if bits < 0 or bits > self.config.max_bits_per_real:
                raise ValueError("allocation contains an invalid bit depth")
            if bits:
                log_terms += cache.log_cell_likelihoods[bits - 1][cell]
        log_likelihood = logsumexp(log_terms, axis=1) - math.log(
            self.nuisance_state_count
        )
        log_posterior = log_likelihood + np.log(self.position_prior)
        log_posterior -= logsumexp(log_posterior)
        posterior = np.exp(log_posterior)
        if not np.all(np.isfinite(posterior)):
            raise RuntimeError("posterior contains non-finite values")
        return posterior

    def posterior_unquantized(
        self,
        profile: PublicProfile,
        observation: Observation,
    ) -> np.ndarray:
        mean_tensor = self._mean_tensor(profile)
        _, noise_std = self.profile_scales(profile)
        values = observation.normalized_samples.reshape(-1)
        log_terms = np.zeros(
            (self.positions_m.shape[0], self.nuisance_state_count), dtype=float
        )
        for cell, value in enumerate(values):
            sensor_index = cell // self.config.n_frequencies
            variance = 2.0 * float(noise_std[sensor_index]) ** 2
            difference = value - mean_tensor[:, :, cell]
            log_terms += -np.abs(difference) ** 2 / variance - math.log(
                math.pi * variance
            )
        log_likelihood = logsumexp(log_terms, axis=1) - math.log(
            self.nuisance_state_count
        )
        log_posterior = log_likelihood + np.log(self.position_prior)
        log_posterior -= logsumexp(log_posterior)
        return np.exp(log_posterior)

    def summarize_posterior(
        self,
        posterior: np.ndarray,
        true_position_index: int,
    ) -> PosteriorSummary:
        posterior = np.asarray(posterior, dtype=float)
        if posterior.shape != (self.positions_m.shape[0],):
            raise ValueError("posterior shape does not match the position grid")
        if not math.isclose(float(np.sum(posterior)), 1.0, abs_tol=1e-9):
            raise ValueError("posterior must be normalized")
        estimate = posterior @ self.positions_m
        deviations = self.positions_m - estimate[None, :]
        posterior_risk = float(np.sum(posterior * np.sum(deviations**2, axis=1)))
        true_position = self.positions_m[int(true_position_index)]
        squared_error = float(np.sum((estimate - true_position) ** 2))
        entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-300))))
        nll = float(-math.log(max(float(posterior[int(true_position_index)]), 1e-300)))
        order = np.argsort(posterior)[::-1]
        cumulative = np.cumsum(posterior[order])
        included = order[: int(np.searchsorted(cumulative, 0.9, side="left")) + 1]
        return PosteriorSummary(
            estimate_m=np.asarray(estimate, dtype=float),
            posterior_risk_m2=posterior_risk,
            squared_error_m2=squared_error,
            posterior_entropy=entropy,
            posterior_nll=nll,
            credible90_contains_true=bool(int(true_position_index) in set(included)),
        )

    def mean_design_risk(
        self,
        caches: Sequence[LikelihoodCache],
        allocation: Sequence[int],
    ) -> float:
        return float(np.mean(self.design_risks(caches, allocation)))

    def design_risks(
        self,
        caches: Sequence[LikelihoodCache],
        allocation: Sequence[int],
    ) -> np.ndarray:
        risks = np.asarray(
            [
            self.summarize_posterior(
                self.posterior_from_cache(cache, allocation),
                cache.true_position_index,
            ).posterior_risk_m2
            for cache in caches
            ],
            dtype=float,
        )
        if risks.size == 0 or not np.all(np.isfinite(risks)):
            raise ValueError("design risks must be non-empty and finite")
        return risks

    def canonical_geometry_profile(self, profile: PublicProfile) -> PublicProfile:
        return PublicProfile(
            profile_id=f"{profile.profile_id}-reliability-surrogate",
            sensor_positions_m=((-100.0, 0.0), (100.0, 0.0), (0.0, 150.0)),
            sensor_snr_db=profile.sensor_snr_db,
            sensor_ber=profile.sensor_ber,
            sensor_capacity_depth=profile.sensor_capacity_depth,
        )

    def nominal_reliability_profile(self, profile: PublicProfile) -> PublicProfile:
        linear = 10.0 ** (np.asarray(profile.sensor_snr_db, dtype=float) / 10.0)
        common_snr = float(10.0 * np.log10(np.mean(linear)))
        common_ber = float(np.mean(profile.sensor_ber))
        return PublicProfile(
            profile_id=f"{profile.profile_id}-geometry-surrogate",
            sensor_positions_m=profile.sensor_positions_m,
            sensor_snr_db=(common_snr,) * self.config.n_sensors,
            sensor_ber=(common_ber,) * self.config.n_sensors,
            sensor_capacity_depth=profile.sensor_capacity_depth,
        )

    def equal_bit_allocation(self) -> tuple[int, ...]:
        bits = self.config.bit_units // self.config.n_cells
        allocation = (bits,) * self.config.n_cells
        if allocation_payload_bits(allocation) != self.config.payload_bits:
            raise RuntimeError("the frozen budget does not admit equal allocation")
        return allocation

    def jiang_style_allocation(self, profile: PublicProfile) -> tuple[int, ...]:
        """Strict-budget adaptation of Jiang's preassigned mixed-depth principle.

        The original paper fixes 1/2/3-bit sensor rows.  Here those three depths
        are assigned by public slow-link capability and repeated at both bins.
        Threshold optimization from Jiang is intentionally not claimed.
        """

        depth_by_sensor = np.asarray(profile.sensor_capacity_depth, dtype=int)
        allocation = tuple(
            int(depth_by_sensor[sensor])
            for sensor in range(self.config.n_sensors)
            for _ in range(self.config.n_frequencies)
        )
        if allocation_payload_bits(allocation) != self.config.payload_bits:
            raise RuntimeError("Jiang-style adaptation violates the payload budget")
        return allocation

    def projected_fisher_score(
        self,
        profile: PublicProfile,
        allocation: Sequence[int],
    ) -> float:
        """Return an ROI-averaged nuisance-projected phase-information proxy."""

        sensors = np.asarray(profile.sensor_positions_m, dtype=float)
        snr = 10.0 ** (np.asarray(profile.sensor_snr_db, dtype=float) / 10.0)
        ber = np.asarray(profile.sensor_ber, dtype=float)
        n_nuisance = self.config.n_frequencies + self.config.n_sensors - 1
        scores: list[float] = []
        for position in self.positions_m:
            rows: list[np.ndarray] = []
            weights: list[float] = []
            distances = np.linalg.norm(position[None, :] - sensors, axis=1)
            unit = (position[None, :] - sensors) / np.maximum(
                distances[:, None], 1e-12
            )
            for sensor in range(self.config.n_sensors):
                for frequency_index, frequency_hz in enumerate(
                    self.config.frequencies_hz
                ):
                    cell = sensor * self.config.n_frequencies + frequency_index
                    bits = int(allocation[cell])
                    if bits == 0:
                        continue
                    gradient = (
                        -2.0
                        * np.pi
                        * frequency_hz
                        / self.config.sound_speed_m_s
                        * (unit[sensor] - unit[0])
                    )
                    row = np.zeros(2 + n_nuisance, dtype=float)
                    row[:2] = gradient
                    row[2 + frequency_index] = 1.0
                    if sensor > 0:
                        row[2 + self.config.n_frequencies + sensor - 1] = 1.0
                    bit_efficiency = 1.0 - 4.0 ** (-bits)
                    reliability = snr[sensor] / (1.0 + snr[sensor])
                    reliability *= (1.0 - 2.0 * ber[sensor]) ** 2
                    rows.append(row)
                    weights.append(float(bit_efficiency * reliability))
            if not rows:
                scores.append(-1e12)
                continue
            design = np.vstack(rows)
            weight = np.diag(np.asarray(weights))
            information = design.T @ weight @ design
            j_pp = information[:2, :2]
            j_pe = information[:2, 2:]
            j_ee = information[2:, 2:]
            projected = j_pp - j_pe @ np.linalg.pinv(j_ee, rcond=1e-10) @ j_pe.T
            projected = 0.5 * (projected + projected.T)
            sign, logdet = np.linalg.slogdet(projected + 1e-12 * np.eye(2))
            scores.append(float(logdet if sign > 0 else -1e12))
        return float(np.mean(scores))


def allocation_label(allocation: Iterable[int]) -> str:
    return "-".join(str(int(value)) for value in allocation)
