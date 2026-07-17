"""Core primitives for posterior-driven active TDOA refinement.

The module is intentionally independent from the main DAE/urban8 pipeline.  It
implements the asymmetric passive two-receiver model used by the frozen PFRS
study, but changes the communication protocol from a fixed mask to sequential
frequency requests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy.linalg import eigvalsh
from scipy.signal import find_peaks


@dataclass(frozen=True)
class SpectralModel:
    bin_indices: np.ndarray
    omega: np.ndarray
    power: np.ndarray
    source: np.ndarray
    rho: complex


@dataclass(frozen=True)
class QuantizedSpectrum:
    values: np.ndarray
    reconstructed_scale: float
    step: float
    complex_error_variance: float
    clipping_rate: float


@dataclass(frozen=True)
class PosteriorMode:
    tau: float
    probability: float
    variance: float
    peak_probability: float


@dataclass(frozen=True)
class PosteriorResult:
    estimate: float
    posterior: np.ndarray
    cost: np.ndarray
    modes: tuple[PosteriorMode, ...]
    boundary: bool


@dataclass
class BitLedger:
    coefficient_bits: int = 0
    scale_bits: int = 0
    action_feedback_bits: int = 0
    decision_feedback_bits: int = 0

    @property
    def total_bits(self) -> int:
        return (
            self.coefficient_bits
            + self.scale_bits
            + self.action_feedback_bits
            + self.decision_feedback_bits
        )


@dataclass(frozen=True)
class ActiveEstimate:
    estimate: float
    selected_positions: tuple[int, ...]
    posterior: PosteriorResult
    ledger: BitLedger
    stop_reason: str
    action_trace: tuple[dict[str, float | int | str], ...] = field(default_factory=tuple)


def _bin_indices(n_bins: int) -> np.ndarray:
    if n_bins < 4 or n_bins % 2:
        raise ValueError("n_bins must be an even integer >= 4")
    return np.arange(-n_bins // 2, n_bins // 2, dtype=int)


def _source_power(indices: np.ndarray, profile: str, floor: float) -> np.ndarray:
    x = indices / max(float(np.max(np.abs(indices))), 1.0)
    if profile == "flat":
        power = np.ones_like(x, dtype=float)
    elif profile == "center_weighted":
        envelope = np.exp(-0.5 * (x / 0.58) ** 2)
        ripple = 1.0 + 0.20 * np.cos(2.3 * np.pi * x + 0.35)
        power = floor + envelope * ripple
    else:
        raise ValueError(f"Unsupported source profile: {profile}")
    power = np.maximum(power, floor)
    return power / np.mean(power)


def build_spectral_model(
    n_bins: int = 16,
    source_profile: str = "center_weighted",
    source_floor: float = 0.15,
    rho_abs: float = 0.85,
    rho_phase_rad: float = 0.40,
    seed: int = 20260716,
) -> SpectralModel:
    indices = _bin_indices(n_bins)
    omega = 2.0 * np.pi * indices / n_bins
    power = _source_power(indices, source_profile, source_floor)
    rng = np.random.default_rng(seed + 17)
    phase = rng.uniform(-np.pi, np.pi, size=n_bins)
    source = np.sqrt(power) * np.exp(1j * phase)
    rho = rho_abs * np.exp(1j * rho_phase_rad)
    return SpectralModel(indices, omega, power, source, rho)


def profiled_cost_grid(
    y_a: np.ndarray,
    y_b: np.ndarray,
    omega: np.ndarray,
    tau_grid: np.ndarray,
    sigma_a2: float,
    sigma_b2: float,
) -> np.ndarray:
    """Evaluate the nuisance-profiled negative log-likelihood on a delay grid."""
    y_a = np.asarray(y_a, dtype=np.complex128)
    y_b = np.asarray(y_b, dtype=np.complex128)
    omega = np.asarray(omega, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if y_a.ndim != 1 or y_b.shape != y_a.shape:
        raise ValueError("y_a and y_b must be one-dimensional arrays of equal length")
    if omega.shape != y_a.shape:
        raise ValueError("omega must match the selected spectrum shape")
    if tau_grid.ndim != 1 or tau_grid.size < 3:
        raise ValueError("tau_grid must be one-dimensional with at least three points")
    if sigma_a2 <= 0.0 or sigma_b2 <= 0.0:
        raise ValueError("noise variances must be positive")

    steering = np.exp(-1j * omega[:, None] * tau_grid[None, :])
    cross = (np.conj(y_b) * y_a) @ steering
    cross *= -1.0 / math.sqrt(sigma_a2 * sigma_b2)
    a = np.sum(np.abs(y_b) ** 2) / sigma_b2
    d = np.sum(np.abs(y_a) ** 2) / sigma_a2
    discriminant = np.sqrt(max((a - d) ** 2, 0.0) + 4.0 * np.abs(cross) ** 2)
    return np.maximum(0.5 * (a + d - discriminant).real, 0.0)


def posterior_from_cost(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 1 or not np.all(np.isfinite(cost)):
        raise ValueError("cost must be a finite one-dimensional array")
    log_weight = -(cost - float(np.min(cost)))
    log_weight -= float(np.max(log_weight))
    weight = np.exp(log_weight)
    total = float(np.sum(weight))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("posterior normalization failed")
    return weight / total


def _parabolic_minimum(tau_grid: np.ndarray, cost: np.ndarray, index: int) -> float:
    if index <= 0 or index >= cost.size - 1:
        return float(tau_grid[index])
    left, middle, right = cost[index - 1 : index + 2]
    denominator = left - 2.0 * middle + right
    scale = max(abs(left), abs(middle), abs(right), 1.0)
    if denominator <= np.finfo(float).eps * scale:
        return float(tau_grid[index])
    offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    return float(tau_grid[index] + offset * (tau_grid[1] - tau_grid[0]))


def extract_posterior_modes(
    tau_grid: np.ndarray,
    posterior: np.ndarray,
    cost: np.ndarray,
    max_modes: int = 5,
    min_separation_samples: float = 0.35,
) -> tuple[PosteriorMode, ...]:
    if max_modes < 1:
        raise ValueError("max_modes must be positive")
    step = float(tau_grid[1] - tau_grid[0])
    min_distance = max(1, int(math.ceil(min_separation_samples / step)))
    peaks, _ = find_peaks(posterior, distance=min_distance)
    endpoint_peaks = []
    if posterior[0] > posterior[1]:
        endpoint_peaks.append(0)
    if posterior[-1] > posterior[-2]:
        endpoint_peaks.append(posterior.size - 1)
    candidates = np.unique(np.concatenate([peaks, endpoint_peaks])).astype(int)
    if candidates.size == 0:
        candidates = np.asarray([int(np.argmax(posterior))], dtype=int)
    selected = np.sort(candidates)

    # Partition the complete posterior at minima between retained peaks.
    boundaries = [0]
    for left_peak, right_peak in zip(selected[:-1], selected[1:], strict=True):
        valley = left_peak + int(np.argmin(posterior[left_peak : right_peak + 1]))
        boundaries.append(valley + 1)
    boundaries.append(posterior.size)

    modes = []
    for region_index, peak in enumerate(selected):
        lo, hi = boundaries[region_index], boundaries[region_index + 1]
        region_probability = posterior[lo:hi]
        mass = float(np.sum(region_probability))
        if mass <= 0.0:
            continue
        mean = float(np.sum(tau_grid[lo:hi] * region_probability) / mass)
        variance = float(
            np.sum((tau_grid[lo:hi] - mean) ** 2 * region_probability) / mass
        )
        modes.append(
            PosteriorMode(
                tau=_parabolic_minimum(tau_grid, cost, int(peak)),
                probability=mass,
                variance=max(variance, 0.0),
                peak_probability=float(posterior[peak]),
            )
        )
    return tuple(
        sorted(modes, key=lambda item: item.probability, reverse=True)[:max_modes]
    )


def estimate_subset(
    y_a: np.ndarray,
    y_b: np.ndarray,
    positions: Sequence[int],
    model: SpectralModel,
    tau_grid: np.ndarray,
    sigma_a2: float,
    sigma_b2: float,
    max_modes: int = 5,
    min_mode_separation_samples: float = 0.35,
) -> PosteriorResult:
    idx = np.asarray(tuple(positions), dtype=int)
    if idx.ndim != 1 or idx.size < 2 or np.unique(idx).size != idx.size:
        raise ValueError("positions must contain at least two unique indices")
    if np.min(idx) < 0 or np.max(idx) >= model.bin_indices.size:
        raise ValueError("selected position lies outside the spectrum")
    cost = profiled_cost_grid(
        y_a[idx], y_b[idx], model.omega[idx], tau_grid, sigma_a2, sigma_b2
    )
    posterior = posterior_from_cost(cost)
    coarse = int(np.argmin(cost))
    estimate = _parabolic_minimum(tau_grid, cost, coarse)
    modes = extract_posterior_modes(
        tau_grid,
        posterior,
        cost,
        max_modes=max_modes,
        min_separation_samples=min_mode_separation_samples,
    )
    return PosteriorResult(
        estimate=estimate,
        posterior=posterior,
        cost=cost,
        modes=modes,
        boundary=coarse in (0, tau_grid.size - 1),
    )


def quantize_complex_spectrum(
    values: np.ndarray,
    qbits: int,
    scale_bits: int,
    rng: np.random.Generator,
    log2_scale_bounds: tuple[float, float] = (-6.0, 4.0),
) -> QuantizedSpectrum:
    """Apply public subtractive-dither I/Q quantization with a coded scale."""
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional complex spectrum")
    if qbits < 2 or scale_bits < 2:
        raise ValueError("qbits and scale_bits must both be at least two")
    lo, hi = log2_scale_bounds
    if not lo < hi:
        raise ValueError("invalid log2 scale range")

    raw_scale = max(
        float(np.max(np.abs(values.real))),
        float(np.max(np.abs(values.imag))),
        2.0**lo,
    )
    levels = 2**scale_bits - 1
    normalized = np.clip((math.log2(raw_scale) - lo) / (hi - lo), 0.0, 1.0)
    # Round upward so quantizing the scale itself does not create avoidable
    # coefficient clipping. Values above the public range remain auditable via
    # clipping_rate.
    scale_code = int(np.ceil(normalized * levels))
    reconstructed_scale = 2.0 ** (lo + (hi - lo) * scale_code / levels)

    signed_peak = 2 ** (qbits - 1) - 1
    step = reconstructed_scale / signed_peak
    dither_real = rng.uniform(-0.5 * step, 0.5 * step, size=values.size)
    dither_imag = rng.uniform(-0.5 * step, 0.5 * step, size=values.size)
    code_real_unclipped = np.rint((values.real + dither_real) / step)
    code_imag_unclipped = np.rint((values.imag + dither_imag) / step)
    code_real = np.clip(code_real_unclipped, -signed_peak, signed_peak)
    code_imag = np.clip(code_imag_unclipped, -signed_peak, signed_peak)
    reconstructed = (code_real * step - dither_real) + 1j * (
        code_imag * step - dither_imag
    )
    clipped = (code_real != code_real_unclipped) | (code_imag != code_imag_unclipped)
    return QuantizedSpectrum(
        values=reconstructed,
        reconstructed_scale=reconstructed_scale,
        step=step,
        complex_error_variance=step**2 / 6.0,
        clipping_rate=float(np.mean(clipped)),
    )


def fisher_information_from_power(
    positions: Sequence[int],
    power: np.ndarray,
    model: SpectralModel,
    sigma_a2: float,
    sigma_b2: float,
    rho_abs: float = 1.0,
) -> float:
    idx = np.asarray(tuple(positions), dtype=int)
    selected_power = np.maximum(np.asarray(power, dtype=float)[idx], 0.0)
    total = float(np.sum(selected_power))
    if total <= 0.0:
        return 0.0
    omega = model.omega[idx]
    mean_omega = float(np.sum(selected_power * omega) / total)
    centered = float(np.sum(selected_power * (omega - mean_omega) ** 2))
    scale = rho_abs**2 / (sigma_b2 + rho_abs**2 * sigma_a2)
    return 2.0 * scale * centered


def profiled_mean_distance_from_power(
    positions: Sequence[int],
    power: np.ndarray,
    model: SpectralModel,
    delay_difference: float,
    sigma_a2: float,
    sigma_b2: float,
    rho_abs: float = 1.0,
) -> float:
    idx = np.asarray(tuple(positions), dtype=int)
    selected_power = np.maximum(np.asarray(power, dtype=float)[idx], 0.0)
    source = np.sqrt(selected_power).astype(np.complex128)
    y_b = rho_abs * source
    steered_a = source * np.exp(-1j * model.omega[idx] * delay_difference)
    cross = np.vdot(y_b, steered_a)
    matrix = np.array(
        [
            [np.vdot(y_b, y_b).real, -cross],
            [-np.conj(cross), np.vdot(steered_a, steered_a).real],
        ],
        dtype=np.complex128,
    )
    metric = np.diag([sigma_b2, sigma_a2]).astype(np.complex128)
    return float(max(eigvalsh(matrix, metric)[0].real, 0.0))


def _action_support(modes: Sequence[PosteriorMode], grid_step: float) -> list[tuple[float, float]]:
    if len(modes) >= 2:
        total = sum(mode.probability for mode in modes)
        return [(mode.tau, mode.probability / total) for mode in modes]
    mode = modes[0]
    spread = max(math.sqrt(mode.variance), grid_step)
    return [(mode.tau - spread, 0.5), (mode.tau + spread, 0.5)]


def action_information_gain(
    current_positions: Sequence[int],
    candidate_position: int,
    modes: Sequence[PosteriorMode],
    estimated_power: np.ndarray,
    model: SpectralModel,
    sigma_a2: float,
    sigma_b2: float,
    action_bits: int,
    grid_step: float,
) -> float:
    if not modes or action_bits <= 0:
        raise ValueError("modes must be non-empty and action_bits must be positive")
    current = tuple(sorted(int(value) for value in current_positions))
    if candidate_position in current:
        raise ValueError("candidate position is already selected")
    updated = tuple(sorted((*current, int(candidate_position))))
    support = _action_support(modes, grid_step)
    gain = 0.0
    for left in range(len(support)):
        for right in range(left + 1, len(support)):
            tau_left, probability_left = support[left]
            tau_right, probability_right = support[right]
            separation = tau_left - tau_right
            current_distance = profiled_mean_distance_from_power(
                current,
                estimated_power,
                model,
                separation,
                sigma_a2,
                sigma_b2,
            )
            updated_distance = profiled_mean_distance_from_power(
                updated,
                estimated_power,
                model,
                separation,
                sigma_a2,
                sigma_b2,
            )
            gain += (
                probability_left
                * probability_right
                * max(updated_distance - current_distance, 0.0)
            )
    return gain / action_bits


def _observed_power(y_a: np.ndarray, sigma_a2: float) -> np.ndarray:
    raw = np.abs(y_a) ** 2 - sigma_a2
    positive = raw[raw > 0.0]
    floor = max(float(np.median(positive)) * 1e-3, 1e-8) if positive.size else 1e-8
    return np.maximum(raw, floor)


def _oracle_next_position(
    current: tuple[int, ...],
    remaining: Sequence[int],
    y_a: np.ndarray,
    y_b: np.ndarray,
    true_tau: float,
    model: SpectralModel,
    tau_grid: np.ndarray,
    sigma_a2: float,
    sigma_b2: float,
    max_modes: int,
    min_mode_separation_samples: float,
) -> int:
    scored = []
    for candidate in remaining:
        result = estimate_subset(
            y_a,
            y_b,
            (*current, candidate),
            model,
            tau_grid,
            sigma_a2,
            sigma_b2,
            max_modes=max_modes,
            min_mode_separation_samples=min_mode_separation_samples,
        )
        scored.append(((result.estimate - true_tau) ** 2, int(candidate)))
    return min(scored)[1]


def run_active_refinement(
    y_a: np.ndarray,
    quantized_b: QuantizedSpectrum,
    base_positions: Sequence[int],
    model: SpectralModel,
    tau_grid: np.ndarray,
    sigma_a2: float,
    sigma_b2: float,
    qbits: int,
    scale_bits: int,
    max_bins: int,
    confidence_threshold: float,
    variance_target: float,
    strategy: Literal[
        "posterior",
        "posterior_oracle_power",
        "fisher",
        "top1",
        "random",
        "fixed",
        "oracle",
    ],
    rng: np.random.Generator,
    fixed_order: Sequence[int] = (),
    true_tau: float | None = None,
    max_modes: int = 5,
    min_mode_separation_samples: float = 0.35,
    allow_early_stop: bool = True,
) -> ActiveEstimate:
    selected = tuple(sorted(int(value) for value in base_positions))
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise ValueError("base_positions must contain at least two unique bins")
    if max_bins < len(selected) or max_bins > model.bin_indices.size:
        raise ValueError("max_bins is incompatible with the base layer")
    if strategy == "oracle" and true_tau is None:
        raise ValueError("oracle strategy requires true_tau")
    if not 0.0 < confidence_threshold < 1.0 or variance_target <= 0.0:
        raise ValueError("invalid stopping thresholds")

    ledger = BitLedger(
        coefficient_bits=len(selected) * 2 * qbits,
        scale_bits=scale_bits,
    )
    trace: list[dict[str, float | int | str]] = []
    effective_sigma_b2 = sigma_b2 + quantized_b.complex_error_variance
    grid_step = float(tau_grid[1] - tau_grid[0])

    while True:
        posterior = estimate_subset(
            y_a,
            quantized_b.values,
            selected,
            model,
            tau_grid,
            sigma_a2,
            effective_sigma_b2,
            max_modes=max_modes,
            min_mode_separation_samples=min_mode_separation_samples,
        )
        dominant = posterior.modes[0]
        confidence_stop = (
            not posterior.boundary
            and dominant.probability >= confidence_threshold
            and dominant.variance <= variance_target
        )
        if len(selected) >= max_bins:
            return ActiveEstimate(
                posterior.estimate,
                selected,
                posterior,
                ledger,
                "max_bins",
                tuple(trace),
            )
        if confidence_stop and allow_early_stop:
            ledger.decision_feedback_bits += 1
            return ActiveEstimate(
                posterior.estimate,
                selected,
                posterior,
                ledger,
                "posterior_confident",
                tuple(trace),
            )

        remaining = tuple(
            position
            for position in range(model.bin_indices.size)
            if position not in selected
        )
        action_bits = (
            0 if strategy == "fixed" else max(1, math.ceil(math.log2(len(remaining))))
        )
        if allow_early_stop:
            ledger.decision_feedback_bits += 1
        ledger.action_feedback_bits += action_bits
        ledger.coefficient_bits += 2 * qbits

        if strategy == "random":
            chosen = int(rng.choice(remaining))
            score = math.nan
        elif strategy == "fixed":
            available = [int(value) for value in fixed_order if value in remaining]
            if not available:
                raise RuntimeError("fixed refinement order ran out of valid positions")
            chosen = available[0]
            score = math.nan
        elif strategy == "oracle":
            chosen = _oracle_next_position(
                selected,
                remaining,
                y_a,
                quantized_b.values,
                float(true_tau),
                model,
                tau_grid,
                sigma_a2,
                effective_sigma_b2,
                max_modes,
                min_mode_separation_samples,
            )
            score = math.nan
        else:
            estimated_power = (
                model.power
                if strategy == "posterior_oracle_power"
                else _observed_power(y_a, sigma_a2)
            )
            if strategy == "fisher":
                current_information = fisher_information_from_power(
                    selected,
                    estimated_power,
                    model,
                    sigma_a2,
                    effective_sigma_b2,
                )
                scores = {
                    int(candidate): (
                        fisher_information_from_power(
                            (*selected, int(candidate)),
                            estimated_power,
                            model,
                            sigma_a2,
                            effective_sigma_b2,
                        )
                        - current_information
                    )
                    / (action_bits + 2 * qbits + int(allow_early_stop))
                    for candidate in remaining
                }
            else:
                action_modes = (
                    posterior.modes[:1] if strategy == "top1" else posterior.modes
                )
                scores = {
                    int(candidate): action_information_gain(
                        selected,
                        int(candidate),
                        action_modes,
                        estimated_power,
                        model,
                        sigma_a2,
                        effective_sigma_b2,
                        action_bits + 2 * qbits + int(allow_early_stop),
                        grid_step,
                    )
                    for candidate in remaining
                }
            chosen = max(scores, key=lambda key: (scores[key], -key))
            score = float(scores[chosen])

        trace.append(
            {
                "layer": len(selected) - len(tuple(base_positions)) + 1,
                "chosen_position": chosen,
                "chosen_bin": int(model.bin_indices[chosen]),
                "mode_probability": dominant.probability,
                "mode_variance": dominant.variance,
                "action_score_per_bit": score,
                "cumulative_bits": ledger.total_bits,
            }
        )
        selected = tuple(sorted((*selected, chosen)))
