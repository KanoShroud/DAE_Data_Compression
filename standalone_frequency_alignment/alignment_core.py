"""频谱嵌入、周期插值、连续时延估计和诊断指标。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class CompletionResult:
    spectrum: np.ndarray
    raw_mapped_spectrum: np.ndarray
    observed_consistency_error: float
    generated_missing_energy: float


@dataclass(frozen=True)
class DelayEstimate:
    delay_samples: float
    peak_value: float
    peak_sidelobe_ratio: float


def shifted_frequencies(n_fft: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(n_fft))


@lru_cache(maxsize=2)
def _cached_delay_steering(
    n_fft: int,
    lag_limit_samples: float,
    grid_step_samples: float,
) -> tuple[np.ndarray, np.ndarray]:
    lags = np.arange(
        -lag_limit_samples,
        lag_limit_samples + 0.5 * grid_step_samples,
        grid_step_samples,
    )
    steering = np.exp(2j * np.pi * np.outer(lags, shifted_frequencies(n_fft)))
    return lags, steering


def embed_observation(
    n_fft: int,
    indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """把实际传输的复频点嵌入共同频率网格，未传输处为零。"""

    idx = np.asarray(indices, dtype=int)
    val = np.asarray(values, dtype=complex)
    if idx.ndim != 1 or val.shape != idx.shape:
        raise ValueError("indices 和 values 必须是一维且形状相同")
    if np.unique(idx).size != idx.size or np.any((idx < 0) | (idx >= n_fft)):
        raise ValueError("indices 必须唯一且位于有效范围")
    spectrum = np.zeros(n_fft, dtype=complex)
    spectrum[idx] = val
    return spectrum


def _map_high_rate_spectrum_to_original(
    high_rate_time: np.ndarray,
    n_fft: int,
) -> np.ndarray:
    high_n = high_rate_time.size
    if high_n % n_fft != 0:
        raise ValueError("高采样率长度必须是原始长度的整数倍")
    factor = high_n // n_fft
    high_spectrum = np.fft.fftshift(np.fft.fft(high_rate_time, norm="ortho"))
    high_harmonics = np.rint(
        np.fft.fftshift(np.fft.fftfreq(high_n)) * high_n
    ).astype(int)
    original_harmonics = np.rint(
        np.fft.fftshift(np.fft.fftfreq(n_fft)) * n_fft
    ).astype(int)
    lookup = {harmonic: index for index, harmonic in enumerate(high_harmonics)}
    mapped = np.array(
        [high_spectrum[lookup[harmonic]] for harmonic in original_harmonics],
        dtype=complex,
    )
    return mapped / np.sqrt(factor)


def _periodic_linear_resample(time_signal: np.ndarray, output_length: int) -> np.ndarray:
    n = time_signal.size
    old_grid = np.arange(n + 1, dtype=float)
    new_grid = np.arange(output_length, dtype=float) * n / output_length
    periodic = np.concatenate([time_signal, time_signal[:1]])
    real = np.interp(new_grid, old_grid, periodic.real)
    imag = np.interp(new_grid, old_grid, periodic.imag)
    return real + 1j * imag


def _periodic_cubic_resample(time_signal: np.ndarray, output_length: int) -> np.ndarray:
    n = time_signal.size
    old_grid = np.arange(n + 1, dtype=float)
    new_grid = np.arange(output_length, dtype=float) * n / output_length
    periodic = np.concatenate([time_signal, time_signal[:1]])
    real_spline = CubicSpline(old_grid, periodic.real, bc_type="periodic")
    imag_spline = CubicSpline(old_grid, periodic.imag, bc_type="periodic")
    return real_spline(new_grid) + 1j * imag_spline(new_grid)


def complete_spectrum(
    sparse_spectrum: np.ndarray,
    observed_indices: np.ndarray,
    method: str,
    interpolation_factor: int,
) -> CompletionResult:
    """执行一种伪补全，并把结果重新映射到原始物理 DFT 网格。"""

    sparse = np.asarray(sparse_spectrum, dtype=complex)
    idx = np.asarray(observed_indices, dtype=int)
    if sparse.ndim != 1 or interpolation_factor < 1:
        raise ValueError("输入必须是一维，插值倍数必须为正整数")
    n_fft = sparse.size
    if method == "zero_fill":
        raw_mapped = sparse.copy()
    else:
        time_signal = np.fft.ifft(np.fft.ifftshift(sparse), norm="ortho")
        output_length = n_fft * interpolation_factor
        if method == "ideal_periodic":
            high_rate_time = signal.resample(time_signal, output_length)
        elif method == "polyphase_fir":
            high_rate_time = signal.resample_poly(
                time_signal,
                interpolation_factor,
                1,
                padtype="wrap",
            )
            high_rate_time = high_rate_time[:output_length]
        elif method == "linear_periodic":
            high_rate_time = _periodic_linear_resample(time_signal, output_length)
        elif method == "cubic_periodic":
            high_rate_time = _periodic_cubic_resample(time_signal, output_length)
        else:
            raise ValueError(f"未知补全方法 {method!r}")
        raw_mapped = _map_high_rate_spectrum_to_original(high_rate_time, n_fft)

    observed_error = float(
        np.linalg.norm(raw_mapped[idx] - sparse[idx])
        / max(np.linalg.norm(sparse[idx]), np.finfo(float).tiny)
    )
    missing_mask = np.ones(n_fft, dtype=bool)
    missing_mask[idx] = False
    generated_energy = float(np.sum(np.abs(raw_mapped[missing_mask]) ** 2))

    # 对比 TDOA 时必须保持传输频点原值不变，避免把滤波失真误称为补全收益。
    consistent = raw_mapped.copy()
    consistent[idx] = sparse[idx]
    return CompletionResult(
        spectrum=consistent,
        raw_mapped_spectrum=raw_mapped,
        observed_consistency_error=observed_error,
        generated_missing_energy=generated_energy,
    )


def estimate_continuous_delay(
    spectrum_i: np.ndarray,
    spectrum_reference: np.ndarray,
    valid_mask: np.ndarray,
    lag_limit_samples: float,
    grid_step_samples: float,
) -> DelayEstimate:
    """用频域连续网格似然估计 delay_i - delay_reference。"""

    x_i = np.asarray(spectrum_i, dtype=complex)
    x_ref = np.asarray(spectrum_reference, dtype=complex)
    mask = np.asarray(valid_mask, dtype=bool)
    if x_i.shape != x_ref.shape or mask.shape != x_i.shape:
        raise ValueError("两个频谱和 valid_mask 的形状必须一致")
    if np.count_nonzero(mask) < 2:
        return DelayEstimate(float("nan"), float("nan"), float("nan"))

    selected_indices = np.flatnonzero(mask)
    cross_spectrum = x_i[mask] * np.conj(x_ref[mask])
    numerical_floor = (
        100.0
        * np.finfo(float).eps
        * np.linalg.norm(x_i[mask])
        * np.linalg.norm(x_ref[mask])
    )
    if np.linalg.norm(cross_spectrum) <= max(numerical_floor, np.finfo(float).tiny):
        return DelayEstimate(float("nan"), float("nan"), float("nan"))
    active_floor = max(
        float(np.max(np.abs(cross_spectrum))) * 1e-12,
        numerical_floor / max(np.sqrt(cross_spectrum.size), 1.0),
    )
    active = np.abs(cross_spectrum) > active_floor
    if np.count_nonzero(active) < 2:
        return DelayEstimate(float("nan"), float("nan"), float("nan"))
    active_indices = selected_indices[active]
    cross_spectrum = cross_spectrum[active]
    lags, full_steering = _cached_delay_steering(
        x_i.size,
        float(lag_limit_samples),
        float(grid_step_samples),
    )
    score = np.abs(full_steering[:, active_indices] @ cross_spectrum)
    peak_index = int(np.argmax(score))
    peak = float(score[peak_index])
    exclusion = max(1, int(np.ceil(1.0 / grid_step_samples)))
    sidelobe_mask = np.ones(score.size, dtype=bool)
    left = max(0, peak_index - exclusion)
    right = min(score.size, peak_index + exclusion + 1)
    sidelobe_mask[left:right] = False
    sidelobe = float(np.max(score[sidelobe_mask])) if np.any(sidelobe_mask) else 0.0
    return DelayEstimate(
        delay_samples=float(lags[peak_index]),
        peak_value=peak,
        peak_sidelobe_ratio=sidelobe / max(peak, np.finfo(float).tiny),
    )


def complex_nmse_db(estimate: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(estimate - target) ** 2))
    denominator = float(np.sum(np.abs(target) ** 2))
    return 10.0 * np.log10(max(numerator, np.finfo(float).tiny) / max(denominator, np.finfo(float).tiny))


def phase_rmse_rad(
    estimate: np.ndarray,
    target: np.ndarray,
    relative_magnitude_floor: float = 0.05,
) -> float:
    magnitude = np.abs(target)
    target_scale = max(float(np.max(magnitude)), np.finfo(float).tiny)
    informative = (
        (magnitude >= relative_magnitude_floor * target_scale)
        & (np.abs(estimate) >= 1e-12 * target_scale)
    )
    if not np.any(informative):
        return float("nan")
    phase_error = np.angle(estimate[informative] * np.conj(target[informative]))
    return float(np.sqrt(np.mean(phase_error * phase_error)))


def complex_correlation(estimate: np.ndarray, target: np.ndarray) -> float:
    denominator = np.linalg.norm(estimate) * np.linalg.norm(target)
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(abs(np.vdot(target, estimate)) / denominator)


def interpolation_operator_rank(
    n_fft: int,
    observed_indices: np.ndarray,
    method: str,
    interpolation_factor: int,
    tolerance: float = 1e-10,
) -> int:
    """显式构造从 M 个观测到 N 个输出的线性算子并计算秩。"""

    idx = np.asarray(observed_indices, dtype=int)
    columns = []
    for observed_index in idx:
        basis = np.zeros(n_fft, dtype=complex)
        basis[observed_index] = 1.0
        columns.append(
            complete_spectrum(basis, idx, method, interpolation_factor).spectrum
        )
    operator = np.column_stack(columns)
    return int(np.linalg.matrix_rank(operator, tol=tolerance))
