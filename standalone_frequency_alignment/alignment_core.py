"""频谱嵌入、周期插值、连续时延估计和诊断指标。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time

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


# 时域补零密集频谱（TZP-DS）属于插值实验的通用线性算子，故与原网格补全
# 和连续时延估计统一维护在本核心模块；旧 tzp_dense.py 仅保留兼容性导出。
def time_zero_padded_dense_spectrum(
    sparse_shifted_spectrum: np.ndarray,
    factor: int,
) -> np.ndarray:
    """IFFT 后在时域末尾补零，返回幅度校正后的 shifted 密集频谱。"""

    spectrum = np.asarray(sparse_shifted_spectrum, dtype=np.complex128)
    if spectrum.ndim < 1 or spectrum.shape[-1] < 2:
        raise ValueError("输入频谱最后一维长度必须至少为 2")
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("时域补零倍数必须为正整数")
    n_fft = spectrum.shape[-1]
    dense_length = n_fft * int(factor)
    time_signal = np.fft.ifft(
        np.fft.ifftshift(spectrum, axes=-1),
        axis=-1,
        norm="ortho",
    )
    padded_time = np.zeros(
        (*time_signal.shape[:-1], dense_length),
        dtype=np.complex128,
    )
    padded_time[..., :n_fft] = time_signal
    dense_unshifted = np.sqrt(float(factor)) * np.fft.fft(
        padded_time,
        axis=-1,
        norm="ortho",
    )
    return np.fft.fftshift(dense_unshifted, axes=-1)


def sample_original_grid(
    dense_shifted_spectrum: np.ndarray,
    original_length: int,
    factor: int,
) -> np.ndarray:
    """从 shifted 密集频谱抽取回原始 shifted DFT 网格。"""

    dense = np.asarray(dense_shifted_spectrum, dtype=np.complex128)
    if original_length < 2 or factor < 1:
        raise ValueError("原始长度必须至少为 2，补零倍数必须为正整数")
    if dense.shape[-1] != original_length * factor:
        raise ValueError("密集频谱长度与 original_length * factor 不一致")
    return dense[..., factor * np.arange(original_length)]


def dense_delay_score_grid(
    shifted_cross_spectra: np.ndarray,
    lag_min_samples: float,
    lag_max_samples: float,
    lag_step_samples: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 TZP-DS 在等间隔连续时延网格上的绝对值得分。"""

    spectra = np.asarray(shifted_cross_spectra, dtype=np.complex128)
    if spectra.ndim == 1:
        spectra = spectra[None, :]
    if spectra.ndim != 2:
        raise ValueError("互谱必须是一维或二维批量数组")
    if lag_step_samples <= 0.0 or lag_max_samples < lag_min_samples:
        raise ValueError("时延搜索范围或步长无效")
    start_tick = int(round(lag_min_samples / lag_step_samples))
    stop_tick = int(round(lag_max_samples / lag_step_samples))
    lags = np.arange(start_tick, stop_tick + 1, dtype=int).astype(float) * lag_step_samples
    dense_length = spectra.shape[-1]
    zoom = signal.ZoomFFT(
        dense_length,
        [-lags[0] / dense_length, -lags[-1] / dense_length],
        m=lags.size,
        fs=1.0,
        endpoint=True,
    )
    return lags, np.abs(zoom(spectra, axis=-1))


def dense_delay_estimates(
    shifted_cross_spectra: np.ndarray,
    valid_input_counts: np.ndarray,
    lag_min_samples: float,
    lag_max_samples: float,
    lag_step_samples: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """在连续时延网格上用 ZoomFFT 计算 TZP-DS 互谱得分。"""

    started = time.perf_counter()
    spectra = np.asarray(shifted_cross_spectra, dtype=np.complex128)
    if spectra.ndim == 1:
        spectra = spectra[None, :]
    if spectra.ndim != 2:
        raise ValueError("互谱必须是一维或二维批量数组")
    counts = np.asarray(valid_input_counts, dtype=int)
    if counts.shape != (spectra.shape[0],):
        raise ValueError("valid_input_counts 与互谱批量维度不一致")

    lags, scores = dense_delay_score_grid(
        spectra,
        lag_min_samples,
        lag_max_samples,
        lag_step_samples,
    )
    peak_indices = np.argmax(scores, axis=1)
    estimates = lags[peak_indices]
    sidelobe_ratios = np.full(spectra.shape[0], np.nan, dtype=float)
    exclusion = max(1, int(round(1.0 / lag_step_samples)))
    norms = np.linalg.norm(spectra, axis=1)
    valid = (counts >= 1) & (norms > np.finfo(float).tiny)
    for row_index in np.flatnonzero(valid):
        peak_index = int(peak_indices[row_index])
        peak_value = float(scores[row_index, peak_index])
        keep = np.ones(scores.shape[1], dtype=bool)
        keep[
            max(0, peak_index - exclusion) : min(
                scores.shape[1], peak_index + exclusion + 1
            )
        ] = False
        sidelobe = float(np.max(scores[row_index, keep])) if np.any(keep) else 0.0
        sidelobe_ratios[row_index] = sidelobe / max(peak_value, np.finfo(float).tiny)
    estimates[~valid] = np.nan
    elapsed_ms_per_row = (
        (time.perf_counter() - started) * 1000.0 / max(spectra.shape[0], 1)
    )
    return estimates, sidelobe_ratios, elapsed_ms_per_row


def direct_dense_delay_estimate(
    shifted_cross_spectrum: np.ndarray,
    lag_min_samples: float,
    lag_max_samples: float,
    lag_step_samples: float,
    chunk_size: int = 256,
) -> float:
    """按定义分块求和，作为 TZP-DS ZoomFFT 估计器的一致性参考。"""

    cross = np.asarray(shifted_cross_spectrum, dtype=np.complex128)
    if cross.ndim != 1:
        raise ValueError("直接求和参考只接受一维互谱")
    if chunk_size < 1:
        raise ValueError("chunk_size 必须为正整数")
    start_tick = int(round(lag_min_samples / lag_step_samples))
    stop_tick = int(round(lag_max_samples / lag_step_samples))
    lags = np.arange(start_tick, stop_tick + 1, dtype=float) * lag_step_samples
    frequencies = np.fft.fftshift(np.fft.fftfreq(cross.size))
    best_score = -np.inf
    best_lag = float("nan")
    for start in range(0, lags.size, chunk_size):
        block = lags[start : start + chunk_size]
        steering = np.exp(2j * np.pi * np.outer(block, frequencies))
        scores = np.abs(steering @ cross)
        local = int(np.argmax(scores))
        score = float(scores[local])
        if score > best_score:
            best_score = score
            best_lag = float(block[local])
    return best_lag
