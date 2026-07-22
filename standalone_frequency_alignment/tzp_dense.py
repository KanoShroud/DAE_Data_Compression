"""时域补零密集频谱（TZP-DS）的独立数值算子。"""

from __future__ import annotations

import time

import numpy as np
from scipy import signal


def time_zero_padded_dense_spectrum(
    sparse_shifted_spectrum: np.ndarray,
    factor: int,
) -> np.ndarray:
    """IFFT 后在时域末尾补零，再返回幅度校正后的 shifted 密集频谱。

    输入最后一维长度为 ``N``，输出最后一维长度为 ``factor * N``。
    使用正交归一化 FFT 时乘以 ``sqrt(factor)``，使原始频点位置
    ``q = factor * k`` 与输入 shifted DFT 频点严格一致。
    """

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
    padded_shape = (*time_signal.shape[:-1], dense_length)
    padded_time = np.zeros(padded_shape, dtype=np.complex128)
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
    """从 shifted 密集频谱取回原始 shifted DFT 网格。"""

    dense = np.asarray(dense_shifted_spectrum, dtype=np.complex128)
    if original_length < 2 or factor < 1:
        raise ValueError("原始长度必须至少为 2，补零倍数必须为正整数")
    if dense.shape[-1] != original_length * factor:
        raise ValueError("密集频谱长度与 original_length * factor 不一致")
    return dense[..., factor * np.arange(original_length)]


def dense_delay_estimates(
    shifted_cross_spectra: np.ndarray,
    valid_input_counts: np.ndarray,
    lag_min_samples: float,
    lag_max_samples: float,
    lag_step_samples: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """在连续时延网格上用 ZoomFFT 计算 TZP-DS 互谱得分。

    shifted 频率索引 ``r`` 对应物理谐波 ``r-P/2``。二者的 Fourier
    和只相差模长为 1 的全局相位，因此对峰值搜索所用的绝对值得分完全
    等价。ZoomFFT 避免显式构造 ``时延点数 × 密集频点数`` 的导向矩阵。
    """

    started = time.perf_counter()
    spectra = np.asarray(shifted_cross_spectra, dtype=np.complex128)
    if spectra.ndim == 1:
        spectra = spectra[None, :]
    if spectra.ndim != 2:
        raise ValueError("互谱必须是一维或二维批量数组")
    counts = np.asarray(valid_input_counts, dtype=int)
    if counts.shape != (spectra.shape[0],):
        raise ValueError("valid_input_counts 与互谱批量维度不一致")
    if lag_step_samples <= 0.0 or lag_max_samples < lag_min_samples:
        raise ValueError("时延搜索范围或步长无效")

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
                scores.shape[1],
                peak_index + exclusion + 1,
            )
        ] = False
        sidelobe = float(np.max(scores[row_index, keep])) if np.any(keep) else 0.0
        sidelobe_ratios[row_index] = sidelobe / max(
            peak_value,
            np.finfo(float).tiny,
        )
    estimates[~valid] = np.nan
    elapsed_ms_per_row = (
        (time.perf_counter() - started) * 1000.0 / max(spectra.shape[0], 1)
    )
    return estimates, sidelobe_ratios, elapsed_ms_per_row


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
    lag_ticks = np.arange(start_tick, stop_tick + 1, dtype=int)
    lags = lag_ticks.astype(float) * lag_step_samples
    dense_length = spectra.shape[-1]
    zoom = signal.ZoomFFT(
        dense_length,
        [-lags[0] / dense_length, -lags[-1] / dense_length],
        m=lags.size,
        fs=1.0,
        endpoint=True,
    )
    return lags, np.abs(zoom(spectra, axis=-1))


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
