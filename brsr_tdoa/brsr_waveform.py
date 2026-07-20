"""BRSR-TDOA Stage B 的完整 BPSK-RRC 波形与候选频点工具。

正式研究对象使用线性分数时延和有限观测窗。循环时延只用于验证 FFT
相移符号与频点索引，不进入正式性能数据。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class WaveformConfig:
    n_time: int = 1024
    fs_hz: float = 40e6
    samples_per_symbol: int = 2
    rrc_rolloff: float = 0.2
    rrc_span_symbols: int = 8
    tau_min: float = -8.0
    tau_max: float = 8.0
    fractional_delay_half_len: int = 32
    candidate_count: int = 12
    base_bins: tuple[int, int] = (31, 94)
    min_candidate_spacing: int = 8

    def validate(self) -> None:
        if self.n_time < 64:
            raise ValueError("n_time must be at least 64")
        if not math.isfinite(self.fs_hz) or self.fs_hz <= 0.0:
            raise ValueError("fs_hz must be positive and finite")
        if self.samples_per_symbol < 2:
            raise ValueError("samples_per_symbol must be at least two")
        if not 0.0 < self.rrc_rolloff <= 1.0:
            raise ValueError("rrc_rolloff must lie in (0, 1]")
        if self.rrc_span_symbols < 2:
            raise ValueError("rrc_span_symbols must be at least two")
        if not self.tau_min < self.tau_max:
            raise ValueError("tau_min must be smaller than tau_max")
        if self.fractional_delay_half_len < 8:
            raise ValueError("fractional_delay_half_len must be at least eight")
        if self.candidate_count < 4:
            raise ValueError("candidate_count must be at least four")
        if len(self.base_bins) != 2 or self.base_bins[0] >= self.base_bins[1]:
            raise ValueError("base_bins must contain two increasing bins")
        if min(self.base_bins) <= 0:
            raise ValueError("Stage B uses positive, non-DC candidate bins")
        if self.min_candidate_spacing < 1:
            raise ValueError("min_candidate_spacing must be positive")
        support = rrc_positive_support_max_bin(self)
        if max(self.base_bins) > support:
            raise ValueError("base_bins lie outside the RRC theoretical support")
        ambiguity_period = self.n_time / (self.base_bins[1] - self.base_bins[0])
        if ambiguity_period <= self.tau_max - self.tau_min:
            raise ValueError("base_bins are ambiguous over the TDOA support")

    @property
    def symbol_rate_hz(self) -> float:
        return self.fs_hz / self.samples_per_symbol

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.fs_hz


@dataclass(frozen=True)
class StructuredTrial:
    """跨 SNR 复用的无噪声波形、真值和标准噪声。"""

    source_long: np.ndarray
    source_a: np.ndarray
    source_b_unit_gain: np.ndarray
    true_tau: float
    true_rho: complex
    noise_a_standard: np.ndarray
    noise_b_standard: np.ndarray


def root_raised_cosine_taps(
    rolloff: float,
    span_symbols: int,
    samples_per_symbol: int,
) -> np.ndarray:
    """返回单位能量、奇数长度的根升余弦 FIR。"""

    if not 0.0 < rolloff <= 1.0:
        raise ValueError("rolloff must lie in (0, 1]")
    if span_symbols < 2 or samples_per_symbol < 2:
        raise ValueError("span_symbols and samples_per_symbol are too small")

    half = span_symbols * samples_per_symbol // 2
    t = np.arange(-half, half + 1, dtype=float) / samples_per_symbol
    taps = np.empty_like(t)
    beta = rolloff
    for index, value in enumerate(t):
        if math.isclose(value, 0.0, abs_tol=1e-14):
            taps[index] = 1.0 + beta * (4.0 / math.pi - 1.0)
        elif math.isclose(abs(value), 1.0 / (4.0 * beta), abs_tol=1e-12):
            taps[index] = (
                beta
                / math.sqrt(2.0)
                * (
                    (1.0 + 2.0 / math.pi)
                    * math.sin(math.pi / (4.0 * beta))
                    + (1.0 - 2.0 / math.pi)
                    * math.cos(math.pi / (4.0 * beta))
                )
            )
        else:
            numerator = (
                math.sin(math.pi * value * (1.0 - beta))
                + 4.0
                * beta
                * value
                * math.cos(math.pi * value * (1.0 + beta))
            )
            denominator = math.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[index] = numerator / denominator
    energy = float(np.sum(taps**2))
    if not math.isfinite(energy) or energy <= 0.0:
        raise RuntimeError("invalid RRC filter energy")
    return taps / math.sqrt(energy)


def fractional_delay_taps(delay_samples: float, half_len: int) -> np.ndarray:
    """构造单位直流增益的窗化 sinc 分数时延 FIR。"""

    if not math.isfinite(delay_samples):
        raise ValueError("delay_samples must be finite")
    if half_len < 8:
        raise ValueError("half_len must be at least eight")
    indices = np.arange(-half_len, half_len + 1, dtype=float)
    window = np.kaiser(indices.size, beta=8.6)
    taps = np.sinc(indices - delay_samples) * window
    gain = float(np.sum(taps))
    if not math.isfinite(gain) or abs(gain) < 1e-12:
        raise RuntimeError("fractional-delay filter has invalid gain")
    return taps / gain


def apply_linear_fractional_delay(
    signal: np.ndarray,
    delay_samples: float,
    half_len: int,
) -> np.ndarray:
    """对长信号施加线性分数时延，返回同长度结果，不发生循环回绕。"""

    signal = np.asarray(signal, dtype=np.complex128)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    taps = fractional_delay_taps(delay_samples, half_len)
    return fftconvolve(signal, taps, mode="same")


def apply_circular_fractional_delay(
    signal: np.ndarray,
    delay_samples: float,
) -> np.ndarray:
    """仅供单元测试使用的精确循环分数时延。"""

    signal = np.asarray(signal, dtype=np.complex128)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    frequencies = np.fft.fftfreq(signal.size)
    spectrum = np.fft.fft(signal, norm="ortho")
    shifted = spectrum * np.exp(-2j * np.pi * frequencies * delay_samples)
    return np.fft.ifft(shifted, norm="ortho")


def rrc_positive_support_max_bin(config: WaveformConfig) -> int:
    bandwidth_hz = (
        (1.0 + config.rrc_rolloff) * config.symbol_rate_hz / 2.0
    )
    return int(math.floor(bandwidth_hz * config.n_time / config.fs_hz))


def raised_cosine_power_at_bins(
    bin_indices: np.ndarray,
    config: WaveformConfig,
) -> np.ndarray:
    """返回RRC成形BPSK的理论升余弦功率谱形状。"""

    bins = np.asarray(bin_indices, dtype=int)
    frequencies = np.abs(bins) * config.fs_hz / config.n_time
    symbol_rate = config.symbol_rate_hz
    low = (1.0 - config.rrc_rolloff) * symbol_rate / 2.0
    high = (1.0 + config.rrc_rolloff) * symbol_rate / 2.0
    power = np.zeros_like(frequencies, dtype=float)
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
    return power


def design_candidate_bins(config: WaveformConfig) -> np.ndarray:
    """按理论 ``power * frequency^2`` 累计分位点构造单边候选池。

    基础频点由无歧义周期约束预注册；其余频点用理论RRC功率谱的信息密度
    分位点覆盖有效中高频。整个过程不读取development或locked性能。
    """

    config.validate()
    support = rrc_positive_support_max_bin(config)
    eligible = np.arange(1, support + 1, dtype=int)
    power = raised_cosine_power_at_bins(eligible, config)
    omega = 2.0 * np.pi * eligible / config.n_time
    information = power * omega**2
    total = float(np.sum(information))
    if total <= 0.0:
        raise RuntimeError("RRC information density is empty")
    cumulative = np.cumsum(information) / total

    selected = [int(value) for value in config.base_bins]
    n_extra = config.candidate_count - len(selected)
    quantiles = (np.arange(n_extra, dtype=float) + 0.5) / n_extra
    for quantile in quantiles:
        target = int(eligible[np.searchsorted(cumulative, quantile, side="left")])
        valid = eligible[
            np.asarray(
                [
                    all(
                        abs(int(candidate) - existing)
                        >= config.min_candidate_spacing
                        for existing in selected
                    )
                    for candidate in eligible
                ],
                dtype=bool,
            )
        ]
        if valid.size == 0:
            raise RuntimeError("candidate-spacing constraint is infeasible")
        chosen = int(valid[np.argmin(np.abs(valid - target))])
        selected.append(chosen)

    result = np.asarray(sorted(selected), dtype=int)
    if result.size != config.candidate_count or np.unique(result).size != result.size:
        raise RuntimeError("candidate design did not produce unique bins")
    if np.any(result <= 0) or np.any(result > support):
        raise RuntimeError("candidate design left the positive RRC support")
    return result


def candidate_fft_values(signal: np.ndarray, bin_indices: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128)
    bins = np.asarray(bin_indices, dtype=int)
    if signal.ndim != 1 or bins.ndim != 1:
        raise ValueError("signal and bin_indices must be one-dimensional")
    if np.any(bins < 0) or np.any(bins >= signal.size):
        raise ValueError("candidate bins lie outside the FFT array")
    return np.fft.fft(signal, norm="ortho")[bins]


def _complex_standard_normal(
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    return (
        rng.standard_normal(size) + 1j * rng.standard_normal(size)
    ) / math.sqrt(2.0)


def generate_structured_trial(
    rng: np.random.Generator,
    config: WaveformConfig,
) -> StructuredTrial:
    """生成跨SNR复用的一条BPSK-RRC双站无噪声轨迹。"""

    config.validate()
    rrc = root_raised_cosine_taps(
        config.rrc_rolloff,
        config.rrc_span_symbols,
        config.samples_per_symbol,
    )
    rrc_half = (rrc.size - 1) // 2
    guard = (
        rrc_half
        + config.fractional_delay_half_len
        + int(math.ceil(max(abs(config.tau_min), abs(config.tau_max))))
        + 16
    )
    long_len = config.n_time + 2 * guard
    n_symbols = int(math.ceil((long_len + rrc.size) / config.samples_per_symbol))
    symbols = rng.choice((-1.0, 1.0), size=n_symbols)
    upsampled = np.zeros(n_symbols * config.samples_per_symbol, dtype=float)
    upsampled[:: config.samples_per_symbol] = symbols
    shaped = fftconvolve(upsampled, rrc, mode="same")
    # 单位能量RRC和单位能量符号使平均样本功率约为1/sps。
    shaped *= math.sqrt(config.samples_per_symbol)

    if shaped.size < long_len:
        raise RuntimeError("generated waveform is shorter than the guarded window")
    start = (shaped.size - long_len) // 2
    source_long = np.asarray(shaped[start : start + long_len], dtype=np.complex128)
    true_tau = float(rng.uniform(config.tau_min, config.tau_max))
    amplitude = float(rng.uniform(0.70, 0.90))
    phase = float(rng.uniform(-np.pi, np.pi))
    true_rho = amplitude * np.exp(1j * phase)

    delayed = apply_linear_fractional_delay(
        source_long,
        true_tau,
        config.fractional_delay_half_len,
    )
    crop = slice(guard, guard + config.n_time)
    source_a = source_long[crop].copy()
    source_b = delayed[crop].copy()
    return StructuredTrial(
        source_long=source_long,
        source_a=source_a,
        source_b_unit_gain=source_b,
        true_tau=true_tau,
        true_rho=complex(true_rho),
        noise_a_standard=_complex_standard_normal(rng, config.n_time),
        noise_b_standard=_complex_standard_normal(rng, config.n_time),
    )


def observe_trial(
    trial: StructuredTrial,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """按A站单位名义信号功率定义加入AWGN。"""

    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    sigma2 = 10.0 ** (-snr_db / 10.0)
    y_a = trial.source_a + math.sqrt(sigma2) * trial.noise_a_standard
    y_b = (
        trial.true_rho * trial.source_b_unit_gain
        + math.sqrt(sigma2) * trial.noise_b_standard
    )
    return y_a, y_b, sigma2


def estimate_candidate_statistics(
    trials: list[StructuredTrial],
    bin_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """从development clean A波形冻结候选频点功率和协方差诊断。"""

    if not trials:
        raise ValueError("at least one development trial is required")
    spectra = np.stack(
        [candidate_fft_values(trial.source_a, bin_indices) for trial in trials],
        axis=0,
    )
    centered = spectra - np.mean(spectra, axis=0, keepdims=True)
    covariance = centered.conj().T @ centered / max(centered.shape[0] - 1, 1)
    power = np.maximum(np.real(np.diag(covariance)), 1e-8)
    standard = np.sqrt(power[:, None] * power[None, :])
    coherence = np.divide(
        np.abs(covariance),
        standard,
        out=np.zeros_like(standard, dtype=float),
        where=standard > 0.0,
    )
    off_diagonal = covariance - np.diag(np.diag(covariance))
    diagnostics = {
        "offdiag_frobenius_ratio": float(
            np.linalg.norm(off_diagonal) / max(np.linalg.norm(covariance), 1e-12)
        ),
        "max_offdiag_coherence": float(
            np.max(coherence - np.eye(coherence.shape[0]))
        ),
        "min_candidate_power": float(np.min(power)),
        "max_candidate_power": float(np.max(power)),
    }
    return power, covariance, diagnostics

