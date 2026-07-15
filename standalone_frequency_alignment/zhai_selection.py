"""按翟靖宇学位论文描述实现严格 0-1 CRLB 逐点交换选频。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZhaiSelectionResult:
    indices: np.ndarray
    objective: float
    initial_objective: float
    sweeps: int
    accepted_swaps: int
    converged: bool


def crlb_information_objective(
    indices: np.ndarray,
    power: np.ndarray,
    frequencies: np.ndarray,
) -> float:
    """计算与时延 CRLB 成反比的角频率加权方差项。"""

    idx = np.asarray(indices, dtype=int)
    p = np.maximum(np.asarray(power, dtype=float)[idx], 0.0)
    omega = 2.0 * np.pi * np.asarray(frequencies, dtype=float)[idx]
    s0 = float(np.sum(p))
    if s0 <= np.finfo(float).tiny:
        return 0.0
    s1 = float(np.sum(p * omega))
    s2 = float(np.sum(p * omega * omega))
    return max(0.0, s2 - s1 * s1 / s0)


def crlb_value(
    indices: np.ndarray,
    power: np.ndarray,
    frequencies: np.ndarray,
    noise_variance_a: float,
    noise_variance_b: float,
    gain_abs: float = 1.0,
) -> float:
    """计算论文模型下的 CRLB；无信息时返回无穷大。"""

    information = crlb_information_objective(indices, power, frequencies)
    if information <= np.finfo(float).tiny or gain_abs <= 0.0:
        return float("inf")
    noise_term = noise_variance_a + noise_variance_b / (gain_abs * gain_abs)
    return 0.5 * noise_term / information


def select_zhai_thesis(
    power: np.ndarray,
    selected_bins: int,
    frequencies: np.ndarray | None = None,
    max_sweeps: int = 50,
    relative_tolerance: float = 1e-10,
) -> ZhaiSelectionResult:
    """幅度最大初始化，再按选中位置逐一寻找最佳交换，直至收敛。"""

    p = np.maximum(np.asarray(power, dtype=float), 0.0)
    if p.ndim != 1:
        raise ValueError("power 必须是一维数组")
    n = p.size
    if not 1 <= selected_bins < n:
        raise ValueError("selected_bins 必须满足 1 <= M < N")
    if frequencies is None:
        frequencies = np.fft.fftshift(np.fft.fftfreq(n))
    f = np.asarray(frequencies, dtype=float)
    if f.shape != p.shape:
        raise ValueError("frequencies 与 power 的形状必须一致")
    omega = 2.0 * np.pi * f

    # mergesort 保证功率并列时仍有确定性的低索引优先结果。
    initial = np.argsort(-p, kind="mergesort")[:selected_bins]
    selected = initial.astype(int).copy()
    current = crlb_information_objective(selected, p, f)
    initial_objective = current
    accepted_swaps = 0
    converged = False
    completed_sweeps = 0

    for sweep in range(1, max_sweeps + 1):
        sweep_start = current
        for position in range(selected_bins):
            selected_mask = np.zeros(n, dtype=bool)
            selected_mask[selected] = True
            candidates = np.flatnonzero(~selected_mask)
            old_index = selected[position]
            selected_power = p[selected]
            selected_frequency = omega[selected]
            s0 = float(np.sum(selected_power))
            s1 = float(np.sum(selected_power * selected_frequency))
            s2 = float(np.sum(selected_power * selected_frequency**2))
            candidate_s0 = s0 - p[old_index] + p[candidates]
            candidate_s1 = (
                s1
                - p[old_index] * omega[old_index]
                + p[candidates] * omega[candidates]
            )
            candidate_s2 = (
                s2
                - p[old_index] * omega[old_index] ** 2
                + p[candidates] * omega[candidates] ** 2
            )
            candidate_objectives = np.where(
                candidate_s0 > np.finfo(float).tiny,
                candidate_s2 - candidate_s1**2 / candidate_s0,
                0.0,
            )
            best_position = int(np.argmax(candidate_objectives))
            best_objective = float(max(candidate_objectives[best_position], 0.0))
            threshold = relative_tolerance * max(1.0, abs(current))
            if best_objective > current + threshold:
                selected[position] = int(candidates[best_position])
                current = best_objective
                accepted_swaps += 1

        completed_sweeps = sweep
        improvement = current - sweep_start
        threshold = relative_tolerance * max(1.0, abs(sweep_start))
        if improvement <= threshold:
            converged = True
            break

    return ZhaiSelectionResult(
        indices=np.sort(selected),
        objective=float(current),
        initial_objective=float(initial_objective),
        sweeps=completed_sweeps,
        accepted_swaps=accepted_swaps,
        converged=converged,
    )
