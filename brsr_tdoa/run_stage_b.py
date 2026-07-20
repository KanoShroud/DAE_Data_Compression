"""BRSR-TDOA 阶段 B：结构化 BPSK-RRC 波形上的开发门禁。

在 PyCharm 中直接运行本文件。顶部 ``PYCHARM_RUN_MODE`` 默认为
``development``；Codex 轻量验证使用 ``--mode smoke``。本脚本只使用
``brsr_tdoa`` 独立目录，不导入或修改主项目的训练、评估和绘图链路。

阶段 B 只运行 development。候选频点、工作先验、拉格朗日乘子和静态
调度均不得读取未来 locked 集合；只有 development 门禁通过后，才能另行
批准和生成全新的 locked 实验。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brsr_tdoa.brsr_core import NestedComplexQuantizer  # noqa: E402
from brsr_tdoa.brsr_stage_b import (  # noqa: E402
    StageBObservation,
    StageBProtocol,
    build_stage_b_model,
    enumerate_static_configurations,
    evaluate_fixed,
    make_observation,
    result_row,
    run_policy,
)
from brsr_tdoa.brsr_waveform import (  # noqa: E402
    StructuredTrial,
    WaveformConfig,
    apply_circular_fractional_delay,
    apply_linear_fractional_delay,
    candidate_fft_values,
    design_candidate_bins,
    estimate_candidate_statistics,
    generate_structured_trial,
    observe_trial,
    raised_cosine_power_at_bins,
    root_raised_cosine_taps,
    rrc_positive_support_max_bin,
)


PYCHARM_RUN_MODE = "development"


@dataclass(frozen=True)
class StageBConfig:
    mode: str
    snr_db: tuple[float, ...]
    prior_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    prior_trials_per_seed: int
    calibration_trials_per_seed: int
    evaluation_trials_per_seed: int
    tau_step: float
    rho_amplitudes: tuple[float, ...]
    rho_phase_count: int
    rollout_samples: int
    rollout_convergence_samples: tuple[int, ...]
    rollout_first_actions: int
    lambda_iterations: int
    lambda_max: float
    lambda_bit_tolerance: float
    static_batch_size: int
    bootstrap_repetitions: int
    progress_every: int


CONFIGS = {
    "smoke": StageBConfig(
        mode="smoke",
        snr_db=(-5.0, 10.0),
        prior_seeds=(20269001,),
        calibration_seeds=(20269101,),
        evaluation_seeds=(20269201,),
        prior_trials_per_seed=4,
        calibration_trials_per_seed=2,
        evaluation_trials_per_seed=2,
        tau_step=1.0,
        rho_amplitudes=(0.70, 0.90),
        rho_phase_count=4,
        rollout_samples=2,
        rollout_convergence_samples=(1, 2, 4),
        rollout_first_actions=2,
        lambda_iterations=2,
        lambda_max=10.0,
        lambda_bit_tolerance=2.0,
        static_batch_size=32,
        bootstrap_repetitions=50,
        progress_every=1,
    ),
    "development": StageBConfig(
        mode="development",
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
        prior_seeds=(20269001, 20269002, 20269003),
        calibration_seeds=(20269101, 20269102),
        evaluation_seeds=(20269201, 20269202, 20269203),
        prior_trials_per_seed=64,
        calibration_trials_per_seed=6,
        evaluation_trials_per_seed=20,
        tau_step=0.25,
        rho_amplitudes=(0.70, 0.80, 0.90),
        rho_phase_count=8,
        rollout_samples=16,
        rollout_convergence_samples=(16, 32, 64, 128),
        rollout_first_actions=3,
        lambda_iterations=6,
        lambda_max=10.0,
        lambda_bit_tolerance=0.75,
        static_batch_size=64,
        bootstrap_repetitions=1000,
        progress_every=10,
    ),
}


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _validate_config(config: StageBConfig) -> None:
    if config.mode not in CONFIGS:
        raise ValueError(f"unsupported Stage B mode: {config.mode}")
    snr = np.asarray(config.snr_db, dtype=float)
    if (
        snr.size == 0
        or not np.all(np.isfinite(snr))
        or np.unique(snr).size != snr.size
    ):
        raise ValueError("snr_db must contain unique finite values")
    pools = (
        config.prior_seeds,
        config.calibration_seeds,
        config.evaluation_seeds,
    )
    if any(not values for values in pools):
        raise ValueError("all development seed pools must be non-empty")
    flattened = [value for values in pools for value in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("prior, calibration and evaluation seeds must be disjoint")
    if min(
        config.prior_trials_per_seed,
        config.calibration_trials_per_seed,
        config.evaluation_trials_per_seed,
    ) <= 0:
        raise ValueError("trial counts must be positive")
    if config.tau_step <= 0.0 or not math.isfinite(config.tau_step):
        raise ValueError("tau_step must be positive and finite")
    if (
        not config.rho_amplitudes
        or min(config.rho_amplitudes) <= 0.0
        or config.rho_phase_count < 4
    ):
        raise ValueError("rho grid is invalid")
    if (
        config.rollout_samples < 1
        or not config.rollout_convergence_samples
        or min(config.rollout_convergence_samples) < 1
        or config.rollout_first_actions < 1
        or config.lambda_iterations < 1
    ):
        raise ValueError("policy-search settings must be positive")
    if len(set(config.rollout_convergence_samples)) != len(
        config.rollout_convergence_samples
    ):
        raise ValueError("rollout convergence sample counts must be unique")
    if config.lambda_max <= 0.0 or config.lambda_bit_tolerance <= 0.0:
        raise ValueError("lambda calibration bounds must be positive")
    if config.static_batch_size < 1 or config.bootstrap_repetitions < 1:
        raise ValueError("batch and bootstrap counts must be positive")
    if config.progress_every < 1:
        raise ValueError("progress_every must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedded_chain_checks(
    waveform: WaveformConfig,
    bins: np.ndarray,
    quantizer: NestedComplexQuantizer,
) -> dict[str, object]:
    """用极小确定性样本检查正式结构化链路的符号和协议不变量。"""

    rrc = root_raised_cosine_taps(
        waveform.rrc_rolloff,
        waveform.rrc_span_symbols,
        waveform.samples_per_symbol,
    )
    rrc_energy_error = abs(float(np.sum(rrc**2)) - 1.0)

    rng = np.random.default_rng(2026071801)
    circular_source = (
        rng.standard_normal(waveform.n_time)
        + 1j * rng.standard_normal(waveform.n_time)
    )
    circular_delay = 2.375
    circular_shifted = apply_circular_fractional_delay(
        circular_source,
        circular_delay,
    )
    frequencies = np.fft.fftfreq(waveform.n_time)
    circular_expected = np.fft.fft(circular_source, norm="ortho") * np.exp(
        -2j * np.pi * frequencies * circular_delay
    )
    circular_error = float(
        np.max(
            np.abs(
                np.fft.fft(circular_shifted, norm="ortho")
                - circular_expected
            )
        )
    )

    tone_bin = int(bins[len(bins) // 2])
    indices = np.arange(4096)
    tone = np.exp(2j * np.pi * tone_bin * indices / waveform.n_time)
    linear_shifted = apply_linear_fractional_delay(
        tone,
        circular_delay,
        waveform.fractional_delay_half_len,
    )
    central = slice(256, -256)
    observed_ratio = np.mean(linear_shifted[central] / tone[central])
    expected_ratio = np.exp(
        -2j * np.pi * tone_bin * circular_delay / waveform.n_time
    )
    linear_phase_error = float(np.angle(observed_ratio / expected_ratio))
    linear_gain_error = float(abs(abs(observed_ratio) - 1.0))

    trial = generate_structured_trial(np.random.default_rng(2026071802), waveform)
    integer_delay = 3.0
    delayed_long = apply_linear_fractional_delay(
        trial.source_long,
        integer_delay,
        waveform.fractional_delay_half_len,
    )
    guard = (trial.source_long.size - waveform.n_time) // 2
    source = trial.source_long[guard : guard + waveform.n_time]
    delayed = delayed_long[guard : guard + waveform.n_time]
    correlation = np.correlate(delayed, source, mode="full")
    lags = np.arange(-waveform.n_time + 1, waveform.n_time)
    integer_estimate = int(lags[np.argmax(np.abs(correlation))])

    values = np.asarray([-3.1 - 0.2j, 0.3 + 1.7j, 3.8 - 4.2j])
    fine_real, fine_imag = quantizer.encode_fine(values)
    coarse_real = quantizer.coarsen(fine_real, 4)
    coarse_imag = quantizer.coarsen(fine_imag, 4)
    nested_ok = all(
        int(fine_code) in quantizer.child_codes(int(coarse_code), 4, 8)
        for fine_code, coarse_code in zip(
            np.r_[fine_real, fine_imag],
            np.r_[coarse_real, coarse_imag],
            strict=True,
        )
    )

    support = rrc_positive_support_max_bin(waveform)
    checks = {
        "status": "PASS",
        "rrc_energy_error": rrc_energy_error,
        "circular_fft_shift_max_error": circular_error,
        "linear_tone_phase_error_rad": linear_phase_error,
        "linear_tone_gain_error": linear_gain_error,
        "integer_delay_true_samples": integer_delay,
        "integer_delay_estimate_samples": integer_estimate,
        "nested_quantization_check": nested_ok,
        "candidate_bins_in_theoretical_band": bool(
            np.all((bins > 0) & (bins <= support))
        ),
        "candidate_bins_unique": bool(np.unique(bins).size == bins.size),
    }
    if (
        rrc_energy_error > 1e-12
        or circular_error > 2e-12
        or abs(linear_phase_error) > 2e-3
        or linear_gain_error > 2e-3
        or integer_estimate != int(integer_delay)
        or not nested_ok
        or not checks["candidate_bins_in_theoretical_band"]
        or not checks["candidate_bins_unique"]
    ):
        checks["status"] = "FAIL"
        raise RuntimeError(f"embedded chain checks failed: {checks}")
    return checks


def _stable_rng(*parts: object) -> np.random.Generator:
    text = "|".join(str(value) for value in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def _generate_trials(
    seeds: Iterable[int],
    trials_per_seed: int,
    waveform: WaveformConfig,
) -> list[tuple[int, int, StructuredTrial]]:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for trial_index in range(trials_per_seed):
            rows.append(
                (
                    seed,
                    trial_index,
                    generate_structured_trial(rng, waveform),
                )
            )
    return rows


def _make_observations(
    *,
    stage: str,
    trials: list[tuple[int, int, StructuredTrial]],
    snr_values: tuple[float, ...],
    bins: np.ndarray,
    model,
    quantizer: NestedComplexQuantizer,
) -> list[StageBObservation]:
    observations = []
    for snr_db in snr_values:
        for seed, trial_index, trial in trials:
            y_a, y_b, sigma2 = observe_trial(trial, snr_db)
            observations.append(
                make_observation(
                    stage=stage,
                    seed=seed,
                    trial=trial_index,
                    snr_db=snr_db,
                    true_tau=trial.true_tau,
                    true_rho=trial.true_rho,
                    y_a_selected=candidate_fft_values(y_a, bins),
                    y_b_selected=candidate_fft_values(y_b, bins),
                    sigma2=sigma2,
                    model=model,
                    quantizer=quantizer,
                )
            )
    return observations


def _policy_rng(
    observation: StageBObservation,
    method: str,
    lambda_per_bit: float,
) -> np.random.Generator:
    return _stable_rng(
        "stage-b-policy",
        observation.seed,
        observation.trial,
        observation.snr_db,
        method,
        f"{lambda_per_bit:.17g}",
    )


def _run_adaptive_rows(
    observations: list[StageBObservation],
    *,
    method: str,
    label: str,
    lambda_per_bit: float,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    model,
    progress_every: int,
) -> list[dict[str, object]]:
    rows = []
    start = time.perf_counter()
    for index, observation in enumerate(observations, start=1):
        result = run_policy(
            observation,
            method=method,
            lambda_per_bit=lambda_per_bit,
            protocol=protocol,
            quantizer=quantizer,
            model=model,
            rng=_policy_rng(observation, method, lambda_per_bit),
        )
        result = replace(result, method=label)
        rows.append(
            result_row(
                observation,
                result,
                lambda_per_bit=lambda_per_bit,
            )
        )
        if index % progress_every == 0 or index == len(observations):
            elapsed = time.perf_counter() - start
            rate = elapsed / index
            eta = rate * (len(observations) - index)
            print(
                f"[Policy] {label}: {index}/{len(observations)} | "
                f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s",
                flush=True,
            )
    return rows


def _rollout_convergence_diagnostic(
    observations: list[StageBObservation],
    *,
    lambda_per_bit: float,
    config: StageBConfig,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    model,
) -> pd.DataFrame:
    """在每个 SNR 的一个预注册样本上检查嵌套蒙特卡洛样本数敏感性。"""

    selected = []
    for snr_db in config.snr_db:
        candidates = [
            item for item in observations if math.isclose(item.snr_db, snr_db)
        ]
        if not candidates:
            raise RuntimeError(f"missing convergence observation at SNR={snr_db}")
        selected.append(
            min(candidates, key=lambda item: (item.seed, item.trial))
        )

    rows = []
    for sample_count in config.rollout_convergence_samples:
        diagnostic_protocol = replace(
            protocol,
            rollout_samples=sample_count,
        )
        for observation in selected:
            start = time.perf_counter()
            result = run_policy(
                observation,
                method="rollout",
                lambda_per_bit=lambda_per_bit,
                protocol=diagnostic_protocol,
                quantizer=quantizer,
                model=model,
                rng=_stable_rng(
                    "rollout-convergence",
                    observation.seed,
                    observation.trial,
                    observation.snr_db,
                    sample_count,
                ),
            )
            first_action = (
                result.action_trace[0]["action"]
                if result.action_trace
                else "none"
            )
            rows.append(
                {
                    "seed": observation.seed,
                    "trial": observation.trial,
                    "snr_db": observation.snr_db,
                    "rollout_samples": sample_count,
                    "estimate_tau": result.estimate,
                    "squared_error_samples2": (
                        result.estimate - observation.true_tau
                    )
                    ** 2,
                    "total_bits": result.total_bits,
                    "first_action": first_action,
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
            print(
                f"[Rollout-Q] Q={sample_count} SNR={observation.snr_db:g} "
                f"bits={result.total_bits} first={first_action}",
                flush=True,
            )
    return pd.DataFrame(rows)


def _mean_bits_for_lambda(
    observations: list[StageBObservation],
    *,
    method: str,
    lambda_per_bit: float,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    model,
) -> float:
    values = []
    for observation in observations:
        result = run_policy(
            observation,
            method=method,
            lambda_per_bit=lambda_per_bit,
            protocol=protocol,
            quantizer=quantizer,
            model=model,
            rng=_policy_rng(observation, method, lambda_per_bit),
        )
        values.append(result.total_bits)
    return float(np.mean(values))


def _calibrate_lambda(
    observations: list[StageBObservation],
    *,
    method: str,
    target_bits: int,
    config: StageBConfig,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    model,
    cache: dict[float, float],
) -> tuple[float, list[dict[str, object]]]:
    def evaluate(value: float) -> float:
        key = float(f"{value:.17g}")
        if key not in cache:
            cache[key] = _mean_bits_for_lambda(
                observations,
                method=method,
                lambda_per_bit=key,
                protocol=protocol,
                quantizer=quantizer,
                model=model,
            )
            print(
                f"[Lambda] {method} target={target_bits} "
                f"lambda={key:.6g} mean_bits={cache[key]:.3f}",
                flush=True,
            )
        return cache[key]

    zero_bits = evaluate(0.0)
    high = config.lambda_max
    high_bits = evaluate(high)
    if zero_bits < high_bits - 1e-9:
        raise RuntimeError("mean communication is not decreasing with lambda bracket")
    if target_bits > zero_bits + config.lambda_bit_tolerance:
        raise RuntimeError("target bits exceed the zero-penalty policy budget")
    if target_bits < high_bits - config.lambda_bit_tolerance:
        raise RuntimeError("lambda_max is too small to reach the target budget")

    # 先按十进制尺度向下寻找跃迁区间，避免在未知量纲上直接线性二分。
    positive = high
    for _ in range(12):
        candidate = positive / 10.0
        candidate_bits = evaluate(candidate)
        if candidate_bits >= target_bits:
            break
        positive = candidate
    else:
        candidate = 0.0
        candidate_bits = zero_bits

    low = candidate
    high = positive
    if candidate_bits < target_bits:
        low = 0.0

    for _ in range(config.lambda_iterations):
        mid = math.sqrt(low * high) if low > 0.0 else 0.5 * high
        mid_bits = evaluate(mid)
        if abs(mid_bits - target_bits) <= config.lambda_bit_tolerance:
            break
        if mid_bits > target_bits:
            low = mid
        else:
            high = mid

    chosen = min(
        cache,
        key=lambda value: (
            abs(cache[value] - target_bits),
            value,
        ),
    )
    rows = [
        {
            "method": method,
            "target_bits": target_bits,
            "lambda_per_bit": value,
            "calibration_mean_bits": bits,
            "absolute_bit_error": abs(bits - target_bits),
            "selected": math.isclose(value, chosen, rel_tol=0.0, abs_tol=1e-15),
        }
        for value, bits in sorted(cache.items())
    ]
    return float(chosen), rows


def _levels_to_features(levels: tuple[int, ...]) -> np.ndarray:
    features = np.zeros(2 * len(levels), dtype=float)
    for position, bits in enumerate(levels):
        if bits == 4:
            features[2 * position] = 1.0
        elif bits == 8:
            features[2 * position + 1] = 1.0
        elif bits != 0:
            raise ValueError("static levels must be 0, 4, or 8")
    return features


def _static_log_likelihood_features(
    observations: list[StageBObservation],
    *,
    quantizer: NestedComplexQuantizer,
    model,
) -> np.ndarray:
    features = np.empty(
        (
            len(observations),
            2 * model.n_bins,
            model.n_tau * model.n_rho,
        ),
        dtype=np.float64,
    )
    for obs_index, observation in enumerate(observations):
        for position in range(model.n_bins):
            for level_index, bits in enumerate((4, 8)):
                real_code = int(
                    quantizer.coarsen(
                        observation.fine_real_codes[position],
                        bits,
                    )
                )
                imag_code = int(
                    quantizer.coarsen(
                        observation.fine_imag_codes[position],
                        bits,
                    )
                )
                features[obs_index, 2 * position + level_index] = (
                    quantizer.log_probability(
                        observation.context.mean_by_bin[position],
                        observation.context.variance_by_bin[position],
                        real_code,
                        imag_code,
                        bits,
                    )
                )
    return features


def _configuration_mse(
    *,
    configurations: list[tuple[int, ...]],
    observations: list[StageBObservation],
    likelihood_features: np.ndarray,
    model,
    batch_size: int,
    progress_label: str,
) -> np.ndarray:
    feature_matrix = np.stack(
        [_levels_to_features(levels) for levels in configurations],
        axis=0,
    )
    true_tau = np.asarray([item.true_tau for item in observations], dtype=float)
    log_prior = np.log(np.maximum(model.joint_prior, 1e-300))
    mse = np.empty((len(observations), len(configurations)), dtype=float)
    start = time.perf_counter()

    for begin in range(0, len(configurations), batch_size):
        end = min(begin + batch_size, len(configurations))
        batch_features = feature_matrix[begin:end]
        log_values = (
            log_prior[None, None, :]
            + np.einsum(
                "bf,ofh->obh",
                batch_features,
                likelihood_features,
                optimize=True,
            )
        )
        maxima = np.max(log_values, axis=2, keepdims=True)
        posterior = np.exp(log_values - maxima)
        posterior /= np.sum(posterior, axis=2, keepdims=True)
        tau_marginal = posterior.reshape(
            len(observations),
            end - begin,
            model.n_tau,
            model.n_rho,
        ).sum(axis=3)
        estimates = np.einsum(
            "obt,t->ob",
            tau_marginal,
            model.tau_values,
            optimize=True,
        )
        mse[:, begin:end] = (estimates - true_tau[:, None]) ** 2
        if end == len(configurations) or (begin // batch_size) % 10 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[Static] {progress_label}: {end}/{len(configurations)} "
                f"configs | elapsed={elapsed:.1f}s",
                flush=True,
            )
    return mse


def _select_static_plans(
    observations: list[StageBObservation],
    *,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
    model,
    config: StageBConfig,
) -> tuple[dict[int, tuple[int, ...]], dict[tuple[float, int], tuple[int, ...]], pd.DataFrame]:
    configurations = enumerate_static_configurations(
        model=model,
        max_payload_bits=max(protocol.target_mean_bits),
    )
    if not configurations:
        raise RuntimeError("no identifiable static configurations are available")
    likelihood_features = _static_log_likelihood_features(
        observations,
        quantizer=quantizer,
        model=model,
    )
    mse = _configuration_mse(
        configurations=configurations,
        observations=observations,
        likelihood_features=likelihood_features,
        model=model,
        batch_size=config.static_batch_size,
        progress_label="development selection",
    )
    payload = np.asarray([2 * sum(levels) for levels in configurations], dtype=int)
    snr = np.asarray([item.snr_db for item in observations], dtype=float)
    pooled_plans = {}
    snr_plans = {}
    plan_rows = []

    for target in protocol.target_mean_bits:
        eligible = np.flatnonzero(payload <= target)
        if eligible.size == 0:
            raise RuntimeError(f"no static plan fits target {target}")
        pooled_index = int(eligible[np.argmin(np.mean(mse[:, eligible], axis=0))])
        pooled_plans[target] = configurations[pooled_index]
        plan_rows.append(
            {
                "method": "Best-Static",
                "snr_db": np.nan,
                "target_bits": target,
                "payload_bits": int(payload[pooled_index]),
                "selection_rmse_samples": math.sqrt(
                    float(np.mean(mse[:, pooled_index]))
                ),
                "levels": " ".join(str(value) for value in configurations[pooled_index]),
            }
        )
        for snr_db in config.snr_db:
            mask = np.isclose(snr, snr_db)
            index = int(
                eligible[np.argmin(np.mean(mse[mask][:, eligible], axis=0))]
            )
            snr_plans[(snr_db, target)] = configurations[index]
            plan_rows.append(
                {
                    "method": "SNR-only",
                    "snr_db": snr_db,
                    "target_bits": target,
                    "payload_bits": int(payload[index]),
                    "selection_rmse_samples": math.sqrt(
                        float(np.mean(mse[mask, index]))
                    ),
                    "levels": " ".join(str(value) for value in configurations[index]),
                }
            )
    return pooled_plans, snr_plans, pd.DataFrame(plan_rows)


def _run_fixed_rows(
    observations: list[StageBObservation],
    *,
    method: str,
    level_for_observation,
    quantizer: NestedComplexQuantizer,
    model,
) -> list[dict[str, object]]:
    rows = []
    for observation in observations:
        levels = tuple(level_for_observation(observation))
        result = evaluate_fixed(
            observation,
            method=method,
            levels=levels,
            quantizer=quantizer,
            model=model,
        )
        rows.append(
            result_row(
                observation,
                result,
                lambda_per_bit=None,
            )
        )
    return rows


def _save_partial_rows(
    rows: list[dict[str, object]],
    output: Path,
) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(
        output / "stage_b_trials.partial.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _metric_row(group: pd.DataFrame) -> pd.Series:
    squared = group["squared_error_samples2"].to_numpy(dtype=float)
    absolute = group["abs_error_samples"].to_numpy(dtype=float)
    posterior_risk = group["posterior_risk_samples2"].to_numpy(dtype=float)
    empirical_mse = float(np.mean(squared))
    mean_risk = float(np.mean(posterior_risk))
    return pd.Series(
        {
            "n": len(group),
            "rmse_samples": math.sqrt(empirical_mse),
            "mae_samples": float(np.mean(absolute)),
            "p90_abs_error_samples": float(np.quantile(absolute, 0.90)),
            "severe_error_rate_gt2": float(np.mean(absolute > 2.0)),
            "mean_bits": float(group["total_bits"].mean()),
            "max_bits": int(group["total_bits"].max()),
            "mean_posterior_risk": mean_risk,
            "risk_to_mse_ratio": mean_risk / max(empirical_mse, 1e-12),
            "credible90_coverage": float(
                group["credible90_contains_true"].mean()
            ),
            "mean_clipping_rate": float(
                group["quantizer_clipping_rate"].mean()
            ),
        }
    )


def _summaries(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_snr = (
        rows.groupby(["method", "snr_db"], sort=False, observed=True)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
    overall = (
        rows.groupby("method", sort=False, observed=True)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
    return by_snr, overall


def _cluster_bootstrap_difference(
    rows: pd.DataFrame,
    method: str,
    baseline: str,
    repetitions: int,
) -> dict[str, float]:
    subset = rows[rows["method"].isin((method, baseline))].copy()
    pivot = subset.pivot_table(
        index=["seed", "trial", "snr_db"],
        columns="method",
        values="squared_error_samples2",
        aggfunc="first",
    ).dropna()
    if pivot.empty:
        raise RuntimeError("paired bootstrap has no aligned trajectories")
    trajectories = (
        pivot.reset_index()
        .groupby(["seed", "trial"], observed=True)
        .apply(
            lambda frame: float(
                np.mean(frame[baseline].to_numpy() - frame[method].to_numpy())
            ),
            include_groups=False,
        )
        .to_numpy()
    )
    rng = np.random.default_rng(20269999)
    sampled = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        draw = rng.integers(0, trajectories.size, size=trajectories.size)
        sampled[index] = float(np.mean(trajectories[draw]))
    return {
        "paired_mse_gain_samples2": float(np.mean(trajectories)),
        "paired_mse_gain_ci95_low": float(np.quantile(sampled, 0.025)),
        "paired_mse_gain_ci95_high": float(np.quantile(sampled, 0.975)),
        "n_trajectory_clusters": int(trajectories.size),
    }


def _validate_rows(
    rows: pd.DataFrame,
    *,
    waveform: WaveformConfig,
    protocol: StageBProtocol,
    candidate_bins: np.ndarray,
) -> dict[str, object]:
    required = {
        "seed",
        "trial",
        "snr_db",
        "method",
        "true_tau",
        "estimate_tau",
        "total_bits",
        "squared_error_samples2",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise RuntimeError(f"trial table is missing columns: {missing}")
    if rows.empty or rows[list(required - {"method"})].isna().any().any():
        raise RuntimeError("trial table is empty or contains non-finite required values")
    if np.any(np.abs(rows["true_tau"]) > waveform.tau_max + 1e-12):
        raise RuntimeError("true TDOA left the registered support")
    if np.any(rows["total_bits"] < 0):
        raise RuntimeError("communication budget became negative")
    adaptive = rows[rows["method"].str.startswith("BRSR-")]
    hard_max = (
        len(waveform.base_bins) * 2 * protocol.coarse_bits
        + protocol.max_decisions
        * (
            2 * (protocol.max_bits - protocol.coarse_bits)
            + protocol.feedback_bits_for(candidate_bins.size)
        )
    )
    if not adaptive.empty and int(adaptive["total_bits"].max()) > hard_max:
        raise RuntimeError("adaptive policy exceeds the registered hard bit cap")
    if np.any(candidate_bins > rrc_positive_support_max_bin(waveform)):
        raise RuntimeError("candidate bins left the RRC support")
    duplicated = rows.duplicated(["seed", "trial", "snr_db", "method"])
    if bool(duplicated.any()):
        raise RuntimeError("duplicate method rows were generated")
    return {
        "status": "PASS",
        "row_count": int(len(rows)),
        "method_count": int(rows["method"].nunique()),
        "max_adaptive_bits": (
            None if adaptive.empty else int(adaptive["total_bits"].max())
        ),
        "registered_adaptive_hard_cap_bits": hard_max,
        "candidate_bins_in_band": True,
        "duplicate_rows": 0,
    }


def _gate(
    rows: pd.DataFrame,
    by_snr: pd.DataFrame,
    overall: pd.DataFrame,
    *,
    config: StageBConfig,
) -> dict[str, object]:
    if config.mode == "smoke":
        return {
            "status": "SMOKE_ONLY",
            "reason": "small-sample smoke validates code and protocol only",
        }
    metrics = overall.set_index("method")
    primary_method = "BRSR-Rollout-T32"
    primary_baselines = ("Best-Static-T32", "SNR-only-T32")
    if primary_method not in metrics.index or any(
        name not in metrics.index for name in primary_baselines
    ):
        raise RuntimeError("development gate methods are missing")
    primary_baseline = min(
        primary_baselines,
        key=lambda name: float(metrics.loc[name, "rmse_samples"]),
    )
    primary_rmse = float(metrics.loc[primary_method, "rmse_samples"])
    primary_baseline_rmse = float(
        metrics.loc[primary_baseline, "rmse_samples"]
    )
    primary_relative_gain = (
        primary_baseline_rmse - primary_rmse
    ) / primary_baseline_rmse
    primary_bit_error = abs(
        float(metrics.loc[primary_method, "mean_bits"]) - 32.0
    )
    accuracy_route_pass = (
        primary_relative_gain >= 0.05
        and primary_bit_error <= config.lambda_bit_tolerance
    )

    savings_candidates = []
    adaptive_methods = [
        f"BRSR-Rollout-T{target}" for target in (24, 32, 40)
    ]
    baseline_methods = [
        f"{family}-T{target}"
        for family in ("Best-Static", "SNR-only")
        for target in (24, 32, 40)
    ]
    for adaptive in adaptive_methods:
        for baseline in baseline_methods:
            if adaptive not in metrics.index or baseline not in metrics.index:
                continue
            adaptive_rmse = float(metrics.loc[adaptive, "rmse_samples"])
            baseline_rmse = float(metrics.loc[baseline, "rmse_samples"])
            if adaptive_rmse > baseline_rmse:
                continue
            adaptive_bits = float(metrics.loc[adaptive, "mean_bits"])
            baseline_bits = float(metrics.loc[baseline, "mean_bits"])
            savings_candidates.append(
                (
                    (baseline_bits - adaptive_bits) / baseline_bits,
                    adaptive,
                    baseline,
                )
            )
    best_savings = max(
        savings_candidates,
        default=(-math.inf, primary_method, primary_baseline),
        key=lambda item: item[0],
    )
    savings_route_pass = best_savings[0] >= 0.10

    if accuracy_route_pass:
        method = primary_method
        strongest = primary_baseline
        selected_route = "same_bit_accuracy"
    elif savings_route_pass:
        _, method, strongest = best_savings
        selected_route = "same_accuracy_bit_saving"
    else:
        method = primary_method
        strongest = primary_baseline
        selected_route = "none"

    adaptive_rmse = float(metrics.loc[method, "rmse_samples"])
    baseline_rmse = float(metrics.loc[strongest, "rmse_samples"])
    relative_gain = (baseline_rmse - adaptive_rmse) / baseline_rmse
    adaptive_snr = by_snr[by_snr["method"] == method].set_index("snr_db")
    baseline_snr = by_snr[by_snr["method"] == strongest].set_index("snr_db")
    relative_by_snr = (
        adaptive_snr["rmse_samples"] / baseline_snr["rmse_samples"] - 1.0
    )
    worst_degradation = float(relative_by_snr.max())
    coverage = float(metrics.loc[method, "credible90_coverage"])
    risk_ratio = float(metrics.loc[method, "risk_to_mse_ratio"])
    clipping = float(metrics.loc[method, "mean_clipping_rate"])
    bootstrap = _cluster_bootstrap_difference(
        rows,
        method,
        strongest,
        config.bootstrap_repetitions,
    )
    passed = (
        (accuracy_route_pass or savings_route_pass)
        and worst_degradation <= 0.05
        and bootstrap["paired_mse_gain_ci95_low"] > 0.0
        and 0.80 <= coverage <= 0.98
        and 0.50 <= risk_ratio <= 2.00
        and clipping <= 0.01
    )
    return {
        "status": (
            "READY_FOR_NEW_LOCKED_DESIGN"
            if passed
            else "DEVELOPMENT_NO_GO"
        ),
        "primary_method": primary_method,
        "primary_strongest_baseline": primary_baseline,
        "comparison_method": method,
        "comparison_baseline": strongest,
        "selected_success_route": selected_route,
        "adaptive_rmse_samples": adaptive_rmse,
        "baseline_rmse_samples": baseline_rmse,
        "relative_rmse_gain": relative_gain,
        "worst_per_snr_relative_degradation": worst_degradation,
        "primary_t32_relative_rmse_gain": primary_relative_gain,
        "primary_t32_mean_bit_target_error": primary_bit_error,
        "best_bit_saving_fraction_at_no_worse_rmse": (
            None if not math.isfinite(best_savings[0]) else best_savings[0]
        ),
        "bit_saving_adaptive_method": best_savings[1],
        "bit_saving_baseline_method": best_savings[2],
        "credible90_coverage": coverage,
        "risk_to_empirical_mse_ratio": risk_ratio,
        "mean_quantizer_clipping_rate": clipping,
        **bootstrap,
        "criteria": {
            "relative_rmse_gain_min": 0.05,
            "same_accuracy_bit_saving_min": 0.10,
            "per_snr_degradation_max": 0.05,
            "mean_bit_target_error_max": config.lambda_bit_tolerance,
            "paired_mse_ci95_low_must_be_positive": True,
            "credible90_coverage_range": [0.80, 0.98],
            "risk_to_empirical_mse_ratio_range": [0.50, 2.00],
            "mean_quantizer_clipping_rate_max": 0.01,
        },
    }


def _plot_rmse(by_snr: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    preferred = [
        "Best-Static-T32",
        "SNR-only-T32",
        "BRSR-Greedy-T32",
        "BRSR-Rollout-T32",
        "Base-8bit",
        "All-8bit",
    ]
    styles = {
        "Best-Static-T32": ("#4c78a8", "o", "-"),
        "SNR-only-T32": ("#f58518", "s", "--"),
        "BRSR-Greedy-T32": ("#54a24b", "^", "-."),
        "BRSR-Rollout-T32": ("#b279a2", "D", "-"),
        "Base-8bit": ("#9d755d", "v", ":"),
        "All-8bit": ("#79706e", "P", "--"),
    }
    for method in preferred:
        subset = by_snr[by_snr["method"] == method].sort_values("snr_db")
        if subset.empty:
            continue
        color, marker, line = styles[method]
        ax.plot(
            subset["snr_db"],
            subset["rmse_samples"],
            label=method,
            color=color,
            marker=marker,
            linestyle=line,
            linewidth=1.5,
            markersize=5,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("TDOA RMSE (samples)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_pareto(overall: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    families = {
        "Best-Static": "#4c78a8",
        "SNR-only": "#f58518",
        "BRSR-Greedy": "#54a24b",
        "BRSR-Rollout": "#b279a2",
        "Reference": "#79706e",
    }
    for family in ("Best-Static", "SNR-only", "BRSR-Greedy", "BRSR-Rollout"):
        subset = overall[
            overall["method"].str.startswith(f"{family}-T")
        ].sort_values("mean_bits")
        if subset.empty:
            continue
        ax.plot(
            subset["mean_bits"],
            subset["rmse_samples"],
            color=families[family],
            marker="o",
            linewidth=1.5,
            markersize=5,
            label=family,
        )
        for _, row in subset.iterrows():
            target = str(row["method"]).rsplit("T", maxsplit=1)[-1]
            ax.annotate(
                target,
                (row["mean_bits"], row["rmse_samples"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    references = overall[overall["method"].isin(("Base-8bit", "All-8bit"))]
    for _, row in references.iterrows():
        ax.scatter(
            row["mean_bits"],
            row["rmse_samples"],
            color=families["Reference"],
            marker="x",
            s=48,
        )
        ax.annotate(
            str(row["method"]),
            (row["mean_bits"], row["rmse_samples"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Mean communication (bits per B-station observation)")
    ax.set_ylabel("TDOA RMSE (samples)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(by_snr: pd.DataFrame, output: Path) -> None:
    methods = ("BRSR-Greedy-T32", "BRSR-Rollout-T32")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for method, color in zip(methods, ("#54a24b", "#b279a2"), strict=True):
        subset = by_snr[by_snr["method"] == method].sort_values("snr_db")
        if subset.empty:
            continue
        axes[0].plot(
            subset["snr_db"],
            subset["risk_to_mse_ratio"],
            marker="o",
            linewidth=1.5,
            color=color,
            label=method,
        )
        axes[1].plot(
            subset["snr_db"],
            subset["credible90_coverage"],
            marker="s",
            linewidth=1.5,
            color=color,
            label=method,
        )
    axes[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("Mean posterior risk / empirical MSE")
    axes[1].axhline(0.90, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Empirical 90% credible coverage")
    for ax in axes:
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_outputs(
    *,
    output: Path,
    rows: pd.DataFrame,
    by_snr: pd.DataFrame,
    overall: pd.DataFrame,
    lambda_rows: pd.DataFrame,
    static_plans: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
    action_summary: pd.DataFrame,
    covariance: np.ndarray,
    covariance_diagnostics: dict[str, float],
    chain_self_check: dict[str, object],
    rollout_convergence: pd.DataFrame,
    validation: dict[str, object],
    gate: dict[str, object],
    config: StageBConfig,
    waveform: WaveformConfig,
    protocol: StageBProtocol,
    elapsed_seconds: float,
) -> None:
    rows.to_csv(output / "stage_b_trials.csv", index=False, encoding="utf-8-sig")
    by_snr.to_csv(
        output / "summary_by_snr.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        output / "summary_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lambda_rows.to_csv(
        output / "lambda_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    static_plans.to_csv(
        output / "static_plans.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_metadata.to_csv(
        output / "candidate_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    action_summary.to_csv(
        output / "action_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rollout_convergence.to_csv(
        output / "rollout_convergence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.save(output / "candidate_covariance.npy", covariance)
    (output / "covariance_diagnostics.json").write_text(
        json.dumps(covariance_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "chain_self_check.json").write_text(
        json.dumps(chain_self_check, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "development_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_rmse(by_snr, output / "FigB1_TDOA_RMSE.svg")
    _plot_pareto(overall, output / "FigB2_Risk_Bit_Pareto.svg")
    _plot_calibration(by_snr, output / "FigB3_Posterior_Calibration.svg")

    source_files = [
        Path(__file__),
        ROOT / "brsr_tdoa" / "brsr_stage_b.py",
        ROOT / "brsr_tdoa" / "brsr_waveform.py",
        ROOT / "brsr_tdoa" / "brsr_core.py",
    ]
    manifest = {
        "experiment": "BRSR-TDOA Stage B development",
        "mode": config.mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "waveform": asdict(waveform),
        "protocol": asdict(protocol),
        "elapsed_seconds": elapsed_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "assumptions": {
            "signal": "BPSK-RRC, single-path AWGN, linear fractional delay",
            "a_station_at_fusion": True,
            "a_station_communication_counted": False,
            "b_station_caches_full_precision_fft": True,
            "candidate_selection_uses_locked_data": False,
            "working_likelihood": "diagonal complex Gaussian approximation",
            "full_covariance_use": "diagnostic only; no approximate correlated quantized likelihood",
            "locked_evaluation_generated": False,
            "communication_scope": "B-station payload plus feedback; not a CR16 comparison",
            "exact_dp_scope": "tiny model-matched unit diagnostic only",
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for partial_name in (
        "stage_b_trials.partial.csv",
        "lambda_calibration.partial.csv",
        "static_plans.partial.csv",
    ):
        partial = output / partial_name
        if partial.exists():
            partial.unlink()


def run(config: StageBConfig, output: Path) -> dict[str, object]:
    _validate_config(config)
    waveform = WaveformConfig()
    waveform.validate()
    bins = design_candidate_bins(waveform)
    protocol = StageBProtocol(
        rollout_samples=config.rollout_samples,
        rollout_first_action_candidates=config.rollout_first_actions,
    )
    protocol.validate(bins.size)
    quantizer = NestedComplexQuantizer(
        max_bits=protocol.max_bits,
        clip_abs=protocol.quantizer_clip_abs,
    )
    chain_self_check = _embedded_chain_checks(waveform, bins, quantizer)

    print(f"Mode: {config.mode}", flush=True)
    print(
        f"Waveform: N={waveform.n_time}, Fs={waveform.fs_hz/1e6:.1f} MHz, "
        f"BPSK-RRC sps={waveform.samples_per_symbol}, "
        f"rolloff={waveform.rrc_rolloff}",
        flush=True,
    )
    print(f"Candidate bins: {bins.tolist()}", flush=True)
    print(
        f"Protocol: 4->8 bit, decisions={protocol.max_decisions}, "
        f"feedback={protocol.feedback_bits_for(bins.size)} bit, "
        f"targets={protocol.target_mean_bits}",
        flush=True,
    )
    print(
        "This run uses development pools only; it does not create locked results.",
        flush=True,
    )

    prior_rows = _generate_trials(
        config.prior_seeds,
        config.prior_trials_per_seed,
        waveform,
    )
    prior_trials = [item[2] for item in prior_rows]
    source_power, covariance, covariance_diagnostics = (
        estimate_candidate_statistics(prior_trials, bins)
    )
    relative_min_power = float(np.min(source_power) / np.max(source_power))
    covariance_diagnostics["relative_min_candidate_power"] = relative_min_power
    if relative_min_power < 0.01:
        raise RuntimeError(
            "a candidate bin is more than 20 dB below the strongest development bin"
        )
    model = build_stage_b_model(
        n_time=waveform.n_time,
        bin_indices=bins,
        source_power=source_power,
        base_bins=waveform.base_bins,
        tau_min=waveform.tau_min,
        tau_max=waveform.tau_max,
        tau_step=config.tau_step,
        rho_amplitudes=config.rho_amplitudes,
        rho_phase_count=config.rho_phase_count,
    )

    calibration_trials = _generate_trials(
        config.calibration_seeds,
        config.calibration_trials_per_seed,
        waveform,
    )
    evaluation_trials = _generate_trials(
        config.evaluation_seeds,
        config.evaluation_trials_per_seed,
        waveform,
    )
    calibration_observations = _make_observations(
        stage="calibration",
        trials=calibration_trials,
        snr_values=config.snr_db,
        bins=bins,
        model=model,
        quantizer=quantizer,
    )
    evaluation_observations = _make_observations(
        stage="evaluation",
        trials=evaluation_trials,
        snr_values=config.snr_db,
        bins=bins,
        model=model,
        quantizer=quantizer,
    )
    print(
        f"Observations: calibration={len(calibration_observations)}, "
        f"evaluation={len(evaluation_observations)}",
        flush=True,
    )

    pooled_plans, snr_plans, static_plan_table = _select_static_plans(
        calibration_observations,
        protocol=protocol,
        quantizer=quantizer,
        model=model,
        config=config,
    )
    static_plan_table.to_csv(
        output / "static_plans.partial.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lambda_choices: dict[tuple[str, int], float] = {}
    calibration_rows = []
    for method in ("greedy", "rollout"):
        method_cache: dict[float, float] = {}
        for target in protocol.target_mean_bits:
            chosen, records = _calibrate_lambda(
                calibration_observations,
                method=method,
                target_bits=target,
                config=config,
                protocol=protocol,
                quantizer=quantizer,
                model=model,
                cache=method_cache,
            )
            lambda_choices[(method, target)] = chosen
            calibration_rows.extend(records)
            pd.DataFrame(calibration_rows).drop_duplicates().to_csv(
                output / "lambda_calibration.partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    rows: list[dict[str, object]] = []
    base4 = tuple(
        4 if position in model.base_positions else 0
        for position in range(model.n_bins)
    )
    base8 = tuple(
        8 if position in model.base_positions else 0
        for position in range(model.n_bins)
    )
    all4 = (4,) * model.n_bins
    all8 = (8,) * model.n_bins
    for label, levels in (
        ("Base-4bit", base4),
        ("Base-8bit", base8),
        ("All-4bit", all4),
        ("All-8bit", all8),
    ):
        rows.extend(
            _run_fixed_rows(
                evaluation_observations,
                method=label,
                level_for_observation=lambda _obs, levels=levels: levels,
                quantizer=quantizer,
                model=model,
            )
        )
    _save_partial_rows(rows, output)

    for target in protocol.target_mean_bits:
        rows.extend(
            _run_fixed_rows(
                evaluation_observations,
                method=f"Best-Static-T{target}",
                level_for_observation=lambda _obs, target=target: pooled_plans[target],
                quantizer=quantizer,
                model=model,
            )
        )
        _save_partial_rows(rows, output)
        rows.extend(
            _run_fixed_rows(
                evaluation_observations,
                method=f"SNR-only-T{target}",
                level_for_observation=lambda obs, target=target: snr_plans[
                    (obs.snr_db, target)
                ],
                quantizer=quantizer,
                model=model,
            )
        )
        _save_partial_rows(rows, output)
        for method in ("greedy", "rollout"):
            label = f"BRSR-{method.capitalize()}-T{target}"
            rows.extend(
                _run_adaptive_rows(
                    evaluation_observations,
                    method=method,
                    label=label,
                    lambda_per_bit=lambda_choices[(method, target)],
                    protocol=protocol,
                    quantizer=quantizer,
                    model=model,
                    progress_every=config.progress_every,
                )
            )
            _save_partial_rows(rows, output)

    trial_table = pd.DataFrame(rows)
    adaptive_rows = trial_table[
        trial_table["method"].str.startswith("BRSR-")
    ]
    action_summary = (
        adaptive_rows.groupby(
            ["method", "snr_db", "first_action"],
            observed=True,
            sort=False,
        )
        .agg(
            n=("first_action", "size"),
            mean_total_bits=("total_bits", "mean"),
            mean_actual_risk_reduction=(
                "actual_posterior_risk_reduction",
                "mean",
            ),
        )
        .reset_index()
    )
    rollout_convergence = _rollout_convergence_diagnostic(
        evaluation_observations,
        lambda_per_bit=lambda_choices[("rollout", 32)],
        config=config,
        protocol=protocol,
        quantizer=quantizer,
        model=model,
    )
    by_snr, overall = _summaries(trial_table)
    validation = _validate_rows(
        trial_table,
        waveform=waveform,
        protocol=protocol,
        candidate_bins=bins,
    )
    gate = _gate(
        trial_table,
        by_snr,
        overall,
        config=config,
    )
    theory_power = raised_cosine_power_at_bins(bins, waveform)
    candidate_metadata = pd.DataFrame(
        {
            "position": np.arange(bins.size),
            "bin_index": bins,
            "frequency_hz": bins * waveform.fs_hz / waveform.n_time,
            "theoretical_rc_power": theory_power,
            "development_power": source_power,
            "is_base": np.isin(bins, waveform.base_bins),
        }
    )
    return {
        "rows": trial_table,
        "by_snr": by_snr,
        "overall": overall,
        "lambda_rows": pd.DataFrame(calibration_rows).drop_duplicates(),
        "static_plans": static_plan_table,
        "candidate_metadata": candidate_metadata,
        "action_summary": action_summary,
        "covariance": covariance,
        "covariance_diagnostics": covariance_diagnostics,
        "chain_self_check": chain_self_check,
        "rollout_convergence": rollout_convergence,
        "validation": validation,
        "gate": gate,
        "waveform": waveform,
        "protocol": protocol,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(CONFIGS), default=PYCHARM_RUN_MODE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "运行结果" / f"BRSR_STAGE_B_{timestamp}_{config.mode}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        tee_out = Tee(sys.stdout, log_handle)
        tee_err = Tee(sys.stderr, log_handle)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Output: {output}", flush=True)
            result = run(config, output)
            elapsed = time.perf_counter() - start
            _write_outputs(
                output=output,
                rows=result["rows"],
                by_snr=result["by_snr"],
                overall=result["overall"],
                lambda_rows=result["lambda_rows"],
                static_plans=result["static_plans"],
                candidate_metadata=result["candidate_metadata"],
                action_summary=result["action_summary"],
                covariance=result["covariance"],
                covariance_diagnostics=result["covariance_diagnostics"],
                chain_self_check=result["chain_self_check"],
                rollout_convergence=result["rollout_convergence"],
                validation=result["validation"],
                gate=result["gate"],
                config=config,
                waveform=result["waveform"],
                protocol=result["protocol"],
                elapsed_seconds=elapsed,
            )
            print("\nOverall summary", flush=True)
            print(
                result["overall"][
                    ["method", "rmse_samples", "mean_bits", "credible90_coverage"]
                ].to_string(index=False),
                flush=True,
            )
            print(
                "\nDevelopment gate\n"
                + json.dumps(result["gate"], ensure_ascii=False, indent=2),
                flush=True,
            )
            print(f"[Done] elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
