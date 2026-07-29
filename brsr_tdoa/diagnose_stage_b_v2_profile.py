"""BRSR Stage B v2 步骤1：连续复增益剖面似然诊断。

本脚本只验证固定频点、固定码率下的估计模型，不修改 Stage B v1，
不重新选择候选池，也不运行 Greedy/Rollout。主方法使用融合端可获得的
完整 A 端观测构造线性有限窗时延模板，并对精确量化区间似然中的连续
复增益进行确定性剖面优化。
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brsr_tdoa.brsr_core import (  # noqa: E402
    BayesianSpectralModel,
    NestedComplexQuantizer,
)
from brsr_tdoa.brsr_stage_b import (  # noqa: E402
    StageBProtocol,
    build_stage_b_model,
    evaluate_fixed,
    make_observation,
)
from brsr_tdoa.brsr_waveform import (  # noqa: E402
    StructuredTrial,
    WaveformConfig,
    apply_linear_fractional_delay,
    candidate_fft_values,
    generate_structured_trial,
    observe_trial,
)


SOURCE_RESULT_DIR = (
    ROOT
    / "运行结果"
    / "BRSR"
    / "BRSR_STAGE_B_20260718_173940_development"
)
PYCHARM_RUN_MODE: Literal["smoke", "research"] = "research"


@dataclass(frozen=True)
class Step1Config:
    mode: str
    snr_db: tuple[float, ...]
    seeds: tuple[int, ...]
    trials_per_seed: int
    profile_iterations: int
    progress_every: int


@dataclass(frozen=True)
class ProfileEstimate:
    estimate: float
    posterior_risk: float
    credible90_contains_true: bool
    profile_log_likelihood: np.ndarray


class _Tee:
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


def config_for_mode(mode: str) -> Step1Config:
    if mode == "smoke":
        return Step1Config(
            mode=mode,
            snr_db=(0.0, 15.0),
            seeds=(20269301,),
            trials_per_seed=2,
            profile_iterations=2,
            progress_every=1,
        )
    if mode == "research":
        return Step1Config(
            mode=mode,
            snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
            seeds=(20269301, 20269302, 20269303),
            trials_per_seed=20,
            profile_iterations=4,
            progress_every=10,
        )
    raise ValueError(f"unsupported mode: {mode}")


def _load_frozen() -> dict[str, object]:
    required = (
        "manifest.json",
        "candidate_metadata.csv",
        "static_plans.csv",
    )
    for name in required:
        if not (SOURCE_RESULT_DIR / name).is_file():
            raise FileNotFoundError(SOURCE_RESULT_DIR / name)
    manifest = json.loads(
        (SOURCE_RESULT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    waveform = WaveformConfig(**manifest["waveform"])
    protocol = StageBProtocol(**manifest["protocol"])
    candidate = pd.read_csv(
        SOURCE_RESULT_DIR / "candidate_metadata.csv", encoding="utf-8-sig"
    ).sort_values("position")
    bins = candidate["bin_index"].to_numpy(dtype=int)
    source_power = candidate["development_power"].to_numpy(dtype=float)
    stage_config = manifest["config"]
    model = build_stage_b_model(
        n_time=waveform.n_time,
        bin_indices=bins,
        source_power=source_power,
        base_bins=waveform.base_bins,
        tau_min=waveform.tau_min,
        tau_max=waveform.tau_max,
        tau_step=float(stage_config["tau_step"]),
        rho_amplitudes=tuple(float(x) for x in stage_config["rho_amplitudes"]),
        rho_phase_count=int(stage_config["rho_phase_count"]),
    )
    quantizer = NestedComplexQuantizer(
        max_bits=protocol.max_bits,
        clip_abs=protocol.quantizer_clip_abs,
    )
    static = pd.read_csv(SOURCE_RESULT_DIR / "static_plans.csv", encoding="utf-8-sig")
    return {
        "manifest": manifest,
        "waveform": waveform,
        "protocol": protocol,
        "bins": bins,
        "source_power": source_power,
        "model": model,
        "quantizer": quantizer,
        "static": static,
    }


def _parse_levels(value: object, n_bins: int) -> tuple[int, ...]:
    levels = tuple(int(x) for x in str(value).split())
    if len(levels) != n_bins or any(x not in (0, 4, 8) for x in levels):
        raise ValueError(f"invalid level vector: {value}")
    return levels


def _frozen_plans(static: pd.DataFrame, model: BayesianSpectralModel) -> dict[str, object]:
    t32 = static[static["target_bits"] == 32].copy()
    best = t32[t32["method"] == "Best-Static"]
    if len(best) != 1:
        raise RuntimeError("frozen Best-Static-T32 plan is missing or duplicated")
    best_levels = _parse_levels(best.iloc[0]["levels"], model.n_bins)
    snr_rows = t32[t32["method"] == "SNR-only"]
    snr_levels = {
        float(row.snr_db): _parse_levels(row.levels, model.n_bins)
        for row in snr_rows.itertuples(index=False)
    }
    if sum(x > 0 for x in best_levels) != 4 or sum(best_levels) != 16:
        raise RuntimeError("Best-Static-T32 must contain four 4-bit complex bins")
    return {"BestStatic-T32": best_levels, "SNR-only-T32": snr_levels}


def _generate_trials(config: Step1Config, waveform: WaveformConfig) -> dict[tuple[int, int], StructuredTrial]:
    trials: dict[tuple[int, int], StructuredTrial] = {}
    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        for trial_index in range(config.trials_per_seed):
            trials[(seed, trial_index)] = generate_structured_trial(rng, waveform)
    return trials


def _posterior_summary(
    log_likelihood: np.ndarray,
    model: BayesianSpectralModel,
    true_tau: float,
) -> ProfileEstimate:
    values = np.asarray(log_likelihood, dtype=float)
    if values.shape != (model.n_tau,) or not np.all(np.isfinite(values)):
        raise ValueError("profile log likelihood must be finite and match tau grid")
    log_posterior = values + np.log(model.tau_prior)
    log_posterior -= float(np.max(log_posterior))
    posterior = np.exp(log_posterior)
    posterior /= float(np.sum(posterior))
    estimate = float(np.sum(model.tau_values * posterior))
    risk = float(np.sum((model.tau_values - estimate) ** 2 * posterior))
    cdf = np.cumsum(posterior)
    lower = model.tau_values[int(np.searchsorted(cdf, 0.05, side="left"))]
    upper = model.tau_values[
        min(int(np.searchsorted(cdf, 0.95, side="left")), model.n_tau - 1)
    ]
    return ProfileEstimate(
        estimate=estimate,
        posterior_risk=max(risk, 0.0),
        credible90_contains_true=bool(lower <= true_tau <= upper),
        profile_log_likelihood=values,
    )


def _quantized_profile_scores(
    *,
    template: np.ndarray,
    source_variance: np.ndarray,
    sigma_b2: float,
    scale: float,
    levels: tuple[int, ...],
    fine_real: np.ndarray,
    fine_imag: np.ndarray,
    quantizer: NestedComplexQuantizer,
    model: BayesianSpectralModel,
    iterations: int,
) -> np.ndarray:
    selected = np.flatnonzero(np.asarray(levels) > 0)
    template = np.asarray(template, dtype=np.complex128)
    source_variance = np.asarray(source_variance, dtype=float)
    if template.shape != (model.n_bins, model.n_tau):
        raise ValueError("template shape does not match model")
    if source_variance.shape != (model.n_bins,):
        raise ValueError("source variance shape does not match model")
    if selected.size < 2:
        raise ValueError("profile likelihood needs at least two selected bins")

    initial_amp = np.abs(model.rho_values)
    initial_phase = np.angle(model.rho_values)
    rho_grid = initial_amp * np.exp(1j * initial_phase)
    scores = _score_rho_grid(
        template=template,
        source_variance=source_variance,
        sigma_b2=sigma_b2,
        scale=scale,
        selected=selected,
        levels=levels,
        fine_real=fine_real,
        fine_imag=fine_imag,
        quantizer=quantizer,
        rho_grid=np.broadcast_to(rho_grid[None, :], (model.n_tau, rho_grid.size)),
    )
    best = np.argmax(scores, axis=1)
    amp = initial_amp[best].copy()
    phase = initial_phase[best].copy()
    amp_step = 0.05
    phase_step = np.pi / 8.0
    offsets = np.asarray([-1.0, 0.0, 1.0])
    for _ in range(iterations):
        amp_candidates = np.clip(
            amp[:, None, None] + amp_step * offsets[None, :, None], 0.70, 0.90
        )
        phase_candidates = phase[:, None, None] + phase_step * offsets[None, None, :]
        rho_candidates = amp_candidates * np.exp(1j * phase_candidates)
        rho_candidates = rho_candidates.reshape(model.n_tau, -1)
        local_scores = _score_rho_grid(
            template=template,
            source_variance=source_variance,
            sigma_b2=sigma_b2,
            scale=scale,
            selected=selected,
            levels=levels,
            fine_real=fine_real,
            fine_imag=fine_imag,
            quantizer=quantizer,
            rho_grid=rho_candidates,
        )
        local_best = np.argmax(local_scores, axis=1)
        chosen = rho_candidates[np.arange(model.n_tau), local_best]
        amp = np.abs(chosen)
        phase = np.angle(chosen)
        amp_step *= 0.5
        phase_step *= 0.5
    return local_scores[np.arange(model.n_tau), local_best]


def _score_rho_grid(
    *,
    template: np.ndarray,
    source_variance: np.ndarray,
    sigma_b2: float,
    scale: float,
    selected: np.ndarray,
    levels: tuple[int, ...],
    fine_real: np.ndarray,
    fine_imag: np.ndarray,
    quantizer: NestedComplexQuantizer,
    rho_grid: np.ndarray,
) -> np.ndarray:
    n_tau, n_rho = rho_grid.shape
    total = np.zeros((n_tau, n_rho), dtype=float)
    for position in selected:
        bits = int(levels[int(position)])
        real_code = int(quantizer.coarsen(fine_real[position], bits))
        imag_code = int(quantizer.coarsen(fine_imag[position], bits))
        mean = template[position, :, None] * rho_grid
        variance = (
            source_variance[position] * np.abs(rho_grid) ** 2 + sigma_b2
        ) / scale**2
        total += quantizer.log_probability(
            mean,
            variance,
            real_code,
            imag_code,
            bits,
        )
    return total


def _unquantized_profile_scores(
    *,
    normalized_b: np.ndarray,
    template: np.ndarray,
    source_variance: np.ndarray,
    sigma_b2: float,
    scale: float,
    levels: tuple[int, ...],
    model: BayesianSpectralModel,
    iterations: int,
) -> np.ndarray:
    selected = np.flatnonzero(np.asarray(levels) > 0)
    initial = np.broadcast_to(model.rho_values[None, :], (model.n_tau, model.n_rho))
    scores = _score_unquantized(
        normalized_b, template, source_variance, sigma_b2, scale, selected, initial
    )
    best = np.argmax(scores, axis=1)
    amp = np.abs(model.rho_values[best])
    phase = np.angle(model.rho_values[best])
    amp_step = 0.05
    phase_step = np.pi / 8.0
    offsets = np.asarray([-1.0, 0.0, 1.0])
    for _ in range(iterations):
        aa = np.clip(amp[:, None, None] + amp_step * offsets[None, :, None], 0.70, 0.90)
        pp = phase[:, None, None] + phase_step * offsets[None, None, :]
        rho = (aa * np.exp(1j * pp)).reshape(model.n_tau, -1)
        local = _score_unquantized(
            normalized_b, template, source_variance, sigma_b2, scale, selected, rho
        )
        idx = np.argmax(local, axis=1)
        chosen = rho[np.arange(model.n_tau), idx]
        amp, phase = np.abs(chosen), np.angle(chosen)
        amp_step *= 0.5
        phase_step *= 0.5
    return local[np.arange(model.n_tau), idx]


def _score_unquantized(
    normalized_b: np.ndarray,
    template: np.ndarray,
    source_variance: np.ndarray,
    sigma_b2: float,
    scale: float,
    selected: np.ndarray,
    rho_grid: np.ndarray,
) -> np.ndarray:
    total = np.zeros(rho_grid.shape, dtype=float)
    for position in selected:
        mean = template[position, :, None] * rho_grid
        variance = (
            source_variance[position] * np.abs(rho_grid) ** 2 + sigma_b2
        ) / scale**2
        residual = normalized_b[position] - mean
        total += -np.log(np.pi * variance) - np.abs(residual) ** 2 / variance
    return total


def _ideal_template(
    y_a_selected: np.ndarray,
    sigma2: float,
    scale: float,
    model: BayesianSpectralModel,
) -> tuple[np.ndarray, np.ndarray]:
    gain = model.source_power / (model.source_power + sigma2)
    source_mean = gain * y_a_selected
    source_variance = model.source_power * sigma2 / (model.source_power + sigma2)
    steering = np.exp(-1j * model.omega[:, None] * model.tau_values[None, :])
    return source_mean[:, None] * steering / scale, source_variance


def _linear_a_template(
    y_a: np.ndarray,
    sigma2: float,
    scale: float,
    waveform: WaveformConfig,
    model: BayesianSpectralModel,
) -> tuple[np.ndarray, np.ndarray]:
    columns = []
    for tau in model.tau_values:
        delayed = apply_linear_fractional_delay(
            y_a, float(tau), waveform.fractional_delay_half_len
        )
        columns.append(candidate_fft_values(delayed, model.bin_indices) / scale)
    template = np.stack(columns, axis=1)
    # y_B - rho*D(y_A) includes B noise and delayed A noise. The delay filter has
    # near-unit energy, so sigma_A^2 is the deployable diagonal approximation.
    source_variance = np.full(model.n_bins, sigma2, dtype=float)
    return template, source_variance


def _oracle_model(model: BayesianSpectralModel, true_rho: complex) -> BayesianSpectralModel:
    from dataclasses import replace

    epsilon = 1e-12
    values = np.asarray([true_rho * (1.0 - epsilon), true_rho * (1.0 + epsilon)])
    oracle = replace(model, rho_values=values, rho_prior=np.asarray([0.5, 0.5]))
    oracle.validate()
    return oracle


def _row(
    *,
    plan: str,
    method: str,
    seed: int,
    trial_index: int,
    snr_db: float,
    true_tau: float,
    estimate: float,
    risk: float,
    coverage: bool,
) -> dict[str, object]:
    error = estimate - true_tau
    return {
        "plan": plan,
        "method": method,
        "seed": seed,
        "trial": trial_index,
        "snr_db": snr_db,
        "true_tau": true_tau,
        "estimate_tau": estimate,
        "error_samples": error,
        "squared_error_samples2": error**2,
        "posterior_risk_samples2": risk,
        "credible90_contains_true": coverage,
    }


def run_step1(config: Step1Config, output: Path) -> pd.DataFrame:
    frozen = _load_frozen()
    waveform = frozen["waveform"]
    protocol = frozen["protocol"]
    model = frozen["model"]
    quantizer = frozen["quantizer"]
    static = frozen["static"]
    if not isinstance(waveform, WaveformConfig) or not isinstance(protocol, StageBProtocol):
        raise TypeError("frozen waveform/protocol reconstruction failed")
    if not isinstance(model, BayesianSpectralModel) or not isinstance(quantizer, NestedComplexQuantizer):
        raise TypeError("frozen model/quantizer reconstruction failed")
    if not isinstance(static, pd.DataFrame):
        raise TypeError("frozen static plans are invalid")
    plans = _frozen_plans(static, model)
    trials = _generate_trials(config, waveform)
    old_seeds = set(frozen["manifest"]["config"]["prior_seeds"])
    old_seeds |= set(frozen["manifest"]["config"]["calibration_seeds"])
    old_seeds |= set(frozen["manifest"]["config"]["evaluation_seeds"])
    if old_seeds.intersection(config.seeds):
        raise RuntimeError("Step1 seeds overlap the frozen Stage B pools")

    rows: list[dict[str, object]] = []
    total = len(config.snr_db) * len(trials)
    done = 0
    started = time.perf_counter()
    for snr_db in config.snr_db:
        for (seed, trial_index), trial in trials.items():
            y_a, y_b, sigma2 = observe_trial(trial, snr_db)
            y_a_selected = candidate_fft_values(y_a, model.bin_indices)
            y_b_selected = candidate_fft_values(y_b, model.bin_indices)
            observation = make_observation(
                stage="stage_b_v2_step1",
                seed=seed,
                trial=trial_index,
                snr_db=snr_db,
                true_tau=trial.true_tau,
                true_rho=trial.true_rho,
                y_a_selected=y_a_selected,
                y_b_selected=y_b_selected,
                sigma2=sigma2,
                model=model,
                quantizer=quantizer,
            )
            normalized_b = y_b_selected / observation.public_b_scale
            ideal_template, ideal_variance = _ideal_template(
                y_a_selected, sigma2, observation.public_b_scale, model
            )
            linear_template, linear_variance = _linear_a_template(
                y_a, sigma2, observation.public_b_scale, waveform, model
            )
            oracle_model = _oracle_model(model, trial.true_rho)
            oracle_observation = make_observation(
                stage="stage_b_v2_step1_oracle",
                seed=seed,
                trial=trial_index,
                snr_db=snr_db,
                true_tau=trial.true_tau,
                true_rho=trial.true_rho,
                y_a_selected=y_a_selected,
                y_b_selected=y_b_selected,
                sigma2=sigma2,
                model=oracle_model,
                quantizer=quantizer,
            )
            for plan_name in ("BestStatic-T32", "SNR-only-T32"):
                raw = plans[plan_name]
                levels = raw[snr_db] if isinstance(raw, dict) else raw
                current = evaluate_fixed(
                    observation,
                    method="DiscreteMarginal24",
                    levels=levels,
                    quantizer=quantizer,
                    model=model,
                )
                rows.append(_row(
                    plan=plan_name, method="DiscreteMarginal24", seed=seed,
                    trial_index=trial_index, snr_db=snr_db, true_tau=trial.true_tau,
                    estimate=current.estimate, risk=current.posterior_risk,
                    coverage=current.credible90_contains_true,
                ))
                oracle = evaluate_fixed(
                    oracle_observation,
                    method="OracleRho-Quantized",
                    levels=levels,
                    quantizer=quantizer,
                    model=oracle_model,
                )
                rows.append(_row(
                    plan=plan_name, method="OracleRho-Quantized", seed=seed,
                    trial_index=trial_index, snr_db=snr_db, true_tau=trial.true_tau,
                    estimate=oracle.estimate, risk=oracle.posterior_risk,
                    coverage=oracle.credible90_contains_true,
                ))
                ideal_scores = _quantized_profile_scores(
                    template=ideal_template, source_variance=ideal_variance,
                    sigma_b2=sigma2, scale=observation.public_b_scale,
                    levels=levels, fine_real=observation.fine_real_codes,
                    fine_imag=observation.fine_imag_codes, quantizer=quantizer,
                    model=model, iterations=config.profile_iterations,
                )
                ideal = _posterior_summary(ideal_scores, model, trial.true_tau)
                rows.append(_row(
                    plan=plan_name, method="IdealPhase-QuantProfile", seed=seed,
                    trial_index=trial_index, snr_db=snr_db, true_tau=trial.true_tau,
                    estimate=ideal.estimate, risk=ideal.posterior_risk,
                    coverage=ideal.credible90_contains_true,
                ))
                linear_scores = _quantized_profile_scores(
                    template=linear_template, source_variance=linear_variance,
                    sigma_b2=sigma2, scale=observation.public_b_scale,
                    levels=levels, fine_real=observation.fine_real_codes,
                    fine_imag=observation.fine_imag_codes, quantizer=quantizer,
                    model=model, iterations=config.profile_iterations,
                )
                linear = _posterior_summary(linear_scores, model, trial.true_tau)
                rows.append(_row(
                    plan=plan_name, method="LinearA-QuantProfile", seed=seed,
                    trial_index=trial_index, snr_db=snr_db, true_tau=trial.true_tau,
                    estimate=linear.estimate, risk=linear.posterior_risk,
                    coverage=linear.credible90_contains_true,
                ))
                unquantized_scores = _unquantized_profile_scores(
                    normalized_b=normalized_b, template=linear_template,
                    source_variance=linear_variance, sigma_b2=sigma2,
                    scale=observation.public_b_scale, levels=levels, model=model,
                    iterations=config.profile_iterations,
                )
                unquantized = _posterior_summary(
                    unquantized_scores, model, trial.true_tau
                )
                rows.append(_row(
                    plan=plan_name, method="LinearA-UnquantizedProfile", seed=seed,
                    trial_index=trial_index, snr_db=snr_db, true_tau=trial.true_tau,
                    estimate=unquantized.estimate, risk=unquantized.posterior_risk,
                    coverage=unquantized.credible90_contains_true,
                ))
            done += 1
            if done % config.progress_every == 0 or done == total:
                elapsed = time.perf_counter() - started
                rate = elapsed / done
                eta = rate * (total - done)
                print(
                    f"[Progress] {done}/{total} SNR={snr_db:g} "
                    f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(output / "step1_rows.partial.csv", index=False)
    return pd.DataFrame(rows)


def _aggregate(group: pd.DataFrame) -> pd.Series:
    mse = float(group["squared_error_samples2"].mean())
    mean_risk = float(group["posterior_risk_samples2"].mean())
    return pd.Series({
        "n": len(group),
        "rmse_samples": math.sqrt(mse),
        "mae_samples": float(group["error_samples"].abs().mean()),
        "severe_rate_gt2": float((group["error_samples"].abs() > 2.0).mean()),
        "coverage90": float(group["credible90_contains_true"].mean()),
        "risk_to_mse": mean_risk / max(mse, 1e-12),
    })


def _summaries(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    overall = rows.groupby(["plan", "method"], observed=True).apply(
        _aggregate, include_groups=False
    ).reset_index()
    by_snr = rows.groupby(["plan", "method", "snr_db"], observed=True).apply(
        _aggregate, include_groups=False
    ).reset_index()
    primary = rows[rows["plan"] == "BestStatic-T32"]
    mse = primary.groupby("method")["squared_error_samples2"].mean()
    current = float(mse["DiscreteMarginal24"])
    candidate = float(mse["LinearA-QuantProfile"])
    oracle = float(mse["OracleRho-Quantized"])
    oracle_gap = current - oracle
    recovered = (current - candidate) / oracle_gap if oracle_gap > 0.0 else None
    high = primary[primary["snr_db"] >= 10.0]
    severe = high.groupby("method")["error_samples"].apply(
        lambda x: float((x.abs() > 2.0).mean())
    )
    severe_current = float(severe["DiscreteMarginal24"])
    severe_candidate = float(severe["LinearA-QuantProfile"])
    severe_reduction = (
        (severe_current - severe_candidate) / severe_current
        if severe_current > 0.0 else (1.0 if severe_candidate == 0.0 else -math.inf)
    )
    wide = by_snr[by_snr["plan"] == "BestStatic-T32"].pivot(
        index="snr_db", columns="method", values="rmse_samples"
    )
    degradation = (
        wide["LinearA-QuantProfile"] / wide["DiscreteMarginal24"] - 1.0
    )
    primary_summary = overall[
        (overall["plan"] == "BestStatic-T32")
        & (overall["method"] == "LinearA-QuantProfile")
    ].iloc[0]
    criteria = {
        "oracle_mse_gap_recovery_min": 0.25,
        "high_snr_severe_error_reduction_min": 0.50,
        "worst_per_snr_rmse_degradation_max": 0.05,
        "coverage90_range": [0.80, 0.98],
        "risk_to_mse_range": [0.50, 2.00],
    }
    checks = {
        "oracle_mse_gap_recovery": recovered is not None
        and recovered >= criteria["oracle_mse_gap_recovery_min"],
        "high_snr_severe_error_reduction": severe_reduction
        >= criteria["high_snr_severe_error_reduction_min"],
        "per_snr_degradation": float(degradation.max())
        <= criteria["worst_per_snr_rmse_degradation_max"],
        "coverage": criteria["coverage90_range"][0]
        <= float(primary_summary["coverage90"])
        <= criteria["coverage90_range"][1],
        "risk_calibration": criteria["risk_to_mse_range"][0]
        <= float(primary_summary["risk_to_mse"])
        <= criteria["risk_to_mse_range"][1],
    }
    gate = {
        "status": "STEP1_GO" if all(checks.values()) else "STEP1_NO_GO",
        "primary_plan": "BestStatic-T32",
        "primary_method": "LinearA-QuantProfile",
        "current_mse": current,
        "candidate_mse": candidate,
        "oracle_mse": oracle,
        "oracle_mse_gap_recovery": recovered,
        "oracle_mse_gap_is_positive": oracle_gap > 0.0,
        "high_snr_severe_current": severe_current,
        "high_snr_severe_candidate": severe_candidate,
        "high_snr_severe_reduction": severe_reduction,
        "worst_per_snr_rmse_degradation": float(degradation.max()),
        "coverage90": float(primary_summary["coverage90"]),
        "risk_to_mse": float(primary_summary["risk_to_mse"]),
        "criteria": criteria,
        "checks": checks,
        "interpretation_boundary": (
            "OracleRho uses truth and unquantized profile removes quantization, but neither "
            "is a guaranteed performance upper bound under likelihood mismatch. They are "
            "diagnostic references only. No candidate pool or active policy is changed."
        ),
    }
    return overall, by_snr, gate


def _paired_diagnostics(rows: pd.DataFrame) -> dict[str, object]:
    primary = rows[rows["plan"] == "BestStatic-T32"]
    wide = primary.pivot(
        index=["seed", "trial", "snr_db"],
        columns="method",
        values="squared_error_samples2",
    ).reset_index()
    rng = np.random.default_rng(20260722)
    result: dict[str, object] = {
        "reference": "DiscreteMarginal24",
        "bootstrap_unit": "trajectory cluster (seed, trial)",
        "bootstrap_repetitions": 5000,
        "posthoc_descriptive_not_gate": True,
    }
    for method in (
        "IdealPhase-QuantProfile",
        "LinearA-QuantProfile",
        "LinearA-UnquantizedProfile",
        "OracleRho-Quantized",
    ):
        delta = wide[method] - wide["DiscreteMarginal24"]
        cluster = (
            wide.assign(delta=delta)
            .groupby(["seed", "trial"], observed=True)["delta"]
            .mean()
            .to_numpy(dtype=float)
        )
        bootstrap = np.empty(5000, dtype=float)
        for index in range(bootstrap.size):
            bootstrap[index] = float(
                rng.choice(cluster, size=cluster.size, replace=True).mean()
            )
        result[method] = {
            "paired_mse_delta_candidate_minus_reference": float(delta.mean()),
            "trajectory_cluster_ci95": np.quantile(
                bootstrap, [0.025, 0.975]
            ).tolist(),
            "fraction_trajectories_candidate_better": float(np.mean(cluster < 0.0)),
        }
    return result


def _validate_rows(rows: pd.DataFrame, config: Step1Config) -> dict[str, object]:
    methods = 5
    plans = 2
    expected = (
        len(config.snr_db)
        * len(config.seeds)
        * config.trials_per_seed
        * methods
        * plans
    )
    duplicate_count = int(
        rows.duplicated(["plan", "method", "seed", "trial", "snr_db"]).sum()
    )
    numeric = rows[
        [
            "true_tau",
            "estimate_tau",
            "error_samples",
            "squared_error_samples2",
            "posterior_risk_samples2",
        ]
    ].to_numpy(dtype=float)
    checks = {
        "row_count": len(rows) == expected,
        "no_duplicate_method_trials": duplicate_count == 0,
        "all_numeric_outputs_finite": bool(np.all(np.isfinite(numeric))),
        "non_negative_squared_error": bool(
            np.all(rows["squared_error_samples2"].to_numpy(dtype=float) >= 0.0)
        ),
        "non_negative_posterior_risk": bool(
            np.all(rows["posterior_risk_samples2"].to_numpy(dtype=float) >= 0.0)
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_rows": expected,
        "actual_rows": len(rows),
        "duplicate_count": duplicate_count,
        "checks": checks,
    }


def _plot(by_snr: pd.DataFrame, output: Path) -> None:
    methods = (
        "DiscreteMarginal24",
        "IdealPhase-QuantProfile",
        "LinearA-QuantProfile",
        "LinearA-UnquantizedProfile",
        "OracleRho-Quantized",
    )
    styles = {
        "DiscreteMarginal24": ("#4c78a8", "o", "-"),
        "IdealPhase-QuantProfile": ("#f58518", "s", "--"),
        "LinearA-QuantProfile": ("#54a24b", "^", "-"),
        "LinearA-UnquantizedProfile": ("#b279a2", "D", ":"),
        "OracleRho-Quantized": ("#e45756", "v", "-."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    data = by_snr[by_snr["plan"] == "BestStatic-T32"]
    for method in methods:
        group = data[data["method"] == method].sort_values("snr_db")
        color, marker, linestyle = styles[method]
        axes[0].plot(
            group["snr_db"], group["rmse_samples"], label=method,
            color=color, marker=marker, linestyle=linestyle, linewidth=1.4,
        )
        axes[1].plot(
            group["snr_db"], group["severe_rate_gt2"], label=method,
            color=color, marker=marker, linestyle=linestyle, linewidth=1.4,
        )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("TDOA RMSE (samples)")
    axes[0].set_title("(a) Fixed BestStatic-T32 estimation")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Severe error rate, |error| > 2 samples")
    axes[1].set_title("(b) Gross-delay failures")
    axes[1].grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    fig.savefig(output / "FigV2S1_Profile_Likelihood_Diagnostic.svg", format="svg")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "运行结果"
        / "BRSR"
        / f"BRSR_STAGE_B_V2_STEP1_{timestamp}_{mode}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "research"), default=PYCHARM_RUN_MODE)
    args = parser.parse_args()
    config = config_for_mode(args.mode)
    output = _output_dir(args.mode)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.log").open("w", encoding="utf-8") as log_file:
        tee_out = _Tee(sys.stdout, log_file)
        tee_err = _Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            started = time.perf_counter()
            print(f"Mode: {config.mode}", flush=True)
            print(f"Output: {output}", flush=True)
            print(
                f"Seeds: {config.seeds}; trials/seed={config.trials_per_seed}",
                flush=True,
            )
            print(f"SNR: {config.snr_db}", flush=True)
            rows = run_step1(config, output)
            overall, by_snr, gate = _summaries(rows)
            paired = _paired_diagnostics(rows)
            validation = _validate_rows(rows, config)
            if validation["status"] != "PASS":
                raise RuntimeError(f"Step1 validation failed: {validation}")
            rows.to_csv(output / "step1_rows.csv", index=False)
            overall.to_csv(output / "step1_summary_overall.csv", index=False)
            by_snr.to_csv(output / "step1_summary_by_snr.csv", index=False)
            (output / "step1_gate.json").write_text(
                json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (output / "step1_paired_diagnostics.json").write_text(
                json.dumps(paired, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (output / "validation.json").write_text(
                json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            _plot(by_snr, output)
            partial = output / "step1_rows.partial.csv"
            if partial.exists():
                partial.unlink()
            manifest = {
                "experiment": "BRSR Stage B v2 Step1 continuous-rho profile diagnostic",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "config": asdict(config),
                "source_result": str(SOURCE_RESULT_DIR),
                "source_manifest_sha256": _sha256(SOURCE_RESULT_DIR / "manifest.json"),
                "python": sys.version,
                "platform": platform.platform(),
                "elapsed_seconds": time.perf_counter() - started,
                "scope": {
                    "candidate_pool_changed": False,
                    "bit_plan_changed": False,
                    "active_policy_run": False,
                    "new_seeds_disjoint_from_stage_b_v1": True,
                    "primary_method": "LinearA-QuantProfile",
                },
            }
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("\nStep1 gate", flush=True)
            print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
            print("\nPaired diagnostics", flush=True)
            print(json.dumps(paired, indent=2, ensure_ascii=False), flush=True)
            print(f"[Done] elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
