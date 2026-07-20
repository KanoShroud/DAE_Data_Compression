"""BRSR-TDOA Stage B 第一阶段失败机理诊断。

本脚本只读取冻结的 Stage B development 配置和结果，并用相同种子重建
结构化波形。它不修改正式候选池、工作后验、通信协议、策略参数或既有结果。

诊断内容：

1. 线性分数时延与有限窗截取相对理想逐频点相移模型的无噪声残差；
2. 固定已传输量化码下，粗复增益网格、细网格和真值复增益上界的敏感性；
3. 固定工作后验下，停止/动作选择相对同一动作空间事后上界的差距；
4. 预注册少量样本上的 Rollout 蒙特卡洛样本数敏感性；
5. 当前候选池的 Schur-Fisher、全局模糊和频点冗余离线审计。

PyCharm 直接运行时默认执行 ``diagnostic``。Codex 冒烟使用
``--mode smoke``。所有真值辅助结果都明确标记为 oracle/diagnostic，
不得作为可部署方法的正式性能。
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
from typing import TextIO

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
    build_likelihood_context,
)
from brsr_tdoa.brsr_stage_b import (  # noqa: E402
    StageBObservation,
    StageBProtocol,
    ambiguity_period_samples,
    build_stage_b_model,
    evaluate_fixed,
    make_observation,
    run_policy,
)
from brsr_tdoa.brsr_waveform import (  # noqa: E402
    StructuredTrial,
    WaveformConfig,
    apply_circular_fractional_delay,
    candidate_fft_values,
    generate_structured_trial,
    observe_trial,
)


PYCHARM_RUN_MODE = "diagnostic"
SOURCE_RESULT_DIR = (
    ROOT / "运行结果" / "BRSR_STAGE_B_20260718_173940_development"
)


@dataclass(frozen=True)
class DiagnosticConfig:
    mode: str
    model_mismatch_trials_per_seed: int
    selected_trials_per_seed: int
    fine_rho_amplitude_count: int
    fine_rho_phase_count: int
    rollout_snr_db: tuple[float, ...]
    rollout_samples: tuple[int, ...]


CONFIGS = {
    "smoke": DiagnosticConfig(
        mode="smoke",
        model_mismatch_trials_per_seed=1,
        selected_trials_per_seed=1,
        fine_rho_amplitude_count=5,
        fine_rho_phase_count=16,
        rollout_snr_db=(15.0,),
        rollout_samples=(2, 4),
    ),
    "diagnostic": DiagnosticConfig(
        mode="diagnostic",
        model_mismatch_trials_per_seed=20,
        selected_trials_per_seed=1,
        fine_rho_amplitude_count=9,
        fine_rho_phase_count=64,
        rollout_snr_db=(5.0, 10.0, 15.0),
        rollout_samples=(16, 64),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_rng(*parts: object) -> np.random.Generator:
    text = "|".join(str(value) for value in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


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


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_source() -> dict[str, object]:
    required = (
        "manifest.json",
        "development_gate.json",
        "stage_b_trials.csv",
        "candidate_metadata.csv",
        "static_plans.csv",
        "lambda_calibration.csv",
    )
    missing = [name for name in required if not (SOURCE_RESULT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source result is incomplete: {missing}")

    manifest = _load_json(SOURCE_RESULT_DIR / "manifest.json")
    if manifest.get("mode") != "development":
        raise RuntimeError("source result is not the frozen Stage B development run")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("source manifest has no code fingerprints")
    for relative, expected in source_hashes.items():
        path = ROOT / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"frozen source file is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                "frozen Stage B source fingerprint changed: "
                f"{relative}, expected={expected}, actual={actual}"
            )

    return {
        "manifest": manifest,
        "gate": _load_json(SOURCE_RESULT_DIR / "development_gate.json"),
        "rows": pd.read_csv(
            SOURCE_RESULT_DIR / "stage_b_trials.csv",
            encoding="utf-8-sig",
        ),
        "candidate": pd.read_csv(
            SOURCE_RESULT_DIR / "candidate_metadata.csv",
            encoding="utf-8-sig",
        ),
        "static": pd.read_csv(
            SOURCE_RESULT_DIR / "static_plans.csv",
            encoding="utf-8-sig",
        ),
        "lambda": pd.read_csv(
            SOURCE_RESULT_DIR / "lambda_calibration.csv",
            encoding="utf-8-sig",
        ),
    }


def _build_frozen_objects(source: dict[str, object]) -> dict[str, object]:
    manifest = source["manifest"]
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")
    waveform = WaveformConfig(**manifest["waveform"])
    protocol = StageBProtocol(**manifest["protocol"])
    candidate = source["candidate"]
    if not isinstance(candidate, pd.DataFrame):
        raise TypeError("candidate metadata must be a table")
    candidate = candidate.sort_values("position")
    expected_positions = np.arange(len(candidate), dtype=int)
    if not np.array_equal(candidate["position"].to_numpy(dtype=int), expected_positions):
        raise RuntimeError("candidate positions are not contiguous")
    bins = candidate["bin_index"].to_numpy(dtype=int)
    source_power = candidate["development_power"].to_numpy(dtype=float)
    config = manifest["config"]
    model = build_stage_b_model(
        n_time=waveform.n_time,
        bin_indices=bins,
        source_power=source_power,
        base_bins=waveform.base_bins,
        tau_min=waveform.tau_min,
        tau_max=waveform.tau_max,
        tau_step=float(config["tau_step"]),
        rho_amplitudes=tuple(float(value) for value in config["rho_amplitudes"]),
        rho_phase_count=int(config["rho_phase_count"]),
    )
    quantizer = NestedComplexQuantizer(
        max_bits=protocol.max_bits,
        clip_abs=protocol.quantizer_clip_abs,
    )
    return {
        "waveform": waveform,
        "protocol": protocol,
        "bins": bins,
        "source_power": source_power,
        "model": model,
        "quantizer": quantizer,
        "stage_config": config,
    }


def _generate_frozen_trials(
    seeds: tuple[int, ...],
    trials_per_seed: int,
    waveform: WaveformConfig,
) -> dict[tuple[int, int], StructuredTrial]:
    trials: dict[tuple[int, int], StructuredTrial] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for trial_index in range(trials_per_seed):
            trials[(seed, trial_index)] = generate_structured_trial(rng, waveform)
    return trials


def _make_frozen_observation(
    *,
    seed: int,
    trial_index: int,
    snr_db: float,
    trial: StructuredTrial,
    bins: np.ndarray,
    model: BayesianSpectralModel,
    quantizer: NestedComplexQuantizer,
) -> tuple[StageBObservation, np.ndarray, float]:
    y_a, y_b, sigma2 = observe_trial(trial, snr_db)
    y_a_selected = candidate_fft_values(y_a, bins)
    observation = make_observation(
        stage="diagnostic",
        seed=seed,
        trial=trial_index,
        snr_db=snr_db,
        true_tau=trial.true_tau,
        true_rho=trial.true_rho,
        y_a_selected=y_a_selected,
        y_b_selected=candidate_fft_values(y_b, bins),
        sigma2=sigma2,
        model=model,
        quantizer=quantizer,
    )
    return observation, y_a_selected, sigma2


def _model_mismatch_rows(
    trials: dict[tuple[int, int], StructuredTrial],
    *,
    waveform: WaveformConfig,
    bins: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    omega = 2.0 * np.pi * bins / waveform.n_time
    trial_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    for (seed, trial_index), trial in trials.items():
        spectrum_a = candidate_fft_values(trial.source_a, bins)
        spectrum_b = candidate_fft_values(trial.source_b_unit_gain, bins)
        ideal = spectrum_a * np.exp(-1j * omega * trial.true_tau)
        circular = candidate_fft_values(
            apply_circular_fractional_delay(trial.source_a, trial.true_tau),
            bins,
        )
        denominator = max(float(np.sum(np.abs(spectrum_b) ** 2)), 1e-12)
        ideal_denominator = max(float(np.sum(np.abs(ideal) ** 2)), 1e-12)
        profiled_gain = np.vdot(ideal, spectrum_b) / ideal_denominator
        linear_nmse = float(np.sum(np.abs(spectrum_b - ideal) ** 2) / denominator)
        profiled_nmse = float(
            np.sum(np.abs(spectrum_b - profiled_gain * ideal) ** 2) / denominator
        )
        circular_nmse = float(
            np.sum(np.abs(circular - ideal) ** 2) / ideal_denominator
        )
        edge = waveform.fractional_delay_half_len + int(
            math.ceil(abs(trial.true_tau))
        )
        edge_energy = float(
            np.sum(np.abs(trial.source_a[:edge]) ** 2)
            + np.sum(np.abs(trial.source_a[-edge:]) ** 2)
        )
        total_energy = max(float(np.sum(np.abs(trial.source_a) ** 2)), 1e-12)
        phase_error = np.angle(spectrum_b * np.conj(ideal))
        trial_rows.append(
            {
                "seed": seed,
                "trial": trial_index,
                "true_tau": trial.true_tau,
                "abs_tau": abs(trial.true_tau),
                "tau_region": "boundary" if abs(trial.true_tau) > 6.0 else "interior",
                "linear_model_nmse": linear_nmse,
                "profiled_scalar_nmse": profiled_nmse,
                "circular_control_nmse": circular_nmse,
                "profiled_gain_real": float(profiled_gain.real),
                "profiled_gain_imag": float(profiled_gain.imag),
                "profiled_gain_abs_error": abs(abs(profiled_gain) - 1.0),
                "phase_rmse_rad": float(np.sqrt(np.mean(phase_error**2))),
                "edge_energy_fraction": edge_energy / total_energy,
            }
        )
        for position, bin_index in enumerate(bins):
            scale = max(
                float(abs(spectrum_b[position]) ** 2 + abs(ideal[position]) ** 2),
                1e-12,
            )
            bin_rows.append(
                {
                    "seed": seed,
                    "trial": trial_index,
                    "true_tau": trial.true_tau,
                    "tau_region": (
                        "boundary" if abs(trial.true_tau) > 6.0 else "interior"
                    ),
                    "position": position,
                    "bin_index": int(bin_index),
                    "frequency_mhz": (
                        bin_index * waveform.fs_hz / waveform.n_time / 1e6
                    ),
                    "symmetric_residual": (
                        float(abs(spectrum_b[position] - ideal[position]) ** 2)
                        / scale
                    ),
                    "phase_error_rad": float(phase_error[position]),
                    "amplitude_ratio": float(
                        abs(spectrum_b[position])
                        / max(abs(ideal[position]), 1e-12)
                    ),
                }
            )
    return pd.DataFrame(trial_rows), pd.DataFrame(bin_rows)


def _candidate_set_metrics(
    label: str,
    levels: tuple[int, ...],
    *,
    model: BayesianSpectralModel,
    snr_db: float | None,
) -> dict[str, object]:
    selected = np.flatnonzero(np.asarray(levels, dtype=int) > 0)
    if selected.size < 2:
        raise ValueError("candidate audit requires at least two selected bins")
    omega = model.omega[selected]
    weight = model.source_power[selected]
    weight_sum = float(np.sum(weight))
    weighted_mean = float(np.sum(weight * omega) / weight_sum)
    projected_information = float(
        np.sum(weight * (omega - weighted_mean) ** 2)
    )

    delay_difference = np.linspace(-16.0, 16.0, 8193)
    ambiguity = np.abs(
        np.sum(
            weight[:, None]
            * np.exp(-1j * omega[:, None] * delay_difference[None, :]),
            axis=0,
        )
    ) / weight_sum
    row = {
        "set": label,
        "snr_db": snr_db,
        "selected_count": int(selected.size),
        "payload_bits": int(2 * sum(levels)),
        "selected_positions": " ".join(str(value) for value in selected),
        "selected_bins": " ".join(str(value) for value in model.bin_indices[selected]),
        "unknown_gain_projected_information_proxy": projected_information,
        "ambiguity_period_samples": ambiguity_period_samples(levels, model),
    }
    for threshold in (1, 2, 4):
        global_mask = np.abs(delay_difference) >= float(threshold)
        global_values = ambiguity[global_mask]
        global_delays = delay_difference[global_mask]
        maximum_index = int(np.argmax(global_values))
        maximum = float(global_values[maximum_index])
        row[f"max_ambiguity_abs_ge_{threshold}"] = maximum
        row[f"max_ambiguity_delay_ge_{threshold}"] = float(
            global_delays[maximum_index]
        )
        row[f"min_projected_distance_ge_{threshold}"] = float(
            max(0.0, 1.0 - maximum**2)
        )
    return row


def _parse_levels(text: object, n_bins: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in str(text).split())
    if len(values) != n_bins or any(value not in (0, 4, 8) for value in values):
        raise ValueError(f"invalid selected levels: {text}")
    return values


def _candidate_pool_audit(
    source: dict[str, object],
    *,
    model: BayesianSpectralModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    static = source["static"]
    candidate = source["candidate"]
    if not isinstance(static, pd.DataFrame) or not isinstance(candidate, pd.DataFrame):
        raise TypeError("source candidate/static tables are invalid")

    sets: list[tuple[str, float | None, tuple[int, ...]]] = []
    base = tuple(4 if index in model.base_positions else 0 for index in range(model.n_bins))
    sets.append(("Base-4bit", None, base))
    sets.append(("All-4bit", None, (4,) * model.n_bins))

    best = static[
        (static["method"] == "Best-Static") & (static["target_bits"] == 32)
    ]
    if len(best) != 1:
        raise RuntimeError("frozen Best-Static-T32 plan is missing or duplicated")
    sets.append(("Best-Static-T32", None, _parse_levels(best.iloc[0]["levels"], model.n_bins)))
    snr_only = static[
        (static["method"] == "SNR-only") & (static["target_bits"] == 32)
    ]
    for row in snr_only.itertuples(index=False):
        sets.append(
            (
                f"SNR-only-T32-{float(row.snr_db):g}dB",
                float(row.snr_db),
                _parse_levels(row.levels, model.n_bins),
            )
        )

    set_rows = [
        _candidate_set_metrics(label, levels, model=model, snr_db=snr)
        for label, snr, levels in sets
    ]
    all_levels = (4,) * model.n_bins
    all_metric = _candidate_set_metrics(
        "All-4bit", all_levels, model=model, snr_db=None
    )
    all_information = float(
        all_metric["unknown_gain_projected_information_proxy"]
    )
    for row in set_rows:
        row["projected_information_ratio_to_all"] = (
            float(row["unknown_gain_projected_information_proxy"])
            / max(all_information, 1e-12)
        )

    full_weight = model.source_power
    full_mean = float(np.sum(full_weight * model.omega) / np.sum(full_weight))
    full_fisher = float(np.sum(full_weight * (model.omega - full_mean) ** 2))
    bin_rows = []
    for position in range(model.n_bins):
        keep = np.ones(model.n_bins, dtype=bool)
        keep[position] = False
        weight = model.source_power[keep]
        omega = model.omega[keep]
        mean = float(np.sum(weight * omega) / np.sum(weight))
        fisher_without = float(np.sum(weight * (omega - mean) ** 2))
        nearest_spacing = int(
            np.min(
                np.abs(
                    model.bin_indices[position]
                    - np.delete(model.bin_indices, position)
                )
            )
        )
        bin_rows.append(
            {
                "position": position,
                "bin_index": int(model.bin_indices[position]),
                "development_power": float(model.source_power[position]),
                "local_power_omega2": float(
                    model.source_power[position] * model.omega[position] ** 2
                ),
                "leave_one_out_projected_information_loss": (
                    full_fisher - fisher_without
                ),
                "leave_one_out_projected_information_loss_ratio": (
                    (full_fisher - fisher_without) / max(full_fisher, 1e-12)
                ),
                "nearest_bin_spacing": nearest_spacing,
                "is_base": bool(candidate.iloc[position]["is_base"]),
            }
        )
    return pd.DataFrame(set_rows), pd.DataFrame(bin_rows)


def _alternate_model(
    formal_model: BayesianSpectralModel,
    rho_values: np.ndarray,
) -> BayesianSpectralModel:
    rho_values = np.asarray(rho_values, dtype=np.complex128)
    if rho_values.ndim != 1 or rho_values.size < 2:
        raise ValueError("alternate rho grid must contain at least two values")
    model = replace(
        formal_model,
        rho_values=rho_values,
        rho_prior=np.full(rho_values.size, 1.0 / rho_values.size),
    )
    model.validate()
    return model


def _observation_for_model(
    formal: StageBObservation,
    *,
    y_a_selected: np.ndarray,
    sigma2: float,
    model: BayesianSpectralModel,
) -> StageBObservation:
    context = build_likelihood_context(
        y_a_selected,
        model,
        sigma2,
        sigma2,
        b_public_scale=formal.public_b_scale,
    )
    return replace(formal, context=context)


def _selected_keys(
    source_rows: pd.DataFrame,
    *,
    stage_config: dict[str, object],
    selected_trials_per_seed: int,
    mode: str,
) -> pd.DataFrame:
    methods = ("BRSR-Greedy-T32", "BRSR-Rollout-T32")
    rows = source_rows[source_rows["method"].isin(methods)].copy()
    evaluation_seeds = tuple(int(value) for value in stage_config["evaluation_seeds"])
    if mode == "smoke":
        allowed_snr = (-5.0, 15.0)
        evaluation_seeds = evaluation_seeds[:1]
    else:
        allowed_snr = tuple(float(value) for value in stage_config["snr_db"])
    selected = rows[
        rows["seed"].isin(evaluation_seeds)
        & rows["snr_db"].isin(allowed_snr)
        & (rows["trial"] < selected_trials_per_seed)
    ].copy()
    selected["selection_role"] = "pre_registered"

    if mode == "diagnostic":
        failure = rows[
            (rows["method"] == "BRSR-Rollout-T32")
            & (rows["snr_db"] == 15.0)
        ].nlargest(1, "squared_error_samples2")
        failure_key = failure[["seed", "trial", "snr_db"]]
        case = rows.merge(failure_key, on=["seed", "trial", "snr_db"], how="inner")
        case = case[
            ~case.set_index(["seed", "trial", "snr_db", "method"]).index.isin(
                selected.set_index(["seed", "trial", "snr_db", "method"]).index
            )
        ].copy()
        case["selection_role"] = "known_failure_case"
        selected = pd.concat([selected, case], ignore_index=True)

    expected = selected.groupby(
        ["seed", "trial", "snr_db", "method"],
        observed=True,
    ).size()
    if expected.empty or int(expected.max()) != 1:
        raise RuntimeError("diagnostic source rows are missing or duplicated")
    return selected.sort_values(["selection_role", "snr_db", "seed", "trial", "method"])


def _rho_grid_sensitivity(
    selected_rows: pd.DataFrame,
    trials: dict[tuple[int, int], StructuredTrial],
    *,
    config: DiagnosticConfig,
    bins: np.ndarray,
    formal_model: BayesianSpectralModel,
    quantizer: NestedComplexQuantizer,
) -> pd.DataFrame:
    amplitudes = np.linspace(0.70, 0.90, config.fine_rho_amplitude_count)
    phases = np.linspace(-np.pi, np.pi, config.fine_rho_phase_count, endpoint=False)
    fine_values = np.asarray(
        [amplitude * np.exp(1j * phase) for amplitude in amplitudes for phase in phases],
        dtype=np.complex128,
    )
    fine_model = _alternate_model(formal_model, fine_values)
    rows: list[dict[str, object]] = []
    grouped = selected_rows.groupby(["seed", "trial", "snr_db"], sort=False)
    for index, ((seed, trial_index, snr_db), group) in enumerate(grouped, start=1):
        trial = trials[(int(seed), int(trial_index))]
        formal, y_a_selected, sigma2 = _make_frozen_observation(
            seed=int(seed),
            trial_index=int(trial_index),
            snr_db=float(snr_db),
            trial=trial,
            bins=bins,
            model=formal_model,
            quantizer=quantizer,
        )
        fine_observation = _observation_for_model(
            formal,
            y_a_selected=y_a_selected,
            sigma2=sigma2,
            model=fine_model,
        )
        epsilon = 1e-12
        oracle_values = np.asarray(
            [
                trial.true_rho * (1.0 - epsilon),
                trial.true_rho * (1.0 + epsilon),
            ],
            dtype=np.complex128,
        )
        oracle_model = _alternate_model(formal_model, oracle_values)
        oracle_observation = _observation_for_model(
            formal,
            y_a_selected=y_a_selected,
            sigma2=sigma2,
            model=oracle_model,
        )
        for source_row in group.itertuples(index=False):
            levels = _parse_levels(source_row.selected_levels, formal_model.n_bins)
            model_cases = (
                ("formal-coarse-rho", formal_model, formal),
                ("fine-rho-grid", fine_model, fine_observation),
                ("oracle-true-rho", oracle_model, oracle_observation),
            )
            for label, model, observation in model_cases:
                result = evaluate_fixed(
                    observation,
                    method=label,
                    levels=levels,
                    quantizer=quantizer,
                    model=model,
                )
                error = result.estimate - trial.true_tau
                rows.append(
                    {
                        "selection_role": source_row.selection_role,
                        "seed": int(seed),
                        "trial": int(trial_index),
                        "snr_db": float(snr_db),
                        "source_method": source_row.method,
                        "rho_model": label,
                        "n_rho": model.n_rho,
                        "true_tau": trial.true_tau,
                        "estimate_tau": result.estimate,
                        "error_samples": error,
                        "squared_error_samples2": error**2,
                        "posterior_risk_samples2": result.posterior_risk,
                        "credible90_contains_true": result.credible90_contains_true,
                        "source_estimate_tau": float(source_row.estimate_tau),
                        "formal_alignment_abs": (
                            abs(result.estimate - float(source_row.estimate_tau))
                            if label == "formal-coarse-rho"
                            else np.nan
                        ),
                        "nearest_rho_grid_distance": float(
                            np.min(np.abs(model.rho_values - trial.true_rho))
                        ),
                    }
                )
        print(
            f"[Rho] {index}/{len(grouped)} seed={seed} trial={trial_index} "
            f"SNR={snr_db:g}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    maximum_alignment = float(table["formal_alignment_abs"].max(skipna=True))
    if maximum_alignment > 1e-10:
        raise RuntimeError(
            "reconstructed formal posterior does not align with frozen Stage B: "
            f"max diff={maximum_alignment:.6g}"
        )
    return table


def _reachable_level_configs(
    model: BayesianSpectralModel,
    *,
    coarse_bits: int,
    max_bits: int,
    max_decisions: int,
) -> dict[tuple[int, ...], int]:
    initial = tuple(
        coarse_bits if index in model.base_positions else 0
        for index in range(model.n_bins)
    )
    depths = {initial: 0}
    frontier = {initial}
    for depth in range(1, max_decisions + 1):
        next_frontier: set[tuple[int, ...]] = set()
        for levels in frontier:
            for position, current in enumerate(levels):
                if current == 0:
                    target = coarse_bits
                elif current < max_bits:
                    target = max_bits
                else:
                    continue
                updated = list(levels)
                updated[position] = target
                item = tuple(updated)
                if item not in depths:
                    depths[item] = depth
                    next_frontier.add(item)
        frontier = next_frontier
    return depths


def _policy_counterfactual(
    selected_rows: pd.DataFrame,
    trials: dict[tuple[int, int], StructuredTrial],
    *,
    bins: np.ndarray,
    model: BayesianSpectralModel,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
) -> pd.DataFrame:
    configurations = _reachable_level_configs(
        model,
        coarse_bits=protocol.coarse_bits,
        max_bits=protocol.max_bits,
        max_decisions=protocol.max_decisions,
    )
    rows: list[dict[str, object]] = []
    grouped = selected_rows.groupby(["seed", "trial", "snr_db"], sort=False)
    for index, ((seed, trial_index, snr_db), group) in enumerate(grouped, start=1):
        trial = trials[(int(seed), int(trial_index))]
        observation, _, _ = _make_frozen_observation(
            seed=int(seed),
            trial_index=int(trial_index),
            snr_db=float(snr_db),
            trial=trial,
            bins=bins,
            model=model,
            quantizer=quantizer,
        )
        evaluated: dict[tuple[int, ...], tuple[float, float]] = {}
        for levels in configurations:
            result = evaluate_fixed(
                observation,
                method="oracle-action-diagnostic",
                levels=levels,
                quantizer=quantizer,
                model=model,
            )
            error = result.estimate - trial.true_tau
            evaluated[levels] = (result.estimate, error**2)

        for source_row in group.itertuples(index=False):
            current_levels = _parse_levels(source_row.selected_levels, model.n_bins)
            if current_levels not in configurations:
                raise RuntimeError(
                    "frozen policy selected a state outside the two-decision action space"
                )
            current_depth = configurations[current_levels]
            if current_depth != int(source_row.n_actions):
                raise RuntimeError("frozen action count does not match selected levels")
            current_estimate, current_error = evaluated[current_levels]
            if abs(current_estimate - float(source_row.estimate_tau)) > 1e-10:
                raise RuntimeError("counterfactual reconstruction does not align")
            same_depth = [
                (levels, values)
                for levels, values in evaluated.items()
                if configurations[levels] == current_depth
            ]
            full_depth = list(evaluated.items())
            best_same = min(same_depth, key=lambda item: item[1][1])
            best_full = min(full_depth, key=lambda item: item[1][1])
            rows.append(
                {
                    "selection_role": source_row.selection_role,
                    "seed": int(seed),
                    "trial": int(trial_index),
                    "snr_db": float(snr_db),
                    "method": source_row.method,
                    "true_tau": trial.true_tau,
                    "n_actions": current_depth,
                    "stopped_by_policy": bool(source_row.stopped_by_policy),
                    "current_squared_error": current_error,
                    "oracle_same_depth_squared_error": best_same[1][1],
                    "oracle_full_depth_squared_error": best_full[1][1],
                    "action_selection_regret": max(
                        current_error - best_same[1][1], 0.0
                    ),
                    "remaining_information_regret": max(
                        current_error - best_full[1][1], 0.0
                    ),
                    "oracle_same_depth_levels": " ".join(
                        str(value) for value in best_same[0]
                    ),
                    "oracle_full_depth_levels": " ".join(
                        str(value) for value in best_full[0]
                    ),
                    "oracle_full_depth": configurations[best_full[0]],
                }
            )
        print(
            f"[OracleAction] {index}/{len(grouped)} seed={seed} "
            f"trial={trial_index} SNR={snr_db:g}",
            flush=True,
        )
    return pd.DataFrame(rows)


def _selected_lambda(source: dict[str, object], method: str, target: int) -> float:
    table = source["lambda"]
    if not isinstance(table, pd.DataFrame):
        raise TypeError("lambda table is invalid")
    selected = table[
        (table["method"] == method)
        & (table["target_bits"] == target)
        & table["selected"].astype(bool)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"missing frozen lambda for {method}-T{target}")
    return float(selected.iloc[0]["lambda_per_bit"])


def _rollout_stability(
    source: dict[str, object],
    trials: dict[tuple[int, int], StructuredTrial],
    *,
    config: DiagnosticConfig,
    stage_config: dict[str, object],
    bins: np.ndarray,
    model: BayesianSpectralModel,
    protocol: StageBProtocol,
    quantizer: NestedComplexQuantizer,
) -> pd.DataFrame:
    source_rows = source["rows"]
    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("source rows are invalid")
    evaluation_seed = int(stage_config["evaluation_seeds"][0])
    lambda_rollout = _selected_lambda(source, "rollout", 32)
    lambda_greedy = _selected_lambda(source, "greedy", 32)
    rows: list[dict[str, object]] = []
    keys = [(evaluation_seed, 0, snr) for snr in config.rollout_snr_db]
    if config.mode == "diagnostic":
        failure = source_rows[
            (source_rows["method"] == "BRSR-Rollout-T32")
            & (source_rows["snr_db"] == 15.0)
        ].nlargest(1, "squared_error_samples2")
        failure_key = (
            int(failure.iloc[0]["seed"]),
            int(failure.iloc[0]["trial"]),
            float(failure.iloc[0]["snr_db"]),
        )
        if failure_key not in keys:
            keys.append(failure_key)

    for key_index, (seed, trial_index, snr_db) in enumerate(keys, start=1):
        trial = trials[(seed, trial_index)]
        observation, _, _ = _make_frozen_observation(
            seed=seed,
            trial_index=trial_index,
            snr_db=snr_db,
            trial=trial,
            bins=bins,
            model=model,
            quantizer=quantizer,
        )
        greedy = run_policy(
            observation,
            method="greedy",
            lambda_per_bit=lambda_greedy,
            protocol=protocol,
            quantizer=quantizer,
            model=model,
            rng=_policy_rng(observation, "greedy", lambda_greedy),
        )
        rows.append(
            {
                "seed": seed,
                "trial": trial_index,
                "snr_db": snr_db,
                "method": "greedy",
                "rollout_samples": 0,
                "first_action": (
                    greedy.action_trace[0]["action"] if greedy.action_trace else "none"
                ),
                "estimate_tau": greedy.estimate,
                "squared_error_samples2": (greedy.estimate - trial.true_tau) ** 2,
                "total_bits": greedy.total_bits,
                "source_alignment_abs": np.nan,
            }
        )
        for sample_count in config.rollout_samples:
            diagnostic_protocol = replace(protocol, rollout_samples=sample_count)
            start = time.perf_counter()
            result = run_policy(
                observation,
                method="rollout",
                lambda_per_bit=lambda_rollout,
                protocol=diagnostic_protocol,
                quantizer=quantizer,
                model=model,
                rng=_policy_rng(observation, "rollout", lambda_rollout),
            )
            source_match = source_rows[
                (source_rows["method"] == "BRSR-Rollout-T32")
                & (source_rows["seed"] == seed)
                & (source_rows["trial"] == trial_index)
                & (source_rows["snr_db"] == snr_db)
            ]
            alignment = np.nan
            if sample_count == protocol.rollout_samples:
                if len(source_match) != 1:
                    raise RuntimeError("frozen Rollout-T32 row is missing")
                alignment = abs(
                    result.estimate - float(source_match.iloc[0]["estimate_tau"])
                )
                if alignment > 1e-10:
                    raise RuntimeError(
                        "Rollout Q16 reconstruction does not align with frozen result"
                    )
            rows.append(
                {
                    "seed": seed,
                    "trial": trial_index,
                    "snr_db": snr_db,
                    "method": "rollout",
                    "rollout_samples": sample_count,
                    "first_action": (
                        result.action_trace[0]["action"]
                        if result.action_trace
                        else "none"
                    ),
                    "estimate_tau": result.estimate,
                    "squared_error_samples2": (
                        result.estimate - trial.true_tau
                    )
                    ** 2,
                    "total_bits": result.total_bits,
                    "source_alignment_abs": alignment,
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
        print(
            f"[Rollout] {key_index}/{len(keys)} seed={seed} "
            f"trial={trial_index} SNR={snr_db:g}",
            flush=True,
        )
    return pd.DataFrame(rows)


def _aggregate_summary(
    mismatch: pd.DataFrame,
    rho: pd.DataFrame,
    counterfactual: pd.DataFrame,
    rollout: pd.DataFrame,
    candidate_sets: pd.DataFrame,
) -> dict[str, object]:
    prereg_rho = rho[rho["selection_role"] == "pre_registered"]
    prereg_counterfactual = counterfactual[
        counterfactual["selection_role"] == "pre_registered"
    ]
    rho_summary = {}
    for (method, rho_model), group in prereg_rho.groupby(
        ["source_method", "rho_model"],
        observed=True,
    ):
        rho_summary[f"{method}|{rho_model}"] = {
            "rmse_samples": float(
                np.sqrt(np.mean(group["squared_error_samples2"]))
            ),
            "coverage90": float(group["credible90_contains_true"].mean()),
            "risk_to_mse": float(
                group["posterior_risk_samples2"].mean()
                / max(group["squared_error_samples2"].mean(), 1e-12)
            ),
        }

    counterfactual_summary = {}
    for method, group in prereg_counterfactual.groupby("method", observed=True):
        counterfactual_summary[str(method)] = {
            "mean_action_selection_regret": float(
                group["action_selection_regret"].mean()
            ),
            "mean_remaining_information_regret": float(
                group["remaining_information_regret"].mean()
            ),
            "fraction_with_recoverable_error": float(
                (group["remaining_information_regret"] > 1e-12).mean()
            ),
        }

    rollout_summary = {}
    rollout_only = rollout[rollout["method"] == "rollout"]
    if not rollout_only.empty:
        pivot = rollout_only.pivot_table(
            index=["seed", "trial", "snr_db"],
            columns="rollout_samples",
            values="first_action",
            aggfunc="first",
        )
        samples = sorted(int(value) for value in rollout_only["rollout_samples"].unique())
        if len(samples) >= 2:
            rollout_summary["first_action_agreement_min_vs_max"] = float(
                (pivot[samples[0]] == pivot[samples[-1]]).mean()
            )

    summary = {
        "status": "DIAGNOSTIC_ONLY_NO_POLICY_CHANGE",
        "model_mismatch": {
            "median_linear_nmse": float(mismatch["linear_model_nmse"].median()),
            "median_profiled_nmse": float(mismatch["profiled_scalar_nmse"].median()),
            "max_circular_control_nmse": float(
                mismatch["circular_control_nmse"].max()
            ),
            "interior_mean_linear_nmse": float(
                mismatch.loc[
                    mismatch["tau_region"] == "interior", "linear_model_nmse"
                ].mean()
            ),
            "boundary_mean_linear_nmse": float(
                mismatch.loc[
                    mismatch["tau_region"] == "boundary", "linear_model_nmse"
                ].mean()
            ),
        },
        "rho_grid_sensitivity": rho_summary,
        "policy_counterfactual": counterfactual_summary,
        "rollout_stability": rollout_summary,
        "candidate_set_audit": candidate_sets.to_dict(orient="records"),
        "interpretation_boundary": (
            "Oracle-rho and oracle-action use truth only as diagnostic upper bounds. "
            "Candidate metrics are unquantized projected-information proxies and "
            "offline ambiguity audits, not exact Schur-Fisher values or replacement "
            "pools."
        ),
    }
    return _json_safe(summary)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _plot_diagnostics(
    output: Path,
    mismatch: pd.DataFrame,
    mismatch_bins: pd.DataFrame,
    rho: pd.DataFrame,
    counterfactual: pd.DataFrame,
    candidate_bins: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes[0, 0].scatter(
        mismatch["abs_tau"],
        mismatch["linear_model_nmse"],
        s=24,
        alpha=0.75,
        label="Linear window",
        color="#1f77b4",
    )
    axes[0, 0].scatter(
        mismatch["abs_tau"],
        mismatch["profiled_scalar_nmse"],
        s=20,
        alpha=0.65,
        label="After scalar profiling",
        color="#d62728",
    )
    axes[0, 0].set_xlabel(r"$|\tau|$ (samples)")
    axes[0, 0].set_ylabel("Noiseless spectral NMSE")
    axes[0, 0].set_title("(a) Finite-window model mismatch")
    axes[0, 0].legend(frameon=False, fontsize=8)

    by_bin = mismatch_bins.groupby(
        ["frequency_mhz", "tau_region"],
        observed=True,
    )["symmetric_residual"].median().reset_index()
    for region, color in (("interior", "#2ca02c"), ("boundary", "#9467bd")):
        data = by_bin[by_bin["tau_region"] == region]
        axes[0, 1].plot(
            data["frequency_mhz"],
            data["symmetric_residual"],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=region,
            color=color,
        )
    axes[0, 1].set_xlabel("Candidate frequency (MHz)")
    axes[0, 1].set_ylabel("Median symmetric residual")
    axes[0, 1].set_title("(b) Mismatch across candidate bins")
    axes[0, 1].legend(frameon=False, fontsize=8)

    prereg = rho[rho["selection_role"] == "pre_registered"]
    rho_summary = (
        prereg.groupby(["source_method", "rho_model"], observed=True)
        ["squared_error_samples2"]
        .mean()
        .pow(0.5)
        .reset_index(name="rmse")
    )
    labels = sorted(rho_summary["source_method"].unique())
    models = ("formal-coarse-rho", "fine-rho-grid", "oracle-true-rho")
    x = np.arange(len(labels))
    width = 0.24
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for offset, (model_name, color) in enumerate(zip(models, colors, strict=True)):
        values = [
            float(
                rho_summary[
                    (rho_summary["source_method"] == label)
                    & (rho_summary["rho_model"] == model_name)
                ]["rmse"].iloc[0]
            )
            for label in labels
        ]
        axes[1, 0].bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=model_name,
            color=color,
        )
    axes[1, 0].set_xticks(x, [label.replace("BRSR-", "") for label in labels])
    axes[1, 0].set_ylabel("TDOA RMSE (samples)")
    axes[1, 0].set_title("(c) Fixed-code nuisance-grid sensitivity")
    axes[1, 0].legend(frameon=False, fontsize=7)

    prereg_cf = counterfactual[
        counterfactual["selection_role"] == "pre_registered"
    ]
    cf = (
        prereg_cf.groupby(["method", "snr_db"], observed=True)
        ["remaining_information_regret"]
        .mean()
        .reset_index()
    )
    for method, color in (
        ("BRSR-Greedy-T32", "#17becf"),
        ("BRSR-Rollout-T32", "#e45756"),
    ):
        data = cf[cf["method"] == method]
        axes[1, 1].plot(
            data["snr_db"],
            data["remaining_information_regret"],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=method.replace("BRSR-", ""),
            color=color,
        )
    axes[1, 1].set_xlabel("SNR (dB)")
    axes[1, 1].set_ylabel(r"Oracle recoverable squared error (sample$^2$)")
    axes[1, 1].set_title("(d) Stop/action upper-bound gap")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "FigD1_StageB_Failure_Mechanisms.svg", format="svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    axes[0].bar(
        candidate_bins["bin_index"].astype(str),
        candidate_bins["leave_one_out_projected_information_loss_ratio"],
        color="#4c78a8",
    )
    axes[0].set_xlabel("Candidate FFT bin")
    axes[0].set_ylabel("Leave-one-out projected-information loss ratio")
    axes[0].set_title("(a) Projected local-information contribution")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].scatter(
        candidate_bins["local_power_omega2"],
        candidate_bins["leave_one_out_projected_information_loss_ratio"],
        c=candidate_bins["bin_index"],
        cmap="viridis",
        s=42,
    )
    axes[1].set_xlabel(r"Simplified $P(k)\omega_k^2$")
    axes[1].set_ylabel("Projected contribution")
    axes[1].set_title("(b) Simplified vs nuisance-projected score")
    fig.savefig(output / "FigD2_Candidate_Pool_Audit.svg", format="svg")
    plt.close(fig)


def _write_outputs(
    output: Path,
    *,
    source: dict[str, object],
    config: DiagnosticConfig,
    mismatch: pd.DataFrame,
    mismatch_bins: pd.DataFrame,
    rho: pd.DataFrame,
    counterfactual: pd.DataFrame,
    rollout: pd.DataFrame,
    candidate_sets: pd.DataFrame,
    candidate_bins: pd.DataFrame,
    summary: dict[str, object],
    elapsed_seconds: float,
) -> None:
    tables = {
        "model_mismatch_by_trial.csv": mismatch,
        "model_mismatch_by_bin.csv": mismatch_bins,
        "rho_grid_sensitivity.csv": rho,
        "policy_counterfactual.csv": counterfactual,
        "rollout_stability.csv": rollout,
        "candidate_set_audit.csv": candidate_sets,
        "candidate_bin_audit.csv": candidate_bins,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, encoding="utf-8-sig")
    with (output / "diagnostic_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    manifest = {
        "experiment": "BRSR-TDOA Stage B failure diagnosis",
        "mode": config.mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_result": str(SOURCE_RESULT_DIR),
        "source_gate_status": source["gate"]["status"],
        "diagnostic_config": asdict(config),
        "elapsed_seconds": elapsed_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "assumptions": {
            "formal_stage_b_algorithm_modified": False,
            "source_result_modified": False,
            "oracle_rho_uses_true_value": True,
            "oracle_action_uses_true_tau": True,
            "oracle_outputs_are_deployable_methods": False,
            "candidate_audit_is_unquantized_information_proxy": True,
            "candidate_pool_reselected": False,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                ROOT / "brsr_tdoa" / "brsr_core.py",
                ROOT / "brsr_tdoa" / "brsr_stage_b.py",
                ROOT / "brsr_tdoa" / "brsr_waveform.py",
                ROOT / "brsr_tdoa" / "run_stage_b.py",
            )
        },
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def run(config: DiagnosticConfig, output: Path) -> dict[str, object]:
    source = _load_source()
    frozen = _build_frozen_objects(source)
    waveform = frozen["waveform"]
    protocol = frozen["protocol"]
    bins = frozen["bins"]
    model = frozen["model"]
    quantizer = frozen["quantizer"]
    stage_config = frozen["stage_config"]
    if not isinstance(waveform, WaveformConfig):
        raise TypeError("waveform reconstruction failed")
    if not isinstance(protocol, StageBProtocol):
        raise TypeError("protocol reconstruction failed")
    if not isinstance(bins, np.ndarray):
        raise TypeError("candidate reconstruction failed")
    if not isinstance(model, BayesianSpectralModel):
        raise TypeError("model reconstruction failed")
    if not isinstance(quantizer, NestedComplexQuantizer):
        raise TypeError("quantizer reconstruction failed")
    if not isinstance(stage_config, dict):
        raise TypeError("Stage B config reconstruction failed")

    evaluation_seeds = tuple(
        int(value) for value in stage_config["evaluation_seeds"]
    )
    all_trials = _generate_frozen_trials(
        evaluation_seeds,
        int(stage_config["evaluation_trials_per_seed"]),
        waveform,
    )
    mismatch_trials = {
        key: value
        for key, value in all_trials.items()
        if key[1] < config.model_mismatch_trials_per_seed
    }
    print(
        f"Source: {SOURCE_RESULT_DIR}\n"
        f"Mode: {config.mode}\n"
        f"Mismatch trials: {len(mismatch_trials)}\n"
        f"Formal candidates: {bins.tolist()}\n"
        "No Stage B algorithm or source result will be modified.",
        flush=True,
    )

    mismatch, mismatch_bins = _model_mismatch_rows(
        mismatch_trials,
        waveform=waveform,
        bins=bins,
    )
    mismatch.to_csv(
        output / "model_mismatch_by_trial.partial.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_sets, candidate_bins = _candidate_pool_audit(source, model=model)
    candidate_sets.to_csv(
        output / "candidate_set_audit.partial.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_rows = source["rows"]
    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("source rows are invalid")
    selected_rows = _selected_keys(
        source_rows,
        stage_config=stage_config,
        selected_trials_per_seed=config.selected_trials_per_seed,
        mode=config.mode,
    )
    rho = _rho_grid_sensitivity(
        selected_rows,
        all_trials,
        config=config,
        bins=bins,
        formal_model=model,
        quantizer=quantizer,
    )
    rho.to_csv(
        output / "rho_grid_sensitivity.partial.csv",
        index=False,
        encoding="utf-8-sig",
    )
    counterfactual = _policy_counterfactual(
        selected_rows,
        all_trials,
        bins=bins,
        model=model,
        protocol=protocol,
        quantizer=quantizer,
    )
    counterfactual.to_csv(
        output / "policy_counterfactual.partial.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rollout = _rollout_stability(
        source,
        all_trials,
        config=config,
        stage_config=stage_config,
        bins=bins,
        model=model,
        protocol=protocol,
        quantizer=quantizer,
    )
    rollout.to_csv(
        output / "rollout_stability.partial.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = _aggregate_summary(
        mismatch,
        rho,
        counterfactual,
        rollout,
        candidate_sets,
    )
    _plot_diagnostics(
        output,
        mismatch,
        mismatch_bins,
        rho,
        counterfactual,
        candidate_bins,
    )
    return {
        "source": source,
        "mismatch": mismatch,
        "mismatch_bins": mismatch_bins,
        "rho": rho,
        "counterfactual": counterfactual,
        "rollout": rollout,
        "candidate_sets": candidate_sets,
        "candidate_bins": candidate_bins,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(CONFIGS), default=PYCHARM_RUN_MODE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        ROOT
        / "运行结果"
        / f"BRSR_STAGE_B_DIAG_{timestamp}_{config.mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with (output / "run.log").open(
        "w", encoding="utf-8", buffering=1
    ) as log_handle:
        with redirect_stdout(Tee(sys.stdout, log_handle)), redirect_stderr(
            Tee(sys.stderr, log_handle)
        ):
            print(f"Output: {output}", flush=True)
            result = run(config, output)
            elapsed = time.perf_counter() - start
            _write_outputs(
                output,
                source=result["source"],
                config=config,
                mismatch=result["mismatch"],
                mismatch_bins=result["mismatch_bins"],
                rho=result["rho"],
                counterfactual=result["counterfactual"],
                rollout=result["rollout"],
                candidate_sets=result["candidate_sets"],
                candidate_bins=result["candidate_bins"],
                summary=result["summary"],
                elapsed_seconds=elapsed,
            )
            for partial in output.glob("*.partial.csv"):
                partial.unlink()
            print(
                "\nDiagnostic summary\n"
                + json.dumps(result["summary"], ensure_ascii=False, indent=2),
                flush=True,
            )
            print(f"[Done] elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
