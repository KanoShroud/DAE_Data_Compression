"""BRSR共享状态StateOracle证书的纯函数。

本模块只生成共享状态波形、快速评价冻结4-bit静态配置、冻结planning策略并
计算evaluation统计。它不实现可部署策略，也不读取evaluation结果选择动作。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, firwin
from scipy.special import logsumexp

from brsr_stage2_rate_adaptive.stage2a import (
    cluster_bootstrap_effect,
    temper_probability,
)
from brsr_tdoa.brsr_core import (
    BayesianSpectralModel,
    NestedComplexQuantizer,
)
from brsr_tdoa.brsr_stage_b import StageBObservation
from brsr_tdoa.brsr_waveform import (
    StructuredTrial,
    WaveformConfig,
    apply_linear_fractional_delay,
    generate_structured_trial,
)


SCENE_S0 = "S0-independent-awgn"
SCENE_S1 = "S1-coherent-awgn"
SCENE_S2 = "S2-coherent-notch"
SCENES = (SCENE_S0, SCENE_S1, SCENE_S2)

BASE_PAYLOAD_BITS = 16
ACQUIRED_BIN_PAYLOAD_BITS = 8
ACTION_FEEDBACK_BITS = 5
PAYLOAD_24_BITS = BASE_PAYLOAD_BITS + ACQUIRED_BIN_PAYLOAD_BITS
PAYLOAD_32_BITS = BASE_PAYLOAD_BITS + 2 * ACQUIRED_BIN_PAYLOAD_BITS

NOTCH_CENTER_BINS = (137, 198, 251)
NOTCH_HALF_WIDTH_BINS = 12
NOTCH_NUM_TAPS = 65

KEY_COLUMNS = (
    "scene",
    "state_seed",
    "state_index",
    "state_id",
    "snr_db",
    "packet_index",
)


@dataclass(frozen=True)
class SharedState:
    state_seed: int
    state_index: int
    state_id: str
    true_tau: float
    true_rho: complex
    notch_center_bin: int

    @property
    def context_class(self) -> str:
        return f"notch-{self.notch_center_bin}"


def generate_state_bank(
    seeds: Sequence[int],
    states_per_seed: int,
    *,
    waveform: WaveformConfig,
    candidate_bins: np.ndarray,
) -> list[SharedState]:
    """生成不依赖planning/evaluation结果的连续共享状态原型。"""

    if states_per_seed < 1:
        raise ValueError("states_per_seed must be positive")
    bins = set(np.asarray(candidate_bins, dtype=int).tolist())
    if not set(NOTCH_CENTER_BINS).issubset(bins):
        raise ValueError("pre-registered notch centers are absent from candidate pool")
    states: list[SharedState] = []
    for seed_offset, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 701]))
        for state_index in range(states_per_seed):
            tau = float(rng.uniform(waveform.tau_min, waveform.tau_max))
            amplitude = float(rng.uniform(0.70, 0.90))
            phase = float(rng.uniform(-np.pi, np.pi))
            notch = NOTCH_CENTER_BINS[
                (seed_offset * states_per_seed + state_index)
                % len(NOTCH_CENTER_BINS)
            ]
            states.append(
                SharedState(
                    state_seed=int(seed),
                    state_index=int(state_index),
                    state_id=f"{int(seed)}:{int(state_index)}",
                    true_tau=tau,
                    true_rho=complex(amplitude * np.exp(1j * phase)),
                    notch_center_bin=int(notch),
                )
            )
    return states


def state_rows(states: Sequence[SharedState]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_seed": state.state_seed,
                "state_index": state.state_index,
                "state_id": state.state_id,
                "true_tau": state.true_tau,
                "true_rho_real": state.true_rho.real,
                "true_rho_imag": state.true_rho.imag,
                "notch_center_bin": state.notch_center_bin,
                "context_class": state.context_class,
            }
            for state in states
        ]
    )


def _common_notch(
    signal: np.ndarray,
    *,
    center_bin: int,
    waveform: WaveformConfig,
) -> np.ndarray:
    """施加公共线性相位陷波并保持输入平均功率。"""

    signal = np.asarray(signal, dtype=np.complex128)
    center = center_bin / waveform.n_time
    half_width = NOTCH_HALF_WIDTH_BINS / waveform.n_time
    low = center - half_width
    high = center + half_width
    if not 0.0 < low < high < 0.5:
        raise ValueError("notch band lies outside the discrete-time Nyquist interval")
    taps = firwin(
        NOTCH_NUM_TAPS,
        [low, high],
        pass_zero="bandstop",
        fs=1.0,
        window="hann",
    )
    filtered = fftconvolve(signal, taps, mode="same")
    input_power = float(np.mean(np.abs(signal) ** 2))
    output_power = float(np.mean(np.abs(filtered) ** 2))
    if input_power <= 0.0 or output_power <= 0.0:
        raise RuntimeError("notch filtering produced invalid power")
    return np.asarray(
        filtered * math.sqrt(input_power / output_power),
        dtype=np.complex128,
    )


def _coherent_trial(
    trial: StructuredTrial,
    *,
    state: SharedState,
    scene: str,
    waveform: WaveformConfig,
) -> StructuredTrial:
    source_long = np.asarray(trial.source_long, dtype=np.complex128)
    if scene == SCENE_S2:
        source_long = _common_notch(
            source_long,
            center_bin=state.notch_center_bin,
            waveform=waveform,
        )
    elif scene != SCENE_S1:
        raise ValueError("coherent trial requires S1 or S2")

    delayed = apply_linear_fractional_delay(
        source_long,
        state.true_tau,
        waveform.fractional_delay_half_len,
    )
    guard_numerator = source_long.size - waveform.n_time
    if guard_numerator < 0 or guard_numerator % 2:
        raise RuntimeError("structured waveform has an invalid symmetric guard")
    guard = guard_numerator // 2
    crop = slice(guard, guard + waveform.n_time)
    return replace(
        trial,
        source_long=source_long,
        source_a=source_long[crop].copy(),
        source_b_unit_gain=delayed[crop].copy(),
        true_tau=state.true_tau,
        true_rho=state.true_rho,
    )


def generate_packet_trial(
    state: SharedState,
    *,
    stage: str,
    packet_index: int,
    scene: str,
    waveform: WaveformConfig,
) -> StructuredTrial:
    """生成条件于共享状态、但planning/evaluation相互独立的数据包。"""

    stage_codes = {"planning": 101, "evaluation": 211}
    if stage not in stage_codes:
        raise ValueError("stage must be planning or evaluation")
    if packet_index < 0:
        raise ValueError("packet_index must be non-negative")
    if scene not in SCENES:
        raise ValueError(f"unknown scene: {scene}")
    scene_code = 307 if scene == SCENE_S0 else 401
    seed = np.random.SeedSequence(
        [
            state.state_seed,
            state.state_index,
            stage_codes[stage],
            packet_index,
            scene_code,
        ]
    )
    trial = generate_structured_trial(np.random.default_rng(seed), waveform)
    if scene == SCENE_S0:
        return trial
    return _coherent_trial(
        trial,
        state=state,
        scene=scene,
        waveform=waveform,
    )


def configuration_positions(
    model: BayesianSpectralModel,
    payload_bits: int,
) -> tuple[tuple[int, ...], ...]:
    """枚举固定基础层下的全部B24或B32配置。"""

    base = tuple(sorted(int(value) for value in model.base_positions))
    extras = tuple(
        position for position in range(model.n_bins) if position not in base
    )
    if payload_bits == PAYLOAD_24_BITS:
        additions = combinations(extras, 1)
    elif payload_bits == PAYLOAD_32_BITS:
        additions = combinations(extras, 2)
    else:
        raise ValueError("only B24 and B32 configurations are supported")
    return tuple(tuple(sorted(base + tuple(extra))) for extra in additions)


def configuration_key(positions: Iterable[int]) -> str:
    return " ".join(str(int(value)) for value in sorted(positions))


def parse_configuration_key(value: str) -> tuple[int, ...]:
    positions = tuple(int(item) for item in str(value).split())
    if not positions or tuple(sorted(set(positions))) != positions:
        raise ValueError("configuration key must contain increasing unique positions")
    return positions


def evaluate_configurations(
    observation: StageBObservation,
    *,
    model: BayesianSpectralModel,
    quantizer: NestedComplexQuantizer,
    configurations: Sequence[tuple[int, ...]],
    temperature: float,
) -> list[dict[str, object]]:
    """缓存逐频点量化似然，快速评价多个4-bit配置。"""

    if not configurations:
        raise ValueError("at least one configuration is required")
    observation.context.validate(model)
    all_positions = sorted(
        {
            int(position)
            for configuration in configurations
            for position in configuration
        }
    )
    log_likelihood: dict[int, np.ndarray] = {}
    for position in all_positions:
        real_code = int(
            np.asarray(
                quantizer.coarsen(
                    int(observation.fine_real_codes[position]),
                    4,
                )
            ).item()
        )
        imag_code = int(
            np.asarray(
                quantizer.coarsen(
                    int(observation.fine_imag_codes[position]),
                    4,
                )
            ).item()
        )
        log_likelihood[position] = quantizer.log_probability(
            observation.context.mean_by_bin[position],
            observation.context.variance_by_bin[position],
            real_code,
            imag_code,
            4,
        )

    log_prior = np.log(np.maximum(model.joint_prior, 1e-300))
    rows: list[dict[str, object]] = []
    for positions in configurations:
        joint_log = log_prior.copy()
        for position in positions:
            joint_log += log_likelihood[int(position)]
        posterior = np.exp(joint_log - logsumexp(joint_log))
        tau_probability = posterior.reshape(model.n_tau, model.n_rho).sum(axis=1)
        tau_probability = temper_probability(tau_probability, temperature)
        estimate = float(np.sum(model.tau_values * tau_probability))
        error = estimate - observation.true_tau
        positions_tuple = tuple(int(value) for value in positions)
        rows.append(
            {
                "configuration": configuration_key(positions_tuple),
                "positions": positions_tuple,
                "payload_bits": int(8 * len(positions_tuple)),
                "estimate_tau": estimate,
                "error_samples": error,
                "squared_error_samples2": error**2,
            }
        )
    return rows


def _best_configuration(group: pd.DataFrame) -> pd.Series:
    scores = (
        group.groupby(
            ["payload_bits", "configuration"],
            as_index=False,
            sort=True,
        )["squared_error_samples2"]
        .mean()
        .sort_values(
            ["squared_error_samples2", "configuration"],
            kind="stable",
        )
    )
    if scores.empty:
        raise RuntimeError("planning group contains no configurations")
    return scores.iloc[0]


def freeze_policies(planning: pd.DataFrame) -> pd.DataFrame:
    """只使用planning损失冻结四类B24/B32策略。"""

    required = set(KEY_COLUMNS) | {
        "context_class",
        "payload_bits",
        "configuration",
        "squared_error_samples2",
    }
    missing = required - set(planning.columns)
    if missing:
        raise ValueError(f"planning rows miss columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for scene, scene_rows in planning.groupby("scene", sort=True):
        for payload_bits, payload_rows in scene_rows.groupby(
            "payload_bits",
            sort=True,
        ):
            selected = _best_configuration(payload_rows)
            rows.append(
                {
                    "scene": scene,
                    "policy": "BestStatic",
                    "snr_db": math.nan,
                    "state_id": "",
                    "context_class": "",
                    "payload_bits": int(payload_bits),
                    "configuration": selected["configuration"],
                    "planning_mse": float(selected["squared_error_samples2"]),
                }
            )
            for snr, snr_rows in payload_rows.groupby("snr_db", sort=True):
                selected = _best_configuration(snr_rows)
                rows.append(
                    {
                        "scene": scene,
                        "policy": "SNR-only",
                        "snr_db": float(snr),
                        "state_id": "",
                        "context_class": "",
                        "payload_bits": int(payload_bits),
                        "configuration": selected["configuration"],
                        "planning_mse": float(selected["squared_error_samples2"]),
                    }
                )
                if scene == SCENE_S2:
                    for context, context_rows in snr_rows.groupby(
                        "context_class",
                        sort=True,
                    ):
                        selected = _best_configuration(context_rows)
                        rows.append(
                            {
                                "scene": scene,
                                "policy": "ContextStatic",
                                "snr_db": float(snr),
                                "state_id": "",
                                "context_class": str(context),
                                "payload_bits": int(payload_bits),
                                "configuration": selected["configuration"],
                                "planning_mse": float(
                                    selected["squared_error_samples2"]
                                ),
                            }
                        )
            if payload_bits != PAYLOAD_24_BITS:
                continue
            if scene == SCENE_S0:
                for snr, snr_rows in payload_rows.groupby("snr_db", sort=True):
                    selected = _best_configuration(snr_rows)
                    rows.append(
                        {
                            "scene": scene,
                            "policy": "StateOracle",
                            "snr_db": float(snr),
                            "state_id": "",
                            "context_class": "",
                            "payload_bits": PAYLOAD_24_BITS,
                            "configuration": selected["configuration"],
                            "planning_mse": float(
                                selected["squared_error_samples2"]
                            ),
                        }
                    )
            else:
                for (snr, state_id), state_group in payload_rows.groupby(
                    ["snr_db", "state_id"],
                    sort=True,
                ):
                    selected = _best_configuration(state_group)
                    rows.append(
                        {
                            "scene": scene,
                            "policy": "StateOracle",
                            "snr_db": float(snr),
                            "state_id": str(state_id),
                            "context_class": "",
                            "payload_bits": PAYLOAD_24_BITS,
                            "configuration": selected["configuration"],
                            "planning_mse": float(
                                selected["squared_error_samples2"]
                            ),
                        }
                    )
    policies = pd.DataFrame(rows)
    duplicated = policies.duplicated(
        [
            "scene",
            "policy",
            "snr_db",
            "state_id",
            "context_class",
            "payload_bits",
        ],
        keep=False,
    )
    if duplicated.any():
        raise RuntimeError("frozen policy keys are not unique")
    return policies


def planning_policy_diagnostics(planning: pd.DataFrame) -> pd.DataFrame:
    """报告B24 StateOracle planning动作的裕量和分半稳定性。"""

    required = set(KEY_COLUMNS) | {
        "payload_bits",
        "configuration",
        "squared_error_samples2",
    }
    missing = required - set(planning.columns)
    if missing:
        raise ValueError(f"planning rows miss columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    b24 = planning[planning["payload_bits"].eq(PAYLOAD_24_BITS)]
    for scene, scene_rows in b24.groupby("scene", sort=True):
        if scene == SCENE_S0:
            grouped = scene_rows.groupby(["snr_db"], sort=True)
        else:
            grouped = scene_rows.groupby(["snr_db", "state_id"], sort=True)
        for key, group in grouped:
            key_values = key if isinstance(key, tuple) else (key,)
            snr_db = float(key_values[0])
            state_id = "" if scene == SCENE_S0 else str(key_values[1])
            scores = (
                group.groupby("configuration", as_index=False, sort=True)[
                    "squared_error_samples2"
                ]
                .mean()
                .sort_values(
                    ["squared_error_samples2", "configuration"],
                    kind="stable",
                )
                .reset_index(drop=True)
            )
            if len(scores) < 2:
                raise RuntimeError("planning diagnostics require at least two actions")
            best = scores.iloc[0]
            second = scores.iloc[1]

            half_winners: list[str] = []
            half_counts: list[int] = []
            for parity in (0, 1):
                half = group[group["packet_index"].astype(int) % 2 == parity]
                half_counts.append(
                    int(half[list(KEY_COLUMNS)].drop_duplicates().shape[0])
                )
                half_winners.append(str(_best_configuration(half)["configuration"]))

            best_mse = float(best["squared_error_samples2"])
            second_mse = float(second["squared_error_samples2"])
            margin = second_mse - best_mse
            rows.append(
                {
                    "scene": scene,
                    "snr_db": snr_db,
                    "state_id": state_id,
                    "n_planning_packets": int(
                        group[list(KEY_COLUMNS)].drop_duplicates().shape[0]
                    ),
                    "n_actions": int(len(scores)),
                    "best_configuration": str(best["configuration"]),
                    "second_configuration": str(second["configuration"]),
                    "best_mse_samples2": best_mse,
                    "second_mse_samples2": second_mse,
                    "second_minus_best_mse_samples2": margin,
                    "relative_margin": margin / max(best_mse, 1e-12),
                    "half_a_packets": half_counts[0],
                    "half_b_packets": half_counts[1],
                    "half_a_configuration": half_winners[0],
                    "half_b_configuration": half_winners[1],
                    "half_a_equals_half_b": half_winners[0] == half_winners[1],
                    "both_halves_equal_full": all(
                        value == str(best["configuration"])
                        for value in half_winners
                    ),
                }
            )
    return pd.DataFrame(rows)


def lookup_configuration(
    policies: pd.DataFrame,
    *,
    scene: str,
    policy: str,
    snr_db: float,
    state_id: str,
    context_class: str,
    payload_bits: int,
) -> tuple[int, ...]:
    subset = policies[
        policies["scene"].eq(scene)
        & policies["policy"].eq(policy)
        & policies["payload_bits"].eq(int(payload_bits))
    ]
    if policy == "BestStatic":
        subset = subset[subset["snr_db"].isna()]
    else:
        subset = subset[subset["snr_db"].eq(float(snr_db))]
    if policy == "StateOracle" and scene != SCENE_S0:
        subset = subset[subset["state_id"].eq(state_id)]
    else:
        subset = subset[subset["state_id"].eq("")]
    if policy == "ContextStatic":
        subset = subset[subset["context_class"].eq(context_class)]
    else:
        subset = subset[subset["context_class"].eq("")]
    if len(subset) != 1:
        raise RuntimeError(
            "frozen policy lookup is missing or ambiguous: "
            f"{scene}/{policy}/{snr_db}/{state_id}/{context_class}/{payload_bits}"
        )
    return parse_configuration_key(str(subset.iloc[0]["configuration"]))


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(["scene", "method", "snr_db"], sort=True):
        scene, method, snr = keys
        rows.append(
            {
                "scene": scene,
                "method": method,
                "snr_db": float(snr),
                "n_rows": int(len(group)),
                "rmse_samples": math.sqrt(
                    float(group["squared_error_samples2"].mean())
                ),
                "median_abs_error_samples": float(
                    group["error_samples"].abs().median()
                ),
                "mean_payload_bits": float(group["payload_bits"].mean()),
                "mean_feedback_bits": float(group["feedback_bits"].mean()),
                "mean_total_bits": float(group["total_bits"].mean()),
            }
        )
    for (scene, method), group in results.groupby(
        ["scene", "method"],
        sort=True,
    ):
        rows.append(
            {
                "scene": scene,
                "method": method,
                "snr_db": math.nan,
                "n_rows": int(len(group)),
                "rmse_samples": math.sqrt(
                    float(group["squared_error_samples2"].mean())
                ),
                "median_abs_error_samples": float(
                    group["error_samples"].abs().median()
                ),
                "mean_payload_bits": float(group["payload_bits"].mean()),
                "mean_feedback_bits": float(group["feedback_bits"].mean()),
                "mean_total_bits": float(group["total_bits"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_effects(
    results: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    sample_keys = list(KEY_COLUMNS)
    rows: list[dict[str, object]] = []
    for scene in SCENES:
        target_methods = sorted(
            results[
                results["scene"].eq(scene)
                & results["method"].str.startswith("StateOracle-L")
            ]["method"].unique()
        )
        if not target_methods:
            raise RuntimeError(f"StateOracle rows are missing for {scene}")
        for target_method in target_methods:
            block_length = int(
                target_method.split("-L", 1)[1].split("-", 1)[0]
            )
            scene_baselines = [
                "BestStatic-B24",
                "SNR-only-B24",
                f"BestStatic-RateMatched-L{block_length}",
                f"SNR-only-RateMatched-L{block_length}",
            ]
            if scene == SCENE_S2:
                scene_baselines.extend(
                    (
                        "ContextStatic-B24",
                        f"ContextStatic-RateMatched-L{block_length}",
                    )
                )
            target = results[
                results["scene"].eq(scene)
                & results["method"].eq(target_method)
            ][sample_keys + ["squared_error_samples2"]].rename(
                columns={"squared_error_samples2": "target"}
            )
            for baseline in scene_baselines:
                reference = results[
                    results["scene"].eq(scene)
                    & results["method"].eq(baseline)
                ][sample_keys + ["squared_error_samples2"]].rename(
                    columns={"squared_error_samples2": "baseline"}
                )
                aligned = target.merge(
                    reference,
                    on=sample_keys,
                    validate="one_to_one",
                )
                if len(aligned) != len(target) or len(aligned) != len(reference):
                    raise RuntimeError("paired policy rows are incomplete")
                effect = cluster_bootstrap_effect(
                    aligned["target"].to_numpy(dtype=float),
                    aligned["baseline"].to_numpy(dtype=float),
                    aligned["state_id"].to_numpy(dtype=str),
                    repetitions=repetitions,
                    seed=seed + len(rows),
                )
                seed_gain = (
                    aligned.assign(gain=aligned["baseline"] - aligned["target"])
                    .groupby("state_seed", sort=True)["gain"]
                    .mean()
                )
                rows.append(
                    {
                        "scene": scene,
                        "block_length": block_length,
                        "target": target_method,
                        "baseline": baseline,
                        **effect,
                        "all_state_seeds_positive": bool(
                            (seed_gain > 0.0).all()
                        ),
                        "state_seed_gains": ";".join(
                            f"{int(key)}:{float(value):.9g}"
                            for key, value in seed_gain.items()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def gate_decision(
    effects: pd.DataFrame,
    *,
    mechanical_checks: dict[str, bool],
) -> dict[str, object]:
    if not effects.shape[0]:
        raise ValueError("paired effects are empty")
    mechanical_pass = bool(all(mechanical_checks.values()))
    scene_status: dict[str, str] = {}
    scene_pass: dict[str, bool] = {}
    block_status: dict[str, str] = {}
    for scene in SCENES:
        subset = effects[effects["scene"].eq(scene)]
        if subset.empty:
            raise RuntimeError(f"missing effects for {scene}")
        passed_lengths: list[int] = []
        hold_lengths: list[int] = []
        for block_length, group in subset.groupby("block_length", sort=True):
            point_positive = bool((group["mse_gain"] > 0.0).all())
            interval_positive = bool((group["mse_gain_ci_low"] > 0.0).all())
            seeds_positive = bool(group["all_state_seeds_positive"].all())
            passed = (
                mechanical_pass
                and point_positive
                and interval_positive
                and seeds_positive
            )
            key = f"{scene}|L{int(block_length)}"
            if not mechanical_pass:
                status = "HOLD_MECHANICAL"
            elif passed:
                status = "PASS_STATE_HEADROOM"
                passed_lengths.append(int(block_length))
            elif point_positive:
                status = "HOLD_POWER_STATE_HEADROOM"
                hold_lengths.append(int(block_length))
            else:
                status = "NO_GO_STATE_HEADROOM"
            block_status[key] = status
        scene_pass[scene] = bool(passed_lengths)
        if not mechanical_pass:
            scene_status[scene] = "HOLD_MECHANICAL"
        elif passed_lengths:
            joined = ",".join(str(value) for value in passed_lengths)
            scene_status[scene] = f"PASS_STATE_HEADROOM_L{joined}"
        elif hold_lengths:
            joined = ",".join(str(value) for value in hold_lengths)
            scene_status[scene] = f"HOLD_POWER_STATE_HEADROOM_L{joined}"
        else:
            scene_status[scene] = "NO_GO_STATE_HEADROOM"

    if not mechanical_pass:
        route_status = "HOLD_MECHANICAL"
    elif scene_pass[SCENE_S2]:
        route_status = "GO_OBSERVABLE_POLICY_DESIGN_S2"
    elif scene_pass[SCENE_S1]:
        route_status = "GO_OBSERVABLE_POLICY_DESIGN_S1"
    elif any(
        scene_status[scene].startswith("HOLD_POWER_STATE_HEADROOM")
        for scene in (SCENE_S1, SCENE_S2)
    ):
        route_status = "HOLD_POWER_STATE_HEADROOM"
    else:
        route_status = "NO_GO_ROUTE_STATE_HEADROOM"
    return {
        "status": route_status,
        "mechanical_checks": mechanical_checks,
        "block_status": block_status,
        "scene_status": scene_status,
        "scene_pass": scene_pass,
        "state_oracle_is_deployable": False,
        "outcome_oracle_used": False,
        "locked_data_used": False,
        "interpretation_boundary": (
            "A pass only permits a separately pre-registered ObservablePolicy "
            "development experiment. It does not establish deployable BRSR "
            "performance and does not open locked evaluation."
        ),
    }
