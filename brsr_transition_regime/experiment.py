"""BRSR转折区路线的统计、预算与门禁纯函数。

本模块不生成信号，也不读取locked结果。StateOracle只使用planning数据，
交叉拟合和正式evaluation均使用与动作选择互斥的数据。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import pandas as pd

from brsr_stage2_rate_adaptive.stage2a import (
    cluster_bootstrap_effect,
    cluster_bootstrap_mean,
)
from brsr_state_oracle_certificate.certificate import (
    PAYLOAD_24_BITS,
    SCENE_S1,
    SCENE_S2,
)

PRIMARY_SNR_DB = (0.0, 5.0)
CONTROL_SNR_DB = (-10.0, 15.0)
BLOCK_LENGTH = 16
CONTEXT_INDEX_BITS = 2
ACTION_FEEDBACK_BITS = 5

TARGET_METHOD = "TransitionStateOracle-L16-B24"
TARGET_BASELINE = "ContextStatic-RateMatched-L16"
FLAT_TARGET_METHOD = "TransitionStateOracle-L16-B24"
FLAT_BASELINE = "SNR-only-RateMatched-L16"

PAIR_KEYS = (
    "state_seed",
    "state_index",
    "state_id",
    "snr_db",
    "packet_index",
)


def target_total_bits(scene: str) -> float:
    """返回目标策略每包总通信量，包含块级元数据和动作反馈。"""

    if scene == SCENE_S1:
        overhead = ACTION_FEEDBACK_BITS
    elif scene == SCENE_S2:
        overhead = ACTION_FEEDBACK_BITS + CONTEXT_INDEX_BITS
    else:
        raise ValueError(f"unsupported scene: {scene}")
    return PAYLOAD_24_BITS + overhead / BLOCK_LENGTH


def _best_configuration(group: pd.DataFrame) -> str:
    required = {"configuration", "squared_error_samples2"}
    missing = required - set(group.columns)
    if missing:
        raise ValueError(f"configuration rows miss columns: {sorted(missing)}")
    scores = (
        group.groupby("configuration", as_index=False, sort=True)[
            "squared_error_samples2"
        ]
        .mean()
        .sort_values(
            ["squared_error_samples2", "configuration"],
            kind="stable",
        )
    )
    if scores.empty:
        raise RuntimeError("configuration group is empty")
    return str(scores.iloc[0]["configuration"])


def _configuration_mse(group: pd.DataFrame, configuration: str) -> float:
    subset = group[group["configuration"].eq(configuration)]
    if subset.empty:
        raise RuntimeError(f"configuration {configuration!r} is absent")
    return float(subset["squared_error_samples2"].mean())


def crossfit_transition_headroom(
    planning: pd.DataFrame,
    *,
    primary_snr_db: Sequence[float] = PRIMARY_SNR_DB,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """用奇偶包交叉拟合检验状态动作相对ContextStatic的条件余量。"""

    required = {
        "scene",
        "state_seed",
        "state_id",
        "context_class",
        "snr_db",
        "packet_index",
        "payload_bits",
        "configuration",
        "squared_error_samples2",
    }
    missing = required - set(planning.columns)
    if missing:
        raise ValueError(f"planning rows miss columns: {sorted(missing)}")
    subset = planning[
        planning["scene"].eq(SCENE_S2)
        & planning["payload_bits"].eq(PAYLOAD_24_BITS)
        & planning["snr_db"].isin([float(value) for value in primary_snr_db])
    ].copy()
    if subset.empty:
        raise ValueError("transition-scene planning rows are empty")

    rows: list[dict[str, object]] = []
    for selector_parity, evaluator_parity in ((0, 1), (1, 0)):
        selector = subset[
            subset["packet_index"].astype(int).mod(2).eq(selector_parity)
        ]
        evaluator = subset[
            subset["packet_index"].astype(int).mod(2).eq(evaluator_parity)
        ]
        state_actions = {
            (int(seed), str(state_id), float(snr)): _best_configuration(group)
            for (seed, state_id, snr), group in selector.groupby(
                ["state_seed", "state_id", "snr_db"], sort=True
            )
        }
        context_actions = {
            (float(snr), str(context)): _best_configuration(group)
            for (snr, context), group in selector.groupby(
                ["snr_db", "context_class"], sort=True
            )
        }
        for keys, group in evaluator.groupby(
            ["state_seed", "state_id", "snr_db", "context_class"],
            sort=True,
        ):
            seed, state_id, snr, context = keys
            state_configuration = state_actions[(int(seed), str(state_id), float(snr))]
            context_configuration = context_actions[(float(snr), str(context))]
            target_mse = _configuration_mse(group, state_configuration)
            baseline_mse = _configuration_mse(group, context_configuration)
            rows.append(
                {
                    "selector_parity": int(selector_parity),
                    "evaluator_parity": int(evaluator_parity),
                    "state_seed": int(seed),
                    "state_id": str(state_id),
                    "snr_db": float(snr),
                    "context_class": str(context),
                    "state_configuration": state_configuration,
                    "context_configuration": context_configuration,
                    "target_mse": target_mse,
                    "baseline_mse": baseline_mse,
                    "mse_gain": baseline_mse - target_mse,
                    "actions_differ": state_configuration != context_configuration,
                }
            )
    frame = pd.DataFrame(rows)
    effect = cluster_bootstrap_effect(
        frame["target_mse"].to_numpy(dtype=float),
        frame["baseline_mse"].to_numpy(dtype=float),
        frame["state_id"].to_numpy(dtype=str),
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    seed_gains = frame.groupby("state_seed", sort=True)["mse_gain"].mean()
    summary: dict[str, object] = {
        **effect,
        "all_state_seeds_positive": bool((seed_gains > 0.0).all()),
        "state_seed_gains": ";".join(
            f"{int(seed)}:{float(value):.9g}"
            for seed, value in seed_gains.items()
        ),
        "action_difference_rate": float(frame["actions_differ"].mean()),
    }
    return frame, summary


def paired_policy_effect(
    results: pd.DataFrame,
    *,
    scene: str,
    target: str,
    baseline: str,
    snr_db: Iterable[float],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    snr_values = [float(value) for value in snr_db]
    target_rows = results[
        results["scene"].eq(scene)
        & results["method"].eq(target)
        & results["snr_db"].isin(snr_values)
    ][[*PAIR_KEYS, "squared_error_samples2"]].rename(
        columns={"squared_error_samples2": "target"}
    )
    baseline_rows = results[
        results["scene"].eq(scene)
        & results["method"].eq(baseline)
        & results["snr_db"].isin(snr_values)
    ][[*PAIR_KEYS, "squared_error_samples2"]].rename(
        columns={"squared_error_samples2": "baseline"}
    )
    aligned = target_rows.merge(
        baseline_rows,
        on=list(PAIR_KEYS),
        validate="one_to_one",
    )
    if aligned.empty or len(aligned) != len(target_rows) or len(aligned) != len(baseline_rows):
        raise RuntimeError("paired policy rows are incomplete")
    effect = cluster_bootstrap_effect(
        aligned["target"].to_numpy(dtype=float),
        aligned["baseline"].to_numpy(dtype=float),
        aligned["state_id"].to_numpy(dtype=str),
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    gains = aligned.assign(gain=aligned["baseline"] - aligned["target"])
    seed_gains = gains.groupby("state_seed", sort=True)["gain"].mean()
    return {
        "scene": scene,
        "target": target,
        "baseline": baseline,
        "snr_scope": " ".join(f"{value:g}" for value in snr_values),
        **effect,
        "all_state_seeds_positive": bool((seed_gains > 0.0).all()),
        "state_seed_gains": ";".join(
            f"{int(seed)}:{float(value):.9g}"
            for seed, value in seed_gains.items()
        ),
    }


def heterogeneity_attribution(
    results: pd.DataFrame,
    *,
    primary_snr_db: Sequence[float] = PRIMARY_SNR_DB,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """检验目标场景的自适应增益是否显著高于平坦AWGN负对照。"""

    snr_values = [float(value) for value in primary_snr_db]

    def gains_for(scene: str, target: str, baseline: str, label: str) -> pd.DataFrame:
        target_rows = results[
            results["scene"].eq(scene)
            & results["method"].eq(target)
            & results["snr_db"].isin(snr_values)
        ][[*PAIR_KEYS, "squared_error_samples2"]].rename(
            columns={"squared_error_samples2": "target"}
        )
        baseline_rows = results[
            results["scene"].eq(scene)
            & results["method"].eq(baseline)
            & results["snr_db"].isin(snr_values)
        ][[*PAIR_KEYS, "squared_error_samples2"]].rename(
            columns={"squared_error_samples2": "baseline"}
        )
        aligned = target_rows.merge(
            baseline_rows,
            on=list(PAIR_KEYS),
            validate="one_to_one",
        )
        if aligned.empty:
            raise RuntimeError(f"{label} attribution rows are empty")
        return aligned.assign(**{label: aligned["baseline"] - aligned["target"]})[
            [*PAIR_KEYS, label]
        ]

    target_gain = gains_for(SCENE_S2, TARGET_METHOD, TARGET_BASELINE, "target_gain")
    flat_gain = gains_for(SCENE_S1, FLAT_TARGET_METHOD, FLAT_BASELINE, "flat_gain")
    aligned = target_gain.merge(flat_gain, on=list(PAIR_KEYS), validate="one_to_one")
    aligned["attribution_gain"] = aligned["target_gain"] - aligned["flat_gain"]
    summary = cluster_bootstrap_mean(
        aligned["attribution_gain"].to_numpy(dtype=float),
        aligned["state_id"].to_numpy(dtype=str),
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    seed_gains = aligned.groupby("state_seed", sort=True)["attribution_gain"].mean()
    summary = {
        **summary,
        "all_state_seeds_positive": bool((seed_gains > 0.0).all()),
        "state_seed_gains": ";".join(
            f"{int(seed)}:{float(value):.9g}"
            for seed, value in seed_gains.items()
        ),
    }
    return aligned, summary


def gate_decision(
    *,
    mechanical_checks: dict[str, bool],
    crossfit: dict[str, object],
    primary_effect: dict[str, object],
    attribution: dict[str, object],
    smoke: bool,
) -> dict[str, object]:
    mechanical_pass = bool(all(mechanical_checks.values()))
    crossfit_pass = bool(
        float(crossfit["mse_gain"]) > 0.0
        and float(crossfit["mse_gain_ci_low"]) > 0.0
        and bool(crossfit["all_state_seeds_positive"])
    )
    primary_pass = bool(
        float(primary_effect["mse_gain"]) > 0.0
        and float(primary_effect["mse_gain_ci_low"]) > 0.0
        and bool(primary_effect["all_state_seeds_positive"])
    )
    attribution_pass = bool(
        float(attribution["mean"]) > 0.0
        and float(attribution["ci_low"]) > 0.0
        and bool(attribution["all_state_seeds_positive"])
    )
    if not mechanical_pass:
        status = "HOLD_MECHANICAL"
    elif not crossfit_pass:
        status = "NO_GO_TRANSITION_ACTION_IDENTIFIABILITY"
    elif not primary_pass:
        status = "NO_GO_TRANSITION_STATE_HEADROOM"
    elif not attribution_pass:
        status = "NO_GO_HETEROGENEITY_ATTRIBUTION"
    else:
        status = "GO_OBSERVABLE_POLICY_DESIGN_TRANSITION"
    decision = {
        "status": "SMOKE_ONLY" if smoke else status,
        "provisional_status": status if smoke else None,
        "mechanical_checks": mechanical_checks,
        "crossfit_pass": crossfit_pass,
        "primary_headroom_pass": primary_pass,
        "heterogeneity_attribution_pass": attribution_pass,
        "interpretation_boundary": (
            "GO only permits a separately pre-registered deployable policy in the "
            "moderate-SNR, block-coherent spectral-heterogeneity scope. It does not "
            "reverse the frozen broad-SNR BRSR NO-GO result."
        ),
    }
    if smoke:
        decision["smoke_is_performance_evidence"] = False
    return decision


def expected_rate_matched_payload(
    *,
    scene: str,
    family: str,
) -> float:
    """给出静态基线为匹配目标总bit应采用的平均载荷。"""

    total = target_total_bits(scene)
    context = CONTEXT_INDEX_BITS / BLOCK_LENGTH if family == "ContextStatic" else 0.0
    payload = total - context
    if not 24.0 <= payload <= 32.0 or not math.isfinite(payload):
        raise ValueError("rate-matched payload lies outside B24/B32 bracket")
    return payload
