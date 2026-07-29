"""Read-only protocol decomposition for the frozen BRSR Stage 2A.1 result."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from brsr_stage2_rate_adaptive.stage2a import cluster_bootstrap_effect


KEYS = ("seed", "trial", "cluster_id", "snr_db")
BASE_PAYLOAD_BITS = 16
ACTION_PAYLOAD_BITS = 8
ACTION_FEEDBACK_BITS = 5
PAYLOAD_24_BITS = BASE_PAYLOAD_BITS + ACTION_PAYLOAD_BITS
PAYLOAD_32_BITS = PAYLOAD_24_BITS + ACTION_PAYLOAD_BITS
BLOCK_SIZES = (1, 2, 4, 8, 40)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_tables(
    trials: pd.DataFrame,
    actions: pd.DataFrame,
) -> None:
    trial_required = set(KEYS) | {
        "method",
        "squared_error_samples2",
        "total_bits",
    }
    action_required = set(KEYS) | {
        "position",
        "bin_index",
        "calibrated_expected_risk_after_samples2",
        "calibrated_squared_error_samples2",
    }
    missing_trials = trial_required - set(trials.columns)
    missing_actions = action_required - set(actions.columns)
    if missing_trials:
        raise ValueError(f"trial table missing columns: {sorted(missing_trials)}")
    if missing_actions:
        raise ValueError(f"action table missing columns: {sorted(missing_actions)}")
    if trials.duplicated([*KEYS, "method"]).any():
        raise ValueError("trial table has duplicate key/method rows")
    if actions.duplicated([*KEYS, "position"]).any():
        raise ValueError("action table has duplicate key/position rows")
    if trials[list(KEYS)].isna().any().any() or actions[list(KEYS)].isna().any().any():
        raise ValueError("source key columns contain missing values")
    if not np.isfinite(
        trials["squared_error_samples2"].to_numpy(dtype=float)
    ).all():
        raise ValueError("trial squared errors contain non-finite values")
    for column in (
        "calibrated_expected_risk_after_samples2",
        "calibrated_squared_error_samples2",
    ):
        if not np.isfinite(actions[column].to_numpy(dtype=float)).all():
            raise ValueError(f"action column {column} contains non-finite values")

    expected_methods = {
        "BestStatic-B16",
        "BestStatic-B24",
        "BestStatic-B32",
        "SNR-only-B16",
        "SNR-only-B24",
        "SNR-only-B32",
        "FixedAction-B29",
        "PredictedAction-B29",
        "OracleAction-B29",
    }
    missing_methods = expected_methods - set(trials["method"].unique())
    if missing_methods:
        raise ValueError(f"source result missing methods: {sorted(missing_methods)}")

    key_count = trials[list(KEYS)].drop_duplicates().shape[0]
    action_key_count = actions[list(KEYS)].drop_duplicates().shape[0]
    if key_count != action_key_count:
        raise ValueError("trial and action tables cover different sample counts")
    action_counts = actions.groupby(list(KEYS), sort=False)["position"].nunique()
    if action_counts.nunique() != 1 or int(action_counts.iloc[0]) < 2:
        raise ValueError("candidate action count is not constant across samples")


def method_rows(trials: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = trials[trials["method"].eq(method)].copy()
    expected = trials[list(KEYS)].drop_duplicates().shape[0]
    if len(rows) != expected:
        raise ValueError(f"method {method} is incomplete")
    return rows.sort_values(list(KEYS), kind="stable").reset_index(drop=True)


def method_summary(trials: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        subset = method_rows(trials, method)
        payload = (
            subset["payload_bits"]
            if "payload_bits" in subset
            else subset["total_bits"]
        )
        feedback = (
            subset["feedback_bits"]
            if "feedback_bits" in subset
            else pd.Series(0.0, index=subset.index)
        )
        rows.append(
            {
                "method": method,
                "n_rows": int(len(subset)),
                "mean_payload_bits": float(payload.mean()),
                "recorded_mean_feedback_bits": float(feedback.mean()),
                "recorded_mean_total_bits": float(subset["total_bits"].mean()),
                "rmse_samples": math.sqrt(
                    float(subset["squared_error_samples2"].mean())
                ),
                "mse_samples2": float(
                    subset["squared_error_samples2"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def assert_fixed_matches_b24(trials: pd.DataFrame) -> float:
    fixed = method_rows(trials, "FixedAction-B29")
    static = method_rows(trials, "BestStatic-B24")
    merged = fixed[[*KEYS, "squared_error_samples2"]].merge(
        static[[*KEYS, "squared_error_samples2"]],
        on=list(KEYS),
        suffixes=("_fixed", "_b24"),
        validate="one_to_one",
    )
    maximum = float(
        np.max(
            np.abs(
                merged["squared_error_samples2_fixed"]
                - merged["squared_error_samples2_b24"]
            )
        )
    )
    if maximum > 1e-10:
        raise RuntimeError(
            "FixedAction-B29 does not reproduce frozen BestStatic-B24 "
            f"(max squared-error difference={maximum:.3e})"
        )
    return maximum


def expected_static_frontier(
    trials: pd.DataFrame,
    *,
    family: str,
    target_bits: float,
    label: str,
) -> pd.DataFrame:
    if not PAYLOAD_24_BITS <= target_bits <= PAYLOAD_32_BITS:
        raise ValueError("static frontier target must lie in [24, 32] bits")
    low = method_rows(trials, f"{family}-B24")
    high = method_rows(trials, f"{family}-B32")
    merged = low[[*KEYS, "squared_error_samples2"]].merge(
        high[[*KEYS, "squared_error_samples2"]],
        on=list(KEYS),
        suffixes=("_b24", "_b32"),
        validate="one_to_one",
    )
    high_fraction = (target_bits - PAYLOAD_24_BITS) / (
        PAYLOAD_32_BITS - PAYLOAD_24_BITS
    )
    merged["squared_error_samples2"] = (
        (1.0 - high_fraction) * merged["squared_error_samples2_b24"]
        + high_fraction * merged["squared_error_samples2_b32"]
    )
    merged["method"] = label
    merged["total_bits"] = float(target_bits)
    merged["high_payload_fraction"] = float(high_fraction)
    return merged[
        [
            *KEYS,
            "method",
            "squared_error_samples2",
            "total_bits",
            "high_payload_fraction",
        ]
    ]


def _block_assignments(
    actions: pd.DataFrame,
    *,
    block_size: int,
    criterion: str,
) -> pd.DataFrame:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    frame = actions.copy()
    frame["block_index"] = (
        frame["trial"].to_numpy(dtype=int) // int(block_size)
    )
    block_keys = ["seed", "snr_db", "block_index"]
    return (
        frame.groupby([*block_keys, "position"], as_index=False)
        .agg(
            block_score=(criterion, "mean"),
            block_n=("trial", "nunique"),
        )
        .sort_values(
            [*block_keys, "block_score", "position"],
            kind="stable",
        )
        .drop_duplicates(block_keys, keep="first")
        .rename(columns={"position": "selected_position"})
    )


def build_block_policy_rows(
    actions: pd.DataFrame,
    *,
    block_size: int,
    policy: str,
) -> pd.DataFrame:
    if policy == "predicted":
        criterion = "calibrated_expected_risk_after_samples2"
        label = f"BlockPredicted-L{block_size}"
    elif policy == "oracle":
        criterion = "calibrated_squared_error_samples2"
        label = f"BlockOracle-L{block_size}"
    else:
        raise ValueError("policy must be 'predicted' or 'oracle'")

    frame = actions.copy()
    frame["block_index"] = (
        frame["trial"].to_numpy(dtype=int) // int(block_size)
    )
    assignments = _block_assignments(
        frame,
        block_size=block_size,
        criterion=criterion,
    )
    joined = frame.merge(
        assignments[
            ["seed", "snr_db", "block_index", "selected_position", "block_n"]
        ],
        on=["seed", "snr_db", "block_index"],
        validate="many_to_one",
    )
    selected = joined[
        joined["position"].eq(joined["selected_position"])
    ].copy()
    expected_rows = frame[list(KEYS)].drop_duplicates().shape[0]
    if len(selected) != expected_rows:
        raise RuntimeError(f"{label} did not select exactly one action per sample")
    selected["method"] = label
    selected["squared_error_samples2"] = selected[
        "calibrated_squared_error_samples2"
    ]
    selected["total_bits"] = PAYLOAD_24_BITS + (
        ACTION_FEEDBACK_BITS / selected["block_n"].to_numpy(dtype=float)
    )
    selected["feedback_bits_per_sample"] = (
        ACTION_FEEDBACK_BITS / selected["block_n"].to_numpy(dtype=float)
    )
    selected["block_size_requested"] = int(block_size)
    return selected[
        [
            *KEYS,
            "block_index",
            "block_n",
            "method",
            "selected_position",
            "bin_index",
            "squared_error_samples2",
            "feedback_bits_per_sample",
            "total_bits",
            "block_size_requested",
        ]
    ].sort_values(list(KEYS), kind="stable").reset_index(drop=True)


def assert_l1_alignment(
    trials: pd.DataFrame,
    predicted_l1: pd.DataFrame,
    oracle_l1: pd.DataFrame,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for generated, reference_name, label in (
        (predicted_l1, "PredictedAction-B29", "predicted"),
        (oracle_l1, "OracleAction-B29", "oracle"),
    ):
        reference = method_rows(trials, reference_name)
        merged = generated[[*KEYS, "squared_error_samples2"]].merge(
            reference[[*KEYS, "squared_error_samples2"]],
            on=list(KEYS),
            suffixes=("_generated", "_reference"),
            validate="one_to_one",
        )
        maximum = float(
            np.max(
                np.abs(
                    merged["squared_error_samples2_generated"]
                    - merged["squared_error_samples2_reference"]
                )
            )
        )
        if maximum > 1e-10:
            raise RuntimeError(
                f"L=1 {label} policy is misaligned with Stage 2A.1 "
                f"(max squared-error difference={maximum:.3e})"
            )
        values[f"{label}_l1_max_squared_error_difference"] = maximum
    return values


def effect_row(
    target: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    target_method: str,
    baseline_method: str,
    block_size: int,
    realizable: bool,
    repetitions: int,
    seed: int,
    snr_db: float | None = None,
) -> dict[str, float | int | str | bool]:
    left = target
    right = baseline
    if snr_db is not None:
        left = left[left["snr_db"].eq(float(snr_db))]
        right = right[right["snr_db"].eq(float(snr_db))]
    paired = left[[*KEYS, "squared_error_samples2", "total_bits"]].merge(
        right[[*KEYS, "squared_error_samples2", "total_bits"]],
        on=list(KEYS),
        suffixes=("_target", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("protocol comparison is not fully paired")
    summary = cluster_bootstrap_effect(
        paired["squared_error_samples2_target"].to_numpy(dtype=float),
        paired["squared_error_samples2_baseline"].to_numpy(dtype=float),
        paired["cluster_id"].to_numpy(dtype=str),
        repetitions=repetitions,
        seed=seed,
    )
    return {
        "target_method": target_method,
        "baseline_method": baseline_method,
        "block_size": int(block_size),
        "realizable": bool(realizable),
        "snr_db": math.nan if snr_db is None else float(snr_db),
        "target_mean_total_bits": float(
            paired["total_bits_target"].mean()
        ),
        "baseline_mean_total_bits": float(
            paired["total_bits_baseline"].mean()
        ),
        "mean_total_bits_difference": float(
            paired["total_bits_target"].mean()
            - paired["total_bits_baseline"].mean()
        ),
        **summary,
    }


def ideal_per_sample_action_rows(
    trials: pd.DataFrame,
    *,
    method: str,
    block_size: int,
) -> pd.DataFrame:
    rows = method_rows(trials, method)
    rows["method"] = f"IdealAmortized-{method}-L{block_size}"
    rows["total_bits"] = PAYLOAD_24_BITS + ACTION_FEEDBACK_BITS / block_size
    return rows


def action_gap_table(
    trials: pd.DataFrame,
    *,
    repetitions: int,
) -> pd.DataFrame:
    comparisons = (
        ("PredictedAction-B29", "FixedAction-B29", "prediction_over_fixed"),
        ("OracleAction-B29", "PredictedAction-B29", "oracle_over_prediction"),
        ("OracleAction-B29", "FixedAction-B29", "oracle_over_fixed"),
    )
    rows = []
    for index, (target_name, baseline_name, component) in enumerate(comparisons):
        target = method_rows(trials, target_name)
        baseline = method_rows(trials, baseline_name)
        row = effect_row(
            target,
            baseline,
            target_method=target_name,
            baseline_method=baseline_name,
            block_size=1,
            realizable=True,
            repetitions=repetitions,
            seed=2026073101 + index,
        )
        row["component"] = component
        rows.append(row)
    return pd.DataFrame(rows)


def protocol_decision(effects: pd.DataFrame) -> dict[str, object]:
    overall = effects[effects["snr_db"].isna()]
    realizable = overall[
        overall["realizable"]
        & overall["target_method"].str.startswith("Block")
    ]

    def passing_blocks(prefix: str) -> list[int]:
        subset = realizable[realizable["target_method"].str.startswith(prefix)]
        passed = []
        for block_size, group in subset.groupby("block_size"):
            if (
                set(group["baseline_method"])
                == {"BestStatic-Expected", "SNR-only-Expected"}
                and (group["mse_gain_ci_low"] > 0.0).all()
            ):
                passed.append(int(block_size))
        return sorted(passed)

    predicted = passing_blocks("BlockPredicted")
    oracle = passing_blocks("BlockOracle")
    if predicted:
        status = "FEASIBLE-BLOCK-PREDICTED"
    elif oracle:
        status = "UPPER-BOUND-ONLY"
    else:
        status = "NO-FEASIBLE-BLOCK-REGION"
    return {
        "status": status,
        "decision_rule": (
            "A realizable block policy passes only when its paired two-sided "
            "95% cluster-bootstrap MSE-gain CI is strictly above zero against "
            "both expected BestStatic and SNR-only static frontiers at the "
            "same mean total bits."
        ),
        "passing_predicted_block_sizes": predicted,
        "passing_oracle_block_sizes": oracle,
        "route_interpretation": (
            "FEASIBLE-BLOCK-PREDICTED supports preregistering a new block "
            "protocol; UPPER-BOUND-ONLY retains protocol potential but rejects "
            "the current observable block score; NO-FEASIBLE-BLOCK-REGION "
            "rejects the tested one-action shared-feedback protocol space."
        ),
    }
