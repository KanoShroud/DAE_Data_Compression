"""Pure statistics for the pre-registered S1 precision replication."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from brsr_stage2_rate_adaptive.stage2a import cluster_bootstrap_effect


SCENE_S1 = "S1-coherent-awgn"
TARGET_METHOD = "StateOracle-L16-B24"
BASELINES = (
    "BestStatic-B24",
    "SNR-only-B24",
    "BestStatic-RateMatched-L16",
    "SNR-only-RateMatched-L16",
)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _configuration_scores(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("configuration", as_index=False, sort=True)[
            "squared_error_samples2"
        ]
        .mean()
        .sort_values(
            ["squared_error_samples2", "configuration"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _cluster_mean_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    if repetitions < 20:
        raise ValueError("repetitions must be at least 20")
    frame = pd.DataFrame({"value": values, "cluster": clusters})
    cluster_means = (
        frame.groupby("cluster", sort=True)["value"].mean().to_numpy(dtype=float)
    )
    if cluster_means.size < 2:
        value = float(cluster_means.mean())
        return value, value, value
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        cluster_means,
        size=(repetitions, cluster_means.size),
        replace=True,
    ).mean(axis=1)
    return (
        float(cluster_means.mean()),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def planning_convergence_diagnostics(
    selection_rows: pd.DataFrame,
    check_rows: pd.DataFrame,
    *,
    candidate_counts: Sequence[int],
    reference_count: int,
    equivalence_margin: float,
    sensitivity_margins: Sequence[float],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int | None]:
    """Select the smallest planning prefix equivalent to the frozen reference.

    Candidate actions are selected only from ``selection_rows``. Their losses are
    compared on the independent ``check_rows`` stream. The check stream is a
    calibration-only resource and is never reused in the confirmation experiment.
    """

    required = {
        "scene",
        "state_seed",
        "state_id",
        "snr_db",
        "packet_index",
        "payload_bits",
        "configuration",
        "squared_error_samples2",
    }
    _require_columns(selection_rows, required, "selection_rows")
    _require_columns(check_rows, required, "check_rows")
    counts = tuple(int(value) for value in candidate_counts)
    if not counts or tuple(sorted(set(counts))) != counts:
        raise ValueError("candidate_counts must be unique and increasing")
    if counts[-1] >= reference_count:
        raise ValueError("candidate_counts must be smaller than reference_count")
    if equivalence_margin <= 0.0:
        raise ValueError("equivalence_margin must be positive")
    if any(value <= 0.0 for value in sensitivity_margins):
        raise ValueError("sensitivity margins must be positive")

    selection = selection_rows[
        selection_rows["scene"].eq(SCENE_S1)
        & selection_rows["payload_bits"].eq(24)
        & selection_rows["packet_index"].lt(reference_count)
    ].copy()
    check = check_rows[
        check_rows["scene"].eq(SCENE_S1)
        & check_rows["payload_bits"].eq(24)
    ].copy()
    if selection.empty or check.empty:
        raise ValueError("planning convergence inputs contain no S1 B24 rows")

    keys = ["state_seed", "state_id", "snr_db"]
    selection_groups = {
        key: group for key, group in selection.groupby(keys, sort=True)
    }
    check_groups = {key: group for key, group in check.groupby(keys, sort=True)}
    if selection_groups.keys() != check_groups.keys():
        raise RuntimeError("selection and check state/SNR keys are not aligned")

    detail_rows: list[dict[str, object]] = []
    for group_index, (key, select_group) in enumerate(selection_groups.items()):
        state_seed, state_id, snr_db = key
        check_scores = _configuration_scores(check_groups[key]).rename(
            columns={"squared_error_samples2": "check_mse"}
        )
        reference_scores = _configuration_scores(
            select_group[select_group["packet_index"].lt(reference_count)]
        )
        reference_configuration = str(reference_scores.iloc[0]["configuration"])
        check_map = check_scores.set_index("configuration")["check_mse"]
        if reference_configuration not in check_map:
            raise RuntimeError("reference action is absent from check stream")
        crossfit_regrets: list[float] = []
        for selector_parity, evaluator_parity in ((0, 1), (1, 0)):
            selector = check_groups[key][
                check_groups[key]["packet_index"].astype(int) % 2
                == selector_parity
            ]
            evaluator = check_groups[key][
                check_groups[key]["packet_index"].astype(int) % 2
                == evaluator_parity
            ]
            selector_configuration = str(
                _configuration_scores(selector).iloc[0]["configuration"]
            )
            evaluator_scores = _configuration_scores(evaluator).set_index(
                "configuration"
            )["squared_error_samples2"]
            crossfit_regrets.append(
                float(evaluator_scores.loc[reference_configuration])
                - float(evaluator_scores.loc[selector_configuration])
            )
        reference_crossfit_regret = float(np.mean(crossfit_regrets))
        check_best_configuration = str(
            check_scores.sort_values(
                ["check_mse", "configuration"],
                kind="stable",
            ).iloc[0]["configuration"]
        )
        reference_check_mse = float(check_map.loc[reference_configuration])
        check_best_mse = float(check_map.loc[check_best_configuration])

        for count in counts:
            prefix = select_group[select_group["packet_index"].lt(count)]
            prefix_scores = _configuration_scores(prefix).rename(
                columns={"squared_error_samples2": "selection_mse"}
            )
            selected_configuration = str(
                prefix_scores.iloc[0]["configuration"]
            )
            if selected_configuration not in check_map:
                raise RuntimeError("prefix action is absent from check stream")
            aligned_scores = prefix_scores.merge(
                check_scores,
                on="configuration",
                validate="one_to_one",
            )
            rank_spearman = float(
                aligned_scores["selection_mse"].corr(
                    aligned_scores["check_mse"],
                    method="spearman",
                )
            )
            selected_check_mse = float(check_map.loc[selected_configuration])
            detail_rows.append(
                {
                    "state_seed": int(state_seed),
                    "state_id": str(state_id),
                    "snr_db": float(snr_db),
                    "candidate_count": int(count),
                    "reference_count": int(reference_count),
                    "selected_configuration": selected_configuration,
                    "reference_configuration": reference_configuration,
                    "check_best_configuration": check_best_configuration,
                    "selected_equals_reference": (
                        selected_configuration == reference_configuration
                    ),
                    "reference_equals_check_best": (
                        reference_configuration == check_best_configuration
                    ),
                    "selected_check_mse": selected_check_mse,
                    "reference_check_mse": reference_check_mse,
                    "check_best_mse": check_best_mse,
                    "regret_vs_reference_samples2": (
                        selected_check_mse - reference_check_mse
                    ),
                    "reference_regret_vs_check_best_samples2": (
                        reference_check_mse - check_best_mse
                    ),
                    "reference_crossfit_regret_samples2": (
                        reference_crossfit_regret
                    ),
                    "action_rank_spearman": rank_spearman,
                    "group_index": int(group_index),
                }
            )

    details = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    selected_count: int | None = None
    reference_subset = details[
        details["candidate_count"].eq(counts[0])
    ]
    (
        reference_crossfit_regret,
        reference_crossfit_ci_low,
        reference_crossfit_ci_high,
    ) = _cluster_mean_ci(
        reference_subset[
            "reference_crossfit_regret_samples2"
        ].to_numpy(dtype=float),
        reference_subset["state_id"].to_numpy(dtype=str),
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 10_000,
    )
    reference_seed_regret = reference_subset.groupby(
        "state_seed",
        sort=True,
    )["reference_crossfit_regret_samples2"].mean()
    for count in counts:
        subset = details[details["candidate_count"].eq(count)]
        mean_regret, ci_low, ci_high = _cluster_mean_ci(
            subset["regret_vs_reference_samples2"].to_numpy(dtype=float),
            subset["state_id"].to_numpy(dtype=str),
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + int(count),
        )
        seed_regret = subset.groupby("state_seed", sort=True)[
            "regret_vs_reference_samples2"
        ].mean()
        reference_pass = bool(
            reference_crossfit_regret <= equivalence_margin
            and reference_crossfit_ci_high <= equivalence_margin
            and (reference_seed_regret <= equivalence_margin).all()
        )
        primary_pass = bool(
            mean_regret <= equivalence_margin
            and ci_high <= equivalence_margin
            and (seed_regret <= equivalence_margin).all()
            and reference_pass
        )
        summary_rows.append(
            {
                "candidate_count": int(count),
                "reference_count": int(reference_count),
                "n_state_snr_conditions": int(len(subset)),
                "n_state_clusters": int(subset["state_id"].nunique()),
                "mean_regret_samples2": mean_regret,
                "regret_ci_low": ci_low,
                "regret_ci_high": ci_high,
                "equivalence_margin_samples2": float(equivalence_margin),
                "all_seed_mean_regrets_within_margin": bool(
                    (seed_regret <= equivalence_margin).all()
                ),
                "seed_mean_regrets": ";".join(
                    f"{int(seed)}:{float(value):.9g}"
                    for seed, value in seed_regret.items()
                ),
                "selected_equals_reference_rate": float(
                    subset["selected_equals_reference"].mean()
                ),
                "reference_equals_check_best_rate": float(
                    subset["reference_equals_check_best"].mean()
                ),
                "mean_reference_regret_vs_check_best_samples2": float(
                    subset["reference_regret_vs_check_best_samples2"].mean()
                ),
                "reference_crossfit_regret_samples2": (
                    reference_crossfit_regret
                ),
                "reference_crossfit_regret_ci_low": reference_crossfit_ci_low,
                "reference_crossfit_regret_ci_high": reference_crossfit_ci_high,
                "all_seed_reference_regrets_within_margin": bool(
                    (reference_seed_regret <= equivalence_margin).all()
                ),
                "reference_crossfit_equivalence_pass": reference_pass,
                "mean_action_rank_spearman": float(
                    subset["action_rank_spearman"].mean()
                ),
                "primary_equivalence_pass": primary_pass,
            }
        )
        for margin in sensitivity_margins:
            sensitivity_rows.append(
                {
                    "candidate_count": int(count),
                    "equivalence_margin_samples2": float(margin),
                    "mean_within_margin": bool(mean_regret <= margin),
                    "ci_high_within_margin": bool(ci_high <= margin),
                    "all_seed_means_within_margin": bool(
                        (seed_regret <= margin).all()
                    ),
                    "reference_mean_within_margin": bool(
                        reference_crossfit_regret <= margin
                    ),
                    "reference_ci_high_within_margin": bool(
                        reference_crossfit_ci_high <= margin
                    ),
                    "all_seed_reference_means_within_margin": bool(
                        (reference_seed_regret <= margin).all()
                    ),
                    "equivalence_pass": bool(
                        mean_regret <= margin
                        and ci_high <= margin
                        and (seed_regret <= margin).all()
                        and reference_crossfit_regret <= margin
                        and reference_crossfit_ci_high <= margin
                        and (reference_seed_regret <= margin).all()
                    ),
                }
            )
        if selected_count is None and primary_pass:
            selected_count = int(count)

    return (
        details,
        pd.DataFrame(summary_rows),
        pd.DataFrame(sensitivity_rows),
        selected_count,
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    required = {
        "scene",
        "method",
        "snr_db",
        "squared_error_samples2",
        "error_samples",
        "payload_bits",
        "feedback_bits",
        "total_bits",
    }
    _require_columns(results, required, "results")
    rows: list[dict[str, object]] = []
    groupings = (
        (["scene", "method", "snr_db"], False),
        (["scene", "method"], True),
    )
    for columns, aggregate_snr in groupings:
        for keys, group in results.groupby(columns, sort=True):
            values = keys if isinstance(keys, tuple) else (keys,)
            scene = str(values[0])
            method = str(values[1])
            snr_db = math.nan if aggregate_snr else float(values[2])
            rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "snr_db": snr_db,
                    "n_rows": int(len(group)),
                    "n_state_clusters": int(group["state_id"].nunique()),
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
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    required = {
        "scene",
        "state_seed",
        "state_index",
        "state_id",
        "snr_db",
        "packet_index",
        "method",
        "squared_error_samples2",
    }
    _require_columns(results, required, "results")
    sample_keys = [
        "scene",
        "state_seed",
        "state_index",
        "state_id",
        "snr_db",
        "packet_index",
    ]
    target = results[results["method"].eq(TARGET_METHOD)][
        sample_keys + ["squared_error_samples2"]
    ].rename(columns={"squared_error_samples2": "target"})
    if target.empty:
        raise ValueError(f"missing target method: {TARGET_METHOD}")
    rows: list[dict[str, object]] = []
    for index, baseline in enumerate(BASELINES):
        reference = results[results["method"].eq(baseline)][
            sample_keys + ["squared_error_samples2"]
        ].rename(columns={"squared_error_samples2": "baseline"})
        aligned = target.merge(
            reference,
            on=sample_keys,
            validate="one_to_one",
        )
        if len(aligned) != len(target) or len(aligned) != len(reference):
            raise RuntimeError(f"paired rows are incomplete for {baseline}")
        effect = cluster_bootstrap_effect(
            aligned["target"].to_numpy(dtype=float),
            aligned["baseline"].to_numpy(dtype=float),
            aligned["state_id"].to_numpy(dtype=str),
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + index,
        )
        seed_gain = (
            aligned.assign(gain=aligned["baseline"] - aligned["target"])
            .groupby("state_seed", sort=True)["gain"]
            .mean()
        )
        snr_gain = (
            aligned.assign(gain=aligned["baseline"] - aligned["target"])
            .groupby("snr_db", sort=True)["gain"]
            .mean()
        )
        rows.append(
            {
                "scene": SCENE_S1,
                "block_length": 16,
                "target": TARGET_METHOD,
                "baseline": baseline,
                **effect,
                "all_state_seeds_positive": bool((seed_gain > 0.0).all()),
                "state_seed_gains": ";".join(
                    f"{int(seed)}:{float(value):.9g}"
                    for seed, value in seed_gain.items()
                ),
                "snr_gains": ";".join(
                    f"{float(snr):g}:{float(value):.9g}"
                    for snr, value in snr_gain.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def gate_decision(
    *,
    mode: str,
    planning_selected_count: int | None,
    effects: pd.DataFrame | None,
    mechanical_checks: dict[str, bool],
) -> dict[str, object]:
    mechanical_pass = bool(all(mechanical_checks.values()))
    if mode == "smoke":
        return {
            "status": "SMOKE_ONLY",
            "provisional_status": (
                "READY_FOR_CONFIRMATION"
                if planning_selected_count is not None
                else "NO_GO_ROUTE_PLANNING_NONCONVERGENCE"
            ),
            "mechanical_checks": mechanical_checks,
            "smoke_is_performance_evidence": False,
        }
    if not mechanical_pass:
        status = "HOLD_MECHANICAL"
    elif planning_selected_count is None:
        status = "NO_GO_ROUTE_PLANNING_NONCONVERGENCE"
    else:
        if effects is None or effects.empty:
            raise ValueError("confirmation effects are required after convergence")
        point_positive = bool((effects["mse_gain"] > 0.0).all())
        interval_positive = bool((effects["mse_gain_ci_low"] > 0.0).all())
        seeds_positive = bool(effects["all_state_seeds_positive"].all())
        if point_positive and interval_positive and seeds_positive:
            status = "PASS_S1_PRECISION_REPLICATION"
        else:
            status = "NO_GO_S1_PRECISION_REPLICATION"
    return {
        "status": status,
        "mechanical_checks": mechanical_checks,
        "planning_selected_count": planning_selected_count,
        "state_oracle_is_deployable": False,
        "outcome_oracle_used": False,
        "locked_data_used": False,
        "interpretation_boundary": (
            "PASS only establishes repeatable S1 StateOracle headroom and permits "
            "a separately pre-registered ObservablePolicy development. Any other "
            "formal outcome closes the current shared-state BRSR route."
        ),
    }
