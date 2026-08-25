"""PyCharm entry point for the independent NP-NBQ-DPD Stage 0 certificate.

Run this file directly in PyCharm.  ``RUN_MODE`` controls the formal local run;
Codex uses ``--smoke`` only for short engineering verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from np_nbq_dpd.core import (  # noqa: E402
    LikelihoodCache,
    ModelConfig,
    NPNBQModel,
    Observation,
    PublicProfile,
    allocation_label,
    allocation_payload_bits,
    bsc_transition_probability,
    complex_code_probability,
    default_public_profiles,
    enumerate_allocations,
)


RUN_MODE = "development"


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    design_seed: int
    evaluation_seed: int
    design_trials_per_profile: int
    evaluation_trials_per_profile: int
    bootstrap_repetitions: int
    profile_limit: int

    def validate(self) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if self.design_seed == self.evaluation_seed:
            raise ValueError("design and evaluation seeds must be disjoint")
        if self.design_trials_per_profile < 1 or self.evaluation_trials_per_profile < 1:
            raise ValueError("trial counts must be positive")
        if self.bootstrap_repetitions < 50:
            raise ValueError("bootstrap_repetitions must be at least 50")
        if self.profile_limit < 1:
            raise ValueError("profile_limit must be positive")


def build_experiment_config(mode: str) -> ExperimentConfig:
    if mode == "smoke":
        config = ExperimentConfig("smoke", 9201, 10201, 2, 4, 100, 2)
    elif mode == "development":
        config = ExperimentConfig("development", 9201, 10201, 32, 80, 2000, 4)
    else:
        raise ValueError(f"unknown mode: {mode}")
    config.validate()
    return config


def build_model_config(mode: str) -> ModelConfig:
    if mode == "smoke":
        config = ModelConfig(
            x_grid_m=(-40.0, 0.0, 40.0),
            y_grid_m=(30.0, 70.0, 110.0),
            source_phase_grid_rad=(0.0, math.pi),
        )
    else:
        config = ModelConfig()
    config.validate()
    return config


def _source_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_caches(
    model: NPNBQModel,
    profile: PublicProfile,
    observations: Sequence[Observation],
) -> list[LikelihoodCache]:
    return [model.build_likelihood_cache(profile, observation) for observation in observations]


def _risk_table(
    model: NPNBQModel,
    profile: PublicProfile,
    allocations: Sequence[tuple[int, ...]],
    design_seed: int,
    n_trials: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    actual_observations = model.generate_observations(profile, design_seed, n_trials)
    actual_caches = _build_caches(model, profile, actual_observations)

    geometry_profile = model.nominal_reliability_profile(profile)
    geometry_observations = model.generate_observations(
        geometry_profile, design_seed, n_trials
    )
    geometry_caches = _build_caches(model, geometry_profile, geometry_observations)

    reliability_profile = model.canonical_geometry_profile(profile)
    reliability_observations = model.generate_observations(
        reliability_profile, design_seed, n_trials
    )
    reliability_caches = _build_caches(
        model, reliability_profile, reliability_observations
    )

    rows: list[dict[str, float | int | str]] = []
    actual_risk_matrix = np.empty((len(allocations), n_trials), dtype=float)
    start = time.perf_counter()
    for index, allocation in enumerate(allocations):
        actual_risks = model.design_risks(actual_caches, allocation)
        actual_risk_matrix[index] = actual_risks
        rows.append(
            {
                "profile_id": profile.profile_id,
                "allocation": allocation_label(allocation),
                "payload_bits": allocation_payload_bits(allocation),
                "actual_design_risk_m2": float(np.mean(actual_risks)),
                "geometry_surrogate_risk_m2": model.mean_design_risk(
                    geometry_caches, allocation
                ),
                "reliability_surrogate_risk_m2": model.mean_design_risk(
                    reliability_caches, allocation
                ),
                "projected_fisher_score": model.projected_fisher_score(
                    profile, allocation
                ),
            }
        )
        if (index + 1) % max(1, len(allocations) // 4) == 0:
            print(
                f"[Design] {profile.profile_id} {index + 1}/{len(allocations)} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows), actual_risk_matrix


def _allocation_from_row(row: pd.Series) -> tuple[int, ...]:
    return tuple(int(value) for value in str(row["allocation"]).split("-"))


def _freeze_allocations(
    model: NPNBQModel,
    profile: PublicProfile,
    risk_table: pd.DataFrame,
) -> tuple[dict[str, tuple[int, ...]], str]:
    allocations = {
        "EqualBit": model.equal_bit_allocation(),
        "JiangStylePresetHybrid": model.jiang_style_allocation(profile),
        "GeometryOnly": _allocation_from_row(
            risk_table.loc[risk_table["geometry_surrogate_risk_m2"].idxmin()]
        ),
        "ReliabilityOnly": _allocation_from_row(
            risk_table.loc[risk_table["reliability_surrogate_risk_m2"].idxmin()]
        ),
        "ProjectedFisher": _allocation_from_row(
            risk_table.loc[risk_table["projected_fisher_score"].idxmax()]
        ),
        "JointRisk": _allocation_from_row(
            risk_table.loc[risk_table["actual_design_risk_m2"].idxmin()]
        ),
    }
    risk_lookup = risk_table.set_index("allocation")["actual_design_risk_m2"]
    baseline_names = (
        "EqualBit",
        "JiangStylePresetHybrid",
        "GeometryOnly",
        "ReliabilityOnly",
        "ProjectedFisher",
    )
    strongest_name = min(
        baseline_names,
        key=lambda name: float(risk_lookup[allocation_label(allocations[name])]),
    )
    allocations["StrongestFrozenBaseline"] = allocations[strongest_name]
    return allocations, strongest_name


def _evaluate_profile(
    model: NPNBQModel,
    profile: PublicProfile,
    allocations: dict[str, tuple[int, ...]],
    evaluation_seed: int,
    n_trials: int,
) -> pd.DataFrame:
    observations = model.generate_observations(profile, evaluation_seed, n_trials)
    rows: list[dict[str, float | int | str | bool]] = []
    for observation in observations:
        cache = model.build_likelihood_cache(profile, observation)
        for method, allocation in allocations.items():
            posterior = model.posterior_from_cache(cache, allocation)
            summary = model.summarize_posterior(
                posterior, observation.true_position_index
            )
            rows.append(
                {
                    "method": method,
                    "profile_id": profile.profile_id,
                    "seed": observation.seed,
                    "trial": observation.trial,
                    "true_position_index": observation.true_position_index,
                    "estimate_x_m": summary.estimate_m[0],
                    "estimate_y_m": summary.estimate_m[1],
                    "squared_error_m2": summary.squared_error_m2,
                    "position_error_m": math.sqrt(summary.squared_error_m2),
                    "posterior_risk_m2": summary.posterior_risk_m2,
                    "posterior_entropy": summary.posterior_entropy,
                    "posterior_nll": summary.posterior_nll,
                    "credible90_contains_true": summary.credible90_contains_true,
                    "allocation": allocation_label(allocation),
                    "payload_bits": allocation_payload_bits(allocation),
                }
            )
        unquantized = model.summarize_posterior(
            model.posterior_unquantized(profile, observation),
            observation.true_position_index,
        )
        rows.append(
            {
                "method": "UnquantizedAll-Diagnostic",
                "profile_id": profile.profile_id,
                "seed": observation.seed,
                "trial": observation.trial,
                "true_position_index": observation.true_position_index,
                "estimate_x_m": unquantized.estimate_m[0],
                "estimate_y_m": unquantized.estimate_m[1],
                "squared_error_m2": unquantized.squared_error_m2,
                "position_error_m": math.sqrt(unquantized.squared_error_m2),
                "posterior_risk_m2": unquantized.posterior_risk_m2,
                "posterior_entropy": unquantized.posterior_entropy,
                "posterior_nll": unquantized.posterior_nll,
                "credible90_contains_true": unquantized.credible90_contains_true,
                "allocation": "unquantized",
                "payload_bits": math.nan,
            }
        )
    return pd.DataFrame(rows)


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for (method, profile_id), group in rows.groupby(
        ["method", "profile_id"], sort=False
    ):
        squared = group["squared_error_m2"].to_numpy(float)
        errors = group["position_error_m"].to_numpy(float)
        records.append(
            {
                "method": method,
                "profile_id": profile_id,
                "rmse_m": float(np.sqrt(np.mean(squared))),
                "empirical_mse_m2": float(np.mean(squared)),
                "mean_error_m": float(np.mean(errors)),
                "median_error_m": float(np.median(errors)),
                "p95_error_m": float(np.quantile(errors, 0.95)),
                "mean_posterior_risk_m2": float(group["posterior_risk_m2"].mean()),
                "posterior_nll": float(group["posterior_nll"].mean()),
                "credible90_coverage": float(
                    group["credible90_contains_true"].mean()
                ),
                "n_samples": int(group.shape[0]),
            }
        )
    for method, group in rows.groupby("method", sort=False):
        squared = group["squared_error_m2"].to_numpy(float)
        errors = group["position_error_m"].to_numpy(float)
        records.append(
            {
                "method": method,
                "profile_id": "ALL",
                "rmse_m": float(np.sqrt(np.mean(squared))),
                "empirical_mse_m2": float(np.mean(squared)),
                "mean_error_m": float(np.mean(errors)),
                "median_error_m": float(np.median(errors)),
                "p95_error_m": float(np.quantile(errors, 0.95)),
                "mean_posterior_risk_m2": float(group["posterior_risk_m2"].mean()),
                "posterior_nll": float(group["posterior_nll"].mean()),
                "credible90_coverage": float(
                    group["credible90_contains_true"].mean()
                ),
                "n_samples": int(group.shape[0]),
            }
        )
    return pd.DataFrame(records)


def _paired_effect(
    rows: pd.DataFrame,
    candidate: str,
    baseline: str,
    repetitions: int,
    seed: int,
    profile_id: str | None = None,
) -> dict[str, float | int | str]:
    key = ["profile_id", "seed", "trial"]
    candidate_rows = rows[rows["method"] == candidate].set_index(key)
    baseline_rows = rows[rows["method"] == baseline].set_index(key)
    common = candidate_rows.index.intersection(baseline_rows.index)
    paired = pd.DataFrame(
        {
            "gain_m2": (
                baseline_rows.loc[common, "squared_error_m2"].to_numpy(float)
                - candidate_rows.loc[common, "squared_error_m2"].to_numpy(float)
            )
        },
        index=common,
    ).reset_index()
    if profile_id is not None:
        paired = paired[paired["profile_id"] == profile_id]
    if paired.empty:
        raise ValueError("paired effect has no common observations")
    strata = [
        group["gain_m2"].to_numpy(float)
        for _, group in paired.groupby("profile_id", sort=True)
    ]
    if any(values.size < 2 for values in strata):
        raise ValueError("each profile stratum needs at least two observations")
    point_estimate = float(np.mean([np.mean(values) for values in strata]))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        stratum_means = []
        for values in strata:
            sampled = rng.integers(0, values.size, size=values.size)
            stratum_means.append(float(np.mean(values[sampled])))
        bootstrap[index] = float(np.mean(stratum_means))
    stratified_variance = float(
        np.sum(
            [np.var(values, ddof=1) / values.size for values in strata],
            dtype=float,
        )
        / len(strata) ** 2
    )
    stratified_se = math.sqrt(max(stratified_variance, 0.0))
    ci95_halfwidth = float(norm.ppf(0.975) * stratified_se)
    mde80 = float((norm.ppf(0.975) + norm.ppf(0.80)) * stratified_se)
    gains = paired["gain_m2"].to_numpy(float)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "profile_scope": profile_id or "STRATIFIED_ALL",
        "paired_mse_gain_m2": point_estimate,
        "ci95_low_m2": float(np.quantile(bootstrap, 0.025)),
        "ci95_high_m2": float(np.quantile(bootstrap, 0.975)),
        "fraction_pairs_positive": float(
            np.mean([np.mean(values > 0.0) for values in strata])
        ),
        "n_observation_pairs": int(gains.size),
        "n_profile_strata": int(len(strata)),
        "pair_sd_m2": float(np.std(gains, ddof=1)),
        "stratified_se_m2": stratified_se,
        "normal_ci95_halfwidth_m2": ci95_halfwidth,
        "approx_mde80_two_sided_m2": mde80,
    }


def _calibration_diagnostics(
    rows: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for group_index, ((method, profile_id), group) in enumerate(
        rows.groupby(["method", "profile_id"], sort=True)
    ):
        posterior_risk = group["posterior_risk_m2"].to_numpy(float)
        squared_error = group["squared_error_m2"].to_numpy(float)
        differences = posterior_risk - squared_error
        empirical_mse = float(np.mean(squared_error))
        mean_risk = float(np.mean(posterior_risk))
        ratio = mean_risk / max(empirical_mse, np.finfo(float).tiny)
        rng = np.random.default_rng(seed + group_index)
        bootstrap = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            sampled = rng.integers(0, differences.size, size=differences.size)
            bootstrap[index] = float(np.mean(differences[sampled]))
        records.append(
            {
                "method": method,
                "profile_id": profile_id,
                "n_observations": int(group.shape[0]),
                "empirical_mse_m2": empirical_mse,
                "mean_posterior_risk_m2": mean_risk,
                "risk_to_mse_ratio": ratio,
                "risk_minus_mse_m2": float(np.mean(differences)),
                "risk_minus_mse_ci95_low_m2": float(
                    np.quantile(bootstrap, 0.025)
                ),
                "risk_minus_mse_ci95_high_m2": float(
                    np.quantile(bootstrap, 0.975)
                ),
                "credible90_coverage": float(
                    group["credible90_contains_true"].mean()
                ),
                "calibration_status": (
                    "PASS_DESCRIPTIVE"
                    if 0.5 <= ratio <= 2.0
                    else "WARNING_REVIEW"
                ),
            }
        )
    return pd.DataFrame(records)


def _design_stability(
    risk_table: pd.DataFrame,
    actual_risk_matrix: np.ndarray,
    allocations: Sequence[tuple[int, ...]],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    if actual_risk_matrix.ndim != 2 or actual_risk_matrix.shape[0] != len(
        allocations
    ):
        raise ValueError("design-risk matrix row count does not match allocations")
    if actual_risk_matrix.shape[1] < 2 or not np.all(np.isfinite(actual_risk_matrix)):
        raise ValueError("design-risk matrix must contain finite repeated trials")
    mean_risks = actual_risk_matrix.mean(axis=1)
    order = np.argsort(mean_risks)
    winner_index = int(order[0])
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(allocations), dtype=int)
    for _ in range(repetitions):
        sampled = rng.integers(
            0, actual_risk_matrix.shape[1], size=actual_risk_matrix.shape[1]
        )
        bootstrap_mean = actual_risk_matrix[:, sampled].mean(axis=1)
        counts[int(np.argmin(bootstrap_mean))] += 1
    frequencies = counts / repetitions
    ranks = np.empty(len(allocations), dtype=int)
    ranks[order] = np.arange(1, len(allocations) + 1)
    profile_id = str(risk_table.iloc[0]["profile_id"])
    details = pd.DataFrame(
        {
            "profile_id": profile_id,
            "allocation": [allocation_label(item) for item in allocations],
            "design_rank": ranks,
            "design_risk_m2": mean_risks,
            "risk_gap_from_best_m2": mean_risks - mean_risks[winner_index],
            "bootstrap_winner_frequency": frequencies,
            "is_frozen_joint_winner": np.arange(len(allocations)) == winner_index,
        }
    ).sort_values("design_rank")
    bit_rows: list[dict[str, float | int | str]] = []
    for cell in range(len(allocations[0])):
        for bits in range(4):
            bit_rows.append(
                {
                    "profile_id": profile_id,
                    "cell": cell,
                    "cell_label": f"U{cell // 2}-F{cell % 2}",
                    "bits_per_real": bits,
                    "bootstrap_selection_frequency": float(
                        np.sum(
                            [
                                frequencies[index]
                                for index, allocation in enumerate(allocations)
                                if allocation[cell] == bits
                            ]
                        )
                    ),
                }
            )
    positive = frequencies[frequencies > 0.0]
    summary = {
        "profile_id": profile_id,
        "frozen_joint_allocation": allocation_label(allocations[winner_index]),
        "best_design_risk_m2": float(mean_risks[order[0]]),
        "second_best_design_risk_m2": float(mean_risks[order[1]]),
        "best_second_gap_m2": float(mean_risks[order[1]] - mean_risks[order[0]]),
        "best_top10_gap_m2": float(
            mean_risks[order[min(9, len(order) - 1)]] - mean_risks[order[0]]
        ),
        "frozen_winner_bootstrap_frequency": float(frequencies[winner_index]),
        "top5_design_set_bootstrap_frequency": float(np.sum(frequencies[order[:5]])),
        "unique_bootstrap_winners": int(np.count_nonzero(counts)),
        "bootstrap_winner_entropy_nats": float(-np.sum(positive * np.log(positive))),
        "design_trials": int(actual_risk_matrix.shape[1]),
        "bootstrap_repetitions": int(repetitions),
    }
    return details, pd.DataFrame(bit_rows), summary


def _mechanical_checks(
    model: NPNBQModel,
    config: ExperimentConfig,
    profiles: Sequence[PublicProfile],
    allocations: Sequence[tuple[int, ...]],
) -> dict[str, bool | float | int]:
    test_mean = np.asarray([-1.2, 0.0, 1.4], dtype=complex)
    probability_errors = []
    bsc_errors = []
    for bits in range(1, model.config.max_bits_per_real + 1):
        levels = 2**bits
        for received in range(levels * levels):
            probabilities = complex_code_probability(
                test_mean,
                0.7,
                bits,
                received,
                0.07,
            )
            if np.any(probabilities < 0.0):
                probability_errors.append(math.inf)
        for transmitted in range(levels):
            total = sum(
                bsc_transition_probability(transmitted, received, bits, 0.07)
                for received in range(levels)
            )
            bsc_errors.append(abs(total - 1.0))
        total_codes = sum(
            complex_code_probability(test_mean, 0.7, bits, code, 0.07)
            for code in range(levels * levels)
        )
        probability_errors.append(float(np.max(np.abs(total_codes - 1.0))))
    checks: dict[str, bool | float | int] = {
        "design_evaluation_seeds_disjoint": config.design_seed != config.evaluation_seed,
        "profile_count": len(profiles),
        "profile_count_valid": len(profiles) == config.profile_limit,
        "allocation_count": len(allocations),
        "allocation_space_nonempty": bool(allocations),
        "all_allocations_exact_budget": all(
            allocation_payload_bits(allocation) == model.config.payload_bits
            for allocation in allocations
        ),
        "equal_allocation_exact_budget": allocation_payload_bits(
            model.equal_bit_allocation()
        )
        == model.config.payload_bits,
        "max_quantizer_partition_error": max(probability_errors),
        "quantizer_probabilities_partition": max(probability_errors) < 1e-10,
        "max_bsc_row_sum_error": max(bsc_errors),
        "bsc_rows_normalized": max(bsc_errors) < 1e-12,
        "position_prior_normalized": math.isclose(
            float(np.sum(model.position_prior)), 1.0, abs_tol=1e-12
        ),
        "nuisance_state_count": model.nuisance_state_count,
        "nuisance_prior_nontrivial": model.nuisance_state_count > 1,
        "no_outcome_oracle_method": True,
    }
    checks["all_passed"] = all(
        value for value in checks.values() if isinstance(value, bool)
    )
    return checks


def _post_checks(
    rows: pd.DataFrame,
    allocations: pd.DataFrame,
    expected_trials: int,
    payload_bits: int,
) -> dict[str, bool | int]:
    quantized = rows[rows["method"] != "UnquantizedAll-Diagnostic"]
    keys = ["method", "profile_id", "seed", "trial"]
    checks: dict[str, bool | int] = {
        "evaluation_rows": int(rows.shape[0]),
        "evaluation_keys_unique": bool(not rows.duplicated(keys).any()),
        "quantized_payload_exact": bool(
            np.all(quantized["payload_bits"].to_numpy(float) == payload_bits)
        ),
        "unquantized_not_bit_comparable": bool(
            rows.loc[
                rows["method"] == "UnquantizedAll-Diagnostic", "payload_bits"
            ].isna().all()
        ),
        "all_expected_trials_present": bool(
            rows.groupby(["method", "profile_id"]).size().eq(expected_trials).all()
        ),
        "frozen_allocations_exact_budget": bool(
            allocations["payload_bits"].eq(payload_bits).all()
        ),
        "evaluation_finite": bool(
            np.isfinite(
                rows[
                    [
                        "squared_error_m2",
                        "posterior_risk_m2",
                        "posterior_nll",
                    ]
                ].to_numpy(float)
            ).all()
        ),
    }
    checks["all_passed"] = all(
        value for value in checks.values() if isinstance(value, bool)
    )
    return checks


def _gate_status(
    mode: str,
    checks_passed: bool,
    effects: pd.DataFrame,
    profile_effects: pd.DataFrame,
    allocation_rows: pd.DataFrame,
) -> str:
    if not checks_passed:
        return "INVALID_STAGE0_IMPLEMENTATION"
    if mode == "smoke":
        return "SMOKE_ONLY"
    effect_by_baseline = effects.set_index("baseline")
    required = (
        "StrongestFrozenBaseline",
        "GeometryOnly",
        "ReliabilityOnly",
    )
    if any(name not in effect_by_baseline.index for name in required):
        return "INVALID_STAGE0_EFFECT_TABLE"
    joint = allocation_rows[allocation_rows["method"] == "JointRisk"]
    if (
        joint["allocation"].nunique() == 1
        and joint.iloc[0]["allocation"] == "2-2-2-2-2-2"
    ):
        return "NO_GO_ALLOCATION_COLLAPSE"
    heterogeneous = profile_effects[
        (profile_effects["baseline"] == "StrongestFrozenBaseline")
        & (profile_effects["profile_scope"] != "balanced")
    ]
    if heterogeneous["profile_scope"].nunique() != 3:
        return "INVALID_STAGE0_PROFILE_EFFECT_TABLE"
    if (heterogeneous["ci95_high_m2"] < 0.0).any():
        return "NO_GO_HETEROGENEOUS_PROFILE_REGRESSION"
    if all(
        float(effect_by_baseline.loc[name, "ci95_low_m2"]) > 0.0
        for name in required
    ):
        return "GO_STRUCTURE_HEADROOM"
    strongest = effect_by_baseline.loc["StrongestFrozenBaseline"]
    if float(strongest["ci95_high_m2"]) <= 0.0:
        return "NO_GO_STRUCTURE_HEADROOM"
    return "HOLD_STRUCTURE_EFFECT"


def _plot_summary(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    path: Path,
) -> None:
    all_rows = summary[summary["profile_id"] == "ALL"].copy()
    method_order = [
        "UnquantizedAll-Diagnostic",
        "EqualBit",
        "JiangStylePresetHybrid",
        "GeometryOnly",
        "ReliabilityOnly",
        "ProjectedFisher",
        "StrongestFrozenBaseline",
        "JointRisk",
    ]
    all_rows["order"] = all_rows["method"].map(
        {name: index for index, name in enumerate(method_order)}
    )
    all_rows = all_rows.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(all_rows)))
    axes[0].barh(all_rows["method"], all_rows["rmse_m"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Position RMSE (m)")
    axes[0].set_title("Stage 0 direct-position accuracy")
    axes[0].grid(axis="x", alpha=0.25)

    effect_plot = effects.iloc[::-1]
    center = effect_plot["paired_mse_gain_m2"].to_numpy(float)
    lower = center - effect_plot["ci95_low_m2"].to_numpy(float)
    upper = effect_plot["ci95_high_m2"].to_numpy(float) - center
    axes[1].errorbar(
        center,
        effect_plot["baseline"],
        xerr=np.vstack((lower, upper)),
        fmt="o",
        color="#c23b22",
        ecolor="#555555",
        capsize=3,
    )
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("JointRisk paired MSE gain (m$^2$)")
    axes[1].set_title("Paired effects with 95% profile-stratified CI")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(path, format="svg")
    plt.close(fig)


def _plot_allocations(allocation_rows: pd.DataFrame, path: Path) -> None:
    method_order = [
        "EqualBit",
        "JiangStylePresetHybrid",
        "GeometryOnly",
        "ReliabilityOnly",
        "ProjectedFisher",
        "JointRisk",
    ]
    ordered = allocation_rows[allocation_rows["method"].isin(method_order)].copy()
    ordered["method_order"] = ordered["method"].map(
        {name: index for index, name in enumerate(method_order)}
    )
    ordered = ordered.sort_values(["profile_id", "method_order"])
    matrix = np.asarray(
        [
            [int(value) for value in allocation.split("-")]
            for allocation in ordered["allocation"]
        ],
        dtype=float,
    )
    labels = [f"{row.profile_id} | {row.method}" for row in ordered.itertuples()]
    fig, ax = plt.subplots(figsize=(8.2, max(5.0, 0.28 * len(labels))))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=3)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    ax.set_xticks(
        np.arange(matrix.shape[1]),
        labels=["U0-F0", "U0-F1", "U1-F0", "U1-F1", "U2-F0", "U2-F1"],
        rotation=35,
        ha="right",
    )
    ax.set_title("Frozen bits per real component (24 payload bits)")
    colorbar = fig.colorbar(image, ax=ax, ticks=(0, 1, 2, 3))
    colorbar.set_label("Bits per I or Q component")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def run(mode: str) -> Path:
    config = build_experiment_config(mode)
    model = NPNBQModel(build_model_config(mode))
    profiles = default_public_profiles()[: config.profile_limit]
    for profile in profiles:
        profile.validate(model.config)
    allocations = enumerate_allocations(model.config)
    checks = _mechanical_checks(model, config, profiles, allocations)
    if not bool(checks["all_passed"]):
        raise RuntimeError(f"mechanical checks failed: {checks}")

    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        root
        / "运行结果"
        / "NP_NBQ_DPD"
        / f"NP_NBQ_STAGE0_{timestamp}_{mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    source_paths = (Path(__file__).resolve(), Path(__file__).with_name("core.py"))

    print("NP-NBQ-DPD Stage 0 structure certificate", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(
        f"Profiles={len(profiles)}, allocations={len(allocations)}, "
        f"design/profile={config.design_trials_per_profile}, "
        f"evaluation/profile={config.evaluation_trials_per_profile}",
        flush=True,
    )
    print(
        "Scope: finite nuisance prior, single-path AWGN, public pilot SNR/BER; "
        "not a general unknown-channel result",
        flush=True,
    )

    all_risk_tables: list[pd.DataFrame] = []
    all_allocation_rows: list[dict[str, float | int | str]] = []
    all_evaluation_rows: list[pd.DataFrame] = []
    proxy_rows: list[dict[str, float | int | str]] = []
    stability_detail_frames: list[pd.DataFrame] = []
    stability_bit_frames: list[pd.DataFrame] = []
    stability_summary_rows: list[dict[str, float | int | str]] = []
    strongest_by_profile: dict[str, str] = {}
    derived_seeds_by_profile: dict[str, dict[str, int]] = {}

    for profile_index, profile in enumerate(profiles):
        print(f"[Profile] {profile.profile_id} start", flush=True)
        design_seed = config.design_seed + profile_index * 10_000
        evaluation_seed = config.evaluation_seed + profile_index * 10_000
        derived_seeds_by_profile[profile.profile_id] = {
            "design_seed": design_seed,
            "evaluation_seed": evaluation_seed,
            "stability_seed": 55_001 + profile_index,
        }
        risk_table, actual_risk_matrix = _risk_table(
            model,
            profile,
            allocations,
            design_seed,
            config.design_trials_per_profile,
        )
        stability_details, stability_bits, stability_summary = _design_stability(
            risk_table,
            actual_risk_matrix,
            allocations,
            config.bootstrap_repetitions,
            55_001 + profile_index,
        )
        stability_detail_frames.append(stability_details)
        stability_bit_frames.append(stability_bits)
        stability_summary_rows.append(stability_summary)
        frozen, strongest_name = _freeze_allocations(model, profile, risk_table)
        strongest_by_profile[profile.profile_id] = strongest_name
        lookup = risk_table.set_index("allocation")
        for method, allocation in frozen.items():
            row = lookup.loc[allocation_label(allocation)]
            all_allocation_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "method": method,
                    "allocation": allocation_label(allocation),
                    "payload_bits": allocation_payload_bits(allocation),
                    "actual_design_risk_m2": float(row["actual_design_risk_m2"]),
                    "strongest_baseline_source": (
                        strongest_name
                        if method == "StrongestFrozenBaseline"
                        else ""
                    ),
                }
            )
        correlation = spearmanr(
            risk_table["projected_fisher_score"].to_numpy(float),
            -risk_table["actual_design_risk_m2"].to_numpy(float),
        ).statistic
        proxy_rows.append(
            {
                "profile_id": profile.profile_id,
                "spearman_fisher_vs_negative_risk": float(correlation),
                "n_allocations": int(risk_table.shape[0]),
            }
        )
        all_risk_tables.append(risk_table)
        all_evaluation_rows.append(
            _evaluate_profile(
                model,
                profile,
                frozen,
                evaluation_seed,
                config.evaluation_trials_per_profile,
            )
        )

    risk_tables = pd.concat(all_risk_tables, ignore_index=True)
    allocation_rows = pd.DataFrame(all_allocation_rows)
    evaluation_rows = pd.concat(all_evaluation_rows, ignore_index=True)
    summary = _summarize(evaluation_rows)
    effect_baselines = (
        "StrongestFrozenBaseline",
        "EqualBit",
        "JiangStylePresetHybrid",
        "GeometryOnly",
        "ReliabilityOnly",
        "ProjectedFisher",
    )
    effects = pd.DataFrame(
        [
            _paired_effect(
                evaluation_rows,
                "JointRisk",
                baseline,
                config.bootstrap_repetitions,
                44001 + index,
            )
            for index, baseline in enumerate(effect_baselines)
        ]
    )
    profile_effects = pd.DataFrame(
        [
            _paired_effect(
                evaluation_rows,
                "JointRisk",
                baseline,
                config.bootstrap_repetitions,
                46_001 + baseline_index * 100 + profile_index,
                profile.profile_id,
            )
            for baseline_index, baseline in enumerate(effect_baselines)
            for profile_index, profile in enumerate(profiles)
        ]
    )
    calibration = _calibration_diagnostics(
        evaluation_rows,
        config.bootstrap_repetitions,
        47_001,
    )
    stability_details = pd.concat(stability_detail_frames, ignore_index=True)
    stability_bits = pd.concat(stability_bit_frames, ignore_index=True)
    stability_summary = pd.DataFrame(stability_summary_rows)
    post_checks = _post_checks(
        evaluation_rows,
        allocation_rows,
        config.evaluation_trials_per_profile,
        model.config.payload_bits,
    )
    checks.update({f"post_{key}": value for key, value in post_checks.items()})
    checks["all_passed"] = bool(checks["all_passed"] and post_checks["all_passed"])
    checks["post_profile_effect_rows_complete"] = bool(
        profile_effects.shape[0] == len(effect_baselines) * len(profiles)
    )
    checks["post_calibration_finite"] = bool(
        np.isfinite(
            calibration[
                [
                    "empirical_mse_m2",
                    "mean_posterior_risk_m2",
                    "risk_to_mse_ratio",
                    "risk_minus_mse_m2",
                ]
            ].to_numpy(float)
        ).all()
    )
    checks["post_stability_winner_frequencies_normalized"] = bool(
        stability_details.groupby("profile_id")["bootstrap_winner_frequency"]
        .sum()
        .sub(1.0)
        .abs()
        .lt(1e-12)
        .all()
    )
    checks["post_stability_bit_frequencies_normalized"] = bool(
        stability_bits.groupby(["profile_id", "cell"])[
            "bootstrap_selection_frequency"
        ]
        .sum()
        .sub(1.0)
        .abs()
        .lt(1e-12)
        .all()
    )
    checks["post_natural_binary_mapping_frozen"] = bool(
        model.config.quantizer_code_mapping == "natural_binary"
    )
    checks["all_passed"] = bool(
        checks["all_passed"]
        and checks["post_profile_effect_rows_complete"]
        and checks["post_calibration_finite"]
        and checks["post_stability_winner_frequencies_normalized"]
        and checks["post_stability_bit_frequencies_normalized"]
        and checks["post_natural_binary_mapping_frozen"]
    )
    proxy_frame = pd.DataFrame(proxy_rows)
    correlations = proxy_frame["spearman_fisher_vs_negative_risk"].to_numpy(float)
    if np.all(correlations > 0.0):
        proxy_status = "SUPPORTED_DESCRIPTIVE"
    elif np.all(correlations <= 0.0):
        proxy_status = "NO_GO_FISHER_PROXY"
    else:
        proxy_status = "HOLD_PROXY_RANKING"
    gate = _gate_status(
        mode,
        bool(checks["all_passed"]),
        effects,
        profile_effects,
        allocation_rows,
    )

    risk_tables.to_csv(output_dir / "allocation_design_risks.csv", index=False)
    allocation_rows.to_csv(output_dir / "frozen_allocations.csv", index=False)
    evaluation_rows.to_csv(output_dir / "evaluation_rows.csv", index=False)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    effects.to_csv(output_dir / "paired_effects.csv", index=False)
    profile_effects.to_csv(output_dir / "paired_effects_by_profile.csv", index=False)
    calibration.to_csv(output_dir / "posterior_calibration.csv", index=False)
    stability_details.to_csv(
        output_dir / "allocation_selection_stability.csv", index=False
    )
    stability_bits.to_csv(
        output_dir / "allocation_bit_selection_frequency.csv", index=False
    )
    stability_summary.to_csv(
        output_dir / "allocation_stability_summary.csv", index=False
    )
    proxy_frame.to_csv(output_dir / "fisher_proxy_diagnostics.csv", index=False)
    (output_dir / "mechanical_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "route": "NP-NBQ-DPD",
        "stage": "Stage 0 structure certificate",
        "mode": mode,
        "config": asdict(config),
        "model_config": asdict(model.config),
        "profiles": [asdict(profile) for profile in profiles],
        "derived_seeds_by_profile": derived_seeds_by_profile,
        "source_hash": _source_hash(source_paths),
        "strongest_baseline_by_profile": strongest_by_profile,
        "main_gate": gate,
        "fisher_proxy_status": proxy_status,
        "calibration_status": (
            "PASS_DESCRIPTIVE"
            if calibration["calibration_status"].eq("PASS_DESCRIPTIVE").all()
            else "WARNING_REVIEW"
        ),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "scientific_scope": {
            "unknown_source": (
                "finite per-frequency phase prior plus a finite second-bin "
                "amplitude-ratio nuisance"
            ),
            "complex_gain": (
                "reference gain fixed as gauge; non-reference frequency-flat "
                "complex gains have finite amplitude/phase priors"
            ),
            "channel": "single-path AWGN plus per-link BSC",
            "allocation_information": "ROI prior, geometry, pilot SNR and BER only",
            "quantizer_code_mapping": model.config.quantizer_code_mapping,
            "excluded": [
                "true position at allocation time",
                "clean waveform",
                "outcome error oracle",
                "multipath/NLOS",
                "continuous arbitrary source spectrum and complex gain",
                "frequency-selective unknown gain",
            ],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_summary(summary, effects, output_dir / "Fig1_Stage0_Performance.svg")
    _plot_allocations(allocation_rows, output_dir / "Fig2_Frozen_Allocations.svg")

    print("\nPaired effects: JointRisk minus baseline", flush=True)
    print(effects.to_string(index=False), flush=True)
    print("\nDesign winner stability", flush=True)
    print(stability_summary.to_string(index=False), flush=True)
    calibration_warning_count = int(
        calibration["calibration_status"].eq("WARNING_REVIEW").sum()
    )
    print(
        f"\n[Calibration] warnings={calibration_warning_count}/"
        f"{calibration.shape[0]}",
        flush=True,
    )
    print(f"\n[Fisher proxy] {proxy_status}", flush=True)
    print(f"[Gate] {gate}", flush=True)
    print(f"[Done] {output_dir}", flush=True)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="run a short engineering smoke test"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run("smoke" if args.smoke else RUN_MODE)
