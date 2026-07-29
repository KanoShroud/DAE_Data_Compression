"""BRSR 路线可行性阶段1：候选池 x 似然 x 单动作Oracle证书。

PyCharm直接运行本文件时默认使用 ``development``。Codex验证使用
``--mode smoke``。本脚本不读取或覆盖任何历史结果。
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
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brsr_route_feasibility.feasibility import (  # noqa: E402
    ACQUIRED_BIN_PAYLOAD_BITS,
    BASE_PAYLOAD_BITS,
    ONE_ACTION_FEEDBACK_BITS,
    FeasibilityConfig,
    LikelihoodName,
    PoolName,
    action_levels,
    build_candidate_pools,
    build_likelihood_cases,
    candidate_pool_rows,
    config_for_mode,
    evaluate_actual_actions,
    expected_action_risks,
    linear_likelihood_context,
    linear_template,
    observation_from_context,
    oracle_rho_model,
    phase_likelihood_context,
    physical_public_scale,
    public_scale_matches_model,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer  # noqa: E402
from brsr_tdoa.brsr_core import tau_posterior  # noqa: E402
from brsr_tdoa.brsr_stage_b import evaluate_fixed, state_from_levels  # noqa: E402
from brsr_tdoa.brsr_waveform import (  # noqa: E402
    StructuredTrial,
    WaveformConfig,
    candidate_fft_values,
    estimate_candidate_statistics,
    generate_structured_trial,
    observe_trial,
)


PYCHARM_RUN_MODE = "development"
DEPLOYABLE_LIKELIHOODS = (
    "Phase-Discrete24",
    "Phase-Marginal",
    "Linear-Marginal",
)
PRIMARY_LIKELIHOOD = "Linear-Marginal"
QUADRATURE_ESTIMATE_GRID_FRACTION = 0.10
QUADRATURE_RISK_GRID_SQUARED_FRACTION = 0.01
QUADRATURE_POSTERIOR_L1_MAX = 0.10
QUADRATURE_DIAGNOSTIC_GRIDS = (
    (3, 8),
    (5, 16),
    (5, 32),
    (5, 64),
    (9, 128),
)
QUADRATURE_REFERENCE_GRID = QUADRATURE_DIAGNOSTIC_GRIDS[-1]


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


def _generate_trials(
    seeds: tuple[int, ...],
    trials_per_seed: int,
    waveform: WaveformConfig,
) -> dict[tuple[int, int], StructuredTrial]:
    trials = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for trial in range(trials_per_seed):
            trials[(seed, trial)] = generate_structured_trial(rng, waveform)
    return trials


def _build_models(
    pools: dict[PoolName, np.ndarray],
    prior_trials: dict[tuple[int, int], StructuredTrial],
    *,
    waveform: WaveformConfig,
    config: FeasibilityConfig,
) -> tuple[
    dict[PoolName, dict[LikelihoodName, object]],
    pd.DataFrame,
]:
    models: dict[PoolName, dict[LikelihoodName, object]] = {}
    rows = []
    trial_list = list(prior_trials.values())
    for pool, bins in pools.items():
        power, covariance, diagnostics = estimate_candidate_statistics(
            trial_list,
            bins,
        )
        cases = build_likelihood_cases(
            waveform=waveform,
            bins=bins,
            source_power=power,
            tau_step=config.tau_step,
            amplitude_nodes=config.marginal_amplitude_nodes,
            phase_nodes=config.marginal_phase_nodes,
        )
        models[pool] = cases
        rows.append(
            {
                "pool": pool,
                "n_prior_trials": len(trial_list),
                "source_power_min": float(np.min(power)),
                "source_power_max": float(np.max(power)),
                "covariance_rank": int(np.linalg.matrix_rank(covariance)),
                **diagnostics,
            }
        )
    return models, pd.DataFrame(rows)


def _likelihood_observations(
    *,
    stage: str,
    seed: int,
    trial_index: int,
    snr_db: float,
    trial: StructuredTrial,
    y_a: np.ndarray,
    y_b: np.ndarray,
    sigma2: float,
    bins: np.ndarray,
    cases: dict[LikelihoodName, object],
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
) -> dict[LikelihoodName, tuple[object, object, float]]:
    y_a_selected = candidate_fft_values(y_a, bins)
    y_b_selected = candidate_fft_values(y_b, bins)
    discrete = cases["Phase-Discrete24"]
    marginal = cases["Phase-Marginal"]
    scale = physical_public_scale(discrete.source_power, sigma2)
    template = linear_template(y_a, bins, marginal.tau_values, waveform)

    observations: dict[LikelihoodName, tuple[object, object, float]] = {}
    for label, model in (
        ("Phase-Discrete24", discrete),
        ("Phase-Marginal", marginal),
    ):
        context = phase_likelihood_context(
            y_a_selected=y_a_selected,
            sigma2=sigma2,
            scale=scale,
            model=model,
        )
        observation = observation_from_context(
            stage=stage,
            seed=seed,
            trial=trial_index,
            snr_db=snr_db,
            true_tau=trial.true_tau,
            true_rho=trial.true_rho,
            y_b_selected=y_b_selected,
            scale=scale,
            context=context,
            quantizer=quantizer,
        )
        observations[label] = (
            observation,
            model,
            public_scale_matches_model(model.source_power, sigma2, model),
        )

    linear_context = linear_likelihood_context(
        template=template,
        sigma2=sigma2,
        scale=scale,
        model=marginal,
    )
    observations["Linear-Marginal"] = (
        observation_from_context(
            stage=stage,
            seed=seed,
            trial=trial_index,
            snr_db=snr_db,
            true_tau=trial.true_tau,
            true_rho=trial.true_rho,
            y_b_selected=y_b_selected,
            scale=scale,
            context=linear_context,
            quantizer=quantizer,
        ),
        marginal,
        public_scale_matches_model(marginal.source_power, sigma2, marginal),
    )

    oracle_model = oracle_rho_model(marginal, trial.true_rho)
    oracle_context = linear_likelihood_context(
        template=template,
        sigma2=sigma2,
        scale=scale,
        model=oracle_model,
    )
    observations["Linear-OracleRho"] = (
        observation_from_context(
            stage=stage,
            seed=seed,
            trial=trial_index,
            snr_db=snr_db,
            true_tau=trial.true_tau,
            true_rho=trial.true_rho,
            y_b_selected=y_b_selected,
            scale=scale,
            context=oracle_context,
            quantizer=quantizer,
        ),
        oracle_model,
        math.nan,
    )
    return observations


def _collect_actions(
    *,
    stage: str,
    trials: dict[tuple[int, int], StructuredTrial],
    config: FeasibilityConfig,
    waveform: WaveformConfig,
    pools: dict[PoolName, np.ndarray],
    models: dict[PoolName, dict[LikelihoodName, object]],
    quantizer: NestedComplexQuantizer,
    score_pool: PoolName | None,
    output: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(trials) * len(config.snr_db)
    done = 0
    start = time.perf_counter()
    for snr_db in config.snr_db:
        print(f"[{stage}] SNR={snr_db:g} dB start", flush=True)
        for (seed, trial_index), trial in trials.items():
            y_a, y_b, sigma2 = observe_trial(trial, snr_db)
            score_this = (
                stage == "evaluation"
                and score_pool is not None
                and trial_index < config.score_trials_per_seed
            )
            for pool, bins in pools.items():
                observations = _likelihood_observations(
                    stage=stage,
                    seed=seed,
                    trial_index=trial_index,
                    snr_db=snr_db,
                    trial=trial,
                    y_a=y_a,
                    y_b=y_b,
                    sigma2=sigma2,
                    bins=bins,
                    cases=models[pool],
                    waveform=waveform,
                    quantizer=quantizer,
                )
                for likelihood, (observation, model, scale_delta) in observations.items():
                    base, action_rows = evaluate_actual_actions(
                        observation,
                        quantizer=quantizer,
                        model=model,
                    )
                    expected = {}
                    if (
                        score_this
                        and pool == score_pool
                        and likelihood in DEPLOYABLE_LIKELIHOODS
                    ):
                        expected = expected_action_risks(
                            observation,
                            quantizer=quantizer,
                            model=model,
                        )
                    for action in action_rows:
                        rows.append(
                            {
                                "stage": stage,
                                "seed": seed,
                                "trial": trial_index,
                                "cluster_id": f"{seed}:{trial_index}",
                                "snr_db": snr_db,
                                "pool": pool,
                                "likelihood": likelihood,
                                "true_tau": trial.true_tau,
                                "true_rho_real": trial.true_rho.real,
                                "true_rho_imag": trial.true_rho.imag,
                                **base,
                                **action,
                                "predicted_expected_risk": expected.get(
                                    int(action["position"]),
                                    math.nan,
                                ),
                                "public_scale_delta_vs_legacy": scale_delta,
                            }
                        )
            done += 1
            if done % config.progress_every == 0 or done == total:
                elapsed = time.perf_counter() - start
                rate = elapsed / done
                print(
                    f"[{stage}] {done}/{total} elapsed={elapsed:.1f}s "
                    f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(
            output / f"{stage}_action_rows.partial.csv",
            index=False,
        )
    return pd.DataFrame(rows)


def _select_plans(calibration: pd.DataFrame) -> tuple[pd.DataFrame, PoolName]:
    plan_rows = []
    for (pool, likelihood), group in calibration.groupby(
        ["pool", "likelihood"],
        sort=True,
    ):
        overall = group.groupby("position")["squared_error_samples2"].mean()
        best_position = int(overall.sort_values(kind="stable").index[0])
        plan_rows.append(
            {
                "plan": "BestStatic-OneAcquire",
                "pool": pool,
                "likelihood": likelihood,
                "snr_db": math.nan,
                "position": best_position,
                "calibration_mse": float(overall.loc[best_position]),
            }
        )
        for snr_db, snr_group in group.groupby("snr_db", sort=True):
            by_action = snr_group.groupby("position")["squared_error_samples2"].mean()
            position = int(by_action.sort_values(kind="stable").index[0])
            plan_rows.append(
                {
                    "plan": "SNR-only-OneAcquire",
                    "pool": pool,
                    "likelihood": likelihood,
                    "snr_db": snr_db,
                    "position": position,
                    "calibration_mse": float(by_action.loc[position]),
                }
            )
    plans = pd.DataFrame(plan_rows)

    linear = calibration[calibration["likelihood"].eq("Linear-Marginal")]
    pool_scores = []
    for pool, group in linear.groupby("pool", sort=True):
        oracle = group.groupby(["seed", "trial", "snr_db"])[
            "squared_error_samples2"
        ].min()
        snr_plans = plans[
            plans["plan"].eq("SNR-only-OneAcquire")
            & plans["pool"].eq(pool)
            & plans["likelihood"].eq("Linear-Marginal")
        ]
        selected = group.merge(
            snr_plans[["snr_db", "position"]],
            on=["snr_db", "position"],
            validate="many_to_one",
        )
        baseline = selected.set_index(["seed", "trial", "snr_db"])[
            "squared_error_samples2"
        ]
        aligned = oracle.to_frame("oracle").join(
            baseline.rename("baseline"),
            how="inner",
            validate="one_to_one",
        )
        pool_scores.append(
            (
                str(pool),
                float((aligned["baseline"] - aligned["oracle"]).mean()),
            )
        )
    if not pool_scores:
        raise RuntimeError("no calibration pool score was produced")
    score_pool = max(pool_scores, key=lambda item: (item[1], item[0]))[0]
    return plans, score_pool  # type: ignore[return-value]


def _quadrature_convergence(
    *,
    config: FeasibilityConfig,
    waveform: WaveformConfig,
    trials: dict[tuple[int, int], StructuredTrial],
    pool: PoolName,
    bins: np.ndarray,
    source_power: np.ndarray,
    plans: pd.DataFrame,
    quantizer: NestedComplexQuantizer,
) -> pd.DataFrame:
    """Compare the configured marginal quadrature against a denser reference."""

    schemes = QUADRATURE_DIAGNOSTIC_GRIDS
    configured_grid = (
        config.marginal_amplitude_nodes,
        config.marginal_phase_nodes,
    )
    if configured_grid not in schemes:
        raise ValueError(
            f"configured quadrature {configured_grid} is missing from "
            "the convergence diagnostic"
        )
    selected_trials = [
        (key, trial)
        for key, trial in sorted(trials.items())
        if key[1] == 0
    ]
    rows = []
    for snr_db in config.snr_db:
        plan = plans[
            plans["plan"].eq("SNR-only-OneAcquire")
            & plans["pool"].eq(pool)
            & plans["likelihood"].eq("Linear-Marginal")
            & plans["snr_db"].eq(snr_db)
        ]
        if len(plan) != 1:
            raise RuntimeError("quadrature diagnostic plan is missing")
        position = int(plan.iloc[0]["position"])
        per_trial: list[list[dict[str, object]]] = []
        for (seed, trial_index), trial in selected_trials:
            y_a, y_b, sigma2 = observe_trial(trial, snr_db)
            y_b_selected = candidate_fft_values(y_b, bins)
            scale = physical_public_scale(source_power, sigma2)
            trial_rows = []
            for amplitude_nodes, phase_nodes in schemes:
                cases = build_likelihood_cases(
                    waveform=waveform,
                    bins=bins,
                    source_power=source_power,
                    tau_step=config.tau_step,
                    amplitude_nodes=amplitude_nodes,
                    phase_nodes=phase_nodes,
                )
                model = cases["Linear-Marginal"]
                template = linear_template(y_a, bins, model.tau_values, waveform)
                context = linear_likelihood_context(
                    template=template,
                    sigma2=sigma2,
                    scale=scale,
                    model=model,
                )
                observation = observation_from_context(
                    stage="quadrature",
                    seed=seed,
                    trial=trial_index,
                    snr_db=snr_db,
                    true_tau=trial.true_tau,
                    true_rho=trial.true_rho,
                    y_b_selected=y_b_selected,
                    scale=scale,
                    context=context,
                    quantizer=quantizer,
                )
                levels = action_levels(model, position)
                result = evaluate_fixed(
                    observation,
                    method="quadrature",
                    levels=levels,
                    quantizer=quantizer,
                    model=model,
                )
                state = state_from_levels(
                    observation,
                    levels,
                    quantizer=quantizer,
                    model=model,
                )
                trial_rows.append(
                    {
                        "seed": seed,
                        "trial": trial_index,
                        "snr_db": snr_db,
                        "pool": pool,
                        "amplitude_nodes": amplitude_nodes,
                        "phase_nodes": phase_nodes,
                        "n_rho": model.n_rho,
                        "estimate_tau": result.estimate,
                        "posterior_risk": result.posterior_risk,
                        "tau_posterior": tau_posterior(state, model),
                    }
                )
            per_trial.append(trial_rows)
        for trial_rows in per_trial:
            reference = trial_rows[-1]
            for item in trial_rows:
                posterior_l1 = float(
                    np.sum(
                        np.abs(
                            np.asarray(item["tau_posterior"])
                            - np.asarray(reference["tau_posterior"])
                        )
                    )
                )
                rows.append(
                    {
                        key: value
                        for key, value in item.items()
                        if key != "tau_posterior"
                    }
                    | {
                        "reference_amplitude_nodes": reference["amplitude_nodes"],
                        "reference_phase_nodes": reference["phase_nodes"],
                        "estimate_abs_diff_vs_reference": abs(
                            float(item["estimate_tau"])
                            - float(reference["estimate_tau"])
                        ),
                        "risk_abs_diff_vs_reference": abs(
                            float(item["posterior_risk"])
                            - float(reference["posterior_risk"])
                        ),
                        "tau_posterior_l1_vs_reference": posterior_l1,
                    }
                )
    return pd.DataFrame(rows)


def _method_rows(actions: pd.DataFrame, plans: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["seed", "trial", "cluster_id", "snr_db", "pool", "likelihood"]
    for key, group in actions.groupby(keys, sort=False):
        metadata = dict(zip(keys, key, strict=True))
        true_tau = float(group["true_tau"].iloc[0])
        pool = str(metadata["pool"])
        likelihood = str(metadata["likelihood"])

        selections: list[tuple[str, pd.Series]] = [
            (
                "Oracle-OneAcquire",
                group.sort_values(
                    ["squared_error_samples2", "position"],
                    kind="stable",
                ).iloc[0],
            )
        ]
        best_plan = plans[
            plans["plan"].eq("BestStatic-OneAcquire")
            & plans["pool"].eq(pool)
            & plans["likelihood"].eq(likelihood)
        ]
        snr_plan = plans[
            plans["plan"].eq("SNR-only-OneAcquire")
            & plans["pool"].eq(pool)
            & plans["likelihood"].eq(likelihood)
            & plans["snr_db"].eq(float(metadata["snr_db"]))
        ]
        if len(best_plan) != 1 or len(snr_plan) != 1:
            raise RuntimeError("frozen calibration plan is missing or duplicated")
        for label, plan in (
            ("BestStatic-OneAcquire", best_plan),
            ("SNR-only-OneAcquire", snr_plan),
        ):
            position = int(plan.iloc[0]["position"])
            selections.append(
                (label, group[group["position"].eq(position)].iloc[0])
            )
        if likelihood in DEPLOYABLE_LIKELIHOODS:
            scored = group[np.isfinite(group["predicted_expected_risk"])]
            if not scored.empty:
                selections.append(
                    (
                        "RiskGreedy-OneAcquire",
                        scored.sort_values(
                            ["predicted_expected_risk", "position"],
                            kind="stable",
                        ).iloc[0],
                    )
                )

        for method, selected in selections:
            feedback_bits = (
                ONE_ACTION_FEEDBACK_BITS
                if method in ("Oracle-OneAcquire", "RiskGreedy-OneAcquire")
                else 0
            )
            rows.append(
                {
                    **metadata,
                    "method": method,
                    "true_tau": true_tau,
                    "position": int(selected["position"]),
                    "bin_index": int(selected["bin_index"]),
                    "estimate_tau": float(selected["estimate_tau"]),
                    "error_samples": float(selected["error_samples"]),
                    "squared_error_samples2": float(
                        selected["squared_error_samples2"]
                    ),
                    "abs_error_samples": float(selected["abs_error_samples"]),
                    "posterior_risk_samples2": float(
                        selected["posterior_risk_samples2"]
                    ),
                    "posterior_nll": float(selected["posterior_nll"]),
                    "credible90_contains_true": bool(
                        selected["credible90_contains_true"]
                    ),
                    "payload_bits": int(selected["payload_bits"]),
                    "feedback_bits": feedback_bits,
                    "total_bits": int(selected["payload_bits"]) + feedback_bits,
                }
            )
    return pd.DataFrame(rows)


def _cluster_bootstrap(
    paired: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    grouped = paired.groupby("cluster_id", sort=True)
    sums = grouped[["target_sq", "baseline_sq"]].sum()
    counts = grouped.size().to_numpy(dtype=float)
    target = sums["target_sq"].to_numpy(dtype=float)
    baseline = sums["baseline_sq"].to_numpy(dtype=float)
    cluster_gain = (baseline - target) / counts
    if len(counts) < 2:
        return {
            "mse_gain": float(np.mean(paired["baseline_sq"] - paired["target_sq"])),
            "mse_gain_ci_low": math.nan,
            "mse_gain_ci_high": math.nan,
            "rmse_gain": math.sqrt(float(paired["baseline_sq"].mean()))
            - math.sqrt(float(paired["target_sq"].mean())),
            "rmse_gain_ci_low": math.nan,
            "rmse_gain_ci_high": math.nan,
            "cluster_mse_gain_sd": math.nan,
            "approx_mde_mse_gain_80pct": math.nan,
        }
    rng = np.random.default_rng(seed)
    mse_values = np.empty(repetitions)
    rmse_values = np.empty(repetitions)
    for index in range(repetitions):
        selected = rng.integers(0, len(counts), size=len(counts))
        n = float(np.sum(counts[selected]))
        target_mse = float(np.sum(target[selected]) / n)
        baseline_mse = float(np.sum(baseline[selected]) / n)
        mse_values[index] = baseline_mse - target_mse
        rmse_values[index] = math.sqrt(baseline_mse) - math.sqrt(target_mse)
    target_mse = float(paired["target_sq"].mean())
    baseline_mse = float(paired["baseline_sq"].mean())
    cluster_sd = float(np.std(cluster_gain, ddof=1))
    standard_error = cluster_sd / math.sqrt(len(cluster_gain))
    approximate_mde = float(
        (norm.ppf(0.975) + norm.ppf(0.80)) * standard_error
    )
    return {
        "mse_gain": baseline_mse - target_mse,
        "mse_gain_ci_low": float(np.quantile(mse_values, 0.025)),
        "mse_gain_ci_high": float(np.quantile(mse_values, 0.975)),
        "rmse_gain": math.sqrt(baseline_mse) - math.sqrt(target_mse),
        "rmse_gain_ci_low": float(np.quantile(rmse_values, 0.025)),
        "rmse_gain_ci_high": float(np.quantile(rmse_values, 0.975)),
        "cluster_mse_gain_sd": cluster_sd,
        "approx_mde_mse_gain_80pct": approximate_mde,
    }


def _effect_summary(
    methods: pd.DataFrame,
    config: FeasibilityConfig,
) -> pd.DataFrame:
    rows = []
    comparisons = (
        ("Oracle-OneAcquire", "BestStatic-OneAcquire"),
        ("Oracle-OneAcquire", "SNR-only-OneAcquire"),
        ("RiskGreedy-OneAcquire", "BestStatic-OneAcquire"),
        ("RiskGreedy-OneAcquire", "SNR-only-OneAcquire"),
    )
    for pool in sorted(methods["pool"].unique()):
        for likelihood in sorted(methods["likelihood"].unique()):
            subset = methods[
                methods["pool"].eq(pool)
                & methods["likelihood"].eq(likelihood)
            ]
            for target, baseline in comparisons:
                left = subset[subset["method"].eq(target)]
                right = subset[subset["method"].eq(baseline)]
                if left.empty or right.empty:
                    continue
                keys = ["seed", "trial", "cluster_id", "snr_db"]
                paired = left[keys + ["squared_error_samples2"]].rename(
                    columns={"squared_error_samples2": "target_sq"}
                ).merge(
                    right[keys + ["squared_error_samples2"]].rename(
                        columns={"squared_error_samples2": "baseline_sq"}
                    ),
                    on=keys,
                    validate="one_to_one",
                )
                summary = _cluster_bootstrap(
                    paired,
                    repetitions=config.bootstrap_repetitions,
                    seed=2026072300 + len(rows),
                )
                baseline_rmse = math.sqrt(float(paired["baseline_sq"].mean()))
                rows.append(
                    {
                        "pool": pool,
                        "likelihood": likelihood,
                        "target": target,
                        "baseline": baseline,
                        "n_rows": len(paired),
                        "n_clusters": paired["cluster_id"].nunique(),
                        "target_rmse": math.sqrt(float(paired["target_sq"].mean())),
                        "baseline_rmse": baseline_rmse,
                        "relative_rmse_gain": (
                            summary["rmse_gain"] / max(baseline_rmse, 1e-12)
                        ),
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def _calibration_summary(methods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in methods.groupby(
        ["pool", "likelihood", "method", "snr_db"],
        sort=True,
    ):
        mse = float(group["squared_error_samples2"].mean())
        risk = float(group["posterior_risk_samples2"].mean())
        rows.append(
            {
                "pool": keys[0],
                "likelihood": keys[1],
                "method": keys[2],
                "snr_db": keys[3],
                "n": len(group),
                "rmse": math.sqrt(mse),
                "mean_posterior_risk": risk,
                "risk_to_mse_ratio": risk / max(mse, 1e-12),
                "coverage90": float(group["credible90_contains_true"].mean()),
                "mean_posterior_nll": float(group["posterior_nll"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _score_diagnostics(
    actions: pd.DataFrame,
    methods: pd.DataFrame,
    score_pool: PoolName,
) -> pd.DataFrame:
    scored = actions[
        actions["pool"].eq(score_pool)
        & actions["likelihood"].isin(DEPLOYABLE_LIKELIHOODS)
        & np.isfinite(actions["predicted_expected_risk"])
    ].copy()
    rows = []
    keys = ["seed", "trial", "cluster_id", "snr_db", "likelihood"]
    for key, group in scored.groupby(keys, sort=True):
        predicted_gain = (
            group["base_posterior_risk"] - group["predicted_expected_risk"]
        )
        actual_gain = group["base_squared_error"] - group["squared_error_samples2"]
        correlation = spearmanr(predicted_gain, actual_gain).statistic
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "pool": score_pool,
                "n_actions": len(group),
                "spearman_predicted_vs_actual_gain": float(correlation),
                "oracle_position": int(
                    group.sort_values(
                        ["squared_error_samples2", "position"],
                        kind="stable",
                    ).iloc[0]["position"]
                ),
                "risk_position": int(
                    group.sort_values(
                        ["predicted_expected_risk", "position"],
                        kind="stable",
                    ).iloc[0]["position"]
                ),
            }
        )
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics
    diagnostics["action_match"] = (
        diagnostics["oracle_position"] == diagnostics["risk_position"]
    )

    score_methods = methods[
        methods["pool"].eq(score_pool)
        & methods["method"].isin(
            (
                "Oracle-OneAcquire",
                "SNR-only-OneAcquire",
                "RiskGreedy-OneAcquire",
            )
        )
    ]
    pivot = score_methods.pivot_table(
        index=["seed", "trial", "cluster_id", "snr_db", "likelihood"],
        columns="method",
        values="squared_error_samples2",
        aggfunc="first",
    ).reset_index()
    if {
        "Oracle-OneAcquire",
        "SNR-only-OneAcquire",
        "RiskGreedy-OneAcquire",
    }.issubset(pivot.columns):
        denominator = (
            pivot["SNR-only-OneAcquire"] - pivot["Oracle-OneAcquire"]
        )
        pivot["oracle_gap_capture"] = np.where(
            denominator > 1e-12,
            (
                pivot["SNR-only-OneAcquire"]
                - pivot["RiskGreedy-OneAcquire"]
            )
            / denominator,
            np.nan,
        )
        diagnostics = diagnostics.merge(
            pivot[
                [
                    "seed",
                    "trial",
                    "cluster_id",
                    "snr_db",
                    "likelihood",
                    "oracle_gap_capture",
                ]
            ],
            on=["seed", "trial", "cluster_id", "snr_db", "likelihood"],
            validate="one_to_one",
        )
    return diagnostics


def _threshold_sensitivity(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for effect in effects.itertuples(index=False):
        for threshold in (0.00, 0.02, 0.05, 0.10):
            rows.append(
                {
                    "pool": effect.pool,
                    "likelihood": effect.likelihood,
                    "target": effect.target,
                    "baseline": effect.baseline,
                    "minimum_relative_rmse_gain": threshold,
                    "observed_relative_rmse_gain": effect.relative_rmse_gain,
                    "effect_pass": bool(effect.relative_rmse_gain >= threshold),
                    "statistical_pass": bool(effect.mse_gain_ci_low > 0.0),
                    "full_pass": bool(
                        effect.relative_rmse_gain >= threshold
                        and effect.mse_gain_ci_low > 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _gate(
    *,
    config: FeasibilityConfig,
    effects: pd.DataFrame,
    scores: pd.DataFrame,
    validations: dict[str, object],
    score_pool: PoolName,
) -> dict[str, object]:
    passing_likelihoods: list[str] = []
    if config.mode == "smoke":
        status = "SMOKE_ONLY"
    elif not bool(validations["quadrature_convergence_pass"]):
        status = "HOLD-NUMERICAL-QUADRATURE"
    else:
        oracle = effects[
            effects["target"].eq("Oracle-OneAcquire")
            & effects["baseline"].eq("SNR-only-OneAcquire")
            & effects["pool"].eq(score_pool)
            & effects["likelihood"].eq(PRIMARY_LIKELIHOOD)
        ]
        oracle_pass = bool((oracle["mse_gain_ci_low"] > 0.0).any())
        risk = effects[
            effects["target"].eq("RiskGreedy-OneAcquire")
            & effects["baseline"].isin(
                ("BestStatic-OneAcquire", "SNR-only-OneAcquire")
            )
            & effects["pool"].eq(score_pool)
        ]
        primary_risk = risk[risk["likelihood"].eq(PRIMARY_LIKELIHOOD)]
        primary_scores = scores[scores["likelihood"].eq(PRIMARY_LIKELIHOOD)]
        risk_pass = bool(
            len(primary_risk) == 2
            and float(primary_risk["mse_gain_ci_low"].min()) > 0.0
        )
        score_pass = bool(
            not primary_scores.empty
            and float(
                primary_scores["spearman_predicted_vs_actual_gain"].mean()
            )
            > 0.0
        )
        if risk_pass and score_pass:
            passing_likelihoods = [PRIMARY_LIKELIHOOD]
        if not oracle_pass:
            status = "NO-GO-ROUTE-CURRENT-ONE-ACTION-SPACE"
        elif passing_likelihoods:
            status = "STAGE1_PASS"
        else:
            status = "HOLD-LIKELIHOOD-OR-SCORE"
    return {
        "status": status,
        "mode": config.mode,
        "primary_likelihood": PRIMARY_LIKELIHOOD,
        "score_pool_selected_on_calibration": score_pool,
        "deployable_likelihoods_passing_stage1": passing_likelihoods,
        "mechanical_validations": validations,
        "quadrature_main_grid": [
            config.marginal_amplitude_nodes,
            config.marginal_phase_nodes,
        ],
        "quadrature_reference_grid": list(QUADRATURE_REFERENCE_GRID),
        "quadrature_tolerance": {
            "estimate_abs_diff_max": (
                QUADRATURE_ESTIMATE_GRID_FRACTION * config.tau_step
            ),
            "posterior_risk_abs_diff_max": (
                QUADRATURE_RISK_GRID_SQUARED_FRACTION * config.tau_step**2
            ),
            "tau_posterior_l1_max": QUADRATURE_POSTERIOR_L1_MAX,
            "basis": (
                "Estimate tolerance is 10% of the registered TDOA grid spacing; "
                "risk tolerance is 1% of one grid cell squared; posterior L1 "
                "tolerance corresponds to at most 5% total-variation distance "
                "from the denser quadrature reference."
            ),
        },
        "interpretation_boundary": (
            "The oracle uses true TDOA and is not deployable. The certificate "
            "holds forward payload at 24 bits. Adaptive selection adds a 5-bit "
            "feedback command, while frozen static/SNR plans require no feedback; "
            "this is an action-selection certificate, not an equal-total-bit claim."
        ),
    }


def _plot_oracle(effects: pd.DataFrame, output: Path) -> None:
    subset = effects[
        effects["target"].eq("Oracle-OneAcquire")
        & effects["baseline"].eq("SNR-only-OneAcquire")
    ].copy()
    labels = [
        f"{row.pool}\n{row.likelihood}"
        for row in subset.itertuples(index=False)
    ]
    values = subset["rmse_gain"].to_numpy(dtype=float)
    lower = values - subset["rmse_gain_ci_low"].to_numpy(dtype=float)
    upper = subset["rmse_gain_ci_high"].to_numpy(dtype=float) - values
    colors = [
        "#247BA0" if "Marginal" in value else "#6C757D"
        for value in subset["likelihood"]
    ]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    x = np.arange(len(subset))
    ax.errorbar(
        x,
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#6C757D",
        capsize=3,
        linewidth=1.1,
    )
    ax.scatter(x, values, c=colors, s=28, zorder=3)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Oracle RMSE gain vs SNR-only (sample)")
    ax.set_title("One-action information upper bound with cluster 95% CI")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "Fig1_Oracle_ActionSpace_Certificate.svg")
    plt.close(fig)


def _plot_calibration(
    calibration: pd.DataFrame,
    scores: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    selected = calibration[
        calibration["method"].eq("SNR-only-OneAcquire")
        & calibration["likelihood"].isin(DEPLOYABLE_LIKELIHOODS)
    ]
    for likelihood, group in selected.groupby("likelihood", sort=True):
        summary = group.groupby("snr_db", as_index=False)["coverage90"].mean()
        axes[0].plot(
            summary["snr_db"],
            summary["coverage90"],
            marker="o",
            linewidth=1.2,
            label=likelihood,
        )
    axes[0].axhline(0.9, color="#222222", linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("90% credible interval coverage")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    if not scores.empty:
        data = [
            group["spearman_predicted_vs_actual_gain"].to_numpy(dtype=float)
            for _, group in scores.groupby("likelihood", sort=True)
        ]
        labels = [
            label for label, _ in scores.groupby("likelihood", sort=True)
        ]
        axes[1].boxplot(data, tick_labels=labels, showfliers=True)
        axes[1].tick_params(axis="x", rotation=25)
    axes[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes[1].set_ylabel("Within-sample Spearman correlation")
    axes[1].set_title("Predicted vs realized action gain")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "Fig2_Likelihood_And_ActionScore.svg")
    plt.close(fig)


def run(config: FeasibilityConfig, output: Path) -> dict[str, object]:
    config.validate()
    waveform = WaveformConfig()
    waveform.validate()
    quantizer = NestedComplexQuantizer(max_bits=8, clip_abs=4.0)
    pools = build_candidate_pools(waveform)
    pool_table = pd.DataFrame(candidate_pool_rows(pools, waveform))

    prior = _generate_trials(
        config.prior_seeds,
        config.prior_trials_per_seed,
        waveform,
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
    models, prior_table = _build_models(
        pools,
        prior,
        waveform=waveform,
        config=config,
    )

    calibration_actions = _collect_actions(
        stage="calibration",
        trials=calibration_trials,
        config=config,
        waveform=waveform,
        pools=pools,
        models=models,
        quantizer=quantizer,
        score_pool=None,
        output=output,
    )
    plans, score_pool = _select_plans(calibration_actions)
    plans.to_csv(output / "frozen_one_action_plans.csv", index=False)
    print(f"[Selection] score diagnostics pool={score_pool}", flush=True)
    score_model = models[score_pool]["Linear-Marginal"]
    quadrature = _quadrature_convergence(
        config=config,
        waveform=waveform,
        trials=calibration_trials,
        pool=score_pool,
        bins=pools[score_pool],
        source_power=score_model.source_power,
        plans=plans,
        quantizer=quantizer,
    )

    evaluation_actions = _collect_actions(
        stage="evaluation",
        trials=evaluation_trials,
        config=config,
        waveform=waveform,
        pools=pools,
        models=models,
        quantizer=quantizer,
        score_pool=score_pool,
        output=output,
    )
    methods = _method_rows(evaluation_actions, plans)
    effects = _effect_summary(methods, config)
    calibration = _calibration_summary(methods)
    scores = _score_diagnostics(evaluation_actions, methods, score_pool)
    sensitivity = _threshold_sensitivity(effects)

    configured_quadrature = quadrature[
        quadrature["amplitude_nodes"].eq(config.marginal_amplitude_nodes)
        & quadrature["phase_nodes"].eq(config.marginal_phase_nodes)
    ]
    quadrature_estimate_max = float(
        configured_quadrature["estimate_abs_diff_vs_reference"].max()
    )
    quadrature_posterior_l1_max = float(
        configured_quadrature["tau_posterior_l1_vs_reference"].max()
    )
    quadrature_risk_max = float(
        configured_quadrature["risk_abs_diff_vs_reference"].max()
    )
    quadrature_pass = bool(
        quadrature_estimate_max
        <= QUADRATURE_ESTIMATE_GRID_FRACTION * config.tau_step
        and quadrature_risk_max
        <= QUADRATURE_RISK_GRID_SQUARED_FRACTION * config.tau_step**2
        and quadrature_posterior_l1_max <= QUADRATURE_POSTERIOR_L1_MAX
    )
    validations = {
        "seed_pools_disjoint": len(
            set(config.prior_seeds)
            | set(config.calibration_seeds)
            | set(config.evaluation_seeds)
        )
        == (
            len(config.prior_seeds)
            + len(config.calibration_seeds)
            + len(config.evaluation_seeds)
        ),
        "all_candidate_counts_equal_12": bool(
            pool_table["candidate_count"].eq(12).all()
        ),
        "all_action_payload_bits_equal_24": bool(
            calibration_actions["payload_bits"].eq(24).all()
            and evaluation_actions["payload_bits"].eq(24).all()
        ),
        "adaptive_action_total_bits_equal_29": bool(
            calibration_actions["adaptive_total_bits"].eq(29).all()
            and evaluation_actions["adaptive_total_bits"].eq(29).all()
        ),
        "method_specific_feedback_accounting": bool(
            methods.loc[
                methods["method"].isin(
                    ("Oracle-OneAcquire", "RiskGreedy-OneAcquire")
                ),
                "total_bits",
            ]
            .eq(29)
            .all()
            and methods.loc[
                methods["method"].isin(
                    ("BestStatic-OneAcquire", "SNR-only-OneAcquire")
                ),
                "total_bits",
            ]
            .eq(24)
            .all()
        ),
        "oracle_never_worse_than_frozen_baselines": True,
        "finite_action_results": bool(
            np.isfinite(
                evaluation_actions[
                    [
                        "squared_error_samples2",
                        "posterior_risk_samples2",
                        "posterior_nll",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "quantizer_clipping_max": float(
            max(
                calibration_actions["clipping_rate"].max(),
                evaluation_actions["clipping_rate"].max(),
            )
        ),
        "quadrature_convergence_pass": quadrature_pass,
        "quadrature_estimate_max_diff": quadrature_estimate_max,
        "quadrature_posterior_risk_max_diff": quadrature_risk_max,
        "quadrature_tau_posterior_l1_max": quadrature_posterior_l1_max,
    }
    for keys, group in methods.groupby(
        ["seed", "trial", "snr_db", "pool", "likelihood"],
        sort=False,
    ):
        oracle = group[group["method"].eq("Oracle-OneAcquire")]
        baselines = group[
            group["method"].isin(
                ("BestStatic-OneAcquire", "SNR-only-OneAcquire")
            )
        ]
        if not baselines.empty and (
            len(oracle) != 1
            or float(oracle.iloc[0]["squared_error_samples2"])
            > float(baselines["squared_error_samples2"].min()) + 1e-12
        ):
            validations["oracle_never_worse_than_frozen_baselines"] = False
            raise RuntimeError(f"oracle invariant failed for {keys}")
    if (
        not validations["seed_pools_disjoint"]
        or not validations["all_candidate_counts_equal_12"]
        or not validations["all_action_payload_bits_equal_24"]
        or not validations["adaptive_action_total_bits_equal_29"]
        or not validations["method_specific_feedback_accounting"]
        or not validations["finite_action_results"]
    ):
        raise RuntimeError(f"mechanical validation failed: {validations}")

    gate = _gate(
        config=config,
        effects=effects,
        scores=scores,
        validations=validations,
        score_pool=score_pool,
    )

    outputs = {
        "candidate_pools.csv": pool_table,
        "prior_statistics.csv": prior_table,
        "calibration_action_rows.csv": calibration_actions,
        "evaluation_action_rows.csv": evaluation_actions,
        "method_results.csv": methods,
        "effect_summary.csv": effects,
        "likelihood_calibration.csv": calibration,
        "action_score_diagnostics.csv": scores,
        "threshold_sensitivity.csv": sensitivity,
        "quadrature_convergence.csv": quadrature,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, index=False)
    for partial in output.glob("*.partial.csv"):
        partial.unlink()
    (output / "gate_decision.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_oracle(effects, output)
    _plot_calibration(calibration, scores, output)
    return {
        "gate": gate,
        "pools": pool_table,
        "models": models,
        "tables": outputs,
    }


def _output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "运行结果"
        / "BRSR"
        / f"BRSR_ROUTE_STAGE1_{timestamp}_{mode}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=tuple(config_for_mode(name).mode for name in ("smoke", "development")),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode or PYCHARM_RUN_MODE
    config = config_for_mode(mode)
    output = _output_dir(mode)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        tee = Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            print("BRSR route feasibility Stage 1", flush=True)
            print(f"Mode: {mode}", flush=True)
            print(f"Output: {output}", flush=True)
            print(f"Config: {json.dumps(asdict(config), ensure_ascii=False)}", flush=True)
            result = run(config, output)
            elapsed = time.perf_counter() - started
            gate = result["gate"]
            print(f"[Gate] {gate['status']}", flush=True)
            print(f"[Done] elapsed={elapsed:.1f}s", flush=True)

            manifest = {
                "experiment": "BRSR route feasibility Stage 1",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "config": asdict(config),
                "waveform": asdict(WaveformConfig()),
                "protocol": {
                    "base_payload_bits": BASE_PAYLOAD_BITS,
                    "action_feedback_bits": ONE_ACTION_FEEDBACK_BITS,
                    "acquired_complex_bin_payload_bits": (
                        ACQUIRED_BIN_PAYLOAD_BITS
                    ),
                    "common_forward_payload_bits": (
                        BASE_PAYLOAD_BITS + ACQUIRED_BIN_PAYLOAD_BITS
                    ),
                    "adaptive_total_bits": (
                        BASE_PAYLOAD_BITS
                        + ONE_ACTION_FEEDBACK_BITS
                        + ACQUIRED_BIN_PAYLOAD_BITS
                    ),
                    "static_total_bits": (
                        BASE_PAYLOAD_BITS + ACQUIRED_BIN_PAYLOAD_BITS
                    ),
                    "max_actions": 1,
                },
                "score_pool_selected_on_calibration": gate[
                    "score_pool_selected_on_calibration"
                ],
                "gate_status": gate["status"],
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "elapsed_seconds": elapsed,
                "source_files": {
                    str(path.relative_to(ROOT)): _sha256(path)
                    for path in (
                        Path(__file__),
                        Path(__file__).with_name("feasibility.py"),
                        ROOT / "brsr_tdoa" / "brsr_core.py",
                        ROOT / "brsr_tdoa" / "brsr_stage_b.py",
                        ROOT / "brsr_tdoa" / "brsr_waveform.py",
                    )
                },
            }
            (output / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
