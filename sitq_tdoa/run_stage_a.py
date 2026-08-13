"""PyCharm entry point for SITQ-TDOA Stage A.

Run this file directly for the pre-registered development experiment.  Codex
uses ``--smoke`` only for a short engineering check; smoke results are not
research evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sitq_tdoa.core import (  # noqa: E402
    ModelConfig,
    ObservationContext,
    SITQModel,
    ThresholdSearchResult,
    evaluate_context,
    evaluate_threshold_objective,
    quantizer_probability_sum,
    search_threshold,
)


RUN_MODE = "development"


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    snr_db_values: tuple[float, ...]
    design_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    design_trials_per_seed: int
    evaluation_trials_per_seed: int
    threshold_candidates: tuple[float, ...]
    bootstrap_repetitions: int

    def validate(self) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.snr_db_values:
            raise ValueError("snr_db_values must not be empty")
        if set(self.design_seeds) & set(self.evaluation_seeds):
            raise ValueError("design and evaluation seeds must be disjoint")
        if self.design_trials_per_seed < 1 or self.evaluation_trials_per_seed < 1:
            raise ValueError("trial counts must be positive")
        thresholds = np.asarray(self.threshold_candidates, dtype=float)
        if thresholds.size < 3 or np.any(np.diff(thresholds) <= 0.0):
            raise ValueError("threshold candidates must be increasing")
        if self.bootstrap_repetitions < 20:
            raise ValueError("bootstrap_repetitions must be at least 20")


def build_experiment_config(mode: str) -> ExperimentConfig:
    if mode == "smoke":
        config = ExperimentConfig(
            mode="smoke",
            snr_db_values=(-5.0, 5.0),
            design_seeds=(1301,),
            evaluation_seeds=(2301,),
            design_trials_per_seed=3,
            evaluation_trials_per_seed=4,
            threshold_candidates=tuple(
                float(value) for value in np.geomspace(0.25, 1.6, 5)
            ),
            bootstrap_repetitions=50,
        )
    elif mode == "development":
        config = ExperimentConfig(
            mode="development",
            snr_db_values=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
            design_seeds=(1301, 1302),
            evaluation_seeds=(2301, 2302, 2303),
            design_trials_per_seed=24,
            evaluation_trials_per_seed=80,
            threshold_candidates=tuple(
                float(value) for value in np.geomspace(0.15, 2.0, 25)
            ),
            bootstrap_repetitions=2000,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    config.validate()
    return config


def _source_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_contexts(
    model: SITQModel,
    seeds: Sequence[int],
    trials_per_seed: int,
    snr_values: Sequence[float],
) -> list[ObservationContext]:
    contexts: list[ObservationContext] = []
    for seed in seeds:
        trials = model.generate_trials(seed, trials_per_seed)
        for trial in trials:
            contexts.extend(model.build_context(trial, snr) for snr in snr_values)
    return contexts


def _embedded_checks(
    model: SITQModel,
    config: ExperimentConfig,
) -> dict[str, bool | float | int]:
    probability_error = abs(quantizer_probability_sum(0.3, 0.8, 0.7) - 1.0)
    bins = model.bins
    checks: dict[str, bool | float | int] = {
        "quantizer_probability_error": probability_error,
        "quantizer_probability_partition": probability_error < 1e-12,
        "design_evaluation_seed_disjoint": not bool(
            set(config.design_seeds) & set(config.evaluation_seeds)
        ),
        "selected_bins_unique": bool(np.unique(bins).size == bins.size),
        "selected_bins_in_rrc_support": bool(
            np.all(bins <= model.config.rrc_support_max_bin)
        ),
        "payload_bits": model.config.total_payload_bits,
        "payload_is_24_bits": model.config.total_payload_bits == 24,
        "tau_prior_normalized": math.isclose(
            float(np.sum(model.tau_prior)), 1.0, abs_tol=1e-12
        ),
        "rho_prior_normalized": math.isclose(
            float(np.sum(model.rho_prior)), 1.0, abs_tol=1e-12
        ),
    }
    boolean_values = [value for value in checks.values() if isinstance(value, bool)]
    checks["all_passed"] = all(boolean_values)
    return checks


def _evaluate_methods(
    model: SITQModel,
    contexts: Sequence[ObservationContext],
    thresholds: dict[str, float | None],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    total = len(contexts) * len(thresholds)
    done = 0
    start = time.perf_counter()
    for context in contexts:
        for method, threshold in thresholds.items():
            result = evaluate_context(model, context, method, threshold)
            rows.append(
                {
                    "method": method,
                    "seed": context.seed,
                    "trial": context.trial,
                    "snr_db": context.snr_db,
                    "true_tau": context.true_tau,
                    "estimate": result.estimate,
                    "squared_error": result.squared_error,
                    "abs_error": abs(result.estimate - context.true_tau),
                    "posterior_risk": result.posterior_risk,
                    "posterior_entropy": result.entropy,
                    "posterior_nll": result.nll,
                    "credible90_contains_true": result.credible90_contains_true,
                    "threshold": math.nan if threshold is None else threshold,
                    "payload_bits": (
                        math.nan
                        if method == "Unquantized-SelectedBins"
                        else model.config.total_payload_bits
                    ),
                }
            )
            done += 1
        if done % max(1, total // 10) == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[Evaluation] {done}/{total} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for (method, snr_db), group in rows.groupby(["method", "snr_db"], sort=False):
        squared = group["squared_error"].to_numpy(float)
        absolute = group["abs_error"].to_numpy(float)
        records.append(
            {
                "method": method,
                "snr_db": float(snr_db),
                "rmse_samples": float(np.sqrt(np.mean(squared))),
                "mae_samples": float(np.mean(absolute)),
                "p95_abs_error_samples": float(np.quantile(absolute, 0.95)),
                "outlier_gt_1_rate": float(np.mean(absolute > 1.0)),
                "posterior_nll": float(group["posterior_nll"].mean()),
                "credible90_coverage": float(
                    group["credible90_contains_true"].mean()
                ),
                "mean_posterior_risk": float(group["posterior_risk"].mean()),
                "n_samples": int(group.shape[0]),
            }
        )
    return pd.DataFrame(records)


def _post_evaluation_checks(
    rows: pd.DataFrame,
    expected_contexts: int,
    payload_bits: int,
) -> dict[str, bool | int]:
    methods = tuple(rows["method"].drop_duplicates())
    expected_rows = expected_contexts * len(methods)
    quantized = rows[rows["method"] != "Unquantized-SelectedBins"]
    keys = ["method", "seed", "trial", "snr_db"]
    counts = rows.groupby("method").size()
    checks: dict[str, bool | int] = {
        "evaluation_expected_rows": int(expected_rows),
        "evaluation_actual_rows": int(rows.shape[0]),
        "evaluation_row_count_valid": bool(rows.shape[0] == expected_rows),
        "evaluation_keys_unique": bool(not rows.duplicated(keys).any()),
        "all_methods_complete": bool((counts == expected_contexts).all()),
        "quantized_payload_nonmissing": bool(quantized["payload_bits"].notna().all()),
        "quantized_payload_equal": bool(
            np.allclose(quantized["payload_bits"].to_numpy(float), payload_bits)
        ),
        "unquantized_excluded_from_bit_comparison": bool(
            rows.loc[
                rows["method"] == "Unquantized-SelectedBins", "payload_bits"
            ].isna().all()
        ),
    }
    checks["post_evaluation_all_passed"] = all(
        value for value in checks.values() if isinstance(value, bool)
    )
    return checks


def _cluster_bootstrap_gain(
    rows: pd.DataFrame,
    candidate: str,
    baseline: str,
    repetitions: int,
    seed: int = 94021,
) -> dict[str, float | int | str]:
    subset = rows[rows["method"].isin([candidate, baseline])]
    pivot = subset.pivot_table(
        index=["seed", "trial", "snr_db"],
        columns="method",
        values="squared_error",
        aggfunc="first",
    ).dropna()
    if pivot.empty:
        raise RuntimeError("paired bootstrap has no aligned rows")
    delta = pivot[baseline] - pivot[candidate]
    cluster = delta.groupby(level=[0, 1]).mean().to_numpy(float)
    cluster_sd = float(np.std(cluster, ddof=1)) if cluster.size > 1 else math.nan
    mde95 = (
        1.96 * cluster_sd / math.sqrt(cluster.size)
        if cluster.size > 1
        else math.nan
    )
    rng = np.random.default_rng(seed)
    sampled = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled[index] = float(
            np.mean(rng.choice(cluster, size=cluster.size, replace=True))
        )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "paired_mse_gain_samples2": float(np.mean(cluster)),
        "paired_mse_gain_ci95_low": float(np.quantile(sampled, 0.025)),
        "paired_mse_gain_ci95_high": float(np.quantile(sampled, 0.975)),
        "fraction_trajectory_clusters_positive": float(np.mean(cluster > 0.0)),
        "n_trajectory_clusters": int(cluster.size),
        "trajectory_cluster_sd_samples2": cluster_sd,
        "approx_mde95_samples2": mde95,
    }


def _threshold_sensitivity(
    search: ThresholdSearchResult,
) -> dict[str, float | bool]:
    """Report local objective separation around the selected grid point."""

    values = np.asarray([row["objective_value"] for row in search.rows], dtype=float)
    thresholds = np.asarray([row["threshold"] for row in search.rows], dtype=float)
    selected = int(np.argmin(values))
    neighbours = [index for index in (selected - 1, selected + 1) if 0 <= index < values.size]
    local_gap = (
        float(min(values[index] - values[selected] for index in neighbours))
        if neighbours
        else math.nan
    )
    return {
        "selected_threshold": float(thresholds[selected]),
        "at_search_boundary": bool(search.at_search_boundary),
        "nearest_grid_objective_gap": local_gap,
    }


def _select_frozen_baseline(
    model: SITQModel,
    design_contexts: Sequence[ObservationContext],
    thresholds: dict[str, float],
) -> tuple[str, dict[str, float]]:
    risks = {
        method: evaluate_threshold_objective(
            model, design_contexts, threshold, "bayes_risk"
        )
        for method, threshold in thresholds.items()
    }
    return min(risks, key=risks.get), risks


def _plot(summary: pd.DataFrame, rows: pd.DataFrame, output: Path) -> None:
    style = {
        "MSEQuant-24bit": ("#4c78a8", "o", "-"),
        "CondMIQuant-24bit": ("#59a14f", "s", "--"),
        "SITQ-BayesRisk-24bit": ("#e45756", "^", "-"),
        "Unquantized-SelectedBins": ("#767676", "D", ":"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    for method, group in summary.groupby("method", sort=False):
        color, marker, line = style[method]
        axes[0].plot(
            group["snr_db"],
            group["rmse_samples"],
            label=method,
            color=color,
            marker=marker,
            linestyle=line,
            linewidth=1.45,
            markersize=4.5,
        )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("TDOA RMSE (samples)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    baseline = "MSEQuant-24bit"
    for method in ("CondMIQuant-24bit", "SITQ-BayesRisk-24bit"):
        merged = rows[rows["method"].isin([baseline, method])].pivot_table(
            index=["seed", "trial", "snr_db"],
            columns="method",
            values="squared_error",
            aggfunc="first",
        ).dropna()
        delta = (merged[baseline] - merged[method]).groupby(level=2).mean()
        color, marker, line = style[method]
        axes[1].plot(
            delta.index,
            delta.values,
            label=method,
            color=color,
            marker=marker,
            linestyle=line,
            linewidth=1.45,
            markersize=4.5,
        )
    axes[1].axhline(0.0, color="#333333", linewidth=0.9)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("MSE gain over MSEQuant (samples$^2$)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(output / "Fig1_SITQ_StageA.svg", dpi=300)
    plt.close(fig)


def _gate_status(
    checks: dict[str, bool | float | int],
    design_risks: dict[str, float],
    sitq_design_risk: float,
    paired: dict[str, float | int | str],
    boundary_methods: Sequence[str] = (),
) -> str:
    if not bool(checks["all_passed"]):
        return "INVALID_MECHANICAL_CHECK"
    if boundary_methods:
        return "HOLD_THRESHOLD_SEARCH_BOUNDARY"
    best_design_baseline = min(design_risks.values())
    if sitq_design_risk > best_design_baseline + 1e-12:
        return "NO_GO_CONFIG_DESIGN_OBJECTIVE"
    ci_low = float(paired["paired_mse_gain_ci95_low"])
    ci_high = float(paired["paired_mse_gain_ci95_high"])
    if ci_low > 0.0:
        return "GO_STAGE_A_MECHANISM"
    if ci_high <= 0.0:
        return "NO_GO_CONFIG_SITQ_STAGE_A"
    return "HOLD_STAGE_A_UNCERTAIN"


def run(mode: str) -> Path:
    experiment = build_experiment_config(mode)
    model_config = ModelConfig()
    model = SITQModel(model_config)
    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / "运行结果" / "SITQ" / f"SITQ_STAGE_A_{timestamp}_{mode}"
    output.mkdir(parents=True, exist_ok=False)
    print("SITQ-TDOA Stage A", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        f"Fixed bins={list(model_config.selected_bins)}, "
        f"payload={model_config.total_payload_bits} bit, "
        f"SNR={list(experiment.snr_db_values)}",
        flush=True,
    )
    checks = _embedded_checks(model, experiment)
    if not bool(checks["all_passed"]):
        raise RuntimeError(f"embedded checks failed: {checks}")

    print("[Design] building independent contexts", flush=True)
    design_contexts = _build_contexts(
        model,
        experiment.design_seeds,
        experiment.design_trials_per_seed,
        experiment.snr_db_values,
    )
    searches = {
        "MSEQuant-24bit": search_threshold(
            model,
            design_contexts,
            experiment.threshold_candidates,
            "quantization_mse",
        ),
        "CondMIQuant-24bit": search_threshold(
            model,
            design_contexts,
            experiment.threshold_candidates,
            "posterior_entropy",
        ),
        "SITQ-BayesRisk-24bit": search_threshold(
            model,
            design_contexts,
            experiment.threshold_candidates,
            "bayes_risk",
        ),
    }
    for method, search in searches.items():
        print(
            f"[Design] {method}: threshold={search.threshold:.6g}, "
            f"objective={search.objective_value:.6g}, "
            f"boundary={search.at_search_boundary}",
            flush=True,
        )
    frozen_baseline, design_baseline_risks = _select_frozen_baseline(
        model,
        design_contexts,
        {
            method: search.threshold
            for method, search in searches.items()
            if method != "SITQ-BayesRisk-24bit"
        },
    )
    sitq_design_risk = evaluate_threshold_objective(
        model,
        design_contexts,
        searches["SITQ-BayesRisk-24bit"].threshold,
        "bayes_risk",
    )
    print(f"[Design] frozen strongest baseline={frozen_baseline}", flush=True)

    print("[Evaluation] building independent contexts", flush=True)
    evaluation_contexts = _build_contexts(
        model,
        experiment.evaluation_seeds,
        experiment.evaluation_trials_per_seed,
        experiment.snr_db_values,
    )
    thresholds: dict[str, float | None] = {
        method: search.threshold for method, search in searches.items()
    }
    thresholds["Unquantized-SelectedBins"] = None
    rows = _evaluate_methods(model, evaluation_contexts, thresholds)
    post_checks = _post_evaluation_checks(
        rows,
        expected_contexts=len(evaluation_contexts),
        payload_bits=model_config.total_payload_bits,
    )
    checks.update(post_checks)
    checks["all_passed"] = bool(
        checks["all_passed"] and post_checks["post_evaluation_all_passed"]
    )
    if not bool(checks["all_passed"]):
        raise RuntimeError(f"post-evaluation checks failed: {checks}")
    summary = _summarize(rows)
    paired_rows = [
        _cluster_bootstrap_gain(
            rows,
            "SITQ-BayesRisk-24bit",
            baseline,
            experiment.bootstrap_repetitions,
        )
        for baseline in ("MSEQuant-24bit", "CondMIQuant-24bit", frozen_baseline)
    ]
    paired = pd.DataFrame(paired_rows).drop_duplicates(
        subset=["candidate", "baseline"]
    )
    frozen_comparison = next(
        row for row in paired_rows if row["baseline"] == frozen_baseline
    )
    status = _gate_status(
        checks,
        design_baseline_risks,
        sitq_design_risk,
        frozen_comparison,
        boundary_methods=tuple(
            method for method, search in searches.items() if search.at_search_boundary
        ),
    )

    rows.to_csv(output / "stage_a_rows.csv", index=False)
    summary.to_csv(output / "stage_a_summary.csv", index=False)
    paired.to_csv(output / "stage_a_paired_effects.csv", index=False)
    design_rows: list[dict[str, float | str | bool]] = []
    for method, search in searches.items():
        for row in search.rows:
            design_rows.append(
                {
                    "method": method,
                    "objective": search.objective,
                    "threshold": row["threshold"],
                    "objective_value": row["objective_value"],
                    "selected": math.isclose(
                        row["threshold"], search.threshold, abs_tol=1e-12
                    ),
                }
            )
    pd.DataFrame(design_rows).to_csv(output / "threshold_design.csv", index=False)
    _plot(summary, rows, output)

    source_files = [Path(__file__).resolve(), Path(__file__).with_name("core.py")]
    manifest = {
        "experiment": asdict(experiment),
        "model": asdict(model_config),
        "source_hash": _source_hash(source_files),
        "methods": {
            method: {
                "objective": search.objective,
                "threshold": search.threshold,
                "at_search_boundary": search.at_search_boundary,
            }
            for method, search in searches.items()
        },
        "threshold_sensitivity": {
            method: _threshold_sensitivity(search)
            for method, search in searches.items()
        },
        "frozen_strongest_baseline": frozen_baseline,
        "design_baseline_bayes_risks": design_baseline_risks,
        "sitq_design_bayes_risk": sitq_design_risk,
        "information_boundary": {
            "encoder": "remote selected frequency coefficients only",
            "decoder_side_information": "full reference selected coefficients",
            "snr": "public known experimental condition",
            "source_and_rho_priors": "matched and frozen before evaluation",
            "forbidden": "true tau, true rho, evaluation outcomes",
        },
        "budget": {
            "dynamic_index_bits": 0,
            "feedback_bits": 0,
            "payload_bits": model_config.total_payload_bits,
            "fixed_codebook": True,
        },
        "gate": {
            "status": status,
            "rule": (
                "HOLD if any threshold optimum lies on the search boundary. "
                "Otherwise GO iff paired trajectory-cluster 95% CI lower bound for "
                "MSE gain over the design-frozen strongest legal baseline is positive; "
                "HOLD if CI crosses zero; configuration NO-GO if CI is nonpositive."
            ),
        },
    }
    (output / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validation = {
        **checks,
        "threshold_search_boundary_methods": [
            method for method, search in searches.items() if search.at_search_boundary
        ],
        "frozen_strongest_baseline": frozen_baseline,
        "gate_status": status,
        "formal_performance_evidence": mode == "development",
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nPaired effects", flush=True)
    print(paired.to_string(index=False), flush=True)
    print(f"\n[Gate] {status}", flush=True)
    print(f"[Done] {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a tiny engineering check; never use it for performance claims",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run("smoke" if args.smoke else RUN_MODE)


if __name__ == "__main__":
    main()
