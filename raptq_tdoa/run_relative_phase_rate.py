"""PyCharm entry for the final bounded RAPTQ RelativePhase feasibility cycle.

The run uses new trajectory seeds and does not overwrite historical RAPTQ or
SITQ results.  It first freezes a continuous-phase CondMI threshold and runs a
numerical audit.  Only a numerically valid development run is interpreted as a
finite-rate comparison; smoke mode proves the engineering chain only.
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
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raptq_tdoa.core import (  # noqa: E402
    BayesianRAPTQModel,
    ModelConfig,
    ObservationContext,
    QMCConfig,
    posterior_total_variation,
)
from raptq_tdoa.relative_phase_rate import (  # noqa: E402
    ContinuousPhaseCondMI24,
    QuantizedRelativePhase24,
    RelativePhaseCodeConfig,
    RelativePhaseQMCConfig,
    relative_phase_quantization_max_error,
)


RUN_MODE = "development"
DELTA_GUARD_SAMPLES2 = 0.0625


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    snr_db_values: tuple[float, ...]
    design_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    design_trials_per_seed: int
    evaluation_trials_per_seed: int
    threshold_candidates: tuple[float, ...]
    condmi_phase_nodes: int
    condmi_reference_phase_nodes: int
    relative_phase_power: int
    relative_phase_repeats: int
    numerical_trials_per_seed: int
    bootstrap_repetitions: int

    def validate(self) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if set(self.design_seeds) & set(self.evaluation_seeds):
            raise ValueError("design and evaluation seeds must be disjoint")
        if self.design_trials_per_seed < 1 or self.evaluation_trials_per_seed < 1:
            raise ValueError("trial counts must be positive")
        if self.numerical_trials_per_seed < 1:
            raise ValueError("numerical trial count must be positive")
        if self.condmi_reference_phase_nodes <= self.condmi_phase_nodes:
            raise ValueError("CondMI reference quadrature must be denser")
        thresholds = np.asarray(self.threshold_candidates, dtype=float)
        if thresholds.size < 3 or np.any(np.diff(thresholds) <= 0.0):
            raise ValueError("threshold candidates must be increasing")
        if self.bootstrap_repetitions < 20:
            raise ValueError("bootstrap repetitions must be at least 20")


def build_experiment_config(mode: str) -> ExperimentConfig:
    if mode == "smoke":
        config = ExperimentConfig(
            mode=mode,
            snr_db_values=(-5.0, 5.0),
            design_seeds=(7301,),
            evaluation_seeds=(8301,),
            design_trials_per_seed=2,
            evaluation_trials_per_seed=2,
            threshold_candidates=tuple(float(x) for x in np.geomspace(0.3, 1.2, 3)),
            condmi_phase_nodes=16,
            condmi_reference_phase_nodes=32,
            relative_phase_power=6,
            relative_phase_repeats=2,
            numerical_trials_per_seed=1,
            bootstrap_repetitions=50,
        )
    elif mode == "development":
        config = ExperimentConfig(
            mode=mode,
            snr_db_values=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
            design_seeds=(7301, 7302),
            evaluation_seeds=(8301, 8302, 8303, 8304),
            design_trials_per_seed=24,
            evaluation_trials_per_seed=40,
            threshold_candidates=tuple(float(x) for x in np.geomspace(0.15, 2.0, 25)),
            condmi_phase_nodes=128,
            condmi_reference_phase_nodes=256,
            relative_phase_power=14,
            relative_phase_repeats=4,
            numerical_trials_per_seed=2,
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


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _seed(context: ObservationContext, label: str) -> int:
    token = f"{context.seed}:{context.trial}:{context.snr_db}:{label}"
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def _summary_row(
    model: BayesianRAPTQModel,
    context: ObservationContext,
    method: str,
    posterior: np.ndarray,
    payload_bits: float,
    elapsed_seconds: float,
) -> dict[str, float | int | str | bool]:
    summary = model.summarize_posterior(posterior, context.true_tau)
    return {
        "method": method,
        "seed": context.seed,
        "trial": context.trial,
        "snr_db": context.snr_db,
        "true_tau": context.true_tau,
        "estimate": summary.estimate,
        "squared_error": summary.squared_error,
        "abs_error": abs(summary.estimate - context.true_tau),
        "gross_error_gt_1": abs(summary.estimate - context.true_tau) > 1.0,
        "posterior_risk": summary.posterior_risk,
        "posterior_entropy": summary.entropy,
        "posterior_nll": summary.nll,
        "credible90_contains_true": summary.credible90_contains_true,
        "payload_bits": payload_bits,
        "elapsed_seconds": elapsed_seconds,
    }


def _paired_effect(
    rows: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int | str]:
    pivot = rows[rows["method"].isin([candidate, baseline])].pivot_table(
        index=["seed", "trial", "snr_db"],
        columns="method",
        values=metric,
        aggfunc="first",
    ).dropna()
    if pivot.empty:
        raise RuntimeError("paired effect has no aligned rows")
    pivot = pivot.astype(float)
    gain = pivot[baseline] - pivot[candidate]
    cluster = gain.groupby(level=[0, 1]).mean().to_numpy(float)
    cluster_sd = float(np.std(cluster, ddof=1)) if cluster.size > 1 else 0.0
    rng = np.random.default_rng(seed)
    sampled = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled[index] = float(np.mean(rng.choice(cluster, cluster.size, replace=True)))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "paired_gain": float(np.mean(cluster)),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "fraction_clusters_positive": float(np.mean(cluster > 0.0)),
        "n_trajectory_clusters": int(cluster.size),
        "cluster_sd": cluster_sd,
        "approx_mde95": (
            1.96 * cluster_sd / math.sqrt(cluster.size) if cluster.size > 1 else 0.0
        ),
    }


def _numerical_audit(
    model: BayesianRAPTQModel,
    contexts: Sequence[ObservationContext],
    experiment: ExperimentConfig,
    threshold: float,
    rate_model: QuantizedRelativePhase24,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    rows: list[dict[str, float | int | str]] = []
    condmi = ContinuousPhaseCondMI24(model, experiment.condmi_phase_nodes)
    condmi_reference = ContinuousPhaseCondMI24(
        model, experiment.condmi_reference_phase_nodes
    )
    model_qmc = QMCConfig(
        power=experiment.relative_phase_power,
        repeats=experiment.relative_phase_repeats,
        reference_power=experiment.relative_phase_power + 1,
        reference_context_limit=1,
        proposal="model_radial",
    )
    defensive_qmc = QMCConfig(
        power=experiment.relative_phase_power,
        repeats=experiment.relative_phase_repeats,
        reference_power=experiment.relative_phase_power + 1,
        reference_context_limit=1,
        proposal="defensive_observed",
    )
    rate_model_qmc = RelativePhaseQMCConfig(
        experiment.relative_phase_power,
        experiment.relative_phase_repeats,
        "model_radial",
    )
    rate_defensive_qmc = RelativePhaseQMCConfig(
        experiment.relative_phase_power,
        experiment.relative_phase_repeats,
        "defensive_mixture",
    )
    for context in contexts:
        posteriors = {
            "CondMI": condmi.posterior(context, threshold),
            "CondMI-Reference": condmi_reference.posterior(context, threshold),
            "RelativePhase-Model": model.posterior_representation(
                context,
                "relative_phase",
                model_qmc,
                2,
                _seed(context, "relative-phase-model"),
            ),
            "RelativePhase-Defensive": model.posterior_representation(
                context,
                "relative_phase",
                defensive_qmc,
                2,
                _seed(context, "relative-phase-defensive"),
            ),
            "RelativePhase24-Model": rate_model.posterior(
                context,
                rate_model_qmc,
                _seed(context, "relative-phase24-model"),
            ),
            "RelativePhase24-Defensive": rate_model.posterior(
                context,
                rate_defensive_qmc,
                _seed(context, "relative-phase24-defensive"),
            ),
        }
        pairs = (
            ("CondMI", "CondMI-Reference", "CondMI-phase-quadrature"),
            ("RelativePhase-Model", "RelativePhase-Defensive", "RelativePhase-QMC"),
            (
                "RelativePhase24-Model",
                "RelativePhase24-Defensive",
                "RelativePhase24-QMC",
            ),
        )
        for first, second, component in pairs:
            first_summary = model.summarize_posterior(
                posteriors[first], context.true_tau
            )
            second_summary = model.summarize_posterior(
                posteriors[second], context.true_tau
            )
            rows.append(
                {
                    "component": component,
                    "seed": context.seed,
                    "trial": context.trial,
                    "snr_db": context.snr_db,
                    "posterior_tv": posterior_total_variation(
                        posteriors[first], posteriors[second]
                    ),
                    "risk_difference_samples2": abs(
                        first_summary.posterior_risk - second_summary.posterior_risk
                    ),
                    "estimate_difference_samples": abs(
                        first_summary.estimate - second_summary.estimate
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    by_component = frame.groupby("component")["risk_difference_samples2"].max()
    checks = {
        "condmi_max_risk_difference": float(by_component["CondMI-phase-quadrature"]),
        "relative_phase_max_risk_difference": float(by_component["RelativePhase-QMC"]),
        "relative_phase24_max_risk_difference": float(
            by_component["RelativePhase24-QMC"]
        ),
    }
    checks["all_numerical_risk_differences_within_guard"] = bool(
        max(checks.values()) <= DELTA_GUARD_SAMPLES2
    )
    return frame, checks


def _evaluate(
    model: BayesianRAPTQModel,
    contexts: Sequence[ObservationContext],
    experiment: ExperimentConfig,
    threshold: float,
    rate_model: QuantizedRelativePhase24,
    include_rate_method: bool,
    include_reference_methods: bool = True,
    partial_path: Path | None = None,
) -> pd.DataFrame:
    condmi = ContinuousPhaseCondMI24(model, experiment.condmi_phase_nodes)
    continuous_qmc = QMCConfig(
        power=experiment.relative_phase_power,
        repeats=experiment.relative_phase_repeats,
        reference_power=experiment.relative_phase_power + 1,
        reference_context_limit=1,
        proposal="model_radial",
    )
    rate_qmc = RelativePhaseQMCConfig(
        experiment.relative_phase_power,
        experiment.relative_phase_repeats,
        "model_radial",
    )
    rows: list[dict[str, float | int | str | bool]] = []
    total = len(contexts)
    start = time.perf_counter()
    for index, context in enumerate(contexts, start=1):
        methods: tuple[tuple[str, float, Callable[[], np.ndarray]], ...] = ()
        if include_reference_methods:
            methods += (
                ("FullComplex", math.nan, lambda: model.posterior_full(context)),
                (
                    "RelativePhase-Continuous",
                    math.nan,
                    lambda: model.posterior_representation(
                        context,
                        "relative_phase",
                        continuous_qmc,
                        2,
                        _seed(context, "evaluation-relative-phase"),
                    ),
                ),
                (
                    "CondMI-Matched24",
                    24.0,
                    lambda: condmi.posterior(context, threshold),
                ),
            )
        if include_rate_method:
            methods += (
                (
                    "RAPTQ-RelativePhase24",
                    24.0,
                    lambda: rate_model.posterior(
                        context,
                        rate_qmc,
                        _seed(context, "evaluation-relative-phase24"),
                    ),
                ),
            )
        for method, bits, evaluator in methods:
            method_start = time.perf_counter()
            posterior = evaluator()
            rows.append(
                _summary_row(
                    model,
                    context,
                    method,
                    posterior,
                    bits,
                    time.perf_counter() - method_start,
                )
            )
        if index % max(1, total // 20) == 0 or index == total:
            elapsed = time.perf_counter() - start
            if partial_path is not None:
                pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(
                f"[Evaluation] {index}/{total} elapsed={elapsed:.1f}s "
                f"ETA={elapsed / index * (total - index):.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for (method, snr), group in rows.groupby(["method", "snr_db"], sort=False):
        squared = group["squared_error"].to_numpy(float)
        absolute = group["abs_error"].to_numpy(float)
        records.append(
            {
                "method": method,
                "snr_db": float(snr),
                "rmse_samples": float(np.sqrt(np.mean(squared))),
                "mae_samples": float(np.mean(absolute)),
                "p95_abs_error_samples": float(np.quantile(absolute, 0.95)),
                "gross_error_gt_1_rate": float(np.mean(absolute > 1.0)),
                "mean_posterior_risk": float(group["posterior_risk"].mean()),
                "posterior_nll": float(group["posterior_nll"].mean()),
                "credible90_coverage": float(
                    group["credible90_contains_true"].mean()
                ),
                "mean_elapsed_ms_per_context": float(
                    1000.0 * group["elapsed_seconds"].mean()
                ),
                "n_contexts": int(group.shape[0]),
            }
        )
    return pd.DataFrame(records)


def _gate(
    mode: str,
    mechanical: dict[str, bool | float | int],
    threshold_boundary: bool,
    numerical: dict[str, float | bool],
    effects: pd.DataFrame,
) -> str:
    if mode == "smoke":
        return "SMOKE_ONLY"
    if not bool(mechanical["all_passed"]):
        return "INVALID_MECHANICAL_OR_FAIRNESS"
    if threshold_boundary:
        return "HOLD_CONDMI_THRESHOLD_BOUNDARY"
    if not bool(numerical["all_numerical_risk_differences_within_guard"]):
        return "HOLD_NUMERICAL_CONVERGENCE"
    lookup = effects.set_index(["candidate", "baseline", "metric"])
    information_loss = lookup.loc[
        ("RelativePhase-Continuous", "FullComplex", "posterior_risk")
    ]
    if float(information_loss["ci95_low"]) > DELTA_GUARD_SAMPLES2:
        return "INVALID_DATA_PROCESSING_INEQUALITY"
    headroom = lookup.loc[
        ("RelativePhase-Continuous", "CondMI-Matched24", "posterior_risk")
    ]
    rate_risk = lookup.loc[
        ("RAPTQ-RelativePhase24", "CondMI-Matched24", "posterior_risk")
    ]
    rate_mse = lookup.loc[
        ("RAPTQ-RelativePhase24", "CondMI-Matched24", "squared_error")
    ]
    rate_gross = lookup.loc[
        ("RAPTQ-RelativePhase24", "CondMI-Matched24", "gross_error_gt_1")
    ]
    if float(headroom["ci95_high"]) <= 0.0:
        return "NO_GO_RELATIVE_PHASE_RATE_HEADROOM"
    if float(headroom["ci95_low"]) <= DELTA_GUARD_SAMPLES2:
        return "HOLD_RELATIVE_PHASE_HEADROOM"
    if float(rate_risk["ci95_high"]) <= 0.0 or float(rate_mse["ci95_high"]) <= 0.0:
        return "NO_GO_RELATIVE_PHASE24"
    if (
        float(rate_risk["ci95_low"]) <= DELTA_GUARD_SAMPLES2
        or float(rate_mse["ci95_low"]) <= 0.0
    ):
        return "HOLD_RELATIVE_PHASE24_EFFECT"
    if float(rate_gross["ci95_high"]) < 0.0:
        return "NO_GO_RELATIVE_PHASE24_TAIL"
    if float(rate_gross["ci95_low"]) < 0.0:
        return "HOLD_RELATIVE_PHASE24_TAIL"
    return "GO_COMPONENT_RELATIVE_PHASE24"


def _plot(summary: pd.DataFrame, rows: pd.DataFrame, output: Path) -> None:
    style = {
        "FullComplex": ("#555555", "D", ":"),
        "RelativePhase-Continuous": ("#9467bd", "^", "--"),
        "CondMI-Matched24": ("#1f77b4", "s", "-"),
        "RAPTQ-RelativePhase24": ("#d62728", "o", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    for method, group in summary.groupby("method", sort=False):
        color, marker, line = style[method]
        axes[0].plot(
            group["snr_db"],
            group["rmse_samples"],
            color=color,
            marker=marker,
            linestyle=line,
            linewidth=1.35,
            markersize=4.2,
            label=method,
        )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("TDOA RMSE (samples)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=7.5)

    baseline = "CondMI-Matched24"
    for method in ("RelativePhase-Continuous", "RAPTQ-RelativePhase24"):
        if method not in set(rows["method"]):
            continue
        pivot = rows[rows["method"].isin([method, baseline])].pivot_table(
            index=["seed", "trial", "snr_db"],
            columns="method",
            values="posterior_risk",
            aggfunc="first",
        ).dropna()
        gain = (pivot[baseline] - pivot[method]).groupby(level=2).mean()
        color, marker, line = style[method]
        axes[1].plot(
            gain.index,
            gain.values,
            color=color,
            marker=marker,
            linestyle=line,
            linewidth=1.35,
            markersize=4.2,
            label=method,
        )
    axes[1].axhline(0.0, color="#333333", linewidth=0.85)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Bayes-risk gain over CondMI (samples$^2$)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=7.5)
    fig.savefig(output / "Fig1_RAPTQ_RelativePhase24_Feasibility.svg", dpi=300)
    plt.close(fig)


def _write_early_stop(
    output: Path,
    experiment: ExperimentConfig,
    model: BayesianRAPTQModel,
    code_config: RelativePhaseCodeConfig,
    mechanical: dict[str, bool | float | int],
    threshold: float,
    threshold_boundary: bool,
    numerical: dict[str, float | bool],
    source_results: Path,
    source_hash_before: str,
    source_files: Sequence[Path],
    status: str,
    formal_performance_evidence: bool = False,
) -> str:
    source_hash_after = _tree_hash(source_results)
    if source_hash_before != source_hash_after:
        status = "INVALID_FROZEN_SOURCE_MUTATION"
    manifest = {
        "experiment": asdict(experiment),
        "model": asdict(model.config),
        "relative_phase_code": asdict(code_config),
        "condmi_matched24": {
            "threshold": threshold,
            "at_search_boundary": threshold_boundary,
        },
        "delta_guard_samples2": DELTA_GUARD_SAMPLES2,
        "source_result": str(source_results),
        "source_tree_hash_before": source_hash_before,
        "source_tree_hash_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
        "source_hash": _source_hash(source_files),
        "gate_status": status,
        "early_stop_before_performance_evaluation": True,
    }
    validation = {
        **mechanical,
        **numerical,
        "threshold_at_search_boundary": threshold_boundary,
        "formal_performance_evidence": formal_performance_evidence,
        "gate_status": status,
    }
    (output / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return status


def _frozen_model_matches(source_results: Path, model: ModelConfig) -> bool:
    manifest = json.loads(
        (source_results / "config_manifest.json").read_text(encoding="utf-8")
    )
    frozen = manifest.get("model_config")
    if not isinstance(frozen, dict):
        return False
    current = asdict(model)
    normalized_current = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in current.items()
    }
    return frozen == normalized_current


def _headroom_status(effect: dict[str, float | int | str]) -> str:
    if float(effect["ci95_high"]) <= 0.0:
        return "NO_GO_RELATIVE_PHASE_RATE_HEADROOM"
    if float(effect["ci95_low"]) <= DELTA_GUARD_SAMPLES2:
        return "HOLD_RELATIVE_PHASE_HEADROOM"
    return "HEADROOM_CONFIRMED"


def _continuous_stage_status(
    information_effect: dict[str, float | int | str],
    headroom_effect: dict[str, float | int | str],
) -> str:
    if float(information_effect["ci95_low"]) > DELTA_GUARD_SAMPLES2:
        return "INVALID_DATA_PROCESSING_INEQUALITY"
    return _headroom_status(headroom_effect)


def run(mode: str) -> Path:
    experiment = build_experiment_config(mode)
    model = BayesianRAPTQModel(ModelConfig())
    code_config = RelativePhaseCodeConfig()
    rate_model = QuantizedRelativePhase24(model, code_config)
    root = Path(__file__).resolve().parents[1]
    source_results = (
        root
        / "运行结果"
        / "RAPTQ"
        / "RAPTQ_STAGE0_20260805_105037_development"
    )
    if not source_results.exists():
        raise FileNotFoundError(f"frozen Stage 0 source is missing: {source_results}")
    source_hash_before = _tree_hash(source_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        root
        / "运行结果"
        / "RAPTQ"
        / f"RAPTQ_RELATIVE_PHASE_RATE_{timestamp}_{mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    print("RAPTQ RelativePhase final bounded feasibility cycle", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        f"Payloads: CondMI=24 bit, RelativePhase={code_config.total_payload_bits} bit; "
        f"bits={list(code_config.bits_by_relative_phase)}",
        flush=True,
    )

    mechanical: dict[str, bool | float | int] = {
        "design_evaluation_seeds_disjoint": not bool(
            set(experiment.design_seeds) & set(experiment.evaluation_seeds)
        ),
        "condmi_payload_bits": ContinuousPhaseCondMI24(
            model, experiment.condmi_phase_nodes
        ).total_payload_bits,
        "relative_phase_payload_bits": rate_model.total_payload_bits,
        "payloads_equal_24": bool(
            ContinuousPhaseCondMI24(model, experiment.condmi_phase_nodes).total_payload_bits
            == rate_model.total_payload_bits
            == 24
        ),
        "no_index_bits": True,
        "no_feedback_bits": True,
        "fixed_selected_bins": tuple(model.config.selected_bins)
        == (31, 94, 145, 196, 247, 298),
        "frozen_stage0_model_matches": _frozen_model_matches(
            source_results, model.config
        ),
        "minimum_marginal_leverage_phase_gets_four_bits": bool(
            code_config.bits_by_relative_phase[
                int(
                    np.argmin(
                        model.source_power[1:]
                        * (model.omega[1:] - model.omega[0]) ** 2
                    )
                )
            ]
            == 4
            and sum(bits == 4 for bits in code_config.bits_by_relative_phase) == 1
        ),
        "maximum_phase_quantization_error_rad": relative_phase_quantization_max_error(
            code_config.bits_by_relative_phase
        ),
    }
    mechanical["all_passed"] = all(
        value for value in mechanical.values() if isinstance(value, bool)
    )
    if not bool(mechanical["all_passed"]):
        raise RuntimeError(f"mechanical checks failed: {mechanical}")

    design_contexts = model.build_contexts(
        experiment.design_seeds,
        experiment.design_trials_per_seed,
        experiment.snr_db_values,
    )
    condmi = ContinuousPhaseCondMI24(model, experiment.condmi_phase_nodes)
    print("[Design] freezing CondMI-Matched24 threshold", flush=True)
    threshold_search = condmi.search_threshold(
        design_contexts,
        experiment.threshold_candidates,
        progress_callback=lambda index, total, threshold, objective: print(
            f"[Design] {index}/{total} threshold={threshold:.6g} "
            f"entropy={objective:.6g}",
            flush=True,
        ),
    )
    threshold_rows = pd.DataFrame(threshold_search.rows)
    threshold_rows["selected"] = np.isclose(
        threshold_rows["threshold"], threshold_search.threshold
    )
    selected_index = int(
        np.flatnonzero(threshold_rows["selected"].to_numpy(bool))[0]
    )
    local_reference_indices = tuple(
        index
        for index in (selected_index - 1, selected_index, selected_index + 1)
        if 0 <= index < threshold_rows.shape[0]
    )
    condmi_reference = ContinuousPhaseCondMI24(
        model, experiment.condmi_reference_phase_nodes
    )
    threshold_rows["reference_objective_value"] = math.nan
    for index in local_reference_indices:
        threshold_rows.loc[index, "reference_objective_value"] = (
            condmi_reference.mean_entropy(
                design_contexts,
                float(threshold_rows.loc[index, "threshold"]),
            )
        )
    selected_reference = float(
        threshold_rows.loc[selected_index, "reference_objective_value"]
    )
    neighbour_reference = [
        float(threshold_rows.loc[index, "reference_objective_value"])
        for index in local_reference_indices
        if index != selected_index
    ]
    threshold_reference_local_minimum = bool(
        neighbour_reference
        and selected_reference <= min(neighbour_reference) + 1e-12
    )
    threshold_rows.to_csv(output / "condmi_matched24_threshold_design.csv", index=False)
    print(
        f"[Design] threshold={threshold_search.threshold:.6g}, "
        f"entropy={threshold_search.objective_value:.6g}, "
        f"boundary={threshold_search.at_boundary}",
        flush=True,
    )

    numerical_contexts = [
        context
        for context in design_contexts
        if context.trial < experiment.numerical_trials_per_seed
    ]
    print(f"[Numerical] contexts={len(numerical_contexts)}", flush=True)
    numerical_rows, numerical = _numerical_audit(
        model,
        numerical_contexts,
        experiment,
        threshold_search.threshold,
        rate_model,
    )
    numerical["condmi_threshold_reference_local_minimum"] = (
        threshold_reference_local_minimum
    )
    numerical["all_numerical_risk_differences_within_guard"] = bool(
        numerical["all_numerical_risk_differences_within_guard"]
        and threshold_reference_local_minimum
    )
    numerical_rows.to_csv(output / "numerical_audit.csv", index=False)

    source_files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("relative_phase_rate.py"),
        Path(__file__).with_name("core.py"),
    )
    if mode == "development" and (
        threshold_search.at_boundary
        or not bool(numerical["all_numerical_risk_differences_within_guard"])
    ):
        early_status = (
            "HOLD_CONDMI_THRESHOLD_BOUNDARY"
            if threshold_search.at_boundary
            else "HOLD_NUMERICAL_CONVERGENCE"
        )
        early_status = _write_early_stop(
            output,
            experiment,
            model,
            code_config,
            mechanical,
            threshold_search.threshold,
            threshold_search.at_boundary,
            numerical,
            source_results,
            source_hash_before,
            source_files,
            early_status,
        )
        print(f"[Gate] {early_status}", flush=True)
        print("[Stop] performance evaluation was not executed", flush=True)
        print(f"[Done] {output}", flush=True)
        return output

    evaluation_contexts = model.build_contexts(
        experiment.evaluation_seeds,
        experiment.evaluation_trials_per_seed,
        experiment.snr_db_values,
    )
    rows = _evaluate(
        model,
        evaluation_contexts,
        experiment,
        threshold_search.threshold,
        rate_model,
        include_rate_method=mode == "smoke",
        include_reference_methods=True,
        partial_path=output / "partial_continuous_rows.csv",
    )
    continuous_effects = [
        _paired_effect(
            rows,
            "RelativePhase-Continuous",
            "FullComplex",
            "posterior_risk",
            experiment.bootstrap_repetitions,
            95001,
        ),
        _paired_effect(
            rows,
            "RelativePhase-Continuous",
            "CondMI-Matched24",
            "posterior_risk",
            experiment.bootstrap_repetitions,
            95002,
        ),
    ]
    headroom_status = _continuous_stage_status(
        continuous_effects[0], continuous_effects[1]
    )
    if mode == "development" and headroom_status != "HEADROOM_CONFIRMED":
        summary = _summarize(rows)
        effects = pd.DataFrame(continuous_effects)
        rows.to_csv(output / "relative_phase_rate_rows.csv", index=False)
        summary.to_csv(output / "relative_phase_rate_summary_by_snr.csv", index=False)
        effects.to_csv(output / "relative_phase_rate_paired_effects.csv", index=False)
        _plot(summary, rows, output)
        headroom_status = _write_early_stop(
            output,
            experiment,
            model,
            code_config,
            mechanical,
            threshold_search.threshold,
            threshold_search.at_boundary,
            numerical,
            source_results,
            source_hash_before,
            source_files,
            headroom_status,
            formal_performance_evidence=True,
        )
        print("\nContinuous-information effects", flush=True)
        print(effects.to_string(index=False), flush=True)
        print(f"\n[Gate] {headroom_status}", flush=True)
        print("[Stop] 24-bit RelativePhase performance was not executed", flush=True)
        print(f"[Done] {output}", flush=True)
        (output / "partial_continuous_rows.csv").unlink(missing_ok=True)
        return output

    if mode == "development":
        rate_rows = _evaluate(
            model,
            evaluation_contexts,
            experiment,
            threshold_search.threshold,
            rate_model,
            include_rate_method=True,
            include_reference_methods=False,
            partial_path=output / "partial_rate_rows.csv",
        )
        rows = pd.concat(
            [rows, rate_rows[rate_rows["method"] == "RAPTQ-RelativePhase24"]],
            ignore_index=True,
        )
    summary = _summarize(rows)
    effects = pd.DataFrame(
        continuous_effects
        + [
            _paired_effect(
                rows,
                "RAPTQ-RelativePhase24",
                "CondMI-Matched24",
                "posterior_risk",
                experiment.bootstrap_repetitions,
                95003,
            ),
            _paired_effect(
                rows,
                "RAPTQ-RelativePhase24",
                "CondMI-Matched24",
                "squared_error",
                experiment.bootstrap_repetitions,
                95004,
            ),
            _paired_effect(
                rows,
                "RAPTQ-RelativePhase24",
                "CondMI-Matched24",
                "gross_error_gt_1",
                experiment.bootstrap_repetitions,
                95005,
            ),
        ]
    )
    status = _gate(
        mode,
        mechanical,
        threshold_search.at_boundary,
        numerical,
        effects,
    )

    rows.to_csv(output / "relative_phase_rate_rows.csv", index=False)
    summary.to_csv(output / "relative_phase_rate_summary_by_snr.csv", index=False)
    effects.to_csv(output / "relative_phase_rate_paired_effects.csv", index=False)
    _plot(summary, rows, output)

    source_hash_after = _tree_hash(source_results)
    source_unchanged = source_hash_before == source_hash_after
    if not source_unchanged:
        status = "INVALID_FROZEN_SOURCE_MUTATION"
    manifest = {
        "experiment": asdict(experiment),
        "model": asdict(model.config),
        "relative_phase_code": asdict(code_config),
        "condmi_matched24": {
            "threshold": threshold_search.threshold,
            "objective": "conditional posterior entropy",
            "continuous_phase_nodes": experiment.condmi_phase_nodes,
            "reference_phase_nodes": experiment.condmi_reference_phase_nodes,
            "at_search_boundary": threshold_search.at_boundary,
        },
        "budget": {
            "condmi_payload_bits": 24,
            "relative_phase_payload_bits": 24,
            "index_bits": 0,
            "feedback_bits": 0,
            "fixed_bins": list(model.config.selected_bins),
        },
        "information_boundary": {
            "encoder": "remote normalized selected-bin observation only",
            "decoder_side_information": "full reference selected-bin observation",
            "public": "SNR, source-power prior, gain-amplitude prior, fixed codebook",
            "forbidden": "true tau, true gain, clean source, evaluation outcomes",
        },
        "delta_guard_samples2": DELTA_GUARD_SAMPLES2,
        "source_result": str(source_results),
        "source_tree_hash_before": source_hash_before,
        "source_tree_hash_after": source_hash_after,
        "source_unchanged": source_unchanged,
        "source_hash": _source_hash(source_files),
        "gate_status": status,
    }
    (output / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validation = {
        **mechanical,
        **numerical,
        "threshold_at_search_boundary": threshold_search.at_boundary,
        "expected_contexts": len(evaluation_contexts),
        "actual_contexts_per_method": {
            method: int(count) for method, count in rows.groupby("method").size().items()
        },
        "evaluation_keys_unique": bool(
            not rows.duplicated(["method", "seed", "trial", "snr_db"]).any()
        ),
        "all_outputs_finite": bool(
            np.isfinite(
                rows[
                    [
                        "estimate",
                        "squared_error",
                        "posterior_risk",
                        "posterior_entropy",
                        "posterior_nll",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "frozen_source_unchanged": source_unchanged,
        "formal_performance_evidence": mode == "development",
        "gate_status": status,
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nPaired effects", flush=True)
    print(effects.to_string(index=False), flush=True)
    print(f"\n[Gate] {status}", flush=True)
    print(f"[Done] {output}", flush=True)
    (output / "partial_continuous_rows.csv").unlink(missing_ok=True)
    (output / "partial_rate_rows.csv").unlink(missing_ok=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a tiny engineering check; never use it as performance evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run("smoke" if args.smoke else RUN_MODE)


if __name__ == "__main__":
    main()
