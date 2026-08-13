"""Read-only threshold-selection stability certificate for SITQ Stage A.

Run this file directly in PyCharm for the full diagnostic.  The script rebuilds
only the Stage A design trajectories from the frozen manifest.  It never reads
evaluation outcomes when selecting or comparing thresholds and does not modify
the Stage A model, quantizer, frequency set, or source result directory.
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

from sitq_tdoa.core import ModelConfig, ObservationContext, SITQModel  # noqa: E402


SOURCE_RESULT_DIR = Path(
    r"F:\PythonWorkspace\DAE Data Compression\运行结果\SITQ\SITQ_STAGE_A_20260803_132419_development"
)
RUN_MODE = "diagnostic"


@dataclass(frozen=True)
class DiagnosticConfig:
    mode: str
    bootstrap_repetitions: int
    bootstrap_seed: int
    crossfreeze_folds: int
    smoke_trials_per_seed: int | None = None
    smoke_snr_count: int | None = None

    def validate(self) -> None:
        if self.mode not in {"smoke", "diagnostic"}:
            raise ValueError("mode must be smoke or diagnostic")
        if self.bootstrap_repetitions < 100:
            raise ValueError("bootstrap_repetitions must be at least 100")
        if self.crossfreeze_folds < 2:
            raise ValueError("crossfreeze_folds must be at least two")


def build_diagnostic_config(mode: str) -> DiagnosticConfig:
    if mode == "smoke":
        config = DiagnosticConfig(
            mode="smoke",
            bootstrap_repetitions=200,
            bootstrap_seed=202608031,
            crossfreeze_folds=2,
            smoke_trials_per_seed=4,
            smoke_snr_count=2,
        )
    elif mode == "diagnostic":
        config = DiagnosticConfig(
            mode="diagnostic",
            bootstrap_repetitions=5000,
            bootstrap_seed=202608031,
            crossfreeze_folds=6,
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


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.iterdir(), key=lambda value: value.name):
        if item.is_file():
            digest.update(item.name.encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _load_source(source_dir: Path) -> tuple[dict, dict]:
    required = {
        "config_manifest.json",
        "validation.json",
        "stage_a_rows.csv",
        "threshold_design.csv",
    }
    missing = sorted(name for name in required if not (source_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"source result is incomplete: {missing}")
    manifest = json.loads(
        (source_dir / "config_manifest.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (source_dir / "validation.json").read_text(encoding="utf-8")
    )
    return manifest, validation


def _model_config_from_manifest(manifest: dict) -> ModelConfig:
    values = dict(manifest["model"])
    values["selected_bins"] = tuple(int(value) for value in values["selected_bins"])
    values["rho_amplitudes"] = tuple(
        float(value) for value in values["rho_amplitudes"]
    )
    return ModelConfig(**values)


def _design_contexts(
    model: SITQModel,
    experiment: dict,
    config: DiagnosticConfig,
) -> list[ObservationContext]:
    design_seeds = tuple(int(value) for value in experiment["design_seeds"])
    evaluation_seeds = tuple(int(value) for value in experiment["evaluation_seeds"])
    if set(design_seeds) & set(evaluation_seeds):
        raise RuntimeError("source design and evaluation seeds overlap")
    trials_per_seed = int(experiment["design_trials_per_seed"])
    snr_values = tuple(float(value) for value in experiment["snr_db_values"])
    if config.mode == "smoke":
        trials_per_seed = min(trials_per_seed, int(config.smoke_trials_per_seed or 1))
        count = min(len(snr_values), int(config.smoke_snr_count or 1))
        indices = np.linspace(0, len(snr_values) - 1, count, dtype=int)
        snr_values = tuple(snr_values[index] for index in indices)
    contexts: list[ObservationContext] = []
    for seed in design_seeds:
        for trial in model.generate_trials(seed, trials_per_seed):
            contexts.extend(model.build_context(trial, snr) for snr in snr_values)
    return contexts


def _compute_context_metrics(
    model: SITQModel,
    contexts: Sequence[ObservationContext],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    total = len(contexts) * thresholds.size
    done = 0
    start = time.perf_counter()
    for context in contexts:
        for threshold in thresholds:
            posterior = model.posterior_quantized(context, float(threshold))
            summary = model.summarize_posterior(posterior, context.true_tau)
            rows.append(
                {
                    "seed": context.seed,
                    "trial": context.trial,
                    "snr_db": context.snr_db,
                    "threshold": float(threshold),
                    "posterior_risk": summary.posterior_risk,
                    "posterior_entropy": summary.entropy,
                }
            )
            done += 1
        if done % max(thresholds.size, total // 10) == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[Metrics] {done}/{total} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def _cluster_metrics(context_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        context_metrics.groupby(["seed", "trial", "threshold"], as_index=False)
        .agg(
            posterior_risk=("posterior_risk", "mean"),
            posterior_entropy=("posterior_entropy", "mean"),
            n_snr=("snr_db", "nunique"),
        )
        .sort_values(["seed", "trial", "threshold"])
        .reset_index(drop=True)
    )


def _metric_matrix(
    cluster_metrics: pd.DataFrame,
    thresholds: np.ndarray,
    metric: str,
) -> tuple[pd.MultiIndex, np.ndarray]:
    pivot = cluster_metrics.pivot(
        index=["seed", "trial"], columns="threshold", values=metric
    ).sort_index()
    pivot = pivot.reindex(columns=thresholds)
    if pivot.isna().any().any():
        raise RuntimeError(f"{metric} matrix is incomplete")
    return pivot.index, pivot.to_numpy(float)


def _threshold_index(thresholds: np.ndarray, value: float) -> int:
    matches = np.flatnonzero(np.isclose(thresholds, value, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise RuntimeError(f"threshold {value} is not unique in the frozen grid")
    return int(matches[0])


def _bootstrap_selection(
    risk_matrix: np.ndarray,
    entropy_matrix: np.ndarray,
    thresholds: np.ndarray,
    sitq_threshold: float,
    condmi_threshold: float,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, risk_matrix.shape[0], size=(repetitions, risk_matrix.shape[0])
    )
    bootstrap_risk = risk_matrix[sample_indices].mean(axis=1)
    bootstrap_entropy = entropy_matrix[sample_indices].mean(axis=1)
    sitq_selected = np.argmin(bootstrap_risk, axis=1)
    condmi_selected = np.argmin(bootstrap_entropy, axis=1)
    sitq_index = _threshold_index(thresholds, sitq_threshold)
    condmi_index = _threshold_index(thresholds, condmi_threshold)
    fixed_delta = (
        risk_matrix[:, condmi_index] - risk_matrix[:, sitq_index]
    )
    fixed_bootstrap = fixed_delta[sample_indices].mean(axis=1)
    rows = pd.DataFrame(
        {
            "replicate": np.arange(repetitions),
            "sitq_selected_threshold": thresholds[sitq_selected],
            "condmi_selected_threshold": thresholds[condmi_selected],
            "fixed_threshold_risk_gain": fixed_bootstrap,
        }
    )
    summary = {
        "fixed_threshold_risk_gain": float(np.mean(fixed_delta)),
        "fixed_threshold_ci95_low": float(np.quantile(fixed_bootstrap, 0.025)),
        "fixed_threshold_ci95_high": float(np.quantile(fixed_bootstrap, 0.975)),
        "sitq_original_selection_rate": float(np.mean(sitq_selected == sitq_index)),
        "condmi_original_selection_rate": float(
            np.mean(condmi_selected == condmi_index)
        ),
        "sitq_unique_selected_thresholds": int(np.unique(sitq_selected).size),
        "condmi_unique_selected_thresholds": int(np.unique(condmi_selected).size),
    }
    return rows, summary


def _crossfreeze(
    cluster_index: pd.MultiIndex,
    risk_matrix: np.ndarray,
    entropy_matrix: np.ndarray,
    thresholds: np.ndarray,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = cluster_index.get_level_values("trial").to_numpy(int)
    fold_ids = trials % folds
    if np.unique(fold_ids).size != folds:
        raise RuntimeError("not every cross-freeze fold received a trajectory")
    row_records: list[dict[str, float | int]] = []
    fold_records: list[dict[str, float | int]] = []
    for fold in range(folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        sitq_index = int(np.argmin(risk_matrix[train_mask].mean(axis=0)))
        condmi_index = int(np.argmin(entropy_matrix[train_mask].mean(axis=0)))
        deltas = (
            risk_matrix[test_mask, condmi_index]
            - risk_matrix[test_mask, sitq_index]
        )
        test_positions = np.flatnonzero(test_mask)
        for position, delta in zip(test_positions, deltas, strict=True):
            row_records.append(
                {
                    "seed": int(cluster_index[position][0]),
                    "trial": int(cluster_index[position][1]),
                    "fold": fold,
                    "sitq_threshold": float(thresholds[sitq_index]),
                    "condmi_threshold": float(thresholds[condmi_index]),
                    "sitq_risk": float(risk_matrix[position, sitq_index]),
                    "condmi_risk": float(risk_matrix[position, condmi_index]),
                    "risk_gain": float(delta),
                }
            )
        fold_records.append(
            {
                "fold": fold,
                "n_train_clusters": int(np.sum(train_mask)),
                "n_test_clusters": int(np.sum(test_mask)),
                "sitq_threshold": float(thresholds[sitq_index]),
                "condmi_threshold": float(thresholds[condmi_index]),
                "mean_risk_gain": float(np.mean(deltas)),
            }
        )
    rows = pd.DataFrame(row_records).sort_values(["seed", "trial"])
    if rows.duplicated(["seed", "trial"]).any():
        raise RuntimeError("cross-freeze assigned a trajectory more than once")
    if rows.shape[0] != risk_matrix.shape[0]:
        raise RuntimeError("cross-freeze did not cover every trajectory")
    return rows, pd.DataFrame(fold_records)


def _cluster_bootstrap(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("bootstrap values must be a finite one-dimensional sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    sampled = values[indices].mean(axis=1)
    standard_deviation = float(np.std(values, ddof=1))
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "cluster_sd": standard_deviation,
        "approx_mde95": float(1.96 * standard_deviation / math.sqrt(values.size)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "n_clusters": int(values.size),
    }


def _seed_holdout_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("seed", as_index=False)
        .agg(
            mean_risk_gain=("risk_gain", "mean"),
            median_risk_gain=("risk_gain", "median"),
            positive_fraction=("risk_gain", lambda values: float(np.mean(values > 0.0))),
            n_clusters=("risk_gain", "size"),
        )
        .sort_values("seed")
    )


def decide_gate(
    mechanical_passed: bool,
    crossfreeze: dict[str, float | int],
    seed_summary: pd.DataFrame,
) -> str:
    if not mechanical_passed:
        return "INVALID_STAGE_A1_DIAGNOSTIC"
    seed_values = seed_summary["mean_risk_gain"].to_numpy(float)
    mean_gain = float(crossfreeze["mean"])
    ci_low = float(crossfreeze["ci95_low"])
    ci_high = float(crossfreeze["ci95_high"])
    if mean_gain > 0.0 and ci_low > 0.0 and np.all(seed_values > 0.0):
        return "STABLE_SELECTION_SIGNAL"
    if ci_high <= 0.0 or (mean_gain <= 0.0 and np.all(seed_values <= 0.0)):
        return "NO_GO_COMPONENT_SINGLE_GLOBAL_THRESHOLD"
    return "HOLD_SELECTION_UNCERTAIN"


def _selection_frequency(
    bootstrap_rows: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for objective, column in (
        ("SITQ-BayesRisk", "sitq_selected_threshold"),
        ("CondMI", "condmi_selected_threshold"),
    ):
        counts = bootstrap_rows[column].value_counts().sort_index()
        for threshold, count in counts.items():
            records.append(
                {
                    "objective": objective,
                    "threshold": float(threshold),
                    "count": int(count),
                    "selection_rate": float(count / bootstrap_rows.shape[0]),
                }
            )
    return pd.DataFrame(records)


def _plot(
    cluster_metrics: pd.DataFrame,
    selection_frequency: pd.DataFrame,
    output: Path,
) -> None:
    mean_curve = (
        cluster_metrics.groupby("threshold", as_index=False)[
            ["posterior_risk", "posterior_entropy"]
        ]
        .mean()
        .sort_values("threshold")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    axes[0].plot(
        mean_curve["threshold"],
        mean_curve["posterior_risk"],
        color="#d1495b",
        marker="o",
        markersize=3.5,
        linewidth=1.35,
        label="Bayes risk",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Quantizer threshold")
    axes[0].set_ylabel("Mean posterior risk (samples$^2$)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    colors = {"SITQ-BayesRisk": "#d1495b", "CondMI": "#2a9d8f"}
    for objective, group in selection_frequency.groupby("objective", sort=False):
        axes[1].plot(
            group["threshold"],
            group["selection_rate"],
            color=colors[objective],
            marker="o",
            linewidth=1.35,
            markersize=4,
            label=objective,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Selected threshold")
    axes[1].set_ylabel("Bootstrap selection rate")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(output / "Fig1_SITQ_StageA1_ThresholdStability.svg", dpi=300)
    plt.close(fig)


def run(mode: str) -> Path:
    config = build_diagnostic_config(mode)
    source_dir = SOURCE_RESULT_DIR.resolve()
    manifest, source_validation = _load_source(source_dir)
    source_hash_before = _directory_hash(source_dir)
    root = Path(__file__).resolve().parents[1]
    output = (
        root
        / "运行结果"
        / "SITQ"
        / f"SITQ_STAGE_A1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
    )
    output.mkdir(parents=True, exist_ok=False)

    model = SITQModel(_model_config_from_manifest(manifest))
    thresholds = np.asarray(
        manifest["experiment"]["threshold_candidates"], dtype=float
    )
    sitq_threshold = float(
        manifest["methods"]["SITQ-BayesRisk-24bit"]["threshold"]
    )
    condmi_threshold = float(
        manifest["methods"]["CondMIQuant-24bit"]["threshold"]
    )
    current_source_hash = _source_hash(
        [Path(__file__).with_name("run_stage_a.py"), Path(__file__).with_name("core.py")]
    )
    contexts = _design_contexts(model, manifest["experiment"], config)
    design_seed_set = {context.seed for context in contexts}
    evaluation_seed_set = {
        int(value) for value in manifest["experiment"]["evaluation_seeds"]
    }
    checks: dict[str, bool | int | str] = {
        "source_formal_performance_evidence": bool(
            source_validation.get("formal_performance_evidence", False)
        ),
        "source_gate_is_hold": source_validation.get("gate_status")
        == "HOLD_STAGE_A_UNCERTAIN",
        "source_code_hash_matches": current_source_hash == manifest["source_hash"],
        "design_evaluation_seed_disjoint": not bool(
            design_seed_set & evaluation_seed_set
        ),
        "only_design_seeds_rebuilt": design_seed_set
        == {int(value) for value in manifest["experiment"]["design_seeds"]},
        "thresholds_finite_and_increasing": bool(
            np.all(np.isfinite(thresholds)) and np.all(np.diff(thresholds) > 0.0)
        ),
        "frozen_thresholds_in_grid": bool(
            np.any(np.isclose(thresholds, sitq_threshold, atol=1e-12, rtol=0.0))
            and np.any(
                np.isclose(thresholds, condmi_threshold, atol=1e-12, rtol=0.0)
            )
        ),
        "payload_unchanged_24_bits": model.config.total_payload_bits == 24,
    }
    checks["pre_metric_all_passed"] = all(
        value for value in checks.values() if isinstance(value, bool)
    )
    if not bool(checks["pre_metric_all_passed"]):
        raise RuntimeError(f"pre-metric checks failed: {checks}")

    print("SITQ Stage A.1 threshold-selection stability certificate", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Source: {source_dir}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        f"Contexts={len(contexts)}, thresholds={thresholds.size}, "
        f"bootstrap={config.bootstrap_repetitions}, folds={config.crossfreeze_folds}",
        flush=True,
    )
    context_metrics = _compute_context_metrics(model, contexts, thresholds)
    cluster_metrics = _cluster_metrics(context_metrics)
    cluster_index, risk_matrix = _metric_matrix(
        cluster_metrics, thresholds, "posterior_risk"
    )
    entropy_index, entropy_matrix = _metric_matrix(
        cluster_metrics, thresholds, "posterior_entropy"
    )
    if not cluster_index.equals(entropy_index):
        raise RuntimeError("risk and entropy cluster indices differ")

    bootstrap_rows, bootstrap_summary = _bootstrap_selection(
        risk_matrix,
        entropy_matrix,
        thresholds,
        sitq_threshold,
        condmi_threshold,
        config.bootstrap_repetitions,
        config.bootstrap_seed,
    )
    crossfreeze_rows, crossfreeze_folds = _crossfreeze(
        cluster_index,
        risk_matrix,
        entropy_matrix,
        thresholds,
        config.crossfreeze_folds,
    )
    crossfreeze_summary = _cluster_bootstrap(
        crossfreeze_rows["risk_gain"].to_numpy(float),
        config.bootstrap_repetitions,
        config.bootstrap_seed + 1,
    )
    seed_summary = _seed_holdout_summary(crossfreeze_rows)
    selection_frequency = _selection_frequency(bootstrap_rows)

    source_hash_after = _directory_hash(source_dir)
    checks.update(
        {
            "source_result_unchanged": source_hash_before == source_hash_after,
            "context_metric_rows": int(context_metrics.shape[0]),
            "context_metric_expected_rows": int(len(contexts) * thresholds.size),
            "context_metric_complete": bool(
                context_metrics.shape[0] == len(contexts) * thresholds.size
            ),
            "cluster_count": int(risk_matrix.shape[0]),
            "crossfreeze_complete": bool(
                crossfreeze_rows.shape[0] == risk_matrix.shape[0]
                and not crossfreeze_rows.duplicated(["seed", "trial"]).any()
            ),
            "all_metrics_finite": bool(
                np.all(np.isfinite(risk_matrix))
                and np.all(np.isfinite(entropy_matrix))
            ),
        }
    )
    checks["all_passed"] = all(
        value for value in checks.values() if isinstance(value, bool)
    )
    gate = decide_gate(bool(checks["all_passed"]), crossfreeze_summary, seed_summary)

    context_metrics.to_csv(output / "stage_a1_context_metrics.csv", index=False)
    cluster_metrics.to_csv(output / "stage_a1_cluster_metrics.csv", index=False)
    bootstrap_rows.to_csv(output / "stage_a1_bootstrap_selection.csv", index=False)
    selection_frequency.to_csv(
        output / "stage_a1_selection_frequency.csv", index=False
    )
    crossfreeze_rows.to_csv(output / "stage_a1_crossfreeze_rows.csv", index=False)
    crossfreeze_folds.to_csv(output / "stage_a1_crossfreeze_folds.csv", index=False)
    seed_summary.to_csv(output / "stage_a1_seed_summary.csv", index=False)
    summary_rows = [
        {"section": "fixed_threshold", **bootstrap_summary},
        {"section": "crossfreeze", **crossfreeze_summary},
    ]
    pd.DataFrame(summary_rows).to_csv(output / "stage_a1_summary.csv", index=False)
    _plot(cluster_metrics, selection_frequency, output)

    diagnostic_manifest = {
        "diagnostic": asdict(config),
        "source_result": str(source_dir),
        "source_result_hash_before": source_hash_before,
        "source_result_hash_after": source_hash_after,
        "source_code_hash": current_source_hash,
        "source_manifest_code_hash": manifest["source_hash"],
        "information_boundary": {
            "used": "Stage A design seeds, frozen threshold grid, model and priors",
            "not_used": "Stage A evaluation rows, evaluation outcomes, true evaluation TDOA",
        },
        "frozen_thresholds": {
            "SITQ-BayesRisk-24bit": sitq_threshold,
            "CondMIQuant-24bit": condmi_threshold,
        },
        "gate": {
            "status": gate,
            "rule": (
                "STABLE only when cross-frozen mean risk gain and CI lower bound "
                "are positive and every design seed mean is positive; NO-GO when "
                "CI upper is nonpositive or mean and every seed are nonpositive; "
                "otherwise HOLD. Exact threshold identity is descriptive, not a gate."
            ),
        },
    }
    (output / "config_manifest.json").write_text(
        json.dumps(diagnostic_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    validation = {
        **checks,
        "bootstrap_summary": bootstrap_summary,
        "crossfreeze_summary": crossfreeze_summary,
        "seed_mean_risk_gain": {
            str(int(row.seed)): float(row.mean_risk_gain)
            for row in seed_summary.itertuples(index=False)
        },
        "gate_status": gate,
        "formal_performance_evidence": False,
        "diagnostic_evidence": mode == "diagnostic",
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nFixed-threshold bootstrap", flush=True)
    print(json.dumps(bootstrap_summary, indent=2), flush=True)
    print("\nCross-freeze", flush=True)
    print(json.dumps(crossfreeze_summary, indent=2), flush=True)
    print("\nSeed summary", flush=True)
    print(seed_summary.to_string(index=False), flush=True)
    print(f"\n[Gate] {gate}", flush=True)
    print(f"[Done] {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a small engineering check; never use it for research conclusions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run("smoke" if args.smoke else RUN_MODE)


if __name__ == "__main__":
    main()
