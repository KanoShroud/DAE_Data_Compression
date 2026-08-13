"""Numerical convergence certificate for RAPTQ Stage 0-I.

Run this file directly in PyCharm.  The default ``RUN_MODE`` uses the frozen
Stage 0 development manifest and regenerates only its pre-registered diagnostic
contexts.  It does not modify or overwrite the source result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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


RUN_MODE = "development"
SOURCE_RESULT = (
    Path(__file__).resolve().parents[1]
    / "运行结果"
    / "RAPTQ"
    / "RAPTQ_STAGE0_20260805_105037_development"
)


@dataclass(frozen=True)
class NumericalConfig:
    mode: str
    trajectory_selectors: tuple[tuple[int, int], ...]
    snr_db_values: tuple[float, ...]
    candidate_powers: tuple[int, ...]
    candidate_repeats: int
    reference_power: int
    reference_repeats: int
    sparse_coordinate_count: int = 2

    def validate(self) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.trajectory_selectors or not self.snr_db_values:
            raise ValueError("diagnostic contexts must not be empty")
        if tuple(sorted(self.candidate_powers)) != self.candidate_powers:
            raise ValueError("candidate powers must be strictly ordered")
        if len(set(self.candidate_powers)) != len(self.candidate_powers):
            raise ValueError("candidate powers must be unique")
        if self.reference_power <= max(self.candidate_powers):
            raise ValueError("reference power must exceed every candidate power")
        if self.candidate_repeats < 1 or self.reference_repeats < 1:
            raise ValueError("QMC repeats must be positive")


VARIANTS = (
    ("ProjectorOnly", "projector_only", "model_radial"),
    ("RelativePhase-Model", "relative_phase", "model_radial"),
    ("RelativePhase-Defensive", "relative_phase", "defensive_observed"),
    ("RelativePower-Analytic", "relative_power", "model_radial"),
    ("SparsePhase", "sparse_phase", "model_radial"),
    ("SparseAmpPhase", "sparse_amp_phase", "model_radial"),
)
REFERENCE_PROPOSAL = {
    "projector_only": "model_radial",
    "relative_phase": "defensive_observed",
    "relative_power": "model_radial",
    "sparse_phase": "model_radial",
    "sparse_amp_phase": "model_radial",
}


def build_config(mode: str, source_manifest: dict[str, object]) -> NumericalConfig:
    audit = source_manifest["audit_config"]
    if not isinstance(audit, dict):
        raise TypeError("source audit_config must be a dictionary")
    source_seeds = tuple(int(value) for value in audit["seeds"])
    source_snr = tuple(float(value) for value in audit["snr_db_values"])
    if mode == "smoke":
        config = NumericalConfig(
            mode=mode,
            trajectory_selectors=((source_seeds[0], 0),),
            snr_db_values=(5.0, 15.0),
            candidate_powers=(4, 5, 6),
            candidate_repeats=2,
            reference_power=8,
            reference_repeats=2,
        )
    elif mode == "development":
        config = NumericalConfig(
            mode=mode,
            trajectory_selectors=tuple((seed, 0) for seed in source_seeds),
            snr_db_values=source_snr,
            candidate_powers=(8, 10, 12, 14),
            candidate_repeats=4,
            reference_power=15,
            reference_repeats=8,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    config.validate()
    return config


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _source_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_from_manifest(manifest: dict[str, object]) -> BayesianRAPTQModel:
    raw = manifest["model_config"]
    if not isinstance(raw, dict):
        raise TypeError("model_config must be a dictionary")
    values = dict(raw)
    values["selected_bins"] = tuple(int(value) for value in values["selected_bins"])
    values["gain_amplitudes"] = tuple(
        float(value) for value in values["gain_amplitudes"]
    )
    return BayesianRAPTQModel(ModelConfig(**values))


def _contexts(
    model: BayesianRAPTQModel,
    config: NumericalConfig,
) -> list[ObservationContext]:
    contexts: list[ObservationContext] = []
    for seed, trial_index in config.trajectory_selectors:
        trials = model.generate_trials(seed, trial_index + 1)
        trial = trials[trial_index]
        contexts.extend(
            model.build_context(trial, snr_db) for snr_db in config.snr_db_values
        )
    return contexts


def _qmc_config(
    power: int,
    repeats: int,
    proposal: str,
) -> QMCConfig:
    return QMCConfig(
        power=power,
        repeats=repeats,
        reference_power=power + 1,
        reference_context_limit=1,
        proposal=proposal,
    )


def _qmc_seed(
    context: ObservationContext,
    representation: str,
    proposal: str,
    power: int,
) -> int:
    token = f"{context.seed}:{context.trial}:{context.snr_db}:{representation}:{proposal}:{power}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "little")


def _reference_posteriors(
    model: BayesianRAPTQModel,
    context: ObservationContext,
    config: NumericalConfig,
) -> dict[str, np.ndarray]:
    reference: dict[str, np.ndarray] = {}
    for representation, proposal in REFERENCE_PROPOSAL.items():
        reference[representation] = model.posterior_representation(
            context,
            representation,
            _qmc_config(config.reference_power, config.reference_repeats, proposal),
            config.sparse_coordinate_count,
            _qmc_seed(
                context,
                representation,
                proposal,
                config.reference_power,
            ),
        )
    return reference


def _evaluate(
    model: BayesianRAPTQModel,
    contexts: list[ObservationContext],
    config: NumericalConfig,
    partial_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for context_index, context in enumerate(contexts):
        full_posterior = model.posterior_full(context)
        full_risk = model.summarize_posterior(full_posterior, context.true_tau).posterior_risk
        references = _reference_posteriors(model, context, config)
        by_power: dict[int, dict[str, np.ndarray]] = {}
        for power in config.candidate_powers:
            by_power[power] = {}
            for method, representation, proposal in VARIANTS:
                posterior = model.posterior_representation(
                    context,
                    representation,
                    _qmc_config(power, config.candidate_repeats, proposal),
                    config.sparse_coordinate_count,
                    _qmc_seed(context, representation, proposal, power),
                )
                by_power[power][method] = posterior
                summary = model.summarize_posterior(posterior, context.true_tau)
                reference = references[representation]
                reference_summary = model.summarize_posterior(
                    reference,
                    context.true_tau,
                )
                rows.append(
                    {
                        "method": method,
                        "representation": representation,
                        "proposal": proposal,
                        "seed": context.seed,
                        "trial": context.trial,
                        "snr_db": context.snr_db,
                        "qmc_power": power,
                        "qmc_points_per_repeat": 2**power,
                        "estimate": summary.estimate,
                        "posterior_risk": summary.posterior_risk,
                        "full_posterior_risk": full_risk,
                        "risk_excess_vs_full": summary.posterior_risk - full_risk,
                        "reference_posterior_risk": reference_summary.posterior_risk,
                        "tv_to_reference": posterior_total_variation(
                            posterior,
                            reference,
                        ),
                        "risk_difference_to_reference": abs(
                            summary.posterior_risk - reference_summary.posterior_risk
                        ),
                    }
                )
            model_posterior = by_power[power]["RelativePhase-Model"]
            defensive_posterior = by_power[power]["RelativePhase-Defensive"]
            cross_rows.append(
                {
                    "seed": context.seed,
                    "trial": context.trial,
                    "snr_db": context.snr_db,
                    "qmc_power": power,
                    "cross_proposal_tv": posterior_total_variation(
                        model_posterior,
                        defensive_posterior,
                    ),
                    "cross_proposal_risk_difference": abs(
                        model.summarize_posterior(
                            model_posterior,
                            context.true_tau,
                        ).posterior_risk
                        - model.summarize_posterior(
                            defensive_posterior,
                            context.true_tau,
                        ).posterior_risk
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        done = context_index + 1
        elapsed = time.perf_counter() - start
        print(
            f"[Context] {done}/{len(contexts)} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(cross_rows)


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["method", "qmc_power"], as_index=False)
        .agg(
            n_contexts=("tv_to_reference", "size"),
            tv_mean=("tv_to_reference", "mean"),
            tv_median=("tv_to_reference", "median"),
            tv_max=("tv_to_reference", "max"),
            risk_difference_mean=("risk_difference_to_reference", "mean"),
            risk_difference_median=("risk_difference_to_reference", "median"),
            risk_difference_max=("risk_difference_to_reference", "max"),
            mean_risk_excess_vs_full=("risk_excess_vs_full", "mean"),
        )
        .sort_values(["method", "qmc_power"])
    )


def _ordering(rows: pd.DataFrame, highest_power: int) -> pd.DataFrame:
    canonical = rows[
        (rows["qmc_power"] == highest_power)
        & rows["method"].isin(
            [
                "ProjectorOnly",
                "RelativePhase-Defensive",
                "RelativePower-Analytic",
                "SparsePhase",
                "SparseAmpPhase",
            ]
        )
    ].copy()
    records: list[dict[str, object]] = []
    for snr_db, group in canonical.groupby("snr_db"):
        candidate = group.groupby("method")["posterior_risk"].mean()
        reference = group.groupby("method")["reference_posterior_risk"].mean()
        candidate_order = tuple(candidate.sort_values().index)
        reference_order = tuple(reference.sort_values().index)
        records.append(
            {
                "snr_db": float(snr_db),
                "candidate_order": " < ".join(candidate_order),
                "reference_order": " < ".join(reference_order),
                "ordering_identical": candidate_order == reference_order,
            }
        )
    return pd.DataFrame(records)


def _validation(
    model: BayesianRAPTQModel,
    config: NumericalConfig,
    rows: pd.DataFrame,
    cross: pd.DataFrame,
    ordering: pd.DataFrame,
) -> dict[str, object]:
    highest = max(config.candidate_powers)
    previous = config.candidate_powers[-2]
    final_rows = rows[rows["qmc_power"] == highest]
    previous_rows = rows[rows["qmc_power"] == previous]
    final_by_method = final_rows.groupby("method")[
        "risk_difference_to_reference"
    ].max()
    previous_by_method = previous_rows.groupby("method")[
        "risk_difference_to_reference"
    ].max()
    analytic = final_rows[final_rows["method"] == "RelativePower-Analytic"]
    prior_mean = float(np.sum(model.tau_prior * model.tau_values))
    prior_risk = float(
        np.sum(model.tau_prior * (model.tau_values - prior_mean) ** 2)
    )
    analytic_prior_error = float(
        np.max(np.abs(analytic["posterior_risk"].to_numpy(float) - prior_risk))
    )
    target_methods = (
        "ProjectorOnly",
        "RelativePhase-Defensive",
        "SparsePhase",
        "SparseAmpPhase",
    )
    cauchy_direction = {
        method: bool(final_by_method[method] <= previous_by_method[method])
        for method in target_methods
    }
    risk_resolution = model.config.tau_step**2
    resolution_sensitivity = {
        "half_grid_squared": (model.config.tau_step / 2.0) ** 2,
        "one_grid_squared": risk_resolution,
        "two_grid_squared": (2.0 * model.config.tau_step) ** 2,
    }
    final_risk_within_resolution = {
        method: bool(final_by_method[method] <= risk_resolution)
        for method in target_methods
    }
    final_cross = cross[cross["qmc_power"] == highest]
    exact_controls = bool(
        analytic_prior_error < 1e-12
        and np.isfinite(rows["posterior_risk"].to_numpy(float)).all()
        and not rows.duplicated(
            ["method", "seed", "trial", "snr_db", "qmc_power"]
        ).any()
    )
    ordering_stable = bool(ordering["ordering_identical"].all())
    cauchy_stable = all(cauchy_direction.values())
    resolution_stable = all(final_risk_within_resolution.values())
    if config.mode == "smoke":
        gate = "SMOKE_ONLY"
    elif not exact_controls:
        gate = "INVALID_NUMERICAL_CERTIFICATE"
    elif not ordering_stable:
        gate = "HOLD_NUMERICAL_ORDERING"
    elif not cauchy_stable or not resolution_stable:
        gate = "HOLD_NUMERICAL_CONVERGENCE"
    else:
        gate = "NUMERICAL_ORDERING_CERTIFIED_PENDING_STAGE0I_EXPANSION"
    return {
        "source_result": str(SOURCE_RESULT),
        "context_count": int(
            rows[["seed", "trial", "snr_db"]].drop_duplicates().shape[0]
        ),
        "row_keys_unique": bool(
            not rows.duplicated(
                ["method", "seed", "trial", "snr_db", "qmc_power"]
            ).any()
        ),
        "all_metrics_finite": bool(
            np.isfinite(
                rows[
                    [
                        "posterior_risk",
                        "tv_to_reference",
                        "risk_difference_to_reference",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "relative_power_analytic_prior_risk": prior_risk,
        "relative_power_analytic_prior_error": analytic_prior_error,
        "relative_power_exact_control": analytic_prior_error < 1e-12,
        "highest_candidate_power": highest,
        "reference_power": config.reference_power,
        "risk_resolution_sensitivity": resolution_sensitivity,
        "final_max_risk_difference_by_method": {
            key: float(value) for key, value in final_by_method.items()
        },
        "cauchy_direction_by_method": cauchy_direction,
        "final_risk_within_one_grid_squared": final_risk_within_resolution,
        "relative_phase_cross_proposal_max_tv": float(
            final_cross["cross_proposal_tv"].max()
        ),
        "relative_phase_cross_proposal_max_risk_difference": float(
            final_cross["cross_proposal_risk_difference"].max()
        ),
        "ordering_identical_all_snr": ordering_stable,
        "gate_status": gate,
    }


def _plot(summary: pd.DataFrame, cross: pd.DataFrame, output: Path, tau_step: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for method, group in summary.groupby("method"):
        points = 2 ** group["qmc_power"].to_numpy(int)
        axes[0].plot(points, group["tv_max"], marker="o", linewidth=1.2, label=method)
        axes[1].plot(
            points,
            group["risk_difference_max"],
            marker="o",
            linewidth=1.2,
            label=method,
        )
    cross_summary = cross.groupby("qmc_power", as_index=False).agg(
        max_tv=("cross_proposal_tv", "max"),
        max_risk_difference=("cross_proposal_risk_difference", "max"),
    )
    cross_points = 2 ** cross_summary["qmc_power"].to_numpy(int)
    axes[2].plot(
        cross_points,
        cross_summary["max_tv"],
        marker="o",
        linewidth=1.2,
        label="Posterior TV",
    )
    axes[2].plot(
        cross_points,
        cross_summary["max_risk_difference"],
        marker="s",
        linewidth=1.2,
        label="Risk difference",
    )
    axes[1].axhline(tau_step**2, color="black", linestyle="--", linewidth=1.0)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.set_xlabel("QMC points per scramble")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Max posterior TV to independent reference")
    axes[1].set_ylabel("Max posterior-risk difference (samples²)")
    axes[2].set_ylabel("RelativePhase proposal disagreement")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.65, -0.20), ncol=3, frameon=False)
    axes[2].legend(frameon=False)
    fig.suptitle("RAPTQ Stage 0-I numerical convergence certificate")
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def run(mode: str) -> Path:
    if not SOURCE_RESULT.is_dir():
        raise FileNotFoundError(f"source result not found: {SOURCE_RESULT}")
    source_hash_before = _tree_hash(SOURCE_RESULT)
    source_manifest = _json(SOURCE_RESULT / "config_manifest.json")
    source_validation = _json(SOURCE_RESULT / "validation.json")
    if source_manifest.get("route") != "RAPTQ-TDOA":
        raise RuntimeError("source route is not RAPTQ-TDOA")
    if source_manifest.get("mode") != "development":
        raise RuntimeError("source result is not a development run")
    if source_validation.get("gate_status") != (
        "STAGE0_T_PASS_STAGE0_I_PENDING_NUMERICAL_REVIEW"
    ):
        raise RuntimeError("source result has an unexpected gate status")

    model = _model_from_manifest(source_manifest)
    config = build_config(mode, source_manifest)
    contexts = _contexts(model, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        Path(__file__).resolve().parents[1]
        / "运行结果"
        / "RAPTQ"
        / f"RAPTQ_STAGE0_NUMERICAL_{timestamp}_{mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    partial_path = output / "numerical_partial_rows.csv"
    manifest = {
        "route": "RAPTQ-TDOA",
        "stage": "Stage 0-I numerical convergence certificate",
        "mode": mode,
        "source_result": str(SOURCE_RESULT),
        "source_result_hash_before": source_hash_before,
        "source_manifest": source_manifest,
        "numerical_config": asdict(config),
        "proposal_boundary": (
            "defensive_observed changes only the exact importance proposal; "
            "agreement with model_radial remains mandatory"
        ),
        "gate_boundary": (
            "ordering and one-grid-squared numerical resolution only; no "
            "continuous-information equivalence or 24-bit performance claim"
        ),
        "source_hash": _source_hash(
            (
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("core.py"),
            )
        ),
    }
    _write_json(output / "config_manifest.json", manifest)

    print("RAPTQ Stage 0-I numerical convergence certificate", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Source: {SOURCE_RESULT}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        f"Contexts={len(contexts)}, powers={config.candidate_powers}, "
        f"reference=2^{config.reference_power}x{config.reference_repeats}",
        flush=True,
    )
    start = time.perf_counter()
    rows, cross = _evaluate(model, contexts, config, partial_path)
    summary = _summary(rows)
    ordering = _ordering(rows, max(config.candidate_powers))
    validation = _validation(model, config, rows, cross, ordering)
    source_hash_after = _tree_hash(SOURCE_RESULT)
    validation["source_result_hash_after"] = source_hash_after
    validation["source_result_unchanged"] = source_hash_before == source_hash_after
    validation["elapsed_seconds"] = time.perf_counter() - start
    if not validation["source_result_unchanged"]:
        validation["gate_status"] = "INVALID_SOURCE_RESULT_MUTATION"

    rows.to_csv(output / "numerical_convergence_rows.csv", index=False)
    summary.to_csv(output / "numerical_convergence_summary.csv", index=False)
    cross.to_csv(output / "relative_phase_proposal_agreement.csv", index=False)
    ordering.to_csv(output / "representation_ordering_stability.csv", index=False)
    _write_json(output / "validation.json", validation)
    _plot(
        summary,
        cross,
        output / "Fig1_RAPTQ_Numerical_Convergence.svg",
        model.config.tau_step,
    )
    if partial_path.exists():
        partial_path.unlink()
    print(f"[Gate] {validation['gate_status']}", flush=True)
    print(f"[Done] elapsed={validation['elapsed_seconds']:.1f}s", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run("smoke" if arguments.smoke else RUN_MODE)


if __name__ == "__main__":
    main()
