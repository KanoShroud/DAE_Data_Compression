"""Correct the frozen 36-context Stage 0 SparseAmpPhase information audit.

Run this file directly in PyCharm.  It reuses the numerically certified
SparseAmpPhase integrator without modifying the historical Stage 0 results.
The run compares two frozen proposals and two independent QMC seed families,
then pools all likelihood estimates with fixed equal weights.  This is a
corrective information audit, not an independent confirmation or a 24-bit
RAPTQ experiment.
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
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raptq_tdoa.core import (  # noqa: E402
    BayesianRAPTQModel,
    ModelConfig,
    ObservationContext,
    posterior_total_variation,
)
from raptq_tdoa.sparse_integrator import SparseRAPTQIntegrator  # noqa: E402


RUN_MODE = "development"
ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "运行结果" / "RAPTQ"
SOURCE_STAGE0 = RESULT_ROOT / "RAPTQ_STAGE0_20260805_105037_development"
SOURCE_NUMERICAL = RESULT_ROOT / "RAPTQ_STAGE0_NUMERICAL_20260805_115458_development"
SOURCE_CERTIFICATION = (
    RESULT_ROOT / "RAPTQ_SPARSE_INTEGRATOR_20260813_102100_development"
)
REPRESENTATION = "sparse_amp_phase"
PROPOSALS = ("model", "defensive_mixture")
SEED_FAMILIES = ("family_a", "family_b")
SCOPES = (
    ("all_snr", -math.inf, math.inf),
    ("low_snr", -math.inf, 0.0),
    ("transition_5_10_db", 5.0, 10.0),
    ("high_15_db", 15.0, math.inf),
)


@dataclass(frozen=True)
class CorrectionConfig:
    mode: str
    seeds: tuple[int, ...]
    trials_per_seed: int
    snr_db_values: tuple[float, ...]
    sparse_coordinate_count: int
    qmc_power: int
    qmc_repeats: int
    seed_families: tuple[str, ...]
    proposals: tuple[str, ...]
    risk_guard_samples2: float
    bootstrap_repetitions: int

    def validate(self) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.seeds or self.trials_per_seed < 1 or not self.snr_db_values:
            raise ValueError("context configuration must not be empty")
        if self.qmc_power < 4 or self.qmc_repeats < 1:
            raise ValueError("QMC configuration is invalid")
        if self.seed_families != SEED_FAMILIES or self.proposals != PROPOSALS:
            raise ValueError("proposal and seed-family protocols are frozen")
        if self.risk_guard_samples2 != 0.0625:
            raise ValueError("risk guard is frozen at 0.0625 sample^2")
        if self.bootstrap_repetitions < 20:
            raise ValueError("bootstrap repetitions must be at least 20")


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
    raw = manifest.get("model_config")
    if not isinstance(raw, dict):
        raise TypeError("source model_config must be a dictionary")
    values = dict(raw)
    values["selected_bins"] = tuple(int(value) for value in values["selected_bins"])
    values["gain_amplitudes"] = tuple(float(value) for value in values["gain_amplitudes"])
    return BayesianRAPTQModel(ModelConfig(**values))


def build_config(mode: str, manifest: dict[str, object]) -> CorrectionConfig:
    raw = manifest.get("audit_config")
    if not isinstance(raw, dict):
        raise TypeError("source audit_config must be a dictionary")
    frozen_seeds = tuple(int(value) for value in raw["seeds"])
    frozen_snr = tuple(float(value) for value in raw["snr_db_values"])
    frozen_trials = int(raw["trials_per_seed"])
    sparse_count = int(raw["sparse_coordinate_count"])
    if (
        frozen_seeds != (4301, 4302)
        or frozen_snr != (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
        or frozen_trials != 3
        or sparse_count != 2
    ):
        raise RuntimeError("historical Stage 0 protocol does not match the frozen audit")
    if mode == "smoke":
        config = CorrectionConfig(
            mode=mode,
            seeds=(4301,),
            trials_per_seed=1,
            snr_db_values=(-5.0, 10.0),
            sparse_coordinate_count=sparse_count,
            qmc_power=6,
            qmc_repeats=2,
            seed_families=SEED_FAMILIES,
            proposals=PROPOSALS,
            risk_guard_samples2=0.0625,
            bootstrap_repetitions=50,
        )
    elif mode == "development":
        config = CorrectionConfig(
            mode=mode,
            seeds=frozen_seeds,
            trials_per_seed=frozen_trials,
            snr_db_values=frozen_snr,
            sparse_coordinate_count=sparse_count,
            qmc_power=16,
            qmc_repeats=8,
            seed_families=SEED_FAMILIES,
            proposals=PROPOSALS,
            risk_guard_samples2=0.0625,
            bootstrap_repetitions=1000,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")
    config.validate()
    return config


def _qmc_seed(
    context: ObservationContext,
    proposal: str,
    family: str,
    power: int,
) -> int:
    token = (
        f"{context.seed}:{context.trial}:{context.snr_db}:{REPRESENTATION}:"
        f"{proposal}:{family}:{power}:stage0-sparse-correction-v1"
    )
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "little")


def _consensus_posterior(
    model: BayesianRAPTQModel,
    repeat_log_likelihoods: list[np.ndarray],
) -> np.ndarray:
    if not repeat_log_likelihoods:
        raise ValueError("at least one likelihood estimate is required")
    values = np.concatenate(repeat_log_likelihoods, axis=0)
    if values.ndim != 2 or values.shape[1] != model.tau_values.size:
        raise ValueError("likelihood estimates must have shape [repeats, n_tau]")
    pooled = logsumexp(values, axis=0) - math.log(values.shape[0])
    return model._posterior_from_log_likelihood(pooled)


def _max_pairwise_tv(posteriors: list[np.ndarray]) -> float:
    if len(posteriors) < 2:
        return 0.0
    return max(
        posterior_total_variation(first, second)
        for first, second in combinations(posteriors, 2)
    )


def _historical_alignment(
    model: BayesianRAPTQModel,
    contexts: list[ObservationContext],
    mode: str,
) -> dict[str, float | bool]:
    historical = pd.read_csv(SOURCE_STAGE0 / "stage0_information_rows.csv")
    historical = historical[historical["method"] == "FullComplex"].set_index(
        ["seed", "trial", "snr_db"]
    )
    max_tau_difference = 0.0
    max_risk_difference = 0.0
    max_estimate_difference = 0.0
    for context in contexts:
        key = (context.seed, context.trial, context.snr_db)
        if key not in historical.index:
            raise RuntimeError(f"context is absent from historical Stage 0: {key}")
        row = historical.loc[key]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError(f"historical FullComplex key is not unique: {key}")
        summary = model.summarize_posterior(model.posterior_full(context), context.true_tau)
        max_tau_difference = max(
            max_tau_difference,
            abs(float(row["true_tau"]) - context.true_tau),
        )
        max_risk_difference = max(
            max_risk_difference,
            abs(float(row["posterior_risk"]) - summary.posterior_risk),
        )
        max_estimate_difference = max(
            max_estimate_difference,
            abs(float(row["estimate"]) - summary.estimate),
        )
    tolerance = 1e-12 if mode == "development" else 1e-10
    return {
        "max_true_tau_difference": max_tau_difference,
        "max_full_risk_difference": max_risk_difference,
        "max_full_estimate_difference": max_estimate_difference,
        "passed": bool(
            max_tau_difference <= tolerance
            and max_risk_difference <= tolerance
            and max_estimate_difference <= tolerance
        ),
    }


def _evaluate(
    model: BayesianRAPTQModel,
    integrator: SparseRAPTQIntegrator,
    contexts: list[ObservationContext],
    config: CorrectionConfig,
    component_partial: Path,
    context_partial: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for context_index, context in enumerate(contexts, start=1):
        full_posterior = model.posterior_full(context)
        full_summary = model.summarize_posterior(full_posterior, context.true_tau)
        posteriors: list[np.ndarray] = []
        all_repeat_log_likelihoods: list[np.ndarray] = []
        current_components: list[dict[str, object]] = []
        for family in config.seed_families:
            for proposal in config.proposals:
                seed = _qmc_seed(context, proposal, family, config.qmc_power)
                result = integrator.integrate(
                    context,
                    REPRESENTATION,
                    config.qmc_power,
                    config.qmc_repeats,
                    seed,
                    proposal,
                )
                summary = model.summarize_posterior(result.posterior, context.true_tau)
                repeat_risks = np.asarray(
                    [
                        model.summarize_posterior(
                            model._posterior_from_log_likelihood(value),
                            context.true_tau,
                        ).posterior_risk
                        for value in result.repeat_log_likelihoods
                    ],
                    dtype=float,
                )
                row = {
                    "seed": context.seed,
                    "trial": context.trial,
                    "snr_db": context.snr_db,
                    "true_tau": context.true_tau,
                    "representation": REPRESENTATION,
                    "seed_family": family,
                    "proposal": proposal,
                    "qmc_seed": seed,
                    "qmc_power": config.qmc_power,
                    "qmc_points_per_repeat": 2**config.qmc_power,
                    "qmc_repeats": config.qmc_repeats,
                    "posterior_risk": summary.posterior_risk,
                    "risk_excess_vs_full": summary.posterior_risk
                    - full_summary.posterior_risk,
                    "posterior_estimate": summary.estimate,
                    "squared_error": summary.squared_error,
                    "abs_error": abs(summary.estimate - context.true_tau),
                    "posterior_entropy": summary.entropy,
                    "posterior_nll": summary.nll,
                    "credible90_contains_true": summary.credible90_contains_true,
                    "repeat_risk_sd": float(np.std(repeat_risks, ddof=1)),
                    "max_integrand_weight_fraction": (
                        result.max_integrand_weight_fraction
                    ),
                    "min_integrand_effective_sample_size": (
                        result.min_integrand_effective_sample_size
                    ),
                    "posterior_json": json.dumps(result.posterior.tolist()),
                }
                current_components.append(row)
                component_rows.append(row)
                posteriors.append(result.posterior)
                all_repeat_log_likelihoods.append(result.repeat_log_likelihoods)

        component_frame = pd.DataFrame(current_components)
        consensus = _consensus_posterior(model, all_repeat_log_likelihoods)
        consensus_summary = model.summarize_posterior(consensus, context.true_tau)
        indexed_risk = component_frame.set_index(["seed_family", "proposal"])[
            "posterior_risk"
        ]
        proposal_difference = max(
            abs(
                float(indexed_risk.loc[(family, "model")])
                - float(indexed_risk.loc[(family, "defensive_mixture")])
            )
            for family in config.seed_families
        )
        family_difference = max(
            abs(
                float(indexed_risk.loc[("family_a", proposal)])
                - float(indexed_risk.loc[("family_b", proposal)])
            )
            for proposal in config.proposals
        )
        risk_span = float(
            component_frame["posterior_risk"].max()
            - component_frame["posterior_risk"].min()
        )
        context_rows.append(
            {
                "seed": context.seed,
                "trial": context.trial,
                "snr_db": context.snr_db,
                "true_tau": context.true_tau,
                "full_estimate": full_summary.estimate,
                "full_squared_error": full_summary.squared_error,
                "full_abs_error": abs(full_summary.estimate - context.true_tau),
                "full_posterior_risk": full_summary.posterior_risk,
                "full_posterior_nll": full_summary.nll,
                "full_credible90_contains_true": (
                    full_summary.credible90_contains_true
                ),
                "consensus_estimate": consensus_summary.estimate,
                "consensus_squared_error": consensus_summary.squared_error,
                "consensus_abs_error": abs(
                    consensus_summary.estimate - context.true_tau
                ),
                "consensus_posterior_risk": consensus_summary.posterior_risk,
                "consensus_risk_excess_vs_full": (
                    consensus_summary.posterior_risk - full_summary.posterior_risk
                ),
                "consensus_posterior_entropy": consensus_summary.entropy,
                "consensus_posterior_nll": consensus_summary.nll,
                "consensus_credible90_contains_true": (
                    consensus_summary.credible90_contains_true
                ),
                "component_risk_span": risk_span,
                "max_within_family_proposal_risk_difference": proposal_difference,
                "max_within_proposal_seed_family_risk_difference": family_difference,
                "max_pairwise_posterior_tv": _max_pairwise_tv(posteriors),
                "component_estimate_span": float(
                    component_frame["posterior_estimate"].max()
                    - component_frame["posterior_estimate"].min()
                ),
                "max_repeat_risk_sd": float(
                    component_frame["repeat_risk_sd"].max()
                ),
                "max_integrand_weight_fraction": float(
                    component_frame["max_integrand_weight_fraction"].max()
                ),
                "min_integrand_effective_sample_size": float(
                    component_frame["min_integrand_effective_sample_size"].min()
                ),
                "numerical_gate_passed": bool(
                    risk_span <= config.risk_guard_samples2
                ),
                "consensus_posterior_json": json.dumps(consensus.tolist()),
            }
        )
        pd.DataFrame(component_rows).to_csv(component_partial, index=False)
        pd.DataFrame(context_rows).to_csv(context_partial, index=False)
        elapsed = time.perf_counter() - start
        seconds_per_context = elapsed / context_index
        eta = seconds_per_context * (len(contexts) - context_index)
        print(
            f"[Correction] {context_index}/{len(contexts)} "
            f"seed={context.seed} trial={context.trial} SNR={context.snr_db:g} "
            f"span={risk_span:.6g} elapsed={elapsed:.1f}s ETA={eta:.1f}s",
            flush=True,
        )
    return pd.DataFrame(component_rows), pd.DataFrame(context_rows)


def _long_method_rows(contexts: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in contexts.itertuples(index=False):
        for method, prefix in (
            ("FullComplex", "full"),
            ("SparseAmpPhaseConsensus", "consensus"),
        ):
            records.append(
                {
                    "method": method,
                    "seed": row.seed,
                    "trial": row.trial,
                    "snr_db": row.snr_db,
                    "true_tau": row.true_tau,
                    "estimate": getattr(row, f"{prefix}_estimate"),
                    "squared_error": getattr(row, f"{prefix}_squared_error"),
                    "abs_error": getattr(row, f"{prefix}_abs_error"),
                    "posterior_risk": getattr(row, f"{prefix}_posterior_risk"),
                    "posterior_nll": getattr(row, f"{prefix}_posterior_nll"),
                    "credible90_contains_true": getattr(
                        row,
                        f"{prefix}_credible90_contains_true",
                    ),
                }
            )
    return pd.DataFrame(records)


def _summary_by_snr(contexts: pd.DataFrame) -> pd.DataFrame:
    long = _long_method_rows(contexts)
    records: list[dict[str, object]] = []
    for (method, snr_db), group in long.groupby(["method", "snr_db"], sort=False):
        records.append(
            {
                "method": method,
                "snr_db": float(snr_db),
                "rmse_samples": float(np.sqrt(group["squared_error"].mean())),
                "mean_posterior_risk": float(group["posterior_risk"].mean()),
                "mean_abs_error_samples": float(group["abs_error"].mean()),
                "p95_abs_error_samples": float(group["abs_error"].quantile(0.95)),
                "gross_error_gt_1_rate": float(np.mean(group["abs_error"] > 1.0)),
                "posterior_nll": float(group["posterior_nll"].mean()),
                "credible90_coverage": float(
                    group["credible90_contains_true"].mean()
                ),
                "n_contexts": int(group.shape[0]),
            }
        )
    return pd.DataFrame(records)


def _numerical_summary(contexts: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    scopes = [("all_snr", contexts)]
    scopes.extend(
        (f"snr_{snr_db:g}_db", group)
        for snr_db, group in contexts.groupby("snr_db", sort=True)
    )
    for scope, group in scopes:
        records.append(
            {
                "scope": scope,
                "n_contexts": int(group.shape[0]),
                "max_four_component_risk_span_samples2": float(
                    group["component_risk_span"].max()
                ),
                "max_within_family_proposal_risk_difference_samples2": float(
                    group["max_within_family_proposal_risk_difference"].max()
                ),
                "max_within_proposal_seed_family_risk_difference_samples2": float(
                    group["max_within_proposal_seed_family_risk_difference"].max()
                ),
                "max_pairwise_posterior_tv": float(
                    group["max_pairwise_posterior_tv"].max()
                ),
                "max_component_estimate_span_samples": float(
                    group["component_estimate_span"].max()
                ),
                "max_repeat_risk_sd_samples2": float(
                    group["max_repeat_risk_sd"].max()
                ),
                "max_integrand_weight_fraction": float(
                    group["max_integrand_weight_fraction"].max()
                ),
                "min_integrand_effective_sample_size": float(
                    group["min_integrand_effective_sample_size"].min()
                ),
                "contexts_passing_numerical_gate": int(
                    group["numerical_gate_passed"].sum()
                ),
                "all_contexts_passed_numerical_gate": bool(
                    group["numerical_gate_passed"].all()
                ),
            }
        )
    return pd.DataFrame(records)


def _trajectory_effects(contexts: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope, lower, upper in SCOPES:
        selected = contexts[
            (contexts["snr_db"] >= lower) & (contexts["snr_db"] <= upper)
        ]
        for (seed, trial), group in selected.groupby(["seed", "trial"], sort=False):
            records.append(
                {
                    "scope": scope,
                    "seed": int(seed),
                    "trial": int(trial),
                    "mean_risk_excess_samples2": float(
                        group["consensus_risk_excess_vs_full"].mean()
                    ),
                    "mean_squared_error_difference_samples2": float(
                        (
                            group["consensus_squared_error"]
                            - group["full_squared_error"]
                        ).mean()
                    ),
                    "n_snr": int(group.shape[0]),
                }
            )
    return pd.DataFrame(records)


def _cluster_effects(
    trajectory_effects: pd.DataFrame,
    repetitions: int,
    seed: int = 73109,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for scope, group in trajectory_effects.groupby("scope", sort=False):
        values = group["mean_risk_excess_samples2"].to_numpy(float)
        bootstrap = np.asarray(
            [
                np.mean(rng.choice(values, size=values.size, replace=True))
                for _ in range(repetitions)
            ],
            dtype=float,
        )
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        records.append(
            {
                "scope": scope,
                "mean_posterior_risk_excess_samples2": float(np.mean(values)),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "fraction_trajectory_clusters_positive": float(
                    np.mean(values > 0.0)
                ),
                "n_trajectory_clusters": int(values.size),
                "trajectory_cluster_sd_samples2": sd,
                "approx_mde95_samples2": float(
                    1.96 * sd / math.sqrt(values.size)
                ),
            }
        )
    return pd.DataFrame(records)


def _plot(contexts: pd.DataFrame, output: Path, guard: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    for _, group in contexts.groupby(["seed", "trial"]):
        ordered = group.sort_values("snr_db")
        axes[0].plot(
            ordered["snr_db"],
            ordered["consensus_risk_excess_vs_full"],
            color="#8da0ae",
            linewidth=0.8,
            alpha=0.55,
        )
    mean_effect = contexts.groupby("snr_db", as_index=False)[
        "consensus_risk_excess_vs_full"
    ].mean()
    axes[0].plot(
        mean_effect["snr_db"],
        mean_effect["consensus_risk_excess_vs_full"],
        color="#b2182b",
        marker="o",
        linewidth=1.4,
        label="Trajectory mean",
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Posterior-risk excess (samples²)")
    axes[0].set_title("Corrected continuous-information effect")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="best")

    numerical = contexts.groupby("snr_db", as_index=False)[
        "component_risk_span"
    ].max()
    axes[1].plot(
        numerical["snr_db"],
        numerical["component_risk_span"],
        color="#2166ac",
        marker="o",
        linewidth=1.4,
        label="Maximum across contexts",
    )
    axes[1].axhline(
        guard,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="Numerical guard",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Four-component risk span (samples²)")
    axes[1].set_title("Proposal and seed-family agreement")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("RAPTQ SparseAmpPhase 36-context corrective audit")
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _validate_source_results() -> tuple[dict[str, object], dict[str, str]]:
    for source in (SOURCE_STAGE0, SOURCE_NUMERICAL, SOURCE_CERTIFICATION):
        if not source.is_dir():
            raise FileNotFoundError(f"frozen source result not found: {source}")
    stage0_manifest = _json(SOURCE_STAGE0 / "config_manifest.json")
    numerical_validation = _json(SOURCE_NUMERICAL / "validation.json")
    certification_manifest = _json(SOURCE_CERTIFICATION / "config_manifest.json")
    certification_validation = _json(SOURCE_CERTIFICATION / "validation.json")
    if stage0_manifest.get("route") != "RAPTQ-TDOA":
        raise RuntimeError("unexpected historical route")
    if numerical_validation.get("gate_status") != "HOLD_NUMERICAL_CONVERGENCE":
        raise RuntimeError("historical numerical certificate is not the frozen HOLD")
    if (
        certification_validation.get("overall_gate")
        != "CERTIFIED_PARTIAL_SPARSE_REPRESENTATIONS"
    ):
        raise RuntimeError("sparse-integrator certification gate is not frozen")
    development = certification_validation.get("development_status")
    review = certification_validation.get("review_status")
    if not isinstance(development, dict) or not isinstance(review, dict):
        raise TypeError("certification representation states are missing")
    if (
        development.get("sparse_phase") != "HOLD_COMPUTABILITY_DEVELOPMENT"
        or review.get("sparse_amp_phase") != "CERTIFIED"
    ):
        raise RuntimeError("certification representation states do not match")
    current_certification_hash = _source_hash(
        (
            ROOT / "raptq_tdoa" / "diagnose_sparse_integrator.py",
            ROOT / "raptq_tdoa" / "sparse_integrator.py",
            ROOT / "raptq_tdoa" / "core.py",
        )
    )
    if current_certification_hash != certification_manifest.get("source_hash"):
        raise RuntimeError("certified sparse-integrator source hash has changed")
    hashes = {
        "stage0": _tree_hash(SOURCE_STAGE0),
        "numerical": _tree_hash(SOURCE_NUMERICAL),
        "certification": _tree_hash(SOURCE_CERTIFICATION),
    }
    return stage0_manifest, hashes


def run(mode: str) -> Path:
    stage0_manifest, source_hashes_before = _validate_source_results()
    config = build_config(mode, stage0_manifest)
    model = _model_from_manifest(stage0_manifest)
    integrator = SparseRAPTQIntegrator(model, config.sparse_coordinate_count)
    contexts = model.build_contexts(
        config.seeds,
        config.trials_per_seed,
        config.snr_db_values,
    )
    expected_contexts = (
        len(config.seeds) * config.trials_per_seed * len(config.snr_db_values)
    )
    if len(contexts) != expected_contexts:
        raise RuntimeError("context count does not match the frozen protocol")
    alignment = _historical_alignment(model, contexts, mode)
    if not alignment["passed"]:
        raise RuntimeError(f"historical Stage 0 alignment failed: {alignment}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULT_ROOT / f"RAPTQ_STAGE0_SPARSE_CORRECTION_{timestamp}_{mode}"
    output.mkdir(parents=True, exist_ok=False)
    component_partial = output / "component_partial_rows.csv"
    context_partial = output / "context_partial_rows.csv"
    manifest = {
        "route": "RAPTQ-TDOA",
        "stage": "Stage 0-I SparseAmpPhase 36-context corrective audit",
        "mode": mode,
        "config": asdict(config),
        "source_stage0": str(SOURCE_STAGE0),
        "source_numerical": str(SOURCE_NUMERICAL),
        "source_certification": str(SOURCE_CERTIFICATION),
        "source_hashes_before": source_hashes_before,
        "historical_alignment": alignment,
        "consensus_rule": (
            "Fixed equal likelihood pooling over both proposals, both seed "
            "families, and every scramble repeat; no component is selected "
            "after observing its result."
        ),
        "protocol_boundary": (
            "The 36 contexts are the frozen historical development set with "
            "six trajectory clusters. This run corrects numerical estimates "
            "but is not an independent information-sufficiency confirmation."
        ),
        "decision_boundary": (
            "Only four-component numerical agreement is a hard gate. Risk "
            "effects are descriptive and cannot approve CondMI-Matched24, new "
            "trajectories, or Stage 0-R without separate preregistration."
        ),
        "source_hash": _source_hash(
            (
                Path(__file__).resolve(),
                ROOT / "raptq_tdoa" / "sparse_integrator.py",
                ROOT / "raptq_tdoa" / "core.py",
            )
        ),
    }
    _write_json(output / "config_manifest.json", manifest)

    print("RAPTQ SparseAmpPhase 36-context corrective audit", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        f"Contexts={len(contexts)}, QMC=2^{config.qmc_power}x"
        f"{config.qmc_repeats}, components="
        f"{len(config.seed_families) * len(config.proposals)}",
        flush=True,
    )
    start = time.perf_counter()
    components, context_rows = _evaluate(
        model,
        integrator,
        contexts,
        config,
        component_partial,
        context_partial,
    )
    summary = _summary_by_snr(context_rows)
    numerical_summary = _numerical_summary(context_rows)
    trajectory = _trajectory_effects(context_rows)
    effects = _cluster_effects(
        trajectory,
        config.bootstrap_repetitions,
    )

    components.to_csv(output / "sparse_correction_component_rows.csv", index=False)
    context_rows.to_csv(output / "sparse_correction_context_rows.csv", index=False)
    summary.to_csv(output / "sparse_correction_summary_by_snr.csv", index=False)
    numerical_summary.to_csv(
        output / "sparse_correction_numerical_summary.csv",
        index=False,
    )
    trajectory.to_csv(output / "sparse_correction_trajectory_effects.csv", index=False)
    effects.to_csv(output / "sparse_correction_cluster_effects.csv", index=False)
    _plot(
        context_rows,
        output / "Fig1_RAPTQ_SparseAmpPhase_Corrective_Audit.svg",
        config.risk_guard_samples2,
    )

    source_hashes_after = {
        "stage0": _tree_hash(SOURCE_STAGE0),
        "numerical": _tree_hash(SOURCE_NUMERICAL),
        "certification": _tree_hash(SOURCE_CERTIFICATION),
    }
    source_results_unchanged = source_hashes_after == source_hashes_before
    finite_columns = [
        "full_posterior_risk",
        "consensus_posterior_risk",
        "consensus_risk_excess_vs_full",
        "component_risk_span",
        "max_pairwise_posterior_tv",
    ]
    rows_complete = bool(
        context_rows.shape[0] == expected_contexts
        and components.shape[0]
        == expected_contexts * len(config.seed_families) * len(config.proposals)
        and not context_rows.duplicated(["seed", "trial", "snr_db"]).any()
        and np.isfinite(context_rows[finite_columns].to_numpy(float)).all()
    )
    max_risk_span = float(context_rows["component_risk_span"].max())
    numerical_gate = (
        "PASS_36_CONTEXT"
        if max_risk_span <= config.risk_guard_samples2
        else "HOLD_COMPUTABILITY_36_CONTEXT"
    )
    if not source_results_unchanged or not rows_complete:
        overall_gate = "INVALID_CORRECTIVE_AUDIT"
    elif mode == "smoke":
        overall_gate = "SMOKE_ONLY"
    elif numerical_gate != "PASS_36_CONTEXT":
        overall_gate = numerical_gate
    else:
        overall_gate = "CORRECTIVE_INFORMATION_AUDIT_COMPLETE_PENDING_REVIEW"
    validation = {
        "historical_gate_preserved": "HOLD_NUMERICAL_CONVERGENCE",
        "source_results_unchanged": source_results_unchanged,
        "historical_alignment": alignment,
        "rows_complete": rows_complete,
        "context_count": int(context_rows.shape[0]),
        "component_row_count": int(components.shape[0]),
        "representation": REPRESENTATION,
        "sparse_phase_recomputed": False,
        "risk_guard_name": "delta_guard",
        "risk_guard_samples2": config.risk_guard_samples2,
        "risk_guard_is_strict_qmc_error_bound": False,
        "max_four_component_risk_span_samples2": max_risk_span,
        "max_pairwise_posterior_tv": float(
            context_rows["max_pairwise_posterior_tv"].max()
        ),
        "numerical_gate": numerical_gate,
        "independent_trajectory_confirmation_performed": False,
        "information_effect_status": "DESCRIPTIVE_ONLY",
        "condmi_matched24_evaluated": False,
        "stage0_r_approved": False,
        "overall_gate": overall_gate,
        "elapsed_seconds": time.perf_counter() - start,
    }
    _write_json(output / "validation.json", validation)
    component_partial.unlink(missing_ok=True)
    context_partial.unlink(missing_ok=True)
    print(f"[Numerical] max four-component span={max_risk_span:.6g}", flush=True)
    print(f"[Gate] {overall_gate}", flush=True)
    print(f"[Done] elapsed={validation['elapsed_seconds']:.1f}s", flush=True)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAPTQ SparseAmpPhase 36-context corrective audit"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a two-context engineering smoke instead of the frozen audit",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode = "smoke" if args.smoke else RUN_MODE
    run(mode)


if __name__ == "__main__":
    main()
