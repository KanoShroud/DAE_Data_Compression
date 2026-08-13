"""PyCharm entry point for the independent RAPTQ Stage 0 audit.

Direct execution uses ``RUN_MODE`` below.  Codex passes ``--smoke`` only for
engineering verification; smoke results are not research evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raptq_tdoa.core import (  # noqa: E402
    AuditConfig,
    BayesianRAPTQModel,
    ModelConfig,
    ObservationContext,
    QMCConfig,
    finite_window_rotation_certificate,
    posterior_total_variation,
)


RUN_MODE = "development"

REPRESENTATIONS = (
    "projector_only",
    "relative_phase",
    "relative_power",
    "sparse_phase",
    "sparse_amp_phase",
)
METHOD_LABELS = {
    "full_complex": "FullComplex",
    "radius_plus_projector": "Radius+Projector",
    "projector_only": "ProjectorOnly",
    "relative_phase": "RelativePhase",
    "relative_power": "RelativePower",
    "sparse_phase": "SparsePhase",
    "sparse_amp_phase": "SparseAmpPhase",
}
REPRESENTATION_SEED_OFFSET = {
    "projector_only": 101,
    "relative_phase": 211,
    "relative_power": 307,
    "sparse_phase": 401,
    "sparse_amp_phase": 503,
}


def build_run_config(mode: str) -> tuple[AuditConfig, QMCConfig]:
    if mode == "smoke":
        audit = AuditConfig(
            mode="smoke",
            snr_db_values=(-5.0, 5.0),
            seeds=(4301,),
            trials_per_seed=1,
            sparse_coordinate_count=2,
            bootstrap_repetitions=50,
        )
        integration = QMCConfig(
            power=6,
            repeats=2,
            reference_power=8,
            reference_context_limit=2,
        )
    elif mode == "development":
        audit = AuditConfig(
            mode="development",
            snr_db_values=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
            seeds=(4301, 4302),
            trials_per_seed=3,
            sparse_coordinate_count=2,
            bootstrap_repetitions=1000,
        )
        integration = QMCConfig(
            power=10,
            repeats=4,
            reference_power=12,
            reference_context_limit=12,
        )
    else:
        raise ValueError(f"unknown run mode: {mode}")
    audit.validate(len(ModelConfig().selected_bins))
    integration.validate()
    return audit, integration


def _source_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _posterior_row(
    model: BayesianRAPTQModel,
    context: ObservationContext,
    method: str,
    posterior: np.ndarray,
    qmc_power: float = math.nan,
    reference_tv: float = math.nan,
    reference_risk_difference: float = math.nan,
    sparse_coordinate_count: int = 2,
) -> dict[str, float | int | str | bool]:
    summary = model.summarize_posterior(posterior, context.true_tau)
    energy = np.abs(context.y_b_normalized) ** 2
    return {
        "method": method,
        "seed": context.seed,
        "trial": context.trial,
        "snr_db": context.snr_db,
        "true_tau": context.true_tau,
        "estimate": summary.estimate,
        "squared_error": summary.squared_error,
        "abs_error": abs(summary.estimate - context.true_tau),
        "posterior_risk": summary.posterior_risk,
        "posterior_entropy": summary.entropy,
        "posterior_nll": summary.nll,
        "credible90_contains_true": summary.credible90_contains_true,
        "qmc_power": qmc_power,
        "qmc_reference_tv": reference_tv,
        "qmc_reference_risk_difference": reference_risk_difference,
        "minimum_bin_energy": float(np.min(energy)),
        "anchor_bin_energy": float(energy[0]),
        "sparse_endpoint_min_energy": float(
            np.min(energy[: sparse_coordinate_count + 1])
        ),
    }


def _qmc_seed(context: ObservationContext, representation: str) -> int:
    snr_code = int(round((context.snr_db + 50.0) * 10.0))
    return int(
        context.seed * 1000003
        + context.trial * 1009
        + snr_code * 31
        + REPRESENTATION_SEED_OFFSET[representation]
    ) % (2**32 - 1)


def _evaluate(
    model: BayesianRAPTQModel,
    contexts: Sequence[ObservationContext],
    audit: AuditConfig,
    integration: QMCConfig,
    partial_path: Path,
) -> tuple[pd.DataFrame, dict[str, float | bool | int]]:
    rows: list[dict[str, float | int | str | bool]] = []
    rotation_tvs: list[float] = []
    projector_tvs: list[float] = []
    projector_errors: list[float] = []
    start = time.perf_counter()
    for context_index, context in enumerate(contexts):
        full = model.posterior_full(context)
        rows.append(
            _posterior_row(
                model,
                context,
                "FullComplex",
                full,
                sparse_coordinate_count=audit.sparse_coordinate_count,
            )
        )

        radius, projector = model.projector(context.y_b_normalized)
        representative = model.representative_from_projector(radius, projector)
        reconstructed = model.posterior_full(context, representative)
        projector_tvs.append(posterior_total_variation(full, reconstructed))
        reconstructed_radius, reconstructed_projector = model.projector(representative)
        projector_errors.append(
            max(
                abs(radius - reconstructed_radius),
                float(np.max(np.abs(projector - reconstructed_projector))),
            )
        )
        rows.append(
            _posterior_row(
                model,
                context,
                "Radius+Projector",
                reconstructed,
                sparse_coordinate_count=audit.sparse_coordinate_count,
            )
        )

        rng = np.random.default_rng(91001 + context_index)
        rotated = context.y_b_normalized * np.exp(1j * rng.uniform(-np.pi, np.pi))
        rotation_tvs.append(
            posterior_total_variation(full, model.posterior_full(context, rotated))
        )

        for representation in REPRESENTATIONS:
            seed = _qmc_seed(context, representation)
            posterior = model.posterior_representation(
                context,
                representation,
                integration,
                audit.sparse_coordinate_count,
                seed,
            )
            reference_tv = math.nan
            reference_risk_difference = math.nan
            if context_index < integration.reference_context_limit:
                reference_config = QMCConfig(
                    power=integration.reference_power,
                    repeats=integration.repeats,
                    reference_power=integration.reference_power + 1,
                    reference_context_limit=integration.reference_context_limit,
                    proposal=integration.proposal,
                )
                reference = model.posterior_representation(
                    context,
                    representation,
                    reference_config,
                    audit.sparse_coordinate_count,
                    seed + 700001,
                )
                reference_tv = posterior_total_variation(posterior, reference)
                reference_risk_difference = abs(
                    model.summarize_posterior(posterior, context.true_tau).posterior_risk
                    - model.summarize_posterior(
                        reference,
                        context.true_tau,
                    ).posterior_risk
                )
            rows.append(
                _posterior_row(
                    model,
                    context,
                    METHOD_LABELS[representation],
                    posterior,
                    qmc_power=float(integration.power),
                    reference_tv=reference_tv,
                    reference_risk_difference=reference_risk_difference,
                    sparse_coordinate_count=audit.sparse_coordinate_count,
                )
            )

        done = context_index + 1
        elapsed = time.perf_counter() - start
        print(
            f"[Audit] {done}/{len(contexts)} contexts elapsed={elapsed:.1f}s",
            flush=True,
        )
        if done % 2 == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    checks: dict[str, float | bool | int] = {
        "context_count": len(contexts),
        "global_rotation_max_tv": max(rotation_tvs, default=math.inf),
        "global_rotation_invariant": max(rotation_tvs, default=math.inf) < 1e-10,
        "radius_projector_max_tv": max(projector_tvs, default=math.inf),
        "radius_projector_posterior_equivalent": (
            max(projector_tvs, default=math.inf) < 1e-10
        ),
        "projector_roundtrip_max_error": max(projector_errors, default=math.inf),
        "projector_roundtrip_valid": max(projector_errors, default=math.inf) < 1e-10,
    }
    return pd.DataFrame(rows), checks


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for (method, snr_db), group in rows.groupby(["method", "snr_db"], sort=False):
        squared = group["squared_error"].to_numpy(float)
        absolute = group["abs_error"].to_numpy(float)
        records.append(
            {
                "method": method,
                "snr_db": float(snr_db),
                "rmse_samples": float(np.sqrt(np.mean(squared))),
                "mean_posterior_risk": float(group["posterior_risk"].mean()),
                "p95_abs_error_samples": float(np.quantile(absolute, 0.95)),
                "gross_error_gt_1_rate": float(np.mean(absolute > 1.0)),
                "posterior_nll": float(group["posterior_nll"].mean()),
                "credible90_coverage": float(group["credible90_contains_true"].mean()),
                "minimum_bin_energy_median": float(group["minimum_bin_energy"].median()),
                "n_contexts": int(group.shape[0]),
            }
        )
    return pd.DataFrame(records)


def _cluster_bootstrap_effects(
    rows: pd.DataFrame,
    repetitions: int,
    seed: int = 99107,
) -> pd.DataFrame:
    full = rows[rows["method"] == "FullComplex"].set_index(
        ["seed", "trial", "snr_db"]
    )
    records: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)
    for method in rows["method"].drop_duplicates():
        if method in {"FullComplex", "Radius+Projector"}:
            continue
        candidate = rows[rows["method"] == method].set_index(
            ["seed", "trial", "snr_db"]
        )
        joined = candidate[["posterior_risk"]].join(
            full[["posterior_risk"]],
            lsuffix="_candidate",
            rsuffix="_full",
            how="inner",
        )
        joined["risk_excess"] = (
            joined["posterior_risk_candidate"] - joined["posterior_risk_full"]
        )
        cluster = joined.reset_index().groupby(["seed", "trial"])["risk_excess"].mean()
        values = cluster.to_numpy(float)
        bootstrap = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            bootstrap[index] = float(
                np.mean(rng.choice(values, size=values.size, replace=True))
            )
        records.append(
            {
                "method": method,
                "mean_posterior_risk_excess": float(np.mean(values)),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "fraction_clusters_nonpositive": float(np.mean(values <= 0.0)),
                "n_trajectory_clusters": int(values.size),
                "cluster_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(records)


def _coordinate_registry(model: BayesianRAPTQModel, sparse_count: int) -> pd.DataFrame:
    k = model.n_bins
    return pd.DataFrame(
        [
            {
                "representation": "FullComplex",
                "lossy": False,
                "retained_coordinates": "all complex coefficients",
                "marginalized_coordinates": "none",
                "jacobian": "Cartesian",
                "mathematical_integration_dimension": 0,
                "role": "full-information reference",
            },
            {
                "representation": "Radius+Projector",
                "lossy": False,
                "retained_coordinates": "radius and projective direction",
                "marginalized_coordinates": "global phase",
                "jacobian": "U(1) orbit constant",
                "mathematical_integration_dimension": 1,
                "role": "implementation equivalence control",
            },
            {
                "representation": "ProjectorOnly",
                "lossy": True,
                "retained_coordinates": "relative powers and relative phases",
                "marginalized_coordinates": "radius and global phase",
                "jacobian": "2^-K t^(K-1); global phase analytic",
                "mathematical_integration_dimension": 1,
                "role": "radial-information ablation",
            },
            {
                "representation": "RelativePhase",
                "lossy": True,
                "retained_coordinates": f"{k - 1} relative phases",
                "marginalized_coordinates": "K powers, radius, global phase",
                "jacobian": "2^-K t^(K-1), simplex; global phase analytic",
                "mathematical_integration_dimension": k,
                "role": "RAPG continuous upper bound",
            },
            {
                "representation": "RelativePower",
                "lossy": True,
                "retained_coordinates": f"{k - 1} relative powers",
                "marginalized_coordinates": "radius and K phases",
                "jacobian": "analytic tau-invariant magnitude law",
                "mathematical_integration_dimension": 0,
                "role": "exact no-delay-information negative control",
            },
            {
                "representation": "SparsePhase",
                "lossy": True,
                "retained_coordinates": f"{sparse_count} fixed relative phases",
                "marginalized_coordinates": "all powers, radius, global and omitted phases",
                "jacobian": "2^-K t^(K-1), simplex; global phase analytic",
                "mathematical_integration_dimension": 2 * k - 1 - sparse_count,
                "role": "genuinely sparse phase statistic",
            },
            {
                "representation": "SparseAmpPhase",
                "lossy": True,
                "retained_coordinates": (
                    f"{sparse_count} relative powers and {sparse_count} relative phases"
                ),
                "marginalized_coordinates": "radius, global phase, omitted powers and phases",
                "jacobian": "2^-K t^(K-1), constrained simplex; global phase analytic",
                "mathematical_integration_dimension": 2 * k - 2 * sparse_count - 1,
                "role": "genuinely sparse amplitude-phase statistic",
            },
        ]
    )


def _plot(summary: pd.DataFrame, output: Path) -> None:
    methods = tuple(summary["method"].drop_duplicates())
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    metrics = (
        ("rmse_samples", "TDOA RMSE (samples)"),
        ("mean_posterior_risk", "Mean posterior risk (samples²)"),
        ("gross_error_gt_1_rate", "Gross-error rate, |error| > 1 sample"),
    )
    for method_index, method in enumerate(methods):
        group = summary[summary["method"] == method].sort_values("snr_db")
        for axis, (column, label) in zip(axes, metrics, strict=True):
            axis.plot(
                group["snr_db"],
                group[column],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color=colors(method_index % 10),
                label=method,
            )
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.65, -0.20),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("RAPTQ Stage 0 continuous-information audit")
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(mode: str) -> Path:
    audit, integration = build_run_config(mode)
    model_config = ModelConfig()
    model = BayesianRAPTQModel(model_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        Path(__file__).resolve().parents[1]
        / "运行结果"
        / "RAPTQ"
        / f"RAPTQ_STAGE0_{timestamp}_{mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    partial_path = output / "stage0_partial_rows.csv"
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("core.py"),
    )
    manifest = {
        "route": "RAPTQ-TDOA",
        "stage": "Stage 0-T + Stage 0-I",
        "status_before_run": "CANDIDATE_UNDER_THEORY_AND_INFORMATION_AUDIT",
        "mode": mode,
        "model_config": asdict(model_config),
        "audit_config": asdict(audit),
        "qmc_config": asdict(integration),
        "gain_phase_prior": "continuous uniform on [-pi, pi)",
        "gain_scope": "one independent gain per trajectory, shared across its SNR sweep",
        "selected_bins_source": "frozen SITQ Stage A bins",
        "source_hash": _source_hash(source_paths),
        "scientific_boundary": (
            "continuous-statistic audit only; no 24-bit method claim"
        ),
    }
    _write_json(output / "config_manifest.json", manifest)

    contexts = model.build_contexts(
        audit.seeds,
        audit.trials_per_seed,
        audit.snr_db_values,
    )
    print("RAPTQ-TDOA Stage 0", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        f"Contexts={len(contexts)}, QMC=2^{integration.power}, "
        f"reference=2^{integration.reference_power}",
        flush=True,
    )
    start = time.perf_counter()
    rows, exact_checks = _evaluate(
        model,
        contexts,
        audit,
        integration,
        partial_path,
    )
    finite_window = finite_window_rotation_certificate()
    rows.to_csv(output / "stage0_information_rows.csv", index=False)
    summary = _summary(rows)
    summary.to_csv(output / "stage0_information_summary.csv", index=False)
    effects = _cluster_bootstrap_effects(
        rows,
        audit.bootstrap_repetitions,
    )
    effects.to_csv(output / "stage0_paired_risk_effects.csv", index=False)
    registry = _coordinate_registry(model, audit.sparse_coordinate_count)
    registry.to_csv(output / "stage0_coordinate_registry.csv", index=False)
    numerical = rows[
        rows["qmc_reference_tv"].notna()
    ][
        [
            "method",
            "seed",
            "trial",
            "snr_db",
            "qmc_reference_tv",
            "qmc_reference_risk_difference",
            "minimum_bin_energy",
        ]
    ]
    numerical.to_csv(output / "stage0_numerical_diagnostics.csv", index=False)
    _plot(summary, output / "Fig1_RAPTQ_Stage0_InformationAudit.svg")

    row_keys = ["method", "seed", "trial", "snr_db"]
    expected_methods_per_context = len(METHOD_LABELS)
    expected_row_count = len(contexts) * expected_methods_per_context
    rows_per_context = rows.groupby(["seed", "trial", "snr_db"]).size()
    complete_context_rows = bool(
        rows.shape[0] == expected_row_count
        and not rows_per_context.empty
        and (rows_per_context == expected_methods_per_context).all()
    )
    finite_rows = bool(
        np.isfinite(
            rows[
                [
                    "estimate",
                    "squared_error",
                    "posterior_risk",
                    "posterior_nll",
                ]
            ].to_numpy(float)
        ).all()
    )
    checks: dict[str, object] = {
        **exact_checks,
        **finite_window,
        "continuous_uniform_gain_phase": True,
        "gain_amplitude_prior_normalized": math.isclose(
            float(np.sum(model.gain_amplitude_prior)), 1.0, abs_tol=1e-12
        ),
        "tau_prior_normalized": math.isclose(
            float(np.sum(model.tau_prior)), 1.0, abs_tol=1e-12
        ),
        "expected_row_count": expected_row_count,
        "actual_row_count": int(rows.shape[0]),
        "row_keys_unique": bool(not rows.duplicated(row_keys).any()),
        "all_posterior_metrics_finite": finite_rows,
        "low_energy_samples_retained": complete_context_rows,
        "dropped_context_count": int(
            len(contexts) - rows_per_context.index.nunique()
        ),
        "qmc_reference_max_tv": (
            float(numerical["qmc_reference_tv"].max())
            if not numerical.empty
            else math.nan
        ),
        "qmc_reference_max_risk_difference": (
            float(numerical["qmc_reference_risk_difference"].max())
            if not numerical.empty
            else math.nan
        ),
    }
    exact_pass = all(
        bool(checks[key])
        for key in (
            "global_rotation_invariant",
            "radius_projector_posterior_equivalent",
            "projector_roundtrip_valid",
            "finite_window_rotation_invariant",
            "finite_window_radius_is_delay_sensitive",
            "gain_amplitude_prior_normalized",
            "tau_prior_normalized",
            "row_keys_unique",
            "all_posterior_metrics_finite",
            "low_energy_samples_retained",
        )
    )
    if mode == "smoke":
        gate_status = "SMOKE_ONLY"
    elif not exact_pass:
        gate_status = "INVALID_IMPLEMENTATION_OR_MODEL_ASSUMPTION"
    else:
        gate_status = "STAGE0_T_PASS_STAGE0_I_PENDING_NUMERICAL_REVIEW"
    checks["gate_status"] = gate_status
    checks["elapsed_seconds"] = time.perf_counter() - start
    _write_json(output / "validation.json", checks)
    if partial_path.exists():
        partial_path.unlink()
    print(f"[Gate] {gate_status}", flush=True)
    print(f"[Done] elapsed={checks['elapsed_seconds']:.1f}s", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run("smoke" if arguments.smoke else RUN_MODE)


if __name__ == "__main__":
    main()
