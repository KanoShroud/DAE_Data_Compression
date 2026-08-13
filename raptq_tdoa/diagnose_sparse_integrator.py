"""Certify the sparse RAPTQ induced-likelihood integrator.

Run this file directly in PyCharm.  The development mode first executes
mechanical checks and seven frozen stress contexts.  Only representations that
pass that development screen are evaluated on the frozen 12-context protocol.
The script does not modify the historical Stage 0 result or its numerical
certificate, and it does not implement a 24-bit RAPTQ method.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp, roots_genlaguerre
from scipy.stats import qmc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raptq_tdoa.core import (  # noqa: E402
    BayesianRAPTQModel,
    ModelConfig,
    ObservationContext,
    QMCConfig,
    posterior_total_variation,
)
from raptq_tdoa.sparse_integrator import (  # noqa: E402
    SparseRAPTQIntegrator,
    dirichlet_stick_breaking,
)


RUN_MODE = "development"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE0 = (
    ROOT
    / "运行结果"
    / "RAPTQ"
    / "RAPTQ_STAGE0_20260805_105037_development"
)
SOURCE_NUMERICAL = (
    ROOT
    / "运行结果"
    / "RAPTQ"
    / "RAPTQ_STAGE0_NUMERICAL_20260805_115458_development"
)
REPRESENTATIONS = ("sparse_phase", "sparse_amp_phase")
PROPOSALS = ("model", "defensive_mixture")


@dataclass(frozen=True)
class CertificateConfig:
    mode: str
    development_selectors: tuple[tuple[int, int, float], ...]
    review_selectors: tuple[tuple[int, int, float], ...]
    qmc_powers: tuple[int, ...]
    repeats: int
    sparse_coordinate_count: int
    risk_guard_samples2: float
    moment_power: int
    mechanical_power: int
    mechanical_repeats: int

    def validate(self) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.development_selectors or not self.review_selectors:
            raise ValueError("context selectors must not be empty")
        if tuple(sorted(self.qmc_powers)) != self.qmc_powers:
            raise ValueError("QMC powers must be strictly ordered")
        if len(set(self.qmc_powers)) != len(self.qmc_powers):
            raise ValueError("QMC powers must be unique")
        if self.repeats < 1 or self.mechanical_repeats < 2:
            raise ValueError("repeat counts are invalid")
        if self.risk_guard_samples2 <= 0.0:
            raise ValueError("risk guard must be positive")


def build_config(mode: str, manifest: dict[str, object]) -> CertificateConfig:
    raw_audit = manifest.get("audit_config")
    if not isinstance(raw_audit, dict):
        raise TypeError("source audit_config must be a dictionary")
    seeds = tuple(int(value) for value in raw_audit["seeds"])
    snr = tuple(float(value) for value in raw_audit["snr_db_values"])
    sparse_count = int(raw_audit["sparse_coordinate_count"])
    if seeds != (4301, 4302) or snr != (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0):
        raise RuntimeError("source selectors do not match the frozen protocol")
    development = (
        (4301, 0, -5.0),
        (4301, 0, 5.0),
        (4301, 0, 10.0),
        (4301, 0, 15.0),
        (4302, 0, 5.0),
        (4302, 0, 10.0),
        (4302, 0, 15.0),
    )
    review = tuple((seed, 0, value) for seed in seeds for value in snr)
    if mode == "smoke":
        config = CertificateConfig(
            mode=mode,
            development_selectors=development[:2],
            review_selectors=review[:2],
            qmc_powers=(5, 6),
            repeats=2,
            sparse_coordinate_count=sparse_count,
            risk_guard_samples2=0.0625,
            moment_power=10,
            mechanical_power=9,
            mechanical_repeats=3,
        )
    elif mode == "development":
        config = CertificateConfig(
            mode=mode,
            development_selectors=development,
            review_selectors=review,
            qmc_powers=(14, 15, 16),
            repeats=8,
            sparse_coordinate_count=sparse_count,
            risk_guard_samples2=0.0625,
            moment_power=16,
            mechanical_power=14,
            mechanical_repeats=8,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")
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
    raw = manifest.get("model_config")
    if not isinstance(raw, dict):
        raise TypeError("source model_config must be a dictionary")
    values = dict(raw)
    values["selected_bins"] = tuple(int(value) for value in values["selected_bins"])
    values["gain_amplitudes"] = tuple(float(value) for value in values["gain_amplitudes"])
    return BayesianRAPTQModel(ModelConfig(**values))


def _contexts(
    model: BayesianRAPTQModel,
    selectors: tuple[tuple[int, int, float], ...],
) -> list[ObservationContext]:
    trials: dict[tuple[int, int], object] = {}
    contexts: list[ObservationContext] = []
    for seed, trial_index, snr_db in selectors:
        key = (seed, trial_index)
        if key not in trials:
            trials[key] = model.generate_trials(seed, trial_index + 1)[trial_index]
        contexts.append(model.build_context(trials[key], snr_db))
    return contexts


def _seed(
    context: ObservationContext,
    representation: str,
    proposal: str,
    power: int,
    stage: str,
) -> int:
    token = (
        f"{context.seed}:{context.trial}:{context.snr_db}:{representation}:"
        f"{proposal}:{power}:{stage}:sparse-integrator-v1"
    )
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "little")


def _dirichlet_mechanical_check(config: CertificateConfig, n_parts: int) -> dict[str, object]:
    n_points = 2**config.moment_power
    unit = qmc.Sobol(n_parts - 1, scramble=True, seed=97231).random_base2(
        config.moment_power
    )
    unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    shares = dirichlet_stick_breaking(unit, np.ones(n_parts, dtype=float))
    expected_mean = 1.0 / n_parts
    expected_variance = (n_parts - 1.0) / (n_parts**2 * (n_parts + 1.0))
    expected_covariance = -1.0 / (n_parts**2 * (n_parts + 1.0))
    empirical_mean = np.mean(shares, axis=0)
    mean_tolerance = 6.0 * math.sqrt(expected_variance / n_points)
    expected_second_diagonal = 2.0 / (n_parts * (n_parts + 1.0))
    expected_second_off_diagonal = 1.0 / (n_parts * (n_parts + 1.0))
    expected_fourth_diagonal = 24.0 / (
        n_parts * (n_parts + 1.0) * (n_parts + 2.0) * (n_parts + 3.0)
    )
    expected_fourth_off_diagonal = 4.0 / (
        n_parts * (n_parts + 1.0) * (n_parts + 2.0) * (n_parts + 3.0)
    )
    diagonal_se = math.sqrt(
        (expected_fourth_diagonal - expected_second_diagonal**2) / n_points
    )
    off_diagonal_se = math.sqrt(
        (expected_fourth_off_diagonal - expected_second_off_diagonal**2)
        / n_points
    )
    diagonal_second = np.mean(shares**2, axis=0)
    off_diagonal_second = np.asarray(
        [
            np.mean(shares[:, first] * shares[:, second])
            for first in range(n_parts)
            for second in range(first + 1, n_parts)
        ],
        dtype=float,
    )
    max_mean_error = float(np.max(np.abs(empirical_mean - expected_mean)))
    max_diagonal_second_error = float(
        np.max(np.abs(diagonal_second - expected_second_diagonal))
    )
    max_off_diagonal_second_error = float(
        np.max(np.abs(off_diagonal_second - expected_second_off_diagonal))
    )
    return {
        "check": "dirichlet_stick_breaking",
        "n_points": n_points,
        "max_sum_error": float(np.max(np.abs(np.sum(shares, axis=1) - 1.0))),
        "min_share": float(np.min(shares)),
        "max_mean_error": max_mean_error,
        "mean_tolerance": mean_tolerance,
        "expected_covariance": expected_covariance,
        "max_diagonal_second_error": max_diagonal_second_error,
        "diagonal_second_tolerance": 6.0 * diagonal_se,
        "max_off_diagonal_second_error": max_off_diagonal_second_error,
        "off_diagonal_second_tolerance": 6.0 * off_diagonal_se,
        "passed": bool(
            np.all(np.isfinite(shares))
            and np.all(shares > 0.0)
            and max_mean_error <= mean_tolerance
            and max_diagonal_second_error <= 6.0 * diagonal_se
            and max_off_diagonal_second_error <= 6.0 * off_diagonal_se
        ),
    }


def _deterministic_sparse_posterior(
    model: BayesianRAPTQModel,
    context: ObservationContext,
    representation: str,
    radial_nodes: int,
    simplex_nodes: int,
    phase_nodes: int,
) -> np.ndarray:
    if model.n_bins != 3:
        raise ValueError("the deterministic reference is registered for three bins")
    _, observed_power, observed_phase = model.observed_coordinates(
        context.y_b_normalized
    )
    retained = 1
    if representation == "sparse_amp_phase":
        remaining = 1.0 - float(observed_power[retained])
        simplex_dimension = 1
    elif representation == "sparse_phase":
        remaining = 1.0
        simplex_dimension = 2
    else:
        raise ValueError(f"unsupported representation: {representation}")
    radial_x, radial_w = roots_genlaguerre(radial_nodes, model.n_bins - 1)
    simplex_x, simplex_w_raw = leggauss(simplex_nodes)
    simplex = 0.5 * (simplex_x + 1.0)
    simplex_w = 0.5 * simplex_w_raw
    phase = np.linspace(-np.pi, np.pi, phase_nodes, endpoint=False)
    phase_w = 2.0 * np.pi / phase_nodes
    means, variances = model._means_and_variances(context)
    expected_energy = float(
        np.mean(np.sum(np.abs(means) ** 2, axis=2))
        + np.mean(np.sum(variances, axis=1))
    )
    scale = max(expected_energy / model.n_bins, 1e-3)
    log_terms: list[np.ndarray] = []
    for radial_value, radial_weight in zip(radial_x, radial_w, strict=True):
        radius_squared = scale * float(radial_value)
        simplex_point_count = simplex_nodes**simplex_dimension
        powers = np.empty(
            (simplex_point_count * phase_nodes, model.n_bins),
            dtype=float,
        )
        phases = np.zeros_like(powers)
        if representation == "sparse_amp_phase":
            simplex_powers = np.column_stack(
                (remaining * simplex, remaining * (1.0 - simplex))
            )
            simplex_weights = remaining * simplex_w
            powers[:, retained] = observed_power[retained]
            powers[:, 0] = np.repeat(simplex_powers[:, 0], phase_nodes)
            powers[:, 2] = np.repeat(simplex_powers[:, 1], phase_nodes)
        else:
            first, second = np.meshgrid(simplex, simplex, indexing="ij")
            first_weight, second_weight = np.meshgrid(
                simplex_w,
                simplex_w,
                indexing="ij",
            )
            first = first.ravel()
            second = second.ravel()
            simplex_powers = np.column_stack(
                (
                    first,
                    (1.0 - first) * second,
                    (1.0 - first) * (1.0 - second),
                )
            )
            simplex_weights = (
                first_weight.ravel()
                * second_weight.ravel()
                * (1.0 - first)
            )
            for index in range(model.n_bins):
                powers[:, index] = np.repeat(
                    simplex_powers[:, index],
                    phase_nodes,
                )
        phases[:, retained] = observed_phase[retained]
        phases[:, 2] = np.tile(phase, simplex_point_count)
        points = np.sqrt(radius_squared * powers) * np.exp(1j * phases)
        log_quadrature_weight = (
            model.n_bins * math.log(scale)
            + math.log(float(radial_weight))
            + float(radial_value)
            + np.repeat(np.log(simplex_weights), phase_nodes)
            + math.log(phase_w)
        )
        log_terms.append(
            model.log_density_by_tau(points, context)
            + log_quadrature_weight[:, None]
        )
    log_likelihood = logsumexp(np.concatenate(log_terms, axis=0), axis=0)
    return model._posterior_from_log_likelihood(log_likelihood)


def _deterministic_mechanical_checks(config: CertificateConfig) -> list[dict[str, object]]:
    model = BayesianRAPTQModel(
        ModelConfig(selected_bins=(31, 145, 298))
    )
    trial = model.generate_trials(61321, 1)[0]
    context = model.build_context(trial, -5.0)
    integrator = SparseRAPTQIntegrator(model, sparse_coordinate_count=1)
    if config.mode == "smoke":
        coarse_grid = (8, 8, 24)
        reference_grid = (12, 12, 40)
        qmc_power = 9
    else:
        coarse_grid = (12, 12, 40)
        reference_grid = (20, 20, 64)
        qmc_power = 14
    rows: list[dict[str, object]] = []
    for representation in REPRESENTATIONS:
        coarse = _deterministic_sparse_posterior(
            model,
            context,
            representation,
            *coarse_grid,
        )
        reference = _deterministic_sparse_posterior(
            model,
            context,
            representation,
            *reference_grid,
        )
        reference_risk = model.summarize_posterior(
            reference,
            context.true_tau,
        ).posterior_risk
        coarse_risk = model.summarize_posterior(
            coarse,
            context.true_tau,
        ).posterior_risk
        rows.append(
            {
                "check": "low_dimensional_deterministic_reference",
                "representation": representation,
                "proposal": "deterministic_coarse",
                "reference": f"tensor_{reference_grid}",
                "risk_difference": abs(coarse_risk - reference_risk),
                "posterior_tv": posterior_total_variation(coarse, reference),
                "passed": (
                    abs(coarse_risk - reference_risk)
                    <= config.risk_guard_samples2
                ),
            }
        )
        for proposal in PROPOSALS:
            result = integrator.integrate(
                context,
                representation,
                qmc_power,
                config.mechanical_repeats,
                71003,
                proposal,
            )
            risk = model.summarize_posterior(
                result.posterior,
                context.true_tau,
            ).posterior_risk
            rows.append(
                {
                    "check": "low_dimensional_deterministic_reference",
                    "representation": representation,
                    "proposal": proposal,
                    "reference": f"tensor_{reference_grid}",
                    "risk_difference": abs(risk - reference_risk),
                    "posterior_tv": posterior_total_variation(
                        result.posterior,
                        reference,
                    ),
                    "passed": (
                        abs(risk - reference_risk)
                        <= config.risk_guard_samples2
                    ),
                }
            )
    return rows


def _radial_mechanical_checks(
    integrator: SparseRAPTQIntegrator,
    context: ObservationContext,
    config: CertificateConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for representation in REPRESENTATIONS:
        for proposal in PROPOSALS:
            estimates: list[float] = []
            exact = math.nan
            for repeat in range(config.mechanical_repeats):
                estimate, exact = integrator.radial_measure_integral_check(
                    context,
                    representation,
                    config.mechanical_power,
                    8191 + 65537 * repeat,
                    proposal,
                )
                estimates.append(estimate)
            values = np.asarray(estimates, dtype=float)
            mean = float(np.mean(values))
            standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
            tolerance = max(6.0 * standard_error, 1e-10 * abs(exact))
            rows.append(
                {
                    "check": "artificial_radial_measure",
                    "representation": representation,
                    "proposal": proposal,
                    "n_points_per_repeat": 2**config.mechanical_power,
                    "repeats": config.mechanical_repeats,
                    "estimate_mean": mean,
                    "estimate_standard_error": standard_error,
                    "exact": exact,
                    "absolute_error": abs(mean - exact),
                    "six_se_tolerance": tolerance,
                    "passed": bool(
                        np.isfinite(values).all()
                        and math.isfinite(exact)
                        and abs(mean - exact) <= tolerance
                    ),
                }
            )
    return rows


def _mechanical_checks(
    integrator: SparseRAPTQIntegrator,
    context: ObservationContext,
    config: CertificateConfig,
) -> pd.DataFrame:
    rows = [_dirichlet_mechanical_check(config, integrator.model.n_bins)]
    rows.extend(_radial_mechanical_checks(integrator, context, config))
    rows.extend(_deterministic_mechanical_checks(config))
    return pd.DataFrame(rows)


def _evaluate_stage(
    integrator: SparseRAPTQIntegrator,
    contexts: list[ObservationContext],
    config: CertificateConfig,
    stage: str,
    representations: tuple[str, ...],
    partial_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(contexts) * len(representations) * len(PROPOSALS) * len(
        config.qmc_powers
    )
    completed = 0
    start = time.perf_counter()
    for context_index, context in enumerate(contexts, start=1):
        print(
            f"[{stage}] context {context_index}/{len(contexts)} "
            f"seed={context.seed} trial={context.trial} SNR={context.snr_db:g}",
            flush=True,
        )
        full = integrator.model.posterior_full(context)
        full_risk = integrator.model.summarize_posterior(
            full,
            context.true_tau,
        ).posterior_risk
        for representation in representations:
            for power in config.qmc_powers:
                for proposal in PROPOSALS:
                    result = integrator.integrate(
                        context,
                        representation,
                        power,
                        config.repeats,
                        _seed(context, representation, proposal, power, stage),
                        proposal,
                    )
                    summary = integrator.model.summarize_posterior(
                        result.posterior,
                        context.true_tau,
                    )
                    repeat_risks = np.asarray(
                        [
                            integrator.model.summarize_posterior(
                                integrator.model._posterior_from_log_likelihood(value),
                                context.true_tau,
                            ).posterior_risk
                            for value in result.repeat_log_likelihoods
                        ],
                        dtype=float,
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "seed": context.seed,
                            "trial": context.trial,
                            "snr_db": context.snr_db,
                            "true_tau": context.true_tau,
                            "representation": representation,
                            "proposal": proposal,
                            "qmc_power": power,
                            "qmc_points_per_repeat": 2**power,
                            "qmc_repeats": config.repeats,
                            "posterior_risk": summary.posterior_risk,
                            "risk_excess_vs_full": summary.posterior_risk - full_risk,
                            "posterior_estimate": summary.estimate,
                            "posterior_entropy": summary.entropy,
                            "repeat_risk_sd": float(np.std(repeat_risks, ddof=1)),
                            "max_integrand_weight_fraction": (
                                result.max_integrand_weight_fraction
                            ),
                            "min_integrand_effective_sample_size": (
                                result.min_integrand_effective_sample_size
                            ),
                            "posterior_json": json.dumps(result.posterior.tolist()),
                        }
                    )
                    completed += 1
                    if completed % max(1, total // 20) == 0 or completed == total:
                        elapsed = time.perf_counter() - start
                        seconds_per_run = elapsed / completed
                        eta = seconds_per_run * (total - completed)
                        print(
                            f"[{stage}] {completed}/{total} elapsed={elapsed:.1f}s "
                            f"avg={seconds_per_run:.2f}s ETA={eta:.1f}s",
                            flush=True,
                        )
        pd.DataFrame(rows).to_csv(partial_path, index=False)
    return pd.DataFrame(rows)


def _proposal_agreement(rows: pd.DataFrame, highest_power: int) -> pd.DataFrame:
    final = rows[rows["qmc_power"] == highest_power].copy()
    records: list[dict[str, object]] = []
    keys = ["stage", "seed", "trial", "snr_db", "true_tau", "representation"]
    for key, group in final.groupby(keys, sort=False):
        values = group.set_index("proposal")
        if set(values.index) != set(PROPOSALS):
            raise RuntimeError(f"proposal rows are incomplete for {key}")
        model_posterior = np.asarray(json.loads(values.loc["model", "posterior_json"]))
        focused_posterior = np.asarray(
            json.loads(values.loc["defensive_mixture", "posterior_json"])
        )
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "qmc_power": highest_power,
                "model_risk": float(values.loc["model", "posterior_risk"]),
                "defensive_risk": float(
                    values.loc["defensive_mixture", "posterior_risk"]
                ),
                "risk_difference": abs(
                    float(values.loc["model", "posterior_risk"])
                    - float(values.loc["defensive_mixture", "posterior_risk"])
                ),
                "posterior_tv": posterior_total_variation(
                    model_posterior,
                    focused_posterior,
                ),
                "posterior_mean_difference": abs(
                    float(values.loc["model", "posterior_estimate"])
                    - float(values.loc["defensive_mixture", "posterior_estimate"])
                ),
                "max_integrand_weight_fraction": float(
                    values["max_integrand_weight_fraction"].max()
                ),
                "min_integrand_effective_sample_size": float(
                    values["min_integrand_effective_sample_size"].min()
                ),
                "max_repeat_risk_sd": float(values["repeat_risk_sd"].max()),
            }
        )
    return pd.DataFrame(records)


def _legacy_low_snr_check(
    model: BayesianRAPTQModel,
    integrator: SparseRAPTQIntegrator,
    context: ObservationContext,
    config: CertificateConfig,
) -> pd.DataFrame:
    power = max(config.qmc_powers)
    repeats = config.repeats
    records: list[dict[str, object]] = []
    legacy_config = QMCConfig(
        power=power,
        repeats=repeats,
        reference_power=power + 1,
        reference_context_limit=1,
        proposal="model_radial",
    )
    for representation in REPRESENTATIONS:
        legacy = model.posterior_representation(
            context,
            representation,
            legacy_config,
            config.sparse_coordinate_count,
            _seed(context, representation, "legacy", power, "mechanical"),
        )
        current = integrator.integrate(
            context,
            representation,
            power,
            repeats,
            _seed(context, representation, "model", power, "mechanical"),
            "model",
        ).posterior
        legacy_risk = model.summarize_posterior(legacy, context.true_tau).posterior_risk
        current_risk = model.summarize_posterior(current, context.true_tau).posterior_risk
        records.append(
            {
                "representation": representation,
                "seed": context.seed,
                "trial": context.trial,
                "snr_db": context.snr_db,
                "legacy_risk": legacy_risk,
                "nonredundant_risk": current_risk,
                "risk_difference": abs(legacy_risk - current_risk),
                "posterior_tv": posterior_total_variation(legacy, current),
                "passed": abs(legacy_risk - current_risk)
                <= config.risk_guard_samples2,
            }
        )
    return pd.DataFrame(records)


def _representation_status(
    agreement: pd.DataFrame,
    representations: tuple[str, ...],
    guard: float,
    passed_label: str,
    failed_label: str,
) -> dict[str, str]:
    status: dict[str, str] = {}
    for representation in representations:
        group = agreement[agreement["representation"] == representation]
        if group.empty:
            status[representation] = "NOT_EVALUATED"
        elif float(group["risk_difference"].max()) <= guard:
            status[representation] = passed_label
        else:
            status[representation] = failed_label
    return status


def _plot(
    development: pd.DataFrame,
    review: pd.DataFrame,
    output: Path,
    guard: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    for axis, frame, title in (
        (axes[0], development, "Frozen stress contexts"),
        (axes[1], review, "Frozen 12-context review"),
    ):
        if frame.empty:
            axis.text(0.5, 0.5, "Not evaluated", ha="center", va="center")
            axis.set_axis_off()
            continue
        for representation, group in frame.groupby("representation"):
            summary = group.groupby("snr_db", as_index=False)["risk_difference"].max()
            axis.plot(
                summary["snr_db"],
                summary["risk_difference"],
                marker="o",
                linewidth=1.2,
                label=representation,
            )
        axis.axhline(guard, color="black", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("Model/defensive risk difference (samples²)")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="best")
    fig.suptitle("RAPTQ sparse-integrator certification")
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def run(mode: str) -> Path:
    for source in (SOURCE_STAGE0, SOURCE_NUMERICAL):
        if not source.is_dir():
            raise FileNotFoundError(f"frozen source result not found: {source}")
    source_stage0_hash_before = _tree_hash(SOURCE_STAGE0)
    source_numerical_hash_before = _tree_hash(SOURCE_NUMERICAL)
    source_manifest = _json(SOURCE_STAGE0 / "config_manifest.json")
    source_validation = _json(SOURCE_NUMERICAL / "validation.json")
    if source_manifest.get("route") != "RAPTQ-TDOA":
        raise RuntimeError("unexpected source route")
    if source_validation.get("gate_status") != "HOLD_NUMERICAL_CONVERGENCE":
        raise RuntimeError("source numerical certificate is not the frozen HOLD")

    config = build_config(mode, source_manifest)
    model = _model_from_manifest(source_manifest)
    integrator = SparseRAPTQIntegrator(model, config.sparse_coordinate_count)
    development_contexts = _contexts(model, config.development_selectors)
    review_contexts = _contexts(model, config.review_selectors)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        ROOT
        / "运行结果"
        / "RAPTQ"
        / f"RAPTQ_SPARSE_INTEGRATOR_{timestamp}_{mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    development_partial = output / "development_partial_rows.csv"
    review_partial = output / "review_partial_rows.csv"
    manifest = {
        "route": "RAPTQ-TDOA",
        "stage": "Sparse integrator certification",
        "mode": mode,
        "config": asdict(config),
        "source_stage0": str(SOURCE_STAGE0),
        "source_numerical": str(SOURCE_NUMERICAL),
        "source_stage0_hash_before": source_stage0_hash_before,
        "source_numerical_hash_before": source_numerical_hash_before,
        "protocol_boundary": (
            "The seven stress contexts are development cases and overlap the "
            "frozen 12-context review. The review is a frozen-protocol coverage "
            "check, not an independent statistical confirmation."
        ),
        "decision_boundary": (
            "This run certifies numerical computability only. It neither changes "
            "the historical HOLD nor approves a 24-bit RAPTQ method."
        ),
        "source_hash": _source_hash(
            (
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("sparse_integrator.py"),
                Path(__file__).resolve().with_name("core.py"),
            )
        ),
    }
    _write_json(output / "config_manifest.json", manifest)

    print("RAPTQ sparse-integrator certification", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        f"Stress contexts={len(development_contexts)}, "
        f"review contexts={len(review_contexts)}, powers={config.qmc_powers}, "
        f"repeats={config.repeats}",
        flush=True,
    )
    start = time.perf_counter()

    mechanical = _mechanical_checks(integrator, development_contexts[0], config)
    low_snr_context = next(
        context for context in development_contexts if context.snr_db == -5.0
    )
    legacy = _legacy_low_snr_check(model, integrator, low_snr_context, config)
    mechanical_pass = bool(
        mechanical["passed"].fillna(False).all()
        and legacy["passed"].fillna(False).all()
    )
    print(f"[Mechanical] passed={mechanical_pass}", flush=True)

    development = pd.DataFrame()
    development_agreement = pd.DataFrame()
    review = pd.DataFrame()
    review_agreement = pd.DataFrame()
    development_status = {name: "NOT_EVALUATED" for name in REPRESENTATIONS}
    review_status = {name: "NOT_EVALUATED" for name in REPRESENTATIONS}

    if mechanical_pass:
        development = _evaluate_stage(
            integrator,
            development_contexts,
            config,
            "development",
            REPRESENTATIONS,
            development_partial,
        )
        development_agreement = _proposal_agreement(
            development,
            max(config.qmc_powers),
        )
        development_status = _representation_status(
            development_agreement,
            REPRESENTATIONS,
            config.risk_guard_samples2,
            "PASS_DEVELOPMENT_SCREEN",
            "HOLD_COMPUTABILITY_DEVELOPMENT",
        )
        passed = tuple(
            name
            for name in REPRESENTATIONS
            if development_status[name] == "PASS_DEVELOPMENT_SCREEN"
        )
        if mode == "smoke":
            passed = REPRESENTATIONS
        if passed:
            review = _evaluate_stage(
                integrator,
                review_contexts,
                config,
                "review",
                passed,
                review_partial,
            )
            review_agreement = _proposal_agreement(review, max(config.qmc_powers))
            review_status = _representation_status(
                review_agreement,
                REPRESENTATIONS,
                config.risk_guard_samples2,
                "CERTIFIED",
                "HOLD_COMPUTABILITY",
            )
        if mode == "smoke":
            development_status = {
                name: "SMOKE_ONLY_NOT_SCIENTIFIC_GATE" for name in REPRESENTATIONS
            }
            review_status = {
                name: "SMOKE_ONLY_NOT_SCIENTIFIC_GATE" for name in REPRESENTATIONS
            }

    source_stage0_hash_after = _tree_hash(SOURCE_STAGE0)
    source_numerical_hash_after = _tree_hash(SOURCE_NUMERICAL)
    source_unchanged = bool(
        source_stage0_hash_before == source_stage0_hash_after
        and source_numerical_hash_before == source_numerical_hash_after
    )
    if not source_unchanged:
        overall_gate = "INVALID_SOURCE_RESULT_MUTATION"
    elif not mechanical_pass:
        overall_gate = "INVALID_INTEGRATOR"
    elif mode == "smoke":
        overall_gate = "SMOKE_ONLY_ENGINEERING_PASS"
    elif all(review_status[name] == "CERTIFIED" for name in REPRESENTATIONS):
        overall_gate = "CERTIFIED_ALL_SPARSE_REPRESENTATIONS"
    elif any(review_status[name] == "CERTIFIED" for name in REPRESENTATIONS):
        overall_gate = "CERTIFIED_PARTIAL_SPARSE_REPRESENTATIONS"
    else:
        overall_gate = "HOLD_COMPUTABILITY"

    summary_rows: list[dict[str, object]] = []
    for representation in REPRESENTATIONS:
        development_group = development_agreement[
            development_agreement.get("representation", pd.Series(dtype=str))
            == representation
        ]
        review_group = review_agreement[
            review_agreement.get("representation", pd.Series(dtype=str))
            == representation
        ]
        summary_rows.append(
            {
                "representation": representation,
                "development_status": development_status[representation],
                "development_max_risk_difference": (
                    float(development_group["risk_difference"].max())
                    if not development_group.empty
                    else math.nan
                ),
                "review_status": review_status[representation],
                "review_max_risk_difference": (
                    float(review_group["risk_difference"].max())
                    if not review_group.empty
                    else math.nan
                ),
                "risk_guard_samples2": config.risk_guard_samples2,
            }
        )
    representation_summary = pd.DataFrame(summary_rows)
    validation = {
        "historical_gate_preserved": "HOLD_NUMERICAL_CONVERGENCE",
        "mechanical_checks_passed": mechanical_pass,
        "source_results_unchanged": source_unchanged,
        "development_status": development_status,
        "review_status": review_status,
        "risk_guard_name": "delta_guard",
        "risk_guard_samples2": config.risk_guard_samples2,
        "risk_guard_is_strict_qmc_error_bound": False,
        "stress_contexts_overlap_review": True,
        "independent_statistical_confirmation_performed": False,
        "overall_gate": overall_gate,
        "elapsed_seconds": time.perf_counter() - start,
    }

    mechanical.to_csv(output / "mechanical_checks.csv", index=False)
    legacy.to_csv(output / "legacy_low_snr_equivalence.csv", index=False)
    development.to_csv(output / "development_rows.csv", index=False)
    development_agreement.to_csv(
        output / "development_proposal_agreement.csv",
        index=False,
    )
    review.to_csv(output / "review_rows.csv", index=False)
    review_agreement.to_csv(output / "review_proposal_agreement.csv", index=False)
    representation_summary.to_csv(
        output / "representation_gate_summary.csv",
        index=False,
    )
    _write_json(output / "validation.json", validation)
    _plot(
        development_agreement,
        review_agreement,
        output / "Fig1_RAPTQ_Sparse_Integrator_Certificate.svg",
        config.risk_guard_samples2,
    )
    for partial in (development_partial, review_partial):
        if partial.exists():
            partial.unlink()
    print(f"[Gate] {overall_gate}", flush=True)
    print(f"[Done] elapsed={validation['elapsed_seconds']:.1f}s", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run("smoke" if arguments.smoke else RUN_MODE)


if __name__ == "__main__":
    main()
