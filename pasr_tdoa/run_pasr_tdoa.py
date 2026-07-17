"""Standalone PASR-TDOA experiment.

Run this file directly in PyCharm.  The default smoke mode verifies the full
chain without modifying or importing the project's DAE/urban8 pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pasr_tdoa.pasr_core import (  # noqa: E402
    ActiveEstimate,
    SpectralModel,
    build_spectral_model,
    estimate_subset,
    fisher_information_from_power,
    profiled_mean_distance_from_power,
    quantize_complex_spectrum,
    run_active_refinement,
)


PYCHARM_RUN_MODE = "research"


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    snr_db: tuple[float, ...]
    development_seeds: tuple[int, ...]
    locked_seeds: tuple[int, ...]
    development_trials: int
    locked_trials: int
    qbits_values: tuple[int, ...]
    primary_qbits: int
    static_qbits_values: tuple[int, ...]
    scale_bits: int
    tau_lower_samples: float
    tau_upper_samples: float
    grid_step_samples: float
    max_bins: int
    confidence_threshold: float
    variance_target: float
    max_modes: int
    min_mode_separation_samples: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    budget_match_relative_tolerance: float
    progress_every: int


CONFIGS = {
    "smoke": ExperimentConfig(
        mode="smoke",
        snr_db=(-5.0, 5.0, 15.0),
        development_seeds=(20265001,),
        locked_seeds=(20266001,),
        development_trials=12,
        locked_trials=20,
        qbits_values=(6,),
        primary_qbits=6,
        static_qbits_values=(4, 5, 6, 7, 8, 9, 10),
        scale_bits=8,
        tau_lower_samples=-4.0,
        tau_upper_samples=4.0,
        grid_step_samples=1.0 / 32.0,
        max_bins=4,
        confidence_threshold=0.95,
        variance_target=0.04,
        max_modes=5,
        min_mode_separation_samples=0.35,
        bootstrap_repetitions=200,
        bootstrap_seed=2026071604,
        budget_match_relative_tolerance=0.03,
        progress_every=10,
    ),
    "research": ExperimentConfig(
        mode="research",
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0),
        development_seeds=(20265001, 20265002, 20265003),
        locked_seeds=(20266001, 20266002, 20266003, 20266004, 20266005, 20266006),
        development_trials=100,
        locked_trials=300,
        qbits_values=(4, 6, 8),
        primary_qbits=6,
        static_qbits_values=(4, 5, 6, 7, 8, 9, 10),
        scale_bits=8,
        tau_lower_samples=-4.0,
        tau_upper_samples=4.0,
        grid_step_samples=1.0 / 32.0,
        max_bins=4,
        confidence_threshold=0.95,
        variance_target=0.04,
        max_modes=5,
        min_mode_separation_samples=0.35,
        bootstrap_repetitions=5000,
        bootstrap_seed=2026071604,
        budget_match_relative_tolerance=0.03,
        progress_every=50,
    ),
}


KNOWN_MASK_BINS = {
    "PFRS": (-8, -5, 6, 7),
    "Zhai-Fisher": (-8, -7, 6, 7),
    "Power": (-2, -1, 0, 1),
    "Uniform": (-8, -3, 2, 7),
}

ACTIVE_METHODS = {
    "PASR-Observed": ("posterior", True),
    "PASR-Always4": ("posterior", False),
    "PASR-OraclePower": ("posterior_oracle_power", True),
    "Fisher-Only-Request": ("fisher", True),
    "HardTop1-Request": ("top1", True),
    "Random-Request": ("random", True),
    "Oracle-Next-Bin": ("oracle", True),
    "Fixed-Nested": ("fixed", True),
}

STATIC_METHOD_PREFIXES = (
    "Exhaustive-Proxy",
    "PFRS",
    "Zhai-Fisher",
    "Power",
    "Uniform",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tau_grid(config: ExperimentConfig) -> np.ndarray:
    width = config.tau_upper_samples - config.tau_lower_samples
    intervals = int(round(width / config.grid_step_samples))
    if not math.isclose(
        intervals * config.grid_step_samples, width, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("TDOA range is not divisible by the configured grid step")
    return np.linspace(config.tau_lower_samples, config.tau_upper_samples, intervals + 1)


def _complex_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / math.sqrt(2.0)


def _positions_from_bins(bins: Iterable[int], model: SpectralModel) -> tuple[int, ...]:
    lookup = {int(value): index for index, value in enumerate(model.bin_indices)}
    try:
        return tuple(sorted(lookup[int(value)] for value in bins))
    except KeyError as exc:
        raise ValueError(f"Unknown spectral bin: {exc}") from exc


def _continuous_period_samples(
    positions: tuple[int, ...], model: SpectralModel
) -> float:
    selected = model.bin_indices[np.asarray(positions, dtype=int)]
    differences = [abs(int(value - selected[0])) for value in selected[1:]]
    nonzero = [value for value in differences if value]
    if not nonzero:
        return 0.0
    return float(model.bin_indices.size / reduce(math.gcd, nonzero))


def _design_proxy(
    positions: tuple[int, ...], model: SpectralModel, prior_width: float
) -> tuple[float, float, float]:
    h = np.linspace(0.25, prior_width - 0.25, 31)
    triangular = h * (prior_width - h)
    triangular /= np.sum(triangular)
    risks = []
    for snr_db in (-5.0, 0.0, 5.0):
        variance = 10.0 ** (-snr_db / 10.0)
        distances = np.asarray(
            [
                profiled_mean_distance_from_power(
                    positions,
                    model.power,
                    model,
                    float(delay),
                    variance,
                    variance,
                    rho_abs=abs(model.rho),
                )
                for delay in h
            ]
        )
        # This monotone Chernoff-like proxy is used only for design ranking.
        risks.append(float(np.sum(triangular * 0.5 * np.exp(-0.25 * distances))))
    reference_variance = 0.1
    fisher = fisher_information_from_power(
        positions,
        model.power,
        model,
        reference_variance,
        reference_variance,
        rho_abs=abs(model.rho),
    )
    return max(risks), float(np.mean(risks)), fisher


def _design_static_masks(
    model: SpectralModel, config: ExperimentConfig
) -> tuple[pd.DataFrame, dict[str, tuple[int, ...]], tuple[int, ...]]:
    prior_width = config.tau_upper_samples - config.tau_lower_samples
    rows = []
    best: dict[int, tuple[float, float, float, tuple[int, ...]]] = {}
    for m_bins in (2, 3, 4):
        for positions in itertools.combinations(range(model.bin_indices.size), m_bins):
            period = _continuous_period_samples(positions, model)
            if period <= prior_width:
                continue
            robust_risk, mean_risk, fisher = _design_proxy(positions, model, prior_width)
            row = {
                "m_bins": m_bins,
                "positions": " ".join(str(value) for value in positions),
                "mask": " ".join(str(int(model.bin_indices[value])) for value in positions),
                "continuous_period_samples": period,
                "robust_design_risk": robust_risk,
                "mean_design_risk": mean_risk,
                "fisher_information_reference": fisher,
            }
            rows.append(row)
            key = (robust_risk, mean_risk, -fisher, positions)
            if m_bins not in best or key < best[m_bins]:
                best[m_bins] = key
    if set(best) != {2, 3, 4}:
        raise RuntimeError("Could not design identifiable static masks for M=2,3,4")

    base = best[2][3]
    containing_base = [
        row
        for row in rows
        if row["m_bins"] == 4
        and set(base).issubset(int(value) for value in str(row["positions"]).split())
    ]
    if not containing_base:
        raise RuntimeError("No four-bin static extension contains the base layer")
    nested_row = min(
        containing_base,
        key=lambda row: (
            row["robust_design_risk"],
            row["mean_design_risk"],
            -row["fisher_information_reference"],
        ),
    )
    nested = tuple(int(value) for value in str(nested_row["positions"]).split())

    # All identifiable masks are enumerated, but the winner is selected by the
    # deterministic Chernoff/Fisher design proxy rather than Monte Carlo risk.
    # The method name states that limitation explicitly.
    methods = {
        f"Exhaustive-Proxy-M{m_bins}": best[m_bins][3] for m_bins in (2, 3, 4)
    }
    methods["Fixed-Nested-Final"] = nested
    methods.update(
        {
            method: _positions_from_bins(bins, model)
            for method, bins in KNOWN_MASK_BINS.items()
        }
    )
    frame = pd.DataFrame(rows)
    selected_keys = {
        " ".join(str(value) for value in positions) for positions in methods.values()
    }
    frame["selected_as_baseline"] = frame["positions"].isin(selected_keys)
    return frame, methods, base


def _fixed_refinement_order(
    base: tuple[int, ...], final_mask: tuple[int, ...]
) -> tuple[int, ...]:
    missing = [value for value in final_mask if value not in base]
    if len(missing) != 2:
        raise RuntimeError("Fixed nested final mask must add exactly two bins")
    return tuple(missing)


def _method_row(
    *,
    stage: str,
    seed: int,
    snr_db: float,
    trial: int,
    qbits: int,
    method: str,
    true_tau: float,
    estimate: float,
    transmitted_bits: int,
    transmitted_bins: int,
    max_total_bits: int,
    stop_reason: str,
    mode_probability: float,
    mode_variance: float,
    clipping_rate: float,
    coefficient_bits: int,
    scale_bits: int,
    action_feedback_bits: int,
    decision_feedback_bits: int,
) -> dict[str, object]:
    error = estimate - true_tau
    return {
        "stage": stage,
        "seed": seed,
        "snr_db": snr_db,
        "trial": trial,
        "qbits": qbits,
        "method": method,
        "true_tau_samples": true_tau,
        "estimate_tau_samples": estimate,
        "error_samples": error,
        "abs_error_samples": abs(error),
        "squared_error_samples2": error**2,
        "transmitted_bits": transmitted_bits,
        "coefficient_bits": coefficient_bits,
        "scale_bits": scale_bits,
        "action_feedback_bits": action_feedback_bits,
        "decision_feedback_bits": decision_feedback_bits,
        "max_total_bits": max_total_bits,
        "transmitted_complex_bins": transmitted_bins,
        "stop_reason": stop_reason,
        "dominant_mode_probability": mode_probability,
        "dominant_mode_variance": mode_variance,
        "quantizer_clipping_rate": clipping_rate,
    }


def _active_max_bits(
    base_count: int,
    max_bins: int,
    qbits: int,
    scale_bits: int,
    n_bins: int,
    *,
    requires_action_index: bool,
    allows_early_stop: bool,
) -> int:
    total = base_count * 2 * qbits + scale_bits
    remaining = n_bins - base_count
    for _ in range(max_bins - base_count):
        total += int(allows_early_stop) + 2 * qbits
        if requires_action_index:
            total += max(1, math.ceil(math.log2(remaining)))
        remaining -= 1
    return total


def _append_active_row(
    rows: list[dict[str, object]],
    trace_rows: list[dict[str, object]],
    result: ActiveEstimate,
    *,
    config: ExperimentConfig,
    model: SpectralModel,
    stage: str,
    seed: int,
    snr_db: float,
    trial: int,
    qbits: int,
    method: str,
    true_tau: float,
    clipping_rate: float,
    base_count: int,
) -> None:
    dominant = result.posterior.modes[0]
    strategy, allows_early_stop = ACTIVE_METHODS[method]
    rows.append(
        _method_row(
            stage=stage,
            seed=seed,
            snr_db=snr_db,
            trial=trial,
            qbits=qbits,
            method=method,
            true_tau=true_tau,
            estimate=result.estimate,
            transmitted_bits=result.ledger.total_bits,
            transmitted_bins=len(result.selected_positions),
            max_total_bits=_active_max_bits(
                base_count,
                config.max_bins,
                qbits,
                config.scale_bits,
                model.bin_indices.size,
                requires_action_index=strategy != "fixed",
                allows_early_stop=allows_early_stop,
            ),
            stop_reason=result.stop_reason,
            mode_probability=dominant.probability,
            mode_variance=dominant.variance,
            clipping_rate=clipping_rate,
            coefficient_bits=result.ledger.coefficient_bits,
            scale_bits=result.ledger.scale_bits,
            action_feedback_bits=result.ledger.action_feedback_bits,
            decision_feedback_bits=result.ledger.decision_feedback_bits,
        )
    )
    if method not in {"PASR-Observed", "Oracle-Next-Bin"}:
        return
    for action in result.action_trace:
        trace_rows.append(
            {
                "stage": stage,
                "seed": seed,
                "snr_db": snr_db,
                "trial": trial,
                "qbits": qbits,
                "method": method,
                **action,
            }
        )


def _run_stage(
    stage: str,
    seeds: tuple[int, ...],
    n_trials: int,
    config: ExperimentConfig,
    model: SpectralModel,
    tau_grid: np.ndarray,
    static_methods: dict[str, tuple[int, ...]],
    base_positions: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    fixed_final = static_methods["Fixed-Nested-Final"]
    fixed_order = _fixed_refinement_order(base_positions, fixed_final)
    reference_power = float(np.mean(model.power))
    total_conditions = len(seeds) * len(config.snr_db)
    condition_index = 0
    started = time.perf_counter()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        true_tau = rng.uniform(
            config.tau_lower_samples, config.tau_upper_samples, size=n_trials
        )
        standard_a = _complex_noise(rng, (n_trials, model.bin_indices.size))
        standard_b = _complex_noise(rng, (n_trials, model.bin_indices.size))
        for snr_index, snr_db in enumerate(config.snr_db):
            condition_index += 1
            variance = reference_power / (10.0 ** (snr_db / 10.0))
            y_a = model.source[None, :] + math.sqrt(variance) * standard_a
            delayed = model.source[None, :] * np.exp(
                -1j * true_tau[:, None] * model.omega[None, :]
            )
            y_b = model.rho * delayed + math.sqrt(variance) * standard_b
            print(
                f"[{stage}] seed={seed} SNR={snr_db:g} dB "
                f"({condition_index}/{total_conditions})",
                flush=True,
            )

            for trial in range(n_trials):
                # Full16 is an unquantized reference and is recorded once per trial.
                full = estimate_subset(
                    y_a[trial],
                    y_b[trial],
                    tuple(range(model.bin_indices.size)),
                    model,
                    tau_grid,
                    variance,
                    variance,
                    max_modes=config.max_modes,
                    min_mode_separation_samples=config.min_mode_separation_samples,
                )
                rows.append(
                    _method_row(
                        stage=stage,
                        seed=seed,
                        snr_db=snr_db,
                        trial=trial,
                        qbits=32,
                        method="Full16-Uncompressed",
                        true_tau=float(true_tau[trial]),
                        estimate=full.estimate,
                        transmitted_bits=16 * 2 * 32,
                        transmitted_bins=16,
                        max_total_bits=16 * 2 * 32,
                        stop_reason="fixed_full",
                        mode_probability=full.modes[0].probability,
                        mode_variance=full.modes[0].variance,
                        clipping_rate=0.0,
                        coefficient_bits=16 * 2 * 32,
                        scale_bits=0,
                        action_feedback_bits=0,
                        decision_feedback_bits=0,
                    )
                )

                all_qbits = tuple(
                    sorted(set(config.static_qbits_values) | set(config.qbits_values))
                )
                for qbits in all_qbits:
                    quantizer_seed = (
                        seed * 1_000_003 + snr_index * 10_007 + trial * 101 + qbits
                    )
                    q_rng = np.random.default_rng(quantizer_seed)
                    quantized = quantize_complex_spectrum(
                        y_b[trial], qbits, config.scale_bits, q_rng
                    )
                    effective_b_variance = variance + quantized.complex_error_variance

                    for method, positions in static_methods.items():
                        if method == "Fixed-Nested-Final":
                            continue
                        result = estimate_subset(
                            y_a[trial],
                            quantized.values,
                            positions,
                            model,
                            tau_grid,
                            variance,
                            effective_b_variance,
                            max_modes=config.max_modes,
                            min_mode_separation_samples=(
                                config.min_mode_separation_samples
                            ),
                        )
                        rows.append(
                            _method_row(
                                stage=stage,
                                seed=seed,
                                snr_db=snr_db,
                                trial=trial,
                                qbits=qbits,
                                method=method,
                                true_tau=float(true_tau[trial]),
                                estimate=result.estimate,
                                transmitted_bits=(
                                    len(positions) * 2 * qbits + config.scale_bits
                                ),
                                transmitted_bins=len(positions),
                                max_total_bits=(
                                    len(positions) * 2 * qbits + config.scale_bits
                                ),
                                stop_reason="fixed_mask",
                                mode_probability=result.modes[0].probability,
                                mode_variance=result.modes[0].variance,
                                clipping_rate=quantized.clipping_rate,
                                coefficient_bits=len(positions) * 2 * qbits,
                                scale_bits=config.scale_bits,
                                action_feedback_bits=0,
                                decision_feedback_bits=0,
                            )
                        )

                    if qbits not in config.qbits_values:
                        continue
                    for method, (strategy, allow_early_stop) in ACTIVE_METHODS.items():
                        strategy_seed = quantizer_seed + 997 * (
                            list(ACTIVE_METHODS).index(method) + 1
                        )
                        result = run_active_refinement(
                            y_a=y_a[trial],
                            quantized_b=quantized,
                            base_positions=base_positions,
                            model=model,
                            tau_grid=tau_grid,
                            sigma_a2=variance,
                            sigma_b2=variance,
                            qbits=qbits,
                            scale_bits=config.scale_bits,
                            max_bins=config.max_bins,
                            confidence_threshold=config.confidence_threshold,
                            variance_target=config.variance_target,
                            strategy=strategy,  # type: ignore[arg-type]
                            rng=np.random.default_rng(strategy_seed),
                            fixed_order=fixed_order,
                            true_tau=float(true_tau[trial]),
                            max_modes=config.max_modes,
                            min_mode_separation_samples=(
                                config.min_mode_separation_samples
                            ),
                            allow_early_stop=allow_early_stop,
                        )
                        _append_active_row(
                            rows,
                            trace_rows,
                            result,
                            config=config,
                            model=model,
                            stage=stage,
                            seed=seed,
                            snr_db=snr_db,
                            trial=trial,
                            qbits=qbits,
                            method=method,
                            true_tau=float(true_tau[trial]),
                            clipping_rate=quantized.clipping_rate,
                            base_count=len(base_positions),
                        )

                if (trial + 1) % config.progress_every == 0 or trial + 1 == n_trials:
                    elapsed = time.perf_counter() - started
                    print(
                        f"  trial {trial + 1}/{n_trials} | elapsed={elapsed:.1f}s",
                        flush=True,
                    )
    return pd.DataFrame(rows), pd.DataFrame(trace_rows)


def _metric_row(group: pd.DataFrame) -> pd.Series:
    error = group["error_samples"].to_numpy(dtype=float)
    absolute = np.abs(error)
    return pd.Series(
        {
            "n": len(group),
            "rmse_samples": math.sqrt(float(np.mean(error**2))),
            "mae_samples": float(np.mean(absolute)),
            "median_abs_error_samples": float(np.median(absolute)),
            "p95_abs_error_samples": float(np.quantile(absolute, 0.95)),
            "gross_error_rate_gt_1": float(np.mean(absolute > 1.0)),
            "mean_total_bits": float(group["transmitted_bits"].mean()),
            "max_total_bits": int(group["max_total_bits"].max()),
            "mean_complex_bins": float(group["transmitted_complex_bins"].mean()),
            "posterior_stop_rate": float(
                np.mean(group["stop_reason"] == "posterior_confident")
            ),
            "mean_quantizer_clipping_rate": float(
                group["quantizer_clipping_rate"].mean()
            ),
        }
    )


def _summaries(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_snr = (
        rows.groupby(["stage", "method", "qbits", "snr_db"], sort=False)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
    pooled = (
        rows.groupby(["stage", "method", "qbits"], sort=False)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
    return by_snr, pooled


def _validate_trial_rows(
    rows: pd.DataFrame, config: ExperimentConfig
) -> dict[str, object]:
    key = ["stage", "seed", "snr_db", "trial", "qbits", "method"]
    if rows.duplicated(key).any():
        duplicate = rows.loc[rows.duplicated(key, keep=False), key].head(5)
        raise RuntimeError(f"Duplicate trial/method keys detected:\n{duplicate}")
    numeric = [
        "true_tau_samples",
        "estimate_tau_samples",
        "error_samples",
        "squared_error_samples2",
        "transmitted_bits",
        "max_total_bits",
        "quantizer_clipping_rate",
    ]
    if not np.all(np.isfinite(rows[numeric].to_numpy(dtype=float))):
        raise RuntimeError("Trial results contain non-finite numeric values")
    component_sum = rows[
        [
            "coefficient_bits",
            "scale_bits",
            "action_feedback_bits",
            "decision_feedback_bits",
        ]
    ].sum(axis=1)
    if not np.array_equal(component_sum.to_numpy(), rows["transmitted_bits"].to_numpy()):
        raise RuntimeError("Communication bit ledger does not sum to transmitted_bits")
    if (rows["transmitted_bits"] > rows["max_total_bits"]).any():
        raise RuntimeError("Observed communication exceeds the declared maximum budget")
    if not rows["quantizer_clipping_rate"].between(0.0, 1.0).all():
        raise RuntimeError("Quantizer clipping rate lies outside [0, 1]")
    if set(config.development_seeds) & set(config.locked_seeds):
        raise RuntimeError("Development and locked seed sets must be disjoint")
    tau_per_trial = rows.groupby(["stage", "seed", "snr_db", "trial"])[
        "true_tau_samples"
    ].nunique()
    if int(tau_per_trial.max()) != 1:
        raise RuntimeError("Methods do not share the same true TDOA per trial")
    tau_across_snr = rows.groupby(["stage", "seed", "trial"])[
        "true_tau_samples"
    ].nunique()
    if int(tau_across_snr.max()) != 1:
        raise RuntimeError("Common-random-number TDOA pairing across SNR is broken")

    expected_by_stage = {
        "development": len(config.development_seeds)
        * len(config.snr_db)
        * config.development_trials,
        "locked": len(config.locked_seeds) * len(config.snr_db) * config.locked_trials,
    }
    for stage, expected in expected_by_stage.items():
        full_count = len(
            rows[(rows["stage"] == stage) & (rows["method"] == "Full16-Uncompressed")]
        )
        if full_count != expected:
            raise RuntimeError(
                f"{stage} Full16 row count {full_count} != expected {expected}"
            )
        for method in ACTIVE_METHODS:
            for qbits in config.qbits_values:
                count = len(
                    rows[
                        (rows["stage"] == stage)
                        & (rows["method"] == method)
                        & (rows["qbits"] == qbits)
                    ]
                )
                if count != expected:
                    raise RuntimeError(
                        f"{stage} {method}/{qbits}-bit row count {count} != {expected}"
                    )
    return {
        "trial_method_keys_unique": True,
        "all_required_numeric_values_finite": True,
        "bit_component_identity_pass": True,
        "observed_bits_within_declared_maximum": True,
        "development_locked_seeds_disjoint": True,
        "same_true_tdoa_shared_by_all_methods": True,
        "same_true_tdoa_reused_across_snr": True,
        "required_active_method_rows_complete": True,
        "status": "PASS",
    }


def _method_metadata(
    static_methods: dict[str, tuple[int, ...]],
    model: SpectralModel,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows = []
    for method, positions in static_methods.items():
        if method == "Fixed-Nested-Final":
            continue
        ambiguity_period = _continuous_period_samples(positions, model)
        rows.append(
            {
                "method": method,
                "category": "fixed_spectral_mask",
                "adaptive_request": False,
                "uses_true_tdoa": False,
                "uses_true_source_power_at_inference": False,
                "mask_design_uses_reference_source_power": method != "Uniform",
                "allows_early_stop": False,
                "requires_feedback": False,
                "requires_shared_dither_seed": True,
                "ambiguity_period_samples": ambiguity_period,
                "globally_identifiable_over_prior": (
                    ambiguity_period
                    > config.tau_upper_samples - config.tau_lower_samples
                ),
                "mask": " ".join(
                    str(int(model.bin_indices[position])) for position in positions
                ),
                "role": (
                    "proxy_selected_static_candidate"
                    if method.startswith("Exhaustive-Proxy")
                    else "published_or_frozen_baseline"
                ),
                "selection_basis": (
                    "all_identifiable_masks_enumerated_then_ranked_by_design_proxy"
                    if method.startswith("Exhaustive-Proxy")
                    else "published_or_previously_frozen_mask"
                ),
            }
        )
    active_roles = {
        "PASR-Observed": "primary_candidate",
        "PASR-Always4": "stopping_ablation",
        "PASR-OraclePower": "source_power_oracle_ablation",
        "Fisher-Only-Request": "local_information_ablation",
        "HardTop1-Request": "posterior_mode_ablation",
        "Random-Request": "random_action_control",
        "Oracle-Next-Bin": "clairvoyant_action_upper_bound",
        "Fixed-Nested": "fixed_order_control",
    }
    for method, (strategy, early_stop) in ACTIVE_METHODS.items():
        rows.append(
            {
                "method": method,
                "category": "active_successive_refinement",
                "adaptive_request": strategy not in {"fixed", "random"},
                "uses_true_tdoa": strategy == "oracle",
                "uses_true_source_power_at_inference": (
                    strategy == "posterior_oracle_power"
                ),
                "mask_design_uses_reference_source_power": False,
                "allows_early_stop": early_stop,
                "requires_feedback": True,
                "requires_shared_dither_seed": True,
                "ambiguity_period_samples": math.nan,
                "globally_identifiable_over_prior": "sample_dependent",
                "mask": "sample_dependent",
                "role": active_roles[method],
                "selection_basis": "sample_dependent_or_control_policy",
            }
        )
    rows.append(
        {
            "method": "Full16-Uncompressed",
            "category": "uncompressed_reference",
            "adaptive_request": False,
            "uses_true_tdoa": False,
            "uses_true_source_power_at_inference": False,
            "mask_design_uses_reference_source_power": False,
            "allows_early_stop": False,
            "requires_feedback": False,
            "requires_shared_dither_seed": False,
            "ambiguity_period_samples": math.inf,
            "globally_identifiable_over_prior": True,
            "mask": "all_16_bins",
            "role": "reference_only_not_same_budget",
            "selection_basis": "all_bins",
        }
    )
    return pd.DataFrame(rows)


def _static_plan(
    left: pd.Series,
    right: pd.Series,
    right_weight: float,
    *,
    purpose: str,
) -> dict[str, object]:
    weight = float(np.clip(right_weight, 0.0, 1.0))
    left_mse = float(left["rmse_samples"] ** 2)
    right_mse = float(right["rmse_samples"] ** 2)
    return {
        "purpose": purpose,
        "selection_stage": "development",
        "left_method": str(left["method"]),
        "left_qbits": int(left["qbits"]),
        "left_bits": float(left["mean_total_bits"]),
        "left_development_mse": left_mse,
        "right_method": str(right["method"]),
        "right_qbits": int(right["qbits"]),
        "right_bits": float(right["mean_total_bits"]),
        "right_development_mse": right_mse,
        "right_weight": weight,
        "development_expected_bits": (
            (1.0 - weight) * float(left["mean_total_bits"])
            + weight * float(right["mean_total_bits"])
        ),
        "development_expected_mse": (1.0 - weight) * left_mse + weight * right_mse,
        "max_total_bits": int(max(left["max_total_bits"], right["max_total_bits"])),
    }


def _same_bit_static_plan(
    static_development: pd.DataFrame, target_bits: float
) -> dict[str, object] | None:
    """Freeze the best static time-sharing comparator on development data."""
    candidates: list[dict[str, object]] = []
    points = [row for _, row in static_development.iterrows()]
    for index, left in enumerate(points):
        left_bits = float(left["mean_total_bits"])
        if math.isclose(left_bits, target_bits, abs_tol=1e-12):
            candidates.append(_static_plan(left, left, 0.0, purpose="same_mean_bits"))
        for right in points[index + 1 :]:
            right_bits = float(right["mean_total_bits"])
            if math.isclose(left_bits, right_bits, abs_tol=1e-12):
                continue
            if min(left_bits, right_bits) <= target_bits <= max(left_bits, right_bits):
                weight = (target_bits - left_bits) / (right_bits - left_bits)
                candidates.append(
                    _static_plan(left, right, weight, purpose="same_mean_bits")
                )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda plan: (
            float(plan["development_expected_mse"]),
            int(plan["max_total_bits"]),
            str(plan["left_method"]),
            str(plan["right_method"]),
        ),
    )


def _same_risk_static_plan(
    static_development: pd.DataFrame, target_mse: float
) -> dict[str, object] | None:
    """Freeze the least-bit static mixture reaching PASR development risk."""
    candidates: list[dict[str, object]] = []
    points = [row for _, row in static_development.iterrows()]
    for index, left in enumerate(points):
        left_mse = float(left["rmse_samples"] ** 2)
        if left_mse <= target_mse:
            candidates.append(_static_plan(left, left, 0.0, purpose="same_mse"))
        for right in points[index + 1 :]:
            right_mse = float(right["rmse_samples"] ** 2)
            if math.isclose(left_mse, right_mse, abs_tol=1e-15):
                continue
            if min(left_mse, right_mse) <= target_mse <= max(left_mse, right_mse):
                weight = (target_mse - left_mse) / (right_mse - left_mse)
                candidates.append(_static_plan(left, right, weight, purpose="same_mse"))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda plan: (
            float(plan["development_expected_bits"]),
            float(plan["development_expected_mse"]),
            str(plan["left_method"]),
            str(plan["right_method"]),
        ),
    )


def _static_plan_rows(
    rows: pd.DataFrame, stage: str, plan: dict[str, object]
) -> pd.DataFrame:
    keys = ["seed", "snr_db", "trial"]

    def select(side: str) -> pd.DataFrame:
        method = str(plan[f"{side}_method"])
        qbits = int(plan[f"{side}_qbits"])
        selected = rows[
            (rows["stage"] == stage)
            & (rows["method"] == method)
            & (rows["qbits"] == qbits)
        ][keys + ["squared_error_samples2", "abs_error_samples", "transmitted_bits"]]
        if selected.empty:
            raise RuntimeError(
                f"Frozen comparator {method}/{qbits}-bit is absent from {stage} rows"
            )
        return selected.rename(
            columns={
                "squared_error_samples2": f"{side}_squared_error",
                "abs_error_samples": f"{side}_abs_error",
                "transmitted_bits": f"{side}_bits",
            }
        )

    left = select("left")
    same_point = (
        plan["left_method"] == plan["right_method"]
        and int(plan["left_qbits"]) == int(plan["right_qbits"])
    )
    if same_point:
        paired = left.copy()
        paired["right_squared_error"] = paired["left_squared_error"]
        paired["right_abs_error"] = paired["left_abs_error"]
        paired["right_bits"] = paired["left_bits"]
    else:
        paired = left.merge(select("right"), on=keys, validate="one_to_one")
    weight = float(plan["right_weight"])
    paired["expected_squared_error"] = (
        (1.0 - weight) * paired["left_squared_error"]
        + weight * paired["right_squared_error"]
    )
    paired["expected_gross_error"] = (
        (1.0 - weight) * (paired["left_abs_error"] > 1.0).astype(float)
        + weight * (paired["right_abs_error"] > 1.0).astype(float)
    )
    paired["expected_bits"] = (
        (1.0 - weight) * paired["left_bits"] + weight * paired["right_bits"]
    )
    return paired


def _evaluate_static_plan(
    rows: pd.DataFrame, stage: str, plan: dict[str, object]
) -> dict[str, float | int]:
    paired = _static_plan_rows(rows, stage, plan)
    return {
        "n": int(len(paired)),
        "rmse_samples": math.sqrt(float(paired["expected_squared_error"].mean())),
        "mse_samples2": float(paired["expected_squared_error"].mean()),
        "gross_error_rate_gt_1": float(paired["expected_gross_error"].mean()),
        "mean_total_bits": float(paired["expected_bits"].mean()),
        "max_total_bits": int(plan["max_total_bits"]),
    }


def _paired_plan_bootstrap(
    rows: pd.DataFrame,
    qbits: int,
    plan: dict[str, object],
    repetitions: int,
    seed: int,
) -> dict[str, float | int | str]:
    static = _static_plan_rows(rows, "locked", plan)
    active = rows[
        (rows["stage"] == "locked")
        & (rows["method"] == "PASR-Observed")
        & (rows["qbits"] == qbits)
    ][["seed", "snr_db", "trial", "squared_error_samples2"]]
    paired = active.merge(
        static[["seed", "snr_db", "trial", "expected_squared_error"]],
        on=["seed", "snr_db", "trial"],
        validate="one_to_one",
    )
    if paired.empty or len(paired) != len(active):
        raise RuntimeError("Frozen static plan does not align one-to-one with PASR rows")
    paired["improvement"] = (
        paired["expected_squared_error"] - paired["squared_error_samples2"]
    )
    # A trial reuses its TDOA/noise direction across SNR and is therefore one
    # paired cluster. Different trials within each RNG stream are independent.
    improvement = (
        paired.groupby(["seed", "trial"])["improvement"].mean().to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, improvement.size, size=(repetitions, improvement.size))
    samples = np.mean(improvement[draw], axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "n_locked_trial_clusters": int(improvement.size),
        "paired_mse_improvement": float(np.mean(improvement)),
        "paired_mse_improvement_ci_low": float(low),
        "paired_mse_improvement_ci_high": float(high),
        "status": "PASS" if low > 0.0 else "FAIL",
    }


def _gate(
    rows: pd.DataFrame,
    pooled: pd.DataFrame,
    by_snr: pd.DataFrame,
    config: ExperimentConfig,
    protocol_validation: dict[str, object],
) -> dict[str, object]:
    if config.primary_qbits not in config.qbits_values:
        raise RuntimeError("primary_qbits must be included in qbits_values")
    development = pooled[pooled["stage"] == "development"]
    locked = pooled[pooled["stage"] == "locked"]
    outcomes = []
    for qbits in config.qbits_values:
        active_development = development[
            (development["method"] == "PASR-Observed")
            & (development["qbits"] == qbits)
        ]
        active_locked = locked[
            (locked["method"] == "PASR-Observed") & (locked["qbits"] == qbits)
        ]
        if active_development.empty or active_locked.empty:
            continue
        active_dev = active_development.iloc[0]
        active_row = active_locked.iloc[0]
        static_dev = development[
            development["method"].str.startswith(STATIC_METHOD_PREFIXES)
            & (development["max_total_bits"] <= active_dev["max_total_bits"])
        ]
        if static_dev.empty:
            raise RuntimeError(f"No legal development static comparator for qbits={qbits}")

        same_bit_plan = _same_bit_static_plan(
            static_dev, float(active_dev["mean_total_bits"])
        )
        same_risk_plan = _same_risk_static_plan(
            static_dev, float(active_dev["rmse_samples"] ** 2)
        )
        if same_bit_plan is None:
            raise RuntimeError(f"Static frontier cannot match PASR mean bits for qbits={qbits}")
        same_bit_locked = _evaluate_static_plan(rows, "locked", same_bit_plan)
        paired = _paired_plan_bootstrap(
            rows,
            qbits,
            same_bit_plan,
            config.bootstrap_repetitions,
            config.bootstrap_seed + qbits,
        )

        budget_relative_error = abs(
            float(same_bit_locked["mean_total_bits"])
            - float(active_row["mean_total_bits"])
        ) / max(float(active_row["mean_total_bits"]), 1e-12)
        budget_match_pass = (
            budget_relative_error <= config.budget_match_relative_tolerance
            and int(same_bit_locked["max_total_bits"])
            <= int(active_row["max_total_bits"])
        )
        same_bit_pass = (
            budget_match_pass
            and float(active_row["rmse_samples"] ** 2)
            < float(same_bit_locked["mse_samples2"])
        )
        static_gross = float(same_bit_locked["gross_error_rate_gt_1"])
        gross_relative = (
            (static_gross - float(active_row["gross_error_rate_gt_1"]))
            / max(static_gross, 1e-12)
        )

        if same_risk_plan is None:
            same_risk_locked = None
            bit_saving_fraction = math.nan
            risk_match_pass = False
        else:
            same_risk_locked = _evaluate_static_plan(rows, "locked", same_risk_plan)
            static_bits = float(same_risk_locked["mean_total_bits"])
            bit_saving_fraction = (
                1.0 - float(active_row["mean_total_bits"]) / static_bits
                if static_bits > 0.0
                else math.nan
            )
            risk_match_pass = (
                float(same_risk_locked["mse_samples2"])
                <= 1.05 * float(active_row["rmse_samples"] ** 2)
                and int(same_risk_locked["max_total_bits"])
                <= int(active_row["max_total_bits"])
            )
        bit_saving_pass = (
            risk_match_pass
            and math.isfinite(bit_saving_fraction)
            and bit_saving_fraction >= 0.20
        )

        high_snr = max(config.snr_db)
        dev_high_static = by_snr[
            (by_snr["stage"] == "development")
            & (by_snr["snr_db"] == high_snr)
            & by_snr["method"].str.startswith(STATIC_METHOD_PREFIXES)
            & (by_snr["max_total_bits"] <= active_dev["max_total_bits"])
        ]
        if dev_high_static.empty:
            raise RuntimeError("No development high-SNR static reference")
        high_reference_dev = dev_high_static.sort_values(
            ["rmse_samples", "mean_total_bits", "method", "qbits"]
        ).iloc[0]
        high_reference_locked = by_snr[
            (by_snr["stage"] == "locked")
            & (by_snr["snr_db"] == high_snr)
            & (by_snr["method"] == high_reference_dev["method"])
            & (by_snr["qbits"] == high_reference_dev["qbits"])
        ]
        active_high = by_snr[
            (by_snr["stage"] == "locked")
            & (by_snr["method"] == "PASR-Observed")
            & (by_snr["qbits"] == qbits)
            & (by_snr["snr_db"] == high_snr)
        ]
        if len(high_reference_locked) != 1 or len(active_high) != 1:
            raise RuntimeError("Frozen high-SNR reference is not uniquely available")
        reference_rmse = float(high_reference_locked.iloc[0]["rmse_samples"])
        active_high_rmse = float(active_high.iloc[0]["rmse_samples"])
        high_ratio = (
            active_high_rmse / reference_rmse
            if reference_rmse > 0.0
            else (1.0 if active_high_rmse == 0.0 else math.inf)
        )

        gross_pass = gross_relative >= 0.15
        local_pass = high_ratio <= 1.05
        ci_pass = paired.get("status") == "PASS"
        outcomes.append(
            {
                "qbits": qbits,
                "analysis_role": (
                    "primary_pre_registered_test"
                    if qbits == config.primary_qbits
                    else "secondary_sensitivity_analysis"
                ),
                "pasr_development_mean_bits": float(active_dev["mean_total_bits"]),
                "pasr_locked_mean_bits": float(active_row["mean_total_bits"]),
                "pasr_max_bits": int(active_row["max_total_bits"]),
                "pasr_locked_rmse_samples": float(active_row["rmse_samples"]),
                "same_bit_static_plan": same_bit_plan,
                "same_bit_static_locked_evaluation": same_bit_locked,
                "same_risk_static_plan": same_risk_plan,
                "same_risk_static_locked_evaluation": same_risk_locked,
                "budget_relative_error": budget_relative_error,
                "bit_saving_fraction_at_same_risk": (
                    bit_saving_fraction if math.isfinite(bit_saving_fraction) else None
                ),
                "gross_error_relative_improvement": gross_relative,
                "high_snr_reference_selected_on_development": {
                    "method": str(high_reference_dev["method"]),
                    "qbits": int(high_reference_dev["qbits"]),
                },
                "high_snr_rmse_ratio_vs_frozen_static": high_ratio,
                "budget_match_pass": budget_match_pass,
                "same_bit_envelope_pass": same_bit_pass,
                "same_risk_validation_pass": risk_match_pass,
                "bit_saving_20_percent_pass": bit_saving_pass,
                "gross_error_pass": gross_pass,
                "high_snr_local_precision_pass": local_pass,
                "paired_ci_pass": ci_pass,
                "paired_bootstrap": paired,
                "status": (
                    "GO"
                    if same_bit_pass
                    and bit_saving_pass
                    and gross_pass
                    and local_pass
                    and ci_pass
                    else "NO_GO"
                ),
            }
        )
    primary = [item for item in outcomes if item["qbits"] == config.primary_qbits]
    if len(primary) != 1:
        raise RuntimeError("Primary qbits outcome is not uniquely available")
    return {
        "scope": (
            "N=16 asymmetric passive two-receiver model with finite-bit B-side "
            "spectral transmission; this is not an urban8 or localization result."
        ),
        "selection_protocol": (
            "Static time-sharing plans and the high-SNR reference are selected only "
            "on development seeds, then frozen before locked evaluation."
        ),
        "static_pool_limitation": (
            "The candidate pool contains deterministic proxy-selected masks and "
            "published/frozen masks; it is not an empirical exhaustive optimum over "
            "all masks."
        ),
        "protocol_validation": protocol_validation,
        "qbits_outcomes": outcomes,
        "overall_status": (
            "SMOKE_ONLY"
            if config.mode == "smoke"
            else str(primary[0]["status"])
        ),
    }


def _frozen_comparator_table(gate: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in gate["qbits_outcomes"]:  # type: ignore[index]
        qbits = int(outcome["qbits"])
        for field in ("same_bit_static_plan", "same_risk_static_plan"):
            plan = outcome.get(field)
            if plan is not None:
                rows.append({"pasr_qbits": qbits, "comparator": field, **plan})
        high = outcome["high_snr_reference_selected_on_development"]
        rows.append(
            {
                "pasr_qbits": qbits,
                "comparator": "high_snr_static_reference",
                "purpose": "high_snr_local_precision",
                "selection_stage": "development",
                "left_method": high["method"],
                "left_qbits": high["qbits"],
                "right_method": high["method"],
                "right_qbits": high["qbits"],
                "right_weight": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _protocol_validation(
    config: ExperimentConfig,
    model: SpectralModel,
    tau_grid: np.ndarray,
    base: tuple[int, ...],
    nested: tuple[int, ...],
) -> dict[str, object]:
    true_tau = 1.317
    y_a = model.source.copy()
    y_b = model.rho * model.source * np.exp(-1j * model.omega * true_tau)
    max_error = 0.0
    for positions in (base, nested, tuple(range(model.bin_indices.size))):
        result = estimate_subset(y_a, y_b, positions, model, tau_grid, 1.0, 1.0)
        max_error = max(max_error, abs(result.estimate - true_tau))
    one_shot = estimate_subset(y_a, y_b, nested, model, tau_grid, 1.0, 1.0)
    reordered = estimate_subset(
        y_a, y_b, tuple(reversed(nested)), model, tau_grid, 1.0, 1.0
    )
    posterior_difference = float(np.max(np.abs(one_shot.posterior - reordered.posterior)))
    if max_error > config.grid_step_samples or posterior_difference > 1e-12:
        raise RuntimeError("PASR protocol mathematical invariants failed")
    return {
        "noiseless_max_abs_tdoa_error_samples": max_error,
        "fixed_nested_vs_same_union_posterior_max_abs_difference": posterior_difference,
        "bit_accounting_includes_coefficients_scale_actions_and_decisions": True,
        "status": "PASS",
    }


def _plot_results(
    by_snr: pd.DataFrame, pooled: pd.DataFrame, config: ExperimentConfig, output_dir: Path
) -> None:
    locked = by_snr[by_snr["stage"] == "locked"]
    qbits = 6 if 6 in config.qbits_values else config.qbits_values[0]
    methods = [
        "PASR-Observed",
        "Fixed-Nested",
        "Random-Request",
        "Oracle-Next-Bin",
        "Exhaustive-Proxy-M4",
        "PFRS",
        "Zhai-Fisher",
        "Power",
        "Full16-Uncompressed",
    ]
    colors = {
        "PASR-Observed": "#C23B22",
        "Fixed-Nested": "#8C564B",
        "Random-Request": "#7F7F7F",
        "Oracle-Next-Bin": "#9467BD",
        "Exhaustive-Proxy-M4": "#1F77B4",
        "PFRS": "#2CA02C",
        "Zhai-Fisher": "#17BECF",
        "Power": "#FF7F0E",
        "Full16-Uncompressed": "#111111",
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for method in methods:
        method_qbits = 32 if method == "Full16-Uncompressed" else qbits
        data = locked[(locked["method"] == method) & (locked["qbits"] == method_qbits)]
        if data.empty:
            continue
        data = data.sort_values("snr_db")
        ax.plot(
            data["snr_db"],
            data["rmse_samples"],
            marker="o",
            linewidth=1.35,
            markersize=4,
            label=method,
            color=colors[method],
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("TDOA RMSE (samples)")
    ax.set_title(f"PASR-TDOA locked evaluation ({qbits}-bit I/Q)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.savefig(output_dir / "FigPASR1_TDOA_RMSE.svg", bbox_inches="tight")
    plt.close(fig)

    locked_pooled = pooled[
        (pooled["stage"] == "locked")
        & (pooled["method"] != "Full16-Uncompressed")
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for method, data in locked_pooled.groupby("method", sort=False):
        if method not in colors:
            continue
        ax.scatter(
            data["mean_total_bits"],
            data["rmse_samples"],
            s=34,
            color=colors[method],
            label=method,
            alpha=0.9,
        )
    ax.set_xlabel("Mean total communication bits per pair")
    ax.set_ylabel("Pooled TDOA RMSE (samples)")
    ax.set_title("Accuracy-communication trade-off")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(
        unique.values(),
        unique.keys(),
        ncol=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.savefig(output_dir / "FigPASR2_Accuracy_Bit_Pareto.svg", bbox_inches="tight")
    plt.close(fig)

    active = locked[
        locked["method"].isin(ACTIVE_METHODS) & (locked["qbits"] == qbits)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for method, data in active.groupby("method", sort=False):
        data = data.sort_values("snr_db")
        axes[0].plot(
            data["snr_db"], data["mean_complex_bins"], marker="o", label=method
        )
        axes[1].plot(
            data["snr_db"], data["posterior_stop_rate"], marker="o", label=method
        )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Mean transmitted complex bins")
    axes[0].set_title("Requested spectral coefficients")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Posterior stop rate")
    axes[1].set_title("Confidence-based early stopping")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(output_dir / "FigPASR3_Stopping_Behavior.svg", bbox_inches="tight")
    plt.close(fig)


def run(mode: str) -> Path:
    if mode not in CONFIGS:
        raise ValueError(f"Unknown mode {mode}; choose from {tuple(CONFIGS)}")
    config = CONFIGS[mode]
    started = time.perf_counter()
    model = build_spectral_model()
    tau_grid = _tau_grid(config)
    static_design, static_methods, base = _design_static_masks(model, config)
    nested = static_methods["Fixed-Nested-Final"]
    protocol_validation = _protocol_validation(config, model, tau_grid, base, nested)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "运行结果" / f"PASR_{timestamp}_{mode}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print("PASR-TDOA standalone experiment", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(
        f"Base bins: {model.bin_indices[np.asarray(base)].tolist()} | "
        f"fixed final: {model.bin_indices[np.asarray(nested)].tolist()}",
        flush=True,
    )
    print(
        "Budget includes I/Q coefficients, one coded scale, action indices and "
        "continue/stop decisions.",
        flush=True,
    )
    print(
        "Nominal SNR uses mean source power before the B-side |rho| attenuation; "
        "all methods share the same observations.",
        flush=True,
    )

    development, development_trace = _run_stage(
        "development",
        config.development_seeds,
        config.development_trials,
        config,
        model,
        tau_grid,
        static_methods,
        base,
    )
    locked, locked_trace = _run_stage(
        "locked",
        config.locked_seeds,
        config.locked_trials,
        config,
        model,
        tau_grid,
        static_methods,
        base,
    )
    rows = pd.concat([development, locked], ignore_index=True)
    traces = pd.concat([development_trace, locked_trace], ignore_index=True)
    protocol_validation = {
        **protocol_validation,
        "trial_result_validation": _validate_trial_rows(rows, config),
    }
    by_snr, pooled = _summaries(rows)
    method_metadata = _method_metadata(static_methods, model, config)
    gate = _gate(rows, pooled, by_snr, config, protocol_validation)
    frozen_comparators = _frozen_comparator_table(gate)

    compression = "gzip" if mode == "research" else None
    suffix = ".csv.gz" if compression else ".csv"
    rows.to_csv(
        output_dir / f"pasr_trial_results{suffix}",
        index=False,
        encoding="utf-8",
        compression=compression,
    )
    traces.to_csv(
        output_dir / f"pasr_action_trace{suffix}",
        index=False,
        encoding="utf-8",
        compression=compression,
    )
    static_design.to_csv(
        output_dir / "pasr_static_design_search.csv", index=False, encoding="utf-8"
    )
    by_snr.to_csv(output_dir / "pasr_summary_by_snr.csv", index=False, encoding="utf-8")
    pooled.to_csv(output_dir / "pasr_summary_pooled.csv", index=False, encoding="utf-8")
    method_metadata.to_csv(
        output_dir / "pasr_method_metadata.csv", index=False, encoding="utf-8"
    )
    frozen_comparators.to_csv(
        output_dir / "pasr_frozen_comparators.csv", index=False, encoding="utf-8"
    )
    (output_dir / "pasr_go_no_go.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_results(by_snr, pooled, config, output_dir)

    output_files = sorted(path.name for path in output_dir.iterdir())
    manifest = {
        "method": "PASR-TDOA",
        "mode": mode,
        "scope": gate["scope"],
        "config": asdict(config),
        "model": {
            "n_bins": model.bin_indices.size,
            "bin_indices": model.bin_indices.tolist(),
            "source_power": model.power.tolist(),
            "rho_abs": abs(model.rho),
            "rho_phase_rad": float(np.angle(model.rho)),
            "snr_definition": (
                "mean unattenuated source power divided by equal A/B complex-noise "
                "variance; B-side received SNR is lower by |rho|^2"
            ),
        },
        "base_positions": list(base),
        "base_bins": model.bin_indices[np.asarray(base)].tolist(),
        "static_methods": {
            method: model.bin_indices[np.asarray(positions)].tolist()
            for method, positions in static_methods.items()
        },
        "script_sha256": _sha256(Path(__file__)),
        "core_sha256": _sha256(Path(__file__).with_name("pasr_core.py")),
        "protocol_validation": protocol_validation,
        "public_randomness_assumption": (
            "Subtractive-dither seeds and the development-frozen static time-sharing "
            "schedule are pre-shared; their indices are not charged per sample."
        ),
        "go_no_go": gate,
        "elapsed_seconds": time.perf_counter() - started,
        "output_files": output_files + ["pasr_manifest.json"],
    }
    (output_dir / "pasr_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    print(f"[Done] elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone PASR-TDOA experiment")
    parser.add_argument("--mode", choices=tuple(CONFIGS), default=PYCHARM_RUN_MODE)
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
