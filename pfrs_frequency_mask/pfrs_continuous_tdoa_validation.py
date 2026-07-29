"""Continuous-delay validation for frozen PFRS spectral masks.

This standalone script reads the masks frozen by
``运行结果/PFRS/PFRS_20260716_141935_confirmation`` and evaluates actual
continuous TDOA
estimation under the same nuisance-profiled two-receiver model.  It does not
train a model, reselect frequency bins, or import the project's main training
and localization pipeline.

The validation separates three SNR roles:

* -10 dB: stress test only;
* -5 to 10 dB: threshold/outlier region;
* 15 to 20 dB: local Fisher-information region.

Run directly in PyCharm.  The default formal mode uses fresh development and
locked seeds that are disjoint from the PFRS selection/confirmation seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

import pfrs_finite_delay_selection as pfrs


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
PYCHARM_RUN_MODE = "continuous_validation"
PYCHARM_SOURCE_RESULT_DIR = (
    PROJECT_ROOT
    / "运行结果"
    / "PFRS"
    / "PFRS_20260716_141935_confirmation"
)


@dataclass(frozen=True)
class ContinuousValidationConfig:
    mode: str
    base_mode: str
    snr_db: tuple[float, ...]
    development_seeds: tuple[int, ...]
    locked_seeds: tuple[int, ...]
    development_trials: int
    locked_trials: int
    tau_lower_samples: float
    tau_upper_samples: float
    grid_step_samples: float
    gross_thresholds: tuple[float, ...]
    bootstrap_repetitions: int
    bootstrap_seed: int
    fisher_floor: float
    fisher_ratio_tolerance: float


CONFIGS = {
    "continuous_smoke": ContinuousValidationConfig(
        mode="continuous_smoke",
        base_mode="research",
        snr_db=(-10.0, -5.0, 5.0, 15.0),
        development_seeds=(20262991,),
        locked_seeds=(20263991, 20263992),
        development_trials=32,
        locked_trials=48,
        tau_lower_samples=-4.0,
        tau_upper_samples=4.0,
        grid_step_samples=1.0 / 32.0,
        gross_thresholds=(0.25, 0.5, 1.0, 2.0),
        bootstrap_repetitions=200,
        bootstrap_seed=2026071602,
        fisher_floor=0.90,
        fisher_ratio_tolerance=0.10,
    ),
    "continuous_validation": ContinuousValidationConfig(
        mode="continuous_validation",
        base_mode="research",
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0),
        development_seeds=(20262801, 20262802, 20262803),
        locked_seeds=(
            20263801,
            20263802,
            20263803,
            20263804,
            20263805,
            20263806,
            20263807,
            20263808,
            20263809,
            20263810,
        ),
        development_trials=500,
        locked_trials=2000,
        tau_lower_samples=-4.0,
        tau_upper_samples=4.0,
        grid_step_samples=1.0 / 32.0,
        gross_thresholds=(0.25, 0.5, 1.0, 2.0),
        bootstrap_repetitions=5000,
        bootstrap_seed=2026071602,
        fisher_floor=0.90,
        fisher_ratio_tolerance=0.10,
    ),
}


METHOD_ORDER = (
    "PFRS",
    "Eta09-MeanRisk",
    "Zhai-Fisher",
    "PFRS-Eta08",
    "Power",
    "Uniform",
    "Random",
    "Full16-Uncompressed",
)

REQUIRED_FROZEN_METHODS = METHOD_ORDER[:-1]
PRIMARY_CANDIDATE = "PFRS"
SECONDARY_CANDIDATE = "Eta09-MeanRisk"
PRIMARY_BASELINE = "Zhai-Fisher"
POWER_BASELINE = "Power"


@dataclass(frozen=True)
class FrozenProtocol:
    source_dir: Path
    source_manifest: dict[str, object]
    model: pfrs.SpectralModel
    methods: dict[str, tuple[int, ...]]
    metadata: pd.DataFrame
    source_seed_set: frozenset[int]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ready_config(config: ContinuousValidationConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in tuple(payload.items()):
        if isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def _load_frozen_protocol(
    source_dir: Path,
    config: ContinuousValidationConfig,
) -> FrozenProtocol:
    source_dir = source_dir.resolve()
    manifest_path = source_dir / "pfrs_confirmation_manifest.json"
    frozen_path = source_dir / "pfrs_frozen_methods.csv"
    if not manifest_path.is_file() or not frozen_path.is_file():
        raise FileNotFoundError(
            "The frozen PFRS confirmation manifest/method table is missing from "
            f"{source_dir}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("validation", {}).get("paired_error_reconstruction") != "PASS":
        raise RuntimeError("The source confirmation did not pass paired reconstruction")
    if manifest.get("base_config", {}).get("mode") != config.base_mode:
        raise RuntimeError("Continuous validation base_mode mismatches the source result")

    pfrs_script = Path(pfrs.__file__).resolve()
    expected_sha = str(manifest.get("script_sha256", ""))
    current_sha = _sha256(pfrs_script)
    if current_sha != expected_sha:
        raise RuntimeError(
            "pfrs_finite_delay_selection.py differs from the script that generated "
            "the frozen source result"
        )

    base_config = pfrs.CONFIGS[config.base_mode]
    base_manifest = manifest["base_config"]
    for key in (
        "n_bins",
        "m_bins",
        "prior_width_samples",
        "source_profile",
        "source_floor",
        "rho_abs",
        "rho_phase_rad",
        "seed",
    ):
        if base_manifest[key] != getattr(base_config, key):
            raise RuntimeError(f"Frozen source config mismatch for {key}")

    if not math.isclose(
        config.tau_upper_samples - config.tau_lower_samples,
        float(base_config.prior_width_samples),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Continuous TDOA interval must match the frozen prior width")

    model = pfrs.build_model(base_config)
    bin_to_position = {int(value): i for i, value in enumerate(model.bin_indices)}
    selected_masks = manifest.get("selected_masks", {})
    methods: dict[str, tuple[int, ...]] = {}
    for method in REQUIRED_FROZEN_METHODS:
        if method not in selected_masks:
            raise RuntimeError(f"Frozen source is missing method {method}")
        bins = tuple(int(value) for value in selected_masks[method])
        if len(bins) != base_config.m_bins or len(set(bins)) != len(bins):
            raise RuntimeError(f"Frozen method {method} has an invalid bin budget")
        try:
            methods[method] = tuple(bin_to_position[value] for value in bins)
        except KeyError as exc:
            raise RuntimeError(f"Frozen method {method} contains an invalid bin") from exc
    methods["Full16-Uncompressed"] = tuple(range(base_config.n_bins))

    frozen = pd.read_csv(frozen_path)
    if set(frozen["method"]) != set(REQUIRED_FROZEN_METHODS):
        raise RuntimeError("Frozen method CSV does not match the confirmation manifest")

    reference_snr = float(base_config.local_snr_db)
    variance = float(np.mean(model.power)) / (10.0 ** (reference_snr / 10.0))
    j_zhai = pfrs.fisher_information(
        methods[PRIMARY_BASELINE], model, variance, variance
    )
    j_max_four = max(
        pfrs.fisher_information(subset, model, variance, variance)
        for name, subset in methods.items()
        if name != "Full16-Uncompressed"
    )
    rows = []
    for method in METHOD_ORDER:
        subset = methods[method]
        information = pfrs.fisher_information(subset, model, variance, variance)
        continuous_period = _continuous_ambiguity_period_samples(subset, model)
        continuous_identifiable = continuous_period > (
            config.tau_upper_samples - config.tau_lower_samples
        )
        rows.append(
            {
                "method": method,
                "role": (
                    "primary_candidate"
                    if method == PRIMARY_CANDIDATE
                    else "secondary_candidate"
                    if method == SECONDARY_CANDIDATE
                    else "uncompressed_reference"
                    if method == "Full16-Uncompressed"
                    else "ambiguity_stress_baseline"
                    if not continuous_identifiable
                    else "compressed_baseline"
                ),
                "mask": " ".join(str(int(model.bin_indices[i])) for i in subset),
                "positions": " ".join(str(i) for i in subset),
                "transmitted_complex_bins": len(subset),
                "fisher_information_reference_snr": information,
                "fisher_ratio_to_zhai": information / j_zhai,
                "fisher_ratio_to_max_four_bin": information / j_max_four,
                "crlb_ratio_vs_zhai": j_zhai / information,
                "integer_grid_ambiguity_period_samples": pfrs.ambiguity_period_samples(
                    subset, model
                ),
                "continuous_ambiguity_period_samples": continuous_period,
                "continuous_identifiable_over_prior": continuous_identifiable,
            }
        )
    metadata = pd.DataFrame(rows)

    source_config = manifest["confirmation_config"]
    source_seeds = frozenset(
        int(seed)
        for key in ("selection_seeds", "confirmation_seeds")
        for seed in source_config[key]
    )
    return FrozenProtocol(
        source_dir=source_dir,
        source_manifest=manifest,
        model=model,
        methods=methods,
        metadata=metadata,
        source_seed_set=source_seeds,
    )


def _continuous_ambiguity_period_samples(
    subset: tuple[int, ...],
    model: pfrs.SpectralModel,
) -> float:
    """Return the delay period after profiling an arbitrary global phase.

    For continuous delay, the phase pattern repeats when every selected-bin
    difference accumulates an integer multiple of 2*pi.  Unlike the integer
    sample grid criterion used by the source experiment, the DFT length must
    not be included in the gcd of selected-bin differences.
    """
    selected_bins = model.bin_indices[np.asarray(subset, dtype=int)]
    if selected_bins.size < 2:
        return 0.0
    differences = [
        abs(int(value - selected_bins[0])) for value in selected_bins[1:]
    ]
    nonzero = [value for value in differences if value > 0]
    if not nonzero:
        return 0.0
    return float(model.bin_indices.size / reduce(math.gcd, nonzero))


def _tau_grid(config: ContinuousValidationConfig) -> np.ndarray:
    width = config.tau_upper_samples - config.tau_lower_samples
    intervals = int(round(width / config.grid_step_samples))
    if intervals < 4 or not math.isclose(
        intervals * config.grid_step_samples, width, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("The TDOA interval must contain an integer number of grid steps")
    return np.linspace(
        config.tau_lower_samples,
        config.tau_upper_samples,
        intervals + 1,
        dtype=float,
    )


def _standard_complex_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    scale = 1.0 / math.sqrt(2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _profiled_cost_grid(
    y_a: np.ndarray,
    y_b: np.ndarray,
    omega: np.ndarray,
    tau_grid: np.ndarray,
    sigma_a2: float,
    sigma_b2: float,
) -> np.ndarray:
    """Vectorized profiled cost for every trial and candidate delay."""
    if y_a.shape != y_b.shape or y_a.ndim != 2:
        raise ValueError("y_a and y_b must have the same [trials, bins] shape")
    if y_a.shape[1] != omega.size:
        raise ValueError("omega length must match the selected-bin dimension")
    if tau_grid.ndim != 1 or tau_grid.size < 3:
        raise ValueError("tau_grid must be a one-dimensional grid with >=3 points")
    if sigma_a2 <= 0.0 or sigma_b2 <= 0.0:
        raise ValueError("noise variances must be positive")

    steering = np.exp(-1j * omega[:, None] * tau_grid[None, :])
    cross = (np.conj(y_b) * y_a) @ steering
    cross *= -1.0 / math.sqrt(sigma_a2 * sigma_b2)
    a = np.sum(np.abs(y_b) ** 2, axis=1)[:, None] / sigma_b2
    d = np.sum(np.abs(y_a) ** 2, axis=1)[:, None] / sigma_a2
    discriminant = np.sqrt(
        np.maximum((a - d) ** 2 + 4.0 * np.abs(cross) ** 2, 0.0)
    )
    return np.maximum(0.5 * (a + d - discriminant).real, 0.0)


def _estimate_continuous_tdoa(
    y_a: np.ndarray,
    y_b: np.ndarray,
    omega: np.ndarray,
    tau_grid: np.ndarray,
    sigma_a2: float,
    sigma_b2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    costs = _profiled_cost_grid(
        y_a, y_b, omega, tau_grid, sigma_a2, sigma_b2
    )
    coarse_index = np.argmin(costs, axis=1)
    estimate = tau_grid[coarse_index].copy()
    boundary = (coarse_index == 0) | (coarse_index == tau_grid.size - 1)

    interior_rows = np.flatnonzero(~boundary)
    if interior_rows.size:
        centers = coarse_index[interior_rows]
        left = costs[interior_rows, centers - 1]
        middle = costs[interior_rows, centers]
        right = costs[interior_rows, centers + 1]
        denominator = left - 2.0 * middle + right
        scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
        convex = denominator > np.finfo(float).eps * scale
        offsets = np.zeros(interior_rows.size, dtype=float)
        offsets[convex] = 0.5 * (left[convex] - right[convex]) / denominator[convex]
        offsets = np.clip(offsets, -1.0, 1.0)
        step = tau_grid[1] - tau_grid[0]
        estimate[interior_rows] += offsets * step

    estimate = np.clip(estimate, tau_grid[0], tau_grid[-1])
    return estimate, tau_grid[coarse_index], boundary


def _validate_cost_and_estimator(
    protocol: FrozenProtocol,
    config: ContinuousValidationConfig,
) -> dict[str, float | str]:
    model = protocol.model
    subset = protocol.methods[PRIMARY_CANDIDATE]
    idx = np.asarray(subset, dtype=int)
    rng = np.random.default_rng(2026071603)
    y_a = rng.normal(size=(5, idx.size)) + 1j * rng.normal(size=(5, idx.size))
    y_b = rng.normal(size=(5, idx.size)) + 1j * rng.normal(size=(5, idx.size))
    tau_test = np.asarray([-3.25, -0.75, 0.5, 2.125], dtype=float)
    grid_cost = _profiled_cost_grid(
        y_a, y_b, model.omega[idx], tau_test, 0.7, 1.3
    )
    scalar = np.column_stack(
        [
            pfrs.profiled_cost(y_a, y_b, model.omega[idx], tau, 0.7, 1.3)
            for tau in tau_test
        ]
    )
    max_cost_difference = float(np.max(np.abs(grid_cost - scalar)))
    if max_cost_difference > 1e-10:
        raise RuntimeError("Grid-profiled cost disagrees with the validated scalar cost")

    tau_grid = _tau_grid(config)
    true_tau = np.asarray([-3.731, -1.317, 0.0, 2.713, 3.881], dtype=float)
    max_noiseless_error = 0.0
    ambiguous_methods = []
    metadata = protocol.metadata.set_index("method")
    for method in METHOD_ORDER:
        subset = protocol.methods[method]
        idx = np.asarray(subset, dtype=int)
        y_a_clean = np.broadcast_to(model.source[idx], (true_tau.size, idx.size)).copy()
        y_b_clean = (
            model.rho
            * model.source[idx][None, :]
            * np.exp(-1j * true_tau[:, None] * model.omega[idx][None, :])
        )
        estimate, _, _ = _estimate_continuous_tdoa(
            y_a_clean,
            y_b_clean,
            model.omega[idx],
            tau_grid,
            1.0,
            1.0,
        )
        method_error = float(np.max(np.abs(estimate - true_tau)))
        if bool(metadata.at[method, "continuous_identifiable_over_prior"]):
            max_noiseless_error = max(max_noiseless_error, method_error)
        else:
            ambiguous_methods.append(method)
    if max_noiseless_error > config.grid_step_samples:
        raise RuntimeError("Noiseless continuous TDOA recovery failed")
    return {
        "profiled_grid_scalar_max_abs_difference": max_cost_difference,
        "max_noiseless_tdoa_error_identifiable_methods_samples": max_noiseless_error,
        "continuously_ambiguous_methods": ambiguous_methods,
        "status": "PASS",
    }


def _run_stage(
    stage: str,
    seeds: tuple[int, ...],
    n_trials: int,
    protocol: FrozenProtocol,
    config: ContinuousValidationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = protocol.model
    tau_grid = _tau_grid(config)
    n_methods = len(METHOD_ORDER)
    errors = np.empty(
        (len(seeds), len(config.snr_db), n_methods, n_trials), dtype=np.float32
    )
    boundary = np.empty_like(errors, dtype=np.uint8)
    true_tau_all = np.empty((len(seeds), n_trials), dtype=np.float64)
    reference_power = float(np.mean(model.power))
    total = len(seeds) * len(config.snr_db)
    done = 0
    started = time.perf_counter()

    for seed_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        true_tau = rng.uniform(
            config.tau_lower_samples, config.tau_upper_samples, size=n_trials
        )
        noise_a = _standard_complex_noise(rng, (n_trials, model.bin_indices.size))
        noise_b = _standard_complex_noise(rng, (n_trials, model.bin_indices.size))
        true_tau_all[seed_index] = true_tau
        clean_b = (
            model.rho
            * model.source[None, :]
            * np.exp(-1j * true_tau[:, None] * model.omega[None, :])
        )
        for snr_index, snr_db in enumerate(config.snr_db):
            variance = reference_power / (10.0 ** (snr_db / 10.0))
            noise_scale = math.sqrt(variance)
            y_a = model.source[None, :] + noise_scale * noise_a
            y_b = clean_b + noise_scale * noise_b
            for method_index, method in enumerate(METHOD_ORDER):
                idx = np.asarray(protocol.methods[method], dtype=int)
                estimate, _, hit_boundary = _estimate_continuous_tdoa(
                    y_a[:, idx],
                    y_b[:, idx],
                    model.omega[idx],
                    tau_grid,
                    variance,
                    variance,
                )
                errors[seed_index, snr_index, method_index] = (
                    estimate - true_tau
                ).astype(np.float32)
                boundary[seed_index, snr_index, method_index] = hit_boundary.astype(
                    np.uint8
                )
            done += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (total - done)
            print(
                f"[{stage}] seed={seed} SNR={snr_db:g} dB "
                f"{done}/{total} | elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                flush=True,
            )
    return errors, boundary, true_tau_all


def _metric_name(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def _point_metrics(
    error: np.ndarray,
    boundary: np.ndarray,
    thresholds: Iterable[float],
) -> dict[str, float | int]:
    values = np.asarray(error, dtype=float).ravel()
    absolute = np.abs(values)
    result: dict[str, float | int] = {
        "n": int(values.size),
        "mse_samples2": float(np.mean(values**2)),
        "rmse_samples": float(np.sqrt(np.mean(values**2))),
        "mae_samples": float(np.mean(absolute)),
        "bias_samples": float(np.mean(values)),
        "std_samples": float(np.std(values)),
        "median_abs_error_samples": float(np.median(absolute)),
        "p90_abs_error_samples": float(np.quantile(absolute, 0.90)),
        "p95_abs_error_samples": float(np.quantile(absolute, 0.95)),
        "boundary_rate": float(np.mean(np.asarray(boundary).ravel())),
    }
    for threshold in thresholds:
        key = _metric_name(threshold)
        result[f"within_{key}_rate"] = float(np.mean(absolute <= threshold))
        result[f"gross_gt_{key}_rate"] = float(np.mean(absolute > threshold))
    return result


def _snr_groups(config: ContinuousValidationConfig) -> dict[str, np.ndarray]:
    snr = np.asarray(config.snr_db, dtype=float)
    groups = {f"snr_{value:g}": np.flatnonzero(np.isclose(snr, value)) for value in snr}
    groups["stress"] = np.flatnonzero(snr <= -10.0)
    groups["threshold"] = np.flatnonzero((snr >= -5.0) & (snr <= 10.0))
    groups["asymptotic"] = np.flatnonzero(snr >= 15.0)
    groups["all"] = np.arange(snr.size)
    for name, indices in groups.items():
        if indices.size == 0:
            raise RuntimeError(f"SNR group {name} is empty")
    return groups


def _by_seed_snr(
    stage: str,
    seeds: tuple[int, ...],
    errors: np.ndarray,
    boundary: np.ndarray,
    config: ContinuousValidationConfig,
) -> pd.DataFrame:
    rows = []
    for seed_index, seed in enumerate(seeds):
        for snr_index, snr_db in enumerate(config.snr_db):
            for method_index, method in enumerate(METHOD_ORDER):
                row = {
                    "stage": stage,
                    "seed": seed,
                    "snr_db": snr_db,
                    "method": method,
                }
                row.update(
                    _point_metrics(
                        errors[seed_index, snr_index, method_index],
                        boundary[seed_index, snr_index, method_index],
                        config.gross_thresholds,
                    )
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_draw(
    n_seeds: int,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_seeds, size=(repetitions, n_seeds))


def _seed_sufficient(
    error: np.ndarray,
    snr_indices: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    selected = np.asarray(error[:, snr_indices, :], dtype=float)
    flattened = selected.reshape(selected.shape[0], -1)
    return {
        "count": np.full(flattened.shape[0], flattened.shape[1], dtype=float),
        "sum": np.sum(flattened, axis=1),
        "sum_abs": np.sum(np.abs(flattened), axis=1),
        "sum_sq": np.sum(flattened**2, axis=1),
        "gross": np.sum(np.abs(flattened) > threshold, axis=1),
    }


def _bootstrap_metric_samples(
    sufficient: dict[str, np.ndarray],
    draw: np.ndarray,
) -> dict[str, np.ndarray]:
    count = sufficient["count"][draw].sum(axis=1)
    sum_error = sufficient["sum"][draw].sum(axis=1)
    sum_abs = sufficient["sum_abs"][draw].sum(axis=1)
    sum_sq = sufficient["sum_sq"][draw].sum(axis=1)
    gross = sufficient["gross"][draw].sum(axis=1)
    return {
        "rmse": np.sqrt(sum_sq / count),
        "mse": sum_sq / count,
        "mae": sum_abs / count,
        "bias": sum_error / count,
        "gross": gross / count,
    }


def _locked_summary(
    errors: np.ndarray,
    boundary: np.ndarray,
    config: ContinuousValidationConfig,
    draw: np.ndarray,
) -> pd.DataFrame:
    rows = []
    groups = _snr_groups(config)
    for group, snr_indices in groups.items():
        snr_values = " ".join(f"{config.snr_db[i]:g}" for i in snr_indices)
        for method_index, method in enumerate(METHOD_ORDER):
            selected_error = errors[:, snr_indices, method_index, :]
            selected_boundary = boundary[:, snr_indices, method_index, :]
            point = _point_metrics(
                selected_error, selected_boundary, config.gross_thresholds
            )
            sufficient = _seed_sufficient(selected_error, np.arange(snr_indices.size), 1.0)
            samples = _bootstrap_metric_samples(sufficient, draw)
            row: dict[str, object] = {
                "stage": "locked",
                "group": group,
                "snr_db_values": snr_values,
                "method": method,
                **point,
            }
            for metric in ("rmse", "mae", "bias", "gross"):
                low, high = np.quantile(samples[metric], [0.025, 0.975])
                output_name = "gross_gt_1p0_rate" if metric == "gross" else f"{metric}_samples"
                row[f"{output_name}_ci_low"] = float(low)
                row[f"{output_name}_ci_high"] = float(high)
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_comparisons(
    errors: np.ndarray,
    config: ContinuousValidationConfig,
    draw: np.ndarray,
) -> pd.DataFrame:
    rows = []
    method_index = {method: i for i, method in enumerate(METHOD_ORDER)}
    groups = _snr_groups(config)
    pairs = tuple(
        (candidate, baseline)
        for candidate in (PRIMARY_CANDIDATE, SECONDARY_CANDIDATE)
        for baseline in (PRIMARY_BASELINE, POWER_BASELINE)
    )
    for group, snr_indices in groups.items():
        for candidate, baseline in pairs:
            candidate_error = errors[:, snr_indices, method_index[candidate], :]
            baseline_error = errors[:, snr_indices, method_index[baseline], :]
            candidate_sufficient = _seed_sufficient(
                candidate_error, np.arange(snr_indices.size), 1.0
            )
            baseline_sufficient = _seed_sufficient(
                baseline_error, np.arange(snr_indices.size), 1.0
            )
            candidate_samples = _bootstrap_metric_samples(candidate_sufficient, draw)
            baseline_samples = _bootstrap_metric_samples(baseline_sufficient, draw)
            candidate_point = _point_metrics(
                candidate_error,
                np.zeros_like(candidate_error, dtype=np.uint8),
                config.gross_thresholds,
            )
            baseline_point = _point_metrics(
                baseline_error,
                np.zeros_like(baseline_error, dtype=np.uint8),
                config.gross_thresholds,
            )
            point_values = {
                "rmse_gain": baseline_point["rmse_samples"]
                - candidate_point["rmse_samples"],
                "mae_gain": baseline_point["mae_samples"]
                - candidate_point["mae_samples"],
                "gross_gt_1p0_gain": baseline_point["gross_gt_1p0_rate"]
                - candidate_point["gross_gt_1p0_rate"],
                "mse_ratio_candidate_over_baseline": candidate_point["mse_samples2"]
                / max(float(baseline_point["mse_samples2"]), np.finfo(float).tiny),
            }
            sample_values = {
                "rmse_gain": baseline_samples["rmse"] - candidate_samples["rmse"],
                "mae_gain": baseline_samples["mae"] - candidate_samples["mae"],
                "gross_gt_1p0_gain": baseline_samples["gross"]
                - candidate_samples["gross"],
                "mse_ratio_candidate_over_baseline": candidate_samples["mse"]
                / np.maximum(baseline_samples["mse"], np.finfo(float).tiny),
            }
            for metric, point in point_values.items():
                low, high = np.quantile(sample_values[metric], [0.025, 0.975])
                rows.append(
                    {
                        "stage": "locked",
                        "group": group,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "value": float(point),
                        "ci_low": float(low),
                        "ci_high": float(high),
                    }
                )
    return pd.DataFrame(rows)


def _fisher_check(
    errors: np.ndarray,
    protocol: FrozenProtocol,
    config: ContinuousValidationConfig,
) -> pd.DataFrame:
    model = protocol.model
    reference_power = float(np.mean(model.power))
    rows = []
    for snr_index, snr_db in enumerate(config.snr_db):
        if snr_db < 15.0:
            continue
        variance = reference_power / (10.0 ** (snr_db / 10.0))
        for method_index, method in enumerate(METHOD_ORDER):
            information = pfrs.fisher_information(
                protocol.methods[method], model, variance, variance
            )
            method_error = np.asarray(errors[:, snr_index, method_index], float).ravel()
            mse = float(np.mean(method_error**2))
            crlb = 1.0 / information
            local_mask = np.abs(method_error) <= 0.5
            conditional_mse = (
                float(np.mean(method_error[local_mask] ** 2))
                if np.any(local_mask)
                else math.nan
            )
            rows.append(
                {
                    "method": method,
                    "snr_db": snr_db,
                    "empirical_mse_samples2": mse,
                    "fisher_information": information,
                    "crlb_samples2": crlb,
                    "mse_to_crlb_ratio": mse / crlb,
                    "crlb_efficiency": crlb / mse,
                    "local_inlier_rate_abs_le_0p5": float(np.mean(local_mask)),
                    "conditional_local_mse_samples2": conditional_mse,
                    "conditional_local_mse_to_crlb_ratio": conditional_mse / crlb,
                }
            )
    return pd.DataFrame(rows)


def _comparison_row(
    paired: pd.DataFrame,
    group: str,
    candidate: str,
    baseline: str,
    metric: str,
) -> pd.Series:
    rows = paired[
        (paired["group"] == group)
        & (paired["candidate"] == candidate)
        & (paired["baseline"] == baseline)
        & (paired["metric"] == metric)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one paired row for {group}/{candidate}/{baseline}/{metric}"
        )
    return rows.iloc[0]


def _candidate_gate(
    candidate: str,
    paired: pd.DataFrame,
    metadata: pd.DataFrame,
    config: ContinuousValidationConfig,
) -> dict[str, object]:
    meta = metadata.loc[metadata["method"] == candidate].iloc[0]
    threshold_rmse = _comparison_row(
        paired, "threshold", candidate, PRIMARY_BASELINE, "rmse_gain"
    )
    threshold_gross = _comparison_row(
        paired, "threshold", candidate, PRIMARY_BASELINE, "gross_gt_1p0_gain"
    )
    power_rmse = _comparison_row(
        paired, "asymptotic", candidate, POWER_BASELINE, "rmse_gain"
    )
    zhai_ratio = _comparison_row(
        paired,
        "asymptotic",
        candidate,
        PRIMARY_BASELINE,
        "mse_ratio_candidate_over_baseline",
    )
    fisher_ratio = float(meta["fisher_ratio_to_zhai"])
    predicted_ratio = 1.0 / fisher_ratio
    allowed_ratio = predicted_ratio * (1.0 + config.fisher_ratio_tolerance)
    fisher_floor_pass = (
        float(meta["fisher_ratio_to_max_four_bin"]) + 1e-12 >= config.fisher_floor
    )
    global_pass = bool(
        float(threshold_rmse["ci_low"]) > 0.0
        or float(threshold_gross["ci_low"]) > 0.0
    )
    local_power_pass = bool(float(power_rmse["ci_low"]) > 0.0)
    zhai_noninferiority_pass = bool(float(zhai_ratio["ci_high"]) <= allowed_ratio)
    go = fisher_floor_pass and global_pass and local_power_pass and zhai_noninferiority_pass
    clear_global_failure = bool(
        float(threshold_rmse["ci_high"]) <= 0.0
        and float(threshold_gross["ci_high"]) <= 0.0
    )
    clear_local_failure = bool(float(power_rmse["ci_high"]) <= 0.0)
    status = "GO" if go else "NO-GO" if clear_global_failure or clear_local_failure else "HOLD"
    return {
        "candidate": candidate,
        "status": status,
        "fisher_floor_pass": fisher_floor_pass,
        "global_threshold_pass": global_pass,
        "local_precision_vs_power_pass": local_power_pass,
        "high_snr_zhai_noninferiority_pass": zhai_noninferiority_pass,
        "threshold_rmse_gain_vs_zhai": float(threshold_rmse["value"]),
        "threshold_rmse_gain_ci": [
            float(threshold_rmse["ci_low"]),
            float(threshold_rmse["ci_high"]),
        ],
        "threshold_gross_gt_1_gain_vs_zhai": float(threshold_gross["value"]),
        "threshold_gross_gt_1_gain_ci": [
            float(threshold_gross["ci_low"]),
            float(threshold_gross["ci_high"]),
        ],
        "asymptotic_rmse_gain_vs_power": float(power_rmse["value"]),
        "asymptotic_rmse_gain_vs_power_ci": [
            float(power_rmse["ci_low"]),
            float(power_rmse["ci_high"]),
        ],
        "asymptotic_mse_ratio_vs_zhai": float(zhai_ratio["value"]),
        "asymptotic_mse_ratio_vs_zhai_ci": [
            float(zhai_ratio["ci_low"]),
            float(zhai_ratio["ci_high"]),
        ],
        "fisher_predicted_mse_ratio_vs_zhai": predicted_ratio,
        "allowed_mse_ratio_with_tolerance": allowed_ratio,
    }


def _build_gate(
    paired: pd.DataFrame,
    metadata: pd.DataFrame,
    config: ContinuousValidationConfig,
) -> dict[str, object]:
    primary = _candidate_gate(PRIMARY_CANDIDATE, paired, metadata, config)
    secondary = _candidate_gate(SECONDARY_CANDIDATE, paired, metadata, config)
    if primary["status"] == "GO":
        route_status = "GO_PFRS"
    elif secondary["status"] == "GO":
        route_status = "REDIRECT_ETA09"
    elif primary["status"] == "NO-GO" and secondary["status"] == "NO-GO":
        route_status = "NO_GO"
    else:
        route_status = "HOLD"
    groups = _snr_groups(config)
    return {
        "route_status": route_status,
        "primary": primary,
        "secondary": secondary,
        "stress_snr_role": "reported_only_not_used_by_gate",
        "stress_snr_db": [float(config.snr_db[i]) for i in groups["stress"]],
        "threshold_snr_db": [
            float(config.snr_db[i]) for i in groups["threshold"]
        ],
        "asymptotic_snr_db": [
            float(config.snr_db[i]) for i in groups["asymptotic"]
        ],
    }


def _method_styles() -> dict[str, dict[str, object]]:
    return {
        "PFRS": {"color": "#D55E00", "marker": "o", "linestyle": "-"},
        "Eta09-MeanRisk": {"color": "#CC79A7", "marker": "s", "linestyle": "-"},
        "Zhai-Fisher": {"color": "#0072B2", "marker": "^", "linestyle": "--"},
        "PFRS-Eta08": {"color": "#E69F00", "marker": "v", "linestyle": ":"},
        "Power": {"color": "#009E73", "marker": "D", "linestyle": "-."},
        "Uniform": {"color": "#6A3D9A", "marker": "P", "linestyle": ":"},
        "Random": {"color": "#7F7F7F", "marker": "X", "linestyle": ":"},
        "Full16-Uncompressed": {
            "color": "#000000",
            "marker": "x",
            "linestyle": "--",
        },
    }


def _plot_performance(
    summary: pd.DataFrame,
    config: ContinuousValidationConfig,
    output_path: Path,
) -> None:
    snr_rows = summary[summary["group"].str.startswith("snr_")].copy()
    snr_rows["snr_db"] = snr_rows["group"].str.removeprefix("snr_").astype(float)
    styles = _method_styles()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for method in METHOD_ORDER:
        rows = snr_rows.loc[snr_rows["method"] == method].sort_values("snr_db")
        style = styles[method]
        axes[0].plot(
            rows["snr_db"], rows["rmse_samples"], label=method, linewidth=1.5, **style
        )
        axes[1].plot(
            rows["snr_db"],
            rows["gross_gt_1p0_rate"],
            label=method,
            linewidth=1.5,
            **style,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Continuous TDOA RMSE (samples)")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Gross error rate, |error| > 1 sample")
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.set_xticks(config.snr_db)
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_fisher_check(fisher: pd.DataFrame, output_path: Path) -> None:
    styles = _method_styles()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for method in METHOD_ORDER:
        rows = fisher.loc[fisher["method"] == method].sort_values("snr_db")
        ax.plot(
            rows["snr_db"],
            rows["mse_to_crlb_ratio"],
            label=method,
            linewidth=1.5,
            **styles[method],
        )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", label="CRLB")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Empirical MSE / CRLB")
    ax.set_xticks(sorted(fisher["snr_db"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_pareto(summary: pd.DataFrame, output_path: Path) -> None:
    threshold = summary.loc[summary["group"] == "threshold"].set_index("method")
    asymptotic = summary.loc[summary["group"] == "asymptotic"].set_index("method")
    styles = _method_styles()
    label_offsets = {
        "PFRS": (5, 5),
        "Eta09-MeanRisk": (-8, -14),
        "Zhai-Fisher": (5, 7),
        "PFRS-Eta08": (5, -14),
        "Power": (5, 5),
        "Uniform": (-2, 7),
        "Random": (5, 5),
        "Full16-Uncompressed": (5, 5),
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for method in METHOD_ORDER:
        x = float(asymptotic.at[method, "rmse_samples"])
        y = float(threshold.at[method, "gross_gt_1p0_rate"])
        style = styles[method]
        ax.scatter(
            x,
            y,
            s=52,
            color=style["color"],
            marker=style["marker"],
            label=method,
            zorder=3,
        )
        ax.annotate(
            method,
            (x, y),
            xytext=label_offsets[method],
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("High-SNR continuous TDOA RMSE (samples; lower is better)")
    ax.set_ylabel("Threshold-region gross error rate (lower is better)")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _validate_outputs(
    development_errors: np.ndarray,
    locked_errors: np.ndarray,
    development_boundary: np.ndarray,
    locked_boundary: np.ndarray,
    protocol: FrozenProtocol,
    config: ContinuousValidationConfig,
) -> dict[str, object]:
    expected_development = (
        len(config.development_seeds),
        len(config.snr_db),
        len(METHOD_ORDER),
        config.development_trials,
    )
    expected_locked = (
        len(config.locked_seeds),
        len(config.snr_db),
        len(METHOD_ORDER),
        config.locked_trials,
    )
    if development_errors.shape != expected_development:
        raise RuntimeError("Development error tensor has the wrong shape")
    if locked_errors.shape != expected_locked:
        raise RuntimeError("Locked error tensor has the wrong shape")
    if development_boundary.shape != expected_development:
        raise RuntimeError("Development boundary tensor has the wrong shape")
    if locked_boundary.shape != expected_locked:
        raise RuntimeError("Locked boundary tensor has the wrong shape")
    if not np.isfinite(development_errors).all() or not np.isfinite(locked_errors).all():
        raise RuntimeError("Continuous TDOA errors contain non-finite values")
    if not set(np.unique(development_boundary)).issubset({0, 1}):
        raise RuntimeError("Development boundary flags are invalid")
    if not set(np.unique(locked_boundary)).issubset({0, 1}):
        raise RuntimeError("Locked boundary flags are invalid")

    development = set(config.development_seeds)
    locked = set(config.locked_seeds)
    if development & locked:
        raise RuntimeError("Development and locked seeds overlap")
    if development & protocol.source_seed_set or locked & protocol.source_seed_set:
        raise RuntimeError("Continuous validation seeds overlap PFRS source seeds")
    metadata = protocol.metadata.set_index("method")
    for method in (PRIMARY_CANDIDATE, SECONDARY_CANDIDATE, PRIMARY_BASELINE, POWER_BASELINE):
        if not bool(metadata.at[method, "continuous_identifiable_over_prior"]):
            raise RuntimeError(f"Required gate method {method} is continuously ambiguous")
    return {
        "development_shape": list(expected_development),
        "locked_shape": list(expected_locked),
        "seed_partitions_disjoint": True,
        "all_errors_finite": True,
        "boundary_flags_valid": True,
        "required_gate_methods_continuously_identifiable": True,
        "status": "PASS",
    }


def run(
    config: ContinuousValidationConfig,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = _load_frozen_protocol(source_dir, config)
    numerical_validation = _validate_cost_and_estimator(protocol, config)
    print("=" * 78, flush=True)
    print("Frozen-mask continuous TDOA validation", flush=True)
    print("=" * 78, flush=True)
    print(f"Mode: {config.mode}", flush=True)
    print(f"Source: {protocol.source_dir}", flush=True)
    print(f"SNR: {list(config.snr_db)}", flush=True)
    print(
        f"TDOA: [{config.tau_lower_samples}, {config.tau_upper_samples}] samples, "
        f"grid step={config.grid_step_samples:g}",
        flush=True,
    )
    print(
        f"Development: {len(config.development_seeds)} seeds x "
        f"{config.development_trials} trials; locked: {len(config.locked_seeds)} "
        f"seeds x {config.locked_trials} trials",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)

    development_errors, development_boundary, development_tau = _run_stage(
        "development",
        config.development_seeds,
        config.development_trials,
        protocol,
        config,
    )
    locked_errors, locked_boundary, locked_tau = _run_stage(
        "locked",
        config.locked_seeds,
        config.locked_trials,
        protocol,
        config,
    )
    output_validation = _validate_outputs(
        development_errors,
        locked_errors,
        development_boundary,
        locked_boundary,
        protocol,
        config,
    )

    development_by_seed = _by_seed_snr(
        "development",
        config.development_seeds,
        development_errors,
        development_boundary,
        config,
    )
    locked_by_seed = _by_seed_snr(
        "locked",
        config.locked_seeds,
        locked_errors,
        locked_boundary,
        config,
    )
    by_seed = pd.concat([development_by_seed, locked_by_seed], ignore_index=True)
    draw = _bootstrap_draw(
        len(config.locked_seeds),
        config.bootstrap_repetitions,
        config.bootstrap_seed,
    )
    locked_summary = _locked_summary(
        locked_errors, locked_boundary, config, draw
    )
    paired = _paired_comparisons(locked_errors, config, draw)
    fisher = _fisher_check(locked_errors, protocol, config)
    gate = _build_gate(paired, protocol.metadata, config)

    protocol.metadata.to_csv(
        output_dir / "pfrs_continuous_method_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_seed.to_csv(
        output_dir / "pfrs_continuous_by_seed_snr.csv",
        index=False,
        encoding="utf-8-sig",
    )
    locked_summary.to_csv(
        output_dir / "pfrs_continuous_locked_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_dir / "pfrs_continuous_paired_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fisher.to_csv(
        output_dir / "pfrs_continuous_fisher_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        output_dir / "pfrs_continuous_trial_errors.npz",
        development_errors=development_errors,
        locked_errors=locked_errors,
        development_boundary=development_boundary,
        locked_boundary=locked_boundary,
        development_true_tau=development_tau,
        locked_true_tau=locked_tau,
        development_seeds=np.asarray(config.development_seeds, dtype=np.int64),
        locked_seeds=np.asarray(config.locked_seeds, dtype=np.int64),
        snr_db=np.asarray(config.snr_db, dtype=float),
        method_names=np.asarray(METHOD_ORDER),
    )
    _plot_performance(
        locked_summary, config, output_dir / "PFRS_Continuous_TDOA_Performance.svg"
    )
    _plot_fisher_check(
        fisher, output_dir / "PFRS_Continuous_Fisher_Check.svg"
    )
    _plot_pareto(locked_summary, output_dir / "PFRS_Continuous_Pareto.svg")
    (output_dir / "pfrs_continuous_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "source_result_dir": str(protocol.source_dir),
        "source_manifest_sha256": _sha256(
            protocol.source_dir / "pfrs_confirmation_manifest.json"
        ),
        "source_pfrs_script_sha256": protocol.source_manifest["script_sha256"],
        "estimator_semantics": (
            "Continuous TDOA is estimated by minimizing the same nuisance-profiled "
            "likelihood over a fixed fine delay grid followed by local parabolic "
            "refinement. Frequency masks are frozen and never reselected."
        ),
        "snr_roles": {
            "stress": [-10.0],
            "threshold": [-5.0, 0.0, 5.0, 10.0],
            "asymptotic": [15.0, 20.0],
        },
        "config": _json_ready_config(config),
        "selected_masks": {
            method: [int(protocol.model.bin_indices[i]) for i in subset]
            for method, subset in protocol.methods.items()
        },
        "numerical_validation": numerical_validation,
        "output_validation": output_validation,
        "gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
        "output_files": sorted(
            [path.name for path in output_dir.iterdir()]
            + ["pfrs_continuous_manifest.json"]
        ),
    }
    (output_dir / "pfrs_continuous_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    key_rows = locked_summary[
        locked_summary["group"].isin(["stress", "threshold", "asymptotic", "all"])
    ][
        [
            "group",
            "method",
            "rmse_samples",
            "mae_samples",
            "gross_gt_1p0_rate",
            "p95_abs_error_samples",
        ]
    ]
    print("\nLocked continuous-TDOA summary", flush=True)
    print(key_rows.to_string(index=False), flush=True)
    print("\nAutomatic route gate", flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    print(f"[Done] elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous TDOA validation for frozen PFRS masks"
    )
    parser.add_argument("--mode", choices=tuple(CONFIGS), default=PYCHARM_RUN_MODE)
    parser.add_argument("--source-dir", type=Path, default=PYCHARM_SOURCE_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "运行结果"
            / "PFRS"
            / f"PFRS_CONTINUOUS_{timestamp}_{config.mode}"
        )
    run(config, args.source_dir, output_dir.resolve())


if __name__ == "__main__":
    main()
