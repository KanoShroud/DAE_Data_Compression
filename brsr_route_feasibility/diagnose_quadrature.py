"""BRSR阶段1连续复增益边缘积分的独立收敛诊断。

PyCharm直接运行本文件时默认执行 ``diagnostic``。该脚本只重建阶段1正式
development的prior/calibration样本和冻结动作，不读取evaluation性能来选择
数值积分配置，也不修改阶段1、Stage A/B或任何历史结果。
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brsr_route_feasibility.feasibility import (  # noqa: E402
    PoolName,
    action_levels,
    build_candidate_pools,
    build_likelihood_cases,
    linear_likelihood_context,
    linear_template,
    observation_from_context,
    physical_public_scale,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer, tau_posterior  # noqa: E402
from brsr_tdoa.brsr_stage_b import evaluate_fixed, state_from_levels  # noqa: E402
from brsr_tdoa.brsr_waveform import (  # noqa: E402
    StructuredTrial,
    WaveformConfig,
    candidate_fft_values,
    estimate_candidate_statistics,
    generate_structured_trial,
    observe_trial,
)


PYCHARM_RUN_MODE = "diagnostic"
SOURCE_STAGE1_DIR = (
    ROOT
    / "运行结果"
    / "BRSR"
    / "BRSR_ROUTE_STAGE1_20260723_170115_development"
)
CURRENT_GRID = (5, 16)
RUNTIME_SOURCE_FILES = {
    "brsr_route_feasibility/feasibility.py",
    "brsr_tdoa/brsr_core.py",
    "brsr_tdoa/brsr_stage_b.py",
    "brsr_tdoa/brsr_waveform.py",
}
NON_RUNTIME_SOURCE_FILES = {
    "brsr_route_feasibility/run_stage1.py",
}
METRIC_COLUMNS = (
    "seed",
    "trial",
    "cluster_id",
    "snr_db",
    "amplitude_nodes",
    "phase_nodes",
    "n_rho",
    "estimate_tau",
    "posterior_risk",
    "reference_amplitude_nodes",
    "reference_phase_nodes",
    "estimate_abs_diff_vs_reference",
    "risk_abs_diff_vs_reference",
    "tau_posterior_l1_vs_reference",
)
SUMMARY_COLUMNS = (
    "amplitude_nodes",
    "phase_nodes",
    "n_rho",
    "n_conditions",
    "estimate_diff_median",
    "estimate_diff_p95",
    "estimate_diff_max",
    "risk_diff_median",
    "risk_diff_p95",
    "risk_diff_max",
    "posterior_l1_median",
    "posterior_l1_p95",
    "posterior_l1_max",
    "estimate_pass",
    "risk_pass",
    "posterior_l1_pass",
    "full_pass",
)
PER_SNR_COLUMNS = (
    "amplitude_nodes",
    "phase_nodes",
    "n_rho",
    "snr_db",
    "n_conditions",
    "estimate_diff_max",
    "estimate_diff_p95",
    "risk_diff_max",
    "risk_diff_p95",
    "posterior_l1_max",
    "posterior_l1_p95",
)
ATTEMPT_COLUMNS = (
    "attempt_order",
    "amplitude_nodes",
    "phase_nodes",
    "n_rho",
    "estimate_diff_max",
    "risk_diff_max",
    "posterior_l1_max",
    "estimate_pass",
    "risk_pass",
    "posterior_l1_pass",
    "full_pass",
)


@dataclass(frozen=True)
class QuadratureDiagnosticConfig:
    mode: str
    candidate_grids: tuple[tuple[int, int], ...]
    reference_grid: tuple[int, int]
    anchor_grid: tuple[int, int]
    subset_trial_indices: tuple[int, ...]
    full_calibration: bool
    progress_every: int

    def validate(self) -> None:
        if self.mode not in {"smoke", "diagnostic"}:
            raise ValueError("mode must be smoke or diagnostic")
        if not self.candidate_grids:
            raise ValueError("candidate_grids must not be empty")
        grids = self.candidate_grids + (
            self.reference_grid,
            self.anchor_grid,
        )
        if len(grids) != len(set(grids)):
            raise ValueError("quadrature grids must be unique")
        if CURRENT_GRID not in self.candidate_grids:
            raise ValueError("candidate grids must include the current 5x16 grid")
        if any(amplitude < 2 or phase < 4 for amplitude, phase in grids):
            raise ValueError("quadrature grids are too small")
        if not self.subset_trial_indices or min(self.subset_trial_indices) < 0:
            raise ValueError("subset_trial_indices must be non-negative")
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")


CONFIGS = {
    "smoke": QuadratureDiagnosticConfig(
        mode="smoke",
        candidate_grids=((5, 16), (5, 32), (7, 32)),
        reference_grid=(7, 64),
        anchor_grid=(9, 128),
        subset_trial_indices=(0,),
        full_calibration=True,
        progress_every=1,
    ),
    "diagnostic": QuadratureDiagnosticConfig(
        mode="diagnostic",
        candidate_grids=(
            (5, 16),
            (5, 32),
            (5, 64),
            (7, 16),
            (7, 32),
            (7, 64),
        ),
        reference_grid=(9, 128),
        anchor_grid=(11, 256),
        subset_trial_indices=(0,),
        full_calibration=True,
        progress_every=10,
    ),
}


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


def config_for_mode(mode: str) -> QuadratureDiagnosticConfig:
    try:
        config = CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported mode: {mode}") from exc
    config.validate()
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _source_protocol(
    source: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    manifest = _load_json(source / "manifest.json")
    gate = _load_json(source / "gate_decision.json")
    if gate.get("status") != "HOLD-NUMERICAL-QUADRATURE":
        raise RuntimeError(
            "source Stage1 result is not HOLD-NUMERICAL-QUADRATURE"
        )
    if manifest.get("gate_status") != gate.get("status"):
        raise RuntimeError("source manifest and gate status do not match")

    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict):
        raise ValueError("source manifest is missing source_files")
    non_runtime_mismatches = []
    for relative, expected in source_files.items():
        path = ROOT / str(relative)
        normalized = str(relative).replace("\\", "/")
        if not path.is_file():
            raise RuntimeError(f"source code file is missing: {relative}")
        actual = _sha256(path)
        if actual == str(expected):
            continue
        if normalized in RUNTIME_SOURCE_FILES:
            raise RuntimeError(
                f"source code fingerprint mismatch: {relative}"
            )
        if normalized not in NON_RUNTIME_SOURCE_FILES:
            raise RuntimeError(
                f"unclassified source code fingerprint mismatch: {relative}"
            )
        non_runtime_mismatches.append(
            {
                "file": normalized,
                "expected_sha256": str(expected),
                "actual_sha256": actual,
            }
        )

    pools = pd.read_csv(source / "candidate_pools.csv")
    plans = pd.read_csv(source / "frozen_one_action_plans.csv")
    fingerprint_audit = {
        "runtime_dependency_fingerprints_match": True,
        "runtime_dependency_files": sorted(RUNTIME_SOURCE_FILES),
        "non_runtime_source_mismatches": non_runtime_mismatches,
    }
    return manifest, gate, pools, plans, fingerprint_audit


def _generate_trials(
    seeds: tuple[int, ...],
    trials_per_seed: int,
    waveform: WaveformConfig,
) -> dict[tuple[int, int], StructuredTrial]:
    trials: dict[tuple[int, int], StructuredTrial] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for trial_index in range(trials_per_seed):
            trials[(seed, trial_index)] = generate_structured_trial(
                rng,
                waveform,
            )
    return trials


def _tuple_of_ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(int(item) for item in value)


def _tuple_of_floats(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(float(item) for item in value)


def _source_config(manifest: dict[str, object]) -> dict[str, object]:
    raw = manifest.get("config")
    if not isinstance(raw, dict):
        raise ValueError("source manifest is missing config")
    required = {
        "snr_db",
        "prior_seeds",
        "calibration_seeds",
        "prior_trials_per_seed",
        "calibration_trials_per_seed",
        "tau_step",
    }
    if not required.issubset(raw):
        raise ValueError("source config is incomplete")
    return raw


def _parse_bins(text: object) -> np.ndarray:
    values = [int(item) for item in str(text).split()]
    bins = np.asarray(values, dtype=int)
    if bins.ndim != 1 or bins.size != 12:
        raise ValueError("source candidate pool must contain 12 bins")
    return bins


def _rebuild_inputs(
    *,
    source: Path,
    manifest: dict[str, object],
    gate: dict[str, object],
    pool_table: pd.DataFrame,
    waveform: WaveformConfig,
) -> tuple[
    PoolName,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], StructuredTrial],
    tuple[float, ...],
    float,
]:
    raw = _source_config(manifest)
    selected_pool = str(gate.get("score_pool_selected_on_calibration"))
    if selected_pool not in {
        "Current-Fisher",
        "Schur-Fisher",
        "Global-Risk",
        "Local-Global-Pareto",
    }:
        raise ValueError("source selected pool is invalid")

    rebuilt_pools = build_candidate_pools(waveform)
    source_row = pool_table[pool_table["pool"].eq(selected_pool)]
    if len(source_row) != 1:
        raise ValueError("selected pool is missing from source table")
    source_bins = _parse_bins(source_row.iloc[0]["candidate_bins"])
    rebuilt_bins = rebuilt_pools[selected_pool]
    if not np.array_equal(source_bins, rebuilt_bins):
        raise RuntimeError("candidate pool reconstruction does not match source")

    prior_seeds = _tuple_of_ints(raw["prior_seeds"], "prior_seeds")
    prior_count = int(raw["prior_trials_per_seed"])
    prior_trials = _generate_trials(prior_seeds, prior_count, waveform)
    power, _, _ = estimate_candidate_statistics(
        list(prior_trials.values()),
        rebuilt_bins,
    )
    prior_table = pd.read_csv(source / "prior_statistics.csv")
    prior_row = prior_table[prior_table["pool"].eq(selected_pool)]
    if len(prior_row) != 1:
        raise ValueError("selected prior statistics are missing")
    if not math.isclose(
        float(np.min(power)),
        float(prior_row.iloc[0]["source_power_min"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(np.max(power)),
        float(prior_row.iloc[0]["source_power_max"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("prior power reconstruction does not match source")

    calibration_seeds = _tuple_of_ints(
        raw["calibration_seeds"],
        "calibration_seeds",
    )
    calibration_count = int(raw["calibration_trials_per_seed"])
    calibration_trials = _generate_trials(
        calibration_seeds,
        calibration_count,
        waveform,
    )
    snr_db = _tuple_of_floats(raw["snr_db"], "snr_db")
    tau_step = float(raw["tau_step"])
    return (
        selected_pool,  # type: ignore[return-value]
        rebuilt_bins,
        power,
        calibration_trials,
        snr_db,
        tau_step,
    )


def _frozen_positions(
    plans: pd.DataFrame,
    selected_pool: PoolName,
    snr_db: tuple[float, ...],
) -> dict[float, int]:
    selected = plans[
        plans["plan"].eq("SNR-only-OneAcquire")
        & plans["pool"].eq(selected_pool)
        & plans["likelihood"].eq("Linear-Marginal")
    ]
    positions: dict[float, int] = {}
    for value in snr_db:
        row = selected[np.isclose(selected["snr_db"], value)]
        if len(row) != 1:
            raise RuntimeError(f"missing frozen action at SNR={value:g}")
        positions[value] = int(row.iloc[0]["position"])
    return positions


def _models_for_grids(
    grids: tuple[tuple[int, int], ...],
    *,
    waveform: WaveformConfig,
    bins: np.ndarray,
    source_power: np.ndarray,
    tau_step: float,
) -> dict[tuple[int, int], object]:
    models = {}
    for amplitude_nodes, phase_nodes in grids:
        cases = build_likelihood_cases(
            waveform=waveform,
            bins=bins,
            source_power=source_power,
            tau_step=tau_step,
            amplitude_nodes=amplitude_nodes,
            phase_nodes=phase_nodes,
        )
        models[(amplitude_nodes, phase_nodes)] = cases["Linear-Marginal"]
    return models


def _evaluate_grid(
    *,
    trial: StructuredTrial,
    seed: int,
    trial_index: int,
    snr_db: float,
    bins: np.ndarray,
    source_power: np.ndarray,
    position: int,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    model: object,
) -> dict[str, object]:
    y_a, y_b, sigma2 = observe_trial(trial, snr_db)
    scale = physical_public_scale(source_power, sigma2)
    template = linear_template(y_a, bins, model.tau_values, waveform)
    context = linear_likelihood_context(
        template=template,
        sigma2=sigma2,
        scale=scale,
        model=model,
    )
    observation = observation_from_context(
        stage="quadrature_diagnostic",
        seed=seed,
        trial=trial_index,
        snr_db=snr_db,
        true_tau=trial.true_tau,
        true_rho=trial.true_rho,
        y_b_selected=candidate_fft_values(y_b, bins),
        scale=scale,
        context=context,
        quantizer=quantizer,
    )
    levels = action_levels(model, position)
    result = evaluate_fixed(
        observation,
        method="quadrature_diagnostic",
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
    posterior = tau_posterior(state, model)
    return {
        "estimate_tau": float(result.estimate),
        "posterior_risk": float(result.posterior_risk),
        "tau_posterior": posterior,
    }


def _metric_row(
    *,
    seed: int,
    trial_index: int,
    snr_db: float,
    grid: tuple[int, int],
    estimate_tau: float,
    posterior_risk: float,
    posterior: np.ndarray,
    reference_grid: tuple[int, int],
    reference: dict[str, object],
) -> dict[str, object]:
    reference_posterior = np.asarray(
        reference["tau_posterior"],
        dtype=float,
    )
    return {
        "seed": seed,
        "trial": trial_index,
        "cluster_id": f"{seed}:{trial_index}",
        "snr_db": snr_db,
        "amplitude_nodes": grid[0],
        "phase_nodes": grid[1],
        "n_rho": grid[0] * grid[1],
        "estimate_tau": estimate_tau,
        "posterior_risk": posterior_risk,
        "reference_amplitude_nodes": reference_grid[0],
        "reference_phase_nodes": reference_grid[1],
        "estimate_abs_diff_vs_reference": abs(
            estimate_tau - float(reference["estimate_tau"])
        ),
        "risk_abs_diff_vs_reference": abs(
            posterior_risk - float(reference["posterior_risk"])
        ),
        "tau_posterior_l1_vs_reference": float(
            np.sum(np.abs(posterior - reference_posterior))
        ),
    }


def _summarize(
    rows: pd.DataFrame,
    *,
    estimate_tolerance: float,
    risk_tolerance: float,
    posterior_l1_tolerance: float,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records = []
    for key, group in rows.groupby(
        ["amplitude_nodes", "phase_nodes"],
        sort=True,
    ):
        est = group["estimate_abs_diff_vs_reference"].to_numpy(dtype=float)
        risk = group["risk_abs_diff_vs_reference"].to_numpy(dtype=float)
        l1 = group["tau_posterior_l1_vs_reference"].to_numpy(dtype=float)
        records.append(
            {
                "amplitude_nodes": int(key[0]),
                "phase_nodes": int(key[1]),
                "n_rho": int(key[0] * key[1]),
                "n_conditions": len(group),
                "estimate_diff_median": float(np.median(est)),
                "estimate_diff_p95": float(np.quantile(est, 0.95)),
                "estimate_diff_max": float(np.max(est)),
                "risk_diff_median": float(np.median(risk)),
                "risk_diff_p95": float(np.quantile(risk, 0.95)),
                "risk_diff_max": float(np.max(risk)),
                "posterior_l1_median": float(np.median(l1)),
                "posterior_l1_p95": float(np.quantile(l1, 0.95)),
                "posterior_l1_max": float(np.max(l1)),
                "estimate_pass": bool(np.max(est) <= estimate_tolerance),
                "risk_pass": bool(np.max(risk) <= risk_tolerance),
                "posterior_l1_pass": bool(
                    np.max(l1) <= posterior_l1_tolerance
                ),
                "full_pass": bool(
                    np.max(est) <= estimate_tolerance
                    and np.max(risk) <= risk_tolerance
                    and np.max(l1) <= posterior_l1_tolerance
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["n_rho", "amplitude_nodes", "phase_nodes"],
        kind="stable",
    )


def select_smallest_converged(
    summary: pd.DataFrame,
    candidate_grids: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    ordered = ordered_converged_candidates(summary, candidate_grids)
    return ordered[0] if ordered else None


def ordered_converged_candidates(
    summary: pd.DataFrame,
    candidate_grids: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    allowed = set(candidate_grids)
    passing = summary[
        summary.apply(
            lambda row: (
                int(row["amplitude_nodes"]),
                int(row["phase_nodes"]),
            )
            in allowed,
            axis=1,
        )
        & summary["full_pass"].astype(bool)
    ]
    if passing.empty:
        return ()
    ordered = passing.sort_values(
        ["n_rho", "amplitude_nodes", "phase_nodes"],
        kind="stable",
    )
    return tuple(
        (int(row.amplitude_nodes), int(row.phase_nodes))
        for row in ordered.itertuples(index=False)
    )


def diagnostic_status(
    *,
    mode: str,
    reference_pass: bool,
    subset_candidate_count: int,
    selected_grid: tuple[int, int] | None,
) -> str:
    if mode == "smoke":
        return "SMOKE_ONLY"
    if not reference_pass:
        return "HOLD-REFERENCE-NOT-CONVERGED"
    if subset_candidate_count < 1:
        return "HOLD-NO-CANDIDATE-CONVERGED"
    if selected_grid is None:
        return "HOLD-FULL-CALIBRATION"
    return "QUADRATURE_DIAGNOSTIC_PASS"


def _subset_scan(
    *,
    config: QuadratureDiagnosticConfig,
    trials: dict[tuple[int, int], StructuredTrial],
    snr_db: tuple[float, ...],
    positions: dict[float, int],
    bins: np.ndarray,
    source_power: np.ndarray,
    waveform: WaveformConfig,
    tau_step: float,
    quantizer: NestedComplexQuantizer,
    output: Path,
) -> pd.DataFrame:
    grids = (
        config.candidate_grids
        + (config.reference_grid, config.anchor_grid)
    )
    models = _models_for_grids(
        grids,
        waveform=waveform,
        bins=bins,
        source_power=source_power,
        tau_step=tau_step,
    )
    selected_trials = [
        (key, trial)
        for key, trial in sorted(trials.items())
        if key[1] in config.subset_trial_indices
    ]
    if config.mode == "smoke":
        selected_trials = selected_trials[:1]
        snr_values = (max(snr_db),)
    else:
        snr_values = snr_db
    total = len(selected_trials) * len(snr_values)
    done = 0
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for value in snr_values:
        print(f"[subset] SNR={value:g} dB start", flush=True)
        for (seed, trial_index), trial in selected_trials:
            results = {}
            for grid in grids:
                results[grid] = _evaluate_grid(
                    trial=trial,
                    seed=seed,
                    trial_index=trial_index,
                    snr_db=value,
                    bins=bins,
                    source_power=source_power,
                    position=positions[value],
                    waveform=waveform,
                    quantizer=quantizer,
                    model=models[grid],
                )
            anchor = results[config.anchor_grid]
            for grid, result in results.items():
                rows.append(
                    _metric_row(
                        seed=seed,
                        trial_index=trial_index,
                        snr_db=value,
                        grid=grid,
                        estimate_tau=float(result["estimate_tau"]),
                        posterior_risk=float(result["posterior_risk"]),
                        posterior=np.asarray(
                            result["tau_posterior"],
                            dtype=float,
                        ),
                        reference_grid=config.anchor_grid,
                        reference=anchor,
                    )
                )
            done += 1
            if done % config.progress_every == 0 or done == total:
                elapsed = time.perf_counter() - started
                rate = elapsed / max(done, 1)
                print(
                    f"[subset] {done}/{total} elapsed={elapsed:.1f}s "
                    f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(
            output / "subset_grid_scan.partial.csv",
            index=False,
        )
    return pd.DataFrame(rows)


def _full_calibration(
    *,
    config: QuadratureDiagnosticConfig,
    candidate_grids: tuple[tuple[int, int], ...],
    trials: dict[tuple[int, int], StructuredTrial],
    snr_db: tuple[float, ...],
    positions: dict[float, int],
    bins: np.ndarray,
    source_power: np.ndarray,
    waveform: WaveformConfig,
    tau_step: float,
    quantizer: NestedComplexQuantizer,
    tolerances: dict[str, float],
    output: Path,
) -> tuple[
    pd.DataFrame,
    tuple[tuple[int, int], ...],
    tuple[int, int] | None,
]:
    grids = tuple(
        dict.fromkeys(
            (CURRENT_GRID, config.reference_grid) + candidate_grids
        )
    )
    models = _models_for_grids(
        grids,
        waveform=waveform,
        bins=bins,
        source_power=source_power,
        tau_step=tau_step,
    )
    trial_items = sorted(trials.items())
    if config.mode == "smoke":
        trial_items = trial_items[:1]
        snr_values = (max(snr_db),)
    else:
        snr_values = snr_db
    total = len(trial_items) * len(snr_values)
    rows: list[dict[str, object]] = []
    reference_cases: list[
        tuple[
            int,
            int,
            float,
            StructuredTrial,
            dict[str, object],
        ]
    ] = []

    done = 0
    started = time.perf_counter()
    for value in snr_values:
        print(f"[full-reference] SNR={value:g} dB start", flush=True)
        for (seed, trial_index), trial in trial_items:
            reference = _evaluate_grid(
                trial=trial,
                seed=seed,
                trial_index=trial_index,
                snr_db=value,
                bins=bins,
                source_power=source_power,
                position=positions[value],
                waveform=waveform,
                quantizer=quantizer,
                model=models[config.reference_grid],
            )
            current = _evaluate_grid(
                trial=trial,
                seed=seed,
                trial_index=trial_index,
                snr_db=value,
                bins=bins,
                source_power=source_power,
                position=positions[value],
                waveform=waveform,
                quantizer=quantizer,
                model=models[CURRENT_GRID],
            )
            reference_cases.append(
                (seed, trial_index, value, trial, reference)
            )
            for grid, result in (
                (CURRENT_GRID, current),
                (config.reference_grid, reference),
            ):
                rows.append(
                    _metric_row(
                        seed=seed,
                        trial_index=trial_index,
                        snr_db=value,
                        grid=grid,
                        estimate_tau=float(result["estimate_tau"]),
                        posterior_risk=float(result["posterior_risk"]),
                        posterior=np.asarray(
                            result["tau_posterior"],
                            dtype=float,
                        ),
                        reference_grid=config.reference_grid,
                        reference=reference,
                    )
                )
            done += 1
            if done % config.progress_every == 0 or done == total:
                elapsed = time.perf_counter() - started
                rate = elapsed / max(done, 1)
                print(
                    f"[full-reference] {done}/{total} "
                    f"elapsed={elapsed:.1f}s "
                    f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                    flush=True,
                )
    pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(
        output / "full_calibration.partial.csv",
        index=False,
    )

    attempted: list[tuple[int, int]] = []
    selected_grid: tuple[int, int] | None = None
    for grid in candidate_grids:
        attempted.append(grid)
        if grid == CURRENT_GRID:
            candidate_rows = pd.DataFrame(rows, columns=METRIC_COLUMNS)
            candidate_rows = candidate_rows[
                candidate_rows["amplitude_nodes"].eq(grid[0])
                & candidate_rows["phase_nodes"].eq(grid[1])
            ].copy()
        else:
            candidate_records: list[dict[str, object]] = []
            candidate_started = time.perf_counter()
            for index, (
                seed,
                trial_index,
                value,
                trial,
                reference,
            ) in enumerate(reference_cases, start=1):
                if index == 1 or (
                    index > 1
                    and value != reference_cases[index - 2][2]
                ):
                    print(
                        f"[full-{grid[0]}x{grid[1]}] "
                        f"SNR={value:g} dB start",
                        flush=True,
                    )
                result = _evaluate_grid(
                    trial=trial,
                    seed=seed,
                    trial_index=trial_index,
                    snr_db=value,
                    bins=bins,
                    source_power=source_power,
                    position=positions[value],
                    waveform=waveform,
                    quantizer=quantizer,
                    model=models[grid],
                )
                candidate_records.append(
                    _metric_row(
                        seed=seed,
                        trial_index=trial_index,
                        snr_db=value,
                        grid=grid,
                        estimate_tau=float(result["estimate_tau"]),
                        posterior_risk=float(result["posterior_risk"]),
                        posterior=np.asarray(
                            result["tau_posterior"],
                            dtype=float,
                        ),
                        reference_grid=config.reference_grid,
                        reference=reference,
                    )
                )
                if (
                    index % config.progress_every == 0
                    or index == len(reference_cases)
                ):
                    elapsed = time.perf_counter() - candidate_started
                    rate = elapsed / index
                    print(
                        f"[full-{grid[0]}x{grid[1]}] "
                        f"{index}/{len(reference_cases)} "
                        f"elapsed={elapsed:.1f}s avg={rate:.3f}s "
                        f"ETA={rate * (len(reference_cases) - index):.1f}s",
                        flush=True,
                    )
            candidate_rows = pd.DataFrame(
                candidate_records,
                columns=METRIC_COLUMNS,
            )
            rows.extend(candidate_records)

        candidate_summary = _summarize(
            candidate_rows,
            estimate_tolerance=tolerances["estimate"],
            risk_tolerance=tolerances["risk"],
            posterior_l1_tolerance=tolerances["posterior_l1"],
        )
        if len(candidate_summary) != 1:
            raise RuntimeError(
                f"full-calibration summary is invalid for {grid}"
            )
        candidate_pass = bool(candidate_summary.iloc[0]["full_pass"])
        print(
            f"[Full candidate] {grid[0]}x{grid[1]} "
            f"pass={candidate_pass}",
            flush=True,
        )
        pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(
            output / "full_calibration.partial.csv",
            index=False,
        )
        if candidate_pass:
            selected_grid = grid
            break

    return (
        pd.DataFrame(rows, columns=METRIC_COLUMNS),
        tuple(attempted),
        selected_grid,
    )


def _per_snr_summary(
    rows: pd.DataFrame,
    attempted_grids: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    records = []
    for grid in attempted_grids:
        selected = rows[
            rows["amplitude_nodes"].eq(grid[0])
            & rows["phase_nodes"].eq(grid[1])
        ]
        for value, group in selected.groupby("snr_db", sort=True):
            records.append(
                {
                    "amplitude_nodes": grid[0],
                    "phase_nodes": grid[1],
                    "n_rho": grid[0] * grid[1],
                    "snr_db": value,
                    "n_conditions": len(group),
                    "estimate_diff_max": float(
                        group["estimate_abs_diff_vs_reference"].max()
                    ),
                    "estimate_diff_p95": float(
                        group["estimate_abs_diff_vs_reference"].quantile(0.95)
                    ),
                    "risk_diff_max": float(
                        group["risk_abs_diff_vs_reference"].max()
                    ),
                    "risk_diff_p95": float(
                        group["risk_abs_diff_vs_reference"].quantile(0.95)
                    ),
                    "posterior_l1_max": float(
                        group["tau_posterior_l1_vs_reference"].max()
                    ),
                    "posterior_l1_p95": float(
                        group["tau_posterior_l1_vs_reference"].quantile(0.95)
                    ),
                }
            )
    return pd.DataFrame(records, columns=PER_SNR_COLUMNS)


def _plot_subset(
    summary: pd.DataFrame,
    *,
    tolerances: dict[str, float],
    output: Path,
) -> None:
    ordered = summary.sort_values("n_rho", kind="stable")
    labels = [
        f"{int(row.amplitude_nodes)}x{int(row.phase_nodes)}"
        for row in ordered.itertuples(index=False)
    ]
    x = np.arange(len(ordered))
    fields = (
        ("estimate_diff_max", "Estimate max difference (sample)", "estimate"),
        ("risk_diff_max", "Risk max difference (sample^2)", "risk"),
        ("posterior_l1_max", "TDOA posterior L1 max", "posterior_l1"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (field, ylabel, tolerance_name) in zip(
        axes,
        fields,
        strict=True,
    ):
        ax.plot(
            x,
            ordered[field],
            marker="o",
            linewidth=1.2,
            color="#2457A6",
        )
        ax.axhline(
            tolerances[tolerance_name],
            color="#B7352D",
            linestyle="--",
            linewidth=1.0,
            label="Registered tolerance",
        )
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_yscale("symlog", linthresh=max(tolerances[tolerance_name] / 10, 1e-12))
        ax.set_xlabel("Amplitude x phase nodes")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Calibration-subset quadrature convergence")
    fig.tight_layout()
    fig.savefig(output / "Fig1_Quadrature_Grid_Convergence.svg")
    plt.close(fig)


def _plot_full(
    per_snr: pd.DataFrame,
    *,
    tolerances: dict[str, float],
    output: Path,
) -> None:
    if per_snr.empty:
        return
    fields = (
        ("estimate_diff_max", "Estimate max difference (sample)", "estimate"),
        ("risk_diff_max", "Risk max difference (sample^2)", "risk"),
        ("posterior_l1_max", "TDOA posterior L1 max", "posterior_l1"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for ax, (field, ylabel, tolerance_name) in zip(
        axes,
        fields,
        strict=True,
    ):
        for (amplitude, phase), group in per_snr.groupby(
            ["amplitude_nodes", "phase_nodes"],
            sort=False,
        ):
            ax.plot(
                group["snr_db"],
                group[field],
                marker="o",
                linewidth=1.2,
                label=f"{int(amplitude)}x{int(phase)}",
            )
        ax.axhline(
            tolerances[tolerance_name],
            color="#B7352D",
            linestyle="--",
            linewidth=1.0,
            label="Registered tolerance",
        )
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Sequential full-calibration convergence")
    fig.tight_layout()
    fig.savefig(output / "Fig2_Full_Calibration_Convergence.svg")
    plt.close(fig)


def run(
    config: QuadratureDiagnosticConfig,
    *,
    source: Path,
    output: Path,
) -> dict[str, object]:
    config.validate()
    (
        manifest,
        source_gate,
        pool_table,
        plans,
        fingerprint_audit,
    ) = _source_protocol(source)
    mismatches = fingerprint_audit["non_runtime_source_mismatches"]
    if mismatches:
        print(
            "[Source fingerprint] runtime dependencies match; "
            "the non-imported Stage1 orchestrator changed after the source "
            "run and is recorded in recommendation.json.",
            flush=True,
        )
    waveform = WaveformConfig()
    waveform.validate()
    (
        selected_pool,
        bins,
        source_power,
        calibration_trials,
        snr_db,
        tau_step,
    ) = _rebuild_inputs(
        source=source,
        manifest=manifest,
        gate=source_gate,
        pool_table=pool_table,
        waveform=waveform,
    )
    positions = _frozen_positions(plans, selected_pool, snr_db)
    quantizer = NestedComplexQuantizer(max_bits=8, clip_abs=4.0)

    raw_tolerances = source_gate.get("quadrature_tolerance")
    if not isinstance(raw_tolerances, dict):
        raise ValueError("source gate is missing quadrature_tolerance")
    tolerances = {
        "estimate": float(raw_tolerances["estimate_abs_diff_max"]),
        "risk": float(raw_tolerances["posterior_risk_abs_diff_max"]),
        "posterior_l1": float(raw_tolerances["tau_posterior_l1_max"]),
    }

    subset_rows = _subset_scan(
        config=config,
        trials=calibration_trials,
        snr_db=snr_db,
        positions=positions,
        bins=bins,
        source_power=source_power,
        waveform=waveform,
        tau_step=tau_step,
        quantizer=quantizer,
        output=output,
    )
    subset_summary = _summarize(
        subset_rows,
        estimate_tolerance=tolerances["estimate"],
        risk_tolerance=tolerances["risk"],
        posterior_l1_tolerance=tolerances["posterior_l1"],
    )
    reference_row = subset_summary[
        subset_summary["amplitude_nodes"].eq(config.reference_grid[0])
        & subset_summary["phase_nodes"].eq(config.reference_grid[1])
    ]
    if len(reference_row) != 1:
        raise RuntimeError("reference quadrature summary is missing")
    reference_pass = bool(reference_row.iloc[0]["full_pass"])
    subset_candidates = (
        ordered_converged_candidates(
            subset_summary,
            config.candidate_grids,
        )
        if reference_pass
        else ()
    )
    smallest_subset_grid = subset_candidates[0] if subset_candidates else None

    full_rows = pd.DataFrame(columns=METRIC_COLUMNS)
    full_summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    per_snr = pd.DataFrame(columns=PER_SNR_COLUMNS)
    attempts = pd.DataFrame(columns=ATTEMPT_COLUMNS)
    attempted_grids: tuple[tuple[int, int], ...] = ()
    selected_grid: tuple[int, int] | None = None
    if (
        config.full_calibration
        and reference_pass
        and subset_candidates
    ):
        print(
            "[Selection] subset-converged order="
            + " -> ".join(
                f"{grid[0]}x{grid[1]}" for grid in subset_candidates
            ),
            flush=True,
        )
        (
            full_rows,
            attempted_grids,
            selected_grid,
        ) = _full_calibration(
            config=config,
            candidate_grids=subset_candidates,
            trials=calibration_trials,
            snr_db=snr_db,
            positions=positions,
            bins=bins,
            source_power=source_power,
            waveform=waveform,
            tau_step=tau_step,
            quantizer=quantizer,
            tolerances=tolerances,
            output=output,
        )
        full_summary = _summarize(
            full_rows,
            estimate_tolerance=tolerances["estimate"],
            risk_tolerance=tolerances["risk"],
            posterior_l1_tolerance=tolerances["posterior_l1"],
        )
        attempt_records = []
        for order, grid in enumerate(attempted_grids, start=1):
            row = full_summary[
                full_summary["amplitude_nodes"].eq(grid[0])
                & full_summary["phase_nodes"].eq(grid[1])
            ]
            if len(row) != 1:
                raise RuntimeError(
                    f"full-calibration summary is missing for {grid}"
                )
            values = row.iloc[0]
            attempt_records.append(
                {
                    "attempt_order": order,
                    "amplitude_nodes": grid[0],
                    "phase_nodes": grid[1],
                    "n_rho": grid[0] * grid[1],
                    "estimate_diff_max": values["estimate_diff_max"],
                    "risk_diff_max": values["risk_diff_max"],
                    "posterior_l1_max": values["posterior_l1_max"],
                    "estimate_pass": values["estimate_pass"],
                    "risk_pass": values["risk_pass"],
                    "posterior_l1_pass": values["posterior_l1_pass"],
                    "full_pass": values["full_pass"],
                }
            )
        attempts = pd.DataFrame(
            attempt_records,
            columns=ATTEMPT_COLUMNS,
        )
        per_snr = _per_snr_summary(full_rows, attempted_grids)

    status = diagnostic_status(
        mode=config.mode,
        reference_pass=reference_pass,
        subset_candidate_count=len(subset_candidates),
        selected_grid=selected_grid,
    )
    recommendation = {
        "status": status,
        "source_stage1": str(source),
        "source_gate_status": source_gate["status"],
        "source_fingerprint_audit": fingerprint_audit,
        "selected_pool": selected_pool,
        "current_grid": list(CURRENT_GRID),
        "reference_grid": list(config.reference_grid),
        "anchor_grid": list(config.anchor_grid),
        "reference_converged_on_subset": reference_pass,
        "smallest_subset_converged_grid": (
            list(smallest_subset_grid)
            if smallest_subset_grid is not None
            else None
        ),
        "subset_converged_grids_in_cost_order": [
            list(grid) for grid in subset_candidates
        ],
        "full_calibration_attempted_grids": [
            list(grid) for grid in attempted_grids
        ],
        "selected_grid": (
            list(selected_grid) if selected_grid is not None else None
        ),
        "selected_grid_converged_on_full_calibration": bool(
            selected_grid is not None
        ),
        "full_calibration_candidates_exhausted": bool(
            subset_candidates
            and selected_grid is None
            and attempted_grids == subset_candidates
        ),
        "stage1_rerun_allowed": bool(
            status == "QUADRATURE_DIAGNOSTIC_PASS"
        ),
        "tolerances_unchanged_from_source": tolerances,
        "interpretation": (
            "This diagnostic selects a numerical integration resolution only "
            "from source calibration data. It does not change the candidate "
            "pool, frozen action, RiskGreedy rule, statistical gates, or any "
            "historical result."
        ),
    }

    subset_rows.to_csv(output / "subset_grid_scan.csv", index=False)
    subset_summary.to_csv(output / "subset_grid_summary.csv", index=False)
    full_rows.to_csv(output / "full_calibration_rows.csv", index=False)
    full_summary.to_csv(output / "full_calibration_summary.csv", index=False)
    attempts.to_csv(
        output / "full_calibration_attempts.csv",
        index=False,
    )
    per_snr.to_csv(output / "full_calibration_by_snr.csv", index=False)
    (output / "recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_subset(subset_summary, tolerances=tolerances, output=output)
    _plot_full(per_snr, tolerances=tolerances, output=output)
    for partial in output.glob("*.partial.csv"):
        partial.unlink()
    return {
        "recommendation": recommendation,
        "subset_rows": subset_rows,
        "subset_summary": subset_summary,
        "full_rows": full_rows,
        "full_summary": full_summary,
        "attempts": attempts,
        "per_snr": per_snr,
    }


def _output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "运行结果"
        / "BRSR"
        / f"BRSR_ROUTE_QUADRATURE_{timestamp}_{mode}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "diagnostic"),
        default=None,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_STAGE1_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode or PYCHARM_RUN_MODE
    config = config_for_mode(mode)
    source = args.source.resolve()
    output = _output_dir(mode)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        tee = Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            print("BRSR Stage1 quadrature convergence diagnostic", flush=True)
            print(f"Mode: {mode}", flush=True)
            print(f"Source: {source}", flush=True)
            print(f"Output: {output}", flush=True)
            print(
                f"Config: {json.dumps(asdict(config), ensure_ascii=False)}",
                flush=True,
            )
            result = run(config, source=source, output=output)
            elapsed = time.perf_counter() - started
            recommendation = result["recommendation"]
            print(f"[Gate] {recommendation['status']}", flush=True)
            print(
                f"[Reference] converged="
                f"{recommendation['reference_converged_on_subset']}",
                flush=True,
            )
            print(
                "[Full attempted] "
                f"{recommendation['full_calibration_attempted_grids']}",
                flush=True,
            )
            print(
                f"[Selected] {recommendation['selected_grid']} "
                "full="
                f"{recommendation['selected_grid_converged_on_full_calibration']}",
                flush=True,
            )
            print(f"[Done] elapsed={elapsed:.1f}s", flush=True)

            manifest = {
                "experiment": "BRSR Stage1 quadrature convergence diagnostic",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": mode,
                "source_stage1": str(source),
                "source_manifest_sha256": _sha256(source / "manifest.json"),
                "source_gate_sha256": _sha256(source / "gate_decision.json"),
                "config": asdict(config),
                "recommendation": recommendation,
                "elapsed_seconds": elapsed,
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "source_files": {
                    str(path.relative_to(ROOT)): _sha256(path)
                    for path in (
                        Path(__file__).resolve(),
                        ROOT / "brsr_route_feasibility" / "feasibility.py",
                        ROOT / "brsr_route_feasibility" / "run_stage1.py",
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
