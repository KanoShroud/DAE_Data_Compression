"""Standalone PFRS theory gate for receiver-side spectral selection.

This script does not import or modify the project's training/evaluation pipeline.
It studies the asymmetric two-receiver model

    Y_A[k] = S[k] + N_A[k]
    Y_B[k] = rho * S[k] * exp(-j * omega[k] * tau) + N_B[k]

with unknown deterministic source coefficients S[k], unknown shared complex gain
rho, and a selected frequency set at receiver B.  The finite-delay objective is
the Monte Carlo error rate of a nuisance-profiled binary GLRT.  It is a design
risk, not a Bayesian Ziv-Zakai lower bound.

Run directly in PyCharm.  The confirmation modes keep mask selection and final
performance confirmation on disjoint noise seeds, and save paired trial errors
for the frozen PFRS and Zhai-Fisher masks.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYCHARM_RUN_MODE = "confirmation"


@dataclass(frozen=True)
class PFRSConfig:
    mode: str
    n_bins: int
    m_bins: int
    snr_db: tuple[float, ...]
    n_trials: int
    prior_width_samples: int
    eta_values: tuple[float, ...]
    primary_eta: float
    source_profile: str
    source_floor: float
    rho_abs: float
    rho_phase_rad: float
    seed: int
    local_snr_db: float
    local_h_values: tuple[float, ...]
    local_ratio_tolerance: float


@dataclass(frozen=True)
class ConfirmationConfig:
    mode: str
    base_mode: str
    selection_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    selection_trials: int
    confirmation_trials: int
    bootstrap_repetitions: int
    bootstrap_seed: int


CONFIGS = {
    "smoke": PFRSConfig(
        mode="smoke",
        n_bins=12,
        m_bins=4,
        snr_db=(-8.0, 0.0, 8.0),
        n_trials=120,
        prior_width_samples=6,
        eta_values=(0.70, 0.80, 0.90, 0.95, 1.00),
        primary_eta=0.90,
        source_profile="center_weighted",
        source_floor=0.15,
        rho_abs=0.85,
        rho_phase_rad=0.40,
        seed=20260716,
        local_snr_db=8.0,
        local_h_values=(1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2),
        local_ratio_tolerance=2e-3,
    ),
    "research": PFRSConfig(
        mode="research",
        n_bins=16,
        m_bins=4,
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0),
        n_trials=500,
        prior_width_samples=8,
        eta_values=(0.70, 0.80, 0.90, 0.95, 1.00),
        primary_eta=0.90,
        source_profile="center_weighted",
        source_floor=0.15,
        rho_abs=0.85,
        rho_phase_rad=0.40,
        seed=20260716,
        local_snr_db=10.0,
        local_h_values=(1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2),
        local_ratio_tolerance=2e-3,
    ),
}


CONFIRMATION_CONFIGS = {
    "confirmation_smoke": ConfirmationConfig(
        mode="confirmation_smoke",
        base_mode="smoke",
        selection_seeds=(20260801, 20260802),
        confirmation_seeds=(20261801, 20261802, 20261803),
        selection_trials=30,
        confirmation_trials=100,
        bootstrap_repetitions=200,
        bootstrap_seed=2026071601,
    ),
    "confirmation": ConfirmationConfig(
        mode="confirmation",
        base_mode="research",
        selection_seeds=(
            20260801,
            20260802,
            20260803,
            20260804,
            20260805,
            20260806,
        ),
        confirmation_seeds=(
            20261801,
            20261802,
            20261803,
            20261804,
            20261805,
            20261806,
            20261807,
            20261808,
            20261809,
            20261810,
            20261811,
            20261812,
        ),
        selection_trials=500,
        confirmation_trials=5000,
        bootstrap_repetitions=5000,
        bootstrap_seed=2026071601,
    ),
}


RUN_MODES = tuple(CONFIGS) + tuple(CONFIRMATION_CONFIGS)


@dataclass(frozen=True)
class SpectralModel:
    bin_indices: np.ndarray
    omega: np.ndarray
    power: np.ndarray
    source: np.ndarray
    rho: complex


@dataclass(frozen=True)
class NoiseBank:
    a_h0: np.ndarray
    b_h0: np.ndarray
    a_h1: np.ndarray
    b_h1: np.ndarray
    sigma_a2: float
    sigma_b2: float


def _mode_config(mode: str) -> PFRSConfig:
    if mode not in CONFIGS:
        raise ValueError(f"Unknown mode {mode!r}; choose from {tuple(CONFIGS)}")
    return CONFIGS[mode]


def _bin_indices(n_bins: int) -> np.ndarray:
    if n_bins < 4 or n_bins % 2 != 0:
        raise ValueError("n_bins must be an even integer >= 4")
    return np.arange(-n_bins // 2, n_bins // 2, dtype=int)


def _source_power(indices: np.ndarray, profile: str, floor: float) -> np.ndarray:
    x = indices / max(float(np.max(np.abs(indices))), 1.0)
    if profile == "flat":
        power = np.ones_like(x, dtype=float)
    elif profile == "center_weighted":
        envelope = np.exp(-0.5 * (x / 0.58) ** 2)
        ripple = 1.0 + 0.20 * np.cos(2.3 * np.pi * x + 0.35)
        power = floor + envelope * ripple
    else:
        raise ValueError(f"Unsupported source_profile={profile!r}")
    power = np.maximum(power, floor)
    return power / np.mean(power)


def build_model(config: PFRSConfig) -> SpectralModel:
    indices = _bin_indices(config.n_bins)
    omega = 2.0 * np.pi * indices / config.n_bins
    power = _source_power(indices, config.source_profile, config.source_floor)
    rng = np.random.default_rng(config.seed + 17)
    phases = rng.uniform(-np.pi, np.pi, size=config.n_bins)
    source = np.sqrt(power) * np.exp(1j * phases)
    rho = config.rho_abs * np.exp(1j * config.rho_phase_rad)
    return SpectralModel(indices, omega, power, source, rho)


def _complex_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    variance: float,
) -> np.ndarray:
    scale = math.sqrt(variance / 2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def build_noise_banks(
    config: PFRSConfig,
    model: SpectralModel,
    noise_seed: int | None = None,
) -> dict[float, NoiseBank]:
    banks: dict[float, NoiseBank] = {}
    rng_seed = config.seed + 101 if noise_seed is None else int(noise_seed)
    rng = np.random.default_rng(rng_seed)
    reference_power = float(np.mean(model.power))
    shape = (config.n_trials, config.n_bins)
    for snr_db in config.snr_db:
        variance = reference_power / (10.0 ** (snr_db / 10.0))
        banks[float(snr_db)] = NoiseBank(
            a_h0=_complex_noise(rng, shape, variance),
            b_h0=_complex_noise(rng, shape, variance),
            a_h1=_complex_noise(rng, shape, variance),
            b_h1=_complex_noise(rng, shape, variance),
            sigma_a2=variance,
            sigma_b2=variance,
        )
    return banks


def profiled_cost(
    y_a: np.ndarray,
    y_b: np.ndarray,
    omega: np.ndarray,
    tau: float,
    sigma_a2: float,
    sigma_b2: float,
) -> np.ndarray:
    """Return min_{rho,S} negative log-likelihood residual for each row.

    Eliminating S[k] gives
        sum |Y_B[k] - rho exp(-j omega_k tau) Y_A[k]|^2
        --------------------------------------------------- .
                   sigma_b^2 + |rho|^2 sigma_a^2

    Minimizing this generalized Rayleigh quotient over complex rho is the
    smallest eigenvalue of a 2x2 Hermitian generalized eigenproblem.  The
    closed form below vectorizes the calculation over Monte Carlo trials.
    """
    if y_a.shape != y_b.shape or y_a.ndim != 2:
        raise ValueError("y_a and y_b must have the same [trials, bins] shape")
    if y_a.shape[1] != omega.size:
        raise ValueError("omega length must match the selected-bin dimension")
    if sigma_a2 <= 0.0 or sigma_b2 <= 0.0:
        raise ValueError("noise variances must be positive")

    steered_a = y_a * np.exp(-1j * omega[None, :] * tau)
    a = np.sum(np.abs(y_b) ** 2, axis=1) / sigma_b2
    d = np.sum(np.abs(steered_a) ** 2, axis=1) / sigma_a2
    cross = -np.sum(np.conj(y_b) * steered_a, axis=1)
    cross /= math.sqrt(sigma_a2 * sigma_b2)
    discriminant = np.sqrt(np.maximum((a - d) ** 2 + 4.0 * np.abs(cross) ** 2, 0.0))
    values = 0.5 * (a + d - discriminant)
    return np.maximum(values.real, 0.0)


def _scalar_profiled_mean_distance(
    source: np.ndarray,
    rho: complex,
    omega: np.ndarray,
    h: float,
    sigma_a2: float,
    sigma_b2: float,
) -> float:
    y_a = source
    y_b = rho * source
    steered_a = y_a * np.exp(-1j * omega * h)
    cross = np.vdot(y_b, steered_a)
    g = np.array(
        [
            [np.vdot(y_b, y_b).real, -cross],
            [-np.conj(cross), np.vdot(steered_a, steered_a).real],
        ],
        dtype=np.complex128,
    )
    noise_metric = np.diag([sigma_b2, sigma_a2]).astype(np.complex128)
    return float(max(eigvalsh(g, noise_metric)[0].real, 0.0))


def fisher_information(
    subset: tuple[int, ...],
    model: SpectralModel,
    sigma_a2: float,
    sigma_b2: float,
) -> float:
    idx = np.asarray(subset, dtype=int)
    p = model.power[idx]
    omega = model.omega[idx]
    p_sum = float(np.sum(p))
    if p_sum <= 0.0:
        return 0.0
    omega_mean = float(np.sum(p * omega) / p_sum)
    centered_second_moment = float(np.sum(p * (omega - omega_mean) ** 2))
    effective_scale = abs(model.rho) ** 2 / (
        sigma_b2 + abs(model.rho) ** 2 * sigma_a2
    )
    return 2.0 * effective_scale * centered_second_moment


def ambiguity_period_samples(
    subset: tuple[int, ...],
    model: SpectralModel,
) -> float:
    active = [i for i in subset if model.power[i] > 1e-12]
    if len(active) < 2:
        return 0.0
    selected_bins = model.bin_indices[np.asarray(active, dtype=int)]
    differences = [abs(int(v - selected_bins[0])) for v in selected_bins[1:]]
    nonzero = [d for d in differences if d > 0]
    if not nonzero:
        return 0.0
    grid_gcd = reduce(math.gcd, [model.bin_indices.size, *nonzero])
    return float(model.bin_indices.size / grid_gcd)


def is_identifiable(
    subset: tuple[int, ...],
    model: SpectralModel,
    prior_width_samples: int,
) -> bool:
    return ambiguity_period_samples(subset, model) > float(prior_width_samples)


def _uniform_delay_weights(prior_width_samples: int) -> tuple[np.ndarray, np.ndarray]:
    h_values = np.arange(1, prior_width_samples, dtype=float)
    weights = h_values * (prior_width_samples - h_values) / prior_width_samples
    weights /= np.sum(weights)
    return h_values, weights


def _binary_error_samples(
    subset: tuple[int, ...],
    h: float,
    model: SpectralModel,
    bank: NoiseBank,
) -> np.ndarray:
    idx = np.asarray(subset, dtype=int)
    source = model.source[idx]
    omega = model.omega[idx]

    y_a_h0 = source[None, :] + bank.a_h0[:, idx]
    y_b_h0 = model.rho * source[None, :] + bank.b_h0[:, idx]
    cost_h0_given_h0 = profiled_cost(
        y_a_h0, y_b_h0, omega, 0.0, bank.sigma_a2, bank.sigma_b2
    )
    cost_h1_given_h0 = profiled_cost(
        y_a_h0, y_b_h0, omega, h, bank.sigma_a2, bank.sigma_b2
    )

    delayed_source = source * np.exp(-1j * omega * h)
    y_a_h1 = source[None, :] + bank.a_h1[:, idx]
    y_b_h1 = model.rho * delayed_source[None, :] + bank.b_h1[:, idx]
    cost_h1_given_h1 = profiled_cost(
        y_a_h1, y_b_h1, omega, h, bank.sigma_a2, bank.sigma_b2
    )
    cost_h0_given_h1 = profiled_cost(
        y_a_h1, y_b_h1, omega, 0.0, bank.sigma_a2, bank.sigma_b2
    )

    atol = 1e-12
    tie_h0 = np.isclose(
        cost_h1_given_h0, cost_h0_given_h0, rtol=0.0, atol=atol
    )
    tie_h1 = np.isclose(
        cost_h0_given_h1, cost_h1_given_h1, rtol=0.0, atol=atol
    )
    err_h0 = (cost_h1_given_h0 < cost_h0_given_h0).astype(float)
    err_h1 = (cost_h0_given_h1 < cost_h1_given_h1).astype(float)
    err_h0[tie_h0] = 0.5
    err_h1[tie_h1] = 0.5
    return np.concatenate([err_h0, err_h1])


def _binary_error_rate(
    subset: tuple[int, ...],
    h: float,
    model: SpectralModel,
    bank: NoiseBank,
) -> tuple[float, float]:
    errors = _binary_error_samples(subset, h, model, bank)
    probability = float(np.mean(errors))
    standard_error = math.sqrt(max(probability * (1.0 - probability), 0.0) / errors.size)
    return probability, standard_error


def evaluate_subset(
    subset: tuple[int, ...],
    config: PFRSConfig,
    model: SpectralModel,
    banks: dict[float, NoiseBank],
) -> dict[str, object]:
    h_values, h_weights = _uniform_delay_weights(config.prior_width_samples)
    risks: dict[float, float] = {}
    max_mc_se = 0.0
    for snr_db in config.snr_db:
        bank = banks[float(snr_db)]
        probabilities = []
        for h in h_values:
            probability, mc_se = _binary_error_rate(subset, h, model, bank)
            probabilities.append(probability)
            max_mc_se = max(max_mc_se, mc_se)
        risks[float(snr_db)] = float(np.dot(h_weights, probabilities))

    local_bank = banks[min(config.snr_db, key=lambda x: abs(x - config.local_snr_db))]
    local_fisher = fisher_information(
        subset, model, local_bank.sigma_a2, local_bank.sigma_b2
    )
    row: dict[str, object] = {
        "mask": " ".join(str(int(model.bin_indices[i])) for i in subset),
        "positions": " ".join(str(i) for i in subset),
        "identifiable": is_identifiable(subset, model, config.prior_width_samples),
        "ambiguity_period_samples": ambiguity_period_samples(subset, model),
        "fisher_information": local_fisher,
        "robust_max_risk": max(risks.values()),
        "mean_risk": float(np.mean(list(risks.values()))),
        "max_binary_mc_se": max_mc_se,
    }
    for snr_db, risk in risks.items():
        row[f"risk_snr_{snr_db:g}_db"] = risk
    return row


def _all_subsets(config: PFRSConfig) -> Iterable[tuple[int, ...]]:
    return itertools.combinations(range(config.n_bins), config.m_bins)


def _mask_from_positions(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split())


def _pareto_mask(metrics: pd.DataFrame) -> pd.Series:
    order = metrics.sort_values(
        ["fisher_information", "robust_max_risk"], ascending=[False, True]
    ).index
    best_risk = math.inf
    is_pareto = pd.Series(False, index=metrics.index)
    for idx in order:
        risk = float(metrics.at[idx, "robust_max_risk"])
        if risk < best_risk - 1e-12:
            is_pareto.at[idx] = True
            best_risk = risk
    return is_pareto


def _choose_uniform_mask(config: PFRSConfig) -> tuple[int, ...]:
    positions = np.linspace(0, config.n_bins - 1, config.m_bins)
    rounded = np.unique(np.rint(positions).astype(int))
    if rounded.size != config.m_bins:
        raise RuntimeError("Uniform mask construction produced duplicate positions")
    return tuple(int(v) for v in rounded)


def _choose_random_identifiable_mask(
    config: PFRSConfig,
    model: SpectralModel,
) -> tuple[int, ...]:
    rng = np.random.default_rng(config.seed + 909)
    for _ in range(1000):
        subset = tuple(sorted(rng.choice(config.n_bins, config.m_bins, replace=False)))
        if is_identifiable(subset, model, config.prior_width_samples):
            return subset
    raise RuntimeError("Could not draw an identifiable random mask")


def select_methods(
    metrics: pd.DataFrame,
    config: PFRSConfig,
    model: SpectralModel,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[int, ...]]]:
    identifiable = metrics[metrics["identifiable"]].copy()
    if identifiable.empty:
        raise RuntimeError("No identifiable masks exist under the configured delay prior")

    zhai_idx = identifiable["fisher_information"].idxmax()
    zhai_mask = _mask_from_positions(str(identifiable.at[zhai_idx, "positions"]))
    j_max = float(identifiable.at[zhai_idx, "fisher_information"])

    eta_rows = []
    eta_masks: dict[float, tuple[int, ...]] = {}
    for eta in config.eta_values:
        feasible = identifiable[
            identifiable["fisher_information"] >= eta * j_max - 1e-12
        ]
        chosen_idx = feasible.sort_values(
            ["robust_max_risk", "mean_risk", "fisher_information"],
            ascending=[True, True, False],
        ).index[0]
        chosen = feasible.loc[chosen_idx]
        eta_masks[float(eta)] = _mask_from_positions(str(chosen["positions"]))
        eta_rows.append(
            {
                "eta": eta,
                "mask": chosen["mask"],
                "positions": chosen["positions"],
                "fisher_ratio": float(chosen["fisher_information"]) / j_max,
                "robust_max_risk": chosen["robust_max_risk"],
                "mean_risk": chosen["mean_risk"],
                "ambiguity_period_samples": chosen["ambiguity_period_samples"],
            }
        )

    power_positions = tuple(
        sorted(np.argsort(model.power)[-config.m_bins :].astype(int).tolist())
    )
    methods = {
        "PFRS": eta_masks[float(config.primary_eta)],
        "Zhai-Fisher": zhai_mask,
        "Power": power_positions,
        "Uniform": _choose_uniform_mask(config),
        "Random": _choose_random_identifiable_mask(config, model),
    }
    lookup = metrics.set_index("positions", drop=False)
    summary_rows = []
    for method, subset in methods.items():
        key = " ".join(str(i) for i in subset)
        row = lookup.loc[key].to_dict()
        row["method"] = method
        row["fisher_ratio"] = float(row["fisher_information"]) / j_max
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    eta_sweep = pd.DataFrame(eta_rows)
    return summary, eta_sweep, methods


def _risk_columns(config: PFRSConfig) -> list[str]:
    return [f"risk_snr_{snr:g}_db" for snr in config.snr_db]


def _aggregate_selection_metrics(
    seed_metrics: pd.DataFrame,
    config: PFRSConfig,
) -> pd.DataFrame:
    risk_columns = _risk_columns(config)
    static_columns = [
        "mask",
        "positions",
        "identifiable",
        "ambiguity_period_samples",
        "fisher_information",
    ]
    static = seed_metrics.groupby("positions", sort=False)[static_columns].first()
    risks = seed_metrics.groupby("positions", sort=False)[risk_columns].mean()
    mc_se = seed_metrics.groupby("positions", sort=False)["max_binary_mc_se"].max()
    aggregate = static.join(risks).join(mc_se).reset_index(drop=True)
    aggregate["robust_max_risk"] = aggregate[risk_columns].max(axis=1)
    aggregate["mean_risk"] = aggregate[risk_columns].mean(axis=1)
    aggregate["pareto"] = _pareto_mask(aggregate)
    return aggregate


def _method_summary_from_masks(
    metrics: pd.DataFrame,
    methods: dict[str, tuple[int, ...]],
) -> pd.DataFrame:
    lookup = metrics.set_index("positions", drop=False)
    j_max = float(metrics.loc[metrics["identifiable"], "fisher_information"].max())
    rows = []
    for method, subset in methods.items():
        key = " ".join(str(i) for i in subset)
        row = lookup.loc[key].to_dict()
        row["method"] = method
        row["fisher_ratio"] = float(row["fisher_information"]) / j_max
        rows.append(row)
    return pd.DataFrame(rows)


def _run_selection_stage(
    base_config: PFRSConfig,
    confirmation_config: ConfirmationConfig,
    model: SpectralModel,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[int, ...]],
]:
    config = replace(base_config, n_trials=confirmation_config.selection_trials)
    all_rows = []
    stability_rows = []
    mask_count = math.comb(config.n_bins, config.m_bins)
    for seed_index, noise_seed in enumerate(confirmation_config.selection_seeds, start=1):
        print(
            f"[Selection] seed {seed_index}/{len(confirmation_config.selection_seeds)} "
            f"noise_seed={noise_seed}",
            flush=True,
        )
        banks = build_noise_banks(config, model, noise_seed=noise_seed)
        rows = []
        for subset in _all_subsets(config):
            row = evaluate_subset(subset, config, model, banks)
            row["selection_seed"] = noise_seed
            rows.append(row)
        if len(rows) != mask_count:
            raise RuntimeError("Selection seed did not evaluate every mask")
        seed_frame = pd.DataFrame(rows)
        seed_summary, _, _ = select_methods(seed_frame, config, model)
        selected = seed_summary.loc[seed_summary["method"] == "PFRS"].iloc[0]
        stability_rows.append(
            {
                "selection_seed": noise_seed,
                "selected_mask": selected["mask"],
                "positions": selected["positions"],
                "fisher_ratio": selected["fisher_ratio"],
                "robust_max_risk": selected["robust_max_risk"],
                "mean_risk": selected["mean_risk"],
            }
        )
        all_rows.extend(rows)

    seed_metrics = pd.DataFrame(all_rows)
    aggregate = _aggregate_selection_metrics(seed_metrics, config)
    _, eta_sweep, base_methods = select_methods(aggregate, config, model)

    j_max = float(aggregate.loc[aggregate["identifiable"], "fisher_information"].max())
    eta_feasible = aggregate[
        aggregate["identifiable"]
        & (aggregate["fisher_information"] >= config.primary_eta * j_max - 1e-12)
    ]
    mean_idx = eta_feasible.sort_values(
        ["mean_risk", "robust_max_risk", "fisher_information"],
        ascending=[True, True, False],
    ).index[0]
    eta_mean_mask = _mask_from_positions(str(aggregate.at[mean_idx, "positions"]))
    eta08_row = eta_sweep.loc[np.isclose(eta_sweep["eta"], 0.80)].iloc[0]
    eta08_mask = _mask_from_positions(str(eta08_row["positions"]))

    frozen_methods = {
        "PFRS": base_methods["PFRS"],
        "Zhai-Fisher": base_methods["Zhai-Fisher"],
        "Eta09-MeanRisk": eta_mean_mask,
        "PFRS-Eta08": eta08_mask,
        "Power": base_methods["Power"],
        "Uniform": base_methods["Uniform"],
        "Random": base_methods["Random"],
    }
    frozen_summary = _method_summary_from_masks(aggregate, frozen_methods)
    frozen_summary["risk_source"] = "selection_seed_average"
    stability = pd.DataFrame(stability_rows)
    stability["matches_frozen_pfrs"] = (
        stability["positions"]
        == " ".join(str(v) for v in frozen_methods["PFRS"])
    )
    return (
        seed_metrics,
        aggregate,
        stability,
        frozen_summary,
        eta_sweep,
        frozen_methods,
    )


def _bootstrap_confirmation(
    seed_snr: pd.DataFrame,
    methods: dict[str, tuple[int, ...]],
    config: PFRSConfig,
    confirmation_config: ConfirmationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = np.asarray(confirmation_config.confirmation_seeds, dtype=int)
    snrs = np.asarray(config.snr_db, dtype=float)
    n_seeds = seeds.size
    rng = np.random.default_rng(confirmation_config.bootstrap_seed)
    draw = rng.integers(0, n_seeds, size=(confirmation_config.bootstrap_repetitions, n_seeds))

    arrays: dict[str, np.ndarray] = {}
    summary_rows = []
    for method in methods:
        pivot = seed_snr.loc[seed_snr["method"] == method].pivot(
            index="confirmation_seed", columns="snr_db", values="prior_weighted_risk"
        )
        pivot = pivot.reindex(index=seeds, columns=snrs)
        if pivot.isna().any().any():
            raise RuntimeError(f"Incomplete confirmation matrix for {method}")
        values = pivot.to_numpy(dtype=float)
        arrays[method] = values
        boot_snr = values[draw].mean(axis=1)
        point_snr = values.mean(axis=0)
        for snr_index, snr_db in enumerate(snrs):
            low, high = np.quantile(boot_snr[:, snr_index], [0.025, 0.975])
            summary_rows.append(
                {
                    "method": method,
                    "metric": "snr_risk",
                    "snr_db": snr_db,
                    "risk": point_snr[snr_index],
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        boot_mean = boot_snr.mean(axis=1)
        boot_robust = boot_snr.max(axis=1)
        for metric, point, samples in (
            ("mean_risk", float(point_snr.mean()), boot_mean),
            ("robust_max_risk", float(point_snr.max()), boot_robust),
        ):
            low, high = np.quantile(samples, [0.025, 0.975])
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "snr_db": math.nan,
                    "risk": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    pfrs = arrays["PFRS"]
    zhai = arrays["Zhai-Fisher"]
    boot_pfrs = pfrs[draw].mean(axis=1)
    boot_zhai = zhai[draw].mean(axis=1)
    point_gain_snr = (zhai - pfrs).mean(axis=0)
    boot_gain_snr = boot_zhai - boot_pfrs
    paired_rows = []
    for snr_index, snr_db in enumerate(snrs):
        samples = boot_gain_snr[:, snr_index]
        low, high = np.quantile(samples, [0.025, 0.975])
        paired_rows.append(
            {
                "metric": "snr_risk_gain",
                "snr_db": snr_db,
                "gain_zhai_minus_pfrs": point_gain_snr[snr_index],
                "ci_low": low,
                "ci_high": high,
            }
        )
    point_pfrs_snr = pfrs.mean(axis=0)
    point_zhai_snr = zhai.mean(axis=0)
    derived = (
        (
            "mean_risk_gain",
            float(point_zhai_snr.mean() - point_pfrs_snr.mean()),
            boot_zhai.mean(axis=1) - boot_pfrs.mean(axis=1),
        ),
        (
            "robust_max_risk_gain",
            float(point_zhai_snr.max() - point_pfrs_snr.max()),
            boot_zhai.max(axis=1) - boot_pfrs.max(axis=1),
        ),
    )
    for metric, point, samples in derived:
        low, high = np.quantile(samples, [0.025, 0.975])
        paired_rows.append(
            {
                "metric": metric,
                "snr_db": math.nan,
                "gain_zhai_minus_pfrs": point,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def _paired_by_delay_summary(
    by_h: pd.DataFrame,
    confirmation_config: ConfirmationConfig,
) -> pd.DataFrame:
    pivot = by_h.loc[by_h["method"].isin(["PFRS", "Zhai-Fisher"])].pivot(
        index=["confirmation_seed", "snr_db", "h_samples"],
        columns="method",
        values="risk",
    )
    if pivot.isna().any().any():
        raise RuntimeError("Incomplete PFRS/Zhai paired delay-risk table")
    pivot = pivot.reset_index()
    pivot["gain_zhai_minus_pfrs"] = pivot["Zhai-Fisher"] - pivot["PFRS"]
    rng = np.random.default_rng(confirmation_config.bootstrap_seed + 31)
    rows = []
    for (snr_db, h_samples), group in pivot.groupby(
        ["snr_db", "h_samples"], sort=True
    ):
        values = group["gain_zhai_minus_pfrs"].to_numpy(dtype=float)
        draw = rng.integers(
            0,
            values.size,
            size=(confirmation_config.bootstrap_repetitions, values.size),
        )
        samples = values[draw].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "snr_db": snr_db,
                "h_samples": h_samples,
                "pfrs_risk": float(group["PFRS"].mean()),
                "zhai_risk": float(group["Zhai-Fisher"].mean()),
                "gain_zhai_minus_pfrs": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _run_confirmation_stage(
    base_config: PFRSConfig,
    confirmation_config: ConfirmationConfig,
    model: SpectralModel,
    methods: dict[str, tuple[int, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    config = replace(base_config, n_trials=confirmation_config.confirmation_trials)
    h_values, h_weights = _uniform_delay_weights(config.prior_width_samples)
    n_error_samples = 2 * config.n_trials
    pair_shape = (
        len(confirmation_config.confirmation_seeds),
        len(config.snr_db),
        len(h_values),
        n_error_samples,
    )
    paired_arrays = {
        "pfrs_error2": np.empty(pair_shape, dtype=np.uint8),
        "zhai_error2": np.empty(pair_shape, dtype=np.uint8),
        "confirmation_seeds": np.asarray(
            confirmation_config.confirmation_seeds, dtype=np.int64
        ),
        "snr_db": np.asarray(config.snr_db, dtype=float),
        "h_samples": h_values,
        "h_weights": h_weights,
    }
    by_h_rows = []
    seed_snr_rows = []
    for seed_index, noise_seed in enumerate(confirmation_config.confirmation_seeds):
        print(
            f"[Confirmation] seed {seed_index + 1}/"
            f"{len(confirmation_config.confirmation_seeds)} noise_seed={noise_seed}",
            flush=True,
        )
        banks = build_noise_banks(config, model, noise_seed=noise_seed)
        for snr_index, snr_db in enumerate(config.snr_db):
            bank = banks[float(snr_db)]
            method_h_risks = {method: [] for method in methods}
            for h_index, h in enumerate(h_values):
                cache: dict[tuple[int, ...], np.ndarray] = {}
                for method, subset in methods.items():
                    if subset not in cache:
                        cache[subset] = _binary_error_samples(subset, h, model, bank)
                    errors = cache[subset]
                    risk = float(np.mean(errors))
                    se = float(np.std(errors, ddof=1) / math.sqrt(errors.size))
                    method_h_risks[method].append(risk)
                    by_h_rows.append(
                        {
                            "confirmation_seed": noise_seed,
                            "snr_db": snr_db,
                            "h_samples": h,
                            "h_weight": h_weights[h_index],
                            "method": method,
                            "risk": risk,
                            "trial_standard_error": se,
                        }
                    )
                pfrs_errors = cache[methods["PFRS"]]
                zhai_errors = cache[methods["Zhai-Fisher"]]
                paired_arrays["pfrs_error2"][seed_index, snr_index, h_index] = np.rint(
                    2.0 * pfrs_errors
                ).astype(np.uint8)
                paired_arrays["zhai_error2"][seed_index, snr_index, h_index] = np.rint(
                    2.0 * zhai_errors
                ).astype(np.uint8)

            for method, risks in method_h_risks.items():
                seed_snr_rows.append(
                    {
                        "confirmation_seed": noise_seed,
                        "snr_db": snr_db,
                        "method": method,
                        "prior_weighted_risk": float(np.dot(h_weights, risks)),
                    }
                )
    return pd.DataFrame(by_h_rows), pd.DataFrame(seed_snr_rows), paired_arrays


def local_limit_diagnostics(
    methods: dict[str, tuple[int, ...]],
    config: PFRSConfig,
    model: SpectralModel,
) -> pd.DataFrame:
    variance = float(np.mean(model.power)) / (10.0 ** (config.local_snr_db / 10.0))
    rows = []
    for method in ("PFRS", "Zhai-Fisher", "Power"):
        subset = methods[method]
        idx = np.asarray(subset, dtype=int)
        fisher = fisher_information(subset, model, variance, variance)
        for h in config.local_h_values:
            distance = _scalar_profiled_mean_distance(
                model.source[idx],
                model.rho,
                model.omega[idx],
                h,
                variance,
                variance,
            )
            ratio = 2.0 * distance / (h * h * fisher) if fisher > 0.0 else math.nan
            rows.append(
                {
                    "method": method,
                    "h_samples": h,
                    "profiled_mean_distance": distance,
                    "fisher_information": fisher,
                    "local_ratio_2D_over_h2J": ratio,
                }
            )
    return pd.DataFrame(rows)


def _plot_pareto(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    j_max = float(metrics.loc[metrics["identifiable"], "fisher_information"].max())
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    valid = metrics[metrics["identifiable"]]
    ax.scatter(
        valid["fisher_information"] / j_max,
        valid["robust_max_risk"],
        s=14,
        color="#B6BDC6",
        alpha=0.55,
        linewidths=0,
        label="Identifiable masks",
    )
    frontier = valid[valid["pareto"]].sort_values("fisher_information")
    ax.plot(
        frontier["fisher_information"] / j_max,
        frontier["robust_max_risk"],
        color="#202124",
        linewidth=1.2,
        label="Pareto frontier",
    )
    colors = {
        "PFRS": "#D55E00",
        "Zhai-Fisher": "#0072B2",
        "Power": "#009E73",
        "Uniform": "#CC79A7",
        "Random": "#E69F00",
    }
    for row in summary.itertuples(index=False):
        ax.scatter(
            row.fisher_ratio,
            row.robust_max_risk,
            s=58,
            color=colors[row.method],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label=row.method,
        )
    ax.set_xlabel("Normalized profiled Fisher information")
    ax.set_ylabel("Worst-SNR finite-delay GLRT risk")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_risk_curves(
    summary: pd.DataFrame,
    config: PFRSConfig,
    output_path: Path,
) -> None:
    colors = {
        "PFRS": "#D55E00",
        "Zhai-Fisher": "#0072B2",
        "Power": "#009E73",
        "Uniform": "#CC79A7",
        "Random": "#E69F00",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _, row in summary.iterrows():
        values = [row[f"risk_snr_{snr:g}_db"] for snr in config.snr_db]
        ax.plot(
            config.snr_db,
            values,
            marker="o",
            markersize=4,
            linewidth=1.3,
            color=colors[str(row["method"])],
            label=str(row["method"]),
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Prior-weighted binary GLRT error risk")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_local_limit(local: pd.DataFrame, output_path: Path) -> None:
    colors = {"PFRS": "#D55E00", "Zhai-Fisher": "#0072B2", "Power": "#009E73"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, group in local.groupby("method", sort=False):
        ax.semilogx(
            group["h_samples"],
            group["local_ratio_2D_over_h2J"],
            marker="o",
            markersize=4,
            linewidth=1.3,
            color=colors[method],
            label=method,
        )
    ax.axhline(1.0, color="#202124", linewidth=1.0, linestyle="--", label="Local limit")
    ax.set_xlabel("Delay separation h (samples)")
    ax.set_ylabel("2D(h) / [h^2 J]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_confirmation_risk(
    summary: pd.DataFrame,
    config: PFRSConfig,
    output_path: Path,
) -> None:
    colors = {
        "PFRS": "#D55E00",
        "Zhai-Fisher": "#0072B2",
        "Eta09-MeanRisk": "#CC79A7",
        "PFRS-Eta08": "#E69F00",
        "Power": "#009E73",
        "Uniform": "#6A3D9A",
        "Random": "#6B7280",
    }
    snr_rows = summary.loc[summary["metric"] == "snr_risk"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for method, group in snr_rows.groupby("method", sort=False):
        group = group.sort_values("snr_db")
        yerr = np.vstack(
            [group["risk"] - group["ci_low"], group["ci_high"] - group["risk"]]
        )
        ax.errorbar(
            group["snr_db"],
            group["risk"],
            yerr=yerr,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            capsize=2.5,
            color=colors[method],
            label=method,
        )
    ax.set_xticks(config.snr_db)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Prior-weighted binary GLRT error risk")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7.5, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_paired_gain(paired: pd.DataFrame, output_path: Path) -> None:
    rows = paired.loc[paired["metric"] == "snr_risk_gain"].sort_values("snr_db")
    yerr = np.vstack(
        [
            rows["gain_zhai_minus_pfrs"] - rows["ci_low"],
            rows["ci_high"] - rows["gain_zhai_minus_pfrs"],
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.errorbar(
        rows["snr_db"],
        rows["gain_zhai_minus_pfrs"],
        yerr=yerr,
        marker="o",
        markersize=4,
        linewidth=1.3,
        capsize=3,
        color="#D55E00",
        label="Zhai-Fisher risk minus PFRS risk",
    )
    ax.axhline(0.0, color="#202124", linewidth=1.0, linestyle="--")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Paired risk gain (positive favors PFRS)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_paired_gain_by_delay(paired_by_delay: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, paired_by_delay["snr_db"].nunique())
    )
    for color, (snr_db, group) in zip(
        colors, paired_by_delay.groupby("snr_db", sort=True), strict=True
    ):
        group = group.sort_values("h_samples")
        ax.plot(
            group["h_samples"],
            group["gain_zhai_minus_pfrs"],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=color,
            label=f"{snr_db:g} dB",
        )
    ax.axhline(0.0, color="#202124", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Delay separation h (samples)")
    ax.set_ylabel("Paired risk gain (positive favors PFRS)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_selection_stability(stability: pd.DataFrame, output_path: Path) -> None:
    counts = stability["selected_mask"].value_counts().sort_values(ascending=True)
    fig_height = max(3.4, 0.42 * len(counts) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    colors = ["#D55E00" if mask == counts.index[-1] else "#9CA3AF" for mask in counts.index]
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of selection seeds")
    ax.set_ylabel("Selected PFRS mask")
    ax.set_xlim(0.0, max(float(counts.max()) + 0.8, 1.0))
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _validate_confirmation(
    seed_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    frozen_summary: pd.DataFrame,
    by_h: pd.DataFrame,
    seed_snr: pd.DataFrame,
    paired_arrays: dict[str, np.ndarray],
    paired_by_delay: pd.DataFrame,
    paired: pd.DataFrame,
    base_config: PFRSConfig,
    confirmation_config: ConfirmationConfig,
) -> dict[str, object]:
    selection_seeds = set(confirmation_config.selection_seeds)
    confirmation_seeds = set(confirmation_config.confirmation_seeds)
    if selection_seeds & confirmation_seeds:
        raise RuntimeError("Selection and confirmation seeds must be disjoint")
    expected_selection_rows = math.comb(base_config.n_bins, base_config.m_bins) * len(
        selection_seeds
    )
    if len(seed_metrics) != expected_selection_rows:
        raise RuntimeError("Selection-stage mask count is incomplete")
    if len(stability) != len(selection_seeds):
        raise RuntimeError("Selection stability table does not cover every seed")
    if not np.isfinite(seed_metrics.select_dtypes(include=[np.number])).all().all():
        raise RuntimeError("Selection-stage metrics contain non-finite values")
    n_methods = int(frozen_summary["method"].nunique())
    expected_by_h_rows = (
        len(confirmation_seeds)
        * len(base_config.snr_db)
        * (base_config.prior_width_samples - 1)
        * n_methods
    )
    expected_seed_snr_rows = (
        len(confirmation_seeds) * len(base_config.snr_db) * n_methods
    )
    if len(by_h) != expected_by_h_rows:
        raise RuntimeError("Per-delay confirmation table is incomplete")
    if len(seed_snr) != expected_seed_snr_rows:
        raise RuntimeError("Per-seed/SNR confirmation table is incomplete")
    expected_paired_delay_rows = len(base_config.snr_db) * (
        base_config.prior_width_samples - 1
    )
    if len(paired_by_delay) != expected_paired_delay_rows:
        raise RuntimeError("Paired delay summary is incomplete")
    for frame_name, frame in (
        ("by_h", by_h),
        ("seed_snr", seed_snr),
        ("paired_by_delay", paired_by_delay),
        ("paired", paired),
    ):
        numeric = frame.select_dtypes(include=[np.number]).drop(
            columns=["snr_db"], errors="ignore"
        )
        if not np.isfinite(numeric.to_numpy()).all():
            raise RuntimeError(f"{frame_name} contains non-finite values")

    valid_codes = {0, 1, 2}
    for key in ("pfrs_error2", "zhai_error2"):
        codes = paired_arrays[key]
        if not set(np.unique(codes)).issubset(valid_codes):
            raise RuntimeError(f"Unexpected paired error code in {key}")
    pfrs_error = paired_arrays["pfrs_error2"].astype(float) / 2.0
    zhai_error = paired_arrays["zhai_error2"].astype(float) / 2.0
    expected_shape = (
        len(confirmation_config.confirmation_seeds),
        len(base_config.snr_db),
        base_config.prior_width_samples - 1,
        2 * confirmation_config.confirmation_trials,
    )
    if pfrs_error.shape != expected_shape or zhai_error.shape != expected_shape:
        raise RuntimeError("Paired trial-error tensor has the wrong shape")
    weights = paired_arrays["h_weights"]
    pfrs_recomputed = np.tensordot(pfrs_error.mean(axis=-1), weights, axes=([2], [0]))
    zhai_recomputed = np.tensordot(zhai_error.mean(axis=-1), weights, axes=([2], [0]))
    seeds = np.asarray(confirmation_config.confirmation_seeds)
    snrs = np.asarray(base_config.snr_db)
    for method, expected in (
        ("PFRS", pfrs_recomputed),
        ("Zhai-Fisher", zhai_recomputed),
    ):
        pivot = seed_snr.loc[seed_snr["method"] == method].pivot(
            index="confirmation_seed", columns="snr_db", values="prior_weighted_risk"
        )
        actual = pivot.reindex(index=seeds, columns=snrs).to_numpy(dtype=float)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    pfrs = frozen_summary.loc[frozen_summary["method"] == "PFRS"].iloc[0]
    fisher_ok = float(pfrs["fisher_ratio"]) + 1e-12 >= base_config.primary_eta
    mean_row = paired.loc[paired["metric"] == "mean_risk_gain"].iloc[0]
    robust_row = paired.loc[paired["metric"] == "robust_max_risk_gain"].iloc[0]
    snr_rows = paired.loc[paired["metric"] == "snr_risk_gain"]
    positive_snr_count = int((snr_rows["gain_zhai_minus_pfrs"] > 0.0).sum())
    required_positive = math.ceil(0.8 * len(base_config.snr_db))
    significant_degradation_count = int((snr_rows["ci_high"] < 0.0).sum())
    go = bool(
        fisher_ok
        and float(mean_row["ci_low"]) > 0.0
        and float(robust_row["ci_low"]) > 0.0
        and positive_snr_count >= required_positive
        and significant_degradation_count == 0
    )
    if go:
        status = "GO"
    elif float(mean_row["ci_high"]) <= 0.0 or float(robust_row["ci_high"]) <= 0.0:
        status = "NO-GO"
    else:
        status = "HOLD"
    return {
        "status": status,
        "selection_confirmation_seeds_disjoint": True,
        "selection_rows": int(len(seed_metrics)),
        "confirmation_by_h_rows": int(len(by_h)),
        "confirmation_seed_snr_rows": int(len(seed_snr)),
        "paired_by_delay_rows": int(len(paired_by_delay)),
        "paired_tensor_shape": list(expected_shape),
        "paired_error_reconstruction": "PASS",
        "pfrs_fisher_ratio": float(pfrs["fisher_ratio"]),
        "fisher_floor_pass": fisher_ok,
        "positive_snr_count": positive_snr_count,
        "required_positive_snr_count": required_positive,
        "significant_degradation_count": significant_degradation_count,
        "mean_risk_gain": float(mean_row["gain_zhai_minus_pfrs"]),
        "mean_risk_gain_ci": [float(mean_row["ci_low"]), float(mean_row["ci_high"])],
        "robust_risk_gain": float(robust_row["gain_zhai_minus_pfrs"]),
        "robust_risk_gain_ci": [
            float(robust_row["ci_low"]),
            float(robust_row["ci_high"]),
        ],
    }


def _json_ready_confirmation(config: ConfirmationConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in tuple(payload.items()):
        if isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def run_confirmation(
    confirmation_config: ConfirmationConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    base_config = CONFIGS[confirmation_config.base_mode]
    model = build_model(base_config)
    print("=" * 76, flush=True)
    print("PFRS independent selection/confirmation experiment", flush=True)
    print("=" * 76, flush=True)
    print(f"Mode: {confirmation_config.mode}", flush=True)
    print(
        f"Grid: N={base_config.n_bins}, M={base_config.m_bins}, "
        f"masks={math.comb(base_config.n_bins, base_config.m_bins)}",
        flush=True,
    )
    print(
        f"Selection: seeds={len(confirmation_config.selection_seeds)}, "
        f"trials/hypothesis={confirmation_config.selection_trials}",
        flush=True,
    )
    print(
        f"Confirmation: seeds={len(confirmation_config.confirmation_seeds)}, "
        f"trials/hypothesis={confirmation_config.confirmation_trials}",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)
    start = time.perf_counter()

    (
        seed_metrics,
        aggregate,
        stability,
        frozen_summary,
        eta_sweep,
        methods,
    ) = _run_selection_stage(base_config, confirmation_config, model)
    by_h, seed_snr, paired_arrays = _run_confirmation_stage(
        base_config, confirmation_config, model, methods
    )
    confirmation_summary, paired = _bootstrap_confirmation(
        seed_snr, methods, base_config, confirmation_config
    )
    paired_by_delay = _paired_by_delay_summary(by_h, confirmation_config)
    local = local_limit_diagnostics(methods, base_config, model)
    validation = _validate_confirmation(
        seed_metrics,
        stability,
        frozen_summary,
        by_h,
        seed_snr,
        paired_arrays,
        paired_by_delay,
        paired,
        base_config,
        confirmation_config,
    )

    seed_metrics.to_csv(
        output_dir / "pfrs_selection_seed_metrics.csv", index=False, encoding="utf-8-sig"
    )
    aggregate.to_csv(
        output_dir / "pfrs_selection_aggregate.csv", index=False, encoding="utf-8-sig"
    )
    stability.to_csv(
        output_dir / "pfrs_selection_stability.csv", index=False, encoding="utf-8-sig"
    )
    frozen_summary.to_csv(
        output_dir / "pfrs_frozen_methods.csv", index=False, encoding="utf-8-sig"
    )
    eta_sweep.to_csv(
        output_dir / "pfrs_confirmation_eta_sweep.csv", index=False, encoding="utf-8-sig"
    )
    by_h.to_csv(
        output_dir / "pfrs_confirmation_by_h.csv", index=False, encoding="utf-8-sig"
    )
    seed_snr.to_csv(
        output_dir / "pfrs_confirmation_by_seed_snr.csv", index=False, encoding="utf-8-sig"
    )
    confirmation_summary.to_csv(
        output_dir / "pfrs_confirmation_summary.csv", index=False, encoding="utf-8-sig"
    )
    paired.to_csv(
        output_dir / "pfrs_paired_comparison.csv", index=False, encoding="utf-8-sig"
    )
    paired_by_delay.to_csv(
        output_dir / "pfrs_paired_by_delay.csv", index=False, encoding="utf-8-sig"
    )
    local.to_csv(
        output_dir / "pfrs_confirmation_local_limit.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(output_dir / "pfrs_paired_trial_errors.npz", **paired_arrays)

    _plot_confirmation_risk(
        confirmation_summary, base_config, output_dir / "PFRS_Confirmation_Risk.svg"
    )
    _plot_paired_gain(paired, output_dir / "PFRS_Paired_Risk_Gain.svg")
    _plot_paired_gain_by_delay(
        paired_by_delay, output_dir / "PFRS_Paired_Gain_by_Delay.svg"
    )
    _plot_selection_stability(stability, output_dir / "PFRS_Selection_Stability.svg")

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "risk_semantics": (
            "Monte Carlo equal-prior binary error of a nuisance-profiled GLRT; "
            "this is a design-ensemble risk, not a Bayesian lower bound."
        ),
        "selection_confirmation_protocol": (
            "The mask is frozen using selection seeds and evaluated only on disjoint "
            "confirmation seeds. PFRS and Zhai errors share common random numbers."
        ),
        "base_config": _json_ready_config(base_config),
        "confirmation_config": _json_ready_confirmation(confirmation_config),
        "selected_masks": {
            name: [int(model.bin_indices[i]) for i in subset]
            for name, subset in methods.items()
        },
        "validation": validation,
        "elapsed_seconds": time.perf_counter() - start,
        "output_files": sorted(
            [path.name for path in output_dir.iterdir()]
            + ["pfrs_confirmation_gate.json", "pfrs_confirmation_manifest.json"]
        ),
    }
    (output_dir / "pfrs_confirmation_gate.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "pfrs_confirmation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nFrozen methods (selection-stage estimates)", flush=True)
    print(
        frozen_summary[
            ["method", "mask", "fisher_ratio", "robust_max_risk", "mean_risk"]
        ].to_string(index=False),
        flush=True,
    )
    print("\nPaired PFRS vs Zhai-Fisher confirmation", flush=True)
    print(paired.to_string(index=False), flush=True)
    print("\nAutomatic gate", flush=True)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    print(f"[Done] elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
    return manifest


def _json_ready_config(config: PFRSConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in tuple(payload.items()):
        if isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_results(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    local: pd.DataFrame,
    config: PFRSConfig,
) -> dict[str, object]:
    if len(metrics) != math.comb(config.n_bins, config.m_bins):
        raise RuntimeError("The exhaustive mask count is incomplete")
    numeric = metrics.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("Mask metrics contain non-finite values")
    if not metrics["robust_max_risk"].between(0.0, 1.0).all():
        raise RuntimeError("Finite-delay risk must lie in [0, 1]")
    pfrs = summary.loc[summary["method"] == "PFRS"].iloc[0]
    if not bool(pfrs["identifiable"]):
        raise RuntimeError("Selected PFRS mask is not identifiable over the delay prior")
    if float(pfrs["fisher_ratio"]) + 1e-12 < config.primary_eta:
        raise RuntimeError("Selected PFRS mask violates the configured Fisher floor")

    small_h = sorted(config.local_h_values)[:3]
    local_small = local[local["h_samples"].isin(small_h)]
    max_local_deviation = float(
        np.max(np.abs(local_small["local_ratio_2D_over_h2J"] - 1.0))
    )
    if max_local_deviation > config.local_ratio_tolerance:
        raise RuntimeError(
            "Profile-distance local limit failed: "
            f"max deviation {max_local_deviation:.3e} > "
            f"{config.local_ratio_tolerance:.3e}"
        )
    return {
        "mask_count": int(len(metrics)),
        "identifiable_mask_count": int(metrics["identifiable"].sum()),
        "pfrs_fisher_ratio": float(pfrs["fisher_ratio"]),
        "pfrs_robust_max_risk": float(pfrs["robust_max_risk"]),
        "max_small_h_local_ratio_deviation": max_local_deviation,
        "status": "PASS",
    }


def run(config: PFRSConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    model = build_model(config)
    banks = build_noise_banks(config, model)
    mask_count = math.comb(config.n_bins, config.m_bins)
    print("=" * 76, flush=True)
    print("PFRS standalone finite-delay selection", flush=True)
    print("=" * 76, flush=True)
    print(f"Mode: {config.mode}", flush=True)
    print(f"Grid: N={config.n_bins}, M={config.m_bins}, masks={mask_count}", flush=True)
    print(f"SNR: {list(config.snr_db)}, trials/hypothesis={config.n_trials}", flush=True)
    print(
        "Objective: nuisance-profiled finite-delay GLRT design risk "
        "(not a Bayesian ZZB)",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)

    start = time.perf_counter()
    rows = []
    progress_every = max(mask_count // 10, 1)
    for count, subset in enumerate(_all_subsets(config), start=1):
        rows.append(evaluate_subset(subset, config, model, banks))
        if count % progress_every == 0 or count == mask_count:
            elapsed = time.perf_counter() - start
            print(
                f"[Progress] {count}/{mask_count} masks | elapsed={elapsed:.1f}s",
                flush=True,
            )

    metrics = pd.DataFrame(rows)
    metrics["pareto"] = _pareto_mask(metrics)
    summary, eta_sweep, methods = select_methods(metrics, config, model)
    local = local_limit_diagnostics(methods, config, model)
    validation = _validate_results(metrics, summary, local, config)

    metrics_path = output_dir / "pfrs_mask_metrics.csv"
    summary_path = output_dir / "pfrs_method_summary.csv"
    eta_path = output_dir / "pfrs_eta_sweep.csv"
    local_path = output_dir / "pfrs_local_limit.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    eta_sweep.to_csv(eta_path, index=False, encoding="utf-8-sig")
    local.to_csv(local_path, index=False, encoding="utf-8-sig")

    _plot_pareto(metrics, summary, output_dir / "PFRS_Risk_Fisher_Pareto.svg")
    _plot_risk_curves(summary, config, output_dir / "PFRS_Risk_by_SNR.svg")
    _plot_local_limit(local, output_dir / "PFRS_Local_Limit.svg")

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "model_statement": (
            "Unknown deterministic common spectrum and unknown shared complex gain; "
            "receiver A observes the reference spectrum and receiver B sends selected bins."
        ),
        "risk_semantics": (
            "Monte Carlo equal-prior binary error of a nuisance-profiled GLRT; "
            "this is a design-ensemble risk, not a Bayesian lower bound."
        ),
        "config": _json_ready_config(config),
        "bin_indices": model.bin_indices.tolist(),
        "source_power": model.power.tolist(),
        "selected_masks": {
            name: [int(model.bin_indices[i]) for i in subset]
            for name, subset in methods.items()
        },
        "validation": validation,
        "elapsed_seconds": time.perf_counter() - start,
        "output_files": sorted(
            [path.name for path in output_dir.iterdir()] + ["pfrs_manifest.json"]
        ),
    }
    manifest_path = output_dir / "pfrs_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nMethod summary", flush=True)
    print(
        summary[
            [
                "method",
                "mask",
                "fisher_ratio",
                "robust_max_risk",
                "mean_risk",
                "ambiguity_period_samples",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nValidation", flush=True)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    print(f"[Done] elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone nuisance-profiled finite-delay spectral selection"
    )
    parser.add_argument("--mode", choices=RUN_MODES, default=PYCHARM_RUN_MODE)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if args.mode in CONFIRMATION_CONFIGS:
        if args.trials is not None:
            raise ValueError("--trials is only supported by smoke/research modes")
        confirmation_config = CONFIRMATION_CONFIGS[args.mode]
        if output_dir is None:
            output_dir = (
                PROJECT_ROOT
                / "运行结果"
                / "PFRS"
                / f"PFRS_{timestamp}_{confirmation_config.mode}"
            )
        run_confirmation(confirmation_config, output_dir.resolve())
        return

    config = _mode_config(args.mode)
    if args.trials is not None:
        if args.trials < 10:
            raise ValueError("--trials must be >= 10")
        config = replace(config, n_trials=args.trials)
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "运行结果"
            / "PFRS"
            / f"PFRS_{timestamp}_{config.mode}"
        )
    run(config, output_dir.resolve())


if __name__ == "__main__":
    main()
