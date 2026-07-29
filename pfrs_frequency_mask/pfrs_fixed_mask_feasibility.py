"""Exhaustive fixed-mask feasibility certificate for the PFRS route.

The script studies the frozen N=16, M=4 nuisance-profiled spectral model.  It
does not train a model or call the urban8/localization pipeline.  For every
four-bin mask it separates two properties:

1. local delay precision, measured by nuisance-eliminated Fisher information;
2. global ambiguity, measured by the strongest non-mainlobe maximum of the
   exact noiseless profiled-likelihood coherence over the continuous delay
   prior.

The result is a finite-space certificate.  It can establish whether another
fixed four-bin mask improves the PFRS ambiguity at the same Fisher floor.  It
cannot establish an impossibility result for adaptive, layered, or variable-
rate encoders.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pfrs_finite_delay_selection as pfrs


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
PYCHARM_RUN_MODE = "certificate"
PYCHARM_SOURCE_DIR = (
    PROJECT_ROOT
    / "运行结果"
    / "PFRS"
    / "PFRS_20260716_141935_confirmation"
)

PRIMARY_METHOD = "PFRS"
ZHAI_METHOD = "Zhai-Fisher"


@dataclass(frozen=True)
class CertificateConfig:
    mode: str
    ambiguity_grid_step_samples: float
    progress_every: int
    peak_tolerance: float


CONFIGS = {
    "smoke": CertificateConfig(
        mode="smoke",
        ambiguity_grid_step_samples=1.0 / 512.0,
        progress_every=400,
        peak_tolerance=1e-8,
    ),
    "certificate": CertificateConfig(
        mode="certificate",
        ambiguity_grid_step_samples=1.0 / 4096.0,
        progress_every=200,
        peak_tolerance=1e-8,
    ),
}


@dataclass(frozen=True)
class FrozenModel:
    source_dir: Path
    source_manifest: dict[str, object]
    base_config: pfrs.PFRSConfig
    model: pfrs.SpectralModel
    methods: dict[str, tuple[int, ...]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_config(config: pfrs.PFRSConfig) -> dict[str, object]:
    result = asdict(config)
    for key, value in tuple(result.items()):
        if isinstance(value, tuple):
            result[key] = list(value)
    return result


def _positions_from_bins(
    bins: tuple[int, ...], model: pfrs.SpectralModel
) -> tuple[int, ...]:
    lookup = {int(value): index for index, value in enumerate(model.bin_indices)}
    try:
        return tuple(lookup[int(value)] for value in bins)
    except KeyError as exc:
        raise RuntimeError(f"Frozen mask contains an unknown DFT bin: {exc}") from exc


def _load_frozen_model(source_dir: Path) -> FrozenModel:
    source_dir = source_dir.resolve()
    manifest_path = source_dir / "pfrs_confirmation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = manifest.get("validation", {})
    if validation.get("paired_error_reconstruction") != "PASS":
        raise RuntimeError("The frozen confirmation did not pass reconstruction")

    base_values = manifest.get("base_config")
    if not isinstance(base_values, dict):
        raise RuntimeError("Source manifest is missing base_config")
    base_mode = str(base_values.get("mode", ""))
    if base_mode not in pfrs.CONFIGS:
        raise RuntimeError(f"Unsupported frozen base mode: {base_mode!r}")
    base_config = pfrs.CONFIGS[base_mode]
    if _normalized_config(base_config) != base_values:
        raise RuntimeError("Local PFRS config differs from the frozen source config")

    source_script = Path(str(manifest["script"]))
    if not source_script.is_file():
        source_script = MODULE_ROOT / "pfrs_finite_delay_selection.py"
    if _sha256(source_script) != str(manifest["script_sha256"]):
        raise RuntimeError("Frozen PFRS source hash mismatch")

    model = pfrs.build_model(base_config)
    selected_masks = manifest.get("selected_masks")
    if not isinstance(selected_masks, dict):
        raise RuntimeError("Source manifest is missing selected_masks")
    methods = {
        str(name): _positions_from_bins(tuple(int(v) for v in bins), model)
        for name, bins in selected_masks.items()
    }
    for required in (PRIMARY_METHOD, ZHAI_METHOD, "Power"):
        if required not in methods:
            raise RuntimeError(f"Frozen source is missing method {required}")
    return FrozenModel(source_dir, manifest, base_config, model, methods)


def _continuous_period_samples(
    subset: tuple[int, ...], model: pfrs.SpectralModel
) -> float:
    selected = model.bin_indices[np.asarray(subset, dtype=int)]
    differences = [abs(int(value - selected[0])) for value in selected[1:]]
    nonzero = [value for value in differences if value > 0]
    if not nonzero:
        return 0.0
    return float(model.bin_indices.size / reduce(math.gcd, nonzero))


def _parabolic_extremum(
    x: np.ndarray, y: np.ndarray, index: int
) -> tuple[float, float]:
    if index <= 0 or index >= y.size - 1:
        return float(x[index]), float(y[index])
    left, middle, right = y[index - 1 : index + 2]
    denominator = left - 2.0 * middle + right
    scale = max(abs(left), abs(middle), abs(right), 1.0)
    if abs(denominator) <= np.finfo(float).eps * scale:
        return float(x[index]), float(y[index])
    offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    step = float(x[1] - x[0])
    x_peak = float(x[index] + offset * step)
    y_peak = float(middle - 0.25 * (left - right) * offset)
    return x_peak, y_peak


def _ambiguity_curve(
    subset: tuple[int, ...], model: pfrs.SpectralModel, h_grid: np.ndarray
) -> np.ndarray:
    idx = np.asarray(subset, dtype=int)
    power = model.power[idx]
    total_power = float(np.sum(power))
    if total_power <= 0.0:
        raise RuntimeError("Selected source power must be positive")
    phase = np.exp(-1j * model.omega[idx, None] * h_grid[None, :])
    return np.abs(np.sum(power[:, None] * phase, axis=0)) / total_power


def _ambiguity_metrics(
    subset: tuple[int, ...],
    model: pfrs.SpectralModel,
    h_grid: np.ndarray,
) -> dict[str, float | int]:
    ambiguity = _ambiguity_curve(subset, model, h_grid)
    minima = np.flatnonzero(
        (ambiguity[1:-1] <= ambiguity[:-2])
        & (ambiguity[1:-1] < ambiguity[2:])
    ) + 1
    if minima.size:
        first_minimum = int(minima[0])
        mainlobe_end, mainlobe_minimum = _parabolic_extremum(
            h_grid, ambiguity, first_minimum
        )
    else:
        first_minimum = ambiguity.size - 1
        mainlobe_end = float(h_grid[-1])
        mainlobe_minimum = float(ambiguity[-1])

    maxima = np.flatnonzero(
        (ambiguity[1:-1] >= ambiguity[:-2])
        & (ambiguity[1:-1] > ambiguity[2:])
    ) + 1
    sidelobe_indices = maxima[maxima > first_minimum]
    if first_minimum < ambiguity.size - 1:
        sidelobe_indices = np.unique(
            np.concatenate([sidelobe_indices, [ambiguity.size - 1]])
        )
    if sidelobe_indices.size:
        strongest_index = int(
            sidelobe_indices[np.argmax(ambiguity[sidelobe_indices])]
        )
        strongest_location, strongest_value = _parabolic_extremum(
            h_grid, ambiguity, strongest_index
        )
    else:
        strongest_location = math.nan
        strongest_value = 0.0

    if first_minimum < ambiguity.size - 1:
        side_h = h_grid[first_minimum:]
        side_power = ambiguity[first_minimum:] ** 2
        integrated_sidelobe = float(np.trapezoid(side_power, side_h))
        normalized_integrated = integrated_sidelobe / max(
            float(side_h[-1] - side_h[0]), np.finfo(float).eps
        )
    else:
        integrated_sidelobe = 0.0
        normalized_integrated = 0.0

    interior_sidelobes = sidelobe_indices[sidelobe_indices < ambiguity.size - 1]
    sidelobe_values = ambiguity[interior_sidelobes]
    return {
        "mainlobe_end_samples": mainlobe_end,
        "mainlobe_minimum": mainlobe_minimum,
        "strongest_sidelobe": strongest_value,
        "strongest_sidelobe_location_samples": strongest_location,
        "integrated_sidelobe_energy": integrated_sidelobe,
        "mean_sidelobe_energy": normalized_integrated,
        "sidelobe_count": int(interior_sidelobes.size),
        "sidelobe_count_gt_0p5": int(np.sum(sidelobe_values > 0.5)),
        "sidelobe_count_gt_0p8": int(np.sum(sidelobe_values > 0.8)),
        "sidelobe_count_gt_0p9": int(np.sum(sidelobe_values > 0.9)),
    }


def _closed_form_profiled_cost(
    ambiguity: np.ndarray,
    selected_power: float,
    rho_abs: float,
    sigma_a2: float,
    sigma_b2: float,
) -> np.ndarray:
    a = rho_abs**2 * selected_power / sigma_b2
    d = selected_power / sigma_a2
    cross = rho_abs * selected_power * ambiguity / math.sqrt(sigma_a2 * sigma_b2)
    return 0.5 * (a + d - np.sqrt((a - d) ** 2 + 4.0 * cross**2))


def _validate_profiled_ambiguity(frozen: FrozenModel) -> dict[str, float | str]:
    model = frozen.model
    h = np.asarray([0.0, 0.137, 0.5, 1.237, 2.75, 5.5, 7.91])
    variance = float(np.mean(model.power)) / (
        10.0 ** (frozen.base_config.local_snr_db / 10.0)
    )
    max_difference = 0.0
    for method in (PRIMARY_METHOD, ZHAI_METHOD, "Power"):
        subset = frozen.methods[method]
        idx = np.asarray(subset, dtype=int)
        y_a = np.broadcast_to(model.source[idx], (h.size, idx.size)).copy()
        y_b = np.broadcast_to(
            model.rho * model.source[idx], (h.size, idx.size)
        ).copy()
        scalar = np.asarray(
            [
                pfrs.profiled_cost(
                    y_a[row : row + 1],
                    y_b[row : row + 1],
                    model.omega[idx],
                    float(delay),
                    variance,
                    variance,
                )[0]
                for row, delay in enumerate(h)
            ]
        )
        ambiguity = _ambiguity_curve(subset, model, h)
        analytic = _closed_form_profiled_cost(
            ambiguity,
            float(np.sum(model.power[idx])),
            abs(model.rho),
            variance,
            variance,
        )
        max_difference = max(max_difference, float(np.max(np.abs(scalar - analytic))))
    if max_difference > 1e-10:
        raise RuntimeError("Analytic ambiguity does not match the profiled likelihood")
    return {
        "max_closed_form_profiled_cost_difference": max_difference,
        "status": "PASS",
    }


def _enumerate_masks(
    frozen: FrozenModel,
    config: CertificateConfig,
) -> pd.DataFrame:
    model = frozen.model
    base = frozen.base_config
    h_grid = np.arange(
        0.0,
        float(base.prior_width_samples) + 0.5 * config.ambiguity_grid_step_samples,
        config.ambiguity_grid_step_samples,
        dtype=float,
    )
    expected = math.comb(base.n_bins, base.m_bins)
    variance = float(np.mean(model.power)) / (10.0 ** (base.local_snr_db / 10.0))
    subsets = tuple(itertools.combinations(range(base.n_bins), base.m_bins))
    if len(subsets) != expected:
        raise RuntimeError("Unexpected mask count")
    fisher_values = np.asarray(
        [pfrs.fisher_information(mask, model, variance, variance) for mask in subsets]
    )
    fisher_max = float(np.max(fisher_values))
    rows = []
    started = time.perf_counter()
    for index, (subset, information) in enumerate(
        zip(subsets, fisher_values, strict=True), start=1
    ):
        selected_bins = tuple(int(model.bin_indices[position]) for position in subset)
        continuous_period = _continuous_period_samples(subset, model)
        ambiguity = _ambiguity_metrics(subset, model, h_grid)
        rows.append(
            {
                "mask": " ".join(str(value) for value in selected_bins),
                "positions": " ".join(str(value) for value in subset),
                "fisher_information": float(information),
                "fisher_ratio": float(information / fisher_max),
                "selected_power": float(np.sum(model.power[np.asarray(subset)])),
                "integer_grid_period_samples": pfrs.ambiguity_period_samples(
                    subset, model
                ),
                "continuous_period_samples": continuous_period,
                "closed_prior_identifiable": bool(
                    continuous_period > base.prior_width_samples + 1e-12
                ),
                "uniform_prior_ae_identifiable": bool(
                    continuous_period >= base.prior_width_samples - 1e-12
                ),
                **ambiguity,
            }
        )
        if index % config.progress_every == 0 or index == expected:
            print(
                f"[Progress] {index}/{expected} masks | "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    result = pd.DataFrame(rows)
    if len(result) != expected or not np.isfinite(
        result.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise RuntimeError("Mask enumeration produced invalid output")
    return result


def _mark_pareto(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    result["pareto_fisher_vs_sidelobe"] = False
    eligible = result[result["closed_prior_identifiable"]]
    for index, row in eligible.iterrows():
        dominated = eligible[
            (eligible["fisher_ratio"] >= row["fisher_ratio"] - 1e-12)
            & (
                eligible["strongest_sidelobe"]
                <= row["strongest_sidelobe"] + 1e-12
            )
            & (
                (eligible["fisher_ratio"] > row["fisher_ratio"] + 1e-12)
                | (
                    eligible["strongest_sidelobe"]
                    < row["strongest_sidelobe"] - 1e-12
                )
            )
        ]
        if dominated.empty:
            result.at[index, "pareto_fisher_vs_sidelobe"] = True
    return result


def _known_method_rows(
    metrics: pd.DataFrame, frozen: FrozenModel
) -> pd.DataFrame:
    lookup = metrics.set_index("positions", drop=False)
    rows = []
    for method, subset in frozen.methods.items():
        key = " ".join(str(value) for value in subset)
        if key not in lookup.index:
            raise RuntimeError(f"Enumerated metrics are missing {method}")
        row = lookup.loc[key].to_dict()
        row["method"] = method
        rows.append(row)
    return pd.DataFrame(rows)


def _floor_summary(
    metrics: pd.DataFrame,
    frozen: FrozenModel,
    config: CertificateConfig,
) -> pd.DataFrame:
    rows = []
    eligible = metrics[metrics["closed_prior_identifiable"]]
    for eta in frozen.base_config.eta_values:
        feasible = eligible[eligible["fisher_ratio"] >= eta - 1e-12]
        if feasible.empty:
            raise RuntimeError(f"No feasible masks at eta={eta}")
        best = feasible.sort_values(
            ["strongest_sidelobe", "mean_sidelobe_energy", "fisher_ratio"],
            ascending=[True, True, False],
        ).iloc[0]
        rows.append(
            {
                "eta": float(eta),
                "feasible_mask_count": int(len(feasible)),
                "best_mask": best["mask"],
                "best_fisher_ratio": float(best["fisher_ratio"]),
                "best_strongest_sidelobe": float(best["strongest_sidelobe"]),
                "best_sidelobe_location_samples": float(
                    best["strongest_sidelobe_location_samples"]
                ),
                "best_mean_sidelobe_energy": float(best["mean_sidelobe_energy"]),
            }
        )
    return pd.DataFrame(rows)


def _build_certificate(
    metrics: pd.DataFrame,
    known: pd.DataFrame,
    floors: pd.DataFrame,
    frozen: FrozenModel,
    config: CertificateConfig,
    numerical_validation: dict[str, float | str],
) -> dict[str, object]:
    primary_eta = float(frozen.base_config.primary_eta)
    pfrs_row = known.loc[known["method"] == PRIMARY_METHOD].iloc[0]
    feasible = metrics[
        metrics["closed_prior_identifiable"]
        & (metrics["fisher_ratio"] >= primary_eta - 1e-12)
    ]
    better = feasible[
        feasible["strongest_sidelobe"]
        < float(pfrs_row["strongest_sidelobe"]) - config.peak_tolerance
    ]
    dominators = feasible[
        (feasible["fisher_ratio"] >= float(pfrs_row["fisher_ratio"]) - 1e-12)
        & (
            feasible["strongest_sidelobe"]
            <= float(pfrs_row["strongest_sidelobe"]) + config.peak_tolerance
        )
        & (
            (feasible["fisher_ratio"] > float(pfrs_row["fisher_ratio"]) + 1e-12)
            | (
                feasible["strongest_sidelobe"]
                < float(pfrs_row["strongest_sidelobe"]) - config.peak_tolerance
            )
        )
    ]
    best_floor = floors.loc[np.isclose(floors["eta"], primary_eta)].iloc[0]
    pfrs_is_best = str(best_floor["best_mask"]) == str(pfrs_row["mask"])
    pfrs_is_pareto = bool(pfrs_row["pareto_fisher_vs_sidelobe"])
    saturated = pfrs_is_best and better.empty and dominators.empty and pfrs_is_pareto
    return {
        "route_status": (
            "FIXED_MASK_SATURATED" if saturated else "ALTERNATIVE_MASK_EXISTS"
        ),
        "recommendation": (
            "STOP_FIXED_MASK_TUNING_AND_PIVOT"
            if saturated
            else "VALIDATE_ONE_PREREGISTERED_CONTINUOUS_MASK"
        ),
        "scope": (
            "Finite N/M mask space under the frozen deterministic source, gain, "
            "noise, prior, and Fisher-floor model only"
        ),
        "primary_eta": primary_eta,
        "all_mask_count": int(len(metrics)),
        "closed_prior_identifiable_mask_count": int(
            metrics["closed_prior_identifiable"].sum()
        ),
        "feasible_mask_count_at_primary_eta": int(len(feasible)),
        "pfrs": {
            "mask": str(pfrs_row["mask"]),
            "fisher_ratio": float(pfrs_row["fisher_ratio"]),
            "strongest_sidelobe": float(pfrs_row["strongest_sidelobe"]),
            "strongest_sidelobe_location_samples": float(
                pfrs_row["strongest_sidelobe_location_samples"]
            ),
            "mainlobe_end_samples": float(pfrs_row["mainlobe_end_samples"]),
            "is_best_sidelobe_at_floor": pfrs_is_best,
            "is_pareto": pfrs_is_pareto,
        },
        "strictly_lower_sidelobe_masks_at_floor": int(len(better)),
        "masks_dominating_pfrs": int(len(dominators)),
        "best_mask_at_floor": str(best_floor["best_mask"]),
        "best_sidelobe_at_floor": float(best_floor["best_strongest_sidelobe"]),
        "numerical_validation": numerical_validation,
    }


def _plot_pareto(
    metrics: pd.DataFrame, known: pd.DataFrame, output_path: Path
) -> None:
    eligible = metrics[metrics["closed_prior_identifiable"]]
    pareto = eligible[eligible["pareto_fisher_vs_sidelobe"]].sort_values(
        "fisher_ratio"
    )
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    ax.scatter(
        eligible["fisher_ratio"],
        eligible["strongest_sidelobe"],
        s=13,
        color="#B8C2CC",
        alpha=0.45,
        label="All identifiable M=4 masks",
    )
    ax.plot(
        pareto["fisher_ratio"],
        pareto["strongest_sidelobe"],
        color="#111111",
        linewidth=1.2,
        marker=".",
        markersize=4,
        label="Finite-mask Pareto frontier",
    )
    colors = {
        "PFRS": "#D62728",
        "Zhai-Fisher": "#1F77B4",
        "Eta09-MeanRisk": "#FF7F0E",
        "Power": "#2CA02C",
        "PFRS-Eta08": "#9467BD",
        "Uniform": "#7F7F7F",
        "Random": "#17BECF",
    }
    for _, row in known.iterrows():
        method = str(row["method"])
        ax.scatter(
            row["fisher_ratio"],
            row["strongest_sidelobe"],
            s=54,
            color=colors.get(method, "#333333"),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            method,
            (row["fisher_ratio"], row["strongest_sidelobe"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0.9, color="#555555", linestyle="--", linewidth=1.0, label="eta=0.9")
    ax.set_xlabel("Nuisance-eliminated Fisher ratio (higher is better)")
    ax.set_ylabel("Strongest continuous sidelobe (lower is better)")
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.02, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_ambiguity_curves(
    frozen: FrozenModel,
    config: CertificateConfig,
    output_path: Path,
) -> None:
    h_grid = np.arange(
        0.0,
        float(frozen.base_config.prior_width_samples)
        + 0.5 * config.ambiguity_grid_step_samples,
        config.ambiguity_grid_step_samples,
    )
    styles = {
        "PFRS": ("#D62728", "-"),
        "Eta09-MeanRisk": ("#FF7F0E", "-"),
        "Zhai-Fisher": ("#1F77B4", "--"),
        "Power": ("#2CA02C", "-."),
        "Random": ("#17BECF", ":"),
        "Uniform": ("#7F7F7F", "--"),
    }
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for method in ("PFRS", "Eta09-MeanRisk", "Zhai-Fisher", "Power", "Random", "Uniform"):
        if method not in frozen.methods:
            continue
        color, linestyle = styles[method]
        ax.plot(
            h_grid,
            _ambiguity_curve(frozen.methods[method], frozen.model, h_grid),
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=method,
        )
    ax.set_xlabel("Delay separation |h| (samples)")
    ax.set_ylabel("Normalized profiled coherence")
    ax.set_xlim(0.0, float(frozen.base_config.prior_width_samples))
    ax.set_ylim(-0.02, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _validate_results(
    metrics: pd.DataFrame,
    known: pd.DataFrame,
    floors: pd.DataFrame,
    certificate: dict[str, object],
    frozen: FrozenModel,
) -> dict[str, object]:
    expected_masks = math.comb(
        frozen.base_config.n_bins, frozen.base_config.m_bins
    )
    if len(metrics) != expected_masks:
        raise RuntimeError("Certificate does not contain every fixed mask")
    if int(metrics["pareto_fisher_vs_sidelobe"].sum()) < 2:
        raise RuntimeError("Pareto frontier is unexpectedly empty")
    pfrs_row = known.loc[known["method"] == PRIMARY_METHOD].iloc[0]
    zhai_row = known.loc[known["method"] == ZHAI_METHOD].iloc[0]
    if not math.isclose(float(zhai_row["fisher_ratio"]), 1.0, abs_tol=1e-12):
        raise RuntimeError("Zhai-Fisher is not the local-information maximum")
    if str(pfrs_row["mask"]) != "-8 -5 6 7":
        raise RuntimeError("Frozen PFRS mask changed unexpectedly")
    primary = floors.loc[
        np.isclose(floors["eta"], frozen.base_config.primary_eta)
    ].iloc[0]
    if str(primary["best_mask"]) != str(certificate["best_mask_at_floor"]):
        raise RuntimeError("Floor summary disagrees with certificate")
    return {
        "expected_mask_count": expected_masks,
        "pareto_mask_count": int(metrics["pareto_fisher_vs_sidelobe"].sum()),
        "known_method_count": int(len(known)),
        "eta_count": int(len(floors)),
        "all_numeric_finite": bool(
            np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "status": "PASS",
    }


def run(config: CertificateConfig, source_dir: Path, output_dir: Path) -> None:
    started = time.perf_counter()
    frozen = _load_frozen_model(source_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Mode: {config.mode}", flush=True)
    print(f"Source: {frozen.source_dir}", flush=True)
    print(
        f"Grid: N={frozen.base_config.n_bins}, M={frozen.base_config.m_bins}, "
        f"step={config.ambiguity_grid_step_samples:g} sample",
        flush=True,
    )

    numerical = _validate_profiled_ambiguity(frozen)
    metrics = _mark_pareto(_enumerate_masks(frozen, config))
    known = _known_method_rows(metrics, frozen)
    floors = _floor_summary(metrics, frozen, config)
    certificate = _build_certificate(
        metrics, known, floors, frozen, config, numerical
    )
    validation = _validate_results(metrics, known, floors, certificate, frozen)

    metrics.to_csv(
        output_dir / "pfrs_fixed_mask_feasibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    known.to_csv(
        output_dir / "pfrs_fixed_mask_known_methods.csv",
        index=False,
        encoding="utf-8-sig",
    )
    floors.to_csv(
        output_dir / "pfrs_fixed_mask_floor_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.loc[metrics["pareto_fisher_vs_sidelobe"]].sort_values(
        "fisher_ratio"
    ).to_csv(
        output_dir / "pfrs_fixed_mask_pareto.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "pfrs_fixed_mask_certificate.json").write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_pareto(metrics, known, output_dir / "PFRS_FixedMask_Pareto.svg")
    _plot_ambiguity_curves(
        frozen, config, output_dir / "PFRS_FixedMask_Ambiguity_Curves.svg"
    )

    output_files = [
        "PFRS_FixedMask_Ambiguity_Curves.svg",
        "PFRS_FixedMask_Pareto.svg",
        "pfrs_fixed_mask_certificate.json",
        "pfrs_fixed_mask_feasibility.csv",
        "pfrs_fixed_mask_floor_summary.csv",
        "pfrs_fixed_mask_known_methods.csv",
        "pfrs_fixed_mask_manifest.json",
        "pfrs_fixed_mask_pareto.csv",
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "source_result_dir": str(frozen.source_dir),
        "source_manifest_sha256": _sha256(
            frozen.source_dir / "pfrs_confirmation_manifest.json"
        ),
        "source_pfrs_script_sha256": frozen.source_manifest["script_sha256"],
        "semantics": (
            "Strongest non-mainlobe maximum of the exact noiseless "
            "nuisance-profiled coherence over the continuous delay prior"
        ),
        "scope_limit": certificate["scope"],
        "config": asdict(config),
        "base_config": _normalized_config(frozen.base_config),
        "numerical_validation": numerical,
        "output_validation": validation,
        "certificate": certificate,
        "elapsed_seconds": time.perf_counter() - started,
        "output_files": output_files,
    }
    (output_dir / "pfrs_fixed_mask_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != set(output_files):
        raise RuntimeError(
            f"Output manifest mismatch: missing={set(output_files)-actual}, "
            f"extra={actual-set(output_files)}"
        )
    print("\nFixed-mask feasibility certificate", flush=True)
    print(json.dumps(certificate, indent=2, ensure_ascii=False), flush=True)
    print(f"[Saved] {output_dir}", flush=True)
    print(f"[Done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


def _default_output_dir(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        PROJECT_ROOT
        / "运行结果"
        / "PFRS"
        / f"PFRS_FIXED_MASK_{stamp}_{mode}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(CONFIGS), default=PYCHARM_RUN_MODE)
    parser.add_argument("--source-dir", type=Path, default=PYCHARM_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or _default_output_dir(args.mode)
    run(CONFIGS[args.mode], args.source_dir, output_dir.resolve())


if __name__ == "__main__":
    main()
