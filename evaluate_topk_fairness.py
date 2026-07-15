# -*- coding: utf-8 -*-
"""Fair strong-baseline comparison and Top-K solver attribution.

PyCharm direct-run entry. Frozen neural estimators are never retrained. The
historical PCA baseline is deterministically rebuilt once from its recorded
training protocol and then cached. Every strong baseline enters its native
Top1-WLS track; methods with a legitimate delay score also enter identical
CommonTop1/CommonTop3 backends. Historical V5B-Expert64 fusion is retained as
an over-budget CR11.64 diagnostic, while V5B-Shared64 is the strict-CR16
candidate. Native-only development uses snapshot pools disjoint from training,
calibration, and locked evaluation before any shared Top-K backend is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import (
    HadamardProjectionBaseline,
    PCABaseline,
    build_traditional_baselines,
)
from compressed_tdoa import (
    HardBinCompressedTDOALikelihood,
    LearnedHardBinCompressedTDOAEstimator,
    V5BExpertGatedTDOAEstimator,
)
from evaluate import _estimate_delay_from_corr, gcc_standard
from experiment_integrity import (
    assert_non_oracle_normalization,
    validate_model_output,
    validate_urban_batch,
)
from model import DAE
from signal_gen import SignalSimulator
from task_baselines import (
    Cao2017DFTAMLEstimator,
    DirectDFTTDOAEstimator,
    GeoHybridDFTTDOAEstimator,
)
from topk_localization import (
    estimator_score_topk,
    localize_topk,
    posterior_topk,
    top1_pair_estimate,
    waveform_gcc_topk,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PYCHARM_SOURCE_RESULT = PROJECT_ROOT / "运行结果" / "20260711_154042"
PYCHARM_DAE_RESULT = PROJECT_ROOT / "运行结果" / "20260702_000925"
PYCHARM_LEGACY_V5_RESULT = PROJECT_ROOT / "运行结果" / "20260709_120019"
PYCHARM_RUN_MODE = "dev_native_multi_pool"
PYCHARM_PROGRESS_EVERY = 10
PYCHARM_ALLOW_LOCKED_OVERWRITE = False

# These manifests are fixed before calibration/development/locked results are inspected.
CALIBRATION_SNAPSHOT_POOL_SEEDS = (142, 542)
DEVELOPMENT_SNAPSHOT_POOL_SEEDS = (242, 342, 442)
LOCKED_SNAPSHOT_POOL_SEEDS = (1042, 2042, 3042, 4042, 5042)
LOCKED_TRIALS_PER_SNR = 100
DEV_NATIVE_SNR = (-10, -6, -2, 0, 2, 6, 12, 20)
DEV_NATIVE_TRIALS_PER_SNR = 60
DEV_FAIR_SNR = (-10, -8, -6, -4, 0, 8)
DEV_FAIR_TRIALS_PER_SNR = 30
DEV_MID_SNR = (2, 4, 6)
DEV_MID_TRIALS_PER_SNR = 30
COMMON_TOP_K = 3
COMMON_MIN_PEAK_SEPARATION = 2
COMMON_BEAM_MAX_PAIRS = 12
COMMON_BEAM_PASSES = 1
FAIR_METHOD_ORDER = (
    "Raw",
    "DAE-CR16",
    "V5B-Expert64",
    "DFT-train-power",
    "DFT-Fisher-Direct",
    "Cao2017-DFT-AML",
    "GeoHybrid-B64-C40F24",
    "Zhai-CRLB-Decimation",
    "Hadamard",
    "PCA",
)
COMMON_SOLVERS = (
    ("CommonTop1-WLS", "top1"),
    ("CommonTop3-GeoBeam", "geobeam"),
    ("CommonTop3-ScreenRefit", "screen_refit"),
)

METHOD_DISPLAY = {
    "Raw": "Raw",
    "DAE-CR16": "DAE-CR16",
    "V5B-Expert64": "V5B (CR11.6)",
    "V5B-Shared64": "V5B-Shared64 (CR16)",
    "V5A1-PowerShared52": "PowerShared52",
    "V5A1-GeoShared52": "GeoShared52",
    "V5A1-Power64": "Power64 (legacy)",
    "V5A1-GeoHybrid64": "GeoHybrid64 (legacy)",
    "DFT-train-power": "DFT-Power",
    "DFT-Fisher-Direct": "DFT-Fisher",
    "Cao2017-DFT-AML": "Cao2017",
    "GeoHybrid-B64-C40F24": "GeoHybrid",
    "Zhai-CRLB-Decimation": "Zhai-CRLB",
    "Hadamard": "Hadamard",
    "PCA": "PCA",
}

METHOD_STYLE = {
    "Raw": dict(color="#111111", marker="o", linestyle="--"),
    "DAE-CR16": dict(color="#d62728", marker="s", linestyle="-"),
    "V5B-Expert64": dict(color="#008b8b", marker="D", linestyle="-"),
    "V5B-Shared64": dict(color="#008b8b", marker="D", linestyle="-"),
    "V5A1-PowerShared52": dict(color="#56b4e9", marker="^", linestyle="-"),
    "V5A1-GeoShared52": dict(color="#0072b2", marker="v", linestyle="-"),
    "V5A1-Power64": dict(color="#9ad9f3", marker="^", linestyle="--"),
    "V5A1-GeoHybrid64": dict(color="#004b76", marker="v", linestyle="--"),
    "DFT-train-power": dict(color="#2ca02c", marker="^", linestyle="-"),
    "DFT-Fisher-Direct": dict(color="#7fbf7b", marker="v", linestyle=":"),
    "Cao2017-DFT-AML": dict(color="#1f77b4", marker="P", linestyle="-"),
    "GeoHybrid-B64-C40F24": dict(color="#ff7f0e", marker="X", linestyle="-"),
    "Zhai-CRLB-Decimation": dict(color="#9467bd", marker="*", linestyle="-"),
    "Hadamard": dict(color="#8c564b", marker="h", linestyle="-."),
    "PCA": dict(color="#6b6b6b", marker="d", linestyle="-."),
}


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _build_simulator(cfg, pool_seed):
    return SignalSimulator(
        channel_mode=cfg.get("channel_mode", "fixed"),
        n_fixed_channels=int(cfg.get("n_fixed_channels", 50)),
        channel_pool_seed=int(pool_seed),
        nlos_prob=float(cfg.get("nlos_prob", 0.0)),
        delay_label_mode=cfg.get("delay_label_mode", "los"),
        snr_train_range=tuple(cfg.get("train_snr_range", (-10, 20))),
        multipath_scale=float(cfg.get("multipath_scale", 0.2)),
        scenario_mode=cfg.get("scenario_mode", "urban8"),
        normalization_mode=cfg.get(
            "normalization_mode", "per_observation_noisy_rms"
        ),
        urban_base_delay=int(cfg.get("urban_base_delay", 16)),
        urban_min_los=int(cfg.get("urban_min_los", 5)),
        urban_train_los_only=bool(cfg.get("urban_train_los_only", True)),
    )


def _resolve_v5_bins(data, label):
    meta = data.get("v5a1_bin_meta", {})
    bins = meta.get("v5a1_selected_bins_by_method", {}).get(label)
    if bins is None:
        bins = (
            data.get("v5a1_training", {}).get(label, {})
            .get("bin_summary", {}).get("selected_bins")
        )
    if bins is None or len(bins) == 0:
        raise RuntimeError(f"cannot resolve frozen selected bins for {label}")
    return np.asarray(bins, dtype=int)


def _v5b_artifact_spec(data):
    cfg = data.get("config", {})
    if bool(cfg.get("v5b_shared64", False)):
        return {
            "method_label": "V5B-Shared64",
            "power_label": "V5A1-PowerShared52",
            "geo_label": "V5A1-GeoShared52",
            "power_file": "model_v5a1_powershared52.pt",
            "geo_file": "model_v5a1_geoshared52.pt",
        }
    return {
        "method_label": "V5B-Expert64",
        "power_label": "V5A1-Power64",
        "geo_label": "V5A1-GeoHybrid64",
        "power_file": "model_v5a1_power64.pt",
        "geo_file": "model_v5a1_geohybrid64.pt",
    }


def _load_v5b_with_experts(source_dir, data, simulator, device):
    cfg = data.get("config", {})
    spec = _v5b_artifact_spec(data)
    lag_limit = int(cfg.get("tdoa_lag_limit_samples", 48))
    estimators = {}
    for key, label, filename in (
        ("power", spec["power_label"], spec["power_file"]),
        ("geohybrid", spec["geo_label"], spec["geo_file"]),
    ):
        model = HardBinCompressedTDOALikelihood(
            selected_bins=_resolve_v5_bins(data, label),
            signal_len=simulator.signal_len,
            lag_limit_samples=lag_limit,
        ).to(device)
        state_path = source_dir / filename
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        model.load_state_dict(torch.load(state_path, map_location=device))
        model.eval()
        estimators[key] = LearnedHardBinCompressedTDOAEstimator(
            model, device, label=label,
            weight_mode=cfg.get("v5a1_weight_mode", "combined"),
        )
    fused = V5BExpertGatedTDOAEstimator(
        estimators,
        label=spec["method_label"],
        low_confidence_power_prior=float(
            cfg.get("v5b_low_confidence_power_prior", 0.65)
        ),
    )
    experts_by_label = {
        spec["power_label"]: estimators["power"],
        spec["geo_label"]: estimators["geohybrid"],
    }
    return fused, experts_by_label


def _load_v5b(source_dir, data, simulator, device):
    fused, _ = _load_v5b_with_experts(
        source_dir, data, simulator, device,
    )
    return fused


def _expert_feature_budget(estimator, signal_len):
    bins = np.asarray(estimator.model.selected_bins.detach().cpu(), dtype=int)
    unique_bins = int(np.unique(bins).size)
    if unique_bins != int(bins.size):
        raise RuntimeError(f"{estimator.label} contains duplicate selected bins")
    real_scalars = 2 * unique_bins
    return {
        "complex_bins": unique_bins,
        "transmitted_real_scalars": real_scalars,
        "effective_cr_vs_2048_real": float(2 * signal_len / real_scalars),
        "within_cr16_budget": bool(real_scalars <= 2 * signal_len // 16),
        "strict_cr16_budget_passed": bool(real_scalars == 2 * signal_len // 16),
    }


def _v5b_feature_budget(estimator, signal_len):
    budget = estimator.feature_budget()
    return {
        "power_complex_bins": int(budget["expert_complex_bins"]["power"]),
        "geohybrid_complex_bins": int(
            budget["expert_complex_bins"]["geohybrid"]
        ),
        "overlap_complex_bins": int(budget["overlap_complex_bins"]),
        "transmitted_union_complex_bins": int(budget["unique_complex_bins"]),
        "transmitted_real_scalars": int(budget["transmitted_real_scalars"]),
        "effective_cr_vs_2048_real": float(budget["effective_cr"]),
        "within_cr16_budget": bool(budget["within_cr16"]),
        "strict_cr16_budget_passed": bool(budget["exact_cr16"]),
    }


def _load_dae(dae_dir, device):
    path = dae_dir / "model_cr16.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model = DAE(cr=16).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _dft_bins(data, label):
    meta = data.get("traditional_baseline_meta", {})
    by_method = meta.get("dft_selected_bins_by_method", {})
    bins = by_method.get(label)
    if bins is None or len(bins) == 0:
        raise RuntimeError(f"missing frozen DFT bins for {label}")
    return np.asarray(bins, dtype=int)


def _first_unique_bins(ranked, count, exclude=()):
    excluded = set(int(v) for v in np.asarray(exclude, dtype=int).tolist())
    selected = []
    seen = set()
    for value in np.asarray(ranked, dtype=int).tolist():
        value = int(value)
        if value in excluded or value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError(f"could not select {count} unique DFT bins")
    return np.asarray(selected, dtype=int)


def _build_direct_estimators(data, signal_len):
    power = _dft_bins(data, "DFT-train-power")
    fisher = _dft_bins(data, "DFT-Fisher")
    geo = _dft_bins(data, "DFT-GeoAmbi")
    zhai = _dft_bins(data, "DFT-Zhai-CRLB")
    coarse = _first_unique_bins(power, 40)
    fine = _first_unique_bins(geo, 24, exclude=coarse)
    if int(np.union1d(coarse, fine).size) != 64:
        raise RuntimeError("GeoHybrid C40F24 does not meet the CR16 union budget")
    common = {
        "DFT-train-power": DirectDFTTDOAEstimator(
            power, signal_len=signal_len, label="DFT-train-power", phat=False,
        ),
        "DFT-Fisher-Direct": DirectDFTTDOAEstimator(
            fisher, signal_len=signal_len, label="DFT-Fisher-Direct", phat=False,
        ),
        "Cao2017-DFT-AML": Cao2017DFTAMLEstimator(
            power, signal_len=signal_len, label="Cao2017-DFT-AML",
            weight_mode="aml",
        ),
        "Zhai-CRLB-Decimation": Cao2017DFTAMLEstimator(
            zhai, signal_len=signal_len, label="Zhai-CRLB-Decimation",
            weight_mode="aml",
        ),
    }
    native_only = {
        "GeoHybrid-B64-C40F24": GeoHybridDFTTDOAEstimator(
            coarse, fine, signal_len=signal_len,
            label="GeoHybrid-B64-C40F24", coarse_mode="aml", fine_mode="aml",
            refinement_radius=4.0, fine_gain=1.25,
            consistency_sigma=3.0, budget_limit_complex=64,
            use_consistency=True, use_pair_uncertainty=True,
            budget_note="strict CR16 budget: union<=64; coarse=40, fine=24",
        )
    }
    selected = {
        "DFT-train-power": power.tolist(),
        "DFT-Fisher-Direct": fisher.tolist(),
        "Cao2017-DFT-AML": power.tolist(),
        "GeoHybrid-B64-C40F24-coarse": coarse.tolist(),
        "GeoHybrid-B64-C40F24-fine": fine.tolist(),
        "Zhai-CRLB-Decimation": zhai.tolist(),
    }
    return common, native_only, selected


def _load_or_build_waveform_baselines(data, simulator, device, output_dir,
                                      training_pool_seed):
    """Rebuild the recorded PCA once, then freeze it for every fair run."""
    cfg = data.get("config", {})
    cache_path = output_dir / f"pca_cr16_frozen_seed{int(training_pool_seed)}.pt"
    fit_seed = int(cfg.get("seed", 42)) + 6000
    expected = {
        "signal_len": int(simulator.signal_len),
        "requested_components": int(2 * simulator.signal_len // 16),
        "fit_seed": int(fit_seed),
        "training_samples": int(cfg.get("baseline_pca_samples", 10000)),
        "training_source": str(cfg.get("baseline_pca_train_source", "noisy")),
        "fixed_snr_db": float(cfg.get("baseline_pca_fixed_snr_db", 0.0)),
        "training_snapshot_pool_seed": int(training_pool_seed),
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        stored = payload.get("protocol", {})
        if stored != expected:
            raise RuntimeError("cached PCA protocol differs from frozen source config")
        pca = PCABaseline(
            payload["mean"], payload["components"],
            signal_len=expected["signal_len"],
            requested_components=expected["requested_components"],
        ).to(device)
    else:
        print("[Baseline] rebuilding the recorded PCA protocol once...", flush=True)
        models, meta = build_traditional_baselines(
            simulator, cr=16,
            n_pca_samples=expected["training_samples"], seed=fit_seed,
            device=device, dft_mode=str(cfg.get("baseline_dft_mode", "scs_lite")),
            hadamard_mode=str(cfg.get("baseline_hadamard_mode", "salari")),
            pca_train_source=expected["training_source"],
            pca_fixed_snr_db=expected["fixed_snr_db"],
            include_diagnostic_variants=False,
            include_legacy_variants=False,
            include_strong_variants=False,
        )
        pca = models["PCA"]
        if int(meta.get("pca_actual_components", -1)) != expected["requested_components"]:
            raise RuntimeError("rebuilt PCA feature count differs from CR16 budget")
        torch.save({
            "mean": pca.mean.detach().cpu(),
            "components": pca.components.detach().cpu(),
            "protocol": expected,
        }, cache_path)
    pca.eval()
    hadamard = HadamardProjectionBaseline(
        signal_len=simulator.signal_len, cr=16,
        row_mode=str(cfg.get("baseline_hadamard_mode", "salari")),
        seed=fit_seed + 23,
    ).to(device)
    hadamard.eval()
    return {"Hadamard": hadamard, "PCA": pca}, cache_path, expected


def _run_waveform_model(model, x_noisy, device, batch_size=64):
    flat = x_noisy.reshape(-1, 2, x_noisy.shape[-1])
    chunks = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], int(batch_size)):
            batch = flat[start:start + int(batch_size)].to(device)
            output = model(batch)
            validate_model_output(output, batch.shape, label=type(model).__name__)
            chunks.append(output.cpu())
    result = torch.cat(chunks, dim=0).reshape(x_noisy.shape)
    validate_model_output(result, x_noisy.shape, label=type(model).__name__)
    return result.numpy()


def _complex_channel(batch, trial, uav):
    return batch[trial, uav, 0, :] + 1j * batch[trial, uav, 1, :]


def _native_waveform_top1(sig_i, sig_j, lag_limit_samples):
    n = int(np.asarray(sig_i).size)
    lags = np.arange(-(n // 2), n - n // 2, dtype=float)
    lag, weight, _, _ = _estimate_delay_from_corr(
        sig_i, sig_j, lags, gcc_standard, sub_sample=True,
        return_quality=True,
        tdoa_lag_limit_samples=lag_limit_samples,
    )
    return top1_pair_estimate(lag, weight)


def _native_estimator_top1(estimator, sig_i, sig_j, lag_limit_samples):
    lag, weight, _, _ = estimator.estimate_pair(
        sig_i, sig_j, sub_sample=True, return_quality=True,
        lag_limit_samples=lag_limit_samples,
    )
    return top1_pair_estimate(lag, weight)


def _summarize(group):
    errors = group["error_m"].to_numpy(dtype=float)
    valid = errors[np.isfinite(errors)]
    tdoa = group["tdoa_abs_error_mean_samples"].to_numpy(dtype=float)
    tdoa = tdoa[np.isfinite(tdoa)]
    return pd.Series({
        "requested_count": int(len(group)),
        "valid_count": int(valid.size),
        "failure_rate": float(1.0 - valid.size / max(len(group), 1)),
        "rmse_m": float(np.sqrt(np.mean(valid ** 2))) if valid.size else np.nan,
        "median_m": float(np.median(valid)) if valid.size else np.nan,
        "p95_m": float(np.percentile(valid, 95)) if valid.size else np.nan,
        "tdoa_mae_samples": float(np.mean(tdoa)) if tdoa.size else np.nan,
        "mean_solver_calls": float(group["n_solver_calls"].mean()),
        "mean_pairs_changed": float(group["n_pairs_changed"].mean()),
        "mean_elapsed_s": float(group["elapsed_s"].mean()),
    })


def _manifest_hash(manifest):
    raw = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_bootstrap_solver_gain(frame, method, solver, bucket,
                                   baseline_solver="CommonTop1-WLS",
                                   seed=42, n_boot=1000):
    subset = frame[frame["method"] == method].copy()
    if bucket == "low":
        subset = subset[subset["SNR_dB"] <= 0]
    elif bucket == "mid":
        subset = subset[(subset["SNR_dB"] >= 2) & (subset["SNR_dB"] <= 6)]
    elif bucket == "high":
        subset = subset[subset["SNR_dB"] >= 8]
    key = ["pool_seed", "SNR_dB", "trial_idx", "snapshot_id", "cluster_id"]
    wide = subset.pivot_table(index=key, columns="solver", values="error_m", aggfunc="first")
    if baseline_solver not in wide or solver not in wide:
        return np.nan, np.nan, np.nan, 0
    valid = np.isfinite(wide[baseline_solver]) & np.isfinite(wide[solver])
    wide = wide.loc[valid].reset_index()
    clusters = wide["cluster_id"].drop_duplicates().to_numpy()
    if wide.empty or clusters.size == 0:
        return np.nan, np.nan, np.nan, 0

    def gain(values):
        return float(
            np.sqrt(np.mean(values[:, 0] ** 2))
            - np.sqrt(np.mean(values[:, 1] ** 2))
        )

    grouped = {
        cluster: wide.loc[
            wide["cluster_id"] == cluster, [baseline_solver, solver]
        ].to_numpy(dtype=float)
        for cluster in clusters
    }
    point = gain(wide[[baseline_solver, solver]].to_numpy(dtype=float))
    rng = np.random.RandomState(int(seed))
    samples = []
    for _ in range(int(n_boot)):
        selected = rng.choice(clusters, size=clusters.size, replace=True)
        samples.append(gain(np.concatenate([grouped[c] for c in selected], axis=0)))
    return (
        point, float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)), int(clusters.size),
    )


def _cluster_bootstrap_method_gain(frame, target, baseline, solver, bucket,
                                   seed=42, n_boot=1000):
    subset = frame.loc[
        (frame["solver"] == solver)
        & frame["method"].isin((target, baseline))
    ].copy()
    if bucket == "low":
        subset = subset.loc[subset["SNR_dB"] <= 0]
    elif bucket == "mid":
        subset = subset.loc[
            (subset["SNR_dB"] >= 2) & (subset["SNR_dB"] <= 6)
        ]
    elif bucket == "high":
        subset = subset.loc[subset["SNR_dB"] >= 8]
    key = ["pool_seed", "SNR_dB", "trial_idx", "snapshot_id", "cluster_id"]
    error_wide = subset.pivot_table(
        index=key, columns="method", values="error_m", aggfunc="first",
    )
    tdoa_wide = subset.pivot_table(
        index=key, columns="method", values="tdoa_abs_error_mean_samples",
        aggfunc="first",
    )
    if any(name not in error_wide for name in (target, baseline)):
        return {
            "paired_count": 0, "cluster_count": 0,
            "rmse_gain_m": np.nan, "rmse_ci95_low_m": np.nan,
            "rmse_ci95_high_m": np.nan, "tdoa_mae_gain_samples": np.nan,
            "tdoa_ci95_low_samples": np.nan, "tdoa_ci95_high_samples": np.nan,
        }
    joined = pd.DataFrame({
        "target_error": error_wide[target],
        "baseline_error": error_wide[baseline],
        "target_tdoa": tdoa_wide[target],
        "baseline_tdoa": tdoa_wide[baseline],
    }).dropna().reset_index()
    if joined.empty:
        return {
            "paired_count": 0, "cluster_count": 0,
            "rmse_gain_m": np.nan, "rmse_ci95_low_m": np.nan,
            "rmse_ci95_high_m": np.nan, "tdoa_mae_gain_samples": np.nan,
            "tdoa_ci95_low_samples": np.nan, "tdoa_ci95_high_samples": np.nan,
        }
    clusters = joined["cluster_id"].drop_duplicates().to_numpy()
    grouped = {
        cluster: joined.index[joined["cluster_id"] == cluster].to_numpy()
        for cluster in clusters
    }

    def metrics(index):
        target_error = joined.loc[index, "target_error"].to_numpy(dtype=float)
        baseline_error = joined.loc[index, "baseline_error"].to_numpy(dtype=float)
        rmse_gain = (
            np.sqrt(np.mean(baseline_error ** 2))
            - np.sqrt(np.mean(target_error ** 2))
        )
        tdoa_gain = float(np.mean(
            joined.loc[index, "baseline_tdoa"].to_numpy(dtype=float)
            - joined.loc[index, "target_tdoa"].to_numpy(dtype=float)
        ))
        return float(rmse_gain), tdoa_gain

    point_rmse, point_tdoa = metrics(joined.index.to_numpy())
    rng = np.random.RandomState(int(seed))
    rmse_samples, tdoa_samples = [], []
    for _ in range(int(n_boot)):
        selected = rng.choice(clusters, size=clusters.size, replace=True)
        index = np.concatenate([grouped[cluster] for cluster in selected])
        rmse_gain, tdoa_gain = metrics(index)
        rmse_samples.append(rmse_gain)
        tdoa_samples.append(tdoa_gain)
    return {
        "paired_count": int(len(joined)),
        "cluster_count": int(clusters.size),
        "rmse_gain_m": point_rmse,
        "rmse_ci95_low_m": float(np.percentile(rmse_samples, 2.5)),
        "rmse_ci95_high_m": float(np.percentile(rmse_samples, 97.5)),
        "tdoa_mae_gain_samples": point_tdoa,
        "tdoa_ci95_low_samples": float(np.percentile(tdoa_samples, 2.5)),
        "tdoa_ci95_high_samples": float(np.percentile(tdoa_samples, 97.5)),
    }


def _export_method_comparison(frame, output_dir, prefix, target=None):
    """Export paired cross-method evidence; positive gain favors ``target``."""
    if target is None:
        target = (
            "V5B-Shared64"
            if "V5B-Shared64" in set(frame["method"]) else "V5B-Expert64"
        )
    rows = []
    available_solvers = list(dict.fromkeys(frame["solver"].tolist()))
    for solver in available_solvers:
        solver_methods = set(frame.loc[frame["solver"] == solver, "method"])
        if target not in solver_methods:
            continue
        for baseline in FAIR_METHOD_ORDER:
            if baseline == target or baseline not in solver_methods:
                continue
            for bucket in ("low", "mid", "high", "all"):
                stats = _cluster_bootstrap_method_gain(
                    frame, target, baseline, solver, bucket,
                    seed=42 + 97 * len(rows), n_boot=1000,
                )
                rows.append({
                    "target_method": target,
                    "baseline_method": baseline,
                    "solver": solver,
                    "snr_bucket": bucket,
                    "positive_gain_means_target_better": True,
                    **stats,
                })
    output_path = Path(output_dir) / f"{prefix}_method_comparison.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    print(f"[Analysis] saved {output_path}", flush=True)
    return output_path


def _export_poolwise_method_comparison(frame, output_dir, prefix, target=None):
    """Export the same paired evidence separately for every snapshot pool."""
    if target is None:
        target = (
            "V5B-Shared64"
            if "V5B-Shared64" in set(frame["method"]) else "V5B-Expert64"
        )
    rows = []
    for pool_seed in sorted(frame["pool_seed"].unique().tolist()):
        pool_frame = frame.loc[frame["pool_seed"] == pool_seed]
        for solver in list(dict.fromkeys(pool_frame["solver"].tolist())):
            solver_methods = set(
                pool_frame.loc[pool_frame["solver"] == solver, "method"]
            )
            if target not in solver_methods:
                continue
            for baseline in FAIR_METHOD_ORDER:
                if baseline == target or baseline not in solver_methods:
                    continue
                for bucket in ("low", "mid", "high", "all"):
                    stats = _cluster_bootstrap_method_gain(
                        pool_frame, target, baseline, solver, bucket,
                        seed=int(pool_seed) + 97 * len(rows), n_boot=1000,
                    )
                    rows.append({
                        "pool_seed": int(pool_seed),
                        "target": target,
                        "baseline": baseline,
                        "solver": solver,
                        "snr_bucket": bucket,
                        "positive_gain_means_target_better": True,
                        **stats,
                    })
    output_path = Path(output_dir) / f"{prefix}_method_comparison_by_pool.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    print(f"[Analysis] saved {output_path}", flush=True)
    return output_path


def _mode_config(mode, cfg):
    full_snr = tuple(float(v) for v in cfg.get(
        "eval_snr_range", np.arange(-10, 21, 2)
    ))
    if mode == "smoke":
        return (int(cfg.get("seed", 42)),), (-10.0,), 2, "fair_topk_smoke"
    if mode == "dev_native_multi_pool":
        return (
            tuple(int(v) for v in DEVELOPMENT_SNAPSHOT_POOL_SEEDS),
            tuple(float(v) for v in DEV_NATIVE_SNR),
            int(DEV_NATIVE_TRIALS_PER_SNR),
            "fair_native_dev_multi_pool",
        )
    if mode == "dev_fair":
        return (
            (int(cfg.get("seed", 42)),), tuple(float(v) for v in DEV_FAIR_SNR),
            int(DEV_FAIR_TRIALS_PER_SNR), "fair_topk_dev",
        )
    if mode == "dev_mid_gate":
        return (
            (int(cfg.get("seed", 42)),), tuple(float(v) for v in DEV_MID_SNR),
            int(DEV_MID_TRIALS_PER_SNR), "fair_topk_dev_mid",
        )
    if mode == "locked_holdout":
        return (
            tuple(int(v) for v in LOCKED_SNAPSHOT_POOL_SEEDS), full_snr,
            int(LOCKED_TRIALS_PER_SNR), "fair_topk_locked",
        )
    raise ValueError(f"unknown run mode: {mode}")


def _validate_pool_partitions(training_pool_seed):
    partitions = {
        "training": {int(training_pool_seed)},
        "calibration": {int(v) for v in CALIBRATION_SNAPSHOT_POOL_SEEDS},
        "development": {int(v) for v in DEVELOPMENT_SNAPSHOT_POOL_SEEDS},
        "locked": {int(v) for v in LOCKED_SNAPSHOT_POOL_SEEDS},
    }
    names = tuple(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = partitions[left].intersection(partitions[right])
            if overlap:
                raise RuntimeError(
                    f"snapshot-pool leakage between {left} and {right}: "
                    f"{sorted(overlap)}"
                )
    return {name: sorted(values) for name, values in partitions.items()}


def _aggregate_for_plot(frame, snr_filter=None):
    subset = frame if snr_filter is None else frame.loc[snr_filter(frame)]
    records = []
    for (method, solver), group in subset.groupby(["method", "solver"]):
        errors = group["error_m"].to_numpy(dtype=float)
        errors = errors[np.isfinite(errors)]
        records.append({
            "method": method,
            "solver": solver,
            "rmse_m": float(np.sqrt(np.mean(errors ** 2))) if errors.size else np.nan,
            "tdoa_mae_samples": float(group["tdoa_abs_error_mean_samples"].mean()),
            "mean_solver_calls": float(group["n_solver_calls"].mean()),
        })
    return pd.DataFrame(records)


def _merge_existing_development_rows(output_dir):
    output_dir = Path(output_dir)
    base_path = output_dir / "fair_topk_dev_rows.csv"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    paths = [base_path]
    mid_path = output_dir / "fair_topk_dev_mid_rows.csv"
    if mid_path.exists():
        paths.append(mid_path)
    frames = [pd.read_csv(path) for path in paths]
    legacy_missing = [
        path.name for path, frame in zip(paths, frames)
        if {"eval_seed", "protocol_id"}.difference(frame.columns)
    ]
    if legacy_missing:
        print(
            "[Legacy] protocol fields are missing; plotting the base development "
            "rows only and excluding the old mid-SNR rows because their fixed-set "
            f"identity cannot be proven: {legacy_missing}",
            flush=True,
        )
        return frames[0], "fair_topk_dev_legacy"
    for path, frame in zip(paths, frames):
        required = {"eval_seed", "protocol_id"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(
                f"{path.name} predates the fixed-evaluation protocol fields "
                f"{missing}; rerun the corresponding development mode"
            )
    frame = pd.concat(frames, ignore_index=True)
    if frame["eval_seed"].nunique() != 1 or frame["protocol_id"].nunique() != 1:
        raise RuntimeError(
            "development rows use different fixed-evaluation seeds or protocols"
        )
    key = ["pool_seed", "SNR_dB", "trial_idx", "snapshot_id", "method", "solver"]
    duplicates = frame.duplicated(key, keep=False)
    if bool(duplicates.any()):
        raise RuntimeError("development rows overlap; refusing an ambiguous merge")
    frame = frame.sort_values(key).reset_index(drop=True)
    prefix = "fair_topk_dev_complete" if len(paths) == 2 else "fair_topk_dev"
    if len(paths) == 2:
        frame.to_csv(
            output_dir / f"{prefix}_rows.csv", index=False, encoding="utf-8",
        )
    return frame, prefix


def _export_basic_summaries(frame, output_dir, prefix):
    output_dir = Path(output_dir)
    summary = (
        frame.groupby(["pool_seed", "SNR_dB", "method", "solver"], dropna=False)
        .apply(_summarize, include_groups=False).reset_index()
    )
    aggregate = (
        frame.groupby(["method", "solver"], dropna=False)
        .apply(_summarize, include_groups=False).reset_index()
    )
    summary.to_csv(
        output_dir / f"{prefix}_summary_by_pool_snr.csv",
        index=False, encoding="utf-8",
    )
    aggregate.to_csv(
        output_dir / f"{prefix}_summary_all.csv", index=False, encoding="utf-8",
    )
    aggregate.loc[aggregate["solver"] == "OriginalTop1-WLS"].to_csv(
        output_dir / f"{prefix}_broad_leaderboard.csv",
        index=False, encoding="utf-8",
    )
    return summary, aggregate


def _plot_fairness_results(frame, output_dir, prefix):
    """Generate readable SVG summaries from an existing sample-level CSV."""
    required = {
        "SNR_dB", "method", "solver", "error_m",
        "tdoa_abs_error_mean_samples", "n_solver_calls",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fairness rows are missing plot fields: {missing}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    native = frame.loc[frame["solver"] == "OriginalTop1-WLS"]
    native_summary = []
    for (snr, method), group in native.groupby(["SNR_dB", "method"]):
        errors = group["error_m"].to_numpy(dtype=float)
        errors = errors[np.isfinite(errors)]
        native_summary.append({
            "SNR_dB": float(snr),
            "method": method,
            "rmse_m": float(np.sqrt(np.mean(errors ** 2))) if errors.size else np.nan,
            "tdoa_mae_samples": float(group["tdoa_abs_error_mean_samples"].mean()),
        })
    native_summary = pd.DataFrame(native_summary)
    snr_values = sorted(native_summary["SNR_dB"].unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.25), constrained_layout=False)
    for method in FAIR_METHOD_ORDER:
        method_rows = native_summary.loc[native_summary["method"] == method].sort_values(
            "SNR_dB"
        )
        if method_rows.empty:
            continue
        style = METHOD_STYLE[method]
        for ax, metric in zip(axes, ("rmse_m", "tdoa_mae_samples")):
            ax.plot(
                method_rows["SNR_dB"], method_rows[metric],
                label=METHOD_DISPLAY[method], linewidth=1.35,
                markersize=4.5, markeredgewidth=0.7, **style,
            )
    axes[0].set_title("Localization accuracy", fontsize=11)
    axes[0].set_ylabel("Localization RMSE (m)")
    axes[1].set_title("Pairwise TDOA accuracy", fontsize=11)
    axes[1].set_ylabel("TDOA MAE (samples)")
    for ax in axes:
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(snr_values)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, 0.01), fontsize=8.4,
    )
    fig.suptitle("Native Top-1 Strong-Baseline Comparison", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.94))
    suffix = {
        "fair_topk_smoke": "_Smoke",
        "fair_native_dev_multi_pool": "_NativeMultiPool",
        "fair_topk_dev_mid": "_Mid",
        "fair_topk_locked": "_Locked",
    }.get(str(prefix), "")
    native_path = output_dir / f"FigFair1_Native_Strong_Baselines{suffix}.svg"
    fig.savefig(native_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    available_solvers = set(frame["solver"].unique())
    solver_order = [
        label for label, _ in COMMON_SOLVERS if label in available_solvers
    ]
    if not solver_order:
        print(f"[Plot] saved {native_path}", flush=True)
        return native_path, None

    common_methods = [
        method for method in FAIR_METHOD_ORDER
        if method != "GeoHybrid-B64-C40F24"
        and method in set(frame["method"].unique())
    ]
    all_summary = _aggregate_for_plot(frame)
    low_summary = _aggregate_for_plot(frame, lambda x: x["SNR_dB"] <= 0)
    solver_style = {
        "CommonTop1-WLS": ("#8c8c8c", ""),
        "CommonTop3-GeoBeam": ("#0072b2", "//"),
        "CommonTop3-ScreenRefit": ("#e69f00", "xx"),
    }
    panels = (
        (all_summary, "rmse_m", "All development SNRs", "Localization RMSE (m)"),
        (low_summary, "rmse_m", "Low SNR (SNR <= 0 dB)", "Localization RMSE (m)"),
        (all_summary, "tdoa_mae_samples", "TDOA accuracy", "TDOA MAE (samples)"),
        (all_summary, "mean_solver_calls", "Solver complexity", "Mean WLS calls / sample"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), constrained_layout=False)
    x = np.arange(len(common_methods), dtype=float)
    width = 0.72 / len(solver_order)
    for ax, (summary, metric, title, ylabel) in zip(axes.ravel(), panels):
        for solver_index, solver in enumerate(solver_order):
            lookup = summary.loc[summary["solver"] == solver].set_index("method")[metric]
            values = [float(lookup.get(method, np.nan)) for method in common_methods]
            color, hatch = solver_style[solver]
            ax.bar(
                x + (solver_index - (len(solver_order) - 1) / 2.0) * width,
                values, width=width,
                color=color, edgecolor="#303030", linewidth=0.45,
                hatch=hatch, label=solver.replace("Common", ""),
            )
        ax.set_title(title, fontsize=10.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_DISPLAY[method] for method in common_methods],
            rotation=24, ha="right", fontsize=8.2,
        )
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_yticks([1, 2, 5, 10, 20])
    axes[1, 1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.01), fontsize=9,
    )
    fig.suptitle("Common Top-K Solver Attribution", fontsize=13)
    fig.text(
        0.5, 0.055,
        "GeoHybrid is omitted because its fine score is conditioned on a coarse local window.",
        ha="center", fontsize=8.2, color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95), h_pad=2.0, w_pad=1.2)
    attribution_path = output_dir / f"FigFair2_Common_TopK_Attribution{suffix}.svg"
    fig.savefig(attribution_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved {native_path}", flush=True)
    print(f"[Plot] saved {attribution_path}", flush=True)
    return native_path, attribution_path


def _active_method_order(v5b_spec, run_mode):
    if run_mode in ("smoke", "dev_native_multi_pool"):
        return (
            "Raw", "DAE-CR16",
            v5b_spec["power_label"], v5b_spec["geo_label"],
            v5b_spec["method_label"],
            "V5A1-Power64", "V5A1-GeoHybrid64",
            "DFT-train-power", "Cao2017-DFT-AML",
            "Zhai-CRLB-Decimation",
        )
    return tuple(
        v5b_spec["method_label"] if name == "V5B-Expert64" else name
        for name in FAIR_METHOD_ORDER
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode", choices=(
        "smoke", "dev_native_multi_pool", "dev_fair", "dev_mid_gate",
        "plot_existing", "locked_holdout"
    ),
                        default=PYCHARM_RUN_MODE)
    parser.add_argument("--source-result", default=str(PYCHARM_SOURCE_RESULT))
    parser.add_argument("--dae-result", default=str(PYCHARM_DAE_RESULT))
    parser.add_argument("--progress-every", type=int, default=PYCHARM_PROGRESS_EVERY)
    args = parser.parse_args()

    source_dir = Path(args.source_result).resolve()
    dae_dir = Path(args.dae_result).resolve()
    data = _load_pickle(source_dir / "plot_data.pkl")
    cfg = data.get("config", {})
    v5b_spec = _v5b_artifact_spec(data)
    v5b_method_label = v5b_spec["method_label"]
    global FAIR_METHOD_ORDER
    FAIR_METHOD_ORDER = _active_method_order(v5b_spec, args.run_mode)
    native_only_run = args.run_mode in ("smoke", "dev_native_multi_pool")
    output_dir = source_dir / "tables" / "topk_fairness"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_mode == "plot_existing":
        frame, plot_prefix = _merge_existing_development_rows(output_dir)
        _export_basic_summaries(frame, output_dir, plot_prefix)
        _export_method_comparison(frame, output_dir, plot_prefix)
        _plot_fairness_results(frame, output_dir, plot_prefix)
        print(
            f"[Done] existing fairness results analyzed/replotted: {plot_prefix}",
            flush=True,
        )
        return
    pool_seeds, snr_values, trials, prefix = _mode_config(args.run_mode, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_rows_path = output_dir / f"{prefix}_rows.csv"
    active_common_solvers = list(COMMON_SOLVERS)
    if args.run_mode == "locked_holdout":
        decision_path = output_dir / "fair_topk_dev_solver_decision.json"
        if not (output_dir / "fair_topk_dev_rows.csv").exists() or not decision_path.exists():
            raise RuntimeError(
                "run dev_fair and create fair_topk_dev_solver_decision.json before "
                "opening the locked holdout"
            )
        with open(decision_path, "r", encoding="utf-8") as handle:
            decision = json.load(handle)
        if not bool(decision.get("budget_audit_passed", False)):
            raise RuntimeError(
                "locked holdout blocked: solver decision has not passed the feature-budget audit"
            )
        selected_solver = str(decision.get("selected_common_solver", ""))
        allowed = {label: mode for label, mode in COMMON_SOLVERS}
        if selected_solver not in allowed:
            raise RuntimeError("locked solver decision is missing or invalid")
        active_common_solvers = [("CommonTop1-WLS", "top1")]
        if selected_solver != "CommonTop1-WLS":
            active_common_solvers.append((selected_solver, allowed[selected_solver]))
    if (args.run_mode == "locked_holdout" and final_rows_path.exists()
            and not PYCHARM_ALLOW_LOCKED_OVERWRITE):
        raise RuntimeError(
            "locked holdout results already exist; refusing to overwrite them. "
            "Do not rerun or tune against a locked test without an explicit protocol revision."
        )

    training_pool_seed = int(cfg.get("channel_pool_seed", cfg.get("seed", 42)))
    pool_partitions = _validate_pool_partitions(training_pool_seed)
    training_sim = _build_simulator(cfg, training_pool_seed)
    base_sim = _build_simulator(cfg, pool_seeds[0])
    assert_non_oracle_normalization(training_sim, context=args.run_mode)
    assert_non_oracle_normalization(base_sim, context=args.run_mode)
    if (training_sim.signal_len != base_sim.signal_len
            or tuple(training_sim.area_size) != tuple(base_sim.area_size)):
        raise RuntimeError("training and evaluation simulators use different dimensions")
    v5b, current_experts = _load_v5b_with_experts(
        source_dir, data, training_sim, device,
    )
    v5b_budget = _v5b_feature_budget(v5b, base_sim.signal_len)
    if args.run_mode in (
        "dev_native_multi_pool", "dev_fair", "dev_mid_gate", "locked_holdout"
    ) and not v5b_budget[
        "strict_cr16_budget_passed"
    ]:
        raise RuntimeError(
            f"{args.run_mode} blocked: V5B expert-bin union does not match "
            "the strict 64-complex-bin CR16 budget"
        )
    dae = _load_dae(dae_dir, device)
    legacy_experts = {}
    legacy_data = None
    legacy_dir = PYCHARM_LEGACY_V5_RESULT.resolve()
    if native_only_run:
        legacy_data = _load_pickle(legacy_dir / "plot_data.pkl")
        legacy_fused, legacy_experts = _load_v5b_with_experts(
            legacy_dir, legacy_data, training_sim, device,
        )
        if str(legacy_fused.label) != "V5B-Expert64":
            raise RuntimeError("legacy V5 source is not the expected V5B-Expert64 run")
        for label in ("V5A1-Power64", "V5A1-GeoHybrid64"):
            if label not in legacy_experts:
                raise RuntimeError(f"legacy exact-CR16 expert missing: {label}")
            budget = _expert_feature_budget(
                legacy_experts[label], base_sim.signal_len,
            )
            if not budget["strict_cr16_budget_passed"]:
                raise RuntimeError(f"legacy standalone expert is not CR16: {label}")
    if native_only_run:
        waveform_models, pca_cache_path, pca_protocol = {}, None, None
    else:
        waveform_models, pca_cache_path, pca_protocol = (
            _load_or_build_waveform_baselines(
                data, training_sim, device, output_dir, training_pool_seed,
            )
        )
    common_estimators, native_only_estimators, selected_bins = (
        _build_direct_estimators(data, base_sim.signal_len)
    )
    if native_only_run:
        common_estimators = {
            label: estimator for label, estimator in common_estimators.items()
            if label in FAIR_METHOD_ORDER
        }
        native_only_estimators = {
            label: estimator for label, estimator in native_only_estimators.items()
            if label in FAIR_METHOD_ORDER
        }
        selected_bins = {
            label: bins for label, bins in selected_bins.items()
            if label in FAIR_METHOD_ORDER
        }
        common_methods = set()
        active_common_solvers = []
    else:
        common_methods = {
            "Raw", "DAE-CR16", v5b_method_label, "Hadamard", "PCA",
            *common_estimators.keys(),
        }
    method_protocols = {
        name: {
            "native_top1": True,
            "common_topk": name in common_methods,
            "role": "uncompressed_reference" if name == "Raw" else "compressed_method",
        }
        for name in FAIR_METHOD_ORDER
    }
    if "GeoHybrid-B64-C40F24" in method_protocols:
        method_protocols["GeoHybrid-B64-C40F24"]["common_topk_reason"] = (
            "disabled: the native method uses a coarse-conditioned local fine window; "
            "a global Top-K curve would change the published implementation"
        )
    method_protocols[v5b_method_label]["feature_budget"] = v5b_budget
    method_protocols[v5b_method_label]["role"] = (
        "over_budget_diagnostic"
        if not v5b_budget["strict_cr16_budget_passed"] else "compressed_method"
    )
    for label, estimator in {**current_experts, **legacy_experts}.items():
        if label in method_protocols:
            method_protocols[label]["feature_budget"] = _expert_feature_budget(
                estimator, base_sim.signal_len,
            )
            method_protocols[label]["role"] = (
                "legacy_standalone_exact_cr16"
                if label in legacy_experts else "shared64_component_expert"
            )
    artifact_paths = {
        "v5_power": source_dir / v5b_spec["power_file"],
        "v5_geohybrid": source_dir / v5b_spec["geo_file"],
        "dae_cr16": dae_dir / "model_cr16.pt",
        "plot_data": source_dir / "plot_data.pkl",
        "fairness_code": Path(__file__).resolve(),
        "topk_code": PROJECT_ROOT / "topk_localization.py",
        "shared_evaluator_code": PROJECT_ROOT / "evaluate.py",
        "task_baseline_code": PROJECT_ROOT / "task_baselines.py",
    }
    if pca_cache_path is not None:
        artifact_paths["pca_cr16"] = pca_cache_path
    if legacy_experts:
        artifact_paths.update({
            "legacy_v5_power64": legacy_dir / "model_v5a1_power64.pt",
            "legacy_v5_geohybrid64": legacy_dir / "model_v5a1_geohybrid64.pt",
            "legacy_v5_plot_data": legacy_dir / "plot_data.pkl",
        })
    manifest = {
        "run_mode": args.run_mode,
        "source_result": str(source_dir),
        "dae_result": str(dae_dir),
        "snapshot_pool_seeds": list(pool_seeds),
        "pool_partitions": pool_partitions,
        "development_pool_set_id": _manifest_hash({
            "pool_seeds": list(DEVELOPMENT_SNAPSHOT_POOL_SEEDS),
            "snr_values": list(DEV_NATIVE_SNR),
            "trials_per_snr": int(DEV_NATIVE_TRIALS_PER_SNR),
        }),
        "snr_values": list(snr_values),
        "trials_per_snr": int(trials),
        "methods": list(FAIR_METHOD_ORDER),
        "method_protocols": method_protocols,
        "solvers": ["OriginalTop1-WLS", *[item[0] for item in active_common_solvers]],
        "top_k": COMMON_TOP_K,
        "min_peak_separation_bins": COMMON_MIN_PEAK_SEPARATION,
        "beam_max_pairs": COMMON_BEAM_MAX_PAIRS,
        "beam_passes": COMMON_BEAM_PASSES,
        "selected_bins_by_method": selected_bins,
        "pca_rebuild_protocol": pca_protocol,
        "v5b_feature_budget": v5b_budget,
        "comparison_rule": (
            "Use OriginalTop1-WLS for the broad method leaderboard. Use the "
            "CommonTop1/CommonTop3 rows only to attribute a shared backend gain."
        ),
        "locked_results_must_not_be_used_for_tuning": args.run_mode == "locked_holdout",
        "evaluation_seed_rule": (
            "smoke/legacy development use cfg.seed; independent native and "
            "locked pools use pool_seed + 100000"
        ),
        "artifact_sha256": {
            key: _file_sha256(path) for key, path in artifact_paths.items()
        },
    }
    evaluation_protocol = {
        "source_result": str(source_dir),
        "dae_result": str(dae_dir),
        "development_eval_seed": int(cfg.get("seed", 42)),
        "pool_partitions": pool_partitions,
        "active_pool_seeds": list(pool_seeds),
        "trials_per_snr": int(trials),
        "top_k": COMMON_TOP_K,
        "min_peak_separation_bins": COMMON_MIN_PEAK_SEPARATION,
        "beam_max_pairs": COMMON_BEAM_MAX_PAIRS,
        "beam_passes": COMMON_BEAM_PASSES,
        "artifact_sha256": manifest["artifact_sha256"],
    }
    protocol_id = _manifest_hash(evaluation_protocol)
    manifest["evaluation_protocol_id"] = protocol_id
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    with open(output_dir / f"{prefix}_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    lag_limit = float(cfg.get("tdoa_lag_limit_samples", 48))
    area_size = (float(base_sim.area_size[0]), float(base_sim.area_size[1]))

    total = len(pool_seeds) * len(snr_values) * int(trials)
    print(f"[Config] mode={args.run_mode} device={device} total_trials={total}", flush=True)
    print(f"[Config] manifest={manifest['manifest_sha256']}", flush=True)
    print(f"[Config] output={output_dir}", flush=True)
    rows = []
    done = 0
    started = time.time()

    for pool_seed in pool_seeds:
        sim = _build_simulator(cfg, pool_seed)
        if sim.signal_len != base_sim.signal_len or tuple(sim.area_size) != tuple(base_sim.area_size):
            raise RuntimeError("locked simulator dimensions differ from frozen estimator")
        for snr_db in snr_values:
            eval_seed = (
                int(cfg.get("seed", 42))
                if args.run_mode in ("smoke", "dev_fair", "dev_mid_gate")
                else int(pool_seed) + 100000
            )
            x_noisy, x_clean, meta = sim.generate_urban_batch(
                int(trials), snr_db=float(snr_db), seed=eval_seed
            )
            validate_urban_batch(
                x_noisy, x_clean, meta, sim, expected_batch=int(trials)
            )
            raw = x_noisy.numpy()
            waveform_outputs = {
                "Raw": raw,
                "DAE-CR16": _run_waveform_model(dae, x_noisy, device),
            }
            for label, model in waveform_models.items():
                waveform_outputs[label] = _run_waveform_model(
                    model, x_noisy, device,
                )
            for trial in range(int(trials)):
                trial_started = time.time()
                los_idx = [int(v) for v in np.where(meta["los"][trial])[0]]
                pairs = [
                    (los_idx[a], los_idx[b])
                    for a in range(len(los_idx))
                    for b in range(a + 1, len(los_idx))
                ]
                native_by_method = {name: [] for name in FAIR_METHOD_ORDER}
                common_by_method = {name: [] for name in common_methods}
                true_lags = []
                for ui, uj in pairs:
                    raw_i = _complex_channel(raw, trial, ui)
                    raw_j = _complex_channel(raw, trial, uj)
                    native_by_method[v5b_method_label].append(
                        _native_estimator_top1(v5b, raw_i, raw_j, lag_limit)
                    )
                    if v5b_method_label in common_methods:
                        common_by_method[v5b_method_label].append(
                            posterior_topk(
                                v5b, raw_i, raw_j, lag_limit, COMMON_TOP_K,
                                COMMON_MIN_PEAK_SEPARATION,
                            )
                        )
                    for label, estimator in current_experts.items():
                        if label in native_by_method:
                            native_by_method[label].append(
                                _native_estimator_top1(
                                    estimator, raw_i, raw_j, lag_limit,
                                )
                            )
                    for label, estimator in legacy_experts.items():
                        if label in native_by_method:
                            native_by_method[label].append(
                                _native_estimator_top1(
                                    estimator, raw_i, raw_j, lag_limit,
                                )
                            )
                    for label, batch in waveform_outputs.items():
                        sig_i = _complex_channel(batch, trial, ui)
                        sig_j = _complex_channel(batch, trial, uj)
                        native_by_method[label].append(
                            _native_waveform_top1(sig_i, sig_j, lag_limit)
                        )
                        if label in common_methods:
                            common_by_method[label].append(
                                waveform_gcc_topk(
                                    sig_i, sig_j, lag_limit, COMMON_TOP_K,
                                    COMMON_MIN_PEAK_SEPARATION,
                                )
                            )
                    for label, estimator in common_estimators.items():
                        native_by_method[label].append(
                            _native_estimator_top1(
                                estimator, raw_i, raw_j, lag_limit,
                            )
                        )
                        if label in common_methods:
                            common_by_method[label].append(
                                estimator_score_topk(
                                    estimator, raw_i, raw_j, lag_limit,
                                    COMMON_TOP_K, COMMON_MIN_PEAK_SEPARATION,
                                )
                            )
                    for label, estimator in native_only_estimators.items():
                        native_by_method[label].append(
                            _native_estimator_top1(
                                estimator, raw_i, raw_j, lag_limit,
                            )
                        )
                    true_lags.append(float(
                        meta["delay_float"][trial, ui]
                        - meta["delay_float"][trial, uj]
                    ))

                snapshot = int(meta.get("snapshot_id", np.arange(trials))[trial])
                for method in FAIR_METHOD_ORDER:
                    solver_runs = [
                        ("OriginalTop1-WLS", "top1", native_by_method[method], "native"),
                    ]
                    if method in common_methods:
                        solver_runs.extend(
                            (label, mode, common_by_method[method], "common_score")
                            for label, mode in active_common_solvers
                        )
                    for solver_label, solver_mode, estimates, evidence_protocol in solver_runs:
                        solve_started = time.time()
                        result = localize_topk(
                            meta["uavs"][trial], pairs, estimates,
                            sim.fs, sim.c, area_size, mode=solver_mode,
                            max_pairs=COMMON_BEAM_MAX_PAIRS,
                            max_passes=COMMON_BEAM_PASSES,
                        )
                        pos = result["position"]
                        error = (
                            float(np.linalg.norm(pos - meta["source"][trial]))
                            if pos is not None else np.nan
                        )
                        selected = np.asarray(result["selected_lags"], dtype=float)
                        pair_error = np.abs(selected - np.asarray(true_lags, dtype=float))
                        rows.append({
                            "pool_seed": int(pool_seed),
                            "eval_seed": int(eval_seed),
                            "protocol_id": protocol_id,
                            "SNR_dB": float(snr_db),
                            "trial_idx": int(trial),
                            "snapshot_id": snapshot,
                            "cluster_id": f"{pool_seed}:{snapshot}",
                            "method": method,
                            "solver": solver_label,
                            "evidence_protocol": evidence_protocol,
                            "error_m": error,
                            "solver_success": bool(pos is not None),
                            "tdoa_abs_error_mean_samples": float(np.mean(pair_error)),
                            "tdoa_abs_error_median_samples": float(np.median(pair_error)),
                            "tdoa_within_1_rate": float(np.mean(pair_error <= 1.0)),
                            "tdoa_within_2_rate": float(np.mean(pair_error <= 2.0)),
                            "n_pairs": int(len(pairs)),
                            "n_pairs_changed": int(result["n_pairs_changed"]),
                            "n_solver_calls": int(result["n_solver_calls"]),
                            "wls_retry_used": bool((result.get("info") or {}).get(
                                "solver_retry_used", False
                            )),
                            "elapsed_s": float(time.time() - solve_started),
                        })
                done += 1
                if done % max(1, int(args.progress_every)) == 0:
                    elapsed = time.time() - started
                    eta = elapsed / done * (total - done)
                    print(
                        f"[Progress] {done}/{total} pool={pool_seed} snr={snr_db:g} "
                        f"trial={trial + 1}/{trials} elapsed={elapsed:.0f}s ETA={eta:.0f}s "
                        f"trial_s={time.time() - trial_started:.2f}",
                        flush=True,
                    )
            pd.DataFrame(rows).to_csv(
                output_dir / f"{prefix}_partial_rows.csv", index=False,
                encoding="utf-8",
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(final_rows_path, index=False, encoding="utf-8")
    summary = (
        frame.groupby(["pool_seed", "SNR_dB", "method", "solver"], dropna=False)
        .apply(_summarize, include_groups=False).reset_index()
    )
    summary.to_csv(
        output_dir / f"{prefix}_summary_by_pool_snr.csv", index=False,
        encoding="utf-8",
    )
    aggregate = (
        frame.groupby(["method", "solver"], dropna=False)
        .apply(_summarize, include_groups=False).reset_index()
    )
    aggregate.to_csv(
        output_dir / f"{prefix}_summary_all.csv", index=False,
        encoding="utf-8",
    )
    aggregate.loc[aggregate["solver"] == "OriginalTop1-WLS"].to_csv(
        output_dir / f"{prefix}_broad_leaderboard.csv", index=False,
        encoding="utf-8",
    )
    alignment_rows = []
    alignment_key = ["pool_seed", "SNR_dB", "trial_idx", "snapshot_id", "method"]
    for method in sorted(common_methods):
        subset = frame[frame["method"] == method]
        wide = subset.pivot_table(
            index=alignment_key, columns="solver",
            values=["error_m", "tdoa_abs_error_mean_samples"], aggfunc="first",
        )
        required = ("OriginalTop1-WLS", "CommonTop1-WLS")
        if any(("error_m", item) not in wide.columns for item in required):
            raise RuntimeError(f"missing native/common Top1 alignment rows for {method}")
        err_diff = np.abs(
            wide[("error_m", "OriginalTop1-WLS")]
            - wide[("error_m", "CommonTop1-WLS")]
        ).to_numpy(dtype=float)
        tdoa_diff = np.abs(
            wide[("tdoa_abs_error_mean_samples", "OriginalTop1-WLS")]
            - wide[("tdoa_abs_error_mean_samples", "CommonTop1-WLS")]
        ).to_numpy(dtype=float)
        alignment_rows.append({
            "method": method,
            "sample_count": int(len(wide)),
            "max_abs_localization_error_diff_m": float(np.nanmax(err_diff)),
            "mean_abs_localization_error_diff_m": float(np.nanmean(err_diff)),
            "max_abs_tdoa_mae_diff_samples": float(np.nanmax(tdoa_diff)),
            "mean_abs_tdoa_mae_diff_samples": float(np.nanmean(tdoa_diff)),
        })
    pd.DataFrame(alignment_rows).to_csv(
        output_dir / f"{prefix}_native_common_alignment.csv", index=False,
        encoding="utf-8",
    )
    attribution_rows = []
    frame["snr_bucket"] = np.where(
        frame["SNR_dB"] <= 0, "low",
        np.where(frame["SNR_dB"] >= 8, "high", "mid"),
    )
    for method in FAIR_METHOD_ORDER:
        if method not in common_methods:
            continue
        for bucket in ("low", "mid", "high", "all"):
            subset = frame[frame["method"] == method]
            if bucket != "all":
                subset = subset[subset["snr_bucket"] == bucket]
            stats = {
                solver: _summarize(subset[subset["solver"] == solver])
                for solver in manifest["solvers"]
            }
            top1_rmse = float(stats["CommonTop1-WLS"]["rmse_m"])
            geo_gain = (
                top1_rmse - float(stats["CommonTop3-GeoBeam"]["rmse_m"])
                if "CommonTop3-GeoBeam" in stats else np.nan
            )
            screen_gain = (
                top1_rmse - float(stats["CommonTop3-ScreenRefit"]["rmse_m"])
                if "CommonTop3-ScreenRefit" in stats else np.nan
            )
            enhanced_solvers = [
                label for label, _ in active_common_solvers
                if label != "CommonTop1-WLS"
            ]
            for solver in enhanced_solvers:
                gain, ci_low, ci_high, n_clusters = _cluster_bootstrap_solver_gain(
                    frame, method, solver, bucket,
                    baseline_solver="CommonTop1-WLS",
                    seed=int(cfg.get("seed", 42)) + len(attribution_rows) * 97,
                )
                attribution_rows.append({
                    "method": method,
                    "snr_bucket": bucket,
                    "solver": solver,
                    "rmse_gain_m": gain,
                    "cluster_ci95_low_m": ci_low,
                    "cluster_ci95_high_m": ci_high,
                    "cluster_count": n_clusters,
                    "mean_solver_calls": float(stats[solver]["mean_solver_calls"]),
                    "screen_gain_retention": (
                        screen_gain / geo_gain
                        if solver == "CommonTop3-ScreenRefit" and geo_gain > 1e-12
                        else np.nan
                    ),
                })
    attribution = pd.DataFrame(attribution_rows)
    attribution.to_csv(
        output_dir / f"{prefix}_solver_attribution.csv", index=False,
        encoding="utf-8",
    )
    _export_method_comparison(frame, output_dir, prefix)
    if args.run_mode == "dev_native_multi_pool":
        _export_poolwise_method_comparison(frame, output_dir, prefix)
    _plot_fairness_results(frame, output_dir, prefix)
    if prefix == "fair_topk_dev_mid" and (output_dir / "fair_topk_dev_rows.csv").exists():
        complete_frame, complete_prefix = _merge_existing_development_rows(output_dir)
        _export_basic_summaries(complete_frame, output_dir, complete_prefix)
        _export_method_comparison(complete_frame, output_dir, complete_prefix)
        _plot_fairness_results(complete_frame, output_dir, complete_prefix)
        print("[Analysis] base dev and mid gate merged", flush=True)
    print("\n" + aggregate.to_string(index=False), flush=True)
    print(f"[Done] elapsed={(time.time() - started) / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
