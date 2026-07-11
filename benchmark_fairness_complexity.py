# -*- coding: utf-8 -*-
"""PyCharm entry for development-stage accuracy/complexity accounting.

The benchmark reuses frozen models and the existing fair-development rows. It
measures the current implementation, including NumPy/Torch transfer overhead,
but excludes model loading and offline training. Results are development
diagnostics rather than hardware-independent theoretical FLOP counts.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import evaluate_topk_fairness as fair
from topk_localization import estimator_score_topk, posterior_topk, waveform_gcc_topk


PROJECT_ROOT = Path(__file__).resolve().parent
PYCHARM_SOURCE_RESULT = PROJECT_ROOT / "运行结果" / "20260709_120019"
PYCHARM_DAE_RESULT = PROJECT_ROOT / "运行结果" / "20260702_000925"
PYCHARM_SNR_DB = 0.0
PYCHARM_TRIALS = 10
PYCHARM_REPEATS = 3
PYCHARM_WARMUP_TRIALS = 1


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_batch(fn, denominator, repeats, device):
    result = fn()
    _sync(device)
    samples = []
    peak_extra = 0.0
    for _ in range(int(repeats)):
        if device.type == "cuda":
            before = float(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            before = 0.0
        _sync(device)
        started = time.perf_counter()
        result = fn()
        _sync(device)
        samples.append(1000.0 * (time.perf_counter() - started) / denominator)
        if device.type == "cuda":
            peak = float(torch.cuda.max_memory_allocated(device))
            peak_extra = max(peak_extra, max(0.0, peak - before) / (1024.0 ** 2))
    return result, np.asarray(samples, dtype=float), peak_extra


def _time_trial_builder(builder, trial_count, repeats, warmup, device):
    for trial in range(min(int(warmup), int(trial_count))):
        builder(trial)
    _sync(device)
    samples = []
    peak_extra = 0.0
    for _ in range(int(repeats)):
        for trial in range(int(trial_count)):
            if device.type == "cuda":
                before = float(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
            else:
                before = 0.0
            _sync(device)
            started = time.perf_counter()
            builder(trial)
            _sync(device)
            samples.append(1000.0 * (time.perf_counter() - started))
            if device.type == "cuda":
                peak = float(torch.cuda.max_memory_allocated(device))
                peak_extra = max(peak_extra, max(0.0, peak - before) / (1024.0 ** 2))
    return np.asarray(samples, dtype=float), peak_extra


def _module_counts(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in module.buffers())
    return int(trainable), int(buffers)


def _file_mb(path):
    path = Path(path)
    return float(path.stat().st_size / (1024.0 ** 2)) if path.exists() else np.nan


def _pairs_for_trial(meta, trial):
    los_idx = [int(v) for v in np.where(meta["los"][trial])[0]]
    return [
        (los_idx[a], los_idx[b])
        for a in range(len(los_idx)) for b in range(a + 1, len(los_idx))
    ]


def _complex(batch, trial, uav):
    return batch[trial, uav, 0, :] + 1j * batch[trial, uav, 1, :]


def _runtime_accuracy_rows(dev_rows, frontend, evidence):
    records = []
    for (method, solver), group in dev_rows.groupby(["method", "solver"]):
        protocol = "native" if solver == "OriginalTop1-WLS" else "common"
        evidence_key = (method, protocol)
        if evidence_key not in evidence:
            continue
        errors = group["error_m"].to_numpy(dtype=float)
        errors = errors[np.isfinite(errors)]
        frontend_ms = float(frontend.get(method, {}).get("mean_ms_per_trial", 0.0))
        evidence_ms = float(evidence[evidence_key]["mean_ms_per_trial"])
        solver_ms = float(1000.0 * group["elapsed_s"].mean())
        records.append({
            "method": method,
            "solver": solver,
            "evidence_protocol": protocol,
            "frontend_ms_per_trial": frontend_ms,
            "pair_evidence_ms_per_trial": evidence_ms,
            "solver_ms_per_trial": solver_ms,
            "total_online_ms_per_trial": frontend_ms + evidence_ms + solver_ms,
            "mean_wls_calls": float(group["n_solver_calls"].mean()),
            "development_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "development_tdoa_mae_samples": float(
                group["tdoa_abs_error_mean_samples"].mean()
            ),
            "valid_count": int(errors.size),
        })
    return pd.DataFrame(records)


def _plot_pareto(runtime, output_dir):
    methods = [
        "DAE-CR16", "V5B-Expert64", "DFT-train-power",
        "Cao2017-DFT-AML", "Zhai-CRLB-Decimation", "Hadamard", "PCA",
    ]
    solvers = ("OriginalTop1-WLS", "CommonTop3-GeoBeam", "CommonTop3-ScreenRefit")
    markers = {
        "OriginalTop1-WLS": "o",
        "CommonTop3-GeoBeam": "^",
        "CommonTop3-ScreenRefit": "s",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for method in methods:
        method_rows = runtime.loc[runtime["method"] == method]
        color = fair.METHOD_STYLE[method]["color"]
        for solver in solvers:
            row = method_rows.loc[method_rows["solver"] == solver]
            if row.empty:
                continue
            x = float(row["total_online_ms_per_trial"].iloc[0])
            for ax, metric in zip(
                axes, ("development_rmse_m", "development_tdoa_mae_samples")
            ):
                y = float(row[metric].iloc[0])
                ax.scatter(
                    x, y, s=48, marker=markers[solver], color=color,
                    edgecolor="#222222", linewidth=0.55, zorder=3,
                )
    axes[0].set_ylabel("Development localization RMSE (m)")
    axes[0].set_title("Localization accuracy vs. online runtime")
    axes[1].set_ylabel("Development TDOA MAE (samples)")
    axes[1].set_title("TDOA accuracy vs. online runtime")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Measured online time per multi-UAV trial (ms, log scale)")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    method_handles = [
        Line2D([0], [0], color=fair.METHOD_STYLE[m]["color"], marker="o",
               linestyle="", markersize=6, label=(
                   "V5B (CR11.6)" if m == "V5B-Expert64"
                   else fair.METHOD_DISPLAY[m]
               ))
        for m in methods
    ]
    solver_handles = [
        Line2D([0], [0], color="#555555", marker=markers[s], linestyle="",
               markersize=6, label=s.replace("Common", ""))
        for s in solvers
    ]
    fig.legend(
        handles=method_handles + solver_handles, loc="lower center", ncol=5,
        frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=8.2,
    )
    fig.suptitle("Development Accuracy-Complexity Pareto View", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.94))
    path = Path(output_dir) / "FigFair3_Accuracy_Complexity_Pareto.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    source_dir = PYCHARM_SOURCE_RESULT.resolve()
    dae_dir = PYCHARM_DAE_RESULT.resolve()
    output_dir = source_dir / "tables" / "topk_fairness"
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_rows_path = output_dir / "fair_topk_dev_complete_rows.csv"
    if not dev_rows_path.exists():
        dev_rows_path = output_dir / "fair_topk_dev_rows.csv"
    if not dev_rows_path.exists():
        raise FileNotFoundError(dev_rows_path)
    data = fair._load_pickle(source_dir / "plot_data.pkl")
    cfg = data.get("config", {})
    seed = int(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = fair._build_simulator(cfg, seed)
    v5b = fair._load_v5b(source_dir, data, simulator, device)
    dae = fair._load_dae(dae_dir, device)
    waveform_models, pca_cache, pca_protocol = fair._load_or_build_waveform_baselines(
        data, simulator, device, output_dir, seed,
    )
    direct_estimators, native_only, _ = fair._build_direct_estimators(
        data, simulator.signal_len,
    )
    x_noisy, _, meta = simulator.generate_urban_batch(
        int(PYCHARM_TRIALS), snr_db=float(PYCHARM_SNR_DB), seed=seed,
    )
    raw = x_noisy.numpy()
    n_observations = int(x_noisy.shape[0] * x_noisy.shape[1])
    lag_limit = float(cfg.get("tdoa_lag_limit_samples", 48))
    print(
        f"[Config] device={device} snr={PYCHARM_SNR_DB:g} "
        f"trials={PYCHARM_TRIALS} repeats={PYCHARM_REPEATS}",
        flush=True,
    )

    waveform_outputs = {"Raw": raw}
    frontend = {"Raw": {"mean_ms_per_trial": 0.0, "peak_extra_gpu_mb": 0.0}}
    for label, model in {
        "DAE-CR16": dae,
        "Hadamard": waveform_models["Hadamard"],
        "PCA": waveform_models["PCA"],
    }.items():
        output, timing, peak = _time_batch(
            lambda model=model: fair._run_waveform_model(model, x_noisy, device),
            denominator=n_observations, repeats=PYCHARM_REPEATS, device=device,
        )
        waveform_outputs[label] = output
        frontend[label] = {
            "mean_ms_per_observation": float(np.mean(timing)),
            "median_ms_per_observation": float(np.median(timing)),
            "p95_ms_per_observation": float(np.percentile(timing, 95)),
            "mean_ms_per_trial": float(np.mean(timing) * x_noisy.shape[1]),
            "peak_extra_gpu_mb": float(peak),
        }

    pairs_by_trial = [_pairs_for_trial(meta, trial) for trial in range(PYCHARM_TRIALS)]
    evidence = {}

    def benchmark(label, protocol, builder):
        timing, peak = _time_trial_builder(
            builder, PYCHARM_TRIALS, PYCHARM_REPEATS,
            PYCHARM_WARMUP_TRIALS, device,
        )
        evidence[(label, protocol)] = {
            "mean_ms_per_trial": float(np.mean(timing)),
            "median_ms_per_trial": float(np.median(timing)),
            "p95_ms_per_trial": float(np.percentile(timing, 95)),
            "peak_extra_gpu_mb": float(peak),
            "mean_pairs_per_trial": float(np.mean([len(v) for v in pairs_by_trial])),
        }

    for label, batch in waveform_outputs.items():
        def native_builder(trial, batch=batch):
            return [
                fair._native_waveform_top1(
                    _complex(batch, trial, i), _complex(batch, trial, j), lag_limit,
                )
                for i, j in pairs_by_trial[trial]
            ]

        def common_builder(trial, batch=batch):
            return [
                waveform_gcc_topk(
                    _complex(batch, trial, i), _complex(batch, trial, j),
                    lag_limit, fair.COMMON_TOP_K, fair.COMMON_MIN_PEAK_SEPARATION,
                )
                for i, j in pairs_by_trial[trial]
            ]

        benchmark(label, "native", native_builder)
        benchmark(label, "common", common_builder)

    def v5_native(trial):
        return [
            fair._native_estimator_top1(
                v5b, _complex(raw, trial, i), _complex(raw, trial, j), lag_limit,
            )
            for i, j in pairs_by_trial[trial]
        ]

    def v5_common(trial):
        return [
            posterior_topk(
                v5b, _complex(raw, trial, i), _complex(raw, trial, j),
                lag_limit, fair.COMMON_TOP_K, fair.COMMON_MIN_PEAK_SEPARATION,
            )
            for i, j in pairs_by_trial[trial]
        ]

    benchmark("V5B-Expert64", "native", v5_native)
    benchmark("V5B-Expert64", "common", v5_common)

    for label, estimator in direct_estimators.items():
        def native_builder(trial, estimator=estimator):
            return [
                fair._native_estimator_top1(
                    estimator, _complex(raw, trial, i), _complex(raw, trial, j),
                    lag_limit,
                )
                for i, j in pairs_by_trial[trial]
            ]

        def common_builder(trial, estimator=estimator):
            return [
                estimator_score_topk(
                    estimator, _complex(raw, trial, i), _complex(raw, trial, j),
                    lag_limit, fair.COMMON_TOP_K,
                    fair.COMMON_MIN_PEAK_SEPARATION,
                )
                for i, j in pairs_by_trial[trial]
            ]

        benchmark(label, "native", native_builder)
        benchmark(label, "common", common_builder)

    for label, estimator in native_only.items():
        def native_builder(trial, estimator=estimator):
            return [
                fair._native_estimator_top1(
                    estimator, _complex(raw, trial, i), _complex(raw, trial, j),
                    lag_limit,
                )
                for i, j in pairs_by_trial[trial]
            ]

        benchmark(label, "native", native_builder)

    dae_params, dae_buffers = _module_counts(dae)
    dae_encoder_params = sum(
        p.numel() for module in (dae.enc, dae.fc_enc) for p in module.parameters()
        if p.requires_grad
    )
    v5_models = [v5b.experts[name].model for name in ("power", "geohybrid")]
    v5_params = sum(_module_counts(model)[0] for model in v5_models)
    v5_buffers = sum(_module_counts(model)[1] for model in v5_models)
    v5_budget = fair._v5b_feature_budget(v5b, simulator.signal_len)
    pca_model = waveform_models["PCA"]
    had_model = waveform_models["Hadamard"]
    static = {
        "Raw": (0, 0, 2048, False, "none", np.nan),
        "DAE-CR16": (
            dae_params, dae_buffers, 128, True, "supervised neural training",
            _file_mb(dae_dir / "model_cr16.pt"),
        ),
        "V5B-Expert64": (
            v5_params, v5_buffers, v5_budget["transmitted_real_scalars"], False,
            "two hard-bin likelihood experts; over-budget CR16 diagnostic",
            _file_mb(source_dir / "model_v5a1_power64.pt")
            + _file_mb(source_dir / "model_v5a1_geohybrid64.pt"),
        ),
        "Hadamard": (
            0, int(had_model.rows.numel()), 128, True, "fixed transform", np.nan,
        ),
        "PCA": (
            0, int(pca_model.mean.numel() + pca_model.components.numel()),
            128, True, "offline PCA fit", _file_mb(pca_cache),
        ),
    }
    for label in direct_estimators:
        static[label] = (0, 64, 128, False, "fixed selected-bin estimator", np.nan)
    for label in native_only:
        static[label] = (0, 64, 128, False, "fixed hybrid estimator", np.nan)

    model_rows = []
    for method in fair.FAIR_METHOD_ORDER:
        params, buffers, feature_dim, reconstructs, fit, artifact_mb = static[method]
        peak = max(
            [frontend.get(method, {}).get("peak_extra_gpu_mb", 0.0)]
            + [value["peak_extra_gpu_mb"] for key, value in evidence.items()
               if key[0] == method]
        )
        model_rows.append({
            "method": method,
            "transmitted_real_scalars": int(feature_dim),
            "effective_cr_vs_2048_real": float(2048 / feature_dim),
            "trainable_parameters": int(params),
            "encoder_trainable_parameters": (
                int(dae_encoder_params) if method == "DAE-CR16" else np.nan
            ),
            "stored_nontrainable_scalars": int(buffers),
            "parameter_memory_mb_fp32": float(4 * params / (1024.0 ** 2)),
            "artifact_size_mb": artifact_mb,
            "reconstructs_waveform": bool(reconstructs),
            "offline_fit_or_training": fit,
            "measured_peak_extra_gpu_mb": float(peak),
            "strict_cr16_budget_passed": bool(
                method != "V5B-Expert64" or v5_budget["strict_cr16_budget_passed"]
            ),
        })
    model_frame = pd.DataFrame(model_rows)
    runtime_frame = _runtime_accuracy_rows(
        pd.read_csv(dev_rows_path), frontend, evidence,
    )
    frontend_rows = [
        {"method": method, **values} for method, values in frontend.items()
    ]
    evidence_rows = [
        {"method": method, "protocol": protocol, **values}
        for (method, protocol), values in evidence.items()
    ]
    model_frame.to_csv(
        output_dir / "fairness_model_complexity.csv", index=False, encoding="utf-8",
    )
    pd.DataFrame(frontend_rows).to_csv(
        output_dir / "fairness_frontend_runtime.csv", index=False, encoding="utf-8",
    )
    pd.DataFrame(evidence_rows).to_csv(
        output_dir / "fairness_evidence_runtime.csv", index=False, encoding="utf-8",
    )
    runtime_frame.to_csv(
        output_dir / "fairness_end_to_end_runtime.csv", index=False, encoding="utf-8",
    )
    pareto_path = _plot_pareto(runtime_frame, output_dir)
    print("\n" + model_frame.to_string(index=False), flush=True)
    print("\n" + runtime_frame.to_string(index=False), flush=True)
    print(f"[Plot] saved {pareto_path}", flush=True)
    print("[Done] complexity benchmark complete", flush=True)


if __name__ == "__main__":
    main()
