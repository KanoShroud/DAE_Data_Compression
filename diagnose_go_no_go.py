"""
Go/no-go diagnostics for the current task-aware DAE route.

This script does not train models. It loads an existing NestedFrequencyPairwiseDAE
result directory, evaluates the latent task head as a direct TDOA estimator, and
exports a compact decision package:

  - learned task-head direct TDOA + WLS against classical direct baselines
  - waveform decoder reference from the same v3 model
  - fake-signal sanity checks for phase randomization, mismatched pairs, and
    alternate geometry seed
  - a go/no-go decision table

The intent is to decide whether the next research step should be task-direct,
waveform residual nested decoding, frequency-selection baselines, or stopping the
current DAE route.
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baselines import HadamardProjectionBaseline
from evaluate import run_urban_method_comparison
from export_results import export_method_baselines
from model import DAE, NestedFrequencyPairwiseDAE
from signal_gen import SignalSimulator
from task_baselines import (
    Cao2017DFTAMLEstimator,
    Cao2020SegmentedFCEstimator,
    DirectDFTTDOAEstimator,
    GeoHybridDFTTDOAEstimator,
    ZhaiPhaseSuperpositionEstimator,
    _peak_from_score,
    _quality_from_score,
)


DEFAULT_SOURCE_RESULT = (
    Path("运行结果")
    / "FreqDAE"
    / "20260707_143945_FreqDAE_v3嵌套任务头"
)
DEFAULT_OUTPUT_ROOT = Path("运行结果")
DEFAULT_SNRS = [-10.0, 0.0, 10.0, 20.0]
DEFAULT_CR_LIST = [4, 8, 16]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _latest_result_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "plot_data.pkl").exists()]
    if not candidates:
        raise FileNotFoundError(f"No result directory with plot_data.pkl under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _read_plot_data(result_dir: Path) -> dict:
    path = result_dir / "plot_data.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing plot_data.pkl: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _write_df(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        md_path.write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {md_path}")


def _load_state_dict(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported model checkpoint format: {path}")
    return state


def _load_dae_model(model_dir: Path, cr: int, signal_len: int, device: torch.device) -> DAE:
    model_path = model_dir / f"model_cr{cr}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing DAE model: {model_path}")
    # The Chen-style DAE class is fixed to the project signal length.
    if int(signal_len) != 1024:
        raise ValueError("The current DAE baseline implementation expects signal_len=1024")
    model = DAE(cr=cr).to(device)
    model.load_state_dict(_load_state_dict(model_path, device))
    model.eval()
    return model


def _load_nested_model(model_dir: Path, cr: int, signal_len: int,
                       device: torch.device) -> NestedFrequencyPairwiseDAE:
    model_path = model_dir / f"model_cr{cr}.pt"
    if not model_path.exists():
        # v3 uses one shared supernet. Fall back to CR4 if only that file exists.
        model_path = model_dir / "model_cr4.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing nested model: {model_dir}/model_cr{cr}.pt")
    model = NestedFrequencyPairwiseDAE(cr=cr, signal_len=signal_len).to(device)
    model.load_state_dict(_load_state_dict(model_path, device))
    model.set_active_cr(cr)
    model.eval()
    return model


def _build_simulator(cfg: dict, seed: int | None = None) -> SignalSimulator:
    return SignalSimulator(
        signal_len=int(cfg.get("signal_len") or 1024),
        channel_mode=str(cfg.get("channel_mode", "fixed")),
        n_fixed_channels=int(cfg.get("n_fixed_channels", 50)),
        channel_pool_seed=int(seed if seed is not None else cfg.get("seed", 42)),
        nlos_prob=float(cfg.get("nlos_prob", 0.0)),
        delay_label_mode=str(cfg.get("delay_label_mode", "los")),
        snr_train_range=tuple(cfg.get("train_snr_range", (-10, 20))),
        multipath_scale=float(cfg.get("multipath_scale", 0.2)),
        scenario_mode=str(cfg.get("scenario_mode", "urban8")),
        normalization_mode=str(cfg.get("normalization_mode", "per_observation_noisy_rms")),
        urban_base_delay=float(cfg.get("urban_base_delay", 16)),
        urban_min_los=int(cfg.get("urban_min_los", 5)),
        urban_train_los_only=bool(cfg.get("urban_train_los_only", True)),
    )


def _sig_to_tensor(sig: np.ndarray, device: torch.device) -> torch.Tensor:
    sig = np.asarray(sig, dtype=np.complex64)
    x = np.stack([sig.real, sig.imag], axis=0)[None, :, :]
    return torch.from_numpy(x).float().to(device)


def _phase_randomize(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.complex64)
    spec = np.fft.fft(sig)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=sig.shape[-1])
    out = np.fft.ifft(np.abs(spec) * np.exp(1j * phase))
    in_rms = np.sqrt(np.mean(np.abs(sig) ** 2)) + 1e-12
    out_rms = np.sqrt(np.mean(np.abs(out) ** 2)) + 1e-12
    return (out * (in_rms / out_rms)).astype(np.complex64)


class LearnedTaskHeadTDOAEstimator:
    """Direct TDOA estimator using NestedFrequencyPairwiseDAE task-head logits."""

    def __init__(self, model: NestedFrequencyPairwiseDAE, device: torch.device,
                 signal_len: int = 1024, label: str = "TaskHead-CR16",
                 input_mode: str = "normal", seed: int = 42):
        self.model = model
        self.device = device
        self.signal_len = int(signal_len)
        self.label = str(label)
        self.input_mode = str(input_mode)
        self.rng = np.random.default_rng(int(seed))
        self.lags = np.arange(self.signal_len, dtype=float) - (self.signal_len // 2)
        self.last_info = {}

    def _transform(self, sig: np.ndarray) -> np.ndarray:
        if self.input_mode == "phase_randomized":
            return _phase_randomize(sig, self.rng)
        if self.input_mode == "zero":
            return np.zeros_like(sig, dtype=np.complex64)
        return np.asarray(sig, dtype=np.complex64)

    @staticmethod
    def _softmax_np(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / (np.sum(exp) + 1e-12)

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        sig_i = self._transform(sig_i)
        sig_j = self._transform(sig_j)
        with torch.no_grad():
            x_i = _sig_to_tensor(sig_i, self.device)
            x_j = _sig_to_tensor(sig_j, self.device)
            _, z_i = self.model(x_i, return_latent=True)
            _, z_j = self.model(x_j, return_latent=True)
            logits = self.model.pair_task_logits_from_latent(z_i, z_j)
            logits_np = logits.detach().cpu().numpy()[0]

        prob = self._softmax_np(logits_np)
        search_mask = np.ones_like(prob, dtype=bool)
        if lag_limit_samples is not None:
            search_mask = np.abs(self.lags) <= float(lag_limit_samples)
        idx, lag, search_mask = _peak_from_score(
            self.lags, prob, search_mask, sub_sample=sub_sample
        )
        weight, peak, sidelobe_ratio = _quality_from_score(prob, idx, search_mask)
        active = prob[search_mask] if np.any(search_mask) else prob
        entropy = -float(np.sum(active * np.log(active + 1e-12)))
        entropy_norm = entropy / (math.log(max(active.size, 2)) + 1e-12)
        half_width = float(np.sum(active >= 0.5 * float(prob[idx])))
        self.last_info = {
            "entropy_norm": entropy_norm,
            "peak_prob": float(prob[idx]),
            "half_width_lags": half_width,
            "sidelobe_ratio": float(sidelobe_ratio),
            "weight": float(weight),
        }
        if not return_quality:
            return lag
        return lag, weight, peak, sidelobe_ratio

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        metrics = _task_head_pair_metrics(
            self, batch_np, meta, fs, c,
            use_los_only=use_los_only,
            sub_sample=sub_sample,
            lag_limit_samples=lag_limit_samples,
            mode="normal",
        )
        return {f"direct_{k}": v for k, v in metrics.items()}


def _task_head_pair_metrics(estimator: LearnedTaskHeadTDOAEstimator, batch_np, meta,
                            fs, c, use_los_only=True, sub_sample=True,
                            lag_limit_samples=None, mode="normal",
                            seed=42) -> dict:
    rng = np.random.default_rng(int(seed))
    abs_err = []
    within1 = []
    within2 = []
    entropy = []
    peak_prob = []
    width = []
    side = []
    weights = []
    n = int(batch_np.shape[0])
    for bi in range(n):
        los_idx = np.where(meta["los"][bi])[0] if use_los_only else np.arange(batch_np.shape[1])
        los_idx = [int(v) for v in los_idx]
        if len(los_idx) < 2:
            continue
        for a in range(len(los_idx)):
            for b in range(a + 1, len(los_idx)):
                ui, uj = los_idx[a], los_idx[b]
                sig_i = batch_np[bi, ui, 0, :] + 1j * batch_np[bi, ui, 1, :]
                sig_j = batch_np[bi, uj, 0, :] + 1j * batch_np[bi, uj, 1, :]
                if mode == "shuffle_second_obs":
                    sj = (bi + 1 + int(rng.integers(0, max(n - 1, 1)))) % n
                    sig_j = batch_np[sj, uj, 0, :] + 1j * batch_np[sj, uj, 1, :]
                elif mode == "phase_randomized":
                    sig_i = _phase_randomize(sig_i, rng)
                    sig_j = _phase_randomize(sig_j, rng)
                elif mode == "zero":
                    sig_i = np.zeros_like(sig_i, dtype=np.complex64)
                    sig_j = np.zeros_like(sig_j, dtype=np.complex64)

                tau, weight, _, sidelobe_ratio = estimator.estimate_pair(
                    sig_i, sig_j,
                    sub_sample=sub_sample,
                    return_quality=True,
                    lag_limit_samples=lag_limit_samples,
                )
                true_tau = (meta["distances"][bi, ui] - meta["distances"][bi, uj]) / (c / fs)
                err = float(abs(float(tau) - float(true_tau)))
                abs_err.append(err)
                within1.append(err <= 1.0)
                within2.append(err <= 2.0)
                info = getattr(estimator, "last_info", {})
                entropy.append(float(info.get("entropy_norm", np.nan)))
                peak_prob.append(float(info.get("peak_prob", np.nan)))
                width.append(float(info.get("half_width_lags", np.nan)))
                side.append(float(sidelobe_ratio))
                weights.append(float(weight))

    arr = np.asarray(abs_err, dtype=float)
    warr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return {
            "mean_abs_tdoa_error_samples": float("nan"),
            "median_abs_tdoa_error_samples": float("nan"),
            "within_1_sample_rate": float("nan"),
            "within_2_sample_rate": float("nan"),
            "weighted_abs_tdoa_error_samples": float("nan"),
            "entropy_norm": float("nan"),
            "peak_prob": float("nan"),
            "half_width_lags": float("nan"),
            "sidelobe_ratio": float("nan"),
            "n_pairs": 0,
        }
    weighted = float(np.sum(warr * arr) / (np.sum(warr) + 1e-12))
    return {
        "mean_abs_tdoa_error_samples": float(np.mean(arr)),
        "median_abs_tdoa_error_samples": float(np.median(arr)),
        "within_1_sample_rate": float(np.mean(within1)),
        "within_2_sample_rate": float(np.mean(within2)),
        "weighted_abs_tdoa_error_samples": weighted,
        "entropy_norm": float(np.nanmean(entropy)),
        "peak_prob": float(np.nanmean(peak_prob)),
        "half_width_lags": float(np.nanmean(width)),
        "sidelobe_ratio": float(np.nanmean(side)),
        "n_pairs": int(arr.size),
    }


def _make_direct_estimators(meta: dict, signal_len: int) -> dict:
    selected = meta.get("dft_selected_bins_by_method", {}) or {}
    dft_source = selected.get("DFT-train-power")
    if dft_source is None:
        dft_source = selected.get("DFT-SCS-lite")
    estimators = {}
    if dft_source is not None:
        estimators["DFT"] = DirectDFTTDOAEstimator(
            dft_source, signal_len=signal_len, label="DFT"
        )
        estimators["Cao2017-DFT-AML"] = Cao2017DFTAMLEstimator(
            dft_source, signal_len=signal_len, label="Cao2017-DFT-AML"
        )
    if selected.get("DFT-Fisher") is not None:
        estimators["DFT-Fisher-Direct"] = DirectDFTTDOAEstimator(
            selected["DFT-Fisher"], signal_len=signal_len, label="DFT-Fisher-Direct"
        )
    if selected.get("DFT-GeoAmbi") is not None:
        if dft_source is not None:
            estimators["GeoHybrid-OverBudget-C64F64"] = GeoHybridDFTTDOAEstimator(
                dft_source, selected["DFT-GeoAmbi"],
                signal_len=signal_len, label="GeoHybrid-OverBudget-C64F64",
                budget_limit_complex=signal_len // 16,
                budget_note="legacy go/no-go over-budget diagnostic"
            )
            estimators["GeoHybrid-OverBudget-C64F64-PHAT"] = GeoHybridDFTTDOAEstimator(
                dft_source, selected["DFT-GeoAmbi"],
                signal_len=signal_len, label="GeoHybrid-OverBudget-C64F64-PHAT",
                fine_mode="phat_gated", fine_gain=1.10,
                budget_limit_complex=signal_len // 16,
                budget_note="legacy go/no-go PHAT over-budget diagnostic"
            )
        estimators["GeoAmbi-DFT-Direct"] = DirectDFTTDOAEstimator(
            selected["DFT-GeoAmbi"], signal_len=signal_len,
            label="GeoAmbi-DFT-Direct"
        )
        estimators["GeoAmbi-DFT-AML"] = Cao2017DFTAMLEstimator(
            selected["DFT-GeoAmbi"], signal_len=signal_len,
            label="GeoAmbi-DFT-AML"
        )
        estimators["GeoAmbi-DFT-PHAT"] = DirectDFTTDOAEstimator(
            selected["DFT-GeoAmbi"], signal_len=signal_len,
            label="GeoAmbi-DFT-PHAT", phat=True
        )
    if selected.get("DFT-Cao2020-CRB") is not None:
        estimators["Cao2020-HighFC"] = Cao2020SegmentedFCEstimator(
            selected["DFT-Cao2020-CRB"], signal_len=signal_len,
            label="Cao2020-HighFC"
        )
    if selected.get("DFT-Zhai-CRLB") is not None:
        estimators["Zhai-CRLB-Decimation"] = Cao2017DFTAMLEstimator(
            selected["DFT-Zhai-CRLB"], signal_len=signal_len,
            label="Zhai-CRLB-Decimation"
        )
        estimators["Zhai-Phase-Superposition"] = ZhaiPhaseSuperpositionEstimator(
            selected["DFT-Zhai-CRLB"], signal_len=signal_len,
            label="Zhai-Phase-Superposition"
        )
    return estimators


def _method_segment_summary(method_results, high_snr_min=10.0, low_snr_max=0.0) -> pd.DataFrame:
    results, snr_range = method_results
    rows = []
    for label in results.get("method_order", []):
        method = results.get("methods", {}).get(label, {})
        if not method:
            continue
        for metric_key, metric_name in [
            ("rmse", "mean_rmse_m"),
            ("median", "median_rmse_m"),
            ("trimmed_rmse", "trimmed_rmse_m"),
            ("outlier_gt20_rate", "outlier_gt20_rate"),
        ]:
            vals = np.asarray(method.get(metric_key, []), dtype=float)
            if vals.size != len(snr_range):
                continue
            rows.append({
                "method": label,
                "metric": metric_name,
                "all": float(np.nanmean(vals)),
                "low_snr": float(np.nanmean(vals[np.asarray(snr_range) <= low_snr_max])),
                "high_snr": float(np.nanmean(vals[np.asarray(snr_range) >= high_snr_min])),
            })
    return pd.DataFrame(rows)


def _diagnostic_segment_summary(method_results, high_snr_min=10.0, low_snr_max=0.0) -> pd.DataFrame:
    results, snr_range = method_results
    rows = []
    snr_arr = np.asarray(snr_range, dtype=float)
    for label, diag in results.get("method_diagnostics", {}).items():
        for key, values in diag.items():
            try:
                vals = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                continue
            if vals.size != len(snr_arr):
                continue
            rows.append({
                "method": label,
                "metric": key,
                "all": float(np.nanmean(vals)),
                "low_snr": float(np.nanmean(vals[snr_arr <= low_snr_max])),
                "high_snr": float(np.nanmean(vals[snr_arr >= high_snr_min])),
            })
    return pd.DataFrame(rows)


def _plot_go_no_go(method_results, out_dir: Path, filename="Fig_GoNoGo_TaskHead_Direct.svg"):
    results, snr_range = method_results
    order = results.get("method_order", [])
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    palette = {
        "Raw": "#4d4d4d",
        "DAE-CR16": "#3182bd",
        "DAE-CR4": "#08519c",
        "Hadamard": "#8c510a",
        "DFT": "#756bb1",
        "Cao2017-DFT-AML": "#bcbddc",
        "Zhai-CRLB-Decimation": "#31a354",
        "GeoHybrid-OverBudget-C64F64": "#636363",
        "GeoHybrid-OverBudget-C64F64-PHAT": "#969696",
        "TaskHead-CR4": "#e6550d",
        "TaskHead-CR8": "#fd8d3c",
        "TaskHead-CR16": "#fdae6b",
        "V3-Waveform-CR16": "#636363",
    }
    for label in order:
        if label in ("Clean", "Geometry"):
            continue
        method = results["methods"].get(label)
        if not method:
            continue
        y = np.asarray(method.get("rmse", []), dtype=float)
        if y.size != len(snr_range):
            continue
        lw = 1.4 if label.startswith("TaskHead") else 1.15
        ax.plot(snr_range, y, marker="o", markersize=3.0, linewidth=lw,
                label=label, color=palette.get(label, None))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean localization RMSE (m)")
    ax.set_title("Go/No-Go: Learned Task-Head Direct TDOA vs Baselines")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.92)
    path = out_dir / filename
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


def _value(summary: pd.DataFrame, method: str, metric: str, segment: str) -> float:
    row = summary[(summary["method"] == method) & (summary["metric"] == metric)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][segment])


def _build_decision_table(method_summary: pd.DataFrame,
                          diag_summary: pd.DataFrame,
                          sanity_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    task_methods = [m for m in method_summary["method"].unique() if str(m).startswith("TaskHead")]
    task_high = {
        m: _value(method_summary, m, "mean_rmse_m", "high_snr") for m in task_methods
    }
    best_task = min(task_high, key=task_high.get) if task_high else None
    best_task_high = task_high.get(best_task, float("nan")) if best_task else float("nan")

    classical_candidates = [
        "DFT", "Cao2017-DFT-AML", "Cao2020-HighFC",
        "Zhai-CRLB-Decimation", "Zhai-Phase-Superposition", "Hadamard"
    ]
    classical_high = {
        m: _value(method_summary, m, "mean_rmse_m", "high_snr")
        for m in classical_candidates
        if np.isfinite(_value(method_summary, m, "mean_rmse_m", "high_snr"))
    }
    best_classical = min(classical_high, key=classical_high.get) if classical_high else None
    best_classical_high = (
        classical_high.get(best_classical, float("nan")) if best_classical else float("nan")
    )

    v3_wf_high = _value(method_summary, "V3-Waveform-CR16", "mean_rmse_m", "high_snr")
    task_tdoa_high = (
        _value(diag_summary, best_task, "direct_mean_abs_tdoa_error_samples", "high_snr")
        if best_task else float("nan")
    )
    wf_tdoa_high = _value(
        diag_summary, "V3-Waveform-CR16", "gcc_mean_abs_tdoa_error_samples", "high_snr"
    )
    beats_wf = (
        np.isfinite(best_task_high) and np.isfinite(v3_wf_high)
        and best_task_high < 0.8 * v3_wf_high
        and (not np.isfinite(wf_tdoa_high) or task_tdoa_high < 0.7 * wf_tdoa_high)
    )
    rows.append({
        "test": "G1_task_head_beats_v3_waveform",
        "status": "PASS" if beats_wf else "FAIL",
        "metric": (
            f"best_task={best_task}, task_high_rmse={best_task_high:.3g}, "
            f"v3_waveform_high_rmse={v3_wf_high:.3g}, "
            f"task_high_tdoa={task_tdoa_high:.3g}, waveform_high_tdoa={wf_tdoa_high:.3g}"
        ),
        "decision_meaning": (
            "PASS supports a task-direct route; FAIL means the task head has not "
            "yet justified replacing the waveform decoder."
        ),
    })

    competes = (
        np.isfinite(best_task_high) and np.isfinite(best_classical_high)
        and best_task_high <= 1.25 * best_classical_high
    )
    rows.append({
        "test": "G2_task_head_competes_with_classical_direct",
        "status": "PASS" if competes else "WARN",
        "metric": (
            f"best_task={best_task}:{best_task_high:.3g}, "
            f"best_classical={best_classical}:{best_classical_high:.3g}"
        ),
        "decision_meaning": (
            "PASS means learned latent is in the direct-baseline competition zone; "
            "WARN means frequency-selection/hybrid methods may deserve priority."
        ),
    })

    cr4 = _value(method_summary, "TaskHead-CR4", "mean_rmse_m", "high_snr")
    cr16 = _value(method_summary, "TaskHead-CR16", "mean_rmse_m", "high_snr")
    cr_sep = np.isfinite(cr4) and np.isfinite(cr16) and cr4 <= cr16 - 1.0
    rows.append({
        "test": "G3_cr_extra_dimensions_used",
        "status": "PASS" if cr_sep else "FAIL",
        "metric": f"TaskHead-CR4_high_rmse={cr4:.3g}, TaskHead-CR16_high_rmse={cr16:.3g}",
        "decision_meaning": (
            "FAIL means nested extra dimensions still do not create a useful CR ladder."
        ),
    })

    for sanity_name in ["phase_randomized", "shuffle_second_obs", "zero"]:
        normal = sanity_df[(sanity_df["test"] == "normal") & (sanity_df["cr"] == 16)]
        sanity = sanity_df[(sanity_df["test"] == sanity_name) & (sanity_df["cr"] == 16)]
        if normal.empty or sanity.empty:
            status = "NOT_RUN"
            metric = ""
        else:
            n_err = float(normal.iloc[0]["mean_abs_tdoa_error_samples"])
            s_err = float(sanity.iloc[0]["mean_abs_tdoa_error_samples"])
            n_w1 = float(normal.iloc[0]["within_1_sample_rate"])
            s_w1 = float(sanity.iloc[0]["within_1_sample_rate"])
            degraded = (s_err >= 2.0 * max(n_err, 1e-9)) or (s_w1 <= n_w1 - 0.2)
            status = "PASS" if degraded else "WARN"
            metric = (
                f"normal_err={n_err:.3g}, {sanity_name}_err={s_err:.3g}, "
                f"normal_within1={n_w1:.3g}, {sanity_name}_within1={s_w1:.3g}"
            )
        rows.append({
            "test": f"G4_sanity_{sanity_name}",
            "status": status,
            "metric": metric,
            "decision_meaning": (
                "PASS means task head is sensitive to signal consistency; WARN "
                "suggests geometry/lag-prior shortcuts may be present."
            ),
        })

    normal = sanity_df[(sanity_df["test"] == "normal") & (sanity_df["cr"] == 16)]
    alt = sanity_df[(sanity_df["test"] == "geometry_seed_shift") & (sanity_df["cr"] == 16)]
    if normal.empty or alt.empty:
        geom_status = "NOT_RUN"
        geom_metric = ""
    else:
        n_err = float(normal.iloc[0]["mean_abs_tdoa_error_samples"])
        a_err = float(alt.iloc[0]["mean_abs_tdoa_error_samples"])
        geom_status = "PASS" if a_err <= 1.5 * max(n_err, 1e-9) else "WARN"
        geom_metric = f"normal_err={n_err:.3g}, alt_geometry_err={a_err:.3g}"
    rows.append({
        "test": "G5_geometry_seed_shift_generalization",
        "status": geom_status,
        "metric": geom_metric,
        "decision_meaning": (
            "WARN means geometry/channel changes should be treated as a blocking "
            "risk before claiming learned compression generality."
        ),
    })

    rows.append({
        "test": "G6_latent_quantization_8bit_4bit",
        "status": "NOT_RUN",
        "metric": "not implemented in first diagnostic package",
        "decision_meaning": (
            "Run only if task-head direct passes G1-G5; otherwise bit-budget tests "
            "are not informative."
        ),
    })
    return pd.DataFrame(rows)


def run_go_no_go(args) -> Path:
    source_result = Path(args.source_result)
    if not source_result.exists():
        source_result = _latest_result_dir(Path("运行结果"))
    data = _read_plot_data(source_result)
    cfg = dict(data.get("config", {}))
    meta = dict(data.get("traditional_baseline_meta", {}))
    signal_len = int(cfg.get("signal_len") or 1024)
    seed = int(cfg.get("seed", 42))
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_OUTPUT_ROOT / "FreqDAE" / f"{_timestamp()}_TaskHead直接诊断"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GoNoGo] source_result={source_result}")
    print(f"[GoNoGo] output_dir={out_dir}")
    print(f"[GoNoGo] device={device}")

    sim = _build_simulator(cfg, seed=seed)
    lag_limit = (
        None if cfg.get("tdoa_lag_limit_samples") is None
        else float(cfg.get("tdoa_lag_limit_samples"))
    )
    snrs = [float(v) for v in args.snrs]

    models = {}
    ref_dir = Path(cfg.get("reference_model_dir") or source_result)
    for cr in args.reference_crs:
        model_path = ref_dir / f"model_cr{int(cr)}.pt"
        if model_path.exists():
            models[f"DAE-CR{int(cr)}"] = _load_dae_model(ref_dir, int(cr), signal_len, device)
    hadamard = HadamardProjectionBaseline(
        signal_len=signal_len,
        cr=int(cfg.get("baseline_cr", 16)),
        row_mode=str(meta.get("hadamard_mode", cfg.get("baseline_hadamard_mode", "salari"))),
    ).to(device)
    hadamard.eval()
    models["Hadamard"] = hadamard

    # Waveform decoder reference from v3, useful to compare against task-head direct.
    for cr in args.v3_waveform_crs:
        models[f"V3-Waveform-CR{int(cr)}"] = _load_nested_model(
            source_result, int(cr), signal_len, device
        )

    direct_estimators = _make_direct_estimators(meta, signal_len)
    for cr in args.task_head_crs:
        model = _load_nested_model(source_result, int(cr), signal_len, device)
        direct_estimators[f"TaskHead-CR{int(cr)}"] = LearnedTaskHeadTDOAEstimator(
            model, device, signal_len=signal_len, label=f"TaskHead-CR{int(cr)}",
            seed=seed + int(cr),
        )

    method_results = run_urban_method_comparison(
        models, sim, device, seed=seed,
        snr_range=np.asarray(snrs, dtype=float),
        num_trials=int(args.num_trials),
        sub_sample=True,
        use_los_only=True,
        batch_size=int(args.batch_size),
        fixed_eval_set=True,
        estimator="all_pair_wls",
        title="GoNoGo_TaskHead_Direct",
        direct_estimators=direct_estimators,
        tdoa_lag_limit_samples=lag_limit,
    )
    method_results[0]["config"].update({
        "source_result": str(source_result),
        "diagnostic_type": "go_no_go_task_head_direct",
        "task_head_crs": [int(v) for v in args.task_head_crs],
        "v3_waveform_crs": [int(v) for v in args.v3_waveform_crs],
        "reference_crs": [int(v) for v in args.reference_crs],
    })
    export_method_baselines(
        method_results, tables_dir, table_prefix="go_no_go", export_figures=False
    )
    method_summary = _method_segment_summary(method_results)
    diag_summary = _diagnostic_segment_summary(method_results)
    _write_df(method_summary, tables_dir, "go_no_go_segment_summary")
    _write_df(diag_summary, tables_dir, "go_no_go_diagnostic_segment_summary")

    # Sanity checks use a single SNR to keep the diagnostic package quick.
    sanity_rows = []
    sanity_snr = float(args.sanity_snr)
    X_noisy, _, meta_batch = sim.generate_urban_batch(
        int(args.sanity_trials), snr_db=sanity_snr, seed=seed
    )
    batch_np = X_noisy.numpy()
    alt_sim = _build_simulator(cfg, seed=seed + int(args.alt_geometry_seed_offset))
    X_alt, _, meta_alt = alt_sim.generate_urban_batch(
        int(args.sanity_trials), snr_db=sanity_snr, seed=seed
    )
    batch_alt_np = X_alt.numpy()
    for cr in args.task_head_crs:
        estimator = LearnedTaskHeadTDOAEstimator(
            _load_nested_model(source_result, int(cr), signal_len, device),
            device, signal_len=signal_len, label=f"TaskHead-CR{int(cr)}",
            seed=seed + 1000 + int(cr),
        )
        for test_name, batch_cur, meta_cur, sim_cur, mode in [
            ("normal", batch_np, meta_batch, sim, "normal"),
            ("phase_randomized", batch_np, meta_batch, sim, "phase_randomized"),
            ("shuffle_second_obs", batch_np, meta_batch, sim, "shuffle_second_obs"),
            ("zero", batch_np, meta_batch, sim, "zero"),
            ("geometry_seed_shift", batch_alt_np, meta_alt, alt_sim, "normal"),
        ]:
            metrics = _task_head_pair_metrics(
                estimator, batch_cur, meta_cur, sim_cur.fs, sim_cur.c,
                use_los_only=True, sub_sample=True,
                lag_limit_samples=lag_limit, mode=mode, seed=seed + int(cr),
            )
            metrics.update({
                "test": test_name,
                "cr": int(cr),
                "snr_db": sanity_snr,
                "num_trials": int(args.sanity_trials),
            })
            sanity_rows.append(metrics)
    sanity_df = pd.DataFrame(sanity_rows)
    _write_df(sanity_df, tables_dir, "go_no_go_sanity")

    decision_df = _build_decision_table(method_summary, diag_summary, sanity_df)
    _write_df(decision_df, tables_dir, "go_no_go_decision")
    _plot_go_no_go(method_results, out_dir)

    with open(out_dir / "go_no_go_data.pkl", "wb") as f:
        pickle.dump({
            "method_results": method_results,
            "method_summary": method_summary,
            "diagnostic_summary": diag_summary,
            "sanity": sanity_df,
            "decision": decision_df,
            "source_result": str(source_result),
            "config": vars(args),
        }, f)
    print(f"[Saved] {out_dir / 'go_no_go_data.pkl'}")
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run go/no-go diagnostics for FreqDAE v3.")
    parser.add_argument("--source-result", type=str, default=str(DEFAULT_SOURCE_RESULT),
                        help="Existing v3 result directory containing model_cr*.pt and plot_data.pkl.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Default: 运行结果/FreqDAE/"
                             "YYYYMMDD_HHMMSS_TaskHead直接诊断")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device. Default: cuda if available else cpu.")
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS,
                        help="SNR points for the direct comparison.")
    parser.add_argument("--num-trials", type=int, default=60,
                        help="Urban observations per SNR for the direct comparison.")
    parser.add_argument("--sanity-snr", type=float, default=10.0,
                        help="SNR used for fake-signal sanity tests.")
    parser.add_argument("--sanity-trials", type=int, default=60,
                        help="Urban observations for sanity tests.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--task-head-crs", type=int, nargs="+", default=DEFAULT_CR_LIST)
    parser.add_argument("--v3-waveform-crs", type=int, nargs="+", default=[16])
    parser.add_argument("--reference-crs", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--alt-geometry-seed-offset", type=int, default=777)
    return parser.parse_args()


if __name__ == "__main__":
    run_go_no_go(parse_args())
