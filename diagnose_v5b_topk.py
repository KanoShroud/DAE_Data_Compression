# -*- coding: utf-8 -*-
"""V5-B pair-level top-K posterior diagnostic (eval-only, no retraining).

Usage from PyCharm:  Run this file directly (default result dir is
``运行结果/压缩域TDOA_V5/20260709_120019_V5B专家门控``).

Usage from terminal::

    D:\\Software\\anaconda3\\envs\\PyTorch\\python.exe diagnose_v5b_topk.py "运行结果/压缩域TDOA_V5/20260709_120019_V5B专家门控"

Output: four CSV files in ``{result_dir}/tables/`` plus a summary table
printed to stdout grouped by low / mid / high SNR.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch

from signal_gen import SignalSimulator
from compressed_tdoa import (
    HardBinCompressedTDOALikelihood,
    LearnedHardBinCompressedTDOAEstimator,
    V5BExpertGatedTDOAEstimator,
    run_v5b_topk_diagnostics,
)


def _write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[Saved] {path}")


def _summarize(df):
    """Print hit-rate summary by SNR bucket."""
    buckets = [("low", "SNR <= 0 dB"), ("mid", "2 <= SNR <= 6 dB"),
               ("high", "SNR >= 8 dB"), ("all", "All SNR")]
    print("\n" + "=" * 72)
    print("V5-B Top-K Diagnostic Summary")
    print("=" * 72)
    header = f"{'Bucket':<6s} {'Pairs':>7s}  {'top1':>7s}  {'top3':>7s}  {'top5':>7s}  {'oracle':>7s}"
    print(header)
    print("-" * 72)
    for bucket_key, bucket_label in buckets:
        if bucket_key == "all":
            sub = df
        else:
            sub = df[df["snr_bucket"] == bucket_key]
        n = len(sub)
        if n == 0:
            continue
        t1 = sub["top1_hit1"].mean() * 100
        t3 = sub["top3_hit1"].mean() * 100
        t5 = sub["top5_hit1"].mean() * 100
        ora = sub["oracle_expert_hit1"].mean() * 100
        print(f"{bucket_label:<18s} {n:7d}  {t1:6.1f}%  {t3:6.1f}%  {t5:6.1f}%  {ora:6.1f}%")
    print("-" * 72)
    print("top1/top3/top5 = ±1 sample hit rate of V5-B fused posterior")
    print("oracle = either Power64 OR GeoHybrid64 top-1 is correct (upper bound)")
    # also show ±2 sample
    print("\n±2 sample hit rate:")
    header2 = f"{'Bucket':<18s} {'top1±2':>7s}  {'top3±2':>7s}  {'top5±2':>7s}"
    print(header2)
    print("-" * 50)
    for bucket_key, bucket_label in buckets:
        if bucket_key == "all":
            sub = df
        else:
            sub = df[df["snr_bucket"] == bucket_key]
        n = len(sub)
        if n == 0:
            continue
        t1 = sub["top1_hit2"].mean() * 100
        t3 = sub["top3_hit2"].mean() * 100
        t5 = sub["top5_hit2"].mean() * 100
        print(f"{bucket_label:<18s} {t1:6.1f}%  {t3:6.1f}%  {t5:6.1f}%")
    print()


def _oracle_rejection_upper_bound(df):
    """Compute oracle pair rejection upper bound: what if we could perfectly
    reject pairs where both experts are wrong?"""
    buckets = [("low", "SNR <= 0 dB"), ("mid", "2 <= SNR <= 6 dB"),
               ("high", "SNR >= 8 dB"), ("all", "All SNR")]
    print("\nOracle Pair Rejection Upper Bound")
    print("-" * 55)
    print(f"{'Bucket':<18s} {'total':>6s} {'good':>6s} {'bad':>6s} {'bad%':>7s}")
    print("-" * 55)
    results = []
    for bk, bl in buckets:
        sub = df if bk == "all" else df[df["snr_bucket"] == bk]
        n = len(sub)
        if n == 0:
            continue
        good = sub["oracle_expert_hit1"].sum()
        bad = n - good
        print(f"{bl:<18s} {n:6d} {int(good):6d} {int(bad):6d} {bad/n*100:6.1f}%")
        results.append({"snr_bucket": bk, "total_pairs": n,
                        "good_pairs": int(good), "bad_pairs": int(bad),
                        "bad_pct": round(bad / n * 100, 2)})
    print("-" * 55)
    return pd.DataFrame(results)


def _expert_gating_upper_bound(df):
    """Oracle expert: what if we always pick the better expert?"""
    print("\nOracle Expert Selection Upper Bound")
    print("-" * 60)
    print(f"{'Bucket':<18s} {'V5B top1':>8s} {'Power top1':>10s} {'GeoHyb top1':>11s} {'Oracle':>7s}")
    print("-" * 60)
    results = []
    for bk, bl in [("low", "SNR <= 0 dB"), ("mid", "2 <= SNR <= 6 dB"),
                    ("high", "SNR >= 8 dB"), ("all", "All SNR")]:
        sub = df if bk == "all" else df[df["snr_bucket"] == bk]
        if len(sub) == 0:
            continue
        v5b = sub["top1_hit1"].mean() * 100
        pw = sub["power_top1_hit1"].mean() * 100
        gh = sub["geo_top1_hit1"].mean() * 100
        ora = sub["oracle_expert_hit1"].mean() * 100
        print(f"{bl:<18s} {v5b:7.1f}% {pw:9.1f}% {gh:10.1f}% {ora:7.1f}%")
        results.append({"snr_bucket": bk, "v5b_top1_pct": round(v5b, 1),
                        "power_top1_pct": round(pw, 1),
                        "geohybrid_top1_pct": round(gh, 1),
                        "oracle_top1_pct": round(ora, 1)})
    print("-" * 60)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="V5-B pair-level top-K posterior diagnostic"
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="运行结果/压缩域TDOA_V5/20260709_120019_V5B专家门控",
        help="Path to the result directory containing plot_data.pkl and model_*.pt"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run one SNR and 20 trials; write *_smoke.csv without replacing full results."
    )
    args = parser.parse_args()
    result_dir = os.path.abspath(args.result_dir)

    # --- load config ----------------------------------------------------------
    pkl_path = os.path.join(result_dir, "plot_data.pkl")
    if not os.path.exists(pkl_path):
        sys.exit(f"plot_data.pkl not found in {result_dir}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    cfg = data.get("config", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # --- recreate simulator ---------------------------------------------------
    sim = SignalSimulator(
        channel_mode=cfg.get("channel_mode", "fixed"),
        n_fixed_channels=int(cfg.get("n_fixed_channels", 50)),
        channel_pool_seed=cfg.get("seed", 42),
        nlos_prob=float(cfg.get("nlos_prob", 0.0)),
        delay_label_mode=cfg.get("delay_label_mode", "los"),
        snr_train_range=tuple(cfg.get("train_snr_range", (-10, 20))),
        multipath_scale=float(cfg.get("multipath_scale", 0.2)),
        scenario_mode=cfg.get("scenario_mode", "urban8"),
        normalization_mode=cfg.get("normalization_mode", "per_observation_noisy_rms"),
        urban_base_delay=int(cfg.get("urban_base_delay", 16)),
        urban_min_los=int(cfg.get("urban_min_los", 5)),
        urban_train_los_only=bool(cfg.get("urban_train_los_only", True)),
    )

    # --- resolve selected bins from plot_data.pkl ----------------------------
    v5a1_training = data.get("v5a1_training", {})
    bin_meta = data.get("v5a1_bin_meta", {})
    bins_by_method = bin_meta.get("v5a1_selected_bins_by_method", {})

    def _resolve_bins(label):
        # primary: v5a1_selected_bins_by_method
        arr = bins_by_method.get(label)
        if arr is not None and len(arr) > 0:
            return np.asarray(arr, dtype=int)
        # fallback: training history bin_summary
        hist = v5a1_training.get(label, {})
        arr = hist.get("bin_summary", {}).get("selected_bins")
        if arr is not None and len(arr) > 0:
            return np.asarray(arr, dtype=int)
        return None

    power_bins = _resolve_bins("V5A1-Power64")
    geo_bins = _resolve_bins("V5A1-GeoHybrid64")
    if power_bins is None or geo_bins is None or len(power_bins) == 0 or len(geo_bins) == 0:
        sys.exit(
            "Cannot resolve selected bins for V5A1-Power64 / V5A1-GeoHybrid64 "
            "from plot_data.pkl. Check v5a1_bin_meta.v5a1_selected_bins_by_method "
            "or v5a1_training.*.bin_summary.selected_bins."
        )

    # --- verify model files --------------------------------------------------
    power_path = os.path.join(result_dir, "model_v5a1_power64.pt")
    geo_path = os.path.join(result_dir, "model_v5a1_geohybrid64.pt")
    for path, label in [(power_path, "V5A1-Power64"), (geo_path, "V5A1-GeoHybrid64")]:
        if not os.path.exists(path):
            sys.exit(f"Model file not found: {path}")

    lag_limit = int(cfg.get("tdoa_lag_limit_samples", 48))

    power_model = HardBinCompressedTDOALikelihood(
        selected_bins=np.asarray(power_bins, dtype=int),
        signal_len=sim.signal_len,
        lag_limit_samples=lag_limit,
    ).to(device)
    power_model.load_state_dict(torch.load(power_path, map_location=device))
    power_model.eval()

    geo_model = HardBinCompressedTDOALikelihood(
        selected_bins=np.asarray(geo_bins, dtype=int),
        signal_len=sim.signal_len,
        lag_limit_samples=lag_limit,
    ).to(device)
    geo_model.load_state_dict(torch.load(geo_path, map_location=device))
    geo_model.eval()

    v5a1_weight_mode = cfg.get("v5a1_weight_mode", "combined")
    power_est = LearnedHardBinCompressedTDOAEstimator(
        power_model, device, label="V5A1-Power64", weight_mode=v5a1_weight_mode
    )
    geo_est = LearnedHardBinCompressedTDOAEstimator(
        geo_model, device, label="V5A1-GeoHybrid64", weight_mode=v5a1_weight_mode
    )
    v5b_est = V5BExpertGatedTDOAEstimator(
        {"power": power_est, "geohybrid": geo_est},
        label="V5B-Expert64",
        low_confidence_power_prior=cfg.get("v5b_low_confidence_power_prior", 0.65),
    )

    # --- run diagnostics ------------------------------------------------------
    eval_snr = [float(x) for x in cfg.get("eval_snr_range", np.arange(-10, 21, 2))]
    n_trials = int(cfg.get("monte_carlo_trials", 200))
    seed = int(cfg.get("seed", 42))
    if args.smoke:
        eval_snr = eval_snr[:1]
        n_trials = 20
    file_suffix = "_smoke" if args.smoke else ""
    print(f"[Diag] SNR range: {eval_snr}")
    print(f"[Diag] Monte Carlo trials: {n_trials}")
    print("[Diag] Candidate mode: distinct local maxima with per-peak sub-sample refinement")
    print("[Diag] This may take 5-10 minutes (eval-only, no training)...")
    df = run_v5b_topk_diagnostics(
        sim, device, v5b_est,
        snr_range=eval_snr,
        num_trials=n_trials,
        snapshot_indices=None,
        seed=seed,
    )

    # --- save CSVs -----------------------------------------------------------
    tables_dir = os.path.join(result_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    _write_csv(df, os.path.join(
        tables_dir, f"v5b_pair_topk_diagnostics{file_suffix}.csv"
    ))

    # per-SNR summary
    hit_cols = [c for c in df.columns if "_hit" in c or "oracle" in c]
    agg = df.groupby("SNR_dB").agg(
        n_pairs=("pair_id", "count"),
        **{c: (c, "mean") for c in hit_cols},
    ).reset_index()
    _write_csv(agg, os.path.join(
        tables_dir, f"v5b_topk_summary_by_snr{file_suffix}.csv"
    ))

    # oracle expert summary
    ora_exp = _expert_gating_upper_bound(df)
    _write_csv(ora_exp, os.path.join(
        tables_dir, f"v5b_oracle_expert_summary{file_suffix}.csv"
    ))

    # oracle pair rejection summary
    ora_rej = _oracle_rejection_upper_bound(df)
    _write_csv(ora_rej, os.path.join(
        tables_dir, f"v5b_oracle_pair_rejection_summary{file_suffix}.csv"
    ))

    # --- print summary --------------------------------------------------------
    _summarize(df)

    print("[Done] V5-B top-K diagnostic complete.")


if __name__ == "__main__":
    main()
