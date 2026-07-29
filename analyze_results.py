"""
Pandas-based result analyzer for DAE Data Compression experiments.

PyCharm direct use:
    Edit DEFAULT_RESULT_DIR below, then run this file.

Command-line use:
    python analyze_results.py "运行结果/论文复现与传统基线/20260704_195946_强任务基线评估"

The script does not train or re-evaluate models. It only reads an existing
plot_data.pkl and writes compact CSV/Markdown summaries into tables/.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = (
    ROOT
    / "运行结果"
    / "论文复现与传统基线"
    / "20260704_195946_强任务基线评估"
)

RESULT_FIELDS = (
    "fig6_results",
    "fig7_results",
    "fig8_results",
    "fig8_supp_results",
    "baseline_all_results",
)

BANDS = (
    ("all", None, None),
    ("low_snr_le_0", None, 0.0),
    ("mid_snr_2_to_6", 2.0, 6.0),
    ("high_snr_ge_8", 8.0, None),
)


def _load_plot_data(result_dir: Path):
    pkl_path = result_dir / "plot_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def _band_mask(snr: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    mask = np.ones_like(snr, dtype=bool)
    if low is not None:
        mask &= snr >= float(low)
    if high is not None:
        mask &= snr <= float(high)
    return mask


def _safe_float_array(values) -> np.ndarray | None:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr


def _write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with md_path.open("w", encoding="utf-8") as f:
        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            f.write(df.to_string(index=False))
        f.write("\n")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {md_path}")


def summarize_rmse(method_data, result_name: str) -> pd.DataFrame:
    results, snr_range = method_data
    snr = np.asarray(snr_range, dtype=float)
    methods = results.get("methods", {})
    order = results.get("method_order", list(methods.keys()))
    rows = []

    for band_name, low, high in BANDS:
        mask = _band_mask(snr, low, high)
        if not np.any(mask):
            continue
        band_values = {}
        for method in order:
            data = methods.get(method, {})
            rmse = _safe_float_array(data.get("rmse", []))
            if rmse is None or rmse.size != snr.size:
                continue
            vals = rmse[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            band_values[method] = vals

        if band_values:
            stacked = pd.DataFrame({
                method: pd.Series(vals)
                for method, vals in band_values.items()
            })
            best_counts = stacked.idxmin(axis=1).value_counts().to_dict()
        else:
            best_counts = {}

        for method, vals in band_values.items():
            rows.append({
                "result": result_name,
                "method": method,
                "band": band_name,
                "snr_min": float(np.min(snr[mask])),
                "snr_max": float(np.max(snr[mask])),
                "n_points": int(vals.size),
                "mean_rmse_m": float(np.mean(vals)),
                "median_curve_rmse_m": float(np.median(vals)),
                "min_curve_rmse_m": float(np.min(vals)),
                "max_curve_rmse_m": float(np.max(vals)),
                "best_count_in_band": int(best_counts.get(method, 0)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rank_in_band"] = (
        df.groupby(["result", "band"])["mean_rmse_m"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return df.sort_values(["result", "band", "rank_in_band", "method"])


def summarize_diagnostics(method_data, result_name: str) -> pd.DataFrame:
    results, snr_range = method_data
    snr = np.asarray(snr_range, dtype=float)
    diagnostics = results.get("method_diagnostics", {})
    rows = []

    for method, diag in diagnostics.items():
        for metric, values in diag.items():
            arr = _safe_float_array(values)
            if arr is None or arr.size != snr.size:
                continue
            for band_name, low, high in BANDS:
                mask = _band_mask(snr, low, high)
                vals = arr[mask]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                rows.append({
                    "result": result_name,
                    "method": method,
                    "diagnostic": metric,
                    "band": band_name,
                    "n_points": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["result", "method", "diagnostic", "band"])


def analyze(result_dir: Path) -> None:
    result_dir = result_dir.resolve()
    data = _load_plot_data(result_dir)
    out_dir = result_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rmse_frames = []
    diag_frames = []
    for field in RESULT_FIELDS:
        method_data = data.get(field)
        if method_data is None:
            continue
        rmse_df = summarize_rmse(method_data, field)
        diag_df = summarize_diagnostics(method_data, field)
        if not rmse_df.empty:
            _write_table(rmse_df, out_dir, f"analysis_{field}_rmse_by_band")
            rmse_frames.append(rmse_df)
        if not diag_df.empty:
            _write_table(diag_df, out_dir, f"analysis_{field}_diagnostics_by_band")
            diag_frames.append(diag_df)

    if rmse_frames:
        all_rmse = pd.concat(rmse_frames, ignore_index=True)
        _write_table(all_rmse, out_dir, "analysis_all_rmse_by_band")
        print("\nTop methods by all-SNR mean RMSE:")
        top = all_rmse[all_rmse["band"] == "all"].sort_values(
            ["result", "rank_in_band"]
        )
        print(top[["result", "rank_in_band", "method", "mean_rmse_m"]].to_string(index=False))
    else:
        print("[Warn] No method-comparison RMSE data found.")

    if diag_frames:
        all_diag = pd.concat(diag_frames, ignore_index=True)
        _write_table(all_diag, out_dir, "analysis_all_diagnostics_by_band")
    else:
        print("[Warn] No numeric diagnostic time series found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved plot_data.pkl with pandas")
    parser.add_argument("result_dir", nargs="?", default=None,
                        help="Result directory containing plot_data.pkl")
    args = parser.parse_args()
    analyze(Path(args.result_dir) if args.result_dir else DEFAULT_RESULT_DIR)


if __name__ == "__main__":
    main()
