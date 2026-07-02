# export_results.py
"""
Export tables and paper-ready figures from a saved plot_data.pkl result folder.

Default target is the frozen v3.1 paper reproduction baseline:
运行结果/20260702_000925
"""

import csv
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "运行结果", "20260702_000925")


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in fieldnames})
    print(f"[Saved] {path}")


def write_markdown_table(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(k, "")) for k in fieldnames) + " |\n")
    print(f"[Saved] {path}")


def cr_keys_from_results(results):
    keys = [k for k in results if k.startswith("dae_") and k[4:].isdigit()]
    return sorted(keys, key=lambda k: int(k[4:]))


def metric_rows(results, snr_range, suffix=""):
    rows = []
    cr_keys = cr_keys_from_results(results)
    for i, snr in enumerate(snr_range):
        row = {"SNR_dB": float(snr)}
        for base, label in [("geom", "Geometry_oracle"),
                            ("clean", "Clean_oracle"),
                            ("raw", "Raw")]:
            key = base + suffix
            if key in results:
                row[label] = float(results[key][i])
        for key in cr_keys:
            metric_key = key + suffix
            if metric_key in results:
                row[f"CR{key[4:]}"] = float(results[metric_key][i])
        rows.append(row)
    fields = ["SNR_dB", "Geometry_oracle", "Clean_oracle", "Raw"]
    fields.extend(f"CR{k[4:]}" for k in cr_keys)
    return rows, fields


def export_training_summary(cv_results_dict, out_dir):
    rows = []
    for cr in sorted(cv_results_dict, key=lambda x: int(x)):
        cv = cv_results_dict[cr]
        fold_best = cv.get("fold_best_epoch", [])
        fold_val = cv.get("fold_best_val", [])
        row = {
            "CR": cr,
            "training_protocol": cv.get("training_protocol"),
            "final_epoch_policy": cv.get("final_epoch_policy"),
            "final_epochs": cv.get("final_epochs"),
            "fold_best_epoch": fold_best,
            "fold_best_epoch_median": float(np.median(fold_best)) if fold_best else "",
            "fold_best_val_mean": float(np.mean(fold_val)) if fold_val else "",
            "fold_best_val_min": float(np.min(fold_val)) if fold_val else "",
            "final_train_loss_last": cv.get("final_train_loss", [""])[-1],
            "final_val_loss_last": cv.get("final_val_loss", [""])[-1],
            "resample_train_each_epoch": cv.get("resample_train_each_epoch"),
            "resample_interval": cv.get("resample_interval"),
        }
        rows.append(row)
    fields = ["CR", "training_protocol", "final_epoch_policy", "final_epochs",
              "fold_best_epoch", "fold_best_epoch_median", "fold_best_val_mean",
              "fold_best_val_min", "final_train_loss_last", "final_val_loss_last",
              "resample_train_each_epoch", "resample_interval"]
    write_csv(os.path.join(out_dir, "training_summary.csv"), rows, fields)
    write_markdown_table(os.path.join(out_dir, "training_summary.md"), rows, fields)


def export_rmse_tables(mc_results, out_dir):
    results, snr_range = mc_results
    for suffix, name in [("", "mean"), ("_med", "median"), ("_trimmed", "trimmed")]:
        rows, fields = metric_rows(results, snr_range, suffix=suffix)
        write_csv(os.path.join(out_dir, f"rmse_{name}.csv"), rows, fields)
        write_markdown_table(os.path.join(out_dir, f"rmse_{name}.md"), rows, fields)


def export_fig2_diagnostics(snr_data, out_dir):
    rows = []
    fields = ["SNR_dB", "target", "mag_nmse_mean", "complex_nmse_mean",
              "psd_db_mse_mean", "mean_abs_peak_error", "within_2_rate",
              "mean_sidelobe_ratio", "mean_false_peaks_gt_05"]
    for snr in sorted(snr_data, key=float):
        diagnostics = snr_data[snr].get("diagnostics", {})
        for target_key, target_name in [("raw_peak", "Raw"), ("clean_peak", "Clean")]:
            row = {"SNR_dB": float(snr), "target": target_name}
            row.update(diagnostics.get(target_key, {}))
            rows.append(row)
        dae = diagnostics.get("dae", {})
        for cr in sorted(dae, key=lambda x: int(x)):
            row = {"SNR_dB": float(snr), "target": f"CR{cr}"}
            row.update(dae[cr])
            rows.append(row)
    write_csv(os.path.join(out_dir, "fig2_diagnostics.csv"), rows, fields)
    write_markdown_table(os.path.join(out_dir, "fig2_diagnostics.md"), rows, fields)


def plot_rmse(ax, results, snr_range, suffix, title):
    cr_keys = cr_keys_from_results(results)
    ax.plot(snr_range, results["raw" + suffix], "b-s", label="Original data", linewidth=1.5)
    if "clean" + suffix in results:
        ax.plot(snr_range, results["clean" + suffix], "k--", label="Clean oracle", linewidth=1.2)
    if "geom" + suffix in results:
        ax.plot(snr_range, results["geom" + suffix], color="0.45", linestyle=":",
                label="Geometry oracle", linewidth=1.1)
    styles = ["m-*", "g-o", "r-+", "c-^", "y-d"]
    for idx, key in enumerate(cr_keys):
        metric_key = key + suffix
        if metric_key in results:
            ax.plot(snr_range, results[metric_key], styles[idx % len(styles)],
                    label=f"Data with CR={key[4:]}", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Localization RMSE [m]")
    ax.grid(True, alpha=0.45)
    ax.legend(fontsize=8)


def export_paper_figures(mc_results, out_dir):
    results, snr_range = mc_results
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    plot_rmse(ax, results, snr_range, "", "Mean Localization RMSE")
    path = os.path.join(out_dir, "Fig3_Main_Mean_RMSE.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    plot_rmse(axes[0], results, snr_range, "_med", "Median Localization RMSE")
    plot_rmse(axes[1], results, snr_range, "_trimmed", "Trimmed Localization RMSE")
    path = os.path.join(out_dir, "Fig3_Supp_Median_Trimmed.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


def export_summary_md(data, result_dir, out_dir):
    config = data.get("config", {})
    results, _ = data["mc_results"]
    cr_keys = cr_keys_from_results(results)
    mean_avgs = {
        "Raw": float(np.mean(results["raw"])),
        "Clean_oracle": float(np.mean(results["clean"])),
        "Geometry_oracle": float(np.mean(results["geom"])),
    }
    for key in cr_keys:
        mean_avgs[f"CR{key[4:]}"] = float(np.mean(results[key]))

    path = os.path.join(out_dir, "paper_repro_v3_1_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# paper_repro_v3.1 Summary\n\n")
        f.write(f"- Result directory: `{result_dir}`\n")
        f.write(f"- Diagnostics version: `{config.get('diagnostics_version')}`\n")
        f.write(f"- Training protocol: `{config.get('training_protocol')}`\n")
        f.write(f"- Scenario: `{config.get('scenario_mode')}`\n")
        f.write(f"- Evaluation: `{config.get('evaluation_mode')}`, "
                f"`{config.get('localization_estimator')}`\n")
        f.write(f"- Normalization: `{config.get('normalization_mode')}`\n")
        f.write(f"- Resampling: `{config.get('resample_train_each_epoch')}`, "
                f"interval `{config.get('resample_interval')}`\n\n")
        f.write("## Mean RMSE Averages\n\n")
        for key, value in mean_avgs.items():
            f.write(f"- {key}: {value:.4f} m\n")
        f.write("\n## Interpretation\n\n")
        f.write("- This is a synthetic urban8/oracle-LOS approximation of Chen 2025, "
                "not a Wireless InSite reproduction.\n")
        f.write("- Geometry oracle checks the WLS geometry/sign lower bound.\n")
        f.write("- Clean oracle checks the clean waveform GCC + WLS upper bound under finite bandwidth.\n")
        f.write("- Mean RMSE is the main metric; median and trimmed RMSE diagnose outlier sensitivity.\n")
    print(f"[Saved] {path}")


def export(result_dir):
    pkl_path = os.path.join(result_dir, "plot_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    out_dir = os.path.join(result_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)
    export_training_summary(data["cv_results_dict"], out_dir)
    export_rmse_tables(data["mc_results"], out_dir)
    export_fig2_diagnostics(data["snr_data"], out_dir)
    export_paper_figures(data["mc_results"], out_dir)
    export_summary_md(data, result_dir, out_dir)
    print("[Done] Export complete.")


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_RESULT_DIR)
