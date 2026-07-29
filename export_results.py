# export_results.py
"""
Export tables and paper-ready figures from a saved plot_data.pkl result folder.

Default target is the frozen v3.1 paper reproduction baseline:
运行结果/论文复现与传统基线/20260702_000925_论文复现v3.1
"""

import csv
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import (filter_method_comparison_data, plot_method_comparison,
                      plot_method_zoom_pair,
                      build_tdoa_metrics_from_method_comparison)


DEFAULT_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "运行结果", "论文复现与传统基线",
                                  "20260702_000925_论文复现v3.1")
EXPORT_MARKDOWN_TABLES = True


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
    if not EXPORT_MARKDOWN_TABLES:
        print(f"[Skip] Markdown table disabled: {path}")
        return
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


def _collect_group_values(methods, prefix, metric_key):
    labels = [label for label in methods if label.startswith(prefix)]
    values = []
    valid_labels = []
    for label in labels:
        vals = np.asarray(methods[label].get(metric_key, []), dtype=float)
        if vals.size:
            values.append(vals)
            valid_labels.append(label)
    if not values:
        return None, valid_labels
    return np.vstack(values), valid_labels


def export_method_group_summary(method_results, out_dir, table_prefix="fig6"):
    if not method_results:
        return
    results, snr_range = method_results
    methods = results.get("methods", {})
    config = results.get("config", {})
    prefixes = []
    for label in config.get("supplement_method_order", results.get("method_order", [])):
        if isinstance(label, str) and label.endswith("*"):
            prefixes.append(label[:-1])
    rows = []
    for prefix in prefixes:
        stack, labels = _collect_group_values(methods, prefix, "rmse")
        if stack is None:
            continue
        group_name = prefix[:-1] if prefix.endswith("-") else prefix
        for i, snr in enumerate(snr_range):
            vals = stack[:, i]
            rows.append({
                "SNR_dB": float(snr),
                "group": group_name,
                "n_members": len(labels),
                "rmse_mean": float(np.nanmean(vals)),
                "rmse_std": float(np.nanstd(vals)),
                "rmse_min": float(np.nanmin(vals)),
                "rmse_max": float(np.nanmax(vals)),
                "members": ";".join(labels),
            })
    if rows:
        fields = ["SNR_dB", "group", "n_members", "rmse_mean", "rmse_std",
                  "rmse_min", "rmse_max", "members"]
        write_csv(os.path.join(out_dir, f"{table_prefix}_random_group_summary.csv"), rows, fields)
        write_markdown_table(os.path.join(out_dir, f"{table_prefix}_random_group_summary.md"), rows, fields)


def export_method_baselines(method_results, out_dir, table_prefix="fig6",
                            export_figures=True, figure_kinds=None):
    if not method_results:
        print(f"[Skip] No {table_prefix}_results found in plot_data.pkl")
        return
    results, snr_range = method_results
    methods = results.get("methods", {})
    order = results.get("method_order", list(methods.keys()))
    rows = []
    metric_specs = [
        ("rmse", "mean_rmse_m"),
        ("median", "median_rmse_m"),
        ("trimmed_rmse", "trimmed_rmse_m"),
        ("p90", "p90_rmse_m"),
        ("p95", "p95_rmse_m"),
        ("max", "max_rmse_m"),
        ("valid_count", "valid_count"),
        ("failure_rate", "failure_rate"),
        ("outlier_gt20_rate", "outlier_gt20_rate"),
        ("outlier_gt50_rate", "outlier_gt50_rate"),
    ]
    for i, snr in enumerate(snr_range):
        row = {"SNR_dB": float(snr)}
        for label in order:
            if label in methods:
                for key, suffix in metric_specs:
                    vals = methods[label].get(key, [])
                    if i < len(vals):
                        row[f"{label}_{suffix}"] = vals[i]
        rows.append(row)
    fields = ["SNR_dB"]
    for label in order:
        for key, suffix in metric_specs:
            if label in methods and key in methods[label]:
                fields.append(f"{label}_{suffix}")
    baseline_table = (
        "fig6_traditional_baselines"
        if table_prefix == "fig6" else f"{table_prefix}_baselines"
    )
    write_csv(os.path.join(out_dir, f"{baseline_table}.csv"), rows, fields)
    write_markdown_table(os.path.join(out_dir, f"{baseline_table}.md"), rows, fields)

    diagnostics = results.get("method_diagnostics", {})
    diag_rows = []
    diag_keys = []
    for label in order:
        method_diag = diagnostics.get(label, {})
        for key in method_diag.keys():
            if key not in diag_keys:
                diag_keys.append(key)
    for i, snr in enumerate(snr_range):
        for label in order:
            method_diag = diagnostics.get(label, {})
            if not method_diag:
                continue
            row = {"SNR_dB": float(snr), "method": label}
            for key in diag_keys:
                vals = method_diag.get(key, [])
                row[key] = vals[i] if i < len(vals) else ""
            diag_rows.append(row)
    if diag_rows:
        diag_fields = ["SNR_dB", "method"] + diag_keys
        write_csv(os.path.join(out_dir, f"{table_prefix}_method_diagnostics.csv"),
                  diag_rows, diag_fields)
        write_markdown_table(os.path.join(out_dir, f"{table_prefix}_method_diagnostics.md"),
                             diag_rows, diag_fields)

    export_method_outlier_details(method_results, out_dir, table_prefix=table_prefix)
    export_method_group_summary(method_results, out_dir, table_prefix=table_prefix)

    if export_figures:
        config = results.get("config", {})
        baseline_cr = config.get("baseline_cr", 16)
        if figure_kinds is None:
            figure_kinds = [
                {"plot_kind": "main",
                 "filename": config.get("main_figure_filename", f"Fig6_Traditional_Baselines_CR{baseline_cr}")},
                {"plot_kind": "supplement",
                 "filename": config.get("supplement_figure_filename",
                                        f"Fig6_Supp_Baseline_Ablation_CR{baseline_cr}")},
            ]
            if config.get("export_method_zoom_figures", True):
                zoom_order = config.get("supplement_zoom_method_order")
                figure_kinds.append(
                    {"plot_kind": "supplement",
                     "filename": config.get(
                         "supplement_zoom_figure_filename",
                         f"Fig6_Supp_Chen_Baseline_Ablation_CR{baseline_cr}_SNR_Zooms"),
                     "zoom_pair": True,
                     "low_snr_max": config.get("method_zoom_low_snr_max", 0.0),
                     "high_snr_min": config.get("method_zoom_high_snr_min", 8.0),
                     "method_order": zoom_order}
                )
        for spec in figure_kinds:
            if isinstance(spec, dict):
                plot_kind = spec.get("plot_kind", "main")
                filename_stem = spec.get("filename", plot_kind)
                snr_min = spec.get("snr_min")
                snr_max = spec.get("snr_max")
                method_order = spec.get("method_order")
                title_suffix = spec.get("title_suffix")
                zoom_pair = bool(spec.get("zoom_pair", False))
                low_snr_max = spec.get("low_snr_max", 0.0)
                high_snr_min = spec.get("high_snr_min", 8.0)
            else:
                plot_kind, filename_stem = spec
                snr_min = snr_max = method_order = title_suffix = None
                zoom_pair = False
                low_snr_max = 0.0
                high_snr_min = 8.0
            if zoom_pair:
                fig = plot_method_zoom_pair(
                    method_results, plot_kind=plot_kind,
                    low_snr_max=low_snr_max, high_snr_min=high_snr_min,
                    method_order=method_order,
                )
            else:
                fig = plot_method_comparison(
                    method_results, plot_kind=plot_kind,
                    snr_min=snr_min, snr_max=snr_max,
                    method_order=method_order, title_suffix=title_suffix,
                )
            path = os.path.join(out_dir, f"{filename_stem}.svg")
            fig.savefig(path, format="svg", bbox_inches="tight")
            plt.close(fig)
            print(f"[Saved] {path}")
    else:
        print(f"[Skip] {table_prefix} SVG export to tables/ disabled; root result folder already contains SVG.")


def export_method_outlier_details(method_results, out_dir, table_prefix="fig6"):
    results, _ = method_results
    outliers = results.get("outlier_details", {})
    if not outliers:
        return
    rows = []
    for label in results.get("method_order", []):
        for snr_entry in outliers.get(label, []):
            snr = snr_entry.get("snr_db", "")
            for rank, detail in enumerate(snr_entry.get("top", []), start=1):
                row = {
                    "SNR_dB": snr,
                    "method": label,
                    "rank": rank,
                }
                row.update(detail)
                rows.append(row)
    if not rows:
        return
    fields = [
        "SNR_dB", "method", "rank", "sample_index", "error_m",
        "source_x", "source_y", "estimated_x", "estimated_y",
        "los_count", "los_uav_indices", "los_uav_xy",
        "used_uav_indices", "n_pairs", "n_uavs", "wls_cost",
        "wls_normalized_cost", "residual_rmse_m",
        "mean_abs_residual_m", "max_abs_residual_m",
        "geometry_condition", "geometry_gdop", "boundary_hit",
        "candidate_count", "second_best_cost", "second_best_x",
        "second_best_y", "cost_gap", "pruned_pairs_removed",
        "pre_prune_cost", "pre_prune_normalized_cost",
        "max_pair_abs_tdoa_error_samples",
        "mean_pair_abs_tdoa_error_samples",
        "median_pair_abs_tdoa_error_samples",
        "mean_pair_sidelobe_ratio",
        "pair_uav_indices",
        "pair_true_tdoa_samples",
        "pair_est_tdoa_samples",
        "pair_tdoa_error_samples",
        "pair_weight",
        "pair_sidelobe_ratio",
    ]
    write_csv(os.path.join(out_dir, f"{table_prefix}_worst_localization_errors.csv"),
              rows, fields)
    write_markdown_table(os.path.join(out_dir, f"{table_prefix}_worst_localization_errors.md"),
                         rows, fields)


def make_tdoa_results(data, key="tdoa_results", fallback_keys=None):
    tdoa_results = data.get(key)
    if tdoa_results is not None:
        return tdoa_results
    fallback_keys = fallback_keys or [
        "fig9_results", "fig8_results", "fig7_results", "fig6_results",
    ]
    for fallback_key in fallback_keys:
        method_results = data.get(fallback_key)
        if method_results is None:
            continue
        results, _ = method_results
        config = results.get("config", {})
        order = (
            config.get("strong_method_order")
            or config.get("taskaware_method_order")
            or config.get("main_method_order")
            or results.get("method_order", [])
        )
        keep = []
        seen = set()
        for label in ["Raw", "Clean", "Geometry"] + list(order):
            if label not in seen:
                keep.append(label)
                seen.add(label)
        built = build_tdoa_metrics_from_method_comparison(
            method_results, method_order=keep, prefer_direct=True
        )
        if built is not None:
            baseline_cr = config.get("baseline_cr", 16)
            built[0].setdefault("config", {}).update({
                "table_prefix": "tdoa",
                "strong_figure_filename": f"Fig_TDOA_MAE_CR{baseline_cr}",
                "tdoa_within1_figure_filename": f"Fig_TDOA_Within1_CR{baseline_cr}",
                "tdoa_source_view": fallback_key,
            })
            return built
    return None


def export_tdoa_metrics(tdoa_results, out_dir, table_prefix="tdoa",
                        export_figures=True):
    if not tdoa_results:
        print(f"[Skip] No {table_prefix}_results found in plot_data.pkl")
        return
    results, snr_range = tdoa_results
    methods = results.get("methods", {})
    order = results.get("method_order", list(methods.keys()))
    rows = []
    metric_keys = [
        "tdoa_mae_samples",
        "tdoa_weighted_abs_samples",
        "tdoa_median_abs_samples",
        "tdoa_within_1_sample_rate",
        "tdoa_within_2_sample_rate",
        "tdoa_sidelobe_ratio",
    ]
    for i, snr in enumerate(snr_range):
        for label in order:
            vals = methods.get(label, {})
            if not vals:
                continue
            row = {
                "SNR_dB": float(snr),
                "method": label,
                "tdoa_source": (
                    vals.get("tdoa_source", [""] * len(snr_range))[i]
                    if i < len(vals.get("tdoa_source", [])) else ""
                ),
            }
            has_value = False
            for key in metric_keys:
                series = vals.get(key, [])
                value = series[i] if i < len(series) else ""
                row[key] = value
                if value != "":
                    has_value = True
            if has_value:
                rows.append(row)
    fields = ["SNR_dB", "method", "tdoa_source"] + metric_keys
    write_csv(os.path.join(out_dir, f"{table_prefix}_method_tdoa_metrics.csv"),
              rows, fields)
    write_markdown_table(os.path.join(out_dir, f"{table_prefix}_method_tdoa_metrics.md"),
                         rows, fields)

    if export_figures:
        config = results.get("config", {})
        for metric_key, filename in [
            ("tdoa_mae_samples", config.get("strong_figure_filename", "Fig_TDOA_MAE_CR16")),
            ("tdoa_within_1_sample_rate",
             config.get("tdoa_within1_figure_filename", "Fig_TDOA_Within1_CR16")),
        ]:
            fig = plot_method_comparison(
                tdoa_results, metric_key=metric_key, plot_kind="strong"
            )
            path = os.path.join(out_dir, f"{filename}.svg")
            fig.savefig(path, format="svg", bbox_inches="tight")
            plt.close(fig)
            print(f"[Saved] {path}")
    else:
        print(f"[Skip] {table_prefix} TDOA SVG export disabled.")


def export_fig6_baselines(fig6_results, out_dir, export_figures=True):
    export_method_baselines(
        fig6_results, out_dir, table_prefix="fig6",
        export_figures=export_figures,
    )


def export_fig7_baselines(fig7_results, out_dir, export_figures=True):
    if not fig7_results:
        print("[Skip] No fig7_results found in plot_data.pkl")
        return
    results, _ = fig7_results
    config = results.get("config", {})
    baseline_cr = config.get("baseline_cr", 16)
    figure_kinds = [
        {"plot_kind": "taskaware",
         "filename": config.get("taskaware_figure_filename",
                                f"Fig7_TaskAware_Baselines_CR{baseline_cr}")},
    ]
    if config.get("export_method_zoom_figures", True):
        zoom_order = config.get("taskaware_zoom_method_order")
        figure_kinds.append(
            {"plot_kind": "taskaware",
             "filename": config.get(
                 "taskaware_zoom_figure_filename",
                 f"Fig7_TaskAware_Baselines_CR{baseline_cr}_SNR_Zooms"),
             "zoom_pair": True,
             "low_snr_max": config.get("method_zoom_low_snr_max", 0.0),
             "high_snr_min": config.get("method_zoom_high_snr_min", 8.0),
             "method_order": zoom_order}
        )
    export_method_baselines(
        fig7_results, out_dir, table_prefix="fig7",
        export_figures=export_figures,
        figure_kinds=figure_kinds,
    )


def export_fig8_baselines(fig8_results, out_dir, export_figures=True):
    if not fig8_results:
        print("[Skip] No fig8_results found in plot_data.pkl")
        return
    results, _ = fig8_results
    config = results.get("config", {})
    baseline_cr = config.get("baseline_cr", 16)
    figure_kinds = [
        {"plot_kind": "strong",
         "filename": config.get("strong_figure_filename",
                                f"Fig8_Strong_TaskAware_Baselines_CR{baseline_cr}")},
    ]
    if config.get("export_method_zoom_figures", True):
        zoom_order = config.get("strong_zoom_method_order")
        figure_kinds.append(
            {"plot_kind": "strong",
             "filename": config.get(
                 "strong_zoom_figure_filename",
                 f"Fig8_Strong_TaskAware_Baselines_CR{baseline_cr}_SNR_Zooms"),
             "zoom_pair": True,
             "low_snr_max": config.get("method_zoom_low_snr_max", 0.0),
             "high_snr_min": config.get("method_zoom_high_snr_min", 8.0),
             "method_order": zoom_order}
        )
    export_method_baselines(
        fig8_results, out_dir, table_prefix="fig8",
        export_figures=export_figures,
        figure_kinds=figure_kinds,
    )


def export_fig8_supp_baselines(fig8_supp_results, out_dir, export_figures=True):
    if not fig8_supp_results:
        print("[Skip] No fig8_supp_results found in plot_data.pkl")
        return
    results, _ = fig8_supp_results
    config = results.get("config", {})
    baseline_cr = config.get("baseline_cr", 16)
    figure_kinds = [
        {"plot_kind": "strong",
         "filename": config.get(
             "strong_figure_filename",
             f"Fig8_Supp_Strong_Baseline_Diagnostics_CR{baseline_cr}")},
    ]
    if config.get("export_method_zoom_figures", True):
        zoom_order = config.get("strong_zoom_method_order")
        figure_kinds.append(
            {"plot_kind": "strong",
             "filename": config.get(
                 "strong_zoom_figure_filename",
                 f"Fig8_Supp_Strong_Baseline_Diagnostics_CR{baseline_cr}_SNR_Zooms"),
             "zoom_pair": True,
             "low_snr_max": config.get("method_zoom_low_snr_max", 0.0),
             "high_snr_min": config.get("method_zoom_high_snr_min", 8.0),
             "method_order": zoom_order}
        )
    export_method_baselines(
        fig8_supp_results, out_dir, table_prefix="fig8_supp",
        export_figures=export_figures,
        figure_kinds=figure_kinds,
    )


def export_fig9_innovation(fig9_results, out_dir, export_figures=True):
    if not fig9_results:
        print("[Skip] No fig9_results found in plot_data.pkl")
        return
    results, _ = fig9_results
    config = results.get("config", {})
    baseline_cr = config.get("baseline_cr", 16)
    figure_kinds = [
        {"plot_kind": "strong",
         "filename": config.get(
             "strong_figure_filename",
             f"Fig9_FrequencyTaskDAE_vs_Repro_CR{baseline_cr}")},
    ]
    if config.get("export_method_zoom_figures", True):
        zoom_order = config.get("strong_zoom_method_order")
        figure_kinds.append(
            {"plot_kind": "strong",
             "filename": config.get(
                 "strong_zoom_figure_filename",
                 f"Fig9_FrequencyTaskDAE_vs_Repro_CR{baseline_cr}_SNR_Zooms"),
             "zoom_pair": True,
             "low_snr_max": config.get("method_zoom_low_snr_max", 0.0),
             "high_snr_min": config.get("method_zoom_high_snr_min", 8.0),
             "method_order": zoom_order}
        )
    export_method_baselines(
        fig9_results, out_dir, table_prefix="fig9",
        export_figures=export_figures,
        figure_kinds=figure_kinds,
    )


def make_fig9_focus_results(data):
    fig9_focus_results = data.get("fig9_focus_results")
    if fig9_focus_results is not None:
        return fig9_focus_results
    fig9_results = data.get("fig9_results")
    if fig9_results is None:
        return None
    config = fig9_results[0].get("config", {})
    methods = fig9_results[0].get("methods", {})
    innovation_order = [
        label for label in [
            "FreqDAE-CR4", "FreqDAE-CR8", "FreqDAE-CR16",
            "FreqDAE-v2-CR4", "FreqDAE-v2-CR8", "FreqDAE-v2-CR16",
            "FreqDAE-v3-CR4", "FreqDAE-v3-CR8", "FreqDAE-v3-CR16",
            "FreqDAE-v4-CR4", "FreqDAE-v4-CR8", "FreqDAE-v4-CR16",
        ]
        if label in methods
    ]
    compressed_order = [
        label for label in [
            "V5A-CR16",
            "V5A1-Uniform64", "V5A1-Power64", "V5A1-Cao64", "V5A1-GeoHybrid64",
            "V5B-Expert64",
        ]
        if label in methods
    ]
    focus_order = [
        "Raw", "DAE-CR4", "DAE-CR8", "DAE-CR16",
    ] + innovation_order + compressed_order
    focus_keep = ["Raw", "Clean", "Geometry"] + focus_order
    return filter_method_comparison_data(
        fig9_results, focus_keep,
        {
            **config,
            "strong_method_order": focus_order,
            "strong_title": "Figure 9 Focus: Frozen Chen-DAE vs Frequency-Task DAE",
            "strong_figure_filename": "Fig9_Focused_DAE_CR_Comparison",
            "strong_zoom_method_order": focus_order,
            "strong_zoom_figure_filename": "Fig9_Focused_DAE_CR_Comparison_SNR_Zooms",
            "table_prefix": "fig9_focus",
        },
    )


def export_fig9_focus(fig9_focus_results, out_dir, export_figures=True):
    if not fig9_focus_results:
        print("[Skip] No fig9_focus_results found in plot_data.pkl")
        return
    config = fig9_focus_results[0].get("config", {})
    figure_kinds = [
        {"plot_kind": "strong",
         "filename": config.get("strong_figure_filename", "Fig9_Focused_DAE_CR_Comparison")},
    ]
    if config.get("export_method_zoom_figures", True):
        zoom_order = config.get("strong_zoom_method_order")
        figure_kinds.append(
            {"plot_kind": "strong",
             "filename": config.get(
                 "strong_zoom_figure_filename",
                 "Fig9_Focused_DAE_CR_Comparison_SNR_Zooms"),
             "zoom_pair": True,
             "low_snr_max": config.get("method_zoom_low_snr_max", 0.0),
             "high_snr_min": config.get("method_zoom_high_snr_min", 8.0),
             "method_order": zoom_order}
        )
    export_method_baselines(
        fig9_focus_results, out_dir, table_prefix="fig9_focus",
        export_figures=export_figures,
        figure_kinds=figure_kinds,
    )


def export_fig10_localizer(fig10_results, out_dir, export_figures=True):
    if not fig10_results:
        print("[Skip] No fig10_localizer_results found in plot_data.pkl")
        return
    config = fig10_results[0].get("config", {})
    figure_kinds = [
        {"plot_kind": "strong",
         "filename": config.get(
             "strong_figure_filename",
             "Fig10_Localizer_Robustness_Ablation")},
    ]
    if config.get("export_method_zoom_figures", False):
        zoom_order = config.get("strong_zoom_method_order")
        figure_kinds.append(
            {"plot_kind": "strong",
             "filename": config.get(
                 "strong_zoom_figure_filename",
                 "Fig10_Localizer_Robustness_Ablation_SNR_Zooms"),
             "zoom_pair": True,
             "low_snr_max": config.get("method_zoom_low_snr_max", 0.0),
             "high_snr_min": config.get("method_zoom_high_snr_min", 8.0),
             "method_order": zoom_order}
        )
    export_method_baselines(
        fig10_results, out_dir, table_prefix="fig10_localizer",
        export_figures=export_figures,
        figure_kinds=figure_kinds,
    )


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
        if data.get("fig6_results") is not None:
            meta = data.get("traditional_baseline_meta", {})
            f.write("- Fig6 traditional baselines: available; "
                    f"CR `{meta.get('baseline_cr', config.get('baseline_cr'))}`, "
                    f"DFT `{meta.get('dft_mode', config.get('baseline_dft_mode'))}`, "
                    f"DFT selection `{meta.get('dft_train_selection_kind', 'N/A')}`, "
                    f"Hadamard `{meta.get('hadamard_mode', config.get('baseline_hadamard_mode'))}`, "
                    f"Hadamard rule `{meta.get('hadamard_row_rule', 'N/A')}`, "
                    f"PCA source `{meta.get('pca_training_source', config.get('baseline_pca_train_source'))}`, "
                    f"PCA samples `{meta.get('pca_training_samples', config.get('baseline_pca_samples'))}`\n\n")
        if data.get("fig7_results") is not None:
            f.write("- Fig7 task-aware baselines: available; direct DFT methods share the same "
                    "evaluation set, LOS mask, communication budget, and WLS localization chain. "
                    "DFT-Fisher-Direct uses balanced Fisher/FIM bin selection with physical-lag "
                    "sidelobe control.\n\n")
        if data.get("fig8_results") is not None:
            f.write("- Fig8 strong task-aware baselines: available; Cao/Zhai-style direct-TDOA "
                    "methods share the same fixed evaluation set, LOS mask, physical lag gate, "
                    "and all-pair WLS localizer. The main Fig8 view keeps the stable official "
                    "strong-baseline subset. GeoHybrid-B64 variants, when present, are "
                    "strict union-budget coarse-to-fine compressed-domain TDOA estimators; "
                    "GeoHybrid-OverBudget-C64F64 is supplement-only and must not be "
                    "treated as a fair CR16 curve. GeoAmbi-DFT v1 remains a fine-only "
                    "ambiguity-suppression diagnostic/ablation.\n\n")
        if data.get("fig8_supp_results") is not None:
            f.write("- Fig8 supplement strong diagnostics: available; Cao2020 uses segmented "
                    "incoherent high-FC scoring, and Zhai phase-superposition uses "
                    "sqrt-power-weighted delay-compensated phasor superposition. "
                    "GeoAmbi fine-only, coarse-only, over-budget, no-consistency, and "
                    "no-uncertainty variants are kept here to diagnose low-SNR variance, "
                    "deterministic ambiguity, feature budget, and WLS weighting.\n\n")
        if data.get("fig9_results") is not None:
            f.write("- Fig9 innovation comparison: available; the frozen Chen-style DAE "
                    "references (CR4/CR8/CR16, when available) and the current "
                    "FrequencySelectiveDAE models are evaluated "
                    "on the same fixed urban8 set, physical lag gate, and all-pair WLS "
                    "localizer.\n\n")
        if make_fig9_focus_results(data) is not None:
            f.write("- Fig9 focused DAE comparison: available; this view removes "
                    "traditional/strong baselines and keeps Raw plus Chen-DAE/FreqDAE "
                    "CR4/CR8/CR16 for direct new-vs-old CR readability.\n\n")
        if make_tdoa_results(data, key="tdoa_results") is not None:
            f.write("- TDOA-only comparison: available; this view promotes existing "
                    "GCC/direct TDOA diagnostics before WLS localization, so it can "
                    "separate pairwise delay-estimation quality from geometry/localizer "
                    "error amplification.\n\n")
        if data.get("fig10_localizer_results") is not None:
            f.write("- Fig10 localizer robustness ablation: available; this is a "
                    "supplement-only high-SNR check that compares default all-pair WLS "
                    "against grid-start/pruned variants. It does not replace Fig3/Fig6-Fig9 "
                    "main conclusions.\n\n")
        f.write("## Mean RMSE Averages\n\n")
        for key, value in mean_avgs.items():
            f.write(f"- {key}: {value:.4f} m\n")
        f.write("\n## Interpretation\n\n")
        f.write("- This is a synthetic urban8/oracle-LOS approximation of Chen 2025, "
                "not a Wireless InSite reproduction.\n")
        f.write("- Geometry oracle checks the WLS geometry/sign lower bound.\n")
        f.write("- Clean oracle checks the clean waveform GCC + WLS upper bound under finite bandwidth.\n")
        f.write("- Mean RMSE is the main metric; median and trimmed RMSE diagnose outlier sensitivity.\n")
        f.write("- In the strict Chen-style Fig6 track, DFT denotes frequency-domain "
                "cross-spectrum TDOA from selected partial-Fourier coefficients. "
                "This uses the cross-correlation theorem and does not zero-fill "
                "individual waveforms before GCC. DFT waveform-reconstruction variants "
                "and DFT-Fisher variants, when present, are supplement/task-aware "
                "diagnostics rather than the main Chen Fig6 DFT curve.\n")
    print(f"[Saved] {path}")


def export(result_dir, export_fig6_figures=True, export_markdown_tables=True,
           export_core_tables=True, export_core_figures=True,
           export_fig6_results=True):
    global EXPORT_MARKDOWN_TABLES
    EXPORT_MARKDOWN_TABLES = bool(export_markdown_tables)
    pkl_path = os.path.join(result_dir, "plot_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    out_dir = os.path.join(result_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)
    if export_core_tables:
        export_training_summary(data["cv_results_dict"], out_dir)
        export_rmse_tables(data["mc_results"], out_dir)
        export_fig2_diagnostics(data["snr_data"], out_dir)
    else:
        print("[Skip] Core tables disabled (training/rmse/fig2).")
    if export_core_figures:
        export_paper_figures(data["mc_results"], out_dir)
    else:
        print("[Skip] Core paper figures disabled (tables/Fig3_*).")
    if export_fig6_results:
        export_fig6_baselines(data.get("fig6_results"), out_dir,
                              export_figures=bool(export_fig6_figures))
    else:
        print("[Skip] Fig6 export disabled.")
    export_fig7_baselines(data.get("fig7_results"), out_dir,
                          export_figures=bool(export_fig6_figures))
    export_fig8_baselines(data.get("fig8_results"), out_dir,
                          export_figures=bool(export_fig6_figures))
    export_fig8_supp_baselines(data.get("fig8_supp_results"), out_dir,
                               export_figures=bool(export_fig6_figures))
    export_fig9_innovation(data.get("fig9_results"), out_dir,
                           export_figures=bool(export_fig6_figures))
    export_fig9_focus(make_fig9_focus_results(data), out_dir,
                      export_figures=bool(export_fig6_figures))
    export_tdoa_metrics(
        make_tdoa_results(data, key="tdoa_results"),
        out_dir, table_prefix="tdoa",
        export_figures=bool(export_fig6_figures),
    )
    export_tdoa_metrics(
        make_tdoa_results(
            data, key="tdoa_focus_results",
            fallback_keys=["fig9_focus_results", "fig9_results"],
        ),
        out_dir, table_prefix="tdoa_focus",
        export_figures=False,
    )
    export_fig10_localizer(data.get("fig10_localizer_results"), out_dir,
                           export_figures=bool(export_fig6_figures))
    export_summary_md(data, result_dir, out_dir)
    print("[Done] Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export paper tables/figures from plot_data.pkl")
    parser.add_argument("result_dir", nargs="?", default=None,
                        help="Result directory containing plot_data.pkl")
    parser.add_argument("--result-dir", dest="result_dir_flag", default=None,
                        help="Result directory containing plot_data.pkl")
    parser.add_argument("--no-fig6-svg", action="store_true",
                        help="Do not write duplicate Fig6 SVG files into tables/")
    parser.add_argument("--csv-only", action="store_true",
                        help="Do not export Markdown table copies; keep CSV plus summary.md")
    parser.add_argument("--direct-only", action="store_true",
                        help="Skip core Fig1-Fig3/Fig6 table/figure exports")
    args = parser.parse_args()
    export(args.result_dir_flag or args.result_dir or DEFAULT_RESULT_DIR,
           export_fig6_figures=not args.no_fig6_svg,
           export_markdown_tables=not args.csv_only,
           export_core_tables=not args.direct_only,
           export_core_figures=not args.direct_only,
           export_fig6_results=not args.direct_only)
