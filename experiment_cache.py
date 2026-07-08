import copy
import os
import pickle

import numpy as np


def load_plot_data(result_dir):
    """Load a saved plot_data.pkl from a result directory."""
    pkl_path = os.path.join(str(result_dir), "plot_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"plot_data.pkl not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_static_baseline_cache(result_dir):
    """
    Load static baseline views that do not depend on the current innovation model.

    Fig6-Fig8 compare Raw/Clean/Geometry, frozen Chen-style DAE and traditional or
    task-aware baselines. Innovation runs can reuse these fields as long as the
    baseline configuration and evaluation set are intentionally unchanged.
    """
    data = load_plot_data(result_dir)
    return {
        "source_dir": str(result_dir),
        "fig6_results": data.get("fig6_results"),
        "fig7_results": data.get("fig7_results"),
        "fig8_results": data.get("fig8_results"),
        "fig8_supp_results": data.get("fig8_supp_results"),
        "baseline_all_results": data.get("baseline_all_results"),
        "traditional_baseline_meta": data.get("traditional_baseline_meta"),
        "config": data.get("config", {}),
    }


def _snr_compatible(a, b):
    if a is None or b is None:
        return True
    try:
        return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    except Exception:
        return False


def merge_method_comparison_results(base_method_data, overlay_method_data,
                                    overlay_first=True, title=None,
                                    config_updates=None):
    """
    Merge two run_urban_method_comparison outputs.

    Values from overlay_method_data replace methods with the same name. This is
    useful for innovation runs: reuse cached static baselines, but refresh Raw,
    frozen DAE references and current innovation curves under the current code.
    """
    if base_method_data is None:
        return overlay_method_data
    if overlay_method_data is None:
        return base_method_data

    base_results, base_snr = base_method_data
    overlay_results, overlay_snr = overlay_method_data
    if not _snr_compatible(base_snr, overlay_snr):
        raise ValueError("Cannot merge method comparison results with different SNR grids")

    merged = copy.deepcopy(base_results)
    overlay = copy.deepcopy(overlay_results)
    merged.setdefault("methods", {})
    merged.setdefault("method_diagnostics", {})
    merged.setdefault("outlier_details", {})

    for label, values in overlay.get("methods", {}).items():
        merged["methods"][label] = values
    for label, values in overlay.get("method_diagnostics", {}).items():
        merged["method_diagnostics"][label] = values
    for label, values in overlay.get("outlier_details", {}).items():
        merged["outlier_details"][label] = values

    order = []
    seen = set()
    order_sources = []
    if overlay_first:
        order_sources.extend([overlay.get("method_order", []), base_results.get("method_order", [])])
    else:
        order_sources.extend([base_results.get("method_order", []), overlay.get("method_order", [])])
    for src in order_sources:
        for label in src:
            if label in merged["methods"] and label not in seen:
                order.append(label)
                seen.add(label)
    merged["method_order"] = order

    config = dict(base_results.get("config", {}))
    config.update(overlay.get("config", {}))
    if config_updates:
        config.update(config_updates)
    merged["config"] = config
    if title is not None:
        merged["title"] = title
    return merged, base_snr
