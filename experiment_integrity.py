"""Shared experiment-integrity checks for training and evaluation entry points."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PROTOCOL_KEYS = (
    "seed",
    "channel_pool_seed",
    "scenario_mode",
    "channel_mode",
    "n_fixed_channels",
    "nlos_prob",
    "delay_label_mode",
    "multipath_scale",
    "normalization_mode",
    "urban_base_delay",
    "urban_min_los",
    "urban_train_los_only",
    "train_snr_range",
    "eval_snr_range",
    "monte_carlo_trials",
    "fixed_eval_set",
    "use_los_only",
    "tdoa_sub_sample",
    "tdoa_lag_limit_samples",
    "localization_estimator",
    "baseline_cr",
    "baseline_dft_mode",
    "baseline_dft_direct_source",
    "baseline_hadamard_mode",
    "baseline_pca_train_source",
    "baseline_pca_fixed_snr_db",
    "baseline_pca_samples",
    "signal_len",
    "fs_hz",
    "c_mps",
    "area_size",
    "baseline_code_fingerprint",
)


def _canonical(value):
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def protocol_subset(config, keys=PROTOCOL_KEYS):
    config = dict(config or {})
    return {key: _canonical(config.get(key, "<missing>")) for key in keys}


def protocol_fingerprint(config, keys=PROTOCOL_KEYS):
    payload = json.dumps(
        protocol_subset(config, keys=keys), sort_keys=True,
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_protocol_compatible(reference, current, keys=PROTOCOL_KEYS,
                               context="cached experiment"):
    ref = protocol_subset(reference, keys=keys)
    cur = protocol_subset(current, keys=keys)
    mismatches = {
        key: {"reference": ref[key], "current": cur[key]}
        for key in keys if ref[key] != cur[key]
    }
    if mismatches:
        preview = "; ".join(
            f"{key}: {item['reference']!r} != {item['current']!r}"
            for key, item in list(mismatches.items())[:8]
        )
        raise RuntimeError(
            f"{context} protocol mismatch ({len(mismatches)} fields): {preview}"
        )
    return protocol_fingerprint(ref, keys=keys)


def validate_selected_bins(selected_bins, signal_len, expected_count=None,
                           label="selected bins"):
    raw = np.asarray(selected_bins)
    if raw.ndim != 1 or raw.size < 1:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(raw.astype(float))):
        raise ValueError(f"{label} contains non-finite values")
    bins = raw.astype(int)
    if not np.allclose(raw.astype(float), bins.astype(float), rtol=0.0, atol=0.0):
        raise ValueError(f"{label} must contain integer FFT indices")
    if np.any(bins < 0) or np.any(bins >= int(signal_len)):
        raise ValueError(f"{label} must lie in [0, {int(signal_len)})")
    if np.unique(bins).size != bins.size:
        raise ValueError(f"{label} contains duplicate FFT indices")
    if expected_count is not None and bins.size != int(expected_count):
        raise ValueError(
            f"{label} has {bins.size} bins, expected {int(expected_count)}"
        )
    return bins


def feature_budget_from_bins(selected_bins, signal_len, label="method"):
    bins = validate_selected_bins(selected_bins, signal_len, label=f"{label} bins")
    real_scalars = int(2 * bins.size)
    return {
        "label": str(label),
        "unique_complex_bins": int(bins.size),
        "transmitted_real_scalars": real_scalars,
        "effective_cr": float(2 * int(signal_len) / real_scalars),
    }


def assert_exact_cr_budget(real_scalars, signal_len, cr, label="method"):
    target = int(2 * int(signal_len) // int(cr))
    if int(real_scalars) != target:
        raise RuntimeError(
            f"{label} violates exact CR{int(cr)}: transmitted_real_scalars="
            f"{int(real_scalars)}, expected={target}"
        )
    return target


def assert_non_oracle_normalization(simulator, context="formal experiment"):
    if bool(getattr(simulator, "normalization_uses_clean_target", False)):
        raise RuntimeError(
            f"{context} uses normalization_mode="
            f"{getattr(simulator, 'normalization_mode', '<unknown>')!r}, which "
            "requires the unavailable clean target. Use a noisy-input or fixed "
            "dataset normalization protocol for a deployable comparison."
        )
    return True


def validate_urban_batch(noisy, clean, meta, simulator, expected_batch=None,
                         expected_snapshot_indices=None):
    noisy_np = noisy.detach().cpu().numpy() if torch.is_tensor(noisy) else np.asarray(noisy)
    clean_np = clean.detach().cpu().numpy() if torch.is_tensor(clean) else np.asarray(clean)
    expected_shape = (
        noisy_np.shape[0], int(simulator.n_uavs), 2, int(simulator.signal_len)
    )
    if noisy_np.shape != expected_shape or clean_np.shape != expected_shape:
        raise RuntimeError(
            f"urban waveform shape mismatch: noisy={noisy_np.shape}, "
            f"clean={clean_np.shape}, expected={expected_shape}"
        )
    if expected_batch is not None and noisy_np.shape[0] != int(expected_batch):
        raise RuntimeError("urban batch size differs from the requested batch size")
    if not np.all(np.isfinite(noisy_np)) or not np.all(np.isfinite(clean_np)):
        raise RuntimeError("urban batch contains NaN or Inf")

    required = {
        "source", "uavs", "los", "distances", "delays", "delay_float",
        "snapshot_id", "snr_db",
    }
    missing = sorted(required - set(meta))
    if missing:
        raise RuntimeError(f"urban metadata missing fields: {missing}")
    n = noisy_np.shape[0]
    if np.asarray(meta["source"]).shape != (n, 2):
        raise RuntimeError("source metadata shape mismatch")
    if np.asarray(meta["uavs"]).shape != (n, simulator.n_uavs, 2):
        raise RuntimeError("UAV metadata shape mismatch")
    if np.asarray(meta["los"]).shape != (n, simulator.n_uavs):
        raise RuntimeError("LOS metadata shape mismatch")
    if np.any(np.sum(np.asarray(meta["los"], dtype=bool), axis=1) < simulator.urban_min_los):
        raise RuntimeError("urban batch violates the configured minimum LOS count")

    source = np.asarray(meta["source"], dtype=float)
    uavs = np.asarray(meta["uavs"], dtype=float)
    distances = np.asarray(meta["distances"], dtype=float)
    recomputed = np.linalg.norm(uavs - source[:, None, :], axis=-1)
    if not np.allclose(distances, recomputed, rtol=0.0, atol=1e-7):
        raise RuntimeError("distance metadata is inconsistent with source/UAV geometry")
    delay_expected = float(simulator.urban_base_delay) + distances / float(
        simulator.sample_distance_m
    )
    if not np.allclose(
        np.asarray(meta["delay_float"], dtype=float), delay_expected,
        rtol=0.0, atol=1e-7,
    ):
        raise RuntimeError("delay_float is inconsistent with geometry and sampling rate")
    if expected_snapshot_indices is not None:
        expected_ids = np.asarray(expected_snapshot_indices, dtype=int)
        if expected_ids.shape != (n,) or not np.array_equal(
            np.asarray(meta["snapshot_id"], dtype=int), expected_ids
        ):
            raise RuntimeError("snapshot IDs differ from the requested fixed evaluation plan")
    return True


def validate_model_output(output, expected_shape, label="model"):
    if not torch.is_tensor(output):
        raise TypeError(f"{label} output must be a torch.Tensor")
    if tuple(output.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{label} output shape={tuple(output.shape)}, expected={tuple(expected_shape)}"
        )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError(f"{label} output contains NaN or Inf")
    return True


def code_fingerprint(root=None, filenames=None):
    root = Path(root or Path(__file__).resolve().parent)
    names = filenames or (
        "main.py", "signal_gen.py", "train.py", "model.py", "evaluate.py",
        "baselines.py", "task_baselines.py", "compressed_tdoa.py",
        "topk_localization.py", "experiment_integrity.py",
    )
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        if not path.exists():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
