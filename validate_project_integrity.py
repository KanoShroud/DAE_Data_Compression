"""Fast scientific-integrity gate for the DAE/TDOA project.

Run this script after Python edits and before a formal PyCharm experiment. It
does not train a model and does not write result files.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch
from scipy import signal

from baselines import DFTCompressionBaseline, HadamardProjectionBaseline, PCABaseline
from evaluate import _all_pair_robust_mode, _localize_from_tdoa_pairs, gcc_standard
from experiment_integrity import (
    assert_exact_cr_budget,
    assert_non_oracle_normalization,
    assert_protocol_compatible,
    validate_model_output,
    validate_selected_bins,
    validate_urban_batch,
)
from model import DAE
from signal_gen import SignalSimulator
from task_baselines import DirectDFTTDOAEstimator


ROOT = Path(__file__).resolve().parent


def _check(condition, message):
    if not bool(condition):
        raise RuntimeError(message)
    print(f"[PASS] {message}", flush=True)


def _normalized_rows(x):
    flat = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    return flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)


def _static_source_gate():
    files = sorted(ROOT.glob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    _check(len(files) >= 10, f"all {len(files)} root Python files parse as UTF-8")


def _data_protocol_gate():
    sim = SignalSimulator(
        scenario_mode="urban8", channel_mode="fixed", n_fixed_channels=8,
        channel_pool_seed=42, normalization_mode="per_observation_noisy_rms",
        urban_base_delay=16, urban_min_los=5, urban_train_los_only=True,
        nlos_prob=0.0, delay_label_mode="los", multipath_scale=0.2,
    )
    assert_non_oracle_normalization(sim)
    snapshot_indices = np.arange(4, dtype=int)
    low_noisy, low_clean, low_meta = sim.generate_urban_batch(
        4, snr_db=-10.0, seed=123, snapshot_indices=snapshot_indices,
    )
    high_noisy, high_clean, high_meta = sim.generate_urban_batch(
        4, snr_db=20.0, seed=123, snapshot_indices=snapshot_indices,
    )
    validate_urban_batch(
        low_noisy, low_clean, low_meta, sim, expected_batch=4,
        expected_snapshot_indices=snapshot_indices,
    )
    validate_urban_batch(
        high_noisy, high_clean, high_meta, sim, expected_batch=4,
        expected_snapshot_indices=snapshot_indices,
    )
    for key in ("source", "uavs", "los", "distances", "delay_float", "snapshot_id"):
        _check(
            np.array_equal(np.asarray(low_meta[key]), np.asarray(high_meta[key])),
            f"fixed evaluation preserves {key} across SNR",
        )
    _check(
        np.allclose(
            _normalized_rows(low_clean.numpy()),
            _normalized_rows(high_clean.numpy()), atol=2e-6, rtol=0.0,
        ),
        "fixed evaluation preserves clean waveform direction across SNR",
    )
    _check(
        not np.allclose(low_noisy.numpy(), high_noisy.numpy()),
        "fixed evaluation changes noise level across SNR",
    )
    pair_differences = []
    for row in np.asarray(low_meta["delay_float"]):
        pair_differences.extend(
            float(row[i] - row[j])
            for i in range(len(row)) for j in range(i + 1, len(row))
        )
    _check(
        np.any(np.asarray(pair_differences) < 0)
        and np.any(np.asarray(pair_differences) > 0),
        "urban TDOA labels support both signs",
    )
    pair_a = sim.generate_pair_batch(12, snr_db=0.0, seed=321)
    pair_b = sim.generate_pair_batch(12, snr_db=0.0, seed=321)
    _check(
        torch.equal(pair_a[0], pair_b[0])
        and torch.equal(pair_a[2], pair_b[2])
        and np.array_equal(pair_a[4], pair_b[4])
        and np.array_equal(pair_a[5], pair_b[5]),
        "urban pair generation is seed-reproducible",
    )
    _check(
        not torch.equal(pair_a[0], pair_a[2]),
        "urban pair generation uses two distinct receiver waveforms",
    )
    return sim, low_meta


def _tdoa_and_solver_gate(sim, meta):
    _check(
        _all_pair_robust_mode("all_pair_wls") == "standard"
        and _all_pair_robust_mode("single_ref") is None,
        "localization estimator names resolve explicitly",
    )
    try:
        _all_pair_robust_mode("misspelled_solver")
    except ValueError:
        pass
    else:
        raise RuntimeError("unknown localization estimator was silently accepted")
    _check(True, "unknown localization estimator is rejected")

    rng = np.random.default_rng(7)
    reference = rng.normal(size=sim.signal_len) + 1j * rng.normal(size=sim.signal_len)
    delayed = np.zeros_like(reference)
    shift = 7
    delayed[shift:] = reference[:-shift]
    lags = signal.correlation_lags(sim.signal_len, sim.signal_len, mode="same")
    gcc_lag = float(lags[np.argmax(np.abs(gcc_standard(delayed, reference)))])
    _check(gcc_lag == shift, "standard GCC uses delay_i-minus-delay_j sign")

    direct = DirectDFTTDOAEstimator(
        np.arange(sim.signal_len), signal_len=sim.signal_len,
        label="integrity-full-DFT", phat=False,
    )
    direct_lag = float(direct.estimate_pair(
        delayed, reference, sub_sample=False, return_quality=False,
        lag_limit_samples=32,
    ))
    _check(abs(direct_lag - shift) < 1e-8, "direct DFT preserves the GCC sign")

    uavs = np.asarray(meta["uavs"][0], dtype=float)
    source = np.asarray(meta["source"][0], dtype=float)
    los = np.where(np.asarray(meta["los"][0], dtype=bool))[0]
    distances = np.linalg.norm(uavs - source[None, :], axis=1)
    pairs = []
    for ai in range(len(los)):
        for bi in range(ai + 1, len(los)):
            ui, uj = int(los[ai]), int(los[bi])
            tau = (distances[ui] - distances[uj]) / (sim.c / sim.fs)
            pairs.append((ui, uj, float(tau), 1.0))
    estimate, info = _localize_from_tdoa_pairs(
        uavs, pairs, sim.fs, sim.c, sim.area_size, robust_mode="standard",
    )
    _check(info.get("success", False), "geometry-oracle WLS converges")
    _check(
        np.linalg.norm(np.asarray(estimate) - source) < 1e-4,
        "geometry-oracle WLS reproduces the source position",
    )


def _feature_budget_gate(sim):
    signal_len = int(sim.signal_len)
    target_real = int(2 * signal_len // 16)
    dft = DFTCompressionBaseline(signal_len=signal_len, cr=16, mode="uniform")
    hadamard = HadamardProjectionBaseline(signal_len=signal_len, cr=16)
    pca = PCABaseline(
        torch.zeros(2 * signal_len),
        torch.eye(2 * signal_len)[:target_real],
        signal_len=signal_len, requested_components=target_real,
    )
    _check(dft.selected_bins.numel() * 2 == target_real, "DFT meets exact CR16")
    _check(hadamard.rows_per_channel * 2 == target_real, "Hadamard meets exact CR16")
    _check(pca.feature_dim == target_real, "PCA meets exact CR16")
    assert_exact_cr_budget(target_real, signal_len, 16, label="integrity budget")

    over_budget = np.union1d(np.arange(64), np.arange(24, 88))
    _check(over_budget.size == 88, "expert-union accounting detects the CR11.64 case")
    try:
        assert_exact_cr_budget(2 * over_budget.size, signal_len, 16, label="V5-B union")
    except RuntimeError:
        pass
    else:
        raise RuntimeError("over-budget expert union was not rejected")
    validate_selected_bins(np.arange(64), signal_len, expected_count=64)
    _check(True, "strict CR16 accepts one shared 64-complex-bin budget")


def _model_and_cache_gate(sim):
    x = torch.zeros(1, 2, sim.signal_len)
    for cr in (4, 8, 16):
        model = DAE(cr=cr).eval()
        with torch.no_grad():
            output = model(x)
        validate_model_output(output, x.shape, label=f"DAE-CR{cr}")
        _check(model.latent_dim == 2 * sim.signal_len // cr, f"DAE-CR{cr} latent budget")

    reference = {"seed": 42, "channel_pool_seed": 42, "signal_len": sim.signal_len}
    assert_protocol_compatible(reference, dict(reference), keys=tuple(reference))
    changed = dict(reference, channel_pool_seed=1042)
    try:
        assert_protocol_compatible(reference, changed, keys=tuple(reference))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("cache protocol accepted a different snapshot pool")
    _check(True, "cache protocol rejects a different snapshot pool")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    _static_source_gate()
    sim, meta = _data_protocol_gate()
    _tdoa_and_solver_gate(sim, meta)
    _feature_budget_gate(sim)
    _model_and_cache_gate(sim)
    print("[PASS] project scientific-integrity gate completed", flush=True)


if __name__ == "__main__":
    main()
