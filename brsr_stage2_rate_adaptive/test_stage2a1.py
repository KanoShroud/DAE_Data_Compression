"""Unit tests for the preregistered Stage 2A.1 safe gate."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from brsr_stage2_rate_adaptive.stage2a1 import (
    apply_threshold,
    build_action_gate_samples,
    certify_threshold,
    fit_threshold,
)


def _actions(realized_candidate_error: float = 0.25) -> pd.DataFrame:
    rows = []
    for trial in range(12):
        for position, score, error in (
            (3, 0.4, 1.0),
            (4, 0.9 + 0.01 * trial, realized_candidate_error),
        ):
            rows.append(
                {
                    "seed": 1,
                    "trial": trial,
                    "cluster_id": f"1:{trial}",
                    "snr_db": 0.0,
                    "true_tau": 0.0,
                    "position": position,
                    "bin_index": 100 + position,
                    "calibrated_predicted_voi_samples2": score,
                    "calibrated_squared_error_samples2": error,
                }
            )
    return pd.DataFrame(rows)


def test_build_samples_and_threshold_fallback() -> None:
    samples = build_action_gate_samples(_actions(), fixed_position=3)
    selected = apply_threshold(samples, threshold=0.45)
    assert selected["adaptive_selected"].all()
    assert np.allclose(selected["realized_policy_gain_samples2"], 0.75)
    fallback = apply_threshold(samples, threshold=math.inf)
    assert not fallback["adaptive_selected"].any()
    assert np.allclose(fallback["selected_squared_error_samples2"], 1.0)


def test_fit_and_certificate_accept_consistent_gain() -> None:
    samples = build_action_gate_samples(_actions(), fixed_position=3)
    threshold, table = fit_threshold(samples, repetitions=100, seed=11)
    assert math.isfinite(threshold)
    fit_lcb = float(table.loc[table["threshold"].eq(threshold), "lcb"].iloc[0])
    policy = certify_threshold(
        samples,
        snr_db=0.0,
        threshold=threshold,
        fit_lcb=fit_lcb,
        repetitions=100,
        seed=12,
    )
    assert policy.certified
    assert policy.selected_threshold == threshold


def test_fit_rejects_consistently_harmful_candidate() -> None:
    samples = build_action_gate_samples(
        _actions(realized_candidate_error=2.0),
        fixed_position=3,
    )
    threshold, _ = fit_threshold(samples, repetitions=100, seed=21)
    assert math.isinf(threshold)
    policy = certify_threshold(
        samples,
        snr_db=0.0,
        threshold=threshold,
        fit_lcb=0.0,
        repetitions=100,
        seed=22,
    )
    assert not policy.certified
    assert math.isinf(policy.selected_threshold)
