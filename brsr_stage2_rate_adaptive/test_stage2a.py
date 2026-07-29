"""Unit tests for the independent BRSR Stage 2A helpers."""

from __future__ import annotations

import math

import numpy as np

from brsr_stage2_rate_adaptive.stage2a import (
    TARGET_MEAN_BITS,
    config_for_mode,
    freeze_rate_policy,
    rate_match_assignments,
    temper_probability,
)


def test_configs_keep_seed_pools_disjoint_and_certified_grid() -> None:
    for mode in ("smoke", "development"):
        config = config_for_mode(mode)
        groups = (
            config.prior_seeds,
            config.plan_seeds,
            config.policy_calibration_seeds,
            config.development_seeds,
        )
        flattened = [seed for group in groups for seed in group]
        assert len(flattened) == len(set(flattened))
        assert (
            config.marginal_amplitude_nodes,
            config.marginal_phase_nodes,
        ) == (5, 64)


def test_temper_probability_is_normalized_and_identity_at_one() -> None:
    probability = np.asarray([0.1, 0.2, 0.7])
    assert np.allclose(temper_probability(probability, 1.0), probability)
    sharpened = temper_probability(probability, 0.5)
    flattened = temper_probability(probability, 2.0)
    assert math.isclose(float(sharpened.sum()), 1.0)
    assert math.isclose(float(flattened.sum()), 1.0)
    assert sharpened[-1] > probability[-1] > flattened[-1]


def test_rate_policy_calibration_hits_registered_mean_bits() -> None:
    rows = [
        {
            "raw_best_predicted_voi_samples2": float(index),
            "calibrated_best_predicted_voi_samples2": float(index),
            "oracle_best_realized_gain_samples2": float(index),
        }
        for index in range(8)
    ]
    for target in TARGET_MEAN_BITS:
        policy = freeze_rate_policy(
            rows,
            snr_db=0.0,
            temperature_mode="calibrated",
            target_mean_bits=target,
        )
        assert math.isclose(policy.calibration_mean_total_bits, target)


def test_rate_matching_is_deterministic_and_within_one_cluster() -> None:
    clusters = [f"cluster-{index}" for index in range(12)]
    first = rate_match_assignments(
        clusters,
        desired_mean_bits=24.0,
        low_bits=16,
        high_bits=32,
        salt="test",
    )
    second = rate_match_assignments(
        reversed(clusters),
        desired_mean_bits=24.0,
        low_bits=16,
        high_bits=32,
        salt="test",
    )
    assert first == second
    achieved = float(np.mean(list(first.values())))
    assert abs(achieved - 24.0) <= (32 - 16) / len(clusters)
