"""Tests for the read-only BRSR Stage 2A failure certificate."""

from __future__ import annotations

import pandas as pd

from brsr_stage2_rate_adaptive.diagnose_stage2a_failures import (
    _decision,
    _fixed_position,
    _policy_thresholds,
    tail_cluster_diagnostics,
)


def test_fixed_position_requires_single_best_static_action() -> None:
    plans = pd.DataFrame(
        [
            {
                "family": "BestStatic",
                "payload_bits": 24,
                "positions": "3",
                "bins": "137",
            }
        ]
    )
    assert _fixed_position(plans) == (3, 137)


def test_policy_threshold_converts_lambda_to_eight_bit_action() -> None:
    policies = pd.DataFrame(
        [
            {
                "snr_db": snr,
                "temperature_mode": "calibrated",
                "target_mean_bits": 24,
                "lambda_per_payload_bit": 0.25,
            }
            for snr in (-10, -5, 0, 5, 10, 15)
        ]
    )
    assert set(_policy_thresholds(policies).values()) == {2.0}


def test_decision_uses_cluster_ci_not_point_estimate() -> None:
    present = pd.DataFrame(
        [
            {
                "scope": "All",
                "feature": "predicted_advantage_over_fixed",
                "n_rows": 100,
                "n_clusters": 20,
                "spearman": 0.2,
                "spearman_ci_low": 0.01,
                "spearman_ci_high": 0.4,
                "auc_for_predicted_beats_fixed": 0.6,
            }
        ]
    )
    assert _decision(present)["status"] == "SCORE_RELATION_PRESENT"
    present.loc[0, "spearman_ci_low"] = -0.1
    assert _decision(present)["status"] == "HOLD_SCORE_RELATION_UNCERTAIN"
    present.loc[0, "spearman_ci_high"] = 0.0
    assert _decision(present)["status"] == "NO_GO_COMPONENT_CURRENT_SCORER"


def test_tail_cluster_diagnostics_preserves_positive_loss_mass() -> None:
    frame = pd.DataFrame(
        {
            "cluster_id": ["a", "a", "b"],
            "seed": [1, 1, 2],
            "trial": [0, 0, 0],
            "snr_db": [0.0, 5.0, 0.0],
            "predicted_is_fixed": [False, False, False],
            "predicted_advantage_over_fixed_samples2": [1.0, 2.0, 3.0],
            "realized_advantage_over_fixed_samples2": [-2.0, 1.0, -3.0],
            "predicted_vs_fixed_delta_samples2": [2.0, -1.0, 3.0],
            "predicted_vs_oracle_regret_samples2": [2.0, 0.0, 4.0],
        }
    )
    result = tail_cluster_diagnostics(frame)
    assert result["positive_loss_sum_samples2"].sum() == 5.0
    assert result["positive_loss_share"].sum() == 1.0
