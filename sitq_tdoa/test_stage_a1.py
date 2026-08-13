"""Tests for the read-only SITQ Stage A.1 stability certificate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sitq_tdoa.diagnose_stage_a1_threshold_stability import (
    _bootstrap_selection,
    _cluster_bootstrap,
    _crossfreeze,
    decide_gate,
)


def _toy_matrices() -> tuple[pd.MultiIndex, np.ndarray, np.ndarray, np.ndarray]:
    index = pd.MultiIndex.from_tuples(
        [(1, trial) for trial in range(6)] + [(2, trial) for trial in range(6)],
        names=["seed", "trial"],
    )
    thresholds = np.asarray([0.4, 0.5, 0.6])
    risk = np.tile(np.asarray([2.0, 1.0, 1.5]), (index.size, 1))
    entropy = np.tile(np.asarray([1.6, 1.2, 1.0]), (index.size, 1))
    return index, thresholds, risk, entropy


def test_crossfreeze_covers_each_trajectory_once() -> None:
    index, thresholds, risk, entropy = _toy_matrices()
    rows, folds = _crossfreeze(index, risk, entropy, thresholds, folds=3)
    assert rows.shape[0] == index.size
    assert not rows.duplicated(["seed", "trial"]).any()
    assert folds.shape[0] == 3
    assert set(rows["fold"]) == {0, 1, 2}


def test_bootstrap_selection_uses_frozen_grid() -> None:
    _, thresholds, risk, entropy = _toy_matrices()
    rows, summary = _bootstrap_selection(
        risk,
        entropy,
        thresholds,
        sitq_threshold=0.5,
        condmi_threshold=0.6,
        repetitions=200,
        seed=7,
    )
    assert set(rows["sitq_selected_threshold"]) == {0.5}
    assert set(rows["condmi_selected_threshold"]) == {0.6}
    assert summary["fixed_threshold_risk_gain"] == 0.5


def test_cluster_bootstrap_reports_direction_and_mde() -> None:
    summary = _cluster_bootstrap(np.asarray([1.0, 2.0, 3.0]), 200, seed=9)
    assert summary["mean"] == 2.0
    assert summary["ci95_low"] > 0.0
    assert summary["approx_mde95"] > 0.0


def test_gate_requires_crossfrozen_positive_ci_and_seed_direction() -> None:
    positive_seeds = pd.DataFrame(
        {"seed": [1, 2], "mean_risk_gain": [0.2, 0.1]}
    )
    negative_seeds = pd.DataFrame(
        {"seed": [1, 2], "mean_risk_gain": [-0.2, -0.1]}
    )
    assert (
        decide_gate(
            True,
            {"mean": 0.15, "ci95_low": 0.02, "ci95_high": 0.3},
            positive_seeds,
        )
        == "STABLE_SELECTION_SIGNAL"
    )
    assert (
        decide_gate(
            True,
            {"mean": -0.15, "ci95_low": -0.3, "ci95_high": -0.02},
            negative_seeds,
        )
        == "NO_GO_COMPONENT_SINGLE_GLOBAL_THRESHOLD"
    )
    assert (
        decide_gate(
            True,
            {"mean": 0.05, "ci95_low": -0.1, "ci95_high": 0.2},
            positive_seeds,
        )
        == "HOLD_SELECTION_UNCERTAIN"
    )
    assert (
        decide_gate(
            False,
            {"mean": 0.15, "ci95_low": 0.02, "ci95_high": 0.3},
            positive_seeds,
        )
        == "INVALID_STAGE_A1_DIAGNOSTIC"
    )
