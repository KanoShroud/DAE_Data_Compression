"""BRSR可观测性证书的关键协议测试。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from brsr_observability_certificate.audit_oracles import audit_oracles
from brsr_observability_certificate.certificate import (
    STATIC_BASELINES,
    gate_decision,
    prediction_diagnostics,
)
from brsr_observability_certificate.features import feature_columns
from brsr_observability_certificate.run_certificate import (
    CONFIGS,
    TARGET_TOTAL_BITS,
    _simple_static_rate_matched,
)
from brsr_stage2_rate_adaptive.stage2a import rate_match_assignments


def test_feature_whitelist_excludes_truth_and_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "snr_db": [-5.0],
            "base_posterior_risk": [1.5],
            "action_position": [3.0],
            "posterior_p_000": [0.25],
            "base_code_0_real": [0.1],
            "true_tau": [2.0],
            "actual_squared_error_samples2": [4.0],
            "clairvoyant_best_position": [3],
        }
    )
    columns = feature_columns(frame)
    assert "true_tau" not in columns
    assert "actual_squared_error_samples2" not in columns
    assert "clairvoyant_best_position" not in columns
    assert "base_posterior_risk" in columns
    assert "posterior_p_000" in columns


def test_gate_uses_only_observable_model_vs_static_baselines() -> None:
    effects = pd.DataFrame(
        {
            "scope": ["all", "all"],
            "target": ["Observable-Ridge-B29", "Observable-Ridge-B29"],
            "baseline": list(STATIC_BASELINES),
            "paired_mse_improvement_samples2": [0.2, 0.3],
            "ci95_lower_samples2": [0.05, 0.06],
            "positive_seed_fraction": [1.0, 1.0],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "method": ["Observable-Ridge-B29"],
            "mean_within_sample_spearman": [0.2],
        }
    )
    decision = gate_decision(
        mode="development",
        selected_method="Observable-Ridge-B29",
        effects=effects,
        diagnostics=diagnostics,
    )
    assert decision["status"] == "GO_OBSERVABLE"
    assert decision["clairvoyant_is_gate_evidence"] is False


def test_oracle_audit_rejects_iid_block_as_protocol_evidence() -> None:
    audit = audit_oracles().set_index("legacy_label")
    row = audit.loc["BlockOracle"]
    assert row["oracle_class"] == "ClairvoyantBlockOutcomeOracle"
    assert row["protocol_status"] == "NO_GO_PROTOCOL_IID_BLOCK"


def test_smoke_rate_matching_is_exactly_29_bits() -> None:
    config = CONFIGS["smoke"]
    cluster_ids = [
        f"seed={seed}|trial={trial}"
        for seed in config.test_seeds
        for trial in range(config.test_trials_per_seed)
    ]
    assignments = rate_match_assignments(
        cluster_ids,
        desired_mean_bits=TARGET_TOTAL_BITS,
        low_bits=24,
        high_bits=32,
        salt="observability-test",
    )
    assert np.mean(list(assignments.values())) == TARGET_TOTAL_BITS


def test_rate_matched_static_preserves_multi_position_plan() -> None:
    samples = pd.DataFrame(
        {
            "seed": [1] * 8,
            "trial": list(range(8)),
            "cluster_id": [f"c{index}" for index in range(8)],
            "snr_db": [-5.0] * 8,
        }
    )
    static_rows = []
    for _, sample in samples.iterrows():
        for bits, selected in ((24, "3"), (32, "3 5")):
            static_rows.append(
                {
                    **sample.to_dict(),
                    "method": f"BestStatic-B{bits}",
                    "selected_position": selected,
                    "squared_error_samples2": 1.0,
                }
            )
    result = _simple_static_rate_matched(
        pd.DataFrame(static_rows),
        samples,
        family="BestStatic",
    )
    assert result["selected_position"].isin({"3", "3 5"}).all()
    assert result["total_bits"].mean() == TARGET_TOTAL_BITS


def test_constant_action_scores_do_not_emit_spearman_warning() -> None:
    rows = pd.DataFrame(
        {
            "seed": [1, 1],
            "trial": [0, 0],
            "cluster_id": ["c0", "c0"],
            "snr_db": [-5.0, -5.0],
            "action_position": [2.0, 3.0],
            "actual_squared_error_samples2": [1.0, 4.0],
            "clairvoyant_best_position": [2, 2],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        diagnostics = prediction_diagnostics(
            rows,
            {"ConstantPolicy": np.asarray([1.0, 1.0])},
        )
    assert np.isnan(
        diagnostics.iloc[0]["mean_within_sample_spearman"]
    )
