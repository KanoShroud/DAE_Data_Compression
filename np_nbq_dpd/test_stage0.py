"""Mechanical and scientific-invariant tests for NP-NBQ-DPD Stage 0."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from np_nbq_dpd.core import (
    ModelConfig,
    NPNBQModel,
    allocation_payload_bits,
    bsc_transition_probability,
    complex_code_probability,
    default_public_profiles,
    enumerate_allocations,
)
from np_nbq_dpd.run_stage0 import (
    _calibration_diagnostics,
    _design_stability,
    _gate_status,
    _paired_effect,
    build_experiment_config,
)


def test_all_legal_allocations_have_the_same_24_bit_payload() -> None:
    config = ModelConfig()
    allocations = enumerate_allocations(config)
    assert len(allocations) == 336
    assert all(allocation_payload_bits(item) == 24 for item in allocations)
    assert (2, 2, 2, 2, 2, 2) in allocations


def test_quantized_bsc_probabilities_partition_unity() -> None:
    means = np.asarray([-1.3 + 0.4j, 0.2 - 0.7j, 1.1 + 1.4j])
    for bits in (1, 2, 3):
        levels = 2**bits
        total = sum(
            complex_code_probability(means, 0.8, bits, code, 0.09)
            for code in range(levels * levels)
        )
        assert np.allclose(total, 1.0, atol=1e-11, rtol=0.0)
        for transmitted in range(levels):
            row_sum = sum(
                bsc_transition_probability(transmitted, received, bits, 0.09)
                for received in range(levels)
            )
            assert math.isclose(row_sum, 1.0, abs_tol=1e-12)


def test_generated_and_decoded_nuisance_priors_match() -> None:
    model = NPNBQModel(ModelConfig())
    expected = (
        len(model.config.source_phase_grid_rad) ** model.config.n_frequencies
        * len(model.config.source_amplitude_ratio_grid)
        * len(model.config.gain_phase_grid_rad) ** (model.config.n_sensors - 1)
        * len(model.config.gain_amplitude_grid) ** (model.config.n_sensors - 1)
    )
    assert model.nuisance_state_count == expected


def test_quantized_and_unquantized_position_posteriors_are_normalized() -> None:
    model = NPNBQModel(
        ModelConfig(
            x_grid_m=(-40.0, 0.0, 40.0),
            y_grid_m=(30.0, 70.0, 110.0),
            source_phase_grid_rad=(0.0, math.pi),
        )
    )
    profile = default_public_profiles()[1]
    observation = model.generate_observations(profile, 1234, 1)[0]
    cache = model.build_likelihood_cache(profile, observation)
    quantized = model.posterior_from_cache(cache, model.equal_bit_allocation())
    unquantized = model.posterior_unquantized(profile, observation)
    assert np.all(np.isfinite(quantized))
    assert np.all(np.isfinite(unquantized))
    assert math.isclose(float(np.sum(quantized)), 1.0, abs_tol=1e-10)
    assert math.isclose(float(np.sum(unquantized)), 1.0, abs_tol=1e-10)


def test_design_risk_does_not_read_true_position_metadata() -> None:
    model = NPNBQModel(
        ModelConfig(
            x_grid_m=(-40.0, 0.0, 40.0),
            y_grid_m=(30.0, 70.0, 110.0),
            source_phase_grid_rad=(0.0, math.pi),
        )
    )
    profile = default_public_profiles()[0]
    observations = model.generate_observations(profile, 551, 2)
    caches = [model.build_likelihood_cache(profile, item) for item in observations]
    changed = [replace(cache, true_position_index=(cache.true_position_index + 1) % 9) for cache in caches]
    allocation = model.equal_bit_allocation()
    assert math.isclose(
        model.mean_design_risk(caches, allocation),
        model.mean_design_risk(changed, allocation),
        abs_tol=1e-12,
    )


def test_jiang_style_is_a_strict_budget_adaptation() -> None:
    model = NPNBQModel(ModelConfig())
    assert model.config.quantizer_code_mapping == "natural_binary"
    for profile in default_public_profiles():
        allocation = model.jiang_style_allocation(profile)
        assert allocation_payload_bits(allocation) == 24
        per_sensor = np.asarray(allocation).reshape(3, 2)
        assert np.all(per_sensor[:, 0] == per_sensor[:, 1])
        assert sorted(per_sensor[:, 0].tolist()) == [1, 2, 3]


def test_projected_fisher_proxy_is_finite_for_equal_allocation() -> None:
    model = NPNBQModel(ModelConfig())
    score = model.projected_fisher_score(
        default_public_profiles()[1], model.equal_bit_allocation()
    )
    assert math.isfinite(score)


def test_design_and_evaluation_random_streams_are_disjoint() -> None:
    config = build_experiment_config("development")
    assert config.design_seed != config.evaluation_seed


def test_gate_is_directional_and_smoke_never_claims_performance() -> None:
    effects = pd.DataFrame(
        [
            {
                "baseline": baseline,
                "ci95_low_m2": 0.1,
                "ci95_high_m2": 0.5,
            }
            for baseline in (
                "StrongestFrozenBaseline",
                "GeometryOnly",
                "ReliabilityOnly",
            )
        ]
    )
    allocations = pd.DataFrame(
        [{"method": "JointRisk", "allocation": "3-3-3-1-1-1"}]
    )
    profile_effects = pd.DataFrame(
        [
            {
                "baseline": "StrongestFrozenBaseline",
                "profile_scope": profile,
                "ci95_high_m2": 0.5,
            }
            for profile in ("left_reliable", "right_reliable", "top_degraded")
        ]
    )
    assert (
        _gate_status("smoke", True, effects, profile_effects, allocations)
        == "SMOKE_ONLY"
    )
    assert (
        _gate_status("development", True, effects, profile_effects, allocations)
        == "GO_STRUCTURE_HEADROOM"
    )
    failed = effects.copy()
    failed.loc[
        failed["baseline"] == "StrongestFrozenBaseline", "ci95_high_m2"
    ] = 0.0
    failed.loc[
        failed["baseline"] == "StrongestFrozenBaseline", "ci95_low_m2"
    ] = -0.5
    assert (
        _gate_status("development", True, failed, profile_effects, allocations)
        == "NO_GO_STRUCTURE_HEADROOM"
    )
    regressed = profile_effects.copy()
    regressed.loc[
        regressed["profile_scope"] == "right_reliable", "ci95_high_m2"
    ] = -0.1
    assert (
        _gate_status("development", True, effects, regressed, allocations)
        == "NO_GO_HETEROGENEOUS_PROFILE_REGRESSION"
    )


def test_profile_stratified_effect_preserves_equal_profile_weight() -> None:
    rows = []
    for method, errors in {
        "JointRisk": {"a": [0.0, 0.0], "b": [0.0, 0.0, 0.0, 0.0]},
        "Baseline": {"a": [9.0, 11.0], "b": [0.0, 0.0, 0.0, 0.0]},
    }.items():
        for profile, values in errors.items():
            for trial, value in enumerate(values):
                rows.append(
                    {
                        "method": method,
                        "profile_id": profile,
                        "seed": 1,
                        "trial": trial,
                        "squared_error_m2": value,
                    }
                )
    effect = _paired_effect(
        pd.DataFrame(rows), "JointRisk", "Baseline", 200, 77
    )
    assert math.isclose(float(effect["paired_mse_gain_m2"]), 5.0, abs_tol=1e-12)
    assert int(effect["n_profile_strata"]) == 2
    assert float(effect["approx_mde80_two_sided_m2"]) > float(
        effect["normal_ci95_halfwidth_m2"]
    )


def test_calibration_diagnostic_reports_risk_mse_identity() -> None:
    rows = pd.DataFrame(
        {
            "method": ["Matched"] * 4,
            "profile_id": ["p"] * 4,
            "posterior_risk_m2": [1.0, 2.0, 3.0, 4.0],
            "squared_error_m2": [1.0, 2.0, 3.0, 4.0],
            "credible90_contains_true": [True, True, False, True],
        }
    )
    result = _calibration_diagnostics(rows, 100, 91).iloc[0]
    assert math.isclose(float(result["risk_to_mse_ratio"]), 1.0, abs_tol=1e-12)
    assert math.isclose(float(result["risk_minus_mse_m2"]), 0.0, abs_tol=1e-12)
    assert result["calibration_status"] == "PASS_DESCRIPTIVE"


def test_design_stability_probabilities_are_normalized() -> None:
    allocations = ((1, 1), (1, 2), (2, 1))
    matrix = np.asarray(
        [
            [1.0, 1.1, 0.9, 1.0],
            [1.2, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.2, 1.3],
        ]
    )
    table = pd.DataFrame(
        {
            "profile_id": ["p"] * 3,
            "allocation": ["1-1", "1-2", "2-1"],
            "actual_design_risk_m2": matrix.mean(axis=1),
        }
    )
    details, bits, summary = _design_stability(
        table, matrix, allocations, 200, 123
    )
    assert math.isclose(
        float(details["bootstrap_winner_frequency"].sum()), 1.0, abs_tol=1e-12
    )
    assert bits.groupby("cell")["bootstrap_selection_frequency"].sum().sub(1.0).abs().lt(1e-12).all()
    assert summary["frozen_joint_allocation"] == "1-1"


def test_allocation_collapse_has_an_explicit_gate_reason() -> None:
    effects = pd.DataFrame(
        [
            {"baseline": name, "ci95_low_m2": 0.0, "ci95_high_m2": 0.0}
            for name in (
                "StrongestFrozenBaseline",
                "GeometryOnly",
                "ReliabilityOnly",
            )
        ]
    )
    allocations = pd.DataFrame(
        [
            {
                "method": "JointRisk",
                "allocation": "2-2-2-2-2-2",
            }
            for _ in range(4)
        ]
    )
    assert (
        _gate_status("development", True, effects, pd.DataFrame(), allocations)
        == "NO_GO_ALLOCATION_COLLAPSE"
    )
