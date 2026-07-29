"""BRSR路线可行性阶段1的理论、协议与公平性测试。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from brsr_route_feasibility.feasibility import (
    action_levels,
    base_levels,
    build_candidate_pools,
    build_likelihood_cases,
    config_for_mode,
    continuous_rho_quadrature,
    evaluate_actual_actions,
    expected_action_risks,
    observation_from_context,
    one_action_positions,
    phase_likelihood_context,
    physical_public_scale,
)
from brsr_route_feasibility.run_stage1 import (
    QUADRATURE_DIAGNOSTIC_GRIDS,
    QUADRATURE_REFERENCE_GRID,
    _gate,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer
from brsr_tdoa.brsr_stage_b import evaluate_fixed
from brsr_tdoa.brsr_waveform import (
    WaveformConfig,
    candidate_fft_values,
    estimate_candidate_statistics,
    generate_structured_trial,
    observe_trial,
)


def _small_case():
    waveform = WaveformConfig()
    pools = build_candidate_pools(waveform)
    bins = pools["Current-Fisher"]
    rng = np.random.default_rng(2026072391)
    prior = [generate_structured_trial(rng, waveform) for _ in range(4)]
    power, _, _ = estimate_candidate_statistics(prior, bins)
    models = build_likelihood_cases(
        waveform=waveform,
        bins=bins,
        source_power=power,
        tau_step=1.0,
        amplitude_nodes=3,
        phase_nodes=8,
    )
    trial = generate_structured_trial(rng, waveform)
    y_a, y_b, sigma2 = observe_trial(trial, 5.0)
    model = models["Phase-Discrete24"]
    scale = physical_public_scale(power, sigma2)
    context = phase_likelihood_context(
        y_a_selected=candidate_fft_values(y_a, bins),
        sigma2=sigma2,
        scale=scale,
        model=model,
    )
    quantizer = NestedComplexQuantizer(max_bits=8, clip_abs=4.0)
    observation = observation_from_context(
        stage="test",
        seed=1,
        trial=0,
        snr_db=5.0,
        true_tau=trial.true_tau,
        true_rho=trial.true_rho,
        y_b_selected=candidate_fft_values(y_b, bins),
        scale=scale,
        context=context,
        quantizer=quantizer,
    )
    return waveform, pools, model, observation, quantizer


def test_candidate_pools_have_equal_valid_resources() -> None:
    waveform = WaveformConfig()
    pools = build_candidate_pools(waveform)
    assert set(pools) == {
        "Current-Fisher",
        "Schur-Fisher",
        "Global-Risk",
        "Local-Global-Pareto",
    }
    for bins in pools.values():
        assert bins.shape == (12,)
        assert np.unique(bins).size == 12
        assert set(waveform.base_bins).issubset(set(bins.tolist()))
        assert np.min(np.diff(bins)) >= waveform.min_candidate_spacing


def test_continuous_rho_quadrature_matches_uniform_prior() -> None:
    rho, prior = continuous_rho_quadrature(5, 16)
    expected_second_moment = (0.90**3 - 0.70**3) / (3.0 * 0.20)
    assert rho.shape == prior.shape == (80,)
    assert math.isclose(float(np.sum(prior)), 1.0, abs_tol=1e-14)
    assert abs(float(np.sum(prior * np.abs(rho) ** 2)) - expected_second_moment) < 1e-12
    assert abs(complex(np.sum(prior * rho))) < 1e-14


def test_stage1_uses_validated_quadrature_configuration() -> None:
    for mode in ("smoke", "development"):
        config = config_for_mode(mode)
        configured = (
            config.marginal_amplitude_nodes,
            config.marginal_phase_nodes,
        )
        assert configured == (5, 64)
        assert configured in QUADRATURE_DIAGNOSTIC_GRIDS
    assert QUADRATURE_REFERENCE_GRID == (9, 128)
    assert QUADRATURE_DIAGNOSTIC_GRIDS[-1] == QUADRATURE_REFERENCE_GRID


def test_one_action_certificate_has_exact_29_bit_budget_and_oracle_invariant() -> None:
    _, _, model, observation, quantizer = _small_case()
    base, rows = evaluate_actual_actions(
        observation,
        quantizer=quantizer,
        model=model,
    )
    assert len(rows) == 10
    assert all(row["payload_bits"] == 24 for row in rows)
    assert all(row["adaptive_feedback_bits"] == 5 for row in rows)
    assert all(row["adaptive_total_bits"] == 29 for row in rows)
    assert all(action_levels(model, row["position"])[row["position"]] == 4 for row in rows)
    assert sum(base_levels(model)) == 8

    best = min(rows, key=lambda row: row["squared_error_samples2"])
    fixed = rows[3]
    assert best["squared_error_samples2"] <= fixed["squared_error_samples2"]
    result = evaluate_fixed(
        observation,
        method="test",
        levels=action_levels(model, fixed["position"]),
        quantizer=quantizer,
        model=model,
    )
    assert abs(result.estimate - fixed["estimate_tau"]) < 1e-12


def test_expected_information_does_not_increase_bayes_risk() -> None:
    _, _, model, observation, quantizer = _small_case()
    base_result = evaluate_fixed(
        observation,
        method="base",
        levels=base_levels(model),
        quantizer=quantizer,
        model=model,
    )
    scores = expected_action_risks(
        observation,
        quantizer=quantizer,
        model=model,
    )
    assert set(scores) == set(one_action_positions(model))
    assert all(value <= base_result.posterior_risk + 1e-9 for value in scores.values())
    assert all(value >= 0.0 and math.isfinite(value) for value in scores.values())


def test_registered_seed_pools_and_score_subsets_are_valid() -> None:
    for mode in ("smoke", "development"):
        config = config_for_mode(mode)
        flattened = (
            config.prior_seeds
            + config.calibration_seeds
            + config.evaluation_seeds
        )
        assert len(flattened) == len(set(flattened))
        assert config.score_trials_per_seed <= config.evaluation_trials_per_seed


def test_stage1_gate_requires_risk_and_score_to_pass_for_same_likelihood() -> None:
    config = config_for_mode("development")
    effects = pd.DataFrame(
        [
            {
                "target": "Oracle-OneAcquire",
                "baseline": "SNR-only-OneAcquire",
                "likelihood": "Linear-Marginal",
                "pool": "Current-Fisher",
                "mse_gain_ci_low": 1.0,
            },
            {
                "target": "RiskGreedy-OneAcquire",
                "baseline": "BestStatic-OneAcquire",
                "likelihood": "Linear-Marginal",
                "pool": "Current-Fisher",
                "mse_gain_ci_low": 1.0,
            },
            {
                "target": "RiskGreedy-OneAcquire",
                "baseline": "SNR-only-OneAcquire",
                "likelihood": "Linear-Marginal",
                "pool": "Current-Fisher",
                "mse_gain_ci_low": 1.0,
            },
        ]
    )
    validations = {"quadrature_convergence_pass": True}
    mismatched_scores = pd.DataFrame(
        [
            {
                "likelihood": "Phase-Marginal",
                "spearman_predicted_vs_actual_gain": 0.5,
            }
        ]
    )
    mismatched = _gate(
        config=config,
        effects=effects,
        scores=mismatched_scores,
        validations=validations,
        score_pool="Current-Fisher",
    )
    assert mismatched["status"] == "HOLD-LIKELIHOOD-OR-SCORE"

    matched_scores = mismatched_scores.assign(likelihood="Linear-Marginal")
    matched = _gate(
        config=config,
        effects=effects,
        scores=matched_scores,
        validations=validations,
        score_pool="Current-Fisher",
    )
    assert matched["status"] == "STAGE1_PASS"
    assert matched["deployable_likelihoods_passing_stage1"] == [
        "Linear-Marginal"
    ]
    assert matched["quadrature_main_grid"] == [5, 64]
    assert matched["quadrature_reference_grid"] == [9, 128]


def test_stage1_gate_holds_when_marginal_quadrature_is_not_converged() -> None:
    result = _gate(
        config=config_for_mode("development"),
        effects=pd.DataFrame(),
        scores=pd.DataFrame(),
        validations={"quadrature_convergence_pass": False},
        score_pool="Current-Fisher",
    )
    assert result["status"] == "HOLD-NUMERICAL-QUADRATURE"
