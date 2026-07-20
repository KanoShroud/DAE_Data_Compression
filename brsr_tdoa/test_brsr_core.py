"""BRSR-TDOA 阶段 A 的确定性理论与协议不变量测试。"""

from __future__ import annotations

import math

import numpy as np

from brsr_tdoa.brsr_core import (
    NestedComplexQuantizer,
    ambiguity_period_samples,
    available_actions,
    build_default_model,
    build_likelihood_context,
    enumerate_action_outcomes,
    evaluate_dynamic_program,
    evaluate_one_step,
    initialize_state,
    posterior_statistics,
    run_dynamic_policy,
)


def _trial():
    model = build_default_model(tau_step=1.0)
    quantizer = NestedComplexQuantizer(max_bits=3, clip_abs=4.0)
    rng = np.random.default_rng(20260717)
    sigma2 = 0.1
    source = (
        rng.standard_normal(model.n_bins) + 1j * rng.standard_normal(model.n_bins)
    ) / math.sqrt(2.0)
    source *= np.sqrt(model.source_power)
    true_tau = float(model.tau_values[model.n_tau // 2 + 1])
    rho = model.rho_values[2]
    noise_a = (
        rng.standard_normal(model.n_bins) + 1j * rng.standard_normal(model.n_bins)
    ) * math.sqrt(sigma2 / 2.0)
    noise_b = (
        rng.standard_normal(model.n_bins) + 1j * rng.standard_normal(model.n_bins)
    ) * math.sqrt(sigma2 / 2.0)
    y_a = source + noise_a
    y_b = rho * source * np.exp(-1j * model.omega * true_tau) + noise_b
    fine_real, fine_imag = quantizer.encode_fine(y_b)
    context = build_likelihood_context(y_a, model, sigma2, sigma2)
    state = initialize_state(
        fine_real_codes=fine_real,
        fine_imag_codes=fine_imag,
        coarse_bits=2,
        quantizer=quantizer,
        context=context,
        model=model,
    )
    return model, quantizer, context, state, fine_real, fine_imag, true_tau


def test_default_model_has_realistic_time_to_delay_scale() -> None:
    model = build_default_model(tau_step=0.5)
    assert model.n_time == 1024
    assert model.n_time > 20 * model.n_tau
    assert ambiguity_period_samples(model.base_positions, model) > (
        model.tau_values[-1] - model.tau_values[0]
    )
    phases = np.sort(np.mod(np.angle(model.rho_values), 2.0 * np.pi))
    unique_phases = np.unique(np.round(phases, decimals=12))
    circular_gaps = np.diff(np.r_[unique_phases, unique_phases[0] + 2.0 * np.pi])
    np.testing.assert_allclose(circular_gaps, np.pi / 2.0, atol=1e-12)


def test_nested_quantizer_intervals_partition_probability() -> None:
    quantizer = NestedComplexQuantizer(max_bits=4, clip_abs=4.0)
    mean = np.asarray([0.3 - 0.2j, -1.0 + 0.7j])
    variance = np.asarray([0.5, 1.2])
    total = np.zeros(mean.shape, dtype=float)
    for real_code in range(4):
        for imag_code in range(4):
            total += quantizer.probability(
                mean, variance, real_code, imag_code, bits=2
            )
    np.testing.assert_allclose(total, 1.0, rtol=0.0, atol=2e-12)
    assert tuple(quantizer.child_codes(2, 2, 4)) == tuple(range(8, 12))
    fine_codes = np.arange(2**quantizer.max_bits)
    for coarse_bits in (1, 2, 3):
        coarse_codes = quantizer.coarsen(fine_codes, coarse_bits)
        expected = fine_codes >> (quantizer.max_bits - coarse_bits)
        np.testing.assert_array_equal(coarse_codes, expected)


def test_log_quantizer_likelihood_stays_finite_in_extreme_tails() -> None:
    quantizer = NestedComplexQuantizer(max_bits=4, clip_abs=4.0)
    mean = np.asarray([30.0 + 25.0j, -28.0 - 31.0j])
    variance = np.asarray([0.01, 0.02])
    log_probability = quantizer.log_probability(
        mean,
        variance,
        real_code=7,
        imag_code=8,
        bits=4,
    )
    assert np.all(np.isfinite(log_probability))
    assert np.all(log_probability < -700.0)


def test_initial_posterior_and_outcome_probabilities_are_normalized() -> None:
    model, quantizer, context, state, *_ = _trial()
    assert math.isclose(float(np.sum(state.posterior)), 1.0, abs_tol=1e-12)
    action = available_actions(
        state,
        coarse_bits=2,
        max_bits=3,
        feedback_bits=4,
        model=model,
    )[0]
    outcomes = enumerate_action_outcomes(state, action, quantizer, context)
    assert math.isclose(sum(item[2] for item in outcomes), 1.0, abs_tol=2e-10)


def test_public_b_scale_transforms_mean_and_variance_consistently() -> None:
    model = build_default_model(tau_step=1.0)
    y_a = np.linspace(0.1, 0.6, model.n_bins) * (1.0 + 0.5j)
    unscaled = build_likelihood_context(y_a, model, 0.4, 0.4)
    scaled = build_likelihood_context(
        y_a,
        model,
        0.4,
        0.4,
        b_public_scale=2.5,
    )
    np.testing.assert_allclose(scaled.mean_by_bin, unscaled.mean_by_bin / 2.5)
    np.testing.assert_allclose(
        scaled.variance_by_bin,
        unscaled.variance_by_bin / 2.5**2,
    )
    assert scaled.b_public_scale == 2.5


def test_high_snr_outcome_probabilities_are_normalized_without_probability_floor() -> None:
    model = build_default_model(tau_step=1.0)
    quantizer = NestedComplexQuantizer(max_bits=4, clip_abs=4.0)
    rng = np.random.default_rng(20260718)
    sigma2 = 10.0 ** (-15.0 / 10.0)
    source = (
        rng.standard_normal(model.n_bins) + 1j * rng.standard_normal(model.n_bins)
    ) * np.sqrt(model.source_power / 2.0)
    true_tau = float(model.tau_values[3])
    rho = model.rho_values[5]
    noise_a = (
        rng.standard_normal(model.n_bins) + 1j * rng.standard_normal(model.n_bins)
    ) * math.sqrt(sigma2 / 2.0)
    noise_b = (
        rng.standard_normal(model.n_bins) + 1j * rng.standard_normal(model.n_bins)
    ) * math.sqrt(sigma2 / 2.0)
    y_a = source + noise_a
    y_b = rho * source * np.exp(-1j * model.omega * true_tau) + noise_b
    fine_real, fine_imag = quantizer.encode_fine(y_b)
    context = build_likelihood_context(y_a, model, sigma2, sigma2)
    state = initialize_state(
        fine_real_codes=fine_real,
        fine_imag_codes=fine_imag,
        coarse_bits=2,
        quantizer=quantizer,
        context=context,
        model=model,
    )
    for action in available_actions(
        state,
        coarse_bits=2,
        max_bits=4,
        feedback_bits=4,
        model=model,
    ):
        outcomes = enumerate_action_outcomes(state, action, quantizer, context)
        assert math.isclose(sum(item[2] for item in outcomes), 1.0, abs_tol=2e-10)


def test_expected_posterior_risk_is_nonincreasing() -> None:
    model, quantizer, context, state, *_ = _trial()
    current_risk = posterior_statistics(state, model)[1]
    for action in available_actions(
        state,
        coarse_bits=2,
        max_bits=3,
        feedback_bits=4,
        model=model,
    ):
        expected = 0.0
        for real_code, imag_code, probability, posterior in enumerate_action_outcomes(
            state, action, quantizer, context
        ):
            child = state.__class__(
                posterior=posterior,
                levels=tuple(
                    action.target_bits if index == action.position else value
                    for index, value in enumerate(state.levels)
                ),
                real_codes=tuple(
                    real_code if index == action.position else value
                    for index, value in enumerate(state.real_codes)
                ),
                imag_codes=tuple(
                    imag_code if index == action.position else value
                    for index, value in enumerate(state.imag_codes)
                ),
            )
            expected += probability * posterior_statistics(child, model)[1]
        assert expected <= current_risk + 2e-10


def test_dynamic_program_value_is_not_worse_than_one_step() -> None:
    model, quantizer, context, state, *_ = _trial()
    arguments = dict(
        lambda_per_bit=0.01,
        stop_feedback_bits=4,
        coarse_bits=2,
        max_bits=3,
        action_feedback_bits=4,
        quantizer=quantizer,
        context=context,
        model=model,
    )
    one_step = evaluate_one_step(state, **arguments)
    dynamic = evaluate_dynamic_program(state, depth=2, **arguments)
    assert dynamic.value <= one_step.value + 1e-10


def test_policy_bit_ledger_is_exact() -> None:
    model, quantizer, context, state, fine_real, fine_imag, true_tau = _trial()
    result = run_dynamic_policy(
        method="greedy",
        initial_state=state,
        fine_real_codes=fine_real,
        fine_imag_codes=fine_imag,
        true_tau=true_tau,
        lambda_per_bit=0.0,
        max_decisions=2,
        dp_horizon=2,
        coarse_bits=2,
        max_bits=3,
        action_feedback_bits=4,
        stop_feedback_bits=4,
        quantizer=quantizer,
        context=context,
        model=model,
    )
    ledger = result.ledger
    assert ledger.base_payload_bits == len(model.base_positions) * 2 * 2
    assert ledger.total_bits == (
        ledger.base_payload_bits
        + ledger.refinement_payload_bits
        + ledger.action_feedback_bits
        + ledger.stop_feedback_bits
        + ledger.header_bits
        + ledger.scale_bits
    )
    assert np.isfinite(result.posterior_risk)
