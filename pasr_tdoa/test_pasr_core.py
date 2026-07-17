"""Deterministic tests for the standalone PASR-TDOA core."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pasr_tdoa.pasr_core import (
    build_spectral_model,
    estimate_subset,
    profiled_cost_grid,
    quantize_complex_spectrum,
    run_active_refinement,
)
from pasr_tdoa.run_pasr_tdoa import (
    _evaluate_static_plan,
    _same_bit_static_plan,
    _same_risk_static_plan,
)


def _tau_grid() -> np.ndarray:
    return np.linspace(-4.0, 4.0, 257)


def test_profiled_cost_recovers_noiseless_delay() -> None:
    model = build_spectral_model()
    true_tau = 1.375
    positions = (0, 3, 14, 15)
    y_a = model.source.copy()
    y_b = model.rho * model.source * np.exp(-1j * model.omega * true_tau)
    result = estimate_subset(y_a, y_b, positions, model, _tau_grid(), 1.0, 1.0)
    assert abs(result.estimate - true_tau) <= 1.0 / 32.0


def test_profiled_cost_is_invariant_to_selected_order() -> None:
    model = build_spectral_model()
    rng = np.random.default_rng(7)
    y_a = rng.normal(size=16) + 1j * rng.normal(size=16)
    y_b = rng.normal(size=16) + 1j * rng.normal(size=16)
    positions = (0, 3, 14, 15)
    forward = estimate_subset(y_a, y_b, positions, model, _tau_grid(), 0.7, 1.3)
    reverse = estimate_subset(
        y_a, y_b, tuple(reversed(positions)), model, _tau_grid(), 0.7, 1.3
    )
    np.testing.assert_allclose(forward.cost, reverse.cost, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(forward.posterior, reverse.posterior, rtol=0.0, atol=1e-12)


def test_quantizer_is_finite_and_reports_complex_variance() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=16) + 1j * rng.normal(size=16)
    result = quantize_complex_spectrum(values, 6, 8, rng)
    assert result.values.shape == values.shape
    assert np.all(np.isfinite(result.values))
    assert result.step > 0.0
    assert math.isclose(result.complex_error_variance, result.step**2 / 6.0)
    assert 0.0 <= result.clipping_rate <= 1.0


def test_active_ledger_counts_every_communication_component() -> None:
    model = build_spectral_model()
    true_tau = 0.625
    y_a = model.source.copy()
    y_b = model.rho * model.source * np.exp(-1j * model.omega * true_tau)
    quantized = quantize_complex_spectrum(y_b, 6, 8, np.random.default_rng(13))
    result = run_active_refinement(
        y_a=y_a,
        quantized_b=quantized,
        base_positions=(7, 8),
        model=model,
        tau_grid=_tau_grid(),
        sigma_a2=1e-3,
        sigma_b2=1e-3,
        qbits=6,
        scale_bits=8,
        max_bins=4,
        confidence_threshold=0.999999,
        variance_target=1e-12,
        strategy="fixed",
        rng=np.random.default_rng(17),
        fixed_order=(0, 15),
    )
    assert len(result.selected_positions) == 4
    assert result.ledger.coefficient_bits == 4 * 2 * 6
    assert result.ledger.scale_bits == 8
    assert result.ledger.action_feedback_bits == 0
    assert result.ledger.decision_feedback_bits == 2
    assert result.ledger.total_bits == (
        result.ledger.coefficient_bits
        + result.ledger.scale_bits
        + result.ledger.action_feedback_bits
        + result.ledger.decision_feedback_bits
    )


def test_always_four_ablation_does_not_charge_stop_decisions() -> None:
    model = build_spectral_model()
    true_tau = 0.625
    y_a = model.source.copy()
    y_b = model.rho * model.source * np.exp(-1j * model.omega * true_tau)
    quantized = quantize_complex_spectrum(y_b, 6, 8, np.random.default_rng(29))
    result = run_active_refinement(
        y_a=y_a,
        quantized_b=quantized,
        base_positions=(7, 8),
        model=model,
        tau_grid=_tau_grid(),
        sigma_a2=1e-3,
        sigma_b2=1e-3,
        qbits=6,
        scale_bits=8,
        max_bins=4,
        confidence_threshold=0.95,
        variance_target=0.04,
        strategy="posterior",
        rng=np.random.default_rng(31),
        allow_early_stop=False,
    )
    assert len(result.selected_positions) == 4
    assert result.ledger.decision_feedback_bits == 0
    assert result.ledger.action_feedback_bits > 0


def test_boundary_posterior_does_not_trigger_false_confident_stop() -> None:
    model = build_spectral_model()
    true_tau = 4.0
    y_a = model.source.copy()
    y_b = model.rho * model.source * np.exp(-1j * model.omega * true_tau)
    quantized = quantize_complex_spectrum(y_b, 8, 8, np.random.default_rng(19))
    result = run_active_refinement(
        y_a=y_a,
        quantized_b=quantized,
        base_positions=(7, 8),
        model=model,
        tau_grid=_tau_grid(),
        sigma_a2=1e-6,
        sigma_b2=1e-6,
        qbits=8,
        scale_bits=8,
        max_bins=3,
        confidence_threshold=0.1,
        variance_target=100.0,
        strategy="fixed",
        rng=np.random.default_rng(23),
        fixed_order=(0,),
    )
    assert len(result.selected_positions) == 3
    assert result.stop_reason == "max_bins"


def test_grid_profiled_cost_shape() -> None:
    model = build_spectral_model()
    positions = np.asarray((0, 3, 14, 15), dtype=int)
    cost = profiled_cost_grid(
        model.source[positions],
        model.source[positions],
        model.omega[positions],
        _tau_grid(),
        1.0,
        1.0,
    )
    assert cost.shape == _tau_grid().shape
    assert np.all(cost >= 0.0)


def test_development_frozen_static_plans_use_expected_convex_combinations() -> None:
    development = pd.DataFrame(
        [
            {
                "method": "Static-A",
                "qbits": 4,
                "mean_total_bits": 10.0,
                "max_total_bits": 10,
                "rmse_samples": 3.0,
            },
            {
                "method": "Static-B",
                "qbits": 8,
                "mean_total_bits": 30.0,
                "max_total_bits": 30,
                "rmse_samples": 1.0,
            },
            {
                "method": "Static-C",
                "qbits": 6,
                "mean_total_bits": 20.0,
                "max_total_bits": 20,
                "rmse_samples": 2.5,
            },
        ]
    )
    same_bit = _same_bit_static_plan(development, 20.0)
    assert same_bit is not None
    assert same_bit["left_method"] == "Static-A"
    assert same_bit["right_method"] == "Static-B"
    assert math.isclose(float(same_bit["right_weight"]), 0.5)
    assert math.isclose(float(same_bit["development_expected_mse"]), 5.0)

    same_risk = _same_risk_static_plan(development, 5.0)
    assert same_risk is not None
    assert math.isclose(float(same_risk["development_expected_bits"]), 20.0)
    assert math.isclose(float(same_risk["development_expected_mse"]), 5.0)


def test_frozen_static_plan_is_evaluated_without_locked_reselection() -> None:
    plan = {
        "left_method": "Static-A",
        "left_qbits": 4,
        "right_method": "Static-B",
        "right_qbits": 8,
        "right_weight": 0.25,
        "max_total_bits": 30,
    }
    locked = pd.DataFrame(
        [
            {
                "stage": "locked",
                "seed": 1,
                "snr_db": 0.0,
                "trial": 0,
                "method": "Static-A",
                "qbits": 4,
                "squared_error_samples2": 9.0,
                "abs_error_samples": 3.0,
                "transmitted_bits": 10,
            },
            {
                "stage": "locked",
                "seed": 1,
                "snr_db": 0.0,
                "trial": 0,
                "method": "Static-B",
                "qbits": 8,
                "squared_error_samples2": 1.0,
                "abs_error_samples": 1.0,
                "transmitted_bits": 30,
            },
        ]
    )
    result = _evaluate_static_plan(locked, "locked", plan)
    assert math.isclose(float(result["mse_samples2"]), 7.0)
    assert math.isclose(float(result["mean_total_bits"]), 15.0)
    assert math.isclose(float(result["gross_error_rate_gt_1"]), 0.75)
