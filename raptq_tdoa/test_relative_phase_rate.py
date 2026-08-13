from __future__ import annotations

from dataclasses import replace

import numpy as np

from raptq_tdoa.core import BayesianRAPTQModel, ModelConfig, QMCConfig
from raptq_tdoa.relative_phase_rate import (
    ContinuousPhaseCondMI24,
    QuantizedRelativePhase24,
    RelativePhaseCodeConfig,
    RelativePhaseQMCConfig,
)
from raptq_tdoa.run_relative_phase_rate import _continuous_stage_status, _gate


def _context():
    model = BayesianRAPTQModel(ModelConfig())
    trial = model.generate_trials(19, 1)[0]
    return model, model.build_context(trial, 5.0)


def test_payloads_are_exactly_equal_to_24_bits() -> None:
    model, _ = _context()
    condmi = ContinuousPhaseCondMI24(model, phase_nodes=16)
    relative = QuantizedRelativePhase24(model)
    assert condmi.total_payload_bits == relative.total_payload_bits == 24
    assert RelativePhaseCodeConfig().bits_by_relative_phase == (5, 5, 5, 5, 4)


def test_relative_phase_code_is_invariant_to_common_rotation() -> None:
    model, context = _context()
    encoder = QuantizedRelativePhase24(model)
    base = encoder.encode(context.y_b_normalized)
    rotated = encoder.encode(context.y_b_normalized * np.exp(1j * 0.731))
    assert np.array_equal(base, rotated)


def test_relative_phase_cells_contain_observed_phases() -> None:
    model, context = _context()
    encoder = QuantizedRelativePhase24(model)
    codes = encoder.encode(context.y_b_normalized)
    lower, upper = encoder.phase_bounds(codes)
    _, _, phase = model.observed_coordinates(context.y_b_normalized)
    assert np.all(phase[1:] >= lower)
    assert np.all(phase[1:] <= upper)


def test_posteriors_ignore_true_tau_and_remain_normalized() -> None:
    model, context = _context()
    altered = replace(context, true_tau=context.true_tau + 1.0)
    condmi = ContinuousPhaseCondMI24(model, phase_nodes=16)
    rate = QuantizedRelativePhase24(model)
    qmc = RelativePhaseQMCConfig(power=6, repeats=2)
    first = condmi.posterior(context, threshold=0.6)
    second = condmi.posterior(altered, threshold=0.6)
    assert np.allclose(first, second)
    first_rate = rate.posterior(context, qmc, seed=73)
    second_rate = rate.posterior(altered, qmc, seed=73)
    assert np.allclose(first_rate, second_rate)
    assert np.isclose(np.sum(first_rate), 1.0)


def test_condmi_continuous_phase_quadrature_converges_on_small_case() -> None:
    model, context = _context()
    coarse = ContinuousPhaseCondMI24(model, phase_nodes=32).posterior(context, 0.6)
    fine = ContinuousPhaseCondMI24(model, phase_nodes=64).posterior(context, 0.6)
    assert 0.5 * np.sum(np.abs(coarse - fine)) < 1e-5


def test_fine_phase_cells_approach_continuous_relative_phase() -> None:
    model = BayesianRAPTQModel(ModelConfig(selected_bins=(31, 94, 145)))
    context = model.build_context(model.generate_trials(22, 1)[0], 5.0)
    quantized = QuantizedRelativePhase24(
        model,
        RelativePhaseCodeConfig(bits_by_relative_phase=(12, 12)),
    ).posterior(
        context,
        RelativePhaseQMCConfig(power=11, repeats=4, proposal="model_radial"),
        seed=123,
    )
    continuous = model.posterior_representation(
        context,
        "relative_phase",
        QMCConfig(
            power=11,
            repeats=4,
            reference_power=12,
            reference_context_limit=1,
            proposal="model_radial",
        ),
        sparse_coordinate_count=1,
        seed=456,
    )
    assert 0.5 * np.sum(np.abs(quantized - continuous)) < 5e-4


def test_gate_respects_headroom_and_rate_effects() -> None:
    import pandas as pd

    mechanical = {"all_passed": True}
    numerical = {"all_numerical_risk_differences_within_guard": True}
    rows = pd.DataFrame(
        [
            {
                "candidate": "RelativePhase-Continuous",
                "baseline": "FullComplex",
                "metric": "posterior_risk",
                "ci95_low": -0.02,
                "ci95_high": 0.03,
            },
            {
                "candidate": "RelativePhase-Continuous",
                "baseline": "CondMI-Matched24",
                "metric": "posterior_risk",
                "ci95_low": 0.2,
                "ci95_high": 0.4,
            },
            {
                "candidate": "RAPTQ-RelativePhase24",
                "baseline": "CondMI-Matched24",
                "metric": "posterior_risk",
                "ci95_low": 0.1,
                "ci95_high": 0.3,
            },
            {
                "candidate": "RAPTQ-RelativePhase24",
                "baseline": "CondMI-Matched24",
                "metric": "squared_error",
                "ci95_low": 0.05,
                "ci95_high": 0.25,
            },
            {
                "candidate": "RAPTQ-RelativePhase24",
                "baseline": "CondMI-Matched24",
                "metric": "gross_error_gt_1",
                "ci95_low": 0.0,
                "ci95_high": 0.1,
            },
        ]
    )
    assert _gate("development", mechanical, False, numerical, rows) == (
        "GO_COMPONENT_RELATIVE_PHASE24"
    )
    rows.loc[1, "ci95_low"] = 0.01
    assert _gate("development", mechanical, False, numerical, rows) == (
        "HOLD_RELATIVE_PHASE_HEADROOM"
    )


def test_continuous_stage_rejects_data_processing_violation() -> None:
    information = {"ci95_low": 0.1, "ci95_high": 0.2}
    headroom = {"ci95_low": 0.2, "ci95_high": 0.3}
    assert _continuous_stage_status(information, headroom) == (
        "INVALID_DATA_PROCESSING_INEQUALITY"
    )
