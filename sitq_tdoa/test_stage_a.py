"""Mechanical and scientific-invariant tests for SITQ-TDOA Stage A."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sitq_tdoa.core import (
    ModelConfig,
    SITQModel,
    quantizer_probability_sum,
    search_threshold,
)
from sitq_tdoa.run_stage_a import (
    _embedded_checks,
    _gate_status,
    _post_evaluation_checks,
    build_experiment_config,
)


def _model_and_contexts() -> tuple[SITQModel, list]:
    model = SITQModel(ModelConfig())
    trials = model.generate_trials(123, 2)
    contexts = [model.build_context(trial, 0.0) for trial in trials]
    return model, contexts


def test_stage_a_budget_and_frequency_support() -> None:
    config = ModelConfig()
    config.validate()
    assert config.total_payload_bits == 24
    assert max(config.selected_bins) <= config.rrc_support_max_bin
    assert len(set(config.selected_bins)) == len(config.selected_bins)


def test_unknown_gain_ambiguity_uses_pairwise_bin_spacing() -> None:
    config = ModelConfig(selected_bins=(1, 65, 129, 193, 257))
    with pytest.raises(ValueError, match="ambiguous"):
        config.validate()


def test_quantizer_intervals_partition_probability() -> None:
    for mean in (-3.0, 0.0, 2.0):
        total = quantizer_probability_sum(mean, 0.7, 0.65)
        assert math.isclose(total, 1.0, abs_tol=1e-12)


def test_quantized_and_unquantized_posteriors_are_normalized() -> None:
    model, contexts = _model_and_contexts()
    for context in contexts:
        quantized = model.posterior_quantized(context, threshold=0.7)
        unquantized = model.posterior_unquantized(context)
        assert np.all(np.isfinite(quantized))
        assert np.all(np.isfinite(unquantized))
        assert math.isclose(float(np.sum(quantized)), 1.0, abs_tol=1e-10)
        assert math.isclose(float(np.sum(unquantized)), 1.0, abs_tol=1e-10)


def test_delay_phase_sign_matches_generation_and_likelihood() -> None:
    model = SITQModel(ModelConfig())
    trial = model.generate_trials(9981, 1)[0]
    context = model.build_context(trial, 60.0)
    posterior = model.posterior_unquantized(context)
    estimate = model.summarize_posterior(posterior, trial.true_tau).estimate
    assert abs(estimate - trial.true_tau) < 0.3


def test_generated_delay_uses_the_same_finite_prior_as_decoder() -> None:
    model = SITQModel(ModelConfig())
    for trial in model.generate_trials(20260803, 20):
        assert np.any(np.isclose(model.tau_values, trial.true_tau, atol=1e-12))


def test_design_and_evaluation_random_streams_are_disjoint() -> None:
    config = build_experiment_config("development")
    assert set(config.design_seeds).isdisjoint(config.evaluation_seeds)
    model = SITQModel(ModelConfig())
    checks = _embedded_checks(model, config)
    assert checks["all_passed"] is True


def test_threshold_search_uses_only_supplied_contexts() -> None:
    model, contexts = _model_and_contexts()
    result = search_threshold(
        model,
        contexts,
        candidates=(0.4, 0.7, 1.1),
        objective="bayes_risk",
    )
    assert result.threshold in {0.4, 0.7, 1.1}
    assert math.isfinite(result.objective_value)
    assert len(result.rows) == 3


def test_post_evaluation_budget_and_pairing_checks() -> None:
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "method": method,
                "seed": 1,
                "trial": trial,
                "snr_db": 0.0,
                "payload_bits": math.nan if method == "Unquantized-SelectedBins" else 24,
            }
            for trial in (0, 1)
            for method in ("MSEQuant-24bit", "Unquantized-SelectedBins")
        ]
    )
    checks = _post_evaluation_checks(rows, expected_contexts=2, payload_bits=24)
    assert checks["post_evaluation_all_passed"] is True


def test_gate_is_directional_and_does_not_use_an_arbitrary_percent() -> None:
    checks = {"all_passed": True}
    design = {"MSEQuant-24bit": 2.0, "CondMIQuant-24bit": 1.9}
    assert (
        _gate_status(
            checks,
            design,
            1.8,
            {
                "paired_mse_gain_ci95_low": 0.1,
                "paired_mse_gain_ci95_high": 0.5,
            },
        )
        == "GO_STAGE_A_MECHANISM"
    )
    assert (
        _gate_status(
            checks,
            design,
            1.8,
            {
                "paired_mse_gain_ci95_low": -0.1,
                "paired_mse_gain_ci95_high": 0.2,
            },
        )
        == "HOLD_STAGE_A_UNCERTAIN"
    )
    assert (
        _gate_status(
            checks,
            design,
            1.8,
            {
                "paired_mse_gain_ci95_low": -0.4,
                "paired_mse_gain_ci95_high": 0.0,
            },
        )
        == "NO_GO_CONFIG_SITQ_STAGE_A"
    )
    assert (
        _gate_status(
            checks,
            design,
            1.8,
            {
                "paired_mse_gain_ci95_low": 0.1,
                "paired_mse_gain_ci95_high": 0.5,
            },
            boundary_methods=("SITQ-BayesRisk-24bit",),
        )
        == "HOLD_THRESHOLD_SEARCH_BOUNDARY"
    )
