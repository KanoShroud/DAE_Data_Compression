from __future__ import annotations

import numpy as np
import pandas as pd

from raptq_tdoa.core import BayesianRAPTQModel, ModelConfig
from raptq_tdoa.recompute_stage0_sparse_amp_phase import (
    _cluster_effects,
    _consensus_posterior,
    _max_pairwise_tv,
    _numerical_summary,
    _trajectory_effects,
    build_config,
)


def test_consensus_uses_equal_likelihood_pooling() -> None:
    model = BayesianRAPTQModel(ModelConfig())
    first = np.log(np.linspace(1.0, 2.0, model.tau_values.size))[None, :]
    second = np.log(np.linspace(2.0, 1.0, model.tau_values.size))[None, :]
    posterior = _consensus_posterior(model, [first, second])
    expected = model._posterior_from_log_likelihood(
        np.log((np.exp(first[0]) + np.exp(second[0])) / 2.0)
    )
    assert np.allclose(posterior, expected)


def test_max_pairwise_tv_checks_every_component() -> None:
    first = np.asarray([0.7, 0.3])
    second = np.asarray([0.6, 0.4])
    third = np.asarray([0.1, 0.9])
    assert np.isclose(_max_pairwise_tv([first, second, third]), 0.6)


def test_frozen_development_config_matches_stage0_manifest() -> None:
    manifest = {
        "audit_config": {
            "seeds": [4301, 4302],
            "snr_db_values": [-10, -5, 0, 5, 10, 15],
            "trials_per_seed": 3,
            "sparse_coordinate_count": 2,
        }
    }
    config = build_config("development", manifest)
    assert config.qmc_power == 16
    assert config.qmc_repeats == 8
    assert config.seed_families == ("family_a", "family_b")
    assert config.proposals == ("model", "defensive_mixture")


def test_cluster_effects_use_trajectory_as_statistical_unit() -> None:
    rows = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "trial": [0, 0, 0, 0],
            "snr_db": [5.0, 10.0, 5.0, 10.0],
            "consensus_risk_excess_vs_full": [1.0, 3.0, 2.0, 4.0],
            "consensus_squared_error": [2.0, 4.0, 3.0, 5.0],
            "full_squared_error": [1.0, 1.0, 1.0, 1.0],
        }
    )
    trajectory = _trajectory_effects(rows)
    transition = trajectory[trajectory["scope"] == "transition_5_10_db"]
    assert transition.shape[0] == 2
    assert np.allclose(transition["mean_risk_excess_samples2"], [2.0, 3.0])
    effects = _cluster_effects(trajectory, repetitions=50, seed=42)
    transition_effect = effects[effects["scope"] == "transition_5_10_db"].iloc[0]
    assert transition_effect["n_trajectory_clusters"] == 2
    assert np.isclose(
        transition_effect["mean_posterior_risk_excess_samples2"],
        2.5,
    )


def test_numerical_summary_keeps_maxima_and_gate_counts() -> None:
    rows = pd.DataFrame(
        {
            "snr_db": [-5.0, -5.0, 10.0],
            "component_risk_span": [0.01, 0.07, 0.02],
            "max_within_family_proposal_risk_difference": [0.01, 0.04, 0.02],
            "max_within_proposal_seed_family_risk_difference": [0.01, 0.07, 0.01],
            "max_pairwise_posterior_tv": [0.02, 0.10, 0.03],
            "component_estimate_span": [0.1, 0.4, 0.2],
            "max_repeat_risk_sd": [0.01, 0.05, 0.02],
            "max_integrand_weight_fraction": [0.2, 0.8, 0.4],
            "min_integrand_effective_sample_size": [10.0, 2.0, 5.0],
            "numerical_gate_passed": [True, False, True],
        }
    )
    summary = _numerical_summary(rows)
    overall = summary[summary["scope"] == "all_snr"].iloc[0]
    assert np.isclose(overall["max_four_component_risk_span_samples2"], 0.07)
    assert overall["contexts_passing_numerical_gate"] == 2
    assert not bool(overall["all_contexts_passed_numerical_gate"])
