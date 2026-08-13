from __future__ import annotations

import numpy as np
import pytest

from raptq_tdoa.core import (
    BayesianRAPTQModel,
    ModelConfig,
    QMCConfig,
    finite_window_delay_matrix,
    finite_window_rotation_certificate,
    posterior_total_variation,
)


def _context():
    model = BayesianRAPTQModel(ModelConfig())
    trial = model.generate_trials(1234, 1)[0]
    return model, model.build_context(trial, 0.0)


def test_continuous_phase_posterior_is_rotation_invariant() -> None:
    model, context = _context()
    base = model.posterior_full(context)
    rotated = model.posterior_full(
        context,
        context.y_b_normalized * np.exp(1j * 0.731),
    )
    assert posterior_total_variation(base, rotated) < 1e-10


def test_radius_projector_roundtrip_preserves_posterior() -> None:
    model, context = _context()
    radius, projector = model.projector(context.y_b_normalized)
    representative = model.representative_from_projector(radius, projector)
    assert posterior_total_variation(
        model.posterior_full(context),
        model.posterior_full(context, representative),
    ) < 1e-10
    new_radius, new_projector = model.projector(representative)
    assert abs(new_radius - radius) < 1e-12
    assert np.max(np.abs(new_projector - projector)) < 1e-12


def test_qmc_representations_return_normalized_posteriors() -> None:
    model, context = _context()
    config = QMCConfig(
        power=4,
        repeats=1,
        reference_power=5,
        reference_context_limit=1,
    )
    for representation in (
        "projector_only",
        "relative_phase",
        "relative_power",
        "sparse_phase",
        "sparse_amp_phase",
    ):
        posterior = model.posterior_representation(
            context,
            representation,
            config,
            sparse_coordinate_count=2,
            seed=8123,
        )
        assert np.isfinite(posterior).all()
        assert abs(float(np.sum(posterior)) - 1.0) < 1e-12


def test_relative_power_is_exactly_tau_independent() -> None:
    model, context = _context()
    config = QMCConfig(
        power=4,
        repeats=1,
        reference_power=5,
        reference_context_limit=1,
    )
    posterior = model.posterior_representation(
        context,
        "relative_power",
        config,
        sparse_coordinate_count=2,
        seed=111,
    )
    assert np.max(np.abs(posterior - model.tau_prior)) < 1e-15


def test_defensive_proposal_is_scoped_to_relative_phase() -> None:
    model, context = _context()
    config = QMCConfig(
        power=5,
        repeats=2,
        reference_power=6,
        reference_context_limit=1,
        proposal="defensive_observed",
    )
    posterior = model.posterior_representation(
        context,
        "relative_phase",
        config,
        sparse_coordinate_count=2,
        seed=222,
    )
    assert np.isfinite(posterior).all()
    assert abs(float(np.sum(posterior)) - 1.0) < 1e-12
    with pytest.raises(ValueError, match="only defined for relative_phase"):
        model.posterior_representation(
            context,
            "sparse_phase",
            config,
            sparse_coordinate_count=2,
            seed=333,
        )


def test_finite_window_operator_is_nonunitary_and_rotation_invariant() -> None:
    operator = finite_window_delay_matrix(12, 1.25)
    gram = operator.conj().T @ operator
    assert not np.allclose(gram, np.eye(12), atol=1e-6)
    certificate = finite_window_rotation_certificate(7781)
    assert certificate["finite_window_rotation_invariant"]
    assert certificate["finite_window_radius_is_delay_sensitive"]


def test_gain_phase_is_continuous_in_generator() -> None:
    model = BayesianRAPTQModel(ModelConfig())
    phases = np.asarray(
        [trial.true_gain_phase for trial in model.generate_trials(9912, 40)]
    )
    phase_grid = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    distance = np.min(np.abs(phases[:, None] - phase_grid[None, :]), axis=1)
    assert np.any(distance > 1e-3)
