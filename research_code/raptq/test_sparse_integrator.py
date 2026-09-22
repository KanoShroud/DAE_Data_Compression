from __future__ import annotations

# Resolve the repository independently of the entry's directory.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / 'runtime_paths.py').is_file())
if str(_LAYOUT_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT))
from runtime_paths import install_legacy_imports as _install_layout_imports
_install_layout_imports()


import numpy as np
import pytest

from research_code.raptq.core import BayesianRAPTQModel, ModelConfig
from research_code.raptq.sparse_integrator import (
    SparseRAPTQIntegrator,
    dirichlet_stick_breaking,
)


def _setup():
    model = BayesianRAPTQModel(ModelConfig())
    trial = model.generate_trials(4301, 1)[0]
    context = model.build_context(trial, 5.0)
    return model, context, SparseRAPTQIntegrator(model, 2)


def test_stick_breaking_uses_k_minus_one_coordinates() -> None:
    rng = np.random.default_rng(42)
    unit = rng.uniform(1e-8, 1.0 - 1e-8, size=(20000, 5))
    shares = dirichlet_stick_breaking(unit, np.ones(6))
    assert shares.shape == (20000, 6)
    assert np.all(shares > 0.0)
    assert np.max(np.abs(np.sum(shares, axis=1) - 1.0)) < 1e-12
    assert np.max(np.abs(np.mean(shares, axis=0) - 1.0 / 6.0)) < 0.006


def test_sparse_batches_preserve_only_registered_coordinates() -> None:
    model, context, integrator = _setup()
    _, observed_power, observed_phase = model.observed_coordinates(
        context.y_b_normalized
    )
    phase_batch = integrator.draw_batch(
        context,
        "sparse_phase",
        power=5,
        seed=81,
        proposal="model",
    )
    assert np.allclose(
        phase_batch.phases[:, integrator.retained],
        observed_phase[integrator.retained][None, :],
    )
    assert np.std(phase_batch.powers[:, integrator.retained[0]]) > 0.01

    amp_phase_batch = integrator.draw_batch(
        context,
        "sparse_amp_phase",
        power=5,
        seed=82,
        proposal="defensive_mixture",
    )
    assert np.allclose(
        amp_phase_batch.phases[:, integrator.retained],
        observed_phase[integrator.retained][None, :],
    )
    assert np.allclose(
        amp_phase_batch.powers[:, integrator.retained],
        observed_power[integrator.retained][None, :],
    )
    assert np.allclose(np.sum(amp_phase_batch.powers, axis=1), 1.0)
    assert np.any(amp_phase_batch.proposal_component)
    assert np.any(~amp_phase_batch.proposal_component)


@pytest.mark.parametrize("representation", ["sparse_phase", "sparse_amp_phase"])
@pytest.mark.parametrize("proposal", ["model", "defensive_mixture"])
def test_sparse_integrator_returns_finite_normalized_posterior(
    representation: str,
    proposal: str,
) -> None:
    _, context, integrator = _setup()
    result = integrator.integrate(
        context,
        representation,
        power=5,
        repeats=2,
        seed=503,
        proposal=proposal,
    )
    assert np.isfinite(result.posterior).all()
    assert abs(float(np.sum(result.posterior)) - 1.0) < 1e-12
    assert 0.0 < result.max_integrand_weight_fraction <= 1.0
    assert result.min_integrand_effective_sample_size >= 1.0


def test_defensive_mixture_weight_is_frozen() -> None:
    model = BayesianRAPTQModel(ModelConfig())
    with pytest.raises(ValueError, match="fixed at 0.5"):
        SparseRAPTQIntegrator(model, 2, defensive_mixture_weight=0.6)
