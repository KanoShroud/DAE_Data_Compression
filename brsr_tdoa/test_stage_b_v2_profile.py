from __future__ import annotations

import numpy as np

from brsr_tdoa.brsr_core import NestedComplexQuantizer
from brsr_tdoa.brsr_stage_b import build_stage_b_model
from brsr_tdoa.diagnose_stage_b_v2_profile import (
    _posterior_summary,
    _quantized_profile_scores,
    config_for_mode,
)


def _tiny_model():
    bins = np.asarray([3, 8, 13, 19])
    return build_stage_b_model(
        n_time=64,
        bin_indices=bins,
        source_power=np.ones(4),
        base_bins=(3, 8),
        tau_min=-2.0,
        tau_max=2.0,
        tau_step=0.5,
        rho_amplitudes=(0.7, 0.9),
        rho_phase_count=4,
    )


def test_step1_seeds_are_outside_frozen_stage_b_pools() -> None:
    seeds = set(config_for_mode("research").seeds)
    assert not seeds.intersection(
        {20269001, 20269002, 20269003, 20269101, 20269102, 20269201, 20269202, 20269203}
    )


def test_quantized_profile_is_finite_and_prefers_true_delay() -> None:
    model = _tiny_model()
    quantizer = NestedComplexQuantizer(max_bits=8, clip_abs=4.0)
    true_tau = 1.0
    true_rho = 0.8 * np.exp(0.4j)
    steering = np.exp(-1j * model.omega[:, None] * model.tau_values[None, :])
    template = steering
    true_index = int(np.flatnonzero(np.isclose(model.tau_values, true_tau))[0])
    values = template[:, true_index] * true_rho
    fine_real, fine_imag = quantizer.encode_fine(values)
    scores = _quantized_profile_scores(
        template=template,
        source_variance=np.full(model.n_bins, 1e-5),
        sigma_b2=1e-5,
        scale=1.0,
        levels=(4, 4, 4, 4),
        fine_real=fine_real,
        fine_imag=fine_imag,
        quantizer=quantizer,
        model=model,
        iterations=3,
    )
    result = _posterior_summary(scores, model, true_tau)
    assert np.all(np.isfinite(scores))
    assert abs(result.estimate - true_tau) <= 0.5
    assert result.posterior_risk >= 0.0


def test_profile_rejects_invalid_template_shape() -> None:
    model = _tiny_model()
    quantizer = NestedComplexQuantizer(max_bits=8, clip_abs=4.0)
    with np.testing.assert_raises(ValueError):
        _quantized_profile_scores(
            template=np.zeros((3, model.n_tau), dtype=complex),
            source_variance=np.ones(model.n_bins),
            sigma_b2=1.0,
            scale=1.0,
            levels=(4, 4, 4, 4),
            fine_real=np.zeros(model.n_bins, dtype=int),
            fine_imag=np.zeros(model.n_bins, dtype=int),
            quantizer=quantizer,
            model=model,
            iterations=1,
        )
