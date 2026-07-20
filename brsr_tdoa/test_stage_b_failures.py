"""Stage B 失败机理诊断的轻量单元测试。"""

from __future__ import annotations

import math

import numpy as np

from brsr_tdoa.brsr_stage_b import build_stage_b_model
from brsr_tdoa.brsr_waveform import (
    WaveformConfig,
    design_candidate_bins,
    generate_structured_trial,
    raised_cosine_power_at_bins,
)
from brsr_tdoa.diagnose_stage_b_failures import (
    _candidate_set_metrics,
    _model_mismatch_rows,
    _reachable_level_configs,
)


def _small_model():
    waveform = WaveformConfig()
    bins = design_candidate_bins(waveform)
    power = np.maximum(raised_cosine_power_at_bins(bins, waveform), 1e-6)
    model = build_stage_b_model(
        n_time=waveform.n_time,
        bin_indices=bins,
        source_power=power,
        base_bins=waveform.base_bins,
        tau_min=waveform.tau_min,
        tau_max=waveform.tau_max,
        tau_step=1.0,
        rho_amplitudes=(0.70, 0.90),
        rho_phase_count=4,
    )
    return waveform, bins, model


def test_finite_window_diagnostic_has_exact_circular_control() -> None:
    waveform, bins, _ = _small_model()
    trial = generate_structured_trial(np.random.default_rng(2026072001), waveform)
    summary, by_bin = _model_mismatch_rows(
        {(2026072001, 0): trial},
        waveform=waveform,
        bins=bins,
    )
    assert len(summary) == 1
    assert len(by_bin) == bins.size
    assert float(summary.iloc[0]["circular_control_nmse"]) < 1e-24
    assert float(summary.iloc[0]["linear_model_nmse"]) >= 0.0
    assert float(summary.iloc[0]["profiled_scalar_nmse"]) >= 0.0


def test_reachable_configs_preserve_two_decision_protocol() -> None:
    _, _, model = _small_model()
    configs = _reachable_level_configs(
        model,
        coarse_bits=4,
        max_bits=8,
        max_decisions=2,
    )
    assert set(configs.values()) == {0, 1, 2}
    for levels, depth in configs.items():
        assert all(value in (0, 4, 8) for value in levels)
        payload_increment = 2 * sum(levels) - 16
        assert payload_increment == depth * 8


def test_candidate_audit_uses_unknown_gain_schur_projection() -> None:
    _, _, model = _small_model()
    base = tuple(4 if index in model.base_positions else 0 for index in range(model.n_bins))
    all4 = (4,) * model.n_bins
    base_metric = _candidate_set_metrics("base", base, model=model, snr_db=None)
    all_metric = _candidate_set_metrics("all", all4, model=model, snr_db=None)
    base_information = float(
        base_metric["unknown_gain_projected_information_proxy"]
    )
    all_information = float(
        all_metric["unknown_gain_projected_information_proxy"]
    )
    assert base_information > 0.0
    assert all_information > base_information
    assert math.isfinite(float(all_metric["max_ambiguity_abs_ge_4"]))
    distance = float(all_metric["min_projected_distance_ge_4"])
    assert 0.0 <= distance <= 1.0
