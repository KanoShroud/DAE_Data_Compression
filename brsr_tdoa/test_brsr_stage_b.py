"""BRSR-TDOA 阶段 B 的波形、协议与公平性不变量测试。"""

from __future__ import annotations

import math

import numpy as np

from brsr_tdoa.brsr_core import NestedComplexQuantizer
from brsr_tdoa.brsr_stage_b import (
    StageBProtocol,
    ambiguity_period_samples,
    build_stage_b_model,
    evaluate_fixed,
    make_observation,
    run_policy,
)
from brsr_tdoa.brsr_waveform import (
    WaveformConfig,
    apply_circular_fractional_delay,
    apply_linear_fractional_delay,
    candidate_fft_values,
    design_candidate_bins,
    estimate_candidate_statistics,
    generate_structured_trial,
    observe_trial,
    raised_cosine_power_at_bins,
    root_raised_cosine_taps,
    rrc_positive_support_max_bin,
)


def _small_stage_b_trial():
    waveform = WaveformConfig()
    bins = design_candidate_bins(waveform)
    prior_trials = [
        generate_structured_trial(np.random.default_rng(seed), waveform)
        for seed in (20269001, 20269002, 20269003, 20269004)
    ]
    source_power, _, _ = estimate_candidate_statistics(prior_trials, bins)
    model = build_stage_b_model(
        n_time=waveform.n_time,
        bin_indices=bins,
        source_power=source_power,
        base_bins=waveform.base_bins,
        tau_min=waveform.tau_min,
        tau_max=waveform.tau_max,
        tau_step=2.0,
        rho_amplitudes=(0.70, 0.90),
        rho_phase_count=4,
    )
    protocol = StageBProtocol(
        rollout_samples=1,
        rollout_first_action_candidates=1,
    )
    quantizer = NestedComplexQuantizer(max_bits=8, clip_abs=4.0)
    trial = generate_structured_trial(np.random.default_rng(20269201), waveform)
    y_a, y_b, sigma2 = observe_trial(trial, snr_db=0.0)
    observation = make_observation(
        stage="test",
        seed=20269201,
        trial=0,
        snr_db=0.0,
        true_tau=trial.true_tau,
        true_rho=trial.true_rho,
        y_a_selected=candidate_fft_values(y_a, bins),
        y_b_selected=candidate_fft_values(y_b, bins),
        sigma2=sigma2,
        model=model,
        quantizer=quantizer,
    )
    return waveform, bins, model, protocol, quantizer, observation


def test_rrc_and_candidate_pool_are_physically_valid() -> None:
    config = WaveformConfig()
    taps = root_raised_cosine_taps(
        config.rrc_rolloff,
        config.rrc_span_symbols,
        config.samples_per_symbol,
    )
    assert taps.size == config.rrc_span_symbols * config.samples_per_symbol + 1
    assert math.isclose(float(np.sum(taps**2)), 1.0, abs_tol=1e-12)
    np.testing.assert_allclose(taps, taps[::-1], rtol=0.0, atol=1e-14)

    bins = design_candidate_bins(config)
    support = rrc_positive_support_max_bin(config)
    assert bins.size == config.candidate_count
    assert np.unique(bins).size == bins.size
    assert np.all((bins > 0) & (bins <= support))
    assert all(value in bins for value in config.base_bins)
    assert np.all(raised_cosine_power_at_bins(bins, config) > 0.0)
    assert config.n_time / (config.base_bins[1] - config.base_bins[0]) > (
        config.tau_max - config.tau_min
    )


def test_delay_sign_matches_frequency_phase_convention() -> None:
    n_time = 1024
    rng = np.random.default_rng(20260718)
    signal = rng.standard_normal(n_time) + 1j * rng.standard_normal(n_time)
    delay = 2.375
    circular = apply_circular_fractional_delay(signal, delay)
    spectrum = np.fft.fft(signal, norm="ortho")
    shifted = np.fft.fft(circular, norm="ortho")
    expected = spectrum * np.exp(
        -2j * np.pi * np.fft.fftfreq(n_time) * delay
    )
    np.testing.assert_allclose(shifted, expected, rtol=0.0, atol=2e-12)

    tone_bin = 93
    indices = np.arange(4096)
    tone = np.exp(2j * np.pi * tone_bin * indices / n_time)
    linearly_delayed = apply_linear_fractional_delay(tone, delay, half_len=32)
    central = slice(256, -256)
    observed_ratio = np.mean(linearly_delayed[central] / tone[central])
    expected_ratio = np.exp(-2j * np.pi * tone_bin * delay / n_time)
    phase_error = np.angle(observed_ratio / expected_ratio)
    assert abs(phase_error) < 2e-3
    assert abs(abs(observed_ratio) - 1.0) < 2e-3


def test_structured_trial_uses_guarded_linear_delay_and_shared_noise() -> None:
    config = WaveformConfig()
    trial = generate_structured_trial(np.random.default_rng(20269201), config)
    assert trial.source_a.shape == (config.n_time,)
    assert trial.source_b_unit_gain.shape == (config.n_time,)
    assert config.tau_min <= trial.true_tau <= config.tau_max
    assert 0.70 <= abs(trial.true_rho) <= 0.90
    assert 0.75 < float(np.mean(np.abs(trial.source_a) ** 2)) < 1.25

    low_a, low_b, low_sigma2 = observe_trial(trial, -5.0)
    high_a, high_b, high_sigma2 = observe_trial(trial, 10.0)
    assert low_sigma2 > high_sigma2
    clean_a_from_low = low_a - math.sqrt(low_sigma2) * trial.noise_a_standard
    clean_a_from_high = high_a - math.sqrt(high_sigma2) * trial.noise_a_standard
    clean_b_from_low = low_b - math.sqrt(low_sigma2) * trial.noise_b_standard
    clean_b_from_high = high_b - math.sqrt(high_sigma2) * trial.noise_b_standard
    np.testing.assert_allclose(clean_a_from_low, clean_a_from_high, atol=1e-12)
    np.testing.assert_allclose(clean_b_from_low, clean_b_from_high, atol=1e-12)


def test_stage_b_quantization_and_bit_ledger_are_exact() -> None:
    waveform, bins, model, protocol, quantizer, observation = _small_stage_b_trial()
    feedback = protocol.feedback_bits_for(bins.size)
    assert feedback == 5
    assert observation.fine_real_codes.shape == (bins.size,)
    assert observation.fine_imag_codes.shape == (bins.size,)
    assert observation.public_b_scale > 0.0

    high_penalty = run_policy(
        observation,
        method="greedy",
        lambda_per_bit=100.0,
        protocol=protocol,
        quantizer=quantizer,
        model=model,
        rng=np.random.default_rng(1),
    )
    assert high_penalty.stopped_by_policy
    assert high_penalty.total_bits == 2 * 2 * 4 + feedback

    zero_penalty = run_policy(
        observation,
        method="greedy",
        lambda_per_bit=0.0,
        protocol=protocol,
        quantizer=quantizer,
        model=model,
        rng=np.random.default_rng(2),
    )
    assert not zero_penalty.stopped_by_policy
    expected_hard_cap = 2 * 2 * 4 + 2 * (2 * (8 - 4) + feedback)
    assert zero_penalty.total_bits == expected_hard_cap
    assert sum(value > 0 for value in zero_penalty.selected_levels) >= 2

    rollout = run_policy(
        observation,
        method="rollout",
        lambda_per_bit=0.05,
        protocol=protocol,
        quantizer=quantizer,
        model=model,
        rng=np.random.default_rng(3),
    )
    assert 2 * 2 * 4 + feedback <= rollout.total_bits <= expected_hard_cap
    assert math.isfinite(rollout.estimate)
    assert math.isfinite(rollout.posterior_risk)

    base8 = tuple(
        8 if position in model.base_positions else 0
        for position in range(model.n_bins)
    )
    fixed = evaluate_fixed(
        observation,
        method="test-base8",
        levels=base8,
        quantizer=quantizer,
        model=model,
    )
    assert fixed.total_bits == 32
    assert ambiguity_period_samples(base8, model) > (
        waveform.tau_max - waveform.tau_min
    )


def test_prior_and_evaluation_seeds_produce_distinct_waveforms() -> None:
    config = WaveformConfig()
    prior = generate_structured_trial(np.random.default_rng(20269001), config)
    evaluation = generate_structured_trial(np.random.default_rng(20269201), config)
    assert not np.array_equal(prior.source_a, evaluation.source_a)
    assert not math.isclose(prior.true_tau, evaluation.true_tau, abs_tol=0.0)


def test_full_covariance_is_diagnostic_and_finite() -> None:
    config = WaveformConfig()
    bins = design_candidate_bins(config)
    trials = [
        generate_structured_trial(np.random.default_rng(seed), config)
        for seed in range(20269001, 20269009)
    ]
    power, covariance, diagnostics = estimate_candidate_statistics(trials, bins)
    assert power.shape == (bins.size,)
    assert covariance.shape == (bins.size, bins.size)
    assert np.all(np.isfinite(covariance))
    assert np.all(power > 0.0)
    assert 0.0 <= diagnostics["offdiag_frobenius_ratio"] <= 1.0
    assert 0.0 <= diagnostics["max_offdiag_coherence"] <= 1.0 + 1e-12
