"""独立验证路线的理论恒等式和选频算法测试。"""

from __future__ import annotations

import unittest

import numpy as np

from standalone_frequency_alignment.alignment_core import (
    complete_spectrum,
    estimate_continuous_delay,
    interpolation_operator_rank,
    shifted_frequencies,
)
from standalone_frequency_alignment.zhai_selection import (
    crlb_information_objective,
    select_zhai_thesis,
)


class AlignmentCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260714)
        self.n_fft = 32
        self.indices = np.sort(self.rng.choice(self.n_fft, 8, replace=False))
        self.sparse = np.zeros(self.n_fft, dtype=complex)
        self.sparse[self.indices] = self.rng.normal(size=8) + 1j * self.rng.normal(size=8)

    def test_ideal_periodic_equals_zero_fill_on_original_grid(self) -> None:
        zero = complete_spectrum(self.sparse, self.indices, "zero_fill", 4)
        ideal = complete_spectrum(self.sparse, self.indices, "ideal_periodic", 4)
        np.testing.assert_allclose(ideal.spectrum, zero.spectrum, atol=1e-12, rtol=1e-12)
        self.assertLessEqual(ideal.generated_missing_energy, 1e-20)

    def test_all_circular_interpolators_do_not_create_missing_dft_bins(self) -> None:
        for method in ("polyphase_fir", "linear_periodic", "cubic_periodic"):
            result = complete_spectrum(self.sparse, self.indices, method, 4)
            missing = np.ones(self.n_fft, dtype=bool)
            missing[self.indices] = False
            self.assertLess(float(np.max(np.abs(result.spectrum[missing]))), 1e-10)

    def test_operator_rank_does_not_exceed_observation_count(self) -> None:
        rank = interpolation_operator_rank(
            self.n_fft,
            self.indices,
            "ideal_periodic",
            4,
        )
        self.assertLessEqual(rank, self.indices.size)

    def test_continuous_delay_sign_and_fractional_accuracy(self) -> None:
        frequencies = shifted_frequencies(self.n_fft)
        reference = self.rng.normal(size=self.n_fft) + 1j * self.rng.normal(size=self.n_fft)
        true_delay = 2.35
        delayed = reference * np.exp(-2j * np.pi * frequencies * true_delay)
        estimate = estimate_continuous_delay(
            delayed,
            reference,
            np.ones(self.n_fft, dtype=bool),
            5.0,
            0.005,
        )
        self.assertAlmostEqual(estimate.delay_samples, true_delay, places=3)

    def test_disjoint_masks_remain_invalid_after_ideal_interpolation(self) -> None:
        indices_a = np.arange(0, 8)
        indices_b = np.arange(8, 16)
        sparse_a = np.zeros(self.n_fft, dtype=complex)
        sparse_b = np.zeros(self.n_fft, dtype=complex)
        sparse_a[indices_a] = 1.0 + 0.5j
        sparse_b[indices_b] = 0.8 - 0.2j
        completed_a = complete_spectrum(sparse_a, indices_a, "ideal_periodic", 4)
        completed_b = complete_spectrum(sparse_b, indices_b, "ideal_periodic", 4)
        estimate = estimate_continuous_delay(
            completed_b.spectrum,
            completed_a.spectrum,
            np.ones(self.n_fft, dtype=bool),
            5.0,
            0.01,
        )
        self.assertTrue(np.isnan(estimate.delay_samples))

    def test_one_common_frequency_is_not_enough_for_delay(self) -> None:
        indices_a = np.arange(0, 8)
        indices_b = np.arange(7, 15)
        sparse_a = np.zeros(self.n_fft, dtype=complex)
        sparse_b = np.zeros(self.n_fft, dtype=complex)
        sparse_a[indices_a] = 1.0 + 0.5j
        sparse_b[indices_b] = 0.8 - 0.2j
        completed_a = complete_spectrum(sparse_a, indices_a, "ideal_periodic", 4)
        completed_b = complete_spectrum(sparse_b, indices_b, "ideal_periodic", 4)
        estimate = estimate_continuous_delay(
            completed_b.spectrum,
            completed_a.spectrum,
            np.ones(self.n_fft, dtype=bool),
            5.0,
            0.01,
        )
        self.assertTrue(np.isnan(estimate.delay_samples))

    def test_zhai_exchange_never_reduces_objective(self) -> None:
        power = np.abs(self.rng.normal(size=48)) + 0.01
        frequencies = shifted_frequencies(power.size)
        result = select_zhai_thesis(power, 10, frequencies)
        self.assertGreaterEqual(result.objective, result.initial_objective)
        self.assertEqual(np.unique(result.indices).size, 10)
        self.assertAlmostEqual(
            result.objective,
            crlb_information_objective(result.indices, power, frequencies),
        )


if __name__ == "__main__":
    unittest.main()
