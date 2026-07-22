from __future__ import annotations

import unittest

import numpy as np

from standalone_frequency_alignment.tzp_dense import (
    dense_delay_estimates,
    direct_dense_delay_estimate,
    sample_original_grid,
    time_zero_padded_dense_spectrum,
)


class TimeZeroPaddedDenseSpectrumTests(unittest.TestCase):
    def test_original_grid_is_preserved_without_filling_missing_bins(self) -> None:
        rng = np.random.default_rng(20260721)
        n_fft = 32
        factor = 8
        selected = np.sort(rng.choice(n_fft, 9, replace=False))
        sparse = np.zeros(n_fft, dtype=complex)
        sparse[selected] = rng.normal(size=selected.size) + 1j * rng.normal(
            size=selected.size
        )

        dense = time_zero_padded_dense_spectrum(sparse, factor)
        mapped = sample_original_grid(dense, n_fft, factor)

        self.assertLess(float(np.max(np.abs(mapped - sparse))), 1e-12)
        missing = np.ones(n_fft, dtype=bool)
        missing[selected] = False
        self.assertLess(float(np.sum(np.abs(mapped[missing]) ** 2)), 1e-20)

    def test_dense_operator_rank_equals_observed_bin_count(self) -> None:
        n_fft = 24
        factor = 4
        selected = np.array([1, 4, 7, 13, 18, 21])
        basis = np.zeros((selected.size, n_fft), dtype=complex)
        basis[np.arange(selected.size), selected] = 1.0

        dense = time_zero_padded_dense_spectrum(basis, factor)
        operator = dense.T

        self.assertEqual(np.linalg.matrix_rank(operator, tol=1e-10), selected.size)
        gram = operator.conj().T @ operator
        expected = factor * np.eye(selected.size)
        self.assertLess(float(np.max(np.abs(gram - expected))), 1e-12)

    def test_zoomfft_delay_matches_direct_matrix_sum(self) -> None:
        rng = np.random.default_rng(20260722)
        n_fft = 32
        factor = 4
        sparse_a = np.zeros(n_fft, dtype=complex)
        sparse_b = np.zeros(n_fft, dtype=complex)
        mask_a = np.sort(rng.choice(n_fft, 10, replace=False))
        mask_b = np.sort(rng.choice(n_fft, 10, replace=False))
        sparse_a[mask_a] = rng.normal(size=10) + 1j * rng.normal(size=10)
        sparse_b[mask_b] = rng.normal(size=10) + 1j * rng.normal(size=10)
        dense_a = time_zero_padded_dense_spectrum(sparse_a, factor)
        dense_b = time_zero_padded_dense_spectrum(sparse_b, factor)
        cross = dense_b * np.conj(dense_a)

        estimate, _, _ = dense_delay_estimates(
            cross,
            np.array([10]),
            -6.0,
            6.0,
            0.02,
        )
        direct = direct_dense_delay_estimate(cross, -6.0, 6.0, 0.02)

        self.assertLess(abs(float(estimate[0]) - direct), 1e-12)


if __name__ == "__main__":
    unittest.main()
