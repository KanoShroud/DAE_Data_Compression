import numpy as np
from scipy import signal


class DirectDFTTDOAEstimator:
    """
    Estimate pairwise TDOA directly from selected DFT coefficients.

    This is a task-level baseline: it respects the same complex-bin budget as the
    reconstruction DFT baseline, but it does not claim to reconstruct a waveform.
    The estimated pairwise delays still go through the same all-pair WLS
    localization solver used by Raw/DAE/DFT/Hadamard/PCA.
    """

    def __init__(self, selected_bins, signal_len=1024, label="DFT-Direct",
                 phat=False):
        self.selected_bins = np.asarray(selected_bins, dtype=int)
        self.signal_len = int(signal_len)
        self.label = str(label)
        self.phat = bool(phat)
        if self.selected_bins.ndim != 1 or self.selected_bins.size < 1:
            raise ValueError("DirectDFTTDOAEstimator needs at least one DFT bin")
        if np.any(self.selected_bins < 0) or np.any(self.selected_bins >= self.signal_len):
            raise ValueError("selected_bins out of range")

        freqs = np.fft.fftshift(np.fft.fftfreq(self.signal_len))
        self.freqs = freqs[self.selected_bins]
        self.lags = signal.correlation_lags(
            self.signal_len, self.signal_len, mode="same"
        ).astype(float)
        self._steering = np.exp(
            2j * np.pi * np.outer(self.lags, self.freqs)
        ).astype(np.complex64)

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True):
        spec_i = np.fft.fftshift(np.fft.fft(sig_i, norm="ortho"))
        spec_j = np.fft.fftshift(np.fft.fft(sig_j, norm="ortho"))
        cross = spec_i[self.selected_bins] * np.conj(spec_j[self.selected_bins])
        if self.phat:
            cross = cross / (np.abs(cross) + 1e-12)

        score = np.abs(self._steering @ cross)
        idx = int(np.argmax(score))
        lag = float(self.lags[idx])
        if sub_sample and 0 < idx < score.size - 1:
            y0, y1, y2 = score[idx - 1], score[idx], score[idx + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                lag += float(np.clip(delta, -0.5, 0.5))

        if not return_quality:
            return lag

        peak = float(score[idx])
        mask = np.ones_like(score, dtype=bool)
        lo = max(0, idx - 2)
        hi = min(score.size, idx + 3)
        mask[lo:hi] = False
        sidelobe = float(np.max(score[mask])) if np.any(mask) else 0.0
        sidelobe_ratio = sidelobe / (peak + 1e-12)
        weight = float(np.clip(1.0 / (sidelobe_ratio + 1e-3), 0.05, 20.0))
        return lag, weight, peak, sidelobe_ratio

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True):
        abs_err = []
        weights = []
        sidelobes = []
        for bi in range(batch_np.shape[0]):
            los_idx = np.where(meta["los"][bi])[0] if use_los_only else np.arange(batch_np.shape[1])
            los_idx = [int(v) for v in los_idx]
            for a in range(len(los_idx)):
                for b in range(a + 1, len(los_idx)):
                    ui, uj = los_idx[a], los_idx[b]
                    sig_i = batch_np[bi, ui, 0, :] + 1j * batch_np[bi, ui, 1, :]
                    sig_j = batch_np[bi, uj, 0, :] + 1j * batch_np[bi, uj, 1, :]
                    tau, weight, _, side = self.estimate_pair(
                        sig_i, sig_j, sub_sample=sub_sample, return_quality=True
                    )
                    true_tau = (
                        meta["distances"][bi, ui] - meta["distances"][bi, uj]
                    ) / (c / fs)
                    abs_err.append(float(abs(tau - true_tau)))
                    weights.append(float(weight))
                    sidelobes.append(float(side))

        arr = np.asarray(abs_err, dtype=float)
        warr = np.asarray(weights, dtype=float)
        if arr.size == 0:
            return {}
        weighted = (
            float(np.sum(warr * arr) / (np.sum(warr) + 1e-12))
            if np.sum(warr) > 0 else float("nan")
        )
        return {
            "direct_mean_abs_tdoa_error_samples": float(np.mean(arr)),
            "direct_weighted_abs_tdoa_error_samples": weighted,
            "direct_median_abs_tdoa_error_samples": float(np.median(arr)),
            "direct_within_1_sample_rate": float(np.mean(arr <= 1.0)),
            "direct_within_2_sample_rate": float(np.mean(arr <= 2.0)),
            "direct_mean_sidelobe_ratio": float(np.mean(sidelobes)),
        }
