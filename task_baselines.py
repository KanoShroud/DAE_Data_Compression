import numpy as np
from scipy import signal


def _lag_mask(lags, lag_limit_samples=None):
    if lag_limit_samples is None:
        return np.ones_like(lags, dtype=bool)
    return np.abs(lags) <= float(lag_limit_samples)


def _peak_from_score(lags, score, search_mask, sub_sample=True):
    search_idx = np.where(search_mask)[0]
    if search_idx.size == 0:
        search_idx = np.arange(score.size)
        search_mask = np.ones_like(score, dtype=bool)
    idx = int(search_idx[np.argmax(score[search_idx])])
    lag = float(lags[idx])
    if (sub_sample and 0 < idx < score.size - 1
            and search_mask[idx - 1] and search_mask[idx + 1]):
        y0, y1, y2 = float(score[idx - 1]), float(score[idx]), float(score[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            lag += float(np.clip(delta, -0.5, 0.5))
    return idx, lag, search_mask


def _quality_from_score(score, peak_idx, search_mask):
    # Shift for quality statistics only; the peak decision still uses the
    # original objective. This keeps real-valued ML scores numerically stable.
    score_q = np.asarray(score, dtype=float).copy()
    if np.any(search_mask):
        score_q -= float(np.min(score_q[search_mask]))
    peak = float(score_q[peak_idx])
    mask = search_mask.copy()
    lo = max(0, int(peak_idx) - 2)
    hi = min(score_q.size, int(peak_idx) + 3)
    mask[lo:hi] = False
    sidelobe = float(np.max(score_q[mask])) if np.any(mask) else 0.0
    sidelobe_ratio = sidelobe / (peak + 1e-12)
    weight = float(np.clip(1.0 / (sidelobe_ratio + 1e-3), 0.05, 20.0))
    return weight, peak, sidelobe_ratio


def _diagnostics_from_estimator(estimator, batch_np, meta, fs, c,
                                use_los_only=True, sub_sample=True,
                                lag_limit_samples=None):
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
                tau, weight, _, side = estimator.estimate_pair(
                    sig_i, sig_j, sub_sample=sub_sample, return_quality=True,
                    lag_limit_samples=lag_limit_samples
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
        "direct_lag_limit_samples": (
            float(lag_limit_samples) if lag_limit_samples is not None else float("nan")
        ),
    }


class DirectDFTTDOAEstimator:
    """
    Estimate pairwise TDOA directly from selected DFT coefficients.

    This is a task-level baseline: it respects the same complex-bin budget as a
    partial-Fourier compression method, but it does not reconstruct each sensor
    waveform by zero-filling missing bins. It forms the selected-bin
    cross-spectrum X_i(f) conj(X_j(f)) and searches delay with the Fourier
    shift steering vector. Equivalently, it estimates the peak of a partial
    inverse cross-spectrum, which follows the cross-correlation theorem
    R_ij(tau) = IFFT{X_i(f) conj(X_j(f))}. The estimated pairwise delays still
    go through the same all-pair WLS localization solver used by Raw/DAE,
    Hadamard, PCA, and waveform-reconstruction DFT ablations.
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

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        spec_i = np.fft.fftshift(np.fft.fft(sig_i, norm="ortho"))
        spec_j = np.fft.fftshift(np.fft.fft(sig_j, norm="ortho"))
        cross = spec_i[self.selected_bins] * np.conj(spec_j[self.selected_bins])
        if self.phat:
            cross = cross / (np.abs(cross) + 1e-12)

        score = np.abs(self._steering @ cross)
        search_mask = self._lag_mask(lag_limit_samples)
        idx, lag, search_mask = _peak_from_score(
            self.lags, score, search_mask, sub_sample=sub_sample
        )

        if not return_quality:
            return lag

        weight, peak, sidelobe_ratio = _quality_from_score(score, idx, search_mask)
        return lag, weight, peak, sidelobe_ratio

    def _lag_mask(self, lag_limit_samples=None):
        return _lag_mask(self.lags, lag_limit_samples)

    def alias_diagnostics(self, lag_limit_samples=None, threshold=0.8):
        if self.selected_bins.size <= 1:
            spacing = np.asarray([], dtype=int)
        else:
            spacing = np.diff(np.sort(self.selected_bins))
        response = np.abs(
            self._steering @ np.ones(self.selected_bins.size, dtype=np.complex64)
        )
        response = response / (np.max(response) + 1e-12)
        peaks, props = signal.find_peaks(response, height=float(threshold))
        peak_lags = self.lags[peaks].astype(int)
        peak_vals = props.get("peak_heights", np.asarray([], dtype=float))
        nonzero = np.abs(self.lags) > 2
        physical = self._lag_mask(lag_limit_samples)
        outside_physical = nonzero & ~physical
        inside_physical = nonzero & physical
        return {
            "selected_bin_count": int(self.selected_bins.size),
            "selected_bin_min": int(np.min(self.selected_bins)),
            "selected_bin_max": int(np.max(self.selected_bins)),
            "selected_spacing_unique": [int(v) for v in np.unique(spacing).tolist()],
            "selected_spacing_min": int(np.min(spacing)) if spacing.size else None,
            "selected_spacing_max": int(np.max(spacing)) if spacing.size else None,
            "alias_threshold": float(threshold),
            "alias_peak_lags": [int(v) for v in peak_lags.tolist()],
            "alias_peak_values": [float(v) for v in peak_vals.tolist()],
            "max_sidelobe_inside_physical_lag": (
                float(np.max(response[inside_physical])) if np.any(inside_physical) else None
            ),
            "max_sidelobe_outside_physical_lag": (
                float(np.max(response[outside_physical])) if np.any(outside_physical) else None
            ),
        }

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        return _diagnostics_from_estimator(
            self, batch_np, meta, fs, c, use_los_only=use_los_only,
            sub_sample=sub_sample, lag_limit_samples=lag_limit_samples
        )


class Cao2017DFTAMLEstimator(DirectDFTTDOAEstimator):
    """
    Cao-style compressed-frequency TDOA estimator.

    The estimator keeps the same selected DFT-bin budget as DirectDFTTDOA but
    scores each candidate lag with a phase-alignment likelihood objective:
    sum_k w_k cos(angle(X_i[k] conj(X_j[k])) + 2*pi*f_k*tau). This is closer to
    the ML/AML estimators used in compressed DFT TDOA literature than simply
    taking the magnitude of the partial inverse cross-spectrum.
    """

    def __init__(self, selected_bins, signal_len=1024, label="Cao2017-DFT-AML",
                 weight_mode="aml"):
        super().__init__(selected_bins, signal_len=signal_len, label=label, phat=False)
        self.weight_mode = str(weight_mode).lower()

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        spec_i = np.fft.fftshift(np.fft.fft(sig_i, norm="ortho"))
        spec_j = np.fft.fftshift(np.fft.fft(sig_j, norm="ortho"))
        cross = spec_i[self.selected_bins] * np.conj(spec_j[self.selected_bins])
        phase = cross / (np.abs(cross) + 1e-12)
        if self.weight_mode in ("phat", "unit", "unweighted"):
            weights = np.ones_like(np.abs(cross), dtype=float)
        else:
            weights = np.abs(cross).astype(float)
            weights = weights / (np.median(weights) + 1e-12)
            weights = np.clip(weights, 0.05, 20.0)
        score = np.real(self._steering @ (weights * phase))
        search_mask = self._lag_mask(lag_limit_samples)
        idx, lag, search_mask = _peak_from_score(
            self.lags, score, search_mask, sub_sample=sub_sample
        )
        if not return_quality:
            return lag
        weight, peak, sidelobe_ratio = _quality_from_score(score, idx, search_mask)
        return lag, weight, peak, sidelobe_ratio


class ZhaiPhaseSuperpositionEstimator(DirectDFTTDOAEstimator):
    """
    Zhai-style phase-superposition TDOA estimator on selected DFT bins.

    The previous implementation grouped bins by the observed pair phase before
    testing candidate delays. That loses the frequency-dependent phase slope and
    can fail even on a known integer shift. This version follows the physically
    meaningful phase-superposition criterion: for each candidate delay, compensate
    the selected cross-spectrum phases by that delay and coherently superpose the
    compensated unit phasors. The peak is therefore produced by the delay whose
    phase ramp best aligns the selected frequency bins.

    In the current strong-baseline track the selected bins are supplied by the
    Zhai CRLB/FIM decimation selector, so this curve represents "CRLB frequency
    selection + phase-only superposition" under the same all-pair WLS localizer.
    """

    def __init__(self, selected_bins, signal_len=1024,
                 label="Zhai-Phase-Superposition"):
        super().__init__(
            selected_bins=selected_bins,
            signal_len=signal_len,
            label=label,
            phat=True,
        )

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        spec_i = np.fft.fftshift(np.fft.fft(sig_i, norm="ortho"))
        spec_j = np.fft.fftshift(np.fft.fft(sig_j, norm="ortho"))
        cross = spec_i[self.selected_bins] * np.conj(spec_j[self.selected_bins])
        phase = cross / (np.abs(cross) + 1e-12)
        score = np.abs(self._steering @ phase)
        search_mask = self._lag_mask(lag_limit_samples)
        idx, lag, search_mask = _peak_from_score(
            self.lags, score, search_mask, sub_sample=sub_sample
        )
        if not return_quality:
            return lag
        weight, peak, sidelobe_ratio = _quality_from_score(score, idx, search_mask)
        return lag, weight, peak, sidelobe_ratio

    def alias_diagnostics(self, lag_limit_samples=None, threshold=0.8):
        diag = super().alias_diagnostics(
            lag_limit_samples=lag_limit_samples,
            threshold=threshold,
        )
        diag["phase_superposition"] = "candidate_delay_compensated_unit_phasors"
        diag["selection_source"] = "zhai_crlb_decimation_bins"
        return diag

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        diag = _diagnostics_from_estimator(
            self, batch_np, meta, fs, c, use_los_only=use_los_only,
            sub_sample=sub_sample, lag_limit_samples=lag_limit_samples
        )
        diag["direct_phase_bin_count"] = int(self.selected_bins.size)
        return diag
