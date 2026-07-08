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


def _normalize_score(score, search_mask):
    score = np.asarray(score, dtype=float)
    if np.any(search_mask):
        base = score[search_mask]
    else:
        base = score
    lo = float(np.min(base))
    hi = float(np.max(base))
    return (score - lo) / (hi - lo + 1e-12)


def _phase_weights(cross, mode="aml"):
    mag = np.abs(cross).astype(float)
    mode = str(mode).lower()
    if mode in ("unit", "phase", "phat", "unweighted"):
        return np.ones_like(mag, dtype=float)
    scale = float(np.percentile(mag, 75)) + 1e-12
    if mode in ("power", "aml"):
        weights = mag / scale
    else:
        weights = np.sqrt(mag / scale)
    return np.clip(weights, 0.05, 20.0)


def _weighted_phase_score(steering, cross, mode="aml"):
    phase = cross / (np.abs(cross) + 1e-12)
    weights = _phase_weights(cross, mode=mode)
    score = np.real(steering @ (weights * phase))
    coherence = float(
        np.abs(np.sum(weights * phase)) / (np.sum(weights) + 1e-12)
    )
    return score, weights, coherence


def _score_margin(sidelobe_ratio):
    return float(np.clip(1.0 - float(sidelobe_ratio), 0.0, 1.0))


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


def _diagnostics_from_quality_estimator(estimator, batch_np, meta, fs, c,
                                        use_los_only=True, sub_sample=True,
                                        lag_limit_samples=None):
    abs_err = []
    weights = []
    sidelobes = []
    extra = {}
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
                for key, value in getattr(estimator, "last_quality", {}).items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        extra.setdefault(key, []).append(float(value))

    arr = np.asarray(abs_err, dtype=float)
    warr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return {}
    weighted = (
        float(np.sum(warr * arr) / (np.sum(warr) + 1e-12))
        if np.sum(warr) > 0 else float("nan")
    )
    diag = {
        "direct_mean_abs_tdoa_error_samples": float(np.mean(arr)),
        "direct_weighted_abs_tdoa_error_samples": weighted,
        "direct_median_abs_tdoa_error_samples": float(np.median(arr)),
        "direct_within_1_sample_rate": float(np.mean(arr <= 1.0)),
        "direct_within_2_sample_rate": float(np.mean(arr <= 2.0)),
        "direct_mean_sidelobe_ratio": float(np.mean(sidelobes)),
        "direct_mean_weight": float(np.mean(weights)),
        "direct_lag_limit_samples": (
            float(lag_limit_samples) if lag_limit_samples is not None else float("nan")
        ),
    }
    for key, values in extra.items():
        if values:
            diag[f"hybrid_mean_{key}"] = float(np.mean(values))
            diag[f"hybrid_median_{key}"] = float(np.median(values))
    return diag


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


class GeoHybridDFTTDOAEstimator(DirectDFTTDOAEstimator):
    """
    Coherence-aware coarse-to-fine compressed-domain TDOA estimator.

    The coarse stage uses high-power/high-coherence DFT bins over the full
    physical lag window to reduce low-SNR phase variance. The fine stage uses
    geometry/ambiguity-aware bins only near the coarse posterior peak, where
    their lower deterministic sidelobe can improve resolution without jumping to
    distant alias peaks. The final pair weight propagates coherence, peak
    margin, coarse-fine consistency, and sidelobe ambiguity to all-pair WLS.
    """

    def __init__(self, coarse_bins, fine_bins, signal_len=1024,
                 label="GeoHybrid-DFT", coarse_mode="aml",
                 fine_mode="aml", refinement_radius=4.0,
                 fine_gain=1.25, consistency_sigma=3.0,
                 phat_gate=0.35, max_pair_weight=8.0):
        super().__init__(fine_bins, signal_len=signal_len, label=label, phat=False)
        self.coarse_bins = np.asarray(coarse_bins, dtype=int)
        self.fine_bins = np.asarray(fine_bins, dtype=int)
        if self.coarse_bins.ndim != 1 or self.coarse_bins.size < 1:
            raise ValueError("GeoHybridDFTTDOAEstimator needs coarse bins")
        if self.fine_bins.ndim != 1 or self.fine_bins.size < 1:
            raise ValueError("GeoHybridDFTTDOAEstimator needs fine bins")
        if np.any(self.coarse_bins < 0) or np.any(self.coarse_bins >= self.signal_len):
            raise ValueError("coarse_bins out of range")
        if np.any(self.fine_bins < 0) or np.any(self.fine_bins >= self.signal_len):
            raise ValueError("fine_bins out of range")

        freqs = np.fft.fftshift(np.fft.fftfreq(self.signal_len))
        self.coarse_freqs = freqs[self.coarse_bins]
        self.fine_freqs = freqs[self.fine_bins]
        self._coarse_steering = np.exp(
            2j * np.pi * np.outer(self.lags, self.coarse_freqs)
        ).astype(np.complex64)
        self._fine_steering = np.exp(
            2j * np.pi * np.outer(self.lags, self.fine_freqs)
        ).astype(np.complex64)
        self.coarse_mode = str(coarse_mode).lower()
        self.fine_mode = str(fine_mode).lower()
        self.refinement_radius = float(refinement_radius)
        self.fine_gain = float(fine_gain)
        self.consistency_sigma = float(consistency_sigma)
        self.phat_gate = float(phat_gate)
        self.max_pair_weight = float(max_pair_weight)
        self.last_quality = {}

    @staticmethod
    def _reliability(coherence, sidelobe_ratio):
        return float(np.clip(float(coherence) * _score_margin(sidelobe_ratio), 0.0, 1.0))

    def _score_with_mode(self, steering, cross, mode, search_mask):
        score, weights, coherence = _weighted_phase_score(steering, cross, mode=mode)
        idx, lag, _ = _peak_from_score(
            self.lags, score, search_mask, sub_sample=False
        )
        quality_weight, peak, sidelobe_ratio = _quality_from_score(
            score, idx, search_mask
        )
        return {
            "score": score,
            "weights": weights,
            "coherence": float(coherence),
            "idx": int(idx),
            "lag": float(lag),
            "quality_weight": float(quality_weight),
            "peak": float(peak),
            "sidelobe_ratio": float(sidelobe_ratio),
            "margin": _score_margin(sidelobe_ratio),
        }

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        spec_i = np.fft.fftshift(np.fft.fft(sig_i, norm="ortho"))
        spec_j = np.fft.fftshift(np.fft.fft(sig_j, norm="ortho"))
        coarse_cross = spec_i[self.coarse_bins] * np.conj(spec_j[self.coarse_bins])
        fine_cross = spec_i[self.fine_bins] * np.conj(spec_j[self.fine_bins])

        physical_mask = self._lag_mask(lag_limit_samples)
        coarse = self._score_with_mode(
            self._coarse_steering, coarse_cross, self.coarse_mode, physical_mask
        )
        fine_mode = self.fine_mode
        fine = self._score_with_mode(
            self._fine_steering, fine_cross,
            "aml" if fine_mode == "phat_gated" else fine_mode,
            physical_mask
        )
        coarse_rel = self._reliability(coarse["coherence"], coarse["sidelobe_ratio"])
        fine_rel = self._reliability(fine["coherence"], fine["sidelobe_ratio"])
        coarse_fine_dev = float(abs(fine["lag"] - coarse["lag"]))
        consistency = float(np.exp(
            -0.5 * (coarse_fine_dev / max(self.consistency_sigma, 1e-12)) ** 2
        ))

        if fine_mode == "phat_gated" and min(coarse_rel, fine_rel) >= self.phat_gate:
            fine = self._score_with_mode(
                self._fine_steering, fine_cross, "phat", physical_mask
            )
            fine_rel = self._reliability(fine["coherence"], fine["sidelobe_ratio"])
            coarse_fine_dev = float(abs(fine["lag"] - coarse["lag"]))
            consistency = float(np.exp(
                -0.5 * (coarse_fine_dev / max(self.consistency_sigma, 1e-12)) ** 2
            ))

        local_mask = physical_mask & (
            np.abs(self.lags - coarse["lag"]) <= self.refinement_radius
        )
        if not np.any(local_mask):
            local_mask = physical_mask
        fine_beta = float(np.clip(
            self.fine_gain * fine_rel * consistency / (coarse_rel + fine_rel + 1e-12),
            0.0, self.fine_gain
        ))
        coarse_norm = _normalize_score(coarse["score"], physical_mask)
        fine_norm = _normalize_score(fine["score"], local_mask)
        combined_score = coarse_norm + fine_beta * fine_norm
        idx, lag, local_mask = _peak_from_score(
            self.lags, combined_score, local_mask, sub_sample=sub_sample
        )

        if not return_quality:
            return lag

        base_weight, peak, sidelobe_ratio = _quality_from_score(
            combined_score, idx, local_mask
        )
        combined_margin = _score_margin(sidelobe_ratio)
        consistency_weight = 0.25 + 0.75 * consistency
        reliability_weight = 0.25 + 2.0 * coarse_rel + 2.0 * fine_rel * consistency
        weight = float(np.clip(
            base_weight * combined_margin * reliability_weight * consistency_weight,
            0.02, self.max_pair_weight
        ))
        self.last_quality = {
            "coarse_coherence": coarse["coherence"],
            "fine_coherence": fine["coherence"],
            "coarse_margin": coarse["margin"],
            "fine_margin": fine["margin"],
            "coarse_sidelobe_ratio": coarse["sidelobe_ratio"],
            "fine_sidelobe_ratio": fine["sidelobe_ratio"],
            "coarse_reliability": coarse_rel,
            "fine_reliability": fine_rel,
            "coarse_fine_deviation_samples": coarse_fine_dev,
            "coarse_fine_consistency": consistency,
            "fine_beta": fine_beta,
            "combined_margin": combined_margin,
            "pair_weight": weight,
        }
        return lag, weight, peak, sidelobe_ratio

    def alias_diagnostics(self, lag_limit_samples=None, threshold=0.8):
        fine_diag = super().alias_diagnostics(
            lag_limit_samples=lag_limit_samples,
            threshold=threshold,
        )
        coarse_tmp = DirectDFTTDOAEstimator(
            self.coarse_bins, signal_len=self.signal_len,
            label=f"{self.label}-coarse"
        )
        coarse_diag = coarse_tmp.alias_diagnostics(
            lag_limit_samples=lag_limit_samples,
            threshold=threshold,
        )
        fine_diag.update({
            "hybrid_estimator": "coarse_to_fine_geo_coherence",
            "hybrid_coarse_bin_count": int(self.coarse_bins.size),
            "hybrid_fine_bin_count": int(self.fine_bins.size),
            "hybrid_coarse_mode": self.coarse_mode,
            "hybrid_fine_mode": self.fine_mode,
            "hybrid_refinement_radius": float(self.refinement_radius),
            "hybrid_fine_gain": float(self.fine_gain),
            "hybrid_consistency_sigma": float(self.consistency_sigma),
            "hybrid_coarse_max_sidelobe_inside_physical_lag": (
                coarse_diag.get("max_sidelobe_inside_physical_lag")
            ),
            "hybrid_fine_max_sidelobe_inside_physical_lag": (
                fine_diag.get("max_sidelobe_inside_physical_lag")
            ),
        })
        return fine_diag

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        diag = _diagnostics_from_quality_estimator(
            self, batch_np, meta, fs, c, use_los_only=use_los_only,
            sub_sample=sub_sample, lag_limit_samples=lag_limit_samples
        )
        diag["hybrid_coarse_bin_count"] = int(self.coarse_bins.size)
        diag["hybrid_fine_bin_count"] = int(self.fine_bins.size)
        diag["hybrid_refinement_radius"] = float(self.refinement_radius)
        diag["hybrid_fine_gain"] = float(self.fine_gain)
        return diag


class Cao2020SegmentedFCEstimator(DirectDFTTDOAEstimator):
    """
    Cao-2020-style segmented high-FC estimator.

    The selected Fourier coefficients are split into contiguous frequency
    segments. Each segment produces its own delay-alignment likelihood, then the
    segment scores are normalized and added incoherently. This keeps the
    high-FC/CRB motivation while reducing the single-window alias sensitivity
    seen in the earlier Cao2020 proxy.
    """

    def __init__(self, selected_bins, signal_len=1024,
                 label="Cao2020-HighFC", n_segments=4,
                 weight_mode="sqrt_power"):
        super().__init__(
            selected_bins=selected_bins,
            signal_len=signal_len,
            label=label,
            phat=False,
        )
        self.n_segments = max(1, int(n_segments))
        self.weight_mode = str(weight_mode).lower()
        local_idx = np.arange(self.selected_bins.size)
        self._segments = [
            seg.astype(int)
            for seg in np.array_split(local_idx, self.n_segments)
            if seg.size > 0
        ]

    def _coherence_weights(self, cross):
        mag = np.abs(cross).astype(float)
        if self.weight_mode in ("unit", "phat", "phase"):
            return np.ones_like(mag, dtype=float)
        scale = float(np.percentile(mag, 75)) + 1e-12
        if self.weight_mode in ("power", "aml"):
            weights = mag / scale
        else:
            weights = np.sqrt(mag / scale)
        return np.clip(weights, 0.05, 10.0)

    @staticmethod
    def _normalize_score(score, search_mask):
        score = np.asarray(score, dtype=float)
        if np.any(search_mask):
            base = score[search_mask]
        else:
            base = score
        lo = float(np.min(base))
        hi = float(np.max(base))
        return (score - lo) / (hi - lo + 1e-12)

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        spec_i = np.fft.fftshift(np.fft.fft(sig_i, norm="ortho"))
        spec_j = np.fft.fftshift(np.fft.fft(sig_j, norm="ortho"))
        cross = spec_i[self.selected_bins] * np.conj(spec_j[self.selected_bins])
        search_mask = self._lag_mask(lag_limit_samples)
        score = np.zeros(self.lags.size, dtype=float)
        reliability_sum = 0.0

        for seg in self._segments:
            seg_cross = cross[seg]
            if seg_cross.size == 0:
                continue
            phase = seg_cross / (np.abs(seg_cross) + 1e-12)
            weights = self._coherence_weights(seg_cross)
            seg_score = np.real(self._steering[:, seg] @ (weights * phase))
            seg_score = self._normalize_score(seg_score, search_mask)
            coherence = float(np.abs(np.sum(weights * phase)) / (np.sum(weights) + 1e-12))
            energy = float(np.sqrt(np.mean(np.abs(seg_cross) ** 2)))
            reliability = float(np.clip(0.5 + coherence + 0.1 * np.log1p(energy), 0.25, 3.0))
            score += reliability * seg_score
            reliability_sum += reliability

        if reliability_sum <= 0:
            phase = cross / (np.abs(cross) + 1e-12)
            score = np.abs(self._steering @ phase)
        else:
            score = score / reliability_sum

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
        diag["cao2020_estimator"] = "segmented_incoherent_fc"
        diag["cao2020_n_segments"] = int(len(self._segments))
        diag["cao2020_segment_sizes"] = [int(seg.size) for seg in self._segments]
        diag["cao2020_weight_mode"] = self.weight_mode
        return diag

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        diag = _diagnostics_from_estimator(
            self, batch_np, meta, fs, c, use_los_only=use_los_only,
            sub_sample=sub_sample, lag_limit_samples=lag_limit_samples
        )
        diag["direct_cao2020_segments"] = int(len(self._segments))
        return diag


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
    Zhai CRLB/FIM decimation selector. The default implementation uses a
    clipped square-root magnitude weight, so it remains a phase-superposition
    estimator but does not throw away all coherence/SNR information.
    """

    def __init__(self, selected_bins, signal_len=1024,
                 label="Zhai-Phase-Superposition", weight_mode="sqrt_power"):
        super().__init__(
            selected_bins=selected_bins,
            signal_len=signal_len,
            label=label,
            phat=True,
        )
        self.weight_mode = str(weight_mode).lower()

    def _phase_weights(self, cross):
        mag = np.abs(cross).astype(float)
        if self.weight_mode in ("unit", "phase", "phat", "unweighted"):
            return np.ones_like(mag, dtype=float)
        scale = float(np.percentile(mag, 75)) + 1e-12
        if self.weight_mode in ("power", "aml"):
            weights = mag / scale
        else:
            weights = np.sqrt(mag / scale)
        return np.clip(weights, 0.05, 10.0)

    def estimate_pair(self, sig_i, sig_j, sub_sample=True, return_quality=True,
                      lag_limit_samples=None):
        spec_i = np.fft.fftshift(np.fft.fft(sig_i, norm="ortho"))
        spec_j = np.fft.fftshift(np.fft.fft(sig_j, norm="ortho"))
        cross = spec_i[self.selected_bins] * np.conj(spec_j[self.selected_bins])
        phase = cross / (np.abs(cross) + 1e-12)
        weights = self._phase_weights(cross)
        score = np.abs(self._steering @ (weights * phase)) / (np.sum(weights) + 1e-12)
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
        diag["phase_superposition"] = "candidate_delay_compensated_weighted_phasors"
        diag["phase_weight_mode"] = self.weight_mode
        diag["selection_source"] = "zhai_crlb_decimation_bins"
        return diag

    def diagnostics(self, batch_np, meta, fs, c, use_los_only=True,
                    sub_sample=True, lag_limit_samples=None):
        diag = _diagnostics_from_estimator(
            self, batch_np, meta, fs, c, use_los_only=use_los_only,
            sub_sample=sub_sample, lag_limit_samples=lag_limit_samples
        )
        diag["direct_phase_bin_count"] = int(self.selected_bins.size)
        diag["direct_phase_weight_mode"] = self.weight_mode
        return diag
