import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def latent_real_dim(signal_len, cr):
    """Number of transmitted real scalars under the DAE CR convention."""
    return int(2 * signal_len // cr)


def _to_complex(x):
    return torch.complex(x[:, 0, :], x[:, 1, :])


def _from_complex(xc):
    return torch.stack((xc.real, xc.imag), dim=1).float()


def _fftshift(x):
    return torch.roll(x, shifts=x.shape[-1] // 2, dims=-1)


def _ifftshift(x):
    return torch.roll(x, shifts=-(x.shape[-1] // 2), dims=-1)


def _select_indices(total, count, mode, seed=42, center=None):
    count = int(count)
    total = int(total)
    if count < 1 or count > total:
        raise ValueError("Invalid number of selected coefficients")
    mode = str(mode).lower()
    if mode in ("center", "bandlimited", "lowpass"):
        if center is None:
            center = total // 2
        half = count // 2
        start = int(center) - half
        end = start + count
        if start < 0 or end > total:
            raise ValueError("Centered selection exceeds available coefficients")
        return torch.arange(start, end, dtype=torch.long)
    if mode in ("uniform", "uniform_stride"):
        if count == 1:
            return torch.tensor([total // 2], dtype=torch.long)
        # Deterministic full-spectrum sampling, avoiding the bandlimited oracle.
        vals = torch.linspace(0, total - 1, steps=count)
        return torch.unique(vals.round().long(), sorted=True)
    if mode in ("random", "random_fixed_seed"):
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        return torch.randperm(total, generator=gen)[:count].sort().values.long()
    raise ValueError(f"Unknown selection mode: {mode}")


def _top_power_bins(waveforms, count, chunk_size=512):
    """Select FFT bins with the largest average noisy training-set power."""
    avg_power = _average_shifted_fft_power(waveforms, chunk_size=chunk_size)
    return torch.topk(avg_power, k=int(count), largest=True).indices.sort().values.long()


def _fisher_power_bins(waveforms, count, chunk_size=512, min_power_frac=0.05):
    """
    Select DFT bins using a TDOA-aware RMS-bandwidth proxy.

    For delay estimation, Fisher information scales with signal power weighted
    by squared frequency. The power floor prevents the selector from spending
    the feature budget on high-frequency noise-only bins.
    """
    count = int(count)
    avg_power = _average_shifted_fft_power(waveforms, chunk_size=chunk_size)
    if count < 1 or count > avg_power.numel():
        raise ValueError("Invalid DFT Fisher selection size")
    center = avg_power.numel() // 2
    freq = torch.arange(avg_power.numel(), dtype=avg_power.dtype) - float(center)
    freq_weight = (freq.abs() / max(float(center), 1.0)) ** 2
    score = avg_power * freq_weight
    power_floor = torch.max(avg_power) * float(min_power_frac)
    score = torch.where(
        avg_power >= power_floor,
        score,
        torch.full_like(score, -torch.inf),
    )
    if torch.isfinite(score).sum().item() < count:
        score = avg_power * freq_weight
    return torch.topk(score, k=count, largest=True).indices.sort().values.long()


def _balanced_fisher_bins(waveforms, count, chunk_size=512, min_power_frac=0.05,
                          lag_limit_samples=None, candidate_factor=8,
                          sidelobe_weight=0.65, rms_sidelobe_weight=0.25,
                          guard_samples=2):
    """
    Select task-aware DFT bins with Fisher power and physical-lag sidelobe control.

    A pure ``power * f^2`` score over-selects edge-frequency bins. That improves
    nominal delay Fisher information but creates a comb-like partial Fourier
    steering response, so direct TDOA peaks become ambiguous inside the physical
    lag search window. This greedy selector keeps the Fisher-information motive
    while penalizing the selected-bin response sidelobes over valid TDOA lags.
    """
    count = int(count)
    avg_power = _average_shifted_fft_power(waveforms, chunk_size=chunk_size)
    n = int(avg_power.numel())
    if count < 1 or count > n:
        raise ValueError("Invalid balanced DFT Fisher selection size")

    center = n // 2
    freq_bin = torch.arange(n, dtype=avg_power.dtype) - float(center)
    freq_weight = (freq_bin.abs() / max(float(center), 1.0)) ** 2
    fisher_score = avg_power * freq_weight

    power_floor = torch.max(avg_power) * float(min_power_frac)
    valid = avg_power >= power_floor
    if int(torch.sum(valid).item()) < count:
        valid = torch.ones_like(valid, dtype=torch.bool)

    masked_score = torch.where(
        valid, fisher_score, torch.full_like(fisher_score, -torch.inf)
    )
    finite_count = int(torch.isfinite(masked_score).sum().item())
    if finite_count < count:
        masked_score = fisher_score
        finite_count = n

    candidate_count = min(n, max(count, int(count) * int(candidate_factor)))
    candidate_count = min(candidate_count, finite_count)
    candidate_bins = torch.topk(
        masked_score, k=candidate_count, largest=True
    ).indices.sort().values

    # If the power floor produced too narrow a candidate pool, complete it with
    # high-power bins so the greedy step always has enough feasible choices.
    if candidate_bins.numel() < count:
        fallback = torch.topk(avg_power, k=count, largest=True).indices
        candidate_bins = torch.unique(
            torch.cat((candidate_bins, fallback)), sorted=True
        )

    candidate_np = candidate_bins.detach().cpu().numpy().astype(int)
    scores = fisher_score[candidate_bins].detach().cpu().numpy().astype(np.float64)
    score_scale = max(float(np.max(scores)), 1e-12)

    freqs = np.fft.fftshift(np.fft.fftfreq(n))[candidate_np]
    lags = np.arange(-n // 2, n // 2, dtype=float)
    if lag_limit_samples is None:
        lag_limit_samples = min(64, max(4, n // 16))
    physical_mask = np.abs(lags) <= float(lag_limit_samples)
    physical_lags = lags[physical_mask]
    if physical_lags.size == 0:
        physical_lags = lags
    side_mask = np.abs(physical_lags) > float(guard_samples)
    if not np.any(side_mask):
        side_mask = np.ones_like(physical_lags, dtype=bool)

    steering = np.exp(
        2j * np.pi * np.outer(physical_lags, freqs)
    ).astype(np.complex64)

    selected_local = []
    remaining = np.ones(candidate_np.size, dtype=bool)
    response = np.zeros(physical_lags.size, dtype=np.complex64)
    score_sum = 0.0
    for step in range(count):
        remaining_idx = np.where(remaining)[0]
        if remaining_idx.size == 0:
            break
        trial_response = response[:, None] + steering[:, remaining_idx]
        amp = np.abs(trial_response) / float(step + 1)
        max_side = np.max(amp[side_mask], axis=0)
        rms_side = np.sqrt(np.mean(amp[side_mask] ** 2, axis=0))
        mean_score = (score_sum + scores[remaining_idx]) / float(step + 1)
        objective = (
            mean_score / score_scale
            - float(sidelobe_weight) * max_side
            - float(rms_sidelobe_weight) * rms_side
        )
        best_local = int(remaining_idx[int(np.argmax(objective))])
        selected_local.append(best_local)
        remaining[best_local] = False
        response += steering[:, best_local]
        score_sum += float(scores[best_local])

    if len(selected_local) < count:
        fallback = np.where(remaining)[0][:count - len(selected_local)]
        selected_local.extend([int(v) for v in fallback.tolist()])

    selected = torch.as_tensor(candidate_np[selected_local[:count]], dtype=torch.long)
    return selected.sort().values.contiguous()


def _window_lag_sidelobes(selected_bins, signal_len, lag_limit_samples=None,
                          guard_samples=2):
    n = int(signal_len)
    selected = np.asarray(selected_bins, dtype=int)
    if selected.size == 0:
        return 1.0, 1.0
    lags = np.arange(-n // 2, n // 2, dtype=float)
    if lag_limit_samples is None:
        lag_limit_samples = min(64, max(4, n // 16))
    physical_mask = np.abs(lags) <= float(lag_limit_samples)
    physical_lags = lags[physical_mask]
    if physical_lags.size == 0:
        physical_lags = lags
    side_mask = np.abs(physical_lags) > float(guard_samples)
    if not np.any(side_mask):
        side_mask = np.ones_like(physical_lags, dtype=bool)
    freqs = np.fft.fftshift(np.fft.fftfreq(n))[selected]
    response = np.abs(
        np.exp(2j * np.pi * np.outer(physical_lags, freqs))
        @ np.ones(selected.size, dtype=np.complex64)
    ) / float(selected.size)
    return float(np.max(response[side_mask])), float(np.sqrt(np.mean(response[side_mask] ** 2)))


def _geometry_lag_weights(simulator, signal_len, lag_limit_samples=None,
                          use_los_only=True, guard_samples=2):
    """
    Geometry-aware weights over physically valid TDOA lags.

    The selected-bin steering response should not only have low uniform
    sidelobes; it should especially avoid ambiguity at lag values that are
    common and localization-sensitive for the current UAV geometry. The weight
    is a smoothed histogram of LOS pair TDOAs, weighted by the norm of the TDOA
    Jacobian row ||u_i - u_j|| at the source position.
    """
    n = int(signal_len)
    lags = np.arange(-n // 2, n // 2, dtype=float)
    if lag_limit_samples is None:
        lag_limit_samples = min(64, max(4, n // 16))
    physical_mask = np.abs(lags) <= float(lag_limit_samples)
    physical_lags = lags[physical_mask]
    if physical_lags.size == 0:
        physical_lags = lags

    weights = np.ones(physical_lags.size, dtype=np.float64)
    snapshots = getattr(simulator, "_urban_snapshots", None)
    sample_distance = getattr(simulator, "sample_distance_m", None)
    if not snapshots or sample_distance is None:
        return weights / (np.sum(weights) + 1e-12)

    hist = np.zeros_like(weights)
    sigma = 1.5
    for snap in snapshots:
        los = np.asarray(snap.get("los", []), dtype=bool)
        if los.size == 0:
            continue
        candidates = np.where(los)[0] if use_los_only else np.arange(los.size)
        if len(candidates) < 2:
            continue
        source = np.asarray(snap["source"], dtype=float)
        uavs = np.asarray(snap["uavs"], dtype=float)
        distances = np.asarray(snap["distances"], dtype=float)
        for ai in range(len(candidates)):
            for bi in range(ai + 1, len(candidates)):
                ui, uj = int(candidates[ai]), int(candidates[bi])
                tau = (distances[ui] - distances[uj]) / float(sample_distance)
                if abs(tau) > float(lag_limit_samples):
                    continue
                vi = (source - uavs[ui]) / (distances[ui] + 1e-12)
                vj = (source - uavs[uj]) / (distances[uj] + 1e-12)
                sensitivity = float(np.linalg.norm(vi - vj))
                sensitivity = float(np.clip(sensitivity, 0.15, 2.0))
                hist += (sensitivity ** 2) * np.exp(
                    -0.5 * ((physical_lags - float(tau)) / sigma) ** 2
                )

    if np.max(hist) <= 0:
        return weights / (np.sum(weights) + 1e-12)
    hist = hist / (np.max(hist) + 1e-12)
    # Keep a uniform floor so rare but physically valid lags are not ignored.
    weights = 0.35 * np.ones_like(hist) + 0.65 * hist
    side_mask = np.abs(physical_lags) > float(guard_samples)
    if np.any(side_mask):
        weights[side_mask] = weights[side_mask] / (np.mean(weights[side_mask]) + 1e-12)
    return weights / (np.sum(weights) + 1e-12)


def _geo_ambiguity_bins(waveforms, count, simulator=None, chunk_size=512,
                        min_power_frac=0.02, lag_limit_samples=None,
                        candidate_factor=14, sidelobe_weight=0.50,
                        geo_sidelobe_weight=0.55, rms_sidelobe_weight=0.20,
                        cluster_weight=0.18, balance_weight=0.08,
                        guard_samples=2):
    """
    Geometry/ambiguity-aware direct-DFT frequency selection.

    This selector keeps the CRLB/Fisher motivation of high-frequency, high-power
    bins, but it explicitly suppresses partial-Fourier ambiguity in the physical
    TDOA window and discourages frequency clustering. Compared with the older
    balanced Fisher selector, the lag-domain penalty is weighted by the current
    urban UAV geometry and LOS-pair TDOA distribution.
    """
    count = int(count)
    avg_power = _average_shifted_fft_power(waveforms, chunk_size=chunk_size)
    n = int(avg_power.numel())
    if count < 1 or count > n:
        raise ValueError("Invalid GeoAmbi DFT selection size")

    center = n // 2
    freq = (torch.arange(n, dtype=avg_power.dtype) - float(center)) / max(float(center), 1.0)
    freq_abs = torch.abs(freq)
    fisher_score = avg_power * (freq_abs ** 2)
    power_floor = torch.max(avg_power) * float(min_power_frac)
    valid = avg_power >= power_floor
    if int(torch.sum(valid).item()) < count:
        valid = torch.ones_like(valid, dtype=torch.bool)
    masked_score = torch.where(valid, fisher_score, torch.full_like(fisher_score, -torch.inf))
    finite_count = int(torch.isfinite(masked_score).sum().item())
    if finite_count < count:
        masked_score = fisher_score
        finite_count = n

    candidate_count = min(n, max(count, count * int(candidate_factor)))
    candidate_count = min(candidate_count, finite_count)
    top_fisher = torch.topk(masked_score, k=candidate_count, largest=True).indices
    top_power = torch.topk(avg_power, k=min(n, candidate_count), largest=True).indices
    # A sparse uniform scaffold improves conditioning if Fisher candidates
    # collapse into a narrow high-frequency cluster.
    scaffold = torch.linspace(0, n - 1, steps=max(count, min(n, candidate_count // 2)))
    scaffold = scaffold.round().long().clamp(0, n - 1)
    candidate_bins = torch.unique(
        torch.cat((top_fisher, top_power, scaffold)), sorted=True
    )
    if candidate_bins.numel() > candidate_count:
        cand_score = fisher_score[candidate_bins]
        keep = torch.topk(cand_score, k=candidate_count, largest=True).indices
        candidate_bins = candidate_bins[keep].sort().values
    if candidate_bins.numel() < count:
        fallback = torch.topk(fisher_score, k=count, largest=True).indices
        candidate_bins = torch.unique(torch.cat((candidate_bins, fallback)), sorted=True)

    candidate_np = candidate_bins.detach().cpu().numpy().astype(int)
    cand_power = avg_power[candidate_bins].detach().cpu().numpy().astype(np.float64)
    cand_freq = freq[candidate_bins].detach().cpu().numpy().astype(np.float64)
    cand_freq_abs = np.abs(cand_freq)

    lags = np.arange(-n // 2, n // 2, dtype=float)
    if lag_limit_samples is None:
        lag_limit_samples = min(64, max(4, n // 16))
    physical_mask = np.abs(lags) <= float(lag_limit_samples)
    physical_lags = lags[physical_mask]
    if physical_lags.size == 0:
        physical_lags = lags
    side_mask = np.abs(physical_lags) > float(guard_samples)
    if not np.any(side_mask):
        side_mask = np.ones_like(physical_lags, dtype=bool)
    geo_weights = _geometry_lag_weights(
        simulator, n, lag_limit_samples=lag_limit_samples,
        use_los_only=True, guard_samples=guard_samples
    )
    if geo_weights.shape[0] != physical_lags.shape[0]:
        geo_weights = np.ones(physical_lags.size, dtype=np.float64)
        geo_weights = geo_weights / np.sum(geo_weights)
    side_weights = geo_weights[side_mask]
    side_weights = side_weights / (np.sum(side_weights) + 1e-12)

    steering = np.exp(
        2j * np.pi * np.outer(physical_lags, np.fft.fftshift(np.fft.fftfreq(n))[candidate_np])
    ).astype(np.complex64)

    selected_local = []
    remaining = np.ones(candidate_np.size, dtype=bool)
    response = np.zeros(physical_lags.size, dtype=np.complex64)
    w_sum = 0.0
    wf_sum = 0.0
    wf2_sum = 0.0
    power_sum = 0.0
    info_scale = max(float(np.max(cand_power * cand_freq ** 2)), 1e-12)
    power_scale = max(float(np.max(cand_power)), 1e-12)
    cluster_width = max(4.0, 0.025 * n)

    for step in range(count):
        remaining_idx = np.where(remaining)[0]
        if remaining_idx.size == 0:
            break
        w_new = w_sum + cand_power[remaining_idx]
        wf_new = wf_sum + cand_power[remaining_idx] * cand_freq[remaining_idx]
        wf2_new = wf2_sum + cand_power[remaining_idx] * cand_freq[remaining_idx] ** 2
        crlb_info = wf2_new - (wf_new ** 2) / (w_new + 1e-12)
        crlb_info = np.maximum(crlb_info, 0.0)
        trial_response = response[:, None] + steering[:, remaining_idx]
        amp = np.abs(trial_response) / float(step + 1)
        side_amp = amp[side_mask]
        max_side = np.max(side_amp, axis=0)
        rms_side = np.sqrt(np.mean(side_amp ** 2, axis=0))
        geo_side = np.sqrt(np.sum(side_weights[:, None] * (side_amp ** 2), axis=0))

        if selected_local:
            selected_bins = candidate_np[np.asarray(selected_local, dtype=int)]
            dist = np.min(np.abs(candidate_np[remaining_idx, None] - selected_bins[None, :]), axis=1)
            cluster_penalty = np.exp(-(dist / cluster_width) ** 2)
            sign_balance = np.abs(
                (np.sum(np.sign(cand_freq[selected_local]))
                 + np.sign(cand_freq[remaining_idx])) / float(step + 1)
            )
        else:
            cluster_penalty = np.zeros(remaining_idx.size, dtype=np.float64)
            sign_balance = np.abs(np.sign(cand_freq[remaining_idx]))

        power_new = power_sum + cand_power[remaining_idx]
        highness = (
            np.sum(cand_freq_abs[selected_local]) + cand_freq_abs[remaining_idx]
        ) / float(step + 1)
        objective = (
            np.log1p(crlb_info / info_scale)
            + 0.10 * np.log1p(power_new / power_scale)
            + 0.12 * highness
            - float(sidelobe_weight) * max_side
            - float(geo_sidelobe_weight) * geo_side
            - float(rms_sidelobe_weight) * rms_side
            - float(cluster_weight) * cluster_penalty
            - float(balance_weight) * sign_balance
        )
        best_local = int(remaining_idx[int(np.argmax(objective))])
        selected_local.append(best_local)
        remaining[best_local] = False
        response += steering[:, best_local]
        w_sum += float(cand_power[best_local])
        wf_sum += float(cand_power[best_local] * cand_freq[best_local])
        wf2_sum += float(cand_power[best_local] * cand_freq[best_local] ** 2)
        power_sum += float(cand_power[best_local])

    if len(selected_local) < count:
        fallback = np.where(remaining)[0][:count - len(selected_local)]
        selected_local.extend([int(v) for v in fallback.tolist()])
    selected = torch.as_tensor(candidate_np[selected_local[:count]], dtype=torch.long)
    return selected.sort().values.contiguous()


def _cao2020_crb_fc_bins(waveforms, count, chunk_size=512, min_power_frac=0.02,
                         lag_limit_samples=None, sidelobe_weight=0.85,
                         rms_sidelobe_weight=0.30, guard_samples=2):
    """
    Cao-2020-style partial Fourier coefficient selection.

    Cao et al. show that higher Fourier coefficients carry more delay
    sensitivity, but the current task also needs a stable partial-Fourier
    steering response inside the physically valid TDOA window. This selector
    therefore scans contiguous one-sided high-frequency FC blocks and scores
    them by CRB/Fisher delay information, retained signal power, high-frequency
    preference, and physical-lag sidelobe suppression. It avoids the previous
    two-edge ``highest |f|`` selection, which created comb-like ambiguity.
    """
    count = int(count)
    avg_power = _average_shifted_fft_power(waveforms, chunk_size=chunk_size)
    n = int(avg_power.numel())
    if count < 1 or count > n:
        raise ValueError("Invalid Cao2020 DFT selection size")

    power_np = avg_power.detach().cpu().numpy().astype(np.float64)
    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    freq_abs = np.abs(freqs) / (np.max(np.abs(freqs)) + 1e-12)
    power_floor = float(np.max(power_np)) * float(min_power_frac)

    center = n // 2
    # One-sided contiguous FC blocks. Do not join the two Nyquist-edge blocks:
    # that was the source of the previous comb ambiguity.
    ranges = [(0, center), (center + 1, n)]
    fim_scale = max(float(np.max(power_np * freq_abs ** 2)) * float(count), 1e-12)
    power_scale = max(float(np.max(power_np)) * float(count), 1e-12)

    best_obj = -np.inf
    best_bins = None
    for lo, hi in ranges:
        if hi - lo < count:
            continue
        for start in range(lo, hi - count + 1):
            bins = np.arange(start, start + count, dtype=int)
            valid_frac = float(np.mean(power_np[bins] >= power_floor))
            if valid_frac < 0.25:
                continue
            fim_info = float(np.sum(power_np[bins] * freq_abs[bins] ** 2))
            band_power = float(np.sum(power_np[bins]))
            highness = float(np.mean(freq_abs[bins]))
            max_side, rms_side = _window_lag_sidelobes(
                bins, n, lag_limit_samples=lag_limit_samples,
                guard_samples=guard_samples
            )
            objective = (
                np.log1p(fim_info / fim_scale)
                + 0.20 * np.log1p(band_power / power_scale)
                + 0.20 * highness
                + 0.10 * valid_frac
                - float(sidelobe_weight) * max_side
                - float(rms_sidelobe_weight) * rms_side
            )
            if objective > best_obj:
                best_obj = float(objective)
                best_bins = bins

    if best_bins is None:
        # Robust fallback: keep a contiguous high-power band rather than
        # returning to the unstable two-edge highest-|f| selection.
        best_bins = _contiguous_power_bins(waveforms, count, chunk_size=chunk_size)
        return best_bins.long().contiguous()
    return torch.as_tensor(best_bins, dtype=torch.long).contiguous()


def _average_shifted_fft_power(waveforms, chunk_size=512):
    waveforms = waveforms.float().cpu()
    signal_len = int(waveforms.shape[-1])
    total_power = torch.zeros(signal_len, dtype=torch.float64)
    total_count = 0
    for start in range(0, waveforms.shape[0], int(chunk_size)):
        chunk = waveforms[start:start + int(chunk_size)]
        xc = torch.complex(chunk[:, 0, :], chunk[:, 1, :])
        spec = _fftshift(torch.fft.fft(xc, dim=-1, norm="ortho"))
        total_power += torch.sum(torch.abs(spec).double() ** 2, dim=0)
        total_count += int(chunk.shape[0])
    if total_count == 0:
        raise ValueError("DFT train-set power selection needs at least one waveform")
    return total_power / float(total_count)


def select_balanced_fisher_dft_bins(waveforms, signal_len=1024, cr=16,
                                    lag_limit_samples=None, chunk_size=512):
    """Public helper for task-aware direct DFT updates without fitting PCA."""
    n_complex_bins = max(1, latent_real_dim(int(signal_len), int(cr)) // 2)
    return _balanced_fisher_bins(
        waveforms, n_complex_bins, chunk_size=chunk_size,
        lag_limit_samples=lag_limit_samples
    )


def select_geo_ambiguity_dft_bins(waveforms, simulator=None, signal_len=1024,
                                  cr=16, lag_limit_samples=None,
                                  chunk_size=512):
    """Public helper for geometry/ambiguity-aware direct DFT bin selection."""
    n_complex_bins = max(1, latent_real_dim(int(signal_len), int(cr)) // 2)
    return _geo_ambiguity_bins(
        waveforms, n_complex_bins, simulator=simulator,
        chunk_size=chunk_size, lag_limit_samples=lag_limit_samples
    )


def select_cao2020_crb_dft_bins(waveforms, signal_len=1024, cr=16,
                                lag_limit_samples=None, chunk_size=512):
    """Public helper for Cao-2020-style CRB/high-FC direct-TDOA baselines."""
    n_complex_bins = max(1, latent_real_dim(int(signal_len), int(cr)) // 2)
    return _cao2020_crb_fc_bins(
        waveforms, n_complex_bins, chunk_size=chunk_size,
        lag_limit_samples=lag_limit_samples
    )


def _zhai_crlb_bins(waveforms, count, chunk_size=512, min_power_frac=0.02,
                    lag_limit_samples=None, candidate_factor=10,
                    sidelobe_weight=0.45, rms_sidelobe_weight=0.15,
                    guard_samples=2):
    """
    Greedy Zhai-style CRLB/FIM frequency decimation.

    The delay CRLB of a bandlimited signal is governed by the weighted spectral
    second central moment. This selector greedily maximizes that CRLB
    denominator while penalizing ambiguous partial-Fourier sidelobes inside the
    physically valid TDOA window. It is an implementable proxy for Zhai's
    CRLB-driven frequency extraction, not a claim of reproducing every thesis
    optimizer detail.
    """
    count = int(count)
    avg_power = _average_shifted_fft_power(waveforms, chunk_size=chunk_size)
    n = int(avg_power.numel())
    if count < 1 or count > n:
        raise ValueError("Invalid Zhai CRLB DFT selection size")

    center = n // 2
    freq = (torch.arange(n, dtype=avg_power.dtype) - float(center)) / max(float(center), 1.0)
    power_floor = torch.max(avg_power) * float(min_power_frac)
    valid = avg_power >= power_floor
    if int(torch.sum(valid).item()) < count:
        valid = torch.ones_like(valid, dtype=torch.bool)
    fim_seed_score = avg_power * (freq ** 2)
    masked_score = torch.where(valid, fim_seed_score, torch.full_like(fim_seed_score, -torch.inf))
    finite_count = int(torch.isfinite(masked_score).sum().item())
    if finite_count < count:
        masked_score = fim_seed_score
        finite_count = n

    candidate_count = min(n, max(count, count * int(candidate_factor)))
    candidate_count = min(candidate_count, finite_count)
    candidate_bins = torch.topk(masked_score, k=candidate_count, largest=True).indices.sort().values
    candidate_np = candidate_bins.detach().cpu().numpy().astype(int)
    cand_power = avg_power[candidate_bins].detach().cpu().numpy().astype(np.float64)
    cand_freq = freq[candidate_bins].detach().cpu().numpy().astype(np.float64)

    lags = np.arange(-n // 2, n // 2, dtype=float)
    if lag_limit_samples is None:
        lag_limit_samples = min(64, max(4, n // 16))
    physical_mask = np.abs(lags) <= float(lag_limit_samples)
    physical_lags = lags[physical_mask]
    if physical_lags.size == 0:
        physical_lags = lags
    side_mask = np.abs(physical_lags) > float(guard_samples)
    if not np.any(side_mask):
        side_mask = np.ones_like(physical_lags, dtype=bool)
    steering = np.exp(
        2j * np.pi * np.outer(physical_lags, np.fft.fftshift(np.fft.fftfreq(n))[candidate_np])
    ).astype(np.complex64)

    selected_local = []
    remaining = np.ones(candidate_np.size, dtype=bool)
    response = np.zeros(physical_lags.size, dtype=np.complex64)
    w_sum = 0.0
    wf_sum = 0.0
    wf2_sum = 0.0
    max_info_scale = max(float(np.max(cand_power * cand_freq ** 2)), 1e-12)
    for step in range(count):
        remaining_idx = np.where(remaining)[0]
        if remaining_idx.size == 0:
            break
        w_new = w_sum + cand_power[remaining_idx]
        wf_new = wf_sum + cand_power[remaining_idx] * cand_freq[remaining_idx]
        wf2_new = wf2_sum + cand_power[remaining_idx] * cand_freq[remaining_idx] ** 2
        crlb_info = wf2_new - (wf_new ** 2) / (w_new + 1e-12)
        crlb_info = np.maximum(crlb_info, 0.0)
        trial_response = response[:, None] + steering[:, remaining_idx]
        amp = np.abs(trial_response) / float(step + 1)
        max_side = np.max(amp[side_mask], axis=0)
        rms_side = np.sqrt(np.mean(amp[side_mask] ** 2, axis=0))
        objective = (
            np.log1p(crlb_info / max_info_scale)
            - float(sidelobe_weight) * max_side
            - float(rms_sidelobe_weight) * rms_side
        )
        best_local = int(remaining_idx[int(np.argmax(objective))])
        selected_local.append(best_local)
        remaining[best_local] = False
        response += steering[:, best_local]
        w_sum += float(cand_power[best_local])
        wf_sum += float(cand_power[best_local] * cand_freq[best_local])
        wf2_sum += float(cand_power[best_local] * cand_freq[best_local] ** 2)

    if len(selected_local) < count:
        fallback = np.where(remaining)[0][:count - len(selected_local)]
        selected_local.extend([int(v) for v in fallback.tolist()])
    selected = torch.as_tensor(candidate_np[selected_local[:count]], dtype=torch.long)
    return selected.sort().values.contiguous()


def select_zhai_crlb_dft_bins(waveforms, signal_len=1024, cr=16,
                              lag_limit_samples=None, chunk_size=512):
    """Public helper for Zhai-style CRLB/FIM direct-TDOA baselines."""
    n_complex_bins = max(1, latent_real_dim(int(signal_len), int(cr)) // 2)
    return _zhai_crlb_bins(
        waveforms, n_complex_bins, chunk_size=chunk_size,
        lag_limit_samples=lag_limit_samples
    )


def _contiguous_power_bins(waveforms, count, chunk_size=512):
    """Select one contiguous FFT window with the largest average training-set power."""
    count = int(count)
    avg_power = _average_shifted_fft_power(waveforms, chunk_size=chunk_size)
    if count < 1 or count > avg_power.numel():
        raise ValueError("Invalid DFT contiguous window size")
    if count == avg_power.numel():
        return torch.arange(avg_power.numel(), dtype=torch.long)
    prefix = torch.cat((torch.zeros(1, dtype=avg_power.dtype), torch.cumsum(avg_power, dim=0)))
    window_power = prefix[count:] - prefix[:-count]
    start = int(torch.argmax(window_power).item())
    return torch.arange(start, start + count, dtype=torch.long)


class DFTCompressionBaseline(nn.Module):
    """
    Fixed DFT transform-coding baseline.

    A complex coefficient carries two real scalars, so CR=16 with a 1024-sample
    complex baseband waveform keeps 64 complex bins, matching the DAE's
    128-real-scalar latent budget.
    """

    def __init__(self, signal_len=1024, cr=16, mode="uniform", seed=42,
                 selected_bins=None):
        super().__init__()
        self.signal_len = int(signal_len)
        self.cr = int(cr)
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.feature_dim = latent_real_dim(self.signal_len, self.cr)
        self.n_complex_bins = max(1, self.feature_dim // 2)
        if selected_bins is None:
            bins = _select_indices(
                self.signal_len, self.n_complex_bins, self.mode,
                seed=self.seed, center=self.signal_len // 2
            )
        else:
            bins = torch.as_tensor(selected_bins, dtype=torch.long)
            if bins.numel() != self.n_complex_bins:
                raise ValueError("selected_bins length must match the DFT feature budget")
        self.register_buffer("selected_bins", bins.long().contiguous())

    def forward(self, x):
        xc = _to_complex(x)
        spec = _fftshift(torch.fft.fft(xc, dim=-1, norm="ortho"))
        kept = torch.zeros_like(spec)
        kept[:, self.selected_bins] = spec[:, self.selected_bins]
        recon = torch.fft.ifft(_ifftshift(kept), dim=-1, norm="ortho")
        return _from_complex(recon)


def _hadamard_matrix(n):
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard baseline requires a power-of-two signal length")
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat(
            (torch.cat((h, h), dim=1),
             torch.cat((h, -h), dim=1)),
            dim=0,
        )
    return h / math.sqrt(float(n))


def _sequency_order(h):
    signs = torch.sign(h)
    signs[signs == 0] = 1
    changes = torch.sum(signs[:, 1:] != signs[:, :-1], dim=1)
    return torch.argsort(changes)


class HadamardProjectionBaseline(nn.Module):
    """
    Salari-style Hadamard measurement plus projection reconstruction.

    Salari Lemma 1 uses a contiguous partial-Hadamard row block:
    rows (i - 1)M + 1 to iM in one-based notation. ``row_offset`` is the
    zero-based start of that block; the default is i=1. The decoder applies
    H_M^T to form the projection H_M^T H_M x. This is not an oracle waveform
    reconstruction, but it gives a length-N signal that can pass through the
    same GCC/WLS localization chain as DAE, DFT, and PCA.
    """

    def __init__(self, signal_len=1024, cr=16, row_mode="salari",
                 row_offset=0, seed=42):
        super().__init__()
        self.signal_len = int(signal_len)
        self.cr = int(cr)
        self.row_mode = str(row_mode).lower()
        self.row_offset = int(row_offset)
        self.seed = int(seed)
        self.feature_dim = latent_real_dim(self.signal_len, self.cr)
        self.rows_per_channel = max(1, self.feature_dim // 2)
        if self.rows_per_channel > self.signal_len:
            raise ValueError("Hadamard rows exceed signal length")
        h = _hadamard_matrix(self.signal_len)
        if self.row_mode in ("salari", "block", "first"):
            start = self.row_offset
            end = start + self.rows_per_channel
            if start < 0 or end > self.signal_len:
                raise ValueError("Invalid Hadamard row_offset for requested CR")
            row_idx = torch.arange(start, end, dtype=torch.long)
            self.row_block_index = start // self.rows_per_channel + 1
        elif self.row_mode in ("random", "random_fixed_seed"):
            row_idx = _select_indices(
                self.signal_len, self.rows_per_channel, "random", seed=self.seed
            )
            self.row_block_index = None
        elif self.row_mode in ("sequency", "sequency_low"):
            row_idx = _sequency_order(h)[:self.rows_per_channel].sort().values.long()
            self.row_block_index = None
        else:
            raise ValueError(f"Unknown Hadamard row_mode: {row_mode}")
        self.register_buffer("row_indices", row_idx.long().contiguous())
        self.register_buffer("rows", h[row_idx].contiguous())

    def _project_channel(self, x_ch):
        coeff = x_ch @ self.rows.t()
        return coeff @ self.rows

    def forward(self, x):
        real = self._project_channel(x[:, 0, :])
        imag = self._project_channel(x[:, 1, :])
        return torch.stack((real, imag), dim=1).float()


class PCABaseline(nn.Module):
    """Linear PCA compression baseline fitted on noisy received waveforms."""

    def __init__(self, mean, components, signal_len=1024, requested_components=None):
        super().__init__()
        self.signal_len = int(signal_len)
        self.requested_components = int(
            requested_components if requested_components is not None else components.shape[0]
        )
        self.feature_dim = int(components.shape[0])
        self.register_buffer("mean", mean.float().contiguous())
        self.register_buffer("components", components.float().contiguous())

    def forward(self, x):
        b = x.shape[0]
        flat = x.reshape(b, -1)
        centered = flat - self.mean
        coeff = centered @ self.components.t()
        recon = coeff @ self.components + self.mean
        return recon.reshape(b, 2, self.signal_len).float()


def fit_pca_baseline(waveforms, cr=16, seed=42, device=None, niter=3):
    signal_len = int(waveforms.shape[-1])
    requested = latent_real_dim(signal_len, cr)
    flat = waveforms.reshape(waveforms.shape[0], -1).float().cpu()
    if flat.shape[0] < 2:
        raise ValueError("PCA baseline requires at least two training samples")
    mean = flat.mean(dim=0)
    centered = flat - mean
    n_components = min(requested, flat.shape[0] - 1, flat.shape[1])
    torch.manual_seed(int(seed))
    _, _, v = torch.pca_lowrank(centered, q=n_components, center=False, niter=int(niter))
    components = v[:, :n_components].t().contiguous()
    model = PCABaseline(
        mean=mean,
        components=components,
        signal_len=signal_len,
        requested_components=requested,
    )
    if device is not None:
        model = model.to(device)
    return model


def _sample_pca_waveforms(simulator, n_samples, seed=42, source="noisy",
                          fixed_snr_db=0.0):
    source = str(source).lower()
    use_clean = source.startswith("clean")
    use_fixed_snr = "fixed" in source

    if not use_fixed_snr:
        x_noisy, x_clean = simulator.generate_training_dataset(int(n_samples), seed=int(seed))
        return x_clean if use_clean else x_noisy

    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    if simulator.scenario_mode == "urban8":
        snap_idx, uav_idx = simulator.build_urban_training_plan(int(n_samples), seed=int(seed))
        noisy_list, clean_list = [], []
        batch_cap = 500
        for start in range(0, int(n_samples), batch_cap):
            end = min(start + batch_cap, int(n_samples))
            Xn, Xc, _ = simulator.generate_urban_batch(
                end - start, snr_db=float(fixed_snr_db),
                snapshot_indices=snap_idx[start:end]
            )
            rows = torch.arange(end - start, dtype=torch.long)
            cols = torch.tensor(uav_idx[start:end], dtype=torch.long)
            noisy_list.append(Xn[rows, cols])
            clean_list.append(Xc[rows, cols])
        x_noisy = torch.cat(noisy_list, dim=0)
        x_clean = torch.cat(clean_list, dim=0)
        np.random.set_state(rng_state)
        return x_clean if use_clean else x_noisy

    noisy_list, clean_list = [], []
    batch_cap = 500
    for start in range(0, int(n_samples), batch_cap):
        end = min(start + batch_cap, int(n_samples))
        X1_n, X1_c, _, _, _, _ = simulator.generate_pair_batch(
            end - start, snr_db=float(fixed_snr_db)
        )
        noisy_list.append(X1_n)
        clean_list.append(X1_c)
    x_noisy = torch.cat(noisy_list, dim=0)
    x_clean = torch.cat(clean_list, dim=0)
    np.random.set_state(rng_state)
    return x_clean if use_clean else x_noisy


def _baseline_name(base, mode, main_name):
    mode = str(mode).lower()
    if base == "dft":
        if mode in ("scs_lite", "spectral_cs_lite", "partial_fourier",
                    "partial_fourier_uniform", "chen_dft", "uniform"):
            return "DFT-SCS-lite"
        if mode in ("fisher", "fisher_power", "tdoa_fisher", "rms_band", "rms_power", "high_fi"):
            return "DFT-Fisher"
        if mode in ("train_band", "train_contiguous", "train_power_band", "data_power_band"):
            return "DFT-train-band"
        if mode in ("train_power", "power", "data_power"):
            return "DFT-train-power"
        if mode in ("center", "bandlimited", "lowpass"):
            return "DFT-bandlimited"
        if mode in ("random", "random_fixed_seed"):
            return "DFT-random"
        return main_name
    if base == "hadamard":
        if mode in ("random", "random_fixed_seed"):
            return "Hadamard-random"
        if mode in ("sequency", "sequency_low"):
            return "Hadamard-sequency"
        return main_name
    return main_name


def build_traditional_baselines(simulator, cr=16, n_pca_samples=10000, seed=42,
                                device=None, dft_mode="uniform",
                                hadamard_mode="salari", pca_train_source="noisy",
                                pca_fixed_snr_db=0.0,
                                include_diagnostic_variants=False,
                                random_variant_seeds=5,
                                include_legacy_variants=False,
                                dft_lag_limit_samples=None,
                                include_strong_variants=False):
    """
    Build Fig.6 traditional baselines under the same real-scalar budget as DAE.

    PCA is fitted on noisy training waveforms to avoid clean-target oracle leakage.
    """
    signal_len = int(simulator.signal_len)
    baselines = OrderedDict()
    pca_waveforms = _sample_pca_waveforms(
        simulator, int(n_pca_samples), seed=int(seed),
        source=pca_train_source, fixed_snr_db=float(pca_fixed_snr_db)
    )
    dft_label = _baseline_name("dft", dft_mode, "DFT")
    had_label = _baseline_name("hadamard", hadamard_mode, "Hadamard")
    dft_selected_bins = None
    dft_mode_l = str(dft_mode).lower()
    dft_model_mode = "uniform" if dft_mode_l in (
        "scs_lite", "spectral_cs_lite", "partial_fourier",
        "partial_fourier_uniform", "chen_dft"
    ) else dft_mode
    if dft_mode_l in ("fisher", "fisher_power", "tdoa_fisher", "rms_band", "rms_power", "high_fi"):
        dft_selected_bins = select_balanced_fisher_dft_bins(
            pca_waveforms, signal_len=signal_len, cr=cr,
            lag_limit_samples=dft_lag_limit_samples
        )
    elif dft_mode_l in ("train_power", "power", "data_power"):
        dft_selected_bins = _top_power_bins(pca_waveforms, latent_real_dim(signal_len, cr) // 2)
    elif dft_mode_l in ("train_band", "train_contiguous", "train_power_band", "data_power_band"):
        dft_selected_bins = _contiguous_power_bins(
            pca_waveforms, latent_real_dim(signal_len, cr) // 2
        )
    baselines[dft_label] = DFTCompressionBaseline(
        signal_len=signal_len, cr=cr, mode=dft_model_mode, seed=seed + 11,
        selected_bins=dft_selected_bins
    ).to(device)
    baselines[had_label] = HadamardProjectionBaseline(
        signal_len=signal_len, cr=cr, row_mode=hadamard_mode, seed=seed + 23
    ).to(device)

    if include_diagnostic_variants:
        if dft_label != "DFT-SCS-lite":
            baselines["DFT-SCS-lite"] = DFTCompressionBaseline(
                signal_len=signal_len, cr=cr, mode="uniform", seed=seed + 27
            ).to(device)
        if dft_label != "DFT-Fisher":
            selected = select_balanced_fisher_dft_bins(
                pca_waveforms, signal_len=signal_len, cr=cr,
                lag_limit_samples=dft_lag_limit_samples
            )
            baselines["DFT-Fisher"] = DFTCompressionBaseline(
                signal_len=signal_len, cr=cr, mode="center", seed=seed + 28,
                selected_bins=selected
            ).to(device)
        if dft_label != "DFT-train-band":
            selected = _contiguous_power_bins(
                pca_waveforms, latent_real_dim(signal_len, cr) // 2
            )
            baselines["DFT-train-band"] = DFTCompressionBaseline(
                signal_len=signal_len, cr=cr, mode="center", seed=seed + 29,
                selected_bins=selected
            ).to(device)
        if dft_label != "DFT-train-power":
            selected = _top_power_bins(pca_waveforms, latent_real_dim(signal_len, cr) // 2)
            baselines["DFT-train-power"] = DFTCompressionBaseline(
                signal_len=signal_len, cr=cr, mode="center", seed=seed + 30,
                selected_bins=selected
            ).to(device)
        if dft_label != "DFT-bandlimited":
            baselines["DFT-bandlimited"] = DFTCompressionBaseline(
                signal_len=signal_len, cr=cr, mode="center", seed=seed + 31
            ).to(device)
        if include_legacy_variants and dft_label != "DFT-random":
            for ridx in range(int(random_variant_seeds)):
                baselines[f"DFT-random-{ridx + 1}"] = DFTCompressionBaseline(
                    signal_len=signal_len, cr=cr, mode="random",
                    seed=seed + 37 + 101 * ridx
                ).to(device)
        if include_legacy_variants and had_label != "Hadamard-random":
            for ridx in range(int(random_variant_seeds)):
                baselines[f"Hadamard-random-{ridx + 1}"] = HadamardProjectionBaseline(
                    signal_len=signal_len, cr=cr, row_mode="random",
                    seed=seed + 41 + 101 * ridx
                ).to(device)
        if include_legacy_variants and had_label != "Hadamard-sequency":
            baselines["Hadamard-sequency"] = HadamardProjectionBaseline(
                signal_len=signal_len, cr=cr, row_mode="sequency", seed=seed + 43
            ).to(device)
        offset = latent_real_dim(signal_len, cr) // 2
        if offset + (latent_real_dim(signal_len, cr) // 2) <= signal_len:
            block_index = offset // (latent_real_dim(signal_len, cr) // 2) + 1
            baselines[f"Hadamard-block-{block_index}"] = HadamardProjectionBaseline(
                signal_len=signal_len, cr=cr, row_mode="salari",
                row_offset=offset, seed=seed + 47
            ).to(device)

    if include_strong_variants and "DFT-Zhai-CRLB" not in baselines:
        selected = select_geo_ambiguity_dft_bins(
            pca_waveforms, simulator=simulator, signal_len=signal_len, cr=cr,
            lag_limit_samples=dft_lag_limit_samples
        )
        baselines["DFT-GeoAmbi"] = DFTCompressionBaseline(
            signal_len=signal_len, cr=cr, mode="center", seed=seed + 57,
            selected_bins=selected
        ).to(device)

        selected = select_cao2020_crb_dft_bins(
            pca_waveforms, signal_len=signal_len, cr=cr,
            lag_limit_samples=dft_lag_limit_samples
        )
        baselines["DFT-Cao2020-CRB"] = DFTCompressionBaseline(
            signal_len=signal_len, cr=cr, mode="center", seed=seed + 59,
            selected_bins=selected
        ).to(device)

        selected = select_zhai_crlb_dft_bins(
            pca_waveforms, signal_len=signal_len, cr=cr,
            lag_limit_samples=dft_lag_limit_samples
        )
        baselines["DFT-Zhai-CRLB"] = DFTCompressionBaseline(
            signal_len=signal_len, cr=cr, mode="center", seed=seed + 61,
            selected_bins=selected
        ).to(device)

    baselines["PCA"] = fit_pca_baseline(
        pca_waveforms, cr=cr, seed=int(seed), device=device
    )
    dft_selected_bins_by_method = {
        name: [int(v) for v in model.selected_bins.detach().cpu().tolist()]
        for name, model in baselines.items()
        if isinstance(model, DFTCompressionBaseline)
    }
    meta = {
        "baseline_cr": int(cr),
        "latent_real_dim": latent_real_dim(signal_len, cr),
        "dft_mode": str(dft_mode),
        "dft_complex_bins": baselines[dft_label].n_complex_bins,
        "dft_main_label": dft_label,
        "dft_train_selection_kind": (
            "balanced_fisher_lag_sidelobe" if dft_mode_l in (
                "fisher", "fisher_power", "tdoa_fisher", "rms_band", "rms_power", "high_fi"
            )
            else "top_k_power" if dft_mode_l in ("train_power", "power", "data_power")
            else "contiguous_power_band" if dft_mode_l in (
                "train_band", "train_contiguous", "train_power_band", "data_power_band"
            )
            else "uniform_partial_fourier_scs_lite" if dft_mode_l in (
                "scs_lite", "spectral_cs_lite", "partial_fourier",
                "partial_fourier_uniform", "chen_dft", "uniform"
            )
            else str(dft_mode)
        ),
        "dft_literature_basis": (
            "Chen Fig.6-style traditional DFT baseline is treated as fixed "
            "partial-Fourier transform coding under the same real-scalar budget. "
            "The main Fig6 curve estimates TDOA from selected-bin cross-spectra "
            "rather than zero-filled individual waveform reconstruction unless a "
            "source paper explicitly requires waveform reconstruction."
        ),
        "dft_oracle_level": (
            "no train-set target oracle for DFT-SCS-lite; train_power/train_band and "
            "fisher variants are diagnostic or task-aware ablations"
        ),
        "dft_waveform_reconstruction_role": (
            "supplement/diagnostic only; zero-filled IDFT of individual waveforms is "
            "not used as the strict Chen Fig6 DFT main curve"
        ),
        "dft_fisher_role": (
            "task-aware ablation; balanced Fisher/FIM bin selection with physical-lag "
            "sidelobe control, not the strict Chen-style DFT baseline"
        ),
        "dft_fisher_lag_limit_samples": (
            float(dft_lag_limit_samples) if dft_lag_limit_samples is not None else None
        ),
        "dft_fisher_selected_bins": (
            [int(v) for v in dft_selected_bins.tolist()]
            if dft_selected_bins is not None and dft_mode_l in (
                "fisher", "fisher_power", "tdoa_fisher", "rms_band", "rms_power", "high_fi"
            )
            else None
        ),
        "dft_train_power_selected_bins": (
            [int(v) for v in dft_selected_bins.tolist()]
            if dft_selected_bins is not None and dft_mode_l in ("train_power", "power", "data_power")
            else None
        ),
        "dft_train_band_selected_bins": (
            [int(v) for v in dft_selected_bins.tolist()]
            if dft_selected_bins is not None and dft_mode_l in (
                "train_band", "train_contiguous", "train_power_band", "data_power_band"
            )
            else None
        ),
        "dft_selected_bins_by_method": dft_selected_bins_by_method,
        "dft_geoambi_role": (
            "stage-A main candidate for the innovation baseline track; greedy "
            "CRLB/Fisher frequency selection with physical-lag sidelobe, "
            "geometry-weighted ambiguity, clustering, and sign-balance penalties"
        ),
        "dft_geoambi_selected_bins": dft_selected_bins_by_method.get("DFT-GeoAmbi"),
        "dft_zhai_crlb_role": (
            "strong task-aware baseline candidate; greedy CRLB/FIM spectral "
            "variance selection with physical-lag sidelobe control"
        ),
        "dft_cao2020_crb_role": (
            "strong task-aware diagnostic candidate; Cao-2020-style one-sided "
            "partial-Fourier CRB/high-frequency selection with training-power "
            "screening and physical-lag sidelobe control; evaluated by segmented "
            "incoherent high-FC delay scoring in main.py"
        ),
        "dft_cao2020_crb_selected_bins": dft_selected_bins_by_method.get("DFT-Cao2020-CRB"),
        "dft_zhai_crlb_selected_bins": dft_selected_bins_by_method.get("DFT-Zhai-CRLB"),
        "hadamard_mode": str(hadamard_mode),
        "hadamard_rows_per_channel": baselines[had_label].rows_per_channel,
        "hadamard_main_label": had_label,
        "hadamard_row_rule": (
            "Salari Lemma 1 contiguous row block: one-based rows "
            "(i-1)M+1 to iM; row_offset is the zero-based block start"
        ),
        "hadamard_main_block_index": getattr(baselines[had_label], "row_block_index", None),
        "pca_requested_components": baselines["PCA"].requested_components,
        "pca_actual_components": baselines["PCA"].feature_dim,
        "pca_training_samples": int(n_pca_samples),
        "pca_training_source": str(pca_train_source),
        "pca_fixed_snr_db": float(pca_fixed_snr_db),
        "hadamard_reconstruction": "H_M^T H_M projection",
        "include_diagnostic_variants": bool(include_diagnostic_variants),
        "include_legacy_variants": bool(include_legacy_variants),
        "include_strong_variants": bool(include_strong_variants),
        "random_variant_seeds": int(random_variant_seeds),
        "baseline_method_order": list(baselines.keys()),
    }
    return baselines, meta
