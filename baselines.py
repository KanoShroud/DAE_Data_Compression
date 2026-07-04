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
                                dft_lag_limit_samples=None):
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
        "random_variant_seeds": int(random_variant_seeds),
        "baseline_method_order": list(baselines.keys()),
    }
    return baselines, meta
