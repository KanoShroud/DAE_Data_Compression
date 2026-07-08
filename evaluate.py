# evaluate.py
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal, optimize


def gcc_phat(sig1, sig2):
    """
    广义互相关 (相位变换加权) - GCC-PHAT

    PHAT 加权将互功率谱归一化为单位幅度，仅保留相位信息。
    优点：多径环境下锐化相关峰。
    缺点：低 SNR 下噪声频段的相位被过度放大，且与 MSE 训练的 DAE 不兼容
          （DAE 优化幅度重建，PHAT 丢弃幅度信息）。
    """
    n = len(sig1)
    SIG1 = np.fft.fft(sig1, n=2 * n - 1)
    SIG2 = np.fft.fft(sig2, n=2 * n - 1)
    R = SIG1 * np.conj(SIG2)
    epsilon = np.max(np.abs(R)) * 1e-3
    R_weighted = R / (np.abs(R) + epsilon)
    cc = np.fft.fftshift(np.fft.ifft(R_weighted))
    start = (len(cc) - n) // 2
    return cc[start:start + n]


def gcc_standard(sig1, sig2):
    """
    标准广义互相关 (无加权) - Standard GCC

    直接使用互功率谱做逆傅里叶变换，保留复相关的幅度和相位信息。
    调用处统一使用 |GCC| 取峰，与 train.py 中的可微 GCC 定义一致。
    """
    n = len(sig1)
    SIG1 = np.fft.fft(sig1, n=2 * n - 1)
    SIG2 = np.fft.fft(sig2, n=2 * n - 1)
    R = SIG1 * np.conj(SIG2)
    cc = np.fft.fftshift(np.fft.ifft(R))
    start = (len(cc) - n) // 2
    return cc[start:start + n]


def _norm_abs_corr(corr):
    corr_abs = np.abs(corr)
    return corr_abs / (np.max(corr_abs) + 1e-9)


def _corr_peak_metrics(corr_norm, lags, true_tdoa, exclude_radius=2, false_threshold=0.5):
    corr_norm = np.asarray(corr_norm)
    lags = np.asarray(lags)
    true_tdoa = int(true_tdoa)
    peak_idx = int(np.argmax(corr_norm))
    peak_lag = int(lags[peak_idx])
    true_idx = int(np.argmin(np.abs(lags - true_tdoa)))
    near_true = np.abs(lags - true_tdoa) <= exclude_radius
    outside = ~near_true
    sidelobe = float(np.max(corr_norm[outside])) if np.any(outside) else 0.0
    true_amp = float(corr_norm[true_idx])
    peaks, props = signal.find_peaks(corr_norm, height=false_threshold)
    false_peaks = int(np.sum(np.abs(lags[peaks] - true_tdoa) > exclude_radius))
    return {
        'peak_lag': peak_lag,
        'peak_error': int(peak_lag - true_tdoa),
        'abs_peak_error': int(abs(peak_lag - true_tdoa)),
        'true_amp': true_amp,
        'sidelobe_ratio': float(sidelobe / (true_amp + 1e-9)),
        'false_peaks_gt_05': false_peaks,
        'num_peaks_gt_05': int(len(peaks)),
    }


def _summarize_peak_metrics(metrics):
    if not metrics:
        return {}
    abs_err = np.asarray([m['abs_peak_error'] for m in metrics], dtype=float)
    true_amp = np.asarray([m['true_amp'] for m in metrics], dtype=float)
    sidelobe = np.asarray([m['sidelobe_ratio'] for m in metrics], dtype=float)
    false_peaks = np.asarray([m['false_peaks_gt_05'] for m in metrics], dtype=float)
    return {
        'n': int(len(metrics)),
        'match_rate': float(np.mean(abs_err == 0)),
        'within_1_rate': float(np.mean(abs_err <= 1)),
        'within_2_rate': float(np.mean(abs_err <= 2)),
        'mean_abs_peak_error': float(np.mean(abs_err)),
        'median_abs_peak_error': float(np.median(abs_err)),
        'mean_true_amp': float(np.mean(true_amp)),
        'mean_sidelobe_ratio': float(np.mean(sidelobe)),
        'median_sidelobe_ratio': float(np.median(sidelobe)),
        'mean_false_peaks_gt_05': float(np.mean(false_peaks)),
    }


def _batch_peak_diagnostics(batch1, batch2, delays1, delays2, lags, gcc_func=gcc_standard):
    batch1 = np.asarray(batch1)
    batch2 = np.asarray(batch2)
    metrics = []
    for i in range(batch1.shape[0]):
        sig1 = batch1[i, 0, :] + 1j * batch1[i, 1, :]
        sig2 = batch2[i, 0, :] + 1j * batch2[i, 1, :]
        true_tdoa = int(delays1[i] - delays2[i])
        corr_norm = _norm_abs_corr(gcc_func(sig1, sig2))
        metrics.append(_corr_peak_metrics(corr_norm, lags, true_tdoa))
    return _summarize_peak_metrics(metrics)


def clean_peak_consistency(simulator, n_trials=256, snr_db=20, gcc_func=gcc_standard):
    """
    训练前数据/标签自检：统计 clean signal 的 GCC 峰是否落在标签 TDOA。

    paper_repro 模式依赖 LOS 标签。如果 clean GCC 与标签明显不一致，
    说明数据生成或多径设置仍不能作为可靠论文复现基线。
    """
    X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2 = simulator.generate_pair_batch(
        n_trials, snr_db=snr_db
    )
    del X1_noisy, X2_noisy
    lags = signal.correlation_lags(simulator.signal_len, simulator.signal_len, mode='same')
    metrics = _batch_peak_diagnostics(
        X1_clean.numpy(), X2_clean.numpy(), delays1, delays2, lags, gcc_func=gcc_func
    )
    metrics['snr_db'] = float(snr_db)
    return metrics


def _bootstrap_rmse_ci(se_values, rng, n_boot=200):
    se_values = np.asarray(se_values, dtype=float)
    if se_values.size == 0:
        return [float('nan'), float('nan')]
    if se_values.size == 1:
        rmse = float(np.sqrt(se_values[0]))
        return [rmse, rmse]
    idx = rng.integers(0, se_values.size, size=(n_boot, se_values.size))
    rmse = np.sqrt(np.mean(se_values[idx], axis=1))
    return [float(np.percentile(rmse, 2.5)), float(np.percentile(rmse, 97.5))]


def _lag_search_mask(lags, tdoa_lag_limit_samples=None):
    if tdoa_lag_limit_samples is None:
        return np.ones_like(lags, dtype=bool)
    return np.abs(np.asarray(lags, dtype=float)) <= float(tdoa_lag_limit_samples)


def _estimate_delay_from_corr(sig_i, sig_ref, lags, gcc_func, sub_sample=True,
                              return_quality=False, tdoa_lag_limit_samples=None):
    corr = np.abs(gcc_func(sig_i, sig_ref))
    search_mask = _lag_search_mask(lags, tdoa_lag_limit_samples)
    search_idx = np.where(search_mask)[0]
    if search_idx.size == 0:
        search_idx = np.arange(len(corr))
        search_mask = np.ones_like(corr, dtype=bool)
    idx = int(search_idx[np.argmax(corr[search_idx])])
    lag = float(lags[idx])
    if (sub_sample and 0 < idx < len(corr) - 1
            and search_mask[idx - 1] and search_mask[idx + 1]):
        y0, y1, y2 = corr[idx - 1], corr[idx], corr[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            lag += float(np.clip(delta, -0.5, 0.5))
    if not return_quality:
        return lag

    peak = float(corr[idx])
    mask = search_mask.copy()
    lo = max(0, idx - 2)
    hi = min(len(corr), idx + 3)
    mask[lo:hi] = False
    sidelobe = float(np.max(corr[mask])) if np.any(mask) else 0.0
    sidelobe_ratio = sidelobe / (peak + 1e-12)
    weight = float(np.clip(1.0 / (sidelobe_ratio + 1e-3), 0.05, 20.0))
    return lag, weight, peak, sidelobe_ratio


def _localize_from_tdoa(uav_pos, ref_idx, tdoa_samples, fs, c, area_size):
    uav_pos = np.asarray(uav_pos, dtype=float)
    ref = uav_pos[ref_idx]
    other_idx = [i for i in range(len(uav_pos)) if i != ref_idx and i in tdoa_samples]
    if len(other_idx) < 3:
        return None
    delta_ranges = np.asarray([tdoa_samples[i] * c / fs for i in other_idx], dtype=float)
    other_pos = uav_pos[other_idx]

    def residual(p):
        return (np.linalg.norm(p[None, :] - other_pos, axis=1)
                - np.linalg.norm(p - ref)
                - delta_ranges)

    x0 = np.mean(uav_pos[[ref_idx] + other_idx], axis=0)
    bounds = ([0.0, 0.0], [float(area_size[0]), float(area_size[1])])
    try:
        res = optimize.least_squares(residual, x0=x0, bounds=bounds, loss='soft_l1',
                                     max_nfev=100)
    except Exception:
        return None
    return res.x if np.all(np.isfinite(res.x)) else None


def _pair_geometry_metrics(p, pairs, uav_pos, sqrt_w):
    p = np.asarray(p, dtype=float)
    rows = []
    for row_w, (i, j, _, _) in zip(sqrt_w, pairs):
        vi = p - uav_pos[i]
        vj = p - uav_pos[j]
        ni = np.linalg.norm(vi)
        nj = np.linalg.norm(vj)
        if ni < 1e-9 or nj < 1e-9:
            continue
        rows.append(float(row_w) * (vi / ni - vj / nj))
    if len(rows) < 2:
        return float('inf'), float('inf')
    jmat = np.asarray(rows, dtype=float)
    try:
        svals = np.linalg.svd(jmat, compute_uv=False)
        cond = float(svals[0] / max(svals[-1], 1e-12))
        fisher = jmat.T @ jmat
        gdop = float(np.sqrt(np.trace(np.linalg.pinv(fisher))))
    except Exception:
        cond = float('inf')
        gdop = float('inf')
    return cond, gdop


def _localize_from_tdoa_pairs(uav_pos, pair_measurements, fs, c, area_size,
                              robust_mode="standard"):
    """
    WLS-style all-pair TDOA localization.

    pair_measurements contains (i, j, delay_i_minus_j_samples, weight). The residual
    is ||p-u_i|| - ||p-u_j|| - c*tau_ij.
    """
    if len(pair_measurements) < 3:
        return None, {'success': False, 'cost': float('inf'), 'n_pairs': len(pair_measurements)}

    uav_pos = np.asarray(uav_pos, dtype=float)
    pairs = [(int(i), int(j), float(tau), float(w)) for i, j, tau, w in pair_measurements
             if np.isfinite(tau) and np.isfinite(w) and w > 0]
    if len(pairs) < 3:
        return None, {'success': False, 'cost': float('inf'), 'n_pairs': len(pairs)}

    bounds = ([0.0, 0.0], [float(area_size[0]), float(area_size[1])])

    def solve(active_pairs, start_mode):
        idx_used = sorted(set([i for i, _, _, _ in active_pairs]
                              + [j for _, j, _, _ in active_pairs]))
        selected = uav_pos[idx_used]
        delta_ranges = np.asarray([tau * c / fs for _, _, tau, _ in active_pairs], dtype=float)
        weights = np.asarray([w for _, _, _, w in active_pairs], dtype=float)
        weights = weights / (np.median(weights) + 1e-12)
        sqrt_w = np.sqrt(np.clip(weights, 0.05, 20.0))

        def raw_residual(p):
            vals = []
            for (i, j, _, _), dr in zip(active_pairs, delta_ranges):
                vals.append(np.linalg.norm(p - uav_pos[i]) - np.linalg.norm(p - uav_pos[j]) - dr)
            return np.asarray(vals, dtype=float)

        def residual(p):
            return sqrt_w * raw_residual(p)

        starts = [
            np.mean(selected, axis=0),
            np.asarray([area_size[0] / 2.0, area_size[1] / 2.0], dtype=float),
        ]
        starts.extend(selected)
        starts.extend([
            np.asarray([0.15 * area_size[0], 0.15 * area_size[1]]),
            np.asarray([0.85 * area_size[0], 0.15 * area_size[1]]),
            np.asarray([0.15 * area_size[0], 0.85 * area_size[1]]),
            np.asarray([0.85 * area_size[0], 0.85 * area_size[1]]),
        ])
        if "grid" in str(start_mode):
            gx = np.linspace(0.1 * area_size[0], 0.9 * area_size[0], 5)
            gy = np.linspace(0.1 * area_size[1], 0.9 * area_size[1], 5)
            starts.extend(np.asarray([x, y], dtype=float) for x in gx for y in gy)

        candidates = []
        for x0 in starts:
            x0 = np.clip(np.asarray(x0, dtype=float), bounds[0], bounds[1])
            try:
                res = optimize.least_squares(residual, x0=x0, bounds=bounds, loss='linear',
                                             max_nfev=250)
            except Exception:
                continue
            if res.success and np.all(np.isfinite(res.x)):
                raw = raw_residual(res.x)
                candidates.append({
                    'x': np.asarray(res.x, dtype=float),
                    'cost': float(res.cost),
                    'normalized_cost': float(res.cost / max(len(active_pairs), 1)),
                    'raw_residual': raw,
                })
        if not candidates:
            return None, {
                'success': False, 'cost': float('inf'), 'n_pairs': len(active_pairs),
                'n_uavs': len(idx_used), 'candidate_count': 0,
            }
        candidates.sort(key=lambda d: d['cost'])
        best_item = candidates[0]
        raw = best_item['raw_residual']
        cond, gdop = _pair_geometry_metrics(best_item['x'], active_pairs, uav_pos, sqrt_w)
        tol = 1e-6
        boundary_hit = bool(
            np.any(best_item['x'] <= np.asarray(bounds[0]) + tol)
            or np.any(best_item['x'] >= np.asarray(bounds[1]) - tol)
        )
        second = candidates[1] if len(candidates) > 1 else None
        info = {
            'success': True,
            'cost': best_item['cost'],
            'normalized_cost': best_item['normalized_cost'],
            'n_pairs': len(active_pairs),
            'n_uavs': len(idx_used),
            'candidate_count': len(candidates),
            'residual_rmse_m': float(np.sqrt(np.mean(raw ** 2))) if raw.size else float('nan'),
            'mean_abs_residual_m': float(np.mean(np.abs(raw))) if raw.size else float('nan'),
            'max_abs_residual_m': float(np.max(np.abs(raw))) if raw.size else float('nan'),
            'geometry_condition': cond,
            'geometry_gdop': gdop,
            'boundary_hit': boundary_hit,
            'second_best_cost': float(second['cost']) if second is not None else float('nan'),
            'second_best_x': float(second['x'][0]) if second is not None else float('nan'),
            'second_best_y': float(second['x'][1]) if second is not None else float('nan'),
            'cost_gap': (
                float(second['cost'] - best_item['cost'])
                if second is not None else float('nan')
            ),
            'used_uav_indices': ";".join(str(i) for i in idx_used),
        }
        return best_item['x'], info

    mode = str(robust_mode or "standard").lower()
    best, info = solve(pairs, mode)
    if best is not None and "pruned" in mode and len(pairs) > 4:
        initial_info = dict(info)
        delta_ranges = np.asarray([tau * c / fs for _, _, tau, _ in pairs], dtype=float)
        raw_vals = []
        for (i, j, _, _), dr in zip(pairs, delta_ranges):
            raw_vals.append(np.linalg.norm(best - uav_pos[i]) - np.linalg.norm(best - uav_pos[j]) - dr)
        raw_vals = np.asarray(raw_vals, dtype=float)
        remove_n = min(max(1, int(np.ceil(0.15 * len(pairs)))), len(pairs) - 3)
        keep_idx = np.argsort(np.abs(raw_vals))[:len(pairs) - remove_n]
        pruned_pairs = [pairs[int(i)] for i in keep_idx]
        pruned_best, pruned_info = solve(pruned_pairs, mode)
        if pruned_best is not None:
            best = pruned_best
            info = dict(pruned_info)
            info['pruned_pairs_removed'] = int(remove_n)
            info['pre_prune_cost'] = float(initial_info.get('cost', float('nan')))
            info['pre_prune_normalized_cost'] = float(
                initial_info.get('normalized_cost', float('nan'))
            )
    else:
        info['pruned_pairs_removed'] = 0

    return best, info


def _summarize_errors(errors, requested_count):
    arr = np.asarray(errors, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            'rmse': float('nan'), 'median': float('nan'), 'trimmed_rmse': float('nan'),
            'p90': float('nan'), 'p95': float('nan'), 'max': float('nan'),
            'valid_count': 0, 'requested_count': int(requested_count), 'failure_rate': 1.0,
            'outlier_gt20_rate': float('nan'), 'outlier_gt50_rate': float('nan'),
            'se': [],
        }
    p95 = float(np.percentile(finite, 95))
    trimmed = finite[finite <= p95]
    return {
        'rmse': float(np.sqrt(np.mean(finite ** 2))),
        'median': float(np.median(finite)),
        'trimmed_rmse': float(np.sqrt(np.mean(trimmed ** 2))) if trimmed.size else float('nan'),
        'p90': float(np.percentile(finite, 90)),
        'p95': p95,
        'max': float(np.max(finite)),
        'valid_count': int(finite.size),
        'requested_count': int(requested_count),
        'failure_rate': float(1.0 - finite.size / max(int(requested_count), 1)),
        'outlier_gt20_rate': float(np.mean(finite > 20.0)),
        'outlier_gt50_rate': float(np.mean(finite > 50.0)),
        'se': [float(v) for v in finite ** 2],
    }


def _pack_xy(points):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return ""
    return ";".join(f"{float(x):.3f},{float(y):.3f}" for x, y in arr)


def _pack_pair_field(pair_details, key, fmt="{:.6g}"):
    values = []
    for item in pair_details or []:
        value = item.get(key, "")
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value):
                values.append(fmt.format(float(value)))
            else:
                values.append("nan")
        else:
            values.append(str(value))
    return ";".join(values)


def _localization_detail(sample_index, error_m, est, meta, los_idx, pair_abs_errors,
                         pair_sidelobe_ratios, loc_info, pair_details=None):
    pair_abs_errors = np.asarray(pair_abs_errors, dtype=float)
    pair_abs_errors = pair_abs_errors[np.isfinite(pair_abs_errors)]
    pair_sidelobe_ratios = np.asarray(pair_sidelobe_ratios, dtype=float)
    pair_sidelobe_ratios = pair_sidelobe_ratios[np.isfinite(pair_sidelobe_ratios)]
    src = np.asarray(meta['source'][sample_index], dtype=float)
    est = np.asarray(est, dtype=float)
    loc_info = loc_info or {}

    def _stat(values, fn):
        return float(fn(values)) if values.size else float('nan')

    return {
        'sample_index': int(sample_index),
        'error_m': float(error_m),
        'source_x': float(src[0]),
        'source_y': float(src[1]),
        'estimated_x': float(est[0]),
        'estimated_y': float(est[1]),
        'los_count': int(len(los_idx)),
        'los_uav_indices': ";".join(str(int(v)) for v in los_idx),
        'los_uav_xy': _pack_xy(np.asarray(meta['uavs'][sample_index], dtype=float)[los_idx]),
        'used_uav_indices': str(loc_info.get('used_uav_indices', "")),
        'n_pairs': int(loc_info.get('n_pairs', len(pair_abs_errors))),
        'n_uavs': int(loc_info.get('n_uavs', len(los_idx))),
        'wls_cost': float(loc_info.get('cost', float('nan'))),
        'wls_normalized_cost': float(loc_info.get('normalized_cost', float('nan'))),
        'residual_rmse_m': float(loc_info.get('residual_rmse_m', float('nan'))),
        'mean_abs_residual_m': float(loc_info.get('mean_abs_residual_m', float('nan'))),
        'max_abs_residual_m': float(loc_info.get('max_abs_residual_m', float('nan'))),
        'geometry_condition': float(loc_info.get('geometry_condition', float('nan'))),
        'geometry_gdop': float(loc_info.get('geometry_gdop', float('nan'))),
        'boundary_hit': int(bool(loc_info.get('boundary_hit', False))),
        'candidate_count': int(loc_info.get('candidate_count', 0)),
        'second_best_cost': float(loc_info.get('second_best_cost', float('nan'))),
        'second_best_x': float(loc_info.get('second_best_x', float('nan'))),
        'second_best_y': float(loc_info.get('second_best_y', float('nan'))),
        'cost_gap': float(loc_info.get('cost_gap', float('nan'))),
        'pruned_pairs_removed': int(loc_info.get('pruned_pairs_removed', 0)),
        'pre_prune_cost': float(loc_info.get('pre_prune_cost', float('nan'))),
        'pre_prune_normalized_cost': float(
            loc_info.get('pre_prune_normalized_cost', float('nan'))
        ),
        'max_pair_abs_tdoa_error_samples': _stat(pair_abs_errors, np.max),
        'mean_pair_abs_tdoa_error_samples': _stat(pair_abs_errors, np.mean),
        'median_pair_abs_tdoa_error_samples': _stat(pair_abs_errors, np.median),
        'mean_pair_sidelobe_ratio': _stat(pair_sidelobe_ratios, np.mean),
        'pair_uav_indices': _pack_pair_field(pair_details, 'pair'),
        'pair_true_tdoa_samples': _pack_pair_field(pair_details, 'true_tdoa'),
        'pair_est_tdoa_samples': _pack_pair_field(pair_details, 'est_tdoa'),
        'pair_tdoa_error_samples': _pack_pair_field(pair_details, 'tdoa_error'),
        'pair_weight': _pack_pair_field(pair_details, 'weight'),
        'pair_sidelobe_ratio': _pack_pair_field(pair_details, 'sidelobe_ratio'),
    }


def _top_worst_details(details, limit=10):
    valid = [d for d in details if np.isfinite(d.get('error_m', float('nan')))]
    valid.sort(key=lambda d: d.get('error_m', float('-inf')), reverse=True)
    return valid[:int(limit)]


def _all_pair_robust_mode(estimator):
    if estimator == "all_pair_wls":
        return "standard"
    if estimator in {
        "all_pair_wls_grid",
        "all_pair_wls_pruned",
        "all_pair_wls_grid_pruned",
    }:
        raise ValueError(
            f"{estimator} was removed from the active evaluation path. "
            "Fig10 showed it is not a generally valid replacement for all_pair_wls."
        )
    return None


def _localization_errors_from_batch(batch_np, meta, fs, c, area_size, gcc_func,
                                    sub_sample=True, use_los_only=True,
                                    estimator="all_pair_wls", oracle_geometry=False,
                                    tdoa_lag_limit_samples=None,
                                    return_details=False):
    errors = []
    failures = 0
    details = []
    for bi in range(batch_np.shape[0]):
        los_idx = np.where(meta['los'][bi])[0] if use_los_only else np.arange(batch_np.shape[1])
        if len(los_idx) < 4:
            failures += 1
            continue
        lags = signal.correlation_lags(batch_np.shape[-1], batch_np.shape[-1], mode='same')
        pair_measurements = []
        los_idx = [int(v) for v in los_idx]
        pair_abs_errors = []
        pair_sidelobe_ratios = []
        pair_details = []
        for a in range(len(los_idx)):
            for b in range(a + 1, len(los_idx)):
                ui, uj = los_idx[a], los_idx[b]
                true_tau = (meta['distances'][bi, ui] - meta['distances'][bi, uj]) / (c / fs)
                if oracle_geometry:
                    tau = true_tau
                    weight = 20.0
                    sidelobe_ratio = float('nan')
                else:
                    sig_i = batch_np[bi, ui, 0, :] + 1j * batch_np[bi, ui, 1, :]
                    sig_j = batch_np[bi, uj, 0, :] + 1j * batch_np[bi, uj, 1, :]
                    tau, weight, _, sidelobe_ratio = _estimate_delay_from_corr(
                        sig_i, sig_j, lags, gcc_func, sub_sample=sub_sample,
                        return_quality=True,
                        tdoa_lag_limit_samples=tdoa_lag_limit_samples
                    )
                pair_measurements.append((ui, uj, tau, weight))
                tdoa_error = float(tau - true_tau)
                pair_abs_errors.append(float(abs(tdoa_error)))
                pair_sidelobe_ratios.append(float(sidelobe_ratio))
                pair_details.append({
                    'pair': f"{ui}-{uj}",
                    'true_tdoa': float(true_tau),
                    'est_tdoa': float(tau),
                    'tdoa_error': tdoa_error,
                    'weight': float(weight),
                    'sidelobe_ratio': float(sidelobe_ratio),
                })

        robust_mode = _all_pair_robust_mode(estimator)
        if robust_mode is None:
            ref_idx = int(los_idx[0])
            ref_sig = batch_np[bi, ref_idx, 0, :] + 1j * batch_np[bi, ref_idx, 1, :]
            tdoa = {}
            for ui in los_idx:
                if ui == ref_idx:
                    continue
                sig_i = batch_np[bi, ui, 0, :] + 1j * batch_np[bi, ui, 1, :]
                tdoa[ui] = _estimate_delay_from_corr(sig_i, ref_sig, lags, gcc_func,
                                                      sub_sample=sub_sample,
                                                      tdoa_lag_limit_samples=tdoa_lag_limit_samples)
            est = _localize_from_tdoa(meta['uavs'][bi], ref_idx, tdoa, fs, c, area_size)
            loc_info = {'success': est is not None, 'cost': float('nan'),
                        'n_pairs': max(len(tdoa), 0), 'n_uavs': len(los_idx)}
        else:
            est, loc_info = _localize_from_tdoa_pairs(meta['uavs'][bi], pair_measurements,
                                                      fs, c, area_size,
                                                      robust_mode=robust_mode)
        if est is None:
            failures += 1
            continue
        err = float(np.linalg.norm(est - meta['source'][bi]))
        errors.append(err)
        if return_details:
            details.append(_localization_detail(
                bi, err, est, meta, los_idx, pair_abs_errors, pair_sidelobe_ratios,
                loc_info, pair_details=pair_details
            ))
    if return_details:
        return errors, details
    return errors


def _localization_errors_from_direct_estimator(batch_np, meta, fs, c, area_size,
                                               direct_estimator, sub_sample=True,
                                               use_los_only=True,
                                               estimator="all_pair_wls",
                                               tdoa_lag_limit_samples=None,
                                               return_details=False):
    errors = []
    details = []
    for bi in range(batch_np.shape[0]):
        los_idx = np.where(meta['los'][bi])[0] if use_los_only else np.arange(batch_np.shape[1])
        los_idx = [int(v) for v in los_idx]
        if len(los_idx) < 4:
            continue

        robust_mode = _all_pair_robust_mode(estimator)
        pair_abs_errors = []
        pair_sidelobe_ratios = []
        pair_details = []
        loc_info = {'success': False, 'cost': float('nan'),
                    'n_pairs': 0, 'n_uavs': len(los_idx)}
        if robust_mode is None:
            ref_idx = int(los_idx[0])
            ref_sig = batch_np[bi, ref_idx, 0, :] + 1j * batch_np[bi, ref_idx, 1, :]
            tdoa = {}
            for ui in los_idx:
                if ui == ref_idx:
                    continue
                sig_i = batch_np[bi, ui, 0, :] + 1j * batch_np[bi, ui, 1, :]
                tdoa[ui] = direct_estimator.estimate_pair(
                    sig_i, ref_sig, sub_sample=sub_sample, return_quality=False,
                    lag_limit_samples=tdoa_lag_limit_samples
                )
                true_tau = (meta['distances'][bi, ui] - meta['distances'][bi, ref_idx]) / (c / fs)
                tau = float(tdoa[ui])
                pair_abs_errors.append(float(abs(tau - true_tau)))
                pair_sidelobe_ratios.append(float('nan'))
                pair_details.append({
                    'pair': f"{ui}-{ref_idx}",
                    'true_tdoa': float(true_tau),
                    'est_tdoa': tau,
                    'tdoa_error': float(tau - true_tau),
                    'weight': float('nan'),
                    'sidelobe_ratio': float('nan'),
                })
            est = _localize_from_tdoa(meta['uavs'][bi], ref_idx, tdoa, fs, c, area_size)
            loc_info = {'success': est is not None, 'cost': float('nan'),
                        'n_pairs': len(tdoa), 'n_uavs': len(los_idx)}
        else:
            pair_measurements = []
            for a in range(len(los_idx)):
                for b in range(a + 1, len(los_idx)):
                    ui, uj = los_idx[a], los_idx[b]
                    true_tau = (meta['distances'][bi, ui] - meta['distances'][bi, uj]) / (c / fs)
                    sig_i = batch_np[bi, ui, 0, :] + 1j * batch_np[bi, ui, 1, :]
                    sig_j = batch_np[bi, uj, 0, :] + 1j * batch_np[bi, uj, 1, :]
                    tau, weight, _, sidelobe_ratio = direct_estimator.estimate_pair(
                        sig_i, sig_j, sub_sample=sub_sample, return_quality=True,
                        lag_limit_samples=tdoa_lag_limit_samples
                    )
                    pair_measurements.append((ui, uj, tau, weight))
                    tdoa_error = float(tau - true_tau)
                    pair_abs_errors.append(float(abs(tdoa_error)))
                    pair_sidelobe_ratios.append(float(sidelobe_ratio))
                    pair_details.append({
                        'pair': f"{ui}-{uj}",
                        'true_tdoa': float(true_tau),
                        'est_tdoa': float(tau),
                        'tdoa_error': tdoa_error,
                        'weight': float(weight),
                        'sidelobe_ratio': float(sidelobe_ratio),
                    })
            est, loc_info = _localize_from_tdoa_pairs(
                meta['uavs'][bi], pair_measurements, fs, c, area_size,
                robust_mode=robust_mode
            )
        if est is None:
            continue
        err = float(np.linalg.norm(est - meta['source'][bi]))
        errors.append(err)
        if return_details:
            details.append(_localization_detail(
                bi, err, est, meta, los_idx, pair_abs_errors, pair_sidelobe_ratios,
                loc_info, pair_details=pair_details
            ))
    if return_details:
        return errors, details
    return errors


def _corr_half_width(corr_norm, peak_idx):
    corr_norm = np.asarray(corr_norm, dtype=float)
    if corr_norm.size == 0:
        return float('nan')
    peak = float(corr_norm[int(peak_idx)])
    threshold = 0.5 * peak
    left = int(peak_idx)
    right = int(peak_idx)
    while left > 0 and corr_norm[left - 1] >= threshold:
        left -= 1
    while right < corr_norm.size - 1 and corr_norm[right + 1] >= threshold:
        right += 1
    return float(right - left + 1)


def _urban_waveform_diagnostics(batch_np, clean_np, meta, fs, c, gcc_func,
                                sub_sample=True, use_los_only=True,
                                tdoa_lag_limit_samples=None):
    """
    Diagnostics for waveform-output Fig.6 methods under the same urban batch.

    The metrics are not used by the localization solver. They explain whether a
    method fails because of waveform reconstruction, spectral distortion, or GCC
    peak structure.
    """
    batch_np = np.asarray(batch_np, dtype=np.float32)
    clean_np = np.asarray(clean_np, dtype=np.float32)
    y = batch_np[:, :, 0, :] + 1j * batch_np[:, :, 1, :]
    clean = clean_np[:, :, 0, :] + 1j * clean_np[:, :, 1, :]
    sig_power = float(np.mean(np.abs(clean) ** 2)) + 1e-9
    complex_nmse = float(np.mean(np.abs(y - clean) ** 2) / sig_power)
    mag_nmse = float(np.mean((np.abs(y) - np.abs(clean)) ** 2) / sig_power)

    psd_mse = []
    psd_active_mse = []
    psd_linear_nmse = []
    # The system bandwidth is close to the full Nyquist span for Fs=40 MHz; keep
    # the mask explicit so later bandwidth changes remain auditable.
    f_ref = None
    for bi in range(batch_np.shape[0]):
        for ui in range(batch_np.shape[1]):
            f_y, p_y = signal.periodogram(y[bi, ui], fs=fs, return_onesided=False,
                                           scaling='density', detrend=False)
            _, p_c = signal.periodogram(clean[bi, ui], fs=fs, return_onesided=False,
                                         scaling='density', detrend=False)
            if f_ref is None:
                f_ref = np.fft.fftshift(f_y)
                band_mask = np.abs(f_ref) <= (fs / 2.0)
            p_y_db = 10 * np.log10(np.fft.fftshift(p_y) + 1e-30)
            p_c_db = 10 * np.log10(np.fft.fftshift(p_c) + 1e-30)
            psd_mse.append(float(np.mean((p_y_db[band_mask] - p_c_db[band_mask]) ** 2)))
            p_y_shift = np.fft.fftshift(p_y)
            p_c_shift = np.fft.fftshift(p_c)
            active = p_c_shift >= (np.max(p_c_shift) * 1e-4)
            if np.any(active):
                psd_active_mse.append(float(np.mean((p_y_db[active] - p_c_db[active]) ** 2)))
            psd_linear_nmse.append(float(
                np.mean((p_y_shift - p_c_shift) ** 2) / (np.mean(p_c_shift ** 2) + 1e-30)
            ))

    lags = signal.correlation_lags(batch_np.shape[-1], batch_np.shape[-1], mode='same')
    lag_mask = _lag_search_mask(lags, tdoa_lag_limit_samples)
    abs_tdoa_err = []
    tdoa_weights = []
    sidelobe_ratio = []
    peak_width = []
    false_peaks = []
    peak_count = []
    for bi in range(batch_np.shape[0]):
        los_idx = np.where(meta['los'][bi])[0] if use_los_only else np.arange(batch_np.shape[1])
        los_idx = [int(v) for v in los_idx]
        for a in range(len(los_idx)):
            for b in range(a + 1, len(los_idx)):
                ui, uj = los_idx[a], los_idx[b]
                sig_i = y[bi, ui]
                sig_j = y[bi, uj]
                tau, weight, _, side = _estimate_delay_from_corr(
                    sig_i, sig_j, lags, gcc_func, sub_sample=sub_sample,
                    return_quality=True,
                    tdoa_lag_limit_samples=tdoa_lag_limit_samples
                )
                true_tau = ((meta['distances'][bi, ui] - meta['distances'][bi, uj])
                            / (c / fs))
                corr_abs = np.abs(gcc_func(sig_i, sig_j))
                gated_peak = np.max(corr_abs[lag_mask]) if np.any(lag_mask) else np.max(corr_abs)
                corr_norm = corr_abs / (gated_peak + 1e-9)
                search_idx = np.where(lag_mask)[0] if np.any(lag_mask) else np.arange(len(lags))
                peak_idx = int(search_idx[np.argmax(corr_norm[search_idx])])
                peaks, _ = signal.find_peaks(corr_norm, height=0.5)
                peaks = peaks[lag_mask[peaks]]
                err = float(abs(tau - true_tau))
                abs_tdoa_err.append(err)
                tdoa_weights.append(float(weight))
                sidelobe_ratio.append(float(side))
                peak_width.append(_corr_half_width(corr_norm, peak_idx))
                false_peaks.append(float(np.sum(np.abs(lags[peaks] - true_tau) > 2.0)))
                peak_count.append(float(len(peaks)))

    def _mean(values):
        values = np.asarray(values, dtype=float)
        return float(np.mean(values)) if values.size else float('nan')

    err_arr = np.asarray(abs_tdoa_err, dtype=float)
    weight_arr = np.asarray(tdoa_weights, dtype=float)
    if err_arr.size and np.sum(weight_arr) > 0:
        weighted_err = float(np.sum(weight_arr * err_arr) / (np.sum(weight_arr) + 1e-12))
    else:
        weighted_err = float('nan')

    return {
        "complex_nmse": complex_nmse,
        "mag_nmse": mag_nmse,
        "psd_db_mse": _mean(psd_mse),
        "psd_active_db_mse": _mean(psd_active_mse),
        "psd_linear_nmse": _mean(psd_linear_nmse),
        "gcc_mean_abs_tdoa_error_samples": _mean(abs_tdoa_err),
        "gcc_weighted_abs_tdoa_error_samples": weighted_err,
        "gcc_median_abs_tdoa_error_samples": float(np.median(abs_tdoa_err)) if abs_tdoa_err else float('nan'),
        "gcc_within_1_sample_rate": float(np.mean(err_arr <= 1.0)) if err_arr.size else float('nan'),
        "gcc_within_2_sample_rate": float(np.mean(err_arr <= 2.0)) if err_arr.size else float('nan'),
        "gcc_mean_sidelobe_ratio": _mean(sidelobe_ratio),
        "gcc_mean_peak_width_samples": _mean(peak_width),
        "gcc_mean_false_peaks_gt_05": _mean(false_peaks),
        "gcc_mean_num_peaks_gt_05": _mean(peak_count),
        "gcc_lag_limit_samples": (
            float(tdoa_lag_limit_samples) if tdoa_lag_limit_samples is not None else float('nan')
        ),
    }


class UrbanLocalizationExperiment:
    """
    论文近似复现评估：8 UAV + oracle LOS 选择 + 多 TDOA 几何定位误差。
    """

    def __init__(self, models_dict, simulator, device, seed=None, gcc_method='standard',
                 snr_range=None, num_trials=200, sub_sample=True, use_los_only=True,
                 batch_size=64, fixed_eval_set=True, estimator="all_pair_wls",
                 tdoa_lag_limit_samples=None):
        self.models_dict = models_dict
        self.sim = simulator
        self.device = device
        self.seed = seed
        self.snr_range = np.asarray(snr_range if snr_range is not None else np.arange(-10, 21, 2))
        self.num_trials = int(num_trials)
        self.sub_sample = bool(sub_sample)
        self.use_los_only = bool(use_los_only)
        self.batch_size = int(batch_size)
        self.fixed_eval_set = bool(fixed_eval_set)
        self.estimator = estimator
        self.tdoa_lag_limit_samples = (
            None if tdoa_lag_limit_samples is None
            else float(tdoa_lag_limit_samples)
        )
        self.gcc_func = gcc_standard if gcc_method == 'standard' else gcc_phat
        self.gcc_method = gcc_method

    def _run_models(self, X_noisy):
        n_obs, n_uav = X_noisy.shape[:2]
        flat = X_noisy.reshape(n_obs * n_uav, 2, self.sim.signal_len).to(self.device)
        outputs = {}
        for cr, model in self.models_dict.items():
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, flat.shape[0], self.batch_size):
                    chunks.append(model(flat[start:start + self.batch_size]).cpu())
            outputs[cr] = torch.cat(chunks, dim=0).reshape(n_obs, n_uav, 2, self.sim.signal_len).numpy()
        return outputs

    def run(self):
        print(f"Running Urban Localization Sweep (GCC method: {self.gcc_method}, "
              f"sub_sample={self.sub_sample}, LOS-only={self.use_los_only}, "
              f"lag_limit={self.tdoa_lag_limit_samples})...")
        if self.seed is not None:
            np.random.seed(self.seed)
            print(f"  Urban localization random seed set to: {self.seed}")

        ci_rng = np.random.default_rng(self.seed)
        results = {
            'metric': 'localization_m',
            'raw': [], 'raw_med': [],
            'clean': [], 'clean_med': [],
            'raw_trimmed': [], 'clean_trimmed': [],
            'raw_p90': [], 'clean_p90': [],
            'raw_p95': [], 'clean_p95': [],
            'raw_max': [], 'clean_max': [],
            'raw_valid_count': [], 'clean_valid_count': [],
            'raw_failure_rate': [], 'clean_failure_rate': [],
            'geom': [], 'geom_med': [], 'geom_trimmed': [],
            'trial_se': {'raw': [], 'clean': [], 'dae': {}},
            'rmse_ci95': {'raw': [], 'clean': [], 'dae': {}},
            'localization_config': {
                'gcc_method': self.gcc_method,
                'sub_sample': self.sub_sample,
                'use_los_only': self.use_los_only,
                'n_uavs': self.sim.n_uavs,
                'area_size': self.sim.area_size,
                'fixed_eval_set': self.fixed_eval_set,
                'estimator': self.estimator,
                'tdoa_lag_limit_samples': self.tdoa_lag_limit_samples,
            },
        }
        for cr in self.models_dict.keys():
            results[f'dae_{cr}'] = []
            results[f'dae_{cr}_med'] = []
            results[f'dae_{cr}_trimmed'] = []
            results[f'dae_{cr}_p90'] = []
            results[f'dae_{cr}_p95'] = []
            results[f'dae_{cr}_max'] = []
            results[f'dae_{cr}_valid_count'] = []
            results[f'dae_{cr}_failure_rate'] = []
            results['trial_se']['dae'][str(cr)] = []
            results['rmse_ci95']['dae'][str(cr)] = []

        for snr in self.snr_range:
            eval_seed = self.seed if self.fixed_eval_set else None
            X_noisy, X_clean, meta = self.sim.generate_urban_batch(
                self.num_trials, snr_db=float(snr), seed=eval_seed
            )
            raw_np = X_noisy.numpy()
            clean_np = X_clean.numpy()
            geom_err = _localization_errors_from_batch(
                clean_np, meta, self.sim.fs, self.sim.c, self.sim.area_size, self.gcc_func,
                sub_sample=self.sub_sample, use_los_only=self.use_los_only,
                estimator=self.estimator, oracle_geometry=True,
                tdoa_lag_limit_samples=self.tdoa_lag_limit_samples
            )
            raw_err = _localization_errors_from_batch(
                raw_np, meta, self.sim.fs, self.sim.c, self.sim.area_size, self.gcc_func,
                sub_sample=self.sub_sample, use_los_only=self.use_los_only,
                estimator=self.estimator,
                tdoa_lag_limit_samples=self.tdoa_lag_limit_samples
            )
            clean_err = _localization_errors_from_batch(
                clean_np, meta, self.sim.fs, self.sim.c, self.sim.area_size, self.gcc_func,
                sub_sample=self.sub_sample, use_los_only=self.use_los_only,
                estimator=self.estimator,
                tdoa_lag_limit_samples=self.tdoa_lag_limit_samples
            )
            geom_stats = _summarize_errors(geom_err, self.num_trials)
            raw_stats = _summarize_errors(raw_err, self.num_trials)
            clean_stats = _summarize_errors(clean_err, self.num_trials)
            for prefix, stats in [('geom', geom_stats), ('raw', raw_stats), ('clean', clean_stats)]:
                results[prefix].append(stats['rmse'])
                if prefix in ('raw', 'clean'):
                    results[f'{prefix}_med'].append(stats['median'])
                    results[f'{prefix}_trimmed'].append(stats['trimmed_rmse'])
                    results[f'{prefix}_p90'].append(stats['p90'])
                    results[f'{prefix}_p95'].append(stats['p95'])
                    results[f'{prefix}_max'].append(stats['max'])
                    results[f'{prefix}_valid_count'].append(stats['valid_count'])
                    results[f'{prefix}_failure_rate'].append(stats['failure_rate'])
                    results['trial_se'][prefix].append(stats['se'])
                    results['rmse_ci95'][prefix].append(
                        _bootstrap_rmse_ci(np.asarray(stats['se'], dtype=float), ci_rng)
                    )
                else:
                    results['geom_med'].append(stats['median'])
                    results['geom_trimmed'].append(stats['trimmed_rmse'])

            dae_outputs = self._run_models(X_noisy)
            for cr, y_np in dae_outputs.items():
                err = _localization_errors_from_batch(
                    y_np, meta, self.sim.fs, self.sim.c, self.sim.area_size, self.gcc_func,
                    sub_sample=self.sub_sample, use_los_only=self.use_los_only,
                    estimator=self.estimator,
                    tdoa_lag_limit_samples=self.tdoa_lag_limit_samples
                )
                stats = _summarize_errors(err, self.num_trials)
                results[f'dae_{cr}'].append(stats['rmse'])
                results[f'dae_{cr}_med'].append(stats['median'])
                results[f'dae_{cr}_trimmed'].append(stats['trimmed_rmse'])
                results[f'dae_{cr}_p90'].append(stats['p90'])
                results[f'dae_{cr}_p95'].append(stats['p95'])
                results[f'dae_{cr}_max'].append(stats['max'])
                results[f'dae_{cr}_valid_count'].append(stats['valid_count'])
                results[f'dae_{cr}_failure_rate'].append(stats['failure_rate'])
                results['trial_se']['dae'][str(cr)].append(stats['se'])
                results['rmse_ci95']['dae'][str(cr)].append(
                    _bootstrap_rmse_ci(np.asarray(stats['se'], dtype=float), ci_rng)
                )

        return results, self.snr_range


def _run_named_models(models_dict, X_noisy, simulator, device, batch_size=64):
    n_obs, n_uav = X_noisy.shape[:2]
    flat = X_noisy.reshape(n_obs * n_uav, 2, simulator.signal_len).to(device)
    outputs = {}
    for label, model in models_dict.items():
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, flat.shape[0], int(batch_size)):
                y = model(flat[start:start + int(batch_size)])
                chunks.append(y.detach().cpu())
        outputs[label] = torch.cat(chunks, dim=0).reshape(
            n_obs, n_uav, 2, simulator.signal_len
        ).numpy()
    return outputs


def run_urban_method_comparison(models_dict, simulator, device, seed=None,
                                snr_range=None, num_trials=200, sub_sample=True,
                                use_los_only=True, batch_size=64,
                                fixed_eval_set=True, estimator="all_pair_wls",
                                gcc_method="standard", title="Fig6",
                                direct_estimators=None,
                                tdoa_lag_limit_samples=None):
    """
    Generic urban localization comparison by method name.

    This is used for Chen-2025-style Fig.6 baselines and task-aware comparison
    tracks. Waveform methods output shape (B, 2, signal_len) and share the same
    GCC + WLS localization chain. Direct estimators are allowed only when the
    source method estimates TDOA from compressed coefficients by design; they
    return pairwise delays and then use the same all-pair WLS localization
    solver as the waveform methods.
    """
    snr_range = np.asarray(snr_range if snr_range is not None else np.arange(-10, 21, 2))
    gcc_func = gcc_standard if gcc_method == 'standard' else gcc_phat
    direct_estimators = direct_estimators or {}
    method_order = ["Raw", "Clean", "Geometry"] + list(models_dict.keys()) + list(direct_estimators.keys())
    results = {
        "metric": "localization_m",
        "title": title,
        "method_order": method_order,
        "methods": {
            label: {
                "rmse": [], "median": [], "trimmed_rmse": [],
                "p90": [], "p95": [], "max": [],
                "valid_count": [], "failure_rate": [],
                "outlier_gt20_rate": [], "outlier_gt50_rate": [],
            }
            for label in method_order
        },
        "method_diagnostics": {label: {} for label in method_order},
        "outlier_details": {label: [] for label in method_order},
        "config": {
            "gcc_method": gcc_method,
            "sub_sample": bool(sub_sample),
            "use_los_only": bool(use_los_only),
            "n_uavs": simulator.n_uavs,
            "area_size": simulator.area_size,
            "fixed_eval_set": bool(fixed_eval_set),
            "estimator": estimator,
            "num_trials": int(num_trials),
            "tdoa_lag_limit_samples": (
                float(tdoa_lag_limit_samples)
                if tdoa_lag_limit_samples is not None else None
            ),
        },
    }

    print(f"Running {title} method comparison "
          f"(waveform_methods={list(models_dict.keys())}, "
          f"direct_methods={list(direct_estimators.keys())}, "
          f"GCC={gcc_method}, estimator={estimator}, "
          f"lag_limit={tdoa_lag_limit_samples})...")
    for snr in snr_range:
        eval_seed = seed if fixed_eval_set else None
        X_noisy, X_clean, meta = simulator.generate_urban_batch(
            int(num_trials), snr_db=float(snr), seed=eval_seed
        )
        raw_np = X_noisy.numpy()
        clean_np = X_clean.numpy()

        base_inputs = {}
        base_details = {}
        base_inputs["Geometry"], base_details["Geometry"] = _localization_errors_from_batch(
                clean_np, meta, simulator.fs, simulator.c, simulator.area_size, gcc_func,
                sub_sample=sub_sample, use_los_only=use_los_only,
                estimator=estimator, oracle_geometry=True,
                tdoa_lag_limit_samples=tdoa_lag_limit_samples,
                return_details=True
            )
        base_inputs["Raw"], base_details["Raw"] = _localization_errors_from_batch(
                raw_np, meta, simulator.fs, simulator.c, simulator.area_size, gcc_func,
                sub_sample=sub_sample, use_los_only=use_los_only,
                estimator=estimator,
                tdoa_lag_limit_samples=tdoa_lag_limit_samples,
                return_details=True
            )
        base_inputs["Clean"], base_details["Clean"] = _localization_errors_from_batch(
                clean_np, meta, simulator.fs, simulator.c, simulator.area_size, gcc_func,
                sub_sample=sub_sample, use_los_only=use_los_only,
                estimator=estimator,
                tdoa_lag_limit_samples=tdoa_lag_limit_samples,
                return_details=True
            )
        model_outputs = _run_named_models(
            models_dict, X_noisy, simulator, device, batch_size=batch_size
        )
        for label, y_np in model_outputs.items():
            base_inputs[label], base_details[label] = _localization_errors_from_batch(
                y_np, meta, simulator.fs, simulator.c, simulator.area_size, gcc_func,
                sub_sample=sub_sample, use_los_only=use_los_only,
                estimator=estimator,
                tdoa_lag_limit_samples=tdoa_lag_limit_samples,
                return_details=True
            )
        for label, direct_estimator in direct_estimators.items():
            base_inputs[label], base_details[label] = _localization_errors_from_direct_estimator(
                raw_np, meta, simulator.fs, simulator.c, simulator.area_size,
                direct_estimator, sub_sample=sub_sample, use_los_only=use_los_only,
                estimator=estimator,
                tdoa_lag_limit_samples=tdoa_lag_limit_samples,
                return_details=True
            )

        diagnostic_inputs = {"Raw": raw_np, "Clean": clean_np}
        diagnostic_inputs.update(model_outputs)
        for label, y_np in diagnostic_inputs.items():
            diag = _urban_waveform_diagnostics(
                y_np, clean_np, meta, simulator.fs, simulator.c, gcc_func,
                sub_sample=sub_sample, use_los_only=use_los_only,
                tdoa_lag_limit_samples=tdoa_lag_limit_samples
            )
            dst_diag = results["method_diagnostics"].setdefault(label, {})
            for key, value in diag.items():
                dst_diag.setdefault(key, []).append(value)
        for label, direct_estimator in direct_estimators.items():
            if hasattr(direct_estimator, "diagnostics"):
                diag = direct_estimator.diagnostics(
                    raw_np, meta, simulator.fs, simulator.c,
                    use_los_only=use_los_only, sub_sample=sub_sample,
                    lag_limit_samples=tdoa_lag_limit_samples
                )
                dst_diag = results["method_diagnostics"].setdefault(label, {})
                for key, value in diag.items():
                    dst_diag.setdefault(key, []).append(value)

        line = [f"SNR={snr:g}dB"]
        for label in method_order:
            stats = _summarize_errors(base_inputs[label], int(num_trials))
            dst = results["methods"][label]
            for key in dst.keys():
                dst[key].append(stats[key if key != "rmse" else "rmse"])
            top_details = _top_worst_details(base_details.get(label, []), limit=10)
            results["outlier_details"].setdefault(label, []).append({
                "snr_db": float(snr),
                "top": top_details,
            })
            line.append(f"{label}={stats['rmse']:.2f}m")
        print("  " + " | ".join(line))

    return results, snr_range


def filter_method_comparison_data(method_data, method_order, config_updates=None):
    """Create a plotting/export view from a shared method-comparison result."""
    results, snr_range = method_data
    config_updates = config_updates or {}
    methods = results.get("methods", {})
    diagnostics = results.get("method_diagnostics", {})
    outliers = results.get("outlier_details", {})
    order = []
    seen = set()
    for label in method_order:
        if label in methods and label not in seen:
            order.append(label)
            seen.add(label)
    filtered = {
        "metric": results.get("metric", "localization_m"),
        "title": config_updates.get("title", results.get("title", "")),
        "method_order": order,
        "methods": {label: methods[label] for label in order},
        "method_diagnostics": {
            label: diagnostics.get(label, {}) for label in order
            if label in diagnostics
        },
        "outlier_details": {
            label: outliers.get(label, []) for label in order
            if label in outliers
        },
        "config": dict(results.get("config", {})),
    }
    filtered["config"].update(config_updates)
    return filtered, snr_range


class MonteCarloExperiment:
    """
    蒙特卡洛实验类
    功能：在不同 SNR 下进行多次实验，统计 RMSE 和 Loss。
    """

    def __init__(self, models_dict, simulator, device, seed=None, gcc_method='standard',
                 snr_range=None, num_trials=1000):
        """
        参数:
            gcc_method: 'standard' (标准 GCC，默认) 或 'phat' (GCC-PHAT)
        """
        self.models_dict = models_dict
        self.sim = simulator
        self.device = device
        self.snr_range = np.asarray(snr_range if snr_range is not None else np.arange(-10, 11, 1))
        self.num_trials = int(num_trials)
        self.seed = seed
        self.gcc_func = gcc_standard if gcc_method == 'standard' else gcc_phat
        self.gcc_method = gcc_method

    def run(self):
        print(f"Running Monte Carlo Sweep (GCC method: {self.gcc_method})...")
        if self.seed is not None:
            np.random.seed(self.seed)
            print(f"  Monte Carlo random seed set to: {self.seed}")

        ci_rng = np.random.default_rng(self.seed)
        results = {
            'raw': [],
            'raw_med': [],
            'trial_se': {'raw': [], 'dae': {}},
            'rmse_ci95': {'raw': [], 'dae': {}},
            'clean_peak_diagnostics': [],
        }
        for cr in self.models_dict.keys():
            results[f'dae_{cr}'] = []
            results[f'dae_{cr}_med'] = []
            results['trial_se']['dae'][str(cr)] = []
            results['rmse_ci95']['dae'][str(cr)] = []
            self.models_dict[cr].eval()

        signal_len = self.sim.signal_len
        lags = signal.correlation_lags(signal_len, signal_len, mode='same')

        for snr in self.snr_range:
            X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2 = \
                self.sim.generate_pair_batch(self.num_trials, snr_db=snr)
            X1_dev = X1_noisy.to(self.device)
            X2_dev = X2_noisy.to(self.device)
            raw_np1 = X1_noisy.cpu().numpy()
            raw_np2 = X2_noisy.cpu().numpy()
            clean_np1 = X1_clean.cpu().numpy()
            clean_np2 = X2_clean.cpu().numpy()
            clean_diag = _batch_peak_diagnostics(
                clean_np1, clean_np2, delays1, delays2, lags, gcc_func=self.gcc_func
            )
            clean_diag['snr'] = float(snr)
            results['clean_peak_diagnostics'].append(clean_diag)

            se_raw = 0.0
            se_list_raw = []
            for i in range(self.num_trials):
                sig_raw1 = raw_np1[i, 0, :] + 1j * raw_np1[i, 1, :]
                sig_raw2 = raw_np2[i, 0, :] + 1j * raw_np2[i, 1, :]
                corr_raw = self.gcc_func(sig_raw1, sig_raw2)
                delay_raw = lags[np.argmax(np.abs(corr_raw))]
                true_tdoa = delays1[i] - delays2[i]
                se = (delay_raw - true_tdoa) ** 2
                se_raw += se
                se_list_raw.append(se)
            results['raw'].append(np.sqrt(se_raw / self.num_trials))
            results['raw_med'].append(np.sqrt(np.median(se_list_raw)))
            results['trial_se']['raw'].append([float(v) for v in se_list_raw])
            results['rmse_ci95']['raw'].append(_bootstrap_rmse_ci(se_list_raw, ci_rng))

            for cr, model in self.models_dict.items():
                se_dae = 0.0
                se_list_dae = []
                with torch.no_grad():
                    Y1_rec = model(X1_dev).cpu().numpy()
                    Y2_rec = model(X2_dev).cpu().numpy()
                for i in range(self.num_trials):
                    sig_rec1 = Y1_rec[i, 0, :] + 1j * Y1_rec[i, 1, :]
                    sig_rec2 = Y2_rec[i, 0, :] + 1j * Y2_rec[i, 1, :]
                    corr_dae = self.gcc_func(sig_rec1, sig_rec2)
                    delay_dae = lags[np.argmax(np.abs(corr_dae))]
                    true_tdoa = delays1[i] - delays2[i]
                    se = (delay_dae - true_tdoa) ** 2
                    se_dae += se
                    se_list_dae.append(se)

                results[f'dae_{cr}'].append(np.sqrt(se_dae / self.num_trials))
                results[f'dae_{cr}_med'].append(np.sqrt(np.median(se_list_dae)))
                results['trial_se']['dae'][str(cr)].append([float(v) for v in se_list_dae])
                results['rmse_ci95']['dae'][str(cr)].append(_bootstrap_rmse_ci(se_list_dae, ci_rng))

        return results, self.snr_range


# --- 绘图函数 ---

def plot_training_loss(loss_hist):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(loss_hist, linewidth=2)
    plt.title("Figure 1: Training Loss Curve (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    return fig


def generate_snr_data(model, sim, device, snr_list=None, demo_seed=20260701):
    """生成 Figure 2 所需的绘图数据，可保存供 replot.py 重绘"""
    if snr_list is None:
        snr_list = [-10, -3, 3, 10]

    Fs = 40e6
    Ts_us = 1e6 / Fs

    print(f"--- Generating SNR comparison data for {snr_list} ---")
    model.eval()

    data_list = []
    for snr in snr_list:
        X1_n, X1_c, X2_n, X2_c, d1, d2 = sim.generate_pair_batch(
            1, snr_db=snr, seed=demo_seed
        )
        with torch.no_grad():
            x1_rec = model(X1_n.to(device)).cpu()
            x2_rec = model(X2_n.to(device)).cpu()

        noisy1 = X1_n[0, 0, :].numpy() + 1j * X1_n[0, 1, :].numpy()
        clean1 = X1_c[0, 0, :].numpy() + 1j * X1_c[0, 1, :].numpy()
        recon1 = x1_rec[0, 0, :].numpy() + 1j * x1_rec[0, 1, :].numpy()
        noisy2 = X2_n[0, 0, :].numpy() + 1j * X2_n[0, 1, :].numpy()
        clean2 = X2_c[0, 0, :].numpy() + 1j * X2_c[0, 1, :].numpy()
        recon2 = x2_rec[0, 0, :].numpy() + 1j * x2_rec[0, 1, :].numpy()
        n_samples = len(noisy1)

        f_n, P_n = signal.periodogram(noisy1, fs=Fs, return_onesided=False, scaling='density', detrend=False)
        f_c, P_c = signal.periodogram(clean1, fs=Fs, return_onesided=False, scaling='density', detrend=False)
        f_r, P_r = signal.periodogram(recon1, fs=Fs, return_onesided=False, scaling='density', detrend=False)

        lags = signal.correlation_lags(n_samples, n_samples, mode='same')
        corr_n = gcc_standard(noisy1, noisy2)
        corr_c = gcc_standard(clean1, clean2)
        corr_r = gcc_standard(recon1, recon2)

        data_list.append({
            'snr': snr,
            'n_samples': n_samples,
            't_us': np.arange(n_samples) * Ts_us,
            'noisy_mag': np.abs(noisy1), 'clean_mag': np.abs(clean1), 'recon_mag': np.abs(recon1),
            'f_n': np.fft.fftshift(f_n), 'f_c': np.fft.fftshift(f_c), 'f_r': np.fft.fftshift(f_r),
            'P_n_db': 10 * np.log10(np.fft.fftshift(P_n) + 1e-30),
            'P_c_db': 10 * np.log10(np.fft.fftshift(P_c) + 1e-30),
            'P_r_db': 10 * np.log10(np.fft.fftshift(P_r) + 1e-30),
            'lags': lags,
            'corr_n_norm': np.abs(corr_n) / (np.max(np.abs(corr_n)) + 1e-9),
            'corr_c_norm': np.abs(corr_c) / (np.max(np.abs(corr_c)) + 1e-9),
            'corr_r_norm': np.abs(corr_r) / (np.max(np.abs(corr_r)) + 1e-9),
            'true_tdoa': d1[0] - d2[0],
        })

    return data_list


def generate_snr_data_all(models_dict, sim, device, snr_list=None, diagnostic_trials=128,
                          demo_seed=20260701):
    """
    生成所有CR的SNR对比数据（多CR叠加绘图用）。

    参数:
        models_dict: {cr: model} 字典
    返回:
        data_dict: {snr: {'t_us': ..., 'noisy_mag': ..., 'clean_mag': ...,
                          'recon_mag': {cr: ...}, 'corr_r_norm': {cr: ...}, ...}}
    """
    if snr_list is None:
        snr_list = [-10, -3, 3, 10]

    Fs = 40e6
    Ts_us = 1e6 / Fs

    for model in models_dict.values():
        model.eval()

    print(f"--- Generating SNR comparison data (all CRs) for {snr_list} ---")
    data_dict = {}
    for snr in snr_list:
        X1_n, X1_c, X2_n, X2_c, d1, d2 = sim.generate_pair_batch(
            1, snr_db=snr, seed=demo_seed
        )

        noisy1 = X1_n[0, 0, :].numpy() + 1j * X1_n[0, 1, :].numpy()
        clean1 = X1_c[0, 0, :].numpy() + 1j * X1_c[0, 1, :].numpy()
        noisy2 = X2_n[0, 0, :].numpy() + 1j * X2_n[0, 1, :].numpy()
        clean2 = X2_c[0, 0, :].numpy() + 1j * X2_c[0, 1, :].numpy()
        n_samples = len(noisy1)
        lags = signal.correlation_lags(n_samples, n_samples, mode='same')

        # 所有CR的DAE输出
        recon_mag = {}
        recon_real = {}
        recon_imag = {}
        corr_r_norm = {}
        P_r_db = {}
        for cr, model in models_dict.items():
            with torch.no_grad():
                x1_rec = model(X1_n.to(device)).cpu()
                x2_rec = model(X2_n.to(device)).cpu()
            recon1 = x1_rec[0, 0, :].numpy() + 1j * x1_rec[0, 1, :].numpy()
            recon2 = x2_rec[0, 0, :].numpy() + 1j * x2_rec[0, 1, :].numpy()
            recon_mag[cr] = np.abs(recon1)
            recon_real[cr] = recon1.real
            recon_imag[cr] = recon1.imag
            corr_r = gcc_standard(recon1, recon2)
            corr_r_norm[cr] = _norm_abs_corr(corr_r)
            _, P_r = signal.periodogram(recon1, fs=Fs, return_onesided=False,
                                         scaling='density', detrend=False)
            P_r_db[cr] = 10 * np.log10(np.fft.fftshift(P_r) + 1e-30)

        # 频谱（所有CR各自计算PSD，用于频域子图叠加对比）
        f_n, P_n = signal.periodogram(noisy1, fs=Fs, return_onesided=False, scaling='density', detrend=False)
        f_c, P_c = signal.periodogram(clean1, fs=Fs, return_onesided=False, scaling='density', detrend=False)
        corr_n = gcc_standard(noisy1, noisy2)
        corr_c = gcc_standard(clean1, clean2)
        f_shift = np.fft.fftshift(f_n)
        p_clean_db = 10 * np.log10(np.fft.fftshift(P_c) + 1e-30)

        diagnostics = {'diagnostic_trials': int(diagnostic_trials), 'dae': {}}
        if diagnostic_trials and diagnostic_trials > 0:
            X1_nd, X1_cd, X2_nd, X2_cd, d1d, d2d = sim.generate_pair_batch(
                diagnostic_trials, snr_db=snr, seed=demo_seed + 1000
            )
            diagnostics['raw_peak'] = _batch_peak_diagnostics(
                X1_nd.numpy(), X2_nd.numpy(), d1d, d2d, lags, gcc_func=gcc_standard
            )
            diagnostics['clean_peak'] = _batch_peak_diagnostics(
                X1_cd.numpy(), X2_cd.numpy(), d1d, d2d, lags, gcc_func=gcc_standard
            )

            clean_diag_np1 = X1_cd.numpy()
            clean_diag_np2 = X2_cd.numpy()
            band_mask = np.abs(f_shift) <= 20e6
            for cr, model in models_dict.items():
                with torch.no_grad():
                    Y1d = model(X1_nd.to(device)).cpu().numpy()
                    Y2d = model(X2_nd.to(device)).cpu().numpy()

                peak_diag = _batch_peak_diagnostics(Y1d, Y2d, d1d, d2d, lags, gcc_func=gcc_standard)
                mag_nmse_vals = []
                complex_nmse_vals = []
                psd_mse_vals = []
                for j in range(diagnostic_trials):
                    rec1 = Y1d[j, 0, :] + 1j * Y1d[j, 1, :]
                    rec2 = Y2d[j, 0, :] + 1j * Y2d[j, 1, :]
                    clean_j1 = clean_diag_np1[j, 0, :] + 1j * clean_diag_np1[j, 1, :]
                    clean_j2 = clean_diag_np2[j, 0, :] + 1j * clean_diag_np2[j, 1, :]
                    sig_power = np.mean(np.abs(clean_j1) ** 2 + np.abs(clean_j2) ** 2) + 1e-9
                    mag_mse = np.mean((np.abs(rec1) - np.abs(clean_j1)) ** 2
                                      + (np.abs(rec2) - np.abs(clean_j2)) ** 2)
                    complex_mse = np.mean(np.abs(rec1 - clean_j1) ** 2
                                          + np.abs(rec2 - clean_j2) ** 2)
                    _, p_rec = signal.periodogram(rec1, fs=Fs, return_onesided=False,
                                                   scaling='density', detrend=False)
                    _, p_cln = signal.periodogram(clean_j1, fs=Fs, return_onesided=False,
                                                   scaling='density', detrend=False)
                    p_rec_db = 10 * np.log10(np.fft.fftshift(p_rec) + 1e-30)
                    p_cln_db = 10 * np.log10(np.fft.fftshift(p_cln) + 1e-30)
                    mag_nmse_vals.append(float(mag_mse / sig_power))
                    complex_nmse_vals.append(float(complex_mse / sig_power))
                    psd_mse_vals.append(float(np.mean((p_rec_db[band_mask] - p_cln_db[band_mask]) ** 2)))

                peak_diag.update({
                    'mag_nmse_mean': float(np.mean(mag_nmse_vals)),
                    'mag_nmse_median': float(np.median(mag_nmse_vals)),
                    'complex_nmse_mean': float(np.mean(complex_nmse_vals)),
                    'complex_nmse_median': float(np.median(complex_nmse_vals)),
                    'psd_db_mse_mean': float(np.mean(psd_mse_vals)),
                    'psd_db_mse_median': float(np.median(psd_mse_vals)),
                })
                diagnostics['dae'][cr] = peak_diag

        data_dict[snr] = {
            'snr': snr, 'n_samples': n_samples,
            't_us': np.arange(n_samples) * Ts_us,
            'noisy_mag': np.abs(noisy1), 'clean_mag': np.abs(clean1),
            'recon_mag': recon_mag,
            'clean_real': clean1.real, 'clean_imag': clean1.imag,
            'noisy_real': noisy1.real, 'noisy_imag': noisy1.imag,
            'recon_real': recon_real, 'recon_imag': recon_imag,
            'f_n': np.fft.fftshift(f_n), 'f_c': np.fft.fftshift(f_c),
            'P_n_db': 10 * np.log10(np.fft.fftshift(P_n) + 1e-30),
            'P_c_db': p_clean_db,
            'lags': lags,
            'corr_n_norm': _norm_abs_corr(corr_n),
            'corr_c_norm': _norm_abs_corr(corr_c),
            'corr_r_norm': corr_r_norm,
            'P_r_db': P_r_db,
            'true_tdoa': d1[0] - d2[0],
            'diagnostics': diagnostics,
        }

    return data_dict


def plot_snr_comparison(model=None, sim=None, device=None, snr_list=None, cr=None, data=None):
    """
    绘制 Figure 2: SNR 信号对比图。
    可通过 data 参数传入预计算数据（replot.py 用），或传入 model+sim 现场计算。
    """
    if data is None:
        data = generate_snr_data(model, sim, device, snr_list)
    snr_list = [d['snr'] for d in data]

    Fs = 40e6
    num_rows = len(data)
    fig, axes = plt.subplots(num_rows, 3, figsize=(14, 2.1 * num_rows + 0.4))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    cr_str = f" (CR={cr})" if cr is not None else ""
    plt.suptitle(f"Figure 2: Signal Analysis{cr_str}  (Symbol Rate=20 MHz, BW=24 MHz, Fs={Fs / 1e6:.0f} MHz)",
                 fontsize=13, y=0.99)

    cols = ['Time Domain', 'Power Spectral Density', 'Cross-Correlation (X₁ vs X₂)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=10, fontweight='bold')

    for i, d in enumerate(data):
        snr = d['snr']
        t_us = d['t_us']

        # --- 时域 ---
        ax_t = axes[i, 0]
        ax_t.plot(t_us, d['noisy_mag'], color='cornflowerblue', linewidth=0.5, label='Noisy')
        ax_t.plot(t_us, d['clean_mag'], 'k--', linewidth=0.8, label='Clean')
        ax_t.plot(t_us, d['recon_mag'], 'r', alpha=0.8, linewidth=0.7, label='Reconstructed')
        ax_t.set_ylabel(f"SNR={snr}dB\n|s(t)|", fontsize=9, fontweight='bold')
        ax_t.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax_t.set_xlim(0, t_us[min(299, d['n_samples'] - 1)])
        ax_t.grid(alpha=0.3)
        if i == num_rows - 1:
            ax_t.set_xlabel("Time [μs]", fontsize=10)

        # --- 频域 ---
        ax_f = axes[i, 1]
        ax_f.plot(d['f_n'] / 1e6, d['P_n_db'], color='cornflowerblue', linewidth=0.5, label='Noisy')
        ax_f.plot(d['f_c'] / 1e6, d['P_c_db'], 'k--', linewidth=0.8, label='Clean')
        ax_f.plot(d['f_r'] / 1e6, d['P_r_db'], 'r', linewidth=0.7, label='Reconstructed')
        ax_f.legend(loc='upper right', fontsize=6, framealpha=0.8, ncol=2)
        if i == num_rows - 1:
            ax_f.set_xlabel("Frequency [MHz]", fontsize=10)
        ax_f.set_ylabel("PSD [dB/Hz]", fontsize=9)
        ax_f.grid(alpha=0.3)
        ax_f.set_xlim(-20, 20)

        # --- 互相关 ---
        ax_c = axes[i, 2]
        ax_c.plot(d['lags'], d['corr_n_norm'], color='cornflowerblue', linewidth=0.8, label='Noisy')
        ax_c.plot(d['lags'], d['corr_c_norm'], 'k--', linewidth=0.8, label='Clean')
        ax_c.plot(d['lags'], d['corr_r_norm'], 'r', linewidth=1.0, label='Reconstructed')
        ax_c.axvline(d['true_tdoa'], color='blue', linestyle='--', linewidth=0.8,
                     label=f"True TDOA={d['true_tdoa']}")
        ax_c.set_xlim(d['true_tdoa'] - 100, d['true_tdoa'] + 100)
        ax_c.grid(alpha=0.3)
        ax_c.set_ylabel("Norm. GCC", fontsize=9)
        ax_c.legend(loc='upper right', fontsize=7, framealpha=0.8)
        if i == num_rows - 1:
            ax_c.set_xlabel("Lag [samples]", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_monte_carlo(mc_data, show_ci=False):
    results, snr_range = mc_data
    has_median = 'raw_med' in results
    has_trimmed = 'raw_trimmed' in results
    ncols = 3 if has_trimmed else (2 if has_median else 1)
    fig_width = 5.0 * ncols
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    # 动态提取 CR 值（支持任意 CR_LIST）
    cr_keys = sorted([k for k in results.keys() if k.startswith('dae_') and k[4:].isdigit()],
                     key=lambda x: int(x[4:]))

    # 左图: Mean RMSE
    ax = axes[0]
    metric = results.get('metric', 'tdoa_samples')
    y_label = 'Localization RMSE [m]' if metric == 'localization_m' else 'TDOA RMSE [samples]'
    title_prefix = 'Localization' if metric == 'localization_m' else 'TDOA'
    ax.plot(snr_range, results['raw'], 'b-s', label='Original data', linewidth=1.5)
    if 'clean' in results:
        ax.plot(snr_range, results['clean'], 'k--', label='Clean oracle', linewidth=1.2)
    if 'geom' in results:
        ax.plot(snr_range, results['geom'], color='0.45', linestyle=':', label='Geometry oracle',
                linewidth=1.1)
    ci_data = results.get('rmse_ci95', {})
    raw_ci = np.asarray(ci_data.get('raw', []), dtype=float)
    if show_ci and raw_ci.shape == (len(snr_range), 2):
        ax.fill_between(snr_range, raw_ci[:, 0], raw_ci[:, 1],
                        color='blue', alpha=0.10, linewidth=0)
    marker_styles = ['m-*', 'g-o', 'r-+', 'c-^', 'y-d']
    for idx, key in enumerate(cr_keys):
        style = marker_styles[idx % len(marker_styles)]
        cr_val = key[4:]
        ax.plot(snr_range, results[key], style, label=f'Data with CR={cr_val}', linewidth=1.5)
        dae_ci = np.asarray(ci_data.get('dae', {}).get(str(cr_val), []), dtype=float)
        if show_ci and dae_ci.shape == (len(snr_range), 2):
            ax.fill_between(snr_range, dae_ci[:, 0], dae_ci[:, 1],
                            color=ax.lines[-1].get_color(), alpha=0.08, linewidth=0)
    ax.set_xlabel('SNR [dB]', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title('Mean RMSE (sensitive to outliers)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.5)

    # 右图: Median RMSE
    if has_median:
        ax = axes[1]
        ax.plot(snr_range, results['raw_med'], 'b-s', label='Original data', linewidth=1.5)
        if 'clean_med' in results:
            ax.plot(snr_range, results['clean_med'], 'k--', label='Clean oracle', linewidth=1.2)
        if 'geom_med' in results:
            ax.plot(snr_range, results['geom_med'], color='0.45', linestyle=':',
                    label='Geometry oracle', linewidth=1.1)
        for idx, key in enumerate(cr_keys):
            med_key = key + '_med'
            if med_key in results:
                style = marker_styles[idx % len(marker_styles)]
                cr_val = key[4:]
                ax.plot(snr_range, results[med_key], style, label=f'Data with CR={cr_val}', linewidth=1.5)
        ax.set_xlabel('SNR [dB]', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title('Median RMSE (robust, typical performance)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.5)

    # 第三图: Trimmed RMSE（去掉最坏 5% 定位异常值）
    if has_trimmed:
        ax = axes[2]
        ax.plot(snr_range, results['raw_trimmed'], 'b-s', label='Original data', linewidth=1.5)
        if 'clean_trimmed' in results:
            ax.plot(snr_range, results['clean_trimmed'], 'k--', label='Clean oracle', linewidth=1.2)
        if 'geom_trimmed' in results:
            ax.plot(snr_range, results['geom_trimmed'], color='0.45', linestyle=':',
                    label='Geometry oracle', linewidth=1.1)
        for idx, key in enumerate(cr_keys):
            trim_key = key + '_trimmed'
            if trim_key in results:
                style = marker_styles[idx % len(marker_styles)]
                cr_val = key[4:]
                ax.plot(snr_range, results[trim_key], style, label=f'Data with CR={cr_val}',
                        linewidth=1.5)
        ax.set_xlabel('SNR [dB]', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title('Trimmed RMSE (95% inliers)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.5)

    fig.suptitle(f'Figure 3: Comparison of {title_prefix} Performance at Different CRs', fontsize=12)
    return fig


def _collect_group_values(methods, prefix, metric_key):
    labels = [label for label in methods if label.startswith(prefix)]
    if not labels:
        return None, []
    values = [np.asarray(methods[label].get(metric_key, []), dtype=float) for label in labels]
    values = [v for v in values if v.size > 0]
    if not values:
        return None, labels
    return np.vstack(values), labels


def plot_method_comparison(method_data, metric_key="rmse", plot_kind="main",
                           snr_min=None, snr_max=None, method_order=None,
                           title_suffix=None, ax=None, legend=True,
                           show_zoom_inset=True):
    results, snr_range = method_data
    methods = results["methods"]
    config = results.get("config", {})
    if method_order is not None:
        order = method_order
    elif plot_kind == "strong":
        order = config.get("strong_method_order", results.get("method_order", list(methods.keys())))
    elif plot_kind == "taskaware":
        order = config.get("taskaware_method_order", results.get("method_order", list(methods.keys())))
    elif plot_kind == "supplement":
        order = config.get("supplement_method_order", results.get("method_order", list(methods.keys())))
    else:
        order = config.get("main_method_order", results.get("method_order", list(methods.keys())))
    metric = results.get("metric", "localization_m")
    y_label = "Localization RMSE [m]" if metric == "localization_m" else "TDOA RMSE [samples]"
    snr_all = np.asarray(snr_range, dtype=float)
    range_mask = np.ones_like(snr_all, dtype=bool)
    if snr_min is not None:
        range_mask &= snr_all >= float(snr_min)
    if snr_max is not None:
        range_mask &= snr_all <= float(snr_max)
    if np.count_nonzero(range_mask) == 0:
        range_mask = np.ones_like(snr_all, dtype=bool)
    snr_plot = snr_all[range_mask]

    styles = {
        "Raw": dict(color="#222222", marker="s", linestyle="-", linewidth=1.25,
                    label="Original data"),
        "Clean": dict(color="black", marker=None, linestyle="--", linewidth=1.05,
                      label="Clean oracle"),
        "Geometry": dict(color="0.45", marker=None, linestyle=":", linewidth=1.05,
                         label="Geometry oracle"),
        "DAE-CR4": dict(color="#08519c", marker="o", linestyle="-", linewidth=1.35,
                        label="Chen DAE CR=4"),
        "DAE-CR8": dict(color="#3182bd", marker="s", linestyle="--", linewidth=1.30,
                        label="Chen DAE CR=8"),
        "DAE-CR16": dict(color="#6baed6", marker="^", linestyle="-.", linewidth=1.30,
                         label="Chen DAE CR=16"),
        "DFT-Fisher": dict(color="#8c2d04", marker="X", linestyle=":", linewidth=1.25,
                           label="DFT Fisher recon"),
        "DFT-SCS-lite": dict(color="#bcbddc", marker="v", linestyle="--", linewidth=1.20,
                             label="DFT recon"),
        "DFT-SCS-lite-Direct": dict(color="#756bb1", marker=">", linestyle="-", linewidth=1.30,
                                    label="DFT SCS-lite direct"),
        "DFT-Fisher-Direct": dict(color="#762a83", marker="X", linestyle="-", linewidth=1.35,
                                  label="Balanced Fisher direct"),
        "DFT-train-power-Direct": dict(color="#01665e", marker="P", linestyle="-", linewidth=1.35,
                                       label="DFT train-power direct"),
        "DFT-GeoAmbi": dict(color="#1b9e77", marker="H", linestyle=":", linewidth=1.20,
                            label="GeoAmbi DFT recon"),
        "GeoAmbi-DFT-Direct": dict(color="#009e73", marker="H", linestyle="--", linewidth=1.35,
                                   label="GeoAmbi DFT direct"),
        "GeoAmbi-DFT-AML": dict(color="#00441b", marker="H", linestyle="-", linewidth=1.45,
                                label="GeoAmbi DFT-AML"),
        "GeoAmbi-DFT-PHAT": dict(color="#41ab5d", marker="h", linestyle="-.", linewidth=1.30,
                                 label="GeoAmbi DFT-PHAT"),
        "GeoHybrid-DFT": dict(color="#e6550d", marker="D", linestyle="-", linewidth=1.45,
                              label="GeoHybrid DFT v2"),
        "GeoHybrid-DFT-PHAT-gated": dict(color="#fdae6b", marker="d", linestyle="--", linewidth=1.30,
                                         label="GeoHybrid PHAT-gated"),
        "Cao2017-DFT-AML": dict(color="#007c91", marker="h", linestyle="-", linewidth=1.35,
                                label="Cao2017 DFT-AML"),
        "Cao2020-HighFC": dict(color="#56b4e9", marker=">", linestyle=(0, (5, 1)), linewidth=1.25,
                               label="Cao2020 segmented FC"),
        "Zhai-CRLB-Decimation": dict(color="#cc79a7", marker="8", linestyle="-", linewidth=1.35,
                                     label="Zhai CRLB decimation"),
        "Zhai-Phase-Superposition": dict(color="#d55e00", marker="P", linestyle="-.", linewidth=1.30,
                                         label="Zhai weighted phase"),
        "FreqDAE-CR4": dict(color="#b2182b", marker="*", linestyle="-", linewidth=1.45,
                            label="FreqDAE CR=4"),
        "FreqDAE-CR8": dict(color="#d6604d", marker="P", linestyle="--", linewidth=1.40,
                            label="FreqDAE CR=8"),
        "FreqDAE-CR16": dict(color="#f46d43", marker="X", linestyle="-.", linewidth=1.35,
                             label="FreqDAE CR=16"),
        "FreqDAE-v2-CR4": dict(color="#7a0177", marker="*", linestyle="-", linewidth=1.45,
                               label="FreqDAE v2 CR=4"),
        "FreqDAE-v2-CR8": dict(color="#c51b8a", marker="P", linestyle="--", linewidth=1.40,
                               label="FreqDAE v2 CR=8"),
        "FreqDAE-v2-CR16": dict(color="#f768a1", marker="X", linestyle="-.", linewidth=1.35,
                                label="FreqDAE v2 CR=16"),
        "FreqDAE-v3-CR4": dict(color="#006d2c", marker="*", linestyle="-", linewidth=1.45,
                               label="FreqDAE v3 CR=4"),
        "FreqDAE-v3-CR8": dict(color="#31a354", marker="P", linestyle="--", linewidth=1.40,
                               label="FreqDAE v3 CR=8"),
        "FreqDAE-v3-CR16": dict(color="#74c476", marker="X", linestyle="-.", linewidth=1.35,
                                label="FreqDAE v3 CR=16"),
        "DFT-Zhai-CRLB": dict(color="#cc79a7", marker="8", linestyle=":", linewidth=1.15,
                              label="DFT Zhai-CRLB recon"),
        "DFT-train-band": dict(color="#6a51a3", marker="^", linestyle="-.", linewidth=1.20,
                               label="DFT train-band"),
        "DFT-train-power": dict(color="#80cdc1", marker="D", linestyle="--", linewidth=1.15,
                                label="DFT train-power"),
        "DFT": dict(color="#4b0082", marker="^", linestyle="-", linewidth=1.35,
                    label="DFT direct"),
        "DFT-uniform": dict(color="#9467bd", marker="^", linestyle="-.", linewidth=1.15,
                            label="DFT-uniform"),
        "DFT-bandlimited": dict(color="#8c6bb1", marker="v", linestyle=":", linewidth=1.10,
                                label="DFT known-band"),
        "DFT-random": dict(color="#bcbddc", marker="^", linestyle="--", linewidth=1.10,
                           label="DFT-random mean"),
        "Hadamard": dict(color="#ff7f0e", marker="d", linestyle="-.", linewidth=1.20,
                         label="Hadamard Salari"),
        "Hadamard-random": dict(color="#fdae6b", marker="D", linestyle="--", linewidth=1.10,
                                label="Hadamard-random mean"),
        "Hadamard-sequency": dict(color="#e6550d", marker="d", linestyle=":", linewidth=1.10,
                                  label="Hadamard-sequency"),
        "Hadamard-block-2": dict(color="#a63603", marker="p", linestyle=(0, (3, 1, 1, 1)),
                                 linewidth=1.10, label="Hadamard block-2"),
        "PCA": dict(color="#2ca02c", marker="o", linestyle="-.", linewidth=1.20,
                    label="PCA"),
    }
    locator_suffix_styles = {
        "WLS": dict(linestyle="-", marker="o", suffix_label="WLS"),
        "Grid": dict(linestyle="--", marker="s", suffix_label="grid WLS"),
        "Pruned": dict(linestyle="-.", marker="^", suffix_label="pruned WLS"),
        "Grid+Pruned": dict(linestyle=(0, (3, 1, 1, 1)), marker="D",
                             suffix_label="grid+pruned WLS"),
    }

    baseline_cr = config.get("baseline_cr", 16)
    created_fig = ax is None
    if created_fig:
        if plot_kind == "supplement":
            fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.0))
        elif plot_kind == "strong":
            fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.2))
        elif plot_kind == "taskaware":
            fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.0))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    else:
        fig = ax.figure
    plotted_for_zoom = []
    for label in order:
        if isinstance(label, str) and label.endswith("*"):
            prefix = label[:-1]
            stack, group_labels = _collect_group_values(methods, prefix, metric_key)
            if stack is None:
                continue
            mean_y = np.nanmean(stack, axis=0)
            lo_y = np.nanmin(stack, axis=0)
            hi_y = np.nanmax(stack, axis=0)
            base_label = prefix[:-1] if prefix.endswith("-") else prefix
            style = dict(styles.get(base_label, dict(marker="*", linestyle="--",
                                                     linewidth=1.15, label=base_label)))
            marker = style.pop("marker", None)
            color = style.get("color", None)
            ax.plot(snr_plot, mean_y[range_mask], marker=marker,
                    markersize=4.5 if marker else 0, **style)
            plotted_for_zoom.append((mean_y, marker, dict(style)))
            if stack.shape[0] > 1 and color is not None:
                ax.fill_between(snr_plot, lo_y[range_mask], hi_y[range_mask],
                                color=color, alpha=0.16, linewidth=0)
            continue
        if label not in methods:
            continue
        y = np.asarray(methods[label].get(metric_key, []), dtype=float)
        if y.size == 0:
            continue
        base_label = label
        locator_label = None
        if isinstance(label, str) and "|" in label:
            base_label, locator_label = label.split("|", 1)
        style = dict(styles.get(base_label, styles.get(label, dict(
            marker="*", linestyle="-", linewidth=1.20, label=label
        ))))
        if locator_label is not None:
            suffix_style = locator_suffix_styles.get(locator_label, {})
            style.update({k: v for k, v in suffix_style.items() if k != "suffix_label"})
            base_display = styles.get(base_label, {}).get("label", base_label)
            suffix_display = suffix_style.get("suffix_label", locator_label)
            style["label"] = f"{base_display} | {suffix_display}"
            style["linewidth"] = min(float(style.get("linewidth", 1.2)), 1.15)
        if base_label.startswith("DFT") and base_label not in styles and label not in styles:
            style = dict(color="#9467bd", marker="^", linestyle="-.", linewidth=1.15,
                         label=label)
        elif ((base_label.startswith("Hadamard-offset") or base_label.startswith("Hadamard-block"))
              and base_label not in styles and label not in styles):
            style = dict(color="#a63603", marker="p", linestyle=(0, (3, 1, 1, 1)),
                         linewidth=1.10, label=label)
        elif base_label.startswith("Hadamard") and base_label not in styles and label not in styles:
            style = dict(color="#ff7f0e", marker="d", linestyle="-.", linewidth=1.15,
                         label=label)
        elif base_label.startswith("PCA") and base_label not in styles and label not in styles:
            style = dict(color="#2ca02c", marker="o", linestyle="-.", linewidth=1.15,
                         label=label)
        if base_label.startswith("DAE") and base_label not in styles and label not in styles:
            style = dict(color="#3182bd", marker="*", linestyle="-", linewidth=1.25,
                         label=label)
        elif base_label.startswith("FreqDAE") and base_label not in styles and label not in styles:
            style = dict(color="#b2182b", marker="*", linestyle="-", linewidth=1.30,
                         label=label)
        if base_label == "DFT" and locator_label is None:
            source = config.get("chen_fig6_dft_direct_source")
            if source:
                short_source = str(source).replace("DFT-", "")
                style["label"] = f"DFT direct ({short_source} bins)"
        marker = style.pop("marker", None)
        ax.plot(snr_plot, y[range_mask], marker=marker,
                markersize=4.5 if marker else 0, **style)
        plotted_for_zoom.append((y, marker, dict(style)))

    ax.set_xlabel("SNR [dB]", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if plot_kind == "strong":
        title = config.get(
            "strong_title",
            f"Figure 8: Strong Task-Aware Baselines (CR={baseline_cr})"
        )
    elif plot_kind == "taskaware":
        title = config.get(
            "taskaware_title",
            f"Figure 7: Task-Aware Baselines (CR={baseline_cr})"
        )
    elif plot_kind == "supplement":
        title = config.get(
            "supplement_title",
            f"Figure 6 Supplement: Baseline Sensitivity (CR={baseline_cr})"
        )
    else:
        title = config.get(
            "main_title",
            f"Figure 6: Localization Performance with Traditional Baselines (CR={baseline_cr})"
        )
    if title_suffix is None:
        if snr_min is not None and snr_max is not None:
            title_suffix = f"{float(snr_min):g} to {float(snr_max):g} dB"
        elif snr_min is not None:
            title_suffix = f"SNR >= {float(snr_min):g} dB"
        elif snr_max is not None:
            title_suffix = f"SNR <= {float(snr_max):g} dB"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.45)
    if (show_zoom_inset and plot_kind == "main"
            and config.get("fig6_show_zoom_inset", True)
            and snr_min is None and snr_max is None):
        snr_arr = snr_all
        zoom_min = float(config.get("fig6_zoom_snr_min", 8.0))
        mask = snr_arr >= zoom_min
        if np.count_nonzero(mask) >= 2 and plotted_for_zoom:
            y_zoom = []
            for y, _, _ in plotted_for_zoom:
                vals = np.asarray(y, dtype=float)[mask]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    y_zoom.append(vals)
            if y_zoom:
                y_all = np.concatenate(y_zoom)
                y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
                pad = max((y_max - y_min) * 0.18, 0.5)
                inset_ax = ax.inset_axes([0.52, 0.47, 0.43, 0.38])
                for y, marker, style in plotted_for_zoom:
                    inset_style = dict(style)
                    inset_style.pop("label", None)
                    inset_style["linewidth"] = min(float(inset_style.get("linewidth", 1.2)), 1.2)
                    inset_ax.plot(
                        snr_arr[mask], np.asarray(y, dtype=float)[mask],
                        marker=marker, markersize=3.0 if marker else 0,
                        **inset_style
                    )
                inset_ax.set_title(f"SNR >= {zoom_min:g} dB", fontsize=8)
                inset_ax.set_xlim(float(np.min(snr_arr[mask])), float(np.max(snr_arr[mask])))
                inset_ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
                inset_ax.tick_params(axis="both", labelsize=7)
                inset_ax.grid(True, alpha=0.35)
                try:
                    ax.indicate_inset_zoom(inset_ax, edgecolor="0.35", alpha=0.55)
                except Exception:
                    pass
    if legend:
        legend_cols = 5 if plot_kind == "strong" else 3
        ax.legend(fontsize=8.2 if plot_kind == "strong" else 8.8,
                  ncol=legend_cols, framealpha=0.95, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), borderaxespad=0.0)
    if created_fig:
        fig.subplots_adjust(bottom=0.27, left=0.11, right=0.98, top=0.90)
    return fig


def plot_method_zoom_pair(method_data, metric_key="rmse", plot_kind="supplement",
                          low_snr_max=0.0, high_snr_min=8.0,
                          method_order=None):
    """Plot low-SNR and high-SNR zooms in one compact two-panel figure."""
    results, _ = method_data
    config = results.get("config", {})
    baseline_cr = config.get("baseline_cr", 16)
    title_map = {
        "supplement": config.get(
            "supplement_title",
            f"Figure 6 Supplement: Baseline Sensitivity (CR={baseline_cr})",
        ),
        "taskaware": config.get(
            "taskaware_title",
            f"Figure 7: Task-Aware Baselines (CR={baseline_cr})",
        ),
        "strong": config.get(
            "strong_title",
            f"Figure 8: Strong Task-Aware Baselines (CR={baseline_cr})",
        ),
        "main": config.get(
            "main_title",
            f"Figure 6: Localization Performance with Traditional Baselines (CR={baseline_cr})",
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=False)
    plot_method_comparison(
        method_data, metric_key=metric_key, plot_kind=plot_kind,
        snr_max=low_snr_max, method_order=method_order,
        title_suffix=f"SNR <= {float(low_snr_max):g} dB",
        ax=axes[0], legend=False, show_zoom_inset=False,
    )
    plot_method_comparison(
        method_data, metric_key=metric_key, plot_kind=plot_kind,
        snr_min=high_snr_min, method_order=method_order,
        title_suffix=f"SNR >= {float(high_snr_min):g} dB",
        ax=axes[1], legend=False, show_zoom_inset=False,
    )
    axes[0].set_title(f"Low SNR (<= {float(low_snr_max):g} dB)", fontsize=11)
    axes[1].set_title(f"High SNR (>= {float(high_snr_min):g} dB)", fontsize=11)
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for handle, label in zip(h, lab):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    ncol = 5 if plot_kind == "strong" else 3
    fig.legend(handles, labels, fontsize=8.0 if plot_kind == "strong" else 8.5,
               ncol=ncol, framealpha=0.95,
               loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"{title_map.get(plot_kind, 'Method comparison')} - SNR Zooms",
                 fontsize=12)
    fig.subplots_adjust(bottom=0.25, left=0.08, right=0.985,
                        top=0.86, wspace=0.22)
    return fig


def plot_snr_comparison_multi(cr_list, data_dict, title_suffix=""):
    """
    多CR SNR信号对比图（Figure 2改进版）。

    4行(SNR) × 3列(Time/PSD/GCC)，每个子图中叠加Noisy/Clean及各CR的DAE轨迹。

    参数:
        cr_list:   CR列表 e.g. [4, 8, 16]
        data_dict: generate_snr_data_all() 的输出 {snr: {data}}
        title_suffix: 标题后缀
    """
    snr_list = sorted(data_dict.keys())
    num_rows = len(snr_list)
    cr_colors = {4: '#08519c', 8: '#d6604d', 16: '#4b0082'}
    cr_styles = {4: '-', 8: '--', 16: '-.'}

    fig, axes = plt.subplots(num_rows, 3, figsize=(14, 2.1 * num_rows + 0.4))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    plt.suptitle(f"Figure 2: Signal Analysis (All CRs){title_suffix}  "
                 f"(Symbol Rate=20 MHz, BW=24 MHz, Fs=40 MHz)",
                 fontsize=13, y=0.99)

    cols = ['Time Domain', 'Power Spectral Density', 'Cross-Correlation (X₁ vs X₂)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=10, fontweight='bold')

    for i, snr in enumerate(snr_list):
        d = data_dict[snr]
        t_us = d['t_us']

        # --- 时域 ---
        ax_t = axes[i, 0]
        ax_t.plot(t_us, d['noisy_mag'], color='#7aa6dc', linewidth=0.35, label='Noisy', alpha=0.7)
        ax_t.plot(t_us, d['clean_mag'], 'k--', linewidth=0.55, label='Clean')
        for cr in cr_list:
            ax_t.plot(t_us, d['recon_mag'][cr], color=cr_colors.get(cr, 'r'),
                      linewidth=0.55, linestyle=cr_styles.get(cr, '-'),
                      label=f'DAE CR={cr}')
        ax_t.set_ylabel(f"SNR={snr}dB\n|s(t)|", fontsize=9, fontweight='bold')
        ax_t.legend(loc='upper right', fontsize=6, framealpha=0.8, ncol=2)
        ax_t.set_xlim(0, t_us[min(299, d['n_samples'] - 1)])
        ax_t.grid(alpha=0.3)
        if i == num_rows - 1:
            ax_t.set_xlabel("Time [μs]", fontsize=10)

        # --- 频域 (所有CR的PSD叠加) ---
        ax_f = axes[i, 1]
        ax_f.plot(d['f_n'] / 1e6, d['P_n_db'], color='#7aa6dc', linewidth=0.35, label='Noisy', alpha=0.7)
        ax_f.plot(d['f_c'] / 1e6, d['P_c_db'], 'k--', linewidth=0.55, label='Clean')
        for cr in cr_list:
            if 'P_r_db' in d and cr in d['P_r_db']:
                ax_f.plot(d['f_n'] / 1e6, d['P_r_db'][cr], color=cr_colors.get(cr, 'r'),
                          linewidth=0.55, linestyle=cr_styles.get(cr, '-'),
                          alpha=0.9, label=f'DAE CR={cr}')
        ax_f.legend(loc='upper right', fontsize=6, framealpha=0.8, ncol=2)
        ax_f.grid(alpha=0.3)
        ax_f.set_xlim(-20, 20)
        if i == num_rows - 1:
            ax_f.set_xlabel("Frequency [MHz]", fontsize=10)
        ax_f.set_ylabel("PSD [dB/Hz]", fontsize=9)

        # --- 互相关 ---
        ax_c = axes[i, 2]
        ax_c.plot(d['lags'], d['corr_n_norm'], color='#7aa6dc', linewidth=0.45, label='Noisy')
        ax_c.plot(d['lags'], d['corr_c_norm'], 'k--', linewidth=0.55, label='Clean')
        for cr in cr_list:
            ax_c.plot(d['lags'], d['corr_r_norm'][cr], color=cr_colors.get(cr, 'r'),
                      linewidth=0.9, linestyle=cr_styles.get(cr, '-'),
                      label=f'DAE CR={cr}')
        ax_c.axvline(d['true_tdoa'], color='blue', linestyle=':', linewidth=0.6,
                     label=f"True TDOA={d['true_tdoa']}")
        ax_c.set_xlim(d['true_tdoa'] - 100, d['true_tdoa'] + 100)
        ax_c.grid(alpha=0.3)
        ax_c.set_ylabel("Norm. GCC", fontsize=9)
        ax_c.legend(loc='upper right', fontsize=6, framealpha=0.8, ncol=2)
        if i == num_rows - 1:
            ax_c.set_xlabel("Lag [samples]", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
