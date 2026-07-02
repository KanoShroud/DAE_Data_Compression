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


def _estimate_delay_from_corr(sig_i, sig_ref, lags, gcc_func, sub_sample=True,
                              return_quality=False):
    corr = np.abs(gcc_func(sig_i, sig_ref))
    idx = int(np.argmax(corr))
    lag = float(lags[idx])
    if sub_sample and 0 < idx < len(corr) - 1:
        y0, y1, y2 = corr[idx - 1], corr[idx], corr[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            lag += float(np.clip(delta, -0.5, 0.5))
    if not return_quality:
        return lag

    peak = float(corr[idx])
    mask = np.ones_like(corr, dtype=bool)
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


def _localize_from_tdoa_pairs(uav_pos, pair_measurements, fs, c, area_size):
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

    idx_used = sorted(set([i for i, _, _, _ in pairs] + [j for _, j, _, _ in pairs]))
    selected = uav_pos[idx_used]
    delta_ranges = np.asarray([tau * c / fs for _, _, tau, _ in pairs], dtype=float)
    weights = np.asarray([w for _, _, _, w in pairs], dtype=float)
    weights = weights / (np.median(weights) + 1e-12)
    sqrt_w = np.sqrt(np.clip(weights, 0.05, 20.0))

    def residual(p):
        vals = []
        for (i, j, _, _), dr in zip(pairs, delta_ranges):
            vals.append(np.linalg.norm(p - uav_pos[i]) - np.linalg.norm(p - uav_pos[j]) - dr)
        return sqrt_w * np.asarray(vals)

    bounds = ([0.0, 0.0], [float(area_size[0]), float(area_size[1])])
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

    best = None
    best_cost = float('inf')
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, dtype=float), bounds[0], bounds[1])
        try:
            res = optimize.least_squares(residual, x0=x0, bounds=bounds, loss='linear',
                                         max_nfev=200)
        except Exception:
            continue
        if res.success and np.all(np.isfinite(res.x)) and res.cost < best_cost:
            best = res.x
            best_cost = float(res.cost)

    info = {
        'success': best is not None,
        'cost': best_cost,
        'n_pairs': len(pairs),
        'n_uavs': len(idx_used),
    }
    return best, info


def _summarize_errors(errors, requested_count):
    arr = np.asarray(errors, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            'rmse': float('nan'), 'median': float('nan'), 'trimmed_rmse': float('nan'),
            'p90': float('nan'), 'p95': float('nan'), 'max': float('nan'),
            'valid_count': 0, 'requested_count': int(requested_count), 'failure_rate': 1.0,
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
        'se': [float(v) for v in finite ** 2],
    }


def _localization_errors_from_batch(batch_np, meta, fs, c, area_size, gcc_func,
                                    sub_sample=True, use_los_only=True,
                                    estimator="all_pair_wls", oracle_geometry=False):
    errors = []
    failures = 0
    for bi in range(batch_np.shape[0]):
        los_idx = np.where(meta['los'][bi])[0] if use_los_only else np.arange(batch_np.shape[1])
        if len(los_idx) < 4:
            failures += 1
            continue
        lags = signal.correlation_lags(batch_np.shape[-1], batch_np.shape[-1], mode='same')
        pair_measurements = []
        los_idx = [int(v) for v in los_idx]
        for a in range(len(los_idx)):
            for b in range(a + 1, len(los_idx)):
                ui, uj = los_idx[a], los_idx[b]
                if oracle_geometry:
                    tau = (meta['distances'][bi, ui] - meta['distances'][bi, uj]) / (c / fs)
                    weight = 20.0
                else:
                    sig_i = batch_np[bi, ui, 0, :] + 1j * batch_np[bi, ui, 1, :]
                    sig_j = batch_np[bi, uj, 0, :] + 1j * batch_np[bi, uj, 1, :]
                    tau, weight, _, _ = _estimate_delay_from_corr(
                        sig_i, sig_j, lags, gcc_func, sub_sample=sub_sample,
                        return_quality=True
                    )
                pair_measurements.append((ui, uj, tau, weight))

        if estimator != "all_pair_wls":
            ref_idx = int(los_idx[0])
            ref_sig = batch_np[bi, ref_idx, 0, :] + 1j * batch_np[bi, ref_idx, 1, :]
            tdoa = {}
            for ui in los_idx:
                if ui == ref_idx:
                    continue
                sig_i = batch_np[bi, ui, 0, :] + 1j * batch_np[bi, ui, 1, :]
                tdoa[ui] = _estimate_delay_from_corr(sig_i, ref_sig, lags, gcc_func,
                                                      sub_sample=sub_sample)
            est = _localize_from_tdoa(meta['uavs'][bi], ref_idx, tdoa, fs, c, area_size)
        else:
            est, _ = _localize_from_tdoa_pairs(meta['uavs'][bi], pair_measurements,
                                               fs, c, area_size)
        if est is None:
            failures += 1
            continue
        errors.append(float(np.linalg.norm(est - meta['source'][bi])))
    return errors


class UrbanLocalizationExperiment:
    """
    论文近似复现评估：8 UAV + oracle LOS 选择 + 多 TDOA 几何定位误差。
    """

    def __init__(self, models_dict, simulator, device, seed=None, gcc_method='standard',
                 snr_range=None, num_trials=200, sub_sample=True, use_los_only=True,
                 batch_size=64, fixed_eval_set=True, estimator="all_pair_wls"):
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
              f"sub_sample={self.sub_sample}, LOS-only={self.use_los_only})...")
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
                estimator=self.estimator, oracle_geometry=True
            )
            raw_err = _localization_errors_from_batch(
                raw_np, meta, self.sim.fs, self.sim.c, self.sim.area_size, self.gcc_func,
                sub_sample=self.sub_sample, use_los_only=self.use_los_only,
                estimator=self.estimator
            )
            clean_err = _localization_errors_from_batch(
                clean_np, meta, self.sim.fs, self.sim.c, self.sim.area_size, self.gcc_func,
                sub_sample=self.sub_sample, use_los_only=self.use_los_only,
                estimator=self.estimator
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
                    estimator=self.estimator
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
        ax_t.plot(t_us, d['recon_mag'], 'r', alpha=0.8, linewidth=0.8, label='DAE')
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
        ax_f.plot(d['f_r'] / 1e6, d['P_r_db'], 'r', linewidth=0.8, label='DAE')
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
        ax_c.plot(d['lags'], d['corr_r_norm'], 'r', linewidth=1.2, label='DAE')
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
    cr_colors = {4: '#E74C3C', 8: '#2ECC71', 16: '#F39C12'}
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
        ax_t.plot(t_us, d['noisy_mag'], color='cornflowerblue', linewidth=0.4, label='Noisy', alpha=0.7)
        ax_t.plot(t_us, d['clean_mag'], 'k--', linewidth=0.6, label='Clean')
        for cr in cr_list:
            ax_t.plot(t_us, d['recon_mag'][cr], color=cr_colors.get(cr, 'r'),
                      linewidth=0.6, linestyle=cr_styles.get(cr, '-'),
                      label=f'DAE CR={cr}')
        ax_t.set_ylabel(f"SNR={snr}dB\n|s(t)|", fontsize=9, fontweight='bold')
        ax_t.legend(loc='upper right', fontsize=6, framealpha=0.8, ncol=2)
        ax_t.set_xlim(0, t_us[min(299, d['n_samples'] - 1)])
        ax_t.grid(alpha=0.3)
        if i == num_rows - 1:
            ax_t.set_xlabel("Time [μs]", fontsize=10)

        # --- 频域 (所有CR的PSD叠加) ---
        ax_f = axes[i, 1]
        ax_f.plot(d['f_n'] / 1e6, d['P_n_db'], color='cornflowerblue', linewidth=0.4, label='Noisy', alpha=0.7)
        ax_f.plot(d['f_c'] / 1e6, d['P_c_db'], 'k--', linewidth=0.6, label='Clean')
        for cr in cr_list:
            if 'P_r_db' in d and cr in d['P_r_db']:
                ax_f.plot(d['f_n'] / 1e6, d['P_r_db'][cr], color=cr_colors.get(cr, 'r'),
                          linewidth=0.6, linestyle=cr_styles.get(cr, '-'),
                          alpha=0.9, label=f'DAE CR={cr}')
        ax_f.legend(loc='upper right', fontsize=6, framealpha=0.8, ncol=2)
        ax_f.grid(alpha=0.3)
        ax_f.set_xlim(-20, 20)
        if i == num_rows - 1:
            ax_f.set_xlabel("Frequency [MHz]", fontsize=10)
        ax_f.set_ylabel("PSD [dB/Hz]", fontsize=9)

        # --- 互相关 ---
        ax_c = axes[i, 2]
        ax_c.plot(d['lags'], d['corr_n_norm'], color='cornflowerblue', linewidth=0.5, label='Noisy')
        ax_c.plot(d['lags'], d['corr_c_norm'], 'k--', linewidth=0.6, label='Clean')
        for cr in cr_list:
            ax_c.plot(d['lags'], d['corr_r_norm'][cr], color=cr_colors.get(cr, 'r'),
                      linewidth=1.0, linestyle=cr_styles.get(cr, '-'),
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
