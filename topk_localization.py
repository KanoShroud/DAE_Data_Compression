"""Common non-oracle Top-K TDOA localization backends.

This module separates estimator evidence from the localization solver.  GCC
curves, direct-DFT scores, and learned V5 posteriors are converted to the same
``TopKPairEstimate`` representation before Top1-WLS, GeoBeam-WLS, or the
low-complexity Screen-and-Refit solver is applied.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from compressed_tdoa import _distinct_peak_hypotheses
from evaluate import _localize_from_tdoa_pairs, gcc_standard
from task_baselines import _lag_mask, _quality_from_score


@dataclass(frozen=True)
class TopKPairEstimate:
    lags: np.ndarray
    probabilities: np.ndarray
    weight: float
    confidence: float
    entropy: float
    variance: float
    margin: float

    def __post_init__(self):
        lags = np.asarray(self.lags, dtype=float)
        probs = np.asarray(self.probabilities, dtype=float)
        if lags.ndim != 1 or probs.ndim != 1 or lags.size != probs.size:
            raise ValueError("lags and probabilities must be equal-length vectors")
        if lags.size < 1 or not np.all(np.isfinite(lags)):
            raise ValueError("at least one finite Top-K lag is required")
        if not np.all(np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError("Top-K probabilities must be finite and non-negative")
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "probabilities", probs)


def topk_from_score(lags, score, lag_limit_samples=None, top_k=3,
                    min_separation_bins=2, sub_sample=True):
    """Convert one score curve to method-agnostic independent hypotheses."""
    lags = np.asarray(lags, dtype=float)
    score = np.asarray(score, dtype=float)
    if lags.ndim != 1 or score.ndim != 1 or lags.size != score.size:
        raise ValueError("lags and score must be equal-length vectors")
    if not np.any(np.isfinite(score)):
        raise ValueError("score contains no finite values")
    mask = _lag_mask(lags, lag_limit_samples) & np.isfinite(score)
    if not np.any(mask):
        raise ValueError("physical lag window contains no finite score")

    hypotheses = _distinct_peak_hypotheses(
        lags, score, mask, top_k=top_k, sub_sample=sub_sample,
        min_separation_bins=min_separation_bins,
    )
    if not hypotheses:
        raise RuntimeError("independent peak extraction returned no hypotheses")

    valid_score = score[mask]
    shifted = np.clip(valid_score - float(np.min(valid_score)), 0.0, None)
    if float(np.sum(shifted)) <= 1e-15:
        posterior = np.full(valid_score.size, 1.0 / valid_score.size, dtype=float)
    else:
        posterior = shifted / (float(np.sum(shifted)) + 1e-15)
    valid_lags = lags[mask]
    mean = float(np.sum(posterior * valid_lags))
    variance = float(np.sum(posterior * (valid_lags - mean) ** 2))
    entropy = float(
        -np.sum(posterior * np.log(posterior + 1e-15))
        / max(np.log(max(posterior.size, 2)), 1e-12)
    )

    peak_indices = np.asarray([item["index"] for item in hypotheses], dtype=int)
    peak_values = score[peak_indices] - float(np.min(valid_score))
    peak_probs = np.clip(peak_values, 0.0, None)
    if float(np.sum(peak_probs)) <= 1e-15:
        peak_probs = np.full(len(hypotheses), 1.0 / len(hypotheses), dtype=float)
    else:
        peak_probs = peak_probs / (float(np.sum(peak_probs)) + 1e-15)
    margin = float(
        peak_probs[0] - peak_probs[1] if peak_probs.size > 1 else peak_probs[0]
    )
    confidence = float(
        margin * max(0.0, 1.0 - entropy)
        / np.sqrt(max(variance, 0.01) + 1e-12)
    )
    weight, _, _ = _quality_from_score(score, int(peak_indices[0]), mask)
    return TopKPairEstimate(
        lags=np.asarray([item["lag"] for item in hypotheses], dtype=float),
        probabilities=peak_probs,
        weight=float(weight),
        confidence=confidence,
        entropy=entropy,
        variance=variance,
        margin=margin,
    )


def waveform_gcc_topk(sig_i, sig_j, lag_limit_samples=None, top_k=3,
                      min_separation_bins=2):
    sig_i = np.asarray(sig_i)
    sig_j = np.asarray(sig_j)
    if sig_i.ndim != 1 or sig_j.ndim != 1 or sig_i.size != sig_j.size:
        raise ValueError("waveform GCC inputs must be equal-length vectors")
    score = np.abs(gcc_standard(sig_i, sig_j))
    lags = np.arange(-(sig_i.size // 2), sig_i.size - sig_i.size // 2, dtype=float)
    if lags.size != score.size:
        raise RuntimeError("GCC lag and score lengths do not match")
    return topk_from_score(
        lags, score, lag_limit_samples=lag_limit_samples, top_k=top_k,
        min_separation_bins=min_separation_bins,
    )


def top1_pair_estimate(lag, weight, confidence=1.0):
    """Wrap one method-native Top-1 estimate for the common WLS interface."""
    lag = float(lag)
    weight = float(weight)
    if not np.isfinite(lag) or not np.isfinite(weight):
        raise ValueError("method-native lag and weight must be finite")
    return TopKPairEstimate(
        lags=np.asarray([lag], dtype=float),
        probabilities=np.asarray([1.0], dtype=float),
        weight=float(np.clip(weight, 0.02, 20.0)),
        confidence=float(confidence),
        entropy=0.0,
        variance=0.0,
        margin=1.0,
    )


def estimator_score_topk(estimator, sig_i, sig_j, lag_limit_samples=None,
                         top_k=3, min_separation_bins=2):
    """Extract Top-K from an estimator's documented native delay score."""
    if not hasattr(estimator, "score_pair") or not hasattr(estimator, "lags"):
        raise TypeError("estimator must expose score_pair() and lags")
    parameters = inspect.signature(estimator.score_pair).parameters
    if "lag_limit_samples" in parameters:
        score = estimator.score_pair(
            sig_i, sig_j, lag_limit_samples=lag_limit_samples,
        )
    else:
        score = estimator.score_pair(sig_i, sig_j)
    score = np.asarray(score, dtype=float)
    return topk_from_score(
        estimator.lags, score, lag_limit_samples=lag_limit_samples,
        top_k=top_k, min_separation_bins=min_separation_bins,
    )


def direct_dft_topk(estimator, sig_i, sig_j, lag_limit_samples=None, top_k=3,
                    min_separation_bins=2):
    """Extract common Top-K hypotheses from DirectDFTTDOAEstimator evidence."""
    if hasattr(estimator, "score_pair"):
        return estimator_score_topk(
            estimator, sig_i, sig_j,
            lag_limit_samples=lag_limit_samples, top_k=top_k,
            min_separation_bins=min_separation_bins,
        )
    required = ("selected_bins", "_steering", "lags", "signal_len")
    missing = [name for name in required if not hasattr(estimator, name)]
    if missing:
        raise TypeError(f"direct DFT estimator lacks fields: {missing}")
    sig_i = np.asarray(sig_i)
    sig_j = np.asarray(sig_j)
    spec_i = np.fft.fftshift(np.fft.fft(sig_i, norm="ortho"))
    spec_j = np.fft.fftshift(np.fft.fft(sig_j, norm="ortho"))
    cross = spec_i[estimator.selected_bins] * np.conj(spec_j[estimator.selected_bins])
    if bool(getattr(estimator, "phat", False)):
        cross = cross / (np.abs(cross) + 1e-12)
    score = np.abs(estimator._steering @ cross)
    return topk_from_score(
        estimator.lags, score, lag_limit_samples=lag_limit_samples, top_k=top_k,
        min_separation_bins=min_separation_bins,
    )


def posterior_topk(estimator, sig_i, sig_j, lag_limit_samples=None, top_k=3,
                   min_separation_bins=2):
    if not hasattr(estimator, "score_pair") or not hasattr(estimator, "lags"):
        raise TypeError("posterior estimator must expose score_pair() and lags")
    score = np.asarray(estimator.score_pair(sig_i, sig_j), dtype=float)
    return topk_from_score(
        estimator.lags, score, lag_limit_samples=lag_limit_samples, top_k=top_k,
        min_separation_bins=min_separation_bins,
    )


def _weighted_residual_cost(pos, uav_pos, pairs, lags, weights, fs, c):
    if pos is None:
        return float("inf")
    weights = np.asarray(weights, dtype=float)
    weights = weights / (np.median(weights) + 1e-12)
    values = []
    for (i, j), lag, weight in zip(pairs, lags, weights):
        residual = (
            np.linalg.norm(pos - uav_pos[i])
            - np.linalg.norm(pos - uav_pos[j])
            - float(lag) * float(c) / float(fs)
        )
        values.append(float(np.clip(weight, 0.05, 20.0)) * residual ** 2)
    return float(np.sum(values))


def _solve(uav_pos, pairs, lags, weights, fs, c, area_size):
    measurements = [
        (int(i), int(j), float(lag), float(weight))
        for (i, j), lag, weight in zip(pairs, lags, weights)
    ]
    pos, info = _localize_from_tdoa_pairs(
        uav_pos, measurements, fs=float(fs), c=float(c),
        area_size=area_size, robust_mode="standard",
    )
    return pos, info


def localize_topk(uav_pos, pairs, estimates, fs, c, area_size,
                  mode="top1", max_pairs=12, max_passes=1):
    """Run a common non-oracle localization backend over pair hypotheses."""
    if len(pairs) != len(estimates) or len(pairs) < 3:
        raise ValueError("pairs and estimates must match and contain at least 3 pairs")
    uav_pos = np.asarray(uav_pos, dtype=float)
    lags = [float(item.lags[0]) for item in estimates]
    weights = [float(item.weight) for item in estimates]
    confidences = np.asarray([float(item.confidence) for item in estimates])
    top1_pos, top1_info = _solve(uav_pos, pairs, lags, weights, fs, c, area_size)
    solver_calls = 1
    if mode == "top1" or top1_pos is None:
        return {
            "position": top1_pos, "info": top1_info, "selected_lags": lags,
            "n_solver_calls": solver_calls, "n_pairs_considered": 0,
            "n_pairs_changed": 0, "changed_indices": [],
        }

    order = np.argsort(confidences)[:min(int(max_pairs), len(pairs))]
    selected = list(lags)
    changed = set()

    if mode == "geobeam":
        best_pos = top1_pos
        best_info = top1_info
        best_cost = _weighted_residual_cost(
            best_pos, uav_pos, pairs, selected, weights, fs, c
        )
        for _ in range(max(1, int(max_passes))):
            improved = False
            for idx in order:
                current_lag = float(selected[idx])
                local_lag, local_pos = current_lag, best_pos
                local_info, local_cost = best_info, best_cost
                for alt in estimates[int(idx)].lags[1:3]:
                    proposal = list(selected)
                    proposal[int(idx)] = float(alt)
                    pos, info = _solve(
                        uav_pos, pairs, proposal, weights, fs, c, area_size
                    )
                    solver_calls += 1
                    if pos is None:
                        continue
                    cost = _weighted_residual_cost(
                        pos, uav_pos, pairs, proposal, weights, fs, c
                    )
                    if cost < local_cost * 0.999:
                        local_lag, local_pos = float(alt), pos
                        local_info, local_cost = info, cost
                if abs(local_lag - current_lag) > 1e-9:
                    selected[int(idx)] = local_lag
                    best_pos, best_info, best_cost = local_pos, local_info, local_cost
                    changed.add(int(idx))
                    improved = True
            if not improved:
                break
        final_pos, final_info = best_pos, {
            **(best_info or {}), "success": best_pos is not None,
            "common_objective_cost": float(best_cost),
            "common_topk_mode": "geobeam",
        }
    elif mode == "screen_refit":
        reference_pos = top1_pos
        for idx in order:
            idx = int(idx)
            current = float(selected[idx])
            current_cost = _weighted_residual_cost(
                reference_pos, uav_pos, [pairs[idx]], [current],
                [weights[idx]], fs, c,
            )
            best_lag, best_cost = current, current_cost
            for alt in estimates[idx].lags[1:3]:
                cost = _weighted_residual_cost(
                    reference_pos, uav_pos, [pairs[idx]], [float(alt)],
                    [weights[idx]], fs, c,
                )
                if cost < best_cost * 0.999:
                    best_lag, best_cost = float(alt), cost
            if abs(best_lag - current) > 1e-9:
                selected[idx] = best_lag
                changed.add(idx)
        if changed:
            final_pos, final_info = _solve(
                uav_pos, pairs, selected, weights, fs, c, area_size
            )
            solver_calls += 1
            top1_cost = _weighted_residual_cost(
                top1_pos, uav_pos, pairs, lags, weights, fs, c
            )
            final_cost = _weighted_residual_cost(
                final_pos, uav_pos, pairs, selected, weights, fs, c
            )
            if final_pos is None or final_cost >= top1_cost * 0.999:
                final_pos, final_info = top1_pos, top1_info
                selected = list(lags)
                changed.clear()
                final_cost = top1_cost
        else:
            final_pos, final_info = top1_pos, top1_info
            final_cost = _weighted_residual_cost(
                top1_pos, uav_pos, pairs, lags, weights, fs, c
            )
        final_info = {
            **(final_info or {}),
            "common_objective_cost": float(final_cost),
            "common_topk_mode": "screen_refit",
        }
    else:
        raise ValueError("mode must be top1, geobeam, or screen_refit")

    return {
        "position": final_pos,
        "info": final_info,
        "selected_lags": selected,
        "n_solver_calls": int(solver_calls),
        "n_pairs_considered": int(len(order)),
        "n_pairs_changed": int(len(changed)),
        "changed_indices": sorted(changed),
    }
