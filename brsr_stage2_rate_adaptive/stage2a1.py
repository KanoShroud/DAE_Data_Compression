"""Core logic for the preregistered BRSR Stage 2A.1 safe action gate.

Stage 2A.1 keeps the Stage 2A signal model, candidate pool, likelihood,
quadrature, posterior temperature and static plans frozen.  It only asks
whether a predicted one-bin action can safely replace the fixed B24 action.
Threshold selection, safety certification and development evaluation use
mutually disjoint trajectory seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


KEYS = ("seed", "trial", "cluster_id", "snr_db")
THRESHOLD_QUANTILES = tuple(float(value) for value in np.linspace(0.0, 0.9, 10))
ONE_SIDED_ALPHA = 0.05


@dataclass(frozen=True)
class SafeGatePolicy:
    """Frozen per-SNR decision produced before development evaluation."""

    snr_db: float
    selected_threshold: float
    fit_lcb_samples2: float
    certificate_lcb_samples2: float
    certificate_mean_gain_samples2: float
    certificate_activation_fraction: float
    certificate_leave_one_cluster_min_gain_samples2: float
    certified: bool
    decision: str


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def build_action_gate_samples(
    actions: pd.DataFrame,
    *,
    fixed_position: int,
) -> pd.DataFrame:
    """Reduce all action rows to one candidate-versus-fixed row per sample."""

    _require_columns(
        actions,
        {
            *KEYS,
            "true_tau",
            "position",
            "bin_index",
            "calibrated_predicted_voi_samples2",
            "calibrated_squared_error_samples2",
        },
        "actions",
    )
    if actions.duplicated([*KEYS, "position"]).any():
        raise ValueError("actions contain duplicate sample-position rows")

    rows: list[dict[str, object]] = []
    for key, group in actions.groupby(list(KEYS), sort=False):
        fixed = group[group["position"].eq(fixed_position)]
        if len(fixed) != 1:
            raise ValueError("fixed action is missing or duplicated")
        fixed_row = fixed.iloc[0]
        ordered = group.sort_values(
            ["calibrated_predicted_voi_samples2", "position"],
            ascending=[False, True],
            kind="stable",
        )
        candidate = ordered.iloc[0]
        candidate_position = int(candidate["position"])
        fixed_score = float(fixed_row["calibrated_predicted_voi_samples2"])
        candidate_score = float(candidate["calibrated_predicted_voi_samples2"])
        fixed_error = float(fixed_row["calibrated_squared_error_samples2"])
        candidate_error = float(candidate["calibrated_squared_error_samples2"])
        metadata = dict(zip(KEYS, key, strict=True))
        rows.append(
            {
                **metadata,
                "true_tau": float(fixed_row["true_tau"]),
                "fixed_position": int(fixed_position),
                "fixed_bin": int(fixed_row["bin_index"]),
                "candidate_position": candidate_position,
                "candidate_bin": int(candidate["bin_index"]),
                "candidate_is_nonfixed": candidate_position != fixed_position,
                "predicted_advantage_over_fixed_samples2": (
                    candidate_score - fixed_score
                ),
                "fixed_squared_error_samples2": fixed_error,
                "candidate_squared_error_samples2": candidate_error,
                "realized_advantage_over_fixed_samples2": (
                    fixed_error - candidate_error
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no gate samples were built")
    return result


def apply_threshold(samples: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply a frozen score threshold with the fixed action as fallback."""

    required = {
        "candidate_is_nonfixed",
        "predicted_advantage_over_fixed_samples2",
        "fixed_squared_error_samples2",
        "candidate_squared_error_samples2",
        "realized_advantage_over_fixed_samples2",
    }
    _require_columns(samples, required, "samples")
    selected = (
        samples["candidate_is_nonfixed"].astype(bool)
        & samples["predicted_advantage_over_fixed_samples2"].ge(threshold)
    )
    output = samples.copy()
    output["adaptive_selected"] = selected
    output["selected_position"] = np.where(
        selected,
        output["candidate_position"],
        output["fixed_position"],
    ).astype(int)
    output["selected_bin"] = np.where(
        selected,
        output["candidate_bin"],
        output["fixed_bin"],
    ).astype(int)
    output["selected_squared_error_samples2"] = np.where(
        selected,
        output["candidate_squared_error_samples2"],
        output["fixed_squared_error_samples2"],
    )
    output["realized_policy_gain_samples2"] = np.where(
        selected,
        output["realized_advantage_over_fixed_samples2"],
        0.0,
    )
    return output


def cluster_bootstrap_mean_lcb(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    alpha: float = ONE_SIDED_ALPHA,
) -> dict[str, float | int]:
    """One-sided cluster-bootstrap lower confidence bound for a mean."""

    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=str)
    if values.ndim != 1 or values.shape != cluster_ids.shape or values.size == 0:
        raise ValueError("bootstrap values and clusters must be aligned")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be finite")
    if repetitions < 20 or not 0.0 < alpha < 0.5:
        raise ValueError("invalid bootstrap configuration")
    unique = np.unique(cluster_ids)
    by_cluster = {
        cluster: np.flatnonzero(cluster_ids == cluster) for cluster in unique
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[cluster] for cluster in sampled])
        bootstrap[index] = float(np.mean(values[indices]))
    return {
        "n_rows": int(values.size),
        "n_clusters": int(len(unique)),
        "mean": float(np.mean(values)),
        "lcb": float(np.quantile(bootstrap, alpha)),
        "ucb": float(np.quantile(bootstrap, 1.0 - alpha)),
    }


def leave_one_cluster_min_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
) -> float:
    """Smallest mean after removing any one trajectory cluster."""

    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=str)
    if values.ndim != 1 or values.shape != cluster_ids.shape or values.size == 0:
        raise ValueError("leave-one-cluster inputs must be aligned")
    unique = np.unique(cluster_ids)
    if len(unique) < 2:
        return math.nan
    return float(
        min(np.mean(values[cluster_ids != cluster]) for cluster in unique)
    )


def threshold_candidates(samples: pd.DataFrame) -> tuple[float, ...]:
    """Predeclared decile candidates derived from fit-set score support."""

    eligible = samples[
        samples["candidate_is_nonfixed"].astype(bool)
        & samples["predicted_advantage_over_fixed_samples2"].gt(0.0)
    ]["predicted_advantage_over_fixed_samples2"].to_numpy(dtype=float)
    if eligible.size == 0:
        return (math.inf,)
    candidates = np.quantile(eligible, THRESHOLD_QUANTILES)
    return tuple(sorted(set(float(value) for value in candidates)))


def fit_threshold(
    samples: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    """Choose the fit-set threshold with the largest conservative mean LCB."""

    rows = []
    candidates = threshold_candidates(samples)
    for index, threshold in enumerate(candidates):
        applied = apply_threshold(samples, threshold)
        summary = cluster_bootstrap_mean_lcb(
            applied["realized_policy_gain_samples2"].to_numpy(),
            applied["cluster_id"].to_numpy(),
            repetitions=repetitions,
            seed=seed + index,
        )
        rows.append(
            {
                "threshold": threshold,
                "activation_fraction": float(applied["adaptive_selected"].mean()),
                **summary,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["lcb"].gt(0.0)]
    if eligible.empty:
        return math.inf, table
    selected = eligible.sort_values(
        ["lcb", "threshold"],
        ascending=[False, False],
        kind="stable",
    ).iloc[0]
    return float(selected["threshold"]), table


def certify_threshold(
    samples: pd.DataFrame,
    *,
    snr_db: float,
    threshold: float,
    fit_lcb: float,
    repetitions: int,
    seed: int,
) -> SafeGatePolicy:
    """Independently certify one fit-selected threshold or freeze fallback."""

    if math.isinf(threshold):
        return SafeGatePolicy(
            snr_db=float(snr_db),
            selected_threshold=math.inf,
            fit_lcb_samples2=float(fit_lcb),
            certificate_lcb_samples2=0.0,
            certificate_mean_gain_samples2=0.0,
            certificate_activation_fraction=0.0,
            certificate_leave_one_cluster_min_gain_samples2=0.0,
            certified=False,
            decision="FALLBACK_NO_FIT_LCB",
        )
    applied = apply_threshold(samples, threshold)
    gain = applied["realized_policy_gain_samples2"].to_numpy(dtype=float)
    cluster = applied["cluster_id"].to_numpy(dtype=str)
    summary = cluster_bootstrap_mean_lcb(
        gain,
        cluster,
        repetitions=repetitions,
        seed=seed,
    )
    leave_one_min = leave_one_cluster_min_mean(gain, cluster)
    certified = bool(summary["lcb"] > 0.0 and leave_one_min > 0.0)
    return SafeGatePolicy(
        snr_db=float(snr_db),
        selected_threshold=float(threshold if certified else math.inf),
        fit_lcb_samples2=float(fit_lcb),
        certificate_lcb_samples2=float(summary["lcb"]),
        certificate_mean_gain_samples2=float(summary["mean"]),
        certificate_activation_fraction=float(
            applied["adaptive_selected"].mean()
        ),
        certificate_leave_one_cluster_min_gain_samples2=leave_one_min,
        certified=certified,
        decision=(
            "CERTIFIED"
            if certified
            else "FALLBACK_CERTIFICATE_NOT_POSITIVE"
        ),
    )

