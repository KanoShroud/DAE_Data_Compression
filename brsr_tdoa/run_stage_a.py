"""BRSR-TDOA 阶段 A 独立实验入口。

在 PyCharm 中直接运行本文件。顶部 ``PYCHARM_RUN_MODE`` 默认为
``research``；Codex 轻量验证使用命令行 ``--mode smoke``，不修改默认值。
该脚本不导入主项目、PASR、PFRS、urban8 或定位链路。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brsr_tdoa.brsr_core import (  # noqa: E402
    DecisionState,
    NestedComplexQuantizer,
    PolicyResult,
    available_actions,
    build_default_model,
    build_likelihood_context,
    evaluate_dynamic_program,
    evaluate_fixed_levels,
    evaluate_one_step,
    expected_risk_after_action,
    initialize_state,
    posterior_statistics,
    run_dynamic_policy,
    state_after_observed_action,
)


PYCHARM_RUN_MODE = "research"


@dataclass(frozen=True)
class StageAConfig:
    mode: str
    snr_db: tuple[float, ...]
    development_seeds: tuple[int, ...]
    locked_seeds: tuple[int, ...]
    development_trials: int
    locked_trials: int
    dp_trials_per_condition: int
    tau_step: float
    coarse_bits: int
    max_bits: int
    quantizer_clip_abs: float
    use_public_b_variance_normalization: bool
    lambda_per_bit: tuple[float, ...]
    max_decisions: int
    dp_horizon: int
    progress_every: int


CONFIGS = {
    "smoke": StageAConfig(
        mode="smoke",
        snr_db=(-5.0, 5.0),
        development_seeds=(20267001,),
        locked_seeds=(20268001,),
        development_trials=3,
        locked_trials=5,
        dp_trials_per_condition=2,
        tau_step=1.0,
        coarse_bits=2,
        max_bits=3,
        quantizer_clip_abs=4.0,
        use_public_b_variance_normalization=True,
        lambda_per_bit=(0.0, 0.01),
        max_decisions=2,
        dp_horizon=2,
        progress_every=2,
    ),
    "research": StageAConfig(
        mode="research",
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
        development_seeds=(20267001, 20267002),
        locked_seeds=(20268001, 20268002, 20268003),
        development_trials=12,
        locked_trials=30,
        dp_trials_per_condition=5,
        tau_step=0.5,
        coarse_bits=2,
        max_bits=4,
        quantizer_clip_abs=4.0,
        use_public_b_variance_normalization=True,
        lambda_per_bit=(0.0, 0.005, 0.02),
        max_decisions=2,
        dp_horizon=2,
        progress_every=10,
    ),
}


def _validate_config(config: StageAConfig) -> None:
    if config.mode not in CONFIGS:
        raise ValueError(f"unsupported Stage A mode: {config.mode}")
    snr = np.asarray(config.snr_db, dtype=float)
    if snr.size == 0 or not np.all(np.isfinite(snr)) or np.unique(snr).size != snr.size:
        raise ValueError("snr_db must contain unique finite values")
    if not config.development_seeds or not config.locked_seeds:
        raise ValueError("development and locked seeds must both be non-empty")
    if len(set(config.development_seeds)) != len(config.development_seeds):
        raise ValueError("development seeds must be unique")
    if len(set(config.locked_seeds)) != len(config.locked_seeds):
        raise ValueError("locked seeds must be unique")
    if set(config.development_seeds) & set(config.locked_seeds):
        raise ValueError("development and locked seeds must be disjoint")
    if config.development_trials <= 0 or config.locked_trials <= 0:
        raise ValueError("trial counts must be positive")
    if (
        config.dp_trials_per_condition < 0
        or config.dp_trials_per_condition
        > min(config.development_trials, config.locked_trials)
    ):
        raise ValueError("dp_trials_per_condition exceeds an available trial count")
    if not math.isfinite(config.tau_step) or config.tau_step <= 0.0:
        raise ValueError("tau_step must be positive and finite")
    if not 1 <= config.coarse_bits < config.max_bits:
        raise ValueError("bit depths must satisfy 1 <= coarse_bits < max_bits")
    if (
        not math.isfinite(config.quantizer_clip_abs)
        or config.quantizer_clip_abs <= 0.0
    ):
        raise ValueError("quantizer_clip_abs must be positive and finite")
    if not isinstance(config.use_public_b_variance_normalization, bool):
        raise ValueError("use_public_b_variance_normalization must be Boolean")
    lambdas = np.asarray(config.lambda_per_bit, dtype=float)
    if (
        lambdas.size == 0
        or not np.all(np.isfinite(lambdas))
        or np.any(lambdas < 0.0)
        or np.unique(lambdas).size != lambdas.size
    ):
        raise ValueError("lambda_per_bit must contain unique finite non-negative values")
    if config.max_decisions < 0:
        raise ValueError("max_decisions must be non-negative")
    if config.dp_horizon < 1 or config.dp_horizon > max(config.max_decisions, 1):
        raise ValueError("dp_horizon must lie in [1, max_decisions]")
    if config.progress_every <= 0:
        raise ValueError("progress_every must be positive")


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex_normal(
    rng: np.random.Generator, variance: np.ndarray | float, shape: tuple[int, ...]
) -> np.ndarray:
    standard = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / math.sqrt(2.0)
    return standard * np.sqrt(variance)


def _action_feedback_bits(n_bins: int) -> int:
    # 固定动作码本包含 stop、每个频点的 acquire 和 refine。
    return int(math.ceil(math.log2(1 + 2 * n_bins)))


def _public_b_scale(model, sigma2: float, enabled: bool) -> float:
    """由公开噪声方差和先验二阶矩确定 B 端固定条件尺度。"""

    if not enabled:
        return 1.0
    rho_second_moment = float(
        np.sum(model.rho_prior * np.abs(model.rho_values) ** 2)
    )
    mean_signal_power = rho_second_moment * float(np.mean(model.source_power))
    return math.sqrt(mean_signal_power + sigma2)


def _fixed_level_sets(
    coarse_bits: int, max_bits: int, n_bins: int, base: tuple[int, ...]
) -> dict[str, tuple[int, ...]]:
    base_levels = [0] * n_bins
    for position in base:
        base_levels[position] = coarse_bits

    base_max = base_levels.copy()
    for position in base:
        base_max[position] = max_bits

    outer_coarse = base_levels.copy()
    for position in (0, n_bins - 1):
        outer_coarse[position] = coarse_bits

    coarse_all = [coarse_bits] * n_bins
    full_all = [max_bits] * n_bins
    return {
        "Fixed-Base": tuple(base_levels),
        "Fixed-BaseMax": tuple(base_max),
        "Fixed-BaseOuter": tuple(outer_coarse),
        "Fixed-CoarseAll": tuple(coarse_all),
        "Fixed-FullAll": tuple(full_all),
    }


def _policy_row(
    *,
    stage: str,
    seed: int,
    snr_db: float,
    trial: int,
    true_tau: float,
    true_rho_index: int,
    lambda_per_bit: float | None,
    result: PolicyResult,
    clipping_rate: float,
    public_b_scale: float,
    is_dp_subset: bool,
) -> dict[str, object]:
    error = result.estimate - true_tau
    ledger = result.ledger
    return {
        "stage": stage,
        "seed": seed,
        "snr_db": snr_db,
        "trial": trial,
        "method": result.method,
        "lambda_per_bit": lambda_per_bit,
        "true_tau": true_tau,
        "true_rho_index": true_rho_index,
        "estimate_tau": result.estimate,
        "error_samples": error,
        "squared_error_samples2": error**2,
        "abs_error_samples": abs(error),
        "posterior_risk_samples2": result.posterior_risk,
        "posterior_nll": result.posterior_nll,
        "credible90_contains_true": result.credible90_contains_true,
        "total_bits": ledger.total_bits,
        "base_payload_bits": ledger.base_payload_bits,
        "refinement_payload_bits": ledger.refinement_payload_bits,
        "action_feedback_bits": ledger.action_feedback_bits,
        "stop_feedback_bits": ledger.stop_feedback_bits,
        "header_bits": ledger.header_bits,
        "scale_bits": ledger.scale_bits,
        "selected_levels": " ".join(str(value) for value in result.selected_levels),
        "n_actions": sum(
            action.get("action") != "stop" for action in result.action_trace
        ),
        "stopped_by_policy": result.stopped_by_policy,
        "quantizer_clipping_rate": clipping_rate,
        "public_b_scale": public_b_scale,
        "is_dp_subset": is_dp_subset,
    }


def _rename_policy(result: PolicyResult, method: str) -> PolicyResult:
    return PolicyResult(
        method=method,
        estimate=result.estimate,
        posterior_risk=result.posterior_risk,
        posterior_nll=result.posterior_nll,
        credible90_contains_true=result.credible90_contains_true,
        ledger=result.ledger,
        selected_levels=result.selected_levels,
        action_trace=result.action_trace,
        stopped_by_policy=result.stopped_by_policy,
    )


def _initial_action_rows(
    *,
    stage: str,
    seed: int,
    snr_db: float,
    trial: int,
    true_tau: float,
    state: DecisionState,
    fine_real_codes: np.ndarray,
    fine_imag_codes: np.ndarray,
    quantizer: NestedComplexQuantizer,
    context,
    model,
    config: StageAConfig,
    feedback_bits: int,
) -> list[dict[str, object]]:
    estimate_before, risk_before = posterior_statistics(state, model)
    squared_before = (estimate_before - true_tau) ** 2
    rows = []
    for action in available_actions(
        state,
        coarse_bits=config.coarse_bits,
        max_bits=config.max_bits,
        feedback_bits=feedback_bits,
        model=model,
    ):
        expected_after = expected_risk_after_action(
            state, action, quantizer, context, model
        )
        observed = state_after_observed_action(
            state,
            action,
            fine_real_codes,
            fine_imag_codes,
            quantizer,
            context,
        )
        estimate_after, risk_after = posterior_statistics(observed, model)
        squared_after = (estimate_after - true_tau) ** 2
        rows.append(
            {
                "stage": stage,
                "seed": seed,
                "snr_db": snr_db,
                "trial": trial,
                "action": action.label,
                "action_bits": action.total_bits,
                "expected_risk_before": risk_before,
                "expected_risk_after": expected_after,
                "expected_risk_reduction": risk_before - expected_after,
                "realized_posterior_risk_after": risk_after,
                "realized_posterior_risk_reduction": risk_before - risk_after,
                "realized_squared_error_before": squared_before,
                "realized_squared_error_after": squared_after,
                "realized_squared_error_reduction": squared_before - squared_after,
            }
        )
    if rows:
        expected_best = max(rows, key=lambda row: row["expected_risk_reduction"])
        realized_best = max(
            rows, key=lambda row: row["realized_squared_error_reduction"]
        )
        for row in rows:
            row["is_expected_best"] = row["action"] == expected_best["action"]
            row["is_realized_oracle_best"] = row["action"] == realized_best["action"]
    return rows


def _run_stage(
    *,
    stage: str,
    seeds: tuple[int, ...],
    n_trials: int,
    config: StageAConfig,
    model,
    quantizer: NestedComplexQuantizer,
    rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> None:
    feedback_bits = _action_feedback_bits(model.n_bins)
    fixed_levels = _fixed_level_sets(
        config.coarse_bits, config.max_bits, model.n_bins, model.base_positions
    )
    total = len(seeds) * len(config.snr_db) * n_trials
    done = 0
    start = time.perf_counter()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        true_tau_indices = rng.choice(
            model.n_tau,
            size=n_trials,
            replace=True,
            p=model.tau_prior,
        )
        true_rho_indices = rng.choice(
            model.n_rho,
            size=n_trials,
            replace=True,
            p=model.rho_prior,
        )
        source_standard = _complex_normal(
            rng, 1.0, (n_trials, model.n_bins)
        )
        noise_a_standard = _complex_normal(
            rng, 1.0, (n_trials, model.n_bins)
        )
        noise_b_standard = _complex_normal(
            rng, 1.0, (n_trials, model.n_bins)
        )

        for snr_db in config.snr_db:
            sigma2 = 10.0 ** (-snr_db / 10.0)
            public_b_scale = _public_b_scale(
                model,
                sigma2,
                config.use_public_b_variance_normalization,
            )
            source = source_standard * np.sqrt(model.source_power)[None, :]
            y_a = source + math.sqrt(sigma2) * noise_a_standard
            tau = model.tau_values[true_tau_indices]
            rho = model.rho_values[true_rho_indices]
            steering = np.exp(-1j * tau[:, None] * model.omega[None, :])
            y_b = (
                rho[:, None] * source * steering
                + math.sqrt(sigma2) * noise_b_standard
            ) / public_b_scale

            for trial in range(n_trials):
                true_tau = float(tau[trial])
                fine_real, fine_imag = quantizer.encode_fine(y_b[trial])
                clipping_rate = float(
                    np.mean(
                        (np.abs(y_b[trial].real) >= quantizer.clip_abs)
                        | (np.abs(y_b[trial].imag) >= quantizer.clip_abs)
                    )
                )
                context = build_likelihood_context(
                    y_a[trial],
                    model,
                    sigma2,
                    sigma2,
                    b_public_scale=public_b_scale,
                )
                initial = initialize_state(
                    fine_real_codes=fine_real,
                    fine_imag_codes=fine_imag,
                    coarse_bits=config.coarse_bits,
                    quantizer=quantizer,
                    context=context,
                    model=model,
                )

                action_rows.extend(
                    _initial_action_rows(
                        stage=stage,
                        seed=seed,
                        snr_db=snr_db,
                        trial=trial,
                        true_tau=true_tau,
                        state=initial,
                        fine_real_codes=fine_real,
                        fine_imag_codes=fine_imag,
                        quantizer=quantizer,
                        context=context,
                        model=model,
                        config=config,
                        feedback_bits=feedback_bits,
                    )
                )

                for name, levels in fixed_levels.items():
                    fixed = evaluate_fixed_levels(
                        name=name,
                        levels=levels,
                        fine_real_codes=fine_real,
                        fine_imag_codes=fine_imag,
                        true_tau=true_tau,
                        quantizer=quantizer,
                        context=context,
                        model=model,
                    )
                    rows.append(
                        _policy_row(
                            stage=stage,
                            seed=seed,
                            snr_db=snr_db,
                            trial=trial,
                            true_tau=true_tau,
                            true_rho_index=int(true_rho_indices[trial]),
                            lambda_per_bit=None,
                            result=fixed,
                            clipping_rate=clipping_rate,
                            public_b_scale=public_b_scale,
                            is_dp_subset=False,
                        )
                    )

                for lambda_value in config.lambda_per_bit:
                    greedy_eval = evaluate_one_step(
                        initial,
                        lambda_per_bit=lambda_value,
                        stop_feedback_bits=feedback_bits,
                        coarse_bits=config.coarse_bits,
                        max_bits=config.max_bits,
                        action_feedback_bits=feedback_bits,
                        quantizer=quantizer,
                        context=context,
                        model=model,
                    )
                    greedy = run_dynamic_policy(
                        method="greedy",
                        initial_state=initial,
                        fine_real_codes=fine_real,
                        fine_imag_codes=fine_imag,
                        true_tau=true_tau,
                        lambda_per_bit=lambda_value,
                        max_decisions=config.max_decisions,
                        dp_horizon=config.dp_horizon,
                        coarse_bits=config.coarse_bits,
                        max_bits=config.max_bits,
                        action_feedback_bits=feedback_bits,
                        stop_feedback_bits=feedback_bits,
                        quantizer=quantizer,
                        context=context,
                        model=model,
                    )
                    method_suffix = f"{lambda_value:g}"
                    greedy = _rename_policy(
                        greedy, f"BRSR-Greedy-lambda{method_suffix}"
                    )
                    rows.append(
                        _policy_row(
                            stage=stage,
                            seed=seed,
                            snr_db=snr_db,
                            trial=trial,
                            true_tau=true_tau,
                            true_rho_index=int(true_rho_indices[trial]),
                            lambda_per_bit=lambda_value,
                            result=greedy,
                            clipping_rate=clipping_rate,
                            public_b_scale=public_b_scale,
                            is_dp_subset=False,
                        )
                    )

                    if trial < config.dp_trials_per_condition:
                        dp_eval = evaluate_dynamic_program(
                            initial,
                            depth=config.dp_horizon,
                            lambda_per_bit=lambda_value,
                            stop_feedback_bits=feedback_bits,
                            coarse_bits=config.coarse_bits,
                            max_bits=config.max_bits,
                            action_feedback_bits=feedback_bits,
                            quantizer=quantizer,
                            context=context,
                            model=model,
                        )
                        if dp_eval.value > greedy_eval.value + 1e-10:
                            raise RuntimeError(
                                "finite-horizon DP value exceeds the one-step value"
                            )
                        dp = run_dynamic_policy(
                            method="dynamic_program",
                            initial_state=initial,
                            fine_real_codes=fine_real,
                            fine_imag_codes=fine_imag,
                            true_tau=true_tau,
                            lambda_per_bit=lambda_value,
                            max_decisions=config.max_decisions,
                            dp_horizon=config.dp_horizon,
                            coarse_bits=config.coarse_bits,
                            max_bits=config.max_bits,
                            action_feedback_bits=feedback_bits,
                            stop_feedback_bits=feedback_bits,
                            quantizer=quantizer,
                            context=context,
                            model=model,
                        )
                        dp = _rename_policy(
                            dp,
                            f"BRSR-DP-DiagnosticSubset-lambda{method_suffix}",
                        )
                        rows.append(
                            _policy_row(
                                stage=stage,
                                seed=seed,
                                snr_db=snr_db,
                                trial=trial,
                                true_tau=true_tau,
                                true_rho_index=int(true_rho_indices[trial]),
                                lambda_per_bit=lambda_value,
                                result=dp,
                                clipping_rate=clipping_rate,
                                public_b_scale=public_b_scale,
                                is_dp_subset=True,
                            )
                        )
                        decision_rows.append(
                            {
                                "stage": stage,
                                "seed": seed,
                                "snr_db": snr_db,
                                "trial": trial,
                                "lambda_per_bit": lambda_value,
                                "greedy_initial_action": (
                                    "stop"
                                    if greedy_eval.action is None
                                    else greedy_eval.action.label
                                ),
                                "dp_initial_action": (
                                    "stop"
                                    if dp_eval.action is None
                                    else dp_eval.action.label
                                ),
                                "greedy_initial_value": greedy_eval.value,
                                "dp_initial_value": dp_eval.value,
                                "dp_not_worse": dp_eval.value
                                <= greedy_eval.value + 1e-10,
                            }
                        )

                done += 1
                if done % config.progress_every == 0 or done == total:
                    elapsed = time.perf_counter() - start
                    eta = elapsed / done * (total - done)
                    print(
                        f"[{stage}] {done}/{total} SNR={snr_db:g} "
                        f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                        flush=True,
                    )


def _metric_row(group: pd.DataFrame) -> pd.Series:
    squared = group["squared_error_samples2"].to_numpy(dtype=float)
    absolute = group["abs_error_samples"].to_numpy(dtype=float)
    return pd.Series(
        {
            "n_trials": len(group),
            "rmse_samples": math.sqrt(float(np.mean(squared))),
            "mae_samples": float(np.mean(absolute)),
            "p95_abs_error_samples": float(np.quantile(absolute, 0.95)),
            "gross_error_rate_gt_1": float(np.mean(absolute > 1.0)),
            "mean_posterior_risk_samples2": float(
                group["posterior_risk_samples2"].mean()
            ),
            "empirical_mse_samples2": float(np.mean(squared)),
            "risk_calibration_ratio": float(
                np.mean(squared)
                / max(float(group["posterior_risk_samples2"].mean()), 1e-12)
            ),
            "credible90_coverage": float(
                group["credible90_contains_true"].mean()
            ),
            "mean_posterior_nll": float(group["posterior_nll"].mean()),
            "mean_total_bits": float(group["total_bits"].mean()),
            "max_total_bits": int(group["total_bits"].max()),
            "mean_actions": float(group["n_actions"].mean()),
            "stop_rate": float(group["stopped_by_policy"].mean()),
            "mean_clipping_rate": float(
                group["quantizer_clipping_rate"].mean()
            ),
        }
    )


def _summaries(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_snr = (
        rows.groupby(["stage", "method", "snr_db"], sort=False)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
    overall = (
        rows.groupby(["stage", "method"], sort=False)
        .apply(_metric_row, include_groups=False)
        .reset_index()
    )
    return by_snr, overall


def _validate_results(
    rows: pd.DataFrame,
    action_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    config: StageAConfig,
) -> dict[str, object]:
    required_numeric = [
        "seed",
        "snr_db",
        "trial",
        "true_tau",
        "true_rho_index",
        "estimate_tau",
        "error_samples",
        "squared_error_samples2",
        "abs_error_samples",
        "posterior_risk_samples2",
        "posterior_nll",
        "total_bits",
        "base_payload_bits",
        "refinement_payload_bits",
        "action_feedback_bits",
        "stop_feedback_bits",
        "header_bits",
        "scale_bits",
        "n_actions",
        "quantizer_clipping_rate",
        "public_b_scale",
    ]
    if not np.isfinite(rows[required_numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("trial results contain non-finite values")
    dynamic_lambda = rows.loc[
        rows["method"].str.startswith("BRSR"), "lambda_per_bit"
    ].to_numpy(dtype=float)
    if not np.isfinite(dynamic_lambda).all():
        raise RuntimeError("dynamic policy rows must record a finite lambda_per_bit")
    components = rows[
        [
            "base_payload_bits",
            "refinement_payload_bits",
            "action_feedback_bits",
            "stop_feedback_bits",
            "header_bits",
            "scale_bits",
        ]
    ].sum(axis=1)
    if not np.array_equal(components.to_numpy(), rows["total_bits"].to_numpy()):
        raise RuntimeError("communication bit ledger identity failed")
    if (components.to_numpy(dtype=float) < 0.0).any():
        raise RuntimeError("communication bit ledger contains a negative component")
    if set(config.development_seeds) & set(config.locked_seeds):
        raise RuntimeError("development and locked seeds overlap")
    if rows.duplicated(["stage", "seed", "snr_db", "trial", "method"]).any():
        raise RuntimeError("trial results contain duplicate method rows")
    if action_rows.duplicated(
        ["stage", "seed", "snr_db", "trial", "action"]
    ).any():
        raise RuntimeError("action diagnostics contain duplicate action rows")
    if decision_rows.duplicated(
        ["stage", "seed", "snr_db", "trial", "lambda_per_bit"]
    ).any():
        raise RuntimeError("dynamic-program diagnostics contain duplicate rows")
    if not action_rows.empty:
        action_numeric = action_rows.select_dtypes(include=[np.number]).to_numpy(
            dtype=float
        )
        if not np.isfinite(action_numeric).all():
            raise RuntimeError("action diagnostics contain non-finite values")
        expected_after = action_rows["expected_risk_after"].to_numpy(dtype=float)
        expected_before = action_rows["expected_risk_before"].to_numpy(dtype=float)
        max_violation = float(np.max(expected_after - expected_before))
        if max_violation > 2e-10:
            raise RuntimeError("expected posterior risk increased after observation")
    else:
        max_violation = math.nan
    if not decision_rows.empty and not decision_rows["dp_not_worse"].all():
        raise RuntimeError("dynamic-program value ordering failed")

    tau_counts = rows.groupby(["stage", "seed", "snr_db", "trial"])[
        "true_tau"
    ].nunique()
    if not (tau_counts == 1).all():
        raise RuntimeError("methods do not share the same true TDOA")
    rho_counts = rows.groupby(["stage", "seed", "snr_db", "trial"])[
        "true_rho_index"
    ].nunique()
    if not (rho_counts == 1).all():
        raise RuntimeError("methods do not share the same complex-gain state")
    scale_counts = rows.groupby(["stage", "snr_db"])["public_b_scale"].nunique()
    if not (scale_counts == 1).all():
        raise RuntimeError("methods do not share one public B-side scale per SNR")
    unique_trials = rows.drop_duplicates(["stage", "seed", "snr_db", "trial"])
    clipping = float(unique_trials["quantizer_clipping_rate"].mean())
    return {
        "bit_ledger_identity_pass": True,
        "development_locked_seeds_disjoint": True,
        "common_random_tdoa_pass": True,
        "common_random_rho_pass": True,
        "common_public_b_scale_pass": True,
        "unique_trial_method_rows_pass": True,
        "unique_action_rows_pass": True,
        "unique_dp_rows_pass": True,
        "expected_risk_monotonicity_pass": True,
        "max_expected_risk_increase": max_violation,
        "dp_value_not_worse_than_one_step_pass": True,
        "mean_quantizer_clipping_rate": clipping,
        "trial_rows": int(len(rows)),
        "action_rows": int(len(action_rows)),
        "decision_rows": int(len(decision_rows)),
    }


def _action_summary(action_rows: pd.DataFrame) -> dict[str, float]:
    locked = action_rows.loc[action_rows["stage"] == "locked"].copy()
    if locked.empty:
        return {
            "spearman_expected_vs_realized_posterior": math.nan,
            "spearman_expected_vs_realized_squared_error": math.nan,
            "expected_oracle_action_agreement": math.nan,
            "cluster_mean_risk_residual": math.nan,
            "cluster_risk_residual_standard_error": math.nan,
            "cluster_risk_residual_z": math.nan,
            "n_independent_trajectories": 0.0,
        }
    posterior_corr = spearmanr(
        locked["expected_risk_reduction"],
        locked["realized_posterior_risk_reduction"],
    ).statistic
    squared_corr = spearmanr(
        locked["expected_risk_reduction"],
        locked["realized_squared_error_reduction"],
    ).statistic
    expected_actions = locked.loc[locked["is_expected_best"], [
        "seed",
        "snr_db",
        "trial",
        "action",
    ]]
    oracle_actions = locked.loc[locked["is_realized_oracle_best"], [
        "seed",
        "snr_db",
        "trial",
        "action",
    ]]
    merged = expected_actions.merge(
        oracle_actions,
        on=["seed", "snr_db", "trial"],
        suffixes=("_expected", "_oracle"),
        validate="one_to_one",
    )
    locked["risk_prediction_residual"] = (
        locked["realized_posterior_risk_after"]
        - locked["expected_risk_after"]
    )
    # 同一 seed/trial 在多个 SNR 和动作下使用共同随机数，先聚合到独立轨迹，
    # 再检验条件期望残差是否与零相容，避免把相关行误当成独立样本。
    cluster_residual = locked.groupby(["seed", "trial"], sort=False)[
        "risk_prediction_residual"
    ].mean()
    cluster_mean = float(cluster_residual.mean())
    if len(cluster_residual) > 1:
        cluster_se = float(
            cluster_residual.std(ddof=1) / math.sqrt(len(cluster_residual))
        )
    else:
        cluster_se = math.nan
    if math.isfinite(cluster_se) and cluster_se > 0.0:
        cluster_z = cluster_mean / cluster_se
    elif abs(cluster_mean) <= 1e-12:
        cluster_z = 0.0
    else:
        cluster_z = math.copysign(math.inf, cluster_mean)
    return {
        "spearman_expected_vs_realized_posterior": float(posterior_corr),
        "spearman_expected_vs_realized_squared_error": float(squared_corr),
        "expected_oracle_action_agreement": float(
            np.mean(merged["action_expected"] == merged["action_oracle"])
        ),
        "cluster_mean_risk_residual": cluster_mean,
        "cluster_risk_residual_standard_error": cluster_se,
        "cluster_risk_residual_z": float(cluster_z),
        "n_independent_trajectories": float(len(cluster_residual)),
    }


def _stage_a_gate(
    config: StageAConfig,
    by_snr: pd.DataFrame,
    overall: pd.DataFrame,
    action_summary: dict[str, float],
    validation: dict[str, object],
) -> dict[str, object]:
    if config.mode == "smoke":
        return {
            "status": "SMOKE_ONLY",
            "reason": "Smoke validates implementation only; it cannot rank methods.",
        }
    locked_overall = overall.loc[overall["stage"] == "locked"]
    locked_by_snr = by_snr.loc[by_snr["stage"] == "locked"]
    greedy_overall = locked_overall.loc[
        locked_overall["method"].str.startswith("BRSR-Greedy")
    ]
    greedy_by_snr = locked_by_snr.loc[
        locked_by_snr["method"].str.startswith("BRSR-Greedy")
    ]
    if greedy_overall.empty or greedy_by_snr.empty:
        raise RuntimeError("Stage A gate cannot find locked greedy policies")
    overall_coverage_errors = np.abs(
        greedy_overall["credible90_coverage"].to_numpy(dtype=float) - 0.90
    )
    by_snr_undercoverage = np.maximum(
        0.90 - greedy_by_snr["credible90_coverage"].to_numpy(dtype=float),
        0.0,
    )
    worst_overall_calibration_error = float(np.max(overall_coverage_errors))
    worst_by_snr_undercoverage = float(np.max(by_snr_undercoverage))
    action_residual_z = abs(action_summary["cluster_risk_residual_z"])
    checks = {
        "all_greedy_overall_coverage_within_0p10": (
            worst_overall_calibration_error <= 0.10
        ),
        "all_greedy_by_snr_undercoverage_below_0p15": (
            worst_by_snr_undercoverage <= 0.15
        ),
        "action_risk_expectation_cluster_z_below_3p29": (
            math.isfinite(action_residual_z) and action_residual_z <= 3.29
        ),
        "expected_risk_monotonicity": bool(
            validation["expected_risk_monotonicity_pass"]
        ),
        "dp_value_ordering": bool(
            validation["dp_value_not_worse_than_one_step_pass"]
        ),
        "mean_clipping_below_0p02": float(
            validation["mean_quantizer_clipping_rate"]
        )
        <= 0.02,
    }
    return {
        "status": (
            "STAGE_A_MECHANISM_PASS"
            if all(checks.values())
            else "STAGE_A_MECHANISM_NO_GO"
        ),
        "checks": checks,
        "worst_greedy_overall_coverage_error": worst_overall_calibration_error,
        "worst_greedy_by_snr_undercoverage": worst_by_snr_undercoverage,
        "action_summary": action_summary,
        "note": (
            "This gate validates posterior/action mechanics only. Spearman and "
            "oracle-action agreement are descriptive, not pass/fail criteria. "
            "No locked-set lambda is selected. The gate does not approve CR16 "
            "or urban8 experiments."
        ),
    }


def _plot_pareto(overall: pd.DataFrame, output: Path) -> None:
    locked = overall.loc[overall["stage"] == "locked"].copy()
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for _, row in locked.iterrows():
        method = str(row["method"])
        is_dp = method.startswith("BRSR-DP")
        marker = "*" if is_dp else ("o" if method.startswith("BRSR") else "s")
        color = "#C43C39" if method.startswith("BRSR") else "#2474A6"
        axis.scatter(
            row["mean_total_bits"],
            row["rmse_samples"],
            marker=marker,
            s=78 if is_dp else 44,
            color=color,
            alpha=0.9,
        )
        axis.annotate(
            method.replace("BRSR-", ""),
            (row["mean_total_bits"], row["rmse_samples"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Mean transmitted bits")
    axis.set_ylabel("TDOA RMSE (samples)")
    axis.grid(True, alpha=0.25)
    axis.set_title("Stage A risk-bit Pareto (DP uses diagnostic subset)")
    figure.tight_layout()
    figure.savefig(output, format="svg")
    plt.close(figure)


def _plot_calibration(by_snr: pd.DataFrame, output: Path) -> None:
    locked = by_snr.loc[by_snr["stage"] == "locked"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for method, group in locked.groupby("method", sort=False):
        if method.startswith("BRSR-DP"):
            continue
        axes[0].plot(
            group["mean_posterior_risk_samples2"],
            group["empirical_mse_samples2"],
            marker="o",
            linewidth=1.1,
            label=method,
        )
        axes[1].plot(
            group["snr_db"],
            group["credible90_coverage"],
            marker="o",
            linewidth=1.1,
            label=method,
        )
    maximum = max(
        float(locked["mean_posterior_risk_samples2"].max()),
        float(locked["empirical_mse_samples2"].max()),
    )
    axes[0].plot([0.0, maximum], [0.0, maximum], "--", color="black", linewidth=1.0)
    axes[0].set_xlabel("Mean posterior risk")
    axes[0].set_ylabel("Empirical MSE")
    axes[0].grid(True, alpha=0.25)
    axes[1].axhline(0.90, linestyle="--", color="black", linewidth=1.0)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("90% credible interval coverage")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    figure.subplots_adjust(bottom=0.26, wspace=0.28)
    figure.savefig(output, format="svg")
    plt.close(figure)


def _plot_action_calibration(action_rows: pd.DataFrame, output: Path) -> None:
    locked = action_rows.loc[action_rows["stage"] == "locked"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    sample = locked
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=20260717)
    axes[0].scatter(
        sample["expected_risk_reduction"],
        sample["realized_posterior_risk_reduction"],
        s=7,
        alpha=0.18,
        color="#2474A6",
    )
    axes[1].scatter(
        sample["expected_risk_reduction"],
        sample["realized_squared_error_reduction"],
        s=7,
        alpha=0.18,
        color="#C43C39",
    )
    axes[0].set_xlabel("Expected posterior-risk reduction")
    axes[0].set_ylabel("Realized posterior-risk reduction")
    axes[1].set_xlabel("Expected posterior-risk reduction")
    axes[1].set_ylabel("Realized squared-error reduction")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, format="svg")
    plt.close(figure)


def _write_outputs(
    *,
    output_dir: Path,
    config: StageAConfig,
    model,
    quantizer: NestedComplexQuantizer,
    rows: pd.DataFrame,
    action_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    validation: dict[str, object],
    action_summary: dict[str, float],
    gate: dict[str, object],
) -> None:
    by_snr, overall = _summaries(rows)
    rows.to_csv(output_dir / "brsr_stage_a_trials.csv", index=False, encoding="utf-8-sig")
    action_rows.to_csv(
        output_dir / "brsr_stage_a_action_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decision_rows.to_csv(
        output_dir / "brsr_stage_a_dp_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_snr.to_csv(
        output_dir / "brsr_stage_a_summary_by_snr.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        output_dir / "brsr_stage_a_summary_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_pareto(overall, output_dir / "BRSR_StageA_Risk_Bit_Pareto.svg")
    _plot_calibration(by_snr, output_dir / "BRSR_StageA_Posterior_Calibration.svg")
    _plot_action_calibration(
        action_rows, output_dir / "BRSR_StageA_Action_Calibration.svg"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "go_no_go.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "method": "BRSR-TDOA Stage A",
        "mode": config.mode,
        "config": asdict(config),
        "model": {
            "n_time": model.n_time,
            "bin_indices": model.bin_indices.tolist(),
            "source_power": model.source_power.tolist(),
            "tau_values": model.tau_values.tolist(),
            "rho_values": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in model.rho_values
            ],
            "base_positions": list(model.base_positions),
        },
        "quantizer": asdict(quantizer),
        "action_feedback_bits": _action_feedback_bits(model.n_bins),
        "probabilistic_assumptions": {
            "source": "independent circular complex Gaussian per selected bin",
            "rho": "finite discrete prior, unknown to estimator",
            "tau": "finite physical-support prior",
            "quantization": "public fixed-range nested scalar I/Q quantizer",
            "a_side_budget": "A-side spectrum is local at fusion in Stage A only",
            "b_normalization": (
                "Per-SNR public prior-predictive RMS scale; it depends only on "
                "known sigma2 and prior second moments, never on a trial sample."
            ),
            "snr_definition": (
                "Nominal SNR is mean pre-rho source power divided by equal A/B "
                "complex-noise variance; B-side received SNR also includes |rho|^2."
            ),
        },
        "gate_protocol": {
            "locked_lambda_selection": "none",
            "coverage": "all pre-registered greedy lambdas must pass",
            "action_calibration": (
                "clustered mean realized-minus-expected posterior-risk residual"
            ),
            "spearman_role": "descriptive only",
        },
        "communication_accounting": {
            "public_b_scale_bits": 0,
            "reason": (
                "The normalization scale is a deterministic function of the "
                "pre-registered SNR condition and public priors."
            ),
        },
        "reproducibility": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "common_random_numbers_across_snr": True,
            "development_locked_seed_overlap": False,
        },
        "action_summary": action_summary,
        "validation": validation,
        "gate": gate,
        "source_sha256": {
            "brsr_core.py": _sha256(Path(__file__).with_name("brsr_core.py")),
            "run_stage_a.py": _sha256(Path(__file__)),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(config: StageAConfig, output_dir: Path) -> dict[str, object]:
    _validate_config(config)
    model = build_default_model(tau_step=config.tau_step)
    quantizer = NestedComplexQuantizer(
        max_bits=config.max_bits, clip_abs=config.quantizer_clip_abs
    )
    print("BRSR-TDOA Stage A", flush=True)
    print(f"Mode: {config.mode}", flush=True)
    print(
        f"N={model.n_time}, delay hypotheses={model.n_tau}, "
        f"candidate bins={model.bin_indices.tolist()}",
        flush=True,
    )
    print(
        f"Quantizer: coarse={config.coarse_bits} bit, max={config.max_bits} bit, "
        f"fixed normalized range=+/-{config.quantizer_clip_abs:g}, "
        f"public B variance normalization="
        f"{config.use_public_b_variance_normalization}",
        flush=True,
    )
    print(
        f"Policies: lambda={list(config.lambda_per_bit)}, "
        f"max_decisions={config.max_decisions}, DP horizon={config.dp_horizon}",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)

    rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    _run_stage(
        stage="development",
        seeds=config.development_seeds,
        n_trials=config.development_trials,
        config=config,
        model=model,
        quantizer=quantizer,
        rows=rows,
        action_rows=action_rows,
        decision_rows=decision_rows,
    )
    _run_stage(
        stage="locked",
        seeds=config.locked_seeds,
        n_trials=config.locked_trials,
        config=config,
        model=model,
        quantizer=quantizer,
        rows=rows,
        action_rows=action_rows,
        decision_rows=decision_rows,
    )
    rows_frame = pd.DataFrame(rows)
    action_frame = pd.DataFrame(action_rows)
    decision_frame = pd.DataFrame(decision_rows)
    validation = _validate_results(
        rows_frame, action_frame, decision_frame, config
    )
    action_summary = _action_summary(action_frame)
    by_snr, overall = _summaries(rows_frame)
    gate = _stage_a_gate(
        config, by_snr, overall, action_summary, validation
    )
    _write_outputs(
        output_dir=output_dir,
        config=config,
        model=model,
        quantizer=quantizer,
        rows=rows_frame,
        action_rows=action_frame,
        decision_rows=decision_frame,
        validation=validation,
        action_summary=action_summary,
        gate=gate,
    )
    print("\nValidation", flush=True)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    print("\nAction calibration", flush=True)
    print(json.dumps(action_summary, ensure_ascii=False, indent=2), flush=True)
    print("\nStage A gate", flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    print("[Done] BRSR-TDOA Stage A complete.", flush=True)
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone finite-bit Bayesian-risk TDOA Stage A"
    )
    parser.add_argument("--mode", choices=tuple(CONFIGS), default=PYCHARM_RUN_MODE)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            ROOT
            / "运行结果"
            / "BRSR"
            / f"BRSR_STAGE_A_{timestamp}_{config.mode}"
        )
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee_out = Tee(sys.stdout, log_handle)
        tee_err = Tee(sys.stderr, log_handle)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            run(config, output_dir)


if __name__ == "__main__":
    main()
