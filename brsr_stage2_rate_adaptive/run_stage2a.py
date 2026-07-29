"""Run the independent BRSR Stage 2A equal-total-bit development experiment.

PyCharm runs the full development profile by default.  Codex uses ``--mode
smoke`` for the lightweight chain check.  This script never creates locked
results and does not modify Stage 1 artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brsr_route_feasibility.feasibility import (  # noqa: E402
    ACQUIRED_BIN_PAYLOAD_BITS,
    BASE_PAYLOAD_BITS,
    ONE_ACTION_FEEDBACK_BITS,
    base_levels,
    build_candidate_pools,
    one_action_positions,
)
from brsr_route_feasibility.run_stage1 import (  # noqa: E402
    Tee,
    _build_models,
    _generate_trials,
    _likelihood_observations,
    _sha256,
)
from brsr_stage2_rate_adaptive.stage2a import (  # noqa: E402
    PRIMARY_TARGET_BITS,
    TARGET_MEAN_BITS,
    Stage2AConfig,
    cluster_bootstrap_effect,
    cluster_bootstrap_mean,
    config_for_mode,
    evaluate_observation_actions,
    fit_temperature,
    freeze_rate_policy,
    rate_match_assignments,
    select_policy_action,
    tempered_state_metrics,
    wilson_interval,
    within_sample_score_diagnostics,
)
from brsr_tdoa.brsr_core import (  # noqa: E402
    BayesianSpectralModel,
    NestedComplexQuantizer,
    tau_posterior,
)
from brsr_tdoa.brsr_stage_b import (  # noqa: E402
    StageBObservation,
    state_from_levels,
)
from brsr_tdoa.brsr_waveform import WaveformConfig, observe_trial  # noqa: E402


PRIMARY_POOL = "Current-Fisher"
PRIMARY_LIKELIHOOD = "Linear-Marginal"
QUANTIZER_BITS = 8
QUANTIZER_CLIP_ABS = 4.0
STAGE1_CERTIFICATE = (
    ROOT
    / "运行结果"
    / "BRSR"
    / "BRSR_ROUTE_STAGE1_20260724_134327_development"
)
RESULT_COLUMNS = (
    "stage",
    "seed",
    "trial",
    "cluster_id",
    "snr_db",
    "true_tau",
    "method",
    "estimate_tau",
    "error_samples",
    "squared_error_samples2",
    "posterior_risk_samples2",
    "posterior_nll",
    "credible90_contains_true",
    "payload_bits",
    "feedback_bits",
    "total_bits",
    "selected_position",
    "selected_bin",
    "stopped",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "development"),
        default="development",
        help="PyCharm default is the full development profile.",
    )
    return parser.parse_args()


def _output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "运行结果" / "BRSR" / f"BRSR_STAGE2A_{timestamp}_{mode}"


def _json_dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cluster_id(seed: int, trial: int) -> str:
    return f"{seed}:{trial}"


def _levels_for_positions(
    model: BayesianSpectralModel,
    extra_positions: tuple[int, ...],
) -> tuple[int, ...]:
    levels = list(base_levels(model))
    for position in extra_positions:
        if position in model.base_positions:
            raise ValueError("static extra positions cannot include a base position")
        levels[position] = 4
    return tuple(levels)


def _collect_observations(
    *,
    stage: str,
    trials: dict[tuple[int, int], object],
    snr_db: tuple[float, ...],
    bins: np.ndarray,
    cases: dict[str, object],
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    progress_every: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    total = len(trials) * len(snr_db)
    done = 0
    started = time.perf_counter()
    for snr in snr_db:
        print(f"[{stage}] SNR={snr:g} dB start", flush=True)
        for (seed, trial_index), trial in trials.items():
            y_a, y_b, sigma2 = observe_trial(trial, snr)
            observations = _likelihood_observations(
                stage=stage,
                seed=seed,
                trial_index=trial_index,
                snr_db=snr,
                trial=trial,
                y_a=y_a,
                y_b=y_b,
                sigma2=sigma2,
                bins=bins,
                cases=cases,
                waveform=waveform,
                quantizer=quantizer,
            )
            observation, model, scale_check = observations[PRIMARY_LIKELIHOOD]
            records.append(
                {
                    "stage": stage,
                    "seed": seed,
                    "trial": trial_index,
                    "cluster_id": _cluster_id(seed, trial_index),
                    "snr_db": float(snr),
                    "observation": observation,
                    "model": model,
                    "scale_check": float(scale_check),
                }
            )
            done += 1
            if done % progress_every == 0 or done == total:
                elapsed = time.perf_counter() - started
                rate = elapsed / done
                eta = rate * (total - done)
                print(
                    f"[{stage}] {done}/{total} elapsed={elapsed:.1f}s "
                    f"avg={rate:.3f}s ETA={eta:.1f}s",
                    flush=True,
                )
    return records


def _fit_temperatures(
    records: list[dict[str, object]],
    *,
    quantizer: NestedComplexQuantizer,
) -> tuple[dict[float, float], pd.DataFrame]:
    rows = []
    temperature_by_snr: dict[float, float] = {}
    for snr, group in _group_records(records, "snr_db"):
        probabilities = []
        truths = []
        model = group[0]["model"]
        for record in group:
            observation = _as_observation(record["observation"])
            state = state_from_levels(
                observation,
                base_levels(model),
                quantizer=quantizer,
                model=model,
            )
            probabilities.append(tau_posterior(state, model))
            truths.append(observation.true_tau)
        fit = fit_temperature(
            snr_db=float(snr),
            tau_probabilities=probabilities,
            true_tau=truths,
            tau_values=model.tau_values,
        )
        temperature_by_snr[float(snr)] = fit.temperature
        rows.append(asdict(fit))
    return temperature_by_snr, pd.DataFrame(rows)


def _group_records(
    records: list[dict[str, object]],
    key: str,
) -> list[tuple[object, list[dict[str, object]]]]:
    values: dict[object, list[dict[str, object]]] = {}
    for record in records:
        values.setdefault(record[key], []).append(record)
    return sorted(values.items(), key=lambda item: item[0])


def _as_observation(value: object) -> StageBObservation:
    if not isinstance(value, StageBObservation):
        raise TypeError("record does not contain a StageBObservation")
    return value


def _as_model(value: object) -> BayesianSpectralModel:
    if not isinstance(value, BayesianSpectralModel):
        raise TypeError("record does not contain a BayesianSpectralModel")
    return value


def _state_result(
    record: dict[str, object],
    *,
    levels: tuple[int, ...],
    temperature: float,
    quantizer: NestedComplexQuantizer,
) -> dict[str, object]:
    observation = _as_observation(record["observation"])
    model = _as_model(record["model"])
    state = state_from_levels(
        observation,
        levels,
        quantizer=quantizer,
        model=model,
    )
    metrics = tempered_state_metrics(
        state,
        model=model,
        true_tau=observation.true_tau,
        temperature=temperature,
    )
    estimate = float(metrics["estimate_tau"])
    error = estimate - observation.true_tau
    payload_bits = int(2 * sum(levels))
    selected = [
        position
        for position, bits in enumerate(levels)
        if bits > 0 and position not in model.base_positions
    ]
    return {
        "estimate_tau": estimate,
        "error_samples": error,
        "squared_error_samples2": error**2,
        "posterior_risk_samples2": float(metrics["posterior_risk_samples2"]),
        "posterior_nll": float(metrics["posterior_nll"]),
        "credible90_contains_true": bool(
            metrics["credible90_contains_true"]
        ),
        "payload_bits": payload_bits,
        "feedback_bits": 0,
        "total_bits": payload_bits,
        "selected_position": " ".join(str(value) for value in selected),
        "selected_bin": " ".join(
            str(int(model.bin_indices[value])) for value in selected
        ),
        "stopped": False,
    }


def _result_metadata(record: dict[str, object], method: str) -> dict[str, object]:
    observation = _as_observation(record["observation"])
    return {
        "stage": record["stage"],
        "seed": int(record["seed"]),
        "trial": int(record["trial"]),
        "cluster_id": str(record["cluster_id"]),
        "snr_db": float(record["snr_db"]),
        "true_tau": observation.true_tau,
        "method": method,
    }


def _static_configurations(
    model: BayesianSpectralModel,
) -> dict[int, list[tuple[int, ...]]]:
    positions = one_action_positions(model)
    return {
        16: [tuple()],
        24: [(position,) for position in positions],
        32: list(combinations(positions, 2)),
    }


def _evaluate_plan_calibration(
    records: list[dict[str, object]],
    *,
    temperature_by_snr: dict[float, float],
    quantizer: NestedComplexQuantizer,
    progress_every: int,
    partial_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model = _as_model(records[0]["model"])
    configurations = _static_configurations(model)
    started = time.perf_counter()
    total = len(records)
    for index, record in enumerate(records, start=1):
        temperature = temperature_by_snr[float(record["snr_db"])]
        for payload_bits, extras in configurations.items():
            for positions in extras:
                result = _state_result(
                    record,
                    levels=_levels_for_positions(model, positions),
                    temperature=temperature,
                    quantizer=quantizer,
                )
                rows.append(
                    {
                        **_result_metadata(record, "PlanCandidate"),
                        "payload_target_bits": payload_bits,
                        "positions": " ".join(str(value) for value in positions),
                        "bins": " ".join(
                            str(int(model.bin_indices[value]))
                            for value in positions
                        ),
                        **result,
                    }
                )
        if index % progress_every == 0 or index == total:
            elapsed = time.perf_counter() - started
            rate = elapsed / index
            print(
                f"[plan_calibration:evaluate] {index}/{total} "
                f"elapsed={elapsed:.1f}s avg={rate:.3f}s "
                f"ETA={rate * (total - index):.1f}s",
                flush=True,
            )
        next_snr = (
            None if index == total else float(records[index]["snr_db"])
        )
        if next_snr != float(record["snr_db"]):
            pd.DataFrame(rows).to_csv(partial_path, index=False)
    return pd.DataFrame(rows)


def _freeze_static_plans(plan_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload_bits in (16, 24, 32):
        subset = plan_rows[plan_rows["payload_target_bits"].eq(payload_bits)]
        grouped = (
            subset.groupby(["positions", "bins"], as_index=False)
            ["squared_error_samples2"]
            .mean()
            .sort_values(
                ["squared_error_samples2", "positions"],
                kind="stable",
            )
        )
        selected = grouped.iloc[0]
        rows.append(
            {
                "family": "BestStatic",
                "snr_db": math.nan,
                "payload_bits": payload_bits,
                "positions": selected["positions"],
                "bins": selected["bins"],
                "plan_calibration_mse": float(
                    selected["squared_error_samples2"]
                ),
            }
        )
        for snr, snr_subset in subset.groupby("snr_db", sort=True):
            snr_grouped = (
                snr_subset.groupby(["positions", "bins"], as_index=False)
                ["squared_error_samples2"]
                .mean()
                .sort_values(
                    ["squared_error_samples2", "positions"],
                    kind="stable",
                )
            )
            selected = snr_grouped.iloc[0]
            rows.append(
                {
                    "family": "SNR-only",
                    "snr_db": float(snr),
                    "payload_bits": payload_bits,
                    "positions": selected["positions"],
                    "bins": selected["bins"],
                    "plan_calibration_mse": float(
                        selected["squared_error_samples2"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _parse_positions(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return tuple()
    return tuple(int(item) for item in text.split())


def _static_plan_for(
    plans: pd.DataFrame,
    *,
    family: str,
    snr_db: float,
    payload_bits: int,
) -> tuple[int, ...]:
    subset = plans[
        plans["family"].eq(family)
        & plans["payload_bits"].eq(payload_bits)
    ]
    if family == "SNR-only":
        subset = subset[subset["snr_db"].eq(float(snr_db))]
    else:
        subset = subset[subset["snr_db"].isna()]
    if len(subset) != 1:
        raise RuntimeError("frozen static plan is missing or duplicated")
    return _parse_positions(subset.iloc[0]["positions"])


def _evaluate_static_development(
    records: list[dict[str, object]],
    *,
    plans: pd.DataFrame,
    temperature_by_snr: dict[float, float],
    quantizer: NestedComplexQuantizer,
) -> pd.DataFrame:
    rows = []
    for record in records:
        model = _as_model(record["model"])
        snr = float(record["snr_db"])
        temperature = temperature_by_snr[snr]
        for family in ("BestStatic", "SNR-only"):
            for payload_bits in (16, 24, 32):
                positions = _static_plan_for(
                    plans,
                    family=family,
                    snr_db=snr,
                    payload_bits=payload_bits,
                )
                result = _state_result(
                    record,
                    levels=_levels_for_positions(model, positions),
                    temperature=temperature,
                    quantizer=quantizer,
                )
                rows.append(
                    {
                        **_result_metadata(
                            record,
                            f"{family}-B{payload_bits}",
                        ),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _evaluate_action_records(
    records: list[dict[str, object]],
    *,
    temperature_by_snr: dict[float, float],
    quantizer: NestedComplexQuantizer,
    stage_label: str,
    progress_every: int,
    partial_base_path: Path,
    partial_action_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    total = len(records)
    for index, record in enumerate(records, start=1):
        observation = _as_observation(record["observation"])
        model = _as_model(record["model"])
        base, actions = evaluate_observation_actions(
            observation,
            quantizer=quantizer,
            model=model,
            temperature=temperature_by_snr[float(record["snr_db"])],
        )
        metadata = _result_metadata(record, "ActionDiagnostic")
        base_rows.append({**metadata, **base})
        raw_base = float(base["base_raw_squared_error_samples2"])
        calibrated_base = float(
            base["base_calibrated_squared_error_samples2"]
        )
        for action in actions:
            action_rows.append(
                {
                    **metadata,
                    **action,
                    "raw_realized_gain_samples2": (
                        raw_base
                        - float(action["raw_squared_error_samples2"])
                    ),
                    "calibrated_realized_gain_samples2": (
                        calibrated_base
                        - float(action["calibrated_squared_error_samples2"])
                    ),
                    "oracle_realized_gain_samples2": (
                        calibrated_base
                        - float(action["calibrated_squared_error_samples2"])
                    ),
                }
            )
        if index % progress_every == 0 or index == total:
            elapsed = time.perf_counter() - started
            rate = elapsed / index
            print(
                f"[{stage_label}:actions] {index}/{total} "
                f"elapsed={elapsed:.1f}s avg={rate:.3f}s "
                f"ETA={rate * (total - index):.1f}s",
                flush=True,
            )
        next_snr = (
            None if index == total else float(records[index]["snr_db"])
        )
        if next_snr != float(record["snr_db"]):
            pd.DataFrame(base_rows).to_csv(partial_base_path, index=False)
            pd.DataFrame(action_rows).to_csv(
                partial_action_path,
                index=False,
            )
    return pd.DataFrame(base_rows), pd.DataFrame(action_rows)


def _policy_calibration_summary(
    base_rows: pd.DataFrame,
    action_rows: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["seed", "trial", "cluster_id", "snr_db"]
    rows = []
    for key, group in action_rows.groupby(keys, sort=False):
        metadata = dict(zip(keys, key, strict=True))
        base = base_rows
        for name, value in metadata.items():
            base = base[base[name].eq(value)]
        if len(base) != 1:
            raise RuntimeError("policy calibration base row is misaligned")
        best_raw = group.sort_values(
            ["raw_predicted_voi_samples2", "position"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        best_calibrated = group.sort_values(
            ["calibrated_predicted_voi_samples2", "position"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        best_oracle = group.sort_values(
            ["oracle_realized_gain_samples2", "position"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        rows.append(
            {
                **metadata,
                "raw_best_predicted_voi_samples2": float(
                    best_raw["raw_predicted_voi_samples2"]
                ),
                "raw_best_position": int(best_raw["position"]),
                "calibrated_best_predicted_voi_samples2": float(
                    best_calibrated[
                        "calibrated_predicted_voi_samples2"
                    ]
                ),
                "calibrated_best_position": int(
                    best_calibrated["position"]
                ),
                "oracle_best_realized_gain_samples2": float(
                    best_oracle["oracle_realized_gain_samples2"]
                ),
                "oracle_best_position": int(best_oracle["position"]),
            }
        )
    return pd.DataFrame(rows)


def _freeze_rate_policies(
    summary: pd.DataFrame,
) -> tuple[dict[tuple[float, str, int], object], pd.DataFrame]:
    policies: dict[tuple[float, str, int], object] = {}
    rows: list[dict[str, object]] = []
    for snr, subset in summary.groupby("snr_db", sort=True):
        source = subset.to_dict("records")
        for mode in ("raw", "calibrated", "oracle"):
            for target in TARGET_MEAN_BITS:
                policy = freeze_rate_policy(
                    source,
                    snr_db=float(snr),
                    temperature_mode=mode,
                    target_mean_bits=target,
                )
                policies[(float(snr), mode, target)] = policy
                rows.append(asdict(policy))
    return policies, pd.DataFrame(rows)


def _selected_action_result(
    *,
    record: dict[str, object],
    base_row: pd.Series,
    action_group: pd.DataFrame,
    mode: str,
    selected_position: int | None,
    method: str,
) -> dict[str, object]:
    model = _as_model(record["model"])
    if selected_position is None:
        prefix = "raw" if mode == "raw" else "calibrated"
        estimate = float(base_row[f"base_{prefix}_estimate_tau"])
        risk = float(base_row[f"base_{prefix}_posterior_risk_samples2"])
        squared_error = float(
            base_row[f"base_{prefix}_squared_error_samples2"]
        )
        nll = (
            math.nan
            if mode == "raw"
            else float(base_row["base_calibrated_posterior_nll"])
        )
        coverage = (
            False
            if mode == "raw"
            else bool(base_row["base_calibrated_coverage90"])
        )
        selected_bin = ""
        payload_bits = BASE_PAYLOAD_BITS
        stopped = True
    else:
        selected = action_group[
            action_group["position"].eq(selected_position)
        ]
        if len(selected) != 1:
            raise RuntimeError("selected action is missing or duplicated")
        selected = selected.iloc[0]
        prefix = "raw" if mode == "raw" else "calibrated"
        estimate = float(selected[f"{prefix}_estimate_tau"])
        risk = float(selected[f"{prefix}_posterior_risk_samples2"])
        squared_error = float(
            selected[f"{prefix}_squared_error_samples2"]
        )
        nll = float(selected[f"{prefix}_posterior_nll"])
        coverage = bool(selected[f"{prefix}_coverage90"])
        selected_bin = str(int(model.bin_indices[selected_position]))
        payload_bits = BASE_PAYLOAD_BITS + ACQUIRED_BIN_PAYLOAD_BITS
        stopped = False
    observation = _as_observation(record["observation"])
    error = estimate - observation.true_tau
    if not math.isclose(error**2, squared_error, abs_tol=1e-10):
        raise RuntimeError("selected action squared error is inconsistent")
    return {
        **_result_metadata(record, method),
        "estimate_tau": estimate,
        "error_samples": error,
        "squared_error_samples2": squared_error,
        "posterior_risk_samples2": risk,
        "posterior_nll": nll,
        "credible90_contains_true": coverage,
        "payload_bits": payload_bits,
        "feedback_bits": ONE_ACTION_FEEDBACK_BITS,
        "total_bits": payload_bits + ONE_ACTION_FEEDBACK_BITS,
        "selected_position": (
            "" if selected_position is None else str(selected_position)
        ),
        "selected_bin": selected_bin,
        "stopped": stopped,
    }


def _adaptive_development_results(
    records: list[dict[str, object]],
    *,
    base_rows: pd.DataFrame,
    action_rows: pd.DataFrame,
    policies: dict[tuple[float, str, int], object],
) -> pd.DataFrame:
    rows = []
    keys = ["seed", "trial", "cluster_id", "snr_db"]
    records_by_key = {
        (
            int(record["seed"]),
            int(record["trial"]),
            str(record["cluster_id"]),
            float(record["snr_db"]),
        ): record
        for record in records
    }
    for key, group in action_rows.groupby(keys, sort=False):
        record = records_by_key[key]
        base = base_rows
        for name, value in zip(keys, key, strict=True):
            base = base[base[name].eq(value)]
        if len(base) != 1:
            raise RuntimeError("development base row is misaligned")
        base_row = base.iloc[0]
        snr = float(key[-1])
        for mode, label in (
            ("raw", "BRSR-Raw"),
            ("calibrated", "BRSR-Calibrated"),
            ("oracle", "Oracle"),
        ):
            output_mode = "raw" if mode == "raw" else "calibrated"
            for target in TARGET_MEAN_BITS:
                policy = policies[(snr, mode, target)]
                selected = select_policy_action(
                    group.to_dict("records"),
                    temperature_mode=mode,
                    policy=policy,
                )
                rows.append(
                    _selected_action_result(
                        record=record,
                        base_row=base_row,
                        action_group=group,
                        mode=output_mode,
                        selected_position=selected,
                        method=f"{label}-B{target}",
                    )
                )
        risk_greedy = group.sort_values(
            ["raw_predicted_voi_samples2", "position"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        rows.append(
            _selected_action_result(
                record=record,
                base_row=base_row,
                action_group=group,
                mode="raw",
                selected_position=int(risk_greedy["position"]),
                method="RiskGreedy-B29",
            )
        )
    return pd.DataFrame(rows)


def _rate_matched_rows(
    *,
    adaptive_rows: pd.DataFrame,
    static_rows: pd.DataFrame,
    adaptive_method: str,
    family: str,
    label: str,
) -> pd.DataFrame:
    rows = []
    adaptive = adaptive_rows[adaptive_rows["method"].eq(adaptive_method)]
    for snr, target in adaptive.groupby("snr_db", sort=True):
        desired = float(target["total_bits"].mean())
        if desired <= 24.0:
            low_bits, high_bits = 16, 24
        else:
            low_bits, high_bits = 24, 32
        assignments = rate_match_assignments(
            target["cluster_id"],
            desired_mean_bits=desired,
            low_bits=low_bits,
            high_bits=high_bits,
            salt=f"stage2a|{label}|{snr:g}",
        )
        family_rows = static_rows[
            static_rows["method"].str.startswith(f"{family}-")
            & static_rows["snr_db"].eq(float(snr))
        ]
        for _, target_row in target.iterrows():
            cluster_id = str(target_row["cluster_id"])
            bits = assignments[cluster_id]
            candidate = family_rows[
                family_rows["cluster_id"].eq(cluster_id)
                & family_rows["method"].eq(f"{family}-B{bits}")
            ]
            if len(candidate) != 1:
                raise RuntimeError("rate-matched static row is misaligned")
            value = candidate.iloc[0].to_dict()
            value["method"] = label
            rows.append(value)
    return pd.DataFrame(rows)


def _paired_effect(
    results: pd.DataFrame,
    *,
    target_method: str,
    baseline_method: str,
    repetitions: int,
    seed: int,
    snr_db: float | None = None,
) -> dict[str, object]:
    keys = ["seed", "trial", "cluster_id", "snr_db"]
    left = results[results["method"].eq(target_method)]
    right = results[results["method"].eq(baseline_method)]
    if snr_db is not None:
        left = left[left["snr_db"].eq(float(snr_db))]
        right = right[right["snr_db"].eq(float(snr_db))]
    paired = left[
        keys + ["squared_error_samples2", "total_bits"]
    ].rename(
        columns={
            "squared_error_samples2": "target_sq",
            "total_bits": "target_bits",
        }
    ).merge(
        right[
            keys + ["squared_error_samples2", "total_bits"]
        ].rename(
            columns={
                "squared_error_samples2": "baseline_sq",
                "total_bits": "baseline_bits",
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("effect comparison is not fully paired")
    summary = cluster_bootstrap_effect(
        paired["target_sq"].to_numpy(),
        paired["baseline_sq"].to_numpy(),
        paired["cluster_id"].to_numpy(),
        repetitions=repetitions,
        seed=seed,
    )
    return {
        "target_method": target_method,
        "baseline_method": baseline_method,
        "snr_db": math.nan if snr_db is None else float(snr_db),
        "target_mean_total_bits": float(paired["target_bits"].mean()),
        "baseline_mean_total_bits": float(paired["baseline_bits"].mean()),
        "mean_total_bits_difference": float(
            paired["target_bits"].mean()
            - paired["baseline_bits"].mean()
        ),
        **summary,
    }


def _score_diagnostics(
    action_rows: pd.DataFrame,
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["seed", "trial", "cluster_id", "snr_db"]
    rows = []
    for key, group in action_rows.groupby(keys, sort=False):
        metadata = dict(zip(keys, key, strict=True))
        for mode in ("raw", "calibrated"):
            rows.append(
                {
                    **metadata,
                    "temperature_mode": mode,
                    **within_sample_score_diagnostics(
                        group.to_dict("records"),
                        temperature_mode=mode,
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    summaries = []
    for mode, subset in frame.groupby("temperature_mode", sort=True):
        summary = cluster_bootstrap_mean(
            subset["spearman_predicted_vs_actual_gain"].fillna(0.0).to_numpy(),
            subset["cluster_id"].to_numpy(),
            repetitions=repetitions,
            seed=2026072591 if mode == "raw" else 2026072592,
        )
        summaries.append({"temperature_mode": mode, **summary})
    return frame, pd.DataFrame(summaries)


def _calibration_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    calibrated = results[
        results["method"].str.startswith("BRSR-Calibrated-")
    ]
    for (method, snr), group in calibrated.groupby(
        ["method", "snr_db"],
        sort=True,
    ):
        successes = int(group["credible90_contains_true"].sum())
        low, high = wilson_interval(successes, len(group))
        empirical_mse = float(group["squared_error_samples2"].mean())
        posterior_risk = float(group["posterior_risk_samples2"].mean())
        rows.append(
            {
                "method": method,
                "snr_db": float(snr),
                "n_rows": len(group),
                "coverage90": successes / len(group),
                "coverage90_wilson_low": low,
                "coverage90_wilson_high": high,
                "posterior_risk_mean": posterior_risk,
                "empirical_mse": empirical_mse,
                "risk_to_mse_ratio": (
                    posterior_risk / empirical_mse
                    if empirical_mse > 0.0
                    else math.nan
                ),
                "posterior_nll_mean": float(group["posterior_nll"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _method_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, snr), group in results.groupby(
        ["method", "snr_db"],
        sort=True,
    ):
        rows.append(
            {
                "method": method,
                "snr_db": float(snr),
                "n_rows": len(group),
                "rmse_samples": math.sqrt(
                    float(group["squared_error_samples2"].mean())
                ),
                "mae_samples": float(group["error_samples"].abs().mean()),
                "median_abs_error_samples": float(
                    group["error_samples"].abs().median()
                ),
                "mean_total_bits": float(group["total_bits"].mean()),
                "stop_fraction": float(group["stopped"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _threshold_sensitivity(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, effect in effects.iterrows():
        gain = float(effect["relative_rmse_gain"])
        for threshold in (0.0, 0.02, 0.05, 0.10):
            rows.append(
                {
                    "target_method": effect["target_method"],
                    "baseline_method": effect["baseline_method"],
                    "minimum_relative_rmse_gain": threshold,
                    "observed_relative_rmse_gain": gain,
                    "point_estimate_pass": bool(gain >= threshold),
                    "statistical_mse_ci_pass": bool(
                        float(effect["mse_gain_ci_low"]) > 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _stage1_certificate() -> dict[str, object]:
    gate_path = STAGE1_CERTIFICATE / "gate_decision.json"
    manifest_path = STAGE1_CERTIFICATE / "manifest.json"
    if not gate_path.exists() or not manifest_path.exists():
        return {
            "available": False,
            "valid": False,
            "path": str(STAGE1_CERTIFICATE),
        }
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hashes = manifest.get("source_files", {})
    current_hashes = {
        relative: _sha256(ROOT / relative)
        for relative in source_hashes
        if (ROOT / relative).exists()
    }
    sources_match = (
        len(current_hashes) == len(source_hashes)
        and current_hashes == source_hashes
    )
    valid = (
        gate.get("status") == "STAGE1_PASS"
        and gate.get("primary_likelihood") == PRIMARY_LIKELIHOOD
        and gate.get("score_pool_selected_on_calibration") == PRIMARY_POOL
        and gate.get("quadrature_main_grid") == [5, 64]
        and gate.get("quadrature_reference_grid") == [9, 128]
        and sources_match
    )
    return {
        "available": True,
        "valid": bool(valid),
        "path": str(STAGE1_CERTIFICATE),
        "gate_sha256": _sha256(gate_path),
        "manifest_sha256": _sha256(manifest_path),
        "frozen_source_hashes_match": sources_match,
    }


def _gate_decision(
    *,
    mode: str,
    config: Stage2AConfig,
    temperature_rows: pd.DataFrame,
    policies: pd.DataFrame,
    results: pd.DataFrame,
    effects: pd.DataFrame,
    score_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    stage1_certificate: dict[str, object],
) -> dict[str, object]:
    allowed_bits = {21, 29}
    primary = results[
        results["method"].eq(
            f"BRSR-Calibrated-B{PRIMARY_TARGET_BITS}"
        )
    ]
    calibration_policy = policies[
        policies["temperature_mode"].eq("calibrated")
        & policies["target_mean_bits"].eq(PRIMARY_TARGET_BITS)
    ]
    all_seed_groups = (
        config.prior_seeds,
        config.plan_seeds,
        config.policy_calibration_seeds,
        config.development_seeds,
    )
    flattened_seeds = [
        seed for group in all_seed_groups for seed in group
    ]
    calibration_rows_per_snr = (
        len(config.policy_calibration_seeds)
        * config.policy_calibration_trials_per_seed
    )
    target_rate_ok = bool(
        np.allclose(
            calibration_policy["calibration_mean_total_bits"],
            PRIMARY_TARGET_BITS,
            atol=8.0 / calibration_rows_per_snr + 1e-12,
            rtol=0.0,
        )
    )
    primary_rate_stable = True
    for _, group in primary.groupby("snr_db", sort=True):
        acquired = int((group["total_bits"] == 29).sum())
        low, high = wilson_interval(acquired, len(group))
        primary_rate_stable &= low <= 3.0 / 8.0 <= high
    rate_match_ok = True
    for baseline in (
        f"BestStatic-RateMatched-B{PRIMARY_TARGET_BITS}",
        f"SNR-only-RateMatched-B{PRIMARY_TARGET_BITS}",
    ):
        baseline_rows = results[results["method"].eq(baseline)]
        for snr, group in primary.groupby("snr_db", sort=True):
            compared = baseline_rows[baseline_rows["snr_db"].eq(snr)]
            n_clusters = int(group["cluster_id"].nunique())
            allowed_rounding = 16.0 / n_clusters + 1e-12
            rate_match_ok &= (
                abs(
                    float(group["total_bits"].mean())
                    - float(compared["total_bits"].mean())
                )
                <= allowed_rounding
            )
    mechanical = {
        "stage1_certificate_valid": bool(stage1_certificate["valid"]),
        "seed_pools_disjoint": (
            len(flattened_seeds) == len(set(flattened_seeds))
        ),
        "quadrature_is_5x64": (
            config.marginal_amplitude_nodes,
            config.marginal_phase_nodes,
        )
        == (5, 64),
        "only_stop_or_one_acquire_bits": set(
            results[
                results["method"].str.startswith(
                    ("BRSR-", "Oracle-", "RiskGreedy-")
                )
            ]["total_bits"].unique()
        ).issubset(allowed_bits),
        "primary_policy_calibration_rate_matches_target": target_rate_ok,
        "primary_development_rate_consistent_with_target": (
            primary_rate_stable
        ),
        "rate_matched_static_rounding_within_one_cluster": rate_match_ok,
        "all_results_finite": bool(
            np.all(
                np.isfinite(
                    results[
                        [
                            "estimate_tau",
                            "error_samples",
                            "squared_error_samples2",
                            "total_bits",
                        ]
                    ].to_numpy(dtype=float)
                )
            )
        ),
        "primary_development_rows_complete": len(primary)
        == (
            len(config.development_seeds)
            * config.development_trials_per_seed
            * len(config.snr_db)
        ),
        "temperature_fit_does_not_worsen_calibration_nll": bool(
            np.all(
                temperature_rows["nll_calibrated"]
                <= temperature_rows["nll_raw"] + 1e-10
            )
        ),
    }
    primary_effects = effects[
        effects["target_method"].eq(
            f"BRSR-Calibrated-B{PRIMARY_TARGET_BITS}"
        )
    ]
    oracle_effects = effects[
        effects["target_method"].eq(f"Oracle-B{PRIMARY_TARGET_BITS}")
    ]
    score = score_summary[
        score_summary["temperature_mode"].eq("calibrated")
    ]
    statistical = {
        "primary_vs_both_static_mse_ci_positive": bool(
            len(primary_effects) == 2
            and np.all(primary_effects["mse_gain_ci_low"] > 0.0)
        ),
        "oracle_vs_both_static_mse_ci_positive": bool(
            len(oracle_effects) == 2
            and np.all(oracle_effects["mse_gain_ci_low"] > 0.0)
        ),
        "calibrated_score_spearman_ci_positive": bool(
            len(score) == 1 and float(score.iloc[0]["ci_low"]) > 0.0
        ),
        "coverage_wilson_contains_nominal_by_snr": bool(
            np.all(
                (calibration["coverage90_wilson_low"] <= 0.90)
                & (calibration["coverage90_wilson_high"] >= 0.90)
            )
        ),
        "temperature_not_on_search_boundary": bool(
            not temperature_rows[
                ["hit_lower_bound", "hit_upper_bound"]
            ].to_numpy(dtype=bool).any()
        ),
    }
    if mode == "smoke":
        status = "SMOKE_ONLY"
    elif not all(mechanical.values()):
        status = "HOLD-MECHANICAL"
    elif not statistical["oracle_vs_both_static_mse_ci_positive"]:
        status = "NO-GO-COMPONENT-RATE-SPACE"
    elif all(statistical.values()):
        status = "STAGE2A_PASS"
    else:
        status = "HOLD-STAGE2A"
    return {
        "status": status,
        "mode": mode,
        "primary_target_mean_total_bits": PRIMARY_TARGET_BITS,
        "mechanical_validations": mechanical,
        "statistical_validations": statistical,
        "stage1_certificate": stage1_certificate,
        "interpretation_boundary": (
            "Stage 2A evaluates one stop/acquire decision at equal observed "
            "average total bits. Oracle uses true TDOA and is diagnostic only. "
            "A pass permits Stage 2B design, not locked confirmation."
        ),
    }


def _plot_results(
    *,
    output: Path,
    summary: pd.DataFrame,
    temperature: pd.DataFrame,
    policies: pd.DataFrame,
    score_rows: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.5,
        }
    )
    methods = (
        f"BRSR-Calibrated-B{PRIMARY_TARGET_BITS}",
        f"BRSR-Raw-B{PRIMARY_TARGET_BITS}",
        f"BestStatic-RateMatched-B{PRIMARY_TARGET_BITS}",
        f"SNR-only-RateMatched-B{PRIMARY_TARGET_BITS}",
        f"Oracle-B{PRIMARY_TARGET_BITS}",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method in methods:
        subset = summary[summary["method"].eq(method)]
        if subset.empty:
            continue
        ax.plot(
            subset["snr_db"],
            subset["rmse_samples"],
            marker="o",
            label=method,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("TDOA RMSE (samples)")
    ax.legend(loc="best", frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "Fig1_EqualRate_TDOA_RMSE.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(
        temperature["snr_db"],
        temperature["temperature"],
        marker="o",
        color="#1f77b4",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Posterior temperature")
    calibrated = policies[
        policies["temperature_mode"].eq("calibrated")
    ]
    for target, group in calibrated.groupby("target_mean_bits", sort=True):
        axes[1].plot(
            group["snr_db"],
            group["calibration_acquire_fraction"],
            marker="o",
            label=f"B{target}",
        )
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Acquire fraction")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / "Fig2_Calibration_And_RatePolicy.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for mode, group in score_rows.groupby("temperature_mode", sort=True):
        axes[0].scatter(
            group["snr_db"],
            group["spearman_predicted_vs_actual_gain"],
            s=15,
            alpha=0.55,
            label=mode,
        )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Within-trial Spearman")
    axes[0].legend(loc="best")
    labels = [
        f"{row.target_method}\nvs\n{row.baseline_method}"
        for row in effects.itertuples()
    ]
    axes[1].errorbar(
        np.arange(len(effects)),
        effects["mse_gain"],
        yerr=np.vstack(
            [
                effects["mse_gain"] - effects["mse_gain_ci_low"],
                effects["mse_gain_ci_high"] - effects["mse_gain"],
            ]
        ),
        fmt="o",
        capsize=3,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(np.arange(len(effects)), labels, rotation=25, ha="right")
    axes[1].set_ylabel("Paired MSE gain (samples squared)")
    fig.tight_layout()
    fig.savefig(output / "Fig3_Score_And_PairedEffects.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    selected = summary[
        summary["method"].isin(
            [
                f"BRSR-Calibrated-B{value}"
                for value in TARGET_MEAN_BITS
            ]
            + ["BestStatic-B16", "BestStatic-B24", "BestStatic-B32"]
        )
    ]
    aggregate = (
        selected.groupby("method", as_index=False)
        .agg(
            rmse_samples=("squared_error_samples2", "mean")
            if "squared_error_samples2" in selected.columns
            else ("rmse_samples", "mean"),
            mean_total_bits=("mean_total_bits", "mean"),
        )
    )
    if "squared_error_samples2" in selected.columns:
        aggregate["rmse_samples"] = np.sqrt(aggregate["rmse_samples"])
    for _, row in aggregate.iterrows():
        ax.scatter(row["mean_total_bits"], row["rmse_samples"], s=45)
        ax.annotate(
            row["method"],
            (row["mean_total_bits"], row["rmse_samples"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean total communication (bit)")
    ax.set_ylabel("Mean TDOA RMSE across SNR (samples)")
    fig.tight_layout()
    fig.savefig(output / "Fig4_Rate_Accuracy_Pareto.svg")
    plt.close(fig)


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        ROOT / "brsr_stage2_rate_adaptive" / "stage2a.py",
        ROOT / "brsr_route_feasibility" / "feasibility.py",
        ROOT / "brsr_route_feasibility" / "run_stage1.py",
        ROOT / "brsr_tdoa" / "brsr_core.py",
        ROOT / "brsr_tdoa" / "brsr_stage_b.py",
        ROOT / "brsr_tdoa" / "brsr_waveform.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _run(config: Stage2AConfig, output: Path) -> None:
    waveform = WaveformConfig()
    waveform.validate()
    quantizer = NestedComplexQuantizer(
        max_bits=QUANTIZER_BITS,
        clip_abs=QUANTIZER_CLIP_ABS,
    )
    all_pools = build_candidate_pools(waveform)
    pools = {PRIMARY_POOL: all_pools[PRIMARY_POOL]}
    prior_trials = _generate_trials(
        config.prior_seeds,
        config.prior_trials_per_seed,
        waveform,
    )

    class _ModelConfig:
        tau_step = config.tau_step
        marginal_amplitude_nodes = config.marginal_amplitude_nodes
        marginal_phase_nodes = config.marginal_phase_nodes

    models, prior_statistics = _build_models(
        pools,
        prior_trials,
        waveform=waveform,
        config=_ModelConfig(),
    )
    cases = models[PRIMARY_POOL]
    bins = pools[PRIMARY_POOL]

    policy_trials = _generate_trials(
        config.policy_calibration_seeds,
        config.policy_calibration_trials_per_seed,
        waveform,
    )
    policy_records = _collect_observations(
        stage="policy_calibration",
        trials=policy_trials,
        snr_db=config.snr_db,
        bins=bins,
        cases=cases,
        waveform=waveform,
        quantizer=quantizer,
        progress_every=config.progress_every,
    )
    temperature_by_snr, temperature_rows = _fit_temperatures(
        policy_records,
        quantizer=quantizer,
    )
    policy_base, policy_actions = _evaluate_action_records(
        policy_records,
        temperature_by_snr=temperature_by_snr,
        quantizer=quantizer,
        stage_label="policy_calibration",
        progress_every=config.progress_every,
        partial_base_path=output / "partial_policy_base.csv",
        partial_action_path=output / "partial_policy_actions.csv",
    )
    policy_summary = _policy_calibration_summary(
        policy_base,
        policy_actions,
    )
    policies, policy_rows = _freeze_rate_policies(policy_summary)

    plan_trials = _generate_trials(
        config.plan_seeds,
        config.plan_trials_per_seed,
        waveform,
    )
    plan_records = _collect_observations(
        stage="plan_calibration",
        trials=plan_trials,
        snr_db=config.snr_db,
        bins=bins,
        cases=cases,
        waveform=waveform,
        quantizer=quantizer,
        progress_every=config.progress_every,
    )
    plan_rows = _evaluate_plan_calibration(
        plan_records,
        temperature_by_snr=temperature_by_snr,
        quantizer=quantizer,
        progress_every=config.progress_every,
        partial_path=output / "partial_plan_calibration_rows.csv",
    )
    frozen_plans = _freeze_static_plans(plan_rows)

    development_trials = _generate_trials(
        config.development_seeds,
        config.development_trials_per_seed,
        waveform,
    )
    development_records = _collect_observations(
        stage="development",
        trials=development_trials,
        snr_db=config.snr_db,
        bins=bins,
        cases=cases,
        waveform=waveform,
        quantizer=quantizer,
        progress_every=config.progress_every,
    )
    static_results = _evaluate_static_development(
        development_records,
        plans=frozen_plans,
        temperature_by_snr=temperature_by_snr,
        quantizer=quantizer,
    )
    development_base, development_actions = _evaluate_action_records(
        development_records,
        temperature_by_snr=temperature_by_snr,
        quantizer=quantizer,
        stage_label="development",
        progress_every=config.progress_every,
        partial_base_path=output / "partial_development_base.csv",
        partial_action_path=output / "partial_development_actions.csv",
    )
    adaptive_results = _adaptive_development_results(
        development_records,
        base_rows=development_base,
        action_rows=development_actions,
        policies=policies,
    )
    result_frames = [static_results, adaptive_results]
    for target_method, prefix in (
        (
            f"BRSR-Calibrated-B{PRIMARY_TARGET_BITS}",
            f"B{PRIMARY_TARGET_BITS}",
        ),
        (
            f"Oracle-B{PRIMARY_TARGET_BITS}",
            f"OracleB{PRIMARY_TARGET_BITS}",
        ),
    ):
        for family in ("BestStatic", "SNR-only"):
            result_frames.append(
                _rate_matched_rows(
                    adaptive_rows=adaptive_results,
                    static_rows=static_results,
                    adaptive_method=target_method,
                    family=family,
                    label=f"{family}-RateMatched-{prefix}",
                )
            )
    results = pd.concat(result_frames, ignore_index=True)
    results = results[list(RESULT_COLUMNS)]

    effect_rows = []
    comparisons = (
        (
            f"BRSR-Calibrated-B{PRIMARY_TARGET_BITS}",
            f"BestStatic-RateMatched-B{PRIMARY_TARGET_BITS}",
        ),
        (
            f"BRSR-Calibrated-B{PRIMARY_TARGET_BITS}",
            f"SNR-only-RateMatched-B{PRIMARY_TARGET_BITS}",
        ),
        (
            f"Oracle-B{PRIMARY_TARGET_BITS}",
            f"BestStatic-RateMatched-OracleB{PRIMARY_TARGET_BITS}",
        ),
        (
            f"Oracle-B{PRIMARY_TARGET_BITS}",
            f"SNR-only-RateMatched-OracleB{PRIMARY_TARGET_BITS}",
        ),
    )
    for index, (target, baseline) in enumerate(comparisons):
        effect_rows.append(
            _paired_effect(
                results,
                target_method=target,
                baseline_method=baseline,
                repetitions=config.bootstrap_repetitions,
                seed=2026072581 + index,
            )
        )
    effects = pd.DataFrame(effect_rows)
    per_snr_effect_rows = []
    for comparison_index, (target, baseline) in enumerate(comparisons):
        for snr_index, snr in enumerate(config.snr_db):
            per_snr_effect_rows.append(
                _paired_effect(
                    results,
                    target_method=target,
                    baseline_method=baseline,
                    repetitions=config.bootstrap_repetitions,
                    seed=2026072600
                    + 10 * comparison_index
                    + snr_index,
                    snr_db=float(snr),
                )
            )
    per_snr_effects = pd.DataFrame(per_snr_effect_rows)
    score_rows, score_summary = _score_diagnostics(
        development_actions,
        repetitions=config.bootstrap_repetitions,
    )
    calibration = _calibration_summary(results)
    method_summary = _method_summary(results)
    threshold_sensitivity = _threshold_sensitivity(effects)
    certificate = _stage1_certificate()
    gate = _gate_decision(
        mode=config.mode,
        config=config,
        temperature_rows=temperature_rows,
        policies=policy_rows,
        results=results,
        effects=effects,
        score_summary=score_summary,
        calibration=calibration[
            calibration["method"].eq(
                f"BRSR-Calibrated-B{PRIMARY_TARGET_BITS}"
            )
        ],
        stage1_certificate=certificate,
    )

    prior_statistics.to_csv(output / "prior_statistics.csv", index=False)
    temperature_rows.to_csv(
        output / "temperature_calibration.csv",
        index=False,
    )
    policy_summary.to_csv(
        output / "policy_calibration_summary.csv",
        index=False,
    )
    policy_rows.to_csv(output / "frozen_rate_policies.csv", index=False)
    plan_rows.to_csv(output / "plan_calibration_rows.csv", index=False)
    frozen_plans.to_csv(output / "frozen_static_plans.csv", index=False)
    development_actions.to_csv(
        output / "stage2a_action_rows.csv",
        index=False,
    )
    results.to_csv(output / "stage2a_trial_results.csv", index=False)
    method_summary.to_csv(output / "method_summary_by_snr.csv", index=False)
    effects.to_csv(output / "rate_matched_effects.csv", index=False)
    per_snr_effects.to_csv(
        output / "rate_matched_effects_by_snr.csv",
        index=False,
    )
    score_rows.to_csv(output / "action_score_diagnostics.csv", index=False)
    score_summary.to_csv(
        output / "action_score_summary.csv",
        index=False,
    )
    calibration.to_csv(output / "calibration_summary.csv", index=False)
    threshold_sensitivity.to_csv(
        output / "threshold_sensitivity.csv",
        index=False,
    )
    _json_dump(output / "gate_decision.json", gate)
    _json_dump(
        output / "manifest.json",
        {
            "experiment": "BRSR Stage 2A equal-total-bit development",
            "mode": config.mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(config),
            "frozen_protocol": {
                "pool": PRIMARY_POOL,
                "likelihood": PRIMARY_LIKELIHOOD,
                "quadrature": [5, 64],
                "reference_quadrature": [9, 128],
                "base_payload_bits": BASE_PAYLOAD_BITS,
                "feedback_bits": ONE_ACTION_FEEDBACK_BITS,
                "acquire_payload_bits": ACQUIRED_BIN_PAYLOAD_BITS,
                "target_mean_total_bits": list(TARGET_MEAN_BITS),
                "primary_target_mean_total_bits": PRIMARY_TARGET_BITS,
            },
            "stage1_certificate": certificate,
            "source_sha256": _source_hashes(),
        },
    )
    _plot_results(
        output=output,
        summary=method_summary,
        temperature=temperature_rows,
        policies=policy_rows,
        score_rows=score_rows,
        effects=effects,
    )
    for path in output.glob("partial_*.csv"):
        path.unlink()
    print("\nStage 2A gate", flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = _parse_args()
    config = config_for_mode(args.mode)
    output = _output_dir(args.mode)
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    original_stdout = sys.stdout
    log_handle = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_handle)
    started = time.perf_counter()
    try:
        print("BRSR Stage 2A equal-total-bit development", flush=True)
        print(f"Mode: {config.mode}", flush=True)
        print(f"Output: {output}", flush=True)
        print(
            "Frozen: Current-Fisher / Linear-Marginal / 5x64; "
            "feedback=5 bit; primary mean total=24 bit",
            flush=True,
        )
        print(
            "No locked data are generated. Development labels do not fit "
            "temperature, static plans, or rate policies.",
            flush=True,
        )
        _run(config, output)
        print(
            f"[Done] elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
