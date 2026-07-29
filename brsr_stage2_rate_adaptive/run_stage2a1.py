"""Run preregistered BRSR Stage 2A.1 safe action-gating development.

PyCharm runs the full development profile by default.  The script keeps the
formal Stage 2A signal model, posterior temperature and static plans frozen.
New gate-fit, gate-certificate and development trajectory pools are mutually
disjoint.  No locked result is generated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brsr_route_feasibility.feasibility import (  # noqa: E402
    ACQUIRED_BIN_PAYLOAD_BITS,
    BASE_PAYLOAD_BITS,
    ONE_ACTION_FEEDBACK_BITS,
    build_candidate_pools,
)
from brsr_route_feasibility.run_stage1 import (  # noqa: E402
    Tee,
    _build_models,
    _generate_trials,
    _sha256,
)
from brsr_stage2_rate_adaptive.run_stage2a import (  # noqa: E402
    PRIMARY_LIKELIHOOD,
    PRIMARY_POOL,
    QUANTIZER_BITS,
    QUANTIZER_CLIP_ABS,
    RESULT_COLUMNS,
    _collect_observations,
    _evaluate_action_records,
    _evaluate_static_development,
    _json_dump,
    _method_summary,
    _paired_effect,
    _rate_matched_rows,
    _selected_action_result,
)
from brsr_stage2_rate_adaptive.stage2a1 import (  # noqa: E402
    KEYS,
    SafeGatePolicy,
    apply_threshold,
    build_action_gate_samples,
    certify_threshold,
    fit_threshold,
    leave_one_cluster_min_mean,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer  # noqa: E402
from brsr_tdoa.brsr_waveform import WaveformConfig  # noqa: E402


SOURCE_STAGE2A = (
    ROOT
    / "运行结果"
    / "BRSR"
    / "BRSR_STAGE2A_20260724_153144_development"
)
TARGET_TOTAL_BITS = 29
FIXED_PAYLOAD_BITS = 24
BOOTSTRAP_ALPHA = 0.05


@dataclass(frozen=True)
class Stage2A1Config:
    mode: str
    snr_db: tuple[float, ...]
    gate_fit_seeds: tuple[int, ...]
    gate_certificate_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    gate_fit_trials_per_seed: int
    gate_certificate_trials_per_seed: int
    development_trials_per_seed: int
    bootstrap_repetitions: int
    progress_every: int

    def validate(self, source_seeds: set[int]) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        groups = (
            self.gate_fit_seeds,
            self.gate_certificate_seeds,
            self.development_seeds,
        )
        flattened = [seed for group in groups for seed in group]
        if any(not group for group in groups):
            raise ValueError("all Stage 2A.1 seed pools must be non-empty")
        if len(flattened) != len(set(flattened)):
            raise ValueError("Stage 2A.1 seed pools must be disjoint")
        if set(flattened) & source_seeds:
            raise ValueError("Stage 2A.1 seeds overlap frozen Stage 2A")
        if min(
            self.gate_fit_trials_per_seed,
            self.gate_certificate_trials_per_seed,
            self.development_trials_per_seed,
        ) < 1:
            raise ValueError("all Stage 2A.1 trial counts must be positive")
        if self.bootstrap_repetitions < 20 or self.progress_every < 1:
            raise ValueError("invalid bootstrap or progress setting")


CONFIGS = {
    "smoke": Stage2A1Config(
        mode="smoke",
        snr_db=(-5.0, 10.0),
        gate_fit_seeds=(2026072711,),
        gate_certificate_seeds=(2026072721,),
        development_seeds=(2026072731,),
        gate_fit_trials_per_seed=2,
        gate_certificate_trials_per_seed=2,
        development_trials_per_seed=2,
        bootstrap_repetitions=50,
        progress_every=1,
    ),
    "development": Stage2A1Config(
        mode="development",
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
        gate_fit_seeds=(2026072711, 2026072712),
        gate_certificate_seeds=(2026072721, 2026072722),
        development_seeds=(2026072731, 2026072732, 2026072733),
        gate_fit_trials_per_seed=40,
        gate_certificate_trials_per_seed=40,
        development_trials_per_seed=40,
        bootstrap_repetitions=2000,
        progress_every=10,
    ),
}


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
    return ROOT / "运行结果" / "BRSR" / f"BRSR_STAGE2A1_{timestamp}_{mode}"


def _load_source() -> tuple[
    dict[str, object],
    pd.DataFrame,
    dict[float, float],
    dict[str, str],
]:
    required = (
        "manifest.json",
        "gate_decision.json",
        "frozen_static_plans.csv",
        "temperature_calibration.csv",
    )
    for name in required:
        if not (SOURCE_STAGE2A / name).is_file():
            raise FileNotFoundError(f"frozen Stage 2A artifact missing: {name}")
    manifest = json.loads(
        (SOURCE_STAGE2A / "manifest.json").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (SOURCE_STAGE2A / "gate_decision.json").read_text(encoding="utf-8")
    )
    if manifest.get("mode") != "development" or gate.get("mode") != "development":
        raise RuntimeError("Stage 2A.1 requires a formal Stage 2A source")
    protocol = manifest.get("frozen_protocol", {})
    if (
        protocol.get("pool") != PRIMARY_POOL
        or protocol.get("likelihood") != PRIMARY_LIKELIHOOD
        or protocol.get("quadrature") != [5, 64]
    ):
        raise RuntimeError("frozen Stage 2A protocol is inconsistent")
    current_hashes = {}
    for relative, expected in manifest.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen source hash changed: {relative}")
        current_hashes[relative] = actual
    plans = pd.read_csv(SOURCE_STAGE2A / "frozen_static_plans.csv")
    plans["positions"] = plans["positions"].fillna("")
    plans["bins"] = plans["bins"].fillna("")
    temperature = pd.read_csv(SOURCE_STAGE2A / "temperature_calibration.csv")
    temperature_by_snr = {
        float(row["snr_db"]): float(row["temperature"])
        for _, row in temperature.iterrows()
    }
    return manifest, plans, temperature_by_snr, current_hashes


def _source_seed_set(manifest: dict[str, object]) -> set[int]:
    config = manifest.get("config", {})
    keys = (
        "prior_seeds",
        "plan_seeds",
        "policy_calibration_seeds",
        "development_seeds",
    )
    return {
        int(seed)
        for key in keys
        for seed in config.get(key, [])
    }


def _fixed_position(plans: pd.DataFrame) -> tuple[int, int]:
    row = plans[
        plans["family"].eq("BestStatic")
        & plans["payload_bits"].eq(FIXED_PAYLOAD_BITS)
        & plans["snr_db"].isna()
    ]
    if len(row) != 1:
        raise RuntimeError("frozen BestStatic-B24 plan is missing")
    positions = [int(value) for value in str(row.iloc[0]["positions"]).split()]
    bins = [int(value) for value in str(row.iloc[0]["bins"]).split()]
    if len(positions) != 1 or len(bins) != 1:
        raise RuntimeError("BestStatic-B24 must contain one extra position")
    return positions[0], bins[0]


def _build_frozen_model(
    manifest: dict[str, object],
    waveform: WaveformConfig,
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame]:
    source_config = manifest["config"]
    all_pools = build_candidate_pools(waveform)
    bins = all_pools[PRIMARY_POOL]
    prior_trials = _generate_trials(
        tuple(int(seed) for seed in source_config["prior_seeds"]),
        int(source_config["prior_trials_per_seed"]),
        waveform,
    )

    class _ModelConfig:
        tau_step = float(source_config["tau_step"])
        marginal_amplitude_nodes = int(
            source_config["marginal_amplitude_nodes"]
        )
        marginal_phase_nodes = int(source_config["marginal_phase_nodes"])

    models, prior_statistics = _build_models(
        {PRIMARY_POOL: bins},
        prior_trials,
        waveform=waveform,
        config=_ModelConfig(),
    )
    return models[PRIMARY_POOL], bins, prior_statistics


def _records_and_actions(
    *,
    stage: str,
    seeds: tuple[int, ...],
    trials_per_seed: int,
    config: Stage2A1Config,
    cases: dict[str, object],
    bins: np.ndarray,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    temperature_by_snr: dict[float, float],
    output: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    trials = _generate_trials(seeds, trials_per_seed, waveform)
    records = _collect_observations(
        stage=stage,
        trials=trials,
        snr_db=config.snr_db,
        bins=bins,
        cases=cases,
        waveform=waveform,
        quantizer=quantizer,
        progress_every=config.progress_every,
    )
    base, actions = _evaluate_action_records(
        records,
        temperature_by_snr=temperature_by_snr,
        quantizer=quantizer,
        stage_label=stage,
        progress_every=config.progress_every,
        partial_base_path=output / f"partial_{stage}_base.csv",
        partial_action_path=output / f"partial_{stage}_actions.csv",
    )
    return records, base, actions


def _fit_and_certify(
    fit_samples: pd.DataFrame,
    certificate_samples: pd.DataFrame,
    *,
    config: Stage2A1Config,
) -> tuple[dict[float, SafeGatePolicy], pd.DataFrame]:
    policies: dict[float, SafeGatePolicy] = {}
    rows: list[pd.DataFrame] = []
    for snr_index, snr in enumerate(config.snr_db):
        fit = fit_samples[fit_samples["snr_db"].eq(float(snr))]
        certificate = certificate_samples[
            certificate_samples["snr_db"].eq(float(snr))
        ]
        threshold, candidates = fit_threshold(
            fit,
            repetitions=config.bootstrap_repetitions,
            seed=2026072800 + 20 * snr_index,
        )
        candidates.insert(0, "snr_db", float(snr))
        candidates["selected_on_fit"] = candidates["threshold"].eq(threshold)
        rows.append(candidates)
        fit_lcb = 0.0
        selected = candidates[candidates["selected_on_fit"]]
        if len(selected) == 1:
            fit_lcb = float(selected.iloc[0]["lcb"])
        policy = certify_threshold(
            certificate,
            snr_db=float(snr),
            threshold=threshold,
            fit_lcb=fit_lcb,
            repetitions=config.bootstrap_repetitions,
            seed=2026072801 + 20 * snr_index,
        )
        policies[float(snr)] = policy
    return policies, pd.concat(rows, ignore_index=True)


def _records_by_key(
    records: list[dict[str, object]],
) -> dict[tuple[object, ...], dict[str, object]]:
    return {
        (
            int(record["seed"]),
            int(record["trial"]),
            str(record["cluster_id"]),
            float(record["snr_db"]),
        ): record
        for record in records
    }


def _result_for_position(
    *,
    key: tuple[object, ...],
    position: int,
    method: str,
    records: dict[tuple[object, ...], dict[str, object]],
    base_rows: pd.DataFrame,
    action_group: pd.DataFrame,
) -> dict[str, object]:
    base = base_rows
    for name, value in zip(KEYS, key, strict=True):
        base = base[base[name].eq(value)]
    if len(base) != 1:
        raise RuntimeError("base action row is missing or duplicated")
    return _selected_action_result(
        record=records[key],
        base_row=base.iloc[0],
        action_group=action_group,
        mode="calibrated",
        selected_position=int(position),
        method=method,
    )


def _adaptive_results(
    records: list[dict[str, object]],
    base_rows: pd.DataFrame,
    action_rows: pd.DataFrame,
    samples: pd.DataFrame,
    policies: dict[float, SafeGatePolicy],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed_samples = samples.set_index(list(KEYS), verify_integrity=True)
    record_index = _records_by_key(records)
    result_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for key, group in action_rows.groupby(list(KEYS), sort=False):
        snr = float(key[-1])
        policy = policies[snr]
        sample = indexed_samples.loc[key]
        applied = apply_threshold(
            pd.DataFrame([sample.to_dict()]),
            policy.selected_threshold,
        ).iloc[0]
        predicted_position = int(sample["candidate_position"])
        oracle = group.sort_values(
            ["calibrated_squared_error_samples2", "position"],
            ascending=[True, True],
            kind="stable",
        ).iloc[0]
        positions = (
            ("FixedAction-B29", int(sample["fixed_position"])),
            ("PredictedAction-B29", predicted_position),
            ("BRSR-SafeGate-B29", int(applied["selected_position"])),
            ("OracleAction-B29", int(oracle["position"])),
        )
        for method, position in positions:
            result_rows.append(
                _result_for_position(
                    key=key,
                    position=position,
                    method=method,
                    records=record_index,
                    base_rows=base_rows,
                    action_group=group,
                )
            )
        decision_rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "threshold": policy.selected_threshold,
                "snr_certified": policy.certified,
                "fixed_position": int(sample["fixed_position"]),
                "candidate_position": predicted_position,
                "selected_position": int(applied["selected_position"]),
                "predicted_advantage_over_fixed_samples2": float(
                    sample["predicted_advantage_over_fixed_samples2"]
                ),
                "realized_advantage_over_fixed_samples2": float(
                    sample["realized_advantage_over_fixed_samples2"]
                ),
                "adaptive_selected": bool(applied["adaptive_selected"]),
            }
        )
    return pd.DataFrame(result_rows), pd.DataFrame(decision_rows)


def _tail_diagnostic(
    results: pd.DataFrame,
    *,
    baseline: str,
) -> dict[str, object]:
    keys = list(KEYS)
    target = results[results["method"].eq("BRSR-SafeGate-B29")]
    reference = results[results["method"].eq(baseline)]
    paired = target[keys + ["squared_error_samples2"]].rename(
        columns={"squared_error_samples2": "target_sq"}
    ).merge(
        reference[keys + ["squared_error_samples2"]].rename(
            columns={"squared_error_samples2": "baseline_sq"}
        ),
        on=keys,
        validate="one_to_one",
    )
    gain = paired["baseline_sq"].to_numpy() - paired["target_sq"].to_numpy()
    clusters = paired["cluster_id"].to_numpy(dtype=str)
    loss = np.maximum(-gain, 0.0)
    total_loss = float(np.sum(loss))
    cluster_loss = (
        pd.DataFrame({"cluster_id": clusters, "loss": loss})
        .groupby("cluster_id", as_index=False)["loss"]
        .sum()
        .sort_values("loss", ascending=False)
    )
    condition_count = max(1, int(math.ceil(0.10 * len(loss))))
    return {
        "baseline_method": baseline,
        "leave_one_cluster_min_gain_samples2": leave_one_cluster_min_mean(
            gain,
            clusters,
        ),
        "largest_cluster_positive_loss_share": (
            float(cluster_loss.iloc[0]["loss"]) / total_loss
            if total_loss > 0.0
            else 0.0
        ),
        "top3_cluster_positive_loss_share": (
            float(cluster_loss.head(3)["loss"].sum()) / total_loss
            if total_loss > 0.0
            else 0.0
        ),
        "worst10pct_condition_positive_loss_share": (
            float(np.sort(loss)[-condition_count:].sum()) / total_loss
            if total_loss > 0.0
            else 0.0
        ),
    }


def _gate_decision(
    *,
    mode: str,
    config: Stage2A1Config,
    policies: dict[float, SafeGatePolicy],
    results: pd.DataFrame,
    effects: pd.DataFrame,
    per_snr_effects: pd.DataFrame,
    tail: pd.DataFrame,
    source_valid: bool,
    source_seeds: set[int],
) -> dict[str, object]:
    new_seeds = (
        *config.gate_fit_seeds,
        *config.gate_certificate_seeds,
        *config.development_seeds,
    )
    adaptive_methods = {
        "FixedAction-B29",
        "PredictedAction-B29",
        "BRSR-SafeGate-B29",
        "OracleAction-B29",
    }
    adaptive = results[results["method"].isin(adaptive_methods)]
    safe_effects = effects[
        effects["target_method"].eq("BRSR-SafeGate-B29")
        & effects["baseline_method"].str.contains("RateMatched")
    ]
    safe_per_snr = per_snr_effects[
        per_snr_effects["target_method"].eq("BRSR-SafeGate-B29")
    ]
    development_cluster_count = int(
        results[results["method"].eq("BRSR-SafeGate-B29")][
            "cluster_id"
        ].nunique()
    )
    rate_rounding_tolerance = 8.0 / development_cluster_count + 1e-12
    mechanical = {
        "frozen_stage2a_source_valid": source_valid,
        "new_seed_pools_disjoint": len(new_seeds) == len(set(new_seeds)),
        "new_seeds_disjoint_from_stage2a": not (set(new_seeds) & source_seeds),
        "safe_gate_never_stops": bool(
            not results[results["method"].eq("BRSR-SafeGate-B29")][
                "stopped"
            ].any()
        ),
        "adaptive_methods_are_exactly_29_bits": bool(
            set(adaptive["total_bits"].unique()) == {TARGET_TOTAL_BITS}
        ),
        "rate_matched_mean_bits_equal": bool(
            np.all(
                np.abs(safe_effects["mean_total_bits_difference"])
                <= rate_rounding_tolerance
            )
        ),
    }
    statistical = {
        "at_least_one_snr_certified": any(
            policy.certified for policy in policies.values()
        ),
        "safe_gate_vs_both_rate_matched_ci_positive": bool(
            len(safe_effects) == 2
            and np.all(safe_effects["mse_gain_ci_low"] > 0.0)
        ),
        "leave_one_cluster_gain_positive_vs_both": bool(
            len(tail) == 2
            and np.all(
                tail["leave_one_cluster_min_gain_samples2"] > 0.0
            )
        ),
        "no_snr_has_significant_reverse_effect": bool(
            not (safe_per_snr["mse_gain_ci_high"] < 0.0).any()
        ),
    }
    point_gain_positive = bool(
        len(safe_effects) == 2 and np.all(safe_effects["mse_gain"] > 0.0)
    )
    if mode == "smoke":
        status = "SMOKE_ONLY"
    elif not all(mechanical.values()):
        status = "HOLD-MECHANICAL"
    elif not statistical["at_least_one_snr_certified"]:
        status = "NO-GO-COMPONENT-SAFE-GATE"
    elif all(statistical.values()):
        status = "STAGE2A1_PASS"
    elif point_gain_positive:
        status = "HOLD-POWER-STAGE2A1"
    else:
        status = "NO-GO-CONFIG-STAGE2A1"
    return {
        "status": status,
        "mode": mode,
        "target_total_bits": TARGET_TOTAL_BITS,
        "mechanical_validations": mechanical,
        "statistical_validations": statistical,
        "certified_snr_db": [
            snr for snr, policy in policies.items() if policy.certified
        ],
        "interpretation_boundary": (
            "Stage 2A.1 tests a calibration-certified one-action replacement "
            "with fixed-action fallback at 29 total bits. It does not test "
            "stopping, multiple actions, locked performance or route-level "
            "generalization."
        ),
    }


def _plot(
    output: Path,
    summary: pd.DataFrame,
    policies: pd.DataFrame,
    decisions: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.35,
        }
    )
    methods = (
        "BRSR-SafeGate-B29",
        "FixedAction-B29",
        "PredictedAction-B29",
        "BestStatic-RateMatched-B29",
        "SNR-only-RateMatched-B29",
        "OracleAction-B29",
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for method in methods:
        subset = summary[summary["method"].eq(method)]
        if not subset.empty:
            ax.plot(
                subset["snr_db"],
                subset["rmse_samples"],
                marker="o",
                label=method,
            )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("TDOA RMSE (samples)")
    ax.legend(loc="best", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output / "Fig1_Stage2A1_Equal29Bit_RMSE.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    finite = policies[np.isfinite(policies["selected_threshold"])]
    axes[0].plot(
        finite["snr_db"],
        finite["selected_threshold"],
        marker="o",
        color="#1f77b4",
    )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Certified score threshold")
    activation = (
        decisions.groupby("snr_db", as_index=False)["adaptive_selected"].mean()
    )
    axes[1].plot(
        activation["snr_db"],
        activation["adaptive_selected"],
        marker="s",
        color="#d62728",
    )
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Development adaptive fraction")
    axes[1].set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(output / "Fig2_Stage2A1_Gate_And_Activation.svg")
    plt.close(fig)

    safe = effects[effects["target_method"].eq("BRSR-SafeGate-B29")]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    labels = safe["baseline_method"].tolist()
    ax.errorbar(
        np.arange(len(safe)),
        safe["mse_gain"],
        yerr=np.vstack(
            [
                safe["mse_gain"] - safe["mse_gain_ci_low"],
                safe["mse_gain_ci_high"] - safe["mse_gain"],
            ]
        ),
        fmt="o",
        capsize=4,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(safe)), labels, rotation=15, ha="right")
    ax.set_ylabel("Paired MSE gain (samples squared)")
    fig.tight_layout()
    fig.savefig(output / "Fig3_Stage2A1_Paired_Effects.svg")
    plt.close(fig)


def _run(config: Stage2A1Config, output: Path) -> None:
    manifest, plans, temperatures, frozen_hashes = _load_source()
    source_seeds = _source_seed_set(manifest)
    config.validate(source_seeds)
    waveform = WaveformConfig()
    waveform.validate()
    quantizer = NestedComplexQuantizer(
        max_bits=QUANTIZER_BITS,
        clip_abs=QUANTIZER_CLIP_ABS,
    )
    cases, bins, prior_statistics = _build_frozen_model(manifest, waveform)
    fixed_position, fixed_bin = _fixed_position(plans)

    fit_records, fit_base, fit_actions = _records_and_actions(
        stage="gate_fit",
        seeds=config.gate_fit_seeds,
        trials_per_seed=config.gate_fit_trials_per_seed,
        config=config,
        cases=cases,
        bins=bins,
        waveform=waveform,
        quantizer=quantizer,
        temperature_by_snr=temperatures,
        output=output,
    )
    certificate_records, certificate_base, certificate_actions = (
        _records_and_actions(
            stage="gate_certificate",
            seeds=config.gate_certificate_seeds,
            trials_per_seed=config.gate_certificate_trials_per_seed,
            config=config,
            cases=cases,
            bins=bins,
            waveform=waveform,
            quantizer=quantizer,
            temperature_by_snr=temperatures,
            output=output,
        )
    )
    fit_samples = build_action_gate_samples(
        fit_actions,
        fixed_position=fixed_position,
    )
    certificate_samples = build_action_gate_samples(
        certificate_actions,
        fixed_position=fixed_position,
    )
    policies, threshold_candidates = _fit_and_certify(
        fit_samples,
        certificate_samples,
        config=config,
    )

    development_records, development_base, development_actions = (
        _records_and_actions(
            stage="development",
            seeds=config.development_seeds,
            trials_per_seed=config.development_trials_per_seed,
            config=config,
            cases=cases,
            bins=bins,
            waveform=waveform,
            quantizer=quantizer,
            temperature_by_snr=temperatures,
            output=output,
        )
    )
    development_samples = build_action_gate_samples(
        development_actions,
        fixed_position=fixed_position,
    )
    adaptive, decisions = _adaptive_results(
        development_records,
        development_base,
        development_actions,
        development_samples,
        policies,
    )
    static = _evaluate_static_development(
        development_records,
        plans=plans,
        temperature_by_snr=temperatures,
        quantizer=quantizer,
    )
    frames = [adaptive, static]
    for family in ("BestStatic", "SNR-only"):
        frames.append(
            _rate_matched_rows(
                adaptive_rows=adaptive,
                static_rows=static,
                adaptive_method="BRSR-SafeGate-B29",
                family=family,
                label=f"{family}-RateMatched-B29",
            )
        )
    results = pd.concat(frames, ignore_index=True)
    results = results[list(RESULT_COLUMNS)]

    comparisons = (
        ("BRSR-SafeGate-B29", "BestStatic-RateMatched-B29"),
        ("BRSR-SafeGate-B29", "SNR-only-RateMatched-B29"),
        ("BRSR-SafeGate-B29", "FixedAction-B29"),
        ("OracleAction-B29", "BestStatic-RateMatched-B29"),
        ("OracleAction-B29", "SNR-only-RateMatched-B29"),
    )
    effects = pd.DataFrame(
        [
            _paired_effect(
                results,
                target_method=target,
                baseline_method=baseline,
                repetitions=config.bootstrap_repetitions,
                seed=2026072900 + index,
            )
            for index, (target, baseline) in enumerate(comparisons)
        ]
    )
    per_snr = []
    for comparison_index, (target, baseline) in enumerate(comparisons[:2]):
        for snr_index, snr in enumerate(config.snr_db):
            per_snr.append(
                _paired_effect(
                    results,
                    target_method=target,
                    baseline_method=baseline,
                    repetitions=config.bootstrap_repetitions,
                    seed=2026073000 + 20 * comparison_index + snr_index,
                    snr_db=snr,
                )
            )
    per_snr_effects = pd.DataFrame(per_snr)
    tail = pd.DataFrame(
        [
            _tail_diagnostic(results, baseline=baseline)
            for baseline in (
                "BestStatic-RateMatched-B29",
                "SNR-only-RateMatched-B29",
            )
        ]
    )
    policy_rows = pd.DataFrame([asdict(policy) for policy in policies.values()])
    summary = _method_summary(results)
    source_valid = bool(frozen_hashes)
    gate = _gate_decision(
        mode=config.mode,
        config=config,
        policies=policies,
        results=results,
        effects=effects,
        per_snr_effects=per_snr_effects,
        tail=tail,
        source_valid=source_valid,
        source_seeds=source_seeds,
    )

    prior_statistics.to_csv(output / "prior_statistics.csv", index=False)
    threshold_candidates.to_csv(
        output / "gate_fit_threshold_candidates.csv",
        index=False,
    )
    policy_rows.to_csv(output / "frozen_safe_gate_policies.csv", index=False)
    fit_samples.to_csv(output / "gate_fit_samples.csv", index=False)
    certificate_samples.to_csv(
        output / "gate_certificate_samples.csv",
        index=False,
    )
    development_actions.to_csv(
        output / "stage2a1_action_rows.csv",
        index=False,
    )
    decisions.to_csv(output / "stage2a1_gate_decisions.csv", index=False)
    results.to_csv(output / "stage2a1_trial_results.csv", index=False)
    summary.to_csv(output / "method_summary_by_snr.csv", index=False)
    effects.to_csv(output / "rate_matched_effects.csv", index=False)
    per_snr_effects.to_csv(
        output / "rate_matched_effects_by_snr.csv",
        index=False,
    )
    tail.to_csv(output / "tail_safety_diagnostics.csv", index=False)
    _json_dump(output / "gate_decision.json", gate)
    _json_dump(
        output / "manifest.json",
        {
            "experiment": "BRSR Stage 2A.1 safe action gate",
            "mode": config.mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(config),
            "frozen_stage2a_source": str(SOURCE_STAGE2A),
            "frozen_source_sha256": frozen_hashes,
            "protocol": {
                "candidate_pool": PRIMARY_POOL,
                "likelihood": PRIMARY_LIKELIHOOD,
                "quadrature": [5, 64],
                "fixed_position": fixed_position,
                "fixed_bin": fixed_bin,
                "base_payload_bits": BASE_PAYLOAD_BITS,
                "acquired_payload_bits": ACQUIRED_BIN_PAYLOAD_BITS,
                "feedback_bits": ONE_ACTION_FEEDBACK_BITS,
                "adaptive_total_bits": TARGET_TOTAL_BITS,
                "fit_threshold_quantiles": [
                    float(value) for value in np.linspace(0.0, 0.9, 10)
                ],
                "certificate_rule": (
                    "one-sided cluster-bootstrap 95% LCB > 0 and "
                    "leave-one-cluster minimum mean gain > 0"
                ),
            },
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (
                    Path(__file__),
                    ROOT / "brsr_stage2_rate_adaptive" / "stage2a1.py",
                )
            },
        },
    )
    _plot(output, summary, policy_rows, decisions, effects)
    for path in output.glob("partial_*.csv"):
        path.unlink()
    print("\nStage 2A.1 gate", flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = _parse_args()
    config = CONFIGS[args.mode]
    output = _output_dir(args.mode)
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    original_stdout = sys.stdout
    log_handle = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_handle)
    started = time.perf_counter()
    try:
        print("BRSR Stage 2A.1 safe action-gating development", flush=True)
        print(f"Mode: {config.mode}", flush=True)
        print(f"Output: {output}", flush=True)
        print(
            "Frozen: Stage 2A model/temperature/plans; 29-bit one-action "
            "comparison; no stop and no locked data.",
            flush=True,
        )
        print(
            "Gate-fit, certificate and development seeds are disjoint. "
            "Development labels never fit or certify the gate.",
            flush=True,
        )
        _run(config, output)
        print(f"[Done] elapsed={time.perf_counter() - started:.1f}s", flush=True)
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
