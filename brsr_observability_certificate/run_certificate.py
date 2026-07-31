"""运行BRSR动作价值可观测性证书。

PyCharm直接运行默认执行development；Codex仅执行 ``--mode smoke``。
该入口使用冻结Stage 2A物理模型和全新种子，不修改既有BRSR算法。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
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

from brsr_observability_certificate.audit_oracles import (  # noqa: E402
    audit_oracles,
)
from brsr_observability_certificate.certificate import (  # noqa: E402
    STATIC_BASELINES,
    choose_model_on_calibration,
    diagnostic_policy_rows,
    fit_candidate_models,
    gate_decision,
    learned_policy_rows,
    paired_effects,
    POLICY_KEY_COLUMNS,
    prediction_diagnostics,
    summarize_policies,
)
from brsr_observability_certificate.features import (  # noqa: E402
    build_observable_action_rows,
)
from brsr_route_feasibility.run_stage1 import Tee  # noqa: E402
from brsr_stage2_rate_adaptive.run_stage2a import (  # noqa: E402
    QUANTIZER_BITS,
    QUANTIZER_CLIP_ABS,
    _evaluate_static_development,
)
from brsr_stage2_rate_adaptive.run_stage2a1 import (  # noqa: E402
    SOURCE_STAGE2A,
    _build_frozen_model,
    _fixed_position,
    _load_source,
    _records_and_actions,
    _source_seed_set,
)
from brsr_stage2_rate_adaptive.stage2a import (  # noqa: E402
    rate_match_assignments,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer  # noqa: E402
from brsr_tdoa.brsr_waveform import WaveformConfig  # noqa: E402


PYCHARM_RUN_MODE = "development"
TARGET_TOTAL_BITS = 29


@dataclass(frozen=True)
class ObservabilityConfig:
    mode: str
    snr_db: tuple[float, ...]
    train_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    train_trials_per_seed: int
    calibration_trials_per_seed: int
    test_trials_per_seed: int
    bootstrap_repetitions: int
    progress_every: int

    def validate(self, source_seeds: set[int]) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        pools = self.train_seeds, self.calibration_seeds, self.test_seeds
        if any(not pool for pool in pools):
            raise ValueError("all observability seed pools must be non-empty")
        flattened = [seed for pool in pools for seed in pool]
        if len(flattened) != len(set(flattened)):
            raise ValueError("observability seed pools must be disjoint")
        if set(flattened) & source_seeds:
            raise ValueError("observability seeds overlap frozen BRSR sources")
        if min(
            self.train_trials_per_seed,
            self.calibration_trials_per_seed,
            self.test_trials_per_seed,
        ) < 1:
            raise ValueError("all observability trial counts must be positive")
        if self.bootstrap_repetitions < 20 or self.progress_every < 1:
            raise ValueError("invalid bootstrap or progress configuration")
        if len(self.snr_db) < 2:
            raise ValueError("at least two SNR values are required")


CONFIGS = {
    "smoke": ObservabilityConfig(
        mode="smoke",
        snr_db=(-5.0, 10.0),
        train_seeds=(2026073011, 2026073012),
        calibration_seeds=(2026073021,),
        test_seeds=(2026073031,),
        train_trials_per_seed=4,
        calibration_trials_per_seed=4,
        test_trials_per_seed=8,
        bootstrap_repetitions=50,
        progress_every=2,
    ),
    "development": ObservabilityConfig(
        mode="development",
        snr_db=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
        train_seeds=(2026073011, 2026073012, 2026073013, 2026073014),
        calibration_seeds=(2026073021, 2026073022),
        test_seeds=(2026073031, 2026073032, 2026073033),
        train_trials_per_seed=30,
        calibration_trials_per_seed=30,
        test_trials_per_seed=40,
        bootstrap_repetitions=2000,
        progress_every=10,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=tuple(CONFIGS),
        default=PYCHARM_RUN_MODE,
    )
    return parser.parse_args()


def _output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "运行结果"
        / "BRSR"
        / f"BRSR_OBSERVABILITY_{timestamp}_{mode}"
    )


def _json_dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "brsr_observability_certificate" / "audit_oracles.py",
        ROOT / "brsr_observability_certificate" / "certificate.py",
        ROOT / "brsr_observability_certificate" / "features.py",
        ROOT / "brsr_observability_certificate" / "run_certificate.py",
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _run_split(
    *,
    split: str,
    seeds: tuple[int, ...],
    trials_per_seed: int,
    config: ObservabilityConfig,
    cases: dict[str, object],
    bins: np.ndarray,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    temperature_by_snr: dict[float, float],
    fixed_position: int,
    output: Path,
) -> tuple[
    list[dict[str, object]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    records, base, actions = _records_and_actions(
        stage=f"observability_{split}",
        seeds=seeds,
        trials_per_seed=trials_per_seed,
        config=config,
        cases=cases,
        bins=bins,
        waveform=waveform,
        quantizer=quantizer,
        temperature_by_snr=temperature_by_snr,
        output=output,
    )
    base.to_csv(output / f"raw_{split}_base_rows.csv", index=False)
    actions.to_csv(output / f"raw_{split}_action_rows.csv", index=False)
    features = build_observable_action_rows(
        records,
        actions,
        quantizer=quantizer,
        temperature_by_snr=temperature_by_snr,
        fixed_position=fixed_position,
    )
    features.insert(0, "split", split)
    features.to_csv(output / f"observable_{split}_action_rows.csv", index=False)
    return records, base, actions, features


def _simple_static_rate_matched(
    static_rows: pd.DataFrame,
    sample_rows: pd.DataFrame,
    *,
    family: str,
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    label = f"{family}-RateMatched-B29"
    samples = sample_rows[
        list(POLICY_KEY_COLUMNS)
    ].drop_duplicates().sort_values(list(POLICY_KEY_COLUMNS), kind="stable")
    for snr, snr_samples in samples.groupby("snr_db", sort=True):
        assignments = rate_match_assignments(
            snr_samples["cluster_id"],
            desired_mean_bits=TARGET_TOTAL_BITS,
            low_bits=24,
            high_bits=32,
            salt=f"observability|{family}|{snr:g}",
        )
        candidates = static_rows[
            static_rows["snr_db"].eq(float(snr))
            & static_rows["method"].str.startswith(f"{family}-")
        ]
        for _, sample in snr_samples.iterrows():
            cluster_id = str(sample["cluster_id"])
            bits = assignments[cluster_id]
            selected = candidates[
                candidates["seed"].eq(int(sample["seed"]))
                & candidates["trial"].eq(int(sample["trial"]))
                & candidates["cluster_id"].eq(cluster_id)
                & candidates["method"].eq(f"{family}-B{bits}")
            ]
            if len(selected) != 1:
                raise RuntimeError("rate-matched static baseline is misaligned")
            row = selected.iloc[0]
            output.append(
                {
                    **{
                        column: sample[column]
                        for column in POLICY_KEY_COLUMNS
                    },
                    "method": label,
                    "selected_position": str(row["selected_position"])
                    if not pd.isna(row["selected_position"])
                    else "",
                    "squared_error_samples2": float(
                        row["squared_error_samples2"]
                    ),
                    "selected_clairvoyant_action": False,
                    "payload_bits": int(bits),
                    "feedback_bits": 0,
                    "total_bits": int(bits),
                }
            )
    frame = pd.DataFrame(output)
    if len(frame) != len(samples):
        raise RuntimeError("rate-matched static baseline lost samples")
    return frame


def _add_budget_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "payload_bits" not in result:
        result["payload_bits"] = 24
    if "feedback_bits" not in result:
        result["feedback_bits"] = 5
    if "total_bits" not in result:
        result["total_bits"] = 29
    return result


def _plot_certificate(
    *,
    summary: pd.DataFrame,
    test_features: pd.DataFrame,
    selected_prediction: np.ndarray,
    selected_method: str,
    fixed_position: int,
    output: Path,
) -> None:
    selected_methods = (
        "BestStatic-RateMatched-B29",
        "SNR-only-RateMatched-B29",
        "RiskGreedy-B29",
        selected_method,
        "ClairvoyantOutcomeOracle-B29",
    )
    palette = {
        "BestStatic-RateMatched-B29": "#4C78A8",
        "SNR-only-RateMatched-B29": "#72B7B2",
        "RiskGreedy-B29": "#F58518",
        selected_method: "#E45756",
        "ClairvoyantOutcomeOracle-B29": "#7A5195",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    snr_summary = summary[summary["scope"].eq("snr")]
    for method in selected_methods:
        group = snr_summary[snr_summary["method"].eq(method)].sort_values("snr_db")
        if group.empty:
            continue
        axes[0].plot(
            group["snr_db"],
            group["rmse_samples"],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=method,
            color=palette[method],
        )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("TDOA RMSE (samples)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")

    frame = test_features[
        [
            *POLICY_KEY_COLUMNS,
            "action_position",
            "actual_squared_error_samples2",
            "actual_gain_vs_fixed_samples2",
        ]
    ].copy()
    frame["predicted_loss"] = selected_prediction
    fixed_predictions = frame[
        frame["action_position"].astype(int).eq(int(fixed_position))
    ][[*POLICY_KEY_COLUMNS, "predicted_loss"]].rename(
        columns={"predicted_loss": "fixed_predicted_loss"}
    )
    frame = frame.merge(
        fixed_predictions,
        on=list(POLICY_KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    if frame["fixed_predicted_loss"].isna().any():
        raise RuntimeError("fixed-action prediction is missing from plot data")
    frame["predicted_gain_proxy"] = (
        frame["fixed_predicted_loss"] - frame["predicted_loss"]
    )
    try:
        frame["bin"] = pd.qcut(
            frame["predicted_gain_proxy"],
            q=10,
            duplicates="drop",
        )
        calibrated = frame.groupby("bin", observed=True).agg(
            predicted=("predicted_gain_proxy", "mean"),
            actual=("actual_gain_vs_fixed_samples2", "mean"),
        )
        axes[1].plot(
            calibrated["predicted"],
            calibrated["actual"],
            marker="o",
            linewidth=1.2,
            color=palette[selected_method],
        )
    except ValueError:
        axes[1].text(
            0.5,
            0.5,
            "Insufficient variation",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_xlabel("Predicted relative value (binned)")
    axes[1].set_ylabel("Observed relative value (samples²)")
    axes[1].grid(alpha=0.25)

    all_summary = summary[summary["scope"].eq("all")].set_index("method")
    bars = (
        "BestStatic-RateMatched-B29",
        selected_method,
        "ClairvoyantOutcomeOracle-B29",
    )
    values = [float(all_summary.loc[method, "mse_samples2"]) for method in bars]
    axes[2].bar(
        ["Static", "Observable", "Clairvoyant"],
        values,
        color=[palette[method] for method in bars],
        width=0.65,
    )
    axes[2].set_ylabel("Mean squared error (samples²)")
    axes[2].set_title("Observable vs. unobservable gap")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output / "Fig1_BRSR_Observability_Certificate.svg")
    plt.close(fig)


def _cleanup_partials(output: Path) -> None:
    for path in output.glob("partial_*.csv"):
        path.unlink()


def _run(config: ObservabilityConfig, output: Path) -> None:
    started = time.perf_counter()
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
    if not set(config.snr_db).issubset(temperatures):
        raise RuntimeError("frozen temperatures do not cover observability SNRs")

    print("BRSR observable action-value certificate", flush=True)
    print(f"Mode: {config.mode}", flush=True)
    print(f"Output: {output}", flush=True)
    print(f"SNR: {list(config.snr_db)}", flush=True)
    print(
        "Splits: "
        f"train={len(config.train_seeds)}x{config.train_trials_per_seed}, "
        f"calibration={len(config.calibration_seeds)}"
        f"x{config.calibration_trials_per_seed}, "
        f"test={len(config.test_seeds)}x{config.test_trials_per_seed}",
        flush=True,
    )
    print(
        "Frozen protocol: Current-Fisher, Linear-Marginal, 5x64, "
        "29-bit adaptive vs 24/32-bit rate-matched static",
        flush=True,
    )

    split_data = {}
    for split, seeds, trials_per_seed in (
        ("train", config.train_seeds, config.train_trials_per_seed),
        (
            "calibration",
            config.calibration_seeds,
            config.calibration_trials_per_seed,
        ),
        ("test", config.test_seeds, config.test_trials_per_seed),
    ):
        split_data[split] = _run_split(
            split=split,
            seeds=seeds,
            trials_per_seed=trials_per_seed,
            config=config,
            cases=cases,
            bins=bins,
            waveform=waveform,
            quantizer=quantizer,
            temperature_by_snr=temperatures,
            fixed_position=fixed_position,
            output=output,
        )

    train_features = split_data["train"][3]
    calibration_features = split_data["calibration"][3]
    test_records, _, _, test_features = split_data["test"]
    all_features = pd.concat(
        [train_features, calibration_features, test_features],
        ignore_index=True,
    )
    all_features.to_csv(output / "observable_action_rows.csv", index=False)

    calibration_models, columns = fit_candidate_models(train_features)
    calibration_policies, _ = learned_policy_rows(
        calibration_features,
        calibration_models,
        columns,
    )
    selected_method, model_selection = choose_model_on_calibration(
        calibration_policies
    )

    fit_features = pd.concat(
        [train_features, calibration_features],
        ignore_index=True,
    )
    final_models, final_columns = fit_candidate_models(fit_features)
    if final_columns != columns:
        raise RuntimeError("feature schema changed between model fits")
    learned_results, learned_predictions = learned_policy_rows(
        test_features,
        final_models,
        final_columns,
    )
    diagnostic_results, diagnostic_predictions = diagnostic_policy_rows(
        test_features,
        fixed_position=fixed_position,
    )
    adaptive_results = _add_budget_columns(
        pd.concat([learned_results, diagnostic_results], ignore_index=True)
    )

    static_results = _evaluate_static_development(
        test_records,
        plans=plans,
        temperature_by_snr=temperatures,
        quantizer=quantizer,
    )
    rate_matched = [
        _simple_static_rate_matched(
            static_results,
            adaptive_results[adaptive_results["method"].eq(selected_method)],
            family=family,
        )
        for family in ("BestStatic", "SNR-only")
    ]
    results = pd.concat([adaptive_results, *rate_matched], ignore_index=True)
    results.to_csv(output / "observable_policy_results.csv", index=False)

    summary = summarize_policies(results)
    summary.to_csv(output / "observable_policy_summary.csv", index=False)
    all_predictions = {**learned_predictions, **diagnostic_predictions}
    diagnostics = prediction_diagnostics(test_features, all_predictions)
    diagnostics.to_csv(output / "observable_prediction_diagnostics.csv", index=False)
    effects = paired_effects(
        results,
        targets=(
            *tuple(final_models),
            "RiskGreedy-B29",
        ),
        baselines=STATIC_BASELINES,
        repetitions=config.bootstrap_repetitions,
        seed=2026073041,
    )
    effects.to_csv(output / "observable_paired_effects.csv", index=False)
    model_selection.to_csv(output / "model_selection.csv", index=False)
    audit_oracles().to_csv(output / "oracle_audit.csv", index=False)
    prior_statistics.to_csv(output / "prior_statistics.csv", index=False)

    decision = gate_decision(
        mode=config.mode,
        selected_method=selected_method,
        effects=effects,
        diagnostics=diagnostics,
    )
    _json_dump(output / "gate_decision.json", decision)
    _plot_certificate(
        summary=summary,
        test_features=test_features,
        selected_prediction=learned_predictions[selected_method],
        selected_method=selected_method,
        fixed_position=fixed_position,
        output=output,
    )

    budget = (
        results.groupby("method", as_index=False)
        .agg(mean_total_bits=("total_bits", "mean"))
        .sort_values("method", kind="stable")
    )
    budget.to_csv(output / "budget_audit.csv", index=False)
    if not np.allclose(budget["mean_total_bits"], TARGET_TOTAL_BITS, atol=1e-12):
        raise RuntimeError("observability comparison is not rate matched")

    manifest_output = {
        "experiment": "BRSR observable action-value certificate",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "frozen_stage2a_source": str(SOURCE_STAGE2A),
        "frozen_source_sha256": frozen_hashes,
        "source_sha256": _source_hashes(),
        "selected_model_on_calibration": selected_method,
        "feature_columns": final_columns,
        "feature_count": len(final_columns),
        "protocol": {
            "candidate_pool": "Current-Fisher",
            "likelihood": "Linear-Marginal",
            "quadrature": [5, 64],
            "fixed_position": fixed_position,
            "fixed_bin": fixed_bin,
            "adaptive_total_bits": TARGET_TOTAL_BITS,
            "static_rate_match": "24/32-bit cluster-hash time sharing",
            "exact_snr_available": True,
            "locked_data_used": False,
            "clairvoyant_used_for_gate": False,
            "iid_block_status": "NO_GO_PROTOCOL_IID_BLOCK",
        },
        "data_contract": {
            "train_calibration_test_seeds_disjoint": True,
            "all_seeds_disjoint_from_frozen_sources": True,
            "test_not_used_for_model_selection": True,
            "truth_excluded_from_feature_whitelist": True,
        },
        "gate_status": decision["status"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    _json_dump(output / "manifest.json", manifest_output)
    _cleanup_partials(output)

    print("\nGate decision", flush=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print(f"[Done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    args = _parse_args()
    config = CONFIGS[args.mode]
    output = _output_dir(config.mode)
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        tee = Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            _run(config, output)


if __name__ == "__main__":
    main()
