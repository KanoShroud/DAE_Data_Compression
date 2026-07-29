"""Build the read-only Stage 2A.1 feedback-amortization certificate.

The source result is never modified. PyCharm can run this file directly with
the frozen formal Stage 2A.1 development result used by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brsr_route_feasibility.run_stage1 import Tee  # noqa: E402
from brsr_stage2_rate_adaptive.stage2a1_protocol import (  # noqa: E402
    BLOCK_SIZES,
    action_gap_table,
    assert_fixed_matches_b24,
    assert_l1_alignment,
    build_block_policy_rows,
    effect_row,
    expected_static_frontier,
    ideal_per_sample_action_rows,
    method_summary,
    protocol_decision,
    sha256_file,
    validate_source_tables,
)


DEFAULT_SOURCE = (
    ROOT
    / "运行结果"
    / "BRSR"
    / "BRSR_STAGE2A1_20260727_115123_development"
)
SOURCE_FILES = (
    "stage2a1_trial_results.csv",
    "stage2a1_action_rows.csv",
    "manifest.json",
)
BOOTSTRAP_REPETITIONS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Stage 2A.1 protocol feasibility certificate."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Frozen Stage 2A.1 formal result directory.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=BOOTSTRAP_REPETITIONS,
    )
    return parser.parse_args()


def output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "运行结果"
        / "BRSR"
        / f"BRSR_STAGE2A1_PROTOCOL_{timestamp}_diagnostic"
    )


def json_dump(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def source_hashes(source: Path) -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source result missing files: {missing}")
    return {name: sha256_file(source / name) for name in SOURCE_FILES}


def _all_effects(
    trials: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rows = []
    block_rows = []
    alignment: dict[str, float] = {}
    snr_values = sorted(trials["snr_db"].unique())
    seed_counter = 2026073200

    for block_size in BLOCK_SIZES:
        predicted = build_block_policy_rows(
            actions,
            block_size=block_size,
            policy="predicted",
        )
        oracle = build_block_policy_rows(
            actions,
            block_size=block_size,
            policy="oracle",
        )
        block_rows.extend([predicted, oracle])
        if block_size == 1:
            alignment.update(assert_l1_alignment(trials, predicted, oracle))

        target_bits = float(predicted["total_bits"].mean())
        if not math.isclose(
            target_bits,
            float(oracle["total_bits"].mean()),
            abs_tol=1e-12,
        ):
            raise RuntimeError("predicted and oracle block policies have unequal rates")
        for family in ("BestStatic", "SNR-only"):
            static = expected_static_frontier(
                trials,
                family=family,
                target_bits=target_bits,
                label=f"{family}-Expected",
            )
            for target in (predicted, oracle):
                seed_counter += 1
                rows.append(
                    effect_row(
                        target,
                        static,
                        target_method=str(target["method"].iloc[0]),
                        baseline_method=f"{family}-Expected",
                        block_size=block_size,
                        realizable=True,
                        repetitions=repetitions,
                        seed=seed_counter,
                    )
                )
                for snr in snr_values:
                    seed_counter += 1
                    rows.append(
                        effect_row(
                            target,
                            static,
                            target_method=str(target["method"].iloc[0]),
                            baseline_method=f"{family}-Expected",
                            block_size=block_size,
                            realizable=True,
                            repetitions=repetitions,
                            seed=seed_counter,
                            snr_db=float(snr),
                        )
                    )

        for method in ("PredictedAction-B29", "OracleAction-B29"):
            ideal = ideal_per_sample_action_rows(
                trials,
                method=method,
                block_size=block_size,
            )
            for family in ("BestStatic", "SNR-only"):
                static = expected_static_frontier(
                    trials,
                    family=family,
                    target_bits=float(ideal["total_bits"].mean()),
                    label=f"{family}-Expected",
                )
                seed_counter += 1
                rows.append(
                    effect_row(
                        ideal,
                        static,
                        target_method=str(ideal["method"].iloc[0]),
                        baseline_method=f"{family}-Expected",
                        block_size=block_size,
                        realizable=False,
                        repetitions=repetitions,
                        seed=seed_counter,
                    )
                )
    return (
        pd.DataFrame(rows),
        pd.concat(block_rows, ignore_index=True),
        alignment,
    )


def _plot(
    output: Path,
    payload: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for family, color, marker in (
        ("BestStatic", "#1f77b4", "o"),
        ("SNR-only", "#ff7f0e", "s"),
    ):
        subset = payload[payload["method"].str.startswith(f"{family}-B")]
        axes[0].plot(
            subset["mean_payload_bits"],
            subset["rmse_samples"],
            color=color,
            marker=marker,
            linewidth=1.5,
            label=family,
        )
    axes[0].set_xlabel("Forward payload (bit/observation)")
    axes[0].set_ylabel("TDOA RMSE (sample)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    overall = effects[
        effects["snr_db"].isna()
        & effects["realizable"]
        & effects["baseline_method"].eq("BestStatic-Expected")
    ]
    for prefix, color, marker, label in (
        ("BlockPredicted", "#2ca02c", "o", "Block predicted"),
        ("BlockOracle", "#9467bd", "^", "Block oracle"),
    ):
        subset = overall[
            overall["target_method"].str.startswith(prefix)
        ].sort_values("block_size")
        axes[1].plot(
            subset["block_size"],
            subset["target_rmse"],
            color=color,
            marker=marker,
            linewidth=1.5,
            label=label,
        )
    static_subset = overall[
        overall["target_method"].str.startswith("BlockPredicted")
    ].sort_values("block_size")
    axes[1].plot(
        static_subset["block_size"],
        static_subset["baseline_rmse"],
        color="#444444",
        linewidth=1.0,
        linestyle="--",
        label="BestStatic frontier",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(list(BLOCK_SIZES))
    axes[1].set_xticklabels([str(value) for value in BLOCK_SIZES])
    axes[1].set_xlabel("Shared-feedback block length")
    axes[1].set_ylabel("TDOA RMSE (sample)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    for prefix, color, marker, label in (
        ("BlockPredicted", "#2ca02c", "o", "Block predicted"),
        ("BlockOracle", "#9467bd", "^", "Block oracle"),
    ):
        subset = overall[
            overall["target_method"].str.startswith(prefix)
        ].sort_values("block_size")
        yerr = np.vstack(
            [
                subset["mse_gain"] - subset["mse_gain_ci_low"],
                subset["mse_gain_ci_high"] - subset["mse_gain"],
            ]
        )
        axes[2].errorbar(
            subset["block_size"],
            subset["mse_gain"],
            yerr=yerr,
            color=color,
            marker=marker,
            linewidth=1.2,
            capsize=2.5,
            label=label,
        )
    axes[2].axhline(0.0, color="#222222", linewidth=1.0)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(list(BLOCK_SIZES))
    axes[2].set_xticklabels([str(value) for value in BLOCK_SIZES])
    axes[2].set_xlabel("Shared-feedback block length")
    axes[2].set_ylabel("MSE gain vs BestStatic (sample$^2$)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "Fig1_Stage2A1_Protocol_Feasibility.svg")
    plt.close(fig)


def run(source: Path, output: Path, *, repetitions: int) -> dict[str, object]:
    if repetitions < 200:
        raise ValueError("bootstrap repetitions must be at least 200")
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = source_hashes(source)
    trials = pd.read_csv(source / "stage2a1_trial_results.csv")
    actions = pd.read_csv(source / "stage2a1_action_rows.csv")
    validate_source_tables(trials, actions)
    fixed_max_diff = assert_fixed_matches_b24(trials)

    payload_methods = [
        "BestStatic-B16",
        "BestStatic-B24",
        "BestStatic-B32",
        "SNR-only-B16",
        "SNR-only-B24",
        "SNR-only-B32",
        "FixedAction-B29",
        "PredictedAction-B29",
        "OracleAction-B29",
    ]
    payload = method_summary(trials, payload_methods)
    effects, block_rows, alignment = _all_effects(
        trials,
        actions,
        repetitions=repetitions,
    )
    gaps = action_gap_table(trials, repetitions=repetitions)
    decision = protocol_decision(effects)

    block_summary_rows = []
    for (method, block_size), group in block_rows.groupby(
        ["method", "block_size_requested"],
        sort=True,
    ):
        block_summary_rows.append(
            {
                "method": method,
                "block_size_requested": int(block_size),
                "n_rows": int(len(group)),
                "n_blocks": int(
                    group[["seed", "snr_db", "block_index"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "mean_total_bits": float(group["total_bits"].mean()),
                "mse_samples2": float(
                    group["squared_error_samples2"].mean()
                ),
            }
        )
    block_summary = pd.DataFrame(block_summary_rows)
    block_summary["rmse_samples"] = np.sqrt(block_summary["mse_samples2"])

    payload.to_csv(output / "payload_frontier.csv", index=False)
    gaps.to_csv(output / "action_selection_decomposition.csv", index=False)
    block_summary.to_csv(output / "block_protocol_summary.csv", index=False)
    effects[effects["realizable"]].to_csv(
        output / "block_protocol_effects.csv",
        index=False,
    )
    effects[~effects["realizable"]].to_csv(
        output / "ideal_amortization_upper_bound.csv",
        index=False,
    )
    _plot(output, payload, effects)

    after = source_hashes(source)
    if before != after:
        raise RuntimeError("frozen Stage 2A.1 source files changed during analysis")
    decision.update(
        {
            "source": str(source),
            "source_sha256_before": before,
            "source_sha256_after": after,
            "source_immutable": True,
            "fixed_action_matches_beststatic_b24_max_sq_error_diff": (
                fixed_max_diff
            ),
            "l1_alignment": alignment,
            "block_sizes": list(BLOCK_SIZES),
            "feedback_accounting": (
                "One 5-bit action index is shared by all observations in a "
                "block. Per-sample action choices are not treated as "
                "deployable amortized policies."
            ),
            "static_frontier": (
                "Expected randomized time sharing between frozen 24-bit and "
                "32-bit static plans at exactly the same mean total rate."
            ),
            "bootstrap_repetitions": int(repetitions),
        }
    )
    json_dump(output / "protocol_decision.json", decision)
    json_dump(
        output / "manifest.json",
        {
            "experiment": "BRSR Stage 2A.1 read-only protocol certificate",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(source),
            "source_sha256": before,
            "input_files": list(SOURCE_FILES),
            "block_sizes": list(BLOCK_SIZES),
            "bootstrap_repetitions": int(repetitions),
            "uses_new_signal_generation": False,
            "fits_or_changes_thresholds": False,
            "changes_estimator_or_candidate_pool": False,
        },
    )
    return decision


def main() -> None:
    args = parse_args()
    output = output_dir()
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    original_stdout = sys.stdout
    log_handle = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_handle)
    started = time.perf_counter()
    try:
        print("BRSR Stage 2A.1 read-only protocol certificate", flush=True)
        print(f"Source: {args.source.resolve()}", flush=True)
        print(f"Output: {output}", flush=True)
        print(f"Block sizes: {list(BLOCK_SIZES)}", flush=True)
        print(
            f"Bootstrap repetitions: {args.bootstrap_repetitions}",
            flush=True,
        )
        decision = run(
            args.source,
            output,
            repetitions=args.bootstrap_repetitions,
        )
        print("\nProtocol decision", flush=True)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
        print(f"[Done] elapsed={time.perf_counter() - started:.1f}s", flush=True)
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
