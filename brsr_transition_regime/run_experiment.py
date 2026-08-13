"""运行BRSR模糊度转折区与频谱异质性独立可行性证书。

PyCharm直接运行默认执行development；Codex仅执行 ``--mode smoke``。
本脚本不训练模型、不读取locked结果，也不修改既有BRSR路线。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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

from brsr_stage2_rate_adaptive.run_stage2a import (  # noqa: E402
    QUANTIZER_BITS,
    QUANTIZER_CLIP_ABS,
)
from brsr_stage2_rate_adaptive.run_stage2a1 import (  # noqa: E402
    _build_frozen_model,
    _source_seed_set,
)
from brsr_stage2_rate_adaptive.stage2a import (  # noqa: E402
    rate_match_assignments,
)
from brsr_state_oracle_certificate.certificate import (  # noqa: E402
    PAYLOAD_24_BITS,
    PAYLOAD_32_BITS,
    SCENE_S1,
    SCENE_S2,
    configuration_key,
    configuration_positions,
    evaluate_configurations,
    freeze_policies,
    generate_packet_trial,
    generate_state_bank,
    lookup_configuration,
    state_rows,
)
from brsr_state_oracle_certificate.run_certificate import (  # noqa: E402
    Tee,
    _base_metadata,
    _load_frozen_source_line_ending_safe,
    _observation,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer  # noqa: E402
from brsr_tdoa.brsr_waveform import WaveformConfig  # noqa: E402
from brsr_transition_regime.experiment import (  # noqa: E402
    ACTION_FEEDBACK_BITS,
    BLOCK_LENGTH,
    CONTEXT_INDEX_BITS,
    CONTROL_SNR_DB,
    FLAT_BASELINE,
    FLAT_TARGET_METHOD,
    PRIMARY_SNR_DB,
    TARGET_BASELINE,
    TARGET_METHOD,
    crossfit_transition_headroom,
    expected_rate_matched_payload,
    gate_decision,
    heterogeneity_attribution,
    paired_policy_effect,
    target_total_bits,
)

PYCHARM_RUN_MODE = "development"
SCENES = (SCENE_S1, SCENE_S2)


@dataclass(frozen=True)
class TransitionConfig:
    mode: str
    snr_db: tuple[float, ...]
    primary_snr_db: tuple[float, ...]
    control_snr_db: tuple[float, ...]
    state_seeds: tuple[int, ...]
    states_per_seed: int
    planning_packets_per_state: int
    evaluation_packets_per_state: int
    bootstrap_repetitions: int
    progress_every: int

    def validate(self, frozen_source_seeds: set[int]) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if tuple(sorted((*self.primary_snr_db, *self.control_snr_db))) != tuple(
            sorted(self.snr_db)
        ):
            raise ValueError("primary/control SNRs must partition snr_db")
        if set(self.primary_snr_db) & set(self.control_snr_db):
            raise ValueError("primary and control SNRs must be disjoint")
        if not set(self.primary_snr_db).issubset(PRIMARY_SNR_DB):
            raise ValueError("primary SNRs exceed the pre-registered scope")
        if not set(self.control_snr_db).issubset(CONTROL_SNR_DB):
            raise ValueError("control SNRs exceed the pre-registered scope")
        if len(set(self.state_seeds)) != len(self.state_seeds) or not self.state_seeds:
            raise ValueError("state seeds must be non-empty and unique")
        if set(self.state_seeds) & frozen_source_seeds:
            raise ValueError("state seeds overlap frozen source experiments")
        counts = (
            self.states_per_seed,
            self.planning_packets_per_state,
            self.evaluation_packets_per_state,
            self.bootstrap_repetitions,
            self.progress_every,
        )
        if min(counts) < 1 or self.bootstrap_repetitions < 20:
            raise ValueError("invalid experiment count")
        if self.planning_packets_per_state < 2 or self.planning_packets_per_state % 2:
            raise ValueError("planning packet count must be an even number >= 2")
        if self.evaluation_packets_per_state != BLOCK_LENGTH and self.mode != "smoke":
            raise ValueError("development evaluation count must equal block length")


CONFIGS = {
    "smoke": TransitionConfig(
        mode="smoke",
        snr_db=(-10.0, 0.0, 5.0, 15.0),
        primary_snr_db=(0.0, 5.0),
        control_snr_db=(-10.0, 15.0),
        state_seeds=(2026090101,),
        states_per_seed=3,
        planning_packets_per_state=4,
        evaluation_packets_per_state=4,
        bootstrap_repetitions=100,
        progress_every=8,
    ),
    "development": TransitionConfig(
        mode="development",
        snr_db=(-10.0, 0.0, 5.0, 15.0),
        primary_snr_db=(0.0, 5.0),
        control_snr_db=(-10.0, 15.0),
        state_seeds=(2026090111, 2026090112, 2026090113, 2026090114),
        states_per_seed=6,
        planning_packets_per_state=128,
        evaluation_packets_per_state=16,
        bootstrap_repetitions=5000,
        progress_every=128,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=tuple(CONFIGS),
        default=PYCHARM_RUN_MODE,
        help="PyCharm默认执行正式development；smoke仅验证工程链路。",
    )
    return parser.parse_args()


def _output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "运行结果" / "BRSR" / f"BRSR_TRANSITION_{timestamp}_{mode}"


def _json_dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "brsr_transition_regime" / "experiment.py",
        ROOT / "brsr_transition_regime" / "run_experiment.py",
        ROOT / "brsr_transition_regime" / "README.md",
        ROOT / "brsr_state_oracle_certificate" / "certificate.py",
        ROOT / "brsr_state_oracle_certificate" / "run_certificate.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _collect_planning(
    *,
    config: TransitionConfig,
    states: list[object],
    cases: dict[str, object],
    bins: np.ndarray,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    temperature_by_snr: dict[float, float],
    output: Path,
) -> tuple[pd.DataFrame, object]:
    rows: list[dict[str, object]] = []
    total = (
        len(SCENES)
        * len(config.snr_db)
        * len(states)
        * config.planning_packets_per_state
    )
    done = 0
    started = time.perf_counter()
    reference_model = None
    for scene in SCENES:
        for snr_db in config.snr_db:
            print(f"[planning] {scene} SNR={snr_db:g} dB start", flush=True)
            for state in states:
                for packet_index in range(config.planning_packets_per_state):
                    trial, observation, model, scale_delta = _observation(
                        scene=scene,
                        stage="planning",
                        state=state,
                        packet_index=packet_index,
                        snr_db=snr_db,
                        cases=cases,
                        bins=bins,
                        waveform=waveform,
                        quantizer=quantizer,
                    )
                    reference_model = model
                    configurations = (
                        *configuration_positions(model, PAYLOAD_24_BITS),
                        *configuration_positions(model, PAYLOAD_32_BITS),
                    )
                    metadata = _base_metadata(
                        scene=scene,
                        stage="planning",
                        state=state,
                        packet_index=packet_index,
                        snr_db=snr_db,
                        trial=trial,
                        observation=observation,
                        scale_delta=scale_delta,
                    )
                    for result in evaluate_configurations(
                        observation,
                        model=model,
                        quantizer=quantizer,
                        configurations=configurations,
                        temperature=temperature_by_snr[float(snr_db)],
                    ):
                        rows.append(
                            {
                                **metadata,
                                **{
                                    key: value
                                    for key, value in result.items()
                                    if key != "positions"
                                },
                                "bin_indices": " ".join(
                                    str(int(model.bin_indices[position]))
                                    for position in result["positions"]
                                ),
                            }
                        )
                    done += 1
                    if done % config.progress_every == 0 or done == total:
                        elapsed = time.perf_counter() - started
                        rate = elapsed / done
                        print(
                            f"[planning] {done}/{total} elapsed={elapsed:.1f}s "
                            f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                            flush=True,
                        )
            pd.DataFrame(rows).to_csv(
                output / "planning_configuration_losses.partial.csv",
                index=False,
            )
    if reference_model is None:
        raise RuntimeError("planning produced no model")
    return pd.DataFrame(rows), reference_model


def _rate_maps(
    *,
    config: TransitionConfig,
    states: list[object],
) -> dict[tuple[str, float, str], dict[str, int]]:
    maps: dict[tuple[str, float, str], dict[str, int]] = {}
    for scene in SCENES:
        families = ("BestStatic", "SNR-only")
        if scene == SCENE_S2:
            families = (*families, "ContextStatic")
        for snr_db in config.snr_db:
            sample_ids = [
                f"{scene}|{state.state_id}|evaluation|{packet}|{snr_db:g}"
                for state in states
                for packet in range(config.evaluation_packets_per_state)
            ]
            for family in families:
                maps[(scene, float(snr_db), family)] = rate_match_assignments(
                    sample_ids,
                    desired_mean_bits=expected_rate_matched_payload(
                        scene=scene,
                        family=family,
                    ),
                    low_bits=PAYLOAD_24_BITS,
                    high_bits=PAYLOAD_32_BITS,
                    salt=f"transition|{scene}|{snr_db:g}|{family}",
                )
    return maps


def _selected_methods(
    *,
    scene: str,
    snr_db: float,
    state_id: str,
    context_class: str,
    sample_id: str,
    policies: pd.DataFrame,
    rate_maps: dict[tuple[str, float, str], dict[str, int]],
) -> dict[str, tuple[int, ...]]:
    def select(family: str, payload_bits: int) -> tuple[int, ...]:
        return lookup_configuration(
            policies,
            scene=scene,
            policy=family,
            snr_db=snr_db,
            state_id=state_id,
            context_class=context_class,
            payload_bits=payload_bits,
        )

    selected = {
        "BestStatic-B24": select("BestStatic", PAYLOAD_24_BITS),
        "SNR-only-B24": select("SNR-only", PAYLOAD_24_BITS),
        TARGET_METHOD: select("StateOracle", PAYLOAD_24_BITS),
    }
    for family in ("BestStatic", "SNR-only"):
        bits = rate_maps[(scene, float(snr_db), family)][sample_id]
        selected[f"{family}-RateMatched-L16"] = select(family, bits)
    if scene == SCENE_S2:
        selected["ContextStatic-B24"] = select("ContextStatic", PAYLOAD_24_BITS)
        bits = rate_maps[(scene, float(snr_db), "ContextStatic")][sample_id]
        selected[TARGET_BASELINE] = select("ContextStatic", bits)
    return selected


def _method_overheads(scene: str, method: str) -> tuple[float, float]:
    context_bits = 0.0
    feedback_bits = 0.0
    if scene == SCENE_S2 and (
        method.startswith("ContextStatic") or method == TARGET_METHOD
    ):
        context_bits = CONTEXT_INDEX_BITS / BLOCK_LENGTH
    if method == TARGET_METHOD:
        feedback_bits = ACTION_FEEDBACK_BITS / BLOCK_LENGTH
    return context_bits, feedback_bits


def _collect_evaluation(
    *,
    config: TransitionConfig,
    states: list[object],
    cases: dict[str, object],
    bins: np.ndarray,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    temperature_by_snr: dict[float, float],
    policies: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rate_maps = _rate_maps(config=config, states=states)
    total = (
        len(SCENES)
        * len(config.snr_db)
        * len(states)
        * config.evaluation_packets_per_state
    )
    done = 0
    started = time.perf_counter()
    for scene in SCENES:
        for snr_db in config.snr_db:
            print(f"[evaluation] {scene} SNR={snr_db:g} dB start", flush=True)
            for state in states:
                for packet_index in range(config.evaluation_packets_per_state):
                    trial, observation, model, scale_delta = _observation(
                        scene=scene,
                        stage="evaluation",
                        state=state,
                        packet_index=packet_index,
                        snr_db=snr_db,
                        cases=cases,
                        bins=bins,
                        waveform=waveform,
                        quantizer=quantizer,
                    )
                    metadata = _base_metadata(
                        scene=scene,
                        stage="evaluation",
                        state=state,
                        packet_index=packet_index,
                        snr_db=snr_db,
                        trial=trial,
                        observation=observation,
                        scale_delta=scale_delta,
                    )
                    sample_id = str(metadata["sample_id"])
                    selected = _selected_methods(
                        scene=scene,
                        snr_db=snr_db,
                        state_id=str(state.state_id),
                        context_class=str(metadata["context_class"]),
                        sample_id=sample_id,
                        policies=policies,
                        rate_maps=rate_maps,
                    )
                    unique = tuple(sorted(set(selected.values())))
                    evaluated = {
                        configuration_key(result["positions"]): result
                        for result in evaluate_configurations(
                            observation,
                            model=model,
                            quantizer=quantizer,
                            configurations=unique,
                            temperature=temperature_by_snr[float(snr_db)],
                        )
                    }
                    for method, positions in selected.items():
                        result = evaluated[configuration_key(positions)]
                        payload_bits = int(result["payload_bits"])
                        context_bits, feedback_bits = _method_overheads(scene, method)
                        rows.append(
                            {
                                **metadata,
                                "method": method,
                                "configuration": configuration_key(positions),
                                "bin_indices": " ".join(
                                    str(int(model.bin_indices[position]))
                                    for position in positions
                                ),
                                "estimate_tau": float(result["estimate_tau"]),
                                "error_samples": float(result["error_samples"]),
                                "squared_error_samples2": float(
                                    result["squared_error_samples2"]
                                ),
                                "payload_bits": payload_bits,
                                "context_bits": context_bits,
                                "feedback_bits": feedback_bits,
                                "total_bits": payload_bits + context_bits + feedback_bits,
                            }
                        )
                    done += 1
                    if done % config.progress_every == 0 or done == total:
                        elapsed = time.perf_counter() - started
                        rate = elapsed / done
                        print(
                            f"[evaluation] {done}/{total} elapsed={elapsed:.1f}s "
                            f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                            flush=True,
                        )
            pd.DataFrame(rows).to_csv(
                output / "evaluation_results.partial.csv",
                index=False,
            )
    return pd.DataFrame(rows)


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scene, method, snr_db), group in results.groupby(
        ["scene", "method", "snr_db"], sort=True
    ):
        rows.append(
            {
                "scene": scene,
                "method": method,
                "snr_db": float(snr_db),
                "n_rows": int(len(group)),
                "rmse_samples": math.sqrt(
                    float(group["squared_error_samples2"].mean())
                ),
                "median_abs_error_samples": float(group["error_samples"].abs().median()),
                "outlier_gt_2_samples": float((group["error_samples"].abs() > 2.0).mean()),
                "mean_payload_bits": float(group["payload_bits"].mean()),
                "mean_context_bits": float(group["context_bits"].mean()),
                "mean_feedback_bits": float(group["feedback_bits"].mean()),
                "mean_total_bits": float(group["total_bits"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _effect_rows(config: TransitionConfig, results: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("primary_target", SCENE_S2, TARGET_METHOD, TARGET_BASELINE, config.primary_snr_db),
        ("primary_flat_control", SCENE_S1, FLAT_TARGET_METHOD, FLAT_BASELINE, config.primary_snr_db),
    ]
    for scene, baseline in ((SCENE_S1, FLAT_BASELINE), (SCENE_S2, TARGET_BASELINE)):
        for snr_db in config.control_snr_db:
            specifications.append(
                (f"control_{scene}_{snr_db:g}", scene, TARGET_METHOD, baseline, (snr_db,))
            )
    rows = []
    for index, (scope, scene, target, baseline, snr_values) in enumerate(specifications):
        rows.append(
            {
                "scope": scope,
                **paired_policy_effect(
                    results,
                    scene=scene,
                    target=target,
                    baseline=baseline,
                    snr_db=snr_values,
                    bootstrap_repetitions=config.bootstrap_repetitions,
                    bootstrap_seed=2026090901 + index,
                ),
            }
        )
    return pd.DataFrame(rows)


def _mechanical_checks(
    *,
    config: TransitionConfig,
    states: list[object],
    planning: pd.DataFrame,
    evaluation: pd.DataFrame,
    frozen_source_seeds: set[int],
    waveform: WaveformConfig,
) -> dict[str, bool]:
    plan_ids = set(planning["sample_id"].astype(str))
    eval_ids = set(evaluation["sample_id"].astype(str))
    budget_ok = True
    for scene in SCENES:
        families = ("BestStatic", "SNR-only")
        if scene == SCENE_S2:
            families = (*families, "ContextStatic")
        for snr_db in config.snr_db:
            target = evaluation[
                evaluation["scene"].eq(scene)
                & evaluation["snr_db"].eq(float(snr_db))
                & evaluation["method"].eq(TARGET_METHOD)
            ]
            if not math.isclose(
                float(target["total_bits"].mean()),
                target_total_bits(scene),
                abs_tol=1e-12,
            ):
                budget_ok = False
            for family in families:
                method = f"{family}-RateMatched-L16"
                rows = evaluation[
                    evaluation["scene"].eq(scene)
                    & evaluation["snr_db"].eq(float(snr_db))
                    & evaluation["method"].eq(method)
                ]
                tolerance = 4.0 / max(len(rows), 1) + 1e-12
                if (
                    abs(float(rows["total_bits"].mean()) - target_total_bits(scene))
                    > tolerance
                ):
                    budget_ok = False

    shared_action = evaluation[evaluation["method"].eq(TARGET_METHOD)].groupby(
        ["scene", "state_id", "snr_db"], sort=False
    )["configuration"].nunique()
    first_state = states[0]
    plan_trial = generate_packet_trial(
        first_state,
        stage="planning",
        packet_index=0,
        scene=SCENE_S1,
        waveform=waveform,
    )
    eval_trial = generate_packet_trial(
        first_state,
        stage="evaluation",
        packet_index=0,
        scene=SCENE_S1,
        waveform=waveform,
    )
    streams_separate = not (
        np.array_equal(plan_trial.source_a, eval_trial.source_a)
        or np.array_equal(plan_trial.noise_a_standard, eval_trial.noise_a_standard)
        or np.array_equal(plan_trial.noise_b_standard, eval_trial.noise_b_standard)
    )
    s2_contexts = set(
        evaluation[evaluation["scene"].eq(SCENE_S2)]["context_class"].astype(str)
    )
    context_overhead_ok = bool(
        (
            evaluation[
                evaluation["scene"].eq(SCENE_S2)
                & (
                    evaluation["method"].str.startswith("ContextStatic")
                    | evaluation["method"].eq(TARGET_METHOD)
                )
            ]["context_bits"]
            == CONTEXT_INDEX_BITS / BLOCK_LENGTH
        ).all()
        and (
            evaluation[
                evaluation["scene"].eq(SCENE_S1)
                | ~(
                    evaluation["method"].str.startswith("ContextStatic")
                    | evaluation["method"].eq(TARGET_METHOD)
                )
            ]["context_bits"]
            == 0.0
        ).all()
    )
    return {
        "new_state_seeds_disjoint_from_frozen_sources": not (
            set(config.state_seeds) & frozen_source_seeds
        ),
        "primary_and_control_snr_disjoint": not (
            set(config.primary_snr_db) & set(config.control_snr_db)
        ),
        "planning_and_evaluation_ids_disjoint": not (plan_ids & eval_ids),
        "planning_and_evaluation_random_streams_disjoint": streams_separate,
        "only_flat_and_notch_scenes_used": set(evaluation["scene"]) == set(SCENES),
        "all_rate_matched_total_bits_within_integer_quantum": budget_ok,
        "state_action_shared_within_block": bool((shared_action == 1).all()),
        "state_oracle_uses_only_b24_payload": set(
            evaluation[evaluation["method"].eq(TARGET_METHOD)]["payload_bits"]
        )
        == {PAYLOAD_24_BITS},
        "context_index_cost_counted_only_where_available": context_overhead_ok,
        "all_three_notch_contexts_present": len(s2_contexts) == 3,
        "no_quantizer_clipping": bool((planning["clipping_rate"] == 0.0).all())
        and bool((evaluation["clipping_rate"] == 0.0).all()),
        "public_scaling_unchanged": bool(
            np.allclose(planning["public_scale_delta_vs_legacy"], 0.0, atol=0.0)
        )
        and bool(
            np.allclose(evaluation["public_scale_delta_vs_legacy"], 0.0, atol=0.0)
        ),
        "no_outcome_oracle_generated": not evaluation["method"].str.contains(
            "OutcomeOracle", regex=False
        ).any(),
        "no_locked_data_used": True,
    }


def _plot(summary: pd.DataFrame, effects: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    colors = {
        TARGET_METHOD: "#0072B2",
        TARGET_BASELINE: "#D55E00",
        "SNR-only-RateMatched-L16": "#009E73",
    }
    target_summary = summary[summary["scene"].eq(SCENE_S2)]
    for method in (TARGET_METHOD, TARGET_BASELINE, "SNR-only-RateMatched-L16"):
        rows = target_summary[target_summary["method"].eq(method)].sort_values("snr_db")
        if rows.empty:
            continue
        axes[0].plot(
            rows["snr_db"],
            rows["rmse_samples"],
            marker="o",
            linewidth=1.5,
            markersize=4.5,
            color=colors[method],
            label=method,
        )
    axes[0].axvspan(-0.5, 5.5, color="#E69F00", alpha=0.10, label="Primary 0/5 dB")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("TDOA RMSE (samples)")
    axes[0].set_title("Target scene: block-coherent spectral notch")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    control = effects[effects["scope"].str.startswith("control_")].copy()
    control["label"] = control["scene"].map(
        {SCENE_S1: "Flat control", SCENE_S2: "Notch target"}
    )
    for label, rows in control.groupby("label", sort=True):
        axes[1].plot(
            rows["snr_scope"].astype(float),
            rows["mse_gain"],
            marker="o",
            linewidth=1.5,
            markersize=4.5,
            label=label,
        )
    primary = effects[effects["scope"].isin(["primary_target", "primary_flat_control"])]
    for _, row in primary.iterrows():
        x = 2.5
        label = "Primary notch" if row["scope"] == "primary_target" else "Primary flat"
        axes[1].scatter([x], [row["mse_gain"]], s=45, label=label)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xlabel("SNR (dB); primary aggregate shown at 2.5 dB")
    axes[1].set_ylabel("Paired MSE gain (samples²)")
    axes[1].set_title("Scoped gains and negative controls")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(output / "Fig1_Transition_Regime_Certificate.svg", format="svg")
    plt.close(fig)


def _cleanup_partials(output: Path) -> None:
    for path in output.glob("*.partial.csv"):
        path.unlink()


def _run(config: TransitionConfig, output: Path) -> None:
    started = time.perf_counter()
    manifest, _, temperature_by_snr, frozen_hash_audit = (
        _load_frozen_source_line_ending_safe()
    )
    frozen_source_seeds = _source_seed_set(manifest)
    config.validate(frozen_source_seeds)
    missing_temperature = set(config.snr_db) - set(temperature_by_snr)
    if missing_temperature:
        raise RuntimeError(
            f"frozen temperature is missing SNR values: {sorted(missing_temperature)}"
        )

    waveform = WaveformConfig()
    waveform.validate()
    cases, bins, prior_statistics = _build_frozen_model(manifest, waveform)
    quantizer = NestedComplexQuantizer(
        max_bits=QUANTIZER_BITS,
        clip_abs=QUANTIZER_CLIP_ABS,
    )
    states = generate_state_bank(
        config.state_seeds,
        config.states_per_seed,
        waveform=waveform,
        candidate_bins=bins,
    )
    state_rows(states).to_csv(output / "state_definitions.csv", index=False)
    prior_statistics.to_csv(output / "prior_statistics.csv", index=False)

    planning, model = _collect_planning(
        config=config,
        states=states,
        cases=cases,
        bins=bins,
        waveform=waveform,
        quantizer=quantizer,
        temperature_by_snr=temperature_by_snr,
        output=output,
    )
    planning.to_csv(output / "planning_configuration_losses.csv", index=False)
    crossfit_rows, crossfit = crossfit_transition_headroom(
        planning,
        primary_snr_db=config.primary_snr_db,
        bootstrap_repetitions=config.bootstrap_repetitions,
        bootstrap_seed=2026090801,
    )
    crossfit_rows.to_csv(output / "crossfit_transition_headroom.csv", index=False)
    pd.DataFrame([crossfit]).to_csv(output / "crossfit_summary.csv", index=False)

    policies = freeze_policies(planning)
    policies["bin_indices"] = policies["configuration"].map(
        lambda value: " ".join(
            str(int(model.bin_indices[int(position)]))
            for position in str(value).split()
        )
    )
    policies.to_csv(output / "frozen_policies.csv", index=False)

    evaluation = _collect_evaluation(
        config=config,
        states=states,
        cases=cases,
        bins=bins,
        waveform=waveform,
        quantizer=quantizer,
        temperature_by_snr=temperature_by_snr,
        policies=policies,
        output=output,
    )
    evaluation.to_csv(output / "evaluation_results.csv", index=False)
    summary = _summarize(evaluation)
    summary.to_csv(output / "policy_summary.csv", index=False)
    effects = _effect_rows(config, evaluation)
    effects.to_csv(output / "paired_effects.csv", index=False)
    attribution_rows, attribution = heterogeneity_attribution(
        evaluation,
        primary_snr_db=config.primary_snr_db,
        bootstrap_repetitions=config.bootstrap_repetitions,
        bootstrap_seed=2026090909,
    )
    attribution_rows.to_csv(output / "heterogeneity_attribution.csv", index=False)
    pd.DataFrame([attribution]).to_csv(output / "attribution_summary.csv", index=False)

    primary_effect = effects[effects["scope"].eq("primary_target")]
    if len(primary_effect) != 1:
        raise RuntimeError("primary target effect is missing or duplicated")
    mechanical = _mechanical_checks(
        config=config,
        states=states,
        planning=planning,
        evaluation=evaluation,
        frozen_source_seeds=frozen_source_seeds,
        waveform=waveform,
    )
    decision = gate_decision(
        mechanical_checks=mechanical,
        crossfit=crossfit,
        primary_effect=primary_effect.iloc[0].to_dict(),
        attribution=attribution,
        smoke=config.mode == "smoke",
    )
    _json_dump(output / "gate_decision.json", decision)
    _plot(summary, effects, output)

    manifest_out = {
        "mode": config.mode,
        "status": decision["status"],
        "config": asdict(config),
        "pre_registered_hypothesis": (
            "In the 0/5 dB ambiguity-transition regime, block-coherent spectral "
            "heterogeneity creates repeatable state-conditioned B24 action value "
            "beyond an equal-information, rate-matched ContextStatic baseline."
        ),
        "scope_boundary": {
            "primary_snr_db": list(config.primary_snr_db),
            "control_snr_db": list(config.control_snr_db),
            "flat_scene_is_negative_control": True,
            "broad_snr_brsr_no_go_remains_frozen": True,
        },
        "communication_accounting": {
            "base_and_acquired_payload_bits": PAYLOAD_24_BITS,
            "context_index_bits_per_block": CONTEXT_INDEX_BITS,
            "action_feedback_bits_per_block": ACTION_FEEDBACK_BITS,
            "block_length": BLOCK_LENGTH,
            "target_total_bits_flat": target_total_bits(SCENE_S1),
            "target_total_bits_notch": target_total_bits(SCENE_S2),
            "rate_matching_includes_context_and_feedback": True,
        },
        "information_boundary": {
            "state_oracle_uses_planning_only": True,
            "planning_crossfit_before_evaluation": True,
            "context_static_receives_same_notch_class": True,
            "evaluation_outcomes_used_for_action_selection": False,
            "outcome_oracle_used": False,
            "locked_data_used": False,
        },
        "mechanical_checks": mechanical,
        "source_sha256": _source_hashes(),
        "frozen_source_hash_audit": frozen_hash_audit,
        "candidate_bins": [int(value) for value in bins],
        "temperature_by_snr": {
            str(value): float(temperature_by_snr[float(value)])
            for value in config.snr_db
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _json_dump(output / "manifest.json", manifest_out)
    _cleanup_partials(output)

    print("\nBRSR transition-regime certificate", flush=True)
    print(
        f"  crossfit gain={float(crossfit['mse_gain']):.6f} "
        f"CI=[{float(crossfit['mse_gain_ci_low']):.6f}, "
        f"{float(crossfit['mse_gain_ci_high']):.6f}]",
        flush=True,
    )
    row = primary_effect.iloc[0]
    print(
        f"  primary gain={float(row['mse_gain']):.6f} "
        f"CI=[{float(row['mse_gain_ci_low']):.6f}, "
        f"{float(row['mse_gain_ci_high']):.6f}]",
        flush=True,
    )
    print(
        f"  attribution={float(attribution['mean']):.6f} "
        f"CI=[{float(attribution['ci_low']):.6f}, "
        f"{float(attribution['ci_high']):.6f}]",
        flush=True,
    )
    print(f"[Gate] {decision['status']}", flush=True)
    print(f"[Done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    args = _parse_args()
    config = CONFIGS[args.mode]
    output = _output_dir(config.mode)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Mode: {config.mode}", flush=True)
    print(f"Output: {output}", flush=True)
    print(f"Primary SNR: {list(config.primary_snr_db)}", flush=True)
    print(f"Control SNR: {list(config.control_snr_db)}", flush=True)
    print(f"Config: {json.dumps(asdict(config), ensure_ascii=False)}", flush=True)
    with (output / "run.log").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            _run(config, output)


if __name__ == "__main__":
    main()
