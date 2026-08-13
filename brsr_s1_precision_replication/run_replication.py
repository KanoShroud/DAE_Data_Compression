"""Run the pre-registered S1-only StateOracle precision replication.

PyCharm runs ``development`` by default. The experiment first selects the
smallest converged planning packet count on calibration-only states. It then
uses disjoint confirmation states to test the frozen S1/L16 hypothesis.
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

from brsr_route_feasibility.run_stage1 import Tee  # noqa: E402
from brsr_stage2_rate_adaptive.run_stage2a import (  # noqa: E402
    PRIMARY_LIKELIHOOD,
    PRIMARY_POOL,
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
    ACTION_FEEDBACK_BITS,
    PAYLOAD_24_BITS,
    PAYLOAD_32_BITS,
    SCENE_S1,
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
    _base_metadata,
    _load_frozen_source_line_ending_safe,
    _observation,
)
from brsr_s1_precision_replication.replication import (  # noqa: E402
    BASELINES,
    TARGET_METHOD,
    gate_decision,
    paired_effects,
    planning_convergence_diagnostics,
    summarize_results,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer  # noqa: E402
from brsr_tdoa.brsr_waveform import WaveformConfig  # noqa: E402


PYCHARM_RUN_MODE = "development"
SOURCE_CERTIFICATE = (
    ROOT
    / "运行结果"
    / "BRSR"
    / "BRSR_STATE_ORACLE_20260731_170154_development"
)
SOURCE_RESULT_SHA256 = {
    "manifest.json": "c42288c35675ff2294981d0ef78df18b11b51b475bc2a38243668587287c4023",
    "gate_decision.json": "3ade94a6d68edf53684dae6c701be29a4d6c35d86a035c6f86fe22c15bc6b0b6",
    "paired_effects.csv": "8db473b798745e7da5db082f084bfb9b154ad28adb554c83c9be008bc6d1987b",
}
HISTORICAL_STATE_SEEDS = {
    2026080101,
    2026080111,
    2026080112,
    2026080113,
}
HISTORICAL_MIN_S1_L16_MSE_GAIN = 0.30455079267987967
HISTORICAL_MAX_S1_L16_CLUSTER_SD = 0.6935536535200854
PLANNING_EQUIVALENCE_FRACTION = 0.10
PLANNING_SENSITIVITY_FRACTIONS = (0.05, 0.10, 0.20)
BLOCK_LENGTH = 16


@dataclass(frozen=True)
class ReplicationConfig:
    mode: str
    snr_db: tuple[float, ...]
    calibration_state_seeds: tuple[int, ...]
    calibration_states_per_seed: int
    planning_candidate_counts: tuple[int, ...]
    planning_reference_count: int
    planning_check_count: int
    confirmation_state_seeds: tuple[int, ...]
    confirmation_states_per_seed: int
    evaluation_packets_per_state: int
    bootstrap_repetitions: int
    progress_every: int

    def validate(self, frozen_source_seeds: set[int]) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.snr_db or len(set(self.snr_db)) != len(self.snr_db):
            raise ValueError("snr_db must be non-empty and unique")
        calibration = set(self.calibration_state_seeds)
        confirmation = set(self.confirmation_state_seeds)
        if (
            not calibration
            or not confirmation
            or len(calibration) != len(self.calibration_state_seeds)
            or len(confirmation) != len(self.confirmation_state_seeds)
        ):
            raise ValueError("state seed sets must be non-empty and unique")
        forbidden = frozen_source_seeds | HISTORICAL_STATE_SEEDS
        if calibration & confirmation or (calibration | confirmation) & forbidden:
            raise ValueError("calibration/confirmation seeds are not independent")
        counts = (
            self.calibration_states_per_seed,
            self.planning_reference_count,
            self.planning_check_count,
            self.confirmation_states_per_seed,
            self.evaluation_packets_per_state,
            self.bootstrap_repetitions,
            self.progress_every,
        )
        if min(counts) < 1 or self.bootstrap_repetitions < 20:
            raise ValueError("replication count configuration is invalid")
        candidates = self.planning_candidate_counts
        if not candidates or tuple(sorted(set(candidates))) != candidates:
            raise ValueError("planning candidate counts must be increasing")
        if candidates[-1] >= self.planning_reference_count:
            raise ValueError("planning reference must exceed every candidate count")
        if self.evaluation_packets_per_state != BLOCK_LENGTH:
            raise ValueError("formal communication block is frozen at L=16")
        if self.mode == "development":
            frozen_design = (
                self.snr_db == (-10.0, 0.0, 5.0, 15.0)
                and len(calibration) * self.calibration_states_per_seed == 16
                and self.planning_candidate_counts == (32, 64, 128)
                and self.planning_reference_count == 256
                and self.planning_check_count == 128
                and len(confirmation) * self.confirmation_states_per_seed == 48
                and self.bootstrap_repetitions == 5000
            )
            if not frozen_design:
                raise ValueError("formal S1 precision design was modified")


CONFIGS = {
    "smoke": ReplicationConfig(
        mode="smoke",
        snr_db=(-5.0, 10.0),
        calibration_state_seeds=(2026081001,),
        calibration_states_per_seed=1,
        planning_candidate_counts=(2,),
        planning_reference_count=4,
        planning_check_count=2,
        confirmation_state_seeds=(2026081101,),
        confirmation_states_per_seed=1,
        evaluation_packets_per_state=16,
        bootstrap_repetitions=50,
        progress_every=2,
    ),
    "development": ReplicationConfig(
        mode="development",
        snr_db=(-10.0, 0.0, 5.0, 15.0),
        calibration_state_seeds=(
            2026081011,
            2026081012,
            2026081013,
            2026081014,
        ),
        calibration_states_per_seed=4,
        planning_candidate_counts=(32, 64, 128),
        planning_reference_count=256,
        planning_check_count=128,
        confirmation_state_seeds=(
            2026081111,
            2026081112,
            2026081113,
            2026081114,
            2026081115,
            2026081116,
        ),
        confirmation_states_per_seed=8,
        evaluation_packets_per_state=16,
        bootstrap_repetitions=5000,
        progress_every=64,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=tuple(CONFIGS),
        default=PYCHARM_RUN_MODE,
        help="PyCharm default is the formal development profile.",
    )
    return parser.parse_args()


def _output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "运行结果"
        / "BRSR"
        / f"BRSR_S1_PRECISION_{timestamp}_{mode}"
    )


def _json_dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "brsr_s1_precision_replication" / "replication.py",
        ROOT / "brsr_s1_precision_replication" / "run_replication.py",
        ROOT / "brsr_s1_precision_replication" / "README.md",
        ROOT / "brsr_state_oracle_certificate" / "certificate.py",
        ROOT / "brsr_state_oracle_certificate" / "run_certificate.py",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _load_and_audit_sources() -> tuple[
    dict[str, object],
    dict[float, float],
    dict[str, dict[str, object]],
]:
    for name, expected in SOURCE_RESULT_SHA256.items():
        path = SOURCE_CERTIFICATE / name
        if not path.is_file():
            raise FileNotFoundError(f"source StateOracle result is missing: {path}")
        if _sha256(path) != expected:
            raise RuntimeError(f"source StateOracle result changed: {name}")
    source_manifest = json.loads(
        (SOURCE_CERTIFICATE / "manifest.json").read_text(encoding="utf-8")
    )
    source_gate = json.loads(
        (SOURCE_CERTIFICATE / "gate_decision.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("mode") != "development"
        or source_gate.get("status") != "HOLD_POWER_STATE_HEADROOM"
    ):
        raise RuntimeError("source StateOracle certificate is not the frozen formal run")
    source_config = source_manifest.get("config", {})
    if (
        source_config.get("state_seeds")
        != [2026080111, 2026080112, 2026080113]
        or source_config.get("states_per_seed") != 4
        or source_config.get("planning_packets_per_state") != 32
        or source_config.get("evaluation_packets_per_state") != 16
    ):
        raise RuntimeError("source StateOracle design basis is inconsistent")
    source_effects = pd.read_csv(SOURCE_CERTIFICATE / "paired_effects.csv")
    relevant = source_effects[
        source_effects["scene"].eq(SCENE_S1)
        & source_effects["block_length"].eq(BLOCK_LENGTH)
    ]
    rate_matched = relevant[
        relevant["baseline"].isin(
            ("BestStatic-RateMatched-L16", "SNR-only-RateMatched-L16")
        )
    ]
    observed_gain = float(rate_matched["mse_gain"].min())
    observed_sd = float(relevant["cluster_mse_gain_sd"].max())
    if not math.isclose(
        observed_gain,
        HISTORICAL_MIN_S1_L16_MSE_GAIN,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        observed_sd,
        HISTORICAL_MAX_S1_L16_CLUSTER_SD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("historical effect-size design basis changed")

    stage2_manifest, _, temperature_by_snr, frozen_hash_audit = (
        _load_frozen_source_line_ending_safe()
    )
    return stage2_manifest, temperature_by_snr, frozen_hash_audit


def _minimum_state_count() -> int:
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    ratio = (
        (z_alpha + z_power)
        * HISTORICAL_MAX_S1_L16_CLUSTER_SD
        / HISTORICAL_MIN_S1_L16_MSE_GAIN
    )
    return int(math.ceil(ratio**2))


def _collect_configuration_rows(
    *,
    states: list[object],
    stage: str,
    packet_count: int,
    snr_db_values: tuple[float, ...],
    include_b32: bool,
    stream_name: str,
    cases: dict[str, object],
    bins: np.ndarray,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    temperature_by_snr: dict[float, float],
    output: Path,
    partial_name: str,
    progress_every: int,
) -> tuple[pd.DataFrame, object]:
    rows: list[dict[str, object]] = []
    total = len(snr_db_values) * len(states) * packet_count
    done = 0
    started = time.perf_counter()
    reference_model = None
    for snr_db in snr_db_values:
        print(f"[{stream_name}] SNR={snr_db:g} dB start", flush=True)
        for state in states:
            for packet_index in range(packet_count):
                trial, observation, model, scale_delta = _observation(
                    scene=SCENE_S1,
                    stage=stage,
                    state=state,
                    packet_index=packet_index,
                    snr_db=snr_db,
                    cases=cases,
                    bins=bins,
                    waveform=waveform,
                    quantizer=quantizer,
                )
                reference_model = model
                configurations = list(
                    configuration_positions(model, PAYLOAD_24_BITS)
                )
                if include_b32:
                    configurations.extend(
                        configuration_positions(model, PAYLOAD_32_BITS)
                    )
                metadata = _base_metadata(
                    scene=SCENE_S1,
                    stage=stage,
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
                    configurations=tuple(configurations),
                    temperature=temperature_by_snr[float(snr_db)],
                ):
                    rows.append(
                        {
                            **metadata,
                            "stream_name": stream_name,
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
                if done % progress_every == 0 or done == total:
                    elapsed = time.perf_counter() - started
                    rate = elapsed / done
                    print(
                        f"[{stream_name}] {done}/{total} elapsed={elapsed:.1f}s "
                        f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                        flush=True,
                    )
        pd.DataFrame(rows).to_csv(output / partial_name, index=False)
    if reference_model is None:
        raise RuntimeError(f"{stream_name} produced no model")
    return pd.DataFrame(rows), reference_model


def _collect_aggregated_confirmation_planning(
    *,
    states: list[object],
    packet_count: int,
    snr_db_values: tuple[float, ...],
    cases: dict[str, object],
    bins: np.ndarray,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    temperature_by_snr: dict[float, float],
    output: Path,
    progress_every: int,
) -> tuple[pd.DataFrame, object]:
    """Collect sufficient planning statistics without retaining packet rows."""

    accumulators: dict[tuple[object, ...], dict[str, object]] = {}
    total = len(snr_db_values) * len(states) * packet_count
    done = 0
    started = time.perf_counter()
    reference_model = None

    def frame_from_accumulators() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for accumulator in accumulators.values():
            count = int(accumulator["n_planning_packets"])
            rows.append(
                {
                    **{
                        key: value
                        for key, value in accumulator.items()
                        if key != "squared_error_sum"
                    },
                    "squared_error_samples2": (
                        float(accumulator["squared_error_sum"]) / count
                    ),
                }
            )
        return pd.DataFrame(rows)

    for snr_db in snr_db_values:
        print(
            f"[effect-confirmation-planning] SNR={snr_db:g} dB start",
            flush=True,
        )
        for state in states:
            for packet_index in range(packet_count):
                trial, observation, model, scale_delta = _observation(
                    scene=SCENE_S1,
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
                    scene=SCENE_S1,
                    stage="planning",
                    state=state,
                    packet_index=-1,
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
                    key = (
                        int(state.state_seed),
                        int(state.state_index),
                        str(state.state_id),
                        float(snr_db),
                        int(result["payload_bits"]),
                        str(result["configuration"]),
                    )
                    if key not in accumulators:
                        accumulators[key] = {
                            **metadata,
                            "stream_name": "effect-confirmation-planning",
                            "configuration": str(result["configuration"]),
                            "payload_bits": int(result["payload_bits"]),
                            "bin_indices": " ".join(
                                str(int(model.bin_indices[position]))
                                for position in result["positions"]
                            ),
                            "n_planning_packets": 0,
                            "squared_error_sum": 0.0,
                        }
                    accumulators[key]["n_planning_packets"] = (
                        int(accumulators[key]["n_planning_packets"]) + 1
                    )
                    accumulators[key]["squared_error_sum"] = (
                        float(accumulators[key]["squared_error_sum"])
                        + float(result["squared_error_samples2"])
                    )
                done += 1
                if done % progress_every == 0 or done == total:
                    elapsed = time.perf_counter() - started
                    rate = elapsed / done
                    print(
                        "[effect-confirmation-planning] "
                        f"{done}/{total} elapsed={elapsed:.1f}s "
                        f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                        flush=True,
                    )
        frame_from_accumulators().to_csv(
            output / "confirmation_planning.partial.csv",
            index=False,
        )
    if reference_model is None:
        raise RuntimeError("effect confirmation planning produced no model")
    return frame_from_accumulators(), reference_model


def _rate_match_maps(
    *,
    states: list[object],
    snr_db_values: tuple[float, ...],
    evaluation_packets_per_state: int,
) -> dict[float, dict[str, int]]:
    target = PAYLOAD_24_BITS + ACTION_FEEDBACK_BITS / BLOCK_LENGTH
    maps: dict[float, dict[str, int]] = {}
    for snr_db in snr_db_values:
        sample_ids = [
            f"{SCENE_S1}|{state.state_id}|evaluation|{packet}|{snr_db:g}"
            for state in states
            for packet in range(evaluation_packets_per_state)
        ]
        maps[float(snr_db)] = rate_match_assignments(
            sample_ids,
            desired_mean_bits=target,
            low_bits=PAYLOAD_24_BITS,
            high_bits=PAYLOAD_32_BITS,
            salt=f"state-oracle|{SCENE_S1}|L{BLOCK_LENGTH}|{snr_db:g}",
        )
    return maps


def _selected_methods(
    *,
    policies: pd.DataFrame,
    state_id: str,
    snr_db: float,
    sample_id: str,
    rate_maps: dict[float, dict[str, int]],
) -> dict[str, tuple[int, ...]]:
    common = {
        "scene": SCENE_S1,
        "snr_db": snr_db,
        "state_id": state_id,
        "context_class": "",
    }
    rate_bits = rate_maps[float(snr_db)][sample_id]
    return {
        "BestStatic-B24": lookup_configuration(
            policies,
            policy="BestStatic",
            payload_bits=PAYLOAD_24_BITS,
            **common,
        ),
        "SNR-only-B24": lookup_configuration(
            policies,
            policy="SNR-only",
            payload_bits=PAYLOAD_24_BITS,
            **common,
        ),
        TARGET_METHOD: lookup_configuration(
            policies,
            policy="StateOracle",
            payload_bits=PAYLOAD_24_BITS,
            **common,
        ),
        "BestStatic-RateMatched-L16": lookup_configuration(
            policies,
            policy="BestStatic",
            payload_bits=rate_bits,
            **common,
        ),
        "SNR-only-RateMatched-L16": lookup_configuration(
            policies,
            policy="SNR-only",
            payload_bits=rate_bits,
            **common,
        ),
    }


def _collect_confirmation_evaluation(
    *,
    states: list[object],
    config: ReplicationConfig,
    policies: pd.DataFrame,
    cases: dict[str, object],
    bins: np.ndarray,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
    temperature_by_snr: dict[float, float],
    output: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rate_maps = _rate_match_maps(
        states=states,
        snr_db_values=config.snr_db,
        evaluation_packets_per_state=config.evaluation_packets_per_state,
    )
    total = (
        len(config.snr_db)
        * len(states)
        * config.evaluation_packets_per_state
    )
    done = 0
    started = time.perf_counter()
    for snr_db in config.snr_db:
        print(f"[confirmation-evaluation] SNR={snr_db:g} dB start", flush=True)
        for state in states:
            for packet_index in range(config.evaluation_packets_per_state):
                trial, observation, model, scale_delta = _observation(
                    scene=SCENE_S1,
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
                    scene=SCENE_S1,
                    stage="evaluation",
                    state=state,
                    packet_index=packet_index,
                    snr_db=snr_db,
                    trial=trial,
                    observation=observation,
                    scale_delta=scale_delta,
                )
                selected = _selected_methods(
                    policies=policies,
                    state_id=str(state.state_id),
                    snr_db=float(snr_db),
                    sample_id=str(metadata["sample_id"]),
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
                    feedback_bits = (
                        ACTION_FEEDBACK_BITS / BLOCK_LENGTH
                        if method == TARGET_METHOD
                        else 0.0
                    )
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
                            "feedback_bits": feedback_bits,
                            "total_bits": payload_bits + feedback_bits,
                        }
                    )
                done += 1
                if done % config.progress_every == 0 or done == total:
                    elapsed = time.perf_counter() - started
                    rate = elapsed / done
                    print(
                        "[confirmation-evaluation] "
                        f"{done}/{total} elapsed={elapsed:.1f}s "
                        f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                        flush=True,
                    )
        pd.DataFrame(rows).to_csv(
            output / "confirmation_evaluation.partial.csv",
            index=False,
        )
    return pd.DataFrame(rows)


def _stream_is_independent(
    first_state: object,
    *,
    waveform: WaveformConfig,
) -> bool:
    planning = generate_packet_trial(
        first_state,
        stage="planning",
        packet_index=0,
        scene=SCENE_S1,
        waveform=waveform,
    )
    evaluation = generate_packet_trial(
        first_state,
        stage="evaluation",
        packet_index=0,
        scene=SCENE_S1,
        waveform=waveform,
    )
    return not (
        np.array_equal(planning.source_a, evaluation.source_a)
        or np.array_equal(
            planning.noise_a_standard,
            evaluation.noise_a_standard,
        )
        or np.array_equal(
            planning.noise_b_standard,
            evaluation.noise_b_standard,
        )
    )


def _mechanical_checks(
    *,
    config: ReplicationConfig,
    frozen_source_seeds: set[int],
    calibration_states: list[object],
    confirmation_states: list[object],
    calibration_selection: pd.DataFrame,
    calibration_check: pd.DataFrame,
    confirmation_planning: pd.DataFrame | None,
    confirmation_evaluation: pd.DataFrame | None,
    selected_count: int | None,
    waveform: WaveformConfig,
) -> dict[str, bool]:
    calibration_seeds = set(config.calibration_state_seeds)
    confirmation_seeds = set(config.confirmation_state_seeds)
    forbidden = frozen_source_seeds | HISTORICAL_STATE_SEEDS
    checks = {
        "source_certificate_hashes_match": all(
            _sha256(SOURCE_CERTIFICATE / name) == expected
            for name, expected in SOURCE_RESULT_SHA256.items()
        ),
        "calibration_and_confirmation_seeds_disjoint": not (
            calibration_seeds & confirmation_seeds
        ),
        "new_state_seeds_disjoint_from_all_frozen_sources": not (
            (calibration_seeds | confirmation_seeds) & forbidden
        ),
        "calibration_and_confirmation_state_ids_disjoint": not (
            {str(state.state_id) for state in calibration_states}
            & {str(state.state_id) for state in confirmation_states}
        ),
        "calibration_selection_and_check_streams_disjoint": (
            _stream_is_independent(calibration_states[0], waveform=waveform)
        ),
        "confirmation_planning_and_evaluation_streams_disjoint": (
            _stream_is_independent(confirmation_states[0], waveform=waveform)
        ),
        "only_s1_scene_used": True,
        "no_locked_data_used": True,
        "no_outcome_oracle_generated": True,
        "planning_count_selected_only_from_calibration": (
            selected_count in config.planning_candidate_counts
            if selected_count is not None
            else True
        ),
        "power_design_has_at_least_required_states": (
            len(confirmation_states) >= _minimum_state_count()
            if config.mode == "development"
            else True
        ),
        "no_quantizer_clipping_in_calibration": bool(
            max(
                float(calibration_selection["clipping_rate"].max()),
                float(calibration_check["clipping_rate"].max()),
            )
            <= 0.0
        ),
        "public_scaling_matches_frozen_protocol_in_calibration": bool(
            max(
                float(
                    calibration_selection[
                        "public_scale_delta_vs_legacy"
                    ].abs().max()
                ),
                float(
                    calibration_check[
                        "public_scale_delta_vs_legacy"
                    ].abs().max()
                ),
            )
            <= 1e-12
        ),
    }
    if confirmation_planning is None or confirmation_evaluation is None:
        return checks

    planning_keys = set(
        confirmation_planning[
            [
                "stage",
                "scene",
                "state_seed",
                "state_id",
                "snr_db",
                "packet_index",
                "configuration",
            ]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )
    evaluation_keys = set(
        confirmation_evaluation[
            [
                "stage",
                "scene",
                "state_seed",
                "state_id",
                "snr_db",
                "packet_index",
                "configuration",
            ]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )
    target_bits = PAYLOAD_24_BITS + ACTION_FEEDBACK_BITS / BLOCK_LENGTH
    target_rows = confirmation_evaluation[
        confirmation_evaluation["method"].eq(TARGET_METHOD)
    ]
    budget_ok = math.isclose(
        float(target_rows["total_bits"].mean()),
        target_bits,
        abs_tol=1e-12,
    )
    for snr_db in config.snr_db:
        snr_target = target_rows[target_rows["snr_db"].eq(float(snr_db))]
        n_samples = int(snr_target["sample_id"].nunique())
        tolerance = 4.0 / max(n_samples, 1) + 1e-12
        for method in (
            "BestStatic-RateMatched-L16",
            "SNR-only-RateMatched-L16",
        ):
            rows = confirmation_evaluation[
                confirmation_evaluation["method"].eq(method)
                & confirmation_evaluation["snr_db"].eq(float(snr_db))
            ]
            if abs(float(rows["total_bits"].mean()) - target_bits) > tolerance:
                budget_ok = False
    action_counts = target_rows.groupby(
        ["state_id", "snr_db"],
        sort=False,
    )["configuration"].nunique()
    checks.update(
        {
            "confirmation_planning_and_evaluation_rows_disjoint": not (
                planning_keys & evaluation_keys
            ),
            "all_rate_matched_budgets_within_integer_quantum": budget_ok,
            "state_oracle_action_shared_within_each_l16_block": bool(
                (action_counts == 1).all()
            ),
            "state_oracle_uses_only_b24_payload": set(
                target_rows["payload_bits"]
            )
            == {PAYLOAD_24_BITS},
            "all_registered_methods_present": set(
                confirmation_evaluation["method"].unique()
            )
            == {TARGET_METHOD, *BASELINES},
            "no_quantizer_clipping_in_confirmation": bool(
                float(confirmation_evaluation["clipping_rate"].max()) <= 0.0
            ),
            "public_scaling_matches_frozen_protocol_in_confirmation": bool(
                float(
                    confirmation_evaluation[
                        "public_scale_delta_vs_legacy"
                    ].abs().max()
                )
                <= 1e-12
            ),
        }
    )
    return checks


def _plot_planning(
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    counts = summary["candidate_count"].to_numpy(dtype=int)
    mean = summary["mean_regret_samples2"].to_numpy(dtype=float)
    low = summary["regret_ci_low"].to_numpy(dtype=float)
    high = summary["regret_ci_high"].to_numpy(dtype=float)
    axes[0].errorbar(
        counts,
        mean,
        yerr=np.vstack((mean - low, high - mean)),
        color="#0072B2",
        marker="o",
        linewidth=1.25,
        capsize=3,
        label="Regret vs 256-packet reference",
    )
    axes[0].axhline(
        float(summary["equivalence_margin_samples2"].iloc[0]),
        color="#D55E00",
        linestyle="--",
        linewidth=1.1,
        label="Pre-registered equivalence margin",
    )
    axes[0].set_xlabel("Planning packets per state/SNR")
    axes[0].set_ylabel("Independent-check MSE regret (samples²)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    pivot = sensitivity.pivot(
        index="equivalence_margin_samples2",
        columns="candidate_count",
        values="equivalence_pass",
    )
    axes[1].imshow(
        pivot.to_numpy(dtype=float),
        cmap=matplotlib.colors.ListedColormap(("#4D4D4D", "#FFFFFF")),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[1].set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    axes[1].set_yticks(
        np.arange(len(pivot.index)),
        [f"{value:.4f}" for value in pivot.index],
    )
    axes[1].set_xlabel("Planning packets per state/SNR")
    axes[1].set_ylabel("Equivalence margin (samples²)")
    axes[1].set_title("Margin sensitivity (white = pass)")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            axes[1].text(
                column,
                row,
                "PASS" if pivot.iloc[row, column] else "FAIL",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if pivot.iloc[row, column] else "white",
            )
    fig.tight_layout()
    fig.savefig(
        output / "Fig1_S1_Planning_Convergence.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_confirmation(summary: pd.DataFrame, *, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    methods = (
        TARGET_METHOD,
        "BestStatic-B24",
        "SNR-only-B24",
        "BestStatic-RateMatched-L16",
        "SNR-only-RateMatched-L16",
    )
    colors = ("#D55E00", "#0072B2", "#009E73", "#56B4E9", "#CC79A7")
    markers = ("o", "s", "^", "D", "v")
    for method, color, marker in zip(methods, colors, markers, strict=True):
        rows = summary[
            summary["method"].eq(method) & summary["snr_db"].notna()
        ].sort_values("snr_db")
        axis.plot(
            rows["snr_db"],
            rows["rmse_samples"],
            label=method,
            color=color,
            marker=marker,
            linewidth=1.25,
            markersize=4.5,
        )
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("TDOA RMSE (samples)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(
        output / "Fig2_S1_Precision_Replication.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def _cleanup_partials(output: Path) -> None:
    for path in output.glob("*.partial.csv"):
        path.unlink()


def _run(config: ReplicationConfig, output: Path) -> None:
    started = time.perf_counter()
    stage2_manifest, temperature_by_snr, frozen_hash_audit = (
        _load_and_audit_sources()
    )
    frozen_source_seeds = _source_seed_set(stage2_manifest)
    config.validate(frozen_source_seeds)
    missing_temperature = set(config.snr_db) - set(temperature_by_snr)
    if missing_temperature:
        raise RuntimeError(
            f"frozen temperature misses SNR values: {sorted(missing_temperature)}"
        )
    required_states = _minimum_state_count()
    print(
        "Power design: "
        f"historical effect={HISTORICAL_MIN_S1_L16_MSE_GAIN:.6f}, "
        f"cluster SD={HISTORICAL_MAX_S1_L16_CLUSTER_SD:.6f}, "
        f"minimum states={required_states}, "
        f"registered states="
        f"{len(config.confirmation_state_seeds) * config.confirmation_states_per_seed}",
        flush=True,
    )

    waveform = WaveformConfig()
    waveform.validate()
    cases, bins, prior_statistics = _build_frozen_model(
        stage2_manifest,
        waveform,
    )
    quantizer = NestedComplexQuantizer(
        max_bits=QUANTIZER_BITS,
        clip_abs=QUANTIZER_CLIP_ABS,
    )
    prior_statistics.to_csv(output / "prior_statistics.csv", index=False)

    calibration_states = generate_state_bank(
        config.calibration_state_seeds,
        config.calibration_states_per_seed,
        waveform=waveform,
        candidate_bins=bins,
    )
    confirmation_states = generate_state_bank(
        config.confirmation_state_seeds,
        config.confirmation_states_per_seed,
        waveform=waveform,
        candidate_bins=bins,
    )
    state_rows(calibration_states).assign(split="planning-calibration").to_csv(
        output / "calibration_state_definitions.csv",
        index=False,
    )
    state_rows(confirmation_states).assign(split="effect-confirmation").to_csv(
        output / "confirmation_state_definitions.csv",
        index=False,
    )

    selection_rows, _ = _collect_configuration_rows(
        states=calibration_states,
        stage="planning",
        packet_count=config.planning_reference_count,
        snr_db_values=config.snr_db,
        include_b32=False,
        stream_name="planning-calibration-selection",
        cases=cases,
        bins=bins,
        waveform=waveform,
        quantizer=quantizer,
        temperature_by_snr=temperature_by_snr,
        output=output,
        partial_name="planning_calibration_selection.partial.csv",
        progress_every=config.progress_every,
    )
    check_rows, _ = _collect_configuration_rows(
        states=calibration_states,
        stage="evaluation",
        packet_count=config.planning_check_count,
        snr_db_values=config.snr_db,
        include_b32=False,
        stream_name="planning-calibration-check",
        cases=cases,
        bins=bins,
        waveform=waveform,
        quantizer=quantizer,
        temperature_by_snr=temperature_by_snr,
        output=output,
        partial_name="planning_calibration_check.partial.csv",
        progress_every=config.progress_every,
    )
    selection_rows.to_csv(
        output / "planning_calibration_selection.csv",
        index=False,
    )
    check_rows.to_csv(
        output / "planning_calibration_check.csv",
        index=False,
    )
    margin = (
        PLANNING_EQUIVALENCE_FRACTION * HISTORICAL_MIN_S1_L16_MSE_GAIN
    )
    sensitivity_margins = tuple(
        fraction * HISTORICAL_MIN_S1_L16_MSE_GAIN
        for fraction in PLANNING_SENSITIVITY_FRACTIONS
    )
    (
        convergence_details,
        convergence_summary,
        convergence_sensitivity,
        selected_count,
    ) = planning_convergence_diagnostics(
        selection_rows,
        check_rows,
        candidate_counts=config.planning_candidate_counts,
        reference_count=config.planning_reference_count,
        equivalence_margin=margin,
        sensitivity_margins=sensitivity_margins,
        bootstrap_repetitions=config.bootstrap_repetitions,
        bootstrap_seed=2026081091,
    )
    convergence_details.to_csv(
        output / "planning_convergence_details.csv",
        index=False,
    )
    convergence_summary.to_csv(
        output / "planning_convergence_summary.csv",
        index=False,
    )
    convergence_sensitivity.to_csv(
        output / "planning_convergence_margin_sensitivity.csv",
        index=False,
    )
    _plot_planning(
        convergence_summary,
        convergence_sensitivity,
        output=output,
    )
    calibration_selected_count = selected_count
    smoke_forced_planning_count = False
    if config.mode == "smoke" and selected_count is None:
        selected_count = config.planning_candidate_counts[0]
        smoke_forced_planning_count = True
        print(
            "[Smoke] planning convergence did not pass; forcing the sole "
            "candidate only to validate the downstream chain.",
            flush=True,
        )

    confirmation_planning: pd.DataFrame | None = None
    confirmation_evaluation: pd.DataFrame | None = None
    effects: pd.DataFrame | None = None
    summary: pd.DataFrame | None = None
    policies: pd.DataFrame | None = None
    if selected_count is not None:
        confirmation_planning, model = _collect_aggregated_confirmation_planning(
            states=confirmation_states,
            packet_count=selected_count,
            snr_db_values=config.snr_db,
            cases=cases,
            bins=bins,
            waveform=waveform,
            quantizer=quantizer,
            temperature_by_snr=temperature_by_snr,
            output=output,
            progress_every=config.progress_every,
        )
        confirmation_planning.to_csv(
            output / "confirmation_planning_configuration_losses.csv",
            index=False,
        )
        policies = freeze_policies(confirmation_planning)
        policies["bin_indices"] = policies["configuration"].map(
            lambda value: " ".join(
                str(int(model.bin_indices[int(position)]))
                for position in str(value).split()
            )
        )
        policies.to_csv(
            output / "confirmation_frozen_state_policies.csv",
            index=False,
        )
        confirmation_evaluation = _collect_confirmation_evaluation(
            states=confirmation_states,
            config=config,
            policies=policies,
            cases=cases,
            bins=bins,
            waveform=waveform,
            quantizer=quantizer,
            temperature_by_snr=temperature_by_snr,
            output=output,
        )
        confirmation_evaluation.to_csv(
            output / "confirmation_evaluation_results.csv",
            index=False,
        )
        summary = summarize_results(confirmation_evaluation)
        summary.to_csv(output / "confirmation_policy_summary.csv", index=False)
        effects = paired_effects(
            confirmation_evaluation,
            bootstrap_repetitions=config.bootstrap_repetitions,
            bootstrap_seed=2026081191,
        )
        effects.to_csv(output / "confirmation_paired_effects.csv", index=False)
        _plot_confirmation(summary, output=output)

    mechanical = _mechanical_checks(
        config=config,
        frozen_source_seeds=frozen_source_seeds,
        calibration_states=calibration_states,
        confirmation_states=confirmation_states,
        calibration_selection=selection_rows,
        calibration_check=check_rows,
        confirmation_planning=confirmation_planning,
        confirmation_evaluation=confirmation_evaluation,
        selected_count=selected_count,
        waveform=waveform,
    )
    decision = gate_decision(
        mode=config.mode,
        planning_selected_count=calibration_selected_count,
        effects=effects,
        mechanical_checks=mechanical,
    )
    _json_dump(output / "gate_decision.json", decision)
    manifest = {
        "mode": config.mode,
        "status": decision["status"],
        "config": asdict(config),
        "pre_registered_hypothesis": (
            "S1 StateOracle-L16-B24 has repeatable positive paired MSE gain "
            "against all four frozen B24 and rate-matched static baselines."
        ),
        "power_design": {
            "historical_min_s1_l16_mse_gain": (
                HISTORICAL_MIN_S1_L16_MSE_GAIN
            ),
            "historical_max_s1_l16_cluster_sd": (
                HISTORICAL_MAX_S1_L16_CLUSTER_SD
            ),
            "two_sided_alpha": 0.05,
            "target_power": 0.80,
            "minimum_independent_states": required_states,
            "registered_independent_states": len(confirmation_states),
        },
        "planning_convergence_design": {
            "candidate_counts": list(config.planning_candidate_counts),
            "reference_count": config.planning_reference_count,
            "independent_check_count": config.planning_check_count,
            "primary_equivalence_fraction_of_historical_effect": (
                PLANNING_EQUIVALENCE_FRACTION
            ),
            "primary_equivalence_margin_samples2": margin,
            "sensitivity_fractions": list(PLANNING_SENSITIVITY_FRACTIONS),
            "selected_count_from_calibration": calibration_selected_count,
            "execution_planning_count": selected_count,
            "smoke_forced_downstream_count": smoke_forced_planning_count,
            "selection_and_check_states_excluded_from_confirmation": True,
        },
        "frozen_protocol": {
            "scene": SCENE_S1,
            "candidate_pool": PRIMARY_POOL,
            "likelihood": PRIMARY_LIKELIHOOD,
            "quadrature": [5, 64],
            "block_length": BLOCK_LENGTH,
            "base_payload_bits": 16,
            "one_acquire_payload_bits": 8,
            "action_feedback_bits": ACTION_FEEDBACK_BITS,
            "quantizer_bits": QUANTIZER_BITS,
            "quantizer_clip_abs": QUANTIZER_CLIP_ABS,
            "candidate_bins": [int(value) for value in bins],
            "temperature_by_snr": {
                str(snr): float(temperature_by_snr[float(snr)])
                for snr in config.snr_db
            },
        },
        "information_boundary": {
            "state_oracle_knows_shared_tau_and_complex_gain": True,
            "state_oracle_uses_confirmation_outcomes": False,
            "observable_policy_trained": False,
            "locked_data_used": False,
            "outcome_oracle_used": False,
        },
        "mechanical_checks": mechanical,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_sha256": _source_hashes(),
        "frozen_stage2_source_hash_audit": frozen_hash_audit,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _json_dump(output / "manifest.json", manifest)
    _cleanup_partials(output)

    print("\nS1 precision replication decision", flush=True)
    print(
        f"  planning selected count from calibration: "
        f"{calibration_selected_count}",
        flush=True,
    )
    print(f"  execution planning count: {selected_count}", flush=True)
    print(f"  mechanical pass: {all(mechanical.values())}", flush=True)
    print(f"[Gate] {decision['status']}", flush=True)
    print(f"[Done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    args = _parse_args()
    config = CONFIGS[args.mode]
    output = _output_dir(config.mode)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.log").open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_handle)):
            print("BRSR S1-only StateOracle precision replication", flush=True)
            print(f"Mode: {config.mode}", flush=True)
            print(f"Output: {output}", flush=True)
            print(f"Config: {json.dumps(asdict(config))}", flush=True)
            print(
                "Boundary: no ObservablePolicy, no OutcomeOracle, no locked data.",
                flush=True,
            )
            _run(config, output)


if __name__ == "__main__":
    main()
