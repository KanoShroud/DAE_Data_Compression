"""运行BRSR共享状态StateOracle路线可行性证书。

PyCharm直接运行默认执行development；Codex仅执行 ``--mode smoke``。
本入口不训练ObservablePolicy，不修改冻结BRSR，也不读取locked结果。
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

from brsr_observability_certificate.run_certificate import (  # noqa: E402
    _source_hashes as observability_source_hashes,
)
from brsr_route_feasibility.run_stage1 import (  # noqa: E402
    Tee,
    _likelihood_observations,
)
from brsr_stage2_rate_adaptive.run_stage2a import (  # noqa: E402
    PRIMARY_LIKELIHOOD,
    PRIMARY_POOL,
    QUANTIZER_BITS,
    QUANTIZER_CLIP_ABS,
)
from brsr_stage2_rate_adaptive.run_stage2a1 import (  # noqa: E402
    SOURCE_STAGE2A,
    _build_frozen_model,
    _source_seed_set,
)
from brsr_stage2_rate_adaptive.stage2a import (  # noqa: E402
    rate_match_assignments,
)
from brsr_state_oracle_certificate.certificate import (  # noqa: E402
    ACTION_FEEDBACK_BITS,
    KEY_COLUMNS,
    PAYLOAD_24_BITS,
    PAYLOAD_32_BITS,
    SCENES,
    SCENE_S0,
    SCENE_S1,
    SCENE_S2,
    configuration_key,
    configuration_positions,
    evaluate_configurations,
    freeze_policies,
    gate_decision,
    generate_packet_trial,
    generate_state_bank,
    lookup_configuration,
    paired_effects,
    planning_policy_diagnostics,
    state_rows,
    summarize_results,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer  # noqa: E402
from brsr_tdoa.brsr_waveform import (  # noqa: E402
    WaveformConfig,
    observe_trial,
)


PYCHARM_RUN_MODE = "development"


@dataclass(frozen=True)
class StateOracleConfig:
    mode: str
    snr_db: tuple[float, ...]
    state_seeds: tuple[int, ...]
    states_per_seed: int
    planning_packets_per_state: int
    evaluation_packets_per_state: int
    coherent_block_lengths: tuple[int, ...]
    bootstrap_repetitions: int
    progress_every: int

    def validate(self, frozen_source_seeds: set[int]) -> None:
        if self.mode not in {"smoke", "development"}:
            raise ValueError("mode must be smoke or development")
        if not self.snr_db or len(set(self.snr_db)) != len(self.snr_db):
            raise ValueError("snr_db must be non-empty and unique")
        if not self.state_seeds or len(set(self.state_seeds)) != len(
            self.state_seeds
        ):
            raise ValueError("state_seeds must be non-empty and unique")
        if set(self.state_seeds) & frozen_source_seeds:
            raise ValueError("state seeds overlap frozen BRSR sources")
        counts = (
            self.states_per_seed,
            self.planning_packets_per_state,
            self.evaluation_packets_per_state,
            self.bootstrap_repetitions,
            self.progress_every,
        )
        if min(counts) < 1 or self.bootstrap_repetitions < 20:
            raise ValueError("invalid state-oracle count or bootstrap configuration")
        if (
            not self.coherent_block_lengths
            or len(set(self.coherent_block_lengths))
            != len(self.coherent_block_lengths)
            or any(
                value < 2
                or value > self.evaluation_packets_per_state
                or self.evaluation_packets_per_state % value
                for value in self.coherent_block_lengths
            )
        ):
            raise ValueError(
                "coherent_block_lengths must be unique divisors of evaluation count"
            )


CONFIGS = {
    "smoke": StateOracleConfig(
        mode="smoke",
        snr_db=(-5.0, 10.0),
        state_seeds=(2026080101,),
        states_per_seed=1,
        planning_packets_per_state=2,
        evaluation_packets_per_state=4,
        coherent_block_lengths=(2, 4),
        bootstrap_repetitions=50,
        progress_every=1,
    ),
    "development": StateOracleConfig(
        mode="development",
        snr_db=(-10.0, 0.0, 5.0, 15.0),
        state_seeds=(2026080111, 2026080112, 2026080113),
        states_per_seed=4,
        planning_packets_per_state=32,
        evaluation_packets_per_state=16,
        coherent_block_lengths=(4, 8, 16),
        bootstrap_repetitions=2000,
        progress_every=24,
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
        / f"BRSR_STATE_ORACLE_{timestamp}_{mode}"
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
        ROOT / "brsr_state_oracle_certificate" / "certificate.py",
        ROOT / "brsr_state_oracle_certificate" / "run_certificate.py",
        ROOT / "brsr_state_oracle_certificate" / "README.md",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _load_frozen_source_line_ending_safe() -> tuple[
    dict[str, object],
    pd.DataFrame,
    dict[float, float],
    dict[str, dict[str, object]],
]:
    """加载冻结Stage 2A，并区分源码变化与CRLF/LF差异。"""

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
        raise RuntimeError("StateOracle certificate requires formal Stage 2A source")
    protocol = manifest.get("frozen_protocol", {})
    if (
        protocol.get("pool") != PRIMARY_POOL
        or protocol.get("likelihood") != PRIMARY_LIKELIHOOD
        or protocol.get("quadrature") != [5, 64]
    ):
        raise RuntimeError("frozen Stage 2A protocol is inconsistent")

    audit: dict[str, dict[str, object]] = {}
    for relative, expected in manifest.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen source missing: {relative}")
        data = path.read_bytes()
        raw = hashlib.sha256(data).hexdigest()
        normalized = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        if raw == expected:
            match_mode = "raw"
        elif normalized == expected:
            match_mode = "newline-normalized"
        else:
            raise RuntimeError(f"frozen source content changed: {relative}")
        audit[str(relative)] = {
            "expected_sha256": str(expected),
            "current_raw_sha256": raw,
            "current_lf_normalized_sha256": normalized,
            "match_mode": match_mode,
        }

    plans = pd.read_csv(SOURCE_STAGE2A / "frozen_static_plans.csv")
    plans["positions"] = plans["positions"].fillna("")
    plans["bins"] = plans["bins"].fillna("")
    temperature = pd.read_csv(SOURCE_STAGE2A / "temperature_calibration.csv")
    temperature_by_snr = {
        float(row["snr_db"]): float(row["temperature"])
        for _, row in temperature.iterrows()
    }
    return manifest, plans, temperature_by_snr, audit


def _observation(
    *,
    scene: str,
    stage: str,
    state: object,
    packet_index: int,
    snr_db: float,
    cases: dict[str, object],
    bins: np.ndarray,
    waveform: WaveformConfig,
    quantizer: NestedComplexQuantizer,
) -> tuple[object, object, object, float]:
    trial = generate_packet_trial(
        state,
        stage=stage,
        packet_index=packet_index,
        scene=scene,
        waveform=waveform,
    )
    y_a, y_b, sigma2 = observe_trial(trial, snr_db)
    trial_index = (
        int(state.state_index) * 10_000
        + (0 if stage == "planning" else 5_000)
        + packet_index
    )
    observations = _likelihood_observations(
        stage=f"state_oracle_{stage}",
        seed=int(state.state_seed),
        trial_index=trial_index,
        snr_db=float(snr_db),
        trial=trial,
        y_a=y_a,
        y_b=y_b,
        sigma2=sigma2,
        bins=bins,
        cases=cases,
        waveform=waveform,
        quantizer=quantizer,
    )
    observation, model, scale_delta = observations[PRIMARY_LIKELIHOOD]
    return trial, observation, model, float(scale_delta)


def _base_metadata(
    *,
    scene: str,
    stage: str,
    state: object,
    packet_index: int,
    snr_db: float,
    trial: object,
    observation: object,
    scale_delta: float,
) -> dict[str, object]:
    return {
        "scene": scene,
        "stage": stage,
        "state_seed": int(state.state_seed),
        "state_index": int(state.state_index),
        "state_id": str(state.state_id),
        "context_class": (
            str(state.context_class) if scene == SCENE_S2 else "none"
        ),
        "snr_db": float(snr_db),
        "packet_index": int(packet_index),
        "sample_id": (
            f"{scene}|{state.state_id}|{stage}|{packet_index}|{snr_db:g}"
        ),
        "true_tau": float(trial.true_tau),
        "true_rho_real": float(trial.true_rho.real),
        "true_rho_imag": float(trial.true_rho.imag),
        "clipping_rate": float(observation.clipping_rate),
        "public_scale_delta_vs_legacy": float(scale_delta),
    }


def _collect_planning(
    *,
    config: StateOracleConfig,
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


def _rate_match_maps(
    *,
    config: StateOracleConfig,
    states: list[object],
) -> dict[tuple[str, float, int], dict[str, int]]:
    maps: dict[tuple[str, float, int], dict[str, int]] = {}
    for scene in SCENES:
        block_lengths = (
            (1,) if scene == SCENE_S0 else config.coherent_block_lengths
        )
        for block_size in block_lengths:
            target = PAYLOAD_24_BITS + ACTION_FEEDBACK_BITS / block_size
            for snr_db in config.snr_db:
                sample_ids = [
                    f"{scene}|{state.state_id}|evaluation|{packet}|{snr_db:g}"
                    for state in states
                    for packet in range(config.evaluation_packets_per_state)
                ]
                maps[(scene, float(snr_db), block_size)] = rate_match_assignments(
                    sample_ids,
                    desired_mean_bits=target,
                    low_bits=PAYLOAD_24_BITS,
                    high_bits=PAYLOAD_32_BITS,
                    salt=f"state-oracle|{scene}|L{block_size}|{snr_db:g}",
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
    rate_maps: dict[tuple[str, float, int], dict[str, int]],
    coherent_block_lengths: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    selected = {
        "BestStatic-B24": lookup_configuration(
            policies,
            scene=scene,
            policy="BestStatic",
            snr_db=snr_db,
            state_id=state_id,
            context_class=context_class,
            payload_bits=PAYLOAD_24_BITS,
        ),
        "SNR-only-B24": lookup_configuration(
            policies,
            scene=scene,
            policy="SNR-only",
            snr_db=snr_db,
            state_id=state_id,
            context_class=context_class,
            payload_bits=PAYLOAD_24_BITS,
        ),
    }
    if scene == SCENE_S2:
        selected["ContextStatic-B24"] = lookup_configuration(
            policies,
            scene=scene,
            policy="ContextStatic",
            snr_db=snr_db,
            state_id=state_id,
            context_class=context_class,
            payload_bits=PAYLOAD_24_BITS,
        )
    block_lengths = (1,) if scene == SCENE_S0 else coherent_block_lengths
    oracle_configuration = lookup_configuration(
        policies,
        scene=scene,
        policy="StateOracle",
        snr_db=snr_db,
        state_id=state_id,
        context_class=context_class,
        payload_bits=PAYLOAD_24_BITS,
    )
    for block_size in block_lengths:
        rate_bits = rate_maps[(scene, float(snr_db), block_size)][sample_id]
        selected[f"StateOracle-L{block_size}-B24"] = oracle_configuration
        selected[f"BestStatic-RateMatched-L{block_size}"] = lookup_configuration(
            policies,
            scene=scene,
            policy="BestStatic",
            snr_db=snr_db,
            state_id=state_id,
            context_class=context_class,
            payload_bits=rate_bits,
        )
        selected[f"SNR-only-RateMatched-L{block_size}"] = lookup_configuration(
            policies,
            scene=scene,
            policy="SNR-only",
            snr_db=snr_db,
            state_id=state_id,
            context_class=context_class,
            payload_bits=rate_bits,
        )
        if scene == SCENE_S2:
            selected[f"ContextStatic-RateMatched-L{block_size}"] = (
                lookup_configuration(
                    policies,
                    scene=scene,
                    policy="ContextStatic",
                    snr_db=snr_db,
                    state_id=state_id,
                    context_class=context_class,
                    payload_bits=rate_bits,
                )
            )
    return selected


def _collect_evaluation(
    *,
    config: StateOracleConfig,
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
    rate_maps = _rate_match_maps(config=config, states=states)
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
                    selected = _selected_methods(
                        scene=scene,
                        snr_db=snr_db,
                        state_id=str(state.state_id),
                        context_class=str(metadata["context_class"]),
                        sample_id=str(metadata["sample_id"]),
                        policies=policies,
                        rate_maps=rate_maps,
                        coherent_block_lengths=config.coherent_block_lengths,
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
                        feedback_bits = 0.0
                        if method.startswith("StateOracle-L"):
                            block_size = int(
                                method.split("-L", 1)[1].split("-", 1)[0]
                            )
                            feedback_bits = ACTION_FEEDBACK_BITS / block_size
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
                            f"[evaluation] {done}/{total} elapsed={elapsed:.1f}s "
                            f"avg={rate:.3f}s ETA={rate * (total - done):.1f}s",
                            flush=True,
                        )
            pd.DataFrame(rows).to_csv(
                output / "evaluation_results.partial.csv",
                index=False,
            )
    return pd.DataFrame(rows)


def _mechanical_checks(
    *,
    config: StateOracleConfig,
    states: list[object],
    planning: pd.DataFrame,
    policies: pd.DataFrame,
    evaluation: pd.DataFrame,
    frozen_source_seeds: set[int],
    waveform: WaveformConfig,
) -> dict[str, bool]:
    planning_keys = set(
        planning[["stage", *KEY_COLUMNS, "configuration"]]
        .astype(str)
        .agg("|".join, axis=1)
    )
    evaluation_keys = set(
        evaluation[["stage", *KEY_COLUMNS, "configuration"]]
        .astype(str)
        .agg("|".join, axis=1)
    )
    budget_ok = True
    shared_action_ok = True
    for scene in SCENES:
        block_lengths = (
            (1,) if scene == SCENE_S0 else config.coherent_block_lengths
        )
        for block_size in block_lengths:
            target = PAYLOAD_24_BITS + ACTION_FEEDBACK_BITS / block_size
            for snr_db in config.snr_db:
                subset = evaluation[
                    evaluation["scene"].eq(scene)
                    & evaluation["snr_db"].eq(float(snr_db))
                ]
                oracle_method = f"StateOracle-L{block_size}-B24"
                oracle = subset[subset["method"].eq(oracle_method)]
                if not math.isclose(
                    float(oracle["total_bits"].mean()),
                    target,
                    abs_tol=1e-12,
                ):
                    budget_ok = False
                n_samples = oracle["sample_id"].nunique()
                tolerance = 4.0 / max(n_samples, 1) + 1e-12
                static_methods = (
                    f"BestStatic-RateMatched-L{block_size}",
                    f"SNR-only-RateMatched-L{block_size}",
                    *(
                        (f"ContextStatic-RateMatched-L{block_size}",)
                        if scene == SCENE_S2
                        else ()
                    ),
                )
                for method in static_methods:
                    static = subset[subset["method"].eq(method)]
                    if (
                        abs(float(static["total_bits"].mean()) - target)
                        > tolerance
                    ):
                        budget_ok = False
                action_counts = oracle.groupby(
                    ["state_id", "snr_db"],
                    sort=False,
                )["configuration"].nunique()
                if not (action_counts == 1).all():
                    shared_action_ok = False

    s0_oracle = policies[
        policies["scene"].eq(SCENE_S0)
        & policies["policy"].eq("StateOracle")
    ][["snr_db", "configuration"]].sort_values("snr_db", kind="stable")
    s0_snr = policies[
        policies["scene"].eq(SCENE_S0)
        & policies["policy"].eq("SNR-only")
        & policies["payload_bits"].eq(PAYLOAD_24_BITS)
    ][["snr_db", "configuration"]].sort_values("snr_db", kind="stable")

    shared_state_ok = True
    for scene in (SCENE_S1, SCENE_S2):
        for state in states:
            for stage, count in (
                ("planning", config.planning_packets_per_state),
                ("evaluation", config.evaluation_packets_per_state),
            ):
                taus = []
                rhos = []
                for packet in range(count):
                    trial = generate_packet_trial(
                        state,
                        stage=stage,
                        packet_index=packet,
                        scene=scene,
                        waveform=waveform,
                    )
                    taus.append(trial.true_tau)
                    rhos.append(trial.true_rho)
                if not (
                    np.allclose(taus, state.true_tau, atol=0.0)
                    and np.allclose(rhos, state.true_rho, atol=0.0)
                ):
                    shared_state_ok = False

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
        or np.array_equal(
            plan_trial.noise_a_standard,
            eval_trial.noise_a_standard,
        )
        or np.array_equal(
            plan_trial.noise_b_standard,
            eval_trial.noise_b_standard,
        )
    )

    return {
        "new_state_seeds_disjoint_from_frozen_sources": not (
            set(config.state_seeds) & frozen_source_seeds
        ),
        "planning_and_evaluation_rows_disjoint": not (
            planning_keys & evaluation_keys
        ),
        "planning_and_evaluation_random_streams_disjoint": streams_separate,
        "s1_s2_share_tau_and_rho_within_block": shared_state_ok,
        "s0_state_oracle_equals_snr_only": s0_oracle.reset_index(
            drop=True
        ).equals(s0_snr.reset_index(drop=True)),
        "all_rate_matched_budgets_within_integer_quantum": budget_ok,
        "state_oracle_action_shared_within_each_block": shared_action_ok,
        "state_oracle_uses_only_b24_payload": set(
            evaluation[evaluation["method"].str.startswith("StateOracle-L")][
                "payload_bits"
            ]
        )
        == {PAYLOAD_24_BITS},
        "no_locked_data_used": True,
        "no_outcome_oracle_generated": not evaluation["method"].str.contains(
            "OutcomeOracle",
            regex=False,
        ).any(),
    }


def _plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    labels: dict[str, object] = {}
    for axis, scene in zip(axes, SCENES, strict=True):
        subset = summary[
            summary["scene"].eq(scene) & summary["snr_db"].notna()
        ]
        target_methods = sorted(
            subset[
                subset["method"].str.startswith("StateOracle-L")
            ]["method"].unique()
        )
        rate_prefix = (
            "ContextStatic-RateMatched-L"
            if scene == SCENE_S2
            else "SNR-only-RateMatched-L"
        )
        rate_methods = sorted(
            subset[subset["method"].str.startswith(rate_prefix)][
                "method"
            ].unique()
        )
        strongest_b24 = (
            "ContextStatic-B24" if scene == SCENE_S2 else "SNR-only-B24"
        )
        methods = ("BestStatic-B24", strongest_b24, *target_methods, *rate_methods)
        colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(methods)))
        markers = ("s", "^", "o", "D", "P", "v", "X", "<", ">")
        for index, method in enumerate(methods):
            rows = subset[subset["method"].eq(method)].sort_values("snr_db")
            if rows.empty:
                continue
            (line,) = axis.plot(
                rows["snr_db"],
                rows["rmse_samples"],
                color=colors[index],
                marker=markers[index % len(markers)],
                linewidth=1.25,
                markersize=4.2,
                label=method,
            )
            labels.setdefault(method, line)
        axis.set_title(scene)
        axis.set_xlabel("SNR (dB)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("TDOA RMSE (samples)")
    fig.legend(
        labels.values(),
        labels.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(
        output / "Fig1_BRSR_StateOracle_Certificate.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def _cleanup_partials(output: Path) -> None:
    for path in output.glob("*.partial.csv"):
        path.unlink()


def _run(config: StateOracleConfig, output: Path) -> None:
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
    states_frame = state_rows(states)
    states_frame.to_csv(output / "state_definitions.csv", index=False)
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
    policies = freeze_policies(planning)
    planning_diagnostics = planning_policy_diagnostics(planning)
    planning_diagnostics.to_csv(
        output / "planning_policy_diagnostics.csv",
        index=False,
    )
    policies["bin_indices"] = policies["configuration"].map(
        lambda value: " ".join(
            str(int(model.bin_indices[int(position)]))
            for position in str(value).split()
        )
    )
    policies.to_csv(output / "frozen_state_policies.csv", index=False)

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
    summary = summarize_results(evaluation)
    summary.to_csv(output / "policy_summary.csv", index=False)
    effects = paired_effects(
        evaluation,
        repetitions=config.bootstrap_repetitions,
        seed=2026080191,
    )
    effects.to_csv(output / "paired_effects.csv", index=False)
    mechanical = _mechanical_checks(
        config=config,
        states=states,
        planning=planning,
        policies=policies,
        evaluation=evaluation,
        frozen_source_seeds=frozen_source_seeds,
        waveform=waveform,
    )
    decision = gate_decision(effects, mechanical_checks=mechanical)
    if config.mode == "smoke":
        decision["provisional_status"] = decision["status"]
        decision["status"] = "SMOKE_ONLY"
        decision["smoke_is_performance_evidence"] = False
    _json_dump(output / "gate_decision.json", decision)

    budget_rows = summary[
        summary["snr_db"].notna()
        & (
            summary["method"].str.startswith("StateOracle-L")
            | summary["method"].str.contains("-RateMatched-L", regex=False)
        )
    ].copy()
    budget_rows.to_csv(output / "budget_audit.csv", index=False)
    _plot(summary, output)

    manifest_out = {
        "mode": config.mode,
        "status": decision["status"],
        "config": asdict(config),
        "scenes": list(SCENES),
        "frozen_protocol": {
            "candidate_pool": "Current-Fisher",
            "likelihood": PRIMARY_LIKELIHOOD,
            "quadrature": [5, 64],
            "base_payload_bits": 16,
            "one_acquire_payload_bits": 8,
            "block_feedback_bits": ACTION_FEEDBACK_BITS,
            "quantizer_bits": QUANTIZER_BITS,
            "quantizer_clip_abs": QUANTIZER_CLIP_ABS,
            "candidate_bins": [int(value) for value in bins],
            "temperature_by_snr": {
                str(value): float(temperature_by_snr[float(value)])
                for value in config.snr_db
            },
        },
        "state_protocol": {
            "planning_and_evaluation_share_only_physical_state": True,
            "planning_and_evaluation_symbols_and_noise_are_independent": True,
            "state_oracle_uses_evaluation_outcomes": False,
            "s0_block_size": 1,
            "s1_s2_block_lengths": list(config.coherent_block_lengths),
            "evaluation_packets_per_state": config.evaluation_packets_per_state,
            "pilot_cost_included": False,
            "pilot_cost_reason": (
                "StateOracle knows the state and is only a route-headroom "
                "certificate. Any future ObservablePolicy must count added "
                "pilot bits and delay."
            ),
        },
        "mechanical_checks": mechanical,
        "source_sha256": _source_hashes(),
        "frozen_source_hash_audit": frozen_hash_audit,
        "observability_source_sha256_at_run": observability_source_hashes(),
        "locked_data_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _json_dump(output / "manifest.json", manifest_out)
    _cleanup_partials(output)

    print("\nStateOracle scene decision", flush=True)
    for scene, status in decision["scene_status"].items():
        print(f"  {scene}: {status}", flush=True)
    print(f"[Gate] {decision['status']}", flush=True)
    print(f"[Done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    args = _parse_args()
    config = CONFIGS[args.mode]
    output = _output_dir(config.mode)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.log").open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_handle)):
            print("BRSR shared-state StateOracle certificate", flush=True)
            print(f"Mode: {config.mode}", flush=True)
            print(f"Output: {output}", flush=True)
            print(
                "Scope: S0/S1/S2 state-headroom only; no ObservablePolicy, "
                "no OutcomeOracle, no locked data.",
                flush=True,
            )
            print(
                "Config: "
                + json.dumps(asdict(config), ensure_ascii=False),
                flush=True,
            )
            _run(config, output)


if __name__ == "__main__":
    main()
