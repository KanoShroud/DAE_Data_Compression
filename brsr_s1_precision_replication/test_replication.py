"""Unit tests for the S1-only precision replication."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from brsr_s1_precision_replication.run_replication import (
    CONFIGS,
    _mechanical_checks,
)
from brsr_s1_precision_replication.replication import (
    BASELINES,
    SCENE_S1,
    TARGET_METHOD,
    gate_decision,
    paired_effects,
    planning_convergence_diagnostics,
)
from brsr_tdoa.brsr_waveform import WaveformConfig


def _planning_rows(offset: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state_seed in (11, 12):
        for state_index in range(2):
            state_id = f"{state_seed}:{state_index}"
            for snr_db in (-5.0, 5.0):
                for packet_index in range(8):
                    for configuration, base in (("0 1 2", 0.2), ("0 1 3", 1.0)):
                        rows.append(
                            {
                                "scene": SCENE_S1,
                                "state_seed": state_seed,
                                "state_id": state_id,
                                "snr_db": snr_db,
                                "packet_index": packet_index,
                                "payload_bits": 24,
                                "configuration": configuration,
                                "squared_error_samples2": (
                                    base + offset + 0.001 * packet_index
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def _evaluation_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state_seed in (21, 22):
        for state_index in range(2):
            state_id = f"{state_seed}:{state_index}"
            for snr_db in (-5.0, 5.0):
                for packet_index in range(2):
                    target_loss = 1.0 + 0.01 * packet_index
                    for method in (TARGET_METHOD, *BASELINES):
                        loss = target_loss if method == TARGET_METHOD else 1.4
                        rows.append(
                            {
                                "scene": SCENE_S1,
                                "state_seed": state_seed,
                                "state_index": state_index,
                                "state_id": state_id,
                                "snr_db": snr_db,
                                "packet_index": packet_index,
                                "method": method,
                                "squared_error_samples2": loss,
                            }
                        )
    return pd.DataFrame(rows)


class ReplicationTests(unittest.TestCase):
    def test_planning_convergence_selects_smallest_prefix(self) -> None:
        selection = _planning_rows(0.0)
        check = _planning_rows(0.1)
        details, summary, sensitivity, selected = (
            planning_convergence_diagnostics(
                selection,
                check,
                candidate_counts=(2, 4),
                reference_count=8,
                equivalence_margin=0.01,
                sensitivity_margins=(0.005, 0.01),
                bootstrap_repetitions=50,
                bootstrap_seed=7,
            )
        )
        self.assertEqual(selected, 2)
        self.assertTrue(summary["primary_equivalence_pass"].all())
        self.assertTrue(sensitivity["equivalence_pass"].all())
        self.assertEqual(len(details), 16)
        self.assertEqual(
            summary["reference_crossfit_regret_ci_high"].nunique(),
            1,
        )

    def test_paired_effects_detects_positive_gain(self) -> None:
        effects = paired_effects(
            _evaluation_rows(),
            bootstrap_repetitions=50,
            bootstrap_seed=9,
        )
        self.assertEqual(set(effects["baseline"]), set(BASELINES))
        self.assertTrue((effects["mse_gain"] > 0.0).all())
        self.assertTrue(effects["all_state_seeds_positive"].all())

    def test_gate_requires_all_effects(self) -> None:
        effects = paired_effects(
            _evaluation_rows(),
            bootstrap_repetitions=50,
            bootstrap_seed=10,
        )
        decision = gate_decision(
            mode="development",
            planning_selected_count=64,
            effects=effects,
            mechanical_checks={"mechanical": True},
        )
        self.assertEqual(decision["status"], "PASS_S1_PRECISION_REPLICATION")
        failed = effects.copy()
        failed.loc[0, "mse_gain_ci_low"] = -0.1
        decision = gate_decision(
            mode="development",
            planning_selected_count=64,
            effects=failed,
            mechanical_checks={"mechanical": True},
        )
        self.assertEqual(decision["status"], "NO_GO_S1_PRECISION_REPLICATION")

    def test_gate_stops_when_planning_does_not_converge(self) -> None:
        decision = gate_decision(
            mode="development",
            planning_selected_count=None,
            effects=None,
            mechanical_checks={"mechanical": True},
        )
        self.assertEqual(
            decision["status"],
            "NO_GO_ROUTE_PLANNING_NONCONVERGENCE",
        )

    def test_power_design_target_exceeds_historical_minimum(self) -> None:
        effect = 0.30455079267987967
        cluster_sd = 0.6935536535200854
        required = int(
            np.ceil(((1.959963984540054 + 0.8416212335729143) * cluster_sd / effect) ** 2)
        )
        self.assertLessEqual(required, 48)

    def test_mechanical_keys_include_stage_and_budget_uses_each_snr(self) -> None:
        config = CONFIGS["smoke"]

        class State:
            state_seed = 2026081001
            state_index = 0
            state_id = "2026081001:0"
            true_tau = 0.0
            true_rho = 0.8 + 0.0j

        class ConfirmationState(State):
            state_seed = 2026081101
            state_id = "2026081101:0"

        methods = (
            TARGET_METHOD,
            "BestStatic-B24",
            "SNR-only-B24",
            "BestStatic-RateMatched-L16",
            "SNR-only-RateMatched-L16",
        )
        planning_rows = []
        evaluation_rows = []
        for snr_db in config.snr_db:
            for packet_index in range(16):
                sample_id = (
                    f"{SCENE_S1}|2026081101:0|evaluation|"
                    f"{packet_index}|{snr_db:g}"
                )
                planning_rows.append(
                    {
                        "stage": "planning",
                        "scene": SCENE_S1,
                        "state_seed": 2026081101,
                        "state_id": "2026081101:0",
                        "snr_db": snr_db,
                        "packet_index": packet_index,
                        "configuration": "0 1 2",
                    }
                )
                for method in methods:
                    is_target = method == TARGET_METHOD
                    is_rate = "RateMatched" in method
                    payload = 32 if is_rate and packet_index == 0 else 24
                    evaluation_rows.append(
                        {
                            "stage": "evaluation",
                            "scene": SCENE_S1,
                            "state_seed": 2026081101,
                            "state_id": "2026081101:0",
                            "snr_db": snr_db,
                            "packet_index": packet_index,
                            "sample_id": sample_id,
                            "method": method,
                            "configuration": "0 1 2",
                            "payload_bits": 24 if is_target else payload,
                            "total_bits": 24.3125 if is_target else payload,
                            "clipping_rate": 0.0,
                            "public_scale_delta_vs_legacy": 0.0,
                        }
                    )
        calibration_frame = pd.DataFrame(
            {
                "clipping_rate": [0.0],
                "public_scale_delta_vs_legacy": [0.0],
            }
        )
        checks = _mechanical_checks(
            config=config,
            frozen_source_seeds=set(),
            calibration_states=[State()],
            confirmation_states=[ConfirmationState()],
            calibration_selection=calibration_frame,
            calibration_check=calibration_frame,
            confirmation_planning=pd.DataFrame(planning_rows),
            confirmation_evaluation=pd.DataFrame(evaluation_rows),
            selected_count=2,
            waveform=WaveformConfig(),
        )
        self.assertTrue(
            checks["confirmation_planning_and_evaluation_rows_disjoint"]
        )
        self.assertTrue(
            checks["all_rate_matched_budgets_within_integer_quantum"]
        )


if __name__ == "__main__":
    unittest.main()
