"""BRSR共享状态StateOracle证书的轻量单元测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from brsr_state_oracle_certificate.certificate import (
    PAYLOAD_24_BITS,
    PAYLOAD_32_BITS,
    SCENE_S0,
    SCENE_S1,
    SCENE_S2,
    configuration_positions,
    evaluate_configurations,
    freeze_policies,
    generate_packet_trial,
    generate_state_bank,
    gate_decision,
    planning_policy_diagnostics,
)
from brsr_tdoa.brsr_core import NestedComplexQuantizer
from brsr_tdoa.brsr_stage_b import build_stage_b_model, evaluate_fixed, make_observation
from brsr_tdoa.brsr_waveform import (
    WaveformConfig,
    candidate_fft_values,
    design_candidate_bins,
    generate_structured_trial,
    observe_trial,
    raised_cosine_power_at_bins,
)


class StateOracleCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.waveform = WaveformConfig()
        self.bins = design_candidate_bins(self.waveform)
        self.states = generate_state_bank(
            (2026080991,),
            1,
            waveform=self.waveform,
            candidate_bins=self.bins,
        )

    def test_shared_state_and_split_independence(self) -> None:
        state = self.states[0]
        planning_s1 = generate_packet_trial(
            state,
            stage="planning",
            packet_index=0,
            scene=SCENE_S1,
            waveform=self.waveform,
        )
        evaluation_s1 = generate_packet_trial(
            state,
            stage="evaluation",
            packet_index=0,
            scene=SCENE_S1,
            waveform=self.waveform,
        )
        planning_s2 = generate_packet_trial(
            state,
            stage="planning",
            packet_index=0,
            scene=SCENE_S2,
            waveform=self.waveform,
        )
        self.assertEqual(planning_s1.true_tau, state.true_tau)
        self.assertEqual(planning_s1.true_rho, state.true_rho)
        self.assertEqual(evaluation_s1.true_tau, state.true_tau)
        self.assertEqual(evaluation_s1.true_rho, state.true_rho)
        self.assertFalse(
            np.array_equal(planning_s1.source_a, evaluation_s1.source_a)
        )
        self.assertTrue(
            np.array_equal(
                planning_s1.noise_a_standard,
                planning_s2.noise_a_standard,
            )
        )
        self.assertFalse(np.array_equal(planning_s1.source_a, planning_s2.source_a))
        self.assertAlmostEqual(
            float(np.mean(np.abs(planning_s1.source_long) ** 2)),
            float(np.mean(np.abs(planning_s2.source_long) ** 2)),
            places=12,
        )

    def test_s0_does_not_reuse_shared_truth(self) -> None:
        state = self.states[0]
        first = generate_packet_trial(
            state,
            stage="planning",
            packet_index=0,
            scene=SCENE_S0,
            waveform=self.waveform,
        )
        second = generate_packet_trial(
            state,
            stage="planning",
            packet_index=1,
            scene=SCENE_S0,
            waveform=self.waveform,
        )
        self.assertNotEqual(first.true_tau, state.true_tau)
        self.assertNotEqual(first.true_tau, second.true_tau)

    def test_fast_configuration_evaluation_matches_fixed_estimator(self) -> None:
        rng = np.random.default_rng(2026080992)
        trial = generate_structured_trial(rng, self.waveform)
        y_a, y_b, sigma2 = observe_trial(trial, 0.0)
        power = np.maximum(
            raised_cosine_power_at_bins(self.bins, self.waveform),
            1e-4,
        )
        model = build_stage_b_model(
            n_time=self.waveform.n_time,
            bin_indices=self.bins,
            source_power=power,
            base_bins=self.waveform.base_bins,
            tau_min=self.waveform.tau_min,
            tau_max=self.waveform.tau_max,
            tau_step=1.0,
        )
        quantizer = NestedComplexQuantizer(max_bits=8, clip_abs=4.0)
        observation = make_observation(
            stage="test",
            seed=1,
            trial=1,
            snr_db=0.0,
            true_tau=trial.true_tau,
            true_rho=trial.true_rho,
            y_a_selected=candidate_fft_values(y_a, self.bins),
            y_b_selected=candidate_fft_values(y_b, self.bins),
            sigma2=sigma2,
            model=model,
            quantizer=quantizer,
        )
        configurations = (
            configuration_positions(model, PAYLOAD_24_BITS)[0],
            configuration_positions(model, PAYLOAD_32_BITS)[0],
        )
        fast = evaluate_configurations(
            observation,
            model=model,
            quantizer=quantizer,
            configurations=configurations,
            temperature=1.0,
        )
        for configuration, fast_row in zip(configurations, fast, strict=True):
            levels = tuple(
                4 if position in configuration else 0
                for position in range(model.n_bins)
            )
            fixed = evaluate_fixed(
                observation,
                method="test",
                levels=levels,
                quantizer=quantizer,
                model=model,
            )
            fixed_error = fixed.estimate - trial.true_tau
            self.assertAlmostEqual(
                float(fast_row["estimate_tau"]),
                fixed.estimate,
                places=11,
            )
            self.assertAlmostEqual(
                float(fast_row["squared_error_samples2"]),
                fixed_error**2,
                places=11,
            )

    def test_s0_state_oracle_freezes_to_snr_only(self) -> None:
        rows = []
        for snr in (-5.0, 10.0):
            for packet in range(2):
                for payload, configs in (
                    (PAYLOAD_24_BITS, ("0 1 2", "0 1 3")),
                    (PAYLOAD_32_BITS, ("0 1 2 3", "0 1 2 4")),
                ):
                    for index, configuration in enumerate(configs):
                        rows.append(
                            {
                                "scene": SCENE_S0,
                                "state_seed": 1,
                                "state_index": 0,
                                "state_id": "1:0",
                                "snr_db": snr,
                                "packet_index": packet,
                                "context_class": "none",
                                "payload_bits": payload,
                                "configuration": configuration,
                                "squared_error_samples2": float(index + packet),
                            }
                        )
        policies = freeze_policies(pd.DataFrame(rows))
        oracle = policies[
            policies["policy"].eq("StateOracle")
        ][["snr_db", "configuration"]].sort_values("snr_db")
        snr_only = policies[
            policies["policy"].eq("SNR-only")
            & policies["payload_bits"].eq(PAYLOAD_24_BITS)
        ][["snr_db", "configuration"]].sort_values("snr_db")
        pd.testing.assert_frame_equal(
            oracle.reset_index(drop=True),
            snr_only.reset_index(drop=True),
        )

    def test_gate_does_not_use_s0_to_open_route(self) -> None:
        rows = []
        for scene in (SCENE_S0, SCENE_S1, SCENE_S2):
            baseline_count = 6 if scene == SCENE_S2 else 4
            for index in range(baseline_count):
                rows.append(
                    {
                        "scene": scene,
                        "block_length": 1 if scene == SCENE_S0 else 4,
                        "mse_gain": -0.1 if scene == SCENE_S0 else 0.2,
                        "mse_gain_ci_low": -0.2 if scene == SCENE_S0 else 0.1,
                        "all_state_seeds_positive": scene != SCENE_S0,
                        "baseline": f"baseline-{index}",
                    }
                )
        decision = gate_decision(
            pd.DataFrame(rows),
            mechanical_checks={"all": True},
        )
        self.assertEqual(
            decision["status"],
            "GO_OBSERVABLE_POLICY_DESIGN_S2",
        )

    def test_planning_diagnostics_report_split_stability(self) -> None:
        rows = []
        for packet_index in range(4):
            for configuration, loss in (("0 1 2", 1.0), ("0 1 3", 2.0)):
                rows.append(
                    {
                        "scene": SCENE_S1,
                        "state_seed": 1,
                        "state_index": 0,
                        "state_id": "1:0",
                        "snr_db": 0.0,
                        "packet_index": packet_index,
                        "payload_bits": PAYLOAD_24_BITS,
                        "configuration": configuration,
                        "squared_error_samples2": loss,
                    }
                )
        diagnostics = planning_policy_diagnostics(pd.DataFrame(rows))
        self.assertEqual(len(diagnostics), 1)
        result = diagnostics.iloc[0]
        self.assertEqual(result["best_configuration"], "0 1 2")
        self.assertTrue(bool(result["half_a_equals_half_b"]))
        self.assertTrue(bool(result["both_halves_equal_full"]))
        self.assertEqual(int(result["n_planning_packets"]), 4)


if __name__ == "__main__":
    unittest.main()
