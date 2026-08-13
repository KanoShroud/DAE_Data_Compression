import unittest

import pandas as pd

from brsr_transition_regime.experiment import (
    FLAT_BASELINE,
    SCENE_S1,
    SCENE_S2,
    TARGET_METHOD,
    TARGET_BASELINE,
    crossfit_transition_headroom,
    expected_rate_matched_payload,
    gate_decision,
    heterogeneity_attribution,
    target_total_bits,
)


class TransitionExperimentTest(unittest.TestCase):
    def test_total_bit_accounting(self) -> None:
        self.assertAlmostEqual(target_total_bits(SCENE_S1), 24.0 + 5.0 / 16.0)
        self.assertAlmostEqual(target_total_bits(SCENE_S2), 24.0 + 7.0 / 16.0)
        self.assertAlmostEqual(
            expected_rate_matched_payload(scene=SCENE_S2, family="ContextStatic"),
            24.0 + 5.0 / 16.0,
        )
        self.assertAlmostEqual(
            expected_rate_matched_payload(scene=SCENE_S2, family="BestStatic"),
            24.0 + 7.0 / 16.0,
        )

    def test_crossfit_uses_opposite_parity(self) -> None:
        rows: list[dict[str, object]] = []
        for seed in (101, 102):
            for state_index in (0, 1):
                for snr_db in (0.0, 5.0):
                    for parity in (0, 1):
                        for config_id in ("A", "B"):
                            if state_index == 0:
                                loss = 0.0 if config_id == "A" else 100.0
                            else:
                                loss = 4.0 if config_id == "A" else 0.0
                            rows.append(
                                {
                                    "scene": SCENE_S2,
                                    "state_seed": seed,
                                    "state_id": f"{seed}-{state_index}",
                                    "context_class": "notch_mid",
                                    "snr_db": snr_db,
                                    "packet_index": parity,
                                    "payload_bits": 24.0,
                                    "configuration": config_id,
                                    "squared_error_samples2": loss,
                                }
                            )

        frame, result = crossfit_transition_headroom(
            pd.DataFrame(rows),
            primary_snr_db=(0.0, 5.0),
            bootstrap_repetitions=100,
            bootstrap_seed=7,
        )

        self.assertEqual(len(frame), 16)
        self.assertGreater(result["mse_gain"], 0.0)
        self.assertGreater(result["action_difference_rate"], 0.0)

    def test_heterogeneity_attribution_is_difference_in_differences(self) -> None:
        rows: list[dict[str, object]] = []
        for scene, baseline, target_error, baseline_error in (
            (SCENE_S1, FLAT_BASELINE, 3.0, 4.0),
            (SCENE_S2, TARGET_BASELINE, 1.0, 5.0),
        ):
            for method, error in (
                (TARGET_METHOD, target_error),
                (baseline, baseline_error),
            ):
                rows.append(
                    {
                        "scene": scene,
                        "state_seed": 1,
                        "state_index": 0,
                        "state_id": "1-0",
                        "snr_db": 0.0,
                        "packet_index": 0,
                        "method": method,
                        "squared_error_samples2": error,
                    }
                )

        frame, result = heterogeneity_attribution(
            pd.DataFrame(rows),
            primary_snr_db=(0.0,),
            bootstrap_repetitions=100,
            bootstrap_seed=11,
        )

        self.assertAlmostEqual(frame.iloc[0]["target_gain"], 4.0)
        self.assertAlmostEqual(frame.iloc[0]["flat_gain"], 1.0)
        self.assertAlmostEqual(result["mean"], 3.0)

    def test_gate_requires_all_pre_registered_components(self) -> None:
        mechanical = {"budget": True, "separation": True}
        crossfit = {
            "mse_gain": 0.2,
            "mse_gain_ci_low": 0.1,
            "all_state_seeds_positive": True,
        }
        primary = {
            "mse_gain": 0.3,
            "mse_gain_ci_low": 0.2,
            "all_state_seeds_positive": True,
        }
        attribution = {
            "mean": 0.1,
            "ci_low": 0.05,
            "all_state_seeds_positive": True,
        }

        self.assertEqual(
            gate_decision(
                mechanical_checks=mechanical,
                crossfit=crossfit,
                primary_effect=primary,
                attribution=attribution,
                smoke=False,
            )["status"],
            "GO_OBSERVABLE_POLICY_DESIGN_TRANSITION",
        )

        crossfit_failed = dict(crossfit)
        crossfit_failed["mse_gain_ci_low"] = -0.01
        self.assertEqual(
            gate_decision(
                mechanical_checks=mechanical,
                crossfit=crossfit_failed,
                primary_effect=primary,
                attribution=attribution,
                smoke=False,
            )["status"],
            "NO_GO_TRANSITION_ACTION_IDENTIFIABILITY",
        )


if __name__ == "__main__":
    unittest.main()
