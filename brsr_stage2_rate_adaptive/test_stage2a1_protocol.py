from __future__ import annotations

import numpy as np
import pandas as pd

from brsr_stage2_rate_adaptive.stage2a1_protocol import (
    build_block_policy_rows,
    expected_static_frontier,
)


def _keys() -> dict[str, list[object]]:
    return {
        "seed": [1, 1, 1, 1],
        "trial": [0, 1, 2, 3],
        "cluster_id": ["1:0", "1:1", "1:2", "1:3"],
        "snr_db": [0.0, 0.0, 0.0, 0.0],
    }


def test_expected_static_frontier_is_exact_convex_mixture() -> None:
    keys = _keys()
    low = pd.DataFrame(
        {
            **keys,
            "method": ["BestStatic-B24"] * 4,
            "squared_error_samples2": [4.0, 9.0, 16.0, 25.0],
            "total_bits": [24.0] * 4,
        }
    )
    high = pd.DataFrame(
        {
            **keys,
            "method": ["BestStatic-B32"] * 4,
            "squared_error_samples2": [0.0, 1.0, 4.0, 9.0],
            "total_bits": [32.0] * 4,
        }
    )
    result = expected_static_frontier(
        pd.concat([low, high], ignore_index=True),
        family="BestStatic",
        target_bits=28.0,
        label="expected",
    )
    assert np.allclose(
        result["squared_error_samples2"],
        [2.0, 5.0, 10.0, 17.0],
    )
    assert np.allclose(result["total_bits"], 28.0)


def test_block_policy_uses_one_action_and_amortizes_one_feedback() -> None:
    rows = []
    for trial in range(4):
        for position in (2, 3):
            rows.append(
                {
                    "seed": 1,
                    "trial": trial,
                    "cluster_id": f"1:{trial}",
                    "snr_db": 0.0,
                    "position": position,
                    "bin_index": 100 + position,
                    "calibrated_expected_risk_after_samples2": (
                        1.0 if position == (2 if trial < 2 else 3) else 5.0
                    ),
                    "calibrated_squared_error_samples2": (
                        float(position + trial)
                    ),
                }
            )
    actions = pd.DataFrame(rows)
    result = build_block_policy_rows(
        actions,
        block_size=2,
        policy="predicted",
    )
    assert result["selected_position"].tolist() == [2, 2, 3, 3]
    assert np.allclose(result["total_bits"], 26.5)
    assert np.allclose(result["feedback_bits_per_sample"], 2.5)
