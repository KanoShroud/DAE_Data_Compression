from __future__ import annotations

import numpy as np
import pandas as pd

from audit_research_gates import _bootstrap_pair_summary, _mcnemar_severe


def test_bootstrap_pair_summary_detects_positive_gain() -> None:
    rows = pd.DataFrame(
        {
            "cluster": np.repeat(["a", "b", "c", "d"], 3),
            "target_sq": np.tile([1.0, 1.2, 0.8], 4),
            "baseline_sq": np.tile([4.0, 4.2, 3.8], 4),
            "target_abs": np.tile([1.0, 1.1, 0.9], 4),
            "baseline_abs": np.tile([2.0, 2.1, 1.9], 4),
        }
    )
    summary = _bootstrap_pair_summary(
        rows,
        cluster_col="cluster",
        target_sq_col="target_sq",
        baseline_sq_col="baseline_sq",
        target_abs_col="target_abs",
        baseline_abs_col="baseline_abs",
        repetitions=200,
        seed=1,
    )
    assert summary.rmse_gain > 0.0
    assert summary.mse_gain_ci_low > 0.0
    assert summary.mean_abs_gain_ci_low > 0.0
    assert summary.n_clusters == 4


def test_mcnemar_severe_uses_paired_discordance() -> None:
    result = _mcnemar_severe(
        np.array([3.0, 0.0, 0.0, 0.0]),
        np.array([3.0, 3.0, 0.0, 0.0]),
        2.0,
    )
    assert result["target_rate"] == 0.25
    assert result["baseline_rate"] == 0.50
    assert result["target_only_bad"] == 0
    assert result["baseline_only_bad"] == 1
