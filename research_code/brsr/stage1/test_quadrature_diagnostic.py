"""BRSR阶段1数值积分收敛诊断的选择与门禁测试。"""

from __future__ import annotations

# Resolve the repository independently of the entry's directory.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / 'runtime_paths.py').is_file())
if str(_LAYOUT_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT))
from runtime_paths import install_legacy_imports as _install_layout_imports
_install_layout_imports()


import pandas as pd

from research_code.brsr.stage1.diagnose_quadrature import (
    SUMMARY_COLUMNS,
    _summarize,
    config_for_mode,
    diagnostic_status,
    ordered_converged_candidates,
    select_smallest_converged,
)


def test_diagnostic_configs_have_distinct_reference_and_anchor() -> None:
    for mode in ("smoke", "diagnostic"):
        config = config_for_mode(mode)
        assert config.reference_grid not in config.candidate_grids
        assert config.anchor_grid not in config.candidate_grids
        assert config.reference_grid != config.anchor_grid
        assert (5, 16) in config.candidate_grids


def test_select_smallest_converged_uses_hypothesis_count() -> None:
    summary = pd.DataFrame(
        [
            {
                "amplitude_nodes": 5,
                "phase_nodes": 16,
                "n_rho": 80,
                "full_pass": False,
            },
            {
                "amplitude_nodes": 5,
                "phase_nodes": 32,
                "n_rho": 160,
                "full_pass": True,
            },
            {
                "amplitude_nodes": 7,
                "phase_nodes": 16,
                "n_rho": 112,
                "full_pass": True,
            },
        ]
    )
    selected = select_smallest_converged(
        summary,
        ((5, 16), (5, 32), (7, 16)),
    )
    assert selected == (7, 16)


def test_select_smallest_converged_returns_none_without_pass() -> None:
    summary = pd.DataFrame(
        [
            {
                "amplitude_nodes": 5,
                "phase_nodes": 16,
                "n_rho": 80,
                "full_pass": False,
            }
        ]
    )
    assert select_smallest_converged(summary, ((5, 16),)) is None


def test_ordered_converged_candidates_preserves_cost_order() -> None:
    summary = pd.DataFrame(
        [
            {
                "amplitude_nodes": 5,
                "phase_nodes": 64,
                "n_rho": 320,
                "full_pass": True,
            },
            {
                "amplitude_nodes": 5,
                "phase_nodes": 32,
                "n_rho": 160,
                "full_pass": True,
            },
            {
                "amplitude_nodes": 7,
                "phase_nodes": 32,
                "n_rho": 224,
                "full_pass": True,
            },
            {
                "amplitude_nodes": 7,
                "phase_nodes": 64,
                "n_rho": 448,
                "full_pass": False,
            },
        ]
    )
    ordered = ordered_converged_candidates(
        summary,
        ((5, 32), (7, 32), (5, 64), (7, 64)),
    )
    assert ordered == ((5, 32), (7, 32), (5, 64))


def test_empty_summary_keeps_parseable_schema() -> None:
    summary = _summarize(
        pd.DataFrame(),
        estimate_tolerance=0.05,
        risk_tolerance=0.0025,
        posterior_l1_tolerance=0.10,
    )
    assert tuple(summary.columns) == SUMMARY_COLUMNS


def test_diagnostic_gate_stops_in_registered_order() -> None:
    assert (
        diagnostic_status(
            mode="diagnostic",
            reference_pass=False,
            subset_candidate_count=0,
            selected_grid=None,
        )
        == "HOLD-REFERENCE-NOT-CONVERGED"
    )
    assert (
        diagnostic_status(
            mode="diagnostic",
            reference_pass=True,
            subset_candidate_count=0,
            selected_grid=None,
        )
        == "HOLD-NO-CANDIDATE-CONVERGED"
    )
    assert (
        diagnostic_status(
            mode="diagnostic",
            reference_pass=True,
            subset_candidate_count=4,
            selected_grid=None,
        )
        == "HOLD-FULL-CALIBRATION"
    )
    assert (
        diagnostic_status(
            mode="diagnostic",
            reference_pass=True,
            subset_candidate_count=4,
            selected_grid=(7, 64),
        )
        == "QUADRATURE_DIAGNOSTIC_PASS"
    )
    assert (
        diagnostic_status(
            mode="smoke",
            reference_pass=False,
            subset_candidate_count=0,
            selected_grid=None,
        )
        == "SMOKE_ONLY"
    )
