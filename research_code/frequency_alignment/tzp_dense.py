"""TZP-DS 兼容性导出。

时域补零密集频谱已经并入 ``alignment_core``，使其与原网格插值算子和
连续时延估计器由同一核心模块维护。本文件只保留旧导入路径，既有测试、
结果重绘脚本和历史运行记录无需修改。
"""

from __future__ import annotations

# Resolve the repository independently of the entry's directory.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / 'runtime_paths.py').is_file())
if str(_LAYOUT_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT))
from runtime_paths import install_legacy_imports as _install_layout_imports
_install_layout_imports()


try:  # 作为 package 导入时使用相对路径；直接运行历史脚本时使用本地路径。
    from .alignment_core import (
        dense_delay_estimates,
        dense_delay_score_grid,
        direct_dense_delay_estimate,
        sample_original_grid,
        time_zero_padded_dense_spectrum,
    )
except ImportError:  # pragma: no cover - 由历史脚本的直接入口覆盖。
    from research_code.frequency_alignment.alignment_core import (
        dense_delay_estimates,
        dense_delay_score_grid,
        direct_dense_delay_estimate,
        sample_original_grid,
        time_zero_padded_dense_spectrum,
    )

__all__ = [
    "dense_delay_estimates",
    "dense_delay_score_grid",
    "direct_dense_delay_estimate",
    "sample_original_grid",
    "time_zero_padded_dense_spectrum",
]
