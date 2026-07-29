"""TZP-DS 兼容性导出。

时域补零密集频谱已经并入 ``alignment_core``，使其与原网格插值算子和
连续时延估计器由同一核心模块维护。本文件只保留旧导入路径，既有测试、
结果重绘脚本和历史运行记录无需修改。
"""

from __future__ import annotations

try:  # 作为 package 导入时使用相对路径；直接运行历史脚本时使用本地路径。
    from .alignment_core import (
        dense_delay_estimates,
        dense_delay_score_grid,
        direct_dense_delay_estimate,
        sample_original_grid,
        time_zero_padded_dense_spectrum,
    )
except ImportError:  # pragma: no cover - 由历史脚本的直接入口覆盖。
    from alignment_core import (
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
