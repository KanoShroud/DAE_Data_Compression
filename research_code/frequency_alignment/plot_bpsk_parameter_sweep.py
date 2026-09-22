"""BPSK/TZP-DS 绘图的兼容启动器。

正式绘图实现已经并入 ``plot_interpolation_mechanism.py``，使原插值实验和
BPSK 扩展共用同一套图形样式、保存逻辑与入口说明。本文件只兼容历史命令和
冻结结果中的脚本引用。
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


try:  # package 导入与历史直接脚本运行均需兼容。
    from .plot_interpolation_mechanism import main_bpsk_parameter_sweep_plot
except ImportError:  # pragma: no cover - 由直接脚本入口覆盖。
    from research_code.frequency_alignment.plot_interpolation_mechanism import main_bpsk_parameter_sweep_plot


if __name__ == "__main__":
    main_bpsk_parameter_sweep_plot()
