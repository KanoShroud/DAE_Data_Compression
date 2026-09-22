"""BPSK/TZP-DS 扩展的兼容启动器。

实现已经并入 ``run_interpolation_alignment.py``，以便原插值实验使用统一的
核心算子、协议入口和结果根目录。保留本文件仅用于兼容历史运行命令、PyCharm
配置及冻结结果中的脚本引用。
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
    from .run_interpolation_alignment import main_bpsk_parameter_sweep
except ImportError:  # pragma: no cover - 由直接脚本入口覆盖。
    from research_code.frequency_alignment.run_interpolation_alignment import main_bpsk_parameter_sweep


if __name__ == "__main__":
    main_bpsk_parameter_sweep()
