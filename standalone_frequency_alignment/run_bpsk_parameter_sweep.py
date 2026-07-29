"""BPSK/TZP-DS 扩展的兼容启动器。

实现已经并入 ``run_interpolation_alignment.py``，以便原插值实验使用统一的
核心算子、协议入口和结果根目录。保留本文件仅用于兼容历史运行命令、PyCharm
配置及冻结结果中的脚本引用。
"""

from __future__ import annotations

try:  # package 导入与历史直接脚本运行均需兼容。
    from .run_interpolation_alignment import main_bpsk_parameter_sweep
except ImportError:  # pragma: no cover - 由直接脚本入口覆盖。
    from run_interpolation_alignment import main_bpsk_parameter_sweep


if __name__ == "__main__":
    main_bpsk_parameter_sweep()
