"""BPSK/TZP-DS 绘图的兼容启动器。

正式绘图实现已经并入 ``plot_interpolation_mechanism.py``，使原插值实验和
BPSK 扩展共用同一套图形样式、保存逻辑与入口说明。本文件只兼容历史命令和
冻结结果中的脚本引用。
"""

from __future__ import annotations

try:  # package 导入与历史直接脚本运行均需兼容。
    from .plot_interpolation_mechanism import main_bpsk_parameter_sweep_plot
except ImportError:  # pragma: no cover - 由直接脚本入口覆盖。
    from plot_interpolation_mechanism import main_bpsk_parameter_sweep_plot


if __name__ == "__main__":
    main_bpsk_parameter_sweep_plot()
