"""独立验证路线的简洁结果图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def _configure_chinese_font() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_summary_figures(trials: pd.DataFrame, output_directory: Path) -> list[Path]:
    _configure_chinese_font()
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    grouped = (
        trials.groupby(["method", "overlap_fraction"], dropna=False)["absolute_error_samples"]
        .mean()
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for method, method_rows in grouped.groupby("method"):
        method_rows = method_rows.sort_values("overlap_fraction")
        axis.plot(
            method_rows["overlap_fraction"],
            method_rows["absolute_error_samples"],
            marker="o",
            label=method,
        )
    axis.set_xlabel("双站实际选频重叠比例")
    axis.set_ylabel("TDOA 平均绝对误差（采样点）")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = figure_directory / "tdoa_mae_vs_overlap.svg"
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)
    return saved
