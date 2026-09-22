"""BRSR动作价值可观测性证书。

本包不修改现有BRSR估计器或策略，只使用冻结物理模型和全新development
种子，检验动作前可观测信息能否预测不同动作的终端TDOA平方误差。
"""

# Resolve the repository independently of the entry's directory.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / 'runtime_paths.py').is_file())
if str(_LAYOUT_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT))
from runtime_paths import install_legacy_imports as _install_layout_imports
_install_layout_imports()


