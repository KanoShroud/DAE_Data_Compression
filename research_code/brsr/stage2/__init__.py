"""BRSR Stage 2A: feedback-aware one-step rate adaptation."""

# Resolve the repository independently of the entry's directory.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / 'runtime_paths.py').is_file())
if str(_LAYOUT_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT))
from runtime_paths import install_legacy_imports as _install_layout_imports
_install_layout_imports()


from .stage2a import (
    Stage2AConfig,
    config_for_mode,
)

__all__ = ["Stage2AConfig", "config_for_mode"]
