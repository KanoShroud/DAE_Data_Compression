"""Standalone PASR-TDOA research package."""

# Resolve the repository independently of the entry's directory.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / 'runtime_paths.py').is_file())
if str(_LAYOUT_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT))
from runtime_paths import install_legacy_imports as _install_layout_imports
_install_layout_imports()


from .pasr_core import (
    ActiveEstimate,
    BitLedger,
    PosteriorMode,
    PosteriorResult,
    QuantizedSpectrum,
    SpectralModel,
    build_spectral_model,
    estimate_subset,
    quantize_complex_spectrum,
    run_active_refinement,
)

__all__ = [
    "ActiveEstimate",
    "BitLedger",
    "PosteriorMode",
    "PosteriorResult",
    "QuantizedSpectrum",
    "SpectralModel",
    "build_spectral_model",
    "estimate_subset",
    "quantize_complex_spectrum",
    "run_active_refinement",
]
