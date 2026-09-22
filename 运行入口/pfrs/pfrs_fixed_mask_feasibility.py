"""PyCharm entry for research_code/pfrs/pfrs_fixed_mask_feasibility.py; configure the implementation, not this wrapper."""

from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime_paths.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "research_code/pfrs/pfrs_fixed_mask_feasibility.py"), run_name="__main__")
