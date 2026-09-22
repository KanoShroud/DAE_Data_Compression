"""Compatibility entry; experiment settings live in research_code/dae_pipeline/main.py."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "research_code/dae_pipeline/main.py"), run_name="__main__")
