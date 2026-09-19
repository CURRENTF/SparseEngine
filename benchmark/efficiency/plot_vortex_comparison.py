"""Compatibility entrypoint for the paper comparison's combined figure."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts" / "official_experiments"
            / "sparseengine_vs_vortex" / "plot.py"),
        run_name="__main__",
    )
