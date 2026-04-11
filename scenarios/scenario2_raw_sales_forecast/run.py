"""Backward-compatible entrypoint for the forecast baseline block.

This scenario script now delegates to `baselines/forecast_block/run.py` so that
baseline experiments are centralized under `baselines/`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    baseline_run = Path(__file__).resolve().parents[2] / "baselines" / "forecast_block" / "run.py"
    runpy.run_path(str(baseline_run), run_name="__main__")


if __name__ == "__main__":
    main()
