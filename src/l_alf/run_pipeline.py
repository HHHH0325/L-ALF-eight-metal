from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import CatalystActiveLearningPipeline


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the L-ALF active-learning pipeline")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "lalf_pipeline.json"),
        help="JSON run configuration",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    CatalystActiveLearningPipeline(config).run()


if __name__ == "__main__":
    main()
