from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .baselines.common import BaselineConfig, SOURCE_DEFAULTS
from .baselines.evolutionary import run_nsga2, run_random
from .baselines.mobo import run_dg_mobo, run_mobo


RUNNERS = {
    "mobo": run_mobo,
    "dg_mobo": run_dg_mobo,
    "nsga2": run_nsga2,
    "random": run_random,
}


def _load_config(path: Path, method: str) -> BaselineConfig:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    required = ("initial_data_path", "oracle_model_path", "output_dir")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"配置缺少字段: {missing}")
    root = path.parent
    resolved = {}
    for key in required:
        value = Path(raw[key])
        resolved[key] = value if value.is_absolute() else (root / value).resolve()

    source_seed = SOURCE_DEFAULTS[method]["random_seed"]
    random_seed = raw.get("random_seed", source_seed)
    if method in {"nsga2", "random"} and random_seed is None:
        raise ValueError(f"{method} 需要整数 random_seed；原脚本默认 seed=5")
    return BaselineConfig(
        **resolved,
        new_evaluations=int(raw.get("new_evaluations", 150)),
        batch_size=int(raw.get("batch_size", 3)),
        random_seed=None if random_seed is None else int(random_seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one packaged baseline against the frozen GBR. The default budget is "
            "150 new evaluations to match saved B1 records; original source defaults "
            "are MOBO=150, DG-MOBO=300, NSGA-II/Random=600."
        )
    )
    parser.add_argument("--config", required=True, help="JSON paths and budget/seed configuration")
    parser.add_argument("--method", required=True, choices=tuple(RUNNERS))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path, args.method)
    config.validate(args.method)
    manifest = RUNNERS[args.method](config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
