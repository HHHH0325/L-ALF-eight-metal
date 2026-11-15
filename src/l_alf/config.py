from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:
    initial_data_path: Path
    oracle_model_path: Path
    output_dir: Path
    oracle_training_data_path: Path | None = None
    random_seed: int = 42
    num_iterations: int = 10
    candidate_pool_size: int = 10000000
    kdas_select_size: int = 10
    query_top_k: int = 3
    retain_top_k: int = 3
    committee_size: int = 5
    similar_examples_k: int = 3
    lambda_start: float = 0.3
    lambda_end: float = 0.7
    kdas_sigma: float = 1.5
    ucb_exploration_weight: float = 1.0
    performance_weight: float = 0.6
    performance_threshold: float = 200.0
    target_pd_improvement_mean: float = 0.06
    target_pd_improvement_std: float = 0.02
    target_cost_reduction_mean: float = 0.03
    target_cost_reduction_std: float = 0.01

    def validate(self) -> None:
        positive_ints = {
            "num_iterations": self.num_iterations,
            "candidate_pool_size": self.candidate_pool_size,
            "kdas_select_size": self.kdas_select_size,
            "query_top_k": self.query_top_k,
            "retain_top_k": self.retain_top_k,
            "committee_size": self.committee_size,
            "similar_examples_k": self.similar_examples_k,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于0")
        if self.query_top_k > self.kdas_select_size:
            raise ValueError("query_top_k 不能大于 kdas_select_size")
        if self.retain_top_k > self.query_top_k:
            raise ValueError("retain_top_k 不能大于 query_top_k")
        for name, value in {
            "lambda_start": self.lambda_start,
            "lambda_end": self.lambda_end,
            "performance_weight": self.performance_weight,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于[0, 1]")
        if self.kdas_sigma <= 0:
            raise ValueError("kdas_sigma 必须大于0")


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    root = config_path.parent
    for key in ("initial_data_path", "oracle_model_path", "output_dir"):
        value = Path(raw[key])
        raw[key] = value if value.is_absolute() else (root / value).resolve()
    if raw.get("oracle_training_data_path") is not None:
        value = Path(raw["oracle_training_data_path"])
        raw["oracle_training_data_path"] = value if value.is_absolute() else (root / value).resolve()

    config = RunConfig(**raw)
    config.validate()
    return config
