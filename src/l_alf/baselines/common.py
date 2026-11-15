from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..costs import METAL_ORDER, calculate_normalized_cost, normalize_ratio


HISTORICAL_B1_NEW_EVALUATIONS = 150
SOURCE_DEFAULTS = {
    "mobo": {"new_evaluations": 150, "random_seed": None},
    "dg_mobo": {"new_evaluations": 300, "random_seed": None},
    "nsga2": {"new_evaluations": 600, "random_seed": 5},
    "random": {"new_evaluations": 600, "random_seed": 5},
}


@dataclass(frozen=True)
class BaselineConfig:
    initial_data_path: Path
    oracle_model_path: Path
    output_dir: Path
    new_evaluations: int = HISTORICAL_B1_NEW_EVALUATIONS
    batch_size: int = 3
    random_seed: int | None = None

    def validate(self, method: str) -> None:
        if method not in SOURCE_DEFAULTS:
            raise ValueError(f"未知基线方法: {method}")
        if self.new_evaluations < 1:
            raise ValueError("new_evaluations 必须大于0")
        if self.batch_size < 1:
            raise ValueError("batch_size 必须大于0")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise ValueError("random_seed 必须是整数或 null")
        if method in {"mobo", "dg_mobo"}:
            if self.new_evaluations % self.batch_size:
                raise ValueError("MOBO/DG-MOBO 的 new_evaluations 必须能被 batch_size 整除")
        if method in {"nsga2", "random"} and self.random_seed is None:
            raise ValueError(f"{method} 需要整数 random_seed；原脚本默认 seed=5")


def load_initial_examples(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到初始数据文件: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("初始数据只支持 .xlsx, .xls 或 .csv")

    target_column = "power_density" if "power_density" in frame else "max_power"
    required = [*METAL_ORDER, target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"初始数据缺少列: {missing}")

    examples = []
    for row in frame[required].dropna().itertuples(index=False, name=None):
        ratio = normalize_ratio(np.asarray(row[:-1], dtype=float))
        power = float(row[-1])
        if not np.isfinite(power):
            raise ValueError("初始功率数据必须为有限数值")
        examples.append({
            "ratio": ratio,
            "power_density": power,
            "normalized_cost": calculate_normalized_cost(ratio),
        })
    if not examples:
        raise ValueError("初始数据不能为空")
    return examples


def load_frozen_oracle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到冻结 GBR 模型: {path}; 不会自动训练替代模型")
    import joblib

    return joblib.load(path)


def save_results(
    config: BaselineConfig,
    method: str,
    initial: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
    algorithm_settings: dict[str, Any],
) -> dict[str, Any]:
    output_dir = config.output_dir / method
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in [*initial, *evaluated]:
        ratio = normalize_ratio(np.asarray(item["ratio"], dtype=float))
        rows.append({
            "id": len(rows) + 1,
            "ratio": json.dumps(ratio.tolist()),
            "power_density": float(item["power_density"]),
            "normalized_cost": float(item["normalized_cost"]),
        })
    total_pool_path = output_dir / "total_pool.csv"
    pd.DataFrame(rows, columns=["id", "ratio", "power_density", "normalized_cost"]).to_csv(
        total_pool_path, index=False
    )

    source = SOURCE_DEFAULTS[method]
    manifest = {
        "method": method,
        "initial_data_path": str(config.initial_data_path),
        "oracle_model_path": str(config.oracle_model_path),
        "new_evaluations": len(evaluated),
        "batch_size": config.batch_size,
        "random_seed": config.random_seed,
        "source_script_default_new_evaluations": source["new_evaluations"],
        "source_script_default_random_seed": source["random_seed"],
        "historical_b1_new_evaluations": HISTORICAL_B1_NEW_EVALUATIONS,
        "configured_budget_differs_from_source_default": (
            config.new_evaluations != source["new_evaluations"]
        ),
        "note": (
            "CLI/config defaults to the 150 new evaluations in saved B1 records. "
            "Source script defaults are recorded separately and are not inferred "
            "from historical result files."
        ),
        "algorithm_settings": algorithm_settings,
        "total_pool_csv": str(total_pool_path),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def evaluate_batch(model, ratios: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(model.predict(ratios), dtype=float).reshape(-1)
    if len(values) != len(ratios) or not np.all(np.isfinite(values)):
        raise ValueError("冻结 GBR 对候选批次返回了无效预测")
    return [
        {
            "ratio": normalize_ratio(ratio),
            "power_density": float(power),
            "normalized_cost": calculate_normalized_cost(ratio),
        }
        for ratio, power in zip(ratios, values)
    ]
