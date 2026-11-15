from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .costs import METAL_ORDER, calculate_normalized_cost, normalize_ratio


def _read_and_validate(path: str | Path, target_column: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的初始数据格式: {path.suffix}; 仅支持 CSV/Excel")
    if target_column == "power_density" and target_column not in frame and "max_power" in frame:
        frame = frame.rename(columns={"max_power": "power_density"})
    required = [*METAL_ORDER, target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    frame = frame[required].dropna().copy()
    ratios = np.vstack([normalize_ratio(row) for row in frame[list(METAL_ORDER)].to_numpy()])
    frame.loc[:, list(METAL_ORDER)] = ratios
    frame[target_column] = pd.to_numeric(frame[target_column], errors="raise")
    return frame


def load_initial_examples(path: str | Path) -> list[dict]:
    frame = _read_and_validate(path, "power_density")
    examples: list[dict] = []
    for _, row in frame.iterrows():
        ratio = row[list(METAL_ORDER)].to_numpy(dtype=float)
        examples.append(
            {
                "catalyst_ratio": ratio.tolist(),
                "power_density": float(row["power_density"]),
                "normalized_cost": calculate_normalized_cost(ratio),
                "source": "initial",
                "iteration": 0,
            }
        )
    if not examples:
        raise ValueError("初始数据不能为空")
    return examples
