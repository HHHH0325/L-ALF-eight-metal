from __future__ import annotations

import numpy as np


METAL_ORDER = ("Pd", "Pt", "Cu", "Au", "Ir", "Ce", "Nb", "Cr")
METAL_PRICES = {
    "Pd": 26393.0,
    "Pt": 20325.0,
    "Cu": 4.53,
    "Au": 29298.0,
    "Ir": 23926.0,
    "Ce": 5.37,
    "Nb": 32.22,
    "Cr": 5.86,
}


def normalize_ratio(ratio: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(ratio, dtype=float)
    if values.shape != (len(METAL_ORDER),):
        raise ValueError(f"配比必须是{len(METAL_ORDER)}维向量")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("配比必须是有限非负数")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("配比总和必须大于0")
    return values / total


def calculate_normalized_cost(ratio: np.ndarray | list[float]) -> float:
    values = normalize_ratio(ratio)
    prices = np.asarray([METAL_PRICES[name] for name in METAL_ORDER], dtype=float)
    return float(np.dot(values, prices) / METAL_PRICES["Pd"] * 100.0)
