from __future__ import annotations

import joblib
from pathlib import Path


def load_fixed_oracle(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到冻结GBR模型: {model_path}. "
            "请从原作者来源获取已训练模型；本项目不会训练替代模型。"
        )
    return joblib.load(model_path)
