from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from .costs import METAL_ORDER, normalize_ratio
class DesignProvider(Protocol):
    name: str

    def generate_ratio(self, target_pd: float, target_cost: float, references: list[dict]) -> list[float]: ...

    def predict_committee(self, ratio: list[float], similar_examples: list[dict], size: int) -> list[float]: ...


class OpenAIProvider:
    name = "openai"

    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "缺少openai包，请执行：pip install -r requirements.txt"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not model:
            raise RuntimeError("OpenAI模式必须设置 OPENAI_API_KEY 和 OPENAI_MODEL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        kwargs["timeout"] = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "300"))
        kwargs["max_retries"] = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        self.inverse_design_max_tokens = int(os.getenv("OPENAI_INVERSE_DESIGN_MAX_TOKENS", "120"))
        self.prediction_max_tokens = int(os.getenv("OPENAI_PREDICTION_MAX_TOKENS", "20"))
        self.prompt_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"

    def _prompt(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8")

    def _complete(self, prompt: str, max_tokens: int, temperature: float | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_ratio(self, target_pd: float, target_cost: float, references: list[dict]) -> list[float]:
        reference_lines = []
        for item in references[:5]:
            ratio = item.get("catalyst_ratio", item.get("ratio"))
            cost = item.get("normalized_cost", item.get("cost"))
            if ratio is None or cost is None:
                raise ValueError("参考样本必须包含 ratio/catalyst_ratio 与 cost/normalized_cost")
            reference_lines.append(
                f"ratio={ratio}; PD={item['power_density']:.4f}; "
                f"cost={cost:.4f}"
            )
        prompt = self._prompt("inverse_design.txt").format(
            target_pd=target_pd,
            target_cost=target_cost,
            references="\n".join(reference_lines),
        )
        text = self._complete(prompt, max_tokens=self.inverse_design_max_tokens)
        match = re.search(r"\[([^\]]+)\]", text)
        if not match:
            raise ValueError(f"无法从LLM输出解析配比: {text!r}")
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(1))
        if len(numbers) != len(METAL_ORDER):
            raise ValueError(f"LLM输出不是8维配比: {text!r}")
        return normalize_ratio([float(value) for value in numbers]).tolist()

    def predict_committee(self, ratio: list[float], similar_examples: list[dict], size: int) -> list[float]:
        references = "\n".join(
            f"ratio={item.get('catalyst_ratio', item.get('ratio'))}; PD={item['power_density']:.4f}"
            for item in similar_examples
        )
        prompt = self._prompt("performance_prediction.txt").format(
            references=references,
            ratio=ratio,
        )
        predictions = []
        for _ in range(size):
            text = self._complete(prompt, max_tokens=self.prediction_max_tokens)
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if not match:
                raise ValueError(f"无法从LLM输出解析预测值: {text!r}")
            value = float(match.group(0))
            if not 0.0 <= value <= 500.0:
                raise ValueError(f"LLM预测超出允许范围[0, 500]: {value}")
            predictions.append(value)
        return predictions


def create_provider() -> DesignProvider:
    return OpenAIProvider()
