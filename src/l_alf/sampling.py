from __future__ import annotations

import numpy as np


def clr_transform(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    was_vector = array.ndim == 1
    if was_vector:
        array = array.reshape(1, -1)
    array = np.maximum(array, epsilon)
    array = array / array.sum(axis=1, keepdims=True)
    logged = np.log(array)
    transformed = logged - logged.mean(axis=1, keepdims=True)
    return transformed[0] if was_vector else transformed


def find_similar_examples(ratio: list[float], examples: list[dict], k: int) -> list[dict]:
    target = clr_transform(np.asarray(ratio))
    scored = []
    for example in examples:
        distance = float(np.linalg.norm(target - clr_transform(np.asarray(example["catalyst_ratio"]))))
        scored.append((distance, example))
    scored.sort(key=lambda pair: pair[0])
    return [example for _, example in scored[:k]]


class AKDAS:
    def __init__(self, target_point: np.ndarray, lambda_param: float, sigma: float):
        self.target_clr = clr_transform(np.asarray(target_point, dtype=float))
        self.lambda_param = float(lambda_param)
        self.sigma = float(sigma)

    def _kernel(self, matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
        distance_sq = np.sum((matrix - point) ** 2, axis=1)
        return np.exp(-distance_sq / (2.0 * self.sigma**2))

    def select_points(self, candidates: np.ndarray, count: int) -> np.ndarray:
        raw = np.asarray(candidates, dtype=float)
        transformed = clr_transform(raw)
        relevance = self._kernel(transformed, self.target_clr)
        diversity_penalty = np.zeros(len(raw), dtype=float)
        available = np.ones(len(raw), dtype=bool)
        selected: list[int] = []
        for _ in range(min(count, len(raw))):
            scores = self.lambda_param * relevance - (1.0 - self.lambda_param) * diversity_penalty
            scores[~available] = -np.inf
            best_index = int(np.argmax(scores))
            selected.append(best_index)
            available[best_index] = False
            similarity = self._kernel(transformed, transformed[best_index])
            diversity_penalty = np.maximum(diversity_penalty, similarity)
        return raw[selected]
