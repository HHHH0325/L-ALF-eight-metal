from __future__ import annotations

import numpy as np


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    """Return a mask for a minimization Pareto front."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Pareto输入必须是二维数组")
    mask = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominated = np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        if np.any(dominated):
            mask[index] = False
    return mask


def sort_reference_examples(examples: list[dict], front_limit: int = 4) -> list[dict]:
    if not examples:
        return []
    objectives = np.asarray(
        [[-item["power_density"], item["normalized_cost"]] for item in examples],
        dtype=float,
    )
    mask = pareto_mask_minimize(objectives)
    front = sorted(
        [examples[i] for i in range(len(examples)) if mask[i]],
        key=lambda item: item["power_density"],
        reverse=True,
    )
    non_front = sorted(
        [examples[i] for i in range(len(examples)) if not mask[i]],
        key=lambda item: item["power_density"],
        reverse=True,
    )
    return front[:front_limit] + sorted(
        front[front_limit:] + non_front,
        key=lambda item: item["power_density"],
        reverse=True,
    )


def calculate_targets(examples: list[dict], config, rng: np.random.Generator) -> tuple[float, float]:
    leader = max(examples, key=lambda item: item["power_density"])
    pd_gain = abs(rng.normal(config.target_pd_improvement_mean, config.target_pd_improvement_std))
    cost_drop = abs(rng.normal(config.target_cost_reduction_mean, config.target_cost_reduction_std))
    target_pd = min(float(leader["power_density"]) * (1.0 + pd_gain), 400.0)
    target_cost = max(float(leader["normalized_cost"]) * (1.0 - cost_drop), 0.1)
    return target_pd, target_cost


def _weighted_score(candidate: dict, bounds: dict, weight: float) -> float:
    ucb_range = bounds["max_ucb"] - bounds["min_ucb"] or 1.0
    cost_range = bounds["max_cost"] - bounds["min_cost"] or 1.0
    norm_ucb = (candidate["ucb_score"] - bounds["min_ucb"]) / ucb_range
    norm_cost = (candidate["cost"] - bounds["min_cost"]) / cost_range
    return float(weight * norm_ucb - (1.0 - weight) * norm_cost)


def select_pareto_candidates(candidates: list[dict], top_k: int, weight: float) -> list[dict]:
    if not candidates:
        return []
    ucb_values = np.asarray([item["ucb_score"] for item in candidates])
    cost_values = np.asarray([item["cost"] for item in candidates])
    bounds = {
        "min_ucb": float(ucb_values.min()),
        "max_ucb": float(ucb_values.max()),
        "min_cost": float(cost_values.min()),
        "max_cost": float(cost_values.max()),
    }
    remaining = list(candidates)
    selected: list[dict] = []
    while remaining and len(selected) < top_k:
        objectives = np.asarray([[-item["ucb_score"], item["cost"]] for item in remaining])
        mask = pareto_mask_minimize(objectives)
        front = [remaining[i] for i in range(len(remaining)) if mask[i]]
        front.sort(key=lambda item: _weighted_score(item, bounds, weight), reverse=True)
        selected.extend(front[: top_k - len(selected)])
        remaining = [remaining[i] for i in range(len(remaining)) if not mask[i]]
    return selected
