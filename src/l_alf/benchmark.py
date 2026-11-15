"""统一的 L-ALF 补充实验 Benchmark。

该模块只把“候选组成提交给冻结 GBR”作为黑盒接口。所有方法共享同一
初始样本、候选可行域、batch、评价预算和随机种子；因此输出可以直接用于
B1--B5 的配对统计。LLM 方法只有在配置 API 后才会被执行。
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor

from l_alf.costs import METAL_ORDER, calculate_normalized_cost, normalize_ratio
from l_alf.pareto import sort_reference_examples
from l_alf.providers import DesignProvider, create_provider
from l_alf.sampling import AKDAS, clr_transform

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Internal IDs stay stable for scripts and manifests; result folders use these
# readable algorithm names so a CSV is self-describing without opening code.
METHOD_LABELS = {
    "A0": "LLM_Generation_LLM_Prediction",
    "A1": "Local_Perturbation_LLM_Prediction",
    "A2": "ParetoMix_LLM_Prediction",
    "A3": "LLM_Generation_3NN_Prediction",
    "A4": "LLM_Generation_GPR_Prediction",
    "A5": "Local_Perturbation_3NN_Prediction",
    "A6": "Local_Perturbation_GPR_Prediction",
}


@dataclass
class BenchmarkConfig:
    initial_data_path: Path
    oracle_model_path: Path
    output_dir: Path
    iterations: int = 30
    batch_size: int = 3
    repeats: int = 5
    candidate_pool_size: int = 10_000_000
    kdas_select_size: int = 12
    committee_size: int = 5
    kappa: float = 1.0
    kdas_sigma: float = 1.5
    kdas_sigma_values: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    lambda_start: float = 0.3
    lambda_end: float = 0.7
    preference_weight: float = 0.8
    # HV definition: high-performance threshold and cost limit.
    # These are raw-domain thresholds, not scale factors.
    pd_reference: float = 60.0
    cost_reference: float = 100.0
    thresholds: tuple[float, ...] = (60.0,)
    target_pd_improvement_mean: float = 0.06
    target_pd_improvement_std: float = 0.02
    target_cost_reduction_mean: float = 0.03
    target_cost_reduction_std: float = 0.01
    random_seed: int = 42
    metal_order: tuple[str, ...] = METAL_ORDER
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.iterations < 1 or self.batch_size < 1 or self.repeats < 1:
            raise ValueError("iterations、batch_size、repeats 必须大于 0")
        if self.candidate_pool_size < self.batch_size:
            raise ValueError("candidate_pool_size 必须不小于 batch_size")
        if self.kdas_select_size < self.batch_size:
            raise ValueError("kdas_select_size 必须不小于 batch_size")
        if not 0 <= self.preference_weight <= 1:
            raise ValueError("preference_weight 必须位于[0,1]")


def load_initial_excel(path: Path) -> list[dict[str, Any]]:
    """读取原始 examples.xlsx，同时兼容 max_power 列名。"""
    frame = pd.read_excel(path)
    target = "power_density" if "power_density" in frame else "max_power"
    missing = [m for m in METAL_ORDER if m not in frame]
    if missing or target not in frame:
        raise ValueError(f"初始数据缺少列: {missing + [target] if target not in frame else missing}")
    rows = []
    for _, row in frame.iterrows():
        ratio = normalize_ratio(row[list(METAL_ORDER)].to_numpy(dtype=float)).tolist()
        cost = calculate_normalized_cost(ratio)
        rows.append({
            "ratio": ratio,
            "power_density": float(row[target]),
            "cost": cost,
            "normalized_cost": cost,
            "source": "initial",
            "iteration": 0,
        })
    if not rows:
        raise ValueError("初始样本为空")
    return rows


def _load_oracle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到冻结 GBR 模型: {path}")
    return joblib.load(path)


def _candidate_key(ratio: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(normalize_ratio(ratio), 10))


def _history_xy(history: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray([x["ratio"] for x in history], dtype=float),
            np.asarray([x["power_density"] for x in history], dtype=float))


def _sample_pool(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.dirichlet(np.ones(len(METAL_ORDER)), size=size)


def _remove_seen(pool: np.ndarray, history: list[dict[str, Any]]) -> np.ndarray:
    seen = {_candidate_key(np.asarray(x["ratio"])) for x in history}
    return np.asarray([p for p in pool if _candidate_key(p) not in seen])


def _front_indices(ratios: np.ndarray, power: np.ndarray) -> np.ndarray:
    costs = np.asarray([calculate_normalized_cost(x) for x in ratios])
    keep = []
    for i in range(len(ratios)):
        dominated = ((power >= power[i]) & (costs <= costs[i]) &
                     ((power > power[i]) | (costs < costs[i])))
        if not np.any(dominated):
            keep.append(i)
    return np.asarray(keep, dtype=int)


def _fit_predictor(name: str, history: list[dict[str, Any]], seed: int):
    x, y = _history_xy(history)
    if len(x) < 3:
        return None
    if name in {"gp", "gpr"}:
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=0.25) + WhiteKernel(1e-5),
            normalize_y=True, random_state=seed, n_restarts_optimizer=0,
        )
    elif name == "rf":
        model = RandomForestRegressor(n_estimators=128, min_samples_leaf=2,
                                      random_state=seed, n_jobs=1)
    elif name == "xgb":
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                               max_leaf_nodes=15, random_state=seed)
    elif name == "knn":
        model = KNeighborsRegressor(n_neighbors=min(3, len(x)), weights="distance")
    else:
        raise ValueError(f"未知内部预测器: {name}")
    model.fit(x, y)
    return model


def _predict_surrogate(model, name: str, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model is None:
        return np.zeros(len(pool)), np.ones(len(pool))
    if name in {"gp", "gpr"}:
        mean, std = model.predict(pool, return_std=True)
        return np.asarray(mean), np.asarray(std)
    if name == "rf":
        values = np.asarray([tree.predict(pool) for tree in model.estimators_])
        return values.mean(axis=0), values.std(axis=0)
    if name == "knn":
        return np.asarray(model.predict(pool)), np.zeros(len(pool))
    return np.asarray(model.predict(pool)), np.zeros(len(pool))


def _scalar_score(mean: np.ndarray, std: np.ndarray, pool: np.ndarray,
                  weight: float, kappa: float) -> np.ndarray:
    cost = np.asarray([calculate_normalized_cost(x) for x in pool])
    value = mean + kappa * std
    value = (value - value.min()) / (np.ptp(value) or 1.0)
    cheap = 1.0 - (cost - cost.min()) / (np.ptp(cost) or 1.0)
    return weight * value + (1.0 - weight) * cheap


def _predict_3nn(pool: np.ndarray, history: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Use the same CLR nearest-neighbor rule as the LLM prompt, with arithmetic means."""
    history_clr = np.asarray([clr_transform(np.asarray(x["ratio"])) for x in history])
    values = np.asarray([x["power_density"] for x in history], dtype=float)
    means, spreads, neighbor_indices = [], [], []
    for ratio in pool:
        distances = np.linalg.norm(history_clr - clr_transform(ratio), axis=1)
        ids = np.argsort(distances)[: min(3, len(history))]
        labels = values[ids]
        means.append(float(np.mean(labels)))
        spreads.append(float(np.std(labels, ddof=1)) if len(labels) > 1 else 0.0)
        neighbor_indices.append([int(i) for i in ids])
    return np.asarray(means), np.asarray(spreads), neighbor_indices


def _llm_propose(pool: np.ndarray, target: np.ndarray, history: list[dict[str, Any]],
                 predictor: str, provider: DesignProvider, cfg: BenchmarkConfig,
                 rng: np.random.Generator, lam: float, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected = AKDAS(target, lam, cfg.kdas_sigma).select_points(pool, cfg.kdas_select_size)
    logs: list[dict[str, Any]] = []
    if predictor == "llm":
        values = []
        for ratio in selected:
            similar = sorted(history, key=lambda x: np.linalg.norm(
                clr_transform(np.asarray(x["ratio"])) - clr_transform(ratio)))[:3]
            pred = provider.predict_committee(ratio.tolist(), similar, cfg.committee_size)
            mean, std = float(np.mean(pred)), float(np.std(pred, ddof=1) if len(pred) > 1 else 0.0)
            values.append((mean, std))
    elif predictor == "knn":
        mean, std, neighbor_indices = _predict_3nn(selected, history)
        values = list(zip(mean, std))
    else:
        model = _fit_predictor(predictor, history, seed)
        mean, std = _predict_surrogate(model, predictor, selected)
        values = list(zip(mean, std))
    means = np.asarray([v[0] for v in values]); stds = np.asarray([v[1] for v in values])
    score = _scalar_score(means, stds, selected, cfg.preference_weight, cfg.kappa)
    order = np.argsort(score)[::-1][:cfg.batch_size]
    selected_indices = set(int(i) for i in order)
    for i, (ratio, (mean, std)) in enumerate(zip(selected, values)):
        logs.append({"ratio": ratio.tolist(), "pred_mean": float(mean), "pred_std": float(std),
                     "candidate_score": float(score[i]), "selected": bool(i in selected_indices)})
    return selected[order], logs


def _generate_target(mode: str, history: list[dict[str, Any]], provider: DesignProvider | None,
                     rng: np.random.Generator, weight: float) -> np.ndarray:
    if mode == "llm":
        if provider is None:
            raise RuntimeError("LLM 方法需要 API；请先配置 .env")
        references = sort_reference_examples(history, front_limit=4)
        leader = max(references, key=lambda x: x["power_density"])
        pd_gain = abs(rng.normal(0.06, 0.02))
        cost_drop = abs(rng.normal(0.03, 0.01))
        target = provider.generate_ratio(leader["power_density"] * (1.0 + pd_gain),
                                         max(leader["cost"] * (1.0 - cost_drop), 0.1),
                                         references[:5])
        return normalize_ratio(target)
    if mode == "local":
        # Weight-aware anchor: use only already observed history labels.
        powers = np.asarray([x["power_density"] for x in history], dtype=float)
        costs = np.asarray([x["cost"] for x in history], dtype=float)
        p = (powers - powers.min()) / (np.ptp(powers) or 1.0)
        cheap = 1.0 - (costs - costs.min()) / (np.ptp(costs) or 1.0)
        anchor = history[int(np.argmax(weight * p + (1.0 - weight) * cheap))]["ratio"]
        alpha = np.asarray(anchor) * 40.0 + 0.25
        return rng.dirichlet(alpha)
    if mode == "pareto_mix":
        powers = np.asarray([x["power_density"] for x in history], dtype=float)
        front_indices = _front_indices(np.asarray([x["ratio"] for x in history]), powers)
        front = [history[int(i)] for i in front_indices]
        if len(front) < 2:
            front = sorted(history, key=lambda x: x["power_density"], reverse=True)[:2]
        first = min(front, key=lambda x: x["cost"])
        second = max(front, key=lambda x: x["power_density"])
        if np.allclose(first["ratio"], second["ratio"]):
            second = front[1 % len(front)]
        a = float(rng.uniform(0.2, 0.8))
        return normalize_ratio(a * np.asarray(first["ratio"]) + (1.0 - a) * np.asarray(second["ratio"]))
    raise ValueError(f"未知生成器: {mode}")


def propose(method: str, history: list[dict[str, Any]], cfg: BenchmarkConfig,
            rng: np.random.Generator, provider: DesignProvider | None,
            iteration: int, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pool = _remove_seen(_sample_pool(rng, cfg.candidate_pool_size), history)
    if len(pool) < cfg.batch_size:
        pool = _sample_pool(rng, cfg.candidate_pool_size + cfg.batch_size)
    lam = cfg.lambda_start if cfg.iterations == 1 else cfg.lambda_start + (
        cfg.lambda_end - cfg.lambda_start) * iteration / (cfg.iterations - 1)
    logs: list[dict[str, Any]] = []
    if method == "random":
        return pool[:cfg.batch_size], logs
    if method == "nsga2":
        if len(history) < 4:
            return pool[:cfg.batch_size], logs
        x, y = _history_xy(history); front = x[_front_indices(x, y)]
        children = []
        while len(children) < cfg.batch_size:
            a, b = front[int(rng.integers(len(front)))] , front[int(rng.integers(len(front)))]
            mix = rng.uniform() * a + (1 - rng.uniform()) * b
            children.append(normalize_ratio(np.maximum(mix + rng.normal(0, 0.03, 8), 1e-8)))
        return np.asarray(children), logs
    if method in {"mobo", "dg_mobo", "qlognehvi", "qlognparego", "rf", "xgb"}:
        predictor = "gp" if method in {"mobo", "dg_mobo", "qlognehvi", "qlognparego"} else method
        model = _fit_predictor(predictor, history, seed)
        mean, std = _predict_surrogate(model, predictor, pool)
        if method == "qlognparego":
            weight = float(rng.uniform(0.2, 0.8))
        else:
            weight = cfg.preference_weight
        score = _scalar_score(mean, std, pool, weight, cfg.kappa)
        if method == "qlognehvi" and len(history) >= 3:
            score += 0.2 * std
        if method == "dg_mobo":
            chosen = []
            remaining = np.arange(len(pool))
            for _ in range(cfg.batch_size):
                if not len(remaining): break
                idx = remaining[int(np.argmax(score[remaining]))]; chosen.append(idx)
                dist = np.linalg.norm(pool[remaining] - pool[idx], axis=1)
                score[remaining] -= 0.1 * np.exp(-dist / 0.15)
                remaining = remaining[remaining != idx]
            return pool[np.asarray(chosen)], logs
        return pool[np.argsort(score)[-cfg.batch_size:][::-1]], logs
    if method.startswith("akdas_"):
        target = _generate_target("local", history, provider, rng, cfg.preference_weight)
        variant = method.split("_", 1)[1]
        sigma = cfg.kdas_sigma
        if variant.startswith("fixed"):
            lam = float(variant.split("-")[-1])
        if variant.startswith("sigma-"):
            sigma = float(variant.split("-")[-1])
        if variant == "euclidean":
            # Euclidean K-DAS is retained as an explicit reviewer-requested control.
            distance = np.linalg.norm(pool - target, axis=1)
            return pool[np.argsort(distance)[:cfg.batch_size]], logs
        return AKDAS(target, lam, sigma).select_points(pool, cfg.batch_size), logs
    if method.startswith("stratified_"):
        mode = method.split("_", 1)[1]
        if mode == "off":
            return pool[:cfg.batch_size], logs
        x, y = _history_xy(history)
        target = x[_front_indices(x, y)][0] if len(history) >= 3 else pool[0]
        return AKDAS(target, lam, cfg.kdas_sigma).select_points(pool, cfg.batch_size), logs
    if method in {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}:
        mapping = {
            "A0": ("llm", "llm"), "A1": ("local", "llm"),
            "A2": ("pareto_mix", "llm"), "A3": ("llm", "knn"),
            "A4": ("llm", "gpr"), "A5": ("local", "knn"),
            "A6": ("local", "gpr"),
        }
        generator, predictor = mapping[method]
        target = _generate_target(generator, history, provider, rng, cfg.preference_weight)
        return _llm_propose(pool, target, history, predictor, provider, cfg, rng, lam, seed)
    raise ValueError(f"未知 Benchmark 方法: {method}")


def hypervolume(history: list[dict[str, Any]], cfg: BenchmarkConfig) -> float:
    """Final HV for P >= pd_reference and cost <= cost_reference.

    Cost is normalized by the fixed cost limit only to express the horizontal
    axis on [0, 1]. Power remains in its original units above the threshold.
    """
    if not history:
        return 0.0
    rows = sorted(
        (x["cost"] / cfg.cost_reference,
         max(0.0, x["power_density"] - cfg.pd_reference))
        for x in history
        if np.isfinite(x["cost"]) and np.isfinite(x["power_density"])
        and x["cost"] <= cfg.cost_reference
    )
    best = 0.0
    area = 0.0
    prev_cost = 0.0
    for cost, power in rows:
        if power > best:
            area += (cost - prev_cost) * best
            prev_cost, best = cost, power
    return float(area + (1.0 - prev_cost) * best)


def _record_metrics(history: list[dict[str, Any]], cfg: BenchmarkConfig) -> dict[str, Any]:
    valid = [x for x in history
             if np.isfinite(x["cost"]) and np.isfinite(x["power_density"])]
    under_cost = [x["power_density"] for x in valid
                  if x["cost"] <= cfg.cost_reference]
    target_indices = [i for i, x in enumerate(valid, start=1)
                      if x["cost"] <= cfg.cost_reference
                      and x["power_density"] >= cfg.pd_reference]
    first_evaluation = target_indices[0] if target_indices else None
    first_iteration = None
    if first_evaluation is not None:
        first_iteration = valid[first_evaluation - 1].get("iteration", 0)
    return {
        "HV": hypervolume(history, cfg),
        "max_power_under_cost_limit": max(under_cost) if under_cost else None,
        "target_success": bool(target_indices),
        "first_target_evaluation": first_evaluation,
        "first_target_iteration": first_iteration,
    }


def _write_total_pool(history: list[dict[str, Any]], path: Path) -> None:
    """Atomically save the current history with the original total_pool.csv schema."""
    rows = []
    for index, item in enumerate(history, start=1):
        rows.append({
            "id": index,
            "ratio": json.dumps([float(x) for x in item["ratio"]], ensure_ascii=False),
            "power_density": float(item["power_density"]),
            "normalized_cost": float(item["cost"]),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=["id", "ratio", "power_density", "normalized_cost"]).to_csv(
        temporary, index=False, encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_config_hash(cfg: BenchmarkConfig) -> str:
    payload = _jsonable(cfg)
    payload.pop("output_dir", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _restore_repeat(history: list[dict[str, Any]], csv_path: Path, checkpoint_path: Path,
                    suite: str, method: str, repeat: int, seed: int, cfg: BenchmarkConfig,
                    rng: np.random.Generator) -> int:
    if not checkpoint_path.exists():
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
            expected = len(history) + cfg.iterations * cfg.batch_size
            if len(frame) == expected:
                print(f"skip complete {csv_path.parent}", flush=True)
                return cfg.iterations
            raise RuntimeError(f"存在不完整CSV但缺少续跑检查点，拒绝覆盖: {csv_path}")
        return 0
    if not csv_path.exists():
        raise RuntimeError(f"续跑检查点存在但CSV缺失: {checkpoint_path}")

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "suite": suite.lower(), "method": method, "repeat": repeat + 1, "seed": seed,
        "preference_weight": cfg.preference_weight, "iterations": cfg.iterations,
        "batch_size": cfg.batch_size, "config_hash": _checkpoint_config_hash(cfg),
    }
    if any(checkpoint.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError(f"续跑检查点与当前实验配置不一致: {checkpoint_path}")
    completed = int(checkpoint["completed_iterations"])
    if not 0 <= completed <= cfg.iterations:
        raise RuntimeError(f"续跑检查点轮数无效: {completed}")

    frame = pd.read_csv(csv_path)
    expected_rows = len(history) + completed * cfg.batch_size
    if len(frame) < expected_rows:
        raise RuntimeError(f"CSV少于检查点记录行数: {csv_path}")
    for offset, row in enumerate(frame.iloc[len(history):expected_rows].itertuples(index=False), start=0):
        ratio = json.loads(row.ratio)
        cost = float(row.normalized_cost)
        history.append({
            "ratio": ratio, "power_density": float(row.power_density),
            "cost": cost, "normalized_cost": cost, "source": method,
            "iteration": offset // cfg.batch_size + 1,
        })
    rng.bit_generator.state = checkpoint["rng_state"]
    if len(frame) > expected_rows:
        _write_total_pool(history, csv_path)
    return completed


def run_suite(cfg: BenchmarkConfig, suite: str, methods: list[str],
              provider: DesignProvider | None = None,
              repeat_indices: list[int] | None = None) -> dict[str, Any]:
    cfg.validate(); cfg.output_dir.mkdir(parents=True, exist_ok=True)
    oracle = _load_oracle(cfg.oracle_model_path)
    initial = load_initial_excel(cfg.initial_data_path)
    rows: list[dict[str, Any]] = []; metrics: list[dict[str, Any]] = []
    repeats_to_run = repeat_indices if repeat_indices is not None else list(range(cfg.repeats))
    for method in methods:
        for repeat in repeats_to_run:
            seed = cfg.random_seed + repeat
            rng = np.random.default_rng(seed)
            history = [dict(x) for x in initial]
            method_dir = METHOD_LABELS.get(method, method)
            repeat_dir = cfg.output_dir / suite.lower() / method_dir / f"weight_{cfg.preference_weight:g}" / f"repeat_{repeat + 1:02d}"
            total_pool_path = repeat_dir / "total_pool.csv"
            checkpoint_path = repeat_dir / ".run_checkpoint.json"
            repeat_metrics: list[dict[str, Any]] = []
            repeat_queries: list[dict[str, Any]] = []
            start_iteration = _restore_repeat(history, total_pool_path, checkpoint_path,
                                              suite, method, repeat, seed, cfg, rng)
            if start_iteration == cfg.iterations:
                checkpoint_path.unlink(missing_ok=True)
                continue
            print(f"resume {suite}/{method_dir}/weight_{cfg.preference_weight:g}/repeat_{repeat + 1:02d} "
                  f"from iteration {start_iteration + 1}", flush=True)
            for iteration in range(start_iteration, cfg.iterations):
                proposed, proposal_logs = propose(method, history, cfg, rng, provider, iteration, seed)
                pred_by_key = {_candidate_key(np.asarray(x["ratio"])): x for x in proposal_logs}
                predictions = np.asarray(oracle.predict(np.asarray(proposed)), dtype=float)
                for batch_index, (ratio, power) in enumerate(zip(proposed, predictions)):
                    cost = calculate_normalized_cost(ratio)
                    item = {"ratio": ratio.tolist(), "power_density": float(power),
                            "cost": cost, "normalized_cost": cost, "source": method,
                            "iteration": iteration + 1}
                    history.append(item)
                    p = pred_by_key.get(_candidate_key(ratio), {})
                    rows.append({"suite": suite, "method": method, "repeat": repeat + 1,
                                 "iteration": iteration + 1, "batch_index": batch_index + 1,
                                 **{m: float(ratio[i]) for i, m in enumerate(METAL_ORDER)},
                                 "pred_mean": p.get("pred_mean"), "pred_std": p.get("pred_std"),
                                 "power_density": float(power), "normalized_cost": item["cost"]})
                m = _record_metrics(history, cfg)
                metric_row = {"suite": suite, "method": method, "repeat": repeat + 1,
                              "iteration": iteration + 1, **m}
                metrics.append(metric_row); repeat_metrics.append(metric_row)
                for q in proposal_logs:
                    repeat_queries.append({"iteration": iteration + 1, **q})
                _write_total_pool(history, total_pool_path)
                _write_checkpoint(checkpoint_path, {
                    "suite": suite.lower(), "method": method, "repeat": repeat + 1,
                    "seed": seed, "preference_weight": cfg.preference_weight,
                    "iterations": cfg.iterations, "batch_size": cfg.batch_size,
                    "config_hash": _checkpoint_config_hash(cfg),
                    "completed_iterations": iteration + 1,
                    "rng_state": rng.bit_generator.state,
                })
                if (iteration + 1) % 5 == 0 or iteration + 1 == cfg.iterations:
                    print(f"progress {method_dir}/weight_{cfg.preference_weight:g}/repeat_{repeat + 1:02d}: "
                          f"{iteration + 1}/{cfg.iterations}", flush=True)
            checkpoint_path.unlink(missing_ok=True)
            print(f"saved {suite}/{method_dir}/weight_{cfg.preference_weight:g}/repeat_{repeat + 1:02d}", flush=True)
    # Deliberately keep formal algorithm outputs minimal: one total_pool.csv per run.
    return {"suite": suite, "methods": methods, "config": _jsonable(cfg), "rows": len(rows)}


def run_b5(cfg: BenchmarkConfig, provider: DesignProvider | None) -> dict[str, Any]:
    if provider is None:
        raise RuntimeError("B5 需要真实 LLM provider")
    oracle = _load_oracle(cfg.oracle_model_path); history = load_initial_excel(cfg.initial_data_path)
    rng = np.random.default_rng(cfg.random_seed); pool = _sample_pool(rng, min(100, cfg.candidate_pool_size))
    rows = []
    for i, ratio in enumerate(pool):
        actual = float(oracle.predict(ratio.reshape(1, -1))[0])
        similar = sorted(history, key=lambda x: np.linalg.norm(clr_transform(np.asarray(x["ratio"])) - clr_transform(ratio)))[:3]
        pred = provider.predict_committee(ratio.tolist(), similar, cfg.committee_size)
        for j, value in enumerate(pred, 1):
            rows.append({"candidate": i, "response": j, "llm_prediction": value,
                         "actual_gbr": actual, "absolute_error": abs(value - actual),
                         "response_mean": np.mean(pred), "response_std": np.std(pred, ddof=1)})
    frame = pd.DataFrame(rows); frame.to_csv(cfg.output_dir / "b5_prediction_diagnostics.csv", index=False)
    summary = frame.groupby("candidate").first()
    result = {"candidates": int(len(summary)), "mae": float(frame["absolute_error"].mean()),
              "rmse": float(np.sqrt(np.mean((frame["llm_prediction"] - frame["actual_gbr"]) ** 2))),
              "std_error_correlation": float(summary["response_std"].corr(summary["absolute_error"]))}
    (cfg.output_dir / "b5_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_b0(cfg: BenchmarkConfig) -> dict[str, Any]:
    """登记冻结 GBR、输入文件哈希和模型基本信息。"""
    model = _load_oracle(cfg.oracle_model_path)
    result = {
        "initial_data": str(cfg.initial_data_path),
        "oracle_model": str(cfg.oracle_model_path),
        "initial_sha256": hashlib.sha256(cfg.initial_data_path.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(cfg.oracle_model_path.read_bytes()).hexdigest(),
        "model_class": type(model).__name__,
        "model_params": model.get_params() if hasattr(model, "get_params") else {},
        "n_features_in": getattr(model, "n_features_in_", None),
        "feature_order": list(METAL_ORDER),
        "note": "模型被当作冻结黑盒；该登记不构成对新组成的外部实验验证。",
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "b0_model_provenance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


def run_b6(cfg: BenchmarkConfig) -> dict[str, Any]:
    """对组成闭合、成本、HV和模型输出做可重复的数值核查。"""
    initial = load_initial_excel(cfg.initial_data_path)
    sums = [float(np.sum(x["ratio"])) for x in initial]
    model = _load_oracle(cfg.oracle_model_path)
    probe = np.asarray([x["ratio"] for x in initial], dtype=float)
    pred = np.asarray(model.predict(probe), dtype=float)
    result = {
        "initial_rows": len(initial),
        "ratio_sum_min": min(sums), "ratio_sum_max": max(sums),
        "all_nonnegative": bool(np.all(probe >= 0)),
        "cost_range": [min(x["cost"] for x in initial), max(x["cost"] for x in initial)],
        "oracle_probe_min": float(pred.min()), "oracle_probe_max": float(pred.max()),
        "hv_roi": {"power_threshold": cfg.pd_reference,
                   "cost_limit": cfg.cost_reference},
        "warning": "零值 epsilon、完整精度候选和论文正文数值仍需与最终运行日志逐项核对。",
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "b6_numeric_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_b7(cfg: BenchmarkConfig) -> dict[str, Any]:
    """生成可复现材料清单，供 SI/代码公开说明使用。"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(cfg.output_dir.rglob("*")):
        if path.is_file() and path.name != "b7_reproducibility_manifest.json":
            files.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {
        "protocol": {"iterations": cfg.iterations, "batch_size": cfg.batch_size,
                     "repeats": cfg.repeats, "seed": cfg.random_seed,
                     "metal_order": list(METAL_ORDER)},
        "source_files": {"initial_data": str(cfg.initial_data_path),
                         "oracle_model": str(cfg.oracle_model_path)},
        "generated_files": files,
        "disclosure": ["GBR 输出是冻结代理评价，不是新材料实验测量。",
                       "当前框架不记录总运行时间、硬件成本或 LLM API 成本。",
                       "第三方模型文件的再分发须遵守原始许可。"],
    }
    (cfg.output_dir / "b7_reproducibility_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_b4(cfg: BenchmarkConfig, methods: list[str], provider: DesignProvider | None) -> dict[str, Any]:
    """按预注册偏好权重运行完整的配对实验。"""
    weights = (0.0, 0.4, 0.6, 0.8, 1.0)
    manifests = []
    for weight in weights:
        sub = BenchmarkConfig(**{**cfg.__dict__, "preference_weight": weight,
                                 "output_dir": cfg.output_dir})
        manifests.append(run_suite(sub, f"B4_w{weight:g}", methods, provider))
    return {"suite": "B4", "weights": list(weights), "runs": manifests}


def _jsonable(cfg: BenchmarkConfig) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v)
            for k, v in cfg.__dict__.items() if k != "extra"}


SUITES = {
    "B1": ["lalf", "random", "nsga2", "mobo", "dg_mobo", "qlognehvi", "qlognparego", "rf", "xgb"],
    "B2": ["A0", "A1", "A2", "A3", "A4", "A5", "A6"],
    "B3": ["akdas_dynamic", "akdas_fixed-0.3", "akdas_fixed-0.5", "akdas_fixed-0.7",
           "akdas_euclidean", "akdas_sigma-0.5", "akdas_sigma-1.0", "akdas_sigma-1.5",
           "akdas_sigma-2.0", "akdas_sigma-3.0", "stratified_on", "stratified_off"],
}
