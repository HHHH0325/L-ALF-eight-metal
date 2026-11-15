from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig
from .costs import METAL_ORDER, calculate_normalized_cost
from .data import load_initial_examples
from .model import load_fixed_oracle
from .pareto import calculate_targets, select_pareto_candidates, sort_reference_examples
from .providers import DesignProvider, create_provider
from .sampling import AKDAS, find_similar_examples


class CatalystActiveLearningPipeline:
    def __init__(self, config: RunConfig, provider: DesignProvider | None = None):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.examples = load_initial_examples(config.initial_data_path)
        self.oracle = load_fixed_oracle(config.oracle_model_path)
        self.provider = provider or create_provider()
        self.iteration_log: list[dict] = []
        self.query_log: list[dict] = []
        self.generated_log: list[dict] = []
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _current_lambda(self, iteration_index: int) -> float:
        if self.config.num_iterations == 1:
            return self.config.lambda_start
        progress = iteration_index / (self.config.num_iterations - 1)
        return self.config.lambda_start + (
            self.config.lambda_end - self.config.lambda_start
        ) * progress

    def _score_and_retain(self, queried: list[dict]) -> list[dict]:
        power = np.asarray([item["power_density"] for item in queried], dtype=float)
        costs = np.asarray([item["normalized_cost"] for item in queried], dtype=float)
        power_range = float(np.ptp(power)) or 1.0
        cost_range = float(np.ptp(costs)) or 1.0
        for item in queried:
            norm_power = (item["power_density"] - float(power.min())) / power_range
            norm_cost = (item["normalized_cost"] - float(costs.min())) / cost_range
            item["final_score"] = (
                self.config.performance_weight * norm_power
                - (1.0 - self.config.performance_weight) * norm_cost
            )
        return sorted(queried, key=lambda item: item["final_score"], reverse=True)[
            : self.config.retain_top_k
        ]

    def _save(self) -> None:
        pool_rows = []
        for sample_id, item in enumerate(self.examples, start=1):
            row = {
                "id": sample_id,
                "iteration": item["iteration"],
                "source": item["source"],
                **{metal: item["catalyst_ratio"][i] for i, metal in enumerate(METAL_ORDER)},
                "power_density": item["power_density"],
                "normalized_cost": item["normalized_cost"],
            }
            pool_rows.append(row)
        pd.DataFrame(pool_rows).to_csv(self.config.output_dir / "total_pool.csv", index=False)
        pd.DataFrame(self.iteration_log).to_csv(
            self.config.output_dir / "iteration_log.csv", index=False
        )
        pd.DataFrame(self.query_log).to_csv(self.config.output_dir / "query_log.csv", index=False)
        pd.DataFrame(self.generated_log).to_csv(
            self.config.output_dir / "generated_ratios.csv", index=False
        )

    def run(self) -> dict:
        initial_best = max(item["power_density"] for item in self.examples)
        print(
            f"启动流程: provider={self.provider.name}, 初始样本={len(self.examples)}, "
            f"初始最佳PD={initial_best:.4f}"
        )

        for iteration_index in range(self.config.num_iterations):
            iteration = iteration_index + 1
            lambda_value = self._current_lambda(iteration_index)
            references = sort_reference_examples(self.examples)
            target_pd, target_cost = calculate_targets(self.examples, self.config, self.rng)

            generated_ratio = self.provider.generate_ratio(target_pd, target_cost, references)
            generated_cost = calculate_normalized_cost(generated_ratio)
            self.generated_log.append(
                {
                    "iteration": iteration,
                    **{metal: generated_ratio[i] for i, metal in enumerate(METAL_ORDER)},
                    "normalized_cost": generated_cost,
                }
            )

            candidates = self.rng.dirichlet(
                np.ones(len(METAL_ORDER)), self.config.candidate_pool_size
            )
            selected_points = AKDAS(
                np.asarray(generated_ratio), lambda_value, self.config.kdas_sigma
            ).select_points(candidates, self.config.kdas_select_size)

            scored_candidates = []
            for point in selected_points:
                ratio = point.tolist()
                similar = find_similar_examples(
                    ratio, self.examples, self.config.similar_examples_k
                )
                predictions = self.provider.predict_committee(
                    ratio, similar, self.config.committee_size
                )
                mean = float(np.mean(predictions))
                std = float(np.std(predictions, ddof=1)) if len(predictions) > 1 else 0.0
                scored_candidates.append(
                    {
                        "ratio": ratio,
                        "cost": calculate_normalized_cost(ratio),
                        "llm_mean": mean,
                        "llm_std": std,
                        "ucb_score": mean + self.config.ucb_exploration_weight * std,
                    }
                )

            top_candidates = select_pareto_candidates(
                scored_candidates,
                self.config.query_top_k,
                self.config.performance_weight,
            )
            queried = []
            for candidate in top_candidates:
                power_density = float(
                    self.oracle.predict(np.asarray(candidate["ratio"]).reshape(1, -1))[0]
                )
                queried.append(
                    {
                        **candidate,
                        "power_density": power_density,
                        "normalized_cost": candidate["cost"],
                    }
                )

            retained = self._score_and_retain(queried)
            retained_ids = {id(item) for item in retained}
            for rank, item in enumerate(queried, start=1):
                self.query_log.append(
                    {
                        "iteration": iteration,
                        "query_rank": rank,
                        "retained": id(item) in retained_ids,
                        **{metal: item["ratio"][i] for i, metal in enumerate(METAL_ORDER)},
                        "llm_mean": item["llm_mean"],
                        "llm_std": item["llm_std"],
                        "ucb_score": item["ucb_score"],
                        "power_density": item["power_density"],
                        "normalized_cost": item["normalized_cost"],
                        "final_score": item["final_score"],
                    }
                )

            for item in retained:
                self.examples.append(
                    {
                        "catalyst_ratio": item["ratio"],
                        "power_density": item["power_density"],
                        "normalized_cost": item["normalized_cost"],
                        "source": "oracle_query",
                        "iteration": iteration,
                    }
                )

            best_new = max(retained, key=lambda item: item["power_density"])
            global_best = max(item["power_density"] for item in self.examples)
            self.iteration_log.append(
                {
                    "iteration": iteration,
                    "lambda": lambda_value,
                    "target_pd": target_pd,
                    "target_cost": target_cost,
                    "generated_cost": generated_cost,
                    "queried_count": len(queried),
                    "retained_count": len(retained),
                    "best_new_pd": best_new["power_density"],
                    "best_new_cost": best_new["normalized_cost"],
                    "global_best_pd": global_best,
                }
            )
            self._save()
            print(
                f"迭代 {iteration:02d}/{self.config.num_iterations}: lambda={lambda_value:.3f}, "
                f"target_PD={target_pd:.3f}, new_best={best_new['power_density']:.3f}, "
                f"global_best={global_best:.3f}"
            )
            if global_best >= self.config.performance_threshold:
                print(f"达到停止阈值 {self.config.performance_threshold:.3f}")
                break

        summary = {
            "provider": self.provider.name,
            "iterations_completed": len(self.iteration_log),
            "initial_samples": sum(item["source"] == "initial" for item in self.examples),
            "retained_samples": sum(item["source"] == "oracle_query" for item in self.examples),
            "oracle_queries": len(self.query_log),
            "best_power_density": max(item["power_density"] for item in self.examples),
            "output_dir": str(self.config.output_dir),
        }
        with (self.config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
