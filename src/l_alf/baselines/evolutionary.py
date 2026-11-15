from __future__ import annotations

import numpy as np

from .common import BaselineConfig, evaluate_batch, load_frozen_oracle, load_initial_examples, save_results
from ..costs import calculate_normalized_cost, normalize_ratio


SOURCE_POPULATION_SIZE = 40
SOURCE_RANDOM_SEED = 5


def run_random(config: BaselineConfig) -> dict:
    """Run Dirichlet(1,...,1) random search; source default budget is 600, seed 5."""
    config.validate("random")
    initial = load_initial_examples(config.initial_data_path)
    oracle = load_frozen_oracle(config.oracle_model_path)
    rng = np.random.RandomState(config.random_seed)
    ratios = rng.dirichlet(np.ones(8), size=config.new_evaluations)
    evaluated = evaluate_batch(oracle, ratios)
    return save_results(
        config,
        "random",
        initial,
        evaluated,
        {"sampler": "Dirichlet(1,...,1)", "dimension": 8},
    )


def run_nsga2(config: BaselineConfig) -> dict:
    """Run pymoo NSGA-II with the source simplex sampling, SBX, PM and repair."""
    config.validate("nsga2")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.repair import Repair
        from pymoo.core.sampling import Sampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
        from pymoo.termination.max_eval import MaximumFunctionCallTermination
    except ImportError as exc:
        raise RuntimeError("NSGA-II 基线需要 pymoo") from exc

    initial = load_initial_examples(config.initial_data_path)
    oracle = load_frozen_oracle(config.oracle_model_path)
    n_vars = 8
    evaluated = []
    np.random.seed(config.random_seed)

    class DirichletSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            return np.random.dirichlet(np.ones(problem.n_var), n_samples)

    class SimplexRepair(Repair):
        def _do(self, problem, values, **kwargs):
            values[values < 1e-6] = 1e-6
            return values / values.sum(axis=1, keepdims=True)

    class CatalystProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n_vars, n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            ratio = normalize_ratio(np.asarray(x, dtype=float))
            prediction = float(oracle.predict(ratio.reshape(1, -1))[0])
            if not np.isfinite(prediction):
                raise ValueError("冻结 GBR 返回了无效功率预测")
            cost = calculate_normalized_cost(ratio)
            evaluated.append({
                "ratio": ratio,
                "power_density": prediction,
                "normalized_cost": cost,
            })
            out["F"] = [-prediction, cost]

    problem = CatalystProblem()
    population_size = min(SOURCE_POPULATION_SIZE, config.new_evaluations)
    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=population_size,
        sampling=DirichletSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / n_vars, eta=20),
        repair=SimplexRepair(),
        eliminate_duplicates=True,
    )
    termination = MaximumFunctionCallTermination(config.new_evaluations)
    algorithm.setup(problem, termination=termination, seed=config.random_seed, verbose=False)
    while algorithm.has_next():
        if algorithm.is_initialized:
            remaining = config.new_evaluations - len(evaluated)
            algorithm.n_offsprings = min(population_size, remaining)
        algorithm.next()

    if len(evaluated) != config.new_evaluations:
        raise RuntimeError(
            f"NSGA-II 执行了 {len(evaluated)} 次 GBR 评价，配置要求 {config.new_evaluations} 次"
        )
    return save_results(
        config,
        "nsga2",
        initial,
        evaluated,
        {
            "optimizer": "pymoo.NSGA2",
            "population_size": population_size,
            "offspring_size": SOURCE_POPULATION_SIZE,
            "sampling": "Dirichlet(1,...,1)",
            "crossover": {"name": "SBX", "probability": 0.9, "eta": 15},
            "mutation": {"name": "polynomial", "probability": 1.0 / n_vars, "eta": 20},
            "repair": "clip at 1e-6 then close to simplex",
            "eliminate_duplicates": True,
        },
    )
