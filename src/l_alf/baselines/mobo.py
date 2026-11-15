from __future__ import annotations

import logging
import warnings

import numpy as np

from .common import BaselineConfig, evaluate_batch, load_frozen_oracle, load_initial_examples, save_results
from ..costs import METAL_ORDER, METAL_PRICES


MOBO_SOURCE_DEFAULT_NEW_EVALUATIONS = 150
DG_MOBO_SOURCE_DEFAULT_NEW_EVALUATIONS = 300
DG_MOBO_CANDIDATE_POOL_SIZE = 5
MC_SAMPLES = 32
REF_MARGIN_FRAC = 0.05
MOBO_REF_MIN_MARGIN = 1e-4
DG_MOBO_REF_MIN_MARGIN = 1e-3


def _load_botorch():
    try:
        import torch
        from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models.deterministic import GenericDeterministicModel
        from botorch.models.model import ModelList
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.list_sampler import ListSampler
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise RuntimeError(
            "MOBO 基线需要 torch、botorch 和 gpytorch；请安装匹配的实验环境依赖"
        ) from exc
    try:
        from botorch.sampling.deterministic import DeterministicSampler
    except ImportError:
        try:
            from botorch.sampling.index_sampler import IndexSampler as DeterministicSampler
        except ImportError as exc:
            raise RuntimeError(
                "当前 BoTorch 缺少 deterministic posterior sampler"
            ) from exc
    return {
        "torch": torch,
        "SingleTaskGP": SingleTaskGP,
        "ModelList": ModelList,
        "GenericDeterministicModel": GenericDeterministicModel,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "qExpectedHypervolumeImprovement": qExpectedHypervolumeImprovement,
        "FastNondominatedPartitioning": FastNondominatedPartitioning,
        "optimize_acqf": optimize_acqf,
        "Standardize": Standardize,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "ListSampler": ListSampler,
        "SobolQMCNormalSampler": SobolQMCNormalSampler,
        "DeterministicSampler": DeterministicSampler,
    }


def _run_qehvi(config: BaselineConfig, method: str) -> dict:
    config.validate(method)
    deps = _load_botorch()
    torch = deps["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

    warnings.filterwarnings("ignore")
    logging.getLogger("botorch").setLevel(logging.ERROR)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    initial = load_initial_examples(config.initial_data_path)
    oracle = load_frozen_oracle(config.oracle_model_path)
    ratios = np.asarray([item["ratio"] for item in initial], dtype=float)
    power = np.asarray([item["power_density"] for item in initial], dtype=float)
    costs = np.asarray([item["normalized_cost"] for item in initial], dtype=float)
    train_x = torch.tensor(ratios, device=device, dtype=dtype)
    train_y = torch.tensor(np.column_stack((power, -costs)), device=device, dtype=dtype)

    prices = torch.tensor([METAL_PRICES[name] for name in METAL_ORDER], device=device, dtype=dtype)
    base_price = torch.tensor(METAL_PRICES["Pd"], device=device, dtype=dtype)

    def cost_callable(x):
        weights = torch.clamp(x, min=0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cost = (weights * prices).sum(dim=-1) / base_price * 100.0
        return (-cost).unsqueeze(-1)

    candidate_pool_size = DG_MOBO_CANDIDATE_POOL_SIZE if method == "dg_mobo" else config.batch_size
    if method == "dg_mobo" and config.batch_size > candidate_pool_size:
        raise ValueError(
            f"DG-MOBO 源码候选数为 {candidate_pool_size}，不能选择 batch_size={config.batch_size}"
        )

    evaluated = []
    iterations = config.new_evaluations // config.batch_size
    for iteration in range(iterations):
        gp_pd = deps["SingleTaskGP"](
            train_x,
            train_y[:, 0:1],
            outcome_transform=deps["Standardize"](m=1),
        )
        mll = deps["ExactMarginalLogLikelihood"](gp_pd.likelihood, gp_pd)
        deps["fit_gpytorch_mll"](mll)
        cost_model = deps["GenericDeterministicModel"](cost_callable)
        model = deps["ModelList"](gp_pd, cost_model)

        y_min = train_y.min(dim=0).values
        y_max = train_y.max(dim=0).values
        min_margin = DG_MOBO_REF_MIN_MARGIN if method == "dg_mobo" else MOBO_REF_MIN_MARGIN
        ref_point = y_min - REF_MARGIN_FRAC * (y_max - y_min).clamp_min(min_margin)
        partitioning = deps["FastNondominatedPartitioning"](ref_point=ref_point, Y=train_y)
        sample_shape = torch.Size([MC_SAMPLES])
        sampler = deps["ListSampler"](
            deps["SobolQMCNormalSampler"](sample_shape=sample_shape),
            deps["DeterministicSampler"](sample_shape=sample_shape),
        )
        acquisition = deps["qExpectedHypervolumeImprovement"](
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )
        bounds = torch.stack((torch.zeros(len(METAL_ORDER), device=device, dtype=dtype),
                              torch.ones(len(METAL_ORDER), device=device, dtype=dtype)))
        constraints = [(
            torch.arange(len(METAL_ORDER), device=device),
            torch.ones(len(METAL_ORDER), device=device, dtype=dtype),
            1.0,
        )]
        candidates, _ = deps["optimize_acqf"](
            acq_function=acquisition,
            bounds=bounds,
            q=candidate_pool_size,
            num_restarts=4,
            raw_samples=128,
            equality_constraints=constraints,
            sequential=False,
        )

        if method == "dg_mobo":
            pool = candidates.detach().cpu()
            selected = [0]
            for _ in range(config.batch_size - 1):
                remaining = [i for i in range(len(pool)) if i not in selected]
                distances = torch.cdist(pool[remaining], pool[selected], p=2)
                min_distance = distances.min(dim=1).values
                selected.append(remaining[torch.argmax(min_distance).item()])
            candidates = candidates[selected]

        with torch.no_grad():
            predictions = gp_pd.posterior(candidates).mean.squeeze(-1).detach().cpu().numpy()
        batch = candidates.detach().cpu().numpy()
        results = evaluate_batch(oracle, batch)
        for index, item in enumerate(results):
            item["gp_pred_power_density"] = float(predictions[index])
        evaluated.extend(results)

        new_y = torch.tensor(
            [[item["power_density"], -item["normalized_cost"]] for item in results],
            device=device,
            dtype=dtype,
        )
        train_x = torch.cat((train_x, candidates), dim=0)
        train_y = torch.cat((train_y, new_y), dim=0)

    settings = {
        "acquisition": "qExpectedHypervolumeImprovement",
        "model": "SingleTaskGP(power) + deterministic differentiable cost",
        "mc_samples": MC_SAMPLES,
        "q": candidate_pool_size,
        "selected_batch_size": config.batch_size,
        "num_restarts": 4,
        "raw_samples": 128,
        "sequential": False,
        "ref_margin_fraction": REF_MARGIN_FRAC,
        "ref_min_margin": min_margin,
        "dtype": "float32",
        "device": str(device),
        "simplex_equality_constraint": True,
        "candidate_pool_size": candidate_pool_size if method == "dg_mobo" else None,
    }
    return save_results(config, method, initial, evaluated, settings)


def run_mobo(config: BaselineConfig) -> dict:
    """Run the source qEHVI MOBO variant; source script default budget is 150."""
    return _run_qehvi(config, "mobo")


def run_dg_mobo(config: BaselineConfig) -> dict:
    """Run qEHVI followed by source maximin selection; source default is 300."""
    return _run_qehvi(config, "dg_mobo")
