#
import os
import warnings
from typing import Optional, Union, Dict, List, Callable
from copy import deepcopy
import random
import numpy as np
import torch

from lib.problem import RPDataset, eval_rp
from lib.utils.logging import setup_logger
logger = setup_logger(__name__)


def save_results(results: List[Dict], save_dir: str, prefix: str = "", postfix: str = ""):
    pth = os.path.join(save_dir, f"{prefix}eval_results{postfix}.pkl")
    logger.info(f"saving results to {pth}")
    torch.save(results, pth)


def get_p_str(test_ds: RPDataset):
    pth = test_ds.data_pth.lower()
    if "uchoa" in pth:
        return "uchoa"
    elif "arnold" in pth and "square" in pth:
        return "real_square"
    elif "arnold" in pth and "l2d" in pth:
        return "real_l2d"
    else:
        graph_size = test_ds.data[0].graph_size - 1
        return f"{test_ds.problem.lower()}{graph_size}"


def eval_baseline(
        method: Callable,
        dataset: Union[RPDataset, str],
        seeds: Union[int, List[int]] = 1234,
        save_dir: str = "./outputs_eval/baselines/",
        cuda: bool = False,
        strict_feasibility: bool = False,   # no max constraint on k
        sample_cfg: Optional[Dict] = None,
        time_limit: Optional[int] = None,
        raise_errors: bool = True,
        method_str: Optional[str] = None,
        **kwargs
):
    if method_str is None:
        method_str = method.__name__
    if "pomo" in method_str.lower():
        md = kwargs.get("mode", "")
        method_str = f"{method_str}-{md}"
    cuda = cuda and torch.cuda.is_available()
    logger.info(f"evaluating baseline: '{method_str}'")
    kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"cfg: cuda={cuda}, time_limit={time_limit}, ({kwargs_str}).")
    # setup dataset
    if isinstance(dataset, RPDataset):
        ds_str = os.path.basename(os.path.splitext(dataset.data_pth)[0]) if dataset.data_pth is not None else ""
        assert dataset.data is not None and len(dataset.data) > 0
        test_ds = dataset
    else:
        logger.info(f"Dataset path provided. Loading dataset...")
        assert isinstance(dataset, str) and os.path.exists(dataset)
        ds_str = os.path.basename(os.path.splitext(dataset)[0])
        sample_cfg = sample_cfg if sample_cfg is not None else {}
        test_ds = RPDataset(
            problem="cvrp",
            fname=dataset,
        ).sample(**sample_cfg)

    p_str = get_p_str(test_ds)
    logger.info(f"loaded test data: {p_str}/{ds_str} - {dataset.data_pth}")

    save_dir = os.path.join(save_dir, method_str, p_str, ds_str)
    os.makedirs(save_dir, exist_ok=True)
    seeds = seeds if isinstance(seeds, list) else [seeds]

    logger.info(f"running {method_str} for {len(seeds)} seeds on {len(test_ds)} instances...")
    solutions = []
    for sd in seeds:
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        test_ds.seed(sd)
        if cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info(f"Running eval for seed {sd}...")
        try:
            _, sols = method(
                    dataset=deepcopy(test_ds),
                    seed=sd,
                    cuda=cuda,
                    batch_size=1,
                    time_limit=time_limit,
                    **kwargs
                )

            solutions += deepcopy(sols)
            assert len(sols) == len(test_ds)
        except Exception as e:
            logger.debug(f"Encountered Error for seed {sd}: \n{e}")
            if raise_errors:
                raise e

    if len(solutions) != len(test_ds) * len(seeds):
        warnings.warn(f"n solutions {len(solutions)} != {len(test_ds)}*{len(seeds)}={len(test_ds) * len(seeds)} n data")
    save_results(solutions, save_dir=save_dir, postfix=f"_raw_solutions")
    results, summary = eval_rp(
        solutions=solutions,
        problem="cvrp",
        strict_feasibility=strict_feasibility,
    )
    logger.info(f"summary: {summary}")
    save_results(results, save_dir=save_dir, postfix=f"_full_{len(seeds)}")

    return results, summary
