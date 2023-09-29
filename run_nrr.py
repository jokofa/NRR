#
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from copy import deepcopy
import random
import numpy as np
import torch

from lib.problem import RPDataset, RPSolution, eval_rp
from lib.nrr import NRR
from lib.nrr.lkh_wrapper import LKHSolver
from lib.nrr.sgbs_wrapper import SGBSSolver
from lib.nrr.best_insert_wrapper import BestInsertSolver
from lib.model.utils import load_model
from baselines.bl_eval import save_results, get_p_str

logger = logging.getLogger(__name__)
PROBLEM = "cvrp"


@hydra.main(config_path="config/nrr_config", config_name="config", version_base=None)
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info(OmegaConf.to_yaml(cfg))

    dataset = RPDataset(
        problem=cfg.problem,
        data_pth=cfg.dataset,
    ).sample(allow_pickle=True, sample_size=cfg.dataset_size)

    seeds = [cfg.global_seed * (i+1) for i in range(cfg.n_seeds)]
    rp_solutions = []
    for sd in seeds:
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        if cfg.solver_cfg.get('cuda', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        ds = deepcopy(dataset)

        slv = str(cfg.solver).lower()
        if slv == 'lkh':
            solver = LKHSolver(
                seed=sd,
                **cfg.solver_cfg
            )
        elif slv == 'sgbs':
            solver = SGBSSolver(
                seed=sd,
                **cfg.solver_cfg
            )
        elif slv == 'bis':
            solver = BestInsertSolver(
                seed=sd,
                **cfg.solver_cfg
            )
        else:
            raise ValueError(slv)

        model = None
        if cfg.nrr_cfg.scoring_method == "nsf":
            model = load_model(
                ckpt_pth=cfg.nrr_cfg.scoring_method_cfg.ckpt,
                cuda=cfg.nrr_cfg.scoring_method_cfg.cuda
            )

        nrr = NRR(
            sg_solver=solver,
            verbose=cfg.debug_lvl,
            seed=sd+1,
            save_trajectory=True,
            scoring_fn=model,
            **cfg.nrr_cfg
        )
        solutions, runtimes, trajectories = [], [], []
        try:
            for inst in tqdm(ds):
                sol, rt = nrr.solve(
                    instance=inst,
                    max_iters=cfg.nrr_max_iters,
                    time_limit=cfg.nrr_time_limit,
                )
                solutions.append(sol)
                runtimes.append(rt)
                trajectories.append(nrr.get_trajectory())
        except KeyboardInterrupt:
            print(f"num solutions: \n{len(solutions)}")
            cost = [s.total_cost for s in solutions]
            num_v = [s.num_vehicles for s in solutions]
            print(f"mean cost: {np.mean(cost): .5f}")
            print(f"mean num vehicles: {np.mean(num_v): .5f}")
            raise KeyboardInterrupt

        for inst, sol, rt, trj in zip(ds, solutions, runtimes, trajectories):
            rp_solutions.append(RPSolution(
                solution=[r.tolist() for r in sol.routes],
                run_time=rt,
                problem=PROBLEM,
                instance=inst,
                trajectory=trj,
            ))

    assert len(rp_solutions) == len(dataset) * len(seeds)
    results, summary = eval_rp(
        solutions=rp_solutions,
        problem=PROBLEM,
        strict_feasibility=False,   # no max constraint on k
    )
    print(f"summary: {summary}")
    ds_str = os.path.basename(os.path.splitext(cfg.dataset)[0])
    p_str = get_p_str(dataset)
    save_dir = cfg.get("save_dir", f"./outputs_eval/nrr/{cfg.nrr_cfg.scoring_method}")
    save_dir = os.path.join(save_dir, p_str, ds_str)
    os.makedirs(save_dir, exist_ok=True)
    inf_mode = cfg.solver_cfg.get("mode", "")
    fname = f"{slv}{':'+inf_mode if len(inf_mode) > 1 else ''}_" \
            f"{cfg.nrr_cfg.scoring_method}_" \
            f"{cfg.nrr_cfg.init_method}_" \
            f"{cfg.nrr_cfg.sg_construction_method}_" \
            f"{cfg.nrr_cfg.sg_selection_method}_" \
            f"{cfg.nrr_cfg.accept_method}_"
    save_results([cfg, summary] + results, save_dir=save_dir, prefix=fname, postfix=f"_{len(seeds)}")


###
if __name__ == "__main__":
    run()

