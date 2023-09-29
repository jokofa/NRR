#
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from lib.problem import RPDataset
from lib.nrr import NRR
from lib.nrr.lkh_wrapper import LKHSolver
from lib.nrr.sgbs_wrapper import SGBSSolver

logger = logging.getLogger(__name__)


@hydra.main(config_path="config/nrr_config", config_name="scoring_config", version_base=None)
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info(OmegaConf.to_yaml(cfg))

    data_mod = 'mixed' if 'mixed' in cfg.train_dataset else 'unf'
    save_dir = os.path.join(
        cfg.save_dir,
        f"{cfg.problem}{cfg.graph_size}_{data_mod}"
    )

    dataset = RPDataset(
        problem=cfg.problem,
        data_pth=cfg.train_dataset,
    ).sample(allow_pickle=True)

    if str(cfg.solver).lower() == 'lkh':
        solver = LKHSolver(
            seed=cfg.global_seed,
            **cfg.solver_cfg
        )
    elif str(cfg.solver).lower() == 'sgbs':
        solver = SGBSSolver(
            seed=cfg.global_seed,
            **cfg.solver_cfg
        )
    else:
        raise ValueError

    nrr = NRR(
        sg_solver=solver,
        verbose=cfg.debug_lvl,
        seed=cfg.global_seed,
        **cfg.nrr_cfg
    )
    pth = nrr.create_scoring_data(
        instances=dataset,
        max_iters=cfg.nrr_max_iters,
        save_dir=save_dir,
    )
    print(f"saved to pth: {pth}")


if __name__ == "__main__":
    run()
