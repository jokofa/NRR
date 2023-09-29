#
import logging
import os
import psutil
import argparse
from pathlib import Path
import torch

from lib.problem import RPDataset
from baselines.bl_eval import eval_baseline
from baselines.L2D.run_eval import l2d

###
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s: %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


###
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-s', '--save_dir', type=str,
                        default="outputs_eval/baselines/")
    parser.add_argument('--d_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_seeds', type=int, default=1)
    parser.add_argument('-n', '--n_iters', type=int, default=10000)
    parser.add_argument('-t', '--time_limit', type=int, default=None)
    parser.add_argument('-c', '--n_cpus', type=int, default=-1)
    parser.add_argument('--n_trials', type=int, default=500)

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--write_dir', type=Path, default=None)
    parser.add_argument('--shell', action='store_true')

    parser.add_argument('--ckpt', type=str,
                        default="baselines/L2D/exps/clustered_mixed_merge_routeneighbors10/subp_rotate_flip_layers6_heads8_lr0.001_batch2048"
                        )
    parser.add_argument('--ckpt_data', type=str,
                        default="baselines/L2D/generations/mixed_nc5_N500/subproblem_selection_lkh"
                        )
    parser.add_argument('--_data_suffix', type=str,
                        default="_routeneighbors10_beam1"
                        )
    args = parser.parse_args()

    DATASET = args.dataset
    assert isinstance(DATASET, str) and os.path.exists(DATASET)
    ds = RPDataset(
        problem="cvrp",
        data_pth=DATASET,
    ).sample(allow_pickle=True, sample_size=args.d_size)

    phys_cores = psutil.cpu_count(logical=False)
    if args.n_cpus is None or args.n_cpus <= 0:
        num_workers = phys_cores
    else:
        num_workers = min(args.n_cpus, phys_cores)

    if args.shell:
        assert args.write_dir is not None
        os.makedirs(args.write_dir, exist_ok=True)
        wfname = "l2d.res"
        wpth = os.path.join(args.write_dir, wfname)

        _, solutions = l2d(
            dataset=ds,
            cuda=args.cuda,
            seed=args.seed,
            root_dir="./",
            ckpt_pth=args.ckpt,
            ckpt_data_pth=args.ckpt_data,
            batch_size=1,
            num_workers=num_workers,
            num_iters=args.n_iters,
            time_limit=args.time_limit,
            n_lkh_trials=args.n_trials,
            _data_suffix=args._data_suffix,
        )
        logger.info(f'saving solutions to: {wpth}.')
        torch.save(solutions, wpth)

    else:
        SEEDS = [args.seed + i for i in range(args.n_seeds)]
        result, summary = eval_baseline(
            method=l2d,
            dataset=ds,
            seeds=SEEDS,
            save_dir=args.save_dir,
            cuda=not args.no_cuda,
            root_dir="./",
            ckpt_pth=args.ckpt,
            ckpt_data_pth=args.ckpt_data,
            num_workers=num_workers,
            num_iters=args.n_iters,
            time_limit=args.time_limit,
            n_lkh_trials=args.n_trials,
            _data_suffix=args._data_suffix,
        )
        logger.info(summary)
