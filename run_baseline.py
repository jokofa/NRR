#
import argparse
import os

from lib.problem import RPDataset
from baselines import methods_registry
from baselines.methods_registry import METHODS, CUDA_METHODS
from baselines.bl_eval import eval_baseline


###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, default='lkh', choices=METHODS)
    parser.add_argument('-d', '--dataset', type=str,
                        default="data/CVRP/cvrp500/data_val_seed222_size100_mixed_random_k_variant.dat")
    parser.add_argument('--d_size', type=int, default=None)
    parser.add_argument('-s', '--save_dir', type=str,
                        default="outputs_eval/baselines/")
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_seeds', type=int, default=1)
    parser.add_argument('-n', '--n_iters', type=int, default=10000)
    parser.add_argument('-t', '--time_limit', type=int, default=None)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--mode', type=str, default="greedy")
    parser.add_argument('--sample_size', type=int, default=1200)
    args = parser.parse_args()

    SEEDS = [args.seed + i for i in range(args.n_seeds)]
    CUDA = not args.no_cuda and args.method in CUDA_METHODS
    CKPTS = {
        "savings": None,
        "lkh": None,
        "lkh_popmusic": None,
        "pomo": "baselines/SGBS/result/uchoa100/checkpoint-8100.pt",
        "sgbs": "baselines/SGBS/result/uchoa100/checkpoint-8100.pt",
        "neuro_lkh": "baselines/NeuroLKH/NeuroLKH/pretrained/cvrp_neurolkh.pt",
        "dact": "baselines/DACT/DACT/pretrained/cvrp100-epoch-198.pt"
    }

    M = getattr(methods_registry, args.method)
    ckpt = args.ckpt if args.ckpt is not None else CKPTS[args.method.lower()]

    DATASET = args.dataset
    assert isinstance(DATASET, str) and os.path.exists(DATASET)
    dataset = RPDataset(
        problem="cvrp",
        data_pth=DATASET,
    ).sample(allow_pickle=True, sample_size=args.d_size)

    eval_baseline(
        method=M,
        dataset=dataset,
        seeds=SEEDS,
        cuda=CUDA,
        save_dir=args.save_dir,
        time_limit=args.time_limit,
        ckpt_pth=ckpt,
        num_iters=args.n_iters,
        strict_feasibility=False,
        raise_errors=True,  ###
        mode=args.mode,
        sample_size=args.sample_size,
    )
