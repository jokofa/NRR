#
import os
import logging
from copy import deepcopy
from typing import Optional, Union, Dict, List
from warnings import warn
from tqdm import tqdm
from argparse import Namespace
import numpy as np
import torch

from lib.problem import RPDataset, RPSolution
from baselines.savings import eval_savings
# LKH, NeuroLKH
from baselines.NeuroLKH.neuro_lkh import cvrp_inference as lkh_inference
# SGBS / POMO
from baselines.SGBS.sgbs import SGBS, eval_model as sgbs_inference
# DACT
from baselines.DACT.DACT.problems.problem_vrp import CVRP
from baselines.DACT.DACT.agent.ppo import PPO
from baselines.DACT.dact import eval_model as dact_inference


__all__ = [
    "savings",
    "lkh",
    "lkh_popmusic",
    "neuro_lkh",
    "pomo",
    "sgbs",
    "dact",
    "METHODS",
    "CUDA_METHODS",
]

# create logger
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


METHODS = [
    "savings",
    "lkh",
    "lkh_popmusic",
    "neuro_lkh",
    "pomo",
    "sgbs",
    "dact",
]

CUDA_METHODS = [
    "neuro_lkh",
    "pomo",
    "sgbs",
    "dact",
]


def savings(
        dataset: Union[RPDataset, List],
        cuda: bool = False,
        seed: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        num_iters: int = 10000,
        time_limit: Optional[int] = None,
        sample_size: int = 1200,
        savings_function: str = "clarke_wright",
        **kwargs
):
    """Savings construction heuristic
    with either savings function:

        - 'clarke_wright'
        - 'gaskell_lambda'
        - 'gaskell_pi'

    """
    np.random.seed(seed)

    results = {}
    solutions = []
    for inst in dataset:
        sol, rt = eval_savings(
            instance=inst,
            savings_function=savings_function,
            **kwargs
        )
        solutions.append(RPSolution(
            solution=sol,
            run_time=rt,
            instance=inst,
            problem="cvrp"
        ))

    return results, solutions


def pomo(
    dataset: Union[RPDataset, List],
    cuda: bool = True,
    seed: Optional[int] = None,
    ckpt_pth: str = None,
    batch_size: int = 1,
    num_iters: int = 1,
    time_limit: Optional[int] = None,
    sample_size: int = 1200,
    mode: str = "greedy",
    **kwargs
):
    assert mode in ['greedy', 'sampling']
    if not ckpt_pth is not None and os.path.exists(ckpt_pth):
        raise ValueError(ckpt_pth)
    if torch.cuda.is_available() and not cuda:
        warn(f"Cuda GPU is available but not used!")

    model = SGBS(
        ckpt_pth=ckpt_pth,
        cuda=cuda,
        seed=seed,
        mode=mode,
        n_samples=sample_size,
        **kwargs
    )
    results, solutions = sgbs_inference(
        model=model,
        data=dataset
    )
    return results, solutions


def sgbs(
    dataset: Union[RPDataset, List],
    cuda: bool = True,
    seed: Optional[int] = None,
    ckpt_pth: str = None,
    batch_size: int = 1,
    num_iters: int = 1,
    time_limit: Optional[int] = None,
    sample_size: int = 1200,
    mode: str = "sgbs",  # greedy, sampling, obs, mcts, sgbs
    **kwargs
):
    if not ckpt_pth is not None and os.path.exists(ckpt_pth):
        raise ValueError(ckpt_pth)
    if torch.cuda.is_available() and not cuda:
        warn(f"Cuda GPU is available but not used!")

    model = SGBS(
        ckpt_pth=ckpt_pth,
        cuda=cuda,
        seed=seed,
        mode='sgbs',
        n_samples=sample_size,
        **kwargs
    )
    results, solutions = sgbs_inference(
        model=model,
        data=dataset
    )
    return results, solutions


def lkh(
    dataset: Union[RPDataset, List],
    cuda: bool = False,
    seed: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 1,
    num_iters: int = 10000,
    time_limit: Optional[int] = None,
    **kwargs
):
    results, solutions = lkh_inference(
        data=dataset,
        method="LKH",
        model_path=None,    # type: ignore
        lkh_exe_path="baselines/NeuroLKH/LKH",
        batch_size=batch_size,
        num_workers=num_workers,
        max_trials=num_iters,
        time_limit=time_limit,
        seed=seed,
    )

    return results, solutions


def lkh_popmusic(
    dataset: Union[RPDataset, List],
    cuda: bool = False,
    seed: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 1,
    num_iters: int = 10000,
    time_limit: Optional[int] = None,
    **kwargs
):
    results, solutions = lkh_inference(
        data=dataset,
        method="LKH",
        model_path=None,    # type: ignore
        lkh_exe_path="baselines/NeuroLKH/LKH",
        batch_size=batch_size,
        num_workers=num_workers,
        max_trials=num_iters,
        time_limit=time_limit,
        seed=seed,
        popmusic=True
    )

    return results, solutions


def neuro_lkh(
    dataset: Union[RPDataset, List],
    cuda: bool = True,
    seed: Optional[int] = None,
    ckpt_pth: str = None,
    batch_size: int = 1,
    num_workers: int = 1,
    num_iters: int = 10000,
    time_limit: Optional[int] = None,
    sample_size: int = 1200,
    **kwargs
):
    assert ckpt_pth is not None and os.path.exists(ckpt_pth)
    assert 'neurolkh' in ckpt_pth.lower()
    if torch.cuda.is_available() and not cuda:
        warn(f"Cuda GPU is available but not used!")
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    results = {}
    solutions = []
    for d in dataset:
        try:
            _, sol = lkh_inference(
                data=[d],
                method="NeuroLKH",
                model_path=ckpt_pth,
                lkh_exe_path="baselines/NeuroLKH/LKH",
                batch_size=batch_size,
                num_workers=num_workers,
                max_trials=num_iters,
                time_limit=time_limit,
                seed=seed,
                device=device,
                **kwargs
            )
            solutions.append(sol[0])
        except Exception as e:
            logger.warning(f"ERROR: {e}")
            solutions.append(RPSolution(solution=None))

    return results, solutions


def dact(
    dataset: Union[RPDataset, List],
    cuda: bool = True,
    seed: Optional[int] = None,
    ckpt_pth: str = None,
    batch_size: int = 1,
    num_iters: int = 1000,
    time_limit: Optional[int] = None,
    sample_size: int = 1200,
    num_augments: int = 1,
    test_cfg: Optional[Dict] = None,
    verbose: bool = False,
    **kwargs
):
    ### !!!
    # DACT for cvrp4000:
    # --> pair_index = M_table.multinomial(1) - RuntimeError: number of categories cannot exceed 2^24
    ### !!!
    assert ckpt_pth is not None and os.path.exists(ckpt_pth)
    if torch.cuda.is_available() and not cuda:
        warn(f"Cuda GPU is available but not used!")
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    cfg = {
        "v_range": 6.0,  # help='to control the entropy'
        "DACTencoder_head_num": 4,  # help='head number of DACT encoder'
        "DACTdecoder_head_num": 4,  # help='head number of DACT decoder'
        "critic_head_num": 6,  # help='head number of critic encoder'
        "embedding_dim": 64,  # help='dimension of input embeddings (NEF & PFE)'
        "hidden_dim": 64,  # help='dimension of hidden layers in Enc/Dec'
        "n_encode_layers": 3,  # help='number of stacked layers in the encoder'
        "normalization": 'layer',  # help="normalization type, #'layer' (default) or 'batch'"

        # agent params
        "RL_agent": 'ppo',  # help='RL Training algorithm '
        "gamma": 0.999,  # help='reward discount factor for future rewards '
        "K_epochs": 3,  # help='mini PPO epoch '
        "eps_clip": 0.1,  # help='PPO clip ratio '
        "T_train": 200,  # help='number of itrations for training '
        "n_step": 4,  # help='n_step for return estimation '
        "best_cl": False,  # help='use best solution found in CL as initial solution for training '
        "Xi_CL": 0.25,  # help='hyperparameter of CL '
        "lr_model": 1e-4,  # help="learning rate for the actor network")
        "lr_critic": 3e-5,  # help="learning rate for the critic network")
        "lr_decay": 0.985,  # help='learning rate decay per epoch '
        "max_grad_norm": 0.04,  # help='maximum L2 norm for gradient clipping '
        "epoch_end": 200,  # help='maximum training epoch'
        "epoch_size": 12000,  # help='number of instances per epoch during training'

        "problem": "cvrp",
        "graph_size": None,
        "coords_dist": 'uniform',
        "T_max": num_iters,  # number of steps for inference
        "num_augments": num_augments,  # number of data augments (<=8)

        "env_cfg": {
            "step_method": '2_opt',  # ['2_opt','swap','insert']
            "init_val_met": 'greedy',  # ['random','greedy','seq']
            "perturb_eps": 250,  # eval
            "dummy_rate": 0.5,
        },

        "use_cuda": cuda and torch.cuda.is_available(),
        "distributed": False,
        "no_saving": True,
        "device": device.type,
        "no_progress_bar": True,
        "eval_only": True,
    }
    if test_cfg is not None:
        cfg.update(test_cfg)
    cfg = Namespace(**cfg)

    assert batch_size == 1
    results, solutions = {}, []

    for d in tqdm(dataset):
        cfg_ = deepcopy(cfg)
        env_cfg = Namespace(**cfg_.env_cfg)

        cfg_.graph_size = int(d.graph_size - 1)
        env = CVRP(
            p_size=cfg_.graph_size,
            step_method=env_cfg.step_method,
            init_val_met=env_cfg.init_val_met,
            with_assert=False,
            P=env_cfg.perturb_eps,
            DUMMY_RATE=env_cfg.dummy_rate,
            verbose=verbose
        )

        policy = PPO(env.NAME, env.size, cfg_)

        _, sol = dact_inference(
            data=[d],
            problem=env,
            agent=policy,
            opts=cfg_,  # type: ignore
            dummy_rate=env_cfg.dummy_rate,
            device=device,
            batch_size=batch_size,
            time_limit=time_limit,
            verbose=verbose
        )
        solutions.append(sol[0])

    return results, solutions


def nlns(
    dataset: Union[RPDataset, List],
    cuda: bool = True,
    seed: Optional[int] = None,
    ckpt_pth: str = None,
    batch_size: int = 1,
    num_workers: int = 1,
    num_iters: int = 10000,
    time_limit: Optional[int] = None,
    sample_size: int = 1200,
    **kwargs
):
    assert ckpt_pth is not None and os.path.exists(ckpt_pth)
    assert 'nlns' in ckpt_pth.lower()
    if torch.cuda.is_available() and not cuda:
        warn(f"Cuda GPU is available but not used!")
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    results = {}
    solutions = []
    for d in dataset:
        try:
            _, sol = lkh_inference(
                data=[d],
                method="NeuroLKH",
                model_path=ckpt_pth,
                lkh_exe_path="baselines/NeuroLKH/LKH",
                batch_size=batch_size,
                num_workers=num_workers,
                max_trials=num_iters,
                time_limit=time_limit,
                seed=seed,
                device=device,
                **kwargs
            )
            solutions.append(sol[0])
        except Exception as e:
            logger.warning(f"ERROR: {e}")
            solutions.append(RPSolution(solution=None))

    return results, solutions


# ============ #
# ### TEST ### #
# ============ #
def _test():
    from baselines.bl_eval import eval_baseline
    import sys

    this = sys.modules[__name__]
    SIZE = 3
    NSEEDS = 2
    CUDA = True

    T_LIM = 10
    DATASET = "data/CVRP/cvrp200/data_val_seed222_size100_mixed_random_k_variant.dat"
    seeds = [1234 * (i + 1) for i in range(NSEEDS)]
    ds = RPDataset(
        problem="cvrp",
        data_pth=DATASET,
    ).sample(allow_pickle=True, sample_size=SIZE)

    CKPTS = {
        "savings": None,
        "lkh": None,
        "lkh_popmusic": None,
        "pomo": "baselines/SGBS/result/uchoa100/checkpoint-8100.pt",
        "sgbs": "baselines/SGBS/result/uchoa100/checkpoint-8100.pt",
        "neuro_lkh": "baselines/NeuroLKH/NeuroLKH/pretrained/cvrp_neurolkh.pt",
        "dact": "baselines/DACT/DACT/pretrained/cvrp100-epoch-198.pt"
    }

    results, summaries = [], {}
    #for m in METHODS:
    for m in ['sgbs']:
        mthd = getattr(this, m)
        kwargs = {'ckpt_pth': CKPTS[m]}
        res, smr = eval_baseline(
            method=mthd,
            dataset=deepcopy(ds),
            seeds=seeds,
            cuda=CUDA,
            sample_cfg=None,
            time_limit=T_LIM,
            raise_errors=True,
            **kwargs
        )
        results.append(res)
        mid = f"{m}{'_cuda' if CUDA and m in CUDA_METHODS else ''}"
        summaries[mid] = smr

    print(summaries)
