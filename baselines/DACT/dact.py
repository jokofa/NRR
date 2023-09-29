#
import logging
from timeit import default_timer
import itertools as it
from typing import Optional, Dict, Union, List, NamedTuple, Tuple, Any
from omegaconf import DictConfig

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.problem import RPInstance, RPSolution
from baselines.DACT.DACT.problems.problem_vrp import CVRP
from baselines.DACT.DACT.agent.ppo import PPO

logger = logging.getLogger(__name__)


class CVRPDataset(Dataset):
    def __init__(self,
                 data: List[RPInstance],
                 #graph_size: int,
                 dummy_rate: Optional[float] = None
                 ):

        super(CVRPDataset, self).__init__()
        self.dummy_rate = dummy_rate
        self.data = [self.make_instance(d, self.dummy_rate) for d in data]

    @staticmethod
    def make_instance(instance: RPInstance, dummy_rate: float):
        depot = torch.from_numpy(instance.coords[0])
        loc = torch.from_numpy(instance.coords[1:])
        demand = torch.from_numpy(instance.demands[1:])

        graph_size = instance.graph_size-1
        depot_reps = int(np.ceil(graph_size * (1 + dummy_rate))) - graph_size

        return {
            'coordinates': torch.cat((depot.view(-1, 2).repeat(depot_reps, 1), loc), 0),
            'demand': torch.cat((torch.zeros(depot_reps), demand), 0),
            'graph_size': graph_size
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def sol_to_list(sol: np.ndarray, depot_idx: int = 0) -> List[List]:
    lst, sol_lst = [], []
    for n in sol:
        if n == depot_idx:
            if len(lst) > 0:
                sol_lst.append(lst)
                lst = []
        else:
            lst.append(n)
    if len(lst) > 0:
        sol_lst.append(lst)
    return sol_lst


#
def train_model():
    raise NotImplementedError
    agent.start_training(problem, opts.val_dataset, tb_logger)


#
def eval_model(data: List[RPInstance],
               problem: CVRP,
               agent: PPO,
               opts: Union[DictConfig, NamedTuple],
               batch_size: int,
               dummy_rate: Optional[float] = None,
               device=torch.device("cpu"),
               time_limit: Optional[int] = None,
               verbose: bool = True,
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:

    # eval mode
    if device.type != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    agent.eval()
    problem.eval()
    if verbose:
        print(f'Inference with {opts.num_augments} augments...')

    val_dataset = CVRPDataset(data=data, dummy_rate=dummy_rate)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

    sols, times, trajectories = [], [], []

    for batch in val_dataloader:

        t_start = default_timer()
        # bv, cost_hist, best_hist, r, best_sol_hist
        res = agent.rollout(
            problem, opts.num_augments, batch,
            do_sample=True,
            record=True,
            show_bar=True,
            time_limit=time_limit,
        )
        t = default_timer() - t_start
        t_per_inst = t / batch_size
        sols.append(res[-1].cpu().numpy())
        times.append([t_per_inst]*batch_size)
        t_elapsed = res[-2]
        objs = res[-3].cpu().numpy()
        trajectories.append({
            "iter": np.arange(len(t_elapsed)),
            "time": np.array(t_elapsed),
            "cost": objs.reshape(-1) if len(objs) == 1 else objs
        })

    #
    times = list(it.chain.from_iterable(times))
    # parse solutions
    num_dep = problem.dummy_size
    sols = np.concatenate(sols, axis=0)
    s_parsed = []
    for sol_ in sols:
        src = 0
        tour_lst, lst = [], []
        for i in range(len(sol_)):
            tgt = sol_[src]
            if tgt < num_dep:
                if len(lst) > 0:
                    tour_lst.append(lst)
                lst = []
            else:
                lst.append(tgt)
            src = tgt
        s_parsed.append([[e-(num_dep-1) for e in l] for l in tour_lst])

    solutions = [
        RPSolution(
            solution=sol,
            run_time=t,
            problem=opts.problem,
            instance=inst,
            trajectory=trj,
        )
        for sol, t, inst, trj in zip(s_parsed, times, data, trajectories)
    ]

    return {}, solutions
