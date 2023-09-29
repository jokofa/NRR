import os
from typing import List, Union, Optional, Dict
from timeit import default_timer as timer
import numpy as np
import torch

from lib.nrr.base_solver import Solver, SGSolution
from lib.nrr.utils import NF_MAP
from sgbs import CVRPModel, E_CVRPEnv, CVRPTester


def get_sep_tours(
        sols: torch.Tensor,
        max_node_idx: Optional[Union[List, np.ndarray]] = None
) -> List[List]:
    """get solution (res) as List[List]"""
    max_node_idx = [None]*len(sols) if max_node_idx is None else max_node_idx
    # parse solution
    return [sol_to_list(sol, depot_idx=0, max_node_idx=mx) for sol, mx in zip(sols, max_node_idx)]


# Transform solution returned from POMO to List[List]
def sol_to_list(
        sol: Union[torch.tensor, np.ndarray],
        depot_idx: int = 0,
        max_node_idx: int = None
) -> List[List]:
    if isinstance(sol, torch.Tensor):
        sol = sol.cpu().numpy()
    sol_lst, lst = [], [0]
    for e in sol:
        if e == depot_idx or max_node_idx is not None and e >= max_node_idx:
            if len(lst) > 1:
                lst.append(0)
                sol_lst.append(lst)
                lst = [0]
        # elif max_node_idx is not None and e >= max_node_idx:
        #     continue
        else:
            lst.append(e)
    return sol_lst


class SGBSSolver(Solver):

    def __init__(
            self,
            ckpt_pth: str,
            model_cfg: Optional[Dict] = None,
            tester_cfg: Optional[Dict] = None,
            cuda: bool = True,
            seed: int = 1234,
            mode: str = "sgbs",
            beta: int = 4,      # beam width
            gamma: int = 4,     # expansion factor
            n_samples: int = 1200,
    ):
        super().__init__(method="sgbs")
        assert os.path.exists(ckpt_pth)
        self.ckpt_pth = ckpt_pth
        self._device = torch.device("cuda") \
            if cuda and torch.cuda.is_available() \
            else torch.device('cpu')

        self.mode = mode
        self.beta = beta
        self.gamma = gamma
        self.n_samples = n_samples

        tester_cfg = {} if tester_cfg is None else tester_cfg
        augment = tester_cfg.pop('augment', True)
        tester_params = {
            'augmentation_enable': augment,
            'aug_factor': 8 if augment else 0,
            'mode': mode,
            'sampling_num': n_samples,  # for sampling
            'obs_bw': 1200,  # beam_wdith of original beam search
            'mcts_rollout_per_step': 12,  # number of rollout per step of mcts
            'sgbs_beta': beta,
            'sgbs_gamma_minus1': (gamma - 1)
        }
        tester_params.update(tester_cfg)

        model_cfg = {} if model_cfg is None else model_cfg
        model_params = {
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
        }
        model_params.update(model_cfg)

        self.env = E_CVRPEnv(device=self._device)
        self.model = CVRPModel(**model_params)
        ckpt = torch.load(ckpt_pth, map_location=self._device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        print(f"cuda: {cuda}")
        self.runner = CVRPTester(
            env=self.env,
            model=self.model,
            tester_params=tester_params,
            cuda=cuda
        )

    def solve(
            self,
            sg_node_idx: List[tuple],
            sg_node_features: np.ndarray,
            **kwargs
    ) -> List[SGSolution]:
        x, y = NF_MAP["centered_x"], NF_MAP["centered_y"] + 1
        d = NF_MAP['demands']

        if isinstance(sg_node_features, list):
            assert len(sg_node_features) == 1
            coords = sg_node_features[0][None, :, x:y]
            demands = sg_node_features[0][None, :, d]
        else:
            coords = sg_node_features[:, :, x:y]
            demands = sg_node_features[:, :, d]

        t_start = timer()
        scores, solutions = self.runner.run(
            coords=coords,
            demands=demands,
            **kwargs
        )
        t_total = timer()-t_start
        if len(solutions) > 1:
            # when BS > 1, it is hard to quantify the exact runtime per instance
            t_total /= 2

        # for sol, dmd, mxidx in zip(solutions.cpu().numpy(), demands, [len(ni) for ni in sg_node_idx]):
        #     rt = []
        #     for i in sol:
        #         if i == 0:
        #             if dmd[rt].sum() > 1.0001:
        #                 a=1
        #             rt = []
        #         else:
        #             rt.append(i)

        solutions = get_sep_tours(solutions, max_node_idx=[len(ni) for ni in sg_node_idx])
        # if DEBUG:
        #     for sol, dmd, mxidx in zip(solutions, demands, [len(ni) for ni in sg_node_idx]):
        #         for r in sol:
        #             if dmd[r].sum() > 1.0001:
        #                 raise RuntimeError(r, dmd[r].sum(), mxidx)

        return [SGSolution(
            num_nodes=len(sg_node_idx[i]),
            num_vehicles=len(sol),
            routes=sol,
            cost=scores[i],
            runtime=t_total
        ) for i, sol in enumerate(solutions)]
