#
import psutil
from typing import List, Union, Optional
from multiprocessing import Pool
from timeit import default_timer as timer
import numpy as np
from scipy.spatial import distance_matrix as calc_distance_matrix
from verypy.classic_heuristics.cheapest_insertion import cheapest_insertion_init
from verypy.classic_heuristics.mole_jameson_insertion import mole_jameson_insertion_init
from verypy.util import sol2routes

from lib.nrr.base_solver import Solver, SGSolution
from lib.nrr.utils import NoSolutionFoundError, NF_MAP, compute_cost


def solve_best_insert(
        instance: List,
        mole_jameson: bool = True,
        **kwargs
):
    t_start = timer()
    coords, demands, cap = instance
    assert len(coords) == len(demands)

    dist_mat = calc_distance_matrix(coords, coords, p=2)
    solver = cheapest_insertion_init
    if mole_jameson:
        solver = mole_jameson_insertion_init
    solution = solver(
        D=dist_mat,
        d=demands.tolist(),
        C=cap,
        **kwargs
    )
    routes = sol2routes(solution)

    t_total = timer() - t_start
    return {
        "N": len(demands),   # sg size with depot
        "num_vehicles": len(routes),
        "cost": compute_cost(routes, dist_mat).sum(),
        "solution": routes,
        "runtime": t_total,
    }


def _solve(args: tuple):
    return solve_best_insert(*args)


class BestInsertSolver(Solver):

    def __init__(
            self,
            max_num_workers: int = -1,
            seed: int = 1234,
            mole_jameson: bool = True,
    ):
        super().__init__(method="lkh")
        # only use physical cores
        phys_cores = psutil.cpu_count(logical=False)
        if max_num_workers <= 0:
            self.num_workers = phys_cores
        else:
            self.num_workers = min(max_num_workers, phys_cores)
        self.seed = seed
        self.mole_jameson = mole_jameson

    def solve(
            self,
            sg_node_idx: List[tuple],
            sg_node_features: Union[List[np.ndarray], np.ndarray],
            max_trials: Optional[int] = None,
            time_limit: Optional[int] = None,
            **kwargs
    ) -> List[SGSolution]:
        x, y = NF_MAP["x"], NF_MAP["y"] + 1
        d = NF_MAP['demands']

        if len(sg_node_features) == 1:
            coords = [sg_node_features[0][:, x:y]]
            demands = [sg_node_features[0][:, d]]
        else:
            coords = [sg_node_features[i, :len(j), x:y] for i, j in enumerate(sg_node_idx)]
            demands = [sg_node_features[i, :len(j), d] for i, j in enumerate(sg_node_idx)]

        solutions = self._solve_cvrp(
            coords=coords,
            demands=demands,
            mole_jameson=self.mole_jameson,
            **kwargs
        )

        return [SGSolution(
            num_nodes=sol['N'],
            num_vehicles=sol['num_vehicles'],
            routes=sol['solution'],
            cost=sol['cost'],
            runtime=sol['runtime'],
        ) if sol is not None else None for sol in solutions]

    def _load_data(self):
        pass

    def _solve_cvrp(
            self,
            coords: List[np.ndarray],
            demands: List[np.ndarray],
            **kwargs
    ):
        N = len(coords)
        num_workers = min(self.num_workers, N)
        # convert data to input format for LKH
        cap = 1.0
        instances = [
            [c, d, cap]
            for c, d in zip(coords, demands)
        ]

        # run best insertion heuristic
        if num_workers <= 1:
            solutions = [
                solve_best_insert(
                    instance=inst,
                    #seed=self.seed,
                    **kwargs
                )
                for inst in instances
            ]
        else:
            with Pool(num_workers) as pool:
                solutions = list(pool.imap(_solve, [
                    (inst, kwargs) for inst in instances
                ]))

        return solutions
