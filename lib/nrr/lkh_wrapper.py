#
import os
import warnings
import psutil
import tempfile
from typing import List, Union, Optional
from subprocess import check_call
from multiprocessing import Pool
from timeit import default_timer as timer
import numpy as np

from lib.nrr.base_solver import Solver, SGSolution
from lib.nrr.utils import NoSolutionFoundError, NF_MAP

FPP = 1e6
BACKUP_FNAME = "lkh.bkp"


def write_instance(
        problem: str,
        instance: List,
        instance_name: str,
        instance_filename: str,
        k: Optional[int] = None
):
    if problem.lower() == "cvrp":
        with open(instance_filename, "w") as f:
            N = len(instance[0]) - 1
            f.write("NAME : " + instance_name + "\n")
            f.write("COMMENT : blank\n")
            f.write("TYPE : CVRP\n")
            if k is not None:
                f.write("VEHICLES : " + str(int(k)) + "\n")
            f.write("DIMENSION : " + str(len(instance[0])) + "\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write("CAPACITY : " + str(instance[2]) + "\n")
            f.write("NODE_COORD_SECTION\n")
            for i in range(N + 1):
                f.write(" " + str(i + 1) + " " + str(instance[0][i][0])[:15] + " " + str(instance[0][i][1])[:15] + "\n")
            f.write("DEMAND_SECTION\n")
            f.write("1 0\n")
            for i in range(N):
                f.write(str(i + 2)+" "+str(instance[1][i])+"\n")
            f.write("DEPOT_SECTION\n 1\n -1\n")
            f.write("EOF\n")
    else:
        raise NotImplementedError(problem)


def write_params(
        instance_filename: str,
        para_filename: str,
        solution_filename: str,
        max_trials: int = 1000,
        time_limit: Optional[int] = None,
        runs: int = 1,
        seed: int = 1234,
        k: Optional[int] = None,
):
    #sol_backup = os.path.join(os.path.dirname(solution_filename), BACKUP_FNAME)
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = " + str(runs) + "\n")
        f.write("MTSP_MIN_SIZE = 0\n")
        f.write("SEED = " + str(seed) + "\n")
        if time_limit is not None:
            f.write("TIME_LIMIT = " + str(time_limit) + "\n")
        if k is not None:
            f.write(f"VEHICLES = {k}\n")
            f.write(f"SALESMEN = {k}\n")
        f.write("TRACE_LEVEL = 1\n")
        f.write(f"TOUR_FILE = {solution_filename}\n")   # to write best solution to file


def read_results(log_filename, sol_filename):
    # shortcut raising error in case no solution found
    if not os.path.exists(sol_filename):
        raise FileNotFoundError
    ###
    objs = []
    num_vehicles = -1
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:  # read the obj and runtime for each trial
            if "VEHICLES" in line:
                l = line.strip().split(" ")
                num_vehicles = int(l[-1])
            elif line[:6] == "-Trial":
                line = line.strip().split(" ")
                assert len(objs) + 1 == int(line[-4])
                objs.append(int(line[-2]))
            else:
                continue
        if "Best CVRP solution:" in lines[-3]:
            final_obj = int(lines[-2].split()[-1].split("_")[-1])
        else:
            final_obj = None

    if len(objs) > 0:
        assert objs[-1] == final_obj

    tours = []
    dim, total_length = 0, 0
    with open(sol_filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):    # read out solution tours
            if "DIMENSION" in line:
                l = line.strip().split(" ")
                dim = int(l[-1])
            elif "Length" in line:
                l = line.strip().split(" ")
                total_length = int(l[-1])
            elif i > 5 and not "EOF" in line:
                idx = int(line)
                if i == 6:
                    assert idx == 1
                tours.append(idx)

    assert tours[-1] == -1
    assert len(tours) == dim + 1
    N = dim-num_vehicles

    # reformat tours
    tours = (np.array(tours) - 1).tolist()  # reduce idx by 1 (since TSPLIB format starts at 1)
    plan = []
    t = [0]
    for n in tours[1:]:
        if n <= 0 or n > N:
            plan.append(t + [0])
            t = [0]
        else:
            t.append(n)
    assert len(plan) == num_vehicles

    # return objs, penalties, runtimes
    return {
        "objs": objs,
        "final_obj": final_obj/FPP if final_obj is not None else None,
        "N": N+1,   # sg size with depot
        "num_vehicles": num_vehicles,
        "total_length": total_length/FPP,
        "solution": plan,
    }


def find_lkh_k(log_filename):
    """Scan logfile to find number of vehicles k
    currently used by LKH."""
    k1, k2 = None, None
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "VEHICLES" in line:
                l = line.strip().split(" ")
                k1 = int(l[-1])
            elif "SALESMEN" in line:
                l = line.strip().split(" ")
                k2 = int(l[-1])
            else:
                continue

    if k1 is not None and k2 is not None:
        assert k1 == k2
    return k1 if k1 is not None else k2


def solve_LKH(
        exe_path: str,
        problem: str,
        instance: List,
        instance_name: str,
        max_trials: int = 1000,
        time_limit: Optional[int] = None,
        seed: int = 1234,
        k: Optional[int] = None,
        n_tries: int = 3
):
    t_start = timer()
    i = 0
    result = None
    with tempfile.TemporaryDirectory() as DIR:
        params_fname = os.path.join(DIR, "lkh.para")
        log_fname = os.path.join(DIR, "lkh.log")
        inst_fname = os.path.join(DIR, "lkh.cvrp")
        sol_fname = os.path.join(DIR, "lkh.sol")

        write_instance(
            problem=problem,
            instance=instance,
            instance_name=instance_name,
            instance_filename=inst_fname,
            k=k
        )
        write_params(
            instance_filename=inst_fname,
            para_filename=params_fname,
            solution_filename=sol_fname,
            max_trials=max_trials,
            time_limit=time_limit,
            seed=seed,
            runs=1
        )

        while i < n_tries:
            with open(log_fname, "w") as f:
                check_call([str(exe_path), params_fname], stdout=f)
            try:
                result = read_results(
                    log_filename=log_fname,
                    sol_filename=sol_fname
                )
            except FileNotFoundError:
                # could not solve!
                # ----
                # for some reason, the initial LKH method to find
                # the number of required vehicles k sometimes selects
                # a value which is too small to find a feasible solution
                # in reasonable time. Therefore if no solution is found
                # we retry with a larger k
                # ----
                # retry with more vehicles
                #print(f"retry {i}")
                k = find_lkh_k(log_fname)
                write_params(
                    instance_filename=inst_fname,
                    para_filename=params_fname,
                    solution_filename=sol_fname,
                    max_trials=max_trials,
                    time_limit=time_limit,
                    seed=seed,
                    runs=1,
                    k=(k+i+1)
                )
            else:
                break

    t_total = timer() - t_start
    if result is not None:
        result['runtime'] = t_total
    return result


def _lkh(args: tuple):
    return solve_LKH(*args)


class LKHSolver(Solver):

    def __init__(
            self,
            lkh_exe_pth: str,
            max_num_workers: int = -1,
            max_trials: int = 1000,
            time_limit: Optional[int] = None,
            seed: int = 1234,
    ):
        super().__init__(method="lkh")
        assert os.path.exists(lkh_exe_pth)
        self.lkh_exe_pth = lkh_exe_pth
        # only use physical cores
        phys_cores = psutil.cpu_count(logical=False)
        if max_num_workers <= 0:
            self.num_workers = phys_cores
        else:
            self.num_workers = min(max_num_workers, phys_cores)
        self.max_trials = max_trials
        self.time_limit = time_limit
        self.seed = seed

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
            # sg_node_features = [sg_node_features[i, :len(j)] for i, j in enumerate(sg_node_idx)]
            # coords = [f[:, x:y] for f in sg_node_features]
            # demands = [f[:, d] for f in sg_node_features]
            coords = [sg_node_features[i, :len(j), x:y] for i, j in enumerate(sg_node_idx)]
            demands = [sg_node_features[i, :len(j), d] for i, j in enumerate(sg_node_idx)]

        solutions = self._solve_cvrp(
            coords=coords,
            demands=demands,
            max_trials=max_trials if max_trials is not None else self.max_trials,
            time_limit=time_limit if time_limit is not None else self.time_limit,
        )

        return [SGSolution(
            num_nodes=sol['N'],
            num_vehicles=sol['num_vehicles'],
            routes=sol['solution'],
            cost=sol['total_length'],
            runtime=sol['runtime'],
        ) if sol is not None else None for sol in solutions]

    def _load_data(self):
        pass

    def _solve_cvrp(
            self,
            coords: List[np.ndarray],
            demands: List[np.ndarray],
            max_trials: int,
            time_limit: Optional[int] = None,
    ):
        problem = "CVRP"
        N = len(coords)
        num_workers = min(self.num_workers, N)
        # convert data to input format for LKH
        cap = int(FPP)
        instances = [
            [(c * FPP).astype(int).tolist(),
             # [1:, ...] since demand for depot node is always 0 and hardcoded in "write_instance"
             np.ceil(d[1:] * FPP).astype(int).tolist(),
             cap]
            for c, d in zip(coords, demands)
        ]
        K = None

        # run LKH
        if num_workers <= 1:
            solutions = [
                solve_LKH(
                    exe_path=self.lkh_exe_pth,
                    problem=problem,
                    instance=inst,
                    instance_name=str(i),
                    max_trials=max_trials,
                    time_limit=time_limit,
                    seed=self.seed,
                    k=K,
                )
                for i, inst in enumerate(instances)
            ]
        else:
            with Pool(num_workers) as pool:
                solutions = list(pool.imap(_lkh, [
                    (self.lkh_exe_pth, problem, inst, str(i),
                     max_trials, time_limit, self.seed, K)
                    for i, inst in enumerate(instances)
                ]))

        return solutions

