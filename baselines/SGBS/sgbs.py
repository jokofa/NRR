#
import logging
from typing import Dict, List, Tuple, Any
from timeit import default_timer as timer
from tqdm import tqdm
from lib.problem import RPInstance, RPSolution
from lib.nrr.sgbs_wrapper import SGBSSolver, get_sep_tours

logger = logging.getLogger(__name__)
CUDA_DEVICE_NUM = 0


class SGBS(SGBSSolver):

    def solve(
            self,
            instance: RPInstance,
            **kwargs
    ) -> RPSolution:

        t_start = timer()
        scores, solutions = self.runner.run(
            coords=instance.coords[None, :, :],
            demands=instance.demands[None, :],
            **kwargs
        )
        t_total = timer()-t_start
        if len(solutions) > 1:
            raise RuntimeError("should use BS=1 for testing!")

        sol = get_sep_tours(solutions)
        assert len(sol) == 1
        return RPSolution(
            solution=sol[0],
            run_time=t_total,
            problem="cvrp",
            instance=instance,
            trajectory=None,
        )


def eval_model(
        model: SGBS,
        data: List[RPInstance],
        **kwargs
        ) -> Tuple[Dict[str, Any], List[RPSolution]]:

    solutions = []
    for inst in tqdm(data):
        solutions.append(model.solve(inst, **kwargs))

    res = {}
    return res, solutions
