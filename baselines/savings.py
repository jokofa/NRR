#
from timeit import default_timer
from scipy.spatial import distance_matrix as calc_distance_matrix
from verypy.classic_heuristics.parallel_savings import parallel_savings_init, clarke_wright_savings_function
from verypy.classic_heuristics.gaskell_savings import gaskell_lambda_savings_function, gaskell_pi_savings_function
from verypy.util import sol2routes

from lib.problem import RPInstance

SAVINGS_FN = {
    'clarke_wright': clarke_wright_savings_function,
    'gaskell_lambda': gaskell_lambda_savings_function,
    'gaskell_pi': gaskell_pi_savings_function
}


def eval_savings(
        instance: RPInstance,
        min_k: bool = False,
        savings_function: str = 'clarke_wright',
        **kwargs
):
    savings_func = SAVINGS_FN[savings_function]

    demands = instance.demands.copy()
    vehicle_capacity = instance.vehicle_capacity

    t_start = default_timer()
    distances = calc_distance_matrix(instance.coords, instance.coords, p=2)
    solution = parallel_savings_init(
        D=distances, d=demands, C=vehicle_capacity,
        savings_callback=savings_func,
        minimize_K=min_k,
    )
    solution = sol2routes(solution)
    t_total = default_timer() - t_start

    return solution, t_total


def cost(routes, D):
    """calculate the cost of a solution"""
    cost = 0
    for route in routes:
        cost += sum([D[route[i], route[i + 1]] for i in range(len(route) - 1)])
    return cost
