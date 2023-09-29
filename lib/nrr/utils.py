#
from typing import List
import numpy as np
from verypy.classic_heuristics.parallel_savings import clarke_wright_savings_function
from verypy.classic_heuristics.gaskell_savings import gaskell_lambda_savings_function, gaskell_pi_savings_function
from verypy.classic_heuristics.sweep import bisect_angle


SAVINGS_FN = {
    'clarke_wright': clarke_wright_savings_function,
    'gaskell_lambda': gaskell_lambda_savings_function,
    'gaskell_pi': gaskell_pi_savings_function
}

SWEEP_DIRECTIONS = {
    "fw": [1],
    "bw": [-1],
    "both": [1, -1]
}

NODE_FEATURES = [
        "x", "y", "centered_x", "centered_y",
        "rho", "phi", "centered_rho", "centered_phi",
        "demands"
    ]
NF_MAP = {k: v for k, v in zip(NODE_FEATURES, range(len(NODE_FEATURES)))}



class NoSolutionFoundError(Exception):
    """Error class for sub-solver time-outs, etc."""

# class FileWriter:
#     """File writer based on numpy memmap using context manager."""
#     def __init__(self, file_path, nrows: int, **kwargs):
#         assert os.path.splitext(file_path)[-1] in [".dat", ".npy"]
#         assert nrows > 0
#         mode = 'r+' if os.path.exists(file_path) and os.path.isfile(file_path) else 'w+'
#         self.file_path = file_path
#         self.nrows = nrows
#         self._file = np.memmap(self.file_path, dtype=object, mode=mode, shape=(nrows,), **kwargs)
#         self._buffered = False
#         self._idx = 0
#         self._pos = 0
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, type, value, traceback):
#         self._file.flush()
#         del self._file
#         return True
#
#     def __len__(self):
#         return self._idx
#
#     def write_to_buffer(self, row: Any):
#         self._file[self._idx] = row
#         self._idx += 1
#         self._buffered = True
#         if self._idx >= self._pos + self.nrows:
#             # open new memmap slice
#             self.flush()
#             self._file = None
#             self._pos += self.nrows
#             self._file = np.memmap(
#                 self.file_path,
#                 dtype=object,
#                 mode="r+",
#                 shape=(self.nrows,),
#                 offset=self._pos
#             )
#
#     def flush(self):
#         self._file.flush()
#         self._buffered = False
#
#     def read(self, idx: Union[int, np.ndarray]):
#         if not self._buffered:
#             self.flush()
#         return self._file[idx].copy()


def compute_cost(routes: List[List], dist_mat: np.ndarray) -> np.ndarray:
    """calculate the cost of each route in solution."""
    costs = np.zeros(len(routes))
    for i, route in enumerate(routes):
        assert route[0] == route[-1] == 0
        costs[i] = dist_mat[route[:-1], route[1:]].sum()
    return costs


# ============================================================== #
# The code below was taken from the VeRyPy library and adapted
# to select a set of nodes as sub-graph consisting of routes
# https://github.com/yorak/VeRyPy/blob/master/verypy/classic_heuristics/sweep.py
# ============================================================== #
def get_sweep_from_polar_coordinates(rhos,phis):
    N = len(rhos)
    # stack the arrays, so that we can sort them (along different dimensions)
    customer_phirhos = np.stack( (phis, rhos, np.arange(N)) )
    sweep_node_order = np.argsort(customer_phirhos[0])
    sweep = customer_phirhos[:, sweep_node_order]
    return sweep


def _step(current, inc, max_val):
    current += inc
    if current > max_val:
        current = 0
    if current < 0:
        # reverse direction
        current = max_val
    return current


def sg_sweep(
        N: int,
        sizes: np.ndarray,
        target_size: int,
        sweep: np.ndarray,
        start: int,
        step_inc: int,
        debug: bool = False,
) -> List[List[int]]:
    """
    Sweeps a beam around the depot node to select a sub graph
    of size close to the specified target size.

    The provided nodes and their demands are not customer nodes,
    but route nodes, i.e. representing the center of the route and
    its total demand.
    """
    sweep_pos_to_node_idx = lambda idx: int(sweep[2, idx])
    assert len(sweep[0]) == len(sweep[2]) == N
    max_sweep_idx = N-1
    total_to_route = N

    # Routes
    sg_route_sets = []
    selected = np.zeros(N, dtype=bool)
    selected_cnt = 0

    # Emerging route
    current_sg = []
    current_sg_size = 0
    sg_complete = False

    # THE MAIN SWEEP LOOP
    # iterate until a full sweep is done and the backlog is empty
    sweep_pos = start
    sweep_node = sweep_pos_to_node_idx(sweep_pos)
    while True:
        if debug:
            if sweep_node:
                prev_pos = _step(sweep_pos, -step_inc, max_sweep_idx)
                next_pos = _step(sweep_pos, step_inc, max_sweep_idx)
                prev_ray = bisect_angle(sweep[0][prev_pos], sweep[0][sweep_pos], direction=step_inc)
                next_ray = bisect_angle(sweep[0][sweep_pos], sweep[0][next_pos], direction=step_inc)
                print("Considering n%d between rays %.2f, %.2f" % (sweep_node, prev_ray, next_ray))

        # we want at least two tours in each SG,
        # we only allow for 1 if there is only 1 left
        proper = len(current_sg) > 1 or (~selected).sum() == 1
        if not sg_complete and target_size:
            sg_complete = proper and (
                    # is smaller but close to target size
                    current_sg_size > target_size*0.85 or
                    # adding next tour would far exceed target size
                    current_sg_size + sizes[sweep_node] > target_size*1.15
            )

        if sg_complete:
            # If SG is complete, store it and start a new one
            # Check if we have all selected, and can exit the main sweep loop
            if proper:
                selected_cnt += len(current_sg)
                sg_route_sets.append(current_sg)

            if selected_cnt >= total_to_route or selected.all():
                break  # SWEEP

            current_sg = []
            current_sg_size = 0
            sg_complete = False

        if (sweep_node is not None) and (not selected[sweep_node]):
            current_sg.append(sweep_node)
            selected[sweep_node] = True
            if target_size:
                current_sg_size += sizes[sweep_node]

        start_stepping_from = sweep_pos
        while True:
            sweep_pos = _step(sweep_pos, step_inc, max_sweep_idx)
            sweep_node = sweep_pos_to_node_idx(sweep_pos)

            if (not selected[sweep_node]):
                break  # found an unselected node continue with it

            if sweep_pos == start_stepping_from:
                # We checked, and it seems there is no unselected non-blocked
                # nodes left -> start a new route, reset blocked and try again.
                sweep_node = None
                sg_complete = True
                break

    return sg_route_sets
