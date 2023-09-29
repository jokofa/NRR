#
from typing import List, Union, NamedTuple, Optional
import numpy as np


class SGSolution(NamedTuple):
    num_nodes: int
    num_vehicles: int
    routes: List[List[int]]
    cost: Optional[Union[int, float]] = None
    runtime: Optional[Union[int, float]] = None


class Solver:
    def __init__(
            self,
            method: str,
    ):
        self.method = method

    def solve(
            self,
            sg_node_idx: List[tuple],
            sg_node_features: Union[List[np.ndarray], np.ndarray],
            **kwargs
    ) -> List[SGSolution]:
        raise NotImplementedError

