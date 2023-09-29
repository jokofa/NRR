#
from typing import NamedTuple, Union, List, Optional, Dict
from collections import namedtuple
import numpy as np
import torch

__all__ = ["RPInstance", "RPSolution", "ScoringData"]


def format_repr(k, v, space: str = ' '):
    if isinstance(v, int) or isinstance(v, float):
        return f"{space}{k}={v}"
    elif isinstance(v, np.ndarray):
        return f"{space}{k}=ndarray_{list(v.shape)}"
    elif isinstance(v, torch.Tensor):
        return f"{space}{k}=tensor_{list(v.shape)}"
    elif isinstance(v, list) and len(v) > 3:
        return f"{space}{k}=list_{[len(v)]}"
    else:
        return f"{space}{k}={v}"


class RPInstance(NamedTuple):
    """Typed routing problem instance wrapper."""
    coords: Union[np.ndarray, torch.Tensor]
    demands: Union[np.ndarray, torch.Tensor]
    graph_size: int
    depot_idx: List = [0]
    vehicle_capacity: float = -1
    max_num_vehicles: Optional[int] = None
    time_windows: Optional[Union[np.ndarray, torch.Tensor]] = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def __eq__(self, other: NamedTuple):
        if not isinstance(other, type(self)):
            return False
        if list(self._fields) != list(other._fields):
            return False
        for k in self._fields:
            e1, e2 = self[k], other[k]
            e1 = e1.cpu().numpy() if isinstance(e1, torch.Tensor) else e1
            e2 = e2.cpu().numpy() if isinstance(e2, torch.Tensor) else e2
            if isinstance(e1, (np.ndarray, list, tuple)):
                if np.any(e1 != e2):
                    return False
            elif isinstance(e1, (np.generic, int, float)):
                if e1 != e2:
                    return False
            else:
                if e1 is not None and e2 is not None:
                    raise ValueError(e1, e2)
        return True

    def to_dict(self):
        return {k: self[k] for k in self._fields}

    @staticmethod
    def make(**kwargs):
        return RPInstance(**{k: kwargs.get(k, None) for k in RPInstance._fields})


class RPSolution(NamedTuple):
    """Typed wrapper for routing problem solutions."""
    solution: List[List]
    cost: float = None
    num_vehicles: int = None
    run_time: float = None
    problem: str = None
    instance: RPInstance = None
    trajectory: Optional[dict] = None

    def update(self, **kwargs):
        return self._replace(**kwargs)


class ScoringData(NamedTuple):
    """typed tuple for scoring function training data"""
    instance: RPInstance
    sg_old_routes: List[List[int]]
    sg_old_cost: float
    sg_new_cost: float
    # target = old_cost - new_cost
    sg_solver_time: float
    meta_iter: int
    sg_solver_cfg: Optional[Dict] = None
    sg_features: Optional[np.ndarray] = None

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def to_dict(self):
        return {
            k: self[k].to_dict() if k == 'instance' else self[k]
            for k in self._fields
        }

    @staticmethod
    def make(**kwargs):
        d = {k: kwargs.get(k, None) for k in ScoringData._fields}
        d['instance'] = RPInstance.make(**d['instance'])
        return ScoringData(**d)
