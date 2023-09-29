#
import pickle
import os
import shutil
from typing import List, Dict, Optional, Union, Tuple
from copy import deepcopy
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm
import itertools as it
import warnings
import numpy as np
import torch
from scipy.special import softmax
from scipy.spatial import distance_matrix as calc_distance_matrix

from verypy.classic_heuristics.sweep import sweep_init, SMALLEST_ANGLE, cart2pol
from verypy.classic_heuristics.parallel_savings import parallel_savings_init
from verypy.util import sol2routes

from lib.problem import RPInstance, ScoringData
from lib.nrr.utils import (
    SAVINGS_FN,
    compute_cost,
    get_sweep_from_polar_coordinates,
    sg_sweep,
    SWEEP_DIRECTIONS,
)
from lib.nrr.base_solver import Solver
from lib.nrr.sa import SimulatedAnnealing
from lib.model.graph_utils import GraphNeighborhoodSampler


EPS = np.finfo(np.float32).eps
DEBUG = False
if DEBUG:
    warnings.warn(f">>> DEBUG flag = True <<<")


class NRRSolution:
    """Solution buffer providing additional functionality."""
    def __init__(
            self,
            routes: List[List[int]],
            instance: RPInstance,
            route_cost: Union[np.ndarray, List],
    ):
        self.routes = np.array([np.array(r) for r in routes], dtype=object)
        self.padded_routes = self.pad_routes(self.routes)
        self.instance = instance
        self.centered_coords = instance.coords-instance.coords[0, :]
        self.polar_coords = np.stack((
            # polar coords
            *cart2pol(
                instance.coords[:, 0],
                instance.coords[:, 1]
            ),
            # depot centered polar coords
            *cart2pol(
                # instance.coords[:, 0]-instance.coords[0, 0],
                # instance.coords[:, 1]-instance.coords[0, 1]
                self.centered_coords[:, 0], self.centered_coords[:, 1]
            )
        ), axis=-1)
        # gather node features as specified in NODE_FEATURES
        self.nodes = np.concatenate([
            instance.coords,    # x, y
            self.centered_coords,
            self.polar_coords,  # polar rho, phi, centered rho, phi
            instance.demands[:, None]   # demand
        ], axis=-1)

        assert len(route_cost) == len(routes)
        self.route_cost = np.array(route_cost)
        self.total_cost = route_cost.sum()

        self.has_changed = np.ones(self.num_vehicles, dtype=bool)
        self.centers = np.empty((self.num_vehicles, self.instance.coords.shape[-1]), dtype=float)
        self.demands = np.empty(self.num_vehicles, dtype=float)
        self.sizes = np.empty(self.num_vehicles, dtype=int)

    @property
    def num_vehicles(self):
        return len(self.routes)

    def _compute_route_centers(self) -> np.ndarray:
        """Compute the centers of tours."""
        coords = self.instance.coords
        if self.has_changed.any():
            self.centers[self.has_changed] = np.array([
                coords[r[:-1]].mean(axis=0)
                for r in self.routes[self.has_changed]
            ])
        return self.centers.copy()

    def _compute_route_demands(self) -> np.ndarray:
        d = self.instance.demands
        if self.has_changed.any():
            self.demands[self.has_changed] = np.array([
                d[r[:-1]].sum(axis=0)
                for r in self.routes[self.has_changed]
            ])
        return self.demands.copy()

    def _compute_route_sizes(self):
        if self.has_changed.any():
            self.sizes[self.has_changed] = np.array([
                len(r)-2    # not counting depot idx at front and back
                for r in self.routes[self.has_changed]
            ])
        return self.sizes.copy()

    def route_info(self):
        c = self._compute_route_centers()
        d = self._compute_route_demands()
        s = self._compute_route_sizes()
        self.has_changed *= False
        return c, d, s

    @staticmethod
    def pad_routes(
            routes: Union[List[Union[List[int], Tuple]], List[np.ndarray], np.ndarray],
            max_len: Optional[int] = None
    ):
        """Right pad routes with zeros."""
        if max_len is None:
            max_len = max(len(r) for r in routes)
        return np.stack([np.pad(r, (0, max_len-len(r)), 'constant') for r in routes])

    def update(
            self,
            route_idx: np.ndarray,
            route_cost: np.ndarray,
            routes: List[List[int]],
    ):
        """update solution with new routes in SG."""
        # replace old routes
        k = len(routes)
        k_org = len(route_idx)
        max_len = np.max([len(r) for r in routes])
        max_len_org = self.padded_routes.shape[-1]
        assert len(route_cost) == k
        if k > k_org:
            # add new rows at end
            n_insert = k-k_org
            add_idx = len(self.routes)
            self.routes = np.insert(self.routes, add_idx, np.empty(n_insert))
            self.route_cost = np.insert(self.route_cost, add_idx, np.empty(n_insert))
            self.padded_routes = np.append(self.padded_routes,
                                           np.empty((n_insert, self.padded_routes.shape[-1]),
                                                    dtype=self.padded_routes.dtype),
                                           axis=0)
            self.has_changed = np.insert(self.has_changed, add_idx, np.zeros(n_insert))
            self.centers = np.append(self.centers, np.empty((n_insert, 2)), axis=0)
            self.demands = np.insert(self.demands, add_idx, np.empty(n_insert))
            self.sizes = np.insert(self.sizes, add_idx, np.empty(n_insert, dtype=self.sizes.dtype))
            # add new final idx to route_idx
            route_idx = np.insert(route_idx, len(route_idx), np.arange(n_insert)+add_idx)
        
        rep_idx = route_idx[:k]     # replace
        del_idx = route_idx[k:]     # delete
        for i, r in zip(rep_idx, routes):
            self.routes[i] = np.array(r)
        if DEBUG:
            # check if all nodes are routed
            routed = (
                len(np.unique(np.concatenate(self.routes, axis=-1)))
                == len(self.instance.coords)
            )
            if not routed:
                raise RuntimeError("Not all nodes were routed!")

        self.route_cost[rep_idx] = route_cost
        self.total_cost = self.route_cost.sum()
        if max_len > max_len_org:
            # add columns to accommodate longer routes
            self.padded_routes = np.pad(
                self.padded_routes, ((0, 0), (0, max_len-max_len_org))
            )
            max_len_org = max_len
        self.padded_routes[rep_idx] = self.pad_routes(routes, max_len=max_len_org)
        self.has_changed[rep_idx] = True    # flag to recompute centers
        if len(del_idx) > 0:
            # delete redundant rows
            self.routes = np.delete(self.routes, del_idx, axis=0)
            self.route_cost = np.delete(self.route_cost, del_idx, axis=0)
            self.padded_routes = np.delete(self.padded_routes, del_idx, axis=0)
            self.has_changed = np.delete(self.has_changed, del_idx, axis=0)
            self.centers = np.delete(self.centers, del_idx, axis=0)
            self.demands = np.delete(self.demands, del_idx, axis=0)
            self.sizes = np.delete(self.sizes, del_idx, axis=0)

        if DEBUG:
            cum_dmd = np.array([self.instance.demands[r].sum() for r in self.routes])
            violations = (cum_dmd > 1.0001)
            if violations.any():
                raise RuntimeError(violations, self.routes[violations], cum_dmd[violations])

    def multi_update(
            self,
            route_idx: List[np.ndarray],
            route_cost: List[np.ndarray],
            routes: List[List[List[int]]],
    ):
        """Update solution with multiple disjoint SGs."""
        ks = np.array([len(r) for r in routes])
        org_ks = np.array([len(i) for i in route_idx])
        assert all([len(r) == k for r, k in zip(route_cost, ks)])

        ###
        #route_idx = np.array(list(it.chain.from_iterable(route_idx)))
        route_cost = np.array(list(it.chain.from_iterable(route_cost)))
        routes = [r for rts in routes for r in rts]

        max_len = np.max([len(r) for r in routes])
        max_len_org = self.padded_routes.shape[-1]

        if (ks-org_ks > 0).any():
            # add new rows at end
            k_per_sg = ks-org_ks
            insert = k_per_sg > 0
            n_insert = k_per_sg[insert].sum()
            add_idx = len(self.routes)
            self.routes = np.insert(self.routes, add_idx, np.empty(n_insert))
            self.route_cost = np.insert(self.route_cost, add_idx, np.empty(n_insert))
            self.padded_routes = np.append(self.padded_routes,
                                           np.empty((n_insert, self.padded_routes.shape[-1]),
                                                    dtype=self.padded_routes.dtype),
                                           axis=0)
            self.has_changed = np.insert(self.has_changed, add_idx, np.zeros(n_insert))
            self.centers = np.append(self.centers, np.empty((n_insert, 2)), axis=0)
            self.demands = np.insert(self.demands, add_idx, np.empty(n_insert))
            self.sizes = np.insert(self.sizes, add_idx, np.empty(n_insert, dtype=self.sizes.dtype))
            # add new added indices to route_idx
            for i, n_ins in enumerate(k_per_sg):
                if n_ins > 0:
                    route_idx[i] = np.insert(route_idx[i], len(route_idx[i]), np.arange(n_ins)+add_idx)
                    add_idx += n_ins

        rep_idx = np.array(list(it.chain.from_iterable([idx[:k] for idx, k in zip(route_idx, ks)])))  # replace
        del_idx = np.array(list(it.chain.from_iterable([idx[k:] for idx, k in zip(route_idx, ks)])))  # delete
        for i, r in zip(rep_idx, routes):
            self.routes[i] = np.array(r)
        if DEBUG:
            # check if all nodes are routed
            routed = (
                    len(np.unique(np.concatenate(self.routes, axis=-1)))
                    == len(self.instance.coords)
            )
            if not routed:
                raise RuntimeError("Not all nodes were routed!")

        self.route_cost[rep_idx] = route_cost
        self.total_cost = self.route_cost.sum()
        if max_len > max_len_org:
            # add columns to accommodate longer routes
            self.padded_routes = np.pad(
                self.padded_routes, ((0, 0), (0, max_len-max_len_org))
            )
            max_len_org = max_len
        self.padded_routes[rep_idx] = self.pad_routes(routes, max_len=max_len_org)
        self.has_changed[rep_idx] = True    # flag to recompute centers
        if len(del_idx) > 0:
            # delete redundant rows
            self.routes = np.delete(self.routes, del_idx, axis=0)
            self.route_cost = np.delete(self.route_cost, del_idx, axis=0)
            self.padded_routes = np.delete(self.padded_routes, del_idx, axis=0)
            self.has_changed = np.delete(self.has_changed, del_idx, axis=0)
            self.centers = np.delete(self.centers, del_idx, axis=0)
            self.demands = np.delete(self.demands, del_idx, axis=0)
            self.sizes = np.delete(self.sizes, del_idx, axis=0)

        if DEBUG:
            violations = (np.array([self.instance.demands[r].sum() for r in self.routes]) > 1.0)
            if violations.any():
                raise RuntimeError(violations, self.routes[violations])

    def get_nodes_from_route_set(self, idx: Union[np.ndarray, List[np.ndarray]]):

        c, d, s = self.route_info()
        if isinstance(idx, np.ndarray):
            raise NotImplementedError
            assert len(idx.shape) == 2
            assert idx.dtype != object
            rts = self.padded_routes[idx].reshape(self.num_vehicles, -1)
            max_nz = (rts > 0).sum(-1).max() + 1  # +1 for depot
            rts = [np.insert(r[r > 0], 0, 0) for r in rts]  # inserting depot idx at first position
            route_features = [[c[idx], d[idx], s[idx]]]
        else:
            rts = [self.padded_routes[i].reshape(-1) for i in idx]
            rts = [np.insert(r[r>0], 0, 0) for r in rts]    # inserting depot idx at first position
            lns = np.array([len(r) for r in rts])
            max_nz = lns.max()
            min_ln = lns.min()
            if min_ln < 2:
                print(f"min len: {min_ln}")
            route_features = [np.concatenate([
                c[i].mean(0),
                d[i][:, None].sum(0),
                s[i][:, None].sum(0)/self.instance.graph_size
            ], axis=-1) for i in idx]

        nodes = [self.nodes[r] for r in rts]
        nodes = [np.pad(nd, ((0, max_nz-nd.shape[0]), (0, 0)), 'constant') for nd in nodes]
        return rts, nodes, route_features

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(" \
               f"total_cost: {self.total_cost: .8f}, " \
               f"num_vehicles: {self.num_vehicles}, " \
               f"routes: \n{self.routes}" \
               f"\n)"


class NRR:
    """Neural Ruin-Recreate Algorithm."""
    INIT_METHODS = ["sweep", "savings", "pomo"]
    CONSTR_METHODS = ["sweep", "tour_knn", "tour_nn_add"]
    SCORING_METHODS = ["nsf", "rnd"]
    SELECT_METHODS = ["greedy", "sampling", "multi", "disjoint"]
    ACCEPT_METHODS = ["all", "greedy", "sa"]

    def __init__(
            self,
            sg_solver: Solver,
            max_iters: int = 1000,
            time_limit: Optional[Union[int, float]] = None,
            init_method: str = "sweep",
            init_method_cfg: Optional[Dict] = None,
            sg_construction_method: str = "sweep",
            sg_construction_method_cfg: Optional[Dict] = None,
            scoring_method: str = "nsf",
            scoring_method_cfg: Optional[Dict] = None,
            scoring_fn: Optional = None,
            sg_selection_method: str = "greedy",
            sg_selection_method_cfg: Optional[Dict] = None,
            accept_method: str = "greedy",
            accept_method_cfg: Optional[Dict] = None,
            tabu_iters: int = 10,
            scale_trials_by_iter: float = 0.0,
            seed: int = 1234,
            verbose: int = 0,
            save_trajectory: bool = False,
    ):

        self.solver = sg_solver
        self.max_iters = max_iters
        self.time_limit = time_limit-EPS if time_limit is not None else 1e9
        self.init_method = init_method.lower()
        assert self.init_method in self.INIT_METHODS
        self.init_method_cfg = init_method_cfg \
            if init_method_cfg is not None else {}
        self.sg_construction_method = sg_construction_method.lower()
        assert self.sg_construction_method in self.CONSTR_METHODS
        self.sg_construction_method_cfg = sg_construction_method_cfg \
            if sg_construction_method_cfg is not None else {}
        self.scoring_method = scoring_method.lower()
        assert self.scoring_method in self.SCORING_METHODS
        self.scoring_method_cfg = scoring_method_cfg \
            if scoring_method_cfg is not None else {}
        self.scoring_fn = scoring_fn
        self.sg_selection_method = sg_selection_method.lower()
        assert self.sg_selection_method in self.SELECT_METHODS
        if self.sg_selection_method == "disjoint":
            assert self.sg_construction_method == "sweep", \
                f"selection method '{self.sg_selection_method}' " \
                f"requires construction method 'sweep' " \
                f"but got '{self.sg_construction_method}'!"
        self.sg_selection_method_cfg = sg_selection_method_cfg \
            if sg_selection_method_cfg is not None else {}
        if self.sg_selection_method == "multi":
            if hasattr(self.solver, "_device") or hasattr(self.solver, "cuda"):
                if str(self.solver._device) != 'cuda':
                    warnings.warn(f"Using sg_selection_method: 'multi' "
                                  f"without CUDA for solver. "
                                  f"Using CUDA can be significantly faster.")

        self.accept_method = accept_method.lower()
        assert self.accept_method in self.ACCEPT_METHODS
        self.accept_method_cfg = accept_method_cfg \
            if accept_method_cfg is not None else {}
        self.sa = None
        if self.accept_method == "sa":
            self.sa = SimulatedAnnealing(
                num_max_steps=max_iters,
                seed=seed,
                **self.accept_method_cfg
            )
        self.tabu_iters = tabu_iters
        self.scale_trials_by_iter = scale_trials_by_iter
        if self.scale_trials_by_iter > 0:
            raise NotImplementedError   # TODO
        self.seed = seed
        self.verbose = verbose
        self.save_trajectory = save_trajectory

        self.instance = None
        self.dist_mat = None

        self._device = None
        if self.scoring_method == "nsf":
            assert self.scoring_fn is not None
            cuda = self.scoring_method_cfg['cuda'] and torch.cuda.is_available()
            self._device = torch.device("cuda" if cuda else "cpu")
            self.scoring_fn.to(self._device)
            self.scoring_fn.eval()

        self._rng = None
        self._current_iter = None
        self._score_hash_table = None
        self._sg_tabu_list = None
        self._restart_idx = None
        self._t_start = None
        self._data_buffer = None
        self._trj_buffer = None
        self._static_node_emb = None
        self._static_graph_emb = None

    def _reset(self):
        self._rng = np.random.default_rng(self.seed)
        self._current_iter = 0
        self._score_hash_table = {}
        self._sg_tabu_list = {}
        self._restart_idx = 1
        self._t_start = timer()
        self._data_buffer = None
        self._trj_buffer = []
        self._static_node_emb = None
        self._static_graph_emb = None
        self.nbh_graph = None
        if self.sa is not None:
            self.sa.reset()

    def load_instances(self, instance: RPInstance):
        """Load problem instance."""
        if len(instance.depot_idx) > 1:
            raise ValueError("multi depot VRP not supported.")
        self.instance = instance
        # if self._creat_static_graph:
        #     self.construct_static_graph()

    def solve_initial(self) -> NRRSolution:
        """Create an initial feasible solution."""
        coords = self.instance.coords.copy()
        demands = self.instance.demands.copy()
        vehicle_capacity = self.instance.vehicle_capacity
        # TODO: fp16 dist_mat for large problems?
        self.dist_mat = calc_distance_matrix(coords, coords, p=2)

        if self.scoring_method == "pomo":
            # pomo greedy rollout for initial solution
            raise NotImplementedError
        else:
            if self.init_method == "sweep":
                solution = sweep_init(
                    coordinates=coords,
                    D=self.dist_mat,
                    d=demands,
                    C=vehicle_capacity,
                    direction="both",
                    seed_node=SMALLEST_ANGLE,
                    **self.init_method_cfg
                )
            elif self.init_method == "savings":
                sf = self.init_method_cfg.pop('savings_function',
                                              'clarke_wright')
                savings_func = SAVINGS_FN[sf]
                solution = parallel_savings_init(
                    D=self.dist_mat,
                    d=demands,
                    C=vehicle_capacity,
                    savings_callback=savings_func,
                    **self.init_method_cfg
                )
            else:
                raise ValueError(f"unknown init method: "
                                 f"{self.init_method}.")
            routes = sol2routes(solution)

        return NRRSolution(
            routes=routes,
            instance=deepcopy(self.instance),
            route_cost=compute_cost(routes, self.dist_mat)
        )

    def construct_static_graph(self, cur_sol: NRRSolution):
        """Construct static global NBH graph."""
        knn_graph = GraphNeighborhoodSampler(
            graph_size=self.instance.graph_size,
            k_frac=self.scoring_method_cfg['knn'],
            num_workers=4,
        )
        e, w, _ = knn_graph(torch.from_numpy(cur_sol.instance.coords))
        self._static_node_emb, self._static_graph_emb = self.scoring_fn.encode_graph(
                node_features=torch.from_numpy(cur_sol.nodes).to(self._device).to(torch.float).unsqueeze(0),
                edges_e=e.to(self._device).view(2, -1),
                edges_w=w.to(self._device).to(torch.float).view(-1),
            )

    def construct_subgraphs(self, sol: NRRSolution):
        """Retrieve set of subgraphs."""
        if self.verbose > 1:
            print("constructing SGs...")
        # compute tour centers, demands and size (number of nodes)
        r_centers, r_demands, r_sizes = sol.route_info()

        if self.sg_construction_method == "tour_knn":
            # +1 since we also select tour itself
            k = self.sg_construction_method_cfg['knn'] + 1
            k = min(k, len(r_centers)-1)
            # find k nearest neighbors for each
            dists = calc_distance_matrix(r_centers, r_centers, p=2)
            route_idx = np.argpartition(dists, k, axis=0)[:k].T  # KNN

        elif self.sg_construction_method == "tour_nn_add":
            # add tours in vicinity until number of nodes close to target is reached
            n_target = self.sg_construction_method_cfg['n_target']
            # to heuristically determine an appropriate number of nearest tours
            # we simply use their median size
            # +1 since we also select tour itself
            k = int(np.ceil(n_target / np.median(r_sizes))) + 1
            k = min(k, len(r_centers)-1)
            # find k nearest neighbors for each
            dists = calc_distance_matrix(r_centers, r_centers, p=2)
            route_idx = np.argpartition(dists, k, axis=0)[:k].T  # KNN

        elif self.sg_construction_method == "sweep":
            # compute depot-centered tour centers
            c_centers = r_centers-self.instance.coords[0]
            rho, phi = cart2pol(c_centers[:, 0], c_centers[:, 1])
            sweep = get_sweep_from_polar_coordinates(rhos=rho, phis=phi)
            n_nodes_per_route = r_sizes
            #
            n_target = self.sg_construction_method_cfg['n_target']
            direction = self.sg_construction_method_cfg['sweep_direction'].lower()
            if self.sg_selection_method == "disjoint":
                # to create disjoint SGs we only do one sweep.
                # the depot node can be part of every SG.
                n_starts = 1
                direction = "fw"
            else:
                alpha = self.sg_construction_method_cfg.get('alpha', 1.0)
                k = sol.num_vehicles
                gs = self.instance.graph_size
                div = 2 if direction == "both" else 1
                n_starts = int(np.ceil(max(min(alpha * gs/n_target * np.sqrt(k), k+1)/div, 1)))

            step_incs = SWEEP_DIRECTIONS[direction]
            mx = max(sol.num_vehicles-n_starts, 1)
            start = self._rng.integers(0, mx)
            route_idx = []
            for i in range(n_starts):
                for inc in step_incs:
                    route_idx += sg_sweep(
                        N=len(c_centers),
                        sizes=n_nodes_per_route,
                        target_size=n_target,
                        sweep=sweep,
                        start=start+i,
                        step_inc=inc,
                    )
        else:
            raise ValueError(f"unknown sg_construction_method: "
                             f"{self.sg_construction_method}.")

        # remove possible duplicate SGs
        route_idx = set(tuple(r) for r in route_idx)
        sg_route_idx = [np.array(list(r)) for r in route_idx]

        sg_node_idx, sg_node_features, sg_route_features = sol.get_nodes_from_route_set(sg_route_idx)
        # convert to immutable (and hashable!) tuples
        sg_node_idx = [tuple(i.tolist()) for i in sg_node_idx]
        if len(self._sg_tabu_list) > 0:
            # remove entries from tabu list after tabu_iters iterations
            rm_keys = [k for k, v in self._sg_tabu_list.items()
                       if self._current_iter - v >= self.tabu_iters]
            for k in rm_keys:
                self._sg_tabu_list.pop(k)
            # check and possibly remove rejected SGs
            # a SG is uniquely defined by the set of its nodes (indices)
            rm_idx = [i for i, idx in enumerate(sg_node_idx) if hash(idx) in self._sg_tabu_list]
            # check how many SGs would remain
            sg_remaining = len(sg_route_idx) - len(rm_idx)
            if sg_remaining < 2:
                # reinstate 2 oldest SGs from tabu list
                tabu_scores = [
                    self._current_iter - self._sg_tabu_list.get(hash(idx), self._current_iter+1)
                    for idx in sg_node_idx
                ]
                rm_rm_idx = np.argsort(tabu_scores)[-2:]
                for i in rm_rm_idx:
                    try:
                        rm_idx.remove(i)
                    except ValueError:
                        pass

            q = 0
            for i in rm_idx:
                i -= q
                if isinstance(sg_route_idx, list):
                    del sg_route_idx[i]
                del sg_node_idx[i]
                del sg_node_features[i]
                del sg_route_features[i]
                q += 1
            if isinstance(sg_route_idx, np.ndarray):
                sg_route_idx = np.delete(sg_route_idx, rm_idx, axis=0)

        return sg_route_idx, sg_node_idx, np.stack(sg_node_features), np.stack(sg_route_features)

    def score_subgraphs(
            self,
            sg_node_idx: List[tuple],
            sg_route_idx: np.ndarray,
            sg_route_features: np.ndarray,
            cur_sol: NRRSolution,
    ) -> np.ndarray:
        """Score sub-graphs with scoring model."""
        if self.verbose > 1:
            print("scoring SGs...")
        if self.scoring_method == "nsf":
            if self._static_node_emb is None or self._static_graph_emb is None:
                self.construct_static_graph(cur_sol)
            sg_node_idx = NRRSolution.pad_routes(sg_node_idx)
            old_costs = np.array([cur_sol.route_cost[ridx].sum() for ridx in sg_route_idx])
            sg_route_features = np.concatenate((
                np.tile([self._current_iter], len(sg_node_idx))[:, None],    # meta iter
                old_costs[:, None],     # current SG cost
                sg_route_features
            ), axis=-1)
            scores = self.scoring_fn.score(
                sg_node_idx=torch.from_numpy(sg_node_idx).to(self._device, dtype=torch.long),
                sg_meta_features=torch.from_numpy(sg_route_features).to(self._device, dtype=torch.float32),
                node_emb=self._static_node_emb,
                graph_emb=self._static_graph_emb,
            ).cpu().numpy()
        elif self.scoring_method == "rnd":
            scores = np.arange(len(sg_node_idx))/len(sg_node_idx)
            self._rng.shuffle(scores)
        else:
            raise ValueError(f"unknown scoring_method: "
                             f"{self.scoring_method}.")

        return scores

    def select_subgraph(
            self,
            sg_route_idx: np.ndarray,
            sg_node_idx: List[tuple],
            sg_node_features: np.ndarray,
            scores: np.ndarray,
            sg_route_features: Optional[np.ndarray] = None,
    ):
        """Select subgraph according to score and strategy."""
        if self.sg_selection_method == "greedy":
            # greedy selection of SG with highest score
            idx = np.argmax(scores)
            sg_route_idx = sg_route_idx[idx][None, :]
            sg_node_idx = [sg_node_idx[idx]]
            # select only real nodes (skip dummy idx > len(sg_node_idx))
            sg_node_features = [sg_node_features[idx][:len(sg_node_idx[0]), :]]
            if sg_route_features is not None:
                sg_route_features = [sg_route_features[idx]]
        elif self.sg_selection_method == "sampling":
            # select SG according to its probability based on
            # the scores normalized via softmax
            idx = self._rng.choice(np.arange(len(scores)), 1,
                                   p=softmax(scores))[0]
            sg_route_idx = sg_route_idx[idx][None, :]
            sg_node_idx = [sg_node_idx[idx]]
            # select only real nodes (skip dummy idx > len(sg_node_idx))
            sg_node_features = [sg_node_features[idx][:len(sg_node_idx[0]), :]]
            if sg_route_features is not None:
                sg_route_features = [sg_route_features[idx]]
        elif self.sg_selection_method in ["multi", "disjoint"]:
            # sample multiple SGs to solve
            n = max(self.sg_selection_method_cfg['n_select'], 1)
            n = min(n, len(scores))
            idx = self._rng.choice(np.arange(len(scores)), n,
                                   replace=False, p=softmax(scores))
            sg_route_idx = [sg_route_idx[i] for i in idx]
            sg_node_idx = [sg_node_idx[i] for i in idx]
            sg_node_features = sg_node_features[idx]
            if sg_route_features is not None:
                sg_route_features = sg_route_features[idx]

            if DEBUG and self.sg_selection_method == "disjoint":
                # check all SGs are disjoint
                ridx = list(it.chain.from_iterable(sg_route_idx))
                assert len(ridx) == len(np.unique(ridx))
                nidx = np.array(list(it.chain.from_iterable(sg_node_idx)))
                nidx = nidx[nidx > 0]
                assert len(nidx) == len(np.unique(nidx))

        else:
            raise ValueError(f"unknown sg_selection_method: "
                             f"{self.sg_selection_method}.")
        return sg_route_idx, sg_node_idx, sg_node_features, sg_route_features

    def solve_subgraph(
            self,
            sg_route_idx: Union[List[np.ndarray], np.ndarray],
            sg_node_idx: Union[List[tuple], tuple],
            sg_node_features: Union[List[np.ndarray], np.ndarray],
            cur_sol: NRRSolution,
            **kwargs
    ) -> NRRSolution:
        """Ruin and recreate subgraph solution."""
        if self.verbose > 1:
            print("solving SGs...")
        # the full SG is 'ruined' by removing all edges
        # and re-created by routing it from scratch
        t_start = timer()
        sg_solution = self.solver.solve(
            sg_node_idx,
            sg_node_features,
            **kwargs
        )

        multi_update = False
        if self.verbose > 1:
            print(f"sub-solver runtime: {timer()-t_start: .5f}s")
        if len(sg_solution) == 1:
            sg_route_idx = sg_route_idx[0]
            sg_solution = sg_solution[0]
            sg_node_idx = sg_node_idx[0]
            if sg_solution is None:     # could not solve SG
                self.reject([sg_node_idx])
                return cur_sol

            routes = sg_solution.routes
            # map sg indices back to original global indices
            routes = [[sg_node_idx[i] for i in r] for r in routes]
            new_route_cost = compute_cost(routes, self.dist_mat)
            new_cost = new_route_cost.sum()
            old_cost = cur_sol.route_cost[sg_route_idx].sum()
            sg_node_idx = [sg_node_idx]
        else:
            # select sg_solution from multiple available
            if self.sg_selection_method == "multi":
                rm_idx = []
                routes, new_route_costs, new_costs, old_costs = [], [], [], []
                for i, route_idx, node_idx, sol in \
                        zip(range(len(sg_route_idx)), sg_route_idx, sg_node_idx, sg_solution):
                    if sol is None:
                        rm_idx.append(i)
                        self.reject([node_idx])
                        continue
                    rts = [[node_idx[i] for i in r] for r in sol.routes]
                    routes.append(rts)
                    nrc = compute_cost(rts, self.dist_mat)
                    new_route_costs.append(nrc)
                    new_costs.append(nrc.sum())
                    old_costs.append(cur_sol.route_cost[route_idx].sum())

                if len(routes) == 0:    # no SG could be solved
                    return cur_sol
                if len(rm_idx) > 0:
                    sg_route_idx = [i for j, i in enumerate(sg_route_idx) if j not in rm_idx]
                    sg_node_idx = [i for j, i in enumerate(sg_node_idx) if j not in rm_idx]

                new_costs, old_costs = np.array(new_costs), np.array(old_costs)
                improvement = old_costs-new_costs
                # greedily select solution with maximum improvement
                max_idx = np.argmax(improvement)
                routes = routes[max_idx]
                sg_route_idx = sg_route_idx[max_idx]
                sg_node_idx = [sg_node_idx[max_idx]]
                new_route_cost = new_route_costs[max_idx]
                old_cost = old_costs[max_idx]
                new_cost = new_costs[max_idx]
            elif self.sg_selection_method == "disjoint":
                routes, _route_idx, _node_idx, new_route_costs, new_costs, old_costs = [], [], [], [], [], []
                ii = 0
                for route_idx, node_idx, sol in zip(sg_route_idx, sg_node_idx, sg_solution):
                    if sol is None:
                        self.reject([node_idx])
                        continue
                    rts = [[node_idx[i] for i in r] for r in sol.routes]
                    nrc = compute_cost(rts, self.dist_mat)
                    new_cost = nrc.sum()
                    old_cost = cur_sol.route_cost[route_idx].sum()
                    ii += 1
                    if old_cost-new_cost > 0 or (ii == len(sg_solution) and len(routes) == 0):
                        routes.append(rts)
                        _route_idx.append(route_idx)
                        _node_idx.append(node_idx)
                        new_route_costs.append(nrc)
                        new_costs.append(new_cost)
                        old_costs.append(old_cost)

                if len(routes) == 0:    # no SG could be solved
                    return cur_sol
                elif len(routes) == 1:
                    max_idx = 0
                    routes = routes[max_idx]
                    sg_route_idx = _route_idx[max_idx]
                    sg_node_idx = [_node_idx[max_idx]]
                    new_route_cost = new_route_costs[max_idx]
                    old_cost = old_costs[max_idx]
                    new_cost = new_costs[max_idx]
                else:
                    sg_route_idx = _route_idx
                    sg_node_idx = _node_idx
                    old_cost = sum(old_costs)
                    new_cost = sum(new_costs)
                    multi_update = True
            else:
                raise ValueError

        accept, restart = self.check_accept(
            cur_cost=old_cost, new_cost=new_cost
        )
        if restart:
            return self.restart()
        elif accept:
            new_sol = deepcopy(cur_sol)
            if multi_update:
                new_sol.multi_update(
                    route_idx=sg_route_idx,
                    route_cost=new_route_costs,
                    routes=routes
                )
            else:
                new_sol.update(
                    route_idx=sg_route_idx,
                    route_cost=new_route_cost,
                    routes=routes
                )
            return new_sol
        else:
            self.reject(sg_node_idx)
            return cur_sol

    def solve_all_sg(
            self,
            sg_route_idx: Union[List[np.ndarray], np.ndarray],
            sg_node_idx: Union[List[tuple], tuple],
            sg_node_features: Union[List[np.ndarray], np.ndarray],
            cur_sol: NRRSolution,
            sg_route_features: Optional[np.ndarray] = None,
            **kwargs
    ) -> NRRSolution:
        """Ruin and recreate all SGs to create training data."""
        sg_solution = self.solver.solve(
            sg_node_idx,
            sg_node_features,
            **kwargs
        )
        # select sg_solution from multiple available
        assert self.sg_selection_method == "multi"
        rm_idx = []
        routes, new_route_costs, new_costs, old_costs, runtimes = [], [], [], [], []
        for i, route_idx, node_idx, sol in \
                zip(range(len(sg_route_idx)), sg_route_idx, sg_node_idx, sg_solution):
            if sol is None:
                rm_idx.append(i)
                continue
            rts = [[node_idx[ii] for ii in r] for r in sol.routes]
            routes.append(rts)
            nrc = compute_cost(rts, self.dist_mat)
            new_route_costs.append(nrc)
            new_costs.append(nrc.sum())
            old_costs.append(cur_sol.route_cost[route_idx].sum())
            runtimes.append(sol.runtime)

        if len(routes) == 0:    # no SG could be solved
            return cur_sol
        if len(rm_idx) > 0:     # remove corresponding indices
            sg_route_idx = [i for j, i in enumerate(sg_route_idx) if j not in rm_idx]
            sg_node_idx = [i for j, i in enumerate(sg_node_idx) if j not in rm_idx]
            if sg_route_features is not None:
                sg_route_features = np.delete(sg_route_features, rm_idx, axis=0)
            else:
                sg_route_features = [None]*len(sg_route_idx)

        # save results to data buffer
        slv_cfg = self._get_solver_cfg()
        self._data_buffer += [
            ScoringData(
                instance=deepcopy(self.instance),
                sg_old_routes=deepcopy(rts),
                sg_old_cost=oldc,
                sg_new_cost=newc,
                sg_solver_time=rtm,
                meta_iter=self._current_iter,
                sg_solver_cfg=slv_cfg,
                sg_features=rtf
            )
            for rts, oldc, newc, rtm, rtf in
            zip(routes, old_costs, new_costs, runtimes, sg_route_features)
        ]

        new_costs, old_costs = np.array(new_costs), np.array(old_costs)
        improvement = old_costs-new_costs
        # greedily select solution with maximum improvement
        max_idx = np.argmax(improvement)
        routes = routes[max_idx]
        sg_route_idx = sg_route_idx[max_idx]
        sg_node_idx = [sg_node_idx[max_idx]]
        new_route_cost = new_route_costs[max_idx]
        old_cost = old_costs[max_idx]
        new_cost = new_costs[max_idx]

        accept, restart = self.check_accept(
            cur_cost=old_cost, new_cost=new_cost
        )
        if restart:
            return self.restart()
        elif accept:
            new_sol = deepcopy(cur_sol)
            new_sol.update(
                route_idx=sg_route_idx,
                route_cost=new_route_cost,
                routes=routes
            )
            return new_sol
        else:
            self.reject(sg_node_idx)
            return cur_sol

    def _get_solver_cfg(self) -> Dict:
        if self.solver.method.lower() == "lkh":
            cfg = {'max_trials': self.solver.max_trials}
        elif self.solver.method.lower() == "sgbs":
            mode = self.solver.mode.lower()
            if mode == 'sgbs':
                cfg = {'mode': mode, 'beta': self.solver.beta, 'gamma': self.solver.gamma}
            elif mode == 'sampling':
                cfg = {'mode': mode, 'n_samples': self.solver.n_samples}
            else:
                cfg = {'mode': mode}
        else:
            raise ValueError
        return cfg

    def check_accept(
            self,
            cur_cost: float,
            new_cost: float,
    ) -> Tuple[bool, bool]:
        restart = False
        if self.accept_method == "all":
            # always accept
            return True, restart
        elif self.accept_method == "greedy":
            # accept if better
            return new_cost < cur_cost, restart
        elif self.accept_method == "sa":
            # simulated annealing acceptance
            return self.sa.check_accept(
                step=self._current_iter,
                prev_cost=cur_cost,
                new_cost=new_cost,
            )
        else:
            raise ValueError(f"unknown accept_method: "
                             f"{self.accept_method}.")

    def restart(self):
        coords = self.instance.coords.copy()
        demands = self.instance.demands.copy()
        vehicle_capacity = self.instance.vehicle_capacity
        solution = sweep_init(
            coordinates=coords,
            D=self.dist_mat,
            d=demands,
            C=vehicle_capacity,
            direction="both",
            seed_node=self._restart_idx
        )
        self._restart_idx += 1
        if self._restart_idx >= self.instance.graph_size:
            self._restart_idx = 1
        routes = sol2routes(solution)
        return NRRSolution(
            routes=routes,
            instance=deepcopy(self.instance),
            route_cost=compute_cost(routes, self.dist_mat)
        )

    def reject(self, node_idx: List[tuple]):
        if len(self._sg_tabu_list) > 0:
            # remove entries from tabu list after 10 iterations
            rm_keys = [k for k, v in self._sg_tabu_list.items()
                       if self._current_iter-v >= self.tabu_iters]
            for k in rm_keys:
                self._sg_tabu_list.pop(k)
        # add currently rejected SG to tabu list,
        # so that it is not directly selected again
        for idx in node_idx:
            self._sg_tabu_list[hash(idx)] = self._current_iter

    def get_trajectory(self):
        if (
            self.save_trajectory and
            self._trj_buffer is not None and
            len(self._trj_buffer) > 0
        ):
            trj = np.array(self._trj_buffer)
            return {
                "iter": trj[:, 0],
                "time": trj[:, 1],
                "cost": trj[:, 2],
            }
        return None

    def solve(
        self,
        instance: RPInstance,
        time_limit: Optional[Union[int, float]] = None,
        max_iters: int = 1000,
        **kwargs
    ) -> Tuple[NRRSolution, Union[int, float]]:
        """Convenience function wrapping inference functionality"""
        self._reset()
        self.load_instances(instance)
        if time_limit is not None:
            self.time_limit = time_limit-EPS

        cur_sol = self.solve_initial()
        best_sol = deepcopy(cur_sol)
        if self.save_trajectory:
            self._trj_buffer.append([self._current_iter,
                                     timer()-self._t_start,
                                     best_sol.total_cost])
        try:
            while (
                self._current_iter < max_iters and
                timer()-self._t_start < self.time_limit
            ):
                if self.verbose:
                    print(f"iter: {str(self._current_iter).rjust(6)} - "
                          f"cost: {best_sol.total_cost: .6f}")
                sg_rt_idx, sg_nd_idx, sg_n_feat, sg_r_feat = self.construct_subgraphs(cur_sol)
                scores = self.score_subgraphs(
                    sg_nd_idx, sg_rt_idx, sg_r_feat, cur_sol
                )
                sg_rt_idx, sg_nd_idx, sg_n_feat, sg_r_feat = self.select_subgraph(
                    sg_rt_idx, sg_nd_idx, sg_n_feat, scores
                )
                cur_sol = self.solve_subgraph(
                    sg_rt_idx, sg_nd_idx, sg_n_feat, cur_sol
                )
                if cur_sol.total_cost < best_sol.total_cost:
                    best_sol = deepcopy(cur_sol)
                self._current_iter += 1
                if self.save_trajectory:
                    self._trj_buffer.append([self._current_iter,
                                             timer() - self._t_start,
                                             best_sol.total_cost])

        except KeyboardInterrupt:   # ^= SIGINT
            raise KeyboardInterrupt(best_sol, timer()-self._t_start)

        t_total = timer()-self._t_start
        if self.verbose:
            print(f"\nfinished after {self._current_iter} iterations ({t_total: .4f}s).")
            print(f"final cost: {best_sol.total_cost}")
        return best_sol, t_total

    def create_scoring_data(
            self,
            instances: List[RPInstance],
            save_dir: str,
            max_iters: int = 1000,
            data_str: str = "data",
            disable_progressbar: bool = False,
            **kwargs
    ):
        """Create data regarding possible improvement for different SGs."""
        # save cfg of generating process
        run_cfg = {
            'solver': self.solver.method,
            'init_method': self.init_method,
            'init_method_cfg': deepcopy(self.init_method_cfg),
            'sg_construction_method': self.sg_construction_method,
            'sg_construction_method_cfg': deepcopy(self.sg_construction_method_cfg),
            'scoring_method': self.scoring_method,
            'scoring_method_cfg': deepcopy(self.scoring_method_cfg),
            'accept_method': self.accept_method,
            'accept_method_cfg': deepcopy(self.accept_method_cfg),
        }
        assert self.sg_selection_method == "multi"
        self.sg_selection_method_cfg['n_select'] = 1e6  # select all
        # setup directory
        os.makedirs(save_dir, exist_ok=True)
        # format file path
        cfg_str = f"{run_cfg['solver']}_" \
                  f"{run_cfg['init_method']}_" \
                  f"{run_cfg['sg_construction_method']}_" \
                  f"{run_cfg['scoring_method']}_" \
                  f"{run_cfg['accept_method']}"
        slv_cfg = self._get_solver_cfg()
        slv_cfg_str = '_'.join([f"{k}:{v}" for k, v in slv_cfg.items()])
        fname = f"nrr_{data_str}_{len(instances)}_{cfg_str}_{slv_cfg_str}.dat"
        fpth = os.path.join(save_dir, fname)
        if os.path.isfile(fpth) and os.path.exists(fpth):
            print(f'Dataset file with same name exists already: {fpth}')
            pre, ext = os.path.splitext(fpth)
            new_f = pre + '_' + datetime.utcnow().strftime('%Y%m%d%H%M%S%f') + ext
            print(f'archiving existing file to: {new_f}...')
            shutil.copy2(fpth, new_f)
            os.remove(fpth)
        print(f"saving SG solutions to: {fpth}")
        with open(fpth, 'wb+') as fw:
            # save cfg and type as first element
            cfg_dict = {'type': ScoringData.__name__, 'run_cfg': run_cfg}
            pickle.dump(cfg_dict, fw)

            for inst in tqdm(instances, disable=disable_progressbar):
                self._reset()
                self.load_instances(inst)
                self._data_buffer = []

                cur_sol = self.solve_initial()
                best_sol = deepcopy(cur_sol)
                try:
                    while self._current_iter < max_iters:
                        if self.verbose:
                            print(f"iter: {str(self._current_iter).rjust(6)} - "
                                  f"cost: {best_sol.total_cost: .6f}")
                        sg_rt_idx, sg_nd_idx, sg_n_feat, sg_r_feat = self.construct_subgraphs(cur_sol)
                        scores = self.score_subgraphs(
                            sg_nd_idx, sg_rt_idx, sg_r_feat, cur_sol
                        )
                        sg_rt_idx, sg_nd_idx, sg_n_feat, sg_r_feat = self.select_subgraph(
                            sg_rt_idx, sg_nd_idx, sg_n_feat,
                            scores=scores, sg_route_features=sg_r_feat
                        )
                        cur_sol = self.solve_all_sg(
                            sg_rt_idx, sg_nd_idx, sg_n_feat, cur_sol,
                            sg_route_features=sg_r_feat
                        )
                        if cur_sol.total_cost < best_sol.total_cost:
                            best_sol = deepcopy(cur_sol)
                        self._current_iter += 1

                        for d in self._data_buffer:
                            if isinstance(d, ScoringData) and len(d) > 0:
                                pickle.dump(d.to_dict(), fw)
                        self._data_buffer.clear()

                except Exception as e:
                    if DEBUG:
                        raise e
                    warnings.warn(f"ERROR: {e}")

                finally:
                    for d in self._data_buffer:
                        if isinstance(d, ScoringData) and len(d) > 0:
                            pickle.dump(d.to_dict(), fw)

        return fpth


#
# ============= #
# ### TEST #### #
# ============= #
def _test():
    from lib.problem import RPDataset
    from lib.nrr.lkh_wrapper import LKHSolver
    from lib.nrr.sgbs_wrapper import SGBSSolver
    from lib.nrr.best_insert_wrapper import BestInsertSolver
    DSET = "data/CVRP/benchmark/uchoa/n2.dat"
    SIZE = 4
    SINGLE = True
    dataset = RPDataset(
        problem="cvrp",
        data_pth=DSET,
    ).sample(sample_size=SIZE, allow_pickle=True)

    lkh = LKHSolver(
        lkh_exe_pth="lkh3/LKH-3.0.4/LKH",
        #lkh_exe_pth="lkh3/LKH3_c4v4/LKH",
        max_trials=64,
        max_num_workers=4
    )
    sgbs = SGBSSolver(
        ckpt_pth="sgbs/ckpts/uchoa100/checkpoint-8100.pt",
        cuda=True
    )
    bis = BestInsertSolver(max_num_workers=4)
    solvers = [
        bis,
        lkh,
        sgbs
    ]

    if SINGLE:
        nrr = NRR(
            sg_solver=solvers[0],
            init_method="savings",
            sg_construction_method="tour_knn",
            sg_construction_method_cfg={'knn': 3},
            scoring_method="rnd",
            sg_selection_method="greedy",
            sg_selection_method_cfg={'n_select': 4},
            accept_method="sa",
            accept_method_cfg={'restart_at_step': 10},
            verbose=1
        )
        for inst in dataset:
            sol = nrr.solve(
                instance=inst,
                time_limit=None,
                max_iters=50,
            )
            print(f"solution: {sol}")
    else:
        CONSTR_CFG = {
            "tour_knn": {'knn': 3},
            "sweep": {'n_target': 40, 'sweep_direction': 'both', 'alpha': 1.0}
        }
        SELECTR_CFG = {
            "greedy": None,
            "sampling": None,
            "multi": {'n_select': 4},
            "disjoint": {'n_select': 3},
        }

        for solver in solvers:
            for init in NRR.INIT_METHODS:
                for constr in NRR.CONSTR_METHODS:
                    for selectr in NRR.SELECT_METHODS:
                        for scorer in NRR.SCORING_METHODS:
                            # skip not Implemented
                            if init == "pomo" or scorer == "nsf":
                                continue
                            try:
                                nrr = NRR(
                                    sg_solver=solver,
                                    init_method=init,
                                    sg_construction_method=constr,
                                    sg_construction_method_cfg=CONSTR_CFG[constr],
                                    scoring_method=scorer,
                                    sg_selection_method=selectr,
                                    sg_selection_method_cfg=SELECTR_CFG[selectr],
                                    accept_method="sa",
                                    accept_method_cfg={'restart_at_step': 5},
                                    verbose=0
                                )
                                for inst in dataset:
                                    sol = nrr.solve(
                                        instance=inst,
                                        time_limit=None,
                                        max_iters=20,
                                    )
                                    #print(f"solution: {sol.total_cost}")
                            except Exception as e:
                                print("\n=========== ERROR ================")
                                print(f"for cfg: {solver.method} - {init} - {constr} - {selectr} - {scorer}")
                                print(f"Error: {e}")
                                print("=========== ERROR ================\n")


def _test_create():
    from lib.problem import RPDataset
    from lib.nrr.lkh_wrapper import LKHSolver
    from lib.nrr.sgbs_wrapper import SGBSSolver
    #DSET = "data/CVRP/benchmark/uchoa/n2.dat"
    DSET = "data/CVRP/cvrp500/data_train_seed111_size1000_uniform_random_int.dat"
    #DSET = "data/CVRP/cvrp500/data_train_seed111_size100_mixed_random_k_variant.dat"
    SIZE = 2
    SAVE_DIR = "data/_TEST/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = RPDataset(
        problem="cvrp",
        data_pth=DSET,
    ).sample(sample_size=SIZE, allow_pickle=True)

    lkh = LKHSolver(
        lkh_exe_pth="lkh3/LKH-3.0.4/LKH",
        #lkh_exe_pth="lkh3/LKH3_c4v4/LKH",
        max_trials=100,
        max_num_workers=4
    )
    sgbs = SGBSSolver(
        ckpt_pth="sgbs/ckpts/uchoa100/checkpoint-8100.pt",
        cuda=False,
        beta=4,
        gamma=4,
    )
    solvers = [
        lkh,
        sgbs
    ]

    nrr = NRR(
        sg_solver=solvers[0],
        init_method="sweep",
        # sg_construction_method="tour_knn",
        # sg_construction_method_cfg={'knn': 3},
        sg_construction_method="sweep",
        sg_construction_method_cfg={'n_target': 100, 'sweep_direction': 'both', 'alpha': 1.0},
        scoring_method="rnd",
        sg_selection_method="multi",
        sg_selection_method_cfg={'n_select': 4},
        accept_method="all",
        #accept_method_cfg={'restart_at_step': 10},
        verbose=1
    )

    pth = nrr.create_scoring_data(
        instances=dataset,
        max_iters=3,
        save_dir=SAVE_DIR,
    )
    print(f"pth: {pth}")
