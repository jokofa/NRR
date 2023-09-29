#
from warnings import warn
from typing import Union, Optional, Tuple, List, Dict, Any
import sys
import os
import io
import logging
import math
import pickle

import numpy as np
from scipy.linalg import block_diag
from torch.utils.data import Dataset
from omegaconf import DictConfig, ListConfig

from lib.problem.formats import RPInstance, ScoringData


__all__ = [
    "RPGenerator",
    "RPDataset",
]
logger = logging.getLogger(__name__)
#EPS = np.finfo(np.float32).eps


def format_ds_save_path(directory, args=None, affix=None, fname='', ext: str = 'dat'):
    """Format the path for saving datasets."""
    directory = os.path.normpath(os.path.expanduser(directory))

    if args is not None:
        for k, v in args.items():
            if isinstance(v, str):
                fname += f'_{v}'
            else:
                fname += f'_{k}_{v}'

    if affix is not None:
        fname = str(affix) + fname
    if fname != '':
        fpath = os.path.join(directory, fname)
    else:
        fpath = directory
    if fpath[-3:] not in ['.pt', 'dat', 'pkl', 'npz']:
        fpath += ext

    if os.path.isfile(fpath):
        print('Dataset file with same name exists already. Overwrite file? (y/n)')
        a = input()
        if a != 'y':
            print('Could not write to file. Terminating program...')
            sys.exit()

    return fpath


def load_pickle(file_path: str, offset: int = 0, limit: int = None):
    assert os.path.splitext(file_path)[-1] in [".dat", ".pkl"]
    data = []
    i = 1
    with open(file_path, 'rb') as f:
        try:
            # first file is cfg dict
            cfg = pickle.load(f)
            while True:
                if limit and i > limit:
                    break
                if i > offset:
                    data.append(pickle.load(f))
                else:
                    # skip
                    pickle.load(f)
                i += 1
        except EOFError:
            pass
    return cfg, data


def load_cvrplib_instance(filepath: str, specification: str = "standard"):
    """For loading and parsing benchmark instances in CVRPLIB format."""
    with io.open(filepath, 'rt', newline='') as f:
        cap = 1
        n, k = None, None
        coord_flag = True
        idx = 0
        for i, line in enumerate(f):
            data = line.strip().split()
            if i in [1, 4]:
                pass
            elif i == 0:
                assert data[0] == "NAME"
                n_str = data[-1]
                if 'k' in n_str:
                    k = int(n_str.split('k')[-1])
            elif i == 2:
                assert data[0] == "TYPE"
                assert data[-1] == "CVRP"
            elif i == 3:
                assert data[0] == "DIMENSION"
                n = int(data[-1])
                node_features = np.zeros((n, 3), dtype=np.single)
            elif i == 5:
                assert data[0] == "CAPACITY"
                cap = int(data[-1])
            else:
                if data[0] == "DEPOT_SECTION":
                    break
                elif data[0] == "NODE_COORD_SECTION":
                    coord_flag = True
                    idx = 0
                elif data[0] == "DEMAND_SECTION":
                    coord_flag = False
                    idx = 0
                else:
                    if specification.lower() == "standard":
                        if coord_flag:
                            # read coordinates
                            assert len(data) == 3
                            node_features[idx, :2] = np.array(data[1:]).astype(np.single)
                            idx += 1
                        else:
                            # read demands
                            assert len(data) == 2
                            node_features[idx, -1] = np.array(data[-1]).astype(np.single)
                            idx += 1
                    else:
                        raise NotImplementedError(specification)

    # normalize coords and demands
    assert node_features[:, :2].min() >= 0
    if node_features[:, :2].max() <= 1000:
        node_features[:, :2] = node_features[:, :2]/1000
    else:
        mx = node_features[:, :2].max()
        print(f"normalizing with max value of: {mx}")
        node_features[:, :2] = node_features[:, :2] / mx
    node_features[:, -1] = node_features[:, -1]/cap

    return RPInstance(
        coords=node_features[:, :2],
        demands=node_features[:, -1],
        graph_size=n,
        vehicle_capacity=1.0,
        max_num_vehicles=k,
    )


def parse_from_cfg(x):
    if isinstance(x, DictConfig):
        return dict(x)
    elif isinstance(x, ListConfig):
        return list(x)
    else:
        return x


class DataSampler:
    """Sampler implementing different options to generate data for RPs and CCPs."""
    def __init__(self,
                 n_components: Union[int, Tuple[int, int]] = 5,
                 n_dims: int = 2,
                 coords_sampling_dist: str = "uniform",
                 covariance_type: str = "diag",
                 mus: Optional[np.ndarray] = None,
                 sigmas: Optional[np.ndarray] = None,
                 mu_sampling_dist: str = "normal",
                 mu_sampling_params: Tuple = (0, 1),
                 sigma_sampling_dist: str = "uniform",
                 sigma_sampling_params: Tuple = (0.05, 0.1),
                 weights_sampling_dist: str = "random_k_variant",
                 weights_sampling_params: Tuple = (1, 10),
                 uniform_fraction: float = 0.5,
                 max_cap_factor: float = 1.05,
                 random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
                 try_ensure_feasibility: bool = True,
                 verbose: bool = False,
                 ):
        """

        Args:
            n_components: number of mixture components
            n_dims: dimension of sampled features, e.g. 2 for Euclidean coordinates
            coords_sampling_dist: type of distribution to sample coordinates,
                                    one of ['uniform', 'gm', 'mixed]
            covariance_type: type of covariance matrix, one of ['diag', 'full']
            mus: user provided mean values for mixture components
            sigmas: user provided covariance values for mixture components
            mu_sampling_dist: type of distribution to sample initial mus,
                                one of ['uniform', 'normal', 'ring', 'io_ring']
            mu_sampling_params: parameters for mu sampling distribution
            sigma_sampling_dist: type of distribution to sample initial sigmas, one of ['uniform', 'normal']
            sigma_sampling_params: parameters for sigma sampling distribution
            weights_sampling_dist: type of distribution to sample weights,
                                    one of ['random_int', 'random_k_variant', 'uniform', 'gamma']
            weights_sampling_params: parameters for weight sampling distribution
            uniform_fraction: fraction of coordinates to be sampled uniformly for mixed instances
                                or parameter tuple to sample this per instance from a beta distribution
            max_cap_factor: global factor of constraint tightness
            random_state: seed integer or numpy random (state) generator
            try_ensure_feasibility: flag to try to ensure the feasibility of the generated instances
            verbose: verbosity flag to print additional info and warnings
        """
        self.nc = n_components
        self.f = n_dims
        self.coords_sampling_dist = coords_sampling_dist.lower()
        self.covariance_type = covariance_type.lower()
        self.mu_sampling_dist = mu_sampling_dist.lower()
        self.mu_sampling_params = mu_sampling_params
        self.sigma_sampling_dist = sigma_sampling_dist.lower()
        self.sigma_sampling_params = sigma_sampling_params
        self.weights_sampling_dist = weights_sampling_dist.lower()
        self.weights_sampling_params = weights_sampling_params
        self.uniform_fraction = uniform_fraction
        self.max_cap_factor = max_cap_factor
        self.try_ensure_feasibility = try_ensure_feasibility
        self.verbose = verbose
        # set random generator
        if random_state is None or isinstance(random_state, int):
            self.rnd = np.random.default_rng(random_state)
        else:
            self.rnd = random_state

        self._sample_nc, self._nc_params = False, None
        if not isinstance(n_components, int):
            assert isinstance(n_components, (tuple, list))
            self._sample_nc = True
            self._nc_params = n_components
            self.nc = 1
        self._sample_unf_frac, self._unf_frac_params = False, None
        if not isinstance(uniform_fraction, float):
            assert isinstance(uniform_fraction, (tuple, list))
            self._sample_unf_frac = True
            self._unf_frac_params = uniform_fraction
            self.uniform_fraction = None

        if self.coords_sampling_dist in ["gm", "gaussian_mixture", "mixed"]:
            # sample initial mu and sigma if not provided
            if mus is not None:
                assert not self._sample_nc
                assert (
                    (mus.shape[0] == self.nc and mus.shape[1] == self.f) or
                    (mus.shape[0] == self.nc * self.f)
                )
                self.mu = mus.reshape(self.nc * self.f)
            else:
                self.mu = self._sample_mu(mu_sampling_dist.lower(), mu_sampling_params)
            if sigmas is not None:
                assert not self._sample_nc
                assert (
                    (sigmas.shape[0] == self.nc and sigmas.shape[1] == (self.f if covariance_type == "diag" else self.f**2))
                    or (sigmas.shape[0] == (self.nc * self.f if covariance_type == "diag" else self.nc * self.f**2))
                )
                self.sigma = self._create_cov(sigmas, cov_type=covariance_type)
            else:
                covariance_type = covariance_type.lower()
                if covariance_type not in ["diag", "full"]:
                    raise ValueError(f"unknown covariance type: <{covariance_type}>")
                self.sigma = self._sample_sigma(sigma_sampling_dist.lower(), sigma_sampling_params, covariance_type)
        else:
            if not self.coords_sampling_dist == "uniform":
                raise ValueError(f"unknown coords_sampling_dist: '{self.coords_sampling_dist}'")

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.rnd = np.random.default_rng(seed)
        else:
            self.rnd = np.random.default_rng(123)

    def resample_gm(self):
        """Resample initial mus and sigmas."""
        self.mu = self._sample_mu(
            self.mu_sampling_dist,
            self.mu_sampling_params
        )
        self.sigma = self._sample_sigma(
            self.sigma_sampling_dist,
            self.sigma_sampling_params,
            self.covariance_type
        )

    def sample_coords(self,
                      n: int,
                      resample_mixture_components: bool = True,
                      sample_unf_depot: bool = False,
                      return_nc: bool = False,
                      **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        """
        Args:
            n: number of samples to draw
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance
            sample_unf_depot: sample depot from uniform

        Returns:
            coords: (n, n_dims)
        """
        if self.coords_sampling_dist == "uniform":
            coords = self._sample_unf_coords(n, **kwargs)
        else:
            if self._sample_nc:
                self.nc = self.sample_rnd_int(*self._nc_params)
                self.resample_gm()
            elif resample_mixture_components:
                self.resample_gm()

            if self.coords_sampling_dist == "mixed":
                if self._sample_unf_frac:
                    # if specified, sample the fraction value from a beta distribution
                    v = self._sample_beta(1, *self._unf_frac_params)
                    self.uniform_fraction = 0.0 if v <= 0.04 else v
                    #print(self.uniform_fraction)
                n_unf = math.floor(n*self.uniform_fraction)
                n_gm = n - n_unf
                unf_coords = self._sample_unf_coords(n_unf, **kwargs)
                n_per_c = math.ceil(n_gm / self.nc)
                gm_coords = self._sample_gm_coords(n_per_c, n_gm, **kwargs)
                coords = np.vstack((unf_coords, gm_coords))
            else:
                n_per_c = math.ceil(n / self.nc)
                coords = self._sample_gm_coords(n_per_c, n, **kwargs)
            if sample_unf_depot:
                # depot stays uniform!
                coords[0] = self._sample_unf_coords(1, **kwargs)

        if return_nc:
            return coords.astype(np.float32), self.nc
        return coords.astype(np.float32)

    def sample_weights(self,
                       n: int,
                       k: Union[int, Tuple[int, int]],
                       cap: Optional[Union[float, int, Tuple[int, int]]] = None,
                       max_cap_factor: Optional[float] = None,
                       ) -> np.ndarray:
        """
        Args:
            n: number of samples to draw
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle

        Returns:
            weights: (n, )
        """
        n_wo_depot = n-1
        # sample a weight for each point
        if self.weights_sampling_dist in ["random_int", "random_k_variant"]:
            assert cap is not None, \
                f"weight sampling dist 'random_int' requires <cap> to be specified"

            if self.weights_sampling_dist == "random_int":
                # standard integer sampling adapted from Nazari et al. and Kool et al.
                weights = self.rnd.integers(1, 10, size=(n_wo_depot,))
                normalizer = cap + 1
            else:
                weights = self.rnd.integers(1, (cap-1)//2, size=(n_wo_depot, ))
                # normalize weights by total max capacity of vehicles
                # where we sample a random number of vehicles
                _div = self.sample_rnd_int(max(2, k//8), k)
                if max_cap_factor is not None:
                    normalizer = np.ceil((weights.sum(axis=-1)) * max_cap_factor) / _div
                else:
                    normalizer = np.ceil((weights.sum(axis=-1)) * 1.08) / _div
        elif self.weights_sampling_dist in ["uniform", "gamma"]:
            if max_cap_factor is None:
                max_cap_factor = self.max_cap_factor
            if self.weights_sampling_dist == "uniform":
                weights = self._sample_uniform(n_wo_depot, *self.weights_sampling_params)
            elif self.weights_sampling_dist == "gamma":
                weights = self._sample_gamma(n_wo_depot, *self.weights_sampling_params)
            else:
                raise ValueError
            weights = weights.reshape(-1)
            if self.verbose:
                if np.any(weights.max(-1) / weights.min(-1) > 10):
                    warn(f"Largest weight is more than 10-times larger than smallest weight.")
            # normalize weights w.r.t. norm capacity of 1.0 per vehicle and specified max_cap_factor
            # using ceiling adds a slight variability in the total sum of weights,
            # such that not all instances are exactly limited to the max_cap_factor
            normalizer = np.ceil((weights.sum(axis=-1)) * max_cap_factor) / k
        else:
            raise ValueError(f"unknown weight sampling distribution: {self.weights_sampling_dist}")

        weights = weights / normalizer

        if self.weights_sampling_dist != "random_int" and np.sum(weights) > k:
            if self.verbose:
                warn(f"generated instance is infeasible just by demands vs. "
                     f"total available capacity of specified number of vehicles.")
            if self.try_ensure_feasibility:
                raise RuntimeError

        weights = np.concatenate((np.array([0]), weights), axis=-1)     # add 0 weight for depot
        return weights.astype(np.float32)

    def sample_rnd_int(self, lower: int, upper: int) -> int:
        """Sample a single random integer between lower (inc) and upper (excl)."""
        return self.rnd.integers(lower, upper, 1)[0]

    def sample(self,
               n: int,
               k: Union[int, Tuple[int, int]],
               cap: Optional[Union[float, int, Tuple[int, int]]] = None,
               max_cap_factor: Optional[float] = None,
               resample_mixture_components: bool = True,
               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            n: number of samples to draw
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance

        Returns:
            coords: (n, n_dims)
            weights: (n, )
        """
        coords = self.sample_coords(n=n, resample_mixture_components=resample_mixture_components, **kwargs)
        weights = self.sample_weights(n=n, k=k, cap=cap, max_cap_factor=max_cap_factor)
        return coords, weights

    def _sample_mu(self, dist: str, params: Tuple):
        size = self.nc * self.f
        if dist == "uniform":
            return self._sample_uniform(size, params[0], params[1])
        elif dist == "normal":
            return self._sample_normal(size, params[0], params[1])
        elif dist == "ring":
            return self._sample_ring(self.nc, params).reshape(-1)
        elif dist == "io_ring":
            return self._sample_io_ring(self.nc).reshape(-1)
        else:
            raise ValueError(f"unknown sampling distribution: <{dist}>")

    def _sample_sigma(self, dist: str, params: Tuple, cov_type: str):
        if cov_type == "full":
            size = self.nc * self.f**2
        else:
            size = self.nc * self.f
        if dist == "uniform":
            x = self._sample_uniform(size, params[0], params[1])
        elif dist == "normal":
            x = np.abs(self._sample_normal(size, params[0], params[1]))
        else:
            raise ValueError(f"unknown sampling distribution: <{dist}>")
        return self._create_cov(x, cov_type=cov_type)

    def _create_cov(self, x, cov_type: str):
        if cov_type == "full":
            # create block diagonal matrix to model covariance only
            # between features of each individual component
            x = x.reshape((self.nc, self.f, self.f))
            return block_diag(*x.tolist())
        else:
            return np.diag(x.reshape(-1))

    def _sample_uniform(self,
                        size: Union[int, Tuple[int, ...]],
                        low: Union[int, np.ndarray] = 0.0,
                        high: Union[int, np.ndarray] = 1.0):
        return self.rnd.uniform(size=size, low=low, high=high)

    def _sample_normal(self,
                       size: Union[int, Tuple[int, ...]],
                       mu: Union[int, np.ndarray],
                       sigma: Union[int, np.ndarray]):
        return self.rnd.normal(size=size, loc=mu, scale=sigma)

    def _sample_gamma(self,
                      size: Union[int, Tuple[int, ...]],
                      alpha: Union[int, np.ndarray],
                      beta: Union[int, np.ndarray]):
        return self.rnd.gamma(size=size, shape=alpha, scale=beta)

    def _sample_beta(self,
                     size: Union[int, Tuple[int, ...]],
                     alpha: Union[int, np.ndarray],
                     beta: Union[int, np.ndarray]):
        return self.rnd.beta(size=size, a=alpha, b=beta)

    def _sample_unf_coords(self, n: int, **kwargs) -> np.ndarray:
        """Sample coords uniform in [0, 1]."""
        return self.rnd.uniform(size=(n, self.f))

    def _sample_gm_coords(self, n_per_c: int, n: Optional[int] = None, **kwargs) -> np.ndarray:
        """Sample coordinates from k Gaussians."""
        coords = self.rnd.multivariate_normal(
            mean=self.mu,
            cov=self.sigma,
            size=n_per_c,
        ).reshape(-1, self.f)   # (k*n, f)
        if n is not None:
            coords = coords[:n]     # if k % n != 0, some of the components have 1 more sample than others
        # normalize coords in [0, 1]
        return self._normalize_coords(coords)

    def _sample_ring(self, size: int, radius_range: Tuple = (0, 1)):
        """inspired by https://stackoverflow.com/a/41912238"""
        # eps = self.rnd.standard_normal(1)[0]
        if size == 1:
            angle = self.rnd.uniform(0, 2 * np.pi, size)
            # eps = self.rnd.uniform(0, np.pi, size)
        else:
            angle = np.linspace(0, 2 * np.pi, size)
        # angle = np.linspace(0+eps, 2*np.pi+eps, size)
        # angle = rnd.uniform(0, 2*np.pi, size)
        # angle += self.rnd.standard_normal(size)*0.05
        angle += self.rnd.uniform(0, np.pi / 3, size)
        d = np.sqrt(self.rnd.uniform(*radius_range, size))
        # d = np.sqrt(rnd.normal(np.mean(radius_range), (radius_range[1]-radius_range[0])/2, size))
        return np.concatenate((
            (d * np.cos(angle))[:, None],
            (d * np.sin(angle))[:, None]
        ), axis=-1)

    def _sample_io_ring(self, size: int):
        """sample an inner and outer ring."""
        # have approx double the number of points in outer ring than inner ring
        num_inner = size // 3
        num_outer = size - num_inner
        inner = self._sample_ring(num_inner, (0.01, 0.2))
        outer = self._sample_ring(num_outer, (0.21, 0.5))
        return np.vstack((inner, outer))

    @staticmethod
    def _normalize_coords(coords: np.ndarray):
        """Applies joint min-max normalization to x and y coordinates."""
        coords[:, 0] = coords[:, 0] - coords[:, 0].min()
        coords[:, 1] = coords[:, 1] - coords[:, 1].min()
        max_val = coords.max()  # joint max to preserve relative spatial distances
        coords[:, 0] = coords[:, 0] / max_val
        coords[:, 1] = coords[:, 1] / max_val
        return coords


class RPGenerator:
    """Wraps data generation, loading and
    saving functionalities for CO problems."""
    COPS = ['ccp', 'vrp', 'cvrp']  # problem variants currently implemented in generator

    def __init__(self,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 float_prec: np.dtype = np.float32,
                 **kwargs):
        self._seed = seed
        self._rnd = np.random.default_rng(seed)
        self.verbose = verbose
        self.float_prec = float_prec
        self.sampler = DataSampler(verbose=verbose, **kwargs)

    def generate(self,
                 problem: str,
                 sample_size: int = 1000,
                 graph_size: int = 100,
                 **kwargs):
        """Generate data with corresponding RP generator function."""
        try:
            generate = getattr(self, f"generate_{problem.lower()}_data")
        except AttributeError:
            raise ModuleNotFoundError(f"The corresponding generator for the problem <{problem}> does not exist.")
        return generate(size=sample_size, graph_size=graph_size, **kwargs)

    def seed(self, seed: Optional[int] = None):
        """Set generator seed."""
        if seed is not None:
            self._seed = seed
            self._rnd = np.random.default_rng(seed)
            self.sampler.seed(seed)

    @staticmethod
    def load_dataset(filename: Optional[str] = None,
                     offset: int = 0,
                     limit: Optional[int] = None,
                     convert: bool = True,
                     **kwargs) -> List[RPInstance]:
        """Load data from file."""
        f_ext = os.path.splitext(filename)[1]
        filepath = os.path.normpath(os.path.expanduser(filename))
        if len(f_ext) == 0 or f_ext in ['.txt', '.vrp']:
            # benchmark instance
            logger.info(f"Loading benchmark instance from:  {filepath}")
            data = load_cvrplib_instance(filepath, kwargs.get('specification', "standard"))
            return [data]
        else:
            logger.info(f"Loading dataset from:  {filepath}")
            cfg, data = load_pickle(filepath, offset=offset, limit=limit)
            if convert:
                data = RPGenerator._to_instance(cfg['type'], data)
            return data

    @staticmethod
    def _to_instance(tp: str, data: List[dict]) -> List[RPInstance]:
        """Convert file back into list of instances."""
        if "rp" in tp.lower():
            TP = RPInstance
        elif "sc" in tp.lower():
            TP = ScoringData
        else:
            raise ValueError(f"unknown type {tp}")
        return [TP.make(**d) for d in data]

    @staticmethod
    def save_dataset(
            dataset: List[Union[RPInstance, ScoringData]],
            filepath: str,
            **kwargs
    ):
        """Saves dataset to file path"""
        filepath = format_ds_save_path(filepath, **kwargs)
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        assert isinstance(dataset, List) and isinstance(dataset[0], (RPInstance, ScoringData))
        if isinstance(dataset[0], RPInstance):
            tp = RPInstance
        elif isinstance(dataset[0], ScoringData):
            tp = ScoringData
        else:
            raise ValueError

        logger.info(f"Saving dataset to:  {filepath}")
        with open(filepath, 'wb+') as fw:
            pickle.dump({'type': tp.__name__}, fw)
            for inst in dataset:
                pickle.dump(inst.to_dict(), fw)

        return str(filepath)

    def _sample_coords_with_depot(
            self,
            graph_size: int,
            n_depots: int = 1,
            central_depot_frac: float = 0.5,
            **kwargs
    ) -> np.ndarray:
        # decide if to sample a central depot or an outer depot
        # where central means somewhere in the center of the
        # coordinate system and outer closer to the boundaries
        if self._rnd.random(1) < central_depot_frac:
            # central
            d_coords = self._rnd.uniform(0.333, 0.666, size=2*n_depots).reshape(n_depots, 2)
        else:
            # outer
            d_coords = self._rnd.uniform(0.0, 0.666, size=2*n_depots)
            upper = d_coords > 0.333
            d_coords[upper] = d_coords[upper] + 0.333
            assert np.all((d_coords <= 0.333) | ((d_coords >= 0.666) & (d_coords <= 1.0)))
        coords = self.sampler.sample_coords(n=graph_size, **kwargs)
        assert np.all(0 <= coords) and np.all(coords <= 1)

        return np.concatenate((
            d_coords.reshape(n_depots, 2),
            coords
        ), axis=0)

    @staticmethod
    def _distance_matrix(coords: np.ndarray, l_norm: Union[int, float] = 2):
        """Calculate distance matrix with specified norm. Default is l2 = Euclidean distance."""
        return np.linalg.norm(coords[:, :, None] - coords[:, None, :], ord=l_norm, axis=0)[:, :, :, None]

    def state_dict(self):
        """Converts the current generator state to a PyTorch style state dict."""
        return {'seed': self._seed, 'rnd': self._rnd}

    def load_state_dict(self, state_dict):
        """Load state from state dict."""
        self._seed = state_dict['seed']
        self._rnd = state_dict['rnd']

    # Standard CVRP
    def generate_cvrp_data(self,
                           size: int,
                           graph_size: int,
                           k: Union[int, Tuple[int, int]] = None,
                           cap: Optional[Union[float, int, Tuple[int, int]]] = None,
                           max_cap_factor: Optional[float] = None,
                           n_depots: int = 1,
                           **kwargs) -> List[RPInstance]:
        """Generate data for CVRP

        Args:
            size (int): size of dataset (number of problem instances)
            graph_size (int): size of problem instance graph (number of nodes)
            k: number of vehicles
            cap: capacity per vehicle
            max_cap_factor: factor of additional capacity w.r.t. a norm capacity of 1.0 per vehicle
            n_depots: number of depots (default = 1)

        Returns:
            VRP Dataset
        """
        if k is None:
            warn(f"no 'k' for sampling provided. Using k based on graph_size.")
            k = int(np.ceil(np.sqrt(graph_size)**1.2))
        if max_cap_factor is None:
            warn(f"no 'max_cap_factor' for sampling provided. Using default 1.15.")
            max_cap_factor = 1.15

        if self.verbose:
            print(f"Sampling {size} problems with graph of size {graph_size}+{n_depots}.")
            if kwargs:
                print(f"Provided additional kwargs: {kwargs}")
        if n_depots > 1:
            raise NotImplementedError()

        n = graph_size + n_depots
        coords = np.stack([
            #self.sampler.sample_coords(n=n, **kwargs)
            self._sample_coords_with_depot(
                graph_size=graph_size,
                n_depots=n_depots,
                sample_unf_depot=False,
                **kwargs
            )
            for _ in range(size)
        ])
        demands = np.stack([
            self.sampler.sample_weights(n=n, k=k, cap=cap, max_cap_factor=max_cap_factor)
            for _ in range(size)
        ])
        # type cast
        coords = coords.astype(self.float_prec)
        demands = demands.astype(self.float_prec)
        assert coords.shape[1] == demands.shape[1] == graph_size+n_depots

        return [
            RPInstance(
                coords=coords[i],
                demands=demands[i],
                graph_size=graph_size+n_depots,
                vehicle_capacity=1.0,   # demands are normalized
                #num_components=k,
            )
            for i in range(size)
        ]


class RPDataset(Dataset):
    """Routing problem dataset wrapper."""
    def __init__(self,
                 problem: str = None,
                 data_pth: str = None,
                 seed: int = None,
                 **kwargs
                 ):
        """

        Args:
            problem: name/id of problems problem
            data_pth: optional file name to load dataset
            seed: seed for random generator
            **kwargs:  additional kwargs for the generator
        """
        super(RPDataset, self).__init__()
        assert problem is not None or data_pth is not None
        if data_pth is not None:
            logger.info(f"provided dataset '{data_pth}', so no new samples are generated.")
        self.problem = problem
        self.data_pth = data_pth
        self.gen = RPGenerator(seed=seed, **kwargs)

        self.size = None
        self.data = None

    def seed(self, seed: int):
        assert seed is not None
        self.gen.seed(seed)

    def sample(self, sample_size: Union[int, Any] = None, graph_size: int = 100, **kwargs):
        """Loads fixed dataset if filename was provided else
        samples a new dataset based on specified nsf_config."""
        if self.data_pth is not None:   # load data
            self.data = RPGenerator.load_dataset(self.data_pth, limit=sample_size, **kwargs)
        else:
            self.data = self.gen.generate(
                problem=self.problem,
                sample_size=sample_size,
                graph_size=graph_size,
                **kwargs
            )
        self.size = len(self.data)
        return self

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# ============= #
# ### TEST #### #
# ============= #
def _test(
    size: int = 10,
    n: int = 20,
    seed: int = 1,
):
    problems = ['cvrp', 'ccp']
    coord_samp = ['uniform', 'gm', 'mixed']
    weight_samp = ['random_int', 'uniform', 'gamma', 'random_k_variant']
    k = 4
    cap = 30
    max_cap_factor = 1.1

    for p in problems:
        for csmp in coord_samp:
            for wsmp in weight_samp:
                ds = RPDataset(
                    problem=p,
                    seed=seed,
                    coords_sampling_dist=csmp,
                    weights_sampling_dist=wsmp,
                    n_components=3,
                )
                try:
                    ds.sample(sample_size=size, graph_size=n, k=k, cap=cap, max_cap_factor=max_cap_factor)
                except AssertionError as ae:
                    print(f"ERROR:   {p}: csmp={csmp} / wsmp={wsmp}: \n{ae}")
                except RuntimeError as re:
                    print(f"ERROR:   {p}: csmp={csmp} / wsmp={wsmp}")
                    raise re


def _test2():
    PTH = "data/CVRP/benchmark/uchoa/n1/X-n101-k25.vrp"
    inst = RPGenerator.load_dataset(PTH)
    print(inst)


def _test_io():
    #import shutil
    import tempfile

    fname = "tst.dat"
    PROBLEM = "cvrp"
    SIZE = 10
    N = 100

    with tempfile.TemporaryDirectory() as PTH:
        ds = RPDataset(
            problem=PROBLEM,
            seed=123,
            coords_sampling_dist="gm",
            weights_sampling_dist="uniform",
            n_components=(1, 10),
            max_cap_factor=1.05,
        )
        data = ds.sample(sample_size=SIZE, graph_size=N).data

        os.makedirs(PTH, exist_ok=True)
        pth = RPGenerator.save_dataset(dataset=data, filepath=os.path.join(PTH, fname))
        print(f"saved at {pth}")

        ds2 = RPDataset(
            problem=PROBLEM,
            seed=123,
            data_pth=pth
        )
        data2 = ds2.sample(sample_size=SIZE).data
        print(f"loaded from {pth}")

        for d1, d2 in zip(data, data2):
            for e1, e2 in zip(d1, d2):
                if isinstance(e1, np.ndarray):
                    assert np.all(e1 == e2)
                else:
                    assert e1 == e2

    #shutil.rmtree(PTH)
