#
from typing import Optional, Union, Tuple
import numpy as np
import scipy as sp
from matplotlib.path import Path


def knn_sampling(
        n: int,
        rng: np.random.RandomState,
        dist_mat: np.ndarray,
        idx_range: np.ndarray,
        selection_interval: Optional[int] = None
):
    if selection_interval is None:
        selection_interval = min(3*n, len(idx_range)-1)
    # sample seed location
    seed_idx = rng.choice(idx_range, size=1, replace=False)
    # then sample from neighborhood proportional to distance
    dists = dist_mat[seed_idx].copy().reshape(-1)
    max_dist = np.sort(dists)[selection_interval]
    msk = dists > max_dist
    dists[msk] = -float("inf")
    p = sp.special.softmax(dists)
    return rng.choice(idx_range, size=n, p=p, replace=False)


def sub_sample(
    xyw: np.ndarray,
    size: int,
    n: int,
    rng: np.random.RandomState,
    method: str = "rnd",
    weight_factor: Union[float, Tuple[float, float]] = 1.0,
    max_weight: float = 0.2,
    add_depot: bool = True,
):
    """

    Args:
        xyw: array of coordinates and weights
        size: number of samples
        n: size of each sample, i.e. number of nodes
        rng: seeded random generator
        method: sampling method to use ["rnd", "knn", "quadrant_rnd", "quadrant_knn"]
        weight_factor: since sub-samples can require a very small number of clusters
                        because the weights are rather small, scale the weights of the
                        subsample by this factor. when providing a tuple (a, b),
                        will sample the factor for each samples from ~uniform(a, b)
        max_weight: max weight allowed for a single node

    """
    method = method.lower()
    assert method in ["rnd", "knn", "quadrant_rnd", "quadrant_knn"]
    if add_depot:
        depot = xyw[0]
        assert depot[-1] == 0.0
        xyw = xyw[1:]
    if ((xyw[:, -1] >= 1.0) | (xyw[:, -1] <= 0.0)).any():
        raise ValueError(f"demands have to be normalized in (0, 1).")
    N = len(xyw)
    idx_range = np.arange(N)
    samples = []
    if method in ["rnd", "knn"]:
        if isinstance(weight_factor, (int, float)):
            wfacs = [weight_factor] * size
        else:
            a, b = weight_factor
            assert a < b
            wfacs = rng.uniform(a, b, size=size)

        if method == "rnd":
            # random sampling
            indices = [rng.choice(idx_range, size=n, replace=False) for _ in range(size)]
        elif method == "knn":
            # localized sub-sampling based on knn
            if 2*n >= 0.6*N:
                raise ValueError(f"n={n} is too large for localized sampling. simply sample randomly.")
            coords = xyw[:, :2]
            dist_mat = sp.spatial.distance.cdist(coords, coords, 'euclidean')
            idx_range = np.arange(N)
            indices = [knn_sampling(n, rng, dist_mat, idx_range) for _ in range(size)]

        for i, wf in zip(indices, wfacs):
            xyw_sample = xyw[i]
            w_sample = xyw_sample[:, -1]
            w_sample *= wf
            w_sample = np.minimum(w_sample, max_weight)
            xyw_sample[:, -1] = w_sample
            if add_depot:
                xyw_sample = np.insert(xyw_sample, 0, depot, axis=0)
            samples.append(xyw_sample)
    else:
        # localized sub-sampling based on square
        coords = xyw[:, :2]
        x_min = coords[:, 0].min()
        x_med = np.median(coords[:, 0])
        x_max = coords[:, 0].max()
        y_min = coords[:, 1].min()
        y_med = np.median(coords[:, 1])
        y_max = coords[:, 1].max()
        if add_depot:
            # make sure square includes depot
            x_max = max(x_max, depot[0]*1.05)
            y_max = max(y_max, depot[1]*1.05)
            x_min = max(x_min, depot[0]*1.05-x_med)
            y_min = max(y_min, depot[1]*1.05-y_med)
            low = np.array([x_min, y_min])
            high = np.array([min(x_med, depot[0]*0.95), min(y_med, depot[1]*0.95)])
        else:
            low = np.array([x_min, y_min])
            high = np.array([x_med, y_med])

        while len(samples) < size:
            if isinstance(weight_factor, (int, float)):
                wf = weight_factor
            else:
                a, b = weight_factor
                assert a < b
                wf = rng.uniform(a, b, size=1)
                #print(wf)
            # sample square seed point (lower left corner quadrant)
            seed_x, seed_y = rng.uniform(low=low, high=high)
            # create square
            seed_x_max = min(seed_x + x_med, x_max)
            seed_y_max = min(seed_y + y_med, y_max)
            square = Path([     # type: ignore
                (seed_x, seed_y), (seed_x, seed_y_max),
                (seed_x_max, seed_y_max), (seed_x_max, seed_y)
            ])
            square_idx = square.contains_points(coords)
            square_xyw = xyw[square_idx]
            square_idx_range = np.arange(len(square_xyw))
            if method == "quadrant_rnd":
                # sample randomly from square
                if len(square_idx_range) > n:
                    rnd_idx = rng.choice(square_idx_range, size=n, replace=False)
                    ###
                    xyw_sample = square_xyw[rnd_idx]
                    w_sample = xyw_sample[:, -1]
                    w_sample *= wf
                    w_sample = np.minimum(w_sample, max_weight)
                    xyw_sample[:, -1] = w_sample
                    if add_depot:
                        xyw_sample = np.insert(xyw_sample, 0, depot, axis=0)
                    samples.append(xyw_sample)
            elif method == "quadrant_knn":
                # sample from neighborhood of a seed node WITHIN square
                dist_mat = sp.spatial.distance.cdist(square_xyw[:, :-1], square_xyw[:, :-1], 'euclidean')
                rnd_idx = knn_sampling(n, rng, dist_mat, square_idx_range)
                if len(rnd_idx) == n:
                    xyw_sample = square_xyw[rnd_idx]
                    w_sample = xyw_sample[:, -1]
                    w_sample *= wf
                    w_sample = np.minimum(w_sample, max_weight)
                    xyw_sample[:, -1] = w_sample
                    if add_depot:
                        xyw_sample = np.insert(xyw_sample, 0, depot, axis=0)
                    samples.append(xyw_sample)
            elif method == "quadrant_multi_knn":
                # sample in neighborhood of multiple seed nodes WITHIN square
                # needs to check for duplicates, etc. ...
                raise NotImplementedError()
            else:
                raise ValueError(f"unknown sampling method: {method}")

    return samples


# ============= #
# ### TEST #### #
# ============= #
def _test():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from lib.problem.generator import RPGenerator

    DATA_PTH = f"data/CVRP/benchmark/arnold/Flanders2.vrp"
    COLS = ["x_coord", "y_coord", "demand"]

    data = RPGenerator.load_dataset(DATA_PTH)

    inst = data[0]
    xyw = np.concatenate((inst.coords, inst.demands[:, None]), axis=-1)
    N = 2000
    seed = 123
    rng = np.random.RandomState(seed)
    data_samples = sub_sample(xyw, size=5, n=N, rng=rng, method="quadrant_rnd", weight_factor=1.0, add_depot=True)

    for smp in data_samples:
        df_ = pd.DataFrame(data=smp, columns=COLS)
        sns.relplot(x="x_coord", y="y_coord", size="demand",
                    sizes=(10, 100), alpha=.5, #palette="muted",
                    height=6, data=df_.iloc[1:])
        plt.scatter(xyw[0, 0], xyw[0, 1], marker="s", c="red")
        plt.show()
