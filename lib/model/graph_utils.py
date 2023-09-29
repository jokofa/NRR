#
from typing import Union
from torch_geometric.typing import Adj
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import knn_graph


def flip_lr(x: Tensor):
    """
    Flip the first dimension in left-right direction.
    This is used to reverse tours by swapping
    (from, to) edges to (to, from) format.
    """
    return torch.fliplr(x.unsqueeze(0)).squeeze(0)


def negative_nbh_sampling(edge_index: Adj,
                          max_k: int,
                          num_neg_samples: int,
                          loop: bool = False) -> Adj:
    """Takes a sparse neighborhood adjacency matrix and
    adds <num_neg_samples> random edges for each node."""
    _, n, k = edge_index.size()
    # possible range of indices
    idx_range = torch.arange(max_k, device=edge_index.device)
    # get indices not yet in edge_index
    mask = ~(
        edge_index[0][:, :, None].expand(n, k, max_k)
        ==
        idx_range[None, None, :].expand(n, k, max_k)
    ).any(dim=1)
    # mask same node indices (self loops)
    if not loop:
        mask &= (edge_index[1, :, 0][:, None].expand(-1, max_k) != idx_range[None, :].expand(n, max_k))
    # get candidate indices
    candidates = idx_range[None, :].expand(n, -1)[mask].view(n, -1)
    # sample idx and create edge
    i = int(not loop)  # i = 1 when not considering self loops!
    return torch.cat(
        (candidates[:, torch.randperm(max_k-k-i)[:num_neg_samples]].unsqueeze(0),
         edge_index[1, :, 0][:, None].expand(-1, num_neg_samples).unsqueeze(0)),
        dim=0
    )


class GraphNeighborhoodSampler(nn.Module):
    def __init__(self,
                 graph_size: int,
                 k_frac: Union[int, float] = 0.3,
                 rnd_edge_ratio: float = 0.0,
                 num_workers: int = 4,
                 **kwargs):
        """Samples <k_frac> nearest neighbors +
        <rnd_edge_ratio> random nodes as initial graph.

        Args:
            graph_size: size of considered graph
            k_frac: number of neighbors considered
            rnd_edge_ratio: ratio of random edges among neighbors
                            to have connections beyond local neighborhood
            num_workers: number of workers
            **kwargs:
        """
        super(GraphNeighborhoodSampler, self).__init__()
        self.graph_size = graph_size
        self.k_frac = k_frac
        self.rnd_edge_ratio = rnd_edge_ratio
        self.num_workers = num_workers
        self.k, self.max_k, self.k_nn, self.num_rnd = None, None, None, None
        self._infer_k(graph_size)

    def _infer_k(self, n: int):
        self.max_k = n
        if isinstance(self.k_frac, float):
            assert 0.0 < self.k_frac < 1.0
            self.k = int(math.floor(self.k_frac*self.max_k))
        elif isinstance(self.k_frac, int):
            self.k = int(min(self.k_frac, self.max_k))
        else:
            raise ValueError
        # infer how many neighbors are nodes sampled randomly from graph
        assert 0.0 <= self.rnd_edge_ratio <= 1.0
        self.num_rnd = int(math.floor(self.k * self.rnd_edge_ratio))
        self.k_nn = self.k - self.num_rnd

    @torch.no_grad()
    def forward(self, coords: Tensor, loop: bool = True):
        n, d = coords.size()
        if abs(n - self.graph_size) > 1:
            self._infer_k(n)
        # remove depot coords
        coords_ = coords[1:, :].view(-1, d)
        # get k nearest neighbors
        edge_idx = knn_graph(coords_,
                             k=self.k_nn,
                             loop=loop,     # include self-loops flag
                             num_workers=self.num_workers)
        # sample additional edges to random nodes if specified
        if self.num_rnd > 0:
            edge_idx = edge_idx.view(2, -1, self.k_nn)
            rnd_edge_idx = negative_nbh_sampling(edge_index=edge_idx,
                                                 max_k=self.max_k,
                                                 num_neg_samples=self.num_rnd,
                                                 loop=False)
            edge_idx = torch.cat((edge_idx, rnd_edge_idx), dim=-1).view(2, -1)

        # add depot node into nbh of each node
        from_depot_edges = torch.cat((
            torch.zeros(n, dtype=torch.long, device=coords.device)[None, :],
            torch.arange(n, device=coords.device)[None, :]
        ), dim=0)[:, 1:]
        edge_idx = torch.cat((
            edge_idx.view(2, n - 1, -1) + 1,
            from_depot_edges[:, :, None]
        ), dim=-1)
        to_depot_edges = torch.cat((
            torch.zeros(2, dtype=torch.long, device=coords.device)[:, None],
            flip_lr(from_depot_edges)
        ), dim=-1)
        # compute knn for depot
        idx_coords = coords.view(-1, d)[to_depot_edges]
        dists = torch.norm(idx_coords[0] - idx_coords[1], p=2, dim=-1)
        to_depot_edges = to_depot_edges[:, torch.topk(-dists, k=self.k_nn+1).indices]
        edge_idx = torch.cat((
            to_depot_edges.view(2, -1),
            edge_idx.view(2, -1)
        ), dim=-1)
        k = self.k + 1

        # calculate euclidean distances between neighbors as weights
        idx_coords = coords.view(-1, d)[edge_idx]
        edge_weights = torch.norm(idx_coords[0] - idx_coords[1], p=2, dim=-1)
        return edge_idx.view(2, -1, k), edge_weights.view(-1, k), k
