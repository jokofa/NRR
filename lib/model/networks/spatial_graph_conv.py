#
from typing import Union
from torch_geometric.typing import Adj, PairTensor, OptTensor

import torch
from torch import Tensor
from torch_geometric.nn.conv import GraphConv


#
class SpatialGraphConv(GraphConv):
    r"""
    Extension to Pytorch Geometric GraphConv
    which is implementing the operator of
    `"Weisfeiler and Leman Go Neural:
    Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper,
    adding the difference of x_i and x_j in the propagation:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_1 \mathbf{x}_i +
        \mathbf{\Theta}_2 \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot (
        \mathbf{x}_j -  \mathbf{x}_i)

    """

    def forward(self,
                x: Union[Tensor, PairTensor],
                edge_index: Adj,
                edge_weight: OptTensor = None,
                **kwargs) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x,
                             edge_weight=edge_weight,
                             size=None)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_root(x_r)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j-x_i if edge_weight is None else edge_weight.view(-1, 1) * (x_j-x_i)


#
# ============= #
# ### TEST #### #
# ============= #
def _test(
        cuda=False,
        seed=1,
):
    import sys
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    N = 4
    D = 16
    x = torch.randn(N, D).to(device)
    edge_index = torch.tensor([[0, 1, 2, 2, 3, 3], [0, 0, 1, 1, 3, 2]]).to(device)
    edge_weight = torch.randn(edge_index.size(-1)).to(device)
    conv = SpatialGraphConv(D, D).to(device)

    try:
        x = conv(x, edge_index, edge_weight)
        assert x.size() == (N, D)
    except Exception as e:
        raise type(e)(str(e)).with_traceback(sys.exc_info()[2])

