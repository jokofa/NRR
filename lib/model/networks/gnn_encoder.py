#
from typing import Tuple, Union, List
from typing import Optional
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from lib.model.networks.graph_conv import GraphConvBlock
from lib.model.networks.spatial_graph_conv import SpatialGraphConv
from lib.model.utils import NN, COMB_TYPES


class NodeEncoder(nn.Module):
    """Graph neural network encoder model for node embeddings."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 edge_feature_dim: int = 1,
                 num_layers: int = 3,
                 conv_type: str = "SpatialGraphConv",
                 pooling_type: Union[str, List[str], Tuple[str, str]] = ("mean", "max", "sum", "std"),
                 activation: str = "gelu",
                 skip: bool = False,
                 norm_type: Optional[str] = None,
                 dropout: float = 0.0,
                 add_linear: bool = False,
                 vertical_aggregation: bool = False,
                 debug: int = 0,
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
            edge_feature_dim: dimension of edge features
            num_layers: number of conv layers
            conv_type: type of graph convolution
            pooling_type:
            activation: activation function
            skip: flag to use skip (residual) connections
            norm_type: type of norm to use
            dropout:
            add_linear: flag to add linear layer after conv
            vertical_aggregation: flag to save all intermediate embeddings after each layer
                                    to be able to pool and aggregate latent features of
                                    different effective receptive fields
            debug: debug level
        """
        super(NodeEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if edge_feature_dim is not None and edge_feature_dim != 1:
            raise ValueError("encoders currently only work for edge_feature_dim=1")
        self.edge_feature_dim = edge_feature_dim

        self.conv_type = conv_type
        self.activation = activation
        self.skip = skip
        self.norm_type = norm_type
        self.dropout = dropout
        self.add_linear = add_linear

        self.pooling_type = [pooling_type] if isinstance(pooling_type, str) else list(pooling_type)
        for pt in self.pooling_type:
            assert pt in COMB_TYPES
        self.pool_opt = None
        self.vertical_aggregation = vertical_aggregation
        self.debug = debug > 0

        self.input_proj = None
        self.output_proj = None
        self.graph_proj = None
        self.layers = None

        self.create_layers(**kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.output_proj.reset_parameters()
        self._reset_module_list(self.layers)

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # input projection layer
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        conv = (
            SpatialGraphConv if self.conv_type.lower() == "spatialgraphconv"
            else getattr(gnn, self.conv_type)
        )

        def GNN():
            # creates a GNN module with specified parameters
            # all modules are initialized globally with the call to
            # reset_parameters()
            return GraphConvBlock(
                    conv,   # conv
                    self.hidden_dim,
                    self.hidden_dim,
                    activation=self.activation,
                    skip=self.skip,
                    norm_type=self.norm_type,
                    dropout=self.dropout,
                    add_linear=self.add_linear,
                    **kwargs
            )
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(GNN())

        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

        # global pooling operator to pool over graph
        npool = len(self.pooling_type)
        self.pool_opt = [getattr(torch, pt) for pt in self.pooling_type]
        self.graph_proj = NN(
            in_=(npool * self.hidden_dim),
            out_=self.output_dim,
            h_=self.hidden_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            norm_type=self.norm_type,
            dropout=self.dropout,
        )

    def pool(self, x: torch.Tensor, dim=-1) -> List[torch.Tensor]:
        tmp = []
        for pool in self.pool_opt:
            out = pool(x, dim=dim)
            tmp.append(out if isinstance(out, torch.Tensor) else out[0])
        return torch.cat(tmp, dim=-1) if len(tmp) > 1 else tmp[0]

    def forward(self,
                x: torch.Tensor,
                e: torch.Tensor,
                w: torch.Tensor,
                bs: int,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_buffer = []
        x = x.view(-1, x.size(-1))

        # input layer
        x = self.input_proj(x)
        # encode node embeddings
        for layer in self.layers:
            x, w = layer(x, e, w)
            if self.vertical_aggregation:
                emb_buffer.append(self.pool(x.view(bs, -1, self.hidden_dim), dim=1))

        x_ = x
        # output layer
        x = self.output_proj(x)

        # check for NANs
        if self.training and (x != x).any():
            raise RuntimeError(f"Output includes NANs! "
                               f"(e.g. GCNConv can produce NANs when <normalize=True>!)")

        x = x.view(bs, -1, self.output_dim)
        # will not work for batches with differently sized graphs (!)
        if self.vertical_aggregation:
            g = torch.stack(emb_buffer, dim=1).sum(dim=1)
        else:
            g = self.pool(x_.view(bs, -1, self.hidden_dim), dim=1)
        g = self.graph_proj(g)

        return x, g


# ============= #
# ### TEST #### #
# ============= #
def _test(
        bs: int = 5,
        n: int = 30,
        cuda=False,
        seed=1
):
    import sys
    from lib.model.graph_utils import GraphNeighborhoodSampler
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # testing args
    num_layers = [1, 3]
    conv_types = ["GCNConv", "GraphConv", "ResGatedGraphConv",
                  "GATConv", "GATv2Conv", "ClusterGCNConv"]
    norm_types = [None, "ln", "bn"]
    skips = [True, False]
    add_lins = [True, False]
    dropouts = [0.0, 0.5]
    vagg = [True, False]

    # create data
    I = 4
    O = 32
    x = torch.randn(bs, n, I).to(device)

    # sample edges and weights
    sampler = GraphNeighborhoodSampler(graph_size=n, k_frac=0.5)
    coords = x[:, :, -2:]
    edge_idx, edge_weights = [], []
    for c in coords:
        ei, ew, k = sampler(c)
        edge_idx.append(ei.view(2, -1))
        edge_weights.append(ew.view(-1))
    edge_idx = torch.stack(edge_idx, dim=0) #.permute(1, 0, 2).reshape(2, -1)
    # transform to running idx
    idx_inc = (torch.cumsum(torch.tensor([n]*bs), dim=0) - n) #.repeat_interleave(k*n)
    edge_idx += idx_inc[:, None, None]
    edge_idx = edge_idx.permute(1, 0, 2).reshape(2, -1)
    edge_weights = torch.stack(edge_weights).view(-1)

    x, e, w = x.view(-1, I), edge_idx, edge_weights

    for l in num_layers:
        for c_type in conv_types:
            for norm in norm_types:
                for skip in skips:
                    for add_lin in add_lins:
                        for drp in dropouts:
                            for va in vagg:
                                try:
                                    enc = NodeEncoder(
                                        I, O,
                                        num_layers=l,
                                        conv_type=c_type,
                                        norm_type=norm,
                                        dropout=drp,
                                        skip=skip,
                                        add_linear=add_lin,
                                        vertical_aggregation=va,
                                    ).to(device)
                                    out, g = enc(x, e, w, bs=bs)
                                    assert out.size() == torch.empty((bs, n, O)).size()
                                    assert g.size() == torch.empty((bs, O)).size()
                                except Exception as err:
                                    raise type(err)(
                                        str(err) + f" - ("
                                                 f"num_layers: {l}, "
                                                 f"conv_type: {c_type}, "
                                                 f"norm: {norm}, "
                                                 f"skip: {skip}, "
                                                 f"add_lin: {add_lin}, "
                                                 f"dropout: {drp}, "
                                                 f"vagg: {va}"
                                                 f")\n"
                                    ).with_traceback(sys.exc_info()[2])

