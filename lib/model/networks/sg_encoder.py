#
from typing import Tuple, List, Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torch import Tensor

from lib.model.utils import NN, COMB_TYPES


class SGEncoder(nn.Module):
    """Encoder model for sub-graph embeddings."""

    def __init__(self,
                 meta_feature_dim: int,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 # here we use sum and max pooling
                 # since they remain unchanged for 0 padding
                 pooling_type: Union[str, List[str], Tuple[str, str]] = ("sum", "max"),
                 pooling_args: Optional[Dict[str, Any]] = None,
                 pre_proj: bool = False,
                 post_proj: bool = False,
                 num_layers: int = 2,
                 activation: str = "gelu",
                 norm_type: Optional[str] = "ln",
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
            num_layers: number of hidden layers

        """
        super(SGEncoder, self).__init__()
        self.meta_feature_dim = meta_feature_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.pooling_type = [pooling_type] if isinstance(pooling_type, str) else list(pooling_type)
        for pt in self.pooling_type:
            assert pt in COMB_TYPES
        self.pooling_args = pooling_args if pooling_args is not None else {}

        self.pre_proj = pre_proj
        self.post_proj = post_proj
        self.num_layers = num_layers
        self.activation = activation
        self.norm_type = norm_type

        self.pool_opt = None

        self.n_pre_net = None
        self.g_pre_net = None
        self.post_net = None
        self.sg_net = None
        self.meta_feat_net = None
        self.ctxt_net = None

        self.create_layers(**kwargs)

    def reset_parameters(self):
        if self.pre_proj:
            self.n_pre_net.reset_parameters()
            self.g_pre_net.reset_parameters()
        if self.post_proj:
            self.post_net.reset_parameters()
        if self.num_layers > 1:
            self._reset_module_list(self.sg_net)
            self._reset_module_list(self.meta_feat_net)
            self._reset_module_list(self.ctxt_net)
        else:
            self.sg_net.reset_parameters()
            self.meta_feat_net.reset_parameters()
            self.ctxt_net.reset_parameters()

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        if self.pre_proj:
            self.n_pre_net = nn.Linear(self.input_dim, self.hidden_dim)
            self.g_pre_net = nn.Linear(self.input_dim, self.hidden_dim)
        else:
            assert self.input_dim == self.hidden_dim

        if self.post_proj:
            self.post_net = nn.Linear(self.hidden_dim, self.output_dim)
        else:
            assert self.output_dim == self.hidden_dim

        # global pooling operator to pool over graph
        npool = len(self.pooling_type)
        self.pool_opt = [getattr(torch, pt) for pt in self.pooling_type]

        self.sg_net = NN(
            in_=(self.hidden_dim * npool),
            out_=self.hidden_dim,
            h_=self.hidden_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            norm_type=self.norm_type,
        )
        self.meta_feat_net = NN(
            in_=self.meta_feature_dim,
            out_=self.hidden_dim,
            h_=self.hidden_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            norm_type=self.norm_type,
        )
        self.ctxt_net = NN(
            in_=2 * self.hidden_dim,
            out_=self.output_dim,
            h_=self.hidden_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            norm_type=None,
        )

    def pool(self, x: torch.Tensor, dim=-1) -> List[torch.Tensor]:
        tmp = []
        for pool in self.pool_opt:
            out = pool(x, dim=dim)
            tmp.append(out if isinstance(out, Tensor) else out[0])
        return torch.cat(tmp, dim=-1) if len(tmp) > 1 else tmp

    def _get_sg_emb(self,
                    sg_node_idx: Tensor,
                    node_emb: Tensor,
                    meta_features: Tensor,
                    ):
        """

        Args:
            sg_node_idx: (BS, sg_n_max)
            node_emb: (BS, N, D)
            meta_features: (BS, meta_feat_dim)

        Returns:

        """
        bs, n, d = node_emb.size()
        sg_n_max = sg_node_idx.size(-1)
        # dummy value of 0 which does not change the value
        # for sum and max pooling. We need this since node_idx
        # is zero padded to same dimension to create batch tensor!
        node_emb[:, 0] = 0
        x = self.pool(
            node_emb.gather(
                index=sg_node_idx[:, :, None].expand(-1, -1, d),
                dim=1
            ).view(bs, sg_n_max, d),
            dim=1   # node dim
        )
        return self.sg_net(x) + self.meta_feat_net(meta_features)

    def forward(self,
                node_emb: torch.Tensor,
                graph_emb: torch.Tensor,
                sg_node_idx: torch.Tensor,
                sg_meta_features: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pre_proj:
            node_emb = self.n_pre_net(node_emb)
            graph_emb = self.g_pre_net(graph_emb)

        sg_emb = self._get_sg_emb(sg_node_idx, node_emb, sg_meta_features)
        ctxt_emb = self.ctxt_net(torch.cat((sg_emb, graph_emb), dim=-1))

        if self.post_proj:
            sg_emb = self.post_net(sg_emb)

        return sg_emb, ctxt_emb
