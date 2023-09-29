#
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from lib.model.utils import NN

COMB_MODES = ["cat", "sum"]


class Decoder(nn.Module):
    """Simple FF pointwise decoder."""

    def __init__(self,
                 sg_emb_dim: int,
                 ctxt_emb_dim: int,
                 hidden_dim: int = 128,
                 comb_mode: str = "cat",
                 num_layers: int = 2,
                 activation: str = "gelu",
                 norm_type: Optional[str] = "ln",
                 dropout: float = 0.0,
                 **kwargs):
        super(Decoder, self).__init__()
        self.sg_emb_dim = sg_emb_dim
        self.ctxt_emb_dim = ctxt_emb_dim
        self.hidden_dim = hidden_dim
        self.comb_mode = comb_mode.lower()
        self.num_layers = num_layers
        self.activation = activation
        self.norm_type = norm_type.lower() if norm_type is not None else None
        self.dropout = dropout
        assert self.comb_mode in COMB_MODES
        assert sg_emb_dim == ctxt_emb_dim

        self.nn = None
        self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        dim = 2*self.ctxt_emb_dim if self.comb_mode == "cat" else self.ctxt_emb_dim
        self.nn = NN(
            in_=dim,
            out_=1,
            h_=self.hidden_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            norm_type=self.norm_type,
            dropout=self.dropout,
            **kwargs
        )

    def reset_parameters(self):
        self._reset_module_list(self.nn)

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def forward(self,
                sg_emb: Tensor,
                ctxt_emb: Tensor,
                ) -> Tensor:
        """

        Args:
            sg_emb: (BS, k_max, D1)
            ctxt_emb: (BS, D2)

        Returns:
            scores: (BS, k_max, )
        """
        if self.comb_mode == "cat":
            emb = torch.cat((sg_emb, ctxt_emb), dim=-1)
        elif self.comb_mode == "sum":
            emb = sg_emb + ctxt_emb
        else:
            raise ValueError(self.comb_mode)

        return self.nn(emb).squeeze(-1)

