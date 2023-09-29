#
import logging
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from lib.model.utils import count_parameters
from lib.model.networks import NodeEncoder, SGEncoder, Decoder

logger = logging.getLogger(__name__)


class SGScoringModel(nn.Module):
    """
    Model wrapping encoder and decoder models.

    Args:
        input_dim: input dimension of nodes
        node_encoder_args: additional arguments for encoder creation
        sg_encoder_args:
        decoder_args: additional arguments for decoder creation
        embedding_dim: general embedding dimension of model

    """

    def __init__(self,
                 input_dim: int,
                 sg_meta_feature_dim: int,
                 node_encoder_args: Optional[Dict] = None,
                 sg_encoder_args: Optional[Dict] = None,
                 decoder_args: Optional[Dict] = None,
                 embedding_dim: int = 256,
                 **kwargs):
        super(SGScoringModel, self).__init__()

        self.input_dim = input_dim
        self.sg_meta_feature_dim = sg_meta_feature_dim
        self.node_encoder_args = node_encoder_args if node_encoder_args is not None else {}
        self.sg_encoder_args = sg_encoder_args if sg_encoder_args is not None else {}
        self.decoder_args = decoder_args if decoder_args is not None else {}
        self.embedding_dim = embedding_dim

        # initialize encoder models
        self.node_encoder = NodeEncoder(
            input_dim=input_dim,
            output_dim=embedding_dim,
            edge_feature_dim=1,
            **self.node_encoder_args, **kwargs
        )

        self.sg_encoder = SGEncoder(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            meta_feature_dim=sg_meta_feature_dim,
            **self.sg_encoder_args, **kwargs
        )

        # initialize decoder model
        self.decoder = Decoder(
            sg_emb_dim=embedding_dim,
            ctxt_emb_dim=embedding_dim,
            **self.decoder_args, **kwargs
        )

        self.reset_parameters()

    def __repr__(self):
        super_repr = super().__repr__()  # get torch module str repr
        n_enc_p = count_parameters(self.node_encoder)
        sg_enc_p = count_parameters(self.sg_encoder)
        dec_p = count_parameters(self.decoder)
        add_repr = f"\n-----------------------------------" \
                   f"\nNum Parameters: " \
                   f"\n  (node_encoder): {n_enc_p} " \
                   f"\n  (center_encoder): {sg_enc_p} " \
                   f"\n  (decoder): {dec_p} " \
                   f"\n  total: {n_enc_p + sg_enc_p + dec_p}\n"
        return super_repr + add_repr

    def reset_parameters(self):
        """Reset model parameters."""
        self.node_encoder.reset_parameters()
        self.sg_encoder.reset_parameters()
        self.decoder.reset_parameters()

    @torch.no_grad()
    def encode_graph(
            self,
            node_features: Tensor,
            edges_e: Optional[Tensor] = None,
            edges_w: Optional[Tensor] = None,
    ):
        bs, n, d = node_features.size()
        # run node encoder to create node embeddings
        return self.node_encoder(
            x=node_features, e=edges_e, w=edges_w, bs=bs
        )

    @torch.no_grad()
    def score(self,
              sg_node_idx: Tensor,
              sg_meta_features: Tensor,
              node_emb: Tensor,
              graph_emb: Tensor,
              ):
        bs = sg_node_idx.size(0)
        assert sg_meta_features.size(0) == bs
        sg_emb, ctxt_emb = self.sg_encoder(
            node_emb=node_emb.expand(bs, -1, -1),
            graph_emb=graph_emb.expand(bs, -1),
            sg_node_idx=sg_node_idx,
            sg_meta_features=sg_meta_features,
        )
        return self.decoder(
            sg_emb=sg_emb,
            ctxt_emb=ctxt_emb,
        )

    def forward(self,
                node_features: Tensor,
                sg_node_idx: Tensor,
                sg_meta_features: Tensor,
                edges_e: Optional[Tensor] = None,
                edges_w: Optional[Tensor] = None,
                node_emb: Optional[Tensor] = None,
                graph_emb: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Tensor, Tensor]:

        bs, n, d = node_features.size()
        if node_emb is None or graph_emb is None:
            # run node encoder to create node embeddings
            node_emb, graph_emb = self.node_encoder(
                x=node_features, e=edges_e, w=edges_w, bs=bs
            )
        else:
            assert (
                node_emb.size(0) == graph_emb.size(0) == bs and
                node_emb.size(1) == n and
                node_emb.size(2) == graph_emb.size(1) == self.embedding_dim
            )

        # encode centers
        sg_emb, ctxt_emb = self.sg_encoder(
            node_emb=node_emb,
            graph_emb=graph_emb,
            sg_node_idx=sg_node_idx,
            sg_meta_features=sg_meta_features,
        )

        scores = self.decoder(
            sg_emb=sg_emb,
            ctxt_emb=ctxt_emb,
        )
        return scores, node_emb, graph_emb


# ============= #
# ### TEST #### #
# ============= #
def _test():
    from torch.utils.data import DataLoader
    from lib.problem import RPDataset
    from lib.model.utils import collate_batch, NODE_FEATURES

    PTH = "data/_TEST/nrr_data_2_lkh_sweep_sweep_rnd_all_max_trials100.dat"
    BS = 8
    CUDA = True
    SEED = 123
    I_DIM = len(NODE_FEATURES)
    MF_DIM = 5

    device = torch.device("cuda" if CUDA else "cpu")
    torch.manual_seed(SEED-1)
    data = RPDataset(data_pth=PTH).sample(sample_size=2*BS)
    dl = DataLoader(
        data,
        batch_size=BS,
        collate_fn=lambda x: x,  # identity -> returning simple list of instances
        shuffle=False,
        num_workers=0
    )

    model = SGScoringModel(
        input_dim=I_DIM,
        sg_meta_feature_dim=MF_DIM,
        embedding_dim=64,
        sg_encoder_args={'pre_proj': True, 'post_proj': True}
    ).to(device=device)

    for i, batch in enumerate(dl):
        x, y = collate_batch(batch, device=device)
        scores, node_emb, graph_emb = model(
            node_features=x.node_features,
            sg_node_idx=x.sg_node_idx,
            sg_meta_features=x.sg_meta_features,
            edges_e=x.edges,
            edges_w=x.weights,
        )
