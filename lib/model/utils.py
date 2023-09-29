#
from typing import Dict, Optional, List, NamedTuple, Tuple, Union
from copy import deepcopy
import os
import logging
import itertools as it
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from lib.problem.formats import format_repr, ScoringData
from lib.model.graph_utils import GraphNeighborhoodSampler
from lib.nrr.utils import NODE_FEATURES

logger = logging.getLogger(__name__)
COMB_TYPES = ["cat", "proj", "sum", "mean", "max", "min", "std"]


def get_activation_fn(activation: str, module: bool = False, negative_slope: float = 0.01, **kwargs):
    if activation is None:
        return None
    if activation.upper() == "RELU":
        return F.relu if not module else nn.ReLU(**kwargs)
    elif activation.upper() == "GELU":
        return F.gelu if not module else nn.GELU()
    elif activation.upper() == "TANH":
        return F.tanh if not module else nn.Tanh()
    elif activation.upper() == "LEAKYRELU":
        return F.leaky_relu if not module else nn.LeakyReLU(negative_slope, **kwargs)
    else:
        raise ModuleNotFoundError(activation)


def get_norm(norm_type: str, hdim: int, **kwargs):
    if norm_type is None or str(norm_type).lower() == "none":
        return None
    if norm_type.lower() in ['bn', 'batch_norm']:
        return nn.BatchNorm1d(hdim, **kwargs)
    elif norm_type.lower() in ['ln', 'layer_norm']:
        return nn.LayerNorm(hdim, **kwargs)
    else:
        raise ModuleNotFoundError(norm_type)


def get_lambda_decay(schedule_type: str, decay: float, decay_step: Optional[int] = None):
    """Create learning rate scheduler (different strategies)."""

    if schedule_type in ['exponential', 'smooth']:
        assert 1.0 >= decay >= 0.9, \
            f"A decay factor >1 or <0.9 is not useful for {schedule_type} schedule!"

    if schedule_type == 'exponential':
        def decay_(eps):
            """exponential learning rate decay"""
            return (decay ** eps) ** eps
    elif schedule_type == 'linear':
        def decay_(eps):
            """linear learning rate decay"""
            return decay ** eps
    elif schedule_type == 'smooth':
        def decay_(eps):
            """smooth learning rate decay"""
            return (1 / (1 + (1 - decay) * eps)) ** eps
    elif schedule_type == 'step':
        assert decay_step is not None, "need to specify decay step for step decay."
        def decay_(eps):
            """step learning rate decay"""
            return decay ** (eps // decay_step)
    else:
        warnings.warn(f"WARNING: No valid lr schedule specified. Running without schedule.")
        def decay_(eps):
            """constant (no decay)"""
            return 1.0
    return decay_


def count_parameters(model: nn.Module, trainable: bool = True):
    """Count the number of (trainable) parameters of the provided model."""
    if trainable:
        model.train()   # set to train mode
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def NN(in_: int, out_: int, h_: int,
       num_layers: int = 1,
       activation: str = "relu", 
       norm_type: Optional[str] = None,
       dropout: float = 0.0,
       **kwargs):
    """
    Creates a FF neural net.
    Layer ordering: (Lin -> Act -> Norm -> Drop)
    """
    if num_layers == 1:
        return nn.Linear(in_, out_)
    elif num_layers == 2:
        layers = [nn.Linear(in_, h_)]
        layers.append(get_activation_fn(activation, module=True, **kwargs))
        nrm = get_norm(norm_type, hdim=h_, **kwargs)
        if nrm is not None:
            layers.append(nrm)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(h_, out_))
    else:
        layers = [nn.Linear(in_, h_)]
        for _ in range(max(num_layers - 2, 0)):
            layers.append(get_activation_fn(activation, module=True, **kwargs))
            nrm = get_norm(norm_type, hdim=h_, **kwargs)
            if nrm is not None:
                layers.append(nrm)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(h_, h_))
        layers.append(get_activation_fn(activation, module=True, **kwargs))
        nrm = get_norm(norm_type, hdim=h_, **kwargs)
        if nrm is not None:
            layers.append(nrm)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(h_, out_))
    return nn.Sequential(*layers)


def knn_graph(
        coords: torch.Tensor,
        knn: int,
        device: torch.device = "cpu",
        num_init: int = 1,
        num_workers: int = 4,
):
    bs, n, d = coords.size()
    # sample KNN edges for each node
    nbh_sampler = GraphNeighborhoodSampler(
        graph_size=n,
        k_frac=knn,
        num_workers=num_workers,
    )
    # starting node indices of each batch at first node
    bsm = bs * num_init
    ridx = cumsum0(
        torch.from_numpy(np.full(bsm, n))
            .to(dtype=torch.long, device=device)
    )
    stat_edges, stat_weights = [], []
    for i, c in enumerate(coords):
        e, w, _ = nbh_sampler(c)
        if num_init <= 1:
            stat_edges.append(e + ridx[i])  # increase node indices by running idx
            stat_weights.append(w)
        else:
            for j in range(num_init):
                stat_edges.append(e + ridx[(i*num_init)+j])
                stat_weights.append(w)

    edges = torch.stack(stat_edges, dim=1).view(2, -1)
    weights = torch.stack(stat_weights).view(-1)
    return edges, weights


@torch.jit.script
def cumsum0(t: torch.Tensor) -> torch.Tensor:
    """calculate cumsum of t starting at 0."""
    return torch.cat((
        torch.zeros(1, dtype=t.dtype, device=t.device),
        torch.cumsum(t, dim=-1)[:-1]
    ), dim=0)


@torch.jit.script
def torch_cart2pol(x: torch.Tensor):
    """converts x,y to polar coords."""
    assert len(x.shape) == 3
    x, y = x[:, :, 0], x[:, :, 1]
    return torch.stack((
        torch.sqrt(x**2 + y**2),    # rho
        torch.arctan2(y, x)     # phi
    ), dim=-1)


class InputTuple(NamedTuple):
    """Typed constrained clustering problem instance wrapper."""
    node_features: torch.Tensor
    sg_node_idx: torch.Tensor
    sg_meta_features: torch.Tensor
    edges: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    node_emb: Optional[torch.Tensor] = None
    graph_emb: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)


def collate_batch(
        batch: List[ScoringData],
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        knn: int = 25,
) -> Tuple[InputTuple, torch.Tensor]:
    """

    Args:
        batch: list of data instances from dataloader
        device: computing device
        dtype: fp precision data type
        knn: number of nearest neighbors for neighborhood graph

    Returns:
        x: meta-instance with collated attributes
        y: corresponding regression targets
    """
    assert isinstance(batch, List)
    bs = len(batch)
    b = batch[0]
    if isinstance(b, (Tuple, List)) and len(b) == 3:
        b, e, w = b
        i = b.instance
        # starting node indices of each batch at first node
        ridx = cumsum0(
            torch.from_numpy(np.full(bs, i.graph_size))
                .to(dtype=torch.long, device=device)
        )
    else:
        assert isinstance(b, ScoringData)
        i = b.instance
        e, w, ridx = None, None, None

    gs = i.graph_size
    cv = i.vehicle_capacity
    coords = []
    demands = []
    ks = []
    rt_features = []
    flat_rts = []
    targets = []
    edges = []
    weights = []
    for i, b in enumerate(batch):
        if isinstance(b, (Tuple, List)) and len(b) == 3:
            b, e, w = b
            inst = b.instance
        else:
            inst = b.instance
            e, w = None, None
        assert inst.graph_size == gs
        assert inst.vehicle_capacity == cv
        assert len(inst.coords) == len(inst.demands) == gs
        coords.append(inst.coords)
        demands.append(inst.demands)
        ks.append(inst.max_num_vehicles)
        if e is not None and w is not None:
            edges.append(e + ridx[i])   # increase node indices by running idx
            weights.append(w)
        rtf = np.insert(b.sg_features, 0, b.sg_old_cost)
        #rt_features.append(np.insert(b.sg_features, 0, b.sg_old_cost))
        rt_features.append(np.insert(rtf, 0, b.meta_iter))
        rt = np.array(list(it.chain.from_iterable(b.sg_old_routes)))
        flat_rts.append(np.insert(rt[rt > 0], 0, 0))
        targets.append(b.sg_old_cost - b.sg_new_cost)

    coords = torch.from_numpy(np.stack(coords)).to(device=device, dtype=dtype)
    demands = torch.from_numpy(np.stack(demands)).to(device=device, dtype=dtype)
    rt_features = torch.from_numpy(np.stack(rt_features)).to(device=device, dtype=dtype)

    if len(edges) == len(weights) == bs:
        edges = torch.stack(edges, dim=1).view(2, -1).to(device=device)
        weights = torch.stack(weights).view(-1).to(device=device, dtype=dtype)
    else:
        edges, weights = knn_graph(coords, knn=knn, device=device)

    # create node features
    c_coords = coords - coords[:, 0].unsqueeze(1)
    pol_coords = torch_cart2pol(coords)
    c_pol_coords = torch_cart2pol(c_coords)
    # must be in correct order! (NODE_FEATURES)
    nodes = torch.cat((
        coords,
        c_coords,
        pol_coords,
        c_pol_coords,
        demands[:, :, None]
    ), dim=-1)
    assert nodes.size(-1) == len(NODE_FEATURES)

    # create (padded) node idx
    mx_rt = max([len(r) for r in flat_rts])
    flat_rts = torch.from_numpy(np.stack(
        np.pad(rt, (0, mx_rt-len(rt)), mode='constant') for rt in flat_rts
    )).to(device=device, dtype=torch.long)

    x = InputTuple(
        node_features=nodes,
        sg_node_idx=flat_rts,
        sg_meta_features=rt_features,
        edges=edges,
        weights=weights,
    )

    assert len(targets) == bs
    y = torch.from_numpy(np.stack(targets)).to(device=device, dtype=dtype)

    return x, y


class NPZProblemDataset(Dataset):
    """Routing problem dataset wrapper."""
    def __init__(self,
                 npz_file_pth: str,
                 knn: int = 25,
                 limit: Optional[int] = None,
                 ):
        """

        Args:
            npz_file_pth: path to numpy .npz dataset file
            knn: number of nearest neighbors for neighborhood graph
        """
        super(NPZProblemDataset, self).__init__()
        self.data_pth = npz_file_pth
        self.knn = knn
        self.limit = limit

        self.data = None
        self.size = None
        self.nbh_sampler = None
        self._file_path = None
        self._keys = None
        self._instances = None
        self._load()

    def _load(self):
        f_ext = os.path.splitext(self.data_pth)[1]
        self._file_path = os.path.normpath(os.path.expanduser(self.data_pth))
        assert f_ext == ".npz"
        logger.info(f"Loading dataset from:  {self._file_path}")
        with np.load(self._file_path, allow_pickle=True) as data:
            #data = np.load(self._file_path, allow_pickle=True)
            keys = deepcopy(data.files)
            size = data.get('size', None)
            if isinstance(size, np.ndarray):
                if size.size > 0:
                    if len(size.shape) == 1:
                        size = int(size[0])
                    else:
                        size = int(size)
                else:
                    raise ValueError(f"size: {size}")
            self.size = size if self.limit is None else min(size, self.limit)
            keys.remove('cfg')
            keys.remove('size')
            keys.remove('all_instances')
            self._keys = deepcopy(keys)
            self.data = {k: data[k] for k in self._keys}
            self._instances = data['all_instances'][0]

        gs = next(iter(self._instances.values())).get('graph_size', 501)
        self.nbh_sampler = GraphNeighborhoodSampler(
            graph_size=gs,
            k_frac=self.knn,
            num_workers=1
        )

    def _prepare(self, x: Dict):
        """prepare one instance"""
        # get instance by hash
        inst = deepcopy(self._instances[x['instance']])
        e, w, _ = self.nbh_sampler(torch.from_numpy(inst['coords']))
        x['instance'] = inst
        return (x, e, w)

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[ScoringData, torch.Tensor, torch.Tensor]:
        x, e, w = self._prepare({
                k: self.data[k][idx] for k in self._keys
            })
        return ScoringData.make(**x), e, w


def load_model(ckpt_pth: str, cuda: bool = True, **kwargs):
    """Loads and initializes the model from the specified checkpoint path."""
    from lib.model.scoring_model import SGScoringModel

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    print(f"loading model checkpoint: {ckpt_pth}")
    checkpoint = torch.load(ckpt_pth, map_location=device, **kwargs)
    cfg = checkpoint['hyper_parameters']['cfg']

    model = SGScoringModel(
        input_dim=cfg.input_dim,
        sg_meta_feature_dim=cfg.sg_meta_feature_dim,
        embedding_dim=cfg.embedding_dim,
        node_encoder_args=cfg.node_encoder_args,
        sg_encoder_args=cfg.sg_encoder_args,
        decoder_args=cfg.decoder_args,
    )
    sd = checkpoint['state_dict']
    # remove task model prefix if existing
    sd = {k[6:]: v for k, v in sd.items() if k[:6] == "model."}
    model.load_state_dict(sd)   # type: ignore
    return model
