#
import logging
from typing import Union, Any, List, Optional, Dict
from omegaconf import DictConfig

import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
import torch.multiprocessing
from torch.utils.data import DataLoader
from pytorch_lightning.core.module import LightningModule

from lib.model.utils import get_lambda_decay, NPZProblemDataset
from lib.problem import RPDataset, ScoringData

# https://pytorch.org/docs/stable/multiprocessing.html#file-descriptor-file-descriptor
torch.multiprocessing.set_sharing_strategy('file_system')
logger = logging.getLogger('lightning')


class BaseTask(LightningModule):
    """Outer PyTorch lightning module wrapper."""
    def __init__(self, cfg: Union[DictConfig, Dict]):
        super(BaseTask, self).__init__()

        self.cfg = cfg
        self.save_hyperparameters()

        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        #
        self.collate_fn = None
        self.loss_fn = None
        self.mae = None
        self.mse = None

    def forward(self, x):
        raise RuntimeError("Cannot use PLModule forward directly!")

    def setup(self, stage: Optional[str] = None):
        """Initial setup hook."""

        if stage is not None and stage.lower() == 'test':
            raise RuntimeError()
        # TRAINING setup ('fit')
        else:
            #logger.info(f"loading validation data {self.cfg.val_dataset}")
            #logger.info(f"loading training data {self.cfg.train_dataset}")
            if ".npz" in self.cfg.train_dataset or ".npz" in self.cfg.val_dataset:
                self.val_dataset = NPZProblemDataset(
                    npz_file_pth=self.cfg.val_dataset,
                    knn=self.cfg.knn,
                    limit=self.cfg.val_dataset_size,
                )
                self.train_dataset = NPZProblemDataset(
                    npz_file_pth=self.cfg.train_dataset,
                    knn=self.cfg.knn,
                    limit=self.cfg.train_dataset_size,
                )
            else:
                # load validation data
                self.val_dataset = RPDataset(
                    data_pth=self.cfg.val_dataset,
                ).sample(sample_size=self.cfg.val_dataset_size)
                # load training data
                self.train_dataset = RPDataset(
                    data_pth=self.cfg.train_dataset,
                ).sample(sample_size=self.cfg.train_dataset_size)

            # set some attributes
            self.collate_fn = lambda x: x   # identity -> returning simple list of instances
            #self.loss_fn = nn.MSELoss()
            self.loss_fn = getattr(nn, self.cfg.loss_fn)()
            self.mae = tm.MeanAbsoluteError()
            self.mse = tm.MeanSquaredError()
            self._build_model()

    def _build_model(self):
        raise NotImplementedError()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # also need to load the original nsf_config
        cfg_ = checkpoint["hyper_parameters"]["cfg"]
        self.cfg.update(cfg_)
        self._build_model()

    def configure_optimizers(self):
        """Create optimizers and lr-schedulers for model."""
        # create optimizer
        opt = getattr(optim, self.cfg.optimizer)
        # provide model parameters and optionally trainable loss parameters to optimizer
        optimizer = opt(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            **self.cfg.optimizer_cfg
        )
        # create lr scheduler
        if self.cfg.scheduler_cfg.schedule_type is not None:
            decay_ = get_lambda_decay(**self.cfg.scheduler_cfg)
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, decay_)
            return (
               {'optimizer': optimizer, 'lr_scheduler': lr_scheduler},
            )
        return optimizer

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def training_step(self,
                      batch: List[ScoringData],
                      batch_idx: int, *args, **kwargs):
        raise NotImplementedError()

    def validation_step(self,
                        batch: List[ScoringData],
                        batch_idx: int, *args, **kwargs):
        raise NotImplementedError()
