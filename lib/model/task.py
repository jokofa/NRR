#
import logging
from typing import Union, List, Dict
from omegaconf import DictConfig

from lib.problem import ScoringData
from lib.model.utils import collate_batch
from lib.model.base_task import BaseTask
from lib.model.scoring_model import SGScoringModel

logger = logging.getLogger('lightning')


class Task(BaseTask):
    """Outer PyTorch lightning module wrapper."""
    def __init__(self, cfg: Union[DictConfig, Dict]):
        super(Task, self).__init__(cfg)

    def _build_model(self):
        logger.info("building model...")
        self.model = SGScoringModel(
            input_dim=self.cfg.input_dim,
            sg_meta_feature_dim=self.cfg.sg_meta_feature_dim,
            embedding_dim=self.cfg.embedding_dim,
            node_encoder_args=self.cfg.node_encoder_args,
            sg_encoder_args=self.cfg.sg_encoder_args,
            decoder_args=self.cfg.decoder_args,
        )

    def training_step(self,
                      batch: List[ScoringData],
                      batch_idx: int,
                      *args, **kwargs):
        x, y = collate_batch(
            batch,
            device=self.device,
            dtype=self.dtype,
            knn=self.cfg.get("knn", 25),
        )
        y_hat, _, _ = self.model(
            node_features=x.node_features,
            sg_node_idx=x.sg_node_idx,
            sg_meta_features=x.sg_meta_features,
            edges_e=x.edges,
            edges_w=x.weights,
        )
        loss = self.loss_fn(y_hat, y)

        self.log(f"train_loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=self.cfg.train_batch_size)
        return loss

    def validation_step(self,
                        batch: List[ScoringData],
                        batch_idx: int,
                        *args, **kwargs):
        x, y = collate_batch(
            batch,
            device=self.device,
            dtype=self.dtype,
            knn=self.cfg.get("knn", 25),
        )
        y_hat, _, _ = self.model(
            node_features=x.node_features,
            sg_node_idx=x.sg_node_idx,
            sg_meta_features=x.sg_meta_features,
            edges_e=x.edges,
            edges_w=x.weights,
        )
        loss = self.loss_fn(y_hat, y)
        mae, mse = self.mae(y_hat, y), self.mse(y_hat, y)
        self.log(f"val_loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=False, logger=True,
                 batch_size=self.cfg.val_batch_size)
        self.log(f"val_mae", self.mae,
                 on_step=False, on_epoch=True,
                 prog_bar=False, logger=True,
                 batch_size=self.cfg.val_batch_size,
                 metric_attribute='mae')
        self.log(f"val_mse", self.mse,
                 on_step=False, on_epoch=True,
                 prog_bar=False, logger=True,
                 batch_size=self.cfg.val_batch_size,
                 metric_attribute='mse')
        return mse
