#
import os
import logging
from warnings import warn
from typing import Optional, Dict, Union, List
import hydra
from omegaconf import DictConfig, OmegaConf as oc

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

#from lib.utils.runner_utils import parse_cfg
from lib.model.task import Task

logger = logging.getLogger(__name__)


def update_path(cfg: DictConfig):
    """
    Correct the path to data files and checkpoints,
    since CWD is changed by hydra.
    """
    cwd = hydra.utils.get_original_cwd()
    if cfg.model_args.train_dataset is not None:
        cfg.model_args.train_dataset = os.path.normpath(os.path.join(cwd, cfg.model_args.train_dataset))
    if cfg.model_args.val_dataset is not None:
        cfg.model_args.val_dataset = os.path.normpath(os.path.join(cwd, cfg.model_args.val_dataset))

    return cfg


def parse_cfg(cfg: Union[Dict, DictConfig]) -> Dict:
    """Convert an omega nsf_config dict nsf_config to standard dict."""
    return oc.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


class Runner:
    """
    Wraps all setup, training and testing functionality
    and the respective experiment configured by hydra cfg.
    """
    def __init__(self, cfg: DictConfig, auto_setup: bool = True):

        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        if self.debug > 1:
            torch.autograd.set_detect_anomaly(True)

        # check device
        cuda = cfg.trainer_args.accelerator == "gpu" and cfg.trainer_args.devices > 0
        if torch.cuda.is_available() and not cuda:
            warn(f"Cuda GPU is available but not used! Specify <cuda=True> in nsf_config file.")

        # raise error on strange CUDA warnings which are not propagated
        if (self.cfg.run_type == "train") and cuda and not torch.cuda.is_available():
            e = ""
            try:
                torch.zeros(10, device=torch.device("cuda"))
            except Exception as e:
                pass
            raise RuntimeError(f"specified GPU training run but running on CPU! {e}")

        self.task = None
        self.trainer = None
        if auto_setup:
            self.setup()

    def setup(self, add_cb: Optional[List[Callback]] = None):
        """set up all entities."""
        self._dir_setup()
        # set all seeds
        self.seed_all(self.cfg.global_seed)
        # set up all entities
        self._build_model()
        self._build_callbacks(add_cb)
        self._build_trainer()

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        os.makedirs(self.cfg.tb_log_path, exist_ok=True)
        # val log dir
        self.cfg.val_log_path = os.path.join(self._cwd, self.cfg.val_log_path)
        os.makedirs(self.cfg.val_log_path, exist_ok=True)
        # checkpoint save dir
        self.cfg.checkpoint_path = os.path.join(self._cwd, self.cfg.checkpoint_path)
        os.makedirs(self.cfg.checkpoint_path, exist_ok=True)
        # provide to model
        self.cfg.model_args['val_log_path'] = self.cfg.val_log_path

    def _build_model(self):
        """Initialize the model task."""
        self.task = Task(cfg=self.cfg.model_args)

    def _build_callbacks(self, add_cb: Optional[List[Callback]] = None):
        """Create necessary callbacks."""
        self.logger = TensorBoardLogger(
            save_dir=self.cfg.tb_log_path,
            name='',
        )
        self.callbacks = []
        self.callbacks.append(
            pl.callbacks.LearningRateMonitor()
        )
        self.callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=self.cfg.checkpoint_path,
                **self.cfg.checkpoint_args
            )
        )
        device_monitor = self.cfg.trainer_args.pop("device_stats", False)
        if device_monitor:
            self.callbacks.append(
                pl.callbacks.DeviceStatsMonitor()
            )
        if add_cb is not None:
            for cb in add_cb:
                self.callbacks.append(cb)

    def _build_trainer(self):
        """Set up PL trainer."""
        self.trainer = pl.Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            **self.cfg.trainer_args,
        )

    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        seed_everything(seed=seed, workers=True)

    def train(self, **kwargs):
        """Train the specified model."""
        logger.info(F"starting training...")
        self.trainer.fit(self.task, **kwargs)

    def test(self, test_cfg: Optional[Union[DictConfig, Dict]] = None, **kwargs):
        """Test (evaluate) the provided trained model."""
        cfg = self.cfg.test_args
        if test_cfg is not None:
            test_cfg = parse_cfg(test_cfg)
            cfg.update(test_cfg)

        if cfg.test_checkpoint_path is not None:
            logger.info(f"loading checkpoint from:\n   {cfg.test_checkpoint_path}")
            task = Task.load_from_checkpoint(checkpoint_path=cfg.test_checkpoint_path, strict=False)
        else:
            best_ckpt_pth = self.trainer.checkpoint_callback.best_model_path
            if len(best_ckpt_pth) > 0 and os.path.exists(best_ckpt_pth):
                task = Task.load_from_checkpoint(checkpoint_path=best_ckpt_pth, strict=False)
            else:
                task = self.task

        self.seed_all(cfg.global_seed)
        task.setup_test(cfg)
        return self.trainer.test(task, **kwargs)

    def run(self, show: bool = True, **kwargs):
        """Convenience function to train model
        and do final test evaluation."""
        if self.cfg.run_type == 'test':
            return self.test(**kwargs)
        else:
            self.train(**kwargs)
            return self.trainer.validate(
                self.task, self.task.val_dataloader(), "best", verbose=show
            )
