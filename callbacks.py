from typing import Any, Optional

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import make_grid


class SamplingCallback(Callback):
    def __init__(self, n_samples: int = 64):
        super(SamplingCallback, self).__init__()
        self.n_samples = n_samples

    def on_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule
    ) -> None:

        samples = make_grid(pl_module.sample([self.n_samples]))
        samples = torch.clip(samples, 0, 1)
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image('Samples',
                                                samples,
                                                global_step=trainer.global_step)


class ReconstructionCallback(Callback):
    def __init__(self, n_samples: int = 64):
        super(ReconstructionCallback, self).__init__()
        self.n_samples = n_samples
        self.x = None

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:

        if self.x is None:
            x, y = batch
            if x.shape[0]>self.n_samples:
                x = x[:self.n_samples]
            self.x = x

    def on_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule
    ) -> None:

        x_rec = make_grid(pl_module.reconstruct(self.x))
        x_rec = torch.clip(x_rec, 0, 1)
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image('Reconstructions',
                                                x_rec,
                                                global_step=trainer.global_step)

