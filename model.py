from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torch.distributions import Independent, Normal, Distribution


class Apply(nn.Module):
    def __init__(self, f: Callable, *args, **kwargs):
        super(Apply, self).__init__()
        self.f = f
        self.kwargs = kwargs
        self.args = args

    def forward(self, input: Any) -> Any:
        kwargs = {}
        args = []
        if not (self.args is None):
            if isinstance(self.args, list):
                if not hasattr(input, '__getitem__') and len(self.args) > 1:
                    raise Exception('The input needs %d components for %s' % (len(self.args), ', '.join(self.args)))
                if len(input) != len(self.args) and len(self.args) > 1:
                    raise Exception('The input needs %d components for %s' % (len(self.args), ', '.join(self.args)))
                for i, arg in self.args:
                    kwargs[arg] = input[i]
        if len(self.args) == 0:
            args = [input]
        kwargs.update(self.kwargs)

        return self.f(*args, **kwargs)

    def __repr__(self) -> str:
        s = 'Apply(%s' % self.f.__name__
        if len(self.args) > 0:
            s += ', '
            s += ', '.join(self.args)
        if len(self.kwargs) > 0:
            s += ', '
            s += ', '.join(['%s=%s' % (k, str(v)) for k, v in self.kwargs.items()])
        s += ')'
        return s



class VAE(pl.LightningModule):
    def __init__(self, z_dim, beta, lr):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.beta = beta
        self.lr = lr

        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)
        self.prior = Prior(z_dim=z_dim)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        x, _ = batch

        # Encode
        q_z_x = self.encoder(x)

        # Sample
        z = q_z_x.rsample()

        # Decode
        p_x_z = self.decoder(z)

        # KL( q(z|x) || p(z) )
        p_z = self.prior()
        kl_loss = torch.mean(q_z_x.log_prob(z) - p_z.log_prob(z))

        # E[-log p(x|z)]
        rec_loss = torch.mean(- p_x_z.log_prob(x))

        # Loss
        loss = rec_loss + self.beta * kl_loss

        self.log('Reconstruction', rec_loss.item())     # ADD FOR LOGGING
        self.log('Regularization', kl_loss.item())      # ADD FOR LOGGING
        self.log('Loss', kl_loss.item())                # ADD FOR LOGGING

        return loss

    def configure_optimizers(self):
        return Adam([
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ], lr=self.lr)
