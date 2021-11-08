import torch
import pytorch_lightning as pl

from torch.optim import Adam
from pytorch_lightning.utilities.types import STEP_OUTPUT
from architectures import Encoder, Decoder, IndependentGaussian


class VAE(pl.LightningModule):
    def __init__(self, z_dim, beta, lr):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.beta = beta
        self.lr = lr

        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)
        self.prior = IndependentGaussian(z_dim=z_dim)

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

        # Logging
        self.log('Train/loss', loss.item())
        self.log('Train/reconstruction', rec_loss.item())
        self.log('Train/regularization', kl_loss.item())

        return loss

    def sample(self, shape: torch.Size = torch.Size([])) -> torch.Tensor:
        return self.decoder(self.prior().sample(shape)).mean

    def reconstruct(self, x):
        return self.decoder(self.encoder(x).mean).mean

    def configure_optimizers(self):
        return Adam([
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ], lr=self.lr)
