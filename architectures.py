import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent, Normal, Distribution
from utils import Apply


class Encoder(nn.Module):
    def __init__(self, z_dim: int):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            Apply(torch.flatten, start_dim=1),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, z_dim * 2)
        )

    def forward(self, x: torch.Tensor) -> Distribution:
        params = self.net(x)
        mu, log_sigma = torch.chunk(params, 2, dim=1)
        q_sigma = F.softplus(log_sigma) + torch.finfo(torch.float32).eps
        q_z_x = Independent(Normal(loc=mu, scale=q_sigma), 1)

        return q_z_x


class Decoder(nn.Module):
    def __init__(self, z_dim: int):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            Apply(torch.reshape, shape=[-1, 1, 28, 28])
        )

    def forward(self, z: torch.Tensor) -> Distribution:
        mu = self.net(z)
        p_x_z = Independent(Normal(loc=mu, scale=0.1), 3)

        return p_x_z


class IndependentGaussian(nn.Module):
    def __init__(self, z_dim: int):
        super(IndependentGaussian, self).__init__()
        self.register_buffer('prior_mu', torch.zeros(z_dim))
        self.register_buffer('prior_sigma', torch.ones(z_dim))

    def forward(self) -> Distribution:
        return Independent(Normal(loc=self.prior_mu, scale=self.prior_sigma), 1)

    def extra_repr(self) -> str:
        return 'Normal'
