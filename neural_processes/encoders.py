import torch
from torch import nn
from torch.distributions.normal import Normal

from .mlps import MLP, PointwiseMLP
from .aggregators import MeanAggregator


class DeterministicEncoder(PointwiseMLP):
    def __init__(self, x_dim, y_dim, r_dim, hidden_dims):
        super().__init__(input_dim=(x_dim + y_dim),
                         hidden_dims=hidden_dims,
                         output_dim=r_dim)

    def forward(self, x, y):
        """
        Args:
            x: [batch_size, num_points, x_dim]
            y: [batch_size, num_points, y_dim]

        Return:
            r_i: [batch_size, num_points, r_dim]   Pointwise representation
        """

        input = torch.cat((x, y), dim=-1)  # [batch_size, num_points, x_dim + y_dim]
        r_i = super().forward(input)       # [batch_size, num_points, r_dim]

        return r_i  # Pointwise representation


class LatentSampler(nn.Module):
    def __init__(self, s_dim, z_dim, hidden_dims):
        super().__init__()

        self.mlp = MLP(input_dim=s_dim,
                       hidden_dims=hidden_dims[:-1],
                       output_dim=hidden_dims[-1],
                       last_relu=True)

        self.mu_linear = nn.Linear(hidden_dims[-1], z_dim)
        self.sigma_linear = nn.Linear(hidden_dims[-1], z_dim)

    def forward(self, s):
        """
        Args:
            s: [batch_size, s_dim]

        Return:
            z: [batch_size, z_dim]
        """

        hidden = self.mlp(s)                          # [batch_size, s_dim]

        mu = self.mu_linear(hidden)                   # [batch_size, z_dim]
        log_sigma = self.sigma_linear(hidden)         # [batch_size, z_dim]
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)  # [batch_size, z_dim]

        z_dist = Normal(loc=mu, scale=sigma)

        return z_dist


class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, s_dim, z_dim, hidden_dims, sampler_dims):
        super().__init__()

        self.s_encoder = DeterministicEncoder(x_dim, y_dim, s_dim, hidden_dims)
        self.z_sampler = LatentSampler(s_dim, z_dim, sampler_dims)
        self.aggregator = MeanAggregator()

    def forward(self, x, y):
        """
        Args:
            x: [batch_size, num_points, x_dim]
            y: [batch_size, num_points, y_dim]

        Return:
            z_dist: [batch_size, z_dim]
        """

        s_i = self.s_encoder(x, y)            # [batch_size, num_points, s_dim]
        s = self.aggregator(None, None, s_i)  # [batch_size, s_dim]
        z_dist = self.z_sampler(s)            # [batch_size, z_dim]

        return z_dist
