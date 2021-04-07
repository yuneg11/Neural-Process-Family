import torch
from torch.nn import functional as F

from .mlps import PointwiseMLP


class DeterministicDecoder(PointwiseMLP):
    def __init__(self, x_dim, y_dim, repr_dim, hidden_dims):
        super().__init__(input_dim=(x_dim + repr_dim),
                         hidden_dims=hidden_dims,
                         output_dim=(y_dim * 2))

    def forward(self, repr, x_target):
        """
        Args:
            repr    : [batch_size, repr_dim] or
                      [batch_size, num_target_points, repr_dim]
            x_target: [batch_size, num_target_points, y_dim]

        Return:
            mu   : [batch_size, num_target_points, y_dim]
            sigma: [batch_size, num_target_points, y_dim]
        """

        if len(repr.shape) == 2:
            repr = repr.unsqueeze(dim=1).repeat(1, x_target.shape[1], 1)   # [batch_size, num_target_points, repr_dim]

        query = torch.cat((repr, x_target), dim=-1)                        # [batch_size, num_target_points, x_dim + repr_dim]
        mu_log_sigma = super().forward(query)                              # [batch_size, num_target_points, y_dim + y_dim]

        y_dim = mu_log_sigma.shape[2] // 2

        mu, log_sigma = torch.split(mu_log_sigma, (y_dim, y_dim), dim=-1)  # [batch_size, num_target_points, y_dim] x 2
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        return mu, sigma                                                   # [batch_size, num_target_points, y_dim] x 2
