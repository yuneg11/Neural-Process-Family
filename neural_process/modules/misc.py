import torch
from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F
from torch.distributions import Normal


class MeanAggregator(nn.Module):
    @staticmethod
    def forward(r_i):
        r = r_i.mean(dim=1)  # [batch, r_dim]
        return r


class DeterministicLatentConcatenator(nn.Module):
    @staticmethod
    def forward(r, z_samples):
        """
        r: [batch, r_dim] or
           [batch, points, r_dim]
        z_samples: [batch, latents, z_dim]
        """
        num_latents = z_samples.shape[1]

        if r.dim() == 2:
            num_points = None
            r = r.unsqueeze(dim=1).repeat(1, num_latents, 1)
        else:
            num_points = r.shape[1]
            r = r.unsqueeze(dim=1).repeat(1, num_latents, 1, 1)

        if num_points is not None and z_samples.dim() == 3:
            z_samples = z_samples.unsqueeze(dim=2).repeat(1, 1, num_points, 1)

        # [batch, latents, r_dim + z_dim] or
        # [batch, latents, points, r_dim + z_dim]
        r_z_samples = torch.cat((r, z_samples), dim=-1)
        return r_z_samples


class MuSigmaSplitter(nn.Module):
    @staticmethod
    def forward(mu_log_sigma):
        y_dim = mu_log_sigma.shape[-1] // 2

        # [batch, points, y_dim] x 2
        # [batch, latents, points, y_dim] x 2
        mu, log_sigma = torch.split(mu_log_sigma, (y_dim, y_dim), dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        return mu, sigma


class Distributor(nn.Module):
    def __init__(self, net):
        super().__init__()

        self.net = net

    def forward(self, s):
        mu_log_sigma = self.net(s)

        z_dim = mu_log_sigma.shape[-1] // 2

        # [batch_size, num_target_points, z_dim] x 2
        mu, log_sigma = torch.split(mu_log_sigma, (z_dim, z_dim), dim=-1)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        q_dist = Normal(loc=mu, scale=sigma)

        return q_dist


class Sampler(nn.Module):
    @staticmethod
    def forward(q_dist, num_samples=1):
        latents = q_dist.rsample([num_samples])
        z_samples = latents.transpose(1, 0)
        return z_samples


class Discretizer:
    def __init__(self, points_per_unit, multiplier, margin=0.1):
        self.points_per_unit = points_per_unit
        self.multiplier = multiplier
        self.margin = margin

    def __call__(self, x_context, x_target):
        x_min = min(torch.min(x_context).cpu().numpy(),
                    torch.min(x_target).cpu().numpy()) - self.margin
        x_max = max(torch.max(x_context).cpu().numpy(),
                    torch.max(x_target).cpu().numpy()) + self.margin

        x = self.points_per_unit * (x_max - x_min)
        if x % self.multiplier == 0:
            num_points = int(x)
        else:
            num_points = int(x + self.multiplier - x % self.multiplier)

        x_grid = torch.linspace(x_min, x_max, num_points).to(x_context.device)
        x_grid = x_grid[None, :, None].repeat(x_context.shape[0], 1, 1)

        return x_grid
