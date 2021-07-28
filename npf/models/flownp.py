from typing import List, Tuple
from torchtyping import TensorType

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from .base import ConditionalNPF

from ..modules import (
    MLP,
    LogLikelihood,
)

__all__ = [
    "FlowNP",
    "FlowMixtureNP",
    "FlowLeakyNP",
    "FlowAffineNP",
]


class FlowNPBase(nn.Module):
    @property
    def is_latent_model(self):
        return False

    def encode(self, x_context, y_context, x_target, split=1):
        context = torch.cat((x_context, y_context), dim=-1)
        represent = self.encoder(context)

        aggregate = torch.mean(represent, dim=1, keepdim=True)
        aggregate = aggregate.repeat(1, x_target.shape[1], 1)

        query = torch.cat((x_target, aggregate), dim=-1)
        theta = self.transform(query)
        theta_i = torch.split(theta, split, dim=-1)

        return theta_i

    def log_likelihood(self, x_context, y_context, x_target, y_target):
        log_likelihood = -self.loss(x_context, y_context, x_target, y_target)
        return log_likelihood


class FlowNP(FlowNPBase):
    def __init__(self):
        super().__init__()
        self.encoder = MLP(2, [128, 128], 128)
        self.transform = MLP(129, [128, 128], 2)
        self.z_dist = Normal(0, 1)

    def forward(self, x_context, y_context, x_target):
        theta_1, theta_2 = self.encode(x_context, y_context, x_target)
        z_sample = self.z_dist.rsample()
        y_target = theta_1 * z_sample + theta_2
        return y_target

    def loss(self, x_context, y_context, x_target, y_target):
        theta_1, theta_2 = self.encode(x_context, y_context, x_target)
        z_target = (y_target - theta_2) / theta_1
        log_likelihood = self.z_dist.log_prob(z_target) - torch.log(torch.abs(theta_1))
        return -torch.mean(log_likelihood)


class FlowMixtureNP(FlowNPBase):
    def __init__(self):
        super().__init__()
        self.encoder = MLP(2, [128, 128], 128)
        self.transform = MLP(129, [128, 128], 2)

        self.pi_param = nn.Parameter(torch.rand(3))
        self.mean_param = nn.Parameter(torch.randn(3))
        self.log_sigma_param = nn.Parameter(torch.randn(3))

    def forward(self, x_context, y_context, x_target):
        theta_1, theta_2 = self.encode(x_context, y_context, x_target)
        pi = 0.1 + 0.9 * F.softplus(self.pi_param)
        pi = pi / torch.sum(pi)
        sigma = 0.1 + 0.9 * F.softplus(self.log_sigma_param)
        mean = self.mean_param
        z_dist = Normal(mean, sigma)
        z_sample = z_dist.rsample()
        z = torch.sum(z_sample * pi)
        y_target = theta_1 * z + theta_2
        return y_target

    def loss(self, x_context, y_context, x_target, y_target):
        theta_1, theta_2 = self.encode(x_context, y_context, x_target)
        z_target = (y_target - theta_2) / theta_1
        pi = 0.1 + 0.9 * F.softplus(self.pi_param)
        pi = pi/torch.sum(pi)
        sigma = 0.1 + 0.9 * F.softplus(self.log_sigma_param)
        mean = self.mean_param
        z_dist = Normal(mean, sigma)

        log_sum_prod = torch.logsumexp(z_dist.log_prob(z_target) + torch.log(pi[None, None, :]), dim=2, keepdim=True)
        log_likelihood = log_sum_prod - torch.log(torch.abs(theta_1))

        return -torch.mean(log_likelihood)


class FlowLeakyNP(FlowNPBase):
    def __init__(self):
        super().__init__()
        self.encoder = MLP(2, [128, 128], 128)
        self.transform = MLP(129, [128, 128], 3)
        self.z_dist = Normal(0, 1)

    def forward(self, x_context, y_context, x_target):
        a, b, c = self.encode(x_context, y_context, x_target)
        z_sample = self.z_dist.rsample()
        y_target = a * (z_sample - b) ** 3 + c
        return y_target

    def loss(self, x_context, y_context, x_target, y_target):
        a, b, c = self.encode(x_context, y_context, x_target)
        middle = ((y_target - c) / a)
        z_target = middle.sign() * (middle.abs())**(1/3) + b
        jacobian = 3 * a * (z_target - b) ** 2
        log_likelihood = self.z_dist.log_prob(z_target) - torch.log(jacobian.abs())
        return -torch.mean(log_likelihood)


class FlowAffineNP(FlowNPBase):
    dims = [16, 8, 1]

    def __init__(self):
        super().__init__()

        self.w1_dim = self.dims[0] * self.dims[1]
        self.b1_dim = self.dims[1]
        self.w2_dim = self.dims[1] * self.dims[2]
        self.b2_dim = self.dims[2]


        self.encoder = MLP(2, [128, 128], 128)
        self.transform = MLP(129, [128, 128], self.w1_dim + self.b1_dim + self.w2_dim + self.b2_dim + 1)
        self.z_dist = Normal(0, 1)

    def nn(self, x_context, y_context, x_target):
        theta = self.encode(x_context, y_context, x_target, split=(self.w1_dim, self.b1_dim, self.w2_dim, self.b2_dim, 1))
        linear1_weight, linear1_bias, linear2_weight, linear2_bias, theta_1 = theta

        batch_size, target_size = x_target.shape[:2]

        linear1_weight = linear1_weight.view(batch_size, target_size, self.dims[1], self.dims[0])
        linear2_weight = linear2_weight.view(batch_size, target_size, self.dims[2], self.dims[1])

        z_sample = self.z_dist.sample([batch_size, target_size, self.dims[0]]).to(x_target.device)

        theta_2 = torch.einsum("btj,btij->bti", z_sample, linear1_weight) + linear1_bias
        theta_2 = F.relu(theta_2)
        theta_2 = torch.einsum("btj,btij->bti", theta_2, linear2_weight) + linear2_bias

        return theta_1, theta_2

    def forward(self, x_context, y_context, x_target):
        theta_1, theta_2 = self.nn(x_context, y_context, x_target)

        batch_size, target_size = x_target.shape[:2]
        z_sample = self.z_dist.sample([batch_size, target_size, 1]).to(x_target.device)

        y_target = theta_1 * z_sample + theta_2
        return y_target

    def loss(self, x_context, y_context, x_target, y_target):
        theta_1, theta_2 = self.nn(x_context, y_context, x_target)
        z_target = (y_target - theta_2) / theta_1
        log_likelihood = self.z_dist.log_prob(z_target) - torch.log(torch.abs(theta_1))
        return -torch.mean(log_likelihood)
