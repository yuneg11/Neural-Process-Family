import torch
from torch import nn
from torch.distributions import Normal


__all__ = [
    "LogLikelihood",
]


class LogLikelihood(nn.Module):
    @staticmethod
    def forward(y_target, mu, sigma):
        """
        y_target: [batch, target, y_dim] or [batch, latent, target, y_dim]
        mu:       [batch, target, y_dim] or [batch, latent, target, y_dim]
        sigma:    [batch, target, y_dim] or [batch, latent, target, y_dim]
        """

        distribution = Normal(mu, sigma)                                        # [batch, (latent,) target, y_dim]
        log_prob = distribution.log_prob(y_target)                              # [batch, (latent,) target, y_dim]
        log_likelihood = log_prob.sum(dim=-1).mean()                            # [1]
        return log_likelihood
