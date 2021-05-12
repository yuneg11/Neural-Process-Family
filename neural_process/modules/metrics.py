from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class LogLikelihood(nn.Module):
    @staticmethod
    def forward(mu, sigma, y_target):
        """
        mu: [batch, points, y_dim] or
            [batch, latents, points, y_dim]
        y_target: [batch, points, y_dim]
        """
        if mu.dim() == 4 and sigma.dim() == 4:
            num_latents = mu.shape[1]
            y_target = y_target.unsqueeze(dim=1).repeat(1, num_latents, 1, 1)
        dist = Normal(mu, sigma)  # [batch_size, num_target_points, y_dim]
        log_prob = dist.log_prob(y_target)  # [batch_size, num_target_points, y_dim]
        # log_likelihood = log_prob.mean(dim=0).sum()  # [1]
        log_likelihood = log_prob.sum(dim=(-2, -1)).mean()  # [1]
        return log_likelihood


class KLDivergence(nn.Module):
    @staticmethod
    def forward(q_context, q_target):
        # kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        kl = kl_divergence(q_target, q_context).sum(dim=(-2, -1)).mean()
        return kl
