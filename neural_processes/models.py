import math

import torch
from torch import nn

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from .encoders import DeterministicEncoder, LatentEncoder
from .aggregators import MeanAggregator, DotProductAttention
from .decoders import DeterministicDecoder


class NeuralProcessBase(nn.Module):
    def loss(self, x_context, y_context, x_target, y_target):
        raise NotImplementedError

    def predictive_log_likelihood(self, x_context, y_context, x_target, y_target, num_samples=100):
        m = x_context.shape[1]                                               # m = num_context_points
        n = x_target.shape[1]                                                # n = num_target_points

        with torch.no_grad():
            x_target_wo_context = x_target[:, m:, :]                         # [batch_size, n - m, x_dim]
            y_target_wo_context = y_target[:, m:, :]                         # [batch_size, n - m, x_dim]

            log_likelihood_samples = []                                      # [num_samples]

            for _ in range(num_samples):
                mu, sigma = self(x_context, y_context, x_target_wo_context)  # [batch_size, n - m, y_dim]

                dist = Normal(mu, sigma)                                     # [batch_size, n - m, y_dim]
                log_prob = dist.log_prob(y_target_wo_context)                # [batch_size, n - m, y_dim]
                log_likelihood_sample = log_prob.mean(dim=0).sum()           # [1]

                log_likelihood_samples.append(log_likelihood_sample.item())

            log_likelihoods = torch.tensor(log_likelihood_samples)
            log_likelihood = torch.logsumexp(log_likelihoods, dim=0).item() - math.log(num_samples)

            # log_likelihood_sum = 0.

            # for _ in range(num_samples):
            #     mu, sigma = self(x_context, y_context, x_target_wo_context)

            #     dist = Normal(mu, sigma)
            #     log_prob = dist.log_prob(y_target_wo_context)
            #     log_likelihood_sample = log_prob.mean(dim=0).sum()
            #     log_likelihood_sum += log_likelihood_sample.item()

            # log_likelihood = log_likelihood_sum / num_samples

        predictive_log_likelihood = log_likelihood / (n - m)

        return predictive_log_likelihood


class ConditionalNeuralProcess(NeuralProcessBase):
    def __init__(self, x_dim, y_dim, r_dim, encoder_dims, decoder_dims):
        super().__init__()

        self.encoder = DeterministicEncoder(x_dim, y_dim, r_dim, encoder_dims)
        self.decoder = DeterministicDecoder(x_dim, y_dim, r_dim, decoder_dims)
        self.aggregator = MeanAggregator()

    def forward(self, x_context, y_context, x_target):
        """
        Args:
            x_context: [batch_size, num_context_points, x_dim]
            y_context: [batch_size, num_context_points, y_dim]
            x_target : [batch_size,  num_target_points, x_dim]

        Return:
            mu   : [batch_size, num_target_points, y_dim]
            sigma: [batch_size, num_target_points, y_dim]
        """

        r_i = self.encoder(x_context, y_context)       # [batch_size, num_context_points, r_dim]
        r = self.aggregator(x_context, x_target, r_i)  # [batch_size, r_dim] or
                                                       #   [batch_size, num_context_points, r_dim]
        mu, sigma = self.decoder(r, x_target)          # [batch_size, num_target_points, y_dim] x 2

        return mu, sigma

    def loss(self, x_context, y_context, x_target, y_target):
        """
        Args:
            x_context: [batch_size, num_context_points, x_dim]
            y_context: [batch_size, num_context_points, y_dim]
            x_target : [batch_size,  num_target_points, x_dim]
            y_target : [batch_size,  num_target_points, y_dim]

        Return:
            normalized_loss: float
        """

        mu, sigma = self(x_context, y_context, x_target)  # [batch_size, num_target_points, y_dim] x 2

        dist = Normal(mu, sigma)                          # [batch_size, num_target_points, y_dim]
        log_prob = dist.log_prob(y_target)                # [batch_size, num_target_points, y_dim]
        log_likelihood = log_prob.mean(dim=0).sum()       # [1]

        loss = -log_likelihood
        normalized_loss = loss / log_prob.shape[1]

        return normalized_loss


class NeuralProcess(NeuralProcessBase):
    def __init__(self, x_dim, y_dim, r_dim, s_dim, z_dim,
                 latent_dims,
                 sampler_dims,
                 decoder_dims,
                 deterministic_dims=None):
        super().__init__()

        self.use_deterministic = (deterministic_dims is not None)

        if deterministic_dims:
            self.deterministic_encoder = DeterministicEncoder(x_dim, y_dim, r_dim, deterministic_dims)
        self.latent_encoder = LatentEncoder(x_dim, y_dim, s_dim, z_dim, latent_dims, sampler_dims)
        self.decoder = DeterministicDecoder(x_dim, y_dim, r_dim + z_dim, decoder_dims)
        self.aggregator = MeanAggregator()

    def forward(self, x_context, y_context, x_target, return_latent_dist=False):
        """
        Args:
            x_context: [batch_size, num_context_points, x_dim]
            y_context: [batch_size, num_context_points, y_dim]
            x_target : [batch_size,  num_target_points, x_dim]

        Return:
            mu    : [batch_size, num_target_points, y_dim]
            sigma : [batch_size, num_target_points, y_dim]
            z_dist: Normal distribution (Optional)
        """

        z_dist = self.latent_encoder(x_context, y_context)
        z = z_dist.rsample()                                        # [batch_size, z_dim]

        if self.use_deterministic:
            r_i = self.deterministic_encoder(x_context, y_context)  # [batch_size, num_context_points, r_dim]
            r = self.aggregator(x_context, x_target, r_i)           # [batch_size, r_dim] or
                                                                    #   [batch_size, num_context_points, r_dim]
            if len(r.shape) == 3:
                z = z.unsqueeze(dim=1).repeat(1, r.shape[1], 1)     # [batch_size, num_context_points, r_dim]

            repr = torch.cat((r, z), dim=-1)                        # [batch_size, num_context_points, r_dim + z_dim]
        else:
            repr = z                                                # [batch_size, z_dim]

        mu, sigma = self.decoder(repr, x_target)                    # [batch_size, num_target_points, y_dim] x 2

        if return_latent_dist:
            return mu, sigma, z_dist
        else:
            return mu, sigma

    def loss(self, x_context, y_context, x_target, y_target):
        """
        Args:
            x_context: [batch_size, num_context_points, x_dim]
            y_context: [batch_size, num_context_points, y_dim]
            x_target : [batch_size,  num_target_points, x_dim]
            y_target : [batch_size,  num_target_points, y_dim]

        Return:
            normalized_loss: float
        """

        mu, sigma, q_context = self(x_context, y_context, x_target, return_latent_dist=True)
                                                                   # [batch_size, num_target_points, y_dim] x 2

        dist = Normal(mu, sigma)                                   # [batch_size, num_target_points, y_dim]
        log_prob = dist.log_prob(y_target)                         # [batch_size, num_target_points, y_dim]
        log_likelihood = log_prob.mean(dim=0).sum()                # [1]

        q_target = self.latent_encoder(x_target, y_target)         # [batch_size, z_dim]
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()  # [1]

        loss = -log_likelihood + kl
        normalized_loss = loss / log_prob.shape[1]

        return normalized_loss

    def joint_log_likelihood(self, x_target, y_target, num_samples=100):
        n = x_target.shape[1]

        with torch.no_grad():
            log_likelihood_sum = 0.
            z_dist = None

            for _ in range(num_samples):
                mu, sigma, q_dist = self(x_target, y_target, x_target, return_latent_dist=True)

                if z_dist is None:
                    z_dist = Normal(loc=torch.zeros_like(q_dist.loc), scale=torch.ones_like(q_dist.scale))

                dist = Normal(mu, sigma)
                log_prob = dist.log_prob(y_target)
                log_likelihood_sample = log_prob.mean(dim=0).sum()

                q_dist = self.latent_encoder(x_target, y_target)
                kl_sample = kl_divergence(q_dist, z_dist).mean(dim=0).sum()

                log_likelihood_sum += log_likelihood_sample.item() - kl_sample.item()

            log_likelihood = log_likelihood_sum / num_samples

        predictive_log_likelihood = log_likelihood / n

        return predictive_log_likelihood


class AttentiveConditionalNeuralProcess(ConditionalNeuralProcess):
    def __init__(self, x_dim, y_dim, r_dim,
                 encoder_dims,
                 decoder_dims,
                 attention_dims,
                 attention="dot_product"):
        super().__init__(x_dim, y_dim, r_dim,
                         encoder_dims,
                         decoder_dims)

        if attention == "dot_product":
            self.aggregator = DotProductAttention(input_dim=x_dim, hidden_dims=attention_dims, output_dim=r_dim)
        else:
            raise NameError(f"'{attention}' is not supported attention")


class AttentiveNeuralProcess(NeuralProcess):
    def __init__(self, x_dim, y_dim, r_dim, s_dim, z_dim,
                 latent_dims,
                 sampler_dims,
                 decoder_dims,
                 deterministic_dims,
                 attention_dims,
                 attention="dot_product"):
        super().__init__(x_dim, y_dim, r_dim, s_dim, z_dim,
                         latent_dims,
                         sampler_dims,
                         decoder_dims,
                         deterministic_dims)

        if attention == "dot_product":
            self.aggregator = DotProductAttention(input_dim=x_dim, hidden_dims=attention_dims, output_dim=r_dim)
        else:
            raise NameError(f"'{attention}' is not supported attention")
