import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

import pytorch_lightning as pl


def input_reshape(x):
    """
    Args:
        x: [b, n, d]

    Return:
        x_: [b x n, d]
        b: int
        n: int
    """
    b, n, _ = x.shape
    x_ = x.reshape(b * n, -1)
    return x_, b, n


def output_reshape(y, b, n):
    """
    Args:
        y: [b x n, d]
        b: int
        n: int

    Return:
        y_: [b, n, d]
    """
    y_ = y.reshape(b, n, -1)
    return y_


def batch_mlp(input_dim, hidden_dims, output_dim):
    mlp = nn.Sequential()

    for i, next_dim in enumerate(hidden_dims):
        mlp.add_module(f'dense{i}', nn.Linear(input_dim, next_dim))
        mlp.add_module(f'relu{i}', nn.ReLU())
        input_dim = next_dim
    mlp.add_module(f'dense{len(hidden_dims)}', nn.Linear(input_dim, output_dim))

    return mlp


def mean_aggregator(context_x, target_x, hidden):
    return hidden.mean(dim=1)


def dot_product_attention(q, k, v):
    scale = torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float)).to(k.device)

    w = torch.bmm(q, k.transpose(2, 1)) / scale  # b m n
    w = torch.softmax(w, dim=1)  # b m n
    # w = torch.sigmoid(w)   # b m n
    w_v = torch.bmm(w, v)  # b m r_d

    return w_v


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, hidden_dims):
        super().__init__()

        self._mlp = batch_mlp(input_dim=(x_dim + y_dim),
                              hidden_dims=hidden_dims,
                              output_dim=r_dim)

    def forward(self, context_x, context_y):
        """
        Args:
            context_x: [batch_size, num_context_points, x_dim]
            context_y: [batch_size, num_context_points, y_dim]

        Return: (Based on aggregator output)
            r: [batch_size, num_context_points, r_dim]
        """
        encoder_input = torch.cat((context_x, context_y), dim=-1)

        mlp_input, batch_size, num_context_points = input_reshape(encoder_input)
        mlp_output = self._mlp(mlp_input)
        r = output_reshape(mlp_output, batch_size, num_context_points)

        return r


class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dims, s_dim=None):
        super().__init__()

        if s_dim is None:
            s_dim = (x_dim + y_dim + z_dim) // 2

        self._mlp = batch_mlp(input_dim=(x_dim + y_dim),
                              hidden_dims=hidden_dims[:-1],
                              output_dim=hidden_dims[-1])

        self._s_linear = nn.Linear(hidden_dims[-1], s_dim)
        self._s_relu = nn.ReLU()

        self._mu_linear = nn.Linear(s_dim, z_dim)
        self._sigma_linear = nn.Linear(s_dim, z_dim)

    def forward(self, x, y):
        """
        Args:
            x: [batch_size, num_points, x_dim]
            y: [batch_size, num_points, y_dim]

        Return:
            z: Normal distribution
        """
        encoder_input = torch.cat((x, y), dim=-1)

        mlp_input, batch_size, num_context_points = input_reshape(encoder_input)
        mlp_output = self._mlp(mlp_input)
        hidden = output_reshape(mlp_output, batch_size, num_context_points)

        s = hidden.mean(dim=1)  # [batch_size, x_dim + y_dim]

        s = self._s_linear(s)
        s = self._s_relu(s)

        mu = self._mu_linear(s)
        log_sigma = self._sigma_linear(s)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        z = Normal(loc=mu, scale=sigma)

        return z


class DeterministicDecoder(nn.Module):
    def __init__(self, x_dim, y_dim, rep_dim, hidden_dims):
        super().__init__()

        self._mlp = batch_mlp(input_dim=(x_dim + rep_dim),
                              hidden_dims=hidden_dims,
                              output_dim=(y_dim * 2))

    def forward(self, representation, target_x):
        """
        Args:
            representation: [batch_size, rep_dim] or
                            [batch_size, num_target_points, rep_dim]
            target_x: [batch_size, num_target_points, y_dim]
        """

        _, num_target_points, _ = target_x.shape

        if len(representation.shape) == 2:
            representation = representation.unsqueeze(dim=1).repeat(1, num_target_points, 1)

        decoder_input = torch.cat((representation, target_x), dim=-1)

        mlp_input, batch_size, num_target_points = input_reshape(decoder_input)
        mlp_output = self._mlp(mlp_input)
        hidden = output_reshape(mlp_output, batch_size, num_target_points)

        y_dim = hidden.shape[2] // 2

        mu, log_sigma = torch.split(hidden, (y_dim, y_dim), dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        return mu, sigma


class ConditionalNeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, encoder_dims, decoder_dims):
        super().__init__()

        self._encoder = DeterministicEncoder(x_dim, y_dim, r_dim, encoder_dims)
        self._decoder = DeterministicDecoder(x_dim, y_dim, r_dim, decoder_dims)

    def forward(self, context_x, context_y, target_x):
        r = self._encoder(context_x, context_y)
        representation = mean_aggregator(context_x, context_y, r)
        mu, sigma = self._decoder(representation, target_x)

        return mu, sigma


class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim,
                 latent_encoder_dims,
                 decoder_dims,
                 deterministic_encoder_dims=None):
        super().__init__()

        self._use_deterministic_path = (deterministic_encoder_dims is not None)

        if deterministic_encoder_dims:
            self._deterministic_encoder = DeterministicEncoder(x_dim, y_dim, r_dim, deterministic_encoder_dims)
        self._latent_encoder = LatentEncoder(x_dim, y_dim, z_dim, latent_encoder_dims)
        self._decoder = DeterministicDecoder(x_dim, y_dim, r_dim + z_dim, decoder_dims)

    def forward(self, context_x, context_y, target_x):
        z = self._latent_encoder(context_x, context_y)
        latent_r = z.rsample()

        if self._use_deterministic_path:
            r = self._deterministic_encoder(context_x, context_y)
            deterministic_r = mean_aggregator(context_x, context_y, r)

            representation = torch.cat((latent_r, deterministic_r), dim=-1)
        else:
            representation = latent_r

        mu, sigma = self._decoder(representation, target_x)

        return mu, sigma


class AttentiveNeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim,
                 latent_encoder_dims,
                 decoder_dims,
                 deterministic_encoder_dims,
                 attention_dims,
                 attention=dot_product_attention):
        super().__init__()

        self._deterministic_encoder = DeterministicEncoder(x_dim, y_dim, r_dim, deterministic_encoder_dims)
        self._latent_encoder = LatentEncoder(x_dim, y_dim, z_dim, latent_encoder_dims)
        self._decoder = DeterministicDecoder(x_dim, y_dim, r_dim + z_dim, decoder_dims)

        self._q_linear = batch_mlp(x_dim, attention_dims[:-1], attention_dims[-1])
        self._k_linear = batch_mlp(x_dim, attention_dims[:-1], attention_dims[-1])

        self._attention = attention

    def forward(self, context_x, context_y, target_x):
        r = self._deterministic_encoder(context_x, context_y)
        q = self._q_linear(target_x)
        k = self._k_linear(context_x)
        deterministic_r = self._attention(q, k, r)

        z = self._latent_encoder(context_x, context_y)
        latent_r = z.rsample()
        latent_r = latent_r.unsqueeze(dim=1).repeat(1, deterministic_r.shape[1], 1)

        representation = torch.cat((latent_r, deterministic_r), dim=-1)

        mu, sigma = self._decoder(representation, target_x)

        return mu, sigma


class ConditionalNeuralProcessModel(ConditionalNeuralProcess, pl.LightningModule):
    def training_step(self, batch, _):
        (context_x, context_y, target_x), target_y = batch

        mu, sigma = self(context_x, context_y, target_x)

        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(target_y)
        loss = -log_prob.mean(dim=0).sum()

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            (context_x, context_y, target_x), target_y = batch

            mu, sigma = self(context_x, context_y, target_x)

            img = self.plotter(context_x, context_y, target_x, target_y, mu, sigma)
            self.logger.experiment.add_image("test_images", img, self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class NeuralProcessModel(NeuralProcess, pl.LightningModule):
    def training_step(self, batch, _):
        (context_x, context_y, target_x), target_y = batch

        z = self._latent_encoder(context_x, context_y)
        latent_r = z.rsample()

        if self._use_deterministic_path:
            r = self._deterministic_encoder(context_x, context_y)
            deterministic_r = mean_aggregator(context_x, context_y, r)

            representation = torch.cat((latent_r, deterministic_r), dim=-1)
        else:
            representation = latent_r

        mu, sigma = self._decoder(representation, target_x)
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(target_y)

        q = self._latent_encoder(target_x, target_y)
        kl = kl_divergence(z, q)

        loss = -log_prob.mean(dim=0).sum() + kl.mean(dim=0).sum()

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            (context_x, context_y, target_x), target_y = batch

            mu, sigma = self(context_x, context_y, target_x)

            img = self.plotter(context_x, context_y, target_x, target_y, mu, sigma)
            self.logger.experiment.add_image("test_images", img, self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class AttentiveNeuralProcessModel(AttentiveNeuralProcess, pl.LightningModule):
    def training_step(self, batch, _):
        (context_x, context_y, target_x), target_y = batch

        r = self._deterministic_encoder(context_x, context_y)
        q = self._q_linear(target_x)
        k = self._k_linear(context_x)
        deterministic_r = self._attention(q, k, r)

        z = self._latent_encoder(context_x, context_y)
        latent_r = z.rsample()
        latent_r = latent_r.unsqueeze(dim=1).repeat(1, deterministic_r.shape[1], 1)

        representation = torch.cat((latent_r, deterministic_r), dim=-1)

        mu, sigma = self._decoder(representation, target_x)
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(target_y)

        q = self._latent_encoder(target_x, target_y)
        kl = kl_divergence(z, q)

        loss = -log_prob.mean(dim=0).sum() + kl.mean(dim=0).sum()

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            (context_x, context_y, target_x), target_y = batch

            mu, sigma = self(context_x, context_y, target_x)

            img = self.plotter(context_x, context_y, target_x, target_y, mu, sigma)
            self.logger.experiment.add_image("test_images", img, self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
