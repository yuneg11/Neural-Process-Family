import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

import pytorch_lightning as pl


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, hidden_dims):
        super().__init__()

        input_dim = x_dim + y_dim
        for i, output_dim in enumerate(hidden_dims):
            self.add_module(f'dense{i}', nn.Linear(input_dim, output_dim))
            self.add_module(f'relu{i}', nn.ReLU())
            input_dim = output_dim
        self.add_module(f'dense{len(hidden_dims)}', nn.Linear(input_dim, r_dim))

    def forward(self, context_x, context_y):
        """
        Args:
            context_x: [batch_size, num_context_points, x_dim]
            context_y: [batch_size, num_context_points, y_dim]

        Return:
            encoder_r: [batch_size, num_context_points, r_dim]
        """
        x = torch.cat((context_x, context_y), dim=-1)

        batch_size, num_context_points, _ = x.shape
        x = x.reshape(batch_size * num_context_points, -1)

        for _, module in self._modules.items():
            x = module(x)

        x = x.reshape(batch_size, num_context_points, -1)

        return x


class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dims):
        super().__init__()

        self._hidden_layers = nn.Sequential()
        input_dim = x_dim + y_dim
        for i, output_dim in enumerate(hidden_dims):
            self._hidden_layers.add_module(f'dense{i}', nn.Linear(input_dim, output_dim))
            self._hidden_layers.add_module(f'relu{i}', nn.ReLU())
            input_dim = output_dim

        self._mu_linear = nn.Linear(input_dim, z_dim)
        self._sigma_linear = nn.Linear(input_dim, z_dim)

    def forward(self, context_x, context_y):
        """
        Args:
            context_x: [batch_size, num_context_points, x_dim]
            context_y: [batch_size, num_context_points, y_dim]

        Return:
            encoder_z: [batch_size, num_context_points, z_dim]
        """
        x = torch.cat((context_x, context_y), dim=-1)

        batch_size, num_context_points, _ = x.shape
        x = x.reshape(batch_size * num_context_points, -1)

        for _, module in self._hidden_layers._modules.items():
            x = module(x)

        x = x.reshape(batch_size, num_context_points, -1)
        x = x.mean(dim=1)

        mu = self._mu_linear(x)
        log_sigma = self._sigma_linear(x)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        dist = Normal(loc=mu, scale=sigma)

        return dist


class DeterministicDecoder(nn.Module):
    def __init__(self, x_dim, y_dim, rep_dim, hidden_dims):
        super().__init__()

        input_dim = x_dim + rep_dim
        for i, output_dim in enumerate(hidden_dims):
            self.add_module(f'dense{i}', nn.Linear(input_dim, output_dim))
            self.add_module(f'relu{i}', nn.ReLU())
            input_dim = output_dim
        self.add_module(f'dense{len(hidden_dims)}', nn.Linear(input_dim, y_dim + y_dim))

    def forward(self, target_x, context_rep):
        """
        Args:
            context_r: [batch_size, rep_dim]
            target_x: [batch_size, num_target_points, y_dim]
        """
        batch_size, num_target_points, _ = target_x.shape

        if len(context_rep.shape) == 2:
            r = context_rep.unsqueeze(dim=1).repeat(1, num_target_points, 1)
        else:
            r = context_rep
        x = torch.cat((r, target_x), dim=-1)
        x = x.reshape(batch_size * num_target_points, -1)

        for _, module in self._modules.items():
            x = module(x)

        x = x.reshape(batch_size, num_target_points, -1)
        y_dim = x.shape[2] // 2

        mu, log_sigma = torch.split(x, (y_dim, y_dim), dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        return mu, sigma


class ConditionalNeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, encoder_dims, decoder_dims):
        super().__init__()

        self._encoder = DeterministicEncoder(x_dim, y_dim, r_dim, encoder_dims)
        self._decoder = DeterministicDecoder(x_dim, y_dim, r_dim, decoder_dims)

    def _aggregator(self, encoder_r):
        return encoder_r.mean(dim=1)

    def forward(self, input_x):
        context_x, context_y, target_x = input_x

        encoder_r = self._encoder(context_x, context_y)
        context_r = self._aggregator(encoder_r)
        mu, sigma = self._decoder(target_x, context_r)

        return mu, sigma


class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim,
                 deterministic_encoder_dims,
                 latent_encoder_dims,
                 decoder_dims):
        super().__init__()

        self._deterministic_encoder = DeterministicEncoder(x_dim, y_dim, r_dim, deterministic_encoder_dims)
        self._latent_encoder = LatentEncoder(x_dim, y_dim, z_dim, latent_encoder_dims)
        self._decoder = DeterministicDecoder(x_dim, y_dim, r_dim + z_dim, decoder_dims)

    def _aggregator(self, context_x, target_x, encoder_r):
        # return encoder_r.mean(dim=1).unsqueeze(dim=1).repeat(1, encoder_r.shape[1], 1)
        return encoder_r.mean(dim=1)

    def forward(self, input_x):
        context_x, context_y, target_x = input_x

        encoder_r = self._deterministic_encoder(context_x, context_y)
        context_r = self._aggregator(context_x, target_x, encoder_r)

        encoder_z = self._latent_encoder(context_x, context_y)
        context_z = encoder_z.rsample()

        context_rep = torch.cat((context_r, context_z), dim=1)

        mu, sigma = self._decoder(target_x, context_rep)

        return mu, sigma


class AttentiveNeuralProcess(NeuralProcess):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, att_dim, *args, **kwargs):
        super().__init__(x_dim, y_dim, r_dim, z_dim, *args, **kwargs)

        self._q_linear = nn.Linear(x_dim, att_dim)
        self._k_linear = nn.Linear(x_dim, att_dim)

    def _aggregator(self, context_x, target_x, encoder_r):
        q = self._q_linear(target_x)   # b m a_d
        k = self._k_linear(context_x)  # b n a_d
        v = encoder_r                  # b n r_d

        scale = torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float)).to(k.device)

        w = torch.bmm(q, k.transpose(2, 1)) / scale  # b m n
        w = torch.softmax(w, dim=1)  # b m n
        # w = torch.sigmoid(w)   # b m n
        w_v = torch.bmm(w, v)  # b m r_d

        return w_v

    def forward(self, input_x):
        context_x, context_y, target_x = input_x

        encoder_r = self._deterministic_encoder(context_x, context_y)
        context_r = self._aggregator(context_x, target_x, encoder_r)

        encoder_z = self._latent_encoder(context_x, context_y)
        context_z = encoder_z.rsample().unsqueeze(dim=1).repeat(1, context_r.shape[1], 1)

        context_rep = torch.cat((context_r, context_z), dim=-1)

        mu, sigma = self._decoder(target_x, context_rep)

        return mu, sigma


class ConditionalNeuralProcessModel(ConditionalNeuralProcess, pl.LightningModule):
    def training_step(self, batch, _):
        input_x, target_y = batch

        mu, sigma = self(input_x)

        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(target_y)
        loss = -log_prob.mean(dim=0).sum()

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            input_x, target_y = batch

            mu, sigma = self(input_x)

            context_x, context_y, target_x = input_x

            img = self.plotter(context_x, context_y, target_x, target_y, mu, sigma)
            self.logger.experiment.add_image("test_images", img, self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class NeuralProcessModel(NeuralProcess, pl.LightningModule):
    def training_step(self, batch, _):
        (context_x, context_y, target_x), target_y = batch

        encoder_r = self._deterministic_encoder(context_x, context_y)
        context_r = self._aggregator(context_x, target_x, encoder_r)

        encoder_z = self._latent_encoder(context_x, context_y)
        context_z = encoder_z.rsample()

        context_rep = torch.cat((context_r, context_z), dim=1)

        mu, sigma = self._decoder(target_x, context_rep)

        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(target_y)

        encoder_q = self._latent_encoder(target_x, target_y)
        kl = kl_divergence(encoder_z, encoder_q)

        loss = -log_prob.mean(dim=0).sum() + kl.mean(dim=0).sum()

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            input_x, target_y = batch

            mu, sigma = self(input_x)

            context_x, context_y, target_x = input_x

            img = self.plotter(context_x, context_y, target_x, target_y, mu, sigma)
            self.logger.experiment.add_image("test_images", img, self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class AttentiveNeuralProcessModel(AttentiveNeuralProcess, pl.LightningModule):
    def training_step(self, batch, _):
        (context_x, context_y, target_x), target_y = batch

        encoder_r = self._deterministic_encoder(context_x, context_y)
        context_r = self._aggregator(context_x, target_x, encoder_r)

        encoder_z = self._latent_encoder(context_x, context_y)
        context_z = encoder_z.rsample().unsqueeze(dim=1).repeat(1, context_r.shape[1], 1)

        context_rep = torch.cat((context_r, context_z), dim=-1)

        mu, sigma = self._decoder(target_x, context_rep)

        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(target_y)

        encoder_q = self._latent_encoder(target_x, target_y)
        kl = kl_divergence(encoder_z, encoder_q)

        loss = -log_prob.mean(dim=0).sum() + kl.mean(dim=0).sum()

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            input_x, target_y = batch

            mu, sigma = self(input_x)

            context_x, context_y, target_x = input_x

            img = self.plotter(context_x, context_y, target_x, target_y, mu, sigma)
            self.logger.experiment.add_image("test_images", img, self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
