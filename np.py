import torch

from torch import nn
from torch.nn import functional as F

from torch.distributions.normal import Normal


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, hidden_dims):
        super().__init__()

        input_dim = x_dim + y_dim
        for i, output_dim in enumerate(hidden_dims):
            self.add_module(f'dense{i}', nn.Linear(input_dim, output_dim))
            self.add_module(f'relu{i}',  nn.ReLU())
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
            self._hidden_layers.add_module(f'relu{i}',  nn.ReLU())
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
            self.add_module(f'relu{i}',  nn.ReLU())
            input_dim = output_dim
        self.add_module(f'dense{len(hidden_dims)}', nn.Linear(input_dim, y_dim + y_dim))

    def forward(self, target_x, context_rep):
        """
        Args:
            context_r: [batch_size, rep_dim]
            target_x: [batch_size, num_target_points, y_dim]
        """
        batch_size, num_target_points, _ = target_x.shape

        r = context_rep.unsqueeze(dim=1).repeat(1, num_target_points, 1)
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

    def forward(self, context_x, context_y, target_x):
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

    def _aggregator(self, encoder_r):
        return encoder_r.mean(dim=1)

    def forward(self, context_x, context_y, target_x):
        encoder_r = self._deterministic_encoder(context_x, context_y)
        context_r = self._aggregator(encoder_r)

        encoder_z = self._latent_encoder(context_x, context_y)
        context_z = encoder_z.sample()

        context_rep = torch.cat((context_r, context_z), dim=1)

        mu, sigma = self._decoder(target_x, context_rep)

        return mu, sigma
