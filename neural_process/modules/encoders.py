import torch
from torch import nn

import numpy as np


class NNEncoder(nn.Module):
    def __init__(self, net, pointwise=True):
        super().__init__()

        self.net = net
        self.pointwise = pointwise

    def forward(self, x, y):
        input = torch.cat((x, y), dim=-1)

        batch_size, num_points, input_dim = input.shape

        if self.pointwise:
            pointwise_input = input.reshape(batch_size * num_points, input_dim)
            pointwise_output = self.net(pointwise_input)
            output = pointwise_output.reshape(batch_size, num_points, -1)
        else:
            output = self.net(input)

        return output


class SetConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        init_length_scale,
        activation=None,
        learn_length_scale=True,
        latent=False
    ):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.net = nn.Linear(self.in_channels, self.out_channels)
        nn.init.xavier_normal_(self.net.weight, gain=1)
        nn.init.constant_(self.net.bias, val=0.0)

        self.activation = activation

        self.sigma = nn.Parameter(torch.full((self.in_channels,), np.log(init_length_scale)),
                                  requires_grad=learn_length_scale)

        self.latent = latent

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations `t`.
        Args:
            x (tensor): Inputs of observations of shape `(n, 1)`.
            y (tensor): Outputs of observations of shape `(n, in_channels)`.
            t (tensor): Inputs to evaluate function at of shape `(m, 1)`.
        Returns:
            tensor: Outputs of evaluated function at `z` of shape `(m, out_channels)`.
        """
        if self.latent:
            batch_size, num_latents, n_in, in_channels = y.shape
            x = x.repeat_interleave(num_latents, dim=0)
            t = t.repeat_interleave(num_latents, dim=0)
            y = y.reshape(batch_size * num_latents, n_in, in_channels)

        batch_size = x.shape[0]
        n_in = x.shape[1]
        # n_out = t.shape[1]

        dists = self.compute_dists(x, t)  # [batch, n_in, n_out]
        wt = self.rbf(dists)  # [batch, n_in, n_out, in_channels]

        y = self.y_preprocess(y)

        # [batch, n_in, n_out, in_channels]
        y = y.reshape(batch_size, n_in, -1, self.in_channels) * wt
        y_out = y.sum(1)  # [batch, n_out, in_channels]

        y_out = self.y_postprocess(y_out)
        y_out = self.y_transform(y_out)  # [batch, n_out, out_channels]

        if self.activation is not None:
            y_out = self.activation(y_out)

        if self.latent:
            y_out = y_out.reshape(batch_size // num_latents, num_latents, -1, self.out_channels)

        return y_out

    def rbf(self, dists):
        """
        Compute the RBF values for the distances using the correct length scales.
        Args:
            dists (tensor): Pair-wise distances between `x` and `t`.
        Returns:
            tensor: Evaluation of `psi(x, t)` with `psi` an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = torch.exp(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.reshape(a, b, c, -1) / scales ** 2)

    @staticmethod
    def compute_dists(x0, x1):
        """
        Fast computation of pair-wise distances for the 1d case.
        Args:
            x0 (tensor): Inputs of shape `(batch, n, 1)`.
            x1 (tensor): Inputs of shape `(batch, m, 1)`.
        Returns:
            tensor: Pair-wise distances of shape `(batch, n, m)`.
        """
        assert x0.shape[2] == 1 and x1.shape[2] == 1, \
            'The inputs x0 and x1 must be 1-dimensional observations.'
        return (x0 - x1.permute(0, 2, 1)) ** 2

    @staticmethod
    def y_preprocess(y):
        return y

    @staticmethod
    def y_postprocess(y_out):
        return y_out

    def y_transform(self, y_out):
        batch_size, n_out, in_channels = y_out.shape

        pointwise_input = y_out.reshape(batch_size * n_out, in_channels)
        pointwise_output = self.net(pointwise_input)
        y_out = pointwise_output.reshape(batch_size, n_out, -1)

        return y_out


class SetConvEncoder(SetConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        init_length_scale,
        activation=None,
        learn_length_scale=True,
    ):
        super().__init__(
            in_channels=in_channels + 1,
            out_channels=out_channels,
            init_length_scale=init_length_scale,
            activation=activation,
            learn_length_scale=learn_length_scale,
        )

    @staticmethod
    def y_preprocess(y):
        batch_size, n_in = y.shape[:2]
        density = torch.ones(batch_size, n_in, 1).to(y.device)  # [batch, n_in, 1]
        y = torch.cat([density, y], dim=-1)  # [batch, n_in, in_channels]
        return y

    @staticmethod
    def y_postprocess(y_out):
        density, conv = y_out[..., :1], y_out[..., 1:]
        normalized_conv = conv / (density + 1e-8)
        y_out = torch.cat((density, normalized_conv), dim=-1)
        return y_out
