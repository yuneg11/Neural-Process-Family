import torch
from torch import nn


class NNDecoder(nn.Module):
    def __init__(self, net, pointwise=True, latent=False):
        super().__init__()

        self.net = net
        self.pointwise = pointwise
        self.latent = latent

    def get_input_output_reshape(self, input_shape):
        if len(input_shape) == 3:
            # [batch, points, dim] or
            # [batch, latents, dim]
            batch_size, nums, input_dim = input_shape

            if self.pointwise:
                input_reshape = (batch_size * nums, input_dim)
                output_reshape = (batch_size, nums, -1)
            else:
                input_reshape = None
                output_reshape = None

        elif len(input_shape) == 4:
            batch_size, num_latents, num_points, input_dim = input_shape

            if self.pointwise:
                input_reshape = (batch_size * num_latents * num_points, input_dim)
                output_reshape = (batch_size, num_latents, num_points, -1)
            else:
                input_reshape = (batch_size * num_latents, num_points, input_dim)
                output_reshape = (batch_size, num_latents, num_points, -1)

        else:
            raise ValueError(f"input_dim mismatch '{len(input_shape)}'")

        return input_reshape, output_reshape

    def forward(self, x, rep):
        """
        x: [batch, points, x_dim]
        rep: [batch, rep_dim] or
             [batch, points, rep_dim] or
             [batch, latents, rep_dim] or
             [batch, latents, points, rep_dim]
        """
        if rep.dim() == 2:  # [batch, rep_dim]
            num_points = x.shape[1]
            rep = rep.unsqueeze(dim=1).repeat(1, num_points, 1)
        elif rep.dim() == 3 and self.latent:  # [batch, latents, rep_dim]
            num_points = x.shape[1]
            num_latents = rep.shape[1]
            rep = rep.unsqueeze(dim=2).repeat(1, 1, num_points, 1)
            x = x.unsqueeze(dim=1).repeat(1, num_latents, 1, 1)
        elif rep.dim() == 4:  # [batch, latents, points, rep_dim]
            num_latents = rep.shape[1]
            x = x.unsqueeze(dim=1).repeat(1, num_latents, 1, 1)

        # [batch, points, x_dim + rep_dim] or
        # [batch, latents, points, x_dim + rep_dim]
        input = torch.cat((x, rep), dim=-1)

        input_reshape, output_reshape = self.get_input_output_reshape(input.shape)

        if input_reshape is None or output_reshape is None:
            output = self.net(input)
        else:
            shaped_input = input.reshape(*input_reshape)
            shaped_output = self.net(shaped_input)
            output = shaped_output.reshape(*output_reshape)

        # [batch, points, output_dim] or
        # [batch, latents, points, output_dim]
        return output


class CNNDecoder(nn.Module):
    def __init__(self, net, latent=False):
        super().__init__()

        self.net = net
        self.latent = latent

    def forward(self, rep):
        if self.latent:
            batch_size, num_latents, num_points, rep_dim = rep.shape
            rep = rep.reshape(batch_size * num_latents, num_points, rep_dim)
        else:
            batch_size, num_points, rep_dim = rep.shape

        rep = rep.transpose(2, 1)
        rep = self.net(rep)
        f_rep = rep.transpose(1, 2)

        # Check that shape is still fine!
        if f_rep.shape[1] != num_points:
            raise RuntimeError('Shape changed.')

        if self.latent:
            f_rep = f_rep.reshape(batch_size, num_latents, num_points, -1)

        return f_rep
