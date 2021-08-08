from ..type import *

import torch
from torch import distributions as torchdist
from torch.nn import functional as F


from nflows.transforms import LeakyReLU


# Interfaces

class Flow:
    def __init__(self,
        transform,
        distribution,
    ):
        self._params = None

        self.transform = transform
        self.distribution = distribution

    @property
    def param_dim(self) -> int:
        return self.transform.param_dim

    def __call__(self,
        params,
    ):
        self._params = params
        return self

    def log_prob(self,
        y_target: TensorType[B, T, Y],
        params=None,
    ) -> TensorType[B]:

        if params is None:
            if self._params is None:
                raise RuntimeError("params are not set")
            else:
                params = self._params

        noise, logabsdet = self.transform(y_target, params)
        log_prob = self.distribution.log_prob(noise)
        return log_prob + logabsdet


class Transform:
    @property
    def param_dim(self) -> int:
        raise NotImplementedError

    def __call__(self,
        y_target: TensorType[B, T, Y],
        params,
    ) -> TensorType[B, T, Y]:
        raise NotImplementedError

    def inverse(self,
        noise: TensorType[B, T, Y],
        params,
    ) -> TensorType[B, T, Y]:
        raise NotImplementedError


class CompositeTransform(Transform):
    def __init__(self, *transforms):
        super().__init__()
        self._transforms = transforms

    @property
    def param_dim(self) -> int:
        return sum([transform.param_dim for transform in self._transforms])

    @staticmethod
    def _cascade(funcs, inputs, params_list):
        outputs = inputs
        total_logabsdet = inputs.new_zeros(inputs.shape[0])
        for func, params in zip(funcs, params_list):
            outputs, logabsdet = func(outputs, params)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def __call__(self,
        y_target: TensorType[B, T, Y],
        params,
    ):
        transforms = self._transforms
        transform_funcs = transforms
        param_dims = [transform.param_dim for transform in transforms]
        params_list = params.split(param_dims, dim=-1)
        return self._cascade(transform_funcs, y_target, params_list)

    def inverse(self,
        noise: TensorType[B, T, Y],
        params,
    ):
        transforms = [transform for transform in reversed(self._transforms)]
        transform_funcs = [transform.inverse for transform in transforms]
        param_dims = [transform.param_dim for transform in transforms]
        params_list = params.split(param_dims, dim=-1)
        return self._cascade(transform_funcs, noise, params_list)


class Distribution:
    def log_prob(self,
        noise: TensorType[B, T, Y],
    ) -> TensorType[B]:
        raise NotImplementedError


# Implementations

class StandardNormal(Distribution):
    def __init__(self):
        self._dist = torchdist.Normal(0, 1)

    def log_prob(self,
        noise: TensorType[B, T, Y],
    ) -> TensorType[B]:

        log_prob = self._dist.log_prob(noise)
        log_prob = torch.mean(torch.sum(log_prob, dim=2), dim=1)
        return log_prob


class NaiveLinear(Transform):
    def __init__(self,
        dim: int,
    ):
        self._dim = dim

    def convert_params(self, params):
        dim = self._dim
        weight, bias = params.split((dim * dim, dim), dim=-1)
        weight = weight.view(*weight.shape[:-1], dim, dim)
        bias = bias.view(*bias.shape[:-1], dim)
        return weight, bias

    @property
    def param_dim(self) -> int:
        return self._dim * self._dim + self._dim

    def __call__(self,
        y_target: TensorType[B, T, Y],
        params,
    ) -> TensorType[B, T, Y]:
        weight, bias = self.convert_params(params)

        # assert y_dim == 1
        weight = weight.squeeze(-1)
        output = weight * y_target + bias
        logabsdet = torch.log(torch.abs(weight))
        logabsdet = torch.mean(torch.sum(logabsdet, dim=2), dim=1)

        return output, logabsdet

    def inverse(self,
        noise: TensorType[B, T, Y],
        params,
    ) -> TensorType[B, T, Y]:
        raise NotImplementedError


class LeakyReLU(Transform):
    def __init__(self, negative_slope=1e-2):
        if negative_slope <= 0:
            raise ValueError("Slope must be positive.")
        super().__init__()
        self.negative_slope = negative_slope
        self.log_negative_slope = torch.log(torch.as_tensor(self.negative_slope))

    @property
    def param_dim(self) -> int:
        return 0

    def __call__(self, inputs, params):
        outputs = F.leaky_relu(inputs, negative_slope=self.negative_slope)
        mask = (inputs < 0).type(torch.Tensor).to(inputs.device)
        logabsdet = self.log_negative_slope * mask
        logabsdet = torch.mean(torch.sum(logabsdet, dim=2), dim=1)
        return outputs, logabsdet

    def inverse(self, inputs, params):
        outputs = F.leaky_relu(inputs, negative_slope=(1 / self.negative_slope))
        mask = (inputs < 0).type(torch.Tensor).to(inputs.device)
        logabsdet = -self.log_negative_slope * mask
        logabsdet = torch.mean(torch.sum(logabsdet, dim=2), dim=1)
        return outputs, logabsdet


class Tanh(Transform):
    @property
    def param_dim(self) -> int:
        return 0

    def __call__(self, inputs, params):
        outputs = torch.tanh(inputs)
        logabsdet = torch.log(1 - outputs ** 2)
        logabsdet = torch.mean(torch.sum(logabsdet, dim=2), dim=1)
        return outputs, logabsdet

    def inverse(self, inputs, params):
        if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
            raise ValueError()
        outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
        logabsdet = -torch.log(1 - inputs ** 2)
        logabsdet = torch.mean(torch.sum(logabsdet, dim=2), dim=1)
        return outputs, logabsdet


class Sigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.temperature = temperature

    @property
    def param_dim(self) -> int:
        return 0

    def __call__(self, inputs, params):
        temperature = torch.tensor(self.temperature, dtype=torch.float, device=inputs.device)
        inputs = temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torch.log(temperature) \
                  - F.softplus(-inputs) \
                  - F.softplus(inputs)
        logabsdet = torch.mean(torch.sum(logabsdet, dim=2), dim=1)
        return outputs, logabsdet

    def inverse(self, inputs, params):
        temperature = torch.tensor(self.temperature, dtype=torch.float, device=inputs.device)
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise RuntimeError()

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torch.log(temperature) \
                  + F.softplus(-temperature * outputs) \
                  + F.softplus(temperature * outputs)
        logabsdet = torch.mean(torch.sum(logabsdet, dim=2), dim=1)
        return outputs, logabsdet


class Affine(Transform):
    def __init__(self,
        dim: int = 1,
    ):
        self._dim = dim

    @property
    def param_dim(self) -> int:
        return self._dim * 2

    def __call__(self, inputs, params):
        mat = params[..., 0]
        bias = torch.diagonal(params[..., 1], dim1=-2, dim2=-1)[:, :, None]
        _, logabsdet = torch.slogdet(mat)
        outputs = torch.matmul(mat, inputs) + bias  # [B, T, Y]
        logabsdet = logabsdet / inputs.shape[1]
        return outputs, logabsdet

    def inverse(self, inputs, params):
        raise NotImplementedError
