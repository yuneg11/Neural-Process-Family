from __future__ import annotations

from functools import wraps

from jax import numpy as jnp
from jax.scipy import stats

from .data import NPData
from .typing import *
from . import functional as F


__all__ = [
    "MultivariateNormalDiag",
    "input_to_npdata",
    "npf_io",
]


class MultivariateNormalDiag:
    """
    Utility class for multivariate normal distribution.
    Resembles `tensorflow.distributions.MultivariateNormalDiag`.
    """

    def __init__(
        self,
        loc,
        scale_diag,
    ):
        self.loc = loc
        self.scale_diag = scale_diag

    def kl_divergence(self, other: MultivariateNormalDiag):
        per_dim_kld = (
            jnp.log(other.scale_diag) - jnp.log(self.scale_diag)
            + (jnp.square(self.scale_diag) + jnp.square(self.loc - other.loc)) / (2 * jnp.square(other.scale_diag))
            - 0.5
        )
        kld = jnp.sum(per_dim_kld, axis=-1)
        return kld

    def log_prob(self, value):
        per_dim_log_prob = stats.norm.logpdf(value, self.loc, self.scale_diag)
        log_prob = jnp.sum(per_dim_log_prob, axis=-1)
        return log_prob


# TODO: Improve typing

def input_to_npdata(args, kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], NPData):
            data = args[0]
        elif len(args) == 4:
            data = NPData(x=args[0], y=args[1], mask_ctx=args[2], mask_tar=args[3])
        elif len(args) == 6:
            data = NPData(x_ctx=args[0], x_tar=args[1], y_ctx=args[2], y_tar=args[3], mask_ctx=args[4], mask_tar=args[5])
        else:
            raise NotImplementedError("Input to npf_io must be either NPData or 4 or 6 arguments")
    else:
        data = NPData(
            x=kwargs.pop("x", None), x_ctx=kwargs.pop("x_ctx", None), x_tar=kwargs.pop("x_tar", None),
            y=kwargs.pop("y", None), y_ctx=kwargs.pop("y_ctx", None), y_tar=kwargs.pop("y_tar", None),
            mask_ctx=kwargs.pop("mask_ctx", None), mask_tar=kwargs.pop("mask_tar", None),
        )
    return data, kwargs


# TODO: Add docstrings

@overload
def npf_io(func: Callable) -> Callable: ...

@overload
def npf_io(flatten: bool = False) -> Callable: ...

@overload
def npf_io(flatten_input: bool = False) -> Callable: ...

def npf_io(
    func: Optional[Callable] = None,
    flatten: bool = False,
    flatten_input: bool = False,
):
    """Neural Process Family input/output decorator.

    Args:
        func: Function to decorate.
        flatten (bool): If true, the inputs to the decorated function will be flatten, and the outputs
                        from the decorated function will be unflatten. Defaults to False.
        flatten_input (bool): If true, the inputs to the decorated function will be flatten. Defaults to False.

    Returns:
        Callable: The decorated function.
    """

    if func is None:
        return lambda func: npf_io(func, flatten=flatten, flatten_input=flatten_input)

    assert not (flatten and flatten_input), "Cannot use both flatten and flatten_input"

    # NOTE: Find a better way to do this. This version does not work with static methods.
    #       But it may not be the problem because we are not using static methods (yet).
    # skip_self = hasattr(func, "__self__")
    skip_self = True

    if flatten:
        @wraps(func)
        def wrapper(*args, skip_io: bool = False, **kwargs):
            if skip_io:
                mu, sigma, *aux = func(*args, **kwargs)
            else:
                if skip_self:
                    self, *args = args

                data, kwargs = input_to_npdata(args, kwargs)
                flatten_data, _, _, tar_shape = data.flatten(return_shape=True)

                if skip_self:
                    flatten_mu, flatten_sigma, *aux = func(self, flatten_data, **kwargs)
                else:
                    flatten_mu, flatten_sigma, *aux = func(flatten_data, **kwargs)

                mu    = F.unflatten(flatten_mu,    tar_shape, axis=-2)
                sigma = F.unflatten(flatten_sigma, tar_shape, axis=-2)
            return mu, sigma, *aux
        return wrapper

    elif flatten_input:
        @wraps(func)
        def wrapper(*args, skip_io: bool = False, **kwargs):
            if skip_io:
                output = func(*args, **kwargs)
            else:
                if skip_self:
                    self, *args = args

                data, kwargs = input_to_npdata(args, kwargs)
                flatten_data = data.flatten(return_shape=False)

                if skip_self:
                    output = func(self, flatten_data, **kwargs)
                else:
                    output = func(flatten_data, **kwargs)
            return output
        return wrapper

    else:
        @wraps(func)
        def wrapper(*args, skip_io: bool = False, **kwargs):
            if skip_io:
                output = func(*args, **kwargs)
            else:
                if skip_self:
                    self, *args = args

                data, kwargs = input_to_npdata(args, kwargs)

                if skip_self:
                    output = func(self, data, **kwargs)
                else:
                    output = func(data, **kwargs)
            return output
        return wrapper
