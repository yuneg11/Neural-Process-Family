import jax
from jax import numpy as jnp
from jax import random


__all__ = [
    "broadcast_mask",
    "get_mask",
    "apply_mask",
    "masked_sum",
    "masked_mean",
    "repeat_axis",
]


def broadcast_mask(mask, ndim: int, axis: int = 0):
    """
    Broadcast a mask to a given axis.
    """
    axis = axis if axis >= 0 else axis + ndim
    mask = jnp.expand_dims(mask, axis=set(range(ndim)).difference({axis}))
    return mask


def get_mask(n: int, start: int = 0, stop: int = None):
    """
    Get a mask of shape (n,) which filled ones at index [start, stop).
    """
    stop = n if stop is None else stop
    mask = (start <= jnp.arange(n)) & (jnp.arange(n) < stop)
    return mask


def apply_mask(a, mask, axis: int, fill_value: int = 0):
    """
    Apply a mask to an array along a given axis.
    """
    assert a.shape[axis] == mask.shape[0], "Mask shape must match array shape along axis."
    a = jnp.where(broadcast_mask(mask, a.ndim, axis), a, fill_value)
    return a


def masked_sum(a, mask, axis: int, keepdims: bool = False):
    """
    Sum a masked array along a given axis.
    """
    assert a.shape[axis] == mask.shape[0], "Mask shape must match array shape along axis."
    mask = broadcast_mask(mask, a.ndim, axis)
    a = jnp.sum(a, axis=axis, keepdims=keepdims, where=mask)
    return a


def masked_mean(a, mask, axis: int, keepdims: bool = False):
    """
    Mean a masked array along a given axis.
    """
    assert a.shape[axis] == mask.shape[0], "Mask shape must match array shape along axis."
    mask = broadcast_mask(mask, a.ndim, axis)
    a = jnp.mean(a, axis=axis, keepdims=keepdims, where=mask)
    return a


def repeat_axis(a, repeats: int, axis: int):
    """
    Repeat an array along a given axis.
    """
    a = jnp.expand_dims(a, axis=axis)
    a = jnp.repeat(a, repeats, axis=axis)
    return a
