from typing import TYPE_CHECKING

from .misc import LazyModule

if TYPE_CHECKING:
    from . import jax
    # from . import torch
else:
    jax = LazyModule("npf.jax")
    # torch = LazyModule("npf.torch")


__all__ = [
    "jax",
    # "torch",
]

__version__ = "0.1.0"
