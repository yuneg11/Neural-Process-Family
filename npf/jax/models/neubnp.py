from ..typing import *

from jax import numpy as jnp
from jax import random
from jax.scipy import stats
from flax import linen as nn

from .cnp import CNPBase
from .canp import CANPBase
from .. import functional as F
from ..modules import (
    MLP,
    MultiheadAttention,
    MultiheadSelfAttention,
)


__all__ = [
    "NeuBNPBase",
    "NeuBANPBase",
    "NeuBNP",
    "NeuBANP",
]


class NeuBNPMixin(nn.Module):
    """
    Mixins for Neural Bootstrapping Neural Process
    """

    def _sample_weight(
        self,
        mask_ctx: Array[B, C],
        num_samples: int = 1,
    ):

        key = self.make_rng("sample")
        alpha = jnp.expand_dims(mask_ctx, axis=1)                                                   # [batch, 1,      context]
        w_ctx = random.dirichlet(key, alpha=alpha, shape=(mask_ctx.shape[0], num_samples))          # [batch, sample, context]
        w_ctx = jnp.expand_dims(w_ctx, axis=-1)                                                     # [batch, sample, context, 1]
        w_ctx = w_ctx * jnp.expand_dims(jnp.sum(mask_ctx, axis=-1), axis=(1, -2, -1))               # [batch, sample, context, 1]
        return w_ctx

    @nn.compact
    def __call__(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_samples: int = 1,
        return_aux: bool = False,
    ) -> Union[
        Tuple[Array[B, S, [T], Y], Array[B, S, [T], Y]],
        Tuple[Array[B, S, [T], Y], Array[B, S, [T], Y], Array[B, T, R]],
    ]:

        # Flatten
        shape_tar = x_tar.shape[1:-1]
        x_ctx    = F.flatten(x_ctx,    start=1, stop=-1)                                            # [batch, context, x_dim]
        y_ctx    = F.flatten(y_ctx,    start=1, stop=-1)                                            # [batch, context, y_dim]
        x_tar    = F.flatten(x_tar,    start=1, stop=-1)                                            # [batch, target,  x_dim]
        mask_ctx = F.flatten(mask_ctx, start=1)                                                     # [batch, context]
        mask_tar = F.flatten(mask_tar, start=1)                                                     # [batch, target]

        # Algorithm
        w_ctx = self._sample_weight(mask_ctx, num_samples)                                          # [batch, sample, context, 1]

        r_i_ctx = self._encode(x_ctx, y_ctx, mask_ctx)                                              # [batch, context, r_dim]
        s_r_i_ctx = F.repeat_axis(r_i_ctx, num_samples, axis=1)                                     # [batch, sample, context, r_dim]
        b_r_i_ctx = s_r_i_ctx * w_ctx                                                               # [batch, sample, context, r_dim]

        s_x_tar = F.repeat_axis(x_tar, num_samples, axis=1)                                         # [batch, sample, target,  x_dim]
        s_x_ctx = F.repeat_axis(x_ctx, num_samples, axis=1)                                         # [batch, sample, context, x_dim]

        b_r_ctx = self._aggregate(s_x_tar, s_x_ctx, b_r_i_ctx, mask_ctx)                            # [batch x sample, target, r_dim]
        mu, sigma = self._decode(s_x_tar, b_r_ctx, mask_tar)                                        # [batch x sample, target, y_dim] x 2

        # Unflatten and mask
        mu    = F.masked_fill(mu,    mask_tar, fill_value=0.,   non_mask_axis=(1, -1))              # [batch, sample, target, y_dim]
        sigma = F.masked_fill(sigma, mask_tar, fill_value=1e-6, non_mask_axis=(1, -1))              # [batch, sample, target, y_dim]
        mu    = F.unflatten(mu,    shape_tar, axis=-2)                                              # [batch, sample, *target, y_dim]
        sigma = F.unflatten(sigma, shape_tar, axis=-2)                                              # [batch, sample, *target, y_dim]

        if return_aux:
            return mu, sigma, (w_ctx,)                                                              # [batch, sample, *target, y_dim] x 2, [batch, sample, context, 1]
        else:
            return mu, sigma                                                                        # [batch, sample, *target, y_dim] x 2

    def log_likelihood(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_samples: int = 1,
        train: bool = False,
        joint: bool = False,
    ) -> Array:

        mu, sigma = self(x_ctx, y_ctx, x_tar, mask_ctx, mask_tar, num_samples=num_samples)          # [batch, sample, *target, y_dim] x 2

        s_y_tar = jnp.expand_dims(y_tar, axis=1)                                                    # [batch, 1,      *target, y_dim]
        log_prob = stats.norm.logpdf(s_y_tar, mu, sigma)                                            # [batch, sample, *target, y_dim]
        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, sample, *target]

        if train and joint:
            axis = [-d for d in range(1, mask_tar.ndim)]
            ll = F.masked_sum(ll, mask_tar, axis=axis, non_mask_axis=1)                             # [batch, sample]
            ll = F.logmeanexp(ll, axis=1)                                                           # [batch]
            ll = jnp.mean(ll / jnp.sum(mask_tar, axis=axis))                                        # (1)
        else:
            ll = F.logmeanexp(ll, axis=1)                                                           # [batch, *target]
            ll = F.masked_mean(ll, mask_tar)                                                        # (1)

        return ll                                                                                   # (1)

    def loss(
        self,
        x_ctx:    Array[B, [C], X],
        y_ctx:    Array[B, [C], Y],
        x_tar:    Array[B, [T], X],
        y_tar:    Array[B, [T], Y],
        mask_ctx: Array[B, [C]],
        mask_tar: Array[B, [T]],
        *,
        num_samples: int = 1,
        train: bool = True,
        joint: bool = False,
        return_aux: bool = False,
    ) -> Array:

        mu, sigma, (w_ctx,) = self(                                                                 # [batch, sample, *target, y_dim] x 2, ([batch, sample, context, 1],)
            x_ctx, y_ctx, x_tar, mask_ctx, mask_tar,
            num_samples=num_samples, return_aux=True,
        )

        s_y_tar = jnp.expand_dims(y_tar, axis=1)                                                    # [batch, 1,      *target, y_dim]
        log_prob = stats.norm.logpdf(s_y_tar, mu, sigma)                                            # [batch, sample, *target, y_dim]

        ll = jnp.sum(log_prob, axis=-1)                                                             # [batch, sample, *target]

        # TODO: Handle the situation where the number of context are different from the number of target
        assert mask_ctx.shape == mask_tar.shape, "Currently, only support context and target from the same array."

        mask_ex_tar = mask_tar & (~mask_ctx)                                                        # [batch, *point (= *context = *target)]
        w_ctx = F.unflatten(w_ctx[..., 0], ll.shape[2:], axis=2)                                    # [batch, sample, *point (= *context)]

        axis = [-d for d in range(1, mask_tar.ndim)]

        if train and joint:
            ll_tar = F.masked_sum(ll,         mask_ex_tar, axis=axis, non_mask_axis=1)              # [batch, sample]
            ll_ctx = F.masked_sum(ll * w_ctx, mask_ctx,    axis=axis, non_mask_axis=1)              # [batch, sample]
            ll = (ll_tar + ll_ctx)                                                                  # [batch, sample]
            ll = F.logmeanexp(ll, axis=1)                                                           # [batch]
        else:
            ll_tar = F.logmeanexp(ll,         axis=1)                                               # [batch, *point]
            ll_ctx = F.logmeanexp(ll * w_ctx, axis=1)                                               # [batch, *point]

            ll_tar = F.masked_sum(ll_tar, mask_ex_tar, axis=axis)                                   # [batch]
            ll_ctx = F.masked_sum(ll_ctx, mask_ctx,    axis=axis)                                   # [batch]

            ll = (ll_tar + ll_ctx) / jnp.sum(mask_tar, axis=axis)                                   # [batch]

        loss = -jnp.mean(ll)                                                                        # (1)

        if return_aux:
            return loss, dict(ll=ll)
        else:
            return loss


class NeuBNPBase(NeuBNPMixin, CNPBase):
    """
    Base class of Neural Bootstrapping Neural Process
    """


class NeuBANPBase(NeuBNPMixin, CANPBase):
    """
    Base class of Neural Bootstrapping Attentive Neural Process
    """


class NeuBNP:
    """
    Neural Bootstrapping Neural Process
    """

    def __new__(
        cls,
        y_dim: int,
        r_dim: int = 128,
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):
        return NeuBNPBase(
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim),
            decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2)),
        )


class NeuBANP:
    """
    Neural Bootstrapping Attentive Neural Process
    """

    def __new__(
        cls,
        y_dim: int,
        r_dim: int = 128,
        sa_heads: Optional[int] = 8,
        ca_heads: Optional[int] = 8,
        transform_qk_dims: Optional[Sequence[int]] = (128, 128, 128, 128, 128),
        encoder_dims: Sequence[int] = (128, 128, 128, 128, 128),
        decoder_dims: Sequence[int] = (128, 128, 128),
    ):

        if sa_heads is not None:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=True)
            self_attention = MultiheadSelfAttention(dim_out=r_dim, num_heads=sa_heads)
        else:
            encoder = MLP(hidden_features=encoder_dims, out_features=r_dim, last_activation=False)
            self_attention = None

        if transform_qk_dims is not None:
            transform_qk = MLP(hidden_features=transform_qk_dims, out_features=r_dim, last_activation=False)
        else:
            transform_qk = None

        cross_attention = MultiheadAttention(dim_out=r_dim, num_heads=ca_heads)
        decoder = MLP(hidden_features=decoder_dims, out_features=(y_dim * 2))

        return NeuBANPBase(
            encoder=encoder,
            self_attention=self_attention,
            transform_qk=transform_qk,
            cross_attention=cross_attention,
            decoder=decoder,
        )
