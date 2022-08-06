from ..typing import *

from jax import numpy as jnp
from jax import random
from flax import linen as nn

from .cnp import CNPBase
from .canp import CANPBase
from .. import functional as F
from ..data import NPData
from ..utils import npf_io, MultivariateNormalDiag
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
    @npf_io(flatten=True)
    def __call__(
        self,
        data: NPData,
        *,
        num_samples: int = 1,
        return_aux: bool = False,
    ) -> Union[
        Tuple[Array[B, S, [T], Y], Array[B, S, [T], Y]],
        Tuple[Array[B, S, [T], Y], Array[B, S, [T], Y], Array[B, T, R]],
    ]:

        w_ctx = self._sample_weight(data.mask_ctx, num_samples)                                     # [batch, sample, context, 1]

        r_i_ctx = self._encode(data.x_ctx, data.y_ctx, data.mask_ctx)                               # [batch, context, r_dim]
        s_r_i_ctx = F.repeat_axis(r_i_ctx, num_samples, axis=1)                                     # [batch, sample, context, r_dim]
        b_r_i_ctx = s_r_i_ctx * w_ctx                                                               # [batch, sample, context, r_dim]

        s_x     = F.repeat_axis(data.x,     num_samples, axis=1)                                    # [batch, sample, point,   x_dim]
        s_x_ctx = F.repeat_axis(data.x_ctx, num_samples, axis=1)                                    # [batch, sample, context, x_dim]
        b_r_ctx = self._aggregate(s_x, s_x_ctx, b_r_i_ctx, data.mask_ctx)                           # [batch, sample, point,   r_dim]

        query = jnp.concatenate((s_x, b_r_ctx), axis=-1)                                            # [batch, sample, point, x_dim + r_dim]
        mu, sigma = self._decode(query, data.mask)                                                  # [batch, sample, point, y_dim] x 2

        # Mask
        mu    = F.masked_fill(mu,    data.mask, fill_value=0., non_mask_axis=(1, -1))               # [batch, sample, point, y_dim]
        sigma = F.masked_fill(sigma, data.mask, fill_value=0., non_mask_axis=(1, -1))               # [batch, sample, point, y_dim]

        if return_aux:
            w_ctx = jnp.squeeze(w_ctx, axis=-1)                                                     # [batch, sample, point (= context)]
            return mu, sigma, (w_ctx,)                                                              # [batch, sample, point, y_dim] x 2, (aux)
        else:
            return mu, sigma                                                                        # [batch, sample, point, y_dim] x 2

    @npf_io(flatten_input=True)
    def log_likelihood(
        self,
        data: NPData,
        *,
        num_samples: int = 1,
        joint: bool = False,
        split_set: bool = False,
    ) -> Array:

        mu, sigma = self(data, num_samples=num_samples, skip_io=True)                               # [batch, sample, point, y_dim] x 2, (aux)

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      point, y_dim]
        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(s_y)                                  # [batch, sample, point]

        if joint:
            ll = F.masked_sum(log_prob, data.mask, axis=-1, non_mask_axis=1)                        # [batch, sample]
            ll = F.logmeanexp(ll, axis=1) / jnp.sum(data.mask, axis=-1)                             # [batch]

            if split_set:
                ll_ctx = F.masked_sum(log_prob, data.mask_ctx, axis=-1, non_mask_axis=1)            # [batch, sample]
                ll_tar = F.masked_sum(log_prob, data.mask_tar, axis=-1, non_mask_axis=1)            # [batch, sample]
                ll_ctx = F.logmeanexp(ll_ctx, axis=1) / jnp.sum(data.mask_ctx, axis=-1)             # [batch]
                ll_tar = F.logmeanexp(ll_tar, axis=1) / jnp.sum(data.mask_tar, axis=-1)             # [batch]

        else:
            ll_all = F.logmeanexp(log_prob, axis=1)                                                 # [batch, point]
            ll = F.masked_mean(ll_all, data.mask, axis=-1)                                          # [batch]

            if split_set:
                ll_ctx = F.masked_mean(ll_all, data.mask_ctx, axis=-1)                              # [batch]
                ll_tar = F.masked_mean(ll_all, data.mask_tar, axis=-1)                              # [batch]

        ll = jnp.mean(ll)                                                                           # (1)

        if split_set:
            ll_ctx = jnp.mean(ll_ctx)                                                               # (1)
            ll_tar = jnp.mean(ll_tar)                                                               # (1)

            return ll, ll_ctx, ll_tar                                                               # (1) x 3
        else:
            return ll                                                                               # (1)

    @npf_io(flatten_input=True)
    def loss(
        self,
        data: NPData,
        *,
        num_samples: int = 1,
        joint: bool = True,
    ) -> Array:

        mu, sigma, (w_ctx,) = self(data, num_samples=num_samples, return_aux=True, skip_io=True)    # [batch, sample, point, y_dim] x 2, ([batch, sample, point],)

        s_y = jnp.expand_dims(data.y, axis=1)                                                       # [batch, 1,      point, y_dim]
        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(s_y)                                  # [batch, sample, point]
        log_prob_w = log_prob * w_ctx                                                               # [batch, sample, point]

        if joint:
            ll_ctx = F.masked_sum(log_prob_w, data.mask_ctx, axis=-1, non_mask_axis=1)              # [batch, sample]
            ll_tar = F.masked_sum(log_prob,   data.mask_tar, axis=-1, non_mask_axis=1)              # [batch, sample]
            ll_ctx = F.logmeanexp(ll_ctx, axis=1)                                                   # [batch]
            ll_tar = F.logmeanexp(ll_tar, axis=1)                                                   # [batch]

        else:
            ll_ctx = F.logmeanexp(log_prob_w, axis=1)                                               # [batch, point]
            ll_tar = F.logmeanexp(log_prob,   axis=1)                                               # [batch, point]
            ll_ctx = F.masked_sum(ll_ctx, data.mask_ctx, axis=-1)                                   # [batch]
            ll_tar = F.masked_sum(ll_tar, data.mask_tar, axis=-1)                                   # [batch]

        ll = (ll_ctx + ll_tar) / jnp.sum(data.mask, axis=-1)                                        # [batch]
        ll = jnp.mean(ll)                                                                           # (1)
        loss = -ll                                                                                  # (1)

        return loss                                                                                 # (1)


class NeuBNPBase(NeuBNPMixin, CNPBase):
    """
    Base class of Neural Bootstrapping Neural Process
    """
    pass


class NeuBANPBase(NeuBNPMixin, CANPBase):
    """
    Base class of Neural Bootstrapping Attentive Neural Process
    """
    pass


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
        min_sigma: float = 0.1,
    ):
        return NeuBNPBase(
            encoder=MLP(hidden_features=encoder_dims, out_features=r_dim),
            decoder=MLP(hidden_features=decoder_dims, out_features=(y_dim * 2)),
            min_sigma=min_sigma,
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
        min_sigma: float = 0.1,
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
            min_sigma=min_sigma,
        )
