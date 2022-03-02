import math

import jax
import jax.numpy as jnp
import flax.linen as nn


__all__ = [
    "MultiheadAttention",
    "MultiheadSelfAttention",
]


def masked_fill(mask, a, fill):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))


class MultiheadAttention(nn.Module):
    dim_out: int
    num_heads: int = 8

    def setup(self):
        self.fc_q   = nn.Dense(features=self.dim_out)
        self.fc_k   = nn.Dense(features=self.dim_out)
        self.fc_v   = nn.Dense(features=self.dim_out)
        self.fc_out = nn.Dense(features=self.dim_out)
        self.ln1    = nn.LayerNorm()
        self.ln2    = nn.LayerNorm()

    def scatter(self, x):
        return jnp.concatenate(jnp.split(x, self.num_heads, axis=-1), axis=-3)

    def gather(self, x):
        return jnp.concatenate(jnp.split(x, self.num_heads, axis=-3), axis=-1)

    def attend(self, q, k, v, mask = None):
        q_, k_, v_ = self.scatter(q), self.scatter(k), self.scatter(v)
        A_logits = q_ @ k_.swapaxes(-2,-1) / math.sqrt(self.dim_out)

        if mask is not None:
            mask = jnp.bool_(mask)
            mask = jnp.stack([mask]*q.shape[-2], axis = -2)
            mask = jnp.concatenate([mask]*self.num_heads, axis = -3)
            A = jax.nn.softmax(masked_fill(mask, A_logits, -float('inf')), -1)
            A = masked_fill(jnp.isnan(A), A, 0.0)
        else:
            A = jax.nn.softmax(A_logits, -1)
        return self.gather(A @ v_)

    def __call__(self, q, k, v, mask=None):
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        out = self.ln1(q + self.attend(q, k, v, mask))
        out = self.ln2(out + nn.relu(self.fc_out(out)))
        return out


class MultiheadSelfAttention(MultiheadAttention):
    def __call__(self, q, mask=None):
        return super().__call__(q, q, q, mask=mask)





# class CrossAttnEncoder(nn.Module):
#     dim_x: int = 1
#     dim_y: int = 1
#     dim_hid: int = 128
#     dim_lat: int = None
#     use_self_attn: bool = True
#     v_depth: int = 4
#     qk_depth: int = 2

#     def setup(self):
#         self.use_lat = self.dim_lat is not None
#         if not self.use_self_attn:
#             self.net_v = _MLP(
#                 dim_hid = self.dim_hid,
#                 dim_out = self.dim_hid,
#                 depth   = self.v_depth,
#             )
#         else:
#             self.net_v = _MLP(
#                 dim_hid = self.dim_hid,
#                 dim_out = self.dim_hid,
#                 depth   = self.v_depth - 2,
#             )
#             self.self_attn = SelfAttn(dim_out=self.dim_hid)

#         self.net_qk = MLP(
#             dim_hid = self.dim_hid,
#             dim_out = self.dim_hid,
#             depth   = self.qk_depth,
#         )
#         if self.use_lat:
#             self.attn = MultiHeadAttn(dim_out=2*self.dim_lat)
#         else:
#             self.attn = MultiHeadAttn(dim_out=self.dim_hid)

#     def __call__(self, xc, yc, xt, mask = None, **kwargs):
#         q, k = self.net_qk(xt, **kwargs), self.net_qk(xc, **kwargs)
#         v = self.net_v(jnp.concatenate([xc, yc], axis=-1), **kwargs)

#         if self.use_self_attn:
#             v = self.self_attn(v, mask=mask, **kwargs)

#         out = self.attn(q, k , v, mask=mask, **kwargs)
#         if self.use_lat:
#             mu, sigma = jnp.split(out, 2, axis=-1)
#             sigma = 0.1 + 0.9 * nn.sigmoid(sigma)
#             return mu, sigma
#         else:
#             return out


# class NeuCrossAttnEncoder(nn.Module):
#     dim_hid: int = 128
#     dim_lat: int = None
#     use_self_attn: bool = True
#     v_depth: int = 4
#     qk_depth: int = 2

#     def setup(self):
#         self.use_lat = self.dim_lat is not None
#         if not self.use_self_attn:
#             self.net_v = _MLP(dim_hid = self.dim_hid, dim_out = self.dim_hid, depth = self.v_depth)
#         else:
#             self.net_v = _MLP(dim_hid = self.dim_hid, dim_out = self.dim_hid, depth = self.v_depth-2)
#             self.self_attn = SelfAttn(dim_out = self.dim_hid)

#         self.net_qk = _MLP(dim_hid = self.dim_hid, dim_out = self.dim_hid, depth = self.qk_depth)

#         if self.use_lat:
#             self.attn = MultiHeadAttn(dim_out = 2*self.dim_lat)
#         else:
#             self.attn = MultiHeadAttn(dim_out = self.dim_hid)

#     def __call__(self, xc, yc, xt, w, mask = None, **kwargs):
#         q, k = self.net_qk(xt, **kwargs), self.net_qk(xc, **kwargs)
#         v = self.net_v(jnp.concatenate([xc, yc], axis = -1), **kwargs)

#         if self.use_self_attn:
#             v = self.self_attn(v, mask = mask, **kwargs)

#         v = v * w
#         out = self.attn(q, k, v, mask = mask, **kwargs)
#         if self.use_lat:
#             mu, sigma = jnp.split(out, 2, axis=-1)
#             sigma = 0.1 + 0.9 * nn.sigmoid(sigma)
#             return mu, sigma
#         else:
#             return out
