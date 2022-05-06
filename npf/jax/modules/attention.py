import math

import jax
import jax.numpy as jnp
import flax.linen as nn


__all__ = [
    "MultiheadAttention",
    "MultiheadSelfAttention",
]


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
        return jnp.concatenate(jnp.split(x, self.num_heads, axis=-1), axis=0)

    def gather(self, x):
        return jnp.concatenate(jnp.split(x, self.num_heads, axis=0), axis=-1)

    def attend(self, q, k, v, mask=None):
        q_, k_, v_ = self.scatter(q), self.scatter(k), self.scatter(v)
        A_logits = q_ @ k_.swapaxes(-2, -1) / math.sqrt(self.dim_out)

        if mask is not None:
            mask = jnp.bool_(mask)
            mask = jnp.expand_dims(mask, axis=-2)
            # mask = jnp.tile(mask, (self.num_heads, *[1 for _ in range(1, mask.ndim)]))
            mask = jnp.tile(mask, (self.num_heads, 1, 1))  # This only occurs in the NPF package
            if A_logits.ndim == 4:
                mask = jnp.expand_dims(mask, axis=1)

            A = jax.nn.softmax(jnp.where(mask, A_logits, -float('inf')), axis=-1)
            A = jnp.where(jnp.isnan(A), 0., A)

        else:
            A = jax.nn.softmax(A_logits, axis=-1)

        return self.gather(A @ v_)

    def __call__(self, q, k, v, mask=None):
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        out = self.ln1(q + self.attend(q, k, v, mask))
        out = self.ln2(out + nn.relu(self.fc_out(out)))
        return out


class MultiheadSelfAttention(MultiheadAttention):
    def __call__(self, q, mask=None):
        return super().__call__(q, q, q, mask=mask)
