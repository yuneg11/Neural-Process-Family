from typing import Optional, List
from torchtyping import TensorType
from ..type import *

from torch import nn

from .net import MLP


__all__ = [
    "MultiheadAttention",
    "MultiheadCrossAttention",
    "MultiheadSelfAttention",
]


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, num_heads: int, qk_dim: int, v_dim: int, **kwargs):
        super().__init__(
            num_heads=num_heads,
            embed_dim=qk_dim,
            kdim=qk_dim,
            vdim=v_dim,
            batch_first=True,
            **kwargs,
        )

    def forward(self,
        query: TensorType[B, T, Q],
        key:   TensorType[B, S, K],
        value: TensorType[B, S, V],
    ) -> TensorType[B, T, V]:

        return super().forward(query, key, value, need_weights=False)[0]


class MultiheadSelfAttention(MultiheadAttention):
    def __init__(self, num_heads: int, input_dim: int, **kwargs):
        super().__init__(num_heads, input_dim, input_dim, **kwargs)

    def forward(self,
        input: TensorType[B, T, "input_dim"],
    ) -> TensorType[B, T, "input_dim"]:

        return super().forward(input, input, input)


class MultiheadCrossAttention(MultiheadAttention):
    def __init__(self,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            qk_dim=v_dim,
            v_dim=v_dim,
            **kwargs,
        )

        self.qk_transform = MLP(qk_dim, [v_dim], v_dim)
        self.v_transform = nn.Linear(v_dim, v_dim)
        self.ln1 = nn.LayerNorm(v_dim)
        self.ln2 = nn.LayerNorm(v_dim)

        nn.init.xavier_normal_(self.v_transform.weight, gain=1)
        nn.init.constant_(self.v_transform.bias, 0.0)

    def forward(self,
        query: TensorType[B, T, Q],
        key:   TensorType[B, S, K],
        value: TensorType[B, S, V],
    ) -> TensorType[B, T, V]:

        query = self.qk_transform(query)
        key   = self.qk_transform(key)

        attn = super().forward(query, key, value)
        value = self.ln1(attn + query)
        value = self.ln2(value + self.v_transform(value))

        return value
