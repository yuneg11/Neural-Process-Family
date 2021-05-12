import numpy as np

import torch
from torch import nn


# class BaseAttender(nn.Module):
#     def __init__(self, query_net, key_net, value_net):
#         super().__init__()
#
#         self.query_net = query_net
#         self.key_net = key_net
#         self.value_net = value_net
#
#     def forward(self, r_i, x_context, x_target):
#         value = self.value_net(r_i)
#         key = self.key_net(x_context)
#         query = self.query_net(x_target)
#         # TODO
#
#     # TODO: calculate_weight
#
#
# class DotProductAttender(BaseAttender):
#     def calculate_weight(self, query, key):
#         dk = key.shape[-1]
#         logit = torch.bmm(query, key.permute(0, 2, 1)) / np.sqrt(dk)
#         weight = nn.functional.softmax(logit, dim=-1)
#         return weight


"""
    Attention modules for AttnCNP
"""

from .networks import BatchMLP, BatchLinear


class DotProdAttention(nn.Module):
    """
    Simple dot-product attention module. Can be used multiple times for
    multi-head attention.

    Args:
        embedding_dim (int): Dimensionality of embedding for keys and queries.
        values_dim (int): Dimensionality of embedding for values.
        linear_transform (bool, optional): Use a linear for all embeddings
            before operation. Defaults to `False`.
    """

    def __init__(self, embedding_dim, values_dim, linear_transform=False):
        super(DotProdAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.values_dim = values_dim
        self.linear_transform = linear_transform

        if self.linear_transform:
            self.key_transform = BatchLinear(self.embedding_dim,
                                             self.embedding_dim, bias=False)
            self.query_transform = BatchLinear(self.embedding_dim,
                                               self.embedding_dim, bias=False)
            self.value_transform = BatchLinear(self.values_dim,
                                               self.values_dim, bias=False)

    def forward(self, keys, queries, values):
        """Forward pass to implement dot-product attention. Assumes that
        everything is in batch mode.

        Args:
            keys (tensor): Keys of shape
                `(num_functions, num_keys, dim_key)`.
            queries (tensor): Queries of shape
                `(num_functions, num_queries, dim_query)`.
            values (tensor): Values of shape
                `(num_functions, num_values, dim_value)`.

        Returns:
            tensor: Output of shape `(num_functions, num_queries, dim_value)`.
        """
        if self.linear_transform:
            keys = self.key_transform(keys)
            queries = self.query_transform(queries)
            values = self.value_transform(values)

        dk = keys.shape[-1]
        attn_logit = torch.bmm(queries, keys.permute(0, 2, 1)) / np.sqrt(dk)
        attn_weights = nn.functional.softmax(attn_logit, dim=-1)
        return torch.bmm(attn_weights, values)


class MultiHeadAttention(nn.Module):
    """Implementation of multi-head attention in a batch way. Wraps around the
    dot-product attention module.

    Args:
        embedding_dim (int): Dimensionality of embedding for keys, values,
            queries.
        value_dim (int): Dimensionality of values representation. Is same as
            above.
        num_heads (int): Number of dot-product attention heads in module.
    """

    def __init__(self,
                 embedding_dim,
                 value_dim,
                 num_heads):
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.head_size = self.embedding_dim // self.num_heads

        self.key_transform = BatchLinear(self.embedding_dim, self.embedding_dim,
                                         bias=False)
        self.query_transform = BatchLinear(self.embedding_dim,
                                           self.embedding_dim, bias=False)
        self.value_transform = BatchLinear(self.embedding_dim,
                                           self.embedding_dim, bias=False)
        self.attention = DotProdAttention(embedding_dim=self.embedding_dim,
                                          values_dim=self.embedding_dim,
                                          linear_transform=False)
        self.head_combine = BatchLinear(self.embedding_dim, self.embedding_dim)

    def forward(self, keys, queries, values):
        """Forward pass through multi-head attention module.

        Args:
            keys (tensor): Keys of shape
                `(num_functions, num_keys, dim_key)`.
            queries (tensor): Queries of shape
                `(num_functions, num_queries, dim_query)`.
            values (tensor): Values of shape
                `(num_functions, num_values, dim_value)`.

        Returns:
            tensor: Output of shape `(num_functions, num_queries, dim_value)`.
        """
        keys = self.key_transform(keys)
        queries = self.query_transform(queries)
        values = self.value_transform(values)

        # Reshape keys, queries, values into shape
        #     (batch_size * n_heads, num_points, head_size).
        keys = self._reshape_objects(keys)
        queries = self._reshape_objects(queries)
        values = self._reshape_objects(values)

        # Compute attention mechanism, reshape, process, and return.
        attn = self.attention(keys, queries, values)
        attn = self._concat_head_outputs(attn)
        return self.head_combine(attn)

    def _reshape_objects(self, o):
        num_functions = o.shape[0]
        o = o.view(num_functions, -1, self.num_heads, self.head_size)
        o = o.permute(2, 0, 1, 3).contiguous()
        return o.view(num_functions * self.num_heads, -1, self.head_size)

    def _concat_head_outputs(self, attn):
        num_functions = attn.shape[0] // self.num_heads
        attn = attn.view(self.num_heads, num_functions, -1, self.head_size)
        attn = attn.permute(1, 2, 0, 3).contiguous()
        return attn.view(num_functions, -1, self.num_heads * self.head_size)


class CrossAttention(nn.Module):
    """Module for transformer-style cross attention to be used by the AttnCNP.

    Args:
        input_dim (int, optional): Dimensionality of the input locations.
            Defaults to `1`.
        embedding_dim (int, optional): Dimensionality of the embeddings (keys).
            Defaults to `128`.
        values_dim (int, optional): Dimensionality of the embeddings (values).
            Defaults to `128`.
        num_heads (int, optional): Number of attention heads to use. Defaults
            to `8`.
    """

    def __init__(self,
                 input_dim=1,
                 embedding_dim=128,
                 values_dim=128,
                 num_heads=8):
        super(CrossAttention, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.values_dim = values_dim
        self.num_heads = num_heads

        self._attention = MultiHeadAttention(embedding_dim=self.embedding_dim,
                                             value_dim=self.values_dim,
                                             num_heads=self.num_heads)
        self.embedding = BatchMLP(in_features=self.input_dim,
                                  out_features=self.embedding_dim)

        # Additional modules for transformer-style computations:
        self.ln1 = nn.LayerNorm(self.embedding_dim)
        self.ln2 = nn.LayerNorm(self.embedding_dim)
        self.ff = BatchLinear(self.embedding_dim, self.embedding_dim)

    def forward(self, h, x_context, x_target):
        """Forward pass through the cross-attentional mechanism.

        Args:
            h (tensor): Embeddings for context points of shape
                `(batch, num_context, embedding_dim)`.
            x_context (tensor): Context locations of shape
                `(batch, num_context, input_dim)`.
            x_target (tensor): Target locations of shape
                `(batch, num_target, input_dim)`.

        Returns:
            tensor: Result of forward pass.
        """
        keys = self.embedding(x_context)
        queries = self.embedding(x_target)
        attn = self._attention(keys, queries, h)
        out = self.ln1(attn + queries)
        return self.ln2(out + self.ff(out))
