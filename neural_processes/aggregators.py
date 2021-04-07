import torch
from torch import nn

from .mlps import MLP


class MeanAggregator(nn.Module):
    def forward(self, x_context, x_target, r_i):
        return r_i.mean(dim=1)


class AttentionBase(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        self.q_linear = MLP(input_dim, hidden_dims, output_dim)
        self.k_linear = MLP(input_dim, hidden_dims, output_dim)

    def attention(self, q, k, v):
        raise NotImplementedError

    def forward(self, x_context, x_target, r_i):
        q = self.q_linear(x_target)
        k = self.k_linear(x_context)

        w_r_i = self.attention(q, k, r_i)

        return w_r_i


class DotProductAttention(AttentionBase):
    def attention(self, q, k, v):
        scale = torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float)).to(k.device)

        w = torch.bmm(q, k.transpose(2, 1)) / scale  # b m n
        w = torch.softmax(w, dim=1)  # b m n
        # w = torch.sigmoid(w)   # b m n
        w_v = torch.bmm(w, v)  # b m r_d

        return w_v
