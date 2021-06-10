from graph_model import Node

from .base import NeuralProcessBase
from ..modules import (
    MLP,
    NNEncoder,
    NNDecoder,
    CrossAttention,
    MuSigmaSplitter,
    LogLikelihood,
    ConditionalLoss,
)


class AttentiveConditionalNeuralProcess(NeuralProcessBase):
    def __init__(
        self,
        encoder,
        cross_attender,
        decoder,
        loss_function,
        self_attender=None,
    ):
        super().__init__()

        self.encoder = encoder

        if self_attender is not None:
            self.self_attender = self_attender
            self_attention = True
        else:
            self_attention = False

        self.cross_attender = cross_attender
        self.decoder = decoder
        self.loss_function = loss_function

        self.mu_sigma_extractor = MuSigmaSplitter()
        self.log_likelihood = LogLikelihood()

        nodes = self.get_nodes(self_attention)
        self.build_graph(nodes)

    def get_nodes(self, self_attention):
        nodes = [
            Node(inputs=["x_context", "y_context"],
                 outputs=[("r_i_pre" if self_attention else "r_i")],
                 func=self.encoder),
            Node(inputs=["r_i", "x_context", "x_target"],
                 outputs=["representation"],
                 func=self.cross_attender),
            Node(inputs=["x_target", "representation"],
                 outputs=["mu_log_sigma"],
                 func=self.decoder),
            Node(inputs=["mu_log_sigma"],
                 outputs=["mu", "sigma"],
                 func=self.mu_sigma_extractor),
            Node(inputs=["mu", "sigma", "y_target"],
                 outputs=["log_likelihood"],
                 func=self.log_likelihood),
            Node(inputs=["log_likelihood", "x_target"],
                 outputs=["loss"],
                 func=self.loss_function),
        ]

        if self_attention:
            nodes.append(
                Node(inputs=["r_i_pre"],
                     outputs=["r_i"],
                     func=self.self_attender)
            )

        return nodes


def attncnp(x_dim, y_dim, h_dim):
    encoder = NNEncoder(MLP(x_dim + y_dim, [h_dim, h_dim], h_dim))
    cross_attender = CrossAttention(
        input_dim=x_dim,
        embedding_dim=h_dim,
        values_dim=h_dim,
        num_heads=8,
    )
    decoder = NNDecoder(MLP(x_dim + h_dim, [h_dim, h_dim], y_dim + y_dim))
    loss_function = ConditionalLoss()

    return AttentiveConditionalNeuralProcess(
        encoder=encoder,
        cross_attender=cross_attender,
        decoder=decoder,
        loss_function=loss_function,
    )
