from graph_model import Node

from .base import NeuralProcessBase
from ..modules import (
    MLP,
    NNEncoder,
    NNDecoder,
    MeanAggregator,
    MuSigmaSplitter,
    LogLikelihood,
    ConditionalLoss,
)


class ConditionalNeuralProcess(NeuralProcessBase):
    def __init__(
        self,
        encoder,
        decoder,
        loss_function,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss_function = loss_function

        self.aggregator = MeanAggregator()
        self.mu_sigma_extractor = MuSigmaSplitter()
        self.log_likelihood = LogLikelihood()

        nodes = self.get_nodes()
        self.build_graph(nodes)

    def get_nodes(self):
        return [
            Node(inputs=["x_context", "y_context"],
                 outputs=["r_i"],
                 func=self.encoder),
            Node(inputs=["r_i"],
                 outputs=["representation"],
                 func=self.aggregator),
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


def cnp(x_dim, y_dim, h_dim):
    encoder = NNEncoder(MLP(x_dim + y_dim, [h_dim, h_dim], h_dim))
    decoder = NNDecoder(MLP(x_dim + h_dim, [h_dim, h_dim], y_dim + y_dim))
    loss_function = ConditionalLoss()

    return ConditionalNeuralProcess(
        encoder=encoder,
        decoder=decoder,
        loss_function=loss_function,
    )
