from graph_model import Node

from .base import NeuralProcessBase

from .modules.misc import MeanAggregator, MuSigmaSplitter
from .modules.metrics import LogLikelihood


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
