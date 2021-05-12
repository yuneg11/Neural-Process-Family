from graph_model import Node

from .base import NeuralProcessBase

from .modules.misc import Sampler
from .modules.metrics import LogLikelihood


class ConvolutionalNeuralProcess(NeuralProcessBase):
    def __init__(
        self,
        discretizer,
        encoder,
        noise_decoder,
        distributor,
        decoder,
        mu_set_conv,
        sigma_set_conv,
        loss_function,
    ):
        super().__init__()

        self.discretizer = discretizer
        self.encoder = encoder
        self.noise_decoder = noise_decoder
        self.distributor = distributor
        self.decoder = decoder
        self.mu_set_conv = mu_set_conv
        self.sigma_set_conv = sigma_set_conv
        self.loss_function = loss_function

        self.sampler = Sampler()
        self.log_likelihood = LogLikelihood()

        nodes = self.get_nodes()
        self.build_graph(nodes)

    def get_nodes(self):
        return [
            Node(inputs=["x_context", "x_target"],
                 outputs=["x_grid"],
                 func=self.discretizer),
            Node(inputs=["x_context", "y_context", "x_grid"],
                 outputs=["representation"],
                 func=self.encoder),
            Node(inputs=["representation"],
                 outputs=["f_noise"],
                 func=self.noise_decoder),
            Node(inputs=["f_noise"],
                 outputs=["q_context"],
                 func=self.distributor),
            Node(inputs=["q_context", "num_latents"],
                 outputs=["z_samples"],
                 func=self.sampler),
            Node(inputs=["z_samples"],
                 outputs=["f_representation"],
                 func=self.decoder),
            Node(inputs=["x_grid", "f_representation", "x_target"],
                 outputs=["mu"],
                 func=self.mu_set_conv),
            Node(inputs=["x_grid", "f_representation", "x_target"],
                 outputs=["sigma"],
                 func=self.sigma_set_conv),
            Node(inputs=["mu", "sigma", "y_target"],
                 outputs=["log_likelihood"],
                 func=self.log_likelihood),
            Node(inputs=["log_likelihood", "x_target"],
                 outputs=["loss"],
                 func=self.loss_function),
        ]
