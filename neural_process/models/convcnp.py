from torch import nn

from graph_model import Node

from .base import NeuralProcessBase
from ..modules import (
    Discretizer,
    SetConvEncoder,
    CNNDecoder,
    UNet,
    SimpleConv,
    SetConv,
    LogLikelihood,
    ConditionalLoss,
)


class ConvolutionalConditionalNeuralProcess(NeuralProcessBase):
    def __init__(
        self,
        discretizer,
        encoder,
        decoder,
        mu_set_conv,
        sigma_set_conv,
        loss_function,
    ):
        super().__init__()

        self.discretizer = discretizer
        self.encoder = encoder
        self.decoder = decoder
        self.mu_set_conv = mu_set_conv
        self.sigma_set_conv = sigma_set_conv
        self.loss_function = loss_function

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


def convcnp(y_dim, points_per_unit=64, xl=False):
    if xl:
        cnn = UNet()
    else:
        cnn = SimpleConv()

    init_length_scale = 2.0 / points_per_unit

    discretizer = Discretizer(points_per_unit, 2 ** cnn.num_halving_layers)
    encoder = SetConvEncoder(
        in_channels=y_dim,
        out_channels=cnn.in_channels,
        init_length_scale=init_length_scale,
        activation=nn.Sigmoid(),
    )
    decoder = CNNDecoder(cnn)
    mu_set_conv = SetConv(
        in_channels=cnn.out_channels,
        out_channels=y_dim,
        init_length_scale=init_length_scale,
    )
    sigma_set_conv = SetConv(
        in_channels=cnn.out_channels,
        out_channels=y_dim,
        init_length_scale=init_length_scale,
        activation=nn.Softplus(),
    )
    loss_function = ConditionalLoss()

    return ConvolutionalConditionalNeuralProcess(
        discretizer=discretizer,
        encoder=encoder,
        decoder=decoder,
        mu_set_conv=mu_set_conv,
        sigma_set_conv=sigma_set_conv,
        loss_function=loss_function,
    )
