from torch import nn

from graph_model import Node

from .base import NeuralProcessBase
from ..modules import (
    MLP,
    Discretizer,
    SetConvEncoder,
    CNNDecoder,
    UNet,
    SimpleConv,
    LatentSetConv,
    Distributor,
    Sampler,
    LogLikelihood,
    ConditionalLoss,
)


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


def convnp(y_dim, points_per_unit=64, xl=False):
    if xl:
        cnn1 = UNet()
        cnn2 = UNet()
    else:
        cnn1 = SimpleConv()
        cnn2 = SimpleConv()

    init_length_scale = 2.0 / points_per_unit

    discretizer = Discretizer(points_per_unit, 2 ** cnn1.num_halving_layers)
    encoder = SetConvEncoder(
        in_channels=y_dim,
        out_channels=cnn1.in_channels,
        init_length_scale=init_length_scale,
        activation=nn.Sigmoid(),
    )
    noise_decoder = CNNDecoder(cnn1)
    distributor = Distributor(
        # MLP(cnn1.out_channels, [cnn1.out_channels], cnn2.in_channels + cnn2.in_channels)
        nn.Sequential(
            nn.Linear(cnn1.out_channels, cnn1.out_channels),
            nn.ReLU(),
            nn.Linear(cnn1.out_channels, cnn2.in_channels + cnn2.in_channels),
        )
    )
    decoder = CNNDecoder(cnn2, latent=True)
    mu_set_conv = LatentSetConv(
        in_channels=cnn2.out_channels,
        out_channels=y_dim,
        init_length_scale=init_length_scale,
    )
    sigma_set_conv = LatentSetConv(
        in_channels=cnn2.out_channels,
        out_channels=y_dim,
        init_length_scale=init_length_scale,
        activation=nn.Softplus(),
    )
    loss_function = ConditionalLoss()

    return ConvolutionalNeuralProcess(
        discretizer=discretizer,
        encoder=encoder,
        noise_decoder=noise_decoder,
        distributor=distributor,
        decoder=decoder,
        mu_set_conv=mu_set_conv,
        sigma_set_conv=sigma_set_conv,
        loss_function=loss_function,
    )
