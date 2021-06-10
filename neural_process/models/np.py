from graph_model import Node

from .base import NeuralProcessBase


from ..modules import (
    MLP,
    NNEncoder,
    NNDecoder,
    Distributor,
    Sampler,
    MeanAggregator,
    DeterministicLatentConcatenator,
    MuSigmaSplitter,
    LogLikelihood,
    KLDivergence,
    LatentLoss,
)


class NeuralProcess(NeuralProcessBase):
    def __init__(
        self,
        latent_encoder,
        distributor,
        decoder,
        loss_function,
        deterministic_path=True,
        deterministic_encoder=None,
    ):
        super().__init__()

        if deterministic_path:
            if deterministic_encoder is None:
                self.common_encoder = latent_encoder
                deterministic_encoder = latent_encoder
            else:
                self.latent_encoder = latent_encoder
                self.deterministic_encoder = deterministic_encoder

            self.concat = DeterministicLatentConcatenator()
        else:
            self.latent_encoder = latent_encoder

        self.distributor = distributor
        self.decoder = decoder
        self.loss_function = loss_function

        self.aggregator = MeanAggregator()
        self.sampler = Sampler()
        self.mu_sigma_extractor = MuSigmaSplitter()
        self.log_likelihood = LogLikelihood()
        self.kl_divergence = KLDivergence()

        nodes = self.get_nodes(latent_encoder, deterministic_path, deterministic_encoder)
        self.build_graph(nodes)

    def get_nodes(self, latent_encoder, deterministic_path, deterministic_encoder):
        latent_nodes = [
            Node(inputs=["x_context", "y_context"],
                 outputs=["s_i_context"],
                 func=latent_encoder),
            Node(inputs=["s_i_context"],
                 outputs=["s_context"],
                 func=self.aggregator),
            Node(inputs=["s_context"],
                 outputs=["q_context"],
                 func=self.distributor),
            Node(inputs=["q_context", "num_latents"],
                 outputs=[("z_samples" if deterministic_path else "representation")],
                 func=self.sampler),   # [batch, latents, z_dim]
        ]

        if deterministic_path:
            deterministic_nodes = [
                Node(inputs=["x_context", "y_context"],
                     outputs=["r_i"],
                     func=deterministic_encoder),
                Node(inputs=["r_i"],
                     outputs=["r"],  # [batch, r_dim]
                     func=self.aggregator),
                Node(inputs=["r", "z_samples"],
                     outputs=["representation"],
                     func=self.concat),
            ]
        else:
            deterministic_nodes = []

        decoder_nodes = [
            Node(inputs=["x_target", "representation"],
                 outputs=["mu_log_sigma"],
                 func=self.decoder),
            Node(inputs=["mu_log_sigma"],
                 outputs=["mu", "sigma"],
                 func=self.mu_sigma_extractor),
        ]

        latent_target_nodes = [
            Node(inputs=["x_target", "y_target"],
                 outputs=["s_i_target"],
                 func=latent_encoder),
            Node(inputs=["s_i_target"],
                 outputs=["s_target"],
                 func=self.aggregator),
            Node(inputs=["s_target"],
                 outputs=["q_target"],
                 func=self.distributor),
        ]

        metric_nodes = [
            Node(inputs=["mu", "sigma", "y_target"],
                 outputs=["log_likelihood"],
                 func=self.log_likelihood),
            Node(inputs=["q_context", "q_target"],
                 outputs=["kl_divergence"],
                 func=self.kl_divergence),
            Node(inputs=["log_likelihood", "kl_divergence", "x_target"],
                 outputs=["loss"],
                 func=self.loss_function),
        ]

        nodes = (
            latent_nodes
            + deterministic_nodes
            + decoder_nodes
            + latent_target_nodes
            + metric_nodes
        )

        return nodes


def np(x_dim, y_dim, h_dim):
    latent_encoder = NNEncoder(MLP(x_dim + y_dim, [h_dim, h_dim], h_dim))
    deterministic_encoder = NNEncoder(MLP(x_dim + y_dim, [h_dim, h_dim], h_dim))
    distributor = Distributor(MLP(h_dim, [h_dim], h_dim + h_dim))
    decoder = NNDecoder(MLP(x_dim + h_dim + h_dim, [h_dim, h_dim], y_dim + y_dim), latent=True)
    loss_function = LatentLoss()

    return NeuralProcess(
        latent_encoder=latent_encoder,
        deterministic_encoder=deterministic_encoder,
        distributor=distributor,
        decoder=decoder,
        loss_function=loss_function,
        deterministic_path=True,
    )
