import torch
from torch import nn

from graph_model import Graph


class NeuralProcessBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.graph = None

        self.forward_path = self.not_implemented
        self.likelihood_path = self.not_implemented
        self.loss_path = self.not_implemented

    @staticmethod
    def not_implemented(*args, **kwargs):
        raise NotImplementedError

    def build_graph(self, nodes):
        self.graph = Graph(*nodes)

        # Paths
        self.forward_path = self.graph.build_path(
            inputs=["x_context", "y_context", "x_target", "num_latents"],
            outputs=["mu", "sigma"],
        )

        self.likelihood_path = self.graph.build_path(
            inputs=["x_context", "y_context", "x_target", "y_target", "num_latents"],
            outputs=["log_likelihood"],
        )

        self.loss_path = self.graph.build_path(
            inputs=["x_context", "y_context", "x_target", "y_target", "num_latents"],
            outputs=["loss"],
        )

    def forward(self, x_context, y_context, x_target, num_latents=None):
        return self.forward_path(x_context, y_context, x_target, num_latents)

    def loss(self, x_context, y_context, x_target, y_target, num_latents=None):
        return self.loss_path(x_context, y_context, x_target, y_target, num_latents)

    def predictive_ll(self, x_context, y_context, x_target, y_target, num_latents=None):
        # m = x_context.shape[1]
        n = x_target.shape[1]

        with torch.no_grad():
            log_likelihood = self.likelihood_path(
                x_context, y_context, x_target, y_target, num_latents,
            )

            # predictive_ll = log_likelihood / (n - m)
            predictive_ll = log_likelihood / n

        return predictive_ll
