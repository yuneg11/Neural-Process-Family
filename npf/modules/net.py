from torch import nn


__all__ = [
    "MLP"
]


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        bias=True,
        last_activation=False,
    ):
        layers = []

        for input_dim, output_dim in zip([in_features] + hidden_features, hidden_features):
            layers.extend([
                nn.Linear(input_dim, output_dim, bias=bias),
                nn.ReLU(),
            ])

        input_dim = (hidden_features[-1] if hidden_features else in_features)
        layers.append(nn.Linear(input_dim, out_features, bias=bias))

        if last_activation:
            layers.append(nn.ReLU())

        super().__init__(*layers)

        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 1e-3)
