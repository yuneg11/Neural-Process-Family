from torch import nn


class MLP(nn.Sequential):
    def __init__(self, input_dim, hidden_dims, output_dim, last_relu=False):
        super().__init__()

        for i, next_dim in enumerate(hidden_dims):
            self.add_module(f'dense{i}', nn.Linear(input_dim, next_dim))
            self.add_module(f'relu{i}', nn.ReLU())
            input_dim = next_dim
        self.add_module(f'dense{len(hidden_dims)}', nn.Linear(input_dim, output_dim))

        if last_relu:
            self.add_module(f'relu{len(hidden_dims)}', nn.ReLU())


class PointwiseMLP(MLP):
    def forward(self, input):
        batch_size, num_points, input_dim = input.shape

        point_wise_input = input.reshape(batch_size * num_points, input_dim)
        point_wise_output = super().forward(point_wise_input)
        output = point_wise_output.reshape(batch_size, num_points, -1)

        return output
