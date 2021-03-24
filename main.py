import torch

from torch import optim

from torch.distributions.normal import Normal

from torch.utils.data.dataloader import DataLoader

from tqdm import trange

from np import ConditionalNeuralProcess, NeuralProcess

from util import plot_function, get_device
from dataset import get_context_and_target, CosineDataset, CurveDataset



NUM_CONTEXT_RANGE = (5, 10)
NUM_TARGET_RANGE = (3, 10)


if __name__ == "__main__":
    device = get_device("cuda:7")

    print(f"Device: {device}")

    train_dataset = CosineDataset(num_points=20, train=True)
    test_dataset = CosineDataset(num_points=100, train=False)
    # train_dataset = CurveDataset(num_points=20, train=True)
    # test_dataset = CurveDataset(num_points=100, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    # model = ConditionalNeuralProcess(x_dim=1, y_dim=1, r_dim=128,
    #                                  encoder_dims=[128, 128, 128],
    #                                  decoder_dims=[128, 128]).to(device)
    model = NeuralProcess(x_dim=1, y_dim=1, r_dim=64, z_dim=64,
                          deterministic_encoder_dims=[128, 128, 128, 128],
                          latent_encoder_dims=[128, 128, 128, 128],
                          decoder_dims=[128, 128]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    for epoch in trange(int(100000)):
        model.train()

        total_loss = 0.

        for batch_x, batch_y in train_dataloader:
            (context_x, context_y), (target_x, target_y) = \
                get_context_and_target(batch_x, batch_y, NUM_CONTEXT_RANGE, NUM_TARGET_RANGE, device)

            optimizer.zero_grad()

            mu, sigma = model(context_x, context_y, target_x)
            dist = Normal(mu, sigma)
            log_prob = dist.log_prob(target_y)

            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss * batch_x.shape[0]

        if epoch % int(500) == 0:
            print(f"loss: {total_loss.mean():.5f}")
            model.eval()

            batch_x, batch_y = next(iter(test_dataloader))

            (context_x, context_y), (target_x, target_y) = \
                get_context_and_target(batch_x, batch_y, NUM_CONTEXT_RANGE, None, device)

            mu, sigma = model(context_x, context_y, target_x)

            plot_function(f"t.png", context_x, context_y, target_x, target_y, mu, sigma, layout=(2, 6))
