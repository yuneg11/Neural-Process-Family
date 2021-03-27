import sys

import torch
from torch import optim

from tqdm import trange, tqdm

from np import ConditionalNeuralProcess, NeuralProcess

from util import get_device
from plot import plot_function, plot_image

import dataset


# NUM_CONTEXT_RANGE = (30, 100)
# NUM_TARGET_RANGE = (100, 300)
NUM_CONTEXT_RANGE = (5, 10)
NUM_TARGET_RANGE = (5, 10)


if __name__ == "__main__":
    user_device = sys.argv[1] if len(sys.argv) > 1 else None



    device = get_device(user_device)

    print(f"Device: {device}")

    # train_loader = dataset.sine(train=True)
    # test_loader = dataset.sine(train=False)
    train_loader = dataset.celeba("~/data/", train=True)
    test_loader = dataset.celeba("~/data/", train=False)

    # model = ConditionalNeuralProcess(x_dim=1, y_dim=1, r_dim=128,
    #                                  encoder_dims=[128, 128, 128],
    #                                  decoder_dims=[128, 128]).to(device)
    # model = ConditionalNeuralProcess(x_dim=2, y_dim=3, r_dim=512,
    #                                  encoder_dims=[512, 512, 512, 512, 512],
    #                                  decoder_dims=[512, 512, 512]).to(device)
    # model = NeuralProcess(x_dim=1, y_dim=1, r_dim=64, z_dim=64,
    #                       deterministic_encoder_dims=[128, 128, 128],
    #                       latent_encoder_dims=[128, 128, 128],
    #                       decoder_dims=[128, 128]).to(device)
    model = NeuralProcess(x_dim=2, y_dim=3, r_dim=256, z_dim=256,
                          deterministic_encoder_dims=[1024, 1024, 1024, 1024, 1024],
                          latent_encoder_dims=[1024, 1024, 1024, 1024, 1024],
                          decoder_dims=[1024, 1024, 1024]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    t = trange(int(100000))
    for epoch in t:
        model.train()

        total_loss = 0.

        t2 = tqdm(train_loader)
        # t2 = train_loader
        for batch_x, batch_y in t2:
            (context_x, context_y), (target_x, target_y) = \
                dataset.get_context_and_target(batch_x, batch_y, NUM_CONTEXT_RANGE, NUM_TARGET_RANGE, device)

            optimizer.zero_grad()

            mu, sigma, loss = model(context_x, context_y, target_x, target_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach() * batch_x.shape[0]

            t2.set_description(f"Loss: {loss.detach():.2f}")

        if epoch % int(10) == 0 :#or True:
            t.set_description(f"Loss: {total_loss / len(t2):.2f}")

        if epoch % int(500) == 0 and epoch > 0 or True:
            model.eval()

            batch_x, batch_y = next(iter(test_loader))

            (context_x, context_y), (target_x, target_y) = \
                dataset.get_context_and_target(batch_x, batch_y, NUM_CONTEXT_RANGE, None, device)

            mu, sigma = model(context_x, context_y, target_x)


            # plot_function(f"np-1d/{epoch}.png", context_x, context_y, target_x, target_y, mu, sigma, layout=(2, 6))

            # torch.save((context_x, context_y, target_x, target_y, mu, sigma, model), f"r4/{epoch}{total_loss / len(t2):.0f}.pt")
            plot_image(f"t.png", context_x, context_y, target_x, target_y, mu, sigma)
            # plot_image(f"r4/{epoch}.png", context_x, context_y, target_x, target_y, mu, sigma)
