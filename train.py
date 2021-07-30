import argparse

import numpy as np
import stheno.torch as stheno
import torch

from neural_process.utils import device
from neural_process.utils import data
from neural_process.utils.experiment import (
    report_loss,
    generate_root,
    WorkingDirectory,
    save_checkpoint
)


from neural_process import models


def validate(data, model, report_freq=None):
    """Compute the validation loss."""
    model.eval()
    likelihoods = []
    with torch.no_grad():
        for step, task in enumerate(data):
            predictive_ll = model.predictive_ll(
                task['x_context'], task['y_context'],
                task['x_target'], task['y_target'], num_latents=20)
            likelihoods.append(predictive_ll.item())

            if report_freq:
                avg_ll = np.array(likelihoods).mean()
                report_loss('Validation', avg_ll, step, report_freq)
    avg_ll = np.array(likelihoods).mean()
    return avg_ll


def train(data, model, opt, report_freq):
    """Perform a training epoch."""
    model.train()
    losses = []
    for step, task in enumerate(data):
        obj = model.loss(task['x_context'], task['y_context'],
                         task['x_target'], task['y_target'], num_latents=20)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        # Track training progress
        losses.append(obj.item())
        avg_loss = np.array(losses).mean()
        report_loss('Training', avg_loss, step, report_freq)
    return avg_loss


# Parse arguments given to the script.
parser = argparse.ArgumentParser()
parser.add_argument('data',
                    choices=['eq', 'matern', 'noisy-mixture', 'weakly-periodic', 'sawtooth'],
                    help='Data set to train the CNP on. ')
parser.add_argument('model',
                    choices=['convcnp', 'convcnpxl', 'convnp', 'convnpxl', 'cnp', 'attncnp', 'np', 'attnnp', 'gnp'],
                    help='Choice of model. ')
parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')
parser.add_argument('--train',
                    action='store_true',
                    help='Perform training. If this is not specified, '
                         'the model will be attempted to be loaded from the '
                         'experiment root.')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    help='Number of epochs to train for.')
parser.add_argument('--learning_rate',
                    default=1e-3,
                    type=float,
                    help='Learning rate.')
parser.add_argument('--weight_decay',
                    default=1e-5,
                    type=float,
                    help='Weight decay.')
args = parser.parse_args()

# Load working directory.
if args.root:
    wd = WorkingDirectory(root=args.root)
else:
    experiment_name = f'{args.model}-{args.data}'
    wd = WorkingDirectory(root=generate_root(experiment_name))

# Load data generator.
if args.data == 'sawtooth':
    gen = data.SawtoothGenerator()
    gen_val = data.SawtoothGenerator(num_tasks=60)
    gen_test = data.SawtoothGenerator(num_tasks=2048)
else:
    if args.data == 'eq':
        kernel = stheno.EQ().stretch(0.25)
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(0.25)
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(1.) + \
                 stheno.EQ().stretch(.25) + \
                 0.001 * stheno.Delta()
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(0.5) * stheno.EQ().periodic(period=0.25)
    else:
        raise ValueError(f'Unknown data "{args.data}".')

    gen = data.GPGenerator(kernel=kernel, num_tasks=256)
    gen_val = data.GPGenerator(kernel=kernel, num_tasks=60)
    gen_test = data.GPGenerator(kernel=kernel, num_tasks=2048)

# Load model.
if args.model == 'convcnp':
    model = models.convcnp(y_dim=1, xl=False)
elif args.model == 'convcnpxl':
    model = models.convcnp(y_dim=1, xl=True)
elif args.model == 'convnp':
    model = models.convnp(y_dim=1, xl=False)
elif args.model == 'convnpxl':
    model = models.convnp(y_dim=1, xl=True)
elif args.model == 'cnp':
    model = models.cnp(x_dim=1, y_dim=1, h_dim=128)
elif args.model == 'np':
    model = models.np(x_dim=1, y_dim=1, h_dim=128)
elif args.model == 'attncnp':
    model = models.attncnp(x_dim=1, y_dim=1, h_dim=128)
elif args.model == 'attnnp':
    model = models.attnnp(x_dim=1, y_dim=1, h_dim=128)
elif args.model == 'gnp':
    model = models.GNP()
    if args.data != "sawtooth":
        kernel = kernel + 0.05 ** 2 * stheno.Delta()
else:
    raise ValueError(f'Unknown model {args.model}.')

model.to(device)

# Perform training.
if args.train:
    opt = torch.optim.Adam(model.parameters(),
                           args.learning_rate,
                           weight_decay=args.weight_decay)

    # Run the training loop, maintaining the best objective value.
    best_obj = -np.inf
    for epoch in range(args.epochs):
        print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

        # Compute training objective.
        train_obj = train(gen, model, opt, report_freq=50)
        report_loss('Training', train_obj, 'epoch')

        # Compute validation objective.
        val_obj = validate(gen_val, model, report_freq=20)
        report_loss('Validation', val_obj, 'epoch')

        # Update the best objective value and checkpoint the model.
        is_best = False
        if val_obj > best_obj:
            best_obj = val_obj
            is_best = True
        save_checkpoint(wd,
                        {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc_top1': best_obj,
                         'optimizer': opt.state_dict()},
                        is_best=is_best)

else:
    # Load saved model.
    load_dict = torch.load(wd.file('model_best.pth.tar', exists=True))
    model.load_state_dict(load_dict['state_dict'])

# Finally, test model on ~2000 tasks.
test_obj = validate(gen_test, model)
print('Model averages a log-likelihood of %s on unseen tasks.' % test_obj)
with open(wd.file('test_log_likelihood.txt'), 'w') as f:
    f.write(str(test_obj))
