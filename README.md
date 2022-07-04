# Neural Process Family

**This library is in the early stages of development.**

**PyTorch versions are not yet supported.**


## Installation

You can choose between the following installation methods:

### NPF as a library

```bash
# From PyPI (recommended)
pip install np-family

# Latest version (from current branch; dev-v4)
pip install git+https://github.com/yuneg11/Neural-Process-Family@dev-v4

# Specific release (from tag; 0.0.1.dev0)
pip install git+https://github.com/yuneg11/Neural-Process-Family@0.0.1.dev0
```

Then, you can use the library as follows:

```python
from npf.jax.models import CNP

cnp = CNP(y_dim=1)
```

You should handle other logics (include train, evaluation, etc...)

### NPF as an experiment framework

```bash
# Dependencies
pip install rich nxcl==0.0.3.dev3

## And ML frameworks (JAX, PyTorch)
# ex) pip install jax

# Clone the repository
git clone https://github.com/yuneg11/Neural-Process-Family npf
cd npf
```

Then, you can run the experiment, for example:

```bash
python scripts/jax/train.py -f configs/gp/rbf/inf/anp.yaml -lr 0.0001 --model.train_kwargs.num_latents 30
```

The output will be saved under `outs/` directory.
Details will be added in the future.

## Download or build datasets

```bash
python -m npf.jax.data.save \
    --root <dataset-root> \
    --dataset <dataset-name>
```

- `<dataset-root>`: The root path to save dataset. Default: `./datasets/`
- `<dataset-name>`: The name of the dataset to save. See below sections for available datasets.

### Image datasets

You should install `torch` and `torchvision` to download the datastes.
You can find the details in the [download page](https://pytorch.org/get-started/locally/).

For example,

```bash
# CUDA 11.3
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

#### Available datasets

- MNIST: `mnist`
- CIFAR10: `cifar10`
- CIFAR100: `cifar100`
- CelebA: `celeba`
- SVHN: `svhn`

### Sim2Real datasets

You should install `numba` and `wget` to simulate or download the datasets.

```bash
pip install numba wget
```

#### Available datasets

- Lotka Volterra: `lotka_volterra`

  TODO: See `npf.jax.data.save:save_lotka_volterra` for more detailed options.

## Models

- **CNP**: Conditional Neural Process
- **NP**: Neural Process
- **CANP**: Conditional Attentive Neural Process
- **ANP**: Attentive Neural Process
- **BNP**: Bootstrapping Neural Process
- **BANP**: Bootstrapping Attentive Neural Process
- **NeuBNP**: Neural Bootstrapping Neural Process
- **NeuBANP**: Neural Bootstrapping Attentive Neural Process
- **ConvCNP**: Convolutional Conditional Neural Process
- **ConvNP**: Convolutional Neural Process

## Scripts

### Train

```bash
python scripts/jax/train.py -f <config-file> [additional-options]
```

You can use your own config file or use the provided config files in the `configs` directory.
For example, the following command will train a CNP model with learning rate of `0.0001` for `100` epochs:

```bash
python scripts/jax/train.py -f configs/gp/rbf/inf/anp.yaml \
    -lr 0.0001 \
    --train.num_epochs 100
```

You can see the help of the config file by using the following command:

```bash
python scripts/jax/train.py -f <config-file> --help
```

### Test

```bash
# From a trained model directory
python scripts/jax/test.py -d <model-output-dir> [additional-options]

# From a new config file and a trained model checkpoint
python scripts/jax/test.py -f <config-file> -c <checkpoint-file-path> [additional-options]
```

You can directly test the trained model by specifying the output directory.
For example:

```bash
python scripts/jax/test.py -d outs/CNP/Train/RBF/Inf/220704-181313-vweh
```

where `outs/CNP/Train/RBF/Inf/220704-181313-vweh` is the output directory of the trained model.

You can also replace or add the test-specific configs from the config file using the `-tf / --test-config-file` option.
For example:

```bash
python scripts/jax/test.py -d outs/CNP/Train/RBF/Inf/220704-181313-vweh \
    -tf configs/gp/robust/matern.yaml
```

### Test Bayesian optimization

```bash
# From a trained model directory
python scripts/jax/test_bo.py -d <model-output-dir> [additional-options]

# From a new config file and a trained model checkpoint
python scripts/jax/test_bo.py -f <config-file> -c <checkpoint-file-path> [additional-options]
```

Similar to above the `test` script, you can directly test the trained model by specifying the output directory.
For example:

```bash
python scripts/jax/test_bo.py -d outs/CNP/Train/RBF/Inf/220704-181313-vweh
```

You can also replace or add the test-specific configs from the config file using the `-bf / --bo-config-file` option.
For example:

```bash
python scripts/jax/test.py -d outs/CNP/Train/RBF/Inf/220704-181313-vweh \
    -bf configs/gp/rbf/bo_config.yaml
```

<br>
<br>
<br>
<br>

## Appendix

## Datasets

1. 1D regression (`x`: `[B, P, 1]`, `y`: `[B, P, 1]`, `mask`: `[B, P]`)
    - Gaussian processes, etc...

2. 2D Image (`x`: `[B, P, P, 2]`, `y`: `[B, P, P, (1 or 3)]`, `mask`: `[B, P, P]`)
    - Image completion, super resolution, etc...

3. Bayesian optimization (`x`: `[B, P, D]`, `y`: `[B, P, 1]`, `mask`: `[B, P]`)


## Dimension rule

- `x`: `[batch, *data_specific_dims, data_dim]`
- `y`: `[batch, *data_specific_dims, data_dim]`
- `mask`:   `[batch, *data_specific_dims]`
- `outs`:   `[batch, *model_specific_dims, *data_specific_dims, data_dim]`

### Examples

1. At `CNP` 1D regression:
    - `x`:    `[batch, point, 1]`
    - `y`:    `[batch, point, 1]`
    - `mask`: `[batch, point]`
    - `outs`: `[batch, point, 1]`

2. At `NP` 1D regression:
    - `x`:    `[batch, point, 1]`
    - `y`:    `[batch, point, 1]`
    - `mask`: `[batch, point]`
    - `outs`: `[batch, latent, point, 1]`

3. At `CNP` 2D image regression:
    - `x`:    `[batch, height, width, 2]`
    - `y`:    `[batch, height, width, 1 or 3]`
    - `mask`: `[batch, height, width]`
    - `outs`: `[batch, height, width, 1 or 3]`

4. At `NP` 2D image regression:
    - `x`:    `[batch, height, width, 2]`
    - `y`:    `[batch, height, width, 1 or 3]`
    - `mask`: `[batch, height, width]`
    - `outs`: `[batch, latent, height, width, 1 or 3]`

5. At `BNP` 1D regression:
    - `x`:    `[batch, point, 1]`
    - `y`:    `[batch, point, 1]`
    - `mask`: `[batch, point]`
    - `outs`: `[batch, sample, point, 1]`

5. At `BNP` 2D image regression:
    - `x`:    `[batch, height, width, 2]`
    - `y`:    `[batch, height, width, 1 or 3]`
    - `mask`: `[batch, height, width]`
    - `outs`: `[batch, sample, height, width, 1 or 3]`
