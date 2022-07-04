# Neural Process Family

**This library is under construction.**


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
python scripts/jax/train.py -f configs/attnnp.yaml -lr 0.0001 --model.train_kwargs.num_latents 30
```

The output will be saved under `outs/` directory.
Details will be added in the future.

## Download or build datasets

### Image datasets

```bash
python -m npf.jax.data.save --root ./datasets --dataset mnist
```

## Models

- CNP: Conditional Neural Process
- NP: Neural Process
- AttnCNP: Attentive Conditional Neural Process
- AttnNP: Attentive Neural Process
- ConvCNP: Convolutional Conditional Neural Process
- ConvNP: Convolutional Neural Process
- BNP: Bootstrapping Neural Process
- AttnBNP: Attentive Bootstrapping Neural Process


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
