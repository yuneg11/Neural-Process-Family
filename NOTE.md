# Note

## TODOs

1. [ ] (**Important**) Check `iwae_loss` and `elbo_loss` of `ConvNP` model.
2. [ ] Add on-the-grid version of `ConvCNP` and `ConvNP` models.
3. [ ] Implement `UNet` and `ResNet` for large versions of `ConvCNP` and `ConvNP` models.
4. [ ] Change image normalization range from `(-0.5, +0.5)` to `(-1, +1)`.
5. [ ] Refactor usage of `NPData` in datasets.

## Design Notes

### 1. `NPData` class

`NPData` class holds data for Neural Processes.

There are five basic properties:

- `x`: Input data of shape `[batch_size, *task_size, input dim]` ($x = x_\text{context} \cup x_\text{target}$)
- `y`: Output data of shape `[batch_size, *task_size, output dim]` ($y = y_\text{context} \cup y_\text{target}$)
- `mask`: Mask of shape `[batch_size, *task_size]` ($m = m_\text{context} \cup m_\text{target}$)
- `mask_ctx`: Context mask of shape `[batch_size, *task_size]` ($m_\text{context}$)
- `mask_tar`: Target mask of shape `[batch_size, *task_size]` ($m_\text{target}$)

These properties construct the batch of tasks.
`x` represents the input $x$, `y` represents the output $y$, and `mask` represents the index set $c$.
Here, we use the notation $m$ instead of $c$ to denote the mask.
In addition to the above basic properties, `NPData` also provides the following properties:

- `x_ctx`: Context input data of shape `[batch_size, *task_size, input dim]` ($x_\text{context}$)
- `x_tar`: Target input data of shape `[batch_size, *task_size, input dim]` ($x_\text{target}$)
- `y_ctx`: Context output data of shape `[batch_size, *task_size, output dim]` ($y_\text{context}$)
- `y_tar`: Target output data of shape `[batch_size, *task_size, output dim]` ($y_\text{target}$)

Basically, these properties are the mask filled versions of the basic properties (filled with $0$),
and we can build them using the basic properties (e.g. `x_ctx = where(cond=mask_ctx, true=x, false=0)`).
However, the name of the additional properties is more intuitive.
So, we decided to let `NPData` provide them for convenience.

**Note**: The length of `*task_size` can be different (e.g. `*task_size = (28, 28)` in MNIST).
  You can use `NPData.flatten()` to flatten the `*task_size` to a single dimension (e.g. `*task_size = (28, 28)` to `(28 * 28,)`).

#### Examples

- Using `x`, `y` (from same tensors)

  ```python
  batch = NPData(
      x        = normal(key[0], shape=(4, 32, 32, 2)),  # [batch_size, *task_size, input_dim]
      y        = normal(key[1], shape=(4, 32, 32, 3)),  # [batch_size, *task_size, output_dim]
      mask_ctx = bernoulli(key[2], shape=(4, 32, 32)),  # [batch_size, *task_size]
      mask_tar = bernoulli(key[3], shape=(4, 32, 32)),  # [batch_size, *task_size]
  )

  # given properties
  batch.x         # [batch_size, *task_size, input_dim]
  batch.y         # [batch_size, *task_size, output_dim]
  batch.mask_ctx  # [batch_size, *task_size]
  batch.mask_tar  # [batch_size, *task_size]

  # these properties can be infered from the above properties
  batch.mask      # [batch_size, *task_size]             (mask = mask_ctx | mask_tar)
  batch.x_ctx     # [batch_size, *task_size, input_dim]  (x_ctx = where(cond=mask_ctx, true=x, false=0))
  batch.x_tar     # [batch_size, *task_size, input_dim]  (x_tar = where(cond=mask_tar, true=x, false=0))
  batch.y_ctx     # [batch_size, *task_size, output_dim] (y_ctx = where(cond=mask_ctx, true=y, false=0))
  batch.y_tar     # [batch_size, *task_size, output_dim] (y_tar = where(cond=mask_tar, true=y, false=0))

  # flatten the `*task_size` to a single dimension
  flatten_batch = batch.flatten()
  flatten_batch.x         # [batch_size, task_size, input_dim]
  flatten_batch.y         # [batch_size, task_size, output_dim]
  flatten_batch.mask_ctx  # [batch_size, task_size]
  flatten_batch.mask_tar  # [batch_size, task_size]
  flatten_batch.mask      # [batch_size, task_size]
  flatten_batch.x_ctx     # [batch_size, task_size, input_dim]
  flatten_batch.x_tar     # [batch_size, task_size, input_dim]
  flatten_batch.y_ctx     # [batch_size, task_size, output_dim]
  flatten_batch.y_tar     # [batch_size, task_size, output_dim]
  ```

<!--
TODO: Uncomment this after implement.

- Using `x_ctx`, `x_tar`, `y_ctx`, `y_tar` (from different tensors with different shapes)

  ```python
  batch = NPData(
      x_ctx    = normal(key[0], shape=(4, 5, 1)),  # [batch_size, *context_size, input_dim]
      x_tar    = normal(key[1], shape=(4, 7, 1)),  # [batch_size, *target_size,  input_dim]
      y_ctx    = normal(key[2], shape=(4, 5, 2)),  # [batch_size, *context_size, output_dim]
      y_tar    = normal(key[3], shape=(4, 7, 2)),  # [batch_size, *target_size,  output_dim]
      mask_ctx = bernoulli(key[4], shape=(4, 5)),  # [batch_size, *context_size]
      mask_tar = bernoulli(key[5], shape=(4, 7)),  # [batch_size, *target_size]
  )

  # given properties
  batch.x_ctx     # [batch_size, *task_size, input_dim]  (x_ctx = where(cond=mask_ctx, true=x, false=0))
  batch.x_tar     # [batch_size, *task_size, input_dim]  (x_tar = where(cond=mask_tar, true=x, false=0))
  batch.y_ctx     # [batch_size, *task_size, output_dim] (y_ctx = where(cond=mask_ctx, true=y, false=0))
  batch.y_tar     # [batch_size, *task_size, output_dim] (y_tar = where(cond=mask_tar, true=y, false=0))
  batch.mask_ctx  # [batch_size, *task_size]
  batch.mask_tar  # [batch_size, *task_size]

  # these properties can be infered from the above properties
  batch.x         # [batch_size, *task_size, input_dim]  (x = concat((x_ctx, x_tar), axis=1))
  batch.y         # [batch_size, *task_size, output_dim] (y = concat((y_ctx, y_tar), axis=1))
  batch.mask      # [batch_size, *task_size]             (mask = mask_ctx | mask_tar)
  ```
-->

### 2. `npf_io` decorator

When we implement the NP models, we usually expect the input to be a `NPData` object.
However, it is convenient to pass each `x`, `y`, ... directly to the model as a tensor.
So, `npf_io` decorator provides two main features:

1. Auto conversion of direct tensor inputs to `NPData` objects.
2. Auto input flattening of `*task_size` to a single dimension, (and output unflattening if necessary).

Each NP model requires different features, so we provide three variants of `npf_io` decorator:

- `@npf_io`: auto conversion, no input / output flattening
- `@npf_io(flatten=True)`: auto conversion, input / output flattening
- `@npf_io(flatten_input=True)`: auto conversion, input flattening / but no output flattening

This decorator gives some convenience, but it makes some overhead.
If the model calls another `npf_io` decorated function in the `npf_io` decorated function,
you can explicitly disable these features by passing `skip_io=False` to the inner function.

#### Example

```python
class Model:

  @npf_io(flatten=True)
  def __call__(self, data):
      ...
      return mu, sigma

  @npf_io(flatten_input=True)
  def log_likelihood(self, data):
      mu, sigma = self(data, skip_io=True)  # disable `npf_io` decorator when calling `__call__`.
      ...
      return ll

...

ll = model.apply(
    params, method=model.log_likelihood, rngs=rngs,
    x=x, y=y, mask_ctx=mask_ctx, mask_tar=mask_tar,  # pass tensors directly to the model
)
```

### 3. `NPF` models

`NPF` models should implement the three key methods: `__call__`, `log_likelihood` and `loss`.

- `__call__(self, data, **kwargs)`: forward pass. The return should be `(mu, sigma)` or `(mu, sigma, aux)` for `data.x`.
- `log_likelihood(self, data, split_sets: bool, **kwargs)`: log likelihood. The return should be
  - `ll` or `(ll, aux)` if `split_sets=False`
  - `(ll, ll_ctx, ll_tar)` or `(ll, ll_ctx, ll_tar, aux)` if `split_sets=True`
- `loss(self, data, **kwargs)`: loss.  The return should be `loss` or `(loss, aux)`.

Here, `aux` can be an auxiliary data which is used by internal functions or contains debugging metrics.
