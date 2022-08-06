# Note

## Design Notes

### 1. `NPData`

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
