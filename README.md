# NPF

## CNP

### Data

```
- x: [batch, sample, dim]
- y: [batch, sample, dim]
```

### Encoder

```
- input:  [batch, context, x_dim + y_dim]
- output: [batch, context, r_dim]
```

### Decoder

```
- input: [batch, target, x_dim + r_dim]
- output: [batch, target, y_dim * 2]
```

### LogLikelihood

```
- input: y_target, mu, var
- output: ll
```
