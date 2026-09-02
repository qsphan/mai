# Understanding `einops.einsum`

## Explaining `einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")`

Consider this line:

```python
return einops.einsum(
    x,
    self.weight,
    "... d_in, d_out d_in -> ... d_out",
)
```

This is a linear transformation written in `einsum` notation. It is easiest to understand by reading the shapes and then reading the pattern.

### Start with the shapes

Suppose:

- `x.shape = (..., d_in)`
- `self.weight.shape = (d_out, d_in)`

For example, if:

- batch size is `32`
- `d_in = 4`
- `d_out = 3`

then:

- `x.shape = (32, 4)`
- `self.weight.shape = (3, 4)`

If `x` also has a sequence dimension, it might instead be:

- `x.shape = (32, 10, 4)`

In that case, the leading dimensions `(32, 10)` are what the `...` represents.

### Read the `einsum` pattern

The pattern is:

```python
"... d_in, d_out d_in -> ... d_out"
```

There are three important pieces.

First, `...` means "any leading dimensions that should be preserved." Those dimensions appear in the input and also appear in the output unchanged.

Second, `d_in` appears in both inputs:

- `x: ... d_in`
- `weight: d_out d_in`

Because `d_in` does not appear in the output, `einsum` multiplies over that dimension and sums it away.

Third, `d_out` appears in the weight and in the output:

- `weight: d_out d_in`
- `output: ... d_out`

So the result has one output feature dimension, `d_out`.

### The actual computation

For one input vector $\mathbf{x} = [x_1, x_2, \ldots, x_{d_{\text{in}}}]$, the output coordinate $y_j$ is:

$$
y_j = \sum_{i=1}^{d_{\text{in}}} x_i W_{j,i}
$$

That is exactly the computation of a linear layer without bias.

For a concrete example, let:

$$
\mathbf{x} = [x_1, x_2, x_3, x_4]
$$

and let the weight matrix be

$$
W =
\begin{bmatrix}
W_{1,1} & W_{1,2} & W_{1,3} & W_{1,4} \\
W_{2,1} & W_{2,2} & W_{2,3} & W_{2,4} \\
W_{3,1} & W_{3,2} & W_{3,3} & W_{3,4}
\end{bmatrix}
$$

Then the output has three coordinates:

$$
y_1 = x_1W_{1,1} + x_2W_{1,2} + x_3W_{1,3} + x_4W_{1,4}
$$

$$
y_2 = x_1W_{2,1} + x_2W_{2,2} + x_3W_{2,3} + x_4W_{2,4}
$$

$$
y_3 = x_1W_{3,1} + x_2W_{3,2} + x_3W_{3,3} + x_4W_{3,4}
$$

So the shape transformation is:

$$
(\ldots, d_{\text{in}}) \times (d_{\text{out}}, d_{\text{in}})
\rightarrow
(\ldots, d_{\text{out}})
$$

### Why is the weight shaped `(d_out, d_in)`?

This is the part that often looks backward at first.

Many APIs present a linear layer as if it were

```python
y = x @ weight.T
```

with:

- `x.shape = (..., d_in)`
- `weight.shape = (d_out, d_in)`
- `weight.T.shape = (d_in, d_out)`

The `einsum` notation makes that transpose unnecessary to write explicitly, because the pattern already states how the dimensions line up:

```python
"... d_in, d_out d_in -> ... d_out"
```

So this expression is conceptually equivalent to:

```python
x @ self.weight.T
```

### A useful mental model

Read

```python
"... d_in, d_out d_in -> ... d_out"
```

as:

- keep `...`
- match `d_in`
- multiply and sum over `d_in`
- produce `d_out`

That is why this pattern means:

> Apply the same linear transformation independently at every position represented by `...`.

If `x.shape = (batch, seq, d_in)`, then the operation applies the same weight matrix to every token in every batch element and returns shape `(batch, seq, d_out)`.
