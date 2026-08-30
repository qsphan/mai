## Weight Initialization: Forward/Backward Stability

The key idea is:

> Weight initialization tries to keep both **activations** and **gradients** at a reasonable scale as they pass through the network.

There are two competing goals:

1. **Forward stability**: prevent activations from exploding or vanishing.
2. **Backward stability**: prevent gradients from exploding or vanishing.

---

### 1. Forward stability

Consider a linear layer:

$$
y = Wx
$$

where:

$$
W \in \mathbb{R}^{d_{out} \times d_{in}}
$$

For one output:

$$
y_j = \sum_{i=1}^{d_{in}} W_{ji}x_i
$$

Assume:

- $x_i$ has variance approximately 1
- each weight $W_{ji}$ has variance $\mathrm{Var}(W)$
- the terms are approximately independent

Then:

$$
\mathrm{Var}(y_j)
\approx
d_{in}\mathrm{Var}(W)
$$

There are `d_in` terms being summed, so the variance grows roughly with `d_in`.

To keep the output variance around 1:

$$
d_{in}\mathrm{Var}(W) \approx 1
$$

Therefore:

$$
\boxed{
\mathrm{Var}(W) \approx \frac{1}{d_{in}}
}
$$

This is what it means to say:

> **Forward stability favors variance `1 / in_features`.**

"Favors" does NOT mean "smaller is always better."

It means:

> For keeping forward activations stable, a variance around `1 / in_features` is a good scale.

If the variance is:

- **too large** → activations can explode
- **too small** → activations can vanish
- **around `1 / d_in`** → activations tend to stay at a reasonable scale

---

### 2. Backward stability

During backpropagation:

$$
g_x = W^T g_y
$$

For one input dimension:

$$ (g_x)_i = \sum_{j=1}^{d_{out}} W_{ji}(g_y)_j $$

Now notice that there are `d_out` terms being summed.

Therefore:

$$
\mathrm{Var}(g_x)
\approx
d_{out}\mathrm{Var}(W)
$$

To keep the gradient variance around 1:

$$
d_{out}\mathrm{Var}(W) \approx 1
$$

Therefore:

$$
\boxed{
\mathrm{Var}(W) \approx \frac{1}{d_{out}}
}
$$

This is what it means to say:

> **Backward stability favors variance `1 / out_features`.**

Again, "favors" means this is a good scale for preserving gradient magnitude. It does not mean "smaller is better."

---

### 3. The conflict

Forward stability wants:

$$
\boxed{
\mathrm{Var}(W) \approx \frac{1}{d_{in}}
}
$$

Backward stability wants:

$$
\boxed{
\mathrm{Var}(W) \approx \frac{1}{d_{out}}
}
$$

If:

$$
d_{in} = d_{out}
$$

there is no conflict.

But if:

$$
d_{in} \neq d_{out}
$$

we cannot satisfy both exactly.

For example:

```text
d_in  = 100
d_out = 200
