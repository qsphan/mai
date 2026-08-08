# MAI

## Xavier/Glorot normal initialization

A linear layer computes each output as a sum of `fan_in` terms. Assuming independent, zero-mean inputs and
weights,

```text
Var(output) = fan_in * Var(weight) * Var(input)
```

Keeping activation variance stable through the forward pass therefore suggests `Var(weight) = 1 / fan_in`.
Backpropagation applies the transposed weight matrix and sums over `fan_out` terms, so keeping gradient variance
stable through the backward pass instead suggests `Var(weight) = 1 / fan_out`. A non-square layer generally cannot
satisfy both goals exactly.

Xavier/Glorot initialization balances the two by using their average fan size:

```text
Var(weight) = 2 / (fan_in + fan_out)
Std(weight) = sqrt(2 / (fan_in + fan_out))
```

Xavier/Glorot **normal** initialization samples weights from a zero-mean normal distribution with that standard
deviation. This helps prevent activations and gradients from systematically exploding or vanishing as they pass
through many layers. The derivation is approximate and relies on assumptions such as independent inputs, weights,
and gradients, but it is a useful default in practice.

Reference: [Glorot initialization on Wikipedia](https://en.wikipedia.org/wiki/Weight_initialization#Glorot_initialization)
