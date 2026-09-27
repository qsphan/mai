# How do we calculate derivative
There are two ways to calculate a derivative: one is mathematically using calculus, and the second is approximately using finite differences.

Suppose the loss is $L(w) = w^2$ and $w = 2$. We try slightly increase it: $w = 2.001$. So the loss are:
 - L(2) = 4
 - L(2.0001) = 4.004001

So approximately:
```
change in loss
----------------
change in w
```
=>
```
(4.004001 - 4) / (2.001 - 2)

≈ 4.001
```
That's approximately the true derivative: `4`.

So conceptually, gradient is how much does the loss change if I slightly change this parameter.

# Make it a neural network
Suppose our model is $y = wx$, and we have:
 - $x = 3$
 - $w = 2$

There fore $y = 6$.
Suppose the correct answer is $target = 10$. Let's use the squared error:

$$
L = (y - target)^2 = (6 - 10)^2 = 16
$$

No we want $\frac{dL}{dw}$ but the loss doesn't directly depend on $w$. It depends on $w \rightarrow y \rightarrow L$. Specifically, $w \rightarrow y = wx \rightarrow L = (y - target)^2$.

This is where the chain rule comes in.

# Chain rule

$$
\frac{dL}{dw} = \frac{dL}{dy} * \frac{dy}{dw}
$$
