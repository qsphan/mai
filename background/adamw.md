### Stochastic Gradient Descent
Suppose the model has a parameter `w = 2.0`, and the gradient of the loss is `gradient = 0.5`. SGD does:

$$w \leftarrow w - \mathrm{lr} \times g$$

if the learning rate $\mathrm{lr} = 0.1$, then $w \leftarrow 2.0 - 0.1 \times 0.5 = 1.95$.

The problem with SGD is that different parameters can have gradients with very different scales. For example:
```
parameter A: gradient = 0.001
parameter B: gradient = 10
```
Using the same learning rate for both can be problematic.

### Adam (Adaptive Moment Estimation)
Adam addresses the aforementioned issue by keeping track of two things for every parameter:
 - `m`: moving average of gradients. `m` shows which direction have gradients generally been pointing.
 - `v`: moving average of squared gradients. `v` shows how large have the gradients generally been, regardless of whether it is positive or negative.

### Background: moments and smoothing
A statistical moment is an average of powers of a random quantity. The first moment is the mean, $E[g]$. The second raw moment is the mean of squares, $E[g^2]$. Variance is built from both: $\mathrm{Var}(g) = E[g^2] - E[g]^2$.

Adam borrows this language: $m$ estimates $E[g]$ (which way gradients point on average), and $v$ estimates $E[g^2]$ (how large they are on average). The "true" gradient is an expectation over the whole dataset, but each step sees only one noisy minibatch, so Adam averages over time to estimate it.

Smoothing means replacing a noisy single sample with a running average. The simplest version is an equal average over all past gradients, but that never forgets old steps and reacts slowly when the landscape changes. Adam uses an exponential moving average instead:

$$s_t = \beta s_{t-1} + (1 - \beta) x_t$$

Each step keeps a fraction $\beta$ of the old average and mixes in a fraction $(1 - \beta)$ of the new sample. Older samples decay by $\beta^k$ after $k$ steps, so recent gradients matter most. The effective memory is about $1 / (1 - \beta)$ steps: around 10 steps for $\beta_1 = 0.9$ and around 1000 steps for $\beta_2 = 0.999$.

### First moment: m
Suppose gradients over several steps are `+0.5, +0.4, +0.6`. Adam maintains an exponential moving average:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

where each element is:
 - $m_t$: first-moment estimate at step $t$, the smoothed gradient Adam actually steps along.
 - $\beta_1$: decay rate, typically $\beta_1 = 0.9$. Closer to 1 means longer memory and smoother $m$; closer to 0 means faster reaction to new gradients.
 - $m_{t-1}$: previous average, the summary of all past gradients up to step $t - 1$.
 - $g_t$: current gradient, $g_t = \nabla L(w_t)$, what backprop just computed on this batch.
 - $(1 - \beta_1)$: weight on the new gradient. The two weights sum to 1 ($\beta_1 + (1 - \beta_1) = 1$), so $m_t$ is a weighted average rather than a growing sum.

So `m` acts somewhat like momentum. Instead of reacting only to today's gradient, Adam considers the recent history:
```
recent gradient history → m → smoothed direction
```

### Second moment: v
Adam also tracks squared gradients:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Typically $\beta_2 = 0.999$. Squaring removes the sign:
```
gradient = -5 → gradient² = 25
gradient = +5 → gradient² = 25
```
So `v` tells Adam roughly how large the gradients for this parameter have been.

### Adaptive update
After correcting for initialization bias (below), Adam essentially computes:

$$\mathrm{update} = \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

$$w \leftarrow w - \mathrm{lr} \times \mathrm{update}$$

This gives each parameter an adaptive effective learning rate. For example:

$$\mathrm{update}_A = 0.1 / 1 = 0.1, \quad \mathrm{update}_B = 0.1 / 10 = 0.01$$

Adam automatically makes the update smaller for parameters that have historically had large gradients.

### Weight decay: the "W" in AdamW
Without weight decay, the Adam update is approximately:

$$w \leftarrow w - \mathrm{lr} \times \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

AdamW additionally shrinks the parameter itself:

$$w \leftarrow w - \mathrm{lr} \times \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

$$w \leftarrow w - \mathrm{lr} \times \lambda \times w$$

Read these as two forces computed from the same starting $w_0$ and applied together in one fused update, not as two sequential in-place overwrites. A literal sequential execution would feed the result of the first line into the second line and pick up an extra second-order term ($\mathrm{lr}^2$), which the definition does not include.

Equivalently:

$$w \leftarrow (1 - \mathrm{lr} \times \lambda) w - \mathrm{lr} \times \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

**Explanation of that equivalence**:

Name the starting value $w_0$ and name the two amounts being subtracted:

$$a = \mathrm{lr} \times \hat{m} / (\sqrt{\hat{v}} + \epsilon), \quad d = \mathrm{lr} \times \lambda \times w_0$$

where $a$ is the Adam amount and $d$ is the decay amount. Applying both to the same $w_0$ gives:

$$w_{\mathrm{new}} = w_0 - a - d$$

Substitute $a$ and $d$ back in:

$$w_{\mathrm{new}} = w_0 - \mathrm{lr} \times \hat{m} / (\sqrt{\hat{v}} + \epsilon) - \mathrm{lr} \times \lambda \times w_0$$

where $\lambda$ is the weight-decay coefficient. So there are two separate forces:
```
w ──→ Adam adaptive update ──→ new w
         +
w ──→ shrink w toward 0 ──→ new w
```

### Why AdamW differs from "Adam + L2 regularization"
Weight decay can be implemented by adding $\frac{\lambda}{2} w^2$ to the loss. For ordinary SGD this produces:

$$g = g_{\mathrm{loss}} + \lambda w$$

$$w \leftarrow w - \mathrm{lr} \times (g + \lambda w)$$

For SGD, this is equivalent to weight decay. But for Adam, that summed gradient gets passed through Adam's adaptive scaling:
```
gradient + λw → Adam → adaptive scaling
```
So the weight decay becomes entangled with Adam's adaptive mechanism. AdamW instead says:
```
gradient → Adam → gradient update
AND separately:
weight → shrink weight
```
That is the key idea: AdamW decouples weight decay from the gradient-based Adam update.

### Full AdamW algorithm
For each parameter `w`:

Step 1: get gradient

$$g_t = \nabla L(w_t)$$

Step 2: update first moment

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

Step 3: update second moment

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Step 4: bias correction

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Step 5: Adam update

$$w \leftarrow w - \mathrm{lr} \times \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Step 6: weight decay

$$w \leftarrow w - \mathrm{lr} \times \lambda \times w$$

Bias correction divides by $1 - \beta^t$ because `m` and `v` start at zero and would otherwise be biased toward zero in early steps. `t` is the step counter, starting at 1.

### A concrete example
Suppose `w = 2.0`, `gradient = 0.5`, `lr = 0.1`, `weight_decay = 0.01`, and Adam's adaptive part produces $\hat{m} / (\sqrt{\hat{v}} + \epsilon) = 0.2$. Then the Adam update is:

$$w \leftarrow 2.0 - 0.1 \times 0.2 = 1.98$$

Weight decay then does:

$$w \leftarrow 1.98 - 0.1 \times 0.01 \times 1.98 \approx 1.97802$$

So:
```
original:         2.00000
gradient update:  1.98000
weight decay:     1.97802
```
Weight decay is simply pulling the parameter toward zero.

### Connecting to PyTorch code
When you write:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
)
```
`model.parameters()` gives AdamW all trainable tensors (`layer1.weight`, `layer1.bias`, ...). AdamW maintains optimizer state for each parameter, roughly:
```
parameter
 ├── gradient (from loss.backward())
 ├── m
 └── v
```
So a model with a million parameters stores roughly a million parameter values plus a million `m` values plus a million `v` values, which is one reason Adam/AdamW consumes substantially more optimizer memory than SGD.

In `cs336_basics/optimizer.py`, this maps to:
```python
defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
super().__init__(params, defaults)
```
`defaults` registers the hyperparameters, and `self.param_groups` organizes parameters by configuration. `self.state[p]` holds per-parameter memory (`step`, `m`, `v`). Inside `step()`, for each `p` with `p.grad is not None`, increment the counter, update `m` and `v`, apply bias correction and the Adam update plus decoupled weight decay, all under `@torch.no_grad()`.

The mental model:
```
SGD:      "Follow today's gradient."
Momentum: "Follow the recent direction of gradients."
Adam:     "Follow the recent direction, but normalize by how large gradients have been."
AdamW:    "Do Adam, AND independently shrink the weights toward zero."
```
Or even more compactly:
```
AdamW
 ├── gradient → m → direction
 ├── gradient² → v → scale
 └── weight → decay → shrink toward 0
```
Autograd computes the gradient; AdamW uses that gradient to update the parameters. The optimizer's bookkeeping (`m`, `v`, etc.) is not itself part of the model's differentiation graph.
