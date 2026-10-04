### Stochastic Gradient Descent
Suppose the model has a parameter `w = 2.0`, and the gradient of the loss is `gradient = 0.5`. SGD does:
```
w ← w - lr × gradient
```
if the learning rate `lr = 0.1`, then `w ← 2.0 - 0.1 × 0.5 = 1.95`.

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

### First moment: m
Suppose gradients over several steps are `+0.5, +0.4, +0.6`. Adam maintains an exponential moving average:
```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
```
Typically `β₁ = 0.9`. So `m` acts somewhat like momentum. Instead of reacting only to today's gradient, Adam considers the recent history:
```
recent gradient history → m → smoothed direction
```

### Second moment: v
Adam also tracks squared gradients:
```
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
```
Typically `β₂ = 0.999`. Squaring removes the sign:
```
gradient = -5 → gradient² = 25
gradient = +5 → gradient² = 25
```
So `v` tells Adam roughly how large the gradients for this parameter have been.

### Adaptive update
After correcting for initialization bias (below), Adam essentially computes:
```
update = m̂ / (sqrt(v̂) + ε)
w ← w - lr × update
```
This gives each parameter an adaptive effective learning rate. For example:
```
parameter A: m = 0.1, sqrt(v) = 1  → update = 0.1 / 1  = 0.1
parameter B: m = 0.1, sqrt(v) = 10 → update = 0.1 / 10 = 0.01
```
Adam automatically makes the update smaller for parameters that have historically had large gradients.

### Weight decay: the "W" in AdamW
Without weight decay, the Adam update is approximately:
```
w ← w - lr × m̂ / (sqrt(v̂) + ε)
```
AdamW additionally shrinks the parameter itself:
```
w ← w - lr × m̂ / (sqrt(v̂) + ε)
w ← w - lr × λ × w
```
Equivalently:
```
w ← (1 - lr × λ) × w - lr × m̂ / (sqrt(v̂) + ε)
```
where `λ` is the weight-decay coefficient. So there are two separate forces:
```
w ──→ Adam adaptive update ──→ new w
         +
w ──→ shrink w toward 0 ──→ new w
```

### Why AdamW differs from "Adam + L2 regularization"
Weight decay can be implemented by adding `λ/2 × w²` to the loss. For ordinary SGD this produces:
```
gradient = gradient_from_loss + λ × w
w ← w - lr × (gradient + λ × w)
```
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
```
Step 1: get gradient
g_t = ∇L(w_t)

Step 2: update first moment
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t

Step 3: update second moment
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²

Step 4: bias correction
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

Step 5: Adam update
w ← w - lr × m̂_t / (sqrt(v̂_t) + ε)

Step 6: weight decay
w ← w - lr × λ × w
```
Bias correction divides by `(1 - β^t)` because `m` and `v` start at zero and would otherwise be biased toward zero in early steps. `t` is the step counter, starting at 1.

### A concrete example
Suppose `w = 2.0`, `gradient = 0.5`, `lr = 0.1`, `weight_decay = 0.01`, and Adam's adaptive part produces `m̂ / (sqrt(v̂) + ε) = 0.2`. Then the Adam update is:
```
w ← 2.0 - 0.1 × 0.2 = 1.98
```
Weight decay then does:
```
w ← 1.98 - 0.1 × 0.01 × 1.98 ≈ 1.97802
```
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
