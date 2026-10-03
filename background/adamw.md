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
 - `v`: moving average of squared gradients. `v` shows how large have the gradients generally been.
