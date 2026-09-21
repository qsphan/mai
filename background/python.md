### `__call__()`
The method `__call__()` lets you make an object behave like a function. It is useful when you want something that is both:
 1. an object that stores state/configuration, and
 2. something you can invoke like a function.

Suppose we have a normal function

```python
def multiply(x):
    return 2 * x
```

The `2` is fixed inside the function. We can make it flexible with `__call__()`:

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor * x

double = Multiplier(2)
triple = Multiplier(3)

double(10)   # 20
triple(10)   # 30
```

You can argue that passing configuration to a function is perfectly reasonable, like the following:

```python
def multiply(x, factor):
    return x * factor

multiply(10, 2)
multiply(10, 3)
```

However, if we need to call this function 100 times, then we need to pass the `factor` 100 times. On the other hand, the configuration `factor` is attached to `double` or `triple`.

Moreover, it is even more useful in Pytorch. Although it is possible to write:

```python
def linear(x, W, b):
    return x @ W + b

x = linear(x, W1, b1)
x = linear(x, W2, b2)
```

It is not practical, because a neural network might have hundreds of layers, each with its own parameter. Moreover, `nn.Module` in Pytorch isn't just a container for arbitrary configuration. It can contain trainable parameters, buffers, hooks and so on.
So an instance of `nn.Module` represents the entire stateful computational object. You don't want:

```python
forward(
    x,
    layer1_weights,
    layer1_bias,
    layer2_weights,
    layer2_bias,
    ...
)
```

Rule of thumb: use a function when it is stateless, such as `softmax(x)`, `relu(x)` etc. Use an object when you have a persistent thing with state.

### `forward()`
It would be clearer to call:

```python
y = model.forward(x)
```

But in Pytorch:

```python
y = model(x)
```

This is because `nn.module` implements `__call__()`, so `model(x)` will eventually invokes `model.forward(x)`.
However, `__call__()` also handles PyTorch machinery around the forward pass, such as hooks. That's why we should call `model(x)` instead of just `model.forward(x)`.

#### Example

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```
This can be read like the following:

```
input x
  │
  ▼
Linear(10 → 20)
  │
  ▼
ReLU
  │
  ▼
Linear(20 → 5)
  │
  ▼
output
```

### `nn.ModuleList`
PyTorch does not properly register Python list, for example:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = [
            nn.Linear(10, 20),
            nn.Linear(20, 30),
        ]
```

`model.parameters()` will not contain the parameters of those Linear layers. Consequently, an optimizer such as:

```python
optimizer = torch.optin.Adam(model.parameters())
```

will not update them.

With `nn.ModuleList`:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.Linear(20, 30),
        ])
```

PyTorch knows:

```
MyModel
 └── layers (ModuleList)
      ├── Linear
      │    ├── weight
      │    └── bias
      └── Linear
           ├── weight
           └── bias
```

so their paramaters appears in `model.parameters()` and get trained.
