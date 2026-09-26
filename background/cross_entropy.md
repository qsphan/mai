# Cross-entropy and LogSumExp, from probabilities to working code

Cross-entropy turns a model's prediction into a number that training can minimize. LogSumExp lets us compute that number reliably from the model's raw output. Their connection is:

$$
\boxed{\text{loss} = \mathrm{logsumexp}(\text{logits}) - \text{logit of the correct class}}
$$

This tutorial assumes you know arrays, functions, and floating-point arithmetic. It introduces the ML vocabulary as needed, then derives and implements the formula used in CS336's `run_cross_entropy` adapter.

All logarithms below are natural logarithms, written $\ln$ or `log`. Their inverse is the exponential function, $\exp(x) = e^x$.

## 1. Classification is an API that scores possible answers

A classification model chooses among a fixed set of possible answers, called **classes**. For an animal classifier, we might assign these integer IDs:

```python
classes = ["cat", "dog", "bird"]
target = 0  # The observed correct answer is cat.
```

The **target** is the answer supplied by the training data. A model uses adjustable numbers called **parameters** or **weights** to map an input, such as an image, to scores for these classes.

Ideally we want probabilities:

```python
probabilities = [0.7, 0.2, 0.1]
```

These numbers are nonnegative and sum to one. Picking the largest gives `cat`, but training needs more information than whether that one choice was correct. Assigning the correct answer probability `0.51` and assigning it `0.99` should produce different feedback.

A **loss function** supplies that feedback as a scalar: smaller means a better prediction according to the training objective.

## 2. Cross-entropy scores the probability of the observed answer

For one example with one correct class, cross-entropy is:

$$
L = -\ln(p_y)
$$

Here $y$ is the target's integer index and $p_y$ is the probability at that index. In code, the formula is simply `-log(probabilities[target])`.

For our cat example:

$$
L = -\ln(0.7) \approx 0.357.
$$

If the model instead predicts `[0.01, 0.98, 0.01]`, the target is still cat, so the loss is $-\ln(0.01) \approx 4.605$. The high probability assigned to dog doesn't change which array entry we read: we always score the observed target.

Because $0 < p_y \leq 1$, its logarithm is nonpositive. The minus sign makes the loss nonnegative. Probability one gives loss zero; as the correct answer's probability approaches zero, loss grows without bound.

“Confidently wrong gets a large penalty” is useful shorthand. More precisely, the loss penalizes **low probability on the correct answer**, whether the remaining probability is concentrated on one wrong answer or spread across many.

### Why use a logarithm?

Imagine two training examples whose correct answers receive probabilities $0.7$ and $0.4$. The probability assigned to observing both answers, under the corresponding product model, is:

$$
0.7 \times 0.4 = 0.28.
$$

Making observed data more likely is called **maximum likelihood** training. Logarithms turn products into sums:

$$
-\ln(0.7 \times 0.4) = -\ln(0.7) - \ln(0.4).
$$

This gives an additive loss, avoids multiplying thousands of tiny probabilities, and preserves the optimization goal because logarithms are strictly increasing. Minimizing negative log-likelihood maximizes the likelihood of the observed answers.

For a batch of $B$ examples, we usually take the mean:

$$
L_{\text{batch}} = -\frac{1}{B}\sum_{b=1}^{B}\ln(p_{b,y_b}).
$$

Using the mean makes the loss scale easier to compare across batch sizes.

## 3. Why the name “cross-entropy”?

So far the target has been an integer. We can equivalently represent it as a **one-hot vector**: one at the correct index, zero everywhere else.

```python
target_distribution = [1.0, 0.0, 0.0]
prediction = [0.7, 0.2, 0.1]
```

Let $q_i$ be the target probability for class $i$, and $p_i$ the model probability. The general definition is:

$$
H(q,p) = -\sum_i q_i\ln(p_i).
$$

With a one-hot target, only the correct class contributes, reducing the formula to $-\ln(p_y)$. You don't need to allocate a one-hot vector in code; indexing is enough.

In information theory, $-\ln(p_i)$ measures the **surprise** of outcome $i$ under distribution $p$. Cross-entropy is the average surprise using $p$ to predict outcomes drawn from $q$. The “cross” refers to using these two distributions. Using $q$ for both gives its own entropy, $H(q,q)$.

For a fixed target distribution, cross-entropy is minimized when the prediction matches it. With a one-hot target, the minimum is zero. With a genuinely uncertain target distribution, even a perfect match generally has positive loss. Natural logarithms measure this quantity in **nats**; base-two logarithms would measure it in bits.

## 4. Models usually return logits

A neural network's final layer usually returns arbitrary real-valued scores called **logits**, one per class:

```python
logits = [2.0, 1.0, 0.1]
```

Logits are not probabilities: they may be negative and don't have to sum to one. **Softmax** converts them into a probability distribution:

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)},
$$

where $z_i$ is the logit for class $i$, and $j$ ranges over all classes. Exponentials make every term positive; dividing by their sum normalizes the result.

For these logits, softmax is approximately `[0.6590, 0.2424, 0.0986]`. If cat is correct, the loss is approximately $-\ln(0.6590) = 0.4170$.

Only **differences between logits** matter. Adding the same constant to every logit leaves softmax unchanged, because the shared exponential factor cancels from numerator and denominator. Thus `[1002, 1001, 1000.1]` represents the same distribution as `[2, 1, 0.1]`.

This invariance is the key to a stable implementation.

## 5. LogSumExp is a function with a stable evaluation trick

The function's name describes its operations from the outside inward:

$$
\mathrm{LSE}(z) = \ln\left(\sum_j \exp(z_j)\right).
$$

For example:

$$
\mathrm{LSE}([1,2,3])
= \ln(e^1 + e^2 + e^3)
\approx 3.4076.
$$

LogSumExp itself is this mathematical function. Subtracting the maximum is the trick for evaluating it safely.

### The direct implementation can overflow

Consider `[1000, 999, 998]`. Computing `exp(1000)` overflows even float64, yielding infinity. Yet the correct LogSumExp is only about `1000.4076`, which is easy to represent.

This is an intermediate-value problem familiar from numerical software: the final answer fits, but the obvious computation doesn't.

Set $m = \max_j z_j$. Factor out $e^m$:

$$
\begin{aligned}
\mathrm{LSE}(z)
&= \ln\left(e^m\sum_j e^{z_j-m}\right) \\
&= m + \ln\left(\sum_j e^{z_j-m}\right).
\end{aligned}
$$

Our example becomes:

$$
1000 + \ln(e^0 + e^{-1} + e^{-2}) \approx 1000.4076.
$$

Every shifted exponent is at most zero, so each exponential is at most one. At least one is exactly one. For finite logits, this avoids exponential overflow and prevents the entire sum from underflowing to zero. Very small terms can still underflow individually, usually with negligible impact on the sum.

For $C$ classes, the mathematical bound is:

$$
m \leq \mathrm{LSE}(z) \leq m + \ln C.
$$

That explains why LSE is sometimes called a **smooth maximum**: it is close to the largest score when that score dominates, and equals $m + \ln C$ when every score equals $m$.

## 6. Derive cross-entropy directly from logits

Substitute softmax into the loss:

$$
\begin{aligned}
L
&= -\ln(p_y) \\
&= -\ln\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) \\
&= -z_y + \ln\left(\sum_j e^{z_j}\right) \\
&= \mathrm{LSE}(z) - z_y.
\end{aligned}
$$

Related identities are:

$$
\ln(p_i) = z_i - \mathrm{LSE}(z),
\qquad
p_i = \exp\left(z_i - \mathrm{LSE}(z)\right).
$$

These are mathematical identities. In floating-point code, keep the computation shifted for as long as possible. Combining the stable LSE formula with cross-entropy gives:

$$
\boxed{L = \ln\left(\sum_j e^{z_j-m}\right) - (z_y-m)}.
$$

This avoids adding $m$ back just to subtract a potentially similar large number, reducing cancellation error.

### Why not compute stable softmax, then take its logarithm?

Even a stable softmax can round a very small probability to zero. For logits `[0, -1000]` with target index `1`, the true loss is approximately `1000`. But the target probability underflows to zero in common floating-point types, so `-log(softmax(z)[1])` gives infinity.

The direct loss formula preserves the answer:

$$
\ln(e^0 + e^{-1000}) - (-1000) \approx 0 + 1000.
$$

We never need to materialize the tiny target probability.

## 7. A from-scratch PyTorch implementation

For CS336, the input contract is:

- `logits`: floating-point tensor of shape `(batch, vocab)`.
- `targets`: integer tensor of shape `(batch,)`, with one valid class index per row.
- Return: a scalar mean loss.

Each row is a separate prediction. The vocabulary axis is the class axis, so we reduce over the **last dimension**, not across the batch.

```python
import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    # Use float32 for low-precision inputs, preserving float64 when supplied.
    z = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits

    # (batch, vocab) - (batch, 1): broadcast one maximum per row.
    shifted = z - z.max(dim=-1, keepdim=True).values

    # One log-normalizer per example: (batch,).
    log_normalizer = shifted.exp().sum(dim=-1).log()

    # (batch,) -> (batch, 1) -> gather -> (batch,).
    correct_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    per_example_loss = log_normalizer - correct_logits
    return per_example_loss.mean()
```

`keepdim=True` retains a size-one vocabulary axis for broadcasting. `gather` selects a different target column in each row. Targets must be `torch.long`, on the same device as the logits, and in the range `[0, vocab_size)`.

This implementation assumes a nonempty batch, a nonempty vocabulary, and finite logits. It intentionally omits options such as ignored targets and label smoothing. Upcasting improves low-precision arithmetic, but cannot recover information already lost when the input logits were stored.

All operations remain connected to PyTorch's **autograd** system, which records operations and computes derivatives. Don't call `.item()`, detach the tensor, or wrap this function in `torch.no_grad()` during training; the optimizer needs those derivatives.

A few useful checks:

```python
import math

logits = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
targets = torch.tensor([0])
loss = cross_entropy(logits, targets)
assert abs(loss.item() - 0.417030) < 1e-5
loss.backward()
assert logits.grad is not None

# Uniform scores imply probability 1/3 for every class.
uniform = cross_entropy(torch.zeros(1, 3), torch.tensor([2]))
assert abs(uniform.item() - math.log(3)) < 1e-6

# A tiny target probability still gives a finite loss.
extreme = cross_entropy(torch.tensor([[0.0, -1000.0]]), torch.tensor([1]))
assert extreme.item() == 1000.0

# A shared offset must not materially change the loss.
shifted_loss = cross_entropy(logits.detach() + 1000, targets)
torch.testing.assert_close(shifted_loss, loss.detach(), atol=1e-5, rtol=1e-5)
```

In production, PyTorch provides `torch.nn.functional.cross_entropy` with a logits-first interface. For the assignment, implement the formula with core tensor operations as above, then connect it to `run_cross_entropy`. Passing probabilities into a function expecting logits would apply the wrong computation.

## 8. How the loss tells the model what to change

A **gradient** describes how the loss changes when an input changes slightly. For one example, cross-entropy has a particularly simple derivative with respect to each logit:

$$
\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i=y],
$$

where $\mathbb{1}[i=y]$ is one for the target class and zero otherwise. This follows because the derivative of LSE is softmax, and the derivative of $-z_y$ contributes minus one at the target index.

For probabilities `[0.7, 0.2, 0.1]` with target cat, the gradient is `[-0.3, 0.2, 0.1]`. An update that subtracts this gradient would increase the cat logit and decrease the others. For a mean over $B$ examples, each row's gradient is additionally divided by $B$.

Training adjusts shared model weights rather than independent logits. **Backpropagation** applies the chain rule through the operations that produced the logits, translating this signal into gradients for those weights. An optimizer, such as AdamW, uses those gradients to update the parameters.

Although the probability-space formula reads only `p[target]`, every logit affects that probability through softmax's denominator. Every class therefore participates in the training signal.

## 9. Language modeling is classification at each token position

A **token** is an element of a fixed vocabulary. It may represent a word, part of a word, punctuation, or bytes. For illustration, pretend each word below is one token:

```text
Input:   The  cat  sat  on  the
Target:  cat  sat  on   the mat
```

At each position the model predicts the next token. A causal attention mask ensures each prediction can use only the current and earlier input tokens. All these predictions can be computed in a single training forward pass.

At the final position, if the model assigns `mat` probability `0.40`, that position's loss is $-\ln(0.40) \approx 0.9163$. Another continuation, such as `floor`, could be linguistically reasonable, but this training example's observed target is `mat`. Learning across many examples teaches a distribution of continuations.

The model returns logits shaped `(batch, seq, vocab)`, with integer targets shaped `(batch, seq)`. For a batch without padding or ignored positions, use the two-dimensional loss function by flattening the first two axes:

```python
loss = cross_entropy(
    logits.reshape(-1, logits.shape[-1]),  # (batch * seq, vocab)
    targets.reshape(-1),                 # (batch * seq,)
)
```

The targets must already be shifted to the next token, as in the example. With padded sequences, exclude padding positions from both the loss sum and its averaging count.

The training loop is then: compute logits, compute mean cross-entropy, call `loss.backward()`, update the weights, and clear gradients before the next accumulation. For a token sequence, multiplying successive conditional next-token probabilities gives its sequence probability; adding their negative logs gives its negative log-likelihood.

You may also encounter **perplexity**, defined as $\exp(\text{mean token loss})$. A uniform prediction over $V$ tokens has loss $\ln V$ and perplexity $V$. Compare perplexity only when the tokenization and evaluation setup are comparable.

## 10. Keep these distinctions clear

- **Logits** are raw scores; **softmax** turns them into probabilities.
- **Cross-entropy** measures how much probability the model assigns to observed targets, averaged over the examples or token positions being trained on.
- **LogSumExp** is the logarithm of softmax's normalizing sum; subtracting the maximum makes its evaluation stable.
- Compute the loss directly in log space: `logsumexp(logits) - target_logit`, preferably keeping both terms shifted in a manual implementation.
- Low training loss means the model predicts its training targets well. Held-out evaluation is still needed to measure how well it predicts new data.
