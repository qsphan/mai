# SwiGLU in Transformer FFNs

This note explains why modern Transformers often use **SwiGLU** in the feed-forward network (FFN), what problem it addresses, how the idea arises from simpler FFN designs, and how to think about its weight dimensions.

You do not need a research background to read this. Familiarity with vectors, matrix multiplication, and the basic structure of a Transformer block is enough.

## 1. Start with the plain Transformer FFN

In a standard Transformer block, the FFN takes a token representation $\mathbf{x}$, expands it into a larger hidden space, applies a nonlinearity, and projects it back:

$$
\mathrm{FFN}(\mathbf{x}) = W_2 \, \phi(W_1 \mathbf{x})
$$

Here:

- $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ is the token representation;
- $W_1$ maps from $d_{\text{model}}$ to a larger hidden dimension $d_{\text{ff}}$;
- $\phi$ is an element-wise nonlinearity such as ReLU or GELU;
- $W_2$ maps back to $d_{\text{model}}$.

In pipeline form:

```text
x
 ↓
Linear
 ↓
Activation
 ↓
Linear
 ↓
output
```

This works well and is the basic FFN design used in many Transformers.

## 2. What problem does a plain FFN have?

The issue is not that the plain FFN is useless. The issue is that it is somewhat limited in how it decides what information should matter.

After the first projection, each hidden coordinate is passed through the same nonlinear function independently:

$$
\phi(W_1 \mathbf{x})
$$

That gives a nonlinear transformation, but it does not explicitly create a mechanism for one learned signal to control another learned signal. In practice, the model may benefit from a more selective computation:

- some features should be amplified only in certain contexts;
- some features should be suppressed when they are not useful;
- the FFN should not only transform information, but also decide how much of that transformed information to let through.

A useful intuition is:

- plain FFN: transform the token representation;
- gated FFN: transform the token representation and modulate how much of each feature is allowed to pass.

That additional selectivity often makes the FFN more expressive.

## 3. The idea that leads to SwiGLU

One natural way to make the FFN more selective is to split the hidden computation into two branches:

1. a **value branch** that carries candidate information;
2. a **gate branch** that decides how much of that candidate information should pass through.

That leads to a family of **gated linear unit** designs. The general pattern is:

$$
\text{output} = W_2 \left[ \text{gate}(\mathbf{x}) \odot \text{value}(\mathbf{x}) \right]
$$

where $\odot$ means element-wise multiplication.

This is the key architectural move. Instead of applying only one element-wise activation to one hidden vector, the model computes two hidden vectors and uses one to modulate the other.

Historically, SwiGLU is best understood as a specific member of this gated-FFN family:

- first came the ordinary FFN;
- then came gated FFN variants such as GLU-style designs;
- SwiGLU uses **SiLU** as the gate nonlinearity.

So the design logic is:

$$
\text{plain FFN}
\rightarrow
\text{gated FFN}
\rightarrow
\text{SiLU-gated FFN}
\rightarrow
\text{SwiGLU}
$$

## 4. The SwiGLU formula

SwiGLU is commonly written as:

$$
\mathrm{SwiGLU}(\mathbf{x})
=
W_2 \left[
\operatorname{SiLU}(W_g \mathbf{x})
\odot
(W_v \mathbf{x})
\right]
$$

There are two first-layer projections:

$$
W_g \mathbf{x}
\qquad\text{and}\qquad
W_v \mathbf{x}
$$

Then the gate branch is passed through SiLU:

$$
\operatorname{SiLU}(z) = z \, \sigma(z)
$$

and multiplied element-wise with the value branch:

$$
\operatorname{SiLU}(W_g \mathbf{x}) \odot (W_v \mathbf{x})
$$

The result is then projected back to the model dimension by $W_2$.

## 5. Why is the gate useful?

The gate gives the FFN a way to say:

- "this hidden feature matters here, so keep it";
- "this hidden feature is not useful here, so reduce it";
- "this feature should pass partially rather than being simply on or off."

That is the main reason SwiGLU is useful. It lets the FFN act more like a conditional computation.

A simple mental model is:

```text
             gate branch
x ── Linear ── SiLU ──┐
                      × ── Linear ── output
x ── Linear ──────────┘
      value branch
```

The lower branch produces candidate content. The upper branch decides how open the gate should be for each hidden coordinate.

Another way to phrase it is:

- plain FFN: "compute a richer hidden representation";
- SwiGLU: "compute a richer hidden representation and learn how much of each part should flow forward."

## 6. Why SiLU specifically?

The gate uses the **SiLU** activation:

$$
\operatorname{SiLU}(x) = x \sigma(x)
$$

This is different from a hard-threshold activation such as ReLU:

$$
\operatorname{ReLU}(x) = \max(0,x)
$$

SiLU is smooth, which makes the gate smoother as well. Instead of creating a hard cutoff, it allows a softer transition between suppressing and passing information.

That smoothness is one reason SiLU works well as the gating nonlinearity in practice.

You do not need to memorize subtle geometric properties here. The main practical point is:

> SiLU gives the gate a smooth, learned control signal rather than a sharp binary-style cutoff.

## 7. Why not just make the FFN bigger?

Making the FFN wider is one way to increase capacity. SwiGLU is attractive because it improves the **structure** of the FFN computation, not just its width.

There is an important engineering detail: SwiGLU introduces an extra projection, because it uses both $W_g$ and $W_v$ instead of only one first-layer matrix.

That means a direct comparison with a standard FFN would otherwise give SwiGLU more parameters and more FLOPs. To keep the comparison fair, modern architectures usually reduce the hidden width used inside the SwiGLU block so that total compute stays in roughly the same range as a conventional FFN.

So the practical claim is not:

> SwiGLU is better because it is simply larger.

It is closer to:

> For similar compute, a gated FFN often gives better model quality than a plain FFN.

That is why SwiGLU became common in modern LLMs.

## 8. The deeper mental model

A Transformer block has two different kinds of computation:

1. **attention**, which mixes information across tokens;
2. **the FFN**, which transforms each token representation independently.

With that framing:

- attention answers: "which other token information should influence this token?";
- the FFN answers: "how should this token's representation be transformed?";
- SwiGLU answers: "how should this token's representation be transformed, and how much of each transformed feature should pass through?"

So a compact mental model is:

| Component | Main role |
|---|---|
| Attention | Mix information across tokens |
| Plain FFN | Transform a token representation |
| SwiGLU FFN | Transform and selectively gate a token representation |

If you remember only one sentence, remember this:

> SwiGLU is an FFN with a learned gate.

## 9. Dimensions of the three weight matrices

SwiGLU has three learned matrices:

$$
W_g,\qquad W_v,\qquad W_2
$$

Assume:

- $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$;
- $d_{\text{model}}$ is the input and output dimension of the block;
- $d_{\text{ff}}$ is the hidden dimension used inside the FFN.

Then the matrix shapes are:

| Matrix | Shape | Role |
|---|---|---|
| $W_g$ | $d_{\text{ff}} \times d_{\text{model}}$ | Gate projection |
| $W_v$ | $d_{\text{ff}} \times d_{\text{model}}$ | Value projection |
| $W_2$ | $d_{\text{model}} \times d_{\text{ff}}$ | Projection back to model width |

Starting from

$$
\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}
$$

we get:

$$
W_g \mathbf{x} \in \mathbb{R}^{d_{\text{ff}}}
$$

and

$$
W_v \mathbf{x} \in \mathbb{R}^{d_{\text{ff}}}
$$

Then the element-wise product still has the same hidden size:

$$
\operatorname{SiLU}(W_g \mathbf{x}) \odot (W_v \mathbf{x})
\in
\mathbb{R}^{d_{\text{ff}}}
$$

Finally,

$$
W_2
\left[
\operatorname{SiLU}(W_g \mathbf{x}) \odot (W_v \mathbf{x})
\right]
\in
\mathbb{R}^{d_{\text{model}}}
$$

So the dataflow is:

```text
              Wg
d_model ─────────────→ d_ff
   │                    │
   │                 SiLU
   │                    │
   │                    ×
   │                    ↑
   │                    │
   └──── Wv ─────────→ d_ff
                        │
                        │
                        W2
                        ↓
                     d_model
```

## 10. Why three matrices matter

A regular FFN has two matrices:

$$
W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}
\qquad
W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}
$$

SwiGLU has three because the first hidden transformation is split into two roles:

- one matrix creates the gate signal;
- one matrix creates the value signal;
- the final matrix projects the gated hidden vector back to the model dimension.

That extra matrix is exactly why implementations often choose a somewhat smaller $d_{\text{ff}}$ for SwiGLU than for a plain FFN when they want similar parameter count and compute.

## 11. A concrete size example

Suppose:

$$
d_{\text{model}} = 4096
\qquad
d_{\text{ff}} = 11008
$$

Then:

$$
W_g,\ W_v \in \mathbb{R}^{11008 \times 4096}
$$

and

$$
W_2 \in \mathbb{R}^{4096 \times 11008}
$$

This is the kind of structure you will see in LLaMA-style model implementations.

## 12. What to remember

The most useful summary is:

1. A plain FFN transforms a token representation.
2. SwiGLU adds a learned gate to that transformation.
3. The gate makes the FFN more selective and expressive.
4. Modern models often choose SwiGLU because it tends to improve quality for roughly similar compute.
5. The key formula is:

$$
\boxed{
\mathrm{SwiGLU}(\mathbf{x})
=
W_2 \left[
\operatorname{SiLU}(W_g \mathbf{x})
\odot
(W_v \mathbf{x})
\right]
}
$$

If you remember one mental model, use this one:

> SwiGLU = FFN + learned gate.
