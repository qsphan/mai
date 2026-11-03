# Numbat architecture

This document describes the model architecture in the
[burner22/numbat](https://huggingface.co/burner22/numbat) Hugging Face
repository, from the architecture code itself, so that the architecture
can be reimplemented or trained from scratch. It is not a description of
the released weights, and it does not verify the repository's claims
about where those weights came from.

The reference implementation is a checkpoint loader and inference
sketch, not a trainer. Section 13 lists every adaptation required
before it can be used for training. Do not train the downloaded
`modeling_numbat.py` unchanged.

## 1. Source and status

The architecture code was downloaded from the `main` branch of
[burner22/numbat](https://huggingface.co/burner22/numbat) on
2026-11-02. CodeHub / fbsource was searched for `modeling_numbat.py`
and `NumbatForCausalLM`; neither was found there. The Hugging Face
repository is therefore the only source used here.

Downloaded into this folder:

| File | Size | Role |
|---|---|---|
| [configuration_numbat.py](configuration_numbat.py) | 3,166 B | `NumbatConfig` |
| [modeling_numbat.py](modeling_numbat.py) | 22,280 B | Model classes |
| [config.json](config.json) | 4,194 B | Full configuration |
| [tokenizer.json](tokenizer.json) | 8,923,761 B | Tokeniser vocab/merges |
| [tokenizer_config.json](tokenizer_config.json) | 485 B | Tokeniser settings |
| [generation_config.json](generation_config.json) | 230 B | Generation defaults |
| [hf_quant_config.json](hf_quant_config.json) | 915 B | NVFP4 quantisation |

No `*.safetensors` weight shard was downloaded. The 772 shards
(about 7.72 TB) were not inspected. Files that remain upstream only
are the repository README, `ckpt_metadata.json`,
`training_log_tail.txt`, and the safetensors index and shards.

Verification done here, and its limits, are in section 15. In
particular, `torch` and `transformers` were not installed in this
workspace, so the model was never instantiated or run here. The
shapes, counts, and equations below were derived by reading the code
and recomputing them from [config.json](config.json).

## 2. Architecture at a glance

Numbat is a decoder-only causal language model. A token embedding
feeds 114 pre-norm decoder layers, a final RMSNorm, and an untied
`lm_head`. One extra multi-token-prediction (MTP) layer is stored
after the trunk and is used only by the `draft` path, not by the
ordinary forward pass.

| Quantity | Value |
|---|---|
| Vocabulary | 131,072 |
| Hidden size `d` | 12,288 |
| Decoder layers | 114 (indices 0–113) |
| MTP layer | 1 (checkpoint index 114) |
| Maximum positions | 131,072 |
| Tie word embeddings | No |
| Default dtype in config | `bfloat16` |
| Normalisation | RMSNorm, eps `1e-6` |
| Activation (feed-forward) | SiLU (SwiGLU) |
| Mixers | Gated DeltaNet (GDN) and MLA |
| Feed-forward | Dense SwiGLU or MoE |
| KV / recurrence cache | None in the reference code |

The headline parameter count, recomputed from the shapes in
[config.json](config.json) and
[modeling_numbat.py](modeling_numbat.py), is about
13.68 trillion logical parameters. Almost all of it is MoE experts
(see section 12). About 613 billion parameters are exercised per
token in the trunk, before the router and `lm_head`; see section 12
for exactly what that count includes and excludes.

## 3. Overall data flow

```
input_ids (B, L)
  -> embed_tokens                       (B, L, 12288)
  -> for i in 0..113: decoder layer i   (B, L, 12288)
  -> final RMSNorm
  -> lm_head                            (B, L, 131072)  logits

draft path only:
  trunk hidden h, next-token embeddings e
  -> MTP layer (index 114)
  -> final RMSNorm, lm_head             (B, L-1, 131072)
```

Each decoder layer is pre-norm with two residual branches:

```
x = x + mixer(input_layernorm(x), position_ids, attention_mask)
x = x + ffn(post_attention_layernorm(x))
```

There is no norm between the trunk and `lm_head` other than the
final `model.norm`, applied inside `NumbatForCausalLM.forward` and
`draft`. There is no learned positional embedding and no absolute
position term added to the hidden state. Position information
enters only through MLA RoPE (section 7). GDN receives
`position_ids` but ignores them. The MTP block receives neither
`position_ids` nor `attention_mask` (section 10).

`NumbatModel.forward` loops over exactly `num_hidden_layers`
layers. Although `self.layers` holds 115 modules (114 trunk layers
plus the MTP module), the MTP module is `layers[-1]` and is never
visited by the trunk loop. `NumbatForCausalLM.forward` therefore
never runs MTP. Only `draft` runs it, and only `draft` returns MTP
logits.

The reference forward methods take only
`(input_ids, position_ids, attention_mask)`. There is no `labels`
argument, no loss, no `past_key_values`, and no cache, matching
`use_cache: false` in the config. A trainer must add the loss
(section 13.3).

## 4. Layer schedule: which mixer and feed-forward, per layer

Layer type is decided by membership in two index lists in
[config.json](config.json). The two decisions are
independent: a layer can be MLA with a dense feed-forward
(layer 111 is).

| Layer indices | Mixer | Feed-forward | Quantised in checkpoint |
|---|---|---|---|
| 0, 1 | GDN | Dense | No (plain `nn.Linear`) |
| 2, 4–6, 8–10, … (not MLA, not dense) | GDN | MoE | Yes, except §11 |
| 3, 7, 11, …, 107 | MLA | MoE | Yes, except §11 |
| 109, 110, 112, 113 | GDN | Dense | No |
| 111 | MLA | Dense | No |

Rules, stated exactly:

- MLA layers are `mla_layer_indices`: 3, 7, 11, …, 111. That is,
  `i % 4 == 3`, 28 layers in total. Every other trunk layer is GDN:
  86 layers.
- Dense feed-forward layers are `dense_layer_indices`:
  0, 1, 109, 110, 111, 112, 113; 7 layers. Every other layer is
  MoE: 107 layers, agreeing with `moe_config.num_moe_layers`
  and with the `NumbatConfig.num_moe_layers` property, which
  counts trunk layers not in `dense_layer_indices`.
- In `NumbatDecoderLayer`, the mixer is built with
  `quantized=not dense`. So dense layers use plain `nn.Linear`
  everywhere, and MoE layers use `NVFP4Linear` for the mixer
  (subject to the per-module exceptions in section 11). The MTP
  block is always unquantised.
- The dense layers are the first two and the last five. Do not
  "simplify" this to first/last only, and do not assume the
  MLA pattern is "every fourth layer starting at 0": it starts
  at 3.

## 5. Embeddings, normalisation, tokenizer

### 5.1 Embeddings

`embed_tokens` is `nn.Embedding(131072, 12288)`, 1,610,612,736
parameters. `lm_head` is a separate, untied
`nn.Linear(12288, 131072, bias=False)` of the same size.
`tie_word_embeddings` is `false`, so the two matrices must be
trained (or loaded) independently. There is no embedding scale
factor, no dropout, and no positional embedding in the code.

Padding is represented only through `attention_mask` (section 13.4).
The pad token id is 2; the config gives no separate handling of
pad embeddings, so a trainer that trains on padded batches must
mask the loss and, for GDN, rely on the beta-masking in
section 6.4 rather than on any embedding mask.

### 5.2 RMSNorm

All norms are the `RMSNorm` class in
[modeling_numbat.py](modeling_numbat.py), not
`nn.RMSNorm` or LayerNorm:

```
y = x.float()
y = y * rsqrt(mean(y^2, dim=-1) + eps) * weight.float()
return y.to(x.dtype)
```

`weight` has shape `(dim,)` and is initialised to ones by the
class constructor. `eps` is `1e-6` everywhere. The computation
is in float32 and the result is cast back to the input dtype.
There is no bias, no mean subtraction, and no separate
`elementwise_affine` flag. Norm instances are:

- Two per decoder layer (`input_layernorm`,
  `post_attention_layernorm`), dim 12,288.
- One final `model.norm`, dim 12,288.
- Inside MLA: `q_a_layernorm` (dim 1,536) and `kv_a_layernorm`
  (dim 512).
- Inside GDN: `o_norm` (dim 256, the value head dim).
- Inside MTP: `enorm`, `hnorm`, `input_layernorm`,
  `post_attention_layernorm` (dim 12,288 each), plus the MLA
  norms inside its attention block.

Norm weights are in the quantisation ignore list (section 11),
so they stay in the checkpoint default dtype, not NVFP4.

### 5.3 Tokenizer

The downloaded [tokenizer.json](tokenizer.json) is a
BPE tokenizer whose `pre_tokenizer` is a sequence of a
GPT-2-style regex `Split` followed by ByteLevel, with a
ByteLevel decoder, no normalizer, and no post-processor,
with 131,072 vocabulary entries and 130,808 merges. [tokenizer_config.json](tokenizer_config.json)
says `PreTrainedTokenizerFast`, `add_bos_token: false`,
`add_eos_token: false`, left padding, `model_max_length`
131,072, and `clean_up_tokenization_spaces: false`.

Special tokens in the config are `<|begin|>` (0), `<|end|>`
(1), and `<|pad|>` (2). The vocab file additionally contains
`<|role|>` (3), `<|msg|>` (4), `<|tool|>` (5),
`<|thinking|>` (6), and `<|toolcall|>` (7) as added tokens.
A trainer must decide, explicitly, whether those five extra
tokens are ever produced in training data; the architecture
code itself attaches no semantics to them, and no chat
template is shipped (`use_default_template: false`).

Generation defaults in [generation_config.json](generation_config.json)
are sampling with temperature 1.0, top-p 1.0, top-k 0,
`max_length` 4096, `use_cache: false`. Those are inference
defaults for the checkpoint, not training settings. Note the
gap between `max_length` 4096 here and `max_position_embeddings`
131,072 in the config: the reference code imposes no length
limit itself, and long-context training at 131k would face
the MLA full-attention cost in section 13.5.

## 6. Gated DeltaNet (GDN) mixer: 86 layers

`NumbatGatedDeltaNet` is a gated delta rule: a linear recurrent
state per head, updated once per token, with a short causal
convolution on the query, key, and value projections. There is
no attention matrix and no softmax.

Config (`gdn_config` in [config.json](config.json)):

| Key | Value | Meaning |
|---|---|---|
| `num_heads` | 48 | Recurrent heads |
| `head_dim_k` | 128 | Key/query dim per head |
| `head_dim_v` | 256 | Value/output dim per head |
| `short_conv_kernel` | 4 | Depthwise conv kernel |
| `conv_bias` | false | No conv bias |
| `qk_l2norm` | true | L2-normalise q and k |
| `chunk_size` | 64 | Declared, not used (§6.5) |
| `state_dtype` | `float32` | Declared; state is fp32 |

### 6.1 Projections

All projections are bias-free linears from the hidden size.
In MoE layers they are `NVFP4Linear` except `a_proj`/`b_proj`
and the convolutions; in dense layers they are plain
`nn.Linear` (section 11).

| Module | In → out | Per-head shape |
|---|---|---|
| `q_proj` | 12,288 → 6,144 | 48 × 128 |
| `k_proj` | 12,288 → 6,144 | 48 × 128 |
| `v_proj` | 12,288 → 12,288 | 48 × 256 |
| `g_proj` | 12,288 → 12,288 | 48 × 256 |
| `o_proj` | 12,288 → 12,288 | Input is 48 × 256 flattened |
| `a_proj` | 12,288 → 48 | One scalar per head |
| `b_proj` | 12,288 → 48 | One scalar per head |

`a_proj` and `b_proj` are never quantised: the code comment
calls them recurrence-sensitive. They are plain `nn.Linear`
in every layer.

### 6.2 Short causal convolution

`q_conv1d`, `k_conv1d`, `v_conv1d` are depthwise `Conv1d`
modules (`groups = channels`, kernel 4, no bias). Each one
operates on the flattened per-head channels (6,144 for q/k,
12,288 for v). The input is left-padded with `kernel - 1`
zeros (causal: no future leakage), convolved, then passed
through SiLU:

```
y = silu(conv1d(pad_left(x, 3)))
```

The convolution is applied to q, k, and v only. `g_proj`
output is not convolved. After the convolution, q and k are
L2-normalised over the last dimension, per head
(`qk_l2norm: true`), before the recurrence. V is not
normalised.

### 6.3 Recurrence

Per head `h`, the state is a float32 matrix
`S` of shape `(256, 128)` — `(head_dim_v, head_dim_k)` —
initialised to zero for each forward call. For each position
`t`, with `q_t`, `k_t`, `v_t` the head's vectors at `t`:

```
S_t = alpha_t * (S_{t-1} - beta_t * (S_{t-1} k_t) k_t^T)
    + beta_t * v_t k_t^T
o_t = S_t q_t
```

The code implements exactly this, in float32, with
`state.matmul(k_t)`, an outer product with `k_t`, and
`state.matmul(q_t)`.

The gates are scalars per head per token, computed from the
layer input `x` (the post-norm hidden state):

```
alpha_t = exp(-exp(A_log) * softplus(a_proj(x) + dt_bias))
beta_t  = sigmoid(b_proj(x))
```

`A_log` and `dt_bias` are float32 parameters of shape `(48,)`,
both initialised to zeros by the constructor. With those
values, `exp(A_log) = 1` and `alpha_t = exp(-softplus(...))`,
in `(0, 1)`. `alpha` is a decay; `beta` is the write
strength of the delta-rule update. Both depend only on the
current token, not on position. There is no learned
per-layer or per-channel decay beyond the per-head `A_log`.

### 6.4 Output

The per-head outputs are stacked to `(H, B, L, 256)`, passed
through `o_norm` (RMSNorm over the 256 value dim), flattened
to `(B, L, 12288)`, gated, and projected:

```
o = o_proj(o_norm_output * silu(g_proj(x)))
```

Note the order: the norm is applied per head before
flattening and before the output gate, and the gate uses
`g_proj(x)` where `x` is the same layer input used for the
other projections. `o_proj` is the last step; there is no
further norm inside the mixer.

Padding: if `attention_mask` is given, `beta` is multiplied
by the mask, so padded positions never write to the state.
They still read the state and produce an output, and their
hidden states still flow into the next layer. Padding must
therefore also be handled by loss masking and, for packed
sequences, by resetting the state at document boundaries —
the reference code does neither (section 13.4).

### 6.5 What the reference implementation does not do

The reference `forward` is a Python `for t in range(L)` loop
over the sequence, materialising one output per step with
per-step fp32 matmuls. It keeps no state across calls, has
no chunkwise parallel form, and reads `chunk_size` and
`state_dtype` from the config in neither the constructor nor
the forward pass: the state dtype is hard-coded float32 and
there is no chunking. `chunk_size: 64` is therefore a
specification for a training implementation, not behaviour
of the downloaded code. A trainer must implement the
chunkwise form (section 13.5) and prove it matches this
recurrence, including the convolution, the L2 norms, and
the exact `alpha`/`beta` formulas, before using it for
training. The per-layer recurrent state is 48 × 256 × 128 =
1,572,864 float32 scalars per sequence in a batch; across
all 86 GDN layers that is 135,266,304 scalars per sequence.
State is `O(1)` in sequence length per layer, which is the
point of the design — but only if the chunkwise kernel is
implemented; the reference loop is `O(L)` Python steps.

## 7. Multi-head latent attention (MLA): 28 layers

`NumbatLatentAttention` is full causal self-attention with
low-rank latent queries and keys/values, decoupled RoPE, and
a sigmoid output gate. It is the DeepSeek-style MLA pattern:
keys and values are not materialised per head from the
hidden state; a compressed latent is expanded per head.

Config (`mla_config` in [config.json](config.json)):

| Key | Value |
|---|---|
| `num_heads` | 96 |
| `q_lora_rank` | 1,536 |
| `kv_lora_rank` | 512 |
| `qk_nope_head_dim` | 128 |
| `qk_rope_head_dim` | 64 |
| `v_head_dim` | 128 |
| `rope_theta` | 1,000,000.0 |

Per-head dimensions are therefore: query and key 192
(128 content + 64 RoPE), value 128. There is no per-head
key/value projection from the hidden state and no
multi-query / grouped-query grouping beyond the shared
latent.

### 7.1 Projections

All are bias-free linears, `NVFP4Linear` in MoE layers and
plain `nn.Linear` in dense layers and in MTP.

| Module | In → out | Notes |
|---|---|---|
| `q_a_proj` | 12,288 → 1,536 | Query latent |
| `q_a_layernorm` | RMSNorm(1,536) | On the query latent |
| `q_b_proj` | 1,536 → 18,432 | 96 × (128 + 64) |
| `kv_a_proj_with_mqa` | 12,288 → 576 | 512 latent + 64 RoPE key |
| `kv_a_layernorm` | RMSNorm(512) | On the latent part only |
| `kv_b_proj` | 512 → 24,576 | 96 × (128 + 128) |
| `o_gate_proj` | 12,288 → 12,288 | Sigmoid gate, from layer input |
| `o_proj` | 12,288 → 12,288 | Output projection |

Computation order in `forward`:

1. `q = q_b_proj(q_a_layernorm(q_a_proj(x)))`, reshaped to
   `(B, 96, L, 192)`.
2. `compressed = kv_a_proj_with_mqa(x)`, shape `(B, L, 576)`.
   Its first 512 dims go through `kv_a_layernorm` and then
   `kv_b_proj`, reshaped to `(B, 96, L, 256)`; its last 64
   dims are the shared RoPE key `k_pe`, kept as
   `(B, 1, L, 64)` and expanded to all 96 heads.
3. Split the expanded query/key: the first 128 dims are the
   content (NoPE) part; the last 64 dims get RoPE (section
   7.2). The key's content part comes from `kv_b_proj` output;
   its RoPE part comes from `k_pe`. The value is the last
   128 dims of the `kv_b_proj` output.

The latent that would be cached at inference is 576 dims
per token per MLA layer (512 + 64), versus an expanded
96 × (192 + 128) = 30,720 dims for explicit per-head K/V.
The reference code caches nothing (`use_cache: false`) and
recomputes the expansion on every forward call.

### 7.2 RoPE (decoupled, half-split)

RoPE is applied only to the 64-dim RoPE parts of q and k.
Frequencies use `theta = 1,000,000`:

```
freq_j = position * theta^(-2j / 64),  j = 0, 1, ..., 31
```

That gives 32 frequencies. The `_rope` helper splits the
last dimension into halves, not into interleaved pairs:

```
x1, x2 = x[..., :32], x[..., 32:]
out = cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)
```

`cos`/`sin` have shape `(B, 1, L, 32)` and broadcast over
heads. If `position_ids` is `None`, positions default to
`arange(L)` per row. RoPE is applied to the shared `k_pe`
before it is combined with the per-head content key, and to
the query's RoPE part. The content parts receive no
position encoding. There is no attention scaling factor in
the code and no QK-norm in MLA: `scaled_dot_product_attention`
uses its default scale, `1/sqrt(192)`, from the full 192-dim
query/key.

### 7.3 Attention and output gate

Attention is causal `F.scaled_dot_product_attention`. With
no mask it uses `is_causal=True`. With a mask, the code
builds one additive mask combining padding (keys at padded
positions get `-inf`) and a causal upper-triangular `-inf`
mask, and passes it as `attn_mask`. Padding in the query
positions is not masked by the attention itself; it is
handled only by loss masking downstream.

The output is transposed and flattened to `(B, L, 12288)`
(96 heads × 128 value dims), gated, and projected:

```
o = o_proj(o * sigmoid(o_gate_proj(x)))
```

`o_gate_proj` takes the same layer input `x` as the other
projections (the post-norm hidden state from the decoder
layer). The gate is a sigmoid, not SiLU, and it is applied
before `o_proj`, not after.

### 7.4 Cost note

MLA attention is quadratic in sequence length in the
reference code, and it materialises per-head q/k/v. At the
config maximum of 131,072 positions this is not trainable
with the reference SDPA call as written; section 13.5 covers
what a training implementation must change (flash / latent
attention, sequence parallelism, shorter training lengths)
without changing the mathematics above.

## 8. Dense feed-forward: SwiGLU

`NumbatMLP` is a SwiGLU block with three bias-free linears:

```
y = down_proj(silu(gate_proj(x)) * up_proj(x))
```

For dense layers the intermediate size is
`intermediate_size = 24,576` (2 × hidden). `gate_proj` and
`up_proj` are 12,288 → 24,576; `down_proj` is 24,576 →
12,288. The activation function is taken from
`config.hidden_act` (`silu`) via `getattr(F, ...)`, so a
config change there changes the block. In dense layers the
three linears are plain `nn.Linear`; the same class with
`NVFP4Linear` is used for MoE experts (section 9), where
the intermediate size differs. The MTP block uses a dense
MLP of the same 24,576 size (section 10).

## 9. Mixture of Experts (MoE): 107 layers

`NumbatMoE` is one shared expert plus grouped, sigmoid-routed
experts. It is the `mlp` submodule of every non-dense layer.
Dense layers do not have a router, a shared expert, or routed
experts.

Config (`moe_config` in [config.json](config.json)):

| Key | Value |
|---|---|
| `n_routed_experts` | 1,224 |
| `n_shared_experts` | 1 (hard-coded as one `NumbatMLP`) |
| `num_experts_per_tok` (top-k) | 48 |
| `moe_intermediate_size` | 2,816 per expert |
| `scoring_func` | `sigmoid` |
| `norm_topk_prob` | true |
| `routed_scaling_factor` | 1.0 |
| `n_group` / `topk_group` | 72 / 8 |
| `router_dtype` | `bfloat16` (metadata; code uses fp32) |
| `router_score_dtype` | `float32` |
| `dropless` | true (metadata; see §9.4) |
| `freeze_correction_at_token_fraction` | 0.98 (metadata; see §9.4) |

### 9.1 Experts

Each expert, routed or shared, is a SwiGLU `NumbatMLP`
with intermediate size 2,816: `gate_proj` and `up_proj`
12,288 → 2,816, `down_proj` 2,816 → 12,288. One expert
has 103,809,024 parameters. The layer therefore holds
1,225 experts (1,224 routed + 1 shared) in the module
tree, as `experts: ModuleList` plus `shared_experts`.

The shared expert is computed for every token and added
to the routed output:

```
out = shared_experts(x) + sum over selected experts
```

The code constructs both the shared expert and every
routed expert with `NVFP4Linear`, unconditionally — the
`quantized` argument to `NumbatMoE.__init__` is accepted
and ignored. That hard-coding matters for training
(section 13.1): unlike GDN/MLA, there is no
`quantized=False` path for MoE in the downloaded code.

### 9.2 Router: aux-loss-free, grouped, sigmoid

`NumbatRouter` is named `gate` in the checkpoint
(`mlp.gate`). Its weight is a parameter of shape
`(1224, 12288)`, created with `torch.empty` — that is, with
uninitialised contents (section 13.2). Its correction bias
`e_score_correction_bias` is a float32 buffer of shape
`(1224,)`, initialised to zeros.

Forward, per token, in exactly this order:

1. Scores: `scores = sigmoid(x.float() @ weight.float().T)`,
   shape `(B, L, 1224)`, computed in float32 regardless of
   the input dtype or `router_dtype`.
2. Corrected scores: `corrected = scores + bias`, where
   the bias is the `e_score_correction_bias` buffer. The
   bias affects selection only, never the weights.
3. Group ranking: the 1,224 experts are divided into 72
   consecutive groups of 17 experts each. A group's score
   is the sum of its top-2 corrected scores. Keep the
   top 8 groups by that score.
4. Candidate experts are exactly the experts in the kept
   groups (at most 8 × 17 = 136 of 1,224). All other
   experts are masked to `-inf`.
5. Select the top 48 experts among the candidates by
   corrected score. Those are the token's experts.
6. Weights: take the *uncorrected* sigmoid scores of the
   selected experts, normalise them to sum to 1, and
   multiply by `routed_scaling_factor` (1.0):

```
weights = scores[selected]
weights = weights / weights.sum(-1, keepdim=True) * 1.0
```

The router returns `(selection, weights)`, both
`(B, L, 48)`.

Three details are easy to get wrong in a reimplementation.
First, group membership is by consecutive index ranges
(`expert // 17`), not by any learned grouping. Second,
group scores use the top-2 corrected scores per group,
not the group's mean or max. Third, weights are
renormalised uncorrected scores, so the correction bias
can change *which* experts are chosen without changing
how much weight a chosen expert would have received on
its raw score, apart from the shared renormalisation.

### 9.3 Dispatch

`NumbatMoE.forward` flattens tokens, runs the router,
then loops over all 1,224 experts in module order. For
each expert it finds the tokens that selected it, runs
the expert on those tokens only, multiplies by the
token's weight for that expert, and accumulates with
`index_add_` into a zero output. Tokens that selected no
expert (impossible here, since top-48 always selects 48,
unless a mask intervened) would get only the shared
expert. There is no capacity factor, no token dropping,
and no padding to a fixed expert capacity in the code.

That loop is a correctness reference, not a training
implementation: it launches 1,224 expert forwards per
MoE layer per step, most of them on small token subsets.
A trainer needs grouped dispatch (section 13.5).

### 9.4 Load balancing: config only

`load_balancing.dropless: true` and
`freeze_correction_at_token_fraction: 0.98` appear only in
the config. No auxiliary balancing loss, no bias-update
rule, and no freezing logic exists in the downloaded code,
and `NumbatRouter.forward` never modifies
`e_score_correction_bias`. "Aux-loss-free" here therefore
describes the *selection mechanism* (a correction bias
used for routing), not a complete balancing scheme. A
trainer must decide and implement the outer loop that
updates the bias from observed expert loads, the schedule
(if any) at which it freezes, and any auxiliary loss —
or else accept the routing as written, with the bias
frozen at zero and no balancing pressure at all.
Whatever is chosen, it is a training decision, not
something the architecture code specifies.

## 10. Multi-token prediction (MTP): 1 layer

`NumbatMTP` is stored as `model.layers[114]`, one past
the trunk. Config (`mtp_config`):

| Key | Value |
|---|---|
| `depth` | 1 |
| `layer_index_in_checkpoint` | 114 |
| `loss_weight` | 0.1 |
| `inference_use` | Self-speculative drafter (1 draft token) |

The block shares `embed_tokens`, `model.norm`, and
`lm_head` with the trunk; it has no embedding or head of
its own. Its own modules, all unquantised plain linears
and norms, are:

| Module | Shape / role |
|---|---|
| `enorm` | RMSNorm(12,288), on the next-token embedding |
| `hnorm` | RMSNorm(12,288), on the trunk hidden state |
| `eh_proj` | Linear(24,576 → 12,288, bias=False) |
| `input_layernorm` | RMSNorm(12,288) |
| `self_attn` | `NumbatLatentAttention`, `quantized=False` |
| `post_attention_layernorm` | RMSNorm(12,288) |
| `mlp` | Dense SwiGLU, intermediate 24,576 |

Forward, given trunk hidden state `h` and an embedding
`e` of the next token:

```
z = eh_proj(cat([enorm(e), hnorm(h)], dim=-1))
z = z + self_attn(input_layernorm(z))
z = z + mlp(post_attention_layernorm(z))
```

That is one decoder-like block: the MTP attention is an
MLA block with the same config as trunk MLA layers (96
heads, latent 512, RoPE 64), but it is called without
`position_ids` or `attention_mask`, so it uses default
`arange` positions and full causal SDPA within the
shifted sequence it is given. `depth: 1` means one such
block; the code has no loop over multiple MTP layers.

`NumbatForCausalLM.draft` wires it up:

```
h = trunk(input_ids)                    # (B, L, 12288)
e = embed_tokens(input_ids)[:, 1:]      # (B, L-1, 12288)
h = h[:, :-1]                           # (B, L-1, 12288)
m = mtp_layer(h, e)
logits = lm_head(model.norm(m))         # (B, L-1, 131072)
```

The docstring says the MTP layer "predicts the token
two positions ahead", consuming the current-position
hidden state and the embedding of the next token. On the
alignment in `draft`, logit `t` pairs hidden state for
input position `t` with the embedding of input position
`t+1`, and therefore targets input position `t+2`. A
training loss built from `draft` must use that alignment:
the first `L-2` draft logits target tokens `2..L-1`,
and the last draft logit has no target inside the
sequence. `loss_weight: 0.1` is config metadata only;
no loss involving it exists in the code (section 13.3).
At inference the config describes the MTP layer as a
self-speculative drafter producing one draft token; the
reference `draft` method returns full-vocabulary logits,
not a sampled draft token or an acceptance rule.

## 11. NVFP4 checkpoint format (inference only)

The released checkpoint stores most linears in NVFP4
(NVIDIA FP4, via ModelOpt 0.47.0), not in BF16. The
reference code implements dequantisation on every forward
pass. This section describes the format exactly as coded,
because a trainer must understand it to load, inspect, or
re-quantise weights — and must not train in it directly
(section 13.1).

### 11.1 The quad layout

Every quantised linear is stored as four tensors named
after the parent module `X`:

| Tensor | Shape | Dtype | Role |
|---|---|---|---|
| `X.weight` | `(out, in // 2)` | `uint8` | Packed E2M1 payload |
| `X.weight_scale` | `(out, in // 16)` | `float8_e4m3fn` | Block scale per 16 inputs |
| `X.weight_scale_2` | scalar `()` | `float32` | Global weight scale |
| `X.input_scale` | scalar `()` | `float32` | Calibrated activation scale |

Packing: byte `i` holds element `2*i` in its low nibble
and element `2*i+1` in its high nibble. Within a nibble,
bit 3 is the sign and bits 0–2 index the E2M1 magnitude
grid `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`. `unpack_e2m1`
decodes low then high, applies the sign, and flattens
back to `(out, in)`.

Dequantisation (`NVFP4Linear.dequantize`):

```
w = unpack_e2m1(weight).float()            # (out, in)
w = w.view(out, -1, 16) * weight_scale.float().unsqueeze(-1)
w = (w.view(out, -1) * weight_scale_2).to(bfloat16)
```

So the effective weight is
`E2M1 value × per-16 block scale × global scale`, in
BF16. The block scale is per output row and per group
of 16 input features; there is one global scale per
linear.

### 11.2 Activation quantisation in forward

`NVFP4Linear.forward` dequantises the weight every call.
If `input_scale` is non-zero, activations are also
quantised to E2M1, per group of 16, before the matmul:

```
x = x.float() / input_scale
g = x reshaped (..., -1, 16)
scale = (amax(|g|) / 6).clamp(min=1e-8), cast to
        float8_e4m3fn and back to float32
q = bucketize(|g| / scale, E2M1_BOUNDS)    # RTNE grid
x = E2M1_LUT[q] * sign(g) * scale
return F.linear(x.to(bf16), w) * input_scale
```

`E2M1_BOUNDS` are the midpoints between adjacent grid
magnitudes, `[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]`,
with the bounds at 0.75, 1.75, and 3.5 nudged with
`nextafter` toward zero so exact-boundary values land
on the round-to-nearest-even side. The code comment
states `input_scale` is calibrated as
`amax / 448 / 6`, so every block scale fits E4M3
exactly. If `input_scale` is zero, the activation
quantisation path is skipped and the layer is a plain
`F.linear` on the dequantised BF16 weight.

The group size is 16 throughout, matching
`quantization_config.group_size: 16`, 4-bit weights and
4-bit activations, non-dynamic (`dynamic: false`) in
the config. `NVFP4Linear.weight` has
`requires_grad=False` and the three scales are buffers,
not parameters. There is no straight-through estimator
and no gradient path through `bucketize` that a trainer
could rely on.

### 11.3 Which modules are quantised, and which are not

`quantization_config` in [config.json](config.json)
and [hf_quant_config.json](hf_quant_config.json)
agree. The ignore / exclude list is:

- `lm_head`, `model.embed_tokens`, `model.norm`.
- All of layers 0, 1, 109–113, and 114 (MTP).
- In every layer: `mlp.gate` (the router),
  `self_attn.a_proj`, `self_attn.b_proj`,
  `self_attn.*conv1d`, every `*layernorm`, and
  `self_attn.o_norm`.

In code terms: GDN/MLA linears in MoE layers are
`NVFP4Linear`; the same linears in dense layers are
plain `nn.Linear`; `a_proj`/`b_proj`, convolutions, and
the router are always plain; MoE experts (shared and
routed) are always `NVFP4Linear`; MTP is always plain.
`A_log`/`dt_bias` are fp32 parameters and the router
correction bias is an fp32 buffer in every case.

A state dict loaded with `strict=True` must therefore
contain, for each quantised linear, four entries
(`weight`, `weight_scale`, `weight_scale_2`,
`input_scale`), and for each unquantised linear a single
`weight` entry. Mixing the two layouts — for example,
replacing a quantised linear with `nn.Linear` and then
loading the checkpoint — fails or silently mis-loads
unless the loading code is changed with it.

## 12. Parameter counts and per-token cost

Counts below are logical parameters (dequantised shapes),
recomputed from [config.json](config.json). They
count each matrix once, including experts that are rarely
selected. They are not file sizes and not NVFP4 storage.

| Component | Parameters |
|---|---|
| Embeddings (`embed_tokens`) | 1,610,612,736 |
| `lm_head` (untied) | 1,610,612,736 |
| Final norm | 12,288 |
| 86 GDN layers, mixer only | 605,270,112 each |
| 28 MLA layers, mixer only | 368,838,656 each |
| 7 dense MLPs | 905,969,664 each |
| 107 MoE layers, FFN + router | 127,181,096,136 each |
| MTP layer | 1,576,847,360 |
| **Total** | **13,681,900,672,984 (~13.68T)** |

Breakdown of the MoE figure: one expert is
103,809,024 parameters, so 1,225 experts per MoE layer
are 127,166,054,400 parameters. Adding the router
(1,224 × 12,288 = 15,040,512) and its 1,224-element
correction bias gives 127,181,096,136 per MoE layer.
Across 107 MoE layers, experts, router, and bias total
13,608,377,286,552 parameters, 99.5% of the model.
Everything else — embeddings, head, mixers, dense MLPs,
norms, and MTP — is about 73.5 billion parameters
combined.

Per-token activated parameters (trunk only, counting the
full mixer, the shared expert plus the 48 selected routed
experts, and the two RMSNorms per layer) are about
613.0 billion, before the router and `lm_head`. Adding
the router (about 1.6 billion, evaluated for every
token) and the `lm_head` (1.6 billion) gives about
616 billion parameters exercised per token, excluding
the embedding lookup and the MTP layer. The MTP layer,
when run, adds its 1.58 billion parameters on top, but
its forward is a second pass over a shifted sequence,
not part of the trunk cost.

Memory scale, by arithmetic only: 13.68T parameters are
about 27.4 TB in BF16, before gradients, optimizer
state, activations, and the GDN/MLA states in
sections 6.5 and 7.1. A full-parameter AdamW run at
this size, in the usual fp32 master-weight plus two
fp32 moment layout, would need on the order of a few
hundred terabytes of accelerator memory for weights,
gradients, and optimizer state alone. That is a reason
to train a scaled-down variant first (section 13.6),
not a claim about any particular training system. Any
claim to train the full 13.68T model must show its
parallelism and memory plan, not just its architecture.

## 13. Training the architecture: required adaptations

Nothing in the downloaded code trains a model. There is
no optimizer, learning-rate schedule, data loader, loss,
or gradient handling, and no training recipe is shipped
in the config files. This section lists what must be
built or changed. Items marked "must" follow from the
code; items marked "decide" are choices the code does
not make for you. Do not present the "decide" items as
if they came from the repository.

### 13.1 Use trainable linears, not `NVFP4Linear`

`NVFP4Linear` is a frozen checkpoint container:

- Its `weight` has `requires_grad=False`.
- Its scales are buffers, so an optimizer never updates
  them.
- It dequantises the full BF16 weight on every forward
  pass, then (usually) quantises activations with a
  non-differentiable `bucketize`.

For training from scratch, construct the model with
plain `nn.Linear` everywhere. GDN and MLA already have
that path: their `linear` factory returns `nn.Linear`
when `quantized=False`, and dense layers and MTP already
use it. MoE does not: `NumbatMoE` hard-codes
`NVFP4Linear` for the shared and routed experts and
ignores its `quantized` argument. Patch
`NumbatMoE` to accept a linear factory (the way
`NumbatMLP` already accepts one) and pass `nn.Linear`
for training, or post-process the module tree replacing
every `NVFP4Linear` with an `nn.Linear` of the same
in/out shape. Keep a separate quantisation step for
inference: train in BF16/FP32, then calibrate and pack
to NVFP4 with the layout in section 11 if an NVFP4
checkpoint is needed. Quantisation-aware training with
a straight-through estimator is a different design and
is not in the reference code; do not assume the
reference activation-quantisation path can be trained
through as written.

The same replacement changes the state-dict layout:
a training checkpoint has one `weight` per linear,
not a quad. Keep the two layouts, and any conversion
between them, explicitly separate.

### 13.2 Initialisation: the reference code initialises nothing

`NumbatPreTrainedModel._init_weights` is an explicit
no-op, with the comment that every loadable tensor is
fully specified by the checkpoint. For training from
scratch that means there is no usable initialisation
scheme in the code, and the defaults that PyTorch
applies at construction are not a scheme:

- `nn.Linear`, `nn.Embedding`, and `nn.Conv1d` get
  PyTorch defaults.
- `RMSNorm.weight` gets ones (from the class).
- `A_log` and `dt_bias` get zeros (from the class).
- The router correction bias gets zeros (buffer).
- The router `weight` is created with `torch.empty`,
  i.e. uninitialised memory. Training on that value is
  undefined behaviour in practice; it must be
  initialised explicitly before the first step.

Decide and record an initialisation scheme before the
first training run, at minimum for: embeddings and
`lm_head` (untied, so initialise both), all linears,
the router, the depthwise convolutions, and the norms.
Standard practice to consider — not a repository
specification — includes scaled initialisation for the
residual-branch projections (`o_proj`, expert and dense
`down_proj`) by roughly `1/sqrt(2 * num_layers)`,
keeping `A_log`/`dt_bias` at zero as constructed, and
keeping norm weights at one and the correction bias at
zero. Whatever scheme is chosen, verify it on a small
model first: check that hidden-state norms and router
entropy are sane after a few steps, and that no expert
is starved from step zero by an uninitialised router.

Also preserve the dtype discipline in section 13.7:
`A_log`, `dt_bias`, the correction bias, router scores,
GDN state, and norm internals are float32 even when the
rest of the model is BF16. A trainer that casts the
whole model to BF16 after construction will silently
change those parts.

### 13.3 Loss: trunk loss plus MTP loss, both to be added

`NumbatForCausalLM.forward` returns only logits. Add:

1. Trunk loss: standard next-token cross-entropy on
   `forward` logits, shifted by one, with loss masking
   for padding and, if packing is used, for document
   boundaries (section 13.4). The config specifies no
   label smoothing, z-loss, or other auxiliary trunk
   loss; adding any is a training decision.
2. MTP loss: cross-entropy on `draft` logits with the
   alignment in section 10 — the first `L-2` draft
   logits target tokens `2..L-1`. Weight it by
   `mtp_config.loss_weight = 0.1` if the config value is
   to be followed. The config gives the weight and the
   depth; it gives no separate MTP label smoothing or
   loss type. The total loss is then
   `trunk_loss + 0.1 * mtp_loss` only if that
   interpretation is adopted deliberately; the code
   does not compute it.
3. MoE balancing: none in the code (section 9.4).
   Decide whether to add an auxiliary load-balancing
   loss, a correction-bias update rule, or neither, and
   record the choice. If a bias update is used, decide
   its schedule against the config's
   `freeze_correction_at_token_fraction: 0.98`
   metadata explicitly, rather than assuming the
   reference code implements it.

Train the MTP block jointly with the trunk from the
start only after checking that its gradients do not
destabilise the trunk through the shared embedding and
norm: the MTP block consumes trunk hidden states, so
its loss back-propagates into every trunk layer. An
alternative used for some MTP designs — freezing or
down-weighting MTP early — is a training decision, and
the `0.1` weight already embodies a down-weighting.
There is no second MTP stage (`depth: 1`).

### 13.4 Sequences, positions, and masking

- `position_ids`: pass them explicitly in packed or
  padded batches. MLA defaults to `arange(L)` when
  they are `None`, which is wrong for packed sequences
  (positions must restart per document) and for padded
  batches with left padding. GDN ignores them, so a
  bug in `position_ids` is visible only in MLA layers.
- `attention_mask`: the reference MLA mask combines
  padding and causality into one additive mask; the
  reference GDN mask only zeroes `beta` for padded
  positions. For packed sequences, neither mechanism
  resets state or attention at document boundaries:
  GDN state carries across the boundary, MLA attends
  across it, and the MTP block (which receives no mask
  at all) attends across it in `draft`. A trainer using
  packing must implement boundary resets — GDN state
  reset per document and MLA/MTP attention isolation
  per document (for example with a block-diagonal mask
  or position resets plus `cu_seqlens`-style handling
  in the attention kernel). A trainer using padding
  must mask the loss at padded positions in both the
  trunk and MTP losses.
- Sequence length: the config maximum is 131,072, but
  no training length is specified. Train at a chosen
  length (or a curriculum of lengths) and treat RoPE
  positions beyond the trained length as
  extrapolation, to be validated, not assumed.
- The reference code has no packing, no sliding
  window, and no sparse attention. Do not add them
  without checking that GDN state resets and MLA
  masks still match the un-packed mathematics.

### 13.5 Efficiency: replace the reference loops

Three reference implementations must be replaced for
any real training run. Each replacement must be proven
numerically equivalent (within the chosen dtype
tolerance) to the reference on random small inputs
before it is used for training, including the
convolution, norms, gates, and masks.

1. GDN recurrence. Replace the Python per-token loop
   with a chunkwise parallel formulation (intra-chunk
   attention-like computation plus inter-chunk state
   propagation), using the config `chunk_size: 64` as
   the chunk length. Keep the state in float32, keep
   the short convolution and L2 norms exactly as in
   section 6, and keep `alpha`/`beta` in float32.
   Sequence-parallel GDN additionally needs the state
   carried across chunk and device boundaries in the
   right order; a chunkwise kernel that resets the
   state per chunk is a different model.
2. MLA attention. Replace the materialised per-head
   SDPA call with a memory-efficient attention kernel
   (flash-style, or a latent-space formulation that
   never expands K/V per head) that implements the
   same half-split RoPE, the same default
   `1/sqrt(192)` scale, and the same causal/padding
   mask. Keep the latent projections and the sigmoid
   output gate outside the kernel, exactly as coded.
   At long lengths, add sequence parallelism or
   context-parallel attention; do not truncate the
   MLA context silently.
3. MoE dispatch. Replace the per-expert Python loop
   with grouped dispatch: sort/group tokens by expert,
   run each expert once on its token block with a
   grouped GEMM, and scatter the weighted results
   back. Preserve the exact router order of operations
   in section 9.2, including group selection before
   top-k, uncorrected-score weights, and the shared
   expert. Expert parallelism (sharding experts across
   devices, with all-to-all dispatch) and dropless
   dispatch (no capacity cap) are the natural
   counterparts of the 1,224-expert, top-48 design;
   a capacity-capped dispatcher changes the model.

Gradient checkpointing is declared supported
(`supports_gradient_checkpointing: True`) but no
checkpointing call exists in the reference forward
methods. A trainer must add activation checkpointing
at layer boundaries (and, for long sequences, inside
MLA/GDN as needed) and measure it; at 114 layers, full
activation storage without checkpointing is not a
viable default.

Distributed training at full scale needs, at minimum,
a plan for tensor parallelism (the 12,288 hidden size
and the large linears), expert parallelism (1,224
experts per MoE layer, 48 active per token),
pipeline parallelism (114 layers plus MTP), and data
parallelism, with the router and GDN-state
communication accounted for. The reference code has
none of this. This document does not prescribe a
framework or a parallelism layout; any layout must
preserve the per-token mathematics in sections 6–10,
in particular that each token's 48 experts and its
shared expert are combined with the same weights on
whatever device computes them.

### 13.6 Training a scaled-down variant first

The full model is not a sensible first training target
(section 12). To validate the implementation, scale
the architecture down while preserving its defining
ratios and invariants (section 14), for example by
reducing hidden size, layer count, expert count, and
top-k together. A useful small variant keeps:

- The same layer pattern: dense first two and last
  five, MLA on `i % 4 == 3`, GDN elsewhere.
- The same head dims (GDN 128/256, MLA 128/64/128),
  so the recurrence and RoPE code paths are unchanged.
- Grouped routing with the same structure: experts
  divisible by groups, top-k within the kept groups,
  top-2 group scoring, sigmoid scores, renormalised
  uncorrected weights.
- One shared expert, one MTP layer, untied embeddings,
  RMSNorm eps `1e-6`, and the same dtype discipline.

Validate the small variant end-to-end (loss decreases,
router uses a spread of experts, GDN chunkwise matches
sequential, MLA matches a naive attention reference,
MTP alignment is correct) before scaling any dimension
up. Scaling expert count or hidden size changes the
parameter count superlinearly through the experts
(section 12); re-derive the counts for the scaled
variant from its config rather than extrapolating
from the full model.

### 13.7 Dtype discipline

| Part | Dtype in the code |
|---|---|
| Default / linears / embeddings | Config `bfloat16` |
| RMSNorm internals | Float32, cast back |
| GDN state, `A_log`, `dt_bias` | Float32 |
| Router scores, correction bias | Float32 |
| MLA `q_a`/`kv_a` norms | Float32 internals (RMSNorm) |
| NVFP4 payload / block scales | `uint8` / `float8_e4m3fn` (§11) |
| NVFP4 global / input scales | Float32 |

A trainer in BF16 with FP32 master weights (or the
equivalent mixed-precision setup) must keep the
float32 items above in float32, not merely rely on
autocast. In particular, casting `A_log`/`dt_bias` to
BF16 changes `alpha` through `exp(A_log)`, and casting
router scores to BF16 changes top-k selection at
group boundaries. Conversely, the GDN recurrence must
not be run in BF16 "for speed" without re-validating
state accumulation over long sequences: the state is
a sum over the whole prefix, and the delta-rule update
is sensitive to the precision of `S_{t-1} k_t`.

## 14. Invariants and tensor shapes a reimplementation must preserve

### 14.1 Divisibility and index invariants

- `n_routed_experts` divisible by `n_group`:
  1,224 / 72 = 17 experts per group, exactly.
- `topk_group <= n_group` (8 <= 72) and
  `num_experts_per_tok <= topk_group * group_size`
  (48 <= 8 × 17 = 136), or the router's candidate set
  can be smaller than top-k and `topk` will select
  masked entries.
- `qk_rope_head_dim` even (64), so the half-split RoPE
  in section 7.2 is well defined; the frequency count
  is 32.
- Every NVFP4 linear's `in_features` divisible by 16
  (for the block scales) and by 2 (for nibble packing).
  All hidden/intermediate sizes in the config satisfy
  this; a scaled-down variant must keep it.
- `mla_layer_indices` and `dense_layer_indices` are
  lists, not patterns, in the config. Layer 111 is in
  both. A reimplementation that derives the schedule
  from a formula must reproduce the lists exactly,
  including that overlap, and must keep the MTP index
  equal to `num_hidden_layers` (114).
- `num_moe_layers` in `moe_config` (107) must equal
  `num_hidden_layers - len(dense_layer_indices)`.
  `NumbatConfig.num_moe_layers` recomputes the latter;
  a mismatch between the stored value and the lists
  is a config error, not a third schedule.
- `vocab_size`, `hidden_size`, and `num_hidden_layers`
  are explicit `NumbatConfig.__init__` parameters with
  defaults 131,072 / 12,288 / 114, and
  `intermediate_size` is taken from kwargs. When a
  config file is loaded, its values are passed as
  arguments and override those defaults; the defaults
  must not be relied on as the model sizes.

### 14.2 Tensor shapes (batch `B`, length `L`)

| Stage | Shape |
|---|---|
| Embeddings | `(B, L, 12288)` |
| GDN q/k per head | `(B, L, 48, 128)` |
| GDN v/g per head | `(B, L, 48, 256)` |
| GDN state per layer | `(48, B, 256, 128)`, fp32 |
| MLA q per head | `(B, 96, L, 192)` |
| MLA latent `compressed` | `(B, L, 576)` |
| MLA kv expanded | `(B, 96, L, 256)` |
| MLA output, flattened | `(B, L, 12288)` |
| Dense MLP intermediate | `(B, L, 24576)` |
| Expert intermediate | `(B, L, 2816)` per expert |
| Router scores / selection | `(B, L, 1224)` / `(B, L, 48)` |
| Trunk logits | `(B, L, 131072)` |
| MTP `eh_proj` input | `(B, L-1, 24576)` |
| MTP / draft logits | `(B, L-1, 131072)` |

The GDN state shape in the table is the code's
`(H, B, V, K)` ordering after the `permute(2, 0, 1, 3)`.
The MLA shapes are after the `permute(0, 2, 1, 3)` in
the code. A chunkwise or latent-space reimplementation
may store these differently internally, but the
values at layer boundaries must match these shapes'
contents.

### 14.3 Config-only fields (no effect in the code)

The following config entries are read into
`NumbatConfig` or stored in JSON but are never read by
any `forward` in the downloaded code. Changing them
does not change the reference model's behaviour; a
trainer that relies on them must implement them:

- `gdn_config.chunk_size`, `gdn_config.state_dtype`.
- `moe_config.load_balancing` (both keys),
  `moe_config.router_dtype` (code uses fp32),
  `moe_config.num_moe_layers` (stored, not used to
  build layers).
- `mtp_config.loss_weight`, `mtp_config.inference_use`,
  `mtp_config.depth` (the code builds exactly one
  block regardless).
- `quantization_config` as a whole, except that
  `NumbatConfig` stores it; the actual quantised vs
  plain choice is made by the `quantized` flags in
  the module constructors.
- `transformers_version: 5.19.0` in
  [config.json](config.json). The `auto_map`
  in the same file names `configuration_numbat` and
  `modeling_numbat` for `AutoConfig`/`AutoModel`/
  `AutoModelForCausalLM`, so loading the checkpoint
  with `transformers` requires `trust_remote_code=True`
  and executes the repository's Python files. That
  loading path was not exercised here (section 15).

## 15. Verification performed, and limits

Performed for this document:

- Downloaded the seven files in section 1 from the
  Hugging Face `resolve/main` URLs and compared their
  byte sizes with the repository tree from the
  Hugging Face API: configuration 3,166 B, modelling
  22,280 B, config 4,194 B, generation 230 B,
  quantisation 915 B, tokenizer config 485 B,
  tokenizer 8,923,761 B. No `*.safetensors` file is
  present in this folder.
- `python3 -m py_compile` on both Python files, and
  JSON parsing of all five JSON files. The tokenizer
  JSON was checked for 131,072 vocab entries,
  130,808 merges, and the eight added tokens listed
  in section 5.3.
- Searched CodeHub / fbsource with
  `meta codehub.file search --query=modeling_numbat.py`
  and `meta codehub.file grep --pattern=NumbatForCausalLM`
  (both read-only, autonomous): no results. The
  Hugging Face repository is the source.
- Recomputed the parameter counts in section 12 from
  the config shapes with a script, including the
  per-layer mixer, dense, MoE, and MTP figures and
  the 613-billion activated estimate and its stated
  inclusions.
- Read every class and method in
  [modeling_numbat.py](modeling_numbat.py) and
  [configuration_numbat.py](configuration_numbat.py)
  in full, including the NVFP4, GDN, MLA, router, MoE,
  MTP, and top-level forward paths, and checked each
  config key against whether the code reads it.

Not performed, and not claimed:

- The model was not instantiated, run, or trained:
  `torch` and `transformers` are not installed in
  this workspace, and no environment was created to
  install them. The equivalence checks recommended
  in section 13.5 (chunkwise GDN vs sequential,
  efficient MLA vs SDPA, grouped MoE dispatch vs
  the per-expert loop) were not run.
- No weight shard was downloaded or inspected, so no
  tensor name, dtype, scale, or value in the released
  checkpoint was verified, and the repository's
  claims about the checkpoint's origin, training run,
  and evaluation scores were not verified.
- No optimizer, schedule, data, or loss recipe is
  described in this document as coming from the
  repository, because none is in the downloaded
  files. Sections 13.2–13.6 contain training
  decisions and standard practice, labelled as such.

## 16. Starting points

The snippets below are starting points for a training
setup, not tested code — `torch`/`transformers` were
not available here (section 15). Adapt them to the
training framework in use, and apply the patches in
section 13.1 before training.

Loading the shipped configuration (inference path,
requires `transformers` and executes repository code):

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(
    "burner22/numbat", trust_remote_code=True
)
# For inference from the checkpoint only. For training,
# see the patch note below: NVFP4Linear is frozen.
model = AutoModelForCausalLM.from_pretrained(
    "burner22/numbat", trust_remote_code=True,
    torch_dtype="bfloat16",
)
```

Local equivalent, using the downloaded files in
this folder instead of the hub id (run from the
workspace root, the parent of this folder):

```python
import sys
sys.path.insert(0, "numbat")
from configuration_numbat import NumbatConfig
from modeling_numbat import NumbatForCausalLM

config = NumbatConfig.from_pretrained("numbat")
model = NumbatForCausalLM(config)  # still NVFP4/frozen: patch first
```

Training setup, in outline only. The essential patch
is to make MoE experts trainable plain linears, matching
the factory GDN/MLA already have, and to initialise
the router explicitly:

```python
# Outline, not run here.
# 1. Edit modeling_numbat.py: give NumbatMoE a linear
#    factory argument; pass nn.Linear for training,
#    as NumbatMLP already accepts.
# 2. Construct NumbatForCausalLM from NumbatConfig.
# 3. Initialise every parameter explicitly, including
#    the router (torch.empty in the reference code),
#    embeddings, and lm_head (untied).
# 4. Add trunk cross-entropy, MTP cross-entropy with
#    the section 10 alignment weighted 0.1, and the
#    chosen MoE balancing handling (section 9.4).
# 5. Replace the GDN loop, MLA SDPA call, and MoE loop
#    with the equivalent efficient forms (section 13.5)
#    and check equivalence on small random inputs.
```

A reduced-scale training config is a `NumbatConfig`
with smaller `hidden_size`, `num_hidden_layers`,
`n_routed_experts`, and `num_experts_per_tok`, keeping
the invariants in section 14. Re-derive its parameter
count from its own config; do not reuse section 12.

Upstream references for the files described here are
the [burner22/numbat](https://huggingface.co/burner22/numbat)
repository page and its
[file listing](https://huggingface.co/burner22/numbat/tree/main).
The local copies actually used are the seven
downloaded files in this folder, plus this document.
