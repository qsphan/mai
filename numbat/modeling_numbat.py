"""numbat model.

Reference implementation of the numbat architecture: an interleaved
gated-delta / latent-attention stack with grouped MoE feed-forward and one
multi-token-prediction layer stored after the decoder trunk.

Checkpoint layout notes
-----------------------

All quantized matmuls are stored in the NVFP4 quad layout. The four
members of a quad are named after their parent module (``X``):

=====================  ==========================  ======================
member                 tensor                     dtype
=====================  ==========================  ======================
``X.weight``          packed E2M1 nibble pairs   ``uint8``
``X.weight_scale``    per-16-element block scale  ``float8_e4m3fn``
``X.weight_scale_2``  global weight scale         ``float32``
``X.input_scale``     calibrated activation scale ``float32``
=====================  ==========================  ======================

Packing: element ``2*i`` occupies the low nibble of byte ``i``, element
``2*i+1`` the high nibble. Bit 3 of a nibble is the sign; bits 0-2 index
the E2M1 magnitude grid ``0, .5, 1, 1.5, 2, 3, 4, 6``.

Quantization is per-layer: MoE layers carry NVFP4 quads everywhere
except their router; dense layers and the MTP layer are plain bfloat16.
``a_proj``/``b_proj`` in the delta rule and every router also stay in
bfloat16. ``A_log`` and ``dt_bias`` are fp32 parameters. The router
correction bias is a fp32 buffer.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .configuration_numbat import NumbatConfig
except ImportError:
    from configuration_numbat import NumbatConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# boundaries halfway between adjacent magnitudes of the E2M1 grid.
# RTNE ties round to the grid point with an even mantissa bit, which
# shifts ties at 0.75, 1.75, and 3.5 upward; nudge those three bounds
# so bucketize places exact-boundary values on the even side.
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
E2M1_BOUNDS[[1, 3, 5]] = torch.nextafter(
    E2M1_BOUNDS[[1, 3, 5]], torch.zeros(3))


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Byte tensor ``(..., n // 2)`` -> float tensor ``(..., n)``."""
    high = ((packed >> 4) & 0xF).long()
    low = (packed & 0xF).long()
    sign = (torch.stack([low, high], dim=-1) & 8) != 0
    codes = (torch.stack([low, high], dim=-1) & 7).long()
    values = E2M1_LUT.to(packed.device)[codes] * torch.where(
        sign, -1.0, 1.0
    )
    return values.flatten(-2)


class NVFP4Linear(nn.Module):
    """A frozen NVFP4 linear layer, stored as its quad member tensors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(
                out_features, in_features // 16, dtype=torch.float8_e4m3fn
            ),
        )
        self.register_buffer(
            "weight_scale_2", torch.zeros((), dtype=torch.float32)
        )
        self.register_buffer(
            "input_scale", torch.zeros((), dtype=torch.float32)
        )
        self.out_features = out_features
        self.in_features = in_features

    def dequantize(self) -> torch.Tensor:
        w = unpack_e2m1(self.weight).float()  # (o, i)
        w = w.view(self.out_features, -1, 16) \
            * self.weight_scale.to(torch.float32).unsqueeze(-1)
        return (w.view(self.out_features, -1)
                * self.weight_scale_2).to(torch.bfloat16)

    def forward(self, x):
        w = self.dequantize()
        if self.input_scale:
            # NVFP4 GEMM: activations are quantized to E2M1 per 16-element
            # group with a block scale, rescaled by the calibrated input_scale
            # (amax / 448 / 6, so every block scale fits E4M3 exactly)
            orig_shape = x.shape
            x = x.to(torch.float32) / self.input_scale.to(torch.float32)
            g = x.reshape(*x.shape[:-1], -1, 16)
            # block scales are calibrated in the E4M3 scale format
            scale = (g.abs().amax(-1, keepdim=True) / 6.0).clamp(min=1e-8)
            scale = scale.to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-8)
            q = torch.bucketize(g.abs() / scale,
                                E2M1_BOUNDS.to(g.device))  # RTNE onto the grid
            x = (E2M1_LUT.to(q.device)[q] * torch.sign(g) * scale)
            x = x.to(w.dtype).reshape(orig_shape)
            return F.linear(x, w) * self.input_scale.to(w.dtype)
        return F.linear(x.to(w.dtype), w)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class NumbatGatedDeltaNet(nn.Module):
    """Gated delta rule with a short causal convolution.

    Per head ``h``, with fp32 state ``S`` of shape ``(V, K)``::

        S_t = alpha_t * (S_{t-1} - beta_t * (S_{t-1} k_t) k_t^T)
            + beta_t v_t k_t^T
        o_t = S_t q_t

    ``q``/``k`` are L2-normalized per head; the recurrence inputs are

        alpha_t = exp(-exp(A_log) * softplus(a_proj(x) + dt_bias))
        beta_t = sigmoid(b_proj(x))
    """

    def __init__(self, config: NumbatConfig, quantized: bool = True):
        super().__init__()
        g = config.gdn_config
        self.num_heads = g["num_heads"]
        self.head_dim_k = g["head_dim_k"]
        self.head_dim_v = g["head_dim_v"]
        self.kernel = g["short_conv_kernel"]
        self.qk_l2norm = g.get("qk_l2norm", True)
        hidden = config.hidden_size

        def linear(i, o):
            return NVFP4Linear(i, o) if quantized else nn.Linear(
                i, o, bias=False
            )

        self.q_proj = linear(hidden, self.num_heads * self.head_dim_k)
        self.k_proj = linear(hidden, self.num_heads * self.head_dim_k)
        self.v_proj = linear(hidden, self.num_heads * self.head_dim_v)
        self.g_proj = linear(hidden, self.num_heads * self.head_dim_v)
        self.o_proj = linear(hidden, self.num_heads * self.head_dim_v)
        # a_proj / b_proj emit one scalar per head and are never
        # quantized (recurrence-sensitive; ~48 x hidden each).
        self.a_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        for name, dim in (
            ("q_conv1d", self.num_heads * self.head_dim_k),
            ("k_conv1d", self.num_heads * self.head_dim_k),
            ("v_conv1d", self.num_heads * self.head_dim_v),
        ):
            setattr(self, name, nn.Conv1d(
                dim, dim, kernel_size=self.kernel,
                groups=dim, bias=g.get("conv_bias", False),
            ))
        self.A_log = nn.Parameter(
            torch.zeros(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.zeros(self.num_heads, dtype=torch.float32))
        self.o_norm = RMSNorm(self.head_dim_v, config.rms_norm_eps)

    def _short_conv(self, module, x):
        """Causal grouped conv over the flattened per-head channels."""
        B, L = x.shape[0], x.shape[1]
        y = module(F.pad(
            x.reshape(B, L, -1).transpose(1, 2), (self.kernel - 1, 0)
        ))
        return F.silu(y.transpose(1, 2).reshape(x.shape))

    def forward(self, x, position_ids=None, attention_mask=None):
        B, L, _ = x.shape
        H, K, V = self.num_heads, self.head_dim_k, self.head_dim_v

        q = self._short_conv(self.q_conv1d,
                             self.q_proj(x).view(B, L, H, K))
        k = self._short_conv(self.k_conv1d,
                             self.k_proj(x).view(B, L, H, K))
        v = self._short_conv(self.v_conv1d,
                             self.v_proj(x).view(B, L, H, V))
        if self.qk_l2norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        q = q.permute(2, 0, 1, 3)  # (H, B, L, K)
        k = k.permute(2, 0, 1, 3)  # (H, B, L, K)
        v = v.permute(2, 0, 1, 3)  # (H, B, L, V)

        # alpha, beta: (B, L, H) -> (H, B, L)
        alpha = torch.exp(-torch.exp(self.A_log) * F.softplus(
            self.a_proj(x) + self.dt_bias
        )).permute(2, 0, 1)
        beta = torch.sigmoid(self.b_proj(x)).permute(2, 0, 1)
        if attention_mask is not None:
            # padded positions never write to the recurrent state
            beta = beta * attention_mask.to(beta.dtype)[None, :, :]

        state = x.new_zeros(H, B, V, K, dtype=torch.float32)
        outputs = []
        for t in range(L):
            k_t = k[:, :, t].to(torch.float32)  # (H, B, K)
            v_t = v[:, :, t].to(torch.float32)  # (H, B, V)
            a_t = alpha[:, :, t:t + 1].unsqueeze(-1)  # (H, B, 1, 1)
            b_t = beta[:, :, t:t + 1].unsqueeze(-1)  # (H, B, 1, 1)
            state = a_t * (state - b_t * state.matmul(k_t.unsqueeze(-1))
                           .matmul(k_t.unsqueeze(-2)))
            state = state + b_t * v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            outputs.append(state.matmul(
                q[:, :, t].to(torch.float32).unsqueeze(-1)).squeeze(-1))
        o = torch.stack(outputs, dim=2)  # (H, B, L, V)
        o = self.o_norm(o).permute(1, 2, 0, 3).reshape(B, L, H * V)
        o = o.to(x.dtype)
        return self.o_proj(o * F.silu(self.g_proj(x)))


class NumbatLatentAttention(nn.Module):
    """Multi-head latent attention with decoupled rope and output gate."""

    def __init__(self, config: NumbatConfig, quantized: bool = True):
        super().__init__()
        m = config.mla_config
        self.num_heads = m["num_heads"]
        self.qk_nope_head_dim = m["qk_nope_head_dim"]
        self.qk_rope_head_dim = m["qk_rope_head_dim"]
        self.v_head_dim = m["v_head_dim"]
        self.q_lora_rank = m["q_lora_rank"]
        self.kv_lora_rank = m["kv_lora_rank"]
        self.rope_theta = m["rope_theta"]
        hidden = config.hidden_size

        def linear(i, o):
            return NVFP4Linear(i, o) if quantized else nn.Linear(
                i, o, bias=False
            )

        self.q_a_proj = linear(hidden, self.q_lora_rank)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = linear(
            self.q_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
        )
        self.kv_a_proj_with_mqa = linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.o_gate_proj = linear(hidden, hidden)
        self.o_proj = linear(hidden, hidden)

    @staticmethod
    def _rope(x, cos, sin):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self,
        x,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(
                L, device=x.device, dtype=torch.float32
            ).expand(B, L)

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, L, self.num_heads, -1).permute(0, 2, 1, 3)
        compressed = self.kv_a_proj_with_mqa(x)
        kv = self.kv_b_proj(self.kv_a_layernorm(
            compressed[..., : self.kv_lora_rank]
        ))
        kv = kv.view(B, L, self.num_heads, -1).permute(0, 2, 1, 3)
        k_pe = (compressed[..., self.kv_lora_rank:]
                .unsqueeze(1)
                .expand(-1, self.num_heads, -1, -1))

        freqs = position_ids.float()[:, None, :, None] * (
            self.rope_theta ** (-torch.arange(
                0, self.qk_rope_head_dim, 2, dtype=torch.float32,
                device=x.device,
            ) / self.qk_rope_head_dim)
        ).view(1, 1, 1, -1)
        cos, sin = freqs.cos(), freqs.sin()

        q = torch.cat([q[..., : self.qk_nope_head_dim],
                       self._rope(q[..., self.qk_nope_head_dim:], cos, sin)],
                      dim=-1)
        k = torch.cat([kv[..., : self.qk_nope_head_dim],
                       self._rope(k_pe, cos, sin)], dim=-1)
        v = kv[..., self.qk_nope_head_dim:]
        if attention_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # padding (B, L) plus causal masking, as one additive mask
            keep = attention_mask.to(torch.bool)[:, None, None, :]
            mask = torch.zeros((B, 1, 1, L), dtype=q.dtype, device=q.device)
            mask = mask.masked_fill(~keep, float("-inf")) + \
                torch.triu(torch.full((L, L), float("-inf"),
                                      dtype=q.dtype, device=q.device), 1)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(o * torch.sigmoid(self.o_gate_proj(x)))


class NumbatMLP(nn.Module):
    """Dense SwiGLU feed-forward block (``mlp`` in the checkpoint)."""

    def __init__(self, config: NumbatConfig, intermediate_size, linear):
        super().__init__()
        hidden = config.hidden_size
        self.gate_proj = linear(hidden, intermediate_size)
        self.up_proj = linear(hidden, intermediate_size)
        self.down_proj = linear(intermediate_size, hidden)
        self.act_fn = getattr(F, config.hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NumbatRouter(nn.Module):
    """Sigmoid router with an aux-loss-free correction bias (``gate``)."""

    def __init__(self, config: NumbatConfig):
        super().__init__()
        m = config.moe_config
        self.num_experts = m["n_routed_experts"]
        self.top_k = m["num_experts_per_tok"]
        self.n_group = m["group_limited_routing"]["n_group"]
        self.topk_group = m["group_limited_routing"]["topk_group"]
        self.norm_topk_prob = m["norm_topk_prob"]
        self.routed_scaling_factor = m["routed_scaling_factor"]
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )

    def forward(self, x):
        """Return (expert_indices, expert_weights), both (B, L, top_k)."""
        scores = torch.sigmoid(F.linear(x.float(), self.weight.float()))
        corrected = scores + self.e_score_correction_bias

        # rank expert groups by the sum of their top-2 corrected scores
        group_scores = corrected.view(
            *scores.shape[:-1], self.n_group, -1
        ).topk(2, dim=-1).values.sum(dim=-1)
        keep = group_scores.topk(self.topk_group, dim=-1).indices

        # expert -> group membership, checked against the kept groups
        group_of = torch.arange(
            self.num_experts, device=x.device
        ) // (self.num_experts // self.n_group)
        is_candidate = (keep.unsqueeze(-1)
                        == group_of.view(1, 1, -1)).any(dim=2)
        selection = corrected.masked_fill(
            ~is_candidate, float("-inf")
        ).topk(self.top_k, dim=-1).indices

        # weights are the *uncorrected* scores of the selected experts
        weights = scores.gather(-1, selection)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True) \
                * self.routed_scaling_factor
        return selection, weights


class NumbatMoE(nn.Module):
    """One shared expert plus grouped top-k routed experts (``mlp``)."""

    def __init__(self, config: NumbatConfig, quantized: bool = True):
        super().__init__()
        self.config = config
        self.gate = NumbatRouter(config)
        self.shared_experts = NumbatMLP(
            config, config.moe_config["moe_intermediate_size"], NVFP4Linear
        )
        self.experts = nn.ModuleList([
            NumbatMLP(config, config.moe_config["moe_intermediate_size"], NVFP4Linear)
            for _ in range(config.moe_config["n_routed_experts"])
        ])

    def forward(self, x):
        B, L, _ = x.shape
        selection, weights = self.gate(x)
        flat = x.view(-1, x.shape[-1])
        sel = selection.view(-1, selection.shape[-1])
        w = weights.view(-1, weights.shape[-1]).to(x.dtype)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit = sel == e
            rows = hit.any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            we = w[rows][hit[rows]].unsqueeze(-1)
            out.index_add_(0, rows, expert(flat[rows]) * we)
        return self.shared_experts(x) + out.view(B, L, -1)


class NumbatDecoderLayer(nn.Module):
    """A single decoder layer: mixer + feed-forward are both pre-norm."""

    def __init__(self, config: NumbatConfig, layer_idx: int):
        super().__init__()
        dense = layer_idx in config.dense_layer_indices
        if layer_idx in config.mla_layer_indices:
            self.self_attn = NumbatLatentAttention(config, quantized=not dense)
        else:
            self.self_attn = NumbatGatedDeltaNet(config, quantized=not dense)
        if dense:
            self.mlp = NumbatMLP(
                config, config.intermediate_size,
                lambda i, o: nn.Linear(i, o, bias=False)
            )
        else:
            self.mlp = NumbatMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, x, position_ids=None, attention_mask=None):
        x = x + self.self_attn(
            self.input_layernorm(x), position_ids, attention_mask
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class NumbatMTP(nn.Module):
    """Multi-token-prediction layer (``model.layers.<num_hidden_layers>``).

    Consumes the current-position hidden state ``h`` and the embedding
    ``e`` of the next token; predicts the token two positions ahead.
    """

    def __init__(self, config: NumbatConfig):
        super().__init__()
        hidden = config.hidden_size
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)
        self.eh_proj = nn.Linear(hidden * 2, hidden, bias=False)
        self.input_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.self_attn = NumbatLatentAttention(config, quantized=False)
        self.mlp = NumbatMLP(config, config.intermediate_size,
                               lambda i, o: nn.Linear(i, o, bias=False))

    def forward(self, h, e):
        """``h``: hidden state; ``e``: embedding of the next token."""
        z = self.eh_proj(torch.cat([self.enorm(e), self.hnorm(h)], dim=-1))
        z = z + self.self_attn(self.input_layernorm(z))
        z = z + self.mlp(self.post_attention_layernorm(z))
        return z


class NumbatPreTrainedModel(PreTrainedModel):
    """Common base: ties the modeling classes to NumbatConfig so that
    ``AutoModel.from_pretrained(..., trust_remote_code=True)`` works with
    the checkpoint's auto_map, without constraining the module tree."""
    config_class = NumbatConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        # every loadable tensor is fully specified by the checkpoint, so
        # no random init is needed or wanted here
        pass


class NumbatModel(NumbatPreTrainedModel):
    def __init__(self, config: NumbatConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [NumbatDecoderLayer(config, i)
             for i in range(config.num_hidden_layers)]
            + [NumbatMTP(config)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        h = self.embed_tokens(input_ids)
        for i in range(self.config.num_hidden_layers):
            h = self.layers[i](h, position_ids, attention_mask)
        return h


class NumbatForCausalLM(NumbatPreTrainedModel):
    def __init__(self, config: NumbatConfig):
        super().__init__(config)
        self.config = config
        self.model = NumbatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        h = self.model(input_ids, position_ids, attention_mask)
        logits = self.lm_head(self.model.norm(h))
        return CausalLMOutputWithPast(logits=logits)

    def draft(self, input_ids, position_ids=None, attention_mask=None):
        """Return the MTP layer's logits (predicts two positions ahead)."""
        h = self.model(input_ids, position_ids, attention_mask)
        e = self.model.embed_tokens(input_ids)[:, 1:]
        h = h[:, :-1]
        m = self.model.layers[-1](h, e)
        return CausalLMOutputWithPast(
            logits=self.lm_head(self.model.norm(m)))
