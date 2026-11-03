"""numbat model configuration."""

from transformers import PretrainedConfig


class NumbatConfig(PretrainedConfig):
    """Configuration class for the numbat architecture.

    The architecture interleaves two mixer blocks: gated delta rules with
    a short causal convolution (``gdn_config``), and full multi-head
    latent attention with decoupled rope and a sigmoid output gate
    (``mla_config``). Feed-forward blocks are either dense SwiGLU or, for
    the MoE block, one shared expert plus routed experts with grouped
    top-48 routing (``moe_config``). Layer indices for each block are
    listed in ``dense_layer_indices`` and ``mla_layer_indices``.

    A single multi-token-prediction layer is stored after the decoder
    stack, at index ``num_hidden_layers``. It shares ``embed_tokens``,
    ``norm`` and ``lm_head`` with the trunk; ``enorm`` / ``hnorm`` /
    ``eh_proj`` implement the embedding drop and output projection
    (``mtp_config``).

    Quantized linears follow the NVFP4 quad layout: a U8 packed E2M1
    payload, an F8_E4M3 block scale per 16 input elements, and F32
    global weight/input scales. Dequantization happens in
    ``modeling_numbat``; tensors whose names match
    ``fp32_parameter_suffixes`` or ``fp32_buffer_suffixes`` must be loaded
    in FP32 regardless of the checkpoint's default dtype.
    """

    model_type = "numbat"

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=12288,
        num_hidden_layers=114,
        hidden_act="silu",
        rms_norm_eps=1e-06,
        max_position_embeddings=131072,
        tie_word_embeddings=False,
        gdn_config=None,
        mla_config=None,
        moe_config=None,
        mtp_config=None,
        dense_layer_indices=None,
        mla_layer_indices=None,
        fp32_parameter_suffixes=None,
        fp32_buffer_suffixes=None,
        **kwargs,
    ):
        self.gdn_config = gdn_config or {}
        self.mla_config = mla_config or {}
        self.moe_config = moe_config or {}
        self.mtp_config = mtp_config or {}
        self.dense_layer_indices = (
            dense_layer_indices if dense_layer_indices is not None else []
        )
        self.mla_layer_indices = (
            mla_layer_indices if mla_layer_indices is not None else []
        )
        self.fp32_parameter_suffixes = fp32_parameter_suffixes or []
        self.fp32_buffer_suffixes = fp32_buffer_suffixes or []
        self.quantization_config = kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.intermediate_size = kwargs.pop("intermediate_size", None)
        self.mtp_config.setdefault("layer_index_in_checkpoint", 114)

    @property
    def num_moe_layers(self):
        return sum(1 for i in range(self.num_hidden_layers)
                   if i not in self.dense_layer_indices)
