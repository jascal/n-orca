"""Liquid AI LFM3 tri-hybrid decoder adapter.

The emitted document preserves LFM3's config-defined conv/sliding/global
schedule.  RMSNorm and SwiGLU remain n-orca's existing LayerNorm/FeedForward
approximations; the two distinctive mixing operators have semantic op names.
"""
from __future__ import annotations

import warnings
from typing import Any

from n_orca.ast import Architecture, FlowEdge, Hyperparameter, Invariant, Layer, OpCall, Tensor
from n_orca.hf.adapters.base import HfAdapter, register


class Lfm3Adapter(HfAdapter):
    model_types = ("lfm3", "lfm3forcausallm")

    def build(self, config: dict[str, Any], *, name: str | None = None) -> Architecture:
        vocab = int(config.get("vocab_size", 128000))
        d_model = int(config.get("hidden_size", 2048))
        n_layer = int(config.get("num_hidden_layers", 24))
        n_heads = int(config.get("num_attention_heads", 16))
        n_kv_heads = int(config.get("num_key_value_heads", n_heads))
        d_ff = int(config.get("intermediate_size", 7680))
        kernel = int(config.get("conv_L_cache", 7))
        window = int(config.get("sliding_window", 256))
        max_pos = int(config.get("max_position_embeddings", 128000))
        declared_types = config.get("layer_types")
        if declared_types is None:
            warnings.warn(
                "LFM3 config has no layer_types; using the documented default "
                "tri-hybrid schedule",
                UserWarning,
                stacklevel=2,
            )
            layer_types = _default_types(n_layer)
        else:
            layer_types = list(declared_types)

        arch = Architecture(
            name=name or _derive_name(config, n_layer),
            description=(
                "Liquid AI LFM3 causal tri-hybrid decoder. Each block uses sandwich "
                "RMSNorm (rendered as LayerNorm), an exact config-ordered gated causal "
                "short convolution, sliding-window GQA, or global GQA operator, then a "
                "SwiGLU-like FFN (rendered as FeedForward). LFM3 ties the token embedding "
                "and LM head in its checkpoint; n-orca displays them as separate layers and "
                "does not share their generated PyTorch weights. Compiler output is topology-only "
                "for LFM3-specific kernels, GQA gates, RoPE, and masks."
            ),
            hyperparameters=[
                Hyperparameter("vocab_size", "int", vocab),
                Hyperparameter("d_model", "int", d_model),
                Hyperparameter("n_layer", "int", n_layer),
                Hyperparameter("n_heads", "int", n_heads),
                Hyperparameter("n_kv_heads", "int", n_kv_heads),
                Hyperparameter("d_ff", "int", d_ff),
                Hyperparameter("conv_kernel", "int", kernel),
                Hyperparameter("sliding_window", "int", window),
                Hyperparameter("max_pos", "int", max_pos),
            ],
            tensors=[Tensor("input_ids", ("B", "S"), "int64"),
                     Tensor("logits", ("B", "S", "vocab_size"), "float32")],
        )
        arch.layers.extend([
            Layer(name="input_ids", is_input=True, description="Token IDs"),
            Layer(name="tok_embed", description="Token embedding (tied to LM head in checkpoint; shown separately)", op=OpCall("Embedding", ["vocab_size", "d_model"])),
        ])
        arch.flow.append(FlowEdge("input_ids", "tok_embed", "tokens"))

        prev = "tok_embed"
        for i, layer_type in enumerate(layer_types):
            layers, edges, prev = _lfm3_block(i, prev, layer_type)
            arch.layers.extend(layers)
            arch.flow.extend(edges)

        arch.layers.extend([
            Layer(name="final_norm", description="Final RMSNorm (rendered as LayerNorm)", op=OpCall("LayerNorm", ["d_model"])),
            Layer(name="lm_head", description="LM projection (tied to embedding in checkpoint; shown separately)", op=OpCall("Linear", ["d_model", "vocab_size"])),
            Layer(name="logits", is_output=True),
        ])
        arch.flow.extend([FlowEdge(prev, "final_norm", "h_final"), FlowEdge("final_norm", "lm_head", "normed"), FlowEdge("lm_head", "logits", "logits")])
        arch.invariants.append(Invariant("output_shape", "=", ("B", "S", "vocab_size")))
        return arch


def _default_types(n_layer: int) -> list[str]:
    return ["full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv" for i in range(n_layer)]


def _lfm3_block(i: int, prev: str, layer_type: str) -> tuple[list[Layer], list[FlowEdge], str]:
    operator_name = {"conv": "GatedCausalShortConv", "sliding_attention": "HeadGatedGroupedQueryAttention", "full_attention": "HeadGatedGroupedQueryAttention"}.get(layer_type)
    if operator_name is None:
        raise ValueError(f"unsupported LFM3 layer type {layer_type!r}")
    args = ["d_model", "conv_kernel"] if layer_type == "conv" else ["d_model", "n_heads", "n_kv_heads", "sliding_window" if layer_type == "sliding_attention" else "0"]
    detail = (
        "Gated causal depthwise short convolution (kernel 7)" if layer_type == "conv" else
        "Head-gated grouped-query causal attention; 256-token sliding window; full RoPE" if layer_type == "sliding_attention" else
        "Head-gated grouped-query causal attention; global context; half-head partial RoPE"
    )
    names = {k: f"{k}_{i}" for k in ("op_norm", "operator", "post_op_norm", "add_operator", "ffn_norm", "ffn", "post_ffn_norm", "add_ffn")}
    layers = [
        Layer(names["op_norm"], description="Operator RMSNorm (rendered as LayerNorm)", op=OpCall("LayerNorm", ["d_model"])),
        Layer(names["operator"], description=detail, op=OpCall(operator_name, args)),
        Layer(names["post_op_norm"], description="Sandwich post-operator RMSNorm", op=OpCall("LayerNorm", ["d_model"])),
        Layer(names["add_operator"], description="Operator residual", op=OpCall("Add", [])),
        Layer(names["ffn_norm"], description="FFN RMSNorm (rendered as LayerNorm)", op=OpCall("LayerNorm", ["d_model"])),
        Layer(names["ffn"], description="LFM3 SwiGLU MLP (rendered as FeedForward)", op=OpCall("FeedForward", ["d_model", "d_ff", "0.0"])),
        Layer(names["post_ffn_norm"], description="Sandwich post-FFN RMSNorm", op=OpCall("LayerNorm", ["d_model"])),
        Layer(names["add_ffn"], description="FFN residual", op=OpCall("Add", [])),
    ]
    e = FlowEdge
    edges = [e(prev, names["op_norm"], f"h_{i}"), e(names["op_norm"], names["operator"], f"op_in_{i}"), e(names["operator"], names["post_op_norm"], f"op_out_{i}"), e(names["post_op_norm"], names["add_operator"], f"op_normed_{i}"), e(prev, names["add_operator"], f"op_skip_{i}"), e(names["add_operator"], names["ffn_norm"], f"residual_{i}"), e(names["ffn_norm"], names["ffn"], f"ffn_in_{i}"), e(names["ffn"], names["post_ffn_norm"], f"ffn_out_{i}"), e(names["post_ffn_norm"], names["add_ffn"], f"ffn_normed_{i}"), e(names["add_operator"], names["add_ffn"], f"ffn_skip_{i}")]
    return layers, edges, names["add_ffn"]


def _derive_name(config: dict[str, Any], n_layer: int) -> str:
    source = str(config.get("_name_or_path") or "LFM3")
    stem = source.split("/")[-1].replace("-", "_").replace(".", "_")
    return "".join(p[:1].upper() + p[1:] for p in stem.split("_") if p) + f"{n_layer}L"


register(Lfm3Adapter())
