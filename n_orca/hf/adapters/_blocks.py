"""Reusable transformer-block emission helpers.

These produce `Layer` and `FlowEdge` lists for the common patterns
(pre-norm decoder block, post-norm encoder block). The adapters wire them
together at the document level.
"""
from __future__ import annotations

from n_orca.ast import Layer, FlowEdge, OpCall


def pre_norm_decoder_block(
    *,
    index: int,
    prev_layer: str,
    d_model: str,
    n_heads: str,
    d_ff: str,
    dropout: str,
    norm_op: str = "LayerNorm",
    ffn_op: str = "FeedForward",
) -> tuple[list[Layer], list[FlowEdge], str]:
    """Emit a single pre-norm decoder block (GPT-2 / LLaMA style).

    Topology:
        prev -> LN -> attn -> add(prev, attn)
             -> LN -> ffn  -> add(prev', ffn) -> next

    Returns (new_layers, new_edges, name_of_block_output).
    """
    ln_1 = f"ln_1_{index}"
    attn = f"attn_{index}"
    add_attn = f"add_attn_{index}"
    ln_2 = f"ln_2_{index}"
    ffn = f"ffn_{index}"
    add_ffn = f"add_ffn_{index}"

    layers = [
        Layer(name=ln_1, op=OpCall(norm_op, [d_model])),
        Layer(name=attn, op=OpCall("MultiHeadAttention", [d_model, n_heads, dropout])),
        Layer(name=add_attn, op=OpCall("Add", [])),
        Layer(name=ln_2, op=OpCall(norm_op, [d_model])),
        Layer(name=ffn, op=OpCall(ffn_op, [d_model, d_ff, dropout])),
        Layer(name=add_ffn, op=OpCall("Add", [])),
    ]
    edges = [
        FlowEdge(source=prev_layer, target=ln_1, tensor=f"h_{index}"),
        FlowEdge(source=ln_1, target=attn, tensor=f"h_{index}_n1"),
        FlowEdge(source=attn, target=add_attn, tensor=f"attn_out_{index}"),
        FlowEdge(source=prev_layer, target=add_attn, tensor=f"h_{index}_skip1"),
        FlowEdge(source=add_attn, target=ln_2, tensor=f"r_{index}"),
        FlowEdge(source=ln_2, target=ffn, tensor=f"r_{index}_n2"),
        FlowEdge(source=ffn, target=add_ffn, tensor=f"ffn_out_{index}"),
        FlowEdge(source=add_attn, target=add_ffn, tensor=f"r_{index}_skip2"),
    ]
    return layers, edges, add_ffn


def post_norm_encoder_block(
    *,
    index: int,
    prev_layer: str,
    d_model: str,
    n_heads: str,
    d_ff: str,
    dropout: str,
) -> tuple[list[Layer], list[FlowEdge], str]:
    """Emit a single post-norm encoder block (BERT style).

    Topology:
        prev -> attn -> add(prev, attn) -> LN
             -> ffn  -> add(prev', ffn) -> LN -> next

    Returns (new_layers, new_edges, name_of_block_output).
    """
    attn = f"attn_{index}"
    add_attn = f"add_attn_{index}"
    ln_1 = f"ln_1_{index}"
    ffn = f"ffn_{index}"
    add_ffn = f"add_ffn_{index}"
    ln_2 = f"ln_2_{index}"

    layers = [
        Layer(name=attn, op=OpCall("MultiHeadAttention", [d_model, n_heads, dropout])),
        Layer(name=add_attn, op=OpCall("Add", [])),
        Layer(name=ln_1, op=OpCall("LayerNorm", [d_model])),
        Layer(name=ffn, op=OpCall("FeedForward", [d_model, d_ff, dropout])),
        Layer(name=add_ffn, op=OpCall("Add", [])),
        Layer(name=ln_2, op=OpCall("LayerNorm", [d_model])),
    ]
    edges = [
        FlowEdge(source=prev_layer, target=attn, tensor=f"h_{index}"),
        FlowEdge(source=attn, target=add_attn, tensor=f"attn_out_{index}"),
        FlowEdge(source=prev_layer, target=add_attn, tensor=f"h_{index}_skip1"),
        FlowEdge(source=add_attn, target=ln_1, tensor=f"r_{index}"),
        FlowEdge(source=ln_1, target=ffn, tensor=f"r_{index}_n1"),
        FlowEdge(source=ffn, target=add_ffn, tensor=f"ffn_out_{index}"),
        FlowEdge(source=ln_1, target=add_ffn, tensor=f"r_{index}_skip2"),
        FlowEdge(source=add_ffn, target=ln_2, tensor=f"r2_{index}"),
    ]
    return layers, edges, ln_2
