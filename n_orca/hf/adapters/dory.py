"""NVIDIA Nemotron-Labs Dory looped (latent-recurrent) nGPT adapter.

Dory splits its `layer_types` schedule into three contiguous groups:
Input (I), Recurrent (R) and Output (O), sized by `n_input_layers`,
`n_recurrent_layers` and `n_output_layers`. Each config "layer" is ONE
sublayer -- either attention (`sliding_attention` / `full_attention`) or an
`mlp` -- so the 4B checkpoint's 16/40/16 split is 8/20/8 attn+MLP blocks.

The R group is applied `n_recurrent_loops` times with shared weights, and the
Input block's output `h_in` is re-injected (added) at the start of every loop:

    s_0 = I(emb(x));  s_k = R(s_{k-1} + h_in);  logits = H(O(s_K))

n-orca graphs are DAGs, so the loop is UNROLLED: each loop becomes its own
`recurrent_loop_<k>` group of layers. Those copies share weights in the real
model; n-orca has no weight-tying notion, so parameter totals count the
recurrent block once per loop.

Each sublayer follows the normalized-GPT (nGPT) update, kept on the
hypersphere:  h <- Norm(h + alpha * (Norm(sublayer(h)) - h)).  It is drawn as
sublayer -> `Add` (the alpha-scaled lerp, with h as the skip input) -> a
renormalization rendered as LayerNorm. SwiGLU is rendered as FeedForward,
as in the Llama-family adapter.
"""
from __future__ import annotations

from typing import Any

from n_orca.ast import (
    Architecture,
    FlowEdge,
    Group,
    Hyperparameter,
    Invariant,
    Layer,
    OpCall,
    Tensor,
)
from n_orca.hf.adapters.base import HfAdapter, register

_ATTN_TYPES = ("sliding_attention", "full_attention")


class DoryAdapter(HfAdapter):
    model_types = ("dory", "doryforcausallm")

    def build(self, config: dict[str, Any], *, name: str | None = None) -> Architecture:
        vocab = int(config.get("vocab_size", 131072))
        d_model = int(config.get("hidden_size", 2560))
        n_heads = int(config.get("num_attention_heads", 16))
        n_kv_heads = int(config.get("num_key_value_heads", n_heads))
        head_dim = int(config.get("head_dim") or d_model // n_heads)
        d_ff = int(config.get("intermediate_size", 7168))
        window = int(config.get("sliding_window", 1024))
        max_pos = int(config.get("max_position_embeddings", 262144))
        n_in = int(config.get("n_input_layers", 16))
        n_rec = int(config.get("n_recurrent_layers", 40))
        n_out = int(config.get("n_output_layers", 16))
        n_loops = int(config.get("n_recurrent_loops", 2))
        tied = bool(config.get("tie_word_embeddings", False))

        layer_types = list(config.get("layer_types") or _default_types(n_in + n_rec + n_out))
        if len(layer_types) != n_in + n_rec + n_out:
            raise ValueError(
                f"Dory layer_types has {len(layer_types)} entries but "
                f"n_input_layers + n_recurrent_layers + n_output_layers = {n_in + n_rec + n_out}"
            )
        if n_loops < 1:
            raise ValueError(f"Dory n_recurrent_loops must be >= 1, got {n_loops}")
        for t in layer_types:
            if t not in _ATTN_TYPES and t != "mlp":
                raise ValueError(f"unsupported Dory layer type {t!r}")

        rope = config.get("rope_parameters") or {}
        theta_swa = (rope.get("sliding_attention") or {}).get("rope_theta", 10000)
        theta_full = (rope.get("full_attention") or {}).get("rope_theta", 10000000)
        tie_note = (
            "The token embedding and LM head are tied in the checkpoint but drawn separately."
            if tied else "The token embedding and LM head are untied."
        )

        arch = Architecture(
            name=name or _derive_name(config, n_loops),
            description=(
                "NVIDIA Nemotron-Labs Dory: a looped (latent-recurrent) normalized-GPT decoder. "
                f"Input block ({n_in} sublayers) -> shared Recurrent block ({n_rec} sublayers) "
                f"applied {n_loops}x with the input-block output h_in re-injected (added) before "
                f"every loop -> Output block ({n_out} sublayers) -> LM head. Sublayers alternate "
                f"GQA attention ({window}-token sliding window, RoPE theta {theta_swa}; or global, "
                f"RoPE theta {theta_full}) and SwiGLU MLPs (rendered as FeedForward). Each sublayer "
                "uses the nGPT hypersphere update h <- Norm(h + alpha*(Norm(f(h)) - h)), drawn as "
                "f -> Add (alpha-lerp) -> renorm (rendered as LayerNorm). The recurrent loops are "
                "UNROLLED and share weights in the real model; n-orca counts each loop's parameters "
                f"separately. {tie_note} Compiler output is topology-only for GQA, RoPE, "
                "sliding-window masks and nGPT scaling."
            ),
            hyperparameters=[
                Hyperparameter("vocab_size", "int", vocab),
                Hyperparameter("d_model", "int", d_model),
                Hyperparameter("n_heads", "int", n_heads),
                Hyperparameter("n_kv_heads", "int", n_kv_heads),
                Hyperparameter("head_dim", "int", head_dim),
                Hyperparameter("d_ff", "int", d_ff),
                Hyperparameter("sliding_window", "int", window),
                Hyperparameter("max_pos", "int", max_pos),
                Hyperparameter("n_input_layers", "int", n_in),
                Hyperparameter("n_recurrent_layers", "int", n_rec),
                Hyperparameter("n_output_layers", "int", n_out),
                Hyperparameter("n_recurrent_loops", "int", n_loops),
            ],
            tensors=[Tensor("input_ids", ("B", "S"), "int64"),
                     Tensor("logits", ("B", "S", "vocab_size"), "float32")],
        )
        arch.layers.extend([
            Layer(name="input_ids", is_input=True, description="Token IDs"),
            Layer(name="tok_embed", description="Token embedding (nGPT: rows normalized)",
                  op=OpCall("Embedding", ["vocab_size", "d_model"])),
        ])
        arch.flow.append(FlowEdge("input_ids", "tok_embed", "tokens"))

        def run(prefix: str, indices: range, prev: str) -> tuple[list[str], str]:
            members: list[str] = []
            for i in indices:
                layers, edges, prev = _sublayer(prefix, i, layer_types[i], prev)
                arch.layers.extend(layers)
                arch.flow.extend(edges)
                members.extend(ly.name for ly in layers)
            return members, prev

        in_members, h_in = run("in", range(n_in), "tok_embed")
        arch.groups.append(Group("input_block", in_members))

        state = h_in  # s_0 = h_in
        for k in range(1, n_loops + 1):
            inject = f"loop{k}_inject"
            arch.layers.append(Layer(inject, description=f"Loop {k}: s~ = s_{k - 1} + h_in (input re-injection)",
                                     op=OpCall("Add", [])))
            arch.flow.extend([FlowEdge(state, inject, f"s_{k - 1}"), FlowEdge(h_in, inject, f"h_in_{k}")])
            members, state = run(f"loop{k}", range(n_in, n_in + n_rec), inject)
            arch.groups.append(Group(f"recurrent_loop_{k}", [inject, *members]))

        out_members, prev = run("out", range(n_in + n_rec, n_in + n_rec + n_out), state)
        arch.groups.append(Group("output_block", out_members))

        arch.layers.extend([
            Layer(name="lm_head", description="LM projection (nGPT logit scaling not drawn)",
                  op=OpCall("Linear", ["d_model", "vocab_size"])),
            Layer(name="logits", is_output=True),
        ])
        arch.flow.extend([FlowEdge(prev, "lm_head", "h_final"), FlowEdge("lm_head", "logits", "logits")])
        arch.invariants.append(Invariant("output_shape", "=", ("B", "S", "vocab_size")))
        return arch


def _default_types(n: int) -> list[str]:
    """Dory-1.0's schedule: attn/mlp pairs, every 4th attention global."""
    return ["mlp" if i % 2 else ("full_attention" if i % 8 == 6 else "sliding_attention") for i in range(n)]


def _sublayer(prefix: str, i: int, layer_type: str, prev: str) -> tuple[list[Layer], list[FlowEdge], str]:
    if layer_type == "mlp":
        kind, op = "mlp", OpCall("FeedForward", ["d_model", "d_ff", "0.0"])
        detail = "SwiGLU MLP, bias-free (rendered as FeedForward)"
    else:
        sliding = layer_type == "sliding_attention"
        kind = "swa" if sliding else "attn"
        op = OpCall("GroupedQueryAttention",
                    ["d_model", "n_heads", "n_kv_heads", "head_dim", "sliding_window" if sliding else "0"])
        detail = ("Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k" if sliding
                  else "Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k")
    base = f"{prefix}_{kind}_{i}"
    op_name, lerp, norm = base, f"{base}_lerp", f"{base}_norm"
    layers = [
        Layer(op_name, description=detail, op=op),
        Layer(lerp, description="nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)", op=OpCall("Add", [])),
        Layer(norm, description="Renormalize to the hypersphere (rendered as LayerNorm)", op=OpCall("LayerNorm", ["d_model"])),
    ]
    edges = [
        FlowEdge(prev, op_name, f"{base}_in"),
        FlowEdge(op_name, lerp, f"{base}_out"),
        FlowEdge(prev, lerp, f"{base}_skip"),
        FlowEdge(lerp, norm, f"{base}_mix"),
    ]
    return layers, edges, norm


def _derive_name(config: dict[str, Any], n_loops: int) -> str:
    source = str(config.get("_name_or_path") or "Dory")
    stem = source.split("/")[-1].replace("-", "_").replace(".", "_")
    return "".join(p[:1].upper() + p[1:] for p in stem.split("_") if p) + f"{n_loops}Loop"


register(DoryAdapter())
