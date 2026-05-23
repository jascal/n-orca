"""Builders for the non-LLM "world models" used as SAE training substrates.

These cover econ-sae's three variants:

- `world_model`        — baseline 2-hidden-layer MLP (the H1 hidden layer is
                          the canonical SAE training substrate)
- `deep_world_model`   — 3+ hidden layers, otherwise identical
- `attn_world_model`   — per-agent multi-head self-attention before the MLP

These architectures are intentionally small (43-dim input, ~100-dim hidden)
and so verify and render quickly — they're useful as the "ground truth" side
of an SAE pipeline where the SAE features are scored against known structure.
"""
from __future__ import annotations

from n_orca.ast import (
    Architecture,
    Hyperparameter,
    Tensor,
    Layer,
    FlowEdge,
    OpCall,
    Invariant,
)


def world_model(
    *,
    input_dim: int = 43,
    h1_dim: int = 96,
    h2_dim: int = 48,
    out_dim: int = 23,
    name: str = "WorldModel",
) -> Architecture:
    """2-hidden-layer MLP world model. H1 is the SAE training substrate."""
    arch = Architecture(
        name=name,
        description=(
            f"Baseline world-model MLP: {input_dim} -> {h1_dim} -> {h2_dim} -> {out_dim}."
            " The first hidden layer `h1` (post-ReLU) is the canonical SAE"
            " training target — its activations are extracted per agent per"
            " period and fed into the encoder."
        ),
        hyperparameters=[
            Hyperparameter("input_dim", "int", input_dim),
            Hyperparameter("h1_dim", "int", h1_dim),
            Hyperparameter("h2_dim", "int", h2_dim),
            Hyperparameter("out_dim", "int", out_dim),
        ],
        tensors=[
            Tensor("x", ("B", "input_dim"), "float32"),
            Tensor("y", ("B", "out_dim"), "float32"),
        ],
    )
    arch.layers.extend([
        Layer(name="x", is_input=True,
              description="Concatenated [agent_state, macro_state, shock_state]"),
        Layer(name="fc1", op=OpCall("Linear", ["input_dim", "h1_dim"])),
        Layer(name="act1",
              description="ReLU on H1 — SAE training substrate is read here",
              op=OpCall("ReLU", [])),
        Layer(name="fc2", op=OpCall("Linear", ["h1_dim", "h2_dim"])),
        Layer(name="act2", op=OpCall("ReLU", [])),
        Layer(name="head", op=OpCall("Linear", ["h2_dim", "out_dim"])),
        Layer(name="y", is_output=True, description="Predicted next-period agent state"),
    ])
    arch.flow.extend([
        FlowEdge("x", "fc1", "x"),
        FlowEdge("fc1", "act1", "z1"),
        FlowEdge("act1", "fc2", "h1"),
        FlowEdge("fc2", "act2", "z2"),
        FlowEdge("act2", "head", "h2"),
        FlowEdge("head", "y", "y_hat"),
    ])
    arch.invariants.append(Invariant("output_shape", "=", ("B", "out_dim")))
    return arch


def deep_world_model(
    *,
    input_dim: int = 43,
    hidden_dims: tuple[int, ...] = (192, 128, 64),
    out_dim: int = 23,
    name: str = "DeepWorldModel",
) -> Architecture:
    """N-hidden-layer MLP world model. The first hidden layer is the SAE substrate."""
    if not hidden_dims:
        raise ValueError("deep_world_model requires at least one hidden dim")
    arch = Architecture(
        name=name,
        description=(
            f"Deep world-model MLP: {input_dim} -> "
            f"{' -> '.join(str(h) for h in hidden_dims)} -> {out_dim}."
            " The first hidden layer is the SAE training substrate;"
            " later hidden layers compress before the output head."
        ),
        hyperparameters=[
            Hyperparameter("input_dim", "int", input_dim),
            *[Hyperparameter(f"h{i+1}_dim", "int", h) for i, h in enumerate(hidden_dims)],
            Hyperparameter("out_dim", "int", out_dim),
        ],
        tensors=[
            Tensor("x", ("B", "input_dim"), "float32"),
            Tensor("y", ("B", "out_dim"), "float32"),
        ],
    )
    arch.layers.append(Layer(name="x", is_input=True))

    prev_layer = "x"
    prev_dim_token = "input_dim"
    for i, _ in enumerate(hidden_dims, start=1):
        fc = f"fc{i}"
        act = f"act{i}"
        dim_token = f"h{i}_dim"
        arch.layers.append(Layer(name=fc, op=OpCall("Linear", [prev_dim_token, dim_token])))
        arch.layers.append(Layer(name=act, op=OpCall("ReLU", []),
                                  description=(
                                      "ReLU on first hidden layer — SAE substrate"
                                      if i == 1 else None
                                  )))
        arch.flow.append(FlowEdge(prev_layer, fc, f"x{i}"))
        arch.flow.append(FlowEdge(fc, act, f"z{i}"))
        prev_layer = act
        prev_dim_token = dim_token

    arch.layers.append(Layer(name="head", op=OpCall("Linear", [prev_dim_token, "out_dim"])))
    arch.layers.append(Layer(name="y", is_output=True))
    arch.flow.append(FlowEdge(prev_layer, "head", "h_last"))
    arch.flow.append(FlowEdge("head", "y", "y_hat"))

    arch.invariants.append(Invariant("output_shape", "=", ("B", "out_dim")))
    return arch


def attn_world_model(
    *,
    input_dim: int = 43,
    embed_dim: int = 64,
    n_heads: int = 4,
    h1_dim: int = 192,
    h2_dim: int = 128,
    out_dim: int = 23,
    dropout: float = 0.0,
    name: str = "AttnWorldModel",
) -> Architecture:
    """Per-agent self-attention world model.

    Topology:
        x (B, N, input_dim)
          -> embed: Linear(input_dim, embed_dim)
          -> attn:  MultiHeadAttention(embed_dim, n_heads)
          -> add:   residual
          -> ln:    LayerNorm
          -> fc1:   Linear(embed_dim, h1_dim)
          -> act1:  ReLU              [SAE substrate]
          -> fc2:   Linear(h1_dim, h2_dim)
          -> act2:  ReLU
          -> head:  Linear(h2_dim, out_dim)
          -> y (B, N, out_dim)
    """
    arch = Architecture(
        name=name,
        description=(
            f"Per-agent attention world model: embed -> MHA -> +residual -> LN"
            f" -> MLP({h1_dim}->{h2_dim}->{out_dim})."
            " Multi-head self-attention is applied across the agent dimension"
            " each period, so each agent's prediction can attend to every"
            " other agent's state."
        ),
        hyperparameters=[
            Hyperparameter("input_dim", "int", input_dim),
            Hyperparameter("embed_dim", "int", embed_dim),
            Hyperparameter("n_heads", "int", n_heads),
            Hyperparameter("h1_dim", "int", h1_dim),
            Hyperparameter("h2_dim", "int", h2_dim),
            Hyperparameter("out_dim", "int", out_dim),
            Hyperparameter("dropout", "float", dropout),
        ],
        tensors=[
            Tensor("x", ("B", "N", "input_dim"), "float32"),
            Tensor("y", ("B", "N", "out_dim"), "float32"),
        ],
    )
    arch.layers.extend([
        Layer(name="x", is_input=True,
              description="Per-agent state — N agents, input_dim features each"),
        Layer(name="embed", op=OpCall("Linear", ["input_dim", "embed_dim"])),
        Layer(name="attn",
              description="Multi-head self-attention across agents",
              op=OpCall("MultiHeadAttention", ["embed_dim", "n_heads", "dropout"])),
        Layer(name="add_attn", op=OpCall("Add", [])),
        Layer(name="ln", op=OpCall("LayerNorm", ["embed_dim"])),
        Layer(name="fc1", op=OpCall("Linear", ["embed_dim", "h1_dim"])),
        Layer(name="act1",
              description="ReLU on H1 — SAE training substrate is read here",
              op=OpCall("ReLU", [])),
        Layer(name="fc2", op=OpCall("Linear", ["h1_dim", "h2_dim"])),
        Layer(name="act2", op=OpCall("ReLU", [])),
        Layer(name="head", op=OpCall("Linear", ["h2_dim", "out_dim"])),
        Layer(name="y", is_output=True),
    ])
    arch.flow.extend([
        FlowEdge("x", "embed", "x"),
        FlowEdge("embed", "attn", "tokens"),
        FlowEdge("attn", "add_attn", "attn_out"),
        FlowEdge("embed", "add_attn", "tok_skip"),
        FlowEdge("add_attn", "ln", "r"),
        FlowEdge("ln", "fc1", "r_n"),
        FlowEdge("fc1", "act1", "z1"),
        FlowEdge("act1", "fc2", "h1"),
        FlowEdge("fc2", "act2", "z2"),
        FlowEdge("act2", "head", "h2"),
        FlowEdge("head", "y", "y_hat"),
    ])
    arch.invariants.append(Invariant("output_shape", "=", ("B", "N", "out_dim")))
    return arch
