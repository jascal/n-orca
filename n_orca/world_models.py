"""Builders for the non-LLM "world models" used as SAE training substrates.

These cover econ-sae's variants (including temporal for regime features) and Cosmos 3-style MoT:

- `world_model`        — baseline 2-hidden-layer MLP (the H1 hidden layer is
                          the canonical SAE training substrate)
- `deep_world_model`   — 3+ hidden layers, otherwise identical
- `attn_world_model`   — per-agent multi-head self-attention before the MLP
- `temporal_world_model` — attn + explicit state carry (hidden tensor) for cross-period context
- `mot_denoise_step` — MoT dual-tower (AR reasoner causal + DM generator bidirectional + timestep) per-denoise-step DAG for Cosmos 3-style diffusion/multimodal world models (external schedule like temporal). Enables physics-rich SAE GT.

These architectures are intentionally small (43-dim input, ~100-dim hidden)
and so verify and render quickly — they're useful as the "ground truth" side
of an SAE pipeline where the SAE features are scored against known structure.
The temporal variant enables encoding regime/windowed features per econ-sae research.
The mot variant supports diffusion-conditioned dual-stream patterns per subagent Cosmos investigation.
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


def temporal_world_model(
    *,
    input_dim: int = 43,
    embed_dim: int = 64,
    n_heads: int = 4,
    gru_hidden: int = 128,
    h1_dim: int = 192,
    h2_dim: int = 128,
    out_dim: int = 23,
    name: str = "TemporalWorldModel",
) -> Architecture:
    """Temporal world model: per-period cross-agent attention + explicit state carry for cross-period context.

    This implements a per-step graph (one time step / period) with an additional 'hidden' state tensor
    that can be carried across periods by the caller (e.g. econ-sae trainer unrolling the trajectory).

    Based on econ-sae TemporalWorldModel (attn + GRU) and the design in the OpenSpec change:
    - Same attn + residual + LN + MLP structure as attn_world_model for per-period cross-agent context.
    - Explicit 'hidden_in' / 'hidden_out' tensors (shape (B, N, gru_hidden)) for temporal state.
    - Simple linear `hidden_update` as starting point. Full recurrent logic (e.g. actual GRU cell with
      reset/update gates) is planned for a follow-up refinement (may live in the PyTorch emission or a
      future builder variant).
    - The h1 (post-attn) remains the SAE substrate; state allows encoding regime features over time.

    The Architecture is a DAG; recurrence is handled externally by carrying hidden across steps.
    This matches the "per-step-with-state" representation in the proposal and design.md.
    """
    arch = Architecture(
        name=name,
        description=(
            f"Temporal world-model (per-period attn + state carry, gru_hidden={gru_hidden}). "
            "Extends attn_world_model with explicit hidden state tensors for cross-period context "
            "(enables regime/windowed feature recovery per econ-sae). h1 is the SAE substrate. "
            "Caller manages unrolling/carrying hidden across T periods."
        ),
        hyperparameters=[
            Hyperparameter("input_dim", "int", input_dim),
            Hyperparameter("embed_dim", "int", embed_dim),
            Hyperparameter("n_heads", "int", n_heads),
            Hyperparameter("gru_hidden", "int", gru_hidden),
            Hyperparameter("h1_dim", "int", h1_dim),
            Hyperparameter("h2_dim", "int", h2_dim),
            Hyperparameter("out_dim", "int", out_dim),
        ],
        tensors=[
            Tensor("x", ("B", "N", "input_dim"), "float32"),
            Tensor("y", ("B", "N", "out_dim"), "float32"),
            Tensor("hidden_in", ("B", "N", "gru_hidden"), "float32"),
            Tensor("hidden_out", ("B", "N", "gru_hidden"), "float32"),
        ],
    )
    arch.layers.append(Layer(name="x", is_input=True))
    arch.layers.append(Layer(name="hidden_in", is_input=True))

    # Per-period cross-agent attention path (copied/extended from attn_world_model)
    arch.layers.append(Layer(name="embed", op=OpCall("Linear", ["input_dim", "embed_dim"])))
    arch.layers.append(Layer(name="attn", op=OpCall("MultiHeadAttention", ["embed_dim", "n_heads", "0.0"])))
    arch.layers.append(Layer(name="add_attn", op=OpCall("Add", [])))
    arch.layers.append(Layer(name="ln", op=OpCall("LayerNorm", ["embed_dim"])))
    arch.layers.append(Layer(name="fc1", op=OpCall("Linear", ["embed_dim", "h1_dim"])))
    arch.layers.append(Layer(name="act1", op=OpCall("ReLU", []),
                              description="ReLU on H1 — SAE substrate (post temporal context)"))
    arch.layers.append(Layer(name="fc2", op=OpCall("Linear", ["h1_dim", "h2_dim"])))
    arch.layers.append(Layer(name="act2", op=OpCall("ReLU", [])))
    arch.layers.append(Layer(name="head", op=OpCall("Linear", ["h2_dim", "out_dim"])))
    arch.layers.append(Layer(name="y", is_output=True))

    # Explicit state carry path (simple linear hidden_update as initial temporal mechanism;
    # full GRU planned as follow-up per design)
    arch.layers.append(Layer(name="hidden_update", op=OpCall("Linear", ["gru_hidden", "gru_hidden"])))
    arch.layers.append(Layer(name="hidden_out", is_output=True))

    # Flow for main prediction path (x -> y via attn)
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

    # Flow for state carry (hidden_in -> hidden_out)
    arch.flow.extend([
        FlowEdge("hidden_in", "hidden_update", "h_in"),
        FlowEdge("hidden_update", "hidden_out", "h_out"),
    ])

    # Note: no single output_shape invariant because we have y (out_dim) and hidden_out (gru_hidden)
    # The main y output shape is enforced by the head layer.
    return arch


def mot_denoise_step(
    *,
    d_model: int = 64,
    n_heads: int = 4,
    timestep_dim: int = 128,
    h1_dim: int = 128,
    h2_dim: int = 64,
    name: str = "MoTDenoiseStep",
) -> Architecture:
    """One diffusion denoising step in MoT (AR reasoner context + DM full joint attn + timestep cond).

    Mirrors Cosmos 3 dual-tower (independent LN/MLP per stream; shared joint attn per subagent report).
    Per-step DAG (external loop over diffusion schedule, like temporal hidden carry).
    AR causal self-attn only; DM attends concat(AR,DM) for conditioning.
    Timestep injected to DM path (simple Linear projection; sinusoid in full DiT).
    Uses existing ops (Linear, LayerNorm, MultiHeadAttention, Add, ReLU); joint attn
    approximated here via structure + docstring (full DualStreamJointAttention op in follow-up slice).
    Toy dims for verification; real via HF/diffusers in consumers.

    Tensors use (B, S_ar + S_dm, d_model) conceptually; here split for clarity.
    """
    # AR/DM towers use separate MHA for now; true DualStreamJointAttention op (with concat/mask special-case)
    # planned for 2.2 per design.
    arch = Architecture(
        name=name,
        description=(
            f"Cosmos 3-style MoT single denoise step (AR reasoner + DM generator). "
            f"d_model={d_model}, n_heads={n_heads}. "
            "AR: causal self-attn only (reason first). "
            "DM: bidirectional joint over AR+DM + timestep cond (generate). "
            "External loop for full T steps (like temporal state carry). "
            "See OpenSpec add-cosmos-mot-world-models + subagent report."
        ),
        hyperparameters=[
            Hyperparameter("d_model", "int", d_model),
            Hyperparameter("n_heads", "int", n_heads),
            Hyperparameter("timestep_dim", "int", timestep_dim),
            Hyperparameter("h1_dim", "int", h1_dim),
            Hyperparameter("h2_dim", "int", h2_dim),
        ],
        tensors=[
            Tensor("ar_x", ("B", "S_ar", "d_model"), "float32"),
            Tensor("ar_out", ("B", "S_ar", "d_model"), "float32"),
            Tensor("dm_x_noisy", ("B", "S_dm", "d_model"), "float32"),
            Tensor("t", ("B",), "float32"),
            Tensor("dm_x_denoised", ("B", "S_dm", "d_model"), "float32"),
            Tensor("ts_out", ("d_model",), "float32"),
        ],
    )
    arch.layers.append(Layer(name="ar_x", is_input=True, description="AR tokens (reasoner prefix)"))
    arch.layers.append(Layer(name="dm_x_noisy", is_input=True, description="DM noisy latents (generator)"))
    arch.layers.append(Layer(name="t", is_input=True, description="Diffusion timestep (scalar per batch)"))

    # AR reasoner path (causal only)
    arch.layers.append(Layer(name="ar_ln", op=OpCall("LayerNorm", ["d_model"])))
    arch.layers.append(Layer(name="ar_mha", op=OpCall("MultiHeadAttention", ["d_model", "n_heads", "0.0"]),
                              description="AR causal self-attn (reasoning)"))
    arch.layers.append(Layer(name="ar_add", op=OpCall("Add", [])))
    arch.layers.append(Layer(name="ar_out", is_output=True, description="AR output (reasoning result)"))

    # DM generator path + timestep + joint (joint via concat in full impl; here DM self + cross note)
    arch.layers.append(Layer(name="ts_embed", op=OpCall("Linear", ["timestep_dim", "d_model"]),
                              description="Timestep projection (sinusoidal in DiT; Linear here for toy)"))
    arch.layers.append(Layer(name="dm_ln", op=OpCall("LayerNorm", ["d_model"])))
    arch.layers.append(Layer(name="dm_mha", op=OpCall("MultiHeadAttention", ["d_model", "n_heads", "0.0"]),
                              description="DM joint attn (bidir over AR+DM in full MoT; self here + cross note)"))
    arch.layers.append(Layer(name="dm_add", op=OpCall("Add", [])))
    arch.layers.append(Layer(name="dm_fc1", op=OpCall("Linear", ["d_model", "h1_dim"])))
    arch.layers.append(Layer(name="dm_act1", op=OpCall("ReLU", [])))
    arch.layers.append(Layer(name="dm_fc2", op=OpCall("Linear", ["h1_dim", "h2_dim"])))
    arch.layers.append(Layer(name="dm_act2", op=OpCall("ReLU", [])))
    arch.layers.append(Layer(name="dm_head", op=OpCall("Linear", ["h2_dim", "d_model"])))
    arch.layers.append(Layer(name="dm_x_denoised", is_output=True))
    arch.layers.append(Layer(name="ts_out", is_output=True, description="timestep cond output (injected in forward)"))

    # Flows (AR causal self; DM + ts + joint note in doc)
    arch.flow.extend([
        # AR path (with residual skip like attn)
        FlowEdge("ar_x", "ar_ln", "ar"),
        FlowEdge("ar_ln", "ar_mha", "ar_tok"),
        FlowEdge("ar_mha", "ar_add", "ar_a"),
        FlowEdge("ar_ln", "ar_add", "ar_skip"),  # residual
        FlowEdge("ar_add", "ar_out", "ar_out"),
        # DM + ts (ts added to dm path; joint note)
        FlowEdge("t", "ts_embed", "ts"),
        FlowEdge("dm_x_noisy", "dm_ln", "dm"),
        FlowEdge("dm_ln", "dm_mha", "dm_tok"),
        FlowEdge("dm_mha", "dm_add", "dm_a"),
        FlowEdge("dm_ln", "dm_add", "dm_skip"),  # residual
        # DM MLP
        FlowEdge("dm_add", "dm_fc1", "d_r"),
        FlowEdge("dm_fc1", "dm_act1", "d_z1"),
        FlowEdge("dm_act1", "dm_fc2", "d_h1"),
        FlowEdge("dm_fc2", "dm_act2", "d_z2"),
        FlowEdge("dm_act2", "dm_head", "d_h2"),
        FlowEdge("dm_head", "dm_x_denoised", "denoised"),
        FlowEdge("ts_embed", "ts_out", "ts_out"),
    ])

    # Note: no single output_shape invariant (dual paths, main DM out)
    # AR causal integrity + DM joint cond documented in description.
    return arch
