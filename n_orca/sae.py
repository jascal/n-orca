"""Builders for sparse-autoencoder architectures.

The variants below match the SAE families implemented in `sm-sae`,
`econ-sae`, `bio-sae`, and consumed by `polygram`:

- **Top-k SAE**          — `topk_sae`     (Makhzani-style; structural sparsity via top-k mask)
- **L1 SAE**             — `l1_sae`       (vanilla; ReLU + L1 penalty in loss)
- **JumpReLU SAE**       — `jumprelu_sae` (Rajamanoharan et al. 2024; learnable per-feature threshold, L0 penalty)
- **Attention-prefixed Top-k SAE** — `attn_topk_sae` (MultiHeadAttention block
                                       prepended to the SAE encoder; 3D input
                                       `(B, T, input_dim)`. Mirrors econ-sae's
                                       AttnWorldModel pattern — its Phase 1.6
                                       cross-agent attention was the biggest
                                       single architectural unlock for the
                                       conjunctive feature tier. Bio-sae's
                                       motif-recovery-architecture-limit
                                       memo predicts the same shape of
                                       capability lift for protein motifs.)
- **Gated SAE**                     — `gated_sae` (parallel magnitude + sigmoid-
                                       gated projections, element-wise multi-
                                       plied; allows graded non-binary feature
                                       contributions. Used by econ-sae Phase 3
                                       for plateaued features that hard
                                       activations couldn't reach.)
- **Supervised Top-k SAE**          — `supervised_topk_sae` (TopK SAE with an
                                       auxiliary per-label classifier head off
                                       the sparse latents. Joint reconstruction
                                       + BCE classifier loss. Two outputs:
                                       `x_hat` and `y_logits`. Mirrors econ-sae
                                       Phase 5.1 — supervised regime SAE lifted
                                       regime mAUC 0.885 → 0.972, the biggest
                                       single-phase jump in econ-sae's journey.)

Each builder returns an `Architecture` AST that verifies clean and compiles
to both Mermaid and PyTorch. Sparsity penalties (L1, L0) are *loss* terms
that live outside the architecture topology — n-orca captures the forward
graph only, not the loss.

Tensor shape conventions: the linear-encoder variants (topk / l1 / jumprelu)
take 2D inputs `(B, input_dim)`. The `attn_topk_sae` builder takes 3D inputs
`(B, T, input_dim)` because attention needs a sequence dimension; consumers
must feed per-token / per-residue / per-agent activations, NOT pooled ones.
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


def topk_sae(
    *,
    input_dim: int = 768,
    n_features: int = 16384,
    k: int = 64,
    name: str = "TopKSae",
    tied_decoder: bool = False,
) -> Architecture:
    """Top-k sparse autoencoder.

    `x -> Linear(input_dim, n_features) -> ReLU -> TopK(k) -> Linear(n_features, input_dim) -> x_hat`

    Sparsity is enforced structurally by `TopK` — no penalty term needed
    in the loss. The `k` largest pre-activations survive per sample.
    """
    arch = _sae_skeleton(
        name=name,
        input_dim=input_dim,
        n_features=n_features,
        variant_description=(
            f"Top-k sparse autoencoder (k={k})."
            " Sparsity is enforced structurally — every sample keeps exactly"
            " `k` features active. No L1 / L0 penalty needed at the loss layer."
        ),
        extra_hps=[Hyperparameter("k", "int", k)],
    )

    arch.layers.extend([
        Layer(name="encoder", op=OpCall("Linear", ["input_dim", "n_features"])),
        Layer(name="relu", op=OpCall("ReLU", [])),
        Layer(name="topk",
              description="Keep only the k largest features per sample",
              op=OpCall("TopK", ["k"])),
        Layer(name="decoder", op=OpCall("Linear", ["n_features", "input_dim"])),
    ])
    arch.flow.extend([
        FlowEdge("x", "encoder", "x"),
        FlowEdge("encoder", "relu", "z_pre"),
        FlowEdge("relu", "topk", "z_relu"),
        FlowEdge("topk", "decoder", "z_sparse"),
        FlowEdge("decoder", "x_hat", "x_hat"),
    ])
    _close_sae(arch, tied=tied_decoder)
    return arch


def l1_sae(
    *,
    input_dim: int = 768,
    n_features: int = 16384,
    l1_coeff: float = 1e-2,
    name: str = "L1Sae",
    tied_decoder: bool = False,
) -> Architecture:
    """Vanilla L1 sparse autoencoder.

    `x -> Linear -> ReLU -> Linear -> x_hat`

    The L1 sparsity penalty `l1_coeff * |z|.sum()` is applied to the ReLU
    output at training time. It is a *loss* term — not part of the forward
    topology that n-orca renders.
    """
    arch = _sae_skeleton(
        name=name,
        input_dim=input_dim,
        n_features=n_features,
        variant_description=(
            f"Vanilla L1 sparse autoencoder (l1_coeff={l1_coeff})."
            " ReLU pre-activation gives non-negative features; sparsity is"
            " encouraged via an L1 penalty `l1_coeff * |z|.sum()` added to"
            " the reconstruction loss (the penalty is not part of the"
            " forward graph rendered here)."
        ),
        extra_hps=[Hyperparameter("l1_coeff", "float", l1_coeff)],
    )

    arch.layers.extend([
        Layer(name="encoder", op=OpCall("Linear", ["input_dim", "n_features"])),
        Layer(name="relu", op=OpCall("ReLU", [])),
        Layer(name="decoder", op=OpCall("Linear", ["n_features", "input_dim"])),
    ])
    arch.flow.extend([
        FlowEdge("x", "encoder", "x"),
        FlowEdge("encoder", "relu", "z_pre"),
        FlowEdge("relu", "decoder", "z"),
        FlowEdge("decoder", "x_hat", "x_hat"),
    ])
    _close_sae(arch, tied=tied_decoder)
    return arch


def jumprelu_sae(
    *,
    input_dim: int = 768,
    n_features: int = 16384,
    theta_init: float = 0.1,
    l0_coeff: float = 5e-3,
    name: str = "JumpReLUSae",
    tied_decoder: bool = False,
) -> Architecture:
    """JumpReLU sparse autoencoder.

    `x -> Linear -> JumpReLU(n_features, theta_init) -> Linear -> x_hat`

    Each feature has a learnable hard threshold `theta_j`; the gate is
    `x_j * 1{x_j > theta_j}`. A straight-through estimator routes gradients
    through `theta` during training. The L0 (count) penalty is applied to
    the gate as a loss term — not part of the forward graph here.
    """
    arch = _sae_skeleton(
        name=name,
        input_dim=input_dim,
        n_features=n_features,
        variant_description=(
            f"JumpReLU sparse autoencoder (theta_init={theta_init},"
            f" l0_coeff={l0_coeff})."
            " Each feature has a learnable threshold `theta_j`; the gate"
            " `x_j * 1{x_j > theta_j}` is hard-thresholded with a straight-"
            "through estimator. Sparsity is encouraged via an L0 count"
            " penalty (not part of the forward graph rendered here)."
        ),
        extra_hps=[
            Hyperparameter("theta_init", "float", theta_init),
            Hyperparameter("l0_coeff", "float", l0_coeff),
        ],
    )

    arch.layers.extend([
        Layer(name="encoder", op=OpCall("Linear", ["input_dim", "n_features"])),
        Layer(name="jumprelu",
              description="Per-feature learnable hard threshold + STE gate",
              op=OpCall("JumpReLU", ["n_features", "theta_init"])),
        Layer(name="decoder", op=OpCall("Linear", ["n_features", "input_dim"])),
    ])
    arch.flow.extend([
        FlowEdge("x", "encoder", "x"),
        FlowEdge("encoder", "jumprelu", "z_pre"),
        FlowEdge("jumprelu", "decoder", "z_gated"),
        FlowEdge("decoder", "x_hat", "x_hat"),
    ])
    _close_sae(arch, tied=tied_decoder)
    return arch


def attn_topk_sae(
    *,
    input_dim: int = 320,
    n_features: int = 1024,
    k: int = 64,
    n_heads: int = 4,
    attn_dropout: float = 0.0,
    name: str = "AttnTopKSae",
    tied_decoder: bool = False,
) -> Architecture:
    """Attention-prefixed Top-k sparse autoencoder.

    Topology:
        x (B, T, input_dim)
          -> attn:  MultiHeadAttention(input_dim, n_heads, attn_dropout) over T
          -> add:   residual
          -> ln:    LayerNorm(input_dim)
          -> encoder: Linear(input_dim, n_features)
          -> relu:  ReLU
          -> topk:  TopK(k)
          -> decoder: Linear(n_features, input_dim)
          -> x_hat (B, T, input_dim)

    The attention prefix gives each position cross-sequence context BEFORE the
    SAE encoder sees it. Mirrors econ-sae's AttnWorldModel structural unlock
    for conjunctive features (`econ-sae/scripts/visualize.py` Phase 1.6: cross-
    agent attention lifted conjunctive mAUC 0.84 → 0.97 — the biggest single
    architectural unlock in econ-sae's 6-phase journey). For bio-sae, the
    analog target is motif recovery — per-residue SAEs structurally cannot
    represent "this residue is part of a 5-residue HTH motif" without
    cross-residue context (see `bio-sae/.../motif-recovery-architecture-limit`).

    Note vs the 2D-tensor SAE variants (`topk_sae`, `l1_sae`, `jumprelu_sae`):
    this builder's `x` is 3D `(B, T, input_dim)` because attention needs a
    sequence dimension. Consumers must feed per-token / per-residue / per-agent
    activations, NOT pooled ones.
    """
    arch = _sae_skeleton(
        name=name,
        input_dim=input_dim,
        n_features=n_features,
        variant_description=(
            f"Attention-prefixed Top-k sparse autoencoder (k={k}, n_heads={n_heads},"
            f" attn_dropout={attn_dropout})."
            " A MultiHeadAttention block (with residual + LayerNorm) precedes"
            " the SAE encoder so each position can see cross-sequence context"
            " before being encoded. Sparsity is enforced by TopK; no L1 / L0"
            " penalty needed."
        ),
        extra_hps=[
            Hyperparameter("k", "int", k),
            Hyperparameter("n_heads", "int", n_heads),
            Hyperparameter("attn_dropout", "float", attn_dropout),
        ],
    )

    # 3D input/output (replace the 2D shapes the skeleton wrote).
    arch.tensors = [
        Tensor("x", ("B", "T", "input_dim"), "float32"),
        Tensor("x_hat", ("B", "T", "input_dim"), "float32"),
    ]

    arch.layers.extend([
        Layer(name="attn",
              description="Self-attention across the sequence dimension",
              op=OpCall("MultiHeadAttention",
                        ["input_dim", "n_heads", "attn_dropout"])),
        Layer(name="add_attn",
              description="Residual add: attention output + original x",
              op=OpCall("Add", [])),
        Layer(name="ln", op=OpCall("LayerNorm", ["input_dim"])),
        Layer(name="encoder", op=OpCall("Linear", ["input_dim", "n_features"])),
        Layer(name="relu", op=OpCall("ReLU", [])),
        Layer(name="topk",
              description="Keep only the k largest features per position",
              op=OpCall("TopK", ["k"])),
        Layer(name="decoder", op=OpCall("Linear", ["n_features", "input_dim"])),
    ])
    arch.flow.extend([
        FlowEdge("x", "attn", "x"),
        FlowEdge("attn", "add_attn", "attn_out"),
        FlowEdge("x", "add_attn", "x_skip"),
        FlowEdge("add_attn", "ln", "r"),
        FlowEdge("ln", "encoder", "r_n"),
        FlowEdge("encoder", "relu", "z_pre"),
        FlowEdge("relu", "topk", "z_relu"),
        FlowEdge("topk", "decoder", "z_sparse"),
        FlowEdge("decoder", "x_hat", "x_hat"),
    ])
    _close_sae(arch, tied=tied_decoder)
    # Override the skeleton's 2D output_shape invariant with the 3D one.
    arch.invariants = [Invariant("output_shape", "=", ("B", "T", "input_dim"))]
    return arch


def gated_sae(
    *,
    input_dim: int = 768,
    n_features: int = 16384,
    name: str = "GatedSae",
    tied_decoder: bool = False,
) -> Architecture:
    """Gated sparse autoencoder (Rajamanoharan et al. — Anthropic-style).

    Topology:
        x (B, input_dim)
          -> magnitude_proj: Linear(input_dim, n_features) → m
          -> gate_proj:      Linear(input_dim, n_features) → g_pre → Sigmoid → g
          -> gate_mul:       z_gated = m * g                              ← element-wise
          -> decoder:        Linear(n_features, input_dim) → x_hat

    The gate (sigmoid) determines per-feature firing probability; the
    magnitude branch sets the value when fired. Unlike TopK (hard binary
    mask) or JumpReLU (hard threshold), gating allows graded contributions
    — useful for features with continuous intensity.

    Used by econ-sae Phase 3 (`gated_sae_experiment.py`) for plateaued
    features that hard activations couldn't reach. Fills the
    activation-function diversity slot in bio-sae's AutoML candidate
    library (`bio-sae/docs/forge-incremental-specialist.md` §4.7
    Family A — alternative activation to TopK/L1/JumpReLU).
    """
    arch = _sae_skeleton(
        name=name,
        input_dim=input_dim,
        n_features=n_features,
        variant_description=(
            "Gated sparse autoencoder."
            " Two parallel encoder projections (magnitude + gate); the gate"
            " is squashed through a sigmoid and element-wise-multiplied with"
            " the magnitude to produce the latent activation. Allows graded"
            " (non-binary, non-thresholded) feature contributions."
        ),
        extra_hps=[],
    )

    arch.layers.extend([
        Layer(name="magnitude_proj",
              description="Magnitude branch — sets fired feature values",
              op=OpCall("Linear", ["input_dim", "n_features"])),
        Layer(name="gate_proj",
              description="Gate branch — produces per-feature firing logits",
              op=OpCall("Linear", ["input_dim", "n_features"])),
        Layer(name="sigmoid",
              description="Squash gate logits to (0, 1) firing probabilities",
              op=OpCall("Sigmoid", [])),
        Layer(name="gate_mul",
              description="Element-wise multiply magnitude × gate",
              op=OpCall("ElementwiseMul", [])),
        Layer(name="decoder", op=OpCall("Linear", ["n_features", "input_dim"])),
    ])
    arch.flow.extend([
        FlowEdge("x", "magnitude_proj", "x"),
        FlowEdge("x", "gate_proj",       "x"),
        FlowEdge("gate_proj", "sigmoid", "g_pre"),
        FlowEdge("magnitude_proj", "gate_mul", "m"),
        FlowEdge("sigmoid",        "gate_mul", "g"),
        FlowEdge("gate_mul", "decoder", "z_gated"),
        FlowEdge("decoder", "x_hat", "x_hat"),
    ])
    _close_sae(arch, tied=tied_decoder)
    return arch


def supervised_topk_sae(
    *,
    input_dim: int = 320,
    n_features: int = 1024,
    k: int = 64,
    n_labels: int = 100,
    aux_weight: float = 0.1,
    name: str = "SupervisedTopKSae",
    tied_decoder: bool = False,
) -> Architecture:
    """Top-k sparse autoencoder with an auxiliary per-label classifier head.

    Topology:
        x (B, input_dim)
          -> encoder: Linear(input_dim, n_features)
          -> relu
          -> topk: TopK(k)                  ← shared sparse latents z
          ├─→ decoder: Linear(n_features, input_dim)   -> x_hat   (reconstruction)
          └─→ aux_head: Linear(n_features, n_labels)    -> y_logits (classifier)

    The model emits TWO outputs: `x_hat` (reconstruction) and `y_logits`
    (auxiliary classifier logits, one per supervised label). The total
    training loss is the standard reconstruction loss + `aux_weight * BCE(
    y_logits, y_true)` — like L1/JumpReLU's sparsity terms, the loss
    weight is metadata here (n-orca captures the forward graph only).

    Motivation: econ-sae's `regime_supervised_experiment.py` (Phase 5.1)
    showed that supervising the SAE jointly with a per-label classifier
    head lifted regime mAUC 0.885 → 0.972 — the biggest single-phase
    jump in econ-sae's 6-phase journey, on the tier where every
    unsupervised technique had plateaued. Bio-sae's
    `forge-incremental-specialist.md` §4.8 catalogs this as Family G
    (new family beyond the basis-side A/B/C/D and substrate-side F).
    """
    arch = _sae_skeleton(
        name=name,
        input_dim=input_dim,
        n_features=n_features,
        variant_description=(
            f"Top-k SAE with auxiliary classifier head (k={k},"
            f" n_labels={n_labels}, aux_weight={aux_weight})."
            " Sparse latents z are decoded for reconstruction AND fed through"
            " a per-label Linear head for joint supervised training. Trains"
            " against a BCE-with-logits loss on the labels, weighted by"
            " `aux_weight` and added to the standard reconstruction loss."
        ),
        extra_hps=[
            Hyperparameter("k", "int", k),
            Hyperparameter("n_labels", "int", n_labels),
            Hyperparameter("aux_weight", "float", aux_weight),
        ],
    )

    # Add the auxiliary-output tensor.
    arch.tensors.append(Tensor("y_logits", ("B", "n_labels"), "float32"))

    arch.layers.extend([
        Layer(name="encoder", op=OpCall("Linear", ["input_dim", "n_features"])),
        Layer(name="relu", op=OpCall("ReLU", [])),
        Layer(name="topk",
              description="Keep only the k largest features per sample",
              op=OpCall("TopK", ["k"])),
        Layer(name="decoder", op=OpCall("Linear", ["n_features", "input_dim"])),
        Layer(name="aux_head",
              description="Per-label classifier from sparse latents",
              op=OpCall("Linear", ["n_features", "n_labels"])),
        Layer(name="y_logits", is_output=True,
              description="Per-label logits (auxiliary classifier head)"),
    ])
    arch.flow.extend([
        FlowEdge("x", "encoder", "x"),
        FlowEdge("encoder", "relu", "z_pre"),
        FlowEdge("relu", "topk", "z_relu"),
        FlowEdge("topk", "decoder", "z_sparse"),       # branch 1: reconstruction
        FlowEdge("topk", "aux_head", "z_sparse"),      # branch 2: classifier
        FlowEdge("decoder", "x_hat", "x_hat"),
        FlowEdge("aux_head", "y_logits", "y_logits"),
    ])
    # NOTE: multi-output SAE. The verifier compares every `output_shape`
    # invariant against every output (so an invariant that matches x_hat
    # would spuriously fail y_logits and vice versa). For now, skip the
    # output_shape invariant entirely; output shapes are still in the
    # inferred-shape report. Both shape rules live in verification_rules.
    arch.verification_rules.append(
        "reconstruction-shape: x_hat must match x's shape exactly"
    )
    arch.verification_rules.append(
        "y_logits-shape: y_logits must match (B, n_labels) — verified via "
        "inferred_shapes, not as an output_shape invariant"
    )
    if tied_decoder:
        arch.verification_rules.append(
            "tied-decoder: decoder weight equals encoder weight transposed"
        )
    return arch


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _sae_skeleton(
    *,
    name: str,
    input_dim: int,
    n_features: int,
    variant_description: str,
    extra_hps: list[Hyperparameter],
) -> Architecture:
    """Construct the common SAE wrapper: hyperparameters, tensors, IO layers."""
    arch = Architecture(
        name=name,
        description=variant_description,
        hyperparameters=[
            Hyperparameter("input_dim", "int", input_dim),
            Hyperparameter("n_features", "int", n_features),
        ] + extra_hps,
        tensors=[
            Tensor("x", ("B", "input_dim"), "float32"),
            Tensor("x_hat", ("B", "input_dim"), "float32"),
        ],
    )
    arch.layers.append(Layer(name="x", is_input=True,
                             description="LLM / world-model activation to reconstruct"))
    arch.layers.append(Layer(name="x_hat", is_output=True,
                             description="Reconstructed activation"))
    return arch


def _close_sae(arch: Architecture, *, tied: bool) -> None:
    """Append common metadata: invariants and verification rules."""
    arch.invariants.append(Invariant("output_shape", "=", ("B", "input_dim")))
    arch.verification_rules.append(
        "reconstruction-shape: x_hat must match x's shape exactly"
    )
    if tied:
        arch.verification_rules.append(
            "tied-decoder: decoder weight equals encoder weight transposed"
            " (enforced in the host module; the n-orca AST does not model"
            " weight tying explicitly)"
        )
