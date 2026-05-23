"""Builders for sparse-autoencoder architectures.

The three variants below match the SAE families implemented in `sm-sae`,
`econ-sae`, and consumed by `polygram`:

- **Top-k SAE**       — `topk_sae`     (Makhzani-style; sparsity via top-k mask)
- **L1 SAE**          — `l1_sae`       (vanilla; ReLU + L1 penalty in loss)
- **JumpReLU SAE**    — `jumprelu_sae` (Rajamanoharan et al. 2024; learnable
                                        per-feature threshold, L0 penalty)

Each builder returns an `Architecture` AST that verifies clean and compiles
to both Mermaid and PyTorch. The sparsity penalty for L1 and JumpReLU is a
*loss* term and lives outside the architecture topology — n-orca captures
the forward graph only, not the loss.
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
