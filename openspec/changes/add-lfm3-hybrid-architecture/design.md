## Context

LFM3's public configuration and reference implementation define a causal
24-layer, 1.73B-parameter text decoder.  n-orca's HF conversion layer supports
conventional transformer families but has no representation for a gated causal
depthwise convolution or the extra gates/KV-head count of LFM3 attention.

## Goals / Non-Goals

**Goals:**

- Preserve the published LFM3-1.7B layer schedule and material hyperparameters
  in generated, verifiable Markdown.
- Make the two LFM3-specific mixing operators visible in n-orca diagrams and
  descriptions rather than disguising them as ordinary MHA/FFN layers.
- Keep generated PyTorch executable for topology exploration, while marking
  any numerical approximation explicitly.

**Non-Goals:**

- Reproduce Liquid AI checkpoints, kernels, caches, or bit-exact inference.
- Model vision, audio, codec, or LFM3-VL encoder details in this text-backbone
  change.

## Decisions

- Introduce `GatedCausalConv1d(d_model, kernel_size)` and
  `HeadGatedGroupedQueryAttention(d_model, n_heads, n_kv_heads, window)` as
  semantic ops.  Their shape rule preserves `(B, S, d_model)`; their resource
  accounting reflects the named projections where inputs are concrete.
  A generic `Conv1d` or `MultiHeadAttention` would hide the architectural fact
  this change exists to expose.
- Compile these ops to documented topology-only approximations using existing
  PyTorch modules, rather than embedding the remote LFM3 implementation or
  adding a new runtime dependency.  The generated Markdown remains the source
  of truth; users requiring checkpoint-faithful inference load the HF model.
- Emit each decoder block explicitly: operator RMSNorm → selected operator →
  post-operator RMSNorm → residual, then FFN RMSNorm → SwiGLU-like FFN →
  post-FFN RMSNorm → residual.  n-orca's LayerNorm and FeedForward are labeled
  approximations of RMSNorm and gated MLP behavior, consistent with its Llama
  adapter.
- Add an `lfm3` HF adapter, deriving its schedule only from the config's
  `layer_types` field.  This preserves future compatible variants rather than
  hard-coding the 24-layer default.

## Risks / Trade-offs

- [Topology compiler is not numerically faithful] → The descriptions and op
  docs explicitly say so; tests only promise importable, shape-compatible code.
- [Attention window and RoPE semantics are not executable in the generic
  compiler] → Preserve them in layer descriptions and op arguments, and retain
  the exact checkpoint configuration as provenance.
- [An upstream config changes] → Conversion reads the supplied config rather
  than relying on a fixed schedule; regression tests use the current published
  LFM3-1.7B configuration.

## Migration Plan

The addition is registry-only and additive.  Removing the adapter or new op
registrations reverts behavior with no migration for existing documents.
