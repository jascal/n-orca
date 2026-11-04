## Why

Liquid AI's LFM3 checkpoints expose an exact, compact tri-hybrid decoder
topology, but n-orca cannot currently convert or inspect it: its architecture
mixes gated short convolutions, sliding-window grouped-query attention, and
global grouped-query attention in one stack.  Supporting this published model
family makes n-orca useful for studying a non-standard, edge-oriented language
model without representing it as a misleading all-Transformer approximation.

## What Changes

- Add semantic n-orca operations for LFM3's gated causal short-convolution and
  head-gated grouped-query attention, with shape/resource support and clearly
  documented compilation approximations.
- Add an HF adapter for `model_type: "lfm3"`, preserving the checkpoint's
  layer order, dimensions, tied embedding/head, sandwich-norm residual blocks,
  window size, and RoPE distinctions in generated Markdown.
- Add a checked LFM3-1.7B base example, Mermaid visualization, and regression
  tests for conversion and exact tri-hybrid ordering.

## Capabilities

### New Capabilities

- `lfm3-hybrid-architecture`: Convert and verify the published LFM3
  tri-hybrid text backbone as an inspectable n-orca architecture.

### Modified Capabilities

- None.

## Impact

Changes are additive to `n_orca/ops/spec.py` and the HF adapter registry, plus
an example, tests, and documentation.  No existing model conversion surface is
removed or changed.
