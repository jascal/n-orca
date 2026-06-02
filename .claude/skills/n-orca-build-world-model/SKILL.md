---
name: n-orca-build-world-model
description: Generate a verified .n.orca.md for an econ-sae style world model (baseline, deep, attn, temporal) used as SAE substrate. Temporal variant adds explicit state carry (for cross-period regime features per econ-sae). Use when working with econ-sae or similar agent-based / temporal substrates.
argument-hint: [baseline|deep|attn] [params...]
allowed-tools: mcp__n_orca__build_world_model
---

Build a world model architecture (the "host" that produces activations for SAE training in econ-sae style experiments).

Variants:
- baseline (2-hidden MLP)
- deep (more layers)
- attn (adds per-agent / per-position MultiHeadAttention before the MLP)
- temporal (attn + explicit hidden state carry across periods for regime/windowed features)

Example: /n-orca-build-world-model attn --h1_dim 96
Example: /n-orca-build-world-model temporal --gru_hidden 128

Returns the .n.orca.md definition (optionally writes files).

These are intentionally small and verify instantly — perfect for ground-truth SAE substrate work.

After building, you can immediately verify, compile to PyTorch for a trainer, or pair it with a build_sae call for the full pipeline.