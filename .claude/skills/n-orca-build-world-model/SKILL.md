---
name: n-orca-build-world-model
description: Generate a verified .n.orca.md for an econ-sae style world model (baseline, deep, attn, temporal, mot) used as SAE substrate. Temporal adds state carry; mot for Cosmos-style diffusion/MoT. Use when working with econ-sae or similar (incl. q-orca-kb n-orca-lang room).
argument-hint: [baseline|deep|attn] [params...]
allowed-tools: mcp__n_orca__build_world_model
---

Build a world model architecture (the "host" that produces activations for SAE training in econ-sae style experiments).

Variants:
- baseline (2-hidden MLP)
- deep (more layers)
- attn (adds per-agent / per-position MultiHeadAttention before the MLP)
- temporal (attn → GRUCell state update → MLP; hidden_in/hidden_out carried across periods, so h1 holds regime/windowed context)
- mot (MoT dual-tower: causal AR attn + DM joint attn over concat(AR, DM); no DM→AR path, checked in `## properties`; timestep for diffusion/multimodal like Cosmos 3)
- mot_reasoner (MoT AR causal reasoner only / "reasons first" tower, causal-masked)

Example: /n-orca-build-world-model attn --h1_dim 96
Example: /n-orca-build-world-model temporal --gru_hidden 128
Example: /n-orca-build-world-model mot --d_model 64 --timestep_dim 32
 (timestep_dim now exposed via MCP after PR#8)
Example: /n-orca-build-world-model mot_reasoner --d_model 64 --n_heads 4

Returns the .n.orca.md definition (optionally writes files).

These are intentionally small and verify instantly — perfect for ground-truth SAE substrate work.

After building, you can immediately verify, compile to PyTorch for a trainer, or pair it with a build_sae call for the full pipeline.