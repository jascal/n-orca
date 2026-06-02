---
name: n-orca-build-sae
description: Generate a verified .n.orca.md (and optional Mermaid) for one of the supported SAE variants (topk, l1, jumprelu, attn_topk, supervised_topk, gated) using the MCP builder. Use for econ-sae, sm-sae, bio-sae, or polygram work.
argument-hint: [variant] [input_dim] [n_features] [k] ...
allowed-tools: mcp__n_orca__build_sae
---

Build an SAE architecture definition directly via the n-orca MCP tool.

Common variants:
- topk, l1, jumprelu (classic)
- attn_topk (with n_heads, attn_dropout; produces 3D tensors)
- supervised_topk (with n_labels, aux_weight; multi-output)
- gated

Example: /n-orca-build-sae attn_topk 320 1024 64 --n_heads 4

The tool returns the markdown (and can write files if out_markdown/out_mermaid provided).

After generation:
- Verify it (or note that build_sae already produces verifiable output).
- Show the key topology differences.
- Offer to compile to PyTorch or save the files.

This is the canonical way to get consistent, verified SAE specs that match what econ-sae etc. actually use.