---
name: n-orca-build-sae
description: Generate a verified .n.orca.md (and optional Mermaid) for one of the supported SAE variants using the MCP builder. Explicitly supports classic (topk, l1, jumprelu) and advanced variants (attn_topk for cross-sequence context, supervised_topk for multi-output with aux classifier head, gated for parallel magnitude/gate). Use for econ-sae, sm-sae, bio-sae, or polygram work.
argument-hint: [variant] [input_dim] [n_features] [k] ...
allowed-tools: mcp__n_orca__build_sae
---

Build an SAE architecture definition directly via the n-orca MCP tool.

Common variants (all verified + compilable):
- topk, l1, jumprelu (classic families)
- attn_topk (with n_heads, attn_dropout; produces 3D tensors `(B, T, d)` + residual attention prefix — the conjunctive feature unlock from econ-sae)
- supervised_topk (with n_labels, aux_weight; multi-output returning `(x_hat, y_logits)` for joint reconstruction + classification)
- gated (parallel magnitude + sigmoid gate projections with element-wise mul for graded features)

Example: /n-orca-build-sae attn_topk 320 1024 64 --n_heads 4

The tool returns the markdown (and can write files if out_markdown/out_mermaid provided).

After generation:
- Verify it (or note that build_sae already produces verifiable output).
- Show the key topology differences.
- Offer to compile to PyTorch or save the files.

This is the canonical way to get consistent, verified SAE specs that match what econ-sae etc. actually use.