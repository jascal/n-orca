# add-missing-advanced-sae-examples

## Why

The advanced SAE builders (`attn_topk_sae`, `supervised_topk_sae`, `gated_sae`) were implemented in `n_orca/sae.py` and exposed via the MCP `build_sae` tool (and documented in the module docstring) to support findings from econ-sae (and cross-sibling projects like bio-sae/sm-sae).

However, the acceptance criteria documented in `docs/proposed-sae-extensions.md` explicitly required:
- Worked examples in `examples/` (`.n.orca.md` + `.mmd` pairs) for each new builder, mirroring the existing `sae-topk`/`sae-l1`/`sae-jumprelu` examples.
- The CI (`.github/workflows/ci.yml`) runs `verify` over every `examples/*.n.orca.md` (and hf-generated).

Without the examples, the new capabilities were not discoverable by users/agents reading the examples dir, and not covered by the "verify every shipped example" gate.

## What Changes

- Added three new example pairs:
  - `examples/sae-attn-topk.n.orca.md` + `.mmd` (3D tensors, MultiHeadAttention prefix + residual + LN, TopK)
  - `examples/sae-supervised-topk.n.orca.md` + `.mmd` (multi-output: x_hat + y_logits auxiliary head)
  - `examples/sae-gated.n.orca.md` + `.mmd` (parallel magnitude/gate projections with elementwise mul)
- All new examples:
  - Parse, verify cleanly (VALID, with correct param counts and depths)
  - Render proper Markdown with hyperparameters, tensors, layers, flow, invariants
  - Produce valid Mermaid
  - Are now covered by the CI example verification loop
- Updated the "verify every shipped example" coverage implicitly (globs pick them up)
- Minor housekeeping: silenced the runpy CLI warning for `python -m n_orca.cli.main` usage (common in CI)

No breaking changes. Pure additive + coverage.

## Capabilities

- No new top-level specs (this is implementation + docs/examples completeness for existing SAE capability).
- Touches the "sae" and "examples" surface.

## Scope

- In scope: adding the three example artifacts + ensuring they pass verification/Mermaid/PyTorch paths + CI coverage.
- Out of scope: new builders (already done), changes to econ-sae/sm-sae sibling docs (separate follow-up), new world-model variants.

## Dependencies / Related

- Follows up on the work described in `docs/proposed-sae-extensions.md` (2026-05-23).
- Directly enables better agent usage and cross-project consistency with econ-sae (which motivated attn + supervised the most).

## Testing / Acceptance

- `n-orca verify` on each new file succeeds.
- Full `python -m pytest` still passes (140 tests).
- CI example verification loop would now cover the new files (21+ examples).
- The three new variants are now on equal footing with the original three for discoverability.