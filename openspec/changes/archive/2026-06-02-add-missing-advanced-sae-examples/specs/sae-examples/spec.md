# SAE Examples Surface

This capability covers the worked `.n.orca.md` + `.mmd` examples shipped under `examples/sae-*.n.orca.md` (and their use in CI verification, agent discovery, and cross-sibling doc generation).

## ADDED Requirements

### Requirement: Advanced SAE variants have first-class examples

The three advanced SAE variants (attn_topk, supervised_topk, gated) that were added to the builder + MCP surface SHALL have corresponding first-class worked examples.

#### Scenario: AttnTopK example exists and is valid
- Agent / user finds `examples/sae-attn-topk.n.orca.md`
- `n-orca verify` on it succeeds (VALID, demonstrates 3D + attention prefix)
- Matching `.mmd` is present and CI would pick it up

#### Scenario: SupervisedTopK (multi-output) example exists and is valid
- `examples/sae-supervised-topk.n.orca.md` exists
- Verifies cleanly and shows multi-output (x_hat + y_logits)
- Covered by example verification

#### Scenario: Gated example exists and is valid
- `examples/sae-gated.n.orca.md` + `.mmd` present
- Verifies, demonstrates gated topology

### Requirement: Examples are generated from live code
Examples SHALL be generated from the live builders (via render/compile_mermaid) so they stay in sync with sae.py.

#### Scenario: No drift
- Regenerating the examples from the builders produces identical (or equivalent) content to what is checked in
- CI verify passes on the committed versions

### Requirement: CI + discoverability coverage
The new examples SHALL be included in the "verify every shipped example" glob and visible when listing examples/.

#### Scenario: CI green
- The glob in .github/workflows/ci.yml now covers the three new sae-*.n.orca.md files
- Full sweep in CI (or locally) reports all green, including the advanced ones

## Non-Requirements (for this change)

- Changes to the core SAE builder semantics or op registry.
- Updates to generated docs in econ-sae / sm-sae / polygram (separate change or script run).