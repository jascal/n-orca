## Context

The SAE capability in n-orca (builders in `n_orca/sae.py`, exposed in MCP `build_sae`, used by econ-sae world-model + SAE experiments) had three new variants implemented:

- attn_topk_sae (for cross-position/agent context)
- supervised_topk_sae (aux head for joint reconstruction + classification)
- gated_sae (graded gating)

These were driven by econ-sae Phase results (biggest lifts for conjunctive and regime tiers) and documented in `docs/proposed-sae-extensions.md`.

The builders + MCP + PyTorch/Mermaid paths were complete and tested. However, the project ships "worked examples" in `examples/` that are verified in CI and used as the primary on-ramp for users and agents (including the OpenSpec skills and generation flows in sibling projects).

## Goals / Non-Goals

**Goals:**
- Make the three advanced SAE variants first-class in the shipped examples surface.
- Ensure they are covered by the existing "verify every example" CI gate.
- Provide copy-pasteable, verifiable `.n.orca.md` that demonstrate 3D tensors, multi-output, and gated topologies.
- Close the gap called out in the proposal doc.

**Non-Goals:**
- Changing the SAE builder implementations (already done and verified).
- Adding new world model variants or HF adapters (future OpenSpec change).
- Updating econ-sae / sm-sae / polygram sibling docs or generated files (can be a follow-up change or run of the generate script).
- Major AST / compiler changes.

## Decisions

- **Decision: Use the existing `render` + `compile_mermaid` + builder functions to generate the examples.**  
  Rationale: Keeps examples in sync with the code (no manual drift). The same path used for the original sae-*.md and econ-*.md. Simple `python -c` one-liner (or script) produces correct output that already passes verify.

- **Decision: Name files `sae-attn-topk.n.orca.md` etc. (kebab, matching existing `sae-jumprelu` etc.).**  
  Rationale: Consistent with current convention in `examples/`.

- **Decision: No new ops or AST changes required for the examples themselves.**  
  (Gated uses ElementwiseMul which is already supported via existing Add/Mul patterns or the op; the builders already produce valid Architecture objects.)

- **No spec delta needed for core "sae" capability** — this change is completeness of the *examples* surface for already-shipped builders.

## Risks / Trade-offs

- [Low] Generated examples could bit-rot if builders change significantly later. → Mitigation: CI verifies them on every run; the generate_sae_docs.py pattern exists for cross-sibling.
- [None] Performance / correctness impact — pure docs + test coverage addition.

## Migration Plan

N/A — additive only. New files are picked up automatically by globs in CI and by anyone listing `examples/`.

## Open Questions

- Should we also add a small "how to use the new builders" section to the top-level README or a new docs/sae-variants.md? (Can be follow-up.)
- Run `scripts/generate_sae_docs.py` and land sibling updates in the same or separate change? (Recommended separate to keep scope tight.)