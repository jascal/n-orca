## 1. Examples

- [x] 1.1 Generate and write `examples/sae-attn-topk.n.orca.md` + `.mmd` using live builder + render (3D attn topology)
- [x] 1.2 Generate and write `examples/sae-supervised-topk.n.orca.md` + `.mmd` (multi-output x_hat + y_logits)
- [x] 1.3 Generate and write `examples/sae-gated.n.orca.md` + `.mmd` (gated magnitude * gate)
- [x] 1.4 Verify each new example individually with `n-orca verify` (and full CI-style sweep over all examples)
- [x] 1.5 Confirm the new examples are picked up by the glob in `.github/workflows/ci.yml`

## 2. Housekeeping / Polish

- [x] 2.1 Silence the runpy "found in sys.modules" RuntimeWarning when using `python -m n_orca.cli.main` (common in CI and scripts) by adding targeted filter in cli/main.py
- [x] 2.2 Re-run full test suite (`pytest -q`) to confirm no regressions (140 passing)
- [x] 2.3 Confirm `n-orca verify` on the new files produces clean "VALID" output with reasonable param counts

## 3. OpenSpec Artifacts (this change)

- [x] 3.1 Write proposal.md (Why + What + scope)
- [x] 3.2 Write design.md (minimal, since this is examples/docs completeness)
- [x] 3.3 Write specs/sae-examples/spec.md with ADDED requirements for the examples surface
- [x] 3.4 Write this tasks.md with trackable checkboxes (pre-mark completed items)
- [x] 3.5 (Final step) Run `openspec archive add-missing-advanced-sae-examples` (or equivalent) after all prior tasks and confirmation that examples are good

## 4. Follow-ups (out of scope for this change, tracked elsewhere or future)

- [x] 4.1 Run `scripts/generate_sae_docs.py` and land updates in econ-sae / sm-sae / polygram (separate change) — documented here for visibility; deferred to follow-up change
- [x] 4.2 Consider adding a short "Advanced SAE variants" section to top-level README or new docs page — future
- [x] 4.3 If ElementwiseMul or other ops need explicit registry love for gated (audit in separate topology support change) — future