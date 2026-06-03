## 1. Research & Design

- [x] 1.1 Review econ-sae TemporalWorldModel implementation (GRU or hidden state carry across periods) -- done read-only during impl cycle + subagent context
- [x] 1.2 Decide on n-orca representation: per-step graph with explicit `hidden` tensor in context vs. unrolled steps vs. custom "RecurrentCell" op -- decided on explicit hidden_in/out + external carry (per design + PR)
- [x] 1.3 Update design.md for the chosen approach (reference proposal)  -- Created initial design.md based on proposal + acceptance criteria (per-step explicit hidden state + attn composition). See design.md.

## 2. Implementation

- [x] 2.1 Add `temporal_world_model` (or `rnn_world_model`) builder in `world_models.py`
- [x] 2.2 Add corresponding handling in `build_world_model` MCP tool (new variant + kwargs like gru_hidden, etc.)
- [ ] 2.3 Ensure shape inference, verifier, and compilers handle any new state tensors or 3D+ patterns  (basic works; full GRU refinement later)

## 3. Examples & Tests

- [x] 3.1 Add `examples/econ-temporal-world-model.n.orca.md` + `.mmd` (generated from builder)
- [x] 3.2 Add test in `test_sae_and_world_models.py` (verifies, compiles, roundtrips)  -- Enhanced post-PR #3 review for hidden_out shape/dim assert + layer rename consistency.
- [ ] 3.3 Full `n-orca verify` sweep and pytest  (new example verifies; full suite green)

## 4. Docs & Sync

- [x] 4.1 Update README SAE/World Model section and `docs/proposed-sae-extensions.md` or new doc (README builders list + temporal note + econ ref done in 2026-06-03 scheduler cycle / PR#5; proposed is SAE-focused, crossref sufficient).
- [ ] 4.2 Note for econ-sae consumption (the n-orca definition becomes the spec)
- [ ] 4.3 Consider regen of sibling docs

## 5. OpenSpec close

- [ ] 5.1 Mark tasks complete
- [ ] 5.2 Archive the change

(Initial scaffolding created during n-orca handoff session. Impl + PR #3 (with nits addressed per review) merged 2026-06-03. Polish on 2.1/3.2 done; remaining tasks (2.3 shape/GRU, 4.x docs, 5. close/archive) for future cycles. Full GRU refinement tracked in IMPROVEMENT_GOAL + future OpenSpec if needed.)