## 1. Research & Design

- [ ] 1.1 Review econ-sae TemporalWorldModel implementation (GRU or hidden state carry across periods)
- [ ] 1.2 Decide on n-orca representation: per-step graph with explicit `hidden` tensor in context vs. unrolled steps vs. custom "RecurrentCell" op
- [x] 1.3 Update design.md for the chosen approach (reference proposal)  -- Created initial design.md based on proposal + acceptance criteria (per-step explicit hidden state + attn composition). See design.md.

## 2. Implementation

- [x] 2.1 Add `temporal_world_model` (or `rnn_world_model`) builder in `world_models.py`
- [x] 2.2 Add corresponding handling in `build_world_model` MCP tool (new variant + kwargs like gru_hidden, etc.)
- [ ] 2.3 Ensure shape inference, verifier, and compilers handle any new state tensors or 3D+ patterns  (basic works; full GRU refinement later)

## 3. Examples & Tests

- [x] 3.1 Add `examples/econ-temporal-world-model.n.orca.md` + `.mmd` (generated from builder)
- [x] 3.2 Add test in `test_sae_and_world_models.py` (verifies, compiles, roundtrips)
- [ ] 3.3 Full `n-orca verify` sweep and pytest  (new example verifies; full suite green)

## 4. Docs & Sync

- [ ] 4.1 Update README SAE/World Model section and `docs/proposed-sae-extensions.md` or new doc
- [ ] 4.2 Note for econ-sae consumption (the n-orca definition becomes the spec)
- [ ] 4.3 Consider regen of sibling docs

## 5. OpenSpec close

- [ ] 5.1 Mark tasks complete
- [ ] 5.2 Archive the change

(Initial scaffolding created during n-orca handoff session. Real design/implementation in follow-up dev sessions.)