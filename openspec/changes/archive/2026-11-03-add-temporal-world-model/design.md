## Context

n-orca currently provides three world model builders in `world_models.py` (baseline, deep, attn_world_model) and corresponding support in `mcp_server.py::build_world_model` and the `n-orca-build-world-model` skill. These are per-step MLP or attention+MLP graphs used as SAE substrates in econ-sae (H1 activations for feature recovery).

econ-sae research (later phases) demonstrated that temporal/cross-period state (e.g. GRU hidden carry, sentiment over periods) is required to encode regime and windowed features in the substrate. The active OpenSpec change `add-temporal-world-model` aims to add a temporal variant so n-orca remains the canonical source for these architectures.

Current OpenSpec proposal and tasks define the need. No implementation or design.md yet (tasks 1.1-1.3 pending).

This design addresses task 1.3: create design.md after breaking down research.

## Goals / Non-Goals

**Goals:**
- Define a representation for temporal state in n-orca's forward DAG model that allows describing per-period state carry or attention over time.
- Enable a builder that produces verifiable Architecture compatible with existing world models.
- Support composition with attn (per-period attention + temporal state).
- Keep changes minimal and within n-orca (no edits to econ-sae etc.).
- Provide clear path for MCP, skill, example, and tests.

**Non-Goals:**
- Full unrolled multi-step graph or custom RNN op (keep simple per-step with explicit state tensor for now, as n-orca is DAG not execution engine).
- Implementing the actual temporal training loop or loss (stays in econ-sae).
- Changing existing builders or breaking API.
- Adding new ops unless necessary for state (prefer using existing Linear, etc.).

## Decisions

**Decision: Use explicit `hidden` / `state` tensor in the Architecture tensors and flow, passed through layers as additional input/output.**

Rationale: n-orca models forward graphs (DAG of layers with named tensors). Temporal recurrence can be represented as a per-step graph where a `hidden` tensor (e.g. shape (B, hidden_dim)) is input and output of the step. This allows "unrolling" in the calling code (econ-sae trainer) without n-orca needing to know about loops. Matches "per-step-with-state" in proposal. Simpler than full unroll (avoids explosion in layers for long sequences).

Alternatives considered:
- Custom "RecurrentCell" op: Rejected for v1 – would require new op in ops/spec.py, shape rules, PyTorch emission. Too much for initial design.
- Full unrolled steps in one Architecture: Rejected – not scalable, and n-orca shouldn't duplicate trainer sequence logic.
- Pure per-period without state: Doesn't meet acceptance criteria for "state carry".

**Decision: Base the temporal builder on attn_world_model + add state projection layers.**

Rationale: econ-sae temporal used attention + GRU. Start by extending attn (highest value for conjunctive + now temporal). Add Linear for hidden update (e.g. input_proj, hidden_update in impl after PR #3 nit). Use existing ops (Linear, ReLU, etc.). Keep small like other world models (input ~43, hidden ~100s).

**Decision: Expose hyperparameters like `hidden_dim`, `temporal_type="gru_simple"` or similar in builder and MCP.**

Rationale: Allows flexibility. Start with simple "state carry" (e.g. hidden = tanh( linear( concat( h1, hidden ) ) ) or attention-modulated. Aligns with proposal acceptance: "explicit hidden_dim, temporal_type".

**Decision: The Architecture will have additional tensors for state: e.g. "hidden_in", "hidden_out". The forward will take x + optional hidden.**

Rationale: In PyTorch compile, the model __init__ and forward can accept extra state args. Existing multi-output (supervised SAE) shows n-orca supports >1 output. For world models, y is output; add hidden_out.

## Risks / Trade-offs

- [Risk] State tensor shapes must be consistent across calls; user (econ-sae) responsible for init and carry. → Mitigation: Document clearly in builder docstring and example. Add invariant for hidden shape.
- [Trade-off] Not a "true" RNN in n-orca (no internal loop). → Acceptable per proposal: "per-step graph with explicit hidden state tensor". Full semantics in consumer.
- [Risk] Shape inference in verifier/compiler may need extension for optional state. → Mitigation: Make state required in the temporal builder's tensors/flow; test compile.
- Low risk overall: purely additive builder + docs in this change.

## Migration Plan

- Additive: new builder function, new variant in MCP/skill, new example file.
- Existing code unaffected.
- Update tests to include new builder (similar to test_attn_world_model_has_attention_layer).
- After impl, update docs/proposed-sae-extensions.md or README if needed (but keep minimal per scope).
- No rollback needed.

## Open Questions

- Exact update rule for hidden (simple linear + tanh? attention modulated? GRU gates in layers?). Defer to research 1.1/1.2; design assumes configurable via temporal_type.
- Should state be optional in forward for compatibility? For now, make temporal models always take/return hidden.
- How to represent in Mermaid (extra state nodes)? Use existing tensor notation.
- Tie to econ-sae TemporalWorldModel: once design set, read econ-sae code (read-only, no edit) for exact match in follow-up.

This design enables safe incremental implementation starting with a simple state-carry extension to attn_world_model.