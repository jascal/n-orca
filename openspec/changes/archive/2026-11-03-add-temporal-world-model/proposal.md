# add-temporal-world-model

## Why

econ-sae later phases (temporal + sentiment-driven MPC, GRU over periods) showed that cross-time recurrence in the world model is necessary to even have a chance at encoding regime / windowed features in the substrate.

Current n-orca world model builders only cover the per-step (baseline / deep / attn) graphs. A "temporal" variant (or explicit recurrent cell) would let econ-sae (and future work) describe the full unrolled or per-step-with-state architecture in the same verified Markdown format.

This keeps n-orca as the single source of truth for the architectures used across the SAE ground-truth fixtures.

## What Changes

- Add `temporal_world_model` (or `rnn_world_model`) builder in `world_models.py` (e.g. per-period attention or embed + GRU hidden state carried in context, or explicit unrolled steps for the graph).
- Corresponding MCP `build_world_model(variant="temporal", ...)` support.
- Example `examples/econ-temporal-world-model.n.orca.md` (and .mmd).
- Tests + verify coverage.
- Update docs and the econ-sae cross-references.

(Actual topology details to be designed in the change — may be a per-step graph with explicit hidden state tensor passed through, since n-orca describes the forward DAG.)

## Capabilities

Affects the "world model" capability / examples surface.

## Scope

In: the builder, MCP, one example, tests, docs update.
Out: full trainer changes in econ-sae, actual temporal training experiments (those stay in econ-sae).

## Related

Follows econ-sae temporal experiments and the "multi-decoder benchmark" lessons.

This change will be the n-orca side of making temporal world models a shared artifact.

## Initial Acceptance Criteria (for the builder + example)

- The new builder (`temporal_world_model` or equivalent) must produce a verifiable Architecture that can represent per-period state carry (e.g. explicit `hidden` or `state` tensor passed through layers) or unrolled temporal steps.
- Must support composition with existing attention (e.g. per-period attn + temporal state).
- Example `examples/econ-temporal-world-model.n.orca.md` must:
  - Verify cleanly (`n-orca verify`).
  - Compile to runnable PyTorch (with state tensors handled in forward).
  - Demonstrate state tracking patterns (e.g. GRU-like hidden update or attention over time).
- MCP `build_world_model(variant="temporal", ...)` must expose relevant hyperparameters (e.g. `hidden_dim`, `temporal_type`).
- Tests in `test_sae_and_world_models.py` must cover the new variant (AST, verify, compile, roundtrip).
- Docs updated to reference econ-sae use case for regime features.

Focus on patterns that allow the SAE substrate (h1 activations) to capture cross-period information without duplicating the full trainer logic in n-orca.