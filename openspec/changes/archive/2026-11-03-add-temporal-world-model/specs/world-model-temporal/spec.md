## ADDED Requirements

### Requirement: Temporal world-model builder
The system SHALL provide `temporal_world_model(input_dim, embed_dim, n_heads,
gru_hidden, h1_dim, h2_dim, out_dim)` that returns a valid Architecture for ONE
period of an econ-sae style temporal world model: per-period cross-agent
`MultiHeadAttention` plus explicit state carry. Its inputs SHALL be
`x (B, N, input_dim)` and `hidden_in (B, N, gru_hidden)`. Its outputs SHALL be
`y (B, N, out_dim)` and `hidden_out (B, N, gru_hidden)`. Recurrence across
periods is performed by the caller feeding `hidden_out` back as `hidden_in`.

The state update is a linear `hidden_update`, not a GRU cell; the builder's
description states this.

#### Scenario: Default build
- **WHEN** `temporal_world_model()` is built and verified
- **THEN** it is VALID, records `gru_hidden` as a hyperparameter, and its
  compiled PyTorch module runs `forward(x, hidden_in)` returning both `y` and
  `hidden_out`

### Requirement: MCP exposure
`build_world_model` SHALL accept `variant="temporal"` (alias
`temporal_world_model`) and pass `gru_hidden`.

#### Scenario: MCP temporal build
- **WHEN** `build_world_model(variant="temporal", gru_hidden=...)` is called
- **THEN** it builds `temporal_world_model` with that `gru_hidden` (and the
  shared `embed_dim`, `n_heads`, `h1_dim`, `h2_dim`, `out_dim` arguments)

### Requirement: Shipped example
The repository SHALL ship `examples/econ-temporal-world-model.n.orca.md` with a
`.mmd`, and it SHALL verify VALID.

#### Scenario: Example sweep
- **WHEN** the example verification sweep runs
- **THEN** the temporal example reports VALID
