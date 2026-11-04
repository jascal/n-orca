## ADDED Requirements

### Requirement: Temporal step uses a GRU between attention and head
`temporal_world_model` SHALL build one period as
`x → embed → attn → Add(skip) → ln → gru → fc1 → act1 → fc2 → act2 → head → y`,
where `gru` is `GRUCell(embed_dim, gru_hidden)` with inputs `ln` then
`hidden_in`, and `fc1` is `Linear(gru_hidden, h1_dim)`. The `gru` output SHALL
also feed `hidden_out`. The graph SHALL NOT contain a `hidden_update` layer.

This supersedes the archived `add-temporal-world-model` requirement that
described a linear state update.

#### Scenario: Default build
- **WHEN** `temporal_world_model()` is built and verified
- **THEN** it is VALID with 146,519 parameters, `gru`'s inputs are `ln` then
  `hidden_in`, and `gru` feeds exactly `fc1` and `hidden_out`

#### Scenario: State influences the prediction
- **WHEN** the compiled module is run twice with the same `x` and different
  `hidden_in`
- **THEN** both `y` and `hidden_out` differ between the two runs

### Requirement: Public surface unchanged
The system SHALL keep the builder's keyword arguments, tensor names (`x`,
`hidden_in`, `y`, `hidden_out`) and shapes, and the compiled signature
`forward(x, hidden_in) -> (y, hidden_out)` unchanged.

#### Scenario: Forward I/O
- **WHEN** the default compiled module is called with `x (2, 5, 43)` and
  `hidden_in (2, 5, 128)`
- **THEN** it returns `y (2, 5, 23)` and `hidden_out (2, 5, 128)`

### Requirement: MCP temporal variant is tested
`build_world_model(variant="temporal")` SHALL pass `gru_hidden` to the builder
and return a valid report, and a test SHALL cover it.

#### Scenario: Custom gru_hidden over MCP
- **WHEN** `build_world_model(variant="temporal", gru_hidden=32, ...)` is called
- **THEN** the report is valid and the markdown records `gru_hidden` as 32

### Requirement: Regenerated example, isolated baseline change
`examples/econ-temporal-world-model.{n.orca.md,mmd}` SHALL be regenerated from
the builder and verify VALID. In `tests/data/example_{verify,mermaid}_baseline.txt`
only that example's entries SHALL change.

#### Scenario: Baseline diff
- **WHEN** the baseline files are diffed against `main`
- **THEN** every changed line is inside the `econ-temporal-world-model` entry
