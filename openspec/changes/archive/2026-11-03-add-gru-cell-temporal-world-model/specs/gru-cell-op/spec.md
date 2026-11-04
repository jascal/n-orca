## ADDED Requirements

### Requirement: GRUCell shape rule
The system SHALL register an op `GRUCell(input_size, hidden_size)` taking
exactly two inputs in `## flow` row order: `x` with shape `(*, input_size)` and
`h` with shape `(*, hidden_size)`. The leading dims of `x` and `h` MUST be
equal. The output SHALL have `h`'s shape.

#### Scenario: Valid step
- **WHEN** a `GRUCell(embed_dim, gru_hidden)` layer receives `(B, N, embed_dim)`
  then `(B, N, gru_hidden)`
- **THEN** its inferred output shape is `(B, N, gru_hidden)`

#### Scenario: Swapped inputs
- **WHEN** the same layer receives `(B, N, gru_hidden)` then `(B, N, embed_dim)`
  and `embed_dim` differs from `gru_hidden`
- **THEN** verification reports a shape error on that layer

#### Scenario: Wrong arity
- **WHEN** a `GRUCell` layer has one incoming edge, or three
- **THEN** verification reports a shape error on that layer

### Requirement: GRUCell parameter count
The parameter count of `GRUCell(i, h)` SHALL be
`3 * (i*h + h*h + 2*h)`, equal to `torch.nn.GRUCell(i, h)`.

#### Scenario: Default temporal sizes
- **WHEN** parameters are computed for `GRUCell(64, 128)`
- **THEN** the result is 74,496

### Requirement: GRUCell PyTorch emission
The compiled module SHALL instantiate `nn.GRUCell(input_size, hidden_size)` and
apply it over flattened leading dims, returning a tensor with `h`'s shape.

#### Scenario: Batched agents
- **WHEN** a compiled `GRUCell(16, 32)` layer is called with tensors of shape
  `(2, 5, 16)` and `(2, 5, 32)`
- **THEN** it returns shape `(2, 5, 32)`, equal to `nn.GRUCell` applied row by
  row to the flattened `(10, ·)` inputs
