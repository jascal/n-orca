## ADDED Requirements

### Requirement: Dory configuration conversion
The system SHALL convert a Hugging Face configuration whose `model_type` is
`dory` into a valid n-orca architecture whose Input, Recurrent and Output
blocks follow `n_input_layers`, `n_recurrent_layers` and `n_output_layers`
over the config-ordered `layer_types`.

#### Scenario: Published 4B configuration
- **WHEN** conversion receives the published Dory1.0-4B configuration
  (16/40/16 sublayers, 2 loops)
- **THEN** it produces a valid architecture with groups `input_block`,
  `recurrent_loop_1`, `recurrent_loop_2`, `output_block` and 56 attention and
  56 MLP sublayer nodes

#### Scenario: Inconsistent split
- **WHEN** the three block sizes do not sum to `len(layer_types)`
- **THEN** conversion fails with a `ValueError` naming the mismatch

### Requirement: Unrolled loops with input re-injection
The system SHALL unroll the Recurrent block once per `n_recurrent_loops`, and
each loop SHALL begin with an `Add` node fed by the previous loop state and by
the Input block's output.

#### Scenario: Second loop entry
- **WHEN** the 2-loop configuration is converted
- **THEN** `loop2_inject` has exactly two inputs: the last node of loop 1 and
  the last node of the Input block

### Requirement: Explicit head_dim attention parameters
`GroupedQueryAttention` SHALL count parameters as
`2*d*n_heads*head_dim + 2*d*n_kv_heads*head_dim`.

#### Scenario: Dory attention sublayer
- **WHEN** parameters are computed for `(2560, 16, 4, 256, 0)`
- **THEN** the result is 26,214,400
