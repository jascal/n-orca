## ADDED Requirements

### Requirement: LFM3 configuration conversion
The system SHALL convert a Hugging Face configuration whose `model_type` is
`lfm3` into a valid n-orca architecture, preserving its declared dimensions,
layer types, vocabulary size, and tied embedding/head relationship.

#### Scenario: Published text configuration
- **WHEN** conversion receives the published LFM3-1.7B base configuration
- **THEN** it produces a valid 24-layer architecture with 13 convolution, 5
  sliding-attention, and 6 global-attention operator blocks in config order

### Requirement: Explicit hybrid operators
The system SHALL represent LFM3 convolution and attention operators with
distinct semantic operations and preserve their sequence shape.

#### Scenario: Verifying a converted backbone
- **WHEN** the converted LFM3 architecture is verified
- **THEN** no unknown-operation or shape error is reported and every operator
  block preserves `(B, S, d_model)` through its residual path

### Requirement: Inspectable generated example
The system SHALL ship a rendered LFM3-1.7B example and Mermaid visualization
with provenance and approximation boundaries stated in its description.

#### Scenario: Reviewing the example
- **WHEN** a user opens the LFM3 example
- **THEN** they can identify the exact tri-hybrid schedule, sandwich norms,
  grouped-query/head-gated attention, and the differences between global and
  sliding attention without inspecting remote implementation code
