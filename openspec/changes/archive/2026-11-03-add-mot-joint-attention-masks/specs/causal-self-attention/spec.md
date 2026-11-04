## ADDED Requirements

### Requirement: Opt-in causal MultiHeadAttention
The system SHALL accept an optional 4th argument on
`MultiHeadAttention(d, h, dropout, causal)`. When that argument is the literal
token `causal`, the emitted module SHALL apply a causal mask so position `i`
attends only to positions `<= i`. Shape inference and the parameter count SHALL
be the same as the three-argument form.

#### Scenario: Causal emission
- **WHEN** a compiled `MultiHeadAttention(d, h, 0.0, causal)` layer is run and
  the input at sequence position `k` is perturbed
- **THEN** outputs at positions `< k` are unchanged and the output at position
  `k` changes

#### Scenario: Existing three-argument documents unchanged
- **WHEN** the example verification and Mermaid baselines are regenerated
- **THEN** every entry other than the two MoT examples is byte-identical, and
  the reasoner's entry differs only in its `ar_mha` op text
