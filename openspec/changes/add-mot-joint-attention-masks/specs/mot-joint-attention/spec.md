## ADDED Requirements

### Requirement: Two-input DualStreamJointAttention
The system SHALL define `DualStreamJointAttention(d, h, dropout)` with exactly
two inputs in `## flow` row order: `dm (B, S_dm, d)` then `ar (B, S_ar, d)`.
Batch and feature dims MUST match; sequence lengths MAY differ. The output
SHALL have `dm`'s shape. The parameter count SHALL be unchanged (`4d² + 4d`).

#### Scenario: Valid inputs
- **WHEN** the op receives `(B, S_dm, d_model)` then `(B, S_ar, d_model)`
- **THEN** its inferred output is `(B, S_dm, d_model)`

#### Scenario: One input
- **WHEN** a `DualStreamJointAttention` layer has a single incoming edge
- **THEN** verification reports an arity error on that layer

#### Scenario: Mismatched feature dim
- **WHEN** the two inputs' last dims differ
- **THEN** verification reports a shape error

### Requirement: Joint attention emission
The emitted module SHALL compute attention with queries from `dm` and keys and
values from `concat(ar, dm)` along the sequence axis, without a mask.

#### Scenario: DM conditioned on AR
- **WHEN** a compiled `DualStreamJointAttention` layer is run twice with the
  same `dm` and different `ar`
- **THEN** the outputs differ

### Requirement: MoT denoise step wiring and invariants
`mot_denoise_step` SHALL feed `ar_ln` into `dm_mha` (after `dm_ln`), mark
`ar_mha` causal, declare `ar_tower` and `dm_tower` groups, and declare
properties `unreachable: ar_out from dm_x_noisy`, `unreachable: ar_out from t`,
`reachable: dm_x_denoised from ar_x` and
`passes_through: dm_mha for ar_x -> dm_x_denoised`. Its parameter count SHALL be
unchanged.

#### Scenario: Default build verifies with properties
- **WHEN** `mot_denoise_step()` is built and verified
- **THEN** it is VALID, all four properties pass, and `dm_mha`'s inputs are
  `dm_ln` then `ar_ln`

#### Scenario: Compiled masks hold numerically
- **WHEN** the compiled module is run while perturbing (a) a later `ar_x`
  position, (b) `dm_x_noisy`, (c) `ar_x`
- **THEN** (a) earlier `ar_out` positions are unchanged, (b) `ar_out` is
  unchanged, (c) `dm_x_denoised` changes

### Requirement: Causal reasoner-only tower
`mot_reasoner_only` SHALL mark its attention causal, with an unchanged parameter
count.

#### Scenario: Reasoner is causal
- **WHEN** its compiled module is run and a later `ar_x` position is perturbed
- **THEN** earlier `ar_out` positions are unchanged

### Requirement: Regenerated MoT examples
`examples/cosmos-mot-denoise-step.*` and `examples/cosmos-mot-reasoner.*` SHALL
be regenerated from the builders and verify VALID, with the denoise-step
example's properties passing.

#### Scenario: Example sweep
- **WHEN** the example verification sweep runs
- **THEN** both MoT examples report VALID
