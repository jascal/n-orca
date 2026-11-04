## ADDED Requirements

### Requirement: MoT denoise-step builder
The system SHALL provide `mot_denoise_step(d_model, n_heads, timestep_dim,
h1_dim, h2_dim)` that returns a valid Architecture for ONE diffusion denoising
step of a Mixture-of-Transformers dual-tower block. Its inputs SHALL be an AR
stream `ar_x (B, S_ar, d_model)`, a noisy DM stream `dm_x_noisy (B, S_dm,
d_model)` and a timestep `t (B, timestep_dim)`. Its outputs SHALL include
`ar_out` and `dm_x_denoised`, with `dm_x_denoised` the same shape as
`dm_x_noisy`. The diffusion schedule is iterated outside the graph.

#### Scenario: Default build
- **WHEN** `mot_denoise_step()` is built and verified
- **THEN** it is VALID, uses `MultiHeadAttention` on the AR path,
  `DualStreamJointAttention` on the DM path and a timestep embedding, and its
  compiled PyTorch module runs `forward(ar_x, dm_x_noisy, t)`

### Requirement: MoT reasoner-only builder
The system SHALL provide `mot_reasoner_only(d_model, n_heads)` that returns a
valid Architecture for the AR causal tower alone: input `ar_x`, output
`ar_out`, with no DM stream, timestep, `DualStreamJointAttention` or
`TimestepEmbed`.

#### Scenario: Default build
- **WHEN** `mot_reasoner_only()` is built and verified
- **THEN** it is VALID and has neither `t` nor `dm_x_noisy` tensors

### Requirement: MoT ops
The system SHALL register `TimestepEmbed`, which validates its input feature
dimension, and `DualStreamJointAttention`. `DualStreamJointAttention` is
topology-level: its PyTorch emission is modelled on multi-head attention and
does not implement the AR-causal / DM-bidirectional masks or the AR+DM key/value
concatenation.

#### Scenario: Op lookup
- **WHEN** either op is used in a document
- **THEN** verification reports no unknown-op error

### Requirement: MCP and skill exposure
`build_world_model` SHALL accept `variant="mot"` (aliases `mot_denoise`,
`cosmos_mot`, `mot_world_model`) and `variant="mot_reasoner"` (aliases
`reasoner`, `mot_ar`, `cosmos_reasoner`), passing `timestep_dim` for `mot`.
The `n-orca-build-world-model` skill SHALL list both variants.

#### Scenario: MCP build with custom timestep_dim
- **WHEN** `build_world_model(variant="mot", timestep_dim=16, ...)` is called
- **THEN** the report is valid and the architecture name is `MoTDenoiseStep`

### Requirement: Shipped examples
The repository SHALL ship `examples/cosmos-mot-denoise-step.n.orca.md` and
`examples/cosmos-mot-reasoner.n.orca.md`, each with a `.mmd`, and both SHALL
verify VALID.

#### Scenario: Example sweep
- **WHEN** the example verification sweep runs
- **THEN** both MoT examples report VALID
