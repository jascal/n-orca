# architecture SubcorticalValueCortex

> A transformer cortex plus a frozen subcortical value core. The core reads the
> observation directly and influences the cortex only forward, through a
> neuromodulator bus that gates the cortical output. The isolation is the design
> claim, so it is asserted in `## properties` and checked by the verifier.

## hyperparameters

| Name     | Type  | Default |
|----------|-------|---------|
| d_obs    | int   | 64      |
| d_model  | int   | 512     |
| n_heads  | int   | 8       |
| d_ff     | int   | 2048    |
| dropout  | float | 0.1     |
| d_core   | int   | 32      |
| n_affect | int   | 8       |
| d_out    | int   | 64      |

## tensors

| Name  | Shape           | Dtype   |
|-------|-----------------|---------|
| obs   | (B, S, d_obs)   | float32 |
| y     | (B, S, d_out)   | float32 |
| v_hat | (B, S, 1)       | float32 |

## layer obs [input]
> Raw observation, read by both the cortex and the value core.

## layer embed
> Cortical input projection.
- op: Linear(d_obs, d_model)

## layer attn_norm
- op: LayerNorm(d_model)

## layer attn
> Multi-head self-attention over the sequence.
- op: MultiHeadAttention(d_model, n_heads, dropout)

## layer add_attn
> Attention residual.
- op: Add()

## layer ff
- op: FeedForward(d_model, d_ff, dropout)

## layer add_ff
> Feed-forward residual.
- op: Add()

## layer cortex_ln
> Final cortical normalization — the last cortical representation.
- op: LayerNorm(d_model)

## layer core_sense
> Value core's own read of the observation. Deliberately not a cortical
> projection: the core must not depend on cortical state.
- op: Linear(d_obs, d_core)

## layer core_fc
- op: Linear(d_core, d_core)

## layer core_add
> Core residual. Its only inputs are core layers — this is the layer a
> feedback path would have to attack.
- op: Add()

## layer core_act
- op: Tanh()

## layer affect_head
> Affect vector: the core's entire outward channel.
- op: Linear(d_core, n_affect)

## layer value_head
- op: Linear(d_core, 1)

## layer nm_bus
> Neuromodulator bus — the single sanctioned route from core to cortex.
- op: Linear(n_affect, n_affect)

## layer nm_receptor
> Expands the affect vector to cortical width.
- op: Linear(n_affect, d_model)

## layer nm_gain
> Multiplicative gain in [0, 1]; the bus modulates, it cannot inject.
- op: Sigmoid()

## layer modulate
> Cortical output gated by the core's affect signal.
- op: Mul()

## layer head
- op: Linear(d_model, d_out)

## layer y [output]
> Gated cortical output.

## layer v_hat [output]
> Value estimate from the isolated core.

## groups

| Name   | Layers                                                                    |
|--------|---------------------------------------------------------------------------|
| core   | core_sense, core_fc, core_add, core_act, affect_head, value_head          |
| cortex | embed, attn_norm, attn, add_attn, ff, add_ff, cortex_ln, modulate, head    |
| bus    | nm_bus, nm_receptor, nm_gain                                              |

## flow

| Source      | Target      | Tensor      |
|-------------|-------------|-------------|
| obs         | embed       | obs         |
| embed       | attn_norm   | h           |
| attn_norm   | attn        | h_normed    |
| embed       | add_attn    | h_skip      |
| attn        | add_attn    | attn_out    |
| add_attn    | ff          | h_attn      |
| add_attn    | add_ff      | h_attn_skip |
| ff          | add_ff      | ff_out      |
| add_ff      | cortex_ln   | h_ff        |
| cortex_ln   | modulate    | h_cortex    |
| obs         | core_sense  | obs         |
| core_sense  | core_fc     | c           |
| core_sense  | core_add    | c_skip      |
| core_fc     | core_add    | c_fc        |
| core_add    | core_act    | c_res       |
| core_act    | affect_head | c_act       |
| core_act    | value_head  | c_act       |
| value_head  | v_hat       | v           |
| affect_head | nm_bus      | affect      |
| nm_bus      | nm_receptor | nm          |
| nm_receptor | nm_gain     | gain_pre    |
| nm_gain     | modulate    | gain        |
| modulate    | head        | h_mod       |
| head        | y           | y_out       |

## invariants
- param_count <= 4M

## properties
- reachable: v_hat from obs
- unreachable: core from cortex
- passes_through: bus for core -> cortex
- within: y from obs <= 12
- param_fraction: core <= 1%
- out_dim: nm_bus <= n_affect

## verification rules
- core-isolation: NO path exists from any cortical layer to any core layer
- bus-mediation: every path from the core to the cortex passes through the neuromodulator bus
- core-frugality: the value core is a small fraction of the model's parameters
- bus-narrowness: the bus carries no more than the affect vector's width

> These four rules used to be prose the verifier ignored. They are now the
> `## properties` block above, and Stage 6 decides them.
