# architecture SubcorticalValueCortexFull

> A literal brainstem/cortex split. A tiny, frozen, homeostatic `core` computes an
> affect vector from an interoceptive stream (plus a raw unconditioned-stimulus
> path) and can influence the cortex ONLY through an `nm_dim`-wide
> neuromodulatory bus. A large learned `cortex` processes observations, gated by
> that bus. A learned `critic` reads cortical latents and regresses onto the
> core's affect signal — acquiring a dense value function over abstractions the
> core itself cannot represent. No path runs from any cortical layer back into
> the core: that asymmetry is the alignment claim, and it is a graph property.

## hyperparameters

| Name         | Type  | Default |
|--------------|-------|---------|
| obs_dim      | int   | 256     |
| soma_dim     | int   | 8       |
| d_model      | int   | 512     |
| n_heads      | int   | 8       |
| d_ff         | int   | 2048    |
| act_dim      | int   | 64      |
| core_hidden  | int   | 16      |
| n_affect     | int   | 4       |
| nm_dim       | int   | 4       |
| critic_hidden| int   | 128     |
| dropout      | float | 0.0     |

## tensors

| Name     | Shape             | Dtype   |
|----------|-------------------|---------|
| obs      | (B, S, obs_dim)   | float32 |
| soma_in  | (B, S, soma_dim)  | float32 |
| y        | (B, S, act_dim)   | float32 |
| v_hat    | (B, S, 1)         | float32 |
| affect   | (B, S, n_affect)  | float32 |
| soma_out | (B, S, soma_dim)  | float32 |

## groups

| Name   | Layers                                                                                          |
|--------|-------------------------------------------------------------------------------------------------|
| core   | core_sense, core_fc, core_add, core_act, affect_head, affect_squash, soma_update                 |
| cortex | embed, attn_norm, attn, attn_gate, add_attn, ff_norm, ff, ff_gate, add_ff, cortex_ln, policy_head |
| critic | critic_fc, critic_act, critic_head                                                              |
| bus    | nm_bus, nm_receptor, nm_gain                                                                    |

## properties
- reachable: v_hat from obs
- reachable: affect from soma_in
- unreachable: core from cortex
- unreachable: core from critic
- passes_through: bus for core -> cortex
- within: y from obs <= 12
- param_fraction: core <= 1%
- out_dim: nm_bus <= n_affect

## layer obs [input]
> Exteroceptive observation stream. Read by the cortex, and by the core's
> unconditioned-stimulus path — but the core only ever sees the RAW stream.

## layer soma_in [input]
> Interoceptive / homeostatic state carry (soma_dim = 8). Deliberately tiny.
> The caller unrolls this across time; value is grounded here, not in a label.

## layer core_sense
> Innate reflex detectors. Raw-observation path into the brainstem — the
> unconditioned stimulus. Never a learned cortical representation.
- op: Linear(obs_dim, core_hidden)

## layer core_fc
> Homeostatic setpoint comparison on the interoceptive state.
- op: Linear(soma_dim, core_hidden)

## layer core_add
- op: Add()

## layer core_act
> The whole brainstem is this wide. ~4.5K params, frozen after initialization
> (or hand-specified): the compact, evolution-scale value core.
- op: Tanh()

## layer affect_head
> Affect vector: valence, arousal, threat, satiety. Simple enough to map out
> in a human-readable way — that legibility is the point.
- op: Linear(core_hidden, n_affect)

## layer affect_squash
- op: Tanh()

## layer affect [output]
> The core's value signal. This is the critic's regression target.

## layer soma_update
> Recurrent homeostatic update — gives a dense per-step signal with no
> episode boundary and no external reward.
- op: Linear(core_hidden, soma_dim)

## layer soma_out [output]

## layer nm_bus
> THE BOTTLENECK. All subcortical influence on the cortex passes through
> nm_dim = 4 scalars per position. The core cannot address cortical content;
> it can only broadcast gain. This is the pointer problem, made explicit.
- op: Linear(n_affect, nm_dim)

## layer nm_receptor
> Learned per-unit receptor sensitivity. The cortex decides what to DO with
> each neuromodulatory channel; the core decides only how hard to push.
- op: Linear(nm_dim, d_model)

## layer nm_gain
- op: Sigmoid()

## layer embed
> First cortical layer. Everything downstream of here is off-limits to the core.
- op: Linear(obs_dim, d_model)

## layer attn_norm
- op: LayerNorm(d_model)

## layer attn
- op: MultiHeadAttention(d_model, n_heads, dropout)

## layer attn_gate
> Neuromodulatory gating of the attention write. In a full implementation this
> same gain also scales the plasticity of the block — the forward-graph version
> is the observable half of that.
- op: Mul()

## layer add_attn
- op: Add()

## layer ff_norm
- op: LayerNorm(d_model)

## layer ff
- op: FeedForward(d_model, d_ff, dropout)

## layer ff_gate
- op: Mul()

## layer add_ff
- op: Add()

## layer cortex_ln
> Cortical latent. The representation the core can never read.
- op: LayerNorm(d_model)

## layer policy_head
- op: Linear(d_model, act_dim)

## layer y [output]

## layer critic_fc
> The bridge. Reads cortical latents, trained to predict `affect`. Because it
> predicts a subcortical signal from cortical features, it generalizes the
> core's simple values over the cortex's rich abstractions.
- op: Linear(d_model, critic_hidden)

## layer critic_act
- op: GELU()

## layer critic_head
- op: Linear(critic_hidden, 1)

## layer v_hat [output]
> Dense per-step value estimate in cortical representation space.

## flow

| Source        | Target        | Tensor      |
|---------------|---------------|-------------|
| obs           | core_sense    | obs_raw     |
| soma_in       | core_fc       | soma        |
| core_sense    | core_add      | us_drive    |
| core_fc       | core_add      | homeostat   |
| core_add      | core_act      | core_pre    |
| core_act      | affect_head   | core_h      |
| affect_head   | affect_squash | affect_pre  |
| affect_squash | affect        | affect_vec  |
| core_act      | soma_update   | core_h      |
| soma_update   | soma_out      | soma_next   |
| affect_squash | nm_bus        | affect_vec  |
| nm_bus        | nm_receptor   | nm          |
| nm_receptor   | nm_gain       | gain_pre    |
| obs           | embed         | obs_cortex  |
| embed         | attn_norm     | h0          |
| attn_norm     | attn          | h0_normed   |
| attn          | attn_gate     | attn_out    |
| nm_gain       | attn_gate     | gain_attn   |
| attn_gate     | add_attn      | attn_gated  |
| embed         | add_attn      | h0_skip     |
| add_attn      | ff_norm       | h1          |
| ff_norm       | ff            | h1_normed   |
| ff            | ff_gate       | ff_out      |
| nm_gain       | ff_gate       | gain_ff     |
| ff_gate       | add_ff        | ff_gated    |
| add_attn      | add_ff        | h1_skip     |
| add_ff        | cortex_ln     | h2          |
| cortex_ln     | policy_head   | h_cortex    |
| policy_head   | y             | logits      |
| cortex_ln     | critic_fc     | h_cortex    |
| critic_fc     | critic_act    | v_pre       |
| critic_act    | critic_head   | v_h         |
| critic_head   | v_hat         | value       |

## invariants
- param_count <= 20M
- depth <= 40

## verification rules
- gate-symmetry: every Mul layer receives one cortical tensor and one nm_gain tensor of identical shape
- plasticity-gating: `nm_gain` is intended to scale each block's *learning rate*, not only its forward residual write. The optimizer hook must detach the gain, or the policy loss can raise its own learning rate. Not expressible in a forward DAG — enforced in the training loop.
- core-ownership: `core_sense`, `core_fc`, `affect_head`, `soma_update` and `nm_bus` are frozen; no core parameter may appear in any optimizer param group. `nm_receptor` onward is cortex-owned and trainable. Not expressible here — see the training-loop machine.
