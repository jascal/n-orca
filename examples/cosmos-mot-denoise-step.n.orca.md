# architecture MoTDenoiseStep

> Cosmos 3-style MoT single denoise step (AR reasoner + DM generator). d_model=64, n_heads=4. AR: causal self-attn over AR only (reason first). DM: bidirectional joint attn, DM queries over concat(AR, DM) keys/values (generate). No DM -> AR path (checked in ## properties). AR keys/values for the DM tower use the DM tower's K/V weights (real MoT: AR tower's). Timestep embedding is not yet injected into DM. External loop for full T steps (like temporal state carry). See OpenSpec add-cosmos-mot-world-models + subagent report.

## hyperparameters

| Name         | Type | Default |
|--------------|------|---------|
| d_model      | int  | 64      |
| n_heads      | int  | 4       |
| timestep_dim | int  | 128     |
| h1_dim       | int  | 128     |
| h2_dim       | int  | 64      |

## tensors

| Name          | Shape              | Dtype   |
|---------------|--------------------|---------|
| ar_x          | (B, S_ar, d_model) | float32 |
| ar_out        | (B, S_ar, d_model) | float32 |
| dm_x_noisy    | (B, S_dm, d_model) | float32 |
| t             | (B, timestep_dim)  | float32 |
| dm_x_denoised | (B, S_dm, d_model) | float32 |
| ts_out        | (B, d_model)       | float32 |

## layer ar_x [input]
> AR tokens (reasoner prefix)

## layer dm_x_noisy [input]
> DM noisy latents (generator)

## layer t [input]
> Diffusion timestep embedding, (B, timestep_dim) (sinusoidal in full DiT; feature vector per batch for the toy learned Linear)

## layer ar_ln
- op: LayerNorm(d_model)

## layer ar_mha
> AR causal self-attn over AR only (reasoning)
- op: MultiHeadAttention(d_model, n_heads, 0.0, causal)

## layer ar_add
- op: Add()

## layer ar_out [output]
> AR output (reasoning result)

## layer ts_embed
> Timestep projection (sinusoidal in DiT; learned Linear for toy). Not yet injected into DM
- op: TimestepEmbed(timestep_dim, d_model)

## layer dm_ln
- op: LayerNorm(d_model)

## layer dm_mha
> DM joint attn: DM queries over concat(AR, DM) keys/values, unmasked
- op: DualStreamJointAttention(d_model, n_heads, 0.0)

## layer dm_add
- op: Add()

## layer dm_fc1
- op: Linear(d_model, h1_dim)

## layer dm_act1
- op: ReLU()

## layer dm_fc2
- op: Linear(h1_dim, h2_dim)

## layer dm_act2
- op: ReLU()

## layer dm_head
- op: Linear(h2_dim, d_model)

## layer dm_x_denoised [output]

## layer ts_out [output]
> timestep cond output (injected in forward)

## flow

| Source     | Target        | Tensor   |
|------------|---------------|----------|
| ar_x       | ar_ln         | ar       |
| ar_ln      | ar_mha        | ar_tok   |
| ar_mha     | ar_add        | ar_a     |
| ar_ln      | ar_add        | ar_skip  |
| ar_add     | ar_out        | ar_out   |
| t          | ts_embed      | ts       |
| dm_x_noisy | dm_ln         | dm       |
| dm_ln      | dm_mha        | dm_tok   |
| ar_ln      | dm_mha        | ar_kv    |
| dm_mha     | dm_add        | dm_a     |
| dm_ln      | dm_add        | dm_skip  |
| dm_add     | dm_fc1        | d_r      |
| dm_fc1     | dm_act1       | d_z1     |
| dm_act1    | dm_fc2        | d_h1     |
| dm_fc2     | dm_act2       | d_z2     |
| dm_act2    | dm_head       | d_h2     |
| dm_head    | dm_x_denoised | denoised |
| ts_embed   | ts_out        | ts_out   |

## groups

| Name     | Layers                                                                     |
|----------|----------------------------------------------------------------------------|
| ar_tower | ar_ln, ar_mha, ar_add                                                      |
| dm_tower | ts_embed, dm_ln, dm_mha, dm_add, dm_fc1, dm_act1, dm_fc2, dm_act2, dm_head |

## properties
- unreachable: ar_out from dm_x_noisy
- unreachable: ar_out from t
- reachable: dm_x_denoised from ar_x
- passes_through: dm_mha for ar_x -> dm_x_denoised
