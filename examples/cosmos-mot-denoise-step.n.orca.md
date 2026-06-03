# architecture MoTDenoiseStep

> Cosmos 3-style MoT single denoise step (AR reasoner + DM generator). d_model=64, n_heads=4. AR: causal self-attn only (reason first). DM: bidirectional joint over AR+DM + timestep cond (generate). External loop for full T steps (like temporal state carry). See OpenSpec add-cosmos-mot-world-models + subagent report.

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
| t             | (B)                | float32 |
| dm_x_denoised | (B, S_dm, d_model) | float32 |
| ts_out        | (d_model)          | float32 |

## layer ar_x [input]
> AR tokens (reasoner prefix)

## layer dm_x_noisy [input]
> DM noisy latents (generator)

## layer t [input]
> Diffusion timestep (scalar per batch)

## layer ar_ln
- op: LayerNorm(d_model)

## layer ar_mha
> AR causal self-attn (reasoning)
- op: MultiHeadAttention(d_model, n_heads, 0.0)

## layer ar_add
- op: Add()

## layer ar_out [output]
> AR output (reasoning result)

## layer ts_embed
> Timestep projection (sinusoidal in DiT; Linear here for toy)
- op: Linear(timestep_dim, d_model)

## layer dm_ln
- op: LayerNorm(d_model)

## layer dm_mha
> DM joint attn (bidir over AR+DM in full MoT; self here + cross note)
- op: MultiHeadAttention(d_model, n_heads, 0.0)

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
| dm_mha     | dm_add        | dm_a     |
| dm_ln      | dm_add        | dm_skip  |
| dm_add     | dm_fc1        | d_r      |
| dm_fc1     | dm_act1       | d_z1     |
| dm_act1    | dm_fc2        | d_h1     |
| dm_fc2     | dm_act2       | d_z2     |
| dm_act2    | dm_head       | d_h2     |
| dm_head    | dm_x_denoised | denoised |
| ts_embed   | ts_out        | ts_out   |
