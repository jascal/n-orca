# architecture MoTReasonerOnly

> MoT AR causal reasoner only (standalone causal tower). d_model=64, n_heads=4. For 'reasons first' in Cosmos MoT before generation. See OpenSpec add-cosmos-mot-world-models + subagent report.

## hyperparameters

| Name    | Type | Default |
|---------|------|---------|
| d_model | int  | 64      |
| n_heads | int  | 4       |

## tensors

| Name   | Shape              | Dtype   |
|--------|--------------------|---------|
| ar_x   | (B, S_ar, d_model) | float32 |
| ar_out | (B, S_ar, d_model) | float32 |

## layer ar_x [input]
> AR tokens (reasoner prefix)

## layer ar_ln
- op: LayerNorm(d_model)

## layer ar_mha
> AR causal self-attn (reasoning)
- op: MultiHeadAttention(d_model, n_heads, 0.0)

## layer ar_add
- op: Add()

## layer ar_out [output]
> AR output (reasoning result)

## flow

| Source | Target | Tensor  |
|--------|--------|---------|
| ar_x   | ar_ln  | ar      |
| ar_ln  | ar_mha | ar_tok  |
| ar_mha | ar_add | ar_a    |
| ar_ln  | ar_add | ar_skip |
| ar_add | ar_out | ar_out  |
