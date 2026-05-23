# architecture TinyViT

> A minimal Vision Transformer head: patchify -> linear projection -> single
> transformer block -> classification head. Intentionally small so the verify
> output is readable; scale d_model / depth in practice.

## hyperparameters

| Name        | Type  | Default |
|-------------|-------|---------|
| patch_dim   | int   | 48      |
| d_model     | int   | 128     |
| n_heads     | int   | 4       |
| d_ff        | int   | 512     |
| n_classes   | int   | 10      |
| dropout     | float | 0.1     |

## tensors

| Name | Shape                    | Dtype   |
|------|--------------------------|---------|
| x    | (B, S, patch_dim)        | float32 |
| y    | (B, n_classes)           | float32 |

## layer x [input]
> A batch of S flattened patches per image

## layer patch_proj
- op: Linear(patch_dim, d_model)

## layer attn_norm
- op: LayerNorm(d_model)

## layer attn
- op: MultiHeadAttention(d_model, n_heads, dropout)

## layer add_1
- op: Add

## layer ff_norm
- op: LayerNorm(d_model)

## layer ff
- op: FeedForward(d_model, d_ff, dropout)

## layer add_2
- op: Add

## layer pool
> Mean over the sequence dim — standard ViT pooling.
- op: Mean(1)

## layer head
- op: Linear(d_model, n_classes)

## layer y [output]

## flow

| Source     | Target     | Tensor    |
|------------|------------|-----------|
| x          | patch_proj | x         |
| patch_proj | attn_norm  | tokens    |
| attn_norm  | attn       | tokens_n  |
| attn       | add_1      | attn_out  |
| patch_proj | add_1      | tok_skip  |
| add_1      | ff_norm    | h         |
| ff_norm    | ff         | h_n       |
| ff         | add_2      | ff_out    |
| add_1      | add_2      | h_skip    |
| add_2      | pool       | hs        |
| pool       | head       | flat      |
| head       | y          | logits    |
