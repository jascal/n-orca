# architecture TransformerBlock

> A pre-norm transformer encoder block: multi-head self-attention + residual,
> then feed-forward + residual. Identical pattern to GPT-2 / LLaMA blocks.

## hyperparameters

| Name    | Type  | Default |
|---------|-------|---------|
| d_model | int   | 512     |
| n_heads | int   | 8       |
| d_ff    | int   | 2048    |
| dropout | float | 0.1     |

## tensors

| Name | Shape            | Dtype   |
|------|------------------|---------|
| x    | (B, S, d_model)  | float32 |
| y    | (B, S, d_model)  | float32 |

## layer x [input]
> Token embeddings

## layer attn_norm
> Pre-attention layer norm
- op: LayerNorm(d_model)

## layer attn
> Multi-head self-attention
- op: MultiHeadAttention(d_model, n_heads, dropout)

## layer add_1
> Attention residual
- op: Add

## layer ff_norm
> Pre-FFN layer norm
- op: LayerNorm(d_model)

## layer ff
> Position-wise feed-forward
- op: FeedForward(d_model, d_ff, dropout)

## layer add_2
> FFN residual
- op: Add

## layer y [output]

## flow

| Source     | Target    | Tensor       |
|------------|-----------|--------------|
| x          | attn_norm | x            |
| attn_norm  | attn      | x_normed     |
| attn       | add_1     | attn_out     |
| x          | add_1     | x_skip       |
| add_1      | ff_norm   | h            |
| ff_norm    | ff        | h_normed     |
| ff         | add_2     | ff_out       |
| add_1      | add_2     | h_skip       |
| add_2      | y         | y_out        |

## invariants
- output_shape: (B, S, d_model)
