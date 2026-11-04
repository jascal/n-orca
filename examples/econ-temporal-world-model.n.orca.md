# architecture TemporalWorldModel

> Temporal world-model: one period of econ-sae's TemporalWorldModel (per-period cross-agent attn -> GRU state update, gru_hidden=128 -> MLP head). The GRU output is both the new state (hidden_out) and the head's input, so h1, the SAE substrate, carries cross-period (regime) context. Caller carries hidden_out back in as hidden_in across T periods.

## hyperparameters

| Name       | Type | Default |
|------------|------|---------|
| input_dim  | int  | 43      |
| embed_dim  | int  | 64      |
| n_heads    | int  | 4       |
| gru_hidden | int  | 128     |
| h1_dim     | int  | 192     |
| h2_dim     | int  | 128     |
| out_dim    | int  | 23      |

## tensors

| Name       | Shape              | Dtype   |
|------------|--------------------|---------|
| x          | (B, N, input_dim)  | float32 |
| y          | (B, N, out_dim)    | float32 |
| hidden_in  | (B, N, gru_hidden) | float32 |
| hidden_out | (B, N, gru_hidden) | float32 |

## layer x [input]

## layer hidden_in [input]

## layer embed
- op: Linear(input_dim, embed_dim)

## layer attn
- op: MultiHeadAttention(embed_dim, n_heads, 0.0)

## layer add_attn
- op: Add()

## layer ln
- op: LayerNorm(embed_dim)

## layer gru
> GRU state update: (period features, hidden_in) -> new hidden
- op: GRUCell(embed_dim, gru_hidden)

## layer fc1
- op: Linear(gru_hidden, h1_dim)

## layer act1
> ReLU on H1 — SAE substrate (post-GRU, carries temporal context)
- op: ReLU()

## layer fc2
- op: Linear(h1_dim, h2_dim)

## layer act2
- op: ReLU()

## layer head
- op: Linear(h2_dim, out_dim)

## layer y [output]

## layer hidden_out [output]

## flow

| Source    | Target     | Tensor   |
|-----------|------------|----------|
| x         | embed      | x        |
| embed     | attn       | tokens   |
| attn      | add_attn   | attn_out |
| embed     | add_attn   | tok_skip |
| add_attn  | ln         | r        |
| ln        | gru        | r_n      |
| hidden_in | gru        | h_in     |
| gru       | fc1        | h_new    |
| fc1       | act1       | z1       |
| act1      | fc2        | h1       |
| fc2       | act2       | z2       |
| act2      | head       | h2       |
| head      | y          | y_hat    |
| gru       | hidden_out | h_out    |
