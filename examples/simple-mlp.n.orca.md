# architecture SimpleMLP

> A two-layer MLP classifier — the n-orca "hello world".

## hyperparameters

| Name    | Type | Default |
|---------|------|---------|
| in_dim  | int  | 784     |
| hidden  | int  | 256     |
| out_dim | int  | 10      |

## tensors

| Name | Shape         | Dtype   |
|------|---------------|---------|
| x    | (B, in_dim)   | float32 |
| y    | (B, out_dim)  | float32 |

## layer x [input]
> Flattened image batch

## layer fc1
- op: Linear(in_dim, hidden)

## layer act
- op: ReLU()

## layer fc2
- op: Linear(hidden, out_dim)

## layer y [output]
> Logits

## flow

| Source | Target | Tensor |
|--------|--------|--------|
| x      | fc1    | x      |
| fc1    | act    | h      |
| act    | fc2    | h_act  |
| fc2    | y      | logits |

## invariants
- param_count <= 1M
- depth <= 10
- output_shape: (B, out_dim)
