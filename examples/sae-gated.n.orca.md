# architecture GatedSae

> Gated sparse autoencoder. Two parallel encoder projections (magnitude + gate); the gate is squashed through a sigmoid and element-wise-multiplied with the magnitude to produce the latent activation. Allows graded (non-binary, non-thresholded) feature contributions.

## hyperparameters

| Name       | Type | Default |
|------------|------|---------|
| input_dim  | int  | 128     |
| n_features | int  | 512     |

## tensors

| Name  | Shape          | Dtype   |
|-------|----------------|---------|
| x     | (B, input_dim) | float32 |
| x_hat | (B, input_dim) | float32 |

## layer x [input]
> LLM / world-model activation to reconstruct

## layer x_hat [output]
> Reconstructed activation

## layer magnitude_proj
> Magnitude branch — sets fired feature values
- op: Linear(input_dim, n_features)

## layer gate_proj
> Gate branch — produces per-feature firing logits
- op: Linear(input_dim, n_features)

## layer sigmoid
> Squash gate logits to (0, 1) firing probabilities
- op: Sigmoid()

## layer gate_mul
> Element-wise multiply magnitude × gate
- op: ElementwiseMul()

## layer decoder
- op: Linear(n_features, input_dim)

## flow

| Source         | Target         | Tensor  |
|----------------|----------------|---------|
| x              | magnitude_proj | x       |
| x              | gate_proj      | x       |
| gate_proj      | sigmoid        | g_pre   |
| magnitude_proj | gate_mul       | m       |
| sigmoid        | gate_mul       | g       |
| gate_mul       | decoder        | z_gated |
| decoder        | x_hat          | x_hat   |

## invariants
- output_shape: (B, input_dim)

## verification rules
- reconstruction-shape: x_hat must match x's shape exactly
