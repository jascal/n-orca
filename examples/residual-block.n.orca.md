# architecture ResidualBlock

> A pre-activation residual block: two conv layers + skip connection.
> Mirrors the classic ResNet-v2 design at a single resolution.

## hyperparameters

| Name     | Type | Default |
|----------|------|---------|
| channels | int  | 64      |

## tensors

| Name | Shape          | Dtype   |
|------|----------------|---------|
| x    | (B, channels, H, W) | float32 |
| y    | (B, channels, H, W) | float32 |

## layer x [input]
> Feature map

## layer bn1
- op: BatchNorm2d(channels)

## layer act1
- op: ReLU()

## layer conv1
- op: Conv2d(channels, channels, 3, 1, 1)

## layer bn2
- op: BatchNorm2d(channels)

## layer act2
- op: ReLU()

## layer conv2
- op: Conv2d(channels, channels, 3, 1, 1)

## layer add
> Residual add: conv2 output + original input
- op: Add

## layer y [output]

## flow

| Source | Target | Tensor   |
|--------|--------|----------|
| x      | bn1    | x        |
| bn1    | act1   | h_bn1    |
| act1   | conv1  | h_act1   |
| conv1  | bn2    | h_conv1  |
| bn2    | act2   | h_bn2    |
| act2   | conv2  | h_act2   |
| conv2  | add    | h_conv2  |
| x      | add    | skip     |
| add    | y      | y_out    |

## invariants
- output_shape: (B, channels, H, W)
