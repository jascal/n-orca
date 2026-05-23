# architecture UNetStub

> A two-level U-Net skeleton: down-sample -> bottleneck -> up-sample with
> a skip connection. Demonstrates `Concat` along the channel dim across a
> skip edge.

## hyperparameters

| Name | Type | Default |
|------|------|---------|
| in_c | int  | 3       |
| f    | int  | 16      |
| out_c| int  | 1       |

## tensors

| Name | Shape                | Dtype   |
|------|----------------------|---------|
| x    | (B, in_c, 64, 64)    | float32 |
| y    | (B, out_c, 64, 64)   | float32 |

## layer x [input]

## layer enc1_conv
- op: Conv2d(in_c, f, 3, 1, 1)

## layer enc1_act
- op: ReLU()

## layer down
- op: MaxPool2d(2, 2)

## layer bottleneck_conv
- op: Conv2d(f, f, 3, 1, 1)

## layer bottleneck_act
- op: ReLU()

## layer up
> Nearest-neighbor 2x upsample via AdaptiveAvgPool back to 64x64.
> (A learned ConvTranspose2d would also fit here.)
- op: AdaptiveAvgPool2d(64)

## layer skip_cat
> Concatenate encoder skip + decoder branch along the channel dim.
- op: Concat(1)

## layer dec_conv
- op: Conv2d(32, out_c, 3, 1, 1)

## layer y [output]

## flow

| Source           | Target          | Tensor    |
|------------------|-----------------|-----------|
| x                | enc1_conv       | x         |
| enc1_conv        | enc1_act        | e1c       |
| enc1_act         | down            | e1a       |
| down             | bottleneck_conv | d         |
| bottleneck_conv  | bottleneck_act  | bc        |
| bottleneck_act   | up              | ba        |
| up               | skip_cat        | u         |
| enc1_act         | skip_cat        | skip      |
| skip_cat         | dec_conv        | cat       |
| dec_conv         | y               | y_out     |

## invariants
- output_shape: (B, out_c, 64, 64)
