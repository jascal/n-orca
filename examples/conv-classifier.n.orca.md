# architecture ConvClassifier

> A small CNN classifier: two conv blocks, global average pooling, linear head.
> Demonstrates Conv2d shape inference and AdaptiveAvgPool2d.

## hyperparameters

| Name      | Type | Default |
|-----------|------|---------|
| in_c      | int  | 3       |
| n_classes | int  | 10      |

## tensors

| Name | Shape              | Dtype   |
|------|--------------------|---------|
| x    | (B, in_c, 32, 32)  | float32 |
| y    | (B, n_classes)     | float32 |

## layer x [input]

## layer conv1
- op: Conv2d(in_c, 32, 3, 1, 1)

## layer act1
- op: ReLU()

## layer pool1
- op: MaxPool2d(2, 2)

## layer conv2
- op: Conv2d(32, 64, 3, 1, 1)

## layer act2
- op: ReLU()

## layer pool2
- op: MaxPool2d(2, 2)

## layer gap
- op: AdaptiveAvgPool2d(1)

## layer flatten
- op: Flatten(1)

## layer head
- op: Linear(64, n_classes)

## layer y [output]

## flow

| Source | Target  | Tensor |
|--------|---------|--------|
| x      | conv1   | x      |
| conv1  | act1    | c1     |
| act1   | pool1   | a1     |
| pool1  | conv2   | p1     |
| conv2  | act2    | c2     |
| act2   | pool2   | a2     |
| pool2  | gap     | p2     |
| gap    | flatten | g      |
| flatten| head    | f      |
| head   | y       | logits |
