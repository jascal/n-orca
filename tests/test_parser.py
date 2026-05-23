"""Tests for the markdown parser."""
from __future__ import annotations

import pytest

from n_orca.parser import parse, ParseError


MINIMAL = """
# architecture Tiny

## hyperparameters

| Name | Type | Default |
|------|------|---------|
| d | int | 4 |

## tensors

| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, d) | float32 |
| y | (B, d) | float32 |

## layer x [input]
## layer act
- op: ReLU()
## layer y [output]

## flow

| Source | Target | Tensor |
|--------|--------|--------|
| x | act | x |
| act | y | y |
"""


def test_parses_minimal():
    [arch] = parse(MINIMAL)
    assert arch.name == "Tiny"
    assert len(arch.layers) == 3
    assert arch.layer("act").op.name == "ReLU"
    assert arch.layer("act").op.args == []
    assert len(arch.flow) == 2


def test_inputs_and_outputs():
    [arch] = parse(MINIMAL)
    assert [ly.name for ly in arch.inputs()] == ["x"]
    assert [ly.name for ly in arch.outputs()] == ["y"]


def test_hyperparameters_typed():
    [arch] = parse(MINIMAL)
    assert len(arch.hyperparameters) == 1
    hp = arch.hyperparameters[0]
    assert hp.name == "d"
    assert hp.type == "int"
    assert hp.default == 4


def test_tensors_with_shape():
    [arch] = parse(MINIMAL)
    assert len(arch.tensors) == 2
    x = arch.tensor("x")
    assert x is not None
    assert x.shape == ("B", "d")
    assert x.dtype == "float32"


def test_op_with_args():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 10) | float32 |
| y | (B, 5) | float32 |

## layer x [input]
## layer fc
- op: Linear(10, 5)
## layer y [output]

## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | fc | x |
| fc | y | y |
"""
    [arch] = parse(src)
    fc = arch.layer("fc")
    assert fc.op.name == "Linear"
    assert fc.op.args == ["10", "5"]


def test_layer_description_blockquote():
    [arch] = parse(MINIMAL)
    # MINIMAL has no descriptions; assert None.
    assert arch.layer("act").description is None


def test_multi_architecture():
    src = """
# architecture First
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | y | z |

---

# architecture Second
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | y | z |
"""
    archs = parse(src)
    assert [a.name for a in archs] == ["First", "Second"]


def test_invariants_with_unit_suffix():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | y | z |
## invariants
- param_count <= 5M
- depth <= 10
"""
    [arch] = parse(src)
    inv = {i.kind: i for i in arch.invariants}
    assert inv["param_count"].value == 5_000_000
    assert inv["param_count"].op == "<="
    assert inv["depth"].value == 10


def test_op_call_with_nested_parens():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer norm
- op: LayerNorm(d_model)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | norm | x |
| norm | y | y |
"""
    [arch] = parse(src)
    assert arch.layer("norm").op.args == ["d_model"]


def test_parse_raises_when_no_architecture_heading():
    with pytest.raises(ParseError):
        parse("# something else\n\njust prose\n")


def test_declared_shape_bullet():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer reshape
- op: Reshape(B, 1)
- shape: (B, 1)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | reshape | x |
| reshape | y | y |
"""
    [arch] = parse(src)
    assert arch.layer("reshape").declared_shape == ("B", "1")
