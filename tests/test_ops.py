"""Tests for the op-spec library's shape inference and param counting."""
from __future__ import annotations

import pytest

from n_orca.ops import get_op, op_names, UnknownOpError, ShapeRuleError


def test_registered_ops_present():
    names = set(op_names())
    must_have = {
        "Linear", "LayerNorm", "Conv2d", "MultiHeadAttention",
        "Add", "Concat", "Flatten", "ReLU", "GELU", "Embedding", "Mean",
    }
    assert must_have.issubset(names)


def test_unknown_op_raises():
    with pytest.raises(UnknownOpError):
        get_op("NotARealOp")


def test_linear_replaces_last_dim():
    spec = get_op("Linear")
    out = spec.infer(["10", "20"], [("B", "S", "10")])
    assert out == ("B", "S", "20")


def test_linear_params():
    spec = get_op("Linear")
    assert spec.params(["10", "20"], []) == 10 * 20 + 20


def test_conv2d_same_padding_preserves_symbolic_dims():
    spec = get_op("Conv2d")
    out = spec.infer(["3", "16", "3", "1", "1"], [("B", "3", "H", "W")])
    assert out == ("B", "16", "H", "W")


def test_conv2d_strided_same_padding():
    spec = get_op("Conv2d")
    out = spec.infer(["3", "16", "3", "2", "1"], [("B", "3", "H", "W")])
    assert out == ("B", "16", "H/2", "W/2")


def test_conv2d_concrete_dims():
    spec = get_op("Conv2d")
    # 32x32 input, kernel 3, stride 1, padding 0 -> 30x30
    out = spec.infer(["3", "16", "3", "1", "0"], [("B", "3", "32", "32")])
    assert out == ("B", "16", "30", "30")


def test_maxpool_k_equals_s_halves_symbolic():
    spec = get_op("MaxPool2d")
    out = spec.infer(["2", "2"], [("B", "C", "H", "W")])
    assert out == ("B", "C", "H/2", "W/2")


def test_add_requires_same_shapes():
    spec = get_op("Add")
    shapes = [("B", "8"), ("B", "8")]
    assert spec.infer([], shapes) == ("B", "8")
    with pytest.raises(ShapeRuleError):
        spec.infer([], [("B", "8"), ("B", "16")])


def test_add_requires_at_least_two_inputs():
    spec = get_op("Add")
    with pytest.raises(ShapeRuleError):
        spec.infer([], [("B", "8")])


def test_concat_sums_dims():
    spec = get_op("Concat")
    out = spec.infer(["1"], [("B", "8", "H", "W"), ("B", "16", "H", "W")])
    assert out == ("B", "24", "H", "W")


def test_concat_rejects_non_concat_dim_mismatch():
    spec = get_op("Concat")
    with pytest.raises(ShapeRuleError):
        spec.infer(["1"], [("B", "8", "32", "32"), ("B", "16", "16", "16")])


def test_flatten_collapses_tail():
    spec = get_op("Flatten")
    out = spec.infer(["1"], [("B", "16", "8", "8")])
    # 16 * 8 * 8 = 1024
    assert out == ("B", "1024")


def test_embedding_appends_dim():
    spec = get_op("Embedding")
    out = spec.infer(["1000", "64"], [("B", "S")])
    assert out == ("B", "S", "64")


def test_mean_drops_a_dim():
    spec = get_op("Mean")
    out = spec.infer(["1"], [("B", "S", "D")])
    assert out == ("B", "D")


def test_mha_preserves_shape():
    spec = get_op("MultiHeadAttention")
    out = spec.infer(["64", "8", "0.1"], [("B", "S", "64")])
    assert out == ("B", "S", "64")
