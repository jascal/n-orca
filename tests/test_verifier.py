"""Tests for the 5-stage verifier."""
from __future__ import annotations

from n_orca.parser import parse
from n_orca.verifier import verify


def _verify_src(src: str):
    [arch] = parse(src)
    return verify(arch)


# --------------------------------------------------------------------------- #
#  Positive cases — each example should verify clean.
# --------------------------------------------------------------------------- #


VALID_MLP = """
# architecture MLP
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| d | int | 16 |
| h | int | 8 |
| k | int | 4 |
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, d) | float32 |
| y | (B, k) | float32 |
## layer x [input]
## layer fc1
- op: Linear(d, h)
## layer act
- op: ReLU()
## layer fc2
- op: Linear(h, k)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | fc1 | x |
| fc1 | act | h1 |
| act | fc2 | a |
| fc2 | y | logits |
"""


def test_valid_mlp_passes():
    report = _verify_src(VALID_MLP)
    assert report.valid, report.errors
    assert report.param_count == 16 * 8 + 8 + 8 * 4 + 4
    assert report.depth == 4


def test_residual_add_passes():
    src = """
# architecture Res
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
## layer fc
- op: Linear(d, d)
## layer add
- op: Add
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | fc | x |
| fc | add | a |
| x | add | skip |
| add | y | y |
"""
    report = _verify_src(src)
    assert report.valid, report.errors


# --------------------------------------------------------------------------- #
#  Negative cases — each defect should produce the documented error code.
# --------------------------------------------------------------------------- #


def test_unreachable_layer():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer orphan
- op: ReLU()
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | y | z |
"""
    report = _verify_src(src)
    assert not report.valid
    assert any(e.code == "UNREACHABLE_LAYER" for e in report.errors)


def test_layer_not_reaching_output():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer sink
- op: ReLU()
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | sink | a |
| x | y | b |
"""
    report = _verify_src(src)
    assert not report.valid
    assert any(e.code == "LAYER_NOT_REACHING_OUTPUT" for e in report.errors)


def test_unknown_layer_reference():
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
| x | ghost | a |
| ghost | y | b |
"""
    report = _verify_src(src)
    assert not report.valid
    assert any(e.code == "UNKNOWN_LAYER_REFERENCE" for e in report.errors)


def test_cycle_detected():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer a
- op: ReLU()
## layer b
- op: ReLU()
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | a | x |
| a | b | t |
| b | a | back |
| b | y | y |
"""
    report = _verify_src(src)
    assert not report.valid
    assert any(e.code == "CYCLE_DETECTED" for e in report.errors)


def test_no_input_layer():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| y | (B, 1) | float32 |
## layer foo
- op: ReLU()
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| foo | y | z |
"""
    report = _verify_src(src)
    assert not report.valid
    assert any(e.code == "NO_INPUT_LAYER" for e in report.errors)


def test_shape_mismatch_on_add():
    src = """
# architecture A
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
## layer fc
- op: Linear(d, 8)
## layer add
- op: Add
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | fc | x |
| fc | add | a |
| x | add | skip |
| add | y | y |
"""
    report = _verify_src(src)
    assert not report.valid
    assert any(e.code == "SHAPE_MISMATCH" for e in report.errors)


def test_input_arity_mismatch():
    src = """
# architecture A
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
## layer fc
- op: Linear(d, d)
## layer fc2
- op: Linear(d, d)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | fc | x |
| x | fc2 | x2 |
| fc | y | a |
| fc2 | y | b |
"""
    report = _verify_src(src)
    assert not report.valid
    # The output layer must have exactly one predecessor.
    assert any(e.code == "OUTPUT_ARITY_MISMATCH" for e in report.errors)


def test_param_budget_exceeded():
    src = """
# architecture A
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| d | int | 1000 |
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, d) | float32 |
| y | (B, d) | float32 |
## layer x [input]
## layer big
- op: Linear(d, d)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | big | x |
| big | y | y |
## invariants
- param_count <= 10K
"""
    report = _verify_src(src)
    assert not report.valid
    assert any(e.code == "PARAM_BUDGET_EXCEEDED" for e in report.errors)


def test_output_shape_invariant_matches():
    src = """
# architecture A
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| d | int | 8 |
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, d) | float32 |
| y | (B, d) | float32 |
## layer x [input]
## layer fc
- op: Linear(d, d)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | fc | x |
| fc | y | y |
## invariants
- output_shape: (B, d)
"""
    report = _verify_src(src)
    assert report.valid, report.errors


def test_output_shape_invariant_violated():
    src = """
# architecture A
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| d | int | 8 |
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, d) | float32 |
| y | (B, 16) | float32 |
## layer x [input]
## layer fc
- op: Linear(d, 16)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | fc | x |
| fc | y | y |
## invariants
- output_shape: (B, 32)
"""
    report = _verify_src(src)
    assert not report.valid
    assert any(e.code == "OUTPUT_SHAPE_INVARIANT" for e in report.errors)


def test_unknown_op_warns():
    src = """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer custom
- op: MyCustomBlock(x)
- shape: (B, 1)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | custom | x |
| custom | y | y |
"""
    report = _verify_src(src)
    assert any(w.code == "UNKNOWN_OP" for w in report.warnings)


def test_strict_promotes_warnings_to_errors():
    [arch] = parse(
        """
# architecture A
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, 1) | float32 |
| y | (B, 1) | float32 |
## layer x [input]
## layer custom
- op: WeirdOp()
- shape: (B, 1)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | custom | x |
| custom | y | y |
"""
    )
    lax = verify(arch, strict=False)
    strict = verify(arch, strict=True)
    assert lax.valid
    assert not strict.valid


def test_param_count_sums_correctly():
    """Each op spec must report exact param counts for primitive ops."""
    src = """
# architecture A
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| d | int | 100 |
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, d) | float32 |
| y | (B, d) | float32 |
## layer x [input]
## layer fc1
- op: Linear(d, 50)
## layer fc2
- op: Linear(50, d)
## layer ln
- op: LayerNorm(d)
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | fc1 | x |
| fc1 | fc2 | a |
| fc2 | ln | b |
| ln | y | y |
"""
    report = _verify_src(src)
    expected = (100 * 50 + 50) + (50 * 100 + 100) + (2 * 100)
    assert report.param_count == expected


def test_conv2d_same_padding_preserves_shape():
    """Stride 1, p = (k-1)/2 should preserve symbolic H/W exactly."""
    src = """
# architecture A
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| c | int | 8 |
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, c, H, W) | float32 |
| y | (B, c, H, W) | float32 |
## layer x [input]
## layer conv
- op: Conv2d(c, c, 3, 1, 1)
## layer add
- op: Add
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | conv | x |
| conv | add | a |
| x | add | skip |
| add | y | y |
"""
    report = _verify_src(src)
    assert report.valid, report.errors
    assert report.inferred_shapes["conv"] == ("B", "c", "H", "W")
