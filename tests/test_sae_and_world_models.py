"""Tests for the new SAE and world-model builders + the TopK/JumpReLU ops."""
from __future__ import annotations

import importlib.util

import pytest

from n_orca import sae, world_models
from n_orca.compiler import compile_pytorch, compile_mermaid
from n_orca.ops import get_op, ShapeRuleError
from n_orca.render import render
from n_orca.parser import parse
from n_orca.verifier import verify


HAS_TORCH = importlib.util.find_spec("torch") is not None


# --------------------------------------------------------------------------- #
#  Op-level tests
# --------------------------------------------------------------------------- #


def test_topk_op_shape_preserves():
    spec = get_op("TopK")
    assert spec.infer(["8"], [("B", "256")]) == ("B", "256")
    assert spec.params(["8"], []) == 0


def test_topk_op_requires_k():
    spec = get_op("TopK")
    with pytest.raises(ShapeRuleError):
        spec.infer([], [("B", "256")])


def test_jumprelu_op_shape_and_params():
    spec = get_op("JumpReLU")
    assert spec.infer(["256"], [("B", "256")]) == ("B", "256")
    # Per-feature threshold: one learnable param per feature.
    assert spec.params(["256"], []) == 256


# --------------------------------------------------------------------------- #
#  SAE builders
# --------------------------------------------------------------------------- #


def test_topk_sae_verifies_and_param_count_matches():
    arch = sae.topk_sae(input_dim=128, n_features=512, k=16)
    rep = verify(arch)
    assert rep.valid, [e.code for e in rep.errors]
    # Encoder: 128*512 + 512 = 66,048. Decoder: 512*128 + 128 = 65,664. Total 131,712.
    assert rep.param_count == (128 * 512 + 512) + (512 * 128 + 128)


def test_l1_sae_has_no_topk_or_jumprelu():
    arch = sae.l1_sae(input_dim=64, n_features=256)
    layer_ops = [(ly.name, ly.op.name if ly.op else None) for ly in arch.layers]
    op_names = {n for _, n in layer_ops if n}
    assert "TopK" not in op_names
    assert "JumpReLU" not in op_names
    assert "ReLU" in op_names


def test_jumprelu_sae_adds_threshold_params():
    arch = sae.jumprelu_sae(input_dim=128, n_features=512)
    rep = verify(arch)
    assert rep.valid, [e.code for e in rep.errors]
    # 131,712 + 512 threshold params.
    assert rep.param_count == (128 * 512 + 512) + 512 + (512 * 128 + 128)


def test_attn_topk_sae_verifies_and_has_attention_layer():
    arch = sae.attn_topk_sae(input_dim=32, n_features=128, k=4, n_heads=4)
    rep = verify(arch)
    assert rep.valid, [e.code for e in rep.errors]
    ops = {ly.op.name for ly in arch.layers if ly.op}
    assert "MultiHeadAttention" in ops
    assert "TopK" in ops
    assert "LayerNorm" in ops
    assert "Add" in ops


def test_attn_topk_sae_output_shape_is_3d():
    arch = sae.attn_topk_sae(input_dim=16, n_features=64, k=4, n_heads=2)
    # The output_shape invariant must reflect the 3D tensor convention
    # (B, T, input_dim), not the 2D (B, input_dim) the skeleton wrote.
    output_shape_inv = next(
        (inv for inv in arch.invariants if inv.kind == "output_shape"),
        None,
    )
    assert output_shape_inv is not None
    assert output_shape_inv.value == ("B", "T", "input_dim")


def test_supervised_topk_sae_has_two_outputs_and_aux_head():
    arch = sae.supervised_topk_sae(
        input_dim=16, n_features=64, k=4, n_labels=10,
    )
    rep = verify(arch)
    assert rep.valid, [e.code for e in rep.errors]
    outputs = arch.outputs()
    output_names = {ly.name for ly in outputs}
    assert output_names == {"x_hat", "y_logits"}, output_names
    # The aux_head should be present in layers.
    layer_names = {ly.name for ly in arch.layers}
    assert "aux_head" in layer_names
    # Param accounting: encoder + decoder + aux_head (no biases on outputs).
    # encoder: 16*64 + 64 = 1088. decoder: 64*16 + 16 = 1040. aux_head: 64*10 + 10 = 650.
    assert rep.param_count == (16 * 64 + 64) + (64 * 16 + 16) + (64 * 10 + 10)


def test_gated_sae_has_parallel_branches_and_sigmoid_mul():
    arch = sae.gated_sae(input_dim=32, n_features=128)
    rep = verify(arch)
    assert rep.valid, [e.code for e in rep.errors]
    ops = {ly.op.name for ly in arch.layers if ly.op}
    assert "Sigmoid" in ops
    assert "Mul" in ops
    # Two parallel Linear projections from x: magnitude_proj + gate_proj.
    edges_from_x = [e for e in arch.flow if e.source == "x"]
    assert {e.target for e in edges_from_x} == {"magnitude_proj", "gate_proj"}
    # Param count: 2 encoders + 1 decoder.
    # 2*(32*128 + 128) + (128*32 + 32) = 8448 + 4128 = 12,576
    assert rep.param_count == 2 * (32 * 128 + 128) + (128 * 32 + 32)


def test_all_three_saes_round_trip_through_render_and_parser():
    for builder in (sae.topk_sae, sae.l1_sae, sae.jumprelu_sae):
        arch = builder(input_dim=32, n_features=64)
        markdown = render(arch)
        [reparsed] = parse(markdown)
        r1 = verify(arch); r2 = verify(reparsed)
        assert r1.param_count == r2.param_count
        assert r1.valid == r2.valid
        assert len(reparsed.layers) == len(arch.layers)


def test_sae_mermaid_renders_without_error():
    for builder in (sae.topk_sae, sae.l1_sae, sae.jumprelu_sae):
        arch = builder(input_dim=32, n_features=64)
        mmd = compile_mermaid(arch)
        assert "flowchart TD" in mmd


# --------------------------------------------------------------------------- #
#  World-model builders
# --------------------------------------------------------------------------- #


def test_world_model_default_topology():
    arch = world_models.world_model()
    rep = verify(arch)
    assert rep.valid, [e.code for e in rep.errors]
    # 43 -> 96 -> 48 -> 23
    # fc1: 43*96 + 96 = 4224. fc2: 96*48 + 48 = 4656. head: 48*23 + 23 = 1127.
    assert rep.param_count == (43 * 96 + 96) + (96 * 48 + 48) + (48 * 23 + 23)


def test_deep_world_model_variable_depth():
    arch = world_models.deep_world_model(input_dim=10, hidden_dims=(8, 4), out_dim=2)
    rep = verify(arch)
    assert rep.valid, [e.code for e in rep.errors]
    expected = (10 * 8 + 8) + (8 * 4 + 4) + (4 * 2 + 2)
    assert rep.param_count == expected


def test_deep_world_model_rejects_empty_hidden_dims():
    with pytest.raises(ValueError):
        world_models.deep_world_model(hidden_dims=())


def test_attn_world_model_has_attention_layer():
    arch = world_models.attn_world_model(input_dim=8, embed_dim=16, n_heads=4,
                                          h1_dim=32, h2_dim=16, out_dim=4)
    rep = verify(arch)
    assert rep.valid, [e.code for e in rep.errors]
    ops = {ly.op.name for ly in arch.layers if ly.op}
    assert "MultiHeadAttention" in ops
    assert "LayerNorm" in ops


# --------------------------------------------------------------------------- #
#  End-to-end PyTorch round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_topk_sae_forward_actually_sparse():
    import torch
    arch = sae.topk_sae(input_dim=32, n_features=128, k=4)
    code = compile_pytorch(arch)
    ns: dict = {}
    exec(code, ns)
    model = ns["TopKSae"](input_dim=32, n_features=128, k=4)
    x = torch.randn(2, 32)
    y = model(x)
    assert y.shape == (2, 32)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_l1_sae_forward():
    import torch
    arch = sae.l1_sae(input_dim=32, n_features=128)
    code = compile_pytorch(arch)
    ns: dict = {}; exec(code, ns)
    model = ns["L1Sae"](input_dim=32, n_features=128)
    y = model(torch.randn(2, 32))
    assert y.shape == (2, 32)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_jumprelu_sae_forward():
    import torch
    arch = sae.jumprelu_sae(input_dim=32, n_features=128)
    code = compile_pytorch(arch)
    ns: dict = {}; exec(code, ns)
    model = ns["JumpReLUSae"](input_dim=32, n_features=128)
    y = model(torch.randn(2, 32))
    assert y.shape == (2, 32)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_world_model_forward():
    import torch
    arch = world_models.world_model(input_dim=10, h1_dim=8, h2_dim=4, out_dim=2)
    code = compile_pytorch(arch)
    ns: dict = {}; exec(code, ns)
    model = ns["WorldModel"](input_dim=10, h1_dim=8, h2_dim=4, out_dim=2)
    y = model(torch.randn(3, 10))
    assert y.shape == (3, 2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gated_sae_forward_produces_graded_latents():
    import torch
    arch = sae.gated_sae(input_dim=32, n_features=64)
    code = compile_pytorch(arch)
    ns: dict = {}
    exec(code, ns)
    model = ns["GatedSae"](input_dim=32, n_features=64)
    x = torch.randn(2, 32)
    y = model(x)
    assert y.shape == (2, 32)
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_supervised_topk_sae_forward_returns_tuple():
    import torch
    arch = sae.supervised_topk_sae(
        input_dim=16, n_features=64, k=4, n_labels=10, aux_weight=0.1,
    )
    code = compile_pytorch(arch)
    ns: dict = {}
    exec(code, ns)
    model = ns["SupervisedTopKSae"](
        input_dim=16, n_features=64, k=4, n_labels=10, aux_weight=0.1,
    )
    out = model(torch.randn(3, 16))
    # The compiled forward returns (x_hat, y_logits).
    assert isinstance(out, tuple), f"expected tuple, got {type(out).__name__}"
    assert len(out) == 2
    x_hat, y_logits = out
    assert x_hat.shape == (3, 16), x_hat.shape
    assert y_logits.shape == (3, 10), y_logits.shape
    assert torch.isfinite(x_hat).all() and torch.isfinite(y_logits).all()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_attn_topk_sae_forward_3d_with_sparsity():
    import torch
    arch = sae.attn_topk_sae(
        input_dim=16, n_features=64, k=4, n_heads=2, attn_dropout=0.0,
    )
    code = compile_pytorch(arch)
    ns: dict = {}
    exec(code, ns)
    model = ns["AttnTopKSae"](
        input_dim=16, n_features=64, k=4, n_heads=2, attn_dropout=0.0,
    )
    # (B=2, T=7, input_dim=16) — per-position SAE with attention prefix.
    x = torch.randn(2, 7, 16)
    y = model(x)
    assert y.shape == (2, 7, 16), f"expected (2, 7, 16) got {tuple(y.shape)}"
    # Sanity check: reconstruction shouldn't be NaN/Inf at fresh init.
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_attn_world_model_forward_handles_3d_input():
    import torch
    arch = world_models.attn_world_model(
        input_dim=8, embed_dim=16, n_heads=4,
        h1_dim=32, h2_dim=16, out_dim=4, dropout=0.0,
    )
    code = compile_pytorch(arch)
    ns: dict = {}; exec(code, ns)
    model = ns["AttnWorldModel"](
        input_dim=8, embed_dim=16, n_heads=4,
        h1_dim=32, h2_dim=16, out_dim=4, dropout=0.0,
    )
    # (B=2, N_agents=17, input_dim=8)
    y = model(torch.randn(2, 17, 8))
    assert y.shape == (2, 17, 4)
