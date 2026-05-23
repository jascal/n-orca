"""Tests for the PyTorch compiler.

These tests exercise the *end-to-end pipeline*: parse → verify → emit
PyTorch source → exec → instantiate → forward-pass with a dummy tensor.
The forward-pass tests skip gracefully when torch is not installed.
"""
from __future__ import annotations

import importlib.util

import pytest

from n_orca.compiler import compile_pytorch
from n_orca.parser import parse_file


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _exec_compiled(arch):
    code = compile_pytorch(arch)
    ns: dict = {}
    exec(code, ns)
    # The generated class is the only nn.Module subclass we just defined.
    import torch  # noqa: F401
    return ns, code


def test_pytorch_source_is_syntactically_valid_for_all_examples():
    """Even without torch, the emitted source must compile (no syntax errors)."""
    import glob
    examples = sorted(glob.glob("examples/*.n.orca.md"))
    assert examples
    for path in examples:
        for arch in parse_file(path):
            code = compile_pytorch(arch)
            # Will raise SyntaxError on bad output.
            compile(code, path, "exec")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_forward():
    import torch
    arch = parse_file("examples/simple-mlp.n.orca.md")[0]
    ns, _ = _exec_compiled(arch)
    SimpleMLP = ns["SimpleMLP"]
    model = SimpleMLP()
    y = model(torch.randn(4, 784))
    assert y.shape == (4, 10)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_residual_block_forward():
    import torch
    arch = parse_file("examples/residual-block.n.orca.md")[0]
    ns, _ = _exec_compiled(arch)
    ResidualBlock = ns["ResidualBlock"]
    model = ResidualBlock(channels=8)
    y = model(torch.randn(2, 8, 16, 16))
    assert y.shape == (2, 8, 16, 16)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_transformer_block_forward():
    import torch
    arch = parse_file("examples/transformer-block.n.orca.md")[0]
    ns, _ = _exec_compiled(arch)
    TransformerBlock = ns["TransformerBlock"]
    model = TransformerBlock(d_model=32, n_heads=4, d_ff=64, dropout=0.0)
    y = model(torch.randn(2, 7, 32))
    assert y.shape == (2, 7, 32)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_conv_classifier_forward():
    import torch
    arch = parse_file("examples/conv-classifier.n.orca.md")[0]
    ns, _ = _exec_compiled(arch)
    ConvClassifier = ns["ConvClassifier"]
    model = ConvClassifier(in_c=3, n_classes=5)
    y = model(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_tiny_vit_forward():
    import torch
    arch = parse_file("examples/tiny-vit.n.orca.md")[0]
    ns, _ = _exec_compiled(arch)
    TinyViT = ns["TinyViT"]
    model = TinyViT(patch_dim=16, d_model=32, n_heads=4, d_ff=64, n_classes=5, dropout=0.0)
    y = model(torch.randn(2, 9, 16))
    assert y.shape == (2, 5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_unet_stub_forward():
    import torch
    arch = parse_file("examples/unet-stub.n.orca.md")[0]
    ns, _ = _exec_compiled(arch)
    UNetStub = ns["UNetStub"]
    model = UNetStub()  # f=16, in_c=3, out_c=1
    y = model(torch.randn(2, 3, 64, 64))
    assert y.shape == (2, 1, 64, 64)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_param_count_matches_pytorch_truth():
    """Verifier-reported param_count must match torch's reality for SimpleMLP."""
    import torch
    arch = parse_file("examples/simple-mlp.n.orca.md")[0]
    ns, _ = _exec_compiled(arch)
    model = ns["SimpleMLP"]()
    torch_params = sum(p.numel() for p in model.parameters())
    from n_orca.verifier import verify
    report = verify(arch)
    assert report.param_count == torch_params
