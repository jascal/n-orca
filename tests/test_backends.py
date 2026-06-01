"""Tests for the Unsloth runtime-capability backend (Increment A).

Covers the support table, model-type detection, the QLoRA VRAM estimator, the
unified `analyze` entry point, the verifier's Stage 6 (runtime) block, and the
`vram_estimate` invariant.
"""
from __future__ import annotations

from pathlib import Path

from n_orca import sae, world_models
from n_orca.backends import capability
from n_orca.backends.capability import (
    analyze,
    detect_model_type,
    estimate_lora_trainable_params,
    estimate_qlora_vram,
    family_of,
    support_for,
)
from n_orca.parser import parse, parse_file
from n_orca.verifier import verify

REPO_ROOT = Path(__file__).resolve().parents[1]
HF = REPO_ROOT / "examples" / "hf-generated"


def _arch(path: Path):
    archs = parse_file(path)
    return archs[0]


# --------------------------------------------------------------------------- #
#  Support table + family resolution
# --------------------------------------------------------------------------- #


def test_supported_decoder_families():
    for mt in ("llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "phi", "phi3", "mixtral"):
        info = support_for(mt)
        assert info is not None and info.supported, mt
        assert info.loader == "FastLanguageModel"
        assert "q_proj" in info.default_target_modules


def test_model_type_aliases_resolve():
    assert family_of("qwen2_moe") == "qwen2"
    assert family_of("tinyllama") == "llama"
    assert family_of("deepseek") == "llama"
    assert family_of("gemma3") == "gemma2"
    assert family_of("roberta") == "bert"
    assert family_of("vjepa2") == "jepa"


def test_known_unsupported_families():
    for mt in ("gpt2", "bert", "t5", "esm", "jepa"):
        info = support_for(mt)
        assert info is not None and not info.supported, mt
        assert info.loader == ""


def test_unknown_model_type_is_none():
    assert family_of("totally-made-up") is None
    assert support_for("totally-made-up") is None
    assert family_of(None) is None


# --------------------------------------------------------------------------- #
#  Model-type detection from architectures
# --------------------------------------------------------------------------- #


def test_detect_model_type_from_hf_examples():
    assert detect_model_type(_arch(HF / "llama-7b.n.orca.md")) == "llama"
    assert detect_model_type(_arch(HF / "tinyllama-2L.n.orca.md")) == "llama"
    assert detect_model_type(_arch(HF / "gpt2-small.n.orca.md")) == "gpt2"
    assert detect_model_type(_arch(HF / "bert-base-uncased.n.orca.md")) == "bert"
    assert detect_model_type(_arch(HF / "vjepa2.n.orca.md")) == "jepa"


def test_detect_model_type_custom_is_none():
    assert detect_model_type(sae.topk_sae()) is None
    assert detect_model_type(sae.attn_topk_sae()) is None
    assert detect_model_type(world_models.world_model()) is None
    assert detect_model_type(world_models.attn_world_model()) is None


def test_word_boundary_avoids_false_positives():
    # "phi" must not match inside "morphism"; "t5" not inside "test5".
    from n_orca.ast import Architecture

    a = Architecture(name="MorphismNet", description="an isomorphism over test5 data")
    assert detect_model_type(a) is None


# --------------------------------------------------------------------------- #
#  VRAM estimation
# --------------------------------------------------------------------------- #


def test_estimate_qlora_vram_breakdown_sums_and_base_is_half_byte():
    est = estimate_qlora_vram(
        param_count=7_000_000_000,
        hyperparameters={"d_model": 4096, "n_layer": 32, "d_ff": 11008,
                         "num_attention_heads": 32},
        lora_r=16, batch_size=1, max_seq_length=2048,
    )
    assert est.breakdown["base_4bit"] == 3_500_000_000          # 7e9 * 0.5
    assert est.total_bytes == sum(est.breakdown.values())
    assert est.assumptions["lora_r"] == 16
    assert est.assumptions["quantization"] == "4bit-nf4-double-quant"
    # A 7B QLoRA fine-tune should land in a sane single-GPU ballpark.
    assert 3.0 < est.total_gib < 24.0


def test_estimate_lora_trainable_params_formula():
    # per layer: attn = 4 * r*(d+d); mlp = 2*r*(d+d_ff) + r*(d_ff+d)
    tp = estimate_lora_trainable_params(
        {"d_model": 16, "n_layer": 2, "d_ff": 64}, lora_r=8,
    )
    attn = 4 * (8 * (16 + 16))
    mlp = 2 * (8 * (16 + 64)) + (8 * (64 + 16))
    assert tp == 2 * (attn + mlp)


def test_estimate_lora_trainable_params_missing_widths_is_zero():
    assert estimate_lora_trainable_params({"d_model": 16}, lora_r=8) == 0
    assert estimate_lora_trainable_params(None) == 0


# --------------------------------------------------------------------------- #
#  Unified analyze()
# --------------------------------------------------------------------------- #


def test_analyze_llama_is_supported():
    arch = _arch(HF / "llama-7b.n.orca.md")
    rep = verify(arch)
    cap = analyze(arch=arch, param_count=rep.param_count)
    assert cap.unsloth_supported is True
    assert cap.family == "llama"
    assert cap.loader == "FastLanguageModel"
    assert "q_proj" in cap.default_target_modules
    assert cap.vram_estimate.breakdown["base_4bit"] == int(rep.param_count * 0.5)


def test_analyze_gpt2_is_unsupported():
    arch = _arch(HF / "gpt2-small.n.orca.md")
    cap = analyze(arch=arch, param_count=verify(arch).param_count)
    assert cap.unsloth_supported is False
    assert cap.family == "gpt2"
    assert cap.loader == ""


def test_analyze_custom_architecture_is_unsupported():
    arch = sae.topk_sae()
    cap = analyze(arch=arch, param_count=verify(arch).param_count)
    assert cap.unsloth_supported is False
    assert cap.family is None
    assert "custom" in cap.note.lower()
    # Still gets a VRAM estimate (the generic PyTorch path can still be trained).
    assert cap.vram_estimate is not None


def test_analyze_gpu_budget_fit_flag():
    small = analyze(model_type="llama", param_count=1_000_000_000,
                    hyperparameters={"d_model": 2048, "n_layer": 16, "d_ff": 5632},
                    gpu_memory_gb=24)
    assert small.fits_in_gpu is True
    huge = analyze(model_type="llama", param_count=70_000_000_000,
                   hyperparameters={"d_model": 8192, "n_layer": 80, "d_ff": 28672},
                   gpu_memory_gb=8)
    assert huge.fits_in_gpu is False


def test_analyze_to_dict_is_json_shaped():
    cap = analyze(model_type="mistral", param_count=7_000_000_000,
                  hyperparameters={"d_model": 4096, "n_layer": 32, "d_ff": 14336},
                  gpu_memory_gb=16)
    d = cap.to_dict()
    assert d["unsloth_supported"] is True
    assert d["vram_estimate"]["total_gib"] > 0
    assert "fits_in_gpu" in d
    assert isinstance(d["patched_classes"], list)


# --------------------------------------------------------------------------- #
#  Verifier Stage 6 (runtime) + vram_estimate invariant
# --------------------------------------------------------------------------- #


def test_verifier_stage6_populates_runtime_block():
    rep = verify(sae.topk_sae())
    assert rep.runtime["unsloth_supported"] is False
    assert "vram_estimate" in rep.runtime
    assert rep.runtime["vram_estimate"]["total_gib"] > 0

    arch = _arch(HF / "llama-7b.n.orca.md")
    rep = verify(arch)
    assert rep.runtime["unsloth_supported"] is True
    assert rep.runtime["family"] == "llama"
    # Stage 6 is informational — it never invalidates a sound architecture.
    assert not any(e.code.startswith("UNSLOTH") for e in rep.errors)


_TINY = """
# architecture Tiny
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| d | int | 16 |
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
- vram_estimate {OP} {BOUND}
"""


def _verify_tiny(op: str, bound: str):
    [arch] = parse(_TINY.replace("{OP}", op).replace("{BOUND}", bound))
    return verify(arch)


def test_vram_estimate_invariant_fails_under_tiny_budget():
    rep = _verify_tiny("<=", "1K")
    assert not rep.valid
    assert any(e.code == "VRAM_BUDGET_EXCEEDED" for e in rep.errors)
    assert rep.runtime["vram_estimate"]["total_bytes"] > 1000


def test_vram_estimate_invariant_passes_under_generous_budget():
    rep = _verify_tiny("<=", "100G")
    assert rep.valid, rep.errors


def test_vram_estimate_invariant_parses_si_suffix():
    [arch] = parse(_TINY.replace("{OP}", "<=").replace("{BOUND}", "24G"))
    inv = [i for i in arch.invariants if i.kind == "vram_estimate"][0]
    assert inv.value == 24_000_000_000
    assert inv.op == "<="
