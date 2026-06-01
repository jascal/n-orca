"""Tests for the Unsloth runtime-capability backend (Increment A, hardened).

Covers the verified three-state support table, model-type detection (incl. the
authoritative metadata path + `## runtime` round-trip), the calibrated QLoRA
VRAM range estimator and its applicability gating, the unified `analyze` entry
point, the verifier's Stage 6 (runtime), and the `vram_estimate` invariant
(now a warning, not an error).
"""
from __future__ import annotations

from pathlib import Path

from n_orca import sae, world_models
from n_orca.ast import Architecture
from n_orca.backends.capability import (
    analyze,
    detect_model_type,
    estimate_lora_trainable_params,
    estimate_qlora_vram,
    family_of,
    is_decoder_lm,
    support_for,
    vram_applicable,
)
from n_orca.parser import parse, parse_file
from n_orca.render import render
from n_orca.verifier import verify

REPO_ROOT = Path(__file__).resolve().parents[1]
HF = REPO_ROOT / "examples" / "hf-generated"


def _arch(path: Path):
    return parse_file(path)[0]


# --------------------------------------------------------------------------- #
#  Support table — three states, verified entries only
# --------------------------------------------------------------------------- #

_SUPPORTED = ("llama", "mistral", "qwen2", "qwen3", "qwen3_moe",
              "gemma", "gemma2", "cohere", "granite", "falcon_h1")


def test_supported_decoder_families():
    for mt in _SUPPORTED:
        info = support_for(mt)
        assert info is not None and info.status == "supported", mt
        assert info.loader == "FastLanguageModel"
        assert info.is_decoder_lm
        assert "q_proj" in info.default_target_modules


def test_only_llama_lists_patched_classes():
    # We do not fabricate per-family class names — only Llama's are source-stated.
    assert support_for("llama").patched_classes == (
        "LlamaAttention", "LlamaDecoderLayer", "LlamaModel")
    assert support_for("mistral").patched_classes == ()
    # Verified Fast* loader classes are recorded where known.
    assert support_for("gemma").fast_class == "FastGemmaModel"
    assert support_for("qwen3_moe").fast_class == "FastQwen3MoeModel"


def test_structurally_unsupported_families():
    for mt in ("bert", "t5", "esm", "jepa"):
        info = support_for(mt)
        assert info is not None and info.status == "unsupported", mt
        assert not info.is_decoder_lm


def test_decoder_lms_outside_snapshot_are_unknown():
    # Decoder LMs we haven't verified in the snapshot are `unknown`, NOT claimed
    # supported or unsupported — but a QLoRA estimate still applies to them.
    for mt in ("gpt2", "gpt_neox", "phi", "phi3"):
        info = support_for(mt)
        assert info is not None and info.status == "unknown", mt
        assert info.is_decoder_lm


def test_aliases_resolve():
    assert family_of("qwen2_moe") == "qwen3_moe"
    assert family_of("tinyllama") == "llama"
    assert family_of("gemma3") == "gemma2"
    assert family_of("roberta") == "bert"
    assert family_of("vjepa2") == "jepa"


def test_unknown_and_speculative_model_types_are_none():
    assert family_of("totally-made-up") is None
    assert support_for("totally-made-up") is None
    assert family_of(None) is None
    # We intentionally do NOT guess speculative families — better `unknown`
    # than a wrong `supported`.
    assert family_of("deepseek") is None
    assert family_of("starcoder2") is None
    assert family_of("yi") is None


# --------------------------------------------------------------------------- #
#  Model-type detection
# --------------------------------------------------------------------------- #


def test_detect_prefers_explicit_metadata():
    a = Architecture(name="MysteryNet", metadata={"model_type": "qwen2"})
    assert detect_model_type(a) == "qwen2"


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
    a = Architecture(name="MorphismNet", description="an isomorphism over test5 data")
    assert detect_model_type(a) is None


def test_is_decoder_lm():
    assert is_decoder_lm("llama")
    assert is_decoder_lm("gpt2")          # unknown status, but a decoder LM
    assert not is_decoder_lm("bert")
    assert not is_decoder_lm(None)        # custom -> not assumed to be a decoder


# --------------------------------------------------------------------------- #
#  VRAM estimation — range + calibration anchors
# --------------------------------------------------------------------------- #


def test_estimate_range_and_breakdown_8b_anchor():
    # Calibration anchor: Llama-3.1-8B QLoRA (batch=2, r=32, seq=2048) < 10 GB.
    est = estimate_qlora_vram(
        param_count=8_030_000_000,
        hyperparameters={"d_model": 4096, "n_layer": 32, "d_ff": 14336},
        lora_r=32, batch_size=2, max_seq_length=2048,
    )
    assert est.breakdown["base_4bit"] == 4_015_000_000
    assert est.central_bytes == sum(est.breakdown.values())
    assert est.low_bytes < est.central_bytes < est.high_bytes
    assert est.central_gib < 10.0


def test_estimate_calibrates_to_70b_anchor():
    # Anchor: Llama-3.1-70B QLoRA fits on a 48 GB GPU at ~7K ctx (batch=2, r=32).
    est = estimate_qlora_vram(
        param_count=70_600_000_000,
        hyperparameters={"d_model": 8192, "n_layer": 80, "d_ff": 28672},
        lora_r=32, batch_size=2, max_seq_length=7168,
    )
    assert 40.0 < est.central_gib < 56.0


def test_estimate_lora_trainable_params_formula():
    tp = estimate_lora_trainable_params(
        {"d_model": 16, "n_layer": 2, "d_ff": 64}, lora_r=8,
    )
    attn = 4 * (8 * (16 + 16))
    mlp = 2 * (8 * (16 + 64)) + (8 * (64 + 16))
    assert tp == 2 * (attn + mlp)


def test_estimate_lora_trainable_params_missing_widths_is_zero():
    assert estimate_lora_trainable_params({"d_model": 16}, lora_r=8) == 0
    assert estimate_lora_trainable_params(None) == 0


def test_vram_applicable_gating():
    ok, _ = vram_applicable("llama", 7_000_000_000)
    assert ok
    # Non-decoder -> not applicable.
    ok, why = vram_applicable(None, 7_000_000_000)
    assert not ok and "decoder" in why.lower()
    # Below the size floor -> not applicable.
    ok, why = vram_applicable("llama", 5_000_000)
    assert not ok and "floor" in why.lower()


# --------------------------------------------------------------------------- #
#  Unified analyze()
# --------------------------------------------------------------------------- #


def test_analyze_llama_is_supported_with_estimate():
    arch = _arch(HF / "llama-7b.n.orca.md")
    rep = verify(arch)
    cap = analyze(arch=arch, param_count=rep.param_count)
    assert cap.unsloth_status == "supported"
    assert cap.family == "llama"
    assert cap.loader == "FastLanguageModel"
    assert "q_proj" in cap.default_target_modules
    assert cap.vram_estimate is not None
    assert cap.vram_estimate.breakdown["base_4bit"] == int(rep.param_count * 0.5)


def test_analyze_decoder_above_floor_gets_estimate():
    cap = analyze(model_type="gpt2", param_count=124_000_000,
                  hyperparameters={"d_model": 768, "n_layer": 12, "d_ff": 3072})
    assert cap.unsloth_status == "unknown"   # gpt2 not in verified snapshot
    assert cap.is_decoder_lm
    assert cap.vram_estimate is not None


def test_analyze_below_floor_has_no_estimate():
    cap = analyze(model_type="llama", param_count=5_000_000,
                  hyperparameters={"d_model": 256, "n_layer": 4, "d_ff": 1024})
    assert cap.vram_estimate is None
    assert "floor" in cap.vram_note.lower()


def test_analyze_custom_is_unknown_without_estimate():
    arch = sae.topk_sae()
    cap = analyze(arch=arch, param_count=verify(arch).param_count)
    assert cap.unsloth_status == "unknown"
    assert cap.family is None
    assert not cap.is_decoder_lm
    assert cap.vram_estimate is None        # QLoRA estimate doesn't apply to an SAE


def test_analyze_structurally_unsupported_has_no_estimate():
    arch = _arch(HF / "bert-base-uncased.n.orca.md")
    cap = analyze(arch=arch, param_count=verify(arch).param_count)
    assert cap.unsloth_status == "unsupported"
    assert cap.vram_estimate is None


def test_analyze_gpu_budget_flags():
    cap = analyze(model_type="llama", param_count=8_030_000_000,
                  hyperparameters={"d_model": 4096, "n_layer": 32, "d_ff": 14336},
                  lora_r=16, batch_size=1, max_seq_length=2048, gpu_memory_gb=24)
    assert cap.fits_in_gpu is True
    assert cap.fits_confidently is True
    tight = analyze(model_type="llama", param_count=70_600_000_000,
                    hyperparameters={"d_model": 8192, "n_layer": 80, "d_ff": 28672},
                    lora_r=32, batch_size=2, max_seq_length=7168, gpu_memory_gb=48)
    assert tight.fits_in_gpu is True          # central ~46 GiB < 48
    assert tight.fits_confidently is False    # but the high tail exceeds 48


def test_analyze_to_dict_shape():
    cap = analyze(model_type="mistral", param_count=7_000_000_000,
                  hyperparameters={"d_model": 4096, "n_layer": 32, "d_ff": 14336},
                  gpu_memory_gb=16)
    d = cap.to_dict()
    assert d["unsloth_status"] == "supported"
    assert d["vram_estimate"]["central_gib"] > 0
    assert "low_gib" in d["vram_estimate"] and "high_gib" in d["vram_estimate"]
    assert "fits_in_gpu" in d and "fits_confidently" in d
    assert isinstance(d["patched_classes"], list)
    assert "support_verified" in d


# --------------------------------------------------------------------------- #
#  Verifier Stage 6 (runtime) + vram_estimate invariant (warning)
# --------------------------------------------------------------------------- #


def test_stage6_runtime_block():
    rep = verify(sae.topk_sae())
    assert rep.runtime["unsloth_status"] == "unknown"
    assert rep.runtime["vram_estimate"] is None

    arch = _arch(HF / "llama-7b.n.orca.md")
    rep = verify(arch)
    assert rep.runtime["unsloth_status"] == "supported"
    assert rep.runtime["family"] == "llama"
    assert rep.runtime["vram_estimate"]["central_gib"] > 0
    # Stage 6 never invalidates a sound architecture.
    assert not any(e.code.startswith(("UNSLOTH", "VRAM")) for e in rep.errors)


# A synthetic, sizeable "decoder LM" (declared via `## runtime`) so the QLoRA
# estimate is produced and a budget can be exceeded.
_DECODER_TINY = """
# architecture MiniDecoder
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| d | int | 12000 |
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
## runtime
- model_type: llama
"""


def _verify_decoder(op: str, bound: str):
    [arch] = parse(_DECODER_TINY.replace("{OP}", op).replace("{BOUND}", bound))
    return arch, verify(arch)


def test_runtime_section_parsed_from_markdown():
    arch, _ = _verify_decoder("<=", "100G")
    assert arch.metadata["model_type"] == "llama"


def test_vram_budget_exceeded_is_warning_not_error():
    _, rep = _verify_decoder("<=", "1K")
    assert rep.valid                       # warning — never invalidates
    assert any(w.code == "VRAM_BUDGET_EXCEEDED" for w in rep.warnings)
    assert rep.runtime["vram_estimate"]["central_bytes"] > 1000


def test_vram_budget_passes_under_generous_budget():
    _, rep = _verify_decoder("<=", "100G")
    assert rep.valid
    assert not any(w.code == "VRAM_BUDGET_EXCEEDED" for w in rep.warnings)


def test_vram_invariant_not_applicable_for_custom():
    src = """
# architecture TinyMlp
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
- vram_estimate <= 1K
"""
    [arch] = parse(src)
    rep = verify(arch)
    assert rep.valid
    assert any(w.code == "VRAM_ESTIMATE_NOT_APPLICABLE" for w in rep.warnings)
    assert rep.runtime["vram_estimate"] is None


def test_strict_promotes_vram_warning_to_error():
    [arch] = parse(_DECODER_TINY.replace("{OP}", "<=").replace("{BOUND}", "1K"))
    assert verify(arch).valid
    assert not verify(arch, strict=True).valid


# --------------------------------------------------------------------------- #
#  Provenance round-trip + convert stamping
# --------------------------------------------------------------------------- #


def test_runtime_section_round_trips_through_render():
    [arch] = parse(_DECODER_TINY.replace("{OP}", "<=").replace("{BOUND}", "100G"))
    assert arch.metadata == {"model_type": "llama"}
    [again] = parse(render(arch))
    assert again.metadata == {"model_type": "llama"}


def test_convert_stamps_model_type():
    from n_orca.hf import convert

    cfg = {
        "model_type": "gpt2",
        "n_embd": 32, "n_layer": 1, "n_head": 4, "n_inner": 64,
        "n_positions": 16, "vocab_size": 100,
    }
    result = convert(cfg)
    assert result.architecture.metadata.get("model_type") == "gpt2"
    assert "## runtime" in result.markdown
