"""Adapter tests — canned configs, no network."""
from __future__ import annotations

import importlib.util

import pytest

from n_orca.hf import convert, UnsupportedModelError
from n_orca.hf.adapters import get_adapter, list_adapters
from n_orca.hf.adapters.jepa import JepaAdapter


HAS_TORCH = importlib.util.find_spec("torch") is not None


GPT2_CONFIG = {
    "_name_or_path": "gpt2",
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "n_embd": 768, "n_layer": 12, "n_head": 12, "n_inner": 3072,
    "n_positions": 1024, "vocab_size": 50257, "attn_pdrop": 0.1,
}

BERT_CONFIG = {
    "_name_or_path": "bert-base-uncased",
    "model_type": "bert",
    "architectures": ["BertForMaskedLM"],
    "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
    "intermediate_size": 3072, "max_position_embeddings": 512,
    "vocab_size": 30522, "hidden_dropout_prob": 0.1,
}

LLAMA_CONFIG = {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
    "intermediate_size": 11008, "max_position_embeddings": 4096,
    "vocab_size": 32000,
}

MISTRAL_CONFIG = {**LLAMA_CONFIG, "_name_or_path": "mistralai/Mistral-7B-v0.1",
                  "model_type": "mistral", "architectures": ["MistralForCausalLM"]}

QWEN_CONFIG = {**LLAMA_CONFIG, "_name_or_path": "Qwen/Qwen2-7B",
               "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}

ROBERTA_CONFIG = {**BERT_CONFIG, "_name_or_path": "roberta-base",
                  "model_type": "roberta", "architectures": ["RobertaForMaskedLM"]}


def test_adapter_registry_nonempty():
    assert len(list_adapters()) >= 3


def test_dispatch_gpt2():
    a = get_adapter(GPT2_CONFIG)
    assert a is not None
    assert "gpt2" in {k.lower() for k in a.model_types}


def test_dispatch_bert():
    a = get_adapter(BERT_CONFIG)
    assert a is not None
    assert "bert" in {k.lower() for k in a.model_types}


def test_dispatch_llama_family():
    for cfg in (LLAMA_CONFIG, MISTRAL_CONFIG, QWEN_CONFIG):
        a = get_adapter(cfg)
        assert a is not None, f"no adapter for {cfg['model_type']}"


def test_dispatch_roberta_via_bert_adapter():
    a = get_adapter(ROBERTA_CONFIG)
    assert a is not None


def test_unsupported_model_type():
    with pytest.raises(UnsupportedModelError):
        convert({"model_type": "made_up_arch_v2"})


def test_gpt2_conversion_verifies_clean():
    result = convert(GPT2_CONFIG)
    assert result.report.valid, [e.code for e in result.report.errors]
    assert result.architecture.name  # has a name
    # Should have N_layer blocks worth of layers + embeddings + LM head + IO.
    assert len(result.architecture.layers) >= 12 * 6


def test_bert_conversion_verifies_clean():
    result = convert(BERT_CONFIG)
    assert result.report.valid, [e.code for e in result.report.errors]


def test_llama_conversion_verifies_clean():
    result = convert(LLAMA_CONFIG)
    assert result.report.valid, [e.code for e in result.report.errors]
    # 32 layers, each with ~6 ops, plus embed/final/head ~= 196+
    assert len(result.architecture.layers) > 190


def test_conversion_emits_markdown_and_mermaid():
    result = convert(GPT2_CONFIG)
    assert "# architecture" in result.markdown
    assert "## layer input_ids [input]" in result.markdown
    assert "flowchart TD" in result.mermaid


def test_conversion_roundtrips_through_parser():
    """The emitted markdown must re-parse and verify identically."""
    from n_orca.parser import parse
    from n_orca.verifier import verify as v
    result = convert(GPT2_CONFIG)
    [re_arch] = parse(result.markdown)
    assert re_arch.name == result.architecture.name
    assert len(re_arch.layers) == len(result.architecture.layers)
    r2 = v(re_arch)
    assert r2.param_count == result.report.param_count


def test_conversion_name_override():
    result = convert(GPT2_CONFIG, name="MyGPT")
    assert result.architecture.name == "MyGPT"


def test_conversion_includes_source_provenance():
    result = convert({**GPT2_CONFIG, "_name_or_path": "gpt2"})
    # Source name appears in description when derived from config.
    assert "gpt2" in result.architecture.name.lower() or "Gpt2" in result.architecture.name


# --------------------------------------------------------------------------- #
#  JEPA / V-JEPA 2 / I-JEPA / LeWorldModel family
# --------------------------------------------------------------------------- #

# Real field names from facebook/vjepa2-vitl-fpc64-256 (downscaled depths/dims
# so the canned forward pass is fast).
VJEPA2_CONFIG = {
    "_name_or_path": "facebook/vjepa2-tiny-test",
    "model_type": "vjepa2", "architectures": ["VJEPA2Model"],
    "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
    "mlp_ratio": 2, "patch_size": 16, "image_size": 32, "crop_size": 32,
    "in_chans": 3, "frames_per_clip": 4, "tubelet_size": 2,
    "pred_hidden_size": 16, "pred_num_hidden_layers": 2,
    "pred_num_attention_heads": 4, "pred_mlp_ratio": 2, "pred_num_mask_tokens": 10,
    "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
}

# Real field names from facebook/ijepa_vith14_1k (image, encoder-only config).
IJEPA_CONFIG = {
    "_name_or_path": "facebook/ijepa-tiny-test",
    "model_type": "ijepa", "architectures": ["IJepaModel"],
    "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
    "intermediate_size": 64, "mlp_ratio": 2, "patch_size": 16, "image_size": 32,
    "num_channels": 3,
}

# Real (nested, Hydra) schema from quentinll/lewm-pusht — note: NO model_type
# and NO architectures, so dispatch must fall back to structural detection.
LEWM_CONFIG = {
    "_target_": "stable_worldmodel.wm.lewm.LeWM",
    "encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf",
                "size": "tiny", "patch_size": 16, "image_size": 32},
    "predictor": {"_target_": "stable_worldmodel.wm.lewm.module.Predictor",
                  "num_frames": 3, "input_dim": 192, "hidden_dim": 16,
                  "output_dim": 16, "depth": 2, "heads": 4, "mlp_dim": 32,
                  "dim_head": 64, "dropout": 0.0},
    "action_encoder": {"_target_": "stable_worldmodel.wm.lewm.module.Embedder",
                       "input_dim": 6, "emb_dim": 16},
    "projector": {"input_dim": 16, "output_dim": 16, "hidden_dim": 24},
}


def test_dispatch_vjepa2_and_ijepa_by_model_type():
    for cfg in (VJEPA2_CONFIG, IJEPA_CONFIG):
        a = get_adapter(cfg)
        assert isinstance(a, JepaAdapter), cfg["model_type"]


def test_dispatch_lewm_by_structure_without_model_type():
    # LeWM has no model_type / architectures — only a Hydra _target_ + nested
    # encoder/predictor dicts. The adapter must still claim it.
    assert "model_type" not in LEWM_CONFIG
    a = get_adapter(LEWM_CONFIG)
    assert isinstance(a, JepaAdapter)


def test_vjepa2_conversion_verifies_clean():
    result = convert(VJEPA2_CONFIG)
    assert result.report.valid, [e.code for e in result.report.errors]
    # Encoder + predictor blocks (2 each * 6 ops) + stem + heads + IO.
    assert len(result.architecture.layers) > 24
    # Two latent outputs, both in the encoder's embedding space.
    out_names = {ly.name for ly in result.architecture.outputs()}
    assert out_names == {"encoder_latents", "predicted_latents"}


def test_vjepa2_uses_tubelet_embed_and_records_mask_tokens():
    arch = convert(VJEPA2_CONFIG).architecture
    ops = {ly.op.name for ly in arch.layers if ly.op}
    assert "TubeletEmbed" in ops  # video patchifier (Conv3d)
    # Predictor mask-token count is preserved as a hyperparameter.
    assert arch.hyperparameter("pred_mask_tokens").default == 10
    # And surfaced as a verification rule for downstream world-model checks.
    assert any("mask-tokens" in r for r in arch.verification_rules)


def test_ijepa_image_uses_patch_embed_not_tubelet():
    arch = convert(IJEPA_CONFIG).architecture
    ops = {ly.op.name for ly in arch.layers if ly.op}
    assert "PatchEmbed" in ops
    assert "TubeletEmbed" not in ops


def test_lewm_has_action_conditioning_and_projector():
    arch = convert(LEWM_CONFIG).architecture
    layer_names = {ly.name for ly in arch.layers}
    # Action conditioning: a second input + embed + additive merge.
    assert {"actions", "action_embed", "add_action"} <= layer_names
    assert any(ly.name == "actions" and ly.is_input for ly in arch.layers)
    # Projector head off the predictor output.
    assert {"proj_fc1", "proj_act", "proj_fc2"} <= layer_names
    # SIGReg latent regularizer documented as a rule (it's a loss term).
    assert any("sigreg" in r.lower() for r in arch.verification_rules)


def test_lewm_latent_dim_consistency_invariant():
    arch = convert(LEWM_CONFIG).architecture
    # Both outputs share (B, num_patches, d_model).
    n = str(arch.hyperparameter("num_patches").default)
    inv = next(i for i in arch.invariants if i.kind == "output_shape")
    assert inv.value == ("B", n, "d_model")


@pytest.mark.parametrize("cfg", [VJEPA2_CONFIG, IJEPA_CONFIG, LEWM_CONFIG])
def test_jepa_round_trips_through_parser(cfg):
    from n_orca.parser import parse
    from n_orca.verifier import verify as v
    result = convert(cfg)
    [re_arch] = parse(result.markdown)
    assert re_arch.name == result.architecture.name
    assert len(re_arch.layers) == len(result.architecture.layers)
    r2 = v(re_arch)
    assert r2.valid, [e.code for e in r2.errors]
    assert r2.param_count == result.report.param_count


@pytest.mark.parametrize("cfg", [VJEPA2_CONFIG, IJEPA_CONFIG, LEWM_CONFIG])
def test_jepa_mermaid_renders(cfg):
    from n_orca.compiler import compile_mermaid
    mmd = compile_mermaid(convert(cfg).architecture)
    assert "flowchart TD" in mmd
    # The encoder->predictor branch point is present.
    assert "enc_norm" in mmd and "pred_in" in mmd


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_vjepa2_compiles_and_forwards_video_clip():
    import torch
    from n_orca.compiler import compile_pytorch
    arch = convert(VJEPA2_CONFIG, name="VJepa2Tiny").architecture
    ns: dict = {}
    exec(compile_pytorch(arch), ns)
    model = ns["VJepa2Tiny"]()
    # (B, C, T, H, W) video clip.
    enc, pred = model(torch.randn(2, 3, 4, 32, 32))
    # tokens = (4/2)*(32/16)*(32/16) = 2*2*2 = 8; encoder dim 32.
    assert tuple(enc.shape) == (2, 8, 32)
    assert tuple(pred.shape) == (2, 8, 32)
    assert torch.isfinite(enc).all() and torch.isfinite(pred).all()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_ijepa_encoder_only_config_still_compiles_and_forwards():
    # The HF I-JEPA config ships no pred_* fields; the predictor defaults to the
    # encoder width/heads so the compiled attention stays head-divisible.
    import torch
    from n_orca.compiler import compile_pytorch
    arch = convert(IJEPA_CONFIG, name="IJepaTiny").architecture
    ns: dict = {}
    exec(compile_pytorch(arch), ns)
    model = ns["IJepaTiny"]()
    enc, pred = model(torch.randn(2, 3, 32, 32))   # (32/16)^2 = 4 tokens, dim 32
    assert tuple(enc.shape) == (2, 4, 32)
    assert tuple(pred.shape) == (2, 4, 32)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_lewm_compiles_and_forwards_with_actions():
    import torch
    from n_orca.compiler import compile_pytorch
    arch = convert(LEWM_CONFIG, name="LeWmTiny").architecture
    ns: dict = {}
    exec(compile_pytorch(arch), ns)
    model = ns["LeWmTiny"]()
    n = arch.hyperparameter("num_patches").default      # (32/16)^2 = 4
    img = torch.randn(2, 3, 32, 32)                     # (B, C, H, W)
    actions = torch.randn(2, n, 6)                      # (B, N, action_dim)
    enc, pred = model(img, actions)
    d = arch.hyperparameter("d_model").default          # ViT-tiny = 192
    assert tuple(enc.shape) == (2, n, d)
    assert tuple(pred.shape) == (2, n, d)
    assert torch.isfinite(pred).all()
