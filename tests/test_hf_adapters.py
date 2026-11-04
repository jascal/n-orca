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

LFM3_CONFIG = {
    "_name_or_path": "LiquidAI/LFM3-1.7B-Base", "model_type": "lfm3",
    "architectures": ["Lfm3ForCausalLM"], "vocab_size": 128000,
    "hidden_size": 2048, "intermediate_size": 7680, "num_hidden_layers": 24,
    "num_attention_heads": 16, "num_key_value_heads": 4, "conv_L_cache": 7,
    "sliding_window": 256, "max_position_embeddings": 262144,
    "layer_types": [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    ],
}

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


def test_lfm3_conversion_preserves_tri_hybrid_schedule():
    result = convert(LFM3_CONFIG)
    assert result.report.valid, [e.code for e in result.report.errors]
    operators = [ly.op.name for ly in result.architecture.layers if ly.op]
    assert operators.count("GatedCausalShortConv") == 13
    assert operators.count("HeadGatedGroupedQueryAttention") == 11
    assert "256-token sliding window" in result.markdown
    assert "half-head partial RoPE" in result.markdown
    assert "does not share their generated PyTorch weights" in result.markdown


def test_lfm3_missing_layer_types_warns_before_using_default_schedule():
    config = {key: value for key, value in LFM3_CONFIG.items() if key != "layer_types"}
    with pytest.warns(UserWarning, match="no layer_types"):
        result = convert(config)
    assert result.report.valid, [e.code for e in result.report.errors]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_lfm3_generated_pytorch_is_importable_and_shape_compatible():
    result = convert({**LFM3_CONFIG, "hidden_size": 32, "intermediate_size": 64,
                      "num_attention_heads": 4, "num_key_value_heads": 1,
                      "vocab_size": 128})
    from n_orca.compiler import compile_pytorch
    ns: dict = {}
    exec(compile_pytorch(result.architecture), ns)
    model = ns[result.architecture.name](vocab_size=128, d_model=32, n_heads=4,
                                         n_kv_heads=1, d_ff=64)
    import torch
    assert model(torch.ones(2, 3, dtype=torch.long)).shape == (2, 3, 128)


def _dory_config(**overrides):
    from n_orca.hf.adapters.dory import _default_types
    cfg = {
        "_name_or_path": "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
        "model_type": "dory", "architectures": ["DoryForCausalLM"],
        "vocab_size": 131072, "hidden_size": 2560, "num_hidden_layers": 72,
        "intermediate_size": 7168, "num_attention_heads": 16,
        "num_key_value_heads": 4, "head_dim": 256, "sliding_window": 1024,
        "max_position_embeddings": 262144, "tie_word_embeddings": False,
        "n_input_layers": 16, "n_recurrent_layers": 40, "n_output_layers": 16,
        "n_recurrent_loops": 2, "layer_types": _default_types(72),
    }
    cfg.update(overrides)
    return cfg


def test_dory_default_schedule_matches_checkpoint_pattern():
    from n_orca.hf.adapters.dory import _default_types
    types = _default_types(72)
    assert types[:8] == ["sliding_attention", "mlp", "sliding_attention", "mlp",
                         "sliding_attention", "mlp", "full_attention", "mlp"]
    assert types.count("full_attention") == 9 and types.count("mlp") == 36


def test_dory_conversion_unrolls_loops_with_input_reinjection():
    result = convert(_dory_config())
    arch = result.architecture
    assert result.report.valid, [e.code for e in result.report.errors]
    assert [g.name for g in arch.groups] == [
        "input_block", "recurrent_loop_1", "recurrent_loop_2", "output_block"]
    ops = [ly.op.name for ly in arch.layers if ly.op]
    # 8 input + 20 recurrent x 2 loops + 8 output attention/MLP sublayers.
    assert ops.count("GroupedQueryAttention") == 56
    assert ops.count("FeedForward") == 56
    h_in = "in_mlp_15_norm"
    for k, prev in ((1, h_in), (2, "loop1_mlp_55_norm")):
        srcs = sorted(e.source for e in arch.flow if e.target == f"loop{k}_inject")
        assert srcs == sorted([prev, h_in])
    assert {e.source for e in arch.flow if e.target == "out_swa_56"} == {"loop2_mlp_55_norm"}
    assert 'subgraph g_recurrent_loop_2["recurrent_loop_2"]' in result.mermaid


def test_dory_loop_count_follows_config():
    arch = convert(_dory_config(n_recurrent_loops=4)).architecture
    assert sum(g.name.startswith("recurrent_loop_") for g in arch.groups) == 4


def test_dory_rejects_inconsistent_split():
    with pytest.raises(ValueError, match="n_input_layers"):
        convert(_dory_config(n_recurrent_layers=38))


def test_gqa_params_use_explicit_head_dim():
    from n_orca.ops.spec import get_op
    spec = get_op("GroupedQueryAttention")
    # q/o: 2560 x (16*256); k/v: 2560 x (4*256)
    assert spec.params(["2560", "16", "4", "256", "0"], []) == 2 * 2560 * 4096 + 2 * 2560 * 1024


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_dory_generated_pytorch_is_shape_compatible():
    result = convert(_dory_config(hidden_size=32, intermediate_size=64, head_dim=8,
                                  num_attention_heads=4, num_key_value_heads=1,
                                  vocab_size=128))
    from n_orca.compiler import compile_pytorch
    ns: dict = {}
    exec(compile_pytorch(result.architecture), ns)
    model = ns[result.architecture.name](vocab_size=128, d_model=32, n_heads=4,
                                         n_kv_heads=1, head_dim=8, d_ff=64)
    import torch
    assert model(torch.ones(2, 3, dtype=torch.long)).shape == (2, 3, 128)


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
