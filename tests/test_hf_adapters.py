"""Adapter tests — canned configs, no network."""
from __future__ import annotations

import pytest

from n_orca.hf import convert, UnsupportedModelError
from n_orca.hf.adapters import get_adapter, list_adapters


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
