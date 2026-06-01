"""Unsloth runtime-capability analysis + QLoRA VRAM estimation.

Pure-Python, **no torch / transformers / peft / unsloth imports** and no GPU
access. This module answers two questions for the AI design loop, from the
static architecture alone:

1. **Backend coverage** — can the Unsloth runtime load and fine-tune this
   architecture? Unsloth only fast-paths a *curated* set of HuggingFace model
   families (decoder LLMs in the Llama lineage, plus some VLMs); everything
   else falls back to the generic PyTorch path. We mirror Unsloth's
   ``model_type``-based dispatch (its ``unsloth/models/mapper.py``) with a
   static support table so the answer needs nothing installed.

2. **QLoRA memory footprint** — roughly how much GPU memory would a 4-bit
   QLoRA fine-tune take? The estimate is a transparent, documented breakdown
   (4-bit base weights + LoRA adapters + optimizer states + gradients +
   activations). It is **best-effort (treat as ±50%)** — useful as a budget
   gate (``vram_estimate <= 24G``), not a guarantee.

Design-time vs runtime: n-orca declares/verifies; Unsloth loads/trains/serves.
This module is the seam — it reports what the runtime backend *could* do with a
given design, joined at the same ``model_type`` key both projects dispatch on.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
#  Unsloth support table (mirrors unsloth/models/mapper.py dispatch)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SupportInfo:
    """What the Unsloth runtime can do with a given model family."""

    family: str
    supported: bool
    loader: str                              # "FastLanguageModel" | "FastVisionModel" | ""
    patched_classes: tuple[str, ...] = ()    # transformers classes Unsloth replaces
    default_target_modules: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "supported": self.supported,
            "loader": self.loader,
            "patched_classes": list(self.patched_classes),
            "default_target_modules": list(self.default_target_modules),
            "note": self.note,
        }


# Decoder-LLM LoRA targets — Unsloth's default `target_modules` for the Llama
# lineage (attention q/k/v/o + gated-MLP gate/up/down projections).
_LLM_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Family -> what the runtime can do. The Llama lineage (Mistral/Qwen/Gemma/Phi/
# Mixtral/...) is all driven through unsloth/models/llama.py's fast forwards, so
# they share one SupportInfo shape. Encoder-only / seq2seq / JEPA-style models
# are *known but unsupported* — Unsloth has no fast path for them.
_SUPPORT: dict[str, SupportInfo] = {
    "llama": SupportInfo(
        "llama", True, "FastLanguageModel",
        ("LlamaAttention", "LlamaDecoderLayer", "LlamaModel"),
        _LLM_TARGETS,
        "Decoder-only LLM — fused RoPE/RMSNorm/cross-entropy kernels + QLoRA.",
    ),
    "mistral": SupportInfo(
        "mistral", True, "FastLanguageModel",
        ("MistralAttention", "MistralDecoderLayer", "MistralModel"),
        _LLM_TARGETS, "Llama-lineage decoder LLM (sliding-window attention).",
    ),
    "mixtral": SupportInfo(
        "mixtral", True, "FastLanguageModel",
        ("MixtralAttention", "MixtralDecoderLayer", "MixtralModel"),
        _LLM_TARGETS, "Sparse-MoE decoder LLM in the Llama lineage.",
    ),
    "qwen2": SupportInfo(
        "qwen2", True, "FastLanguageModel",
        ("Qwen2Attention", "Qwen2DecoderLayer", "Qwen2Model"),
        _LLM_TARGETS, "Qwen2/2.5 decoder LLM (Llama-style fast path).",
    ),
    "qwen3": SupportInfo(
        "qwen3", True, "FastLanguageModel",
        ("Qwen3Attention", "Qwen3DecoderLayer", "Qwen3Model"),
        _LLM_TARGETS, "Qwen3 decoder LLM (Llama-style fast path).",
    ),
    "gemma": SupportInfo(
        "gemma", True, "FastLanguageModel",
        ("GemmaAttention", "GemmaDecoderLayer", "GemmaModel"),
        _LLM_TARGETS, "Gemma decoder LLM (GeGLU MLP, Llama-style fast path).",
    ),
    "gemma2": SupportInfo(
        "gemma2", True, "FastLanguageModel",
        ("Gemma2Attention", "Gemma2DecoderLayer", "Gemma2Model"),
        _LLM_TARGETS, "Gemma2 decoder LLM (Llama-style fast path).",
    ),
    "phi": SupportInfo(
        "phi", True, "FastLanguageModel",
        ("PhiAttention", "PhiDecoderLayer", "PhiModel"),
        _LLM_TARGETS, "Phi decoder LLM (Llama-style fast path).",
    ),
    "phi3": SupportInfo(
        "phi3", True, "FastLanguageModel",
        ("Phi3Attention", "Phi3DecoderLayer", "Phi3Model"),
        _LLM_TARGETS, "Phi-3 decoder LLM (Llama-style fast path).",
    ),
    "cohere": SupportInfo(
        "cohere", True, "FastLanguageModel",
        ("CohereAttention", "CohereDecoderLayer", "CohereModel"),
        _LLM_TARGETS, "Command-R lineage decoder LLM.",
    ),
    "qwen2_vl": SupportInfo(
        "qwen2_vl", True, "FastVisionModel",
        ("Qwen2VLAttention", "Qwen2VLDecoderLayer"),
        _LLM_TARGETS, "Vision-language model — load via FastVisionModel.",
    ),
    "mllama": SupportInfo(
        "mllama", True, "FastVisionModel",
        ("MllamaSelfAttentionDecoderLayer", "MllamaCrossAttentionDecoderLayer"),
        _LLM_TARGETS, "Llama-3.2 vision — load via FastVisionModel.",
    ),
    "llava": SupportInfo(
        "llava", True, "FastVisionModel",
        (), _LLM_TARGETS, "LLaVA VLM — load via FastVisionModel.",
    ),
    # Known but unsupported — Unsloth has no fused fast path for these.
    "gpt2": SupportInfo(
        "gpt2", False, "",
        note="GPT-2 family is not in Unsloth's fast path; use the generic "
             "PyTorch / transformers path.",
    ),
    "gpt_neox": SupportInfo(
        "gpt_neox", False, "",
        note="GPT-NeoX family is not in Unsloth's fast path.",
    ),
    "bert": SupportInfo(
        "bert", False, "",
        note="Encoder-only (BERT/RoBERTa/ELECTRA) — not a decoder LLM Unsloth "
             "fast-paths; fine-tune with the generic transformers path.",
    ),
    "t5": SupportInfo(
        "t5", False, "",
        note="Encoder-decoder (T5) — outside Unsloth's decoder-LLM fast path.",
    ),
    "esm": SupportInfo(
        "esm", False, "",
        note="ESM protein LM (encoder-only) — outside Unsloth's fast path.",
    ),
    "jepa": SupportInfo(
        "jepa", False, "",
        note="JEPA / V-JEPA / I-JEPA / LeWorldModel are joint-embedding world "
             "models, not HF decoder LLMs — no Unsloth fast path; train with "
             "the generic PyTorch path.",
    ),
}

# Canonical model_type for fuzzy keys (HF `model_type` values not already a key
# above are mapped onto a supported family here).
_MODEL_TYPE_ALIASES: dict[str, str] = {
    "qwen2_moe": "qwen2",
    "qwen3_moe": "qwen3",
    "starcoder2": "llama",
    "granite": "llama",
    "yi": "llama",
    "deepseek": "llama",
    "deepseek_v2": "llama",
    "tinyllama": "llama",
    "smollm": "llama",
    "smollm2": "llama",
    "gemma3": "gemma2",
    "gpt_neo": "gpt2",
    "openai-gpt": "gpt2",
    "roberta": "bert",
    "distilbert": "bert",
    "electra": "bert",
    "albert": "bert",
    "vjepa2": "jepa",
    "v-jepa": "jepa",
    "ijepa": "jepa",
    "i-jepa": "jepa",
    "leworldmodel": "jepa",
    "lewm": "jepa",
}

# Substring keywords -> canonical model_type, scanned over an architecture's
# name + description when no explicit model_type is available. Ordered
# most-specific first so e.g. "mixtral" wins over "mistral" and "qwen2_moe"
# resolves before "qwen".
_KEYWORD_TO_MODEL_TYPE: list[tuple[str, str]] = [
    ("mixtral", "mixtral"),
    ("mistral", "mistral"),
    ("qwen3", "qwen3"),
    ("qwen2", "qwen2"),
    ("qwen", "qwen2"),
    ("gemma2", "gemma2"),
    ("gemma", "gemma"),
    ("phi3", "phi3"),
    ("phi", "phi"),
    ("tinyllama", "llama"),
    ("codellama", "llama"),
    ("llama", "llama"),
    ("cohere", "cohere"),
    ("command-r", "cohere"),
    ("vjepa", "jepa"),
    ("v-jepa", "jepa"),
    ("ijepa", "jepa"),
    ("i-jepa", "jepa"),
    ("leworldmodel", "jepa"),
    ("jepa", "jepa"),
    ("distilbert", "bert"),
    ("roberta", "bert"),
    ("electra", "bert"),
    ("albert", "bert"),
    ("bert", "bert"),
    ("gpt-neox", "gpt_neox"),
    ("gpt_neox", "gpt_neox"),
    ("gpt2", "gpt2"),
    ("gpt-2", "gpt2"),
    ("esm2", "esm"),
    ("esm", "esm"),
    ("t5", "t5"),
]


def family_of(model_type: str | None) -> str | None:
    """Canonical family for an HF ``model_type`` (None if unrecognized)."""
    if not model_type:
        return None
    mt = model_type.strip().lower()
    if mt in _SUPPORT:
        return mt
    return _MODEL_TYPE_ALIASES.get(mt)


def support_for(model_type: str | None) -> SupportInfo | None:
    """SupportInfo for an HF ``model_type`` (None if the family is unknown)."""
    fam = family_of(model_type)
    return _SUPPORT.get(fam) if fam else None


def detect_model_type(arch: Any) -> str | None:
    """Best-effort recover an HF ``model_type`` from an architecture.

    Priority: an explicit ``arch.metadata["model_type"]`` (stamped by HF
    adapters / a future ``## runtime`` section, read defensively so this works
    even when the AST has no such field today), then a substring scan over the
    architecture name + description.
    """
    metadata = getattr(arch, "metadata", None) or {}
    explicit = metadata.get("model_type") if isinstance(metadata, dict) else None
    if explicit:
        return str(explicit).lower()

    hay = f"{getattr(arch, 'name', '')} {getattr(arch, 'description', '') or ''}".lower()
    # Leading word-boundary match (not raw substring): matches "Llama27bHf" and
    # "Llama-family" but not "phi" inside "morphism" or "t5" inside "test5".
    for keyword, model_type in _KEYWORD_TO_MODEL_TYPE:
        if re.search(r"\b" + re.escape(keyword), hay):
            return model_type
    return None


# --------------------------------------------------------------------------- #
#  QLoRA VRAM estimation
# --------------------------------------------------------------------------- #

# Bytes/param for 4-bit nf4 weights (4 bits + small double-quant overhead).
_BYTES_PER_4BIT = 0.5
# Multiplier on a single layer's hidden activations to cover the several
# activation tensors retained per gradient-checkpoint segment (heuristic).
_ACT_CONST = 18
# Fixed runtime overhead: CUDA context, kernels, allocator fragmentation.
_RUNTIME_OVERHEAD = 1 * 1024 ** 3
_GIB = 1024 ** 3

# Hyperparameter name aliases, so the estimator reads the right widths from
# n-orca docs regardless of which adapter / builder produced them.
_HIDDEN_KEYS = ("d_model", "hidden_size", "embed_dim", "embedding_dim", "d", "dim")
_LAYER_KEYS = ("n_layer", "num_hidden_layers", "n_layers", "num_layers", "depth", "layers")
_FF_KEYS = ("d_ff", "intermediate_size", "ffn_dim", "mlp_dim", "ff_dim")
_SEQ_KEYS = ("max_pos", "max_position_embeddings", "max_seq_length", "n_positions", "seq_len")


def _pick(hparams: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for k in keys:
        v = hparams.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v == int(v):
            return int(v)
    return None


@dataclass(frozen=True)
class VramEstimate:
    """A documented QLoRA training-memory breakdown, in bytes."""

    total_bytes: int
    breakdown: dict[str, int]
    assumptions: dict[str, Any]
    note: str = (
        "Best-effort QLoRA estimate (treat as ±50%): 4-bit nf4 base weights + "
        "bf16 LoRA adapters + 8-bit-Adam optimizer states + bf16 grads + "
        "gradient-checkpointed activations + fixed CUDA overhead."
    )

    @property
    def total_gib(self) -> float:
        return round(self.total_bytes / _GIB, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "total_gib": self.total_gib,
            "breakdown_bytes": dict(self.breakdown),
            "breakdown_gib": {k: round(v / _GIB, 3) for k, v in self.breakdown.items()},
            "assumptions": dict(self.assumptions),
            "note": self.note,
        }


def estimate_lora_trainable_params(
    hyperparameters: dict[str, Any] | None,
    *,
    lora_r: int = 16,
    n_heads: int | None = None,
) -> int:
    """Estimate LoRA trainable params for a decoder-transformer architecture.

    Counts the default decoder-LLM targets per block — attention q/k/v/o
    (each ``Linear(d, d)``) and the gated MLP gate/up/down (``Linear(d, d_ff)``
    twice + ``Linear(d_ff, d)``) — where each adapted ``Linear(in, out)`` adds
    ``r * (in + out)`` trainable params. Returns 0 if the widths can't be read.
    """
    hp = hyperparameters or {}
    d = _pick(hp, _HIDDEN_KEYS)
    n_layer = _pick(hp, _LAYER_KEYS)
    if d is None or n_layer is None:
        return 0
    d_ff = _pick(hp, _FF_KEYS) or 4 * d
    attn = 4 * (lora_r * (d + d))          # q, k, v, o : each Linear(d, d)
    mlp = 2 * (lora_r * (d + d_ff)) + (lora_r * (d_ff + d))  # gate, up, down
    return n_layer * (attn + mlp)


def estimate_qlora_vram(
    *,
    param_count: int,
    hyperparameters: dict[str, Any] | None = None,
    lora_r: int = 16,
    batch_size: int = 1,
    max_seq_length: int | None = None,
    trainable_params: int | None = None,
) -> VramEstimate:
    """Estimate peak GPU memory for a 4-bit QLoRA fine-tune.

    The dominant, most-reliable term is the 4-bit base-weight footprint
    (``param_count * 0.5`` bytes); the rest are smaller and heuristic.
    """
    hp = hyperparameters or {}
    hidden = _pick(hp, _HIDDEN_KEYS)
    n_layer = _pick(hp, _LAYER_KEYS)
    n_heads = _pick(hp, ("n_heads", "num_attention_heads", "num_heads"))
    seq = max_seq_length or _pick(hp, _SEQ_KEYS) or 2048

    if trainable_params is None:
        trainable_params = estimate_lora_trainable_params(hp, lora_r=lora_r)

    base_4bit = int(param_count * _BYTES_PER_4BIT)
    lora_adapters = trainable_params * 2          # bf16 adapter weights
    gradients = trainable_params * 2              # bf16 grads (LoRA params only)
    optimizer_states = trainable_params * 2       # 8-bit Adam: 2 states * 1 byte

    if hidden is not None:
        act_linear = batch_size * seq * hidden * 2 * _ACT_CONST
        act_attn = batch_size * (n_heads or 1) * seq * seq * 2 if n_heads else 0
        activations = act_linear + act_attn
    else:
        activations = 2 * _GIB                     # fallback when width unknown

    breakdown = {
        "base_4bit": base_4bit,
        "lora_adapters": lora_adapters,
        "gradients": gradients,
        "optimizer_states": optimizer_states,
        "activations": int(activations),
        "runtime_overhead": _RUNTIME_OVERHEAD,
    }
    total = sum(breakdown.values())
    return VramEstimate(
        total_bytes=total,
        breakdown=breakdown,
        assumptions={
            "quantization": "4bit-nf4-double-quant",
            "optimizer": "adamw_8bit",
            "gradient_checkpointing": True,
            "lora_r": lora_r,
            "batch_size": batch_size,
            "max_seq_length": seq,
            "trainable_params": trainable_params,
            "param_count": param_count,
        },
    )


# --------------------------------------------------------------------------- #
#  Unified capability result
# --------------------------------------------------------------------------- #


@dataclass
class RuntimeCapability:
    """What the Unsloth runtime can do with one architecture, + a VRAM estimate."""

    model_type: str | None
    family: str | None
    unsloth_supported: bool
    loader: str
    patched_classes: tuple[str, ...]
    default_target_modules: tuple[str, ...]
    vram_estimate: VramEstimate | None
    note: str
    fits_in_gpu: bool | None = None        # set when a gpu_memory budget is given
    gpu_memory_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_type": self.model_type,
            "family": self.family,
            "unsloth_supported": self.unsloth_supported,
            "loader": self.loader,
            "patched_classes": list(self.patched_classes),
            "default_target_modules": list(self.default_target_modules),
            "note": self.note,
        }
        if self.vram_estimate is not None:
            out["vram_estimate"] = self.vram_estimate.to_dict()
        if self.gpu_memory_gb is not None:
            out["gpu_memory_gb"] = self.gpu_memory_gb
            out["fits_in_gpu"] = self.fits_in_gpu
        return out


def analyze(
    *,
    arch: Any = None,
    model_type: str | None = None,
    param_count: int = 0,
    hyperparameters: dict[str, Any] | None = None,
    lora_r: int = 16,
    batch_size: int = 1,
    max_seq_length: int | None = None,
    gpu_memory_gb: float | None = None,
) -> RuntimeCapability:
    """Analyze an architecture's Unsloth runtime capability + QLoRA VRAM.

    Either pass an `arch` (the model_type and hyperparameters are recovered from
    it) or pass `model_type` / `hyperparameters` / `param_count` explicitly. The
    MCP tool supplies overrides (`lora_r`, `batch_size`, `gpu_memory_gb`); the
    verifier calls it with documented defaults.
    """
    if model_type is None and arch is not None:
        model_type = detect_model_type(arch)
    if hyperparameters is None and arch is not None:
        hyperparameters = {
            hp.name: hp.default for hp in getattr(arch, "hyperparameters", [])
        }

    info = support_for(model_type)
    if info is None:
        family = None
        supported = False
        loader = ""
        patched: tuple[str, ...] = ()
        targets: tuple[str, ...] = ()
        note = (
            "Custom / unrecognized architecture — no Unsloth fast path. "
            "Compile to the generic PyTorch nn.Module and train with a "
            "standard loop."
        )
    else:
        family = info.family
        supported = info.supported
        loader = info.loader
        patched = info.patched_classes
        targets = info.default_target_modules
        note = info.note

    vram = estimate_qlora_vram(
        param_count=param_count,
        hyperparameters=hyperparameters,
        lora_r=lora_r,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    cap = RuntimeCapability(
        model_type=model_type,
        family=family,
        unsloth_supported=supported,
        loader=loader,
        patched_classes=patched,
        default_target_modules=targets,
        vram_estimate=vram,
        note=note,
    )
    if gpu_memory_gb is not None:
        cap.gpu_memory_gb = gpu_memory_gb
        cap.fits_in_gpu = vram.total_bytes <= gpu_memory_gb * _GIB
    return cap
