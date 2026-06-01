"""Unsloth runtime-capability analysis + QLoRA VRAM estimation.

Pure-Python, **no torch / transformers / peft / unsloth imports** and no GPU
access. Answers two questions for the AI design loop, from the static
architecture alone:

1. **Backend coverage** — does the Unsloth runtime fast-path this architecture?
   This is a *three-state* answer — ``supported`` / ``unsupported`` / ``unknown``
   — not a guess. The ``supported`` set is the verified ``model_type`` list from
   Unsloth's architecture-support docs; ``unsupported`` is reserved for models
   that are *structurally* outside Unsloth's decoder-LLM / VLM scope (encoder-
   only, encoder-decoder, joint-embedding world models — true regardless of
   Unsloth version); everything else is ``unknown`` rather than asserted either
   way. See ``UNSLOTH_SUPPORT_VERIFIED``.

2. **QLoRA memory footprint** — for a transformer decoder LLM, roughly how much
   GPU memory would a 4-bit QLoRA fine-tune take? Reported as a **range**
   (low / central / high), calibrated against published Unsloth benchmarks (see
   ``_CALIBRATION``), because the inputs are approximate (n-orca's param counts
   approximate SwiGLU/GQA; the recipe — optimizer, quant, target modules, seq —
   varies). The estimate is **only produced for decoder LLMs above a size floor**
   — applying it to an SAE / CNN / encoder would be meaningless, so those report
   ``vram_estimate = None``.

Design-time vs runtime: n-orca declares/verifies; Unsloth loads/trains/serves.
This module is the seam — joined at the same ``model_type`` key both dispatch on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Snapshot the support table was checked against. Unsloth adds models almost
#: every release, so treat this as a point-in-time mirror, not a live oracle.
UNSLOTH_SUPPORT_VERIFIED = (
    "Verified mid-2025 against docs.unsloth.ai (model catalog / architecture "
    "support) + deepwiki.com/unslothai/unsloth. Unsloth ships new models "
    "frequently — an `unknown` result means 'not in this snapshot', not "
    "'cannot work'."
)

_GIB = 1024 ** 3

# --------------------------------------------------------------------------- #
#  Unsloth support table — only verified entries; everything else is `unknown`
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SupportInfo:
    """What the Unsloth runtime does with a given model family.

    `status` is one of "supported" / "unsupported" / "unknown". `fast_class` and
    `patched_classes` carry only names that are actually stated in Unsloth's
    source / docs (we do not invent per-family class names).
    """

    family: str
    status: str                              # "supported" | "unsupported" | "unknown"
    is_decoder_lm: bool                       # gates whether a QLoRA VRAM estimate applies
    loader: str = ""                          # public entry: FastLanguageModel | FastVisionModel
    fast_class: str = ""                      # internal Fast* class (verified only)
    patched_classes: tuple[str, ...] = ()     # transformers classes Unsloth replaces (verified only)
    default_target_modules: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "status": self.status,
            "is_decoder_lm": self.is_decoder_lm,
            "loader": self.loader,
            "fast_class": self.fast_class,
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


def _supported_llm(family: str, fast_class: str = "FastLlamaModel", note: str = "") -> SupportInfo:
    return SupportInfo(
        family=family, status="supported", is_decoder_lm=True,
        loader="FastLanguageModel", fast_class=fast_class,
        # Only Llama's patched classes are stated in the source; for the rest we
        # name the verified Fast* loader class and leave patched_classes empty.
        patched_classes=(("LlamaAttention", "LlamaDecoderLayer", "LlamaModel")
                         if fast_class == "FastLlamaModel" and family == "llama" else ()),
        default_target_modules=_LLM_TARGETS,
        note=note or "Decoder LLM in Unsloth's fast path (Llama-lineage kernels).",
    )


# Verified supported model_types (Unsloth architecture-support docs + DeepWiki).
_SUPPORT: dict[str, SupportInfo] = {
    "llama": _supported_llm(
        "llama", "FastLlamaModel",
        "Decoder LLM — fused RoPE/RMSNorm/cross-entropy kernels + QLoRA."),
    "mistral": _supported_llm(
        "mistral", "FastLlamaModel", "Routed through FastLlamaModel."),
    "qwen2": _supported_llm("qwen2", "FastLlamaModel"),
    "qwen3": _supported_llm("qwen3", "FastLlamaModel"),
    "qwen3_moe": SupportInfo(
        "qwen3_moe", "supported", True, "FastLanguageModel", "FastQwen3MoeModel",
        default_target_modules=_LLM_TARGETS,
        note="Sparse-MoE decoder LLM (also routes DeepSeek-MoE)."),
    "gemma": SupportInfo(
        "gemma", "supported", True, "FastLanguageModel", "FastGemmaModel",
        default_target_modules=_LLM_TARGETS, note="Gemma decoder LLM."),
    "gemma2": SupportInfo(
        "gemma2", "supported", True, "FastLanguageModel", "FastGemma2Model",
        default_target_modules=_LLM_TARGETS, note="Gemma2 decoder LLM."),
    "cohere": _supported_llm("cohere", "FastLlamaModel", "Command-R lineage decoder LLM."),
    "granite": _supported_llm("granite", "FastLlamaModel", "Granite decoder LLM."),
    "falcon_h1": _supported_llm("falcon_h1", "FastLlamaModel", "Falcon-H1 decoder LLM."),
    # Vision-language: verified via FastVisionModel; specific patched classes not
    # asserted here.
    "qwen2_vl": SupportInfo(
        "qwen2_vl", "supported", True, "FastVisionModel", "FastVisionModel",
        default_target_modules=_LLM_TARGETS, note="VLM — load via FastVisionModel."),
    "mllama": SupportInfo(
        "mllama", "supported", True, "FastVisionModel", "FastVisionModel",
        default_target_modules=_LLM_TARGETS, note="Llama-3.2 vision — FastVisionModel."),
    "llava": SupportInfo(
        "llava", "supported", True, "FastVisionModel", "FastVisionModel",
        default_target_modules=_LLM_TARGETS, note="LLaVA VLM — FastVisionModel."),
    "paligemma": SupportInfo(
        "paligemma", "supported", True, "FastVisionModel", "FastVisionModel",
        default_target_modules=_LLM_TARGETS, note="PaliGemma VLM — FastVisionModel."),
    # Structurally outside Unsloth's scope — `unsupported` regardless of version,
    # because these are not causal decoder LLMs / VLMs.
    "bert": SupportInfo(
        "bert", "unsupported", False,
        note="Encoder-only (BERT/RoBERTa/ELECTRA/ALBERT) — not a causal decoder "
             "LLM; outside Unsloth's scope. Fine-tune via the generic "
             "transformers path."),
    "t5": SupportInfo(
        "t5", "unsupported", False,
        note="Encoder-decoder (T5) — outside Unsloth's decoder-LLM scope."),
    "esm": SupportInfo(
        "esm", "unsupported", False,
        note="ESM protein LM (encoder-only) — outside Unsloth's scope."),
    "jepa": SupportInfo(
        "jepa", "unsupported", False,
        note="JEPA / V-JEPA / I-JEPA / LeWorldModel are joint-embedding world "
             "models, not decoder LLMs — outside Unsloth's scope. Train with the "
             "generic PyTorch path."),
    # Causal decoder LMs whose Unsloth fast-path status is NOT in this snapshot —
    # reported as `unknown` (a QLoRA estimate still applies; they are decoder LMs).
    "gpt2": SupportInfo(
        "gpt2", "unknown", True, note="Decoder LM; not in the verified Unsloth "
        "support snapshot — status unknown."),
    "gpt_neox": SupportInfo(
        "gpt_neox", "unknown", True, note="Decoder LM; status unknown in snapshot."),
    "phi": SupportInfo(
        "phi", "unknown", True, note="Decoder LM (Phi); Unsloth advertises Phi "
        "support but it is not in this verified model_type snapshot — unknown."),
    "phi3": SupportInfo(
        "phi3", "unknown", True, note="Decoder LM (Phi-3); status unknown in snapshot."),
}

# Aliases: HF model_types that fold onto a table entry. Conservative — only
# well-established equivalences; speculative ones are intentionally omitted so
# they resolve to `unknown` rather than a wrong `supported`.
_MODEL_TYPE_ALIASES: dict[str, str] = {
    "qwen2_moe": "qwen3_moe",
    "gemma3": "gemma2",
    "gemma3_text": "gemma2",
    "tinyllama": "llama",
    "codellama": "llama",
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
    "openai-gpt": "gpt2",
    "gpt_neo": "gpt_neox",
}

# Substring keywords -> canonical model_type, scanned over an architecture's
# name + description when no explicit model_type is recorded. Ordered
# most-specific first. Used only as a *fallback*: `convert()` stamps the real
# model_type into `arch.metadata`, which detection prefers.
_KEYWORD_TO_MODEL_TYPE: list[tuple[str, str]] = [
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
    ("granite", "granite"),
    ("falcon", "falcon_h1"),
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
    """SupportInfo for an HF ``model_type`` (None if unrecognized — i.e. unknown)."""
    fam = family_of(model_type)
    return _SUPPORT.get(fam) if fam else None


def detect_model_type(arch: Any) -> str | None:
    """Best-effort recover an HF ``model_type`` from an architecture.

    Priority: an explicit ``arch.metadata["model_type"]`` (stamped by
    ``convert()`` / a ``## runtime`` section — the authoritative path), then a
    substring scan over the architecture name + description (fallback for
    hand-written docs, with a leading word-boundary so "phi" doesn't match
    "morphism" nor "t5" match "test5").
    """
    metadata = getattr(arch, "metadata", None) or {}
    explicit = metadata.get("model_type") if isinstance(metadata, dict) else None
    if explicit:
        return str(explicit).lower()

    hay = f"{getattr(arch, 'name', '')} {getattr(arch, 'description', '') or ''}".lower()
    for keyword, model_type in _KEYWORD_TO_MODEL_TYPE:
        if re.search(r"\b" + re.escape(keyword), hay):
            return model_type
    return None


def is_decoder_lm(model_type: str | None) -> bool:
    """Whether a QLoRA VRAM estimate is structurally meaningful for this type.

    Unknown / custom architectures return False — we do not assume an unrecognized
    design is a decoder transformer (its activation/LoRA shape is unknown).
    """
    info = support_for(model_type)
    return bool(info and info.is_decoder_lm)


# --------------------------------------------------------------------------- #
#  QLoRA VRAM estimation (calibrated against published Unsloth benchmarks)
# --------------------------------------------------------------------------- #

# Bytes/param for 4-bit nf4 weights (4 bits + small double-quant overhead).
_BYTES_PER_4BIT = 0.5
# Activation multiplier on `batch * seq * hidden * 2` bytes. Deliberately
# *independent of layer count*: Unsloth's gradient checkpointing offloads
# activations to system RAM, so on-GPU activation memory is roughly depth-
# independent — modeling it as `n_layers * ...` (or materializing seq*seq
# attention scores) badly overestimates the flash-attention + offload path.
_ACT_PER_TOKEN = 40
# Fixed runtime overhead: CUDA context, kernels, allocator fragmentation.
_RUNTIME_OVERHEAD = int(1.5 * _GIB)
# LoRA side, bytes per trainable param: bf16 adapter (2) + bf16 grad (2) +
# 8-bit-Adam two states (2).
_LORA_BYTES = 6
# Below this, a QLoRA fine-tune estimate is dominated by fixed overhead and not
# meaningful — we decline to emit a number.
MIN_PARAMS_FOR_ESTIMATE = 100_000_000
# Uncertainty band around the central estimate. Asymmetric: real usage skews
# higher (fragmentation, paged-optimizer spikes) and n-orca's param counts skew
# low (SwiGLU/GQA approximations), so the high tail is wider.
_RANGE_LOW = 0.75
_RANGE_HIGH = 1.40

# Calibration anchors (published Unsloth figures) the constants above reproduce:
_CALIBRATION = (
    "Llama-3.1-8B QLoRA (batch=2, r=32, seq=2048): model -> ~7.4 GiB vs Unsloth "
    "'<10 GB'. Llama-3.1-70B QLoRA (batch=2, r=32, seq~7168): model -> ~49 GiB "
    "vs Unsloth 'fits on a 48 GB GPU'. Sources: unsloth.ai/blog/llama3-1, "
    "unsloth.ai/blog/llama3-3."
)

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
    """A calibrated, best-effort QLoRA training-memory estimate, in bytes.

    The headline is a **range** (`low_bytes` .. `high_bytes`) around `central_bytes`;
    `breakdown` itemizes the central estimate.
    """

    central_bytes: int
    low_bytes: int
    high_bytes: int
    breakdown: dict[str, int]
    assumptions: dict[str, Any]
    note: str = (
        "Best-effort calibrated QLoRA estimate — a RANGE, not a guarantee. "
        "4-bit nf4 base weights + bf16 LoRA adapters + 8-bit-Adam states + bf16 "
        "grads + offload-checkpointed activations + fixed CUDA overhead. Base "
        "uses n-orca's param_count, which approximates SwiGLU/GQA, so the low "
        "tail can undercount; the recipe (optimizer/quant/targets/seq) also moves "
        "the number. Use as a planning gate, then measure."
    )

    @staticmethod
    def _gib(b: int) -> float:
        return round(b / _GIB, 2)

    @property
    def central_gib(self) -> float:
        return self._gib(self.central_bytes)

    @property
    def low_gib(self) -> float:
        return self._gib(self.low_bytes)

    @property
    def high_gib(self) -> float:
        return self._gib(self.high_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "central_bytes": self.central_bytes,
            "low_bytes": self.low_bytes,
            "high_bytes": self.high_bytes,
            "central_gib": self.central_gib,
            "low_gib": self.low_gib,
            "high_gib": self.high_gib,
            "breakdown_bytes": dict(self.breakdown),
            "breakdown_gib": {k: round(v / _GIB, 3) for k, v in self.breakdown.items()},
            "assumptions": dict(self.assumptions),
            "calibration": _CALIBRATION,
            "note": self.note,
        }


def estimate_lora_trainable_params(
    hyperparameters: dict[str, Any] | None,
    *,
    lora_r: int = 16,
) -> int:
    """Estimate LoRA trainable params for a decoder-transformer architecture.

    Counts the default decoder-LLM targets per block — attention q/k/v/o (each
    ``Linear(d, d)``) and the gated MLP gate/up/down — where each adapted
    ``Linear(in, out)`` adds ``r * (in + out)`` params. Returns 0 if the widths
    can't be read. (Note: ignores GQA, so K/V are slightly overcounted; the
    LoRA side is a small fraction of total VRAM, so the effect is minor.)
    """
    hp = hyperparameters or {}
    d = _pick(hp, _HIDDEN_KEYS)
    n_layer = _pick(hp, _LAYER_KEYS)
    if d is None or n_layer is None:
        return 0
    d_ff = _pick(hp, _FF_KEYS) or 4 * d
    attn = 4 * (lora_r * (d + d))                              # q, k, v, o
    mlp = 2 * (lora_r * (d + d_ff)) + (lora_r * (d_ff + d))    # gate, up, down
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
    """Estimate peak GPU memory for a 4-bit QLoRA fine-tune, as a range.

    Pure function — applicability gating (decoder-LM, size floor) lives in
    `analyze()` and the verifier; this just computes the arithmetic.
    """
    hp = hyperparameters or {}
    hidden = _pick(hp, _HIDDEN_KEYS)
    seq = max_seq_length or _pick(hp, _SEQ_KEYS) or 2048

    if trainable_params is None:
        trainable_params = estimate_lora_trainable_params(hp, lora_r=lora_r)

    base_4bit = int(param_count * _BYTES_PER_4BIT)
    lora_adapters = trainable_params * 2
    gradients = trainable_params * 2
    optimizer_states = trainable_params * 2          # 8-bit Adam, 2 states
    if hidden is not None:
        activations = batch_size * seq * hidden * 2 * _ACT_PER_TOKEN
    else:
        activations = 2 * _GIB                         # fallback when width unknown

    breakdown = {
        "base_4bit": base_4bit,
        "lora_adapters": lora_adapters,
        "gradients": gradients,
        "optimizer_states": optimizer_states,
        "activations": int(activations),
        "runtime_overhead": _RUNTIME_OVERHEAD,
    }
    central = sum(breakdown.values())
    return VramEstimate(
        central_bytes=central,
        low_bytes=int(central * _RANGE_LOW),
        high_bytes=int(central * _RANGE_HIGH),
        breakdown=breakdown,
        assumptions={
            "quantization": "4bit-nf4-double-quant",
            "optimizer": "adamw_8bit",
            "gradient_checkpointing": "unsloth (offloaded to RAM)",
            "lora_r": lora_r,
            "lora_target_modules": list(_LLM_TARGETS),
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
    """What the Unsloth runtime does with one architecture, + a VRAM range."""

    model_type: str | None
    family: str | None
    unsloth_status: str                       # "supported" | "unsupported" | "unknown"
    is_decoder_lm: bool
    loader: str
    fast_class: str
    patched_classes: tuple[str, ...]
    default_target_modules: tuple[str, ...]
    vram_estimate: VramEstimate | None
    vram_note: str                            # why an estimate is / isn't present
    note: str
    fits_in_gpu: bool | None = None           # central <= budget; set when gpu given
    fits_confidently: bool | None = None      # high <= budget (pessimistic case fits)
    gpu_memory_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_type": self.model_type,
            "family": self.family,
            "unsloth_status": self.unsloth_status,
            "is_decoder_lm": self.is_decoder_lm,
            "loader": self.loader,
            "fast_class": self.fast_class,
            "patched_classes": list(self.patched_classes),
            "default_target_modules": list(self.default_target_modules),
            "note": self.note,
            "vram_note": self.vram_note,
            "support_verified": UNSLOTH_SUPPORT_VERIFIED,
        }
        out["vram_estimate"] = self.vram_estimate.to_dict() if self.vram_estimate else None
        if self.gpu_memory_gb is not None:
            out["gpu_memory_gb"] = self.gpu_memory_gb
            out["fits_in_gpu"] = self.fits_in_gpu
            out["fits_confidently"] = self.fits_confidently
        return out


def vram_applicable(model_type: str | None, param_count: int) -> tuple[bool, str]:
    """Whether a QLoRA VRAM estimate is meaningful, and why / why not."""
    if not is_decoder_lm(model_type):
        return False, (
            "QLoRA VRAM estimate applies to transformer decoder LLMs; this "
            "architecture is not a recognized decoder LLM (custom / encoder / "
            "world-model), so no estimate is produced."
        )
    if param_count < MIN_PARAMS_FOR_ESTIMATE:
        return False, (
            f"model is below the {MIN_PARAMS_FOR_ESTIMATE // 1_000_000}M-param "
            "floor — a QLoRA estimate would be dominated by fixed overhead and "
            "isn't meaningful."
        )
    return True, "QLoRA VRAM estimate produced (decoder LLM above the size floor)."


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

    Either pass an `arch` (model_type + hyperparameters are recovered from it) or
    pass them explicitly. A VRAM estimate is attached only when it is meaningful
    (decoder LLM above the size floor); otherwise `vram_estimate` is None and
    `vram_note` explains why.
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
        status = "unknown"
        decoder = False
        loader = fast_class = ""
        patched: tuple[str, ...] = ()
        targets: tuple[str, ...] = ()
        note = (
            "Unrecognized model_type — not in the verified Unsloth support "
            "snapshot. If this is a custom design, compile to the generic "
            "PyTorch nn.Module and train with a standard loop."
        )
    else:
        family = info.family
        status = info.status
        decoder = info.is_decoder_lm
        loader = info.loader
        fast_class = info.fast_class
        patched = info.patched_classes
        targets = info.default_target_modules
        note = info.note

    applicable, vram_note = vram_applicable(model_type, param_count)
    vram = (
        estimate_qlora_vram(
            param_count=param_count, hyperparameters=hyperparameters,
            lora_r=lora_r, batch_size=batch_size, max_seq_length=max_seq_length,
        )
        if applicable else None
    )

    cap = RuntimeCapability(
        model_type=model_type, family=family, unsloth_status=status,
        is_decoder_lm=decoder, loader=loader, fast_class=fast_class,
        patched_classes=patched, default_target_modules=targets,
        vram_estimate=vram, vram_note=vram_note, note=note,
    )
    if gpu_memory_gb is not None and vram is not None:
        budget = gpu_memory_gb * _GIB
        cap.gpu_memory_gb = gpu_memory_gb
        cap.fits_in_gpu = vram.central_bytes <= budget
        cap.fits_confidently = vram.high_bytes <= budget
    elif gpu_memory_gb is not None:
        cap.gpu_memory_gb = gpu_memory_gb       # no estimate -> fit flags stay None
    return cap
