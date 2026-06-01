"""Runtime backend analysis for n-orca architectures.

Increment A of the Unsloth integration: make the design loop *runtime-aware*
without importing the heavy training stack (torch / transformers / unsloth) or
touching a GPU. Everything here is pure-Python arithmetic over the static AST.

- `capability.analyze(...)`        — "can the Unsloth runtime load + fine-tune
                                      this, and roughly how much GPU memory
                                      would a QLoRA fine-tune take?"
- `capability.estimate_qlora_vram` — the standalone VRAM estimator the verifier's
                                      `vram_estimate` invariant is evaluated against.

The verifier's Stage 6 (runtime) and the `check_runtime_capability` MCP tool both
go through `analyze`.
"""
from n_orca.backends.capability import (
    RuntimeCapability,
    SupportInfo,
    VramEstimate,
    analyze,
    detect_model_type,
    estimate_lora_trainable_params,
    estimate_qlora_vram,
    family_of,
    support_for,
)

__all__ = [
    "RuntimeCapability",
    "SupportInfo",
    "VramEstimate",
    "analyze",
    "detect_model_type",
    "estimate_lora_trainable_params",
    "estimate_qlora_vram",
    "family_of",
    "support_for",
]
