# Verification — Error Catalog

Every code below is a stable identifier. Tools (LLM refiners, CI checks,
editor integrations) can pattern-match on the code.

Severity column: `E` = error (architecture is invalid); `W` = warning
(architecture is still considered valid unless `--strict` is set).

| Code | Stage | Severity | Meaning | Typical fix |
|------|-------|----------|---------|-------------|
| `NO_LAYERS` | 1 | E | architecture declares no `## layer` headings | add at least one layer |
| `NO_INPUT_LAYER` | 1 | E | no layer marked `[input]` | mark the entry layer `## layer <name> [input]` |
| `NO_OUTPUT_LAYER` | 1 | E | no layer marked `[output]` | mark the exit layer `## layer <name> [output]` |
| `DUPLICATE_LAYER` | 1 | E | a layer name appears twice | rename or merge the duplicate |
| `UNKNOWN_LAYER_REFERENCE` | 1 | E | a `## flow` source or target isn't declared | declare `## layer <name>` or fix the typo |
| `INPUT_NOT_IN_TENSORS` | 1 | W | input layer has no row in `## tensors` | add a row declaring shape and dtype |
| `OUTPUT_NOT_IN_TENSORS` | 1 | W | output layer has no row in `## tensors` | add a row declaring shape and dtype |
| `CYCLE_DETECTED` | 2 | E | the layer graph contains a cycle | redirect an edge to break the cycle |
| `UNREACHABLE_LAYER` | 2 | E | a layer has no path from any `[input]` | connect it via `## flow`, or remove it |
| `LAYER_NOT_REACHING_OUTPUT` | 2 | E | a layer has no path to any `[output]` | add an outgoing edge, or remove the layer |
| `INPUT_HAS_PREDECESSOR` | 2 | W | input layer has incoming edges | remove those edges or unmark `[input]` |
| `OUTPUT_HAS_SUCCESSOR` | 2 | W | output layer has outgoing edges | remove those edges or unmark `[output]` |
| `INPUT_SHAPE_UNDECLARED` | 3 | W | input layer's shape can't be inferred | add a row in `## tensors` |
| `MISSING_OP` | 3 | E | non-IO layer has no `op:` | add `- op: <Op>(args)` |
| `SHAPE_MISMATCH` | 3 | E | op's shape rule rejects its inputs | fix op args or upstream shape chain |
| `INPUT_ARITY_MISMATCH` | 3 | E | op expects N inputs, got M | add/remove `## flow` edges |
| `OUTPUT_ARITY_MISMATCH` | 3 | E | output layer has != 1 incoming edge | merge with `Add`/`Concat` first |
| `OUTPUT_SHAPE_MISMATCH` | 3 | E | inferred output shape differs from `## tensors` | fix the chain or the tensor row |
| `DECLARED_SHAPE_MISMATCH` | 3 | E | layer's explicit `shape:` differs from inferred | remove the bullet or fix the chain |
| `PARAM_BUDGET_EXCEEDED` | 4 | E | total params exceeds `param_count <=` | shrink widths/depths, or relax invariant |
| `DEPTH_BUDGET_EXCEEDED` | 4 | E | longest path exceeds `depth <=` | shorten chain, or relax invariant |
| `OUTPUT_SHAPE_INVARIANT` | 4 | E | inferred output shape differs from `## invariants` | adjust architecture or invariant |
| `FLOPS_NOT_IMPLEMENTED` | 4 | W | `flops` invariants are accepted but not enforced | remove the invariant, or use an external profiler |
| `VRAM_BUDGET_EXCEEDED` | 4 | E | estimated QLoRA VRAM exceeds `vram_estimate <=` | raise the budget, lower `max_seq_length`/batch/LoRA rank, or pick a smaller base model |
| `UNKNOWN_OP` | 5 | W | layer uses an op not in the standard library | pick a standard op, or accept the placeholder and supply a custom module in the host |

## Stage 6 — Runtime (informational)

Stage 6 does not produce error codes. It records a `runtime` block on every
report — which HF model family the architecture maps to, whether the
[Unsloth](https://github.com/unslothai/unsloth) runtime backend can load and
fine-tune it (and via which loader / patched classes), the default LoRA target
modules, and a **best-effort** 4-bit QLoRA GPU-memory estimate. Backend
coverage is *reported, not required*: a custom architecture with no Unsloth
fast path is still a valid design.

The one runtime check that can fail a document is the `vram_estimate` invariant
(Stage 4, `VRAM_BUDGET_EXCEEDED` above) — declare a budget with
`- vram_estimate <= 24G` and the estimator is checked against it. The estimate
is pure static analysis (no torch / transformers / unsloth import, no GPU);
treat it as ±50%, useful as a budget gate rather than a guarantee.

```json
"runtime": {
  "model_type": "llama", "family": "llama", "unsloth_supported": true,
  "loader": "FastLanguageModel",
  "patched_classes": ["LlamaAttention", "LlamaDecoderLayer", "LlamaModel"],
  "default_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "vram_estimate": {"total_gib": 5.25, "breakdown_gib": {"base_4bit": 2.47, ...}, "assumptions": {...}}
}
```

## How the LLM should consume errors

Each error carries:

```json
{
  "code": "SHAPE_MISMATCH",
  "message": "layer 'add': Add inputs have mismatched shapes: [...]",
  "suggestion": "check the layer's op args and incoming shapes",
  "severity": "error"
}
```

The `code` is the stable handle; the `message` describes the specific
instance; the `suggestion` is written in the imperative voice so an LLM
can act on it directly when refining the architecture.

`n-orca verify --json` produces a JSON array of these reports, one per
architecture in the file.
