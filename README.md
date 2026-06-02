# N-Orca — Neural-network Orchestrated Architecture Language

[![PyPI](https://img.shields.io/pypi/v/n-orca.svg)](https://pypi.org/project/n-orca/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/n-orca/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jascal/n-orca/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-139%20passing-brightgreen)](https://github.com/jascal/n-orca/tree/main/tests)

N-Orca is a Markdown DSL for declaring, verifying, visualizing, and executing
neural network architectures. It is a domain-specific dialect of
[Orca](https://github.com/jascal/orca-lang) — Orca describes finite-state
machines, N-Orca describes typed DAGs of tensor-flow layers.

The point of the language is to give LLMs (and humans) **one artifact that
covers all five jobs**:

| Affordance | How |
|------------|-----|
| **Author**    | Plain Markdown — LLMs already produce GitHub-flavored tables and headings reliably. No new tokens to memorize. |
| **Read**      | `## section` headings give strong landmarks; one row = one layer; blockquotes describe intent next to declarations. |
| **Visualize** | Compile to Mermaid `flowchart TD` — any markdown viewer renders the diagram. |
| **Explain**   | The verifier emits structured error codes (e.g. `SHAPE_MISMATCH`, `UNREACHABLE_LAYER`) with `suggestion:` fields written for an LLM to act on. |
| **Execute**   | Compile to a runnable PyTorch `nn.Module`. The verifier guarantees the emitted code's shapes line up before you run it. |

---

## What it looks like

```markdown
# architecture TransformerBlock

## hyperparameters

| Name    | Type  | Default |
|---------|-------|---------|
| d_model | int   | 512     |
| n_heads | int   | 8       |
| d_ff    | int   | 2048    |
| dropout | float | 0.1     |

## tensors

| Name | Shape           | Dtype   |
|------|-----------------|---------|
| x    | (B, S, d_model) | float32 |
| y    | (B, S, d_model) | float32 |

## layer x [input]
## layer attn_norm
- op: LayerNorm(d_model)
## layer attn
- op: MultiHeadAttention(d_model, n_heads, dropout)
## layer add_1
- op: Add
## layer ff_norm
- op: LayerNorm(d_model)
## layer ff
- op: FeedForward(d_model, d_ff, dropout)
## layer add_2
- op: Add
## layer y [output]

## flow

| Source     | Target    | Tensor    |
|------------|-----------|-----------|
| x          | attn_norm | x         |
| attn_norm  | attn      | x_normed  |
| attn       | add_1     | attn_out  |
| x          | add_1     | x_skip    |
| add_1      | ff_norm   | h         |
| ff_norm    | ff        | h_normed  |
| ff         | add_2     | ff_out    |
| add_1      | add_2     | h_skip    |
| add_2      | y         | y_out     |

## invariants
- output_shape: (B, S, d_model)
```

That's a real, verifying example. `n-orca verify` confirms reachability,
shape consistency through every residual, parameter count, and the declared
output-shape invariant — before any code runs.

---

## Install

```bash
pip install n-orca                # core CLI, parser, verifier, compilers
pip install "n-orca[torch]"       # + PyTorch (compile to runnable nn.Module)
pip install "n-orca[hf]"          # + huggingface_hub (search, info, download, convert)
pip install "n-orca[mcp]"         # + MCP server (drive n-orca from Claude / any MCP client)
pip install "n-orca[all]"         # everything
```

Or from source:

```bash
git clone https://github.com/jascal/n-orca && cd n-orca
pip install -e ".[all]"
```

Python 3.10+. The core has zero runtime dependencies; `huggingface_hub`,
`torch`, and `mcp` are optional extras and lazy-imported at the point of use.

---

## Commands

```bash
# Verify a file
n-orca verify examples/transformer-block.n.orca.md
n-orca verify examples/transformer-block.n.orca.md --json
n-orca verify examples/transformer-block.n.orca.md --strict

# Compile
n-orca compile mermaid  examples/transformer-block.n.orca.md
n-orca compile pytorch  examples/transformer-block.n.orca.md
n-orca compile pytorch  examples/transformer-block.n.orca.md --out model.py

# Summarize
n-orca info examples/transformer-block.n.orca.md

# Runtime backend coverage + QLoRA VRAM estimate (no GPU / no torch needed)
n-orca runtime examples/hf-generated/llama-7b.n.orca.md --gpu 24
```

Sample verify output:

```
Architecture: TransformerBlock
  Result: VALID
  Parameters: 3,152,384
  Depth: 7
```

When a check fails, you get a stable error code, a message, and a suggestion:

```
Architecture: ResidualBlock
  Result: INVALID
  [ERR] SHAPE_MISMATCH: layer 'add': Add inputs have mismatched shapes:
        [('B', 'C', 'H', 'W'), ('B', '2C', 'H', 'W')]
        -> check the layer's op args and incoming shapes
```

---

## Verification pipeline

Five stages, run in order. A failure in an earlier stage stops later stages
on that architecture (their preconditions wouldn't hold).

| Stage | What it checks | Example error codes |
|-------|----------------|---------------------|
| 1 — Naming      | every flow-edge endpoint resolves; one `[input]` and `[output]`; no duplicate layer names | `UNKNOWN_LAYER_REFERENCE`, `NO_INPUT_LAYER`, `DUPLICATE_LAYER` |
| 2 — Structural  | DAG (no cycles); every layer reachable from an input; every layer reaches an output | `CYCLE_DETECTED`, `UNREACHABLE_LAYER`, `LAYER_NOT_REACHING_OUTPUT` |
| 3 — Shape       | each layer's input shapes match its op's input rule; declared `shape:` matches inferred | `SHAPE_MISMATCH`, `INPUT_ARITY_MISMATCH`, `DECLARED_SHAPE_MISMATCH` |
| 4 — Resource    | `param_count` / `depth` / `output_shape` against `## invariants` | `PARAM_BUDGET_EXCEEDED`, `DEPTH_BUDGET_EXCEEDED`, `OUTPUT_SHAPE_INVARIANT` |
| 5 — Op coverage | every layer's op exists in the standard library (warning if not) | `UNKNOWN_OP` |
| 6 — Runtime     | *informational, warnings only:* three-state Unsloth coverage + a calibrated QLoRA VRAM range; checks a `vram_estimate` invariant | `VRAM_BUDGET_EXCEEDED` (W), `VRAM_ESTIMATE_NOT_APPLICABLE` (W) |

See [`docs/verification.md`](docs/verification.md) for the full error
catalog with examples.

---

## Runtime backend awareness

N-Orca describes architectures; the [Unsloth](https://github.com/unslothai/unsloth)
runtime *loads, fine-tunes, and serves* a curated set of HuggingFace model
families. The two meet at the `model_type` key both dispatch on. Stage 6 of the
verifier makes a design **runtime-aware** — without importing torch /
transformers / unsloth or touching a GPU:

```bash
n-orca runtime examples/hf-generated/llama-7b.n.orca.md --gpu 24
```

```
Architecture: Llama27bHf
  Unsloth backend: supported (family=llama, loader=FastLanguageModel)
  Fast class:      FastLlamaModel
  Patched classes: LlamaAttention, LlamaDecoderLayer, LlamaModel
  LoRA targets:    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  Est. QLoRA VRAM: ~5.44 GiB  (range 4.08–7.62 GiB; r=16, batch=1, seq=4096)
  GPU budget 24.0 GB: FITS (even pessimistically)
```

It answers two questions from static config alone:

1. **Does the Unsloth backend fast-path this?** A *three-state* answer —
   `supported` (a verified `model_type`: the Llama lineage — Llama/Mistral/Qwen/
   Gemma/Cohere/Granite/Falcon-H1 — plus VLMs via `FastVisionModel`),
   `unsupported` (structurally outside Unsloth's scope — encoder-only,
   encoder-decoder, JEPA world models), or `unknown` (a recognized decoder LM
   not in the verified snapshot, or a custom design). It does not guess; the
   snapshot date is reported in `support_verified`.
2. **Roughly what does a 4-bit QLoRA fine-tune cost?** A **range**
   (low/central/high), produced only for decoder LLMs above a size floor, and
   calibrated against published Unsloth benchmarks (Llama-3.1 8B/70B). It models
   flash-attention + RAM-offloaded gradient checkpointing — treat it as a
   planning gate, then measure. Declarable as an invariant (checked as a
   **warning**, since it's a heuristic):

```markdown
## invariants
- vram_estimate <= 24G
```

The same analysis is available over MCP (`check_runtime_capability`, which also
accepts an HF `model_id`) and in the `runtime` block of every `verify --json`
report. `n-orca hf convert` stamps the model's `model_type` into a `## runtime`
section so the answer is authoritative (not a name guess) and round-trips.

This is a **static, informational pre-flight gate** — most useful inside an
automated design loop, to cheaply filter "could I even fine-tune this here?"
before reaching for a GPU. It deliberately does **not** load weights or train
anything: n-orca stays a design/verify tool, and the actual fine-tuning is left
to Unsloth (for supported LLMs) or plain PyTorch (for everything else).

---

## Standard op library

| Op | Maps to | Use |
|----|---------|-----|
| `Linear(in, out)` | `nn.Linear` | dense layer |
| `LayerNorm(d)`, `BatchNorm1d(c)`, `BatchNorm2d(c)` | matching `nn.*Norm` | normalization |
| `Conv2d(ic, oc, k, s, p)` | `nn.Conv2d` | 2D conv |
| `MaxPool2d(k, s)`, `AvgPool2d(k, s)`, `AdaptiveAvgPool2d(out)` | matching `nn.*Pool*` | spatial pooling |
| `ReLU`, `GELU`, `SiLU`, `Tanh`, `Sigmoid`, `Softmax(dim)` | matching `nn.*` | activations |
| `Dropout(p)` | `nn.Dropout` | regularization |
| `Embedding(n, d)` | `nn.Embedding` | token / position embeddings |
| `PatchEmbed(c, d, p)` | `nn.Conv2d` + flatten + transpose | image → patch-token sequence (ViT / I-JEPA) |
| `TubeletEmbed(c, d, t, p)` | `nn.Conv3d` + flatten + transpose | video clip → tubelet-token sequence (V-JEPA 2) |
| `MultiHeadAttention(d, h, dropout)` | `nn.MultiheadAttention` (batch-first) | self-attention |
| `FeedForward(d, d_ff, dropout)` | `nn.Sequential(Linear, GELU, Dropout, Linear, Dropout)` | transformer FFN |
| `Add`, `Mul` | functional `+` / `*` | residual / gated paths |
| `Concat(dim)` | `torch.cat` | skip connections |
| `Mean(dim)` | `tensor.mean(dim=…)` | sequence / global pooling |
| `Flatten(start_dim)`, `Reshape(shape)`, `Identity` | matching torch ops | shape glue |

Unknown ops are accepted with an `UNKNOWN_OP` warning and emit `nn.Identity()`
placeholders in PyTorch output — useful for prototyping; the verifier still
checks everything around them.

See [`docs/grammar.md`](docs/grammar.md) for the full language specification.

---

## Hugging Face Hub integration

```bash
# Search the Hub
n-orca hf search "gpt2" --limit 10
n-orca hf search "llama" --task text-generation

# Inspect a model (metadata + config.json)
n-orca hf info gpt2
n-orca hf info meta-llama/Llama-2-7b-hf --revision <commit-sha>

# Download just config.json, or the full snapshot
n-orca hf download gpt2 --config-only
n-orca hf download gpt2 --allow "*.safetensors" --local-dir ./gpt2

# Vision / video models: also grab the (video) preprocessor config
n-orca hf download facebook/vjepa2-vitl-fpc64-256 --config-only --include-processor

# Convert an HF model -> .n.orca.md (+ optional Mermaid)
n-orca hf convert gpt2 --out gpt2.n.orca.md --mermaid gpt2.mmd
n-orca hf convert meta-llama/Llama-2-7b-hf --out llama-7b.n.orca.md

# JEPA / V-JEPA 2 / LeWorldModel world models
n-orca hf convert facebook/vjepa2-vitl-fpc64-256 --out vjepa2.n.orca.md --mermaid vjepa2.mmd
n-orca hf convert quentinll/lewm-pusht --out lewm-pusht.n.orca.md
```

The convert command reads `config.json` only — **no model weights or remote
code are executed**. It picks an adapter by matching `config["model_type"]`
or `config["architectures"]` against a registry, builds the topology, and
writes a verified `.n.orca.md`.

### Supported model families

| Adapter | Matches | Notes |
|---------|---------|-------|
| `Gpt2Adapter` | `gpt2`, `gpt_neo`, `gpt_neox`, `openai-gpt` | Decoder-only, pre-LN, learned positional embeddings |
| `LlamaFamilyAdapter` | `llama`, `mistral`, `mixtral`, `qwen2`, `qwen3`, `qwen2_moe`, `gemma`, `gemma2`, `phi`, `phi3` | Decoder-only, RMSNorm + RoPE + SwiGLU (approximated in v1) |
| `BertAdapter` | `bert`, `roberta`, `distilbert`, `electra`, `albert` | Encoder-only, post-LN; segment-type embeddings omitted in v1 |
| `EsmAdapter` | `esm` | ESM-2 protein LM; pre-norm encoder, rotary positions left implicit |
| `JepaAdapter` | `vjepa2`, `ijepa`, `jepa`, `leworldmodel`, `lewm` (+ `VJEPA2Model` / `IJepaModel` / LeWM Hydra configs) | Joint-embedding **world models**: ViT encoder → predictor. Video tubelet (Conv3d) or image patch (Conv2d) embed; optional action conditioning + projector. Mask tokens, EMA target, SIGReg (latent regularizer) captured as verification rules |

`JepaAdapter` normalizes two unrelated config schemas into one encoder →
predictor DAG: the flat `transformers` config (V-JEPA 2 / I-JEPA, with `pred_*`
predictor fields) and the nested Hydra config used by LeWorldModel
(`quentinll/lewm-*`), which carries no `model_type` and is matched structurally.
Both latent outputs (`encoder_latents`, `predicted_latents`) live in the
encoder's embedding space, so the `output_shape` invariant enforces latent-dim
consistency between the representation and the forecast.

Adding a new family is a ~50-line file in `n_orca/hf/adapters/` — declare
`model_types` and implement `build(config, name=...)` to return an
`Architecture` AST.

### Library API

```python
from n_orca.hf import HfClient, convert

client = HfClient()
results = client.search("llama", task="text-generation", limit=5)
for r in results:
    print(r.id, r.downloads)

info = client.info("gpt2")
print(info.config["model_type"], info.config["n_layer"])

# Convert (reads config.json directly — no weights needed)
result = convert("gpt2")
result.write_markdown("gpt2.n.orca.md")
result.write_mermaid("gpt2.mmd")
print(f"params: {result.report.param_count:,}, valid: {result.report.valid}")
```

Pre-generated example outputs live in [`examples/hf-generated/`](examples/hf-generated/):
`gpt2-small.n.orca.md`, `bert-base-uncased.n.orca.md`, `llama-7b.n.orca.md`,
`tinyllama-2L.n.orca.md`, `vjepa2.n.orca.md` (V-JEPA 2 ViT-L, 326M params),
`lewm-pusht.n.orca.md` (action-conditioned LeWorldModel) — each with a matching
`.mmd` diagram.

---

## Examples

All examples in [`examples/`](examples/) verify clean *and* round-trip through
the PyTorch compiler with a real forward pass:

| Example | Architecture | Params (default) |
|---------|--------------|------------------|
| `simple-mlp.n.orca.md` | 2-layer MLP classifier | 203,530 |
| `residual-block.n.orca.md` | Pre-activation ResNet-v2 block | 74,112 |
| `transformer-block.n.orca.md` | GPT-style pre-norm encoder block | 3,152,384 |
| `tiny-vit.n.orca.md` | Minimal Vision Transformer | 205,834 |
| `conv-classifier.n.orca.md` | Small CNN classifier | 20,042 |
| `unet-stub.n.orca.md` | Two-level U-Net skeleton with skip-concat | 3,057 |

---

## Tests

```bash
pip install -e ".[test,torch]"
pytest -v
```

```
tests/test_cli.py                 7 passed
tests/test_hf_adapters.py        28 passed
tests/test_hf_cli.py              4 passed
tests/test_hf_client.py           8 passed
tests/test_mermaid.py             4 passed
tests/test_mcp_server.py          5 passed
tests/test_ops.py                22 passed
tests/test_parser.py             11 passed
tests/test_pytorch.py             8 passed
tests/test_render.py              2 passed
tests/test_sae_and_world_models.py  24 passed
tests/test_verifier.py           16 passed
=========================
139 passed in 13.29s
```

---

## Design philosophy

N-Orca borrows three load-bearing ideas from
[Orca](https://github.com/jascal/orca-lang) and
[Q-Orca](https://github.com/jascal/q-orca-lang):

1. **The spec is the artifact the LLM edits.** Not a Python file with a
   sidecar comment; not a JSON config alongside Python. One Markdown file is
   the source of truth.
2. **Tables, not new syntax.** LLMs hallucinate the least on flat,
   row-oriented tables. Layers are one section; flow edges are one section;
   each row is one fact.
3. **Verify before execute.** The pipeline checks topology (DAG-ness,
   reachability), then shapes (the place neural network code most often
   breaks at runtime), then resource bounds — before the PyTorch compiler
   emits anything. By the time you `forward()` the model, the architecture
   has already been checked.

The net effect: an LLM can write an architecture in Markdown, get structured
verifier feedback, refine, and ship runnable code — without ever editing
Python.
