# N-Orca — Neural-network Orchestrated Architecture Language

[![PyPI](https://img.shields.io/pypi/v/n-orca.svg)](https://pypi.org/project/n-orca/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/n-orca/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jascal/n-orca/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-143%20passing-brightgreen)](https://github.com/jascal/n-orca/tree/main/tests)

> **Current OpenSpec Changes** (active work tracked in `openspec/changes/`):
> - `add-temporal-world-model` — Add support for temporal/recurrent world models (GRU-style state carry or per-period attention) to better support regime/windowed feature recovery from econ-sae. See `openspec/changes/add-temporal-world-model/`.
> - `add-cosmos-mot-world-models` — Add MoT (Mixture of Transformers) dual-tower + joint-attn + diffusion timestep world model builders (Cosmos 3 style for Physical AI / multimodal / diffusion substrates). Enables richer SAE GT (physics, conjunctive cross-modal, diffusion factors) + potential new world-sae sibling. Subagent investigation complete; design/proposal/tasks scaffolded. See `openspec/changes/add-cosmos-mot-world-models/`.

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

See [`docs/verification.md`](docs/verification.md) for the full error
catalog with examples.

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
tests/test_mcp_server.py          6 passed  (incl mot with timestep_dim)
tests/test_ops.py                22 passed
tests/test_parser.py             11 passed
tests/test_pytorch.py             8 passed
tests/test_render.py              2 passed
tests/test_sae_and_world_models.py  25 passed  (incl temporal + mot_denoise)
tests/test_verifier.py           16 passed
=========================
143 passed in 6.83s (clean 3.11 + torch env; +1 for mot MCP timestep test)
```

---

## Development

```bash
# From repo root
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
python -m pytest -q
n-orca verify examples/transformer-block.n.orca.md
```

The repo declares Python >=3.10. CI uses 3.10–3.12 + CPU torch.

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

## SAE & World Model builders (for interpretability research)

n-orca ships ready-to-use builders for the exact architectures used as
**ground-truth substrates and SAE models** in econ-sae (and cross-sibling
sm-sae / bio-sae + polygram work):

World models (SAE training substrates, H1 activations are the target):
- `world_model` (baseline 2-layer MLP)
- `deep_world_model`
- `attn_world_model` (per-agent / per-position MultiHeadAttention + residual + LN before the MLP — the biggest single lift for conjunctive features in econ-sae)
- `temporal_world_model` (attn + explicit `hidden_in`/`hidden_out` state carry for cross-period regime features per econ-sae; see OpenSpec add-temporal-world-model)
- `mot_denoise_step` (MoT dual-tower AR-reasoner + DM-generator + timestep for Cosmos 3-style diffusion/multimodal world models; enables richer physics/conjunctive/diffusion-stage SAE GT; see OpenSpec add-cosmos-mot-world-models + subagent report)

SAE variants (all verify + compile to PyTorch/Mermaid):
- `topk_sae`, `l1_sae`, `jumprelu_sae` (classic families)
- `attn_topk_sae` (attention prefix for cross-position context; 3D `(B,T,d)`)
- `supervised_topk_sae` (auxiliary per-label classifier head off the sparse latents; multi-output)
- `gated_sae` (parallel magnitude + sigmoid gate, elementwise mul)

See:
- `n_orca/sae.py` and `n_orca/world_models.py` (builders + docstrings with econ-sae Phase references)
- `examples/sae-*.n.orca.md`, `examples/econ-*.n.orca.md`, `examples/cosmos-mot-*.n.orca.md` (all verified)
- MCP tools: `build_sae`, `build_world_model`, `verify_markdown`, `compile_*`
- n-orca skills: `/n-orca-build-sae`, `/n-orca-build-world-model`

These are the canonical, verified definitions. Use them from n-orca instead of duplicating topology in the SAE-fixture repos.

See `docs/proposed-sae-extensions.md` for the full motivation + implementation status (landed 2026-06-02, examples + skills added in this handoff).
