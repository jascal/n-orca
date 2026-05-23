# N-Orca — Neural-network Orchestrated Architecture Language

[![PyPI](https://img.shields.io/pypi/v/n-orca.svg)](https://pypi.org/project/n-orca/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/n-orca/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jascal/n-orca/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen)](https://github.com/jascal/n-orca/tree/main/tests)

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

# Convert an HF model -> .n.orca.md (+ optional Mermaid)
n-orca hf convert gpt2 --out gpt2.n.orca.md --mermaid gpt2.mmd
n-orca hf convert meta-llama/Llama-2-7b-hf --out llama-7b.n.orca.md
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
`tinyllama-2L.n.orca.md` — each with a matching `.mmd` diagram.

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
tests/test_cli.py            7 passed
tests/test_hf_adapters.py   13 passed
tests/test_hf_cli.py         4 passed
tests/test_hf_client.py      8 passed
tests/test_mermaid.py        3 passed
tests/test_ops.py           17 passed
tests/test_parser.py        11 passed
tests/test_pytorch.py        9 passed
tests/test_render.py         2 passed
tests/test_verifier.py      15 passed
=========================
88 passed in 3.73s
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
