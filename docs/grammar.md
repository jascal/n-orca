# N-Orca Grammar Specification

N-Orca is a Markdown-based language for declaring neural network architectures as verifiable DAGs of typed layers. It is a dialect of [Orca](https://github.com/jascal/orca-lang): layers play the role of states, and a flow table replaces transitions.

A document declares one or more `# architecture Name` blocks. Files conventionally end with `.n.orca.md`.

---

## Top-level document

```
# architecture <Name>
> optional one-line description
```

Multiple architectures may appear in one file, separated by `---`. Each is parsed independently.

---

## Sections

Sections are `##` headings. All are optional except `## flow` and at least one `## layer`. Order is not significant.

### `## hyperparameters`

Free, named, typed constants the architecture is parameterized by.

```markdown
## hyperparameters

| Name    | Type  | Default |
|---------|-------|---------|
| d_model | int   | 512     |
| n_heads | int   | 8       |
| dropout | float | 0.1     |
```

Supported types: `int`, `float`, `bool`, `string`.

Hyperparameter names are referenced symbolically in op arguments and shape literals (`d_model`, etc.).

### `## tensors`

Declares the named input/output tensors of the architecture, with shapes and dtypes.

```markdown
## tensors

| Name | Shape           | Dtype   |
|------|-----------------|---------|
| x    | (B, S, d_model) | float32 |
| y    | (B, S, d_model) | float32 |
```

**Shape grammar** — a parenthesized comma list of dimensions. Each dimension is one of:
- An integer literal (`3`, `224`, `768`)
- A hyperparameter name (`d_model`)
- An arithmetic expression of the above (`d_model * 4`)
- A symbolic dimension by convention: `B` (batch), `S` (sequence), `H` (height), `W` (width), `C` (channels). These are unbound free variables — the verifier treats them as opaque but consistent.

**Dtypes**: `float32`, `float16`, `bfloat16`, `int32`, `int64`, `bool`.

### `## layer <Name> [markers]`

Each layer is a `## layer` heading. Optional bracketed markers:
- `[input]` — receives an externally-supplied tensor (matches a row in `## tensors`)
- `[output]` — produces a tensor exposed to the outside (matches a row in `## tensors`)

Body uses an optional `> description` blockquote and bulleted attributes:

```markdown
## layer attn
> Multi-head self-attention
- op: MultiHeadAttention(d_model, n_heads, dropout)
- shape: (B, S, d_model)
```

Attributes:
- `op:` — op invocation. Op name plus optional positional args. Required for non-input/output layers.
- `shape:` — output shape. Optional; if omitted, the verifier infers from the op and incoming shapes.
- `params:` — explicit parameter dictionary (rarely needed; usually subsumed by `op:` args).

Input/output layers carry no `op:` — they are sources/sinks.

### `## flow`

The DAG edges. One row per directed edge.

```markdown
## flow

| Source     | Target | Tensor    |
|------------|--------|-----------|
| x          | attn_norm | x       |
| attn_norm  | attn   | x_normed  |
| attn       | add_1  | attn_out  |
| x          | add_1  | x_skip    |
```

- **Source / Target** — layer names declared in `## layer` headings.
- **Tensor** — symbolic name for the value flowing along this edge. Used by error messages and by the PyTorch compiler.

If a layer has multiple incoming edges (e.g. residual add), the edges are passed to its op in source-row order — first row first.

### `## invariants`

Declarative bounds. Verified at the resource-bound stage.

```markdown
## invariants
- param_count <= 50M
- flops <= 1G
- output_shape: (B, S, d_model)
- vram_estimate <= 24G
```

Supported predicates: `param_count`, `flops`, `depth`, `output_shape`,
`vram_estimate`. Suffixes: `K`, `M`, `G` (decimal) on numeric bounds.
`vram_estimate` is a calibrated best-effort 4-bit QLoRA GPU-memory estimate
(Stage 6); it is checked as a **warning** and only applies to decoder-LLM
architectures.

### `## verification rules`

Narrative rules, used by the LLM and reflected in stage-by-stage error reporting.

```markdown
## verification rules
- shape-consistency: all flow edges have matching source/target shapes
- residual-symmetry: every Add layer has at least two inputs of identical shape
- dag-acyclicity: the layer graph is acyclic
```

These mostly mirror the built-in checks; including them gives the LLM the vocabulary to think with.

### `## runtime`

Optional provenance / runtime hints as `- key: value` bullets. `n-orca hf
convert` stamps the source `model_type` here so the runtime-capability backend
(Stage 6) has an authoritative signal rather than guessing from the name. Does
not affect topology/shape verification; round-trips through `render`.

```markdown
## runtime
- model_type: llama
```

---

## Standard op library

Ops accepted by the parser and compiled by the PyTorch backend:

| Op | Signature | PyTorch | Shape rule |
|----|-----------|---------|------------|
| `Linear(in, out)` | `(*, in) -> (*, out)` | `nn.Linear` | replace last dim |
| `LayerNorm(dim)` | `(*, dim) -> (*, dim)` | `nn.LayerNorm` | preserve |
| `BatchNorm1d(c)` | `(B, c, *) -> same` | `nn.BatchNorm1d` | preserve |
| `BatchNorm2d(c)` | `(B, c, H, W) -> same` | `nn.BatchNorm2d` | preserve |
| `Conv2d(ic, oc, k, s, p)` | `(B, ic, H, W) -> (B, oc, H', W')` | `nn.Conv2d` | conv formula |
| `MaxPool2d(k, s)` | `(B, C, H, W) -> (B, C, H', W')` | `nn.MaxPool2d` | pool formula |
| `AvgPool2d(k, s)` | `same` | `nn.AvgPool2d` | pool formula |
| `AdaptiveAvgPool2d(out)` | `(B, C, H, W) -> (B, C, oh, ow)` | `nn.AdaptiveAvgPool2d` | fixed |
| `Dropout(p)` | preserve | `nn.Dropout` | preserve |
| `ReLU()` / `GELU()` / `SiLU()` / `Tanh()` / `Sigmoid()` | preserve | matching nn module | preserve |
| `Softmax(dim)` | preserve | `nn.Softmax` | preserve |
| `Embedding(num, dim)` | `(*) int -> (*, dim)` | `nn.Embedding` | append dim |
| `MultiHeadAttention(d, h, dropout)` | `(B, S, d) -> (B, S, d)` | `nn.MultiheadAttention` (batch_first) | preserve |
| `FeedForward(d, d_ff, dropout)` | `(*, d) -> (*, d)` | Linear-GELU-Dropout-Linear | preserve |
| `Add` | n inputs of shape T -> T | functional `+` | all-same |
| `Mul` | n inputs of shape T -> T | functional `*` | all-same |
| `Concat(dim)` | n inputs differing only in `dim` | `torch.cat` | sum-on-dim |
| `Flatten(start_dim=1)` | `(B, *) -> (B, prod)` | `nn.Flatten` | collapse |
| `Reshape(shape)` | reshape | `torch.reshape` | declared |
| `Identity` | preserve | `nn.Identity` | preserve |

Unknown ops are passed through as `nn.Module` placeholders with a `MISSING_OP` warning at verify time.

---

## Verification pipeline

Stages run in order; failure in an earlier stage skips later ones for that machine.

| Stage | Module | Checks |
|-------|--------|--------|
| 1 — Structural | `structural.py` | every layer reachable from inputs; every layer reaches an output; no cycles |
| 2 — Naming | `naming.py` | every flow Source/Target exists; every input/output layer matches a `## tensors` row |
| 3 — Shape | `shape.py` | input shapes to each layer match its op's input rule; declared `shape:` matches inferred |
| 4 — Resource bounds | `resources.py` | `param_count` / `flops` / `depth` against `## invariants` |
| 5 — Op | `ops.py` | every layer's op is in the standard library or carries an explicit out shape |

Error codes are stable identifiers: `UNREACHABLE_LAYER`, `LAYER_NOT_REACHING_OUTPUT`, `CYCLE_DETECTED`, `UNKNOWN_LAYER_REFERENCE`, `SHAPE_MISMATCH`, `INPUT_ARITY_MISMATCH`, `UNDECLARED_OP`, `PARAM_BUDGET_EXCEEDED`, `OUTPUT_SHAPE_MISMATCH`.

---

## Minimal example — MLP

```markdown
# architecture SimpleMLP

## hyperparameters

| Name    | Type | Default |
|---------|------|---------|
| in_dim  | int  | 784     |
| hidden  | int  | 256     |
| out_dim | int  | 10      |

## tensors

| Name | Shape       | Dtype   |
|------|-------------|---------|
| x    | (B, in_dim) | float32 |
| y    | (B, out_dim)| float32 |

## layer x [input]
## layer fc1
- op: Linear(in_dim, hidden)
## layer act
- op: ReLU()
## layer fc2
- op: Linear(hidden, out_dim)
## layer y [output]

## flow

| Source | Target | Tensor |
|--------|--------|--------|
| x      | fc1    | x      |
| fc1    | act    | h      |
| act    | fc2    | h_act  |
| fc2    | y      | logits |
```
