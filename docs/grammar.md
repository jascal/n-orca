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
```

Supported predicates: `param_count`, `flops`, `depth`, `output_shape`.
Suffixes: `K`, `M`, `G` (decimal) on numeric bounds.

### `## groups`

Names a set of layers, so `## properties` can make claims at subgraph
granularity. A table, like `## tensors`.

```markdown
## groups

| Name   | Layers                                                  |
|--------|---------------------------------------------------------|
| core   | core_sense, core_fc, core_add, core_act, affect_head     |
| cortex | embed, attn_norm, attn, add_attn, ff, add_ff, cortex_ln  |
| bus    | nm_bus, nm_receptor, nm_gain                             |
```

- **Layers** — a comma-separated list of declared layer names. Order is
  preserved but not significant.
- A group is *only* a set of layers: not a subgraph, not a module, not a
  namespace. It has no shape and contributes no parameters of its own.
- A group name may be used anywhere a layer name is accepted in
  `## properties`.
- Groups may overlap, but overlap emits `GROUP_OVERLAP` (warning): it usually
  means the boundary being asserted is not the one intended.
- A group name equal to a layer name is `GROUP_NAME_COLLISION` (error) — a
  property operand must resolve to exactly one thing.

Groups also render as Mermaid `subgraph` blocks, so an asserted boundary is
visible in the diagram.

### `## properties`

Checkable claims about the topology, decided by verification Stage 6. One
bullet per property. Mirrors [Orca](https://github.com/jascal/orca-lang)'s
`properties` block.

```markdown
## properties
- reachable: v_hat from obs
- unreachable: core from cortex
- passes_through: bus for core -> cortex
- within: y from obs <= 12
- param_fraction: core <= 1%
- out_dim: nm_bus <= n_affect
```

| Property | Syntax | Meaning | Method |
|----------|--------|---------|--------|
| Reachability | `reachable: B from A` | ∃ path A→B | BFS from A |
| Exclusion | `unreachable: B from A` | ¬∃ path A→B | BFS from A, assert B unvisited |
| Pass-through | `passes_through: C for A -> B` | every A→B path visits C | delete C, assert B unreachable from A |
| Bounded depth | `within: B from A <= N` | B reachable from A in ≤ N edges | shortest-path BFS |
| Param fraction | `param_fraction: G <= P%` | G's params ≤ P% of total | Σ group params / total |
| Output width | `out_dim: L <= <expr>` | L's inferred last dim ≤ expr | read the inferred shape |

`A`, `B`, `C` and `G` may each be a layer name or a group name. `L` must be a
single layer — a group has no output width. `out_dim` bounds the **trailing**
dimension of `L`'s inferred shape (`D` in `(B, S, D)`); see
[`verification.md`](./verification.md) for what that means when the last axis
is spatial.

**Group quantification.** When an operand is a group, the property ranges over
its members:

- `reachable: B from A` holds iff **some** `a ∈ A` reaches **some** `b ∈ B`.
  One live wire makes influence possible.
- `unreachable: B from A` holds iff **every** `a ∈ A` fails to reach **every**
  `b ∈ B`. It is the exact complement of `reachable`.
- `passes_through: C for A -> B` holds iff deleting **all** of `C`'s layers
  makes **every** `b ∈ B` unreachable from **every** `a ∈ A`. `C` is a cut
  *set*: deleting its members one at a time is a different, weaker check.
- `within: B from A <= N` holds iff **some** pair is within `N` edges — it is
  `reachable` plus a budget.

**The relation is directed.** `from` always names the source, and
`unreachable: core from cortex` is a different claim from
`unreachable: cortex from core`. On a design where a bus carries core →
cortex modulation, the first holds and the second correctly fails.

**These checks are exact, not approximations.** A `## flow` edge carries no
guard: if the row is in the table, the tensor flows along it on every forward
pass. So a reported counterexample path is a real activation and gradient path.

`param_fraction` and `out_dim` have no Orca equivalent — a state has no width
and a machine has no parameters. `within` is the DAG-native rename of Orca's
`responds`: N counts edges, not events, and the claim is about depth rather
than latency.

Failures carry stable codes and a counterexample path where one exists — see
[`verification.md`](./verification.md).

### `## verification rules`

Narrative rules, used by the LLM and reflected in stage-by-stage error reporting.

```markdown
## verification rules
- shape-consistency: all flow edges have matching source/target shapes
- residual-symmetry: every Add layer has at least two inputs of identical shape
- dag-acyclicity: the layer graph is acyclic
```

These mostly mirror the built-in checks; including them gives the LLM the vocabulary to think with.

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
| `MultiHeadAttention(d, h, dropout[, causal])` | `(B, S, d) -> (B, S, d)` | `nn.MultiheadAttention` (batch_first); literal `causal` 4th arg adds a causal mask | preserve |
| `DualStreamJointAttention(d, h, dropout)` | `dm (B, S_dm, d)`, `ar (B, S_ar, d)` -> `(B, S_dm, d)`; inputs in `## flow` row order | `nn.MultiheadAttention`: queries `dm`, keys/values `cat([ar, dm], dim=1)`, no mask | dm's shape |
| `FeedForward(d, d_ff, dropout)` | `(*, d) -> (*, d)` | Linear-GELU-Dropout-Linear | preserve |
| `GRUCell(in, hidden)` | `x (*, in)`, `h (*, hidden)` -> `(*, hidden)`; inputs in `## flow` row order | `nn.GRUCell` over flattened leading dims | h's shape |
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

| Stage | Checks |
|-------|--------|
| 1 — Naming | every flow Source/Target exists; every input/output layer matches a `## tensors` row |
| 2 — Structural | every layer reachable from inputs; every layer reaches an output; no cycles |
| 3 — Shape | input shapes to each layer match its op's input rule; declared `shape:` matches inferred |
| 4 — Resource bounds | `param_count` / `flops` / `depth` / `output_shape` against `## invariants` |
| 5 — Op | every layer's op is in the standard library or carries an explicit out shape |
| 6 — Property | `## properties` claims about the topology, over `## groups` |

Naming runs first: the later stages all assume every name resolves. Stage 6
runs last, because it needs a well-formed acyclic graph (Stages 1–2) *and* the
inferred shapes and parameter counts (Stages 3–4); it is skipped silently when
an earlier stage failed.

Error codes are stable identifiers: `UNREACHABLE_LAYER`, `LAYER_NOT_REACHING_OUTPUT`, `CYCLE_DETECTED`, `UNKNOWN_LAYER_REFERENCE`, `SHAPE_MISMATCH`, `INPUT_ARITY_MISMATCH`, `PARAM_BUDGET_EXCEEDED`, `OUTPUT_SHAPE_MISMATCH`, `PROPERTY_EXCLUSION_FAIL`. The full catalog is in [`verification.md`](./verification.md).

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
