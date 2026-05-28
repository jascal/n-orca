# n-orca SAE extensions — proposal for Families F1, G, and GatedSAE

**Date:** 2026-05-23
**Motivated by:** bio-sae's H-ISF + AutoML framework
(`bio-sae/docs/forge-incremental-specialist.md` §4.6-4.8), validated
empirically by econ-sae's 6-phase journey
(`econ-sae/scripts/visualize.py`).

**TL;DR.** n-orca currently ships three SAE variants (`topk_sae`,
`l1_sae`, `jumprelu_sae`) all sharing the topology
`x → Linear → activation → Linear → x_hat`. Three concrete extensions
are now well-motivated by cross-sibling-fixture findings:

| new builder | family | motivation |
|---|---|---|
| `attn_topk_sae` | **F1** | Econ-sae Phase 1.6: cross-agent attention before SAE was the BIGGEST single architectural unlock (conjunctive mAUC 0.84 → 0.97). Bio-sae predicts the same for motif recovery. |
| `supervised_topk_sae` | **G** (new family) | Econ-sae Phase 5.1: regime-supervised SAE lifted regime mAUC 0.885 → 0.972 (biggest single-phase jump). Bio-sae has 8013 labels ready for the same supervision. |
| `gated_sae` | A | Econ-sae's Phase 3 uses GatedSAE; not currently expressible in n-orca's variant list. Fills the activation-function diversity gap (vs TopK/L1/JumpReLU). |

All three are buildable today by extending `n_orca/sae.py` + adding
their MCP wrappers. Each emits clean Mermaid + PyTorch via the
existing compiler. Estimated implementation: 1 day for all three +
tests.

---

## 1. `attn_topk_sae` — attention-prefixed sparse autoencoder

### Topology

```
x [B, T, d_in]
 → MultiheadAttention(d_in, n_heads, dropout)   ← cross-sequence-position context
 → LayerNorm                                     ← post-attention norm (standard)
 → Linear(d_in, n_features)                     ← encoder
 → ReLU
 → TopK(k)                                       ← sparsity
 → Linear(n_features, d_in)                     ← decoder
 → x_hat [B, T, d_in]
```

### Why this matters

Per-residue SAEs (bio-sae) and per-agent SAEs (econ-sae) both have a
*structural ceiling* on multi-position / multi-agent composition: a
plain encoder can't represent "this residue is part of a 5-residue
helix-turn-helix motif" because it sees one residue at a time.
Attention before encoding lifts that ceiling.

Bio-sae's `motif-recovery-architecture-limit` memory pins this as the
*only* hypothesis still standing after 5 ablation axes ruled out
scale / layer / positional encoding / wildcards / pooled-feed
alternatives. Econ-sae's Phase 1.6 already shipped the equivalent
(`AttnWorldModel`) and saw conjunctive mAUC 0.84 → 0.97 — a +0.13
absolute lift that no other intervention in econ-sae's 6-phase
journey matched.

### MCP API

```python
mcp__n_orca__build_sae(
    variant="attn_topk",
    input_dim=320,
    n_features=1024,
    k=64,
    n_heads=4,                    # NEW kwarg
    attn_dropout=0.0,             # NEW kwarg
    name="AttnTopKSae",
)
```

### Implementation sketch

```python
def attn_topk_sae(*, input_dim, n_features, k, n_heads, attn_dropout, name, tied_decoder):
    arch = _sae_skeleton(name=name, input_dim=input_dim, n_features=n_features,
                          variant_description=f"Attention-prefixed Top-k SAE (k={k}, heads={n_heads})...",
                          extra_hps=[
                              Hyperparameter("k", "int", k),
                              Hyperparameter("n_heads", "int", n_heads),
                              Hyperparameter("attn_dropout", "float", attn_dropout),
                          ])
    # Input tensor changes shape: (B, T, input_dim) instead of (B, input_dim).
    arch.tensors[0] = Tensor("x", ("B", "T", "input_dim"), "float32")
    arch.tensors[1] = Tensor("x_hat", ("B", "T", "input_dim"), "float32")
    arch.layers.extend([
        Layer(name="attn", op=OpCall("MultiheadAttention",
                                      ["input_dim", "n_heads", "attn_dropout"])),
        Layer(name="ln", op=OpCall("LayerNorm", ["input_dim"])),
        Layer(name="encoder", op=OpCall("Linear", ["input_dim", "n_features"])),
        Layer(name="relu", op=OpCall("ReLU", [])),
        Layer(name="topk", op=OpCall("TopK", ["k"])),
        Layer(name="decoder", op=OpCall("Linear", ["n_features", "input_dim"])),
    ])
    arch.flow.extend([
        FlowEdge("x", "attn", "x"),
        FlowEdge("attn", "ln", "x_attn"),
        FlowEdge("ln", "encoder", "x_ln"),
        FlowEdge("encoder", "relu", "z_pre"),
        FlowEdge("relu", "topk", "z_relu"),
        FlowEdge("topk", "decoder", "z_sparse"),
        FlowEdge("decoder", "x_hat", "x_hat"),
    ])
    _close_sae(arch, tied=tied_decoder)
    return arch
```

The `MultiheadAttention` op is already in `n_orca/ops/spec.py:340-350`
(`_torch_init_mha`). The compile path produces:

```python
self.attn = nn.MultiheadAttention(input_dim, n_heads, dropout=attn_dropout, batch_first=True)
# ... in forward:
x_attn, _ = self.attn(x, x, x)   # self-attention; uses _torch_call_mha pattern
```

The `_torch_call_mha` helper already handles the `(x, _) = mha(x, x, x)`
unpacking. No new op-registry entries needed.

### Shape rule notes

- `MultiheadAttention` requires sequence dim `T` ≥ 1. The input tensor
  shape change `(B, input_dim)` → `(B, T, input_dim)` is a breaking
  difference from the other SAE variants — consumers must feed
  per-residue (not pooled) activations.
- Need to verify n-orca's shape inference handles the 3D tensor
  threading. Existing world_models.py uses MultiheadAttention in
  similar 3D-tensor contexts so the inference path likely works.

---

## 2. `supervised_topk_sae` — joint reconstruction + classifier head

### Topology

```
x [B, d_in]
 → Linear(d_in, n_features)
 → ReLU
 → TopK(k)                        ← shared sparse latents
 ├─→ Linear(n_features, d_in)    → x_hat       (reconstruction)
 └─→ Linear(n_features, n_labels) → y_logits   (auxiliary classifier head)
```

### Why this matters

Econ-sae Phase 5.1 (`regime_supervised_experiment.py`) showed that
adding a per-label classifier head trained jointly with the
reconstruction loss lifted regime mAUC 0.885 → 0.972 — the
biggest single-phase jump in econ-sae's 6-phase journey, AND on the
tier where every other technique had plateaued. Bio-sae has 8013
labels in `bundle["labels_protein_Y"]`; the top-100 most-prevalent
would be a natural target subset.

Per-feature AUC is already what bio-sae's evaluation harness scores
against; supervising the SAE jointly with the same labels at train
time directly aligns the basis with what's evaluated.

### MCP API

```python
mcp__n_orca__build_sae(
    variant="supervised_topk",
    input_dim=320,
    n_features=1024,
    k=64,
    n_labels=100,                  # NEW kwarg — auxiliary classifier output dim
    aux_weight=0.1,                # NEW kwarg — loss weight (metadata)
    name="SupervisedTopKSae",
)
```

### AST changes needed

This is the first SAE variant with MULTIPLE outputs (`x_hat` AND
`y_logits`). Need to verify `Architecture.tensors` and the output-edge
schema support multi-output graphs cleanly.

```python
arch.tensors = [
    Tensor("x",       ("B", "input_dim"),  "float32"),
    Tensor("x_hat",   ("B", "input_dim"),  "float32"),
    Tensor("y_logits", ("B", "n_labels"),  "float32"),   # NEW
]
arch.layers.extend([
    Layer(name="encoder", op=OpCall("Linear", ["input_dim", "n_features"])),
    Layer(name="relu",    op=OpCall("ReLU", [])),
    Layer(name="topk",    op=OpCall("TopK", ["k"])),
    Layer(name="decoder", op=OpCall("Linear", ["n_features", "input_dim"])),
    Layer(name="aux_head", op=OpCall("Linear", ["n_features", "n_labels"])),
    Layer(name="y_logits", is_output=True),
])
arch.flow.extend([
    FlowEdge("x", "encoder", "x"),
    FlowEdge("encoder", "relu", "z_pre"),
    FlowEdge("relu", "topk", "z_relu"),
    FlowEdge("topk", "decoder", "z_sparse"),
    FlowEdge("topk", "aux_head", "z_sparse"),         # branch
    FlowEdge("decoder", "x_hat", "x_hat"),
    FlowEdge("aux_head", "y_logits", "y_logits"),
])
```

**If multi-output isn't currently supported**, this is the
forcing-function PR for that feature. (Worth checking
`Architecture.is_output` semantics and the PyTorch compiler's
forward-method codegen.)

### Loss term

Aux-head loss is BCE-with-logits per output dim, multiplied by
`aux_weight`. Like L1/JumpReLU's sparsity terms, n-orca captures only
the forward graph; consumers (bio-sae's trainer) implement the loss.

---

## 3. `gated_sae` — sigmoid-gated activation

### Topology

```
x [B, d_in]
 → Linear(d_in, n_features)              ← magnitude projection
 → Linear(d_in, n_features) → Sigmoid    ← gate projection (parallel)
 → magnitude * gate                       ← element-wise product (z_gated)
 → Linear(n_features, d_in)              → x_hat
```

### Why this matters

Econ-sae's Phase 3 used GatedSAE for plateaued features. The
multiplicative gate is structurally different from TopK (hard
masking) and L1/JumpReLU (additive penalty / hard threshold) —
it allows graded contributions, useful for features that vary in
intensity rather than presence/absence.

For bio-sae, this is an activation-function diversity member of
Family A. The current AutoML library has only TopK-based variants
of basis selection; a GatedSAE shadow would give the ensemble a
fundamentally different activation signature to pick from.

### MCP API

```python
mcp__n_orca__build_sae(
    variant="gated",
    input_dim=320,
    n_features=1024,
    name="GatedSae",
)
```

No new hyperparameters beyond the base SAE.

### AST sketch

```python
arch.layers.extend([
    Layer(name="magnitude_proj", op=OpCall("Linear", ["input_dim", "n_features"])),
    Layer(name="gate_proj",      op=OpCall("Linear", ["input_dim", "n_features"])),
    Layer(name="sigmoid",        op=OpCall("Sigmoid", [])),
    Layer(name="gate_mul",       op=OpCall("ElementwiseMul", [])),
    Layer(name="decoder",        op=OpCall("Linear", ["n_features", "input_dim"])),
])
arch.flow.extend([
    FlowEdge("x", "magnitude_proj", "x"),
    FlowEdge("x", "gate_proj", "x"),
    FlowEdge("gate_proj", "sigmoid", "g_pre"),
    FlowEdge("magnitude_proj", "gate_mul", "m"),
    FlowEdge("sigmoid", "gate_mul", "g"),
    FlowEdge("gate_mul", "decoder", "z_gated"),
    FlowEdge("decoder", "x_hat", "x_hat"),
])
```

Needs an `ElementwiseMul` op in the registry (likely doesn't exist
yet) — `Sigmoid` may also be missing as a standalone op (often baked
into other compositions). Both are 1-line registry additions.

---

## 4. Implementation order + acceptance gates

**Recommended ship order** (per ROI):

1. **`attn_topk_sae`** — econ-sae's biggest lever, well-validated in a
   sibling fixture; bio-sae has a falsifiable acceptance gate from
   `motif-recovery-architecture-limit` (any synthetic motif moving
   above AUC=0.95 = the lever lands).
2. **`supervised_topk_sae`** — net-new family, biggest econ-sae jump;
   acceptance gate: per-label AUC on supervised labels rises
   measurably over the unsupervised baseline.
3. **`gated_sae`** — modest expected lift; mostly useful as an extra
   AutoML library member for activation-function diversity.

**Per-builder PR acceptance:**

- New builder lives in `n_orca/sae.py`.
- MCP server `n_orca/mcp_server.py::build_sae` recognizes the new
  variant strings.
- `tests/test_sae_and_world_models.py` adds: AST verifies, Mermaid
  compiles cleanly, PyTorch compiles to runnable nn.Module.
- `examples/` directory gets a worked-example for each new builder
  (mirroring the existing TopK example shape).
- `scripts/generate_sae_docs.py` regenerates the cross-sibling docs.

## 5. What this unlocks downstream

Per the bio-sae H-ISF + AutoML framework:

| n-orca add | bio-sae downstream | acceptance |
|---|---|---|
| `attn_topk_sae` | Train new bio-sae SAE, add as Family F1 candidate in AutoML library | Pfam beats-host count > 0 |
| `supervised_topk_sae` | Train new bio-sae SAE supervised on top-100 labels, add as Family G candidate | Supervised-label AUCs rise materially |
| `gated_sae` | Train new bio-sae GatedSAE, add as Family A candidate | AutoML picks it for a label subset that TopK variants miss |

Per the cross-sibling pattern: all three siblings (sm-sae, econ-sae,
bio-sae) benefit from the same additions, since all three are
SAE-fixture repos hitting the single-substrate ceiling. econ-sae has
been the lead-implementor of F + G; sm-sae and bio-sae have been
basis-side-only. Landing these in n-orca makes the architectures a
shared resource instead of econ-sae-internal code.

---

## 6. Open design questions

1. **3D-tensor threading for attn_topk_sae**: does n-orca's shape
   inference handle `(B, T, d_in)` cleanly across the
   Attention→LayerNorm→Linear chain? World-models do this already,
   but the SAE shape rules may have a 2D-tensor assumption baked in.
2. **Multi-output architectures**: does
   `Architecture.is_output=True` on multiple layers compile
   correctly? Need to test on a multi-head world model first if not.
3. **n-orca AST capability for skip connections** (relevant if we
   want a residual around the attention block — common in
   transformer-style SAE designs). May be future work.
4. **Tied decoder option** for the new variants — existing variants
   support `tied_decoder=True` as a verification-rule note. Should
   the new variants inherit this?
