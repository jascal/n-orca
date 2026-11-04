## Context

`temporal_world_model` builds ONE period of econ-sae's `TemporalWorldModel`; the
caller carries `hidden_out` back in as `hidden_in` (the "per-step graph +
external carry" decision of the archived `add-temporal-world-model`). That
representation stays. What is wrong is the step's topology:

```
today                                    econ-sae (econsae/sae/world_model.py)
x → embed → attn ─┐                      x → input_proj → attn ─┐
       └──────────Add → ln → fc1 → …     ↳───────────── Add → LayerNorm
hidden_in → Linear → hidden_out                 → GRU(embed → gru_hidden)
                                                → fc1 → fc2 → fc3
```

econ-sae runs `nn.GRU` across T periods per agent. One period of that GRU is
exactly `nn.GRUCell(embed_dim, gru_hidden)` applied to
`(LayerNorm output, previous hidden)`, which is what a per-step DAG needs.

## Goals / Non-Goals

**Goals:**
- A `GRUCell` op whose shape rule, parameter count and PyTorch emission match
  `torch.nn.GRUCell`.
- `temporal_world_model`'s step matches one period of econ-sae's model, so
  `h1` is the post-GRU activation econ-sae extracts.
- Keep the public surface: builder keyword arguments, MCP arguments, tensor
  names, and `forward(x, hidden_in) -> (y, hidden_out)`.

**Non-Goals:**
- Multi-layer GRUs (`n_gru_layers > 1`) and multiple attention layers
  (`n_attn_layers > 1`). econ-sae defaults both to 1; stacking can be added
  later as builder arguments.
- An unrolled-over-T variant, LSTM, or a generic recurrent-cell family.
- Input normalization buffers (`x_mean`/`x_std`), which belong to the trainer.

## Decisions

1. **A dedicated `GRUCell` op, not a composition of existing ops.** A GRU
   needs a sigmoid, a tanh and a `1 - z` term, none of which n-orca has as ops,
   and composing it (~12 nodes) would bury the period's structure in the
   diagram. One op mirrors `nn.GRUCell` and gives exact parameters.
   *Alternative:* express gates with `Linear` + `Mul` + `Add` nodes. Rejected:
   it needs new activation ops anyway, and it cannot be checked against
   PyTorch's parameter count as one unit.

2. **Argument order `(x, h)` comes from `## flow` row order.** The verifier and
   compiler both take a layer's inputs in flow-row order. The builder emits the
   `ln → gru` edge before `hidden_in → gru`, and the shape rule checks it: the
   second input's last dim must equal `hidden_size`. A swapped document fails
   verification rather than silently compiling a wrong cell.

3. **Shape rule.** Inputs `x (*, input_size)` and `h (*, hidden_size)`, with
   equal leading dims; output `h`'s shape. Symbolic dims are compared as
   strings, like the other shape rules.

4. **PyTorch emission over leading dims.** `nn.GRUCell` accepts only
   `(batch, features)`. The call flattens the leading dims and restores them:
   `self.gru(x.reshape(-1, x.shape[-1]), h.reshape(-1, h.shape[-1])).reshape(h.shape)`.
   This is numerically identical to econ-sae's per-agent GRU for one step,
   because econ-sae also flattens `(B, N)` into the GRU's batch.

5. **`fc1` reads the GRU output.** `fc1` becomes
   `Linear(gru_hidden, h1_dim)`, and the GRU output fans out to `fc1` and
   `hidden_out`. This is the econ-sae topology and the reason for the change.

6. **Parameter count is the exact PyTorch count:**
   `3 * (input_size*hidden_size + hidden_size^2 + 2*hidden_size)`. For
   `GRUCell(64, 128)` that is 74,496, which matches `nn.GRUCell(64, 128)`.

## Risks / Trade-offs

- [Graph-level breaking change: layer names, edges and parameter count change;
  the default build goes from 76,247 to 146,519 params] → Signature and
  `forward` I/O are unchanged; the old graph was documented as a placeholder;
  call it out in the PR and README.
- [The frozen example baseline must change for one file] → Regenerate only the
  `econ-temporal-world-model` entries and check in the test that every other
  entry is byte-identical (diff the baseline files in the PR).
- [Argument order mistakes in hand-written documents] → The shape rule
  matches each input's last dim against its op arg by name, so with symbolic
  args (`GRUCell(embed_dim, gru_hidden)`) a swap fails even if the two sizes
  are equal. It cannot catch a swap when both args are the same concrete
  integer (e.g. `GRUCell(64, 64)`); the shape error text says which edge
  goes first.

## Migration Plan

Regenerate the example from the builder. There is no data or format migration;
rollback is reverting the PR.

## Open Questions

- None blocking. Stacked GRU / attention layers are noted as a possible
  follow-up.
