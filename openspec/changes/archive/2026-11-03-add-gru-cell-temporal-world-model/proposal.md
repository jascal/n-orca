## Why

`temporal_world_model` is meant to be the n-orca spec of econ-sae's
`TemporalWorldModel`, but it does not describe that model. Its state path is a
standalone `Linear(gru_hidden, gru_hidden)` from `hidden_in` to `hidden_out`:
the period's input never reaches the state, and the state never reaches the
prediction head. In econ-sae a GRU sits *between* the attention block and the
MLP head (`input_proj → attn → LayerNorm(h + attn) → GRU → fc1 → fc2 → fc3`),
so `h1`, the SAE substrate, is a post-GRU activation. Today's graph puts `h1`
before any recurrence, so it cannot carry cross-period (regime) information.
This is the top follow-up left by the archived `add-temporal-world-model`.

## What Changes

- New op `GRUCell(input_size, hidden_size)`: two inputs `(x, h)`, output shaped
  like `h`, PyTorch-standard parameter count, emitted as `nn.GRUCell` applied
  over the flattened leading dims so `(B, N, ·)` tensors work.
- **BREAKING (generated graph only):** `temporal_world_model` rewires its
  per-period step to econ-sae's topology: `ln` and `hidden_in` feed a `gru`
  (`GRUCell(embed_dim, gru_hidden)`), and its output feeds both `hidden_out` and
  `fc1`, which becomes `Linear(gru_hidden, h1_dim)`. The `hidden_update` layer
  is removed. The builder signature, MCP arguments, tensor names and
  `forward(x, hidden_in) -> (y, hidden_out)` are unchanged; the layer graph,
  parameter count and `h1` semantics change.
- Regenerate `examples/econ-temporal-world-model.{n.orca.md,mmd}`. That example
  is in the frozen regression baseline, so its baseline entries are updated
  deliberately, and no other entry may change.
- Add the missing MCP test for `build_world_model(variant="temporal")` and drop
  the always-true `'gru_hidden' in locals()` guard in `mcp_server.py`.

## Capabilities

### New Capabilities
- `gru-cell-op`: the `GRUCell` op's shape rule, parameter count and PyTorch
  emission.
- `world-model-temporal-gru`: the temporal world model's per-period topology
  with a real GRU state update. It supersedes the state-update requirement in
  the archived `add-temporal-world-model` spec, which described the linear
  placeholder.

### Modified Capabilities
- None. The repo has no `openspec/specs/` (archived specs were not synced), so
  the superseded requirement is recorded above instead of as a delta.

## Impact

- `n_orca/ops/spec.py` (new op), `n_orca/world_models.py`
  (`temporal_world_model`), `n_orca/mcp_server.py` (guard removal).
- `examples/econ-temporal-world-model.*`,
  `tests/data/example_{verify,mermaid}_baseline.txt` (that one entry),
  `tests/test_sae_and_world_models.py`, `tests/test_mcp_server.py`.
- README world-model list and the builder docstring.
- Consumers: any downstream use of the generated temporal graph sees new layer
  names and parameter counts. n-orca is consumed only as a published package,
  so nothing changes for sibling repos until a release.
