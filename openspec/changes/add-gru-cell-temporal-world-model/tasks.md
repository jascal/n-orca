## 1. GRUCell op

- [x] 1.1 Add tests first in `tests/test_ops.py`: shape rule for valid, swapped and wrong-arity inputs; `GRUCell(64, 128)` params == 74,496 == `nn.GRUCell(64, 128)`.
- [x] 1.2 Register `GRUCell` in `n_orca/ops/spec.py` (arity 2): shape rule per design §3, params per design §6.
- [x] 1.3 PyTorch emission: `nn.GRUCell(input_size, hidden_size)` init; call flattens leading dims and restores `h`'s shape. Add a torch test comparing against row-by-row `nn.GRUCell` on `(2, 5, ·)` inputs.

## 2. Temporal builder

- [x] 2.1 Update `tests/test_sae_and_world_models.py::test_temporal_world_model_has_state_tensors`: expect `GRUCell`, no `hidden_update`, `gru` inputs `ln` then `hidden_in`, `gru` successors exactly `{fc1, hidden_out}`, default params 146,519.
- [x] 2.2 Add a torch test: same `x` with different `hidden_in` changes both `y` and `hidden_out`; default forward I/O shapes unchanged.
- [x] 2.3 Rewire `temporal_world_model` in `n_orca/world_models.py` per design §5 (emit `ln → gru` before `hidden_in → gru`); update its docstring and description to drop the placeholder wording.

## 3. MCP

- [x] 3.1 Add `tests/test_mcp_server.py` test for `build_world_model(variant="temporal", gru_hidden=32, ...)`: valid report, `gru_hidden` recorded as 32.
- [x] 3.2 Remove the always-true `'gru_hidden' in locals()` guard in `n_orca/mcp_server.py`.

## 4. Example and baseline

- [x] 4.1 Regenerate `examples/econ-temporal-world-model.{n.orca.md,mmd}` from the builder; `n-orca verify` → VALID.
- [x] 4.2 Regenerate only the `econ-temporal-world-model` entries in `tests/data/example_{verify,mermaid}_baseline.txt`; confirm with `git diff` that no other entry changed.

## 5. Docs and close

- [x] 5.1 Update the README world-model bullet (GRU state update; graph change from the placeholder) and remove the GRU item from "Open follow-ups" in `IMPROVEMENT_GOAL.md`, with a Recent Activity entry.
- [x] 5.2 Run `openspec validate`, the full `pytest -q`, and the example verification sweep; all green.
- [x] 5.3 Branch, commit and open a PR that calls out the graph-level breaking change and shows the baseline diff.
