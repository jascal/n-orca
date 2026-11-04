## 1. Causal MultiHeadAttention

- [x] 1.1 Tests first: compiled `MultiHeadAttention(d, h, 0.0, causal)` leaves earlier positions unchanged when a later one is perturbed; the three-argument form is still unmasked; params identical.
- [x] 1.2 Implement the `causal` token in MHA emission (`n_orca/ops/spec.py` + the attention branch in `n_orca/compiler/pytorch.py`), per design §6.

## 2. Two-input DualStreamJointAttention

- [x] 2.1 Tests first in `tests/test_ops.py`: arity 2, `(dm, ar)` shape rule (valid, one input, mismatched feature dim), params unchanged; update `test_dual_joint_preserves_shape_and_params`.
- [x] 2.2 Implement the shape rule and concat-K/V emission; move the op out of the compiler's single-input attention special case onto the `{NAME}` call template. Add an emission test: different `ar` changes the output.

## 3. Builders

- [x] 3.1 Tests first in `tests/test_sae_and_world_models.py`: `dm_mha` inputs `dm_ln` then `ar_ln`; groups and all four properties pass; params unchanged; numeric mask checks (a)–(c) for `mot_denoise_step` and the causal check for `mot_reasoner_only`.
- [x] 3.2 Rewire `mot_denoise_step` per design §4 and §7, and mark `mot_reasoner_only` causal; update docstrings and descriptions to drop the "placeholder" wording and state the §2 approximation.

## 4. Examples and baseline

- [x] 4.1 Confirm the committed MoT examples equal the current builders' output, then regenerate `examples/cosmos-mot-denoise-step.*` and `examples/cosmos-mot-reasoner.*`; `n-orca verify` → VALID.
- [x] 4.2 Move the denoise-step example into the frozen-baseline exclusion set in `tests/test_properties.py` and remove its baseline entries; regenerate the reasoner's entries (it declares no groups, so it stays); confirm with `git diff` that no other entry changed.

## 5. Docs and close

- [x] 5.1 Update the op tables in README and `docs/grammar.md` (MHA `causal`, two-input joint op), the world-model skill, and `IMPROVEMENT_GOAL.md` (drop this follow-up; add the timestep-injection gap; Recent Activity entry).
- [x] 5.2 Run `openspec validate`, the full `pytest -q`, the CI example sweep, and `ruff` on the changed files.
