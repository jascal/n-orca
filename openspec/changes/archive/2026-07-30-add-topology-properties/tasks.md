# Tasks — add-topology-properties

Work test-first: within each group, the test task comes before the
implementation it drives.

## 0. Preconditions

- [x] 0.1 `gh pr list --state open` — confirm no open PR touches the verifier,
      parser, or AST. (Result: no open PRs.)
- [x] 0.2 Capture the pre-change baseline: `verify` and `compile mermaid`
      output for every file in `examples/` and `examples/hf-generated/`, plus
      the test count (143 passed, 6 skipped).
- [x] 0.3 `openspec new change add-topology-properties`; write `proposal.md`,
      `design.md`, `specs/verification/spec.md`, this file.
- [x] 0.4 `openspec validate add-topology-properties` passes.

## 1. AST

- [x] 1.1 Add `Group(name, layers)` to `n_orca/ast.py`.
- [x] 1.2 Add `Property(kind, source, target, through, subject, op, bound, raw)`
      to `n_orca/ast.py`.
- [x] 1.3 Add `Architecture.groups: list[Group]` and
      `Architecture.properties: list[Property]`, both `default_factory=list`.
- [x] 1.4 Add `Architecture.group(name)` accessor alongside `layer()` /
      `tensor()`.

## 2. Parser

- [x] 2.1 Tests: `## groups` table parses to `Group` objects, member order
      preserved, whitespace tolerated, empty cells skipped.
- [x] 2.2 Tests: each of the six property kinds parses; `raw` is verbatim;
      unparseable and unknown-kind bullets still produce a `Property` carrying
      enough information for Stage 6 to emit `PROPERTY_MALFORMED` (the parser
      does not raise).
- [x] 2.3 Test: a document with an unrelated unknown `##` section still parses
      unchanged (the silent-ignore path is preserved).
- [x] 2.4 Implement `_parse_groups` (reuse `_parse_table`) and
      `_parse_properties`; dispatch `"groups"` and `"properties"` in
      `_parse_document` alongside `invariants` / `verification rules`.

## 3. Render round-trip

- [x] 3.1 Test: an `Architecture` with groups and properties renders and
      re-parses to an equal AST.
- [x] 3.2 Test: an `Architecture` with neither renders byte-identically to
      today.
- [x] 3.3 Emit `## groups` (aligned table, via `_render_table`) and
      `## properties` (bullets, from `Property.raw` when present, else
      reconstructed) in `n_orca/render.py`.

## 4. Stage 6 — the checker

- [x] 4.1 Tests: group diagnostics — `GROUP_OVERLAP` (warning, names the layer
      and both groups), `GROUP_NAME_COLLISION` (error),
      `PROPERTY_UNKNOWN_REFERENCE` for a group member that is not a layer.
- [x] 4.2 Tests: `reachable` passes and fails.
- [x] 4.3 Tests: `unreachable` passes and fails; group-to-group universal
      quantification; the asymmetric pair (`core from cortex` passes,
      `cortex from core` fails); counterexample path present and correct.
- [x] 4.4 Tests: `passes_through` passes; catches a bypass edge added around
      the named cut with the bypassing path in the message; multi-member cut
      set deleted as a whole; vacuous case holds.
- [x] 4.5 Tests: `within` passes; fails-too-far reports the actual distance;
      fails-no-path says so distinctly.
- [x] 4.6 Tests: `param_fraction` passes and fails; failure message carries
      both the percentage and the raw counts; missing `%` is
      `PROPERTY_MALFORMED`; zero total is skipped.
- [x] 4.7 Tests: `out_dim` passes against a hyperparameter bound and against an
      arithmetic expression; fails; a group operand is `PROPERTY_MALFORMED`;
      an unresolvable bound or missing inferred shape is skipped silently.
- [x] 4.8 Tests: `PROPERTY_MALFORMED` for each malformed shape;
      `PROPERTY_UNKNOWN_REFERENCE` for an unknown operand.
- [x] 4.9 Tests: degenerate operands per design §4.6 — empty group, overlapping
      operands, `A == B`.
- [x] 4.10 Tests: Stage 6 skipped silently when Stage 1 fails and when Stage 2
      fails; Stage 6 still runs when only Stage 3 fails.
- [x] 4.11 Test: every one of the ten codes carries a non-empty `suggestion`.
- [x] 4.12 Test: counterexample paths are deterministic across repeated runs
      and shortest.
- [x] 4.13 Implement `n_orca/verifier/properties.py`: adjacency in flow-row
      order, `_bfs(succ, sources, exclude=...)` returning `(dist, parent)`,
      `_witness(dist, parent, targets)` for the shortest counterexample,
      operand resolution, one checker per kind, `check_properties(arch, report)`.
      (No `max_depth` argument in the end — `within` needs the *actual* shortest
      distance for its message, so the search is unbounded and the bound is a
      comparison; see design §4.4.)
- [x] 4.14 Add `PropertyResult` to the report (`kind`, `raw`, `passed`,
      `code`, `path`) plus `properties_checked` / `properties_failed`, and
      extend `VerificationReport.to_dict()` with `properties` and `groups`.
- [x] 4.15 Wire one `check_properties(arch, report)` call into `verify()` inside
      the existing naming/structural nest, after Stage 4. Update the module
      docstring to six stages. The import is function-local, with a comment:
      `properties.py` needs `_resolve_arg` / `layer_params` from `verifier.py`,
      so a module-level import in both directions would be a cycle.

## 5. Mermaid

- [x] 5.1 Test: an architecture with groups emits one `subgraph` per group with
      its member nodes inside, all edges preserved, ungrouped nodes declared
      outside.
- [x] 5.2 Test: a layer in two groups appears exactly once, in the first group.
- [x] 5.3 Test: with no groups, output is byte-identical to the baseline for
      every pre-existing example.
- [x] 5.4 Implement subgraph emission in `n_orca/compiler/mermaid.py`
      (`g_`-prefixed ids; grouped nodes, then ungrouped nodes, then edges, then
      the unchanged `classDef` block).

## 6. CLI

- [x] 6.1 Test: `Properties: N checked, M failed` printed when properties
      exist; absent when they do not.
- [x] 6.2 Test: `--json` payload carries `properties` and `groups`.
- [x] 6.3 Test: `--strict` promotes `GROUP_OVERLAP` to an error and the file
      becomes invalid.
- [x] 6.4 Test: `verify` exit code is 1 on the negative example, 0 on the
      positive one.
- [x] 6.5 Implement in `n_orca/cli/main.py` (`_print_human_report`).

## 7. MCP

- [x] 7.1 Test: `verify_markdown` returns `properties` and `groups` in each
      architecture dict.
- [x] 7.2 Confirm no change is needed beyond `to_dict()` — and if none is,
      record that in the test rather than editing `mcp_server.py`.

## 8. Examples

- [x] 8.1 Author `examples/subcortical-value-cortex.n.orca.md`: transformer
      cortex, isolated value core, neuromodulator bus, `## groups`,
      `## properties` exercising all six kinds. Must verify `VALID`.
- [x] 8.2 Generate `examples/subcortical-value-cortex.mmd` (and `…-leak.mmd`)
      via `compile mermaid`, matching the convention of the other examples. The
      leak's diagram is worth shipping: the leaking edge is visibly the only one
      crossing from the `cortex` subgraph into the `core` subgraph.
- [x] 8.3 Author `examples/subcortical-value-cortex-leak.n.orca.md`: the same
      file plus `critic_fc` + `feedback_proj -> core_add`. Must fail
      `PROPERTY_EXCLUSION_FAIL` with
      `cortex_ln -> critic_fc -> feedback_proj -> core_add` in the message, and
      must carry **no** shape/naming/structural error.
- [x] 8.4 Test: both examples behave as specified, asserted on the message text
      not just the code.
- [x] 8.5 Note: the leak example is INVALID by design, so it must be excluded
      from the CI "verify every shipped example" sweep — update the glob in
      `.github/workflows/ci.yml` accordingly.

## 9. Regression

- [x] 9.1 Test: every pre-existing file in `examples/` and
      `examples/hf-generated/` produces `verify` output identical to the
      committed baseline.
- [x] 9.2 Test: the same files produce identical `compile mermaid` output.
- [x] 9.3 `python -m pytest -q` clean: **231 passed, 7 skipped** (143 pre-existing
      + 88 new; the 7th skip is the new MCP test, which skips locally for the
      same reason the existing 6 do — `mcp` is an optional extra and is not
      installed in `.venv`. Verified separately against `mcp<2` from a scratch
      install: it passes).
- [x] 9.4 Lint: **`ruff` is not installed in `.venv` and is not declared in
      `pyproject.toml`, so this repo has no ruff gate** (CI runs `pytest` only) —
      the workspace-level `CLAUDE.md` instruction does not apply here. Checked
      anyway with a scratch-installed ruff: the two new files
      (`verifier/properties.py`, `tests/test_properties.py`) are clean under its
      full default rule set. ~80 pre-existing findings remain across untouched
      files (import sorting, blind `except`, etc.) and were deliberately left
      alone rather than reformatting the codebase inside this change.

## 10. Docs and skill

- [x] 10.1 `docs/grammar.md`: document `## groups` and `## properties` with the
      semantics table and the group-quantification rules.
- [x] 10.2 `docs/grammar.md`: **bonus fix** — the pipeline table lists
      "1 — Structural / 2 — Naming"; the code and `docs/verification.md` both
      have Naming first. Correct the order and add Stage 6.
- [x] 10.3 `docs/verification.md`: add the ten codes to the catalog with
      Stage 6, and document the counterexample format.
- [x] 10.4 `.claude/skills/n-orca-verify/SKILL.md`: teach the new codes and
      what to do about each.
- [x] 10.5 `AGENTS.md` / `README.md`: mention property checking where the
      verifier's stages are listed.
