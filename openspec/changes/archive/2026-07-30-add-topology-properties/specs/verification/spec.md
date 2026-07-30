# Verification — Topology Properties

This capability covers the `## groups` and `## properties` sections, the
Stage 6 property checker (`n_orca/verifier/properties.py`), and the propagation
of property results through the render, Mermaid, CLI, MCP and skill surfaces.

## ADDED Requirements

### Requirement: `## groups` names sets of layers

The document grammar SHALL accept an optional `## groups` section, parsed as a
markdown table with columns `Name` and `Layers`, where `Layers` is a
comma-separated list of declared layer names. Each row SHALL become a `Group`
in `Architecture.groups`.

#### Scenario: A group table parses
- A document declares `## groups` with a row `core | core_sense, core_fc`
- `parse()` yields one `Group(name="core", layers=["core_sense", "core_fc"])`
- Member order is preserved

#### Scenario: A group name may be used wherever a layer name may be used
- A property reads `unreachable: core from cortex` where both are group names
- The checker resolves each operand to its member layers and checks the
  property over member pairs

#### Scenario: Overlapping groups warn
- Layer `nm_bus` is listed in both `bus` and `cortex`
- Verification emits `GROUP_OVERLAP` at severity `warning`, naming the layer
  and both groups
- The architecture remains `valid` unless `--strict` is set

#### Scenario: A group name shadowing a layer name is an error
- A group is named `attn` and a layer is named `attn`
- Verification emits `GROUP_NAME_COLLISION` at severity `error`
- The architecture is `invalid`

#### Scenario: A group naming an undeclared layer is an error
- Group `core` lists `core_ghost`, which is not a declared layer
- Verification emits `PROPERTY_UNKNOWN_REFERENCE` naming `core_ghost`

### Requirement: `## properties` declares checkable topology claims

The document grammar SHALL accept an optional `## properties` section: a bullet
list where each bullet declares one property. Six kinds SHALL be supported —
`reachable`, `unreachable`, `passes_through`, `within`, `param_fraction`,
`out_dim` — with the surface syntax below. Each bullet SHALL become a
`Property` in `Architecture.properties`.

| Kind | Syntax |
|------|--------|
| `reachable` | `reachable: B from A` |
| `unreachable` | `unreachable: B from A` |
| `passes_through` | `passes_through: C for A -> B` |
| `within` | `within: B from A <= N` |
| `param_fraction` | `param_fraction: G <= P%` |
| `out_dim` | `out_dim: L <= <expr>` |

#### Scenario: All six kinds parse
- A document declares one bullet of each kind
- `parse()` yields six `Property` objects with the correct `kind` and operands
- Each `Property.raw` holds the verbatim bullet text

#### Scenario: A malformed bullet is reported, not ignored
- A bullet reads `- reachable: v_hat` (missing `from A`)
- Verification emits `PROPERTY_MALFORMED` at severity `error`, quoting the
  bullet

#### Scenario: An unrecognized property kind is reported
- A bullet reads `- eventually: y from obs`
- Verification emits `PROPERTY_MALFORMED`, quoting the bullet

#### Scenario: A property naming a nonexistent layer or group is reported
- A bullet reads `- reachable: nowhere from obs`
- Verification emits `PROPERTY_UNKNOWN_REFERENCE` naming `nowhere`

### Requirement: Reachability and exclusion are checked over member pairs

`reachable: B from A` SHALL hold iff there exist `a` in the resolution of `A`
and `b` in the resolution of `B` such that a directed path `a -> b` exists in
the layer DAG. `unreachable: B from A` SHALL hold iff no such pair exists —
i.e. for every `a` in `A` and every `b` in `B` there is no path `a -> b`.

A layer SHALL be considered to reach itself (path of length zero).

#### Scenario: Reachability passes
- `reachable: v_hat from obs` where a path exists
- No error is emitted

#### Scenario: Reachability fails
- `reachable: v_hat from unrelated_input` where no path exists
- `PROPERTY_REACHABILITY_FAIL` is emitted at severity `error`

#### Scenario: Exclusion passes on a real isolation boundary
- Groups `core` and `cortex` are declared; no edge leaves any `cortex` layer
  toward any `core` layer
- `unreachable: core from cortex` emits no error

#### Scenario: Exclusion is directed and asymmetric
- The same architecture carries a bus edge from `core` into `cortex`
- `unreachable: core from cortex` passes
- `unreachable: cortex from core` fails with `PROPERTY_EXCLUSION_FAIL`
- Both are checked independently; neither is inferred from the other

#### Scenario: Exclusion failure carries the counterexample path
- A leak exists from `cortex_ln` into `core_add` via `critic_fc` and
  `feedback_proj`
- The `PROPERTY_EXCLUSION_FAIL` message contains the literal substring
  `cortex_ln -> critic_fc -> feedback_proj -> core_add`

### Requirement: Pass-through treats the named operand as a cut set

`passes_through: C for A -> B` SHALL hold iff, after deleting **every** layer
in the resolution of `C` and all its incident edges, no `b` in `B` is
reachable from any `a` in `A`.

The property SHALL hold vacuously when `B` is already unreachable from `A` in
the unmodified graph.

#### Scenario: Pass-through passes when the cut set is the only route
- Group `bus` is the only route from `core` into `cortex`
- `passes_through: bus for core -> cortex` emits no error

#### Scenario: Pass-through catches a bypass edge around the named cut
- An edge is added from a `core` layer directly to a `cortex` layer, bypassing
  `bus`
- `PROPERTY_PATH_FAIL` is emitted, and its message contains the bypassing path
  as an arrow chain

#### Scenario: A multi-layer cut set must be deleted as a whole
- Group `bus` has two members forming two parallel routes
- Deleting either alone leaves a route; deleting both severs it
- `passes_through: bus for core -> cortex` passes

### Requirement: `within` bounds the shortest path length

`within: B from A <= N` SHALL hold iff some `b` in `B` is reachable from some
`a` in `A` in at most `N` edges. `N` SHALL be a non-negative integer literal.

#### Scenario: Bounded depth passes
- The shortest `obs -> y` path is 7 edges and the bullet reads `<= 12`
- No error is emitted

#### Scenario: Bounded depth fails, reporting the actual distance
- The shortest path is 9 edges and the bullet reads `<= 4`
- `PROPERTY_DEPTH_FAIL` is emitted and the message reports the actual shortest
  distance

#### Scenario: Bounded depth fails when no path exists at all
- No path exists between the operands
- `PROPERTY_DEPTH_FAIL` is emitted and the message states that no path exists,
  distinguishing it from the too-far case

### Requirement: `param_fraction` bounds a group's share of total parameters

`param_fraction: G <= P%` SHALL hold iff the summed parameter count of the
layers in the resolution of `G` is at most `P` percent of the architecture's
total parameter count. The numerator SHALL be computed with the same op
`params` call and the same hyperparameter substitution map that the resource
stage uses for the denominator.

The check SHALL be skipped when the total parameter count is zero.

#### Scenario: Param fraction passes
- Group `core` holds 3,433 of 3,226,609 parameters and the bullet reads `<= 1%`
- No error is emitted

#### Scenario: Param fraction fails with both the percentage and the raw counts
- Group `cortex` holds most of the parameters and the bullet reads `<= 1%`
- `PROPERTY_PARAM_FRACTION_FAIL` is emitted
- The message reports the actual percentage and the raw group and total counts

#### Scenario: A percent sign is required
- A bullet reads `- param_fraction: core <= 1`
- `PROPERTY_MALFORMED` is emitted

### Requirement: `out_dim` bounds a layer's inferred last dimension

`out_dim: L <= <expr>` SHALL hold iff the last dimension of `L`'s inferred
shape is at most the value of `<expr>`. `L` SHALL be a single layer. The
right-hand side SHALL be resolved through the same hyperparameter substitution
path the resource stage uses for op arguments, so a hyperparameter name or a
simple arithmetic expression over hyperparameters is accepted.

The check SHALL be skipped when either side does not resolve to an integer.

#### Scenario: Output width passes against a hyperparameter bound
- `nm_bus` infers last dim `n_affect` = 8 and the bullet reads `<= n_affect`
- No error is emitted

#### Scenario: Output width fails
- A layer infers last dim 512 and the bullet reads `<= 64`
- `PROPERTY_WIDTH_FAIL` is emitted, naming the layer, the actual width and the
  bound

#### Scenario: Naming a group where a layer is required is malformed
- A bullet reads `- out_dim: cortex <= 512` where `cortex` is a group
- `PROPERTY_MALFORMED` is emitted, since a group has no output width

#### Scenario: An unresolvable bound is skipped, not failed
- The bound is a free symbolic dimension the verifier treats as opaque
- No error is emitted

### Requirement: Stage 6 runs last and is skipped on an ill-formed graph

Property checking SHALL be a distinct stage (Stage 6) in
`n_orca/verifier/properties.py`, exposing `check_properties(arch, report)`,
dispatched from `verify()` in the same manner as the existing stages.

It SHALL run only when the naming and structural stages both passed, and SHALL
run after the shape and resource stages so that inferred shapes and parameter
counts are available. When an earlier stage failed, Stage 6 SHALL be skipped
**silently**, emitting no property or group diagnostics.

#### Scenario: Stage 6 is skipped when the structural stage fails
- A document declares properties and also contains a cycle
- Verification reports `CYCLE_DETECTED`
- No `PROPERTY_*` and no `GROUP_*` diagnostic appears in the report

#### Scenario: Stage 6 is skipped when the naming stage fails
- A document declares properties and a flow edge naming an undeclared layer
- Verification reports `UNKNOWN_LAYER_REFERENCE` only

#### Scenario: Stage 6 runs on a well-formed graph even when a shape error exists
- Naming and structural pass; a shape error is present
- Property results are still reported, since operand resolution and BFS only
  need a well-formed graph

### Requirement: Property results are reported on every surface

Property outcomes SHALL be surfaced through the report, the CLI (human and
`--json`), and the MCP `verify_markdown` payload. The CLI SHALL print a
`Properties: N checked, M failed` line, and SHALL honour `--strict` for
property warnings.

#### Scenario: The CLI prints a property summary line
- `n-orca verify` on a file with six passing properties
- Output includes `Properties: 6 checked, 0 failed`

#### Scenario: The CLI omits the line when no properties are declared
- `n-orca verify` on a file with no `## properties` section
- Output is byte-identical to the pre-change output

#### Scenario: JSON output carries the property results
- `n-orca verify --json` on a file with properties
- The payload includes a `properties` object with `checked`, `passed`,
  `failed`, and per-property results
- The payload includes a `groups` mapping of group name to member layers

#### Scenario: MCP verify_markdown carries the property results
- `verify_markdown` on a document with properties
- The returned architecture dict includes the same `properties` and `groups`
  keys the CLI JSON exposes

#### Scenario: `--strict` promotes a group warning
- A file with overlapping groups is verified with `--strict`
- `GROUP_OVERLAP` appears as an error and the file is reported invalid

### Requirement: Errors keep the three-part contract

Every new diagnostic SHALL carry a stable `code`, an instance-specific
`message`, and an imperative `suggestion`, as documented in
`docs/verification.md`. The codes SHALL be:
`PROPERTY_REACHABILITY_FAIL`, `PROPERTY_EXCLUSION_FAIL`, `PROPERTY_PATH_FAIL`,
`PROPERTY_DEPTH_FAIL`, `PROPERTY_PARAM_FRACTION_FAIL`, `PROPERTY_WIDTH_FAIL`,
`PROPERTY_UNKNOWN_REFERENCE`, `PROPERTY_MALFORMED`, `GROUP_NAME_COLLISION`
(all errors) and `GROUP_OVERLAP` (warning).

#### Scenario: Every new diagnostic has a suggestion
- Each of the ten codes is triggered by a fixture
- Every emitted diagnostic has a non-empty `suggestion`

#### Scenario: Messages quote the offending property
- Any property failure
- The message includes the property as written, so the author sees their own
  words

### Requirement: Groups render as Mermaid subgraphs

`compile_mermaid` SHALL render each declared group as a Mermaid `subgraph`
containing its member layers, so that an asserted boundary is visible in the
diagram. A layer listed in more than one group SHALL be emitted in the first
group that lists it.

#### Scenario: Groups become subgraphs
- An architecture with groups `core`, `cortex`, `bus` is compiled to Mermaid
- The output contains one `subgraph` block per group, each containing its
  member node declarations, and all edges still appear

#### Scenario: Ungrouped layers are still emitted
- Some layers belong to no group
- Those nodes are declared outside any subgraph and their edges are unchanged

### Requirement: The change is behaviour-preserving for existing documents

Documents without `## groups` or `## properties` SHALL verify and compile
exactly as before. Since unknown `##` sections were previously ignored
silently, adding these two section kinds SHALL NOT change the meaning of any
existing document.

#### Scenario: Every shipped example is unchanged
- `n-orca verify` is run over every file in `examples/` and
  `examples/hf-generated/` that existed before this change
- Output is byte-identical to the pre-change baseline

#### Scenario: Mermaid output is unchanged for ungrouped architectures
- `n-orca compile mermaid` is run over the same pre-existing examples
- Output is byte-identical to the pre-change baseline

#### Scenario: Render round-trips the new sections
- An architecture with groups and properties is rendered and re-parsed
- The resulting AST has equal `groups` and `properties`

### Requirement: Worked examples cover the positive and negative case

A positive example SHALL ship whose prose isolation claims are expressed as
real `## groups` and `## properties`, exercising all six property kinds and
verifying clean. A deliberately-broken twin SHALL ship that differs only by a
feedback path into the isolated core.

#### Scenario: The positive example verifies clean
- `n-orca verify examples/subcortical-value-cortex.n.orca.md`
- Result is `VALID` with `Properties: 6 checked, 0 failed`

#### Scenario: The negative example fails for exactly the right reason
- `n-orca verify examples/subcortical-value-cortex-leak.n.orca.md`
- Result is `INVALID`
- `PROPERTY_EXCLUSION_FAIL` is present with the bypass path in its message
- No shape, naming, or structural error is present — the file is well-formed
  and wrong only in the asserted boundary

#### Scenario: Both examples are covered by the CI example sweep
- The examples live in `examples/` and match the CI glob
- The negative example is exercised by a test asserting its failure, not by the
  CI pass-sweep

## Non-Requirements (for this change)

- `live` and `invariant` property kinds (superseded by Stage 2 and
  `## invariants` respectively).
- Guard-style pruning or any guard disclaimer in messages — flow edges are
  unconditional, so the checks are exact.
- A `max_layers` graph size limit.
- Any new op, any change to shape inference, any change to an existing error
  code or to existing verify output.
